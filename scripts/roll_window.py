"""Walk-forward retraining hook -- mechanism now, human-triggered.

`--dry-run` (default): reports how many graded prospective fights have
accumulated since the current model's data cutoff and, if >= PROMOTION_
THRESHOLD, prints the pre-registered promotion protocol. Never touches any
file. This is what the weekly refresh-data.yml Action runs and only prints.

`--execute`: actually runs the protocol once the threshold is met. It gates
on the TORCH ENSEMBLE -- the exact model the app serves -- not a proxy:

  1. Windows: NEW_CUTOFF = latest data date - VAL_WINDOW_YEARS; the held-
     forward validation slice is [NEW_CUTOFF, latest]. Both incumbent and
     candidate are scored on this same slice. It is out-of-time for the
     incumbent (its training cutoff is earlier), so there is no leakage.
  2. Incumbent: load the committed models/torch ensemble (per-seed
     temperatures included), predict on the new val slice, winner log-loss.
  3. Candidate: retrain the full 5-seed ensemble via scripts/train_torch.py
     into a TEMP dir (train < NEW_CUTOFF, val = the slice), load it, predict
     on the SAME slice, winner log-loss. The candidate fits its own per-seed
     temperatures on its own val slice -- each model is scored as its
     complete, self-contained artifact.
  4. Promote iff candidate_log_loss < incumbent_log_loss - PROMOTION_MARGIN.

If promoted, the candidate ensemble artifacts are STAGED into models/torch
(the incumbent is backed up on disk first) but NOTHING is committed: this
command performs NO git writes. A human then runs the full test suite,
reviews the metrics diff, and commits by hand -- that commit's git sha
becomes the new model_version, which starts a fresh track_record.json
section automatically (prospective predictions already key every fight by
the model_version active when it was made). If rejected, models/torch is
left untouched and the negative result is printed.

This is a manual, stage-only mechanism: it is deliberately NOT wired into
CI auto-promotion. The weekly Action only ever runs it in `--dry-run`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PREDICTIONS_DIR = ROOT / "predictions"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PROMOTION_THRESHOLD = 150
PROMOTION_MARGIN = 0.002
VAL_WINDOW_YEARS = 2
TORCH_SUBDIR = "torch"
CANDIDATE_DIR_NAME = "_roll_window_candidate"
BACKUP_DIR_NAME = "_roll_window_torch_backup"


def current_data_cutoff(features: pd.DataFrame) -> pd.Timestamp:
    """The freshest fight date the current model's artifacts have ever seen
    (train, val, or test), used as "how much data is genuinely new" -- the
    weekly refresh retrains models/ whenever new Kaggle data arrives, so
    this is always a good proxy for the incumbent's data cutoff."""
    return pd.Timestamp(features["date"].max())


def graded_fights_since(event_records: list[dict], cutoff: pd.Timestamp) -> list[dict]:
    """Flatten graded, non-skipped fights from events dated after `cutoff`."""
    graded = []
    for record in event_records:
        if pd.Timestamp(record["event_date"]) <= cutoff:
            continue
        for fight in record.get("fights", []):
            if fight.get("skipped"):
                continue
            if "actual_winner" in fight:
                graded.append(fight)
    return graded


def promotion_protocol_text(n_accumulated: int, cutoff: pd.Timestamp) -> str:
    return (
        f"{n_accumulated} graded prospective fight(s) have accumulated since the "
        f"current model's data cutoff ({cutoff.date()}), meeting the "
        f"{PROMOTION_THRESHOLD}-fight promotion-review threshold.\n\n"
        "Pre-registered promotion protocol:\n"
        "  1. Retrain with train < NEW_CUTOFF (NEW_CUTOFF = latest data date - "
        f"{VAL_WINDOW_YEARS} years).\n"
        "  2. Validate on the newest 2 years (NEW_CUTOFF to latest data date) -- "
        "a slice the retrained model never trained on.\n"
        "  3. Re-evaluate the CURRENT incumbent model on that SAME newest-2-years "
        "slice (not its original historic validation number).\n"
        f"  4. Promote only if new_log_loss < incumbent_log_loss - {PROMOTION_MARGIN} "
        "on that slice.\n"
        "  5. If promoted, the new model_version (next commit's git sha) starts a "
        "fresh track_record.json section automatically -- prospective predictions "
        "already key every fight by the model_version active when it was made.\n\n"
        "Run `python scripts/roll_window.py --execute` to run this end to end. "
        "The promotion gate retrains and scores the full 5-seed TORCH ENSEMBLE "
        "(the model the app serves) directly. On promotion the candidate ensemble "
        "is STAGED into models/torch -- this command makes NO git commit; a human "
        "runs the suite, reviews the diff, and commits by hand."
    )


def decide_promotion(new_log_loss: float, incumbent_log_loss: float,
                      margin: float = PROMOTION_MARGIN) -> bool:
    return new_log_loss < incumbent_log_loss - margin


def _load_event_records(predictions_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(predictions_dir.glob("UFC_*.json"))
    ]


def _report(features: pd.DataFrame, predictions_dir: Path) -> tuple[pd.Timestamp, list[dict]]:
    cutoff = current_data_cutoff(features)
    event_records = _load_event_records(predictions_dir)
    graded = graded_fights_since(event_records, cutoff)
    print(f"Current model's data cutoff: {cutoff.date()}")
    print(f"Graded prospective fights since cutoff: {len(graded)} "
          f"(threshold: {PROMOTION_THRESHOLD})")
    if len(graded) < PROMOTION_THRESHOLD:
        print("Below threshold -- no action. Nothing to do until more prospective "
              "fights are graded.")
    else:
        print()
        print(promotion_protocol_text(len(graded), cutoff))
    return cutoff, graded


def _ensemble_val_log_loss(
    ensemble_dir: Path, features: pd.DataFrame, val_start: str, val_end: str
) -> float:
    """Winner log-loss of the ensemble in `ensemble_dir` on the val slice.

    Loads the ensemble exactly as the app and scripts/final_test_eval.py do
    (per-seed checkpoints + committed temperatures), predicts the mean
    calibrated winner probability on the date-masked slice of
    features.parquet, and scores it against y_winner. This is the single
    evaluation both the incumbent (models/torch) and the freshly retrained
    candidate (a temp dir) go through, so each is measured as its complete,
    self-contained artifact on the identical held-forward slice.
    """
    from mma.evaluate import log_loss as compute_log_loss
    from mma.inference import Ensemble

    val_mask = (features["date"] >= val_start) & (features["date"] <= val_end)
    val = features.loc[val_mask]
    ensemble = Ensemble.load(ensemble_dir)
    p = ensemble.predict(val)["winner_prob"]
    y = val["y_winner"].to_numpy(dtype=float)
    return compute_log_loss(y, p)


def _retrain_candidate(
    out_dir: Path, train_end: str, val_start: str, val_end: str
) -> None:
    """Retrain the full 5-seed ensemble into `out_dir` (minutes-long)."""
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "train_torch.py"),
            "--train-end", train_end,
            "--val-start", val_start,
            "--val-end", val_end,
            "--out-dir", str(out_dir),
        ],
        cwd=ROOT, check=True,
    )


def _rebuild_display_priors() -> None:
    """Regenerate models/torch/display_priors.json from the staged ensemble.

    scripts/build_display_priors.py loads the committed ensemble
    (Ensemble.load() -> models/torch) and writes models/torch/
    display_priors.json, so running it AFTER the candidate is staged into
    models/torch makes the priors match the new ensemble. The display priors
    are train-split base-rate correction factors -- a new train window
    changes them, so a promotion leaves them stale unless rebuilt.
    """
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_display_priors.py")],
        cwd=ROOT, check=True,
    )


def _execute(features: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    latest = pd.Timestamp(features["date"].max())
    new_train_end = (latest - pd.DateOffset(years=VAL_WINDOW_YEARS)).date().isoformat()
    new_val_start = new_train_end
    new_val_end = latest.date().isoformat()
    print(f"\nExecuting walk-forward ensemble retrain: train < {new_train_end}, "
          f"val [{new_val_start}, {new_val_end}]")

    torch_dir = MODELS_DIR / TORCH_SUBDIR
    if not list(torch_dir.glob("net_seed*.pt")):
        print(f"No incumbent ensemble in {torch_dir} -- nothing to compare "
              "against. Aborting.")
        return

    incumbent_ll = _ensemble_val_log_loss(
        torch_dir, features, new_val_start, new_val_end
    )
    print(f"Incumbent ensemble log-loss on the new val slice: {incumbent_ll:.4f}")

    candidate_dir = MODELS_DIR / CANDIDATE_DIR_NAME
    candidate_torch_dir = candidate_dir / TORCH_SUBDIR
    shutil.rmtree(candidate_dir, ignore_errors=True)  # clear any stale run
    candidate_torch_dir.mkdir(parents=True, exist_ok=True)
    try:
        _retrain_candidate(
            candidate_torch_dir, new_train_end, new_val_start, new_val_end
        )
        candidate_ll = _ensemble_val_log_loss(
            candidate_torch_dir, features, new_val_start, new_val_end
        )
        print(f"Candidate ensemble log-loss on the same slice: {candidate_ll:.4f}")

        if decide_promotion(candidate_ll, incumbent_ll):
            backup_dir = MODELS_DIR / BACKUP_DIR_NAME
            shutil.rmtree(backup_dir, ignore_errors=True)
            shutil.copytree(torch_dir, backup_dir)  # incumbent safety copy
            for src in candidate_torch_dir.iterdir():
                if src.is_file():
                    shutil.copy2(src, torch_dir / src.name)

            # Keep the staged artifact set internally consistent: the display
            # priors are train-split base rates, now stale for the new
            # ensemble. Regenerate them from the just-staged ensemble. A
            # rebuild failure must NOT unstage the model -- warn and continue.
            try:
                _rebuild_display_priors()
                priors_line = (
                    "  1. display_priors.json has been regenerated for the new "
                    "ensemble (consistent with the staged artifacts)."
                )
            except Exception as exc:  # noqa: BLE001 -- graceful, never abort promotion
                priors_line = (
                    f"  1. WARNING: display_priors.json rebuild FAILED ({exc}). "
                    "The ensemble is still staged -- regenerate the priors "
                    "manually with `python scripts/build_display_priors.py` "
                    "before committing."
                )

            print(
                f"\nPROMOTED: {candidate_ll:.4f} beats incumbent "
                f"{incumbent_ll:.4f} by more than {PROMOTION_MARGIN}.\n"
                f"Candidate ensemble artifacts are STAGED into {torch_dir} "
                f"(incumbent backed up in {backup_dir}). Nothing has been "
                "committed -- this command makes no git writes. Next steps "
                "(manual, by design -- a human reviews before this ships):\n"
                f"{priors_line}\n"
                "  2. Run the full test suite.\n"
                "  3. Review the metrics diff (models/torch/metrics_val.json), then "
                "commit -- the new commit's git sha becomes the model_version for "
                "future predictions and starts a fresh track_record.json section.\n"
                f"  (To abandon: restore from {backup_dir}. Both {backup_dir} and "
                f"{candidate_dir} are gitignored.)"
            )
        else:
            print(
                f"\nREJECTED: {candidate_ll:.4f} does not beat incumbent "
                f"{incumbent_ll:.4f} by more than {PROMOTION_MARGIN}. "
                f"Leaving {torch_dir} untouched (negative result -- documented, "
                "not shipped)."
            )
    finally:
        shutil.rmtree(candidate_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                         help="run the protocol end to end instead of only reporting")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()

    features = pd.read_parquet(PROCESSED / "features.parquet")
    cutoff, graded = _report(features, args.predictions_dir)

    if args.execute:
        if len(graded) < PROMOTION_THRESHOLD:
            print("--execute requested but below threshold; nothing to do.")
            return
        _execute(features, cutoff)


if __name__ == "__main__":
    main()
