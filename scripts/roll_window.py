"""Walk-forward retraining hook -- mechanism now, human-triggered.

`--dry-run` (default): reports how many graded prospective fights have
accumulated since the current model's data cutoff and, if >= PROMOTION_
THRESHOLD, prints the pre-registered promotion protocol. Never touches any
file. This is what the weekly refresh-data.yml Action runs and only prints.

`--execute`: runs the split-protocol comparison once the threshold is met.
It gates on the TORCH ENSEMBLE, which is one of the four models the app now
serves. Since SP2.2 the winner probability is a blend of that ensemble and a
5-seed XGBoost ensemble; since SP3 the deployed scorer is the HYBRID
(`mma.inference.SimulatorPredictor`), which takes its winner from that blend
and its method, finish round and joint distribution from a Monte Carlo
simulator built on two more models this gate never touches. So it cannot
answer "does a fresh fit beat the incumbent?" about the served model at all,
and `--execute` refuses to run against a blended or hybrid incumbent
(`blended_incumbent` / `hybrid_incumbent` below) rather than reporting a
number about part of a model as if it were about the model.
`incumbent_coverage_text` writes that sentence from the artifacts on disk, so
it stays true as the deployment changes. Bringing the gate onto the served
scorer, or onto the walk-forward harness, is SP4 work. What it does when it
does run:

  1. Windows: NEW_CUTOFF = latest data date - VAL_WINDOW_YEARS; the held-
     forward validation slice is [NEW_CUTOFF, latest]. Both incumbent and
     candidate are scored on this same slice.
  2. Incumbent: load the committed models/torch ensemble (per-seed
     temperatures included), predict on the new val slice, winner log-loss.
  3. Candidate: retrain the full 5-seed ensemble via scripts/train_torch.py
     into a TEMP dir (split protocol: train < NEW_CUTOFF, early-stop and
     fit temperatures on the slice), load it, predict on the SAME slice,
     winner log-loss.
  4. Promote iff candidate_log_loss < incumbent_log_loss - PROMOTION_MARGIN.

CAVEAT 1 -- the candidate is one member of four. See above: `--execute` aborts
on a blended or hybrid incumbent.

CAVEAT 2 -- the comparison is only valid when the slice is OUT-OF-TIME for the
incumbent. Under the default deployment recipe (refit_through_latest, see
models/walkforward/refit_decision_v3.json and the train scripts) the incumbent
is trained on EVERY fight through the latest data date, so its training
cutoff equals the latest date and the two-year "held-forward" slice is
IN-SAMPLE for it: the incumbent would be scored on fights it trained on,
which biases the gate against any candidate. `--execute` therefore reads
models/torch/metrics_val.json and ABORTS when the incumbent's `mode` is
"refit_through" and its `train_through` reaches into the slice. Until SP4
moves this gate onto the walk-forward harness (scripts/run_walkforward.py,
reports under models/walkforward/), the harness -- not this two-year slice
-- is the primary comparison between model recipes.

A candidate produced here must never ship as-is: it is a split-protocol
model (early-stopped, per-seed temperatures fit on the slice, two years
less training data than the refit recipe). Its job is only to answer "does
a fresh fit beat the incumbent on held-forward data?". If promoted, the
candidate ensemble artifacts are STAGED into models/torch (the incumbent is
backed up on disk first) but NOTHING is committed: this command performs NO
git writes. After any promotion decision the human re-runs the refit recipe
-- bare `scripts/train_xgb.py`, `scripts/train_torch.py`,
`scripts/train_hazard.py`, then `scripts/check_display_calibration.py` -- so the
deployed model includes the
newest fights, runs the full test suite, reviews the metrics diff, and
commits by hand. The `model_version` that starts a fresh track_record.json
section is the artifact hash from `mma.versioning.model_version` (over every
member: the torch weights and preprocessing stats, the XGBoost seed boosters,
the simulator's hazard and decision models, and the committed blend and
simulator configuration), NOT a commit sha; prospective
predictions already key every fight by the model_version active when it was
made. If rejected, models/torch is left untouched and the negative result
is printed.

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


def promotion_protocol_text(n_accumulated: int, cutoff: pd.Timestamp,
                            models_dir: Path = None) -> str:
    """The dry-run text. `models_dir` decides how CAVEAT 1 describes the
    incumbent (see `incumbent_coverage_text`) -- it is read from the artifacts
    so the caveat cannot go stale the way a hand-written sentence about "the
    blend" did when the simulator shipped."""
    models_dir = MODELS_DIR if models_dir is None else models_dir
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
        "  5. If promoted, the new model_version (the artifact hash from "
        "mma.versioning.model_version, not a commit sha) starts a fresh "
        "track_record.json section automatically -- prospective predictions "
        "already key every fight by the model_version active when it was made.\n\n"
        "Run `python scripts/roll_window.py --execute` to run this end to end. "
        "The promotion gate retrains and scores the full 5-seed TORCH ENSEMBLE "
        "directly. On promotion the candidate ensemble is STAGED into "
        "models/torch -- this command makes NO git commit; a human runs the "
        "suite, reviews the diff, and commits by hand.\n\n"
        f"CAVEAT 1: {incumbent_coverage_text(models_dir)}, so --execute aborts "
        "rather than reporting a number about part of a model as if it were "
        "about the model.\n"
        "CAVEAT 2: under the default refit_through_latest recipe the incumbent is "
        "trained through the latest data date, so the newest-2-years slice is "
        "IN-SAMPLE for it and --execute aborts for that reason too (steps 2-4 are "
        "not a valid gate). "
        "The walk-forward harness (models/walkforward/, scripts/run_walkforward.py, "
        "including --candidate blend and --candidate hybrid) is the primary "
        "comparison until SP4 moves "
        "this gate onto it. A split-protocol candidate never ships as-is: after "
        "any promotion decision, re-run bare scripts/train_xgb.py, "
        "scripts/train_torch.py, scripts/train_hazard.py and "
        "scripts/check_display_calibration.py (the refit recipe) before committing."
    )


def decide_promotion(new_log_loss: float, incumbent_log_loss: float,
                      margin: float = PROMOTION_MARGIN) -> bool:
    return new_log_loss < incumbent_log_loss - margin


def blended_incumbent(models_dir: Path) -> bool:
    """True when the deployed scorer is a BLEND, not the torch ensemble alone.

    Detected from the artifacts themselves rather than a flag file: a deployed
    XGBoost seed ensemble (`xgb_<head>_seed*.json`) next to `models/torch` is
    what `mma.inference.BlendedPredictor.load` reads, and what
    `mma.versioning.MODEL_ARTIFACT_GLOBS` hashes. When it is there, this
    module's split-protocol gate retrains and scores half the served model, so
    the comparison is not a promotion gate for anything that ships.
    """
    return bool(list(Path(models_dir).glob("xgb_winner_seed*.json")))


def hybrid_incumbent(models_dir: Path) -> bool:
    """True when the deployed scorer is the SP3 HYBRID.

    Same artifact-detection rule as `blended_incumbent`: the simulator's two
    members (`xgb_hazard_seed*.json`, `xgb_decision_seed*.json`) are what
    `mma.inference.SimulatorPredictor.load` reads. It matters separately from
    the blend because it makes this gate's coverage smaller again, and in a
    different way: the blend still supplies the whole winner probability, so a
    torch-only number is half of THAT -- but method, finish round and the
    joint distribution the app and the prediction records now show come from
    the simulator, which this gate does not touch at all.
    """
    return bool(list(Path(models_dir).glob("xgb_hazard_seed*.json"))
                and list(Path(models_dir).glob("xgb_decision_seed*.json")))


def incumbent_coverage_text(models_dir: Path) -> str:
    """What fraction of the served scorer this gate actually retrains, in
    words, given the artifacts on disk. Used by both the dry-run protocol text
    and the --execute abort, so the two cannot describe different models."""
    if hybrid_incumbent(models_dir):
        return (
            "the deployed scorer is the SP3 HYBRID: the blend (a 5-seed XGBoost "
            "ensemble plus the 5-seed torch ensemble, temperature-scaled) supplies "
            "P(A wins), and the Monte Carlo fight simulator (a 5-seed hazard model "
            "and a 5-seed decision model) supplies P(method, round | winner). This "
            "gate retrains and scores the TORCH ENSEMBLE alone -- half of the winner "
            "probability, and none of the method, round or joint distribution the "
            "app and the prediction records show"
        )
    if blended_incumbent(models_dir):
        return (
            "the deployed scorer is a BLEND of the torch ensemble and a 5-seed "
            "XGBoost ensemble, and this gate retrains and scores the torch member "
            "alone -- half the served model"
        )
    return "the deployed scorer is the torch ensemble this gate retrains"


def incumbent_in_sample(metrics: dict, new_val_start) -> bool:
    """True when the incumbent's training data reaches into the held-forward
    slice starting at `new_val_start`, i.e. the split-protocol comparison
    would score the incumbent on fights it trained on.

    `metrics` is models/torch/metrics_val.json. Only the refit_through mode
    records `train_through`; a split-mode file (no `mode` key) trained on
    `date < train_end` with the slice after it, so it is never in-sample
    here."""
    if metrics.get("mode") != "refit_through":
        return False
    return pd.Timestamp(metrics["train_through"]) >= pd.Timestamp(new_val_start)


def _load_incumbent_metrics(torch_dir: Path) -> dict:
    path = torch_dir / "metrics_val.json"
    return json.loads(path.read_text()) if path.exists() else {}


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
    """Winner log-loss of the TORCH ensemble in `ensemble_dir` on the val slice.

    Loads the ensemble from its per-seed checkpoints and committed
    temperatures, predicts the mean calibrated winner probability on the
    date-masked slice of features.parquet, and scores it against y_winner.
    Both the incumbent (models/torch) and the freshly retrained candidate (a
    temp dir) go through this identical evaluation.

    Note what it is NOT: since SP2.2 the served model is a blend, and this
    scores the torch member on its own. `_execute` refuses to run against a
    blended incumbent for exactly that reason (`blended_incumbent`), so this
    function is only ever reached for a torch-only deployment.
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


def _remeasure_display_calibration() -> None:
    """Re-measure models/display_calibration.json against the staged ensemble.

    scripts/check_display_calibration.py loads the committed scorer and
    measures its aggregate method/round marginals against the base rates on
    its own training rows (mma.inference.deployed_training_mask, read from the
    staged metrics_val.json). A new ensemble and a new training window both
    change that measurement, so a promotion leaves it stale unless it is run
    again. Since SP3 it is a measurement rather than a correction the app
    applies, so a stale file misleads a reader rather than changing a
    prediction -- which is why a failure here warns and never unstages the
    model.
    """
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_display_calibration.py")],
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

    if blended_incumbent(MODELS_DIR) or hybrid_incumbent(MODELS_DIR):
        raise SystemExit(
            f"--execute aborted: {incumbent_coverage_text(MODELS_DIR)} (artifacts "
            f"in {MODELS_DIR}, torch ensemble in {torch_dir}). A number about part "
            "of the served model is not a promotion gate for the served model, so "
            "it is not produced. Compare recipes with the walk-forward harness "
            "instead (scripts/run_walkforward.py --candidate blend or --candidate "
            "hybrid, reports under models/walkforward/); moving this gate onto the "
            "served scorer is SP4 work."
        )

    incumbent_metrics = _load_incumbent_metrics(torch_dir)
    if incumbent_in_sample(incumbent_metrics, new_val_start):
        raise SystemExit(
            f"--execute aborted: the incumbent in {torch_dir} was trained with the "
            f"refit_through recipe through {incumbent_metrics['train_through']}, "
            f"which is inside the held-forward slice [{new_val_start}, "
            f"{new_val_end}]. Scoring it there would be in-sample, so the "
            "split-protocol comparison is not a valid promotion gate. Compare "
            "recipes with the walk-forward harness instead (scripts/"
            "run_walkforward.py, reports under models/walkforward/); moving this "
            "gate onto the harness is SP4 work."
        )

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
            # the display calibration measured the OLD ensemble on its own
            # training rows, and is now stale. Re-measure it against the
            # just-staged ensemble. A failure must NOT unstage the model --
            # warn and continue.
            try:
                _remeasure_display_calibration()
                priors_line = (
                    "  1. display_calibration.json has been re-measured for the new "
                    "ensemble (consistent with the staged artifacts)."
                )
            except Exception as exc:  # noqa: BLE001 -- graceful, never abort promotion
                priors_line = (
                    f"  1. WARNING: display_calibration.json rebuild FAILED ({exc}). "
                    "The ensemble is still staged -- re-measure manually with "
                    "`python scripts/check_display_calibration.py` before committing."
                )

            print(
                f"\nPROMOTED: {candidate_ll:.4f} beats incumbent "
                f"{incumbent_ll:.4f} by more than {PROMOTION_MARGIN}.\n"
                f"Candidate ensemble artifacts are STAGED into {torch_dir} "
                f"(incumbent backed up in {backup_dir}). Nothing has been "
                "committed -- this command makes no git writes. Next steps "
                "(manual, by design -- a human reviews before this ships):\n"
                f"{priors_line}\n"
                "  2. Do NOT ship this split-protocol candidate as-is: re-run the "
                "refit recipe (bare scripts/train_xgb.py, scripts/train_torch.py, "
                "scripts/train_hazard.py, scripts/check_display_calibration.py) so "
                "the deployed model includes "
                "the newest fights.\n"
                "  3. Run the full test suite, review the metrics diff "
                "(models/torch/metrics_val.json), then commit -- the artifact hash "
                "(mma.versioning.model_version) becomes the model_version for "
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
