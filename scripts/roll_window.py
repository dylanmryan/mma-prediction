"""Walk-forward retraining hook -- mechanism now, human-triggered.

`--dry-run` (default): reports how many graded prospective fights have
accumulated since the current model's data cutoff and, if >= PROMOTION_
THRESHOLD, prints the pre-registered promotion protocol. Never touches any
file. This is what the weekly refresh-data.yml Action runs and only prints.

`--execute`: actually runs the protocol once the threshold is met --
retrains the XGBoost winner model with train < NEW_CUTOFF, validates on the
newest 2 years, and promotes (keeps the new artifacts) only if the new
validation log-loss beats the incumbent's -- RE-EVALUATED ON THE SAME NEW
SLICE, not its original historic number -- by more than PROMOTION_MARGIN.

Scope note: --execute's promotion gate is decided on the XGBoost winner
model only (fast, deterministic-ish, and this is a rarely-run, human-
triggered mechanism, not something that needs the full 5-seed torch
ensemble on every check). If promotion is approved, it prints the exact
`scripts/train_torch.py` invocation (same train/val window) to rebuild the
full ensemble before committing -- deliberately manual, per the design, to
keep a human reviewing the diff before anything ships.
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
MODEL_ARTIFACTS = (
    "xgb_winner.json", "xgb_method.json", "xgb_round.json", "xgb_metrics_val.json",
)


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
        "Run `python scripts/roll_window.py --execute` to run this end to end "
        "(promotion gate uses the XGBoost winner model; a full ensemble retrain "
        "if promoted is a manual follow-up step -- see printed instructions)."
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


def _execute(features: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    latest = pd.Timestamp(features["date"].max())
    new_train_end = (latest - pd.DateOffset(years=VAL_WINDOW_YEARS)).date().isoformat()
    new_val_start = new_train_end
    new_val_end = latest.date().isoformat()
    print(f"\nExecuting walk-forward retrain: train < {new_train_end}, "
          f"val [{new_val_start}, {new_val_end}]")

    from mma.evaluate import log_loss as compute_log_loss
    from mma.models.xgb import feature_frame
    import xgboost as xgb

    val_mask = (features["date"] >= new_val_start) & (features["date"] <= new_val_end)
    x_new_val = feature_frame(features[val_mask])
    y_new_val = features.loc[val_mask, "y_winner"]

    incumbent_path = MODELS_DIR / "xgb_winner.json"
    if not incumbent_path.exists():
        print("No incumbent xgb_winner.json found -- nothing to compare against. Aborting.")
        return
    incumbent = xgb.XGBClassifier()
    incumbent.load_model(incumbent_path)
    incumbent_p = incumbent.predict_proba(x_new_val)[:, 1]
    incumbent_log_loss = compute_log_loss(y_new_val, incumbent_p)
    print(f"Incumbent log-loss re-evaluated on the new val slice: {incumbent_log_loss:.4f}")

    backup_dir = MODELS_DIR / "_roll_window_backup"
    backup_dir.mkdir(exist_ok=True)
    for name in MODEL_ARTIFACTS:
        src = MODELS_DIR / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)

    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "train_xgb.py"),
            "--train-end", new_train_end,
            "--val-start", new_val_start,
            "--val-end", new_val_end,
        ],
        cwd=ROOT, check=True,
    )

    new_metrics = json.loads((MODELS_DIR / "xgb_metrics_val.json").read_text())
    new_log_loss = new_metrics["winner"]["log_loss"]
    print(f"Retrained model log-loss on the same slice: {new_log_loss:.4f}")

    if decide_promotion(new_log_loss, incumbent_log_loss):
        print(
            f"\nPROMOTED: {new_log_loss:.4f} beats incumbent "
            f"{incumbent_log_loss:.4f} by more than {PROMOTION_MARGIN}.\n"
            "New XGBoost artifacts are staged in models/. Next steps (manual, "
            "by design -- a human reviews before this ships):\n"
            f"  1. Rebuild the full ensemble: python scripts/train_torch.py "
            f"--train-end {new_train_end} --val-start {new_val_start} "
            f"--val-end {new_val_end}\n"
            "  2. Run the full test suite.\n"
            "  3. Review the metrics diff, then commit -- the new commit's git sha "
            "becomes the new model_version for future predictions."
        )
        shutil.rmtree(backup_dir, ignore_errors=True)
    else:
        print(
            f"\nREJECTED: {new_log_loss:.4f} does not beat incumbent "
            f"{incumbent_log_loss:.4f} by more than {PROMOTION_MARGIN}. "
            "Restoring incumbent artifacts (negative result -- documented, not shipped)."
        )
        for name in MODEL_ARTIFACTS:
            backup = backup_dir / name
            if backup.exists():
                shutil.copy2(backup, MODELS_DIR / name)
        shutil.rmtree(backup_dir, ignore_errors=True)


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
