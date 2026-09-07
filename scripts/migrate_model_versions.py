"""One-time migration: re-key prediction records to an artifact-hash model_version.

Before this migration, `model_version` was the git HEAD short sha at
prediction time, which changed with every weekly commit even though the
model never did. Every record carrying one of the old commit shas was
produced by the same artifacts, so they all collapse to that single
artifact hash. Idempotent: values already equal to `--to` are left alone.

Usage:
    python scripts/migrate_model_versions.py --to <12-hex hash> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "predictions"
_GIT_SHORT_SHA = re.compile(r"^[0-9a-f]{7}$")


def rekey_record(record: dict, new_version: str) -> int:
    """Replace every git-sha-shaped model_version in place; return count changed."""
    changed = 0
    if _GIT_SHORT_SHA.match(str(record.get("model_version", ""))):
        record["model_version"] = new_version
        changed += 1
    for fight in record.get("fights", []):
        if _GIT_SHORT_SHA.match(str(fight.get("model_version", ""))):
            fight["model_version"] = new_version
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="artifact hash from mma.versioning")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{12}", args.to):
        raise SystemExit(f"--to must be a 12-hex artifact hash, got {args.to!r}")

    total = 0
    for path in sorted(args.predictions_dir.glob("*.json")):
        if path.name == "track_record.json":
            continue
        record = json.loads(path.read_text())
        changed = rekey_record(record, args.to)
        total += changed
        print(f"{path.name}: {changed} value(s) re-keyed")
        if changed and not args.dry_run:
            path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"total re-keyed: {total}{' (dry run)' if args.dry_run else ''}")
    print("now run scripts/grade_predictions.py to regenerate track_record.json")


if __name__ == "__main__":
    main()
