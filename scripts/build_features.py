"""Build the model-ready feature table from processed parquet.

    python scripts/build_features.py                       # the committed contract
    python scripts/build_features.py --blocks in_fight     # base + in_fight

`--blocks` names blocks from `mma.feature_blocks`; `base` is always on and
does not need naming. The sidecar `features_blocks.json` records which
blocks the table was built from, so a walk-forward report can be traced
back to the table it was computed on. It records nothing time-varying: it
is committed alongside the table, and a rebuild that changes no feature
must leave `git status` clean, which is how the byte-identity check on
features.parquet is read.

WITHOUT `--blocks` this rebuilds THE BLOCKS THE COMMITTED TABLE WAS BUILT
FROM, read from that sidecar (`feature_blocks.table_blocks()`), falling back
to the v1 `base` contract in a checkout that has never built one. That is not
cosmetic: the weekly refresh Action runs this script bare, so a `base` default
would quietly rebuild the shipped table without the blocks the deployed model
was trained and measured on. Nothing would crash -- the preprocessor and the
XGBoost matrix are both fitted on whatever columns exist -- and the next
retrain would simply deploy a worse model. Named blocks always win, so an
experiment still builds exactly what it asks for.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mma.feature_blocks import BLOCKS_SIDECAR, resolve_blocks, table_blocks
from mma.features import build_features
from mma.history import build_history

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUT = PROCESSED / "features.parquet"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--blocks", default=None,
        help="comma-separated feature blocks to build (base is always included; "
             "default: the blocks the committed table was built from)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def resolve_blocks_arg(spec: str | None, sidecar=BLOCKS_SIDECAR) -> tuple[str, ...]:
    """The blocks to build: those named, or the committed table's own."""
    if spec is None:
        return table_blocks(sidecar)
    return resolve_blocks([b.strip() for b in spec.split(",") if b.strip()])


def main(argv=None) -> None:
    args = parse_args(argv)
    blocks = resolve_blocks_arg(args.blocks)

    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")

    history = build_history(fights, stats, ratings)
    features = build_features(fights, fighters, ratings, history, blocks=blocks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)

    sidecar = args.out.parent / f"{args.out.stem}_blocks.json"
    sidecar.write_text(json.dumps({
        "blocks": list(blocks),
        "n_rows": int(len(features)),
        "n_columns": int(features.shape[1]),
    }, indent=2) + "\n")

    print(f"blocks: {', '.join(blocks)} -> {args.out}")
    print(f"{len(features)} rows, {features.shape[1]} columns")
    print("y_winner balance:", features["y_winner"].mean().round(4))
    print("swapped share:", features["swapped"].mean().round(4))
    per_year = features.groupby(features["date"].dt.year).size()
    print("rows/year (last 6):")
    print(per_year.tail(6).to_string())


if __name__ == "__main__":
    main()
