import os

# torch and xgboost each bundle an OpenMP runtime; loading both in one
# process segfaults on this platform unless OpenMP threading is pinned.
# Must be set before either library is imported, hence conftest.
os.environ.setdefault("OMP_NUM_THREADS", "1")


import json
from pathlib import Path

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def table_blocks() -> tuple[str, ...]:
    """The feature blocks `data/processed/features.parquet` was built from.

    A test that rebuilds the table and compares it against the committed one
    has to ask for the same blocks, or it compares a base-only rebuild
    against a block-enabled table and fails on shape rather than on the thing
    it is checking. `scripts/build_features.py` writes the sidecar next to the
    table for exactly this: it is the record of what is on disk.
    """
    from mma.feature_blocks import BASE_BLOCK

    sidecar = _PROCESSED / "features_blocks.json"
    if not sidecar.exists():
        return (BASE_BLOCK,)
    return tuple(json.loads(sidecar.read_text())["blocks"])
