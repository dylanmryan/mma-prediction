"""Deployed-model identity for the prospective track record.

The track record must split by *model*, not by git commit: the weekly
Action commits predictions every week, so a HEAD sha changes weekly while
the model does not. Hashing the artifact bytes that inference actually
loads gives a version that changes exactly when the model changes.

The version identifies the *scorer*: since SP2.2 that is a blend, so the
glob set covers BOTH members -- the torch ensemble weights plus the
preprocessing statistics, and the per-seed XGBoost boosters -- because
``mma.inference.BlendedPredictor.load`` reads all of them and every one of
them moves a recorded probability. Before SP2.2 only the torch half was
hashed, which was correct then (the XGBoost models were an explainer, not a
scorer) and would have covered half of what serves now.

Display priors (applied only in ``app.py``, after the probability the track
record stores) remain deliberately excluded, so regenerating them does not
open a new track-record section for byte-identical predictions. The hash
changes exactly when a retrain or a walk-forward promotion changes what gets
scored, on either side of the blend.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ARTIFACT_GLOBS = (
    "models/torch/net_seed*.pt",
    "models/torch/preprocess.json",
    "models/xgb_winner_seed*.json",
    "models/xgb_method_seed*.json",
    "models/xgb_round_seed*.json",
)


def model_version(root: Path) -> str:
    """12-hex sha256 prefix over the sorted (path, bytes) of every artifact."""
    root = Path(root)
    paths = sorted(
        {path for pattern in MODEL_ARTIFACT_GLOBS for path in root.glob(pattern)}
    )
    if not paths:
        raise FileNotFoundError(f"no model artifacts under {root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]
