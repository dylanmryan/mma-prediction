"""Deployed-model identity for the prospective track record.

The track record must split by *model*, not by git commit: the weekly
Action commits predictions every week, so a HEAD sha changes weekly while
the model does not. Hashing the artifact bytes that inference actually
loads gives a version that changes exactly when the model changes
(retrain, promotion, display-prior regeneration).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ARTIFACT_GLOBS = (
    "models/torch/net_seed*.pt",
    "models/torch/preprocess.json",
    "models/torch/display_priors.json",
    "models/xgb_winner.json",
    "models/xgb_method.json",
    "models/xgb_round.json",
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
