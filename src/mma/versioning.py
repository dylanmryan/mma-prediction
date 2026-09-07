"""Deployed-model identity for the prospective track record.

The track record must split by *model*, not by git commit: the weekly
Action commits predictions every week, so a HEAD sha changes weekly while
the model does not. Hashing the artifact bytes that inference actually
loads gives a version that changes exactly when the model changes.

The version identifies the *scorer* -- the torch ensemble weights plus the
preprocessing statistics that ``mma.inference.Ensemble.load`` reads --
because those are the only inputs to the recorded probabilities. Display
priors (applied only in ``app.py``) and the XGBoost explainer models
(used only by the app's SHAP explainer and ``scripts/final_test_eval.py``)
are deliberately excluded, so regenerating them does not open a new
track-record section for byte-identical predictions. The hash changes
exactly when a retrain or walk-forward promotion changes what gets scored.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ARTIFACT_GLOBS = (
    "models/torch/net_seed*.pt",
    "models/torch/preprocess.json",
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
