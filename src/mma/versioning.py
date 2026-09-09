"""Deployed-model identity for the prospective track record.

The track record must split by *model*, not by git commit: the weekly
Action commits predictions every week, so a HEAD sha changes weekly while
the model does not. Hashing the artifact bytes that inference actually
loads gives a version that changes exactly when the model changes.

The version identifies the *scorer*: since SP2.2 that is a blend, so the
glob set covers every artifact ``mma.inference.BlendedPredictor.load`` reads
-- the torch ensemble weights plus the preprocessing statistics, the
per-seed XGBoost boosters, AND ``models/blend.json``, the committed weight
and post-average temperature the two members are combined with -- because
every one of them moves a recorded probability. Before SP2.2 only the torch
half was hashed, which was correct then (the XGBoost models were an
explainer, not a scorer) and would have covered half of what serves now.

``models/blend.json`` is hashed for the same reason as the model weights: the
blend's mixing weight and post-average temperature are as much a part of the
scorer as any trained parameter is -- changing either one changes every
recorded probability just as retraining a booster does, even though neither
number is learned by gradient descent. Before that file existed, both were
hand-set module constants that this hash never covered, so editing one
silently changed what every future prediction meant while leaving this
version byte-identical; see ``scripts/build_blend_config.py``, which derives
the file from the walk-forward report rather than letting it drift.

Display priors (applied only in ``app.py``, after the probability the track
record stores) remain deliberately excluded, so regenerating them does not
open a new track-record section for byte-identical predictions. The hash
changes exactly when a retrain, a walk-forward promotion, or a change to the
blend's own weight or temperature changes what gets scored, on either side
of the blend.
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
    "models/blend.json",
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
