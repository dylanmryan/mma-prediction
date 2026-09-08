"""The one place a (fighter A, fighter B, context) pair becomes a feature row.

Both paths use it:
  * training  -- `mma.features.build_features` feeds it per-fight PRE-fight
    accumulator snapshots from `mma.history`;
  * serving   -- `mma.inference.build_matchup` feeds it current-state
    snapshots from `mma.snapshots`.
Keeping one implementation is what stops a feature from existing in the
training table but silently missing (or differing) at prediction time; the
value-level guard is `tests/test_serving_parity.py`.

`feature_row` is deliberately dtype-polymorphic: each state value may be a
scalar (serving builds one row) or a pandas Series (training builds every
row at once). The arithmetic below is written so both cases go through the
same expressions, so there is one contract and one set of semantics, not
two implementations that happen to agree today.

*Which* columns it builds is not decided here: `mma.feature_blocks` owns the
registry of named, switchable blocks and `feature_row` walks the spec it
assembles. Callers that pass nothing get the `base` block -- the v1 feature
contract -- unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mma.feature_blocks import BASE_BLOCK, Spec, spec_for


def minus(a, b):
    """A minus B, NaN when either side is missing.

    Scalars follow `float(a) - float(b)`; Series go through `pd.to_numeric`
    so integer inputs stay integer (that is what keeps `career_fights_diff`
    and friends int64 in the training table) and missing values propagate.
    """
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def as_flag(value):
    """Coerce a state value to a strict boolean (Series stay a bool Series)."""
    if isinstance(value, pd.Series):
        return value.fillna(False).astype(bool)
    return bool(value) if not pd.isna(value) else False


def debut_flag(career_fights):
    """No recorded fights yet. Missing count reads as a debut, as in training."""
    if isinstance(career_fights, pd.Series):
        return pd.to_numeric(career_fights, errors="coerce").fillna(0) == 0
    if career_fights is None or pd.isna(career_fights):
        return True
    return float(career_fights) == 0


def feature_row(state_a, state_b, context: dict | None = None,
                blocks=(BASE_BLOCK,)) -> dict:
    """One matchup -> {column: value} following the feature contract.

    `state_a` / `state_b` are anything supporting `.get(key)` over the state
    keys the enabled blocks name -- a plain dict of scalars at serving time,
    a DataFrame of columns at training time. `context` is the fight context
    (weight class, title flag, scheduled rounds, plus any fight-level block
    columns), passed through verbatim ahead of the features. `blocks` is a
    requested block list (or an already-assembled `Spec`); it defaults to the
    v1 `base` block, and the column order is exactly `feature_blocks.
    columns_for(blocks)`.
    """
    spec = blocks if isinstance(blocks, Spec) else spec_for(blocks)
    row: dict = dict(context) if context else {}
    for key, stem in spec.differentials:
        row[f"{stem}_diff"] = minus(state_a.get(key), state_b.get(key))
    for corner, state in (("a", state_a), ("b", state_b)):
        for stem in spec.absolutes:
            row[f"{stem}_{corner}"] = state.get(stem)
        for stem in spec.booleans:
            row[f"{stem}_{corner}"] = as_flag(state.get(stem))
    for name, stem in spec.derived_booleans:
        row[name] = row[f"{stem}_a"] ^ row[f"{stem}_b"]
    return row
