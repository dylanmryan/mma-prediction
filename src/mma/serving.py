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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (state key, output stem) -> "<stem>_diff", A minus B.
# The order here is the column order of the feature table.
DIFFERENTIALS: tuple[tuple[str, str], ...] = (
    ("career_fights", "career_fights"),
    ("career_wins", "career_wins"),
    ("career_win_rate", "career_win_rate"),
    ("career_finish_rate", "career_finish_rate"),
    ("kd_pf", "kd_pf"),
    ("sub_att_pf", "sub_att_pf"),
    ("td_landed_pf", "td_landed_pf"),
    ("td_acc", "td_acc"),
    ("td_def", "td_def"),
    ("sig_pm", "sig_pm"),
    ("sig_absorbed_pm", "sig_absorbed_pm"),
    ("ctrl_share", "ctrl_share"),
    ("streak", "streak"),
    ("days_since_last", "days_since_last"),
    ("last5_win_rate", "last5_win_rate"),
    ("last5_avg_opp_elo", "last5_avg_opp_elo"),
    ("pre_overall", "elo"),
    ("pre_striking", "striking_elo"),
    ("pre_grappling", "grappling_elo"),
    ("pre_fights", "elo_fights"),
    ("height_cm", "height"),
    ("reach_cm", "reach"),
    ("age", "age"),
)

# Emitted per corner as "<stem>_a" / "<stem>_b", unchanged from the state.
ABSOLUTES: tuple[str, ...] = ("age", "career_fights")

# Per-corner booleans, likewise "<stem>_a" / "<stem>_b".
BOOLEANS: tuple[str, ...] = ("reach_missing", "dob_missing", "southpaw", "debut")

# Derived corner-symmetric flags: (output column, per-corner stem XOR-ed).
DERIVED_BOOLEANS: tuple[tuple[str, str], ...] = (
    ("debut_matchup", "debut"),
    ("stance_mismatch", "southpaw"),
)


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


def feature_row(state_a, state_b, context: dict | None = None) -> dict:
    """One matchup -> {column: value} following the feature contract.

    `state_a` / `state_b` are anything supporting `.get(key)` over the state
    keys named in `DIFFERENTIALS`, `ABSOLUTES` and `BOOLEANS` -- a plain dict
    of scalars at serving time, a DataFrame of columns at training time.
    `context` is the fight context (weight class, title flag, scheduled
    rounds), passed through verbatim ahead of the features.
    """
    row: dict = dict(context) if context else {}
    for key, stem in DIFFERENTIALS:
        row[f"{stem}_diff"] = minus(state_a.get(key), state_b.get(key))
    for corner, state in (("a", state_a), ("b", state_b)):
        for stem in ABSOLUTES:
            row[f"{stem}_{corner}"] = state.get(stem)
        for stem in BOOLEANS:
            row[f"{stem}_{corner}"] = as_flag(state.get(stem))
    for name, stem in DERIVED_BOOLEANS:
        row[name] = row[f"{stem}_a"] ^ row[f"{stem}_b"]
    return row
