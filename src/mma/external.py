"""Load the derived external snapshot and turn it into feature-block state.

`data/external/fighter_external.parquet` (built by `scripts/build_external.py`
from the MIT-licensed `ehan03/jds-mma-data` snapshot) holds one row per
matched fighter describing what they did BEFORE the UFC: their pre-UFC record,
how they won and lost it, the quality of the opposition, their pro debut date,
and their nationality and gym. This module is the only place that table is
read, and it does two things with it.

**Join by id, never by name.** The derived table carries no name column at
all, so there is nothing to fall back to: a fighter whose ufcstats
`fighter_id` is not in the table is simply missing. That matters because the
snapshot's UFC coverage ends 2024-12-14 -- roughly a fifth of our feature rows
have at least one corner the snapshot has never seen, and a name-based rescue
attempt would trade honest missingness for silent mismatches.

**Missing means missing.** Every state value for an unmatched corner is NaN
and its `external_missing` flag is True; the row-level `external_missing`
(either corner missing) is the fight-level column
`mma.walkforward.slice_masks` picks up, so the post-snapshot rows are reported
as their own walk-forward slice rather than being averaged away.

A fighter whose recorded pro debut post-dates their first UFC bout is a source
data error -- the pre-UFC window it implies cannot be right, and a naive
`fight_date - pro_debut` would still be positive while summarising the wrong
bouts. `load_table` drops those fighters, so they take the ordinary
missing-fighter path instead of contributing a wrong number.

`days_since_pro_debut` is the one column that is not fighter-static: the table
stores the DATE and this module subtracts it from each fight's own date,
exactly as `days_since_last` is handled.

**Why this block is point-in-time safe**, stated honestly. It is NOT because
`tests/test_processed_features.py::test_no_leakage_truncation_invariance`
passes. That test rebuilds the feature table from a truncated fights table and
compares every column; the external table is a static committed artifact that
the truncation does not touch, so the test is structurally incapable of
detecting a leak here -- it would pass just as happily on a column derived
from next year's results. The argument is the cut itself:
`scripts/build_external.derive` selects `history["date"] < first_ufc_date`,
where `first_ufc_date` is the fighter's first bout in OUR fights table. Every
`pre_ufc_*` value therefore summarises a window that closed strictly before
the fighter's UFC career began, which makes it CONSTANT across all of that
fighter's UFC fights -- so it cannot carry information from one of their
fights into another, let alone from the future. That constancy is the
falsifiable form of the claim, and
`tests/test_external.py::test_pre_ufc_values_are_constant_across_a_fighters_ufc_career`
asserts it directly on the real table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "external" / "fighter_external.parquet"

# The pre-UFC aggregates, all fighter-static.
NUMERIC_STATE_KEYS = (
    "pre_ufc_wins",
    "pre_ufc_losses",
    "pre_ufc_finish_rate",
    "pre_ufc_finish_loss_rate",
    "pre_ufc_avg_opp_wins",
)
# Everything the `external` block declares: the aggregates, the as-of duration
# and the per-corner missingness flag. `mma.features` and
# `mma.inference.build_matchup` both build their state from this tuple.
STATE_KEYS = NUMERIC_STATE_KEYS + ("days_since_pro_debut", "external_missing")
# Carried alongside the state so `fight_context` can compare origins; not a
# feature on its own (a high-cardinality string).
NATIONALITY = "nationality"

_MISSING_STATE = {key: np.nan for key in NUMERIC_STATE_KEYS} | {
    "days_since_pro_debut": np.nan,
    "external_missing": True,
    NATIONALITY: None,
}


def drop_source_errors(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only fighters whose pro debut is a real date on or before their
    first UFC bout.

    Dropping rather than repairing is deliberate: "not in the table" is then
    the single missingness rule the rest of the module implements, and a
    fighter whose debut date disagrees with our own fights table is exactly a
    fighter whose derived pre-UFC window should not be trusted.

    The condition is stated positively (`notna` on both dates AND ordered)
    rather than as `~(debut > first_ufc)`. The negated form let a MISSING date
    through: `NaT > date` is False, so `~False` kept the fighter, whose
    `days_since_pro_debut` then came out NaN while `external_missing` said
    False -- breaking this module's one invariant, that a present flag means
    present values. A fighter with no usable debut date takes the ordinary
    missing-fighter path like any other.

    `attach` and `state_for` apply this themselves rather than trusting their
    caller, so passing a raw table in cannot bypass the guard. It is
    idempotent, and the table is a few thousand rows.
    """
    debut, first_ufc = table["pro_debut_date"], table["first_ufc_date"]
    sane = debut.notna() & first_ufc.notna() & (debut <= first_ufc)
    return table[sane].reset_index(drop=True)


def load_table(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """The committed derived table, source-error fighters already dropped."""
    return drop_source_errors(pd.read_parquet(path))


def attach(side: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """One corner's per-fight frame plus every key the `external` block reads.

    `side` must carry `fighter_id` and `date`; row order and length are
    preserved. The merge is a left join on `fighter_id` alone, validated
    many-to-one so a duplicated fighter in the derived table would raise
    rather than quietly multiply rows.
    """
    table = load_table() if table is None else drop_source_errors(table)
    columns = ["fighter_id", "pro_debut_date", NATIONALITY] + list(NUMERIC_STATE_KEYS)
    merged = side.merge(
        table[columns], on="fighter_id", how="left",
        validate="many_to_one", indicator="_external",
    )
    missing = (merged["_external"] != "both").to_numpy()
    merged["days_since_pro_debut"] = (
        merged["date"] - merged["pro_debut_date"]
    ).dt.days
    for column in NUMERIC_STATE_KEYS + ("days_since_pro_debut",):
        merged.loc[missing, column] = np.nan
    merged.loc[missing, NATIONALITY] = None
    merged["external_missing"] = missing
    return merged.drop(columns=["_external", "pro_debut_date"])


def state_for(fighter_id, as_of: pd.Timestamp,
              table: pd.DataFrame | None = None) -> dict:
    """The same values `attach` produces, for one fighter at serving time.

    Returns the missing state (all NaN, flag set) for a fighter the snapshot
    does not carry -- which is every fighter who debuted after its coverage
    ends, i.e. a real and growing share of the fights actually served.
    """
    table = load_table() if table is None else drop_source_errors(table)
    row = table[table["fighter_id"] == fighter_id]
    if len(row) != 1:
        return dict(_MISSING_STATE)
    row = row.iloc[0]
    debut = row["pro_debut_date"]
    state = {key: row[key] for key in NUMERIC_STATE_KEYS}
    state["days_since_pro_debut"] = (
        float((pd.Timestamp(as_of) - debut).days) if pd.notna(debut) else np.nan
    )
    state["external_missing"] = False
    state[NATIONALITY] = row[NATIONALITY] if pd.notna(row[NATIONALITY]) else None
    return state


def fight_context(state_a, state_b) -> dict:
    """The block's fight-level columns from the two corners' external state.

    Works on scalars (serving) and on whole columns (training), like
    `mma.serving.feature_row` itself. `same_country` is False whenever either
    nationality is unknown; those rows are exactly the ones `external_missing`
    marks, so the two columns are read together rather than `same_country`
    pretending to distinguish "different" from "unknown".
    """
    missing_a, missing_b = state_a["external_missing"], state_b["external_missing"]
    country_a, country_b = state_a[NATIONALITY], state_b[NATIONALITY]
    if isinstance(missing_a, pd.Series) or isinstance(missing_b, pd.Series):
        def flags(value) -> np.ndarray:
            return pd.Series(value).fillna(True).to_numpy(dtype=bool)

        def countries(value) -> pd.Series:
            return pd.Series(value).astype("string").reset_index(drop=True)

        left, right = countries(country_a), countries(country_b)
        # `left.eq(right)` is NA-valued where either side is unknown, and
        # `False & NA` is False in pandas' nullable boolean logic, so the
        # notna() guards do the work and fillna only mops up NA-vs-NA.
        same = (left.notna() & right.notna() & left.eq(right)).fillna(False)
        return {
            "external_missing": flags(missing_a) | flags(missing_b),
            "same_country": same.to_numpy(dtype=bool),
        }
    return {
        "external_missing": bool(missing_a) or bool(missing_b),
        "same_country": bool(
            country_a is not None
            and country_b is not None
            and pd.notna(country_a)
            and pd.notna(country_b)
            and country_a == country_b
        ),
    }
