"""Short-notice replacements and missed weight, joined per (fight, corner).

`data/external/fight_notice.parquet` (built by `scripts/build_external.py`
from the MIT-licensed `ehan03/jds-mma-data` snapshot's Bet MMA tables) holds
one row per CORNER of every fight the source has actually looked at. This
module is the only place that table is read.

**Why membership is the whole design.** Bet MMA publishes two lists:
`late_replacements.csv` (fighter, bout, days of notice) and
`missed_weights.csv` (fighter, bout, weigh-in weight). Both list only the
fighters the thing HAPPENED to. Taken on their own, "no row" is ambiguous
between "trained a full camp" and "nobody recorded it" -- and a feature that
cannot tell those apart is not a feature, it is a coverage flag wearing one.

The disambiguation is the bout universe. Bet MMA covers its own event list
bout by bout (2013-04-20 .. 2024-12-14 in the `ec77f537` snapshot), and
`bout_mapping.csv` says which of those bouts are ours. Inside that set,
absence of a replacement row is an OBSERVATION; outside it, absence is
ignorance. So the derivation emits a row for both corners of every bout it
can see and nothing at all for the rest, which makes membership of this
table the three-state boundary:

    row present, `notice_days` NaN   -> observed: full camp
    row present, `notice_days` = n   -> observed: replaced on n days' notice
    no row                           -> unknown

`missed_weight` is its own boolean column rather than "the overage is not
null", because three of the source's 223 recorded misses have no believable
magnitude (two catchweight bouts with no limit, one 273 lb bantamweight
weigh-in). Those keep the flag and take 0 pounds over, which understates a
magnitude nobody knows rather than inventing one.

**Missingness is fight-level and symmetric**, for the reason the `external`
block was corrected for (SP2 Task 11): a per-corner "we have no data on this
fighter" flag is a look-ahead feature whenever source membership tracks how
a career turned out. Here the derivation guarantees both corners or neither,
so a block built on this carries only a fight-level `notice_unknown`, which
cannot say which corner wins.
`tests/test_notice.py::test_an_unknown_fight_carries_exactly_one_state_tuple`
asserts that property on the real table, one level below the feature block.

**Point-in-time.** A replacement is booked and a weigh-in happens before the
bout, so every column here is known on fight morning. Nothing is accumulated
across fights: each row describes its own bout only.

**Serving.** At prediction time these facts cannot be looked up at all -- a
future bout has no row in any source -- so they would come from the event's
Wikipedia Background prose (`mma.wiki_cards.parse_background`) via
`from_observation`, and `unknown_state()` whenever the page says nothing,
which is the common case.

**Status: the `notice` feature block was measured and REJECTED** (SP2 Task 12);
this module and the table it reads are kept because they are correct and
reusable, not because anything currently consumes them. The block's torch
pooled log-loss was 0.6472 against an incumbent 0.6476 -- a seventh of the
0.003 bar -- and the reason is coverage: the source's bout list ends
2024-12-14, so every 2025 and 2026 row is unknown. See
`mma.feature_blocks` (the commented-out registration) and the SP2 plan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "external" / "fight_notice.parquet"

# `notice_shortfall_days` is a hinge, not the raw notice period: max(0, HINGE -
# notice_days), so a full camp is 0 and a one-day replacement is HINGE. The raw
# `notice_days` cannot be a differential at all, because a fighter who was not a
# late replacement has no notice period -- only an unbounded one -- and any
# sentinel for "full camp" would be invented rather than measured. Hinging at
# the same 30 days as `short_notice_30` keeps the numeric column and the flag
# describing one boundary instead of two.
NOTICE_HINGE_DAYS = 30.0
SHORT_NOTICE_DAYS = (7.0, 30.0)

NUMERIC_STATE_KEYS = ("notice_shortfall_days", "missed_weight_over_lbs")
BOOLEAN_STATE_KEYS = ("short_notice_7", "short_notice_30", "missed_weight")
STATE_KEYS = NUMERIC_STATE_KEYS + BOOLEAN_STATE_KEYS

# What a corner the source has not seen looks like. Numerics are NaN (not 0 --
# "no data" is not "full camp") and the flags are False, which is also what an
# observed clean corner has; the numerics are what separates them, plus the
# fight-level flag.
_UNKNOWN_STATE = (
    {key: np.nan for key in NUMERIC_STATE_KEYS}
    | {key: False for key in BOOLEAN_STATE_KEYS}
    | {"notice_missing": True}
)

def load_table(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """The committed derived table: one row per corner of every seen fight."""
    return pd.read_parquet(path)


def _derive_state(frame: pd.DataFrame, missing: np.ndarray) -> pd.DataFrame:
    """Turn raw `notice_days` / `missed_weight_over_lbs` into the state keys."""
    days = pd.to_numeric(frame["notice_days"], errors="coerce")
    over = pd.to_numeric(frame["missed_weight_over_lbs"], errors="coerce")
    missed = frame["missed_weight"].fillna(False).astype(bool)
    # Within an observed fight a fighter with no replacement row had a full
    # camp, so the hinge is 0 rather than NaN; `fillna` after the hinge keeps
    # the unknown rows NaN via the mask below.
    frame["notice_shortfall_days"] = (
        (NOTICE_HINGE_DAYS - days).clip(lower=0.0).fillna(0.0)
    )
    frame["missed_weight_over_lbs"] = over.fillna(0.0)
    frame["short_notice_7"] = (days <= SHORT_NOTICE_DAYS[0]).fillna(False)
    frame["short_notice_30"] = (days <= SHORT_NOTICE_DAYS[1]).fillna(False)
    frame["missed_weight"] = missed
    for key in NUMERIC_STATE_KEYS:
        frame.loc[missing, key] = np.nan
    for key in BOOLEAN_STATE_KEYS:
        frame[key] = frame[key].astype(bool)
        frame.loc[missing, key] = False
    frame["notice_missing"] = missing
    return frame.drop(columns=["notice_days"])


def attach(side: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """One corner's per-fight frame plus every key the `notice` block reads.

    `side` must carry `fight_id` and `fighter_id`; row order and length are
    preserved. The join is on the PAIR, validated many-to-one, so a corner
    that was not in the bout the source recorded reads as unknown rather than
    borrowing the other corner's camp.
    """
    table = load_table() if table is None else table
    columns = ["fight_id", "fighter_id", "notice_days", "missed_weight",
               "missed_weight_over_lbs"]
    source = table[columns].copy()
    merged = side.merge(
        source, on=["fight_id", "fighter_id"], how="left",
        validate="many_to_one", indicator="_notice",
    )
    missing = (merged["_notice"] != "both").to_numpy()
    return _derive_state(merged.drop(columns=["_notice"]), missing)


def state_for(fight_id, fighter_id, table: pd.DataFrame | None = None) -> dict:
    """The same values `attach` produces, for one corner of one fight.

    Serving does NOT go through this -- a future fight has no id in the
    source -- but the parity test does, which is what keeps the training row
    and the served row comparable given the same known facts.
    """
    table = load_table() if table is None else table
    row = table[
        (table["fight_id"] == fight_id) & (table["fighter_id"] == fighter_id)
    ]
    if len(row) != 1:
        return dict(_UNKNOWN_STATE)
    return from_observation(
        notice_days=row.iloc[0]["notice_days"],
        missed_weight=bool(row.iloc[0]["missed_weight"]),
        missed_weight_over_lbs=row.iloc[0]["missed_weight_over_lbs"],
    )


def from_observation(notice_days=np.nan, missed_weight: bool = False,
                     missed_weight_over_lbs=np.nan) -> dict:
    """The state for a corner whose camp IS known, from the raw facts.

    This is the serving entry point: a caller turns a Wikipedia Background
    note (`mma.wiki_cards.parse_background`) into these arguments -- notice
    days if the prose stated a number, whether a weigh-in was missed, and by
    how much if that was stated -- and hands the result to `build_matchup`.
    A corner nothing is known about takes `unknown_state()` instead; the
    difference between the two is the whole point of this module.
    """
    days = float(notice_days) if pd.notna(notice_days) else np.nan
    over = float(missed_weight_over_lbs) if pd.notna(missed_weight_over_lbs) else np.nan
    return {
        "notice_shortfall_days": (
            max(0.0, NOTICE_HINGE_DAYS - days) if pd.notna(days) else 0.0
        ),
        "missed_weight_over_lbs": over if pd.notna(over) else 0.0,
        "short_notice_7": bool(pd.notna(days) and days <= SHORT_NOTICE_DAYS[0]),
        "short_notice_30": bool(pd.notna(days) and days <= SHORT_NOTICE_DAYS[1]),
        "missed_weight": bool(missed_weight),
        "notice_missing": False,
    }


def unknown_state() -> dict:
    """The state for a corner the source (or the parser) has not seen."""
    return dict(_UNKNOWN_STATE)


def fight_context(state_a, state_b) -> dict:
    """The block's one fight-level column, from the two corners' state.

    Works on scalars (serving) and on whole columns (training). The OR is
    what makes the flag symmetric in the corners: it says the source has not
    seen this fight, never which corner it has not seen.
    """
    missing_a, missing_b = state_a["notice_missing"], state_b["notice_missing"]
    if isinstance(missing_a, pd.Series) or isinstance(missing_b, pd.Series):
        def flags(value) -> np.ndarray:
            return pd.Series(value).fillna(True).to_numpy(dtype=bool)

        return {"notice_unknown": flags(missing_a) | flags(missing_b)}
    return {"notice_unknown": bool(missing_a) or bool(missing_b)}
