"""Official UFC divisional rankings, joined as of the week before each fight.

`data/external/rankings.parquet` (built by `scripts/build_rankings.py` from the
CC0 Kaggle dataset `jerzyszocik/ufc-rankings-history`) holds one row per
(fighter, division, publication date) with `rank`, champion = 0. This module is
the only place it is read.

**Point-in-time is the whole safety argument, and it is a join rule.** Unlike
the pre-UFC aggregates next door, a ranking is not a fixed fact about a
fighter: it moves every week, and it moves *because of fight results*. Reading
the ranking published on or after a fight's date would therefore leak that
fight's own outcome. Every lookup here takes the most recent publication
**strictly before** the fight date -- `merge_asof(..., allow_exact_matches=
False)` for the training table and the same rule spelled out for one fighter in
`state_for`. `tests/test_rankings.py` pins it with a fixture whose ranking
changes on the fight day itself.

**Matching is by name, once, in the derivation.** The source has no ids, so
`scripts/build_rankings.py` resolves names through the prospective pipeline's
never-guess matcher and drops what it cannot resolve; by the time this module
sees the table it is keyed by ufcstats `fighter_id` like everything else. The
cost is that an unresolved name reads as *unranked* rather than as *unknown* --
see `data/external/RANKINGS.md`.

**Unranked is not missing.** Most fighters in most weeks are outside the top
15, and that is an observation, not a coverage gap: the source publishes the
whole list every week from 2013 on. So `is_ranked` is False, `rank` is NaN,
and the fight-level `rank_missing` says the difference is not computable for
this pair -- it does not say the data is absent. The one genuine coverage edge
is a fight before the first publication (2013-02-04) or in a division the UFC
does not rank, and those look the same as unranked, which is the correct
reading: there was no ranking to have.

**Status: the `rankings` feature block was measured and REJECTED** (SP2 Task
12). All three variants were worse on torch than the incumbent, because
`rank_diff` reaches only 13.8% of rows and correlates -0.40 with `elo_diff`
where it does. The signal itself is real -- the better-ranked corner wins
0.563 of the both-ranked fights against `elo_diff`'s 0.533 on the same rows --
so this module and its table are kept for a later block that can widen the
coverage (a divisional rank for everyone, say, rather than a top-15 flag).
See the commented-out registration in `mma.feature_blocks`.

**The 2026 regime change.** On 2026-06-20 the UFC replaced its media-panel
rankings with an Elo-derived list, so a rank means a different thing either
side of that date. `RANKING_REGIME_CHANGE` marks it as a fight-level flag
rather than being silently averaged across.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "external" / "rankings.parquet"

# The date the UFC's rankings became Elo-derived rather than media-panel voted.
# A rank either side of it is not the same measurement, so the fight-level flag
# lets the model treat the two regimes separately instead of pooling them.
RANKING_REGIME_CHANGE = pd.Timestamp("2026-06-20")
CHAMPION_RANK = 0

NUMERIC_STATE_KEYS = ("rank",)
BOOLEAN_STATE_KEYS = ("is_champion", "is_ranked")
STATE_KEYS = NUMERIC_STATE_KEYS + BOOLEAN_STATE_KEYS

_UNRANKED_STATE = {"rank": np.nan, "is_champion": False, "is_ranked": False}


def load_table(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """The committed rankings table, sorted for the as-of join."""
    table = pd.read_parquet(path)
    table["date"] = pd.to_datetime(table["date"])
    return table.sort_values("date", kind="stable").reset_index(drop=True)


def attach(side: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """One corner's per-fight frame plus every key the `rankings` block reads.

    `side` must carry `fighter_id`, `date` and `weight_class`; row order and
    length are preserved. The join is `merge_asof` on date, by (fighter,
    division), with `allow_exact_matches=False` -- a ranking published ON the
    fight date already knows how the week's earlier fights went, so it is not
    admissible even though it is not literally after the fight.
    """
    table = load_table() if table is None else table
    ordered = side.reset_index(drop=True)
    ordered["_row"] = np.arange(len(ordered))
    left = ordered.sort_values("date", kind="stable")
    left["weight_class"] = left["weight_class"].astype("string")
    right = table.copy()
    right["weight_class"] = right["weight_class"].astype("string")
    right["fighter_id"] = right["fighter_id"].astype("string")
    left["fighter_id"] = left["fighter_id"].astype("string")
    merged = pd.merge_asof(
        left, right[["fighter_id", "weight_class", "date", "rank"]],
        on="date", by=["fighter_id", "weight_class"],
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("_row", kind="stable").drop(columns=["_row"])
    merged = merged.reset_index(drop=True)
    rank = pd.to_numeric(merged["rank"], errors="coerce")
    merged["rank"] = rank
    merged["is_ranked"] = rank.notna()
    merged["is_champion"] = (rank == CHAMPION_RANK).fillna(False)
    return merged


def state_for(fighter_id, weight_class, as_of,
              table: pd.DataFrame | None = None) -> dict:
    """The same values `attach` produces, for one fighter at serving time.

    Strictly-before is applied here too, so the row served for a future fight
    uses the ranking that was public the week the card was built.
    """
    table = load_table() if table is None else table
    as_of = pd.Timestamp(as_of)
    rows = table[
        (table["fighter_id"] == fighter_id)
        & (table["weight_class"] == weight_class)
        & (table["date"] < as_of)
    ]
    if rows.empty:
        return dict(_UNRANKED_STATE)
    rank = float(rows.sort_values("date").iloc[-1]["rank"])
    return {
        "rank": rank,
        "is_champion": rank == CHAMPION_RANK,
        "is_ranked": True,
    }


def fight_context(state_a, state_b, date) -> dict:
    """The block's fight-level columns, from the two corners' state.

    Works on scalars (serving) and on whole columns (training).
    `rank_missing` is the symmetric either-corner OR, so it says the rank
    DIFFERENCE is not computable for this pair, never which corner is the
    unranked one -- the same shape as `external_missing`, for the same reason.
    """
    ranked_a, ranked_b = state_a["is_ranked"], state_b["is_ranked"]
    if isinstance(ranked_a, pd.Series) or isinstance(ranked_b, pd.Series):
        def flags(value) -> np.ndarray:
            return pd.Series(value).fillna(False).to_numpy(dtype=bool)

        return {
            "rank_missing": ~(flags(ranked_a) & flags(ranked_b)),
            "ranking_regime_post_2026_06": (
                pd.to_datetime(pd.Series(date).reset_index(drop=True))
                >= RANKING_REGIME_CHANGE
            ).to_numpy(dtype=bool),
        }
    return {
        "rank_missing": not (bool(ranked_a) and bool(ranked_b)),
        "ranking_regime_post_2026_06": pd.Timestamp(date) >= RANKING_REGIME_CHANGE,
    }
