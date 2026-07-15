"""Pure helpers for converting betting odds into devigged implied probabilities.

EVALUATION-ONLY: nothing in this module (or its callers) may become a model
FEATURE -- betting odds are a comparator for the "does the model beat the
market?" benchmark, never an input to prediction. No network access here;
`scripts/build_odds_benchmark.py` does the kagglehub download and calls into
these pure functions.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Iterable, Mapping

# ufcstats URLs end in a 16-character lowercase hex id, e.g.
# http://ufcstats.com/fight-details/d215c4e6dc1346ae
_FIGHT_ID_RE = re.compile(r"([0-9a-f]{16})/?$")


def decimal_to_implied(d: float) -> float:
    """Decimal odds -> raw (vig-included) implied probability: 1/d."""
    return 1.0 / d


def american_to_implied(a: float) -> float:
    """American odds -> raw (vig-included) implied probability.

    Positive (underdog) odds: 100 / (a + 100).
    Negative (favorite) odds: -a / (-a + 100).
    American odds can never be 0 (there is no "even money" American line).
    """
    if a > 0:
        return 100.0 / (a + 100.0)
    if a < 0:
        return -a / (-a + 100.0)
    raise ValueError("American odds cannot be 0")


def devig_pair(p1_raw: float, p2_raw: float) -> tuple[float, float]:
    """Proportional ("multiplicative") devig: normalize a two-way market to sum to 1.

    Bookmaker lines overround (p1_raw + p2_raw > 1) to bake in their margin;
    this splits the overround proportionally across both sides rather than
    subtracting it from one side, which is the standard "multiplicative"
    devig method.
    """
    total = p1_raw + p2_raw
    return p1_raw / total, p2_raw / total


def extract_fight_id(fight_url: str | None) -> str | None:
    """Pull the 16-hex ufcstats fight id from a fight-details URL.

    Returns None if `fight_url` is falsy or doesn't end in a 16-character
    hex id (the exact scheme used by `fights.parquet`'s `fight_id` column).
    """
    if not fight_url:
        return None
    match = _FIGHT_ID_RE.search(str(fight_url).strip())
    return match.group(1) if match else None


def _valid_odds(rows: Iterable[Mapping], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key) if hasattr(row, "get") else row[key]
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        values.append(value)
    return values


def consensus_odds(rows_for_one_fight: Iterable[Mapping]) -> tuple[float, float]:
    """Median decimal odds across books per side, then devigged.

    `rows_for_one_fight` is an iterable of dict-likes (or objects supporting
    `.get`/`[]`) with `"odds_1"` and `"odds_2"` decimal-odds keys -- all rows
    for a single fight, e.g. one row per bookmaker. NaN/None entries are
    ignored. The returned pair mirrors the input's side-1/side-2 order; it is
    the caller's job (see `scripts/build_odds_benchmark.py`) to map that onto
    the project's fighter_a/fighter_b convention.

    Raises ValueError if either side has no valid odds to take a median of.
    """
    rows = list(rows_for_one_fight)
    side1 = _valid_odds(rows, "odds_1")
    side2 = _valid_odds(rows, "odds_2")
    if not side1 or not side2:
        raise ValueError("no valid odds rows for this fight")
    p1_raw = decimal_to_implied(statistics.median(side1))
    p2_raw = decimal_to_implied(statistics.median(side2))
    return devig_pair(p1_raw, p2_raw)
