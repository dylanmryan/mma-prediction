"""The fight's setting: who is refereeing, where it is, and who gets paid extra.

Three facts that describe the BOUT rather than either fighter's record, and
that the base block has no channel for at all.

**Referee tendency.** Referees differ in how early they stop a fight, and a
fight's method is downstream of that. `referee_history` walks the fights table
chronologically and gives each fight the referee's finish and decision rates
over their PRIOR fights only -- a referee's first bout gets NaN, not the rate
they would go on to have. The pass is a mirror of `mma.history`: snapshot
first, then fold the fight in.

**Home advantage.** `home_country` is True when a corner's nationality equals
the country the event is being held in. Nationality comes from the `external`
block's snapshot (`mma.external`), so this column exists only when that block
is enabled; the event country is the last comma-separated component of
`fights.location`, normalised. Both sides are frequently unknown -- nationality
for the ~20% of corners the snapshot never mapped, location for ~23% of fights
-- and `home_country_unknown` is the fight-level flag that says so. It is
deliberately FIGHT-level and symmetric: a per-corner "we don't know where this
one is from" pair would be the `external` block's measured leak in a second
channel, because snapshot membership tracks how long a fighter's UFC career
turned out to be. See the SP2 plan's amended missingness rule.

**Bonus record.** `bonus_rate` is the share of a fighter's prior bouts that
they won AND that were awarded a post-fight bonus. `bonuses.parquet` records
bonuses per FIGHT, not per fighter, so "Performance of the Night" cannot be
attributed to a corner from the data alone; crediting the winner is the honest
approximation, and it makes "Fight of the Night" -- which both corners share --
count for the winner only. That is a definition, not a measurement, and it is
stated here rather than buried in the accumulator.

**Serving asymmetry, stated up front.** Wikipedia fight cards do not carry
referees, so at prediction time the three referee columns are ALWAYS missing
while in training they are present on 98% of rows. A feature that is always
there in training and never there in serving does not help; it just moves the
served rows off the training distribution. This block therefore ships only if
it clears the pre-registered bar AND a re-run with the referee columns held out
of the model does not lose more than sigma_seed against the incumbent -- and if
only the held-out variant clears, that is the variant that ships (columns in
the table, excluded from both model matrices, as `external`'s flags are).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BONUSES_PATH = ROOT / "data" / "processed" / "bonuses.parquet"

# The block's fight-level columns, in emission order.
FIGHT_LEVEL = (
    "referee_finish_rate",
    "referee_decision_rate",
    "referee_missing",
    "home_country_unknown",
)
# The per-corner state key the block reads. `bonus_rate` is not here: it is a
# `mma.history` accumulator field like the other career rates.
STATE_KEYS = ("home_country",)

_FINISH_METHODS = ("ko_tko", "submission")
_DECISION_METHODS = ("decision",)

# Nationalities and event locations come from different sources with different
# vocabularies. Only two collisions actually occur in the data, and both are
# spelled out rather than guessed at: the snapshot writes both "United States"
# and "USA", and it names the four UK home nations while an event location
# only ever says "United Kingdom". Everything else matches on a casefold.
_COUNTRY_ALIASES = {
    "usa": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "england": "united kingdom",
    "scotland": "united kingdom",
    "wales": "united kingdom",
    "northern ireland": "united kingdom",
    "great britain": "united kingdom",
}


def normalise_country(value) -> str | None:
    """A country name reduced to the form the two sources can be compared in."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    folded = text.casefold()
    return _COUNTRY_ALIASES.get(folded, folded)


def event_country(location) -> str | None:
    """The country a fight took place in, from `fights.location`.

    ufcstats writes locations as "City, State, Country" or "City, Country",
    so the country is the last comma-separated component. A location with no
    comma is a city on its own and gives nothing usable -- returning it as a
    country would silently compare a city against a nationality.
    """
    if location is None or (not isinstance(location, str) and pd.isna(location)):
        return None
    parts = [part.strip() for part in str(location).split(",") if part.strip()]
    if len(parts) < 2:
        return None
    return normalise_country(parts[-1])


def home_country(nationality, country) -> bool:
    """True when a corner is fighting in their own country.

    False when either side is unknown -- which is why the fight-level
    `home_country_unknown` flag has to be read alongside it. A boolean cannot
    carry "we don't know", so the flag is where that fact lives.
    """
    left, right = normalise_country(nationality), normalise_country(country)
    return bool(left is not None and right is not None and left == right)


class _RefereeCounts:
    """One referee's running (fights, finishes, decisions)."""

    __slots__ = ("fights", "finishes", "decisions")

    def __init__(self) -> None:
        self.fights = 0
        self.finishes = 0
        self.decisions = 0

    def rates(self) -> tuple[float, float]:
        if not self.fights:
            return (np.nan, np.nan)
        return (self.finishes / self.fights, self.decisions / self.fights)

    def add(self, method) -> None:
        self.fights += 1
        if method in _FINISH_METHODS:
            self.finishes += 1
        elif method in _DECISION_METHODS:
            self.decisions += 1


def _counts(fights: pd.DataFrame) -> dict[str, _RefereeCounts]:
    counts: dict[str, _RefereeCounts] = {}
    for fight in fights.itertuples(index=False):
        referee = fight.referee
        if referee is None or pd.isna(referee):
            continue
        counts.setdefault(str(referee), _RefereeCounts()).add(
            fight.method if pd.notna(fight.method) else None
        )
    return counts


def referee_history(fights: pd.DataFrame) -> pd.DataFrame:
    """Per-fight referee rates over that referee's PRIOR fights only.

    Returns `fight_id` plus the block's three referee columns, one row per
    input fight and in the input's own row order. Point-in-time by
    construction: the rates are read before the fight is folded in, so
    truncating the fights table cannot change a surviving row.
    """
    counts: dict[str, _RefereeCounts] = {}
    rows = []
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for fight in ordered.itertuples(index=False):
        referee = fight.referee
        missing = referee is None or pd.isna(referee)
        if missing:
            finish_rate, decision_rate = np.nan, np.nan
        else:
            state = counts.setdefault(str(referee), _RefereeCounts())
            finish_rate, decision_rate = state.rates()
            state.add(fight.method if pd.notna(fight.method) else None)
        rows.append({
            "fight_id": fight.fight_id,
            "referee_finish_rate": finish_rate,
            "referee_decision_rate": decision_rate,
            "referee_missing": bool(missing),
        })
    history = pd.DataFrame(rows, columns=["fight_id"] + list(FIGHT_LEVEL[:3]))
    history["fight_id"] = history["fight_id"].astype("string")
    return history


def referee_rates(fights: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Every referee's CAREER rates as of the end of `fights`, for serving.

    The serving counterpart of `referee_history`: a future fight's referee has
    officiated everything in the table, so their rate is the whole-table one.
    Pass the fights known as of the prediction date -- that is what makes the
    served value equal the value the same fight would train on.
    """
    return {name: state.rates() for name, state in _counts(fights).items()}


def referee_context(referee, rates: dict | None) -> dict:
    """The three referee columns for one served fight.

    `referee` is None whenever the card does not name one -- which, for a
    Wikipedia-sourced future card, is always. That path is the default, and it
    produces exactly what training produces for a fight with no recorded
    referee: two NaNs and the flag.
    """
    known = referee is not None and not (
        not isinstance(referee, str) and pd.isna(referee)
    )
    if not known:
        return {"referee_finish_rate": np.nan, "referee_decision_rate": np.nan,
                "referee_missing": True}
    finish_rate, decision_rate = (rates or {}).get(str(referee), (np.nan, np.nan))
    return {"referee_finish_rate": finish_rate,
            "referee_decision_rate": decision_rate,
            "referee_missing": False}


def home_context(nationality_a, nationality_b, country) -> dict:
    """The fight-level home-advantage flag, for scalars or whole columns.

    True when the pair `home_country_a`/`home_country_b` cannot mean what it
    says: either nationality unknown, or the event's country unknown. Both
    inputs are symmetric in the corners, which is the property that keeps this
    flag from re-opening the `external` block's per-corner coverage leak.
    """
    if any(isinstance(v, pd.Series) for v in (nationality_a, nationality_b, country)):
        def known(value) -> np.ndarray:
            return pd.Series(value).astype("string").reset_index(drop=True).notna().to_numpy()

        unknown = ~(known(nationality_a) & known(nationality_b) & known(country))
        return {"home_country_unknown": unknown}
    return {"home_country_unknown": bool(
        normalise_country(nationality_a) is None
        or normalise_country(nationality_b) is None
        or normalise_country(country) is None
    )}


@lru_cache(maxsize=1)
def _committed_bonus_fights() -> frozenset:
    """The fight ids in the committed bonuses table (cached, read once)."""
    if not BONUSES_PATH.exists():
        return frozenset()
    return frozenset(pd.read_parquet(BONUSES_PATH)["fight_id"].astype(str))


def bonus_fights(table: pd.DataFrame | None = None) -> frozenset:
    """Fight ids that carried a post-fight bonus.

    Defaults to the committed `data/processed/bonuses.parquet`, the way
    `mma.external.load_table` defaults to its own committed table, so the
    history accumulator and the snapshot builder pick it up without every
    caller having to thread it through. A table can be passed for tests.
    """
    if table is None:
        return _committed_bonus_fights()
    return frozenset(table["fight_id"].astype(str))
