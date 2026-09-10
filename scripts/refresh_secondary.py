"""Adapt the daily ufcstats scrape and merge it on top of the Kaggle primary.

The primary source is a Kaggle mirror of ufcstats that is rebuilt whenever
its maintainer gets round to it; as of 2026-09-09 its newest fight is dated
2026-08-08, so a month of results -- and the fighter form they carry -- is
missing from the deployed model and from the prospective track record's
grading. `Greco1899/scrape_ufc_stats` (GPL-3.0) scrapes the same site on a
daily schedule and publishes plain CSVs, so it can close that gap.

Used as a DATA source, not as code: this script fetches the published CSVs
over HTTPS; no GPL-licensed code is vendored into this repo.

The adaptation deliberately goes through `mma.dataset.build_fighters` /
`build_fights` / `build_fight_stats` / `build_round_stats` rather than
re-deriving our columns: the upstream CSVs are reshaped into the Kaggle
`master.csv` / `fighter.csv` / `round.csv` column names first, so the winner
codes, method mapping, unit conversions and `duration_sec` semantics are the
tested ones by construction rather than by a second implementation that can
drift.

THIS SCRIPT NEVER WRITES. `scripts/make_dataset.py` owns the one write path:
it builds the primary tables from `data/raw/`, calls `merge_into` here as a
standing stage of every rebuild, and runs its integrity checks on the union.
That is what makes the arrangement safe to leave switched on -- a rebuild can
never drop what a previous rebuild added, so `make_dataset.py`'s regression
guard keeps its meaning, and there is no special state to reason about.
Running this script directly reports what the stage would contribute.

Primary always wins on any overlapping fight: it is the reconciled, tested
source, so a fight both sources have keeps the primary's row even where they
disagree. The secondary can only ever append.

Usage:
    python scripts/refresh_secondary.py     # report what the merge would add
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple

import pandas as pd

from mma.dataset import (
    _CORE_STAT_SUFFIXES,
    _TARGET_STAT_SUFFIXES,
    build_fight_stats,
    build_fighters,
    build_fights,
    build_round_stats,
)
from mma.labels import is_title_fight
from mma.parsing import parse_landed_attempted, parse_mmss_seconds, parse_reach_inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_sources import reconcile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SOURCE_URL = "https://github.com/Greco1899/scrape_ufc_stats"
SOURCE_LICENCE = "GPL-3.0"
RAW_BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
FILES = {
    "events": "ufc_event_details.csv",
    "results": "ufc_fight_results.csv",
    "stats": "ufc_fight_stats.csv",
    "fighter_details": "ufc_fighter_details.csv",
    "fighter_tott": "ufc_fighter_tott.csv",
}
# ufcstats ids are the 16-hex tail of every URL the scrape stores.
_UFCSTATS_ID = r"([0-9a-f]{16})$"

# Winner agreement below this on the fights BOTH sources have means the
# adaptation is wrong or the source disagrees materially; either way it is
# not safe to append its rows to a table our models train on.
AGREEMENT_FLOOR = 0.99

PRIMARY = "primary"
SECONDARY = "secondary"
PROVENANCE_COLUMNS = ["table", "row_id", "source"]


class UnknownFighterError(ValueError):
    """A secondary fight names a fighter absent from `fighters.parquet`.

    We cannot build features for a fighter with no biographical row and no
    history, so such a fight is rejected rather than merged with holes.
    """


class SecondaryRefused(RuntimeError):
    """The secondary source is not in a state we are willing to merge."""


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def fetch_csv(filename: str, base: str | None = None, timeout: int = 60) -> pd.DataFrame:
    """Plain HTTPS GET of one published CSV. No auth, no credentials."""
    url = (base or RAW_BASE) + filename
    request = urllib.request.Request(url, headers={"User-Agent": "mma-prediction/refresh-secondary"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https host)
        payload = response.read()
    return pd.read_csv(io.BytesIO(payload))


def fetch_source(base: str | None = None) -> dict[str, pd.DataFrame]:
    """Fetch every CSV the adaptation reads, keyed by role."""
    return {role: fetch_csv(name, base=base) for role, name in FILES.items()}


# ---------------------------------------------------------------------------
# schema adaptation
# ---------------------------------------------------------------------------


def _ids_from_urls(urls: pd.Series) -> pd.Series:
    return urls.astype("string").str.strip().str.extract(_UFCSTATS_ID, expand=False)


def _fighter_id_frame(fighter_tott: pd.DataFrame, fighter_details: pd.DataFrame) -> pd.DataFrame:
    """(name, fighter_id) pairs from the source's own two fighter files.

    Both files are read because they disagree on a handful of spellings
    ("Zach Reese" / "Zachary Reese", "Waldo Cortes Acosta" /
    "Waldo Cortes-Acosta"); `fighter_tott` is listed first so it wins any
    tie on a shared id.
    """
    tott = pd.DataFrame(
        {
            "name": fighter_tott["FIGHTER"].astype("string").str.strip(),
            "fighter_id": _ids_from_urls(fighter_tott["URL"]),
        }
    )
    details = pd.DataFrame(
        {
            "name": (
                fighter_details["FIRST"].fillna("").astype(str).str.strip()
                + " "
                + fighter_details["LAST"].fillna("").astype(str).str.strip()
            ).str.strip(),
            "fighter_id": _ids_from_urls(fighter_details["URL"]),
        }
    )
    pairs = pd.concat([tott, details], ignore_index=True).dropna()
    return pairs[pairs["name"] != ""].drop_duplicates()


def build_name_index(fighter_tott: pd.DataFrame, fighter_details: pd.DataFrame) -> pd.Series:
    """Fighter name -> ufcstats id, from the source's own two fighter files.

    The scrape's fight tables key on the bout string ("A vs. B"), not on
    fighter URLs, so resolving a corner to an id needs a name lookup. It is
    a WITHIN-source lookup against ufcstats' own spellings, not cross-source
    name matching. Any name that maps to more than one id is dropped
    entirely: guessing between two real fighters would attach one man's
    history to another.
    """
    pairs = _fighter_id_frame(fighter_tott, fighter_details)
    distinct = pairs.groupby("name")["fighter_id"].nunique()
    unambiguous = pairs[pairs["name"].isin(distinct[distinct == 1].index)]
    return unambiguous.drop_duplicates("name").set_index("name")["fighter_id"]


def build_secondary_fighters(
    fighter_tott: pd.DataFrame, fighter_details: pd.DataFrame
) -> pd.DataFrame:
    """The source's fighter files -> our `fighters.parquet` schema.

    This is what closes the debutant gap: a fighter making his UFC debut
    after the primary mirror's last rebuild has no row in `fighters.parquet`,
    so his fight would be rejected for having no biographical row to build
    features from. The scrape publishes the same ufcstats fighter pages.

    Ids that appear only in the details file get a name and nothing else --
    that file carries no biography. Missing height/reach/stance/DOB stay
    missing (ufcstats genuinely leaves them blank for most debutants); they
    are never invented.
    """
    tott = pd.DataFrame(
        {
            "fighter_id": _ids_from_urls(fighter_tott["URL"]),
            "fighter_name": fighter_tott["FIGHTER"].astype("string").str.strip(),
            "height": fighter_tott["HEIGHT"],
            "reach_inches": fighter_tott["REACH"].map(parse_reach_inches),
            "stance": fighter_tott["STANCE"].map(_blank_to_na),
            "dob": fighter_tott["DOB"].map(_blank_to_na),
        }
    ).dropna(subset=["fighter_id"])
    names = _fighter_id_frame(fighter_tott, fighter_details).drop_duplicates("fighter_id")
    detail_only = names[~names["fighter_id"].isin(set(tott["fighter_id"]))]
    extra = pd.DataFrame(
        {
            "fighter_id": detail_only["fighter_id"],
            "fighter_name": detail_only["name"],
            "height": pd.NA,
            "reach_inches": pd.NA,
            "stance": pd.NA,
            "dob": pd.NA,
        }
    )
    raw = pd.concat([tott, extra], ignore_index=True).drop_duplicates("fighter_id")
    return build_fighters(raw)


def _blank_to_na(value):
    """ufcstats writes an absent field as '--'; that is missing, not a value."""
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, float) and value != value:  # NaN
        return pd.NA
    text = str(value).strip()
    return pd.NA if text in {"", "--", "---"} else text


_OUTCOME_STATUS = {"W/L": "win", "L/W": "win", "D/D": "draw", "NC/NC": "no_contest"}

# Raw scrape stat column -> our fight_stats column name(s). "45 of 118" pairs
# expand to (landed, attempted); the rest are scalars.
_PAIRED_STATS = {
    "SIG.STR.": ("sig_landed", "sig_attempted"),
    "TOTAL STR.": ("total_landed", "total_attempted"),
    "TD": ("td_landed", "td_attempted"),
    "HEAD": ("head_landed", "head_attempted"),
    "BODY": ("body_landed", "body_attempted"),
    "LEG": ("leg_landed", "leg_attempted"),
    "DISTANCE": ("distance_landed", "distance_attempted"),
    "CLINCH": ("clinch_landed", "clinch_attempted"),
    "GROUND": ("ground_landed", "ground_attempted"),
}
_SCALAR_STATS = {"KD": "kd", "SUB.ATT": "sub_att", "REV.": "rev"}


def _master_column(out_name: str, prefix: str) -> str:
    """Our fight_stats column name -> the Kaggle master.csv column it lives in."""
    if out_name in _CORE_STAT_SUFFIXES:
        return prefix + _CORE_STAT_SUFFIXES[out_name]
    if out_name in _TARGET_STAT_SUFFIXES:
        return prefix + _TARGET_STAT_SUFFIXES[out_name]
    if out_name == "ctrl_sec":
        return prefix + "ctrl_seconds"
    if out_name == "rev":
        return prefix + "rev"
    raise KeyError(out_name)


def _round_column(out_name: str, prefix: str) -> str:
    """Our round_stats column name -> the Kaggle round.csv column it lives in.

    Control time is the one shape difference between the two raw layouts:
    master.csv stores fight-total seconds, round.csv the per-round "m:ss"
    text the scrape also publishes, so the raw string is carried through
    rather than seconds re-serialized back into it.
    """
    if out_name in _CORE_STAT_SUFFIXES:
        return prefix + _CORE_STAT_SUFFIXES[out_name]
    if out_name in _TARGET_STAT_SUFFIXES:
        return prefix + _TARGET_STAT_SUFFIXES[out_name]
    if out_name == "ctrl_raw":
        return prefix + "ctrl"
    if out_name == "rev":
        return prefix + "rev"
    raise KeyError(out_name)


_STAT_OUTPUTS = (
    [name for pair in _PAIRED_STATS.values() for name in pair]
    + list(_SCALAR_STATS.values())
    + ["ctrl_sec"]
)
# Everything `build_round_stats` reads for one corner, in round.csv terms.
_ROUND_OUTPUTS = [name for name in _STAT_OUTPUTS if name != "ctrl_sec"] + ["ctrl_raw"]
# The columns the primary's round_stats carries as integers: a null in any of
# them would make the merged column float, so a fight with one is not merged.
_ROUND_INTEGER_COLUMNS = (
    list(_CORE_STAT_SUFFIXES) + ["rev"] + list(_TARGET_STAT_SUFFIXES)
)


def _round_stats_long(stats_raw: pd.DataFrame, bout_key: pd.DataFrame) -> pd.DataFrame:
    """Per-round scrape rows -> one tidy row per (fight_id, corner, round)."""
    frame = pd.DataFrame(
        {
            "EVENT": stats_raw["EVENT"].astype("string").str.strip(),
            "BOUT": stats_raw["BOUT"].astype("string").str.strip(),
            "FIGHTER": stats_raw["FIGHTER"].astype("string").str.strip(),
            "round_no": pd.to_numeric(
                stats_raw["ROUND"].astype("string").str.extract(r"(\d+)", expand=False),
                errors="coerce",
            ),
        }
    )
    for raw_column, (landed, attempted) in _PAIRED_STATS.items():
        parsed = stats_raw[raw_column].map(parse_landed_attempted)
        frame[landed] = pd.to_numeric([value[0] for value in parsed], errors="coerce")
        frame[attempted] = pd.to_numeric([value[1] for value in parsed], errors="coerce")
    for raw_column, out_name in _SCALAR_STATS.items():
        frame[out_name] = pd.to_numeric(stats_raw[raw_column], errors="coerce")
    frame["ctrl_raw"] = stats_raw["CTRL"].astype("string").str.strip()
    frame["ctrl_sec"] = pd.to_numeric(
        stats_raw["CTRL"].map(parse_mmss_seconds), errors="coerce"
    )
    frame = frame.merge(bout_key, on=["EVENT", "BOUT"], how="inner")
    corner = pd.Series(pd.NA, index=frame.index, dtype="string")
    corner = corner.mask(frame["FIGHTER"] == frame["a_name"], "a")
    corner = corner.mask(frame["FIGHTER"] == frame["b_name"], "b")
    frame["corner"] = corner
    frame = frame.dropna(subset=["fight_id", "corner", "round_no"])
    # The co-branded cards are listed under two event names, so the same
    # round appears twice; keep one.
    return frame.drop_duplicates(["fight_id", "corner", "round_no"])


def _stat_totals(rounds: pd.DataFrame) -> pd.DataFrame:
    """Sum the per-round values into the master.csv fight-total columns."""
    grouped = rounds.groupby(["fight_id", "corner"])[_STAT_OUTPUTS].sum(min_count=1)
    rounds_fought = rounds.groupby("fight_id")["round_no"].nunique().rename("rounds_fought")
    wide = pd.DataFrame(index=rounds_fought.index)
    for corner, prefix in (("a", "r_total_"), ("b", "b_total_")):
        side = grouped.xs(corner, level="corner") if corner in grouped.index.get_level_values("corner") else None
        for out_name in _STAT_OUTPUTS:
            column = _master_column(out_name, prefix)
            wide[column] = side[out_name] if side is not None else pd.NA
    wide["rounds_fought"] = rounds_fought
    return wide.reset_index()


def _round_stats_wide(rounds: pd.DataFrame, corner_ids: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-corner round rows -> the Kaggle round.csv column layout.

    Only rounds carrying BOTH corners survive: `build_round_stats` needs a
    fighter id on each side, and half a round is not a record that can be
    reconciled against the primary source's.
    """
    frame = rounds.assign(_present=1)
    values = _ROUND_OUTPUTS + ["_present"]
    if not len(frame):
        return pd.DataFrame(
            columns=["fight_id", "round_no", "r_id", "b_id"]
            + [_round_column(n, p) for p in ("r_", "b_") for n in _ROUND_OUTPUTS]
        )
    pivoted = frame.pivot(index=["fight_id", "round_no"], columns="corner", values=values)
    wide = pd.DataFrame(index=pivoted.index)
    for corner, prefix in (("a", "r_"), ("b", "b_")):
        for out_name in _ROUND_OUTPUTS:
            key = (out_name, corner)
            wide[_round_column(out_name, prefix)] = (
                pivoted[key] if key in pivoted.columns else pd.NA
            )
    both = pd.DataFrame(
        {
            side: (pivoted[("_present", side)] if ("_present", side) in pivoted.columns
                   else pd.Series(pd.NA, index=pivoted.index))
            for side in ("a", "b")
        }
    )
    wide = wide[both.notna().all(axis=1)].reset_index()
    wide = wide.merge(corner_ids, on="fight_id", how="inner")
    return wide.rename(columns={"a_id": "r_id", "b_id": "b_id"})


def complete_round_fights(rounds: pd.DataFrame) -> set[str]:
    """Fight ids whose per-round record is whole enough to merge.

    Whole means: rounds numbered contiguously from 1, both corners on every
    round (enforced upstream in `_round_stats_wide`), and no null in any
    column the primary's `round_stats` carries as an integer -- a null there
    would silently widen the merged column to float for all 11k fights.
    """
    if not len(rounds):
        return set()
    span = rounds.groupby("fight_id")["round_no"].agg(["min", "max", "nunique"])
    whole = (span["min"] == 1) & (span["nunique"] == span["max"])
    holes = rounds.groupby("fight_id")[_ROUND_INTEGER_COLUMNS].apply(
        lambda block: bool(block.isna().to_numpy().any())
    )
    return set(span.index[whole & ~holes.reindex(span.index).fillna(False)])


class Adapted(NamedTuple):
    """The secondary source expressed in our processed schema."""

    fights: pd.DataFrame
    stats: pd.DataFrame
    rounds: pd.DataFrame
    fighters: pd.DataFrame
    report: dict


def adapt(source: dict[str, pd.DataFrame]) -> Adapted:
    """Upstream CSVs -> our four processed tables, plus a report.

    Reshapes into the Kaggle raw column names and hands the result to the
    tested builders, so every mapped value is identical in kind to the
    primary source's.
    """
    results = source["results"].copy()
    events = source["events"].copy()
    name_index = build_name_index(source["fighter_tott"], source["fighter_details"])

    results["EVENT"] = results["EVENT"].astype("string").str.strip()
    results["BOUT"] = results["BOUT"].astype("string").str.strip()
    results["fight_id"] = _ids_from_urls(results["URL"])
    n_source_rows = len(results)
    results = results.dropna(subset=["fight_id"])
    n_duplicate_fight_ids = int(results["fight_id"].duplicated().sum())
    bout_key = results[["EVENT", "BOUT", "fight_id"]].drop_duplicates(["EVENT", "BOUT"])
    results = results.drop_duplicates("fight_id")

    corners = results["BOUT"].str.split(" vs. ", regex=False)
    results["a_name"] = corners.str[0].str.strip()
    results["b_name"] = corners.str[1].str.strip()
    results["a_id"] = results["a_name"].map(name_index)
    results["b_id"] = results["b_name"].map(name_index)
    unresolved = results["a_id"].isna() | results["b_id"].isna()
    n_unresolved_name = int(unresolved.sum())
    unresolved_names = sorted(
        set(results.loc[results["a_id"].isna(), "a_name"].dropna())
        | set(results.loc[results["b_id"].isna(), "b_name"].dropna())
    )
    results = results[~unresolved]

    event_key = pd.DataFrame(
        {
            "EVENT": events["EVENT"].astype("string").str.strip(),
            "event_id": _ids_from_urls(events["URL"]),
            "event_date": events["DATE"].astype("string").str.strip(),
            "event_location": events["LOCATION"],
        }
    ).drop_duplicates("EVENT")
    results = results.merge(event_key, on="EVENT", how="left")

    outcome = results["OUTCOME"].astype("string").str.strip()
    # Anything the scrape does not label W/L, L/W or D/D is treated as a
    # no-contest rather than guessed -- the same conservative rule
    # `mma.dataset._winner_code` applies to the primary source.
    master = pd.DataFrame(
        {
            "fight_id": results["fight_id"],
            "event_date": results["event_date"],
            "event_id": results["event_id"],
            "event_location": results["event_location"],
            "r_fighter_id": results["a_id"],
            "b_fighter_id": results["b_id"],
            "result_status": outcome.map(_OUTCOME_STATUS).fillna("no_contest"),
            "winner_id": results["a_id"].where(
                outcome == "W/L", results["b_id"].where(outcome == "L/W")
            ),
            "method": results["METHOD"].astype("string").str.strip(),
            "time_format": results["TIME FORMAT"],
            "weight_class": results["WEIGHTCLASS"],
            "title_fight": results["WEIGHTCLASS"].map(is_title_fight).astype(int),
            "referee": results["REFEREE"],
            "finish_round": pd.to_numeric(results["ROUND"], errors="coerce"),
            "finish_time": results["TIME"],
        }
    )

    corner_ids = results[["fight_id", "a_id", "b_id"]].drop_duplicates("fight_id")
    bout_key = bout_key.merge(
        results[["fight_id", "a_name", "b_name"]], on="fight_id", how="inner"
    )
    rounds_long = _round_stats_long(source["stats"], bout_key)
    totals = _stat_totals(rounds_long)
    master = master.merge(totals, on="fight_id", how="left")
    for out_name in _STAT_OUTPUTS:
        for prefix in ("r_total_", "b_total_"):
            column = _master_column(out_name, prefix)
            if column not in master.columns:
                master[column] = pd.NA
    master["rounds_fought"] = master["rounds_fought"].fillna(0)

    fights = build_fights(master)
    stats = build_fight_stats(master)
    rounds = build_round_stats(_round_stats_wide(rounds_long, corner_ids))
    fighters = build_secondary_fighters(source["fighter_tott"], source["fighter_details"])
    report = {
        "n_source_rows": n_source_rows,
        "n_duplicate_fight_ids": n_duplicate_fight_ids,
        "n_unresolved_name": n_unresolved_name,
        "unresolved_names": unresolved_names,
        "n_fights": len(fights),
        "n_secondary_fighters": len(fighters),
        "max_date": fights["date"].max() if len(fights) else pd.NaT,
    }
    return Adapted(fights, stats, rounds, fighters, report)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def merge_secondary(
    primary_fights: pd.DataFrame,
    secondary_fights: pd.DataFrame,
    known_fighter_ids,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Append only what the primary source does not already have.

    Primary always wins: it is the reconciled, tested source, so a fight the
    two sources share keeps the primary's row even where they disagree. A
    fight naming a fighter absent from `fighters.parquet` is rejected --
    there is no biographical row or history to build features from -- and
    counted in the report; `strict=True` raises `UnknownFighterError`
    instead, naming the ids.
    """
    known = set(known_fighter_ids)
    max_date_before = primary_fights["date"].max() if len(primary_fights) else pd.NaT
    report = {
        "n_secondary": len(secondary_fights),
        "n_added": 0,
        "n_dropped_duplicate": 0,
        "n_rejected_unknown_fighter": 0,
        "rejected_fighter_ids": [],
        "max_date_before": max_date_before,
        "max_date_after": max_date_before,
    }
    if not len(secondary_fights):
        return primary_fights.copy(), report

    missing = [c for c in primary_fights.columns if c not in secondary_fights.columns]
    if missing:
        raise ValueError(
            f"secondary fights are missing columns present in the primary table: {missing}"
        )

    duplicate = secondary_fights["fight_id"].isin(set(primary_fights["fight_id"]))
    report["n_dropped_duplicate"] = int(duplicate.sum())
    candidates = secondary_fights[~duplicate]

    unknown_a = ~candidates["fighter_a_id"].isin(known)
    unknown_b = ~candidates["fighter_b_id"].isin(known)
    rejected = unknown_a | unknown_b
    report["n_rejected_unknown_fighter"] = int(rejected.sum())
    report["rejected_fighter_ids"] = sorted(
        set(candidates.loc[unknown_a, "fighter_a_id"].dropna())
        | set(candidates.loc[unknown_b, "fighter_b_id"].dropna())
    )
    if strict and report["rejected_fighter_ids"]:
        raise UnknownFighterError(
            "secondary fights name fighters absent from fighters.parquet: "
            + ", ".join(report["rejected_fighter_ids"])
        )

    added = _align_dtypes(candidates[~rejected], primary_fights)
    report["n_added"] = len(added)
    merged = (
        pd.concat([primary_fights, added], ignore_index=True)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    report["max_date_after"] = merged["date"].max() if len(merged) else pd.NaT
    return merged, report


def _align_dtypes(addition: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Take the template's columns, in its order, with its dtypes.

    Two tables built by the same builder from differently shaped inputs can
    still differ in dtype -- an all-present integer column against one that
    saw a null, or `datetime64[us]` against `[ns]` -- and concatenating them
    would silently rewrite the dtype of all 11k primary rows. Casting here
    keeps a rebuild byte-identical, and raises loudly if a value genuinely
    does not fit.
    """
    aligned = addition[list(template.columns)].copy()
    for column, dtype in template.dtypes.items():
        if aligned[column].dtype != dtype:
            aligned[column] = aligned[column].astype(dtype)
    return aligned


def _append(primary: pd.DataFrame, addition: pd.DataFrame, sort_by: list[str]) -> pd.DataFrame:
    """Concatenate and re-sort on the primary table's own sort key."""
    if not len(addition):
        return primary.copy()
    stacked = pd.concat([primary, _align_dtypes(addition, primary)], ignore_index=True)
    return stacked.sort_values(sort_by, kind="stable").reset_index(drop=True)


def resolve_new_fighters(
    primary_fighters: pd.DataFrame,
    secondary_fighters: pd.DataFrame,
    needed_ids,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Extend the fighters table with the debutants the new fights need.

    Only ids that some candidate fight actually names are added: the point
    is to stop rejecting fights, not to import a second fighter directory
    whose rows the primary will supply on its own next rebuild. An id the
    secondary cannot describe either is returned as unresolved, and its
    fights stay rejected -- a fabricated row would be worse than a gap.
    """
    known = set(primary_fighters["fighter_id"])
    wanted = sorted(set(needed_ids) - known)
    describable = secondary_fighters[secondary_fighters["fighter_id"].isin(wanted)]
    added_ids = sorted(describable["fighter_id"])
    unresolved = [fighter_id for fighter_id in wanted if fighter_id not in set(added_ids)]
    merged = _append(primary_fighters, describable, ["fighter_id"])
    return merged, added_ids, unresolved


class SecondaryResult(NamedTuple):
    """The four processed tables after the standing merge, and what it did."""

    fighters: pd.DataFrame
    fights: pd.DataFrame
    stats: pd.DataFrame
    rounds: pd.DataFrame
    secondary_fight_ids: frozenset
    secondary_fighter_ids: frozenset
    report: dict


def apply_secondary(
    fighters: pd.DataFrame,
    fights: pd.DataFrame,
    stats: pd.DataFrame,
    rounds: pd.DataFrame,
    source: dict[str, pd.DataFrame],
    floor: float = AGREEMENT_FLOOR,
) -> SecondaryResult:
    """Merge the secondary source on top of freshly built primary tables.

    A candidate fight is merged only when it brings the whole shape of
    record the primary provides -- the fight row, both stat rows, and a
    complete per-round record -- and only when both corners can be named in
    `fighters`. Anything less is rejected and counted rather than merged
    with a hole, which is what lets `make_dataset.py` run its integrity
    checks on the union without relaxing a single one.

    Raises `SecondaryRefused` if the two sources disagree on the fights they
    share; `merge_into` turns that into a primary-only build with a warning.
    """
    adapted = adapt(source)
    agreement = reconcile(adapted.fights, fights)
    winner_agreement = agreement["agreement"]["winner"]
    report = dict(adapted.report)
    report.update(
        {
            "n_overlap": agreement["n_overlap"],
            "winner_agreement": winner_agreement,
        }
    )
    if agreement["n_overlap"] and winner_agreement is not None and winner_agreement < floor:
        raise SecondaryRefused(
            f"winner agreement on the {agreement['n_overlap']}-fight overlap is "
            f"{winner_agreement:.4f}, below the {floor} floor"
        )

    known_fights = set(fights["fight_id"])
    candidates = adapted.fights[~adapted.fights["fight_id"].isin(known_fights)]
    report["n_dropped_duplicate"] = len(adapted.fights) - len(candidates)

    whole = complete_round_fights(adapted.rounds)
    with_rounds = candidates[candidates["fight_id"].isin(whole)]
    report["n_rejected_no_round_record"] = len(candidates) - len(with_rounds)

    needed = set(with_rounds["fighter_a_id"]) | set(with_rounds["fighter_b_id"])
    merged_fighters, added_fighter_ids, unresolved = resolve_new_fighters(
        fighters, adapted.fighters, needed
    )
    report["unresolved_fighter_ids"] = unresolved
    report["n_unresolved_fighter"] = len(unresolved)

    merged_fights, merge_report = merge_secondary(
        fights, with_rounds, set(merged_fighters["fighter_id"])
    )
    for key in ("n_added", "n_rejected_unknown_fighter", "rejected_fighter_ids",
                "max_date_before", "max_date_after"):
        report[key] = merge_report[key]

    added_fight_ids = set(merged_fights["fight_id"]) - known_fights
    # A fighter wanted only by a fight that was then rejected must not be
    # left behind in the fighters table.
    used_fighter_ids = set(
        merged_fights.loc[merged_fights["fight_id"].isin(added_fight_ids), "fighter_a_id"]
    ) | set(
        merged_fights.loc[merged_fights["fight_id"].isin(added_fight_ids), "fighter_b_id"]
    )
    kept_fighter_ids = sorted(set(added_fighter_ids) & used_fighter_ids)
    merged_fighters, _, _ = resolve_new_fighters(
        fighters, adapted.fighters, kept_fighter_ids
    )
    report["n_added_fighters"] = len(kept_fighter_ids)

    merged_stats = _append(
        stats, adapted.stats[adapted.stats["fight_id"].isin(added_fight_ids)],
        ["fight_id", "corner"],
    )
    merged_rounds = _append(
        rounds, adapted.rounds[adapted.rounds["fight_id"].isin(added_fight_ids)],
        ["fight_id", "round_no", "corner"],
    )
    return SecondaryResult(
        fighters=merged_fighters,
        fights=merged_fights,
        stats=merged_stats,
        rounds=merged_rounds,
        secondary_fight_ids=frozenset(added_fight_ids),
        secondary_fighter_ids=frozenset(kept_fighter_ids),
        report=report,
    )


def merge_into(
    fighters: pd.DataFrame,
    fights: pd.DataFrame,
    stats: pd.DataFrame,
    rounds: pd.DataFrame,
    fetch: Callable[..., dict[str, pd.DataFrame]] = fetch_source,
    base: str | None = None,
    stream=None,
) -> SecondaryResult:
    """The standing stage: fetch, adapt, merge -- and never break the build.

    Every failure mode of an external daily scrape degrades to the
    primary-only tables with a loud warning on stderr: an unreachable host,
    a renamed column, a truncated CSV, a source that disagrees with ours on
    the fights we both have. None of them may turn the weekly Action red or
    leave a half-built table behind, because the primary source alone is
    always a correct (if staler) answer.
    """
    stream = sys.stderr if stream is None else stream
    unchanged = SecondaryResult(
        fighters=fighters, fights=fights, stats=stats, rounds=rounds,
        secondary_fight_ids=frozenset(), secondary_fighter_ids=frozenset(),
        report={"applied": False, "error": None},
    )
    try:
        source = fetch(base=base)
        result = apply_secondary(fighters, fights, stats, rounds, source)
    except Exception as error:  # noqa: BLE001 -- fail-soft is the whole point
        message = f"{type(error).__name__}: {error}"
        print(
            f"WARNING: the secondary source ({SOURCE_URL}) contributed nothing "
            f"to this build -- {message}. Building from the primary source alone; "
            "the tables will simply be as fresh as the Kaggle mirror.",
            file=stream,
        )
        return unchanged._replace(report={"applied": False, "error": message})
    return result._replace(report=dict(result.report, applied=True, error=None))


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def build_provenance(
    fights: pd.DataFrame,
    fighters: pd.DataFrame,
    secondary_fight_ids,
    secondary_fighter_ids,
) -> pd.DataFrame:
    """One row per processed `fights` / `fighters` row, naming its source.

    A SIDECAR rather than a column on those tables, for three reasons. The
    builders in `mma.dataset` have a fixed column contract that the feature
    builder, the serving-parity check and several tests assert on, and
    provenance is not a property of a fight. It is a near-perfect proxy for
    recency -- "secondary" means "in the last few weeks" -- so keeping it out
    of the tables `build_features.py` reads means no future block can pick it
    up by iterating columns. And an audit wants one place to look, not two.

    `fight_stats` and `round_stats` are keyed by `fight_id` and inherit their
    fight's row here; they carry no rows for a fight `fights` does not have.
    """
    secondary_fights = set(secondary_fight_ids)
    secondary_fighters = set(secondary_fighter_ids)
    frames = [
        pd.DataFrame(
            {
                "table": table,
                "row_id": ids.astype("string"),
                "source": pd.Series(
                    [SECONDARY if row_id in secondary else PRIMARY for row_id in ids],
                    dtype="string",
                ).values,
            }
        )
        for table, ids, secondary in (
            ("fights", fights["fight_id"], secondary_fights),
            ("fighters", fighters["fighter_id"], secondary_fighters),
        )
    ]
    provenance = pd.concat(frames, ignore_index=True)
    provenance["table"] = provenance["table"].astype("string")
    return (
        provenance[PROVENANCE_COLUMNS]
        .sort_values(["table", "row_id"], kind="stable")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI (report only -- make_dataset.py owns the write path)
# ---------------------------------------------------------------------------


def load_primary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(PROCESSED / "fighters.parquet"),
        pd.read_parquet(PROCESSED / "fights.parquet"),
        pd.read_parquet(PROCESSED / "fight_stats.parquet"),
        pd.read_parquet(PROCESSED / "round_stats.parquet"),
    )


def print_report(report: dict, stream=None) -> None:
    """The line `make_dataset.py` and the weekly Action both print."""
    stream = sys.stdout if stream is None else stream
    print("\n=== SECONDARY SOURCE ===", file=stream)
    print(f"source:  {SOURCE_URL}  ({SOURCE_LICENCE}, data only)", file=stream)
    if not report.get("applied", True):
        print(f"applied: NO -- {report.get('error')}", file=stream)
        return
    rows = [
        ("source rows", report["n_source_rows"]),
        ("duplicate fight ids", f"{report['n_duplicate_fight_ids']} (co-branded cards listed twice)"),
        ("unresolved bout names", report["n_unresolved_name"]),
        ("adapted fights", report["n_fights"]),
        ("adapted fighters", report["n_secondary_fighters"]),
        ("overlap with primary", report["n_overlap"]),
        ("winner agreement", f"{report['winner_agreement']:.4f}"
            if report.get("winner_agreement") is not None else "n/a"),
        ("fights already primary", report["n_dropped_duplicate"]),
        ("fights added", report["n_added"]),
        ("fighters added", report["n_added_fighters"]),
        ("rejected: no round record", report["n_rejected_no_round_record"]),
        ("rejected: unknown fighter", report["n_rejected_unknown_fighter"]),
        ("unresolvable fighter ids", report["n_unresolved_fighter"]),
    ]
    for label, value in rows:
        print(f"  {label:<27} {value}", file=stream)
    if report.get("rejected_fighter_ids"):
        print(f"  rejected fighter ids:       {report['rejected_fighter_ids']}", file=stream)
    for key in ("max_date_before", "max_date_after"):
        value = report[key]
        text = f"{value:%Y-%m-%d}" if pd.notna(value) else "n/a"
        print(f"  {key:<27} {text}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="accepted for compatibility; this script never writes either way",
    )
    parser.add_argument("--base-url", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    # `fetch_source` by name, not by default argument, so the module attribute
    # is what runs -- that is what a test can stand in for.
    result = merge_into(*load_primary(), fetch=fetch_source, base=args.base_url)
    print_report(result.report)
    if not result.report.get("applied"):
        return 1
    print(
        "\nreport only: nothing written. `scripts/make_dataset.py` runs this same "
        "merge as a standing stage of every rebuild.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
