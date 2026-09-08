"""Fill the gap between irregular Kaggle refreshes from a daily ufcstats scrape.

The primary source is a Kaggle mirror of ufcstats that is rebuilt whenever
its maintainer gets round to it; as of 2026-09-08 its newest fight is dated
2026-08-08, so a month of results -- and the fighter form they carry -- is
missing from the deployed model and from the prospective track record's
grading. `Greco1899/scrape_ufc_stats` (GPL-3.0) scrapes the same site on a
daily schedule and publishes plain CSVs, so it can close that gap.

Used as a DATA source, not as code: this script fetches the published CSVs
over HTTPS; no GPL-licensed code is vendored into this repo.

The adaptation deliberately goes through `mma.dataset.build_fights` /
`build_fight_stats` rather than re-deriving our columns: the upstream CSVs
are reshaped into the Kaggle `master.csv` column names first, so the winner
codes, method mapping and `duration_sec` semantics are the tested ones by
construction rather than by a second implementation that can drift.

Default is a DRY RUN. `--enable` is required for any write, because a fetch
failure or an upstream schema change must never break the weekly Action.

Usage:
    python scripts/refresh_secondary.py              # dry run: fetch, adapt, report
    python scripts/refresh_secondary.py --enable     # ... and write the merge
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from mma.dataset import (
    _CORE_STAT_SUFFIXES,
    _TARGET_STAT_SUFFIXES,
    build_fight_stats,
    build_fights,
)
from mma.labels import is_title_fight
from mma.parsing import parse_landed_attempted, parse_mmss_seconds

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


class UnknownFighterError(ValueError):
    """A secondary fight names a fighter absent from `fighters.parquet`.

    We cannot build features for a fighter with no biographical row and no
    history, so such a fight is rejected rather than merged with holes.
    """


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


def build_name_index(fighter_tott: pd.DataFrame, fighter_details: pd.DataFrame) -> pd.Series:
    """Fighter name -> ufcstats id, from the source's own two fighter files.

    The scrape's fight tables key on the bout string ("A vs. B"), not on
    fighter URLs, so resolving a corner to an id needs a name lookup. It is
    a WITHIN-source lookup against ufcstats' own spellings, not cross-source
    name matching -- but the two fighter files still disagree on a handful of
    names ("Zach Reese" / "Zachary Reese", "Waldo Cortes Acosta" /
    "Waldo Cortes-Acosta"), so both are indexed. Any name that maps to more
    than one id is dropped entirely: guessing between two real fighters would
    attach one man's history to another.
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
    pairs = pairs[pairs["name"] != ""].drop_duplicates()
    distinct = pairs.groupby("name")["fighter_id"].nunique()
    unambiguous = pairs[pairs["name"].isin(distinct[distinct == 1].index)]
    return unambiguous.drop_duplicates("name").set_index("name")["fighter_id"]


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


_STAT_OUTPUTS = (
    [name for pair in _PAIRED_STATS.values() for name in pair]
    + list(_SCALAR_STATS.values())
    + ["ctrl_sec"]
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


def adapt(source: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Upstream CSVs -> our `fights` / `fight_stats` tables, plus a report.

    Reshapes into the Kaggle `master.csv` column names and hands the result
    to the tested builders, so every mapped value is identical in kind to
    the primary source's.
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

    bout_key = bout_key.merge(
        results[["fight_id", "a_name", "b_name"]], on="fight_id", how="inner"
    )
    rounds = _round_stats_long(source["stats"], bout_key)
    totals = _stat_totals(rounds)
    master = master.merge(totals, on="fight_id", how="left")
    for out_name in _STAT_OUTPUTS:
        for prefix in ("r_total_", "b_total_"):
            column = _master_column(out_name, prefix)
            if column not in master.columns:
                master[column] = pd.NA
    master["rounds_fought"] = master["rounds_fought"].fillna(0)

    fights = build_fights(master)
    stats = build_fight_stats(master)
    report = {
        "n_source_rows": n_source_rows,
        "n_duplicate_fight_ids": n_duplicate_fight_ids,
        "n_unresolved_name": n_unresolved_name,
        "unresolved_names": unresolved_names,
        "n_fights": len(fights),
        "max_date": fights["date"].max() if len(fights) else pd.NaT,
    }
    return fights, stats, report


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

    added = candidates[~rejected][list(primary_fights.columns)]
    report["n_added"] = len(added)
    merged = (
        pd.concat([primary_fights, added], ignore_index=True)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    report["max_date_after"] = merged["date"].max() if len(merged) else pd.NaT
    return merged, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_primary() -> tuple[pd.DataFrame, pd.DataFrame, set]:
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    return fights, stats, set(fighters["fighter_id"])


def _print_report(adapt_report: dict, merge_report: dict, agreement: dict) -> None:
    print("\n=== SECONDARY SOURCE ===")
    print(f"source:  {SOURCE_URL}  ({SOURCE_LICENCE}, data only)")
    print(f"files:   {', '.join(FILES.values())}")
    print(f"source rows:            {adapt_report['n_source_rows']}")
    print(f"duplicate fight ids:    {adapt_report['n_duplicate_fight_ids']} (co-branded cards listed twice)")
    print(f"unresolved bout names:  {adapt_report['n_unresolved_name']}")
    if adapt_report["unresolved_names"]:
        print(f"  sample: {adapt_report['unresolved_names'][:5]}")
    print(f"adapted fights:         {adapt_report['n_fights']} through {adapt_report['max_date']:%Y-%m-%d}")
    print("\nreconciliation on the overlap:")
    print(f"  n_overlap:            {agreement['n_overlap']}")
    for column, rate in agreement["agreement"].items():
        print(f"  {column:<21} {rate:.4f}" if rate is not None else f"  {column:<21} n/a")
    print("\nmerge:")
    for key in (
        "n_secondary", "n_added", "n_dropped_duplicate", "n_rejected_unknown_fighter",
    ):
        print(f"  {key:<28} {merge_report[key]}")
    if merge_report["rejected_fighter_ids"]:
        print(f"  rejected fighter ids:        {merge_report['rejected_fighter_ids']}")
    for key in ("max_date_before", "max_date_after"):
        value = merge_report[key]
        print(f"  {key:<28} {value:%Y-%m-%d}" if pd.notna(value) else f"  {key:<28} n/a")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="the default: fetch, adapt, merge in memory and print the report without writing",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="actually write the merged fights/fight_stats tables (off by default)",
    )
    parser.add_argument("--base-url", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    source = fetch_source(base=args.base_url)
    secondary_fights, secondary_stats, adapt_report = adapt(source)
    primary_fights, primary_stats, known_fighter_ids = load_primary()

    # Reconciliation guard: compare the secondary against ours on the fights
    # BOTH have. Low winner agreement means the adaptation is wrong or the
    # source disagrees materially -- either way, do not merge.
    agreement = reconcile(secondary_fights, primary_fights)
    merged_fights, merge_report = merge_secondary(
        primary_fights, secondary_fights, known_fighter_ids
    )
    _print_report(adapt_report, merge_report, agreement)

    winner_agreement = agreement["agreement"]["winner"]
    if agreement["n_overlap"] and winner_agreement is not None and winner_agreement < AGREEMENT_FLOOR:
        print(
            f"\nREFUSING TO MERGE: winner agreement on the overlap is "
            f"{winner_agreement:.4f}, below the {AGREEMENT_FLOOR} floor."
        )
        return 1

    if not args.enable:
        print("\ndry run: nothing written (pass --enable to write)")
        return 0

    added_ids = set(merged_fights["fight_id"]) - set(primary_fights["fight_id"])
    merged_stats = (
        pd.concat(
            [primary_stats, secondary_stats[secondary_stats["fight_id"].isin(added_ids)]],
            ignore_index=True,
        )
        .sort_values(["fight_id", "corner"])
        .reset_index(drop=True)
    )
    merged_fights.to_parquet(PROCESSED / "fights.parquet", index=False)
    merged_stats.to_parquet(PROCESSED / "fight_stats.parquet", index=False)
    print(f"\nwrote {len(merged_fights)} fights and {len(merged_stats)} stat rows")
    if merge_report["n_rejected_unknown_fighter"]:
        print(
            "warning: rejected fights involve fighters absent from fighters.parquet; "
            "that table is NOT extended by this script (see SP4 follow-up)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
