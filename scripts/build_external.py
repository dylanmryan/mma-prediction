"""Derive the committed external per-fighter table from an open MIT snapshot.

    python scripts/build_external.py                     # clones into a temp dir
    python scripts/build_external.py --snapshot /path/to/jds-mma-data

Source: https://github.com/ehan03/jds-mma-data (MIT, Eugene Han). The raw
snapshot is ~130 MB and is NEVER committed; only the derived aggregates in
`data/external/fighter_external.parquet` (plus `data/external/README.md`,
which records the snapshot commit and the coverage dates) are.

What is derived, keyed by OUR ufcstats `fighter_id`:
  pre_ufc_wins, pre_ufc_losses            counts before the first UFC bout
  pre_ufc_finish_rate                     share of those wins that were finishes
  pre_ufc_finish_loss_rate                share of those losses that were finishes
  pre_ufc_avg_opp_wins                    mean career wins of the pre-UFC
                                          opponents AS OF each of those bouts
  pro_debut_date                          first recorded pro bout (a DATE)
  first_ufc_date                          first bout in our own fights table
  nationality, gym_id                     origin descriptors, not features

POINT-IN-TIME ARGUMENT (the reason this block is safe):

  Every `pre_ufc_*` column is computed over the bouts dated STRICTLY BEFORE
  the fighter's first UFC bout. Our feature table contains only UFC fights,
  so for every row in which a fighter appears, the whole window the column
  summarises lies strictly in that row's past. The values are therefore
  constant across all of a fighter's UFC fights and cannot carry information
  from one UFC fight into another, let alone from the future. `pro_debut_date`
  is a fixed date for the same reason; the feature layer turns it into
  `days_since_pro_debut` relative to each fight's own date, exactly as
  `days_since_last` is handled.

  `pre_ufc_avg_opp_wins` is the one column that needs care, because it reads
  the OPPONENT's record. It is evaluated as of the pre-UFC bout's own date
  (wins strictly before it), not as of the snapshot, so it too describes only
  the pre-UFC window.

  Mechanically this is checked by
  `tests/test_processed_features.py::test_no_leakage_truncation_invariance`:
  the table is a static committed artifact, so truncating the fights table
  cannot change any column derived from it.

  The one thing that would break the argument is a fighter whose recorded pro
  debut post-dates their first UFC bout -- the window would then be empty or
  wrong. That is a source data error (2 fighters in this snapshot) and
  `mma.external` flags those fighters as missing rather than emitting a
  negative duration; the flag is not applied here so that the raw disagreement
  stays visible in the committed table.

GYM DATA: `Tapology/fighter_gyms.csv` is keyed by (fighter, bout), so gym
affiliation IS dated in this source -- it is not an as-of-scrape snapshot, and
451 of 2,253 fighters change gym over their career in it. It is therefore not
leaky in principle. It is still not turned into a feature: a gym win-rate is
accumulated per-fight state rather than fighter-static data, and the snapshot's
UFC coverage stops at 2024-12-14, so the affiliation would be unknown for every
future fight the deployed model actually serves. `gym_id` is recorded here (the
gym at the fighter's earliest UFC bout) for a later block that can date it live.
"""
from __future__ import annotations

import argparse
import bisect
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "external"
SOURCE_URL = "https://github.com/ehan03/jds-mma-data"
FINISH_METHODS = ("KO/TKO", "Submission")


def clone_snapshot(destination: Path) -> None:
    """Shallow-clone the source repo if it is not already there."""
    if (destination / "data" / "clean").is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", SOURCE_URL, str(destination)],
        check=True,
    )


def snapshot_commit(snapshot: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(snapshot), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _dates(series: pd.Series) -> pd.Series:
    """The snapshot mixes bare dates and midnight timestamps in one column."""
    return pd.to_datetime(series, format="ISO8601", errors="coerce")


def first_ufc_dates(fights: pd.DataFrame) -> pd.Series:
    """fighter_id -> the date of their first bout in OUR fights table.

    Our table is the UFC universe, so this is the boundary the pre-UFC window
    is cut at. Taking it from our own data rather than the snapshot's keeps
    the cut consistent with the rows the features are built for.
    """
    corners = pd.concat([
        fights[["fighter_a_id", "date"]].rename(columns={"fighter_a_id": "fighter_id"}),
        fights[["fighter_b_id", "date"]].rename(columns={"fighter_b_id": "fighter_id"}),
    ])
    return corners.groupby("fighter_id")["date"].min()


def win_date_index(histories: pd.DataFrame) -> dict[str, list]:
    """sherdog fighter id -> their win dates, ascending (for as-of counting)."""
    wins = histories.loc[histories["outcome"] == "W", ["fighter_id", "date"]]
    index: dict[str, list] = defaultdict(list)
    for fighter_id, date in wins.sort_values("date").itertuples(index=False):
        index[fighter_id].append(date)
    return index


def derive(snapshot: Path, fights: pd.DataFrame) -> pd.DataFrame:
    clean = snapshot / "data" / "clean"
    mapping = pd.read_csv(clean / "fighter_mapping.csv", dtype=str)
    histories = pd.read_csv(
        clean / "Sherdog" / "fighter_histories.csv",
        dtype={"fighter_id": str, "opponent_id": str},
    )
    histories["date"] = _dates(histories["date"])
    bios = pd.read_csv(clean / "Sherdog" / "fighters.csv", dtype={"id": str})

    debut = first_ufc_dates(fights)
    wins_before = win_date_index(histories)
    by_fighter = dict(tuple(histories.groupby("fighter_id")))
    nationality = bios.set_index("id")["nationality"].to_dict()
    gyms = gym_at_ufc_debut(clean, mapping, fights)

    rows = []
    for entry in mapping.itertuples(index=False):
        first_ufc = debut.get(entry.ufcstats_id)
        history = by_fighter.get(entry.sherdog_id)
        if pd.isna(first_ufc) or history is None:
            continue
        pre = history[history["date"] < first_ufc]
        wins = pre[pre["outcome"] == "W"]
        losses = pre[pre["outcome"] == "L"]
        opponent_wins = [
            bisect.bisect_left(wins_before[opponent], date) if opponent in wins_before else 0
            for opponent, date in pre[["opponent_id", "date"]].itertuples(index=False)
            if pd.notna(opponent)
        ]
        rows.append({
            "fighter_id": entry.ufcstats_id,
            "pre_ufc_wins": len(wins),
            "pre_ufc_losses": len(losses),
            "pre_ufc_finish_rate": (
                float(wins["outcome_method_broad"].isin(FINISH_METHODS).mean())
                if len(wins) else np.nan
            ),
            "pre_ufc_finish_loss_rate": (
                float(losses["outcome_method_broad"].isin(FINISH_METHODS).mean())
                if len(losses) else np.nan
            ),
            "pre_ufc_avg_opp_wins": float(np.mean(opponent_wins)) if opponent_wins else np.nan,
            "pro_debut_date": history["date"].min(),
            "first_ufc_date": first_ufc,
            "nationality": nationality.get(entry.sherdog_id),
            "gym_id": gyms.get(entry.ufcstats_id),
        })

    table = pd.DataFrame(rows).sort_values("fighter_id").reset_index(drop=True)
    table["pre_ufc_wins"] = table["pre_ufc_wins"].astype("int64")
    table["pre_ufc_losses"] = table["pre_ufc_losses"].astype("int64")
    for column in ("nationality", "gym_id"):
        table[column] = table[column].astype("string")
    return table


def gym_at_ufc_debut(clean: Path, mapping: pd.DataFrame, fights: pd.DataFrame) -> dict[str, str]:
    """fighter_id -> the gym recorded at their earliest dated UFC bout.

    `Tapology/fighter_gyms.csv` is per (fighter, bout); its bouts carry a
    ufcstats id, so each affiliation can be dated from our own fights table.
    Taking the earliest keeps the value point-in-time-safe for every fight
    after the debut (it is not used as a feature either way -- see the module
    docstring).
    """
    fighter_gyms = pd.read_csv(clean / "Tapology" / "fighter_gyms.csv", dtype=str)
    bouts = pd.read_csv(clean / "Tapology" / "bouts.csv", dtype=str)
    dated = fighter_gyms.merge(
        bouts[["id", "ufcstats_id"]].rename(columns={"id": "bout_id"}),
        on="bout_id", how="inner",
    ).merge(
        fights[["fight_id", "date"]].rename(columns={"fight_id": "ufcstats_id"}),
        on="ufcstats_id", how="inner",
    )
    dated = dated.merge(
        mapping[["ufcstats_id", "tapology_id"]].rename(
            columns={"ufcstats_id": "our_id", "tapology_id": "fighter_id"}
        ),
        on="fighter_id", how="inner",
    )
    dated = dated[dated["gym_id"].notna()].sort_values(["date", "ufcstats_id"], kind="stable")
    return dated.drop_duplicates("our_id", keep="first").set_index("our_id")["gym_id"].to_dict()


def write_readme(path: Path, table: pd.DataFrame, commit: str,
                 coverage_end, fighters: pd.DataFrame) -> None:
    matched = fighters["fighter_id"].isin(set(table["fighter_id"]))
    coverage = "\n".join(
        f"| `{column}` | {int(table[column].notna().sum())} | "
        f"{table[column].notna().mean():.3f} |"
        for column in table.columns if column != "fighter_id"
    )
    path.write_text(f"""# `data/external/` — derived open-snapshot aggregates

| | |
|---|---|
| Source | <{SOURCE_URL}> |
| Licence | MIT (Copyright (c) 2025 Eugene Han) |
| Snapshot commit | `{commit}` |
| Coverage end (source UFC events) | {coverage_end:%Y-%m-%d} |
| Rows in `fighter_external.parquet` | {len(table)} |
| Match rate vs `data/processed/fighters.parquet` | {int(matched.sum())} / {len(fighters)} ({matched.mean():.3f}) |

**Only derived aggregates are committed here, never the raw snapshot** (~130 MB);
regenerate with `python scripts/build_external.py`, which clones the source into
a scratch directory.

The snapshot is static: its last commit is December 2025 and its UFC event
coverage ends {coverage_end:%Y-%m-%d}, so fighters who debuted after that are
unmatched by construction. That is what the row-level `external_missing` flag
and the walk-forward slice of the same name exist to measure.

Per-column non-null counts over the {len(table)} matched fighters:

| column | non-null | share |
|---|---|---|
{coverage}

## Membership of this table is not itself a feature

Being in the source's cross-source fighter mapping requires the fighter to be
findable on every one of its sites, which in practice means having had a UFC
career. Among fighters who debuted 2013-2022 -- well inside the coverage
window -- 24% of the one-and-done fighters are in the mapping against 100% of
those with 11 or more bouts. Membership is therefore a look-ahead variable:
whether a fighter is here depends on fights that had not happened yet.

Consequently the `external` feature block carries missingness only at fight
level, as the symmetric either-corner OR. A per-corner
`external_missing_a`/`_b` pair was measured first and leaks badly: the unmapped
corner loses 76% of the time overall and 90% of the time in the debut slice,
which bought a spurious -0.0143 pooled log-loss on the 2018-2023 folds and then
cost +0.0487 on 2025, where missingness means "debuted after the snapshot"
instead. See the SP2 plan's Task 11 notes.

See `scripts/build_external.py` for the point-in-time argument behind every
column, and `src/mma/external.py` for the join (by fighter id only, never by
name) and the missingness rules.
""")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--snapshot", type=Path, default=Path("/tmp/jds-mma-data"),
        help="path to a clone of the source repo (cloned there if absent)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    clone_snapshot(args.snapshot)
    commit = snapshot_commit(args.snapshot)

    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    events = pd.read_csv(args.snapshot / "data" / "clean" / "UFC Stats" / "events.csv")
    coverage_end = _dates(events["date"]).max()

    table = derive(args.snapshot, fights)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out_dir / "fighter_external.parquet", index=False)
    write_readme(args.out_dir / "README.md", table, commit, coverage_end, fighters)

    matched = fighters["fighter_id"].isin(set(table["fighter_id"]))
    errors = (table["pro_debut_date"] > table["first_ufc_date"]).sum()
    print(f"snapshot {commit[:10]} -> {args.out_dir / 'fighter_external.parquet'}")
    print(f"{len(table)} fighters; coverage end {coverage_end:%Y-%m-%d}")
    print(f"match rate vs fighters.parquet: {int(matched.sum())}/{len(fighters)} "
          f"({matched.mean():.4f})")
    print(f"pro debut after first UFC bout (source errors, flagged at load): {int(errors)}")
    print(table.notna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
