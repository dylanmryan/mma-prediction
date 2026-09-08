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
                                          (see the CONVENTION note below)
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

  What does NOT check this is
  `tests/test_processed_features.py::test_no_leakage_truncation_invariance`.
  That test rebuilds the feature table from a truncated fights table; this
  table is a static committed artifact, so truncation cannot change it and the
  test passes whatever is in here -- including, hypothetically, a column
  derived from next year's results. The falsifiable form of the argument is
  the constancy above, and it is checked directly by
  `tests/test_external.py::test_pre_ufc_values_are_constant_across_a_fighters_ufc_career`,
  which asserts that every `pre_ufc_*` value a fighter takes is the same at
  every one of their UFC fights.

  The one thing that would break the argument is a fighter whose recorded pro
  debut post-dates their first UFC bout -- the window would then be empty or
  wrong. That is a source data error (2 fighters in this snapshot) and
  `mma.external` flags those fighters as missing rather than emitting a
  negative duration; the flag is not applied here so that the raw disagreement
  stays visible in the committed table.

CONVENTION -- `pre_ufc_avg_opp_wins` scores an unrecorded opponent as 0 wins:

  The opponent's win count is `bisect` over their own recorded win dates, so
  an opponent with no wins before the bout contributes 0. That is right for
  the common case -- 3,482 of the 26,336 pre-UFC bouts (13.2%) are against an
  opponent the source knows and who had genuinely not won yet -- and wrong for
  one case: an opponent with no history rows at all is "unknown", not "zero".
  In this snapshot that is exactly **1 bout of 26,336 (0.004%), affecting 1
  fighter of 2,553**, so the convention is documented and kept rather than
  changed: switching to NaN would require regenerating this table and
  re-running every walk-forward evaluation the `external` shipping decision
  rests on, to move one value for one fighter.

  It is kept SAFE rather than merely documented: `derive` counts the unknown
  opponents and raises if they exceed `MAX_UNKNOWN_OPPONENT_SHARE`, so a
  future snapshot in which the conflation actually matters fails loudly here
  instead of quietly biasing the column downwards.

SECOND OUTPUT -- `data/external/fight_notice.parquet` (SP2 Task 12):

  Keyed by (our `fight_id`, our `fighter_id`), one row per CORNER of every
  bout the Bet MMA tables actually cover:

    notice_days             days of notice for a late replacement, NaN if the
                            fighter was not one (an OBSERVED full camp)
    missed_weight           whether the source records a missed weigh-in
    missed_weight_over_lbs  pounds over the divisional limit, NaN when the
                            magnitude is not quantifiable (see below)

  `missed_weight` is a separate column from the overage rather than "the
  overage is not null", because three of the 223 recorded misses cannot be
  quantified: two are at `Catch Weight`, which has no limit, and one records
  273 lb at a bantamweight bout -- a source transcription error that would
  otherwise enter the feature table as 138 pounds over. The fact of the miss
  is what the source asserts and is kept; the magnitude is NaN when it is not
  believable, bounded by `MAX_MISSED_WEIGHT_OVER_LBS`.

  The point of emitting both corners of a covered bout and nothing at all for
  the rest is that `late_replacements.csv` lists only fighters who WERE
  replacements, so "no row" is otherwise ambiguous between "full camp" and
  "nobody looked". Bet MMA's own bout list is what resolves it: inside it,
  absence is an observation; outside it, absence is ignorance. `mma.notice`
  carries the resulting three-state rule.

  Both facts precede the bout, so the block is point-in-time by construction.
  Corner assignment never uses names: `bout_mapping.csv` gives our `fight_id`
  and both Bet MMA fighter ids resolve to ufcstats ids, and the derivation
  asserts the resulting pair is exactly the pair our own fights table records.

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
sys.path.insert(0, str(ROOT / "src"))

from mma.external import drop_source_errors  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "external"
SOURCE_URL = "https://github.com/ehan03/jds-mma-data"
# Divisional weigh-in limits in pounds. Non-title bouts allow one extra pound,
# but a fighter listed in `missed_weights.csv` missed by the source's own
# reckoning either way, so the base limit is what the overage is measured
# against. `Catch Weight` and `Open Weight` have no limit and are left NaN.
# A recorded weigh-in more than this far over the divisional limit is a source
# transcription error, not a miss: the worst genuine UFC misses are around ten
# pounds. One row in the ec77f537 snapshot reads 273 lb at a bantamweight bout.
MAX_MISSED_WEIGHT_OVER_LBS = 25.0
WEIGHT_LIMIT_LBS = {
    "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
    "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
    "Light Heavyweight": 205, "Heavyweight": 265,
    "Women's Strawweight": 115, "Women's Flyweight": 125,
    "Women's Bantamweight": 135, "Women's Featherweight": 145,
}
FINISH_METHODS = ("KO/TKO", "Submission")
# See the CONVENTION note in the module docstring: an opponent with no history
# rows at all is scored as 0 wins, which is only tolerable while it is
# vanishingly rare (0.004% of pre-UFC bouts in the ec77f537 snapshot).
MAX_UNKNOWN_OPPONENT_SHARE = 0.001


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
    # Fighters the source has ANY history for. `wins_before` alone cannot tell
    # "no wins yet" from "no such fighter"; this set can. See the CONVENTION
    # note in the module docstring.
    known_fighters = set(histories["fighter_id"].dropna())
    n_opponent_bouts = n_unknown_opponents = 0
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
        opponent_wins = []
        for opponent, date in pre[["opponent_id", "date"]].itertuples(index=False):
            if pd.isna(opponent):
                continue
            n_opponent_bouts += 1
            if opponent not in known_fighters:
                # Unknown, scored as zero -- the documented convention, held
                # to a rate the assertion below enforces.
                n_unknown_opponents += 1
            opponent_wins.append(
                bisect.bisect_left(wins_before[opponent], date)
                if opponent in wins_before else 0
            )
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

    share = n_unknown_opponents / n_opponent_bouts if n_opponent_bouts else 0.0
    print(f"pre_ufc_avg_opp_wins: {n_unknown_opponents}/{n_opponent_bouts} pre-UFC bouts "
          f"({share:.5%}) are against an opponent with no history rows, scored as 0 wins")
    if share > MAX_UNKNOWN_OPPONENT_SHARE:
        raise ValueError(
            f"{share:.4%} of pre-UFC bouts have an unrecorded opponent, above the "
            f"{MAX_UNKNOWN_OPPONENT_SHARE:.4%} the zero-wins convention tolerates; "
            "emit NaN for those bouts and average over the known ones instead "
            "(see the CONVENTION note in this module's docstring)"
        )

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


def betmma_fighter_map(clean: Path) -> dict[str, str]:
    """Bet MMA fighter id -> our ufcstats id, from the two sources that carry it.

    `Bet MMA/fighters.csv` has a `ufcstats_id` column and `fighter_mapping.csv`
    has a `betmma_id` column; they agree on all 1,765 ids they share, and the
    union covers 1,874. Using both matters: with only the first, 14 of the
    covered bouts' replacement/miss rows have an unresolvable fighter, and a
    covered bout with one unresolvable corner would have to be dropped -- which
    is exactly the per-corner asymmetry `mma.notice` is built to avoid.
    """
    mapping = pd.read_csv(clean / "fighter_mapping.csv", dtype=str)
    betmma = pd.read_csv(clean / "Bet MMA" / "fighters.csv", dtype=str)
    resolved = mapping.dropna(subset=["betmma_id"]).set_index("betmma_id")[
        "ufcstats_id"].to_dict()
    resolved.update(
        betmma.dropna(subset=["ufcstats_id"]).set_index("id")["ufcstats_id"].to_dict()
    )
    return resolved


def derive_notice(snapshot: Path, fights: pd.DataFrame) -> pd.DataFrame:
    """One row per corner of every UFC bout the Bet MMA tables cover.

    See the SECOND OUTPUT note in the module docstring for why the covered
    bouts, rather than the replacement rows, are the unit here.
    """
    clean = snapshot / "data" / "clean"
    bout_mapping = pd.read_csv(clean / "bout_mapping.csv", dtype=str)
    bouts = pd.read_csv(clean / "Bet MMA" / "bouts.csv", dtype=str)
    replacements = pd.read_csv(
        clean / "Bet MMA" / "late_replacements.csv",
        dtype={"fighter_id": str, "bout_id": str},
    )
    misses = pd.read_csv(
        clean / "Bet MMA" / "missed_weights.csv",
        dtype={"fighter_id": str, "bout_id": str},
    )
    to_fighter = betmma_fighter_map(clean)
    to_fight = (
        bout_mapping.dropna(subset=["betmma_id"])
        .set_index("betmma_id")["ufcstats_id"].to_dict()
    )
    ours = fights.set_index("fight_id")[
        ["fighter_a_id", "fighter_b_id", "weight_class"]
    ]

    covered = bouts[bouts["id"].isin(to_fight)].copy()
    covered["fight_id"] = covered["id"].map(to_fight)
    covered = covered[covered["fight_id"].isin(ours.index)]

    rows = []
    for bout in covered.itertuples(index=False):
        corners = [to_fighter.get(bout.fighter_1_id), to_fighter.get(bout.fighter_2_id)]
        ours_row = ours.loc[bout.fight_id]
        if set(corners) != {ours_row["fighter_a_id"], ours_row["fighter_b_id"]}:
            # Either corner unresolvable, or the source disagrees with us about
            # who fought. Both corners are dropped together: half a bout is not
            # an observation (see `mma.notice`).
            continue
        limit = WEIGHT_LIMIT_LBS.get(ours_row["weight_class"])
        for betmma_id, fighter_id in zip(
            (bout.fighter_1_id, bout.fighter_2_id), corners
        ):
            rows.append({
                "fight_id": bout.fight_id,
                "fighter_id": fighter_id,
                "_bout_id": bout.id,
                "_betmma_fighter_id": betmma_id,
                "_limit": limit,
            })

    table = pd.DataFrame(rows)
    table = table.merge(
        replacements.rename(columns={
            "bout_id": "_bout_id", "fighter_id": "_betmma_fighter_id",
            "notice_time_days": "notice_days",
        }),
        on=["_bout_id", "_betmma_fighter_id"], how="left", validate="one_to_one",
    )
    table = table.merge(
        misses.rename(columns={
            "bout_id": "_bout_id", "fighter_id": "_betmma_fighter_id",
        }),
        on=["_bout_id", "_betmma_fighter_id"], how="left", validate="one_to_one",
    )
    table["missed_weight"] = table["weight_lbs"].notna()
    over = (
        table["weight_lbs"].astype(float) - table["_limit"].astype(float)
    ).clip(lower=0.0)
    # A recorded miss whose overage is not positive means our `weight_class` is
    # the bout's RENEGOTIATED class, not the one the fighter missed; the miss
    # still happened, so it stays at 0 pounds over. An overage past the
    # plausibility bound, or a class with no limit, leaves the MAGNITUDE
    # unknown -- NaN -- while the boolean still records the miss.
    over[over > MAX_MISSED_WEIGHT_OVER_LBS] = np.nan
    table["missed_weight_over_lbs"] = over.where(table["missed_weight"], np.nan)
    unquantified = int((table["missed_weight"] & over.isna()).sum())
    print(f"missed weight: {int(table['missed_weight'].sum())} recorded, "
          f"{unquantified} without a believable overage")
    table["notice_days"] = table["notice_days"].astype(float)
    table = table[["fight_id", "fighter_id", "notice_days",
                   "missed_weight", "missed_weight_over_lbs"]]
    table = table.sort_values(["fight_id", "fighter_id"]).reset_index(drop=True)

    per_fight = table.groupby("fight_id").size()
    if not (per_fight == 2).all():
        raise ValueError(
            "every covered fight must contribute exactly two corners; "
            f"{int((per_fight != 2).sum())} do not"
        )
    return table


def write_readme(path: Path, table: pd.DataFrame, commit: str,
                 coverage_end, fighters: pd.DataFrame,
                 notice: pd.DataFrame, fights: pd.DataFrame) -> None:
    """Both match rates are reported, pre- and post-drop.

    The committed table keeps the source-error fighters so the raw
    disagreement stays visible, but `mma.external.load_table` drops them, so
    the number that actually reaches the feature table is the post-drop one.
    Quoting only the pre-drop rate overstates coverage by however many
    fighters the guard removes.
    """
    matched = fighters["fighter_id"].isin(set(table["fighter_id"]))
    usable = drop_source_errors(table)
    matched_usable = fighters["fighter_id"].isin(set(usable["fighter_id"]))
    coverage = "\n".join(
        f"| `{column}` | {int(table[column].notna().sum())} | "
        f"{table[column].notna().mean():.3f} |"
        for column in table.columns if column != "fighter_id"
    )
    seen = fights["fight_id"].isin(set(notice["fight_id"]))
    window = fights[
        fights["fight_id"].isin(set(notice["fight_id"]))
    ]["date"]
    in_window = fights[(fights["date"] >= window.min()) & (fights["date"] <= window.max())]
    n_notice_rows = len(notice)
    n_notice_fights = int(notice["fight_id"].nunique())
    notice_share = float(seen.mean())
    n_fights = len(fights)
    window_share = float(in_window["fight_id"].isin(set(notice["fight_id"])).mean())
    window_label = f"{window.min():%Y-%m-%d} .. {window.max():%Y-%m-%d}"
    n_replacements = int(notice["notice_days"].notna().sum())
    n_misses = int(notice["missed_weight"].sum())
    notice_lo = float(notice["notice_days"].min())
    notice_hi = float(notice["notice_days"].max())
    path.write_text(f"""# `data/external/` — derived open-snapshot aggregates

This file covers the two tables derived from the `ehan03/jds-mma-data`
snapshot: `fighter_external.parquet` and `fight_notice.parquet`. The third
table in this directory, `rankings.parquet`, comes from a different source
under a different licence and is documented in
[`RANKINGS.md`](RANKINGS.md) (built by `scripts/build_rankings.py`).

| | |
|---|---|
| Source | <{SOURCE_URL}> |
| Licence | MIT (Copyright (c) 2025 Eugene Han) |
| Snapshot commit | `{commit}` |
| Coverage end (source UFC events) | {coverage_end:%Y-%m-%d} |
| Licence notice | vendored verbatim as [`LICENSE-jds-mma-data`](LICENSE-jds-mma-data) |
| Rows in `fighter_external.parquet` | {len(table)} as committed; **{len(usable)} usable** |
| Match rate vs `data/processed/fighters.parquet` | {int(matched.sum())} / {len(fighters)} ({matched.mean():.3f}) as committed; **{int(matched_usable.sum())} / {len(fighters)} ({matched_usable.mean():.3f}) usable** |

The two rows differ by the {len(table) - len(usable)} fighter(s) whose recorded pro debut
post-dates their first UFC bout. They are kept in the committed parquet so the
source disagreement stays visible, and dropped by
`mma.external.drop_source_errors` before anything reads it -- so the *usable*
figure is the one the feature table actually sees, and the one to quote.

**Only derived aggregates are committed here, never the raw snapshot** (~130 MB);
regenerate with `python scripts/build_external.py`, which clones the source into
a scratch directory.

The snapshot is static: its last commit is December 2025 and its UFC event
coverage ends {coverage_end:%Y-%m-%d}, so fighters who debuted after that are
unmatched by construction. That is what the row-level `external_missing` flag
and the walk-forward slice of the same name exist to measure.

Per-column non-null counts over the {len(table)} committed rows:

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

The shipped block goes further: both fight-level flags stay in the feature
table (so the `external_missing` walk-forward slice is reportable) and are
excluded from every model matrix, because modelling them makes the block decay
as the snapshot ages. What makes the selection effect *unreachable* rather
than merely unmodelled is that on the rows where exactly one corner is
unmapped, every one of the block's columns takes the same value -- one
distinct value tuple over all 1,445 such rows, asserted by
`tests/test_external.py::test_a_half_matched_fight_carries_exactly_one_value_tuple`.

## Point-in-time

Every `pre_ufc_*` column is cut at `history["date"] < first_ufc_date`, so it
summarises a window that closed before the fighter's UFC career began and is
therefore constant across all of their UFC fights. That constancy -- not the
truncation-invariance test, which cannot see a static artifact at all -- is
the checkable form of the claim, and
`tests/test_external.py::test_pre_ufc_values_are_constant_across_a_fighters_ufc_career`
asserts it on the real table. `scripts/build_external.py` carries the full
argument, including the zero-wins convention for an unrecorded opponent;
`src/mma/external.py` carries the join (by fighter id only, never by name) and
the missingness rules.

# `fight_notice.parquet` — short notice and missed weight

One row per CORNER of every UFC bout the snapshot's Bet MMA tables cover:
{n_notice_rows} rows over {n_notice_fights} fights, {notice_share:.3f} of our
{n_fights} fights ({window_share:.3f} of the {window_label} window the source spans).
{n_replacements} of those corners were late replacements (notice
{notice_lo:.0f}-{notice_hi:.0f} days) and {n_misses} missed weight.

| | |
|---|---|
| Source tables | `Bet MMA/late_replacements.csv`, `Bet MMA/missed_weights.csv`, `Bet MMA/bouts.csv`, `bout_mapping.csv`, `Bet MMA/fighters.csv` |
| Licence | MIT, same snapshot and commit as above |
| Coverage | {window_label} |

**Membership is the three-state boundary.** The two source lists name only the
fighters a thing happened to, so "no row" would otherwise be ambiguous between
"trained a full camp" and "nobody recorded it". Bet MMA's own bout list
resolves it: inside the bouts it covers, absence of a replacement row is an
observation; outside them, absence is ignorance. The derivation therefore emits
BOTH corners of a covered bout or neither, which also means the block carries
no per-corner missingness channel at all — the failure mode the `external`
section above describes. `src/mma/notice.py` carries the rule and
`tests/test_notice.py` checks it against the committed table.

**Corner assignment never uses names.** `bout_mapping.csv` gives our
`fight_id`, both Bet MMA fighter ids resolve to ufcstats ids through
`fighter_mapping.csv` and `Bet MMA/fighters.csv`, and the derivation asserts
the resulting pair is exactly the pair our own fights table records for that
fight; a bout where it is not is dropped whole.

**Point-in-time.** A replacement is booked and a weigh-in happens before the
bout, so both facts are known on fight morning, and neither is accumulated
across fights.
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
    notice = derive_notice(args.snapshot, fights)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out_dir / "fighter_external.parquet", index=False)
    notice.to_parquet(args.out_dir / "fight_notice.parquet", index=False)
    write_readme(args.out_dir / "README.md", table, commit, coverage_end, fighters,
                 notice, fights)

    matched = fighters["fighter_id"].isin(set(table["fighter_id"]))
    usable = drop_source_errors(table)
    matched_usable = fighters["fighter_id"].isin(set(usable["fighter_id"]))
    print(f"snapshot {commit[:10]} -> {args.out_dir / 'fighter_external.parquet'}")
    print(f"{len(table)} fighters committed, {len(usable)} usable; "
          f"coverage end {coverage_end:%Y-%m-%d}")
    print(f"match rate vs fighters.parquet: {int(matched.sum())}/{len(fighters)} "
          f"({matched.mean():.4f}) committed, {int(matched_usable.sum())}/{len(fighters)} "
          f"({matched_usable.mean():.4f}) usable")
    print("dropped at load (pro debut after first UFC bout, or a missing date): "
          f"{len(table) - len(usable)}")
    print(table.notna().mean().round(4).to_string())

    seen = fights["fight_id"].isin(set(notice["fight_id"]))
    print(f"\nfight_notice.parquet: {len(notice)} corner rows over "
          f"{notice['fight_id'].nunique()} fights "
          f"({seen.mean():.4f} of our {len(fights)} fights)")
    print(f"late replacements: {int(notice['notice_days'].notna().sum())}; "
          f"missed weight: {int(notice['missed_weight'].sum())}")


if __name__ == "__main__":
    main()
