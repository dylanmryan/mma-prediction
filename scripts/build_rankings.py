"""Derive the committed official-rankings table from an open CC0 dataset.

    python scripts/build_rankings.py            # kagglehub-downloads the source
    python scripts/build_rankings.py --csv /path/to/UFC_rankings_history.csv

Source: Kaggle `jerzyszocik/ufc-rankings-history` ("UFC Rankings History
(2013-ongoing)", CC0: Public Domain), a weekly snapshot of the UFC's own
divisional rankings: `date, weightclass, fighter, rank`, champion = rank 0.
Downloaded through `kagglehub`, which the project already depends on for the
primary fights dataset. The raw CSV (~4 MB) is not committed; the derived
`data/external/rankings.parquet` is.

WHY THIS SOURCE RATHER THAN THE GITHUB MIRROR: `martj42/ufc_rankings_history`
carries the same columns but no LICENSE file at all, which makes redistributing
anything derived from it a guess. This one states CC0 on the dataset itself and
is refreshed weekly, so unlike the `external` block's static snapshot its
coverage does not decay away from the fights we actually serve.

WHAT IS DERIVED, keyed by OUR ufcstats `fighter_id`:
  fighter_id, date, weight_class, rank

MATCHING IS BY NAME, because the source has nothing else -- so it uses the
prospective pipeline's never-guess matcher (`mma.prospective.match_fighter_id`:
exact unicode-normalised, then accent-folded, and a name that is ambiguous
under either tier is UNMATCHED rather than assigned). A name that does not
match is dropped and counted; the fighter then reads as "not ranked", which is
the same state as a genuinely unranked fighter. That conflation is the honest
cost of a name-keyed source and is reported here and in the README rather than
papered over with a fuzzy match.

POUND-FOR-POUND ROWS ARE DROPPED. They are not a division a fight happens in,
they duplicate fighters already ranked in their own division, and the source
spells them four different ways over the years. Only the twelve divisional
rankings survive, and their names are already exactly our `weight_class`
values.

POINT-IN-TIME: nothing here is as-of-scrape. Each row carries the date the
ranking was PUBLISHED, and `mma.rankings` joins the most recent publication
STRICTLY BEFORE each fight date, so a fight can never see the ranking its own
result produced. That is the whole safety argument for this block, and
`tests/test_rankings.py::test_the_ranking_used_is_the_last_one_published_before_the_fight`
asserts it directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.prospective import build_name_index, match_fighter_id  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "external"
DATASET = "jerzyszocik/ufc-rankings-history"
SOURCE_URL = f"https://www.kaggle.com/datasets/{DATASET}"
LICENCE = "CC0: Public Domain"
# Everything the source publishes that is not a division a bout takes place in.
# Spelled four ways across the years ("Men's Pound-for-Pound", "...Top Rank",
# "...PoundTop Rank", bare "Pound-for-Pound"), so the test is a substring.
NOT_A_DIVISION = "pound-for-pound"


def download_csv() -> Path:
    """kagglehub-download the rankings CSV (public dataset, no credentials)."""
    import kagglehub

    cache = Path(kagglehub.dataset_download(DATASET))
    candidates = sorted(cache.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no CSV found in kagglehub cache {cache}")
    return candidates[0]


def derive(raw: pd.DataFrame, fighters: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(rankings keyed by our fighter_id, a match report).

    The report carries the counts the README and the SP2 notes quote: how many
    distinct names the source has, how many matched at each tier, and which
    ones did not.
    """
    rows = raw.dropna(subset=["fighter", "weightclass", "date", "rank"]).copy()
    rows["date"] = pd.to_datetime(rows["date"])
    divisional = ~rows["weightclass"].str.lower().str.contains(NOT_A_DIVISION)
    rows = rows[divisional]

    index = build_name_index(fighters)
    names = sorted(rows["fighter"].unique())
    matched: dict[str, str] = {}
    tiers: dict[str, str] = {}
    unmatched: dict[str, str] = {}
    for name in names:
        fighter_id, tier, reason = match_fighter_id(name, index)
        if fighter_id is None:
            unmatched[name] = reason
            continue
        matched[name] = fighter_id
        tiers[name] = tier

    rows["fighter_id"] = rows["fighter"].map(matched)
    kept = rows[rows["fighter_id"].notna()]
    table = (
        kept.rename(columns={"weightclass": "weight_class"})[
            ["fighter_id", "date", "weight_class", "rank"]
        ]
        .drop_duplicates(["fighter_id", "date", "weight_class"])
        .sort_values(["fighter_id", "weight_class", "date"])
        .reset_index(drop=True)
    )
    table["rank"] = table["rank"].astype("int16")
    table["weight_class"] = table["weight_class"].astype("string")
    table["fighter_id"] = table["fighter_id"].astype("string")

    report = {
        "n_names": len(names),
        "n_matched": len(matched),
        "n_exact": sum(1 for t in tiers.values() if t == "exact"),
        "n_accent_folded": sum(1 for t in tiers.values() if t == "accent_folded"),
        "unmatched": unmatched,
        "row_match_rate": float(rows["fighter_id"].notna().mean()),
        "n_rows_in": int(len(rows)),
        "n_rows_out": int(len(table)),
        "weight_classes": sorted(table["weight_class"].dropna().unique().tolist()),
        "date_min": table["date"].min(),
        "date_max": table["date"].max(),
        "n_snapshots": int(table["date"].nunique()),
    }
    return table, report


def write_readme(path: Path, table: pd.DataFrame, report: dict,
                 fighters: pd.DataFrame, version: str) -> None:
    ranked = fighters["fighter_id"].isin(set(table["fighter_id"]))
    unmatched = "\n".join(
        f"| `{name}` | {reason} |" for name, reason in sorted(report["unmatched"].items())
    )
    path.write_text(f"""# `data/external/rankings.parquet` — official UFC divisional rankings

| | |
|---|---|
| Source | <{SOURCE_URL}> |
| Licence | {LICENCE} |
| Dataset version | {version} |
| Coverage | {report['date_min']:%Y-%m-%d} .. {report['date_max']:%Y-%m-%d} ({report['n_snapshots']} weekly snapshots) |
| Rows | {len(table)} (from {report['n_rows_in']} divisional source rows) |
| Divisions | {len(report['weight_classes'])}, named exactly as our `weight_class` |
| Fighters | {table['fighter_id'].nunique()} of our {len(fighters)} ({ranked.mean():.4f}) appear at least once |

Rank 0 is the champion. Pound-for-pound rows are dropped: they are not a
division a bout takes place in, they re-list fighters already ranked in their
own division, and the source spells them four different ways.

Regenerate with `python scripts/build_rankings.py`; the raw CSV is
kagglehub-downloaded and never committed.

## Name matching, and what it costs

The source identifies fighters by NAME only, so this is the one derived table
that cannot be joined by id. It uses the prospective pipeline's never-guess
matcher (`mma.prospective.match_fighter_id`): exact unicode-normalised match,
then accent-folded, and anything ambiguous under either tier is left unmatched.

**{report['n_matched']} of {report['n_names']} distinct names matched**
({report['n_exact']} exact, {report['n_accent_folded']} accent-folded), covering
{report['row_match_rate']:.4f} of the divisional source rows.

An unmatched name is dropped, so that fighter reads as *unranked* — the same
state as someone genuinely outside the top 15. That conflation is the honest
cost of a name-keyed source; it is not repaired with a fuzzy match, because a
wrong id is worse than a missing one. The names it affects are ring names the
UFC uses and our fights table does not ("Rampage Jackson", "Cris Cyborg",
"Minotauro Nogueira"), spellings ("Georges St. Pierre" vs "Georges St-Pierre"),
and names shared by two of our fighters, which are ambiguous by construction:

| name | why it did not match |
|---|---|
{unmatched}

Three of these are mechanical near-misses rather than genuinely different
people: `Jan Błachowicz` / `Jan Blachowicz` and `Klaudia Syguła` /
`Klaudia Sygula` differ by a stroked Latin letter, which NFKD does not
decompose the way it decomposes an accent, and `Lone’er Kavanagh` /
`Lone'er Kavanagh` differ by a curly versus a straight apostrophe. Extending
`mma.prospective.fold_accents` with a stroked-letter map and apostrophe
normalisation would close all three without weakening the never-guess rule
(a folded match must still be unique). It is left as a follow-up rather than
done here, because `fold_accents` is on the live prediction path and this
table is not.

## Point-in-time

Every row carries the date the ranking was PUBLISHED. `mma.rankings` joins the
most recent publication **strictly before** each fight's date
(`pandas.merge_asof(..., allow_exact_matches=False)`), so a fight can never see
a ranking that its own result produced. Unlike the pre-UFC snapshot next door,
this source is refreshed weekly, so its coverage does not decay away from the
fights the deployed model actually serves.
""")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=None,
                        help="path to UFC_rankings_history.csv (downloaded if absent)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    csv = args.csv or download_csv()
    raw = pd.read_csv(csv)
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")

    table, report = derive(raw, fighters)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out_dir / "rankings.parquet", index=False)
    version = csv.parent.name if csv.parent.name.isdigit() else "local"
    write_readme(args.out_dir / "RANKINGS.md", table, report, fighters, version)

    print(f"{csv} -> {args.out_dir / 'rankings.parquet'}")
    print(f"{len(table)} rows, {report['n_snapshots']} weekly snapshots, "
          f"{report['date_min']:%Y-%m-%d}..{report['date_max']:%Y-%m-%d}")
    print(f"name match: {report['n_matched']}/{report['n_names']} names "
          f"({report['n_exact']} exact, {report['n_accent_folded']} accent-folded); "
          f"{report['row_match_rate']:.4f} of rows")
    print("unmatched names:", ", ".join(sorted(report["unmatched"])) or "none")


if __name__ == "__main__":
    main()
