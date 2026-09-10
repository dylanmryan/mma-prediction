"""Is the deployed recipe's walk-forward evidence older than the data it trains on?

The deployed models are refit through the latest event, and the weekly Action
retrains them every time new fights land. The walk-forward reports that
*justified* the recipe were run once, on the table as it stood that day. So
within a week of any decision the models train on fights no committed harness
report has ever seen, and they stay that way until something re-measures.

That gap is a true fact and worth saying. What it is NOT, on its own, is
actionable -- because `scripts/revalidate_recipe.py` exists precisely to close
it: it re-applies every recorded bar to the current table and writes
`models/walkforward/recipe_revalidation.json`. A passing re-validation whose
table reaches the training cutoff IS the missing measurement, even though each
member's own harness report is still older and still records the older table.

This module is the one place that read lives, so every consumer answers the
question the same way. There are four:

* the three refit scripts (`scripts/train_xgb.py`, `train_torch.py`,
  `train_hazard.py`), which print the warning as they train;
* `tests/test_processed_xgb.py`, which checks the committed metrics file's
  provenance and must not fail the suite -- and therefore the Action's commit
  and predict steps -- merely because a week has passed;
* `scripts/revalidate_recipe.py`'s own `--check-staleness`, the cheap weekly
  read, which imports `revalidation_cover` from here.

Each of those grew its own copy of the naive comparison, and each of them was
still firing after the 2026-09-09 re-validation had already answered it. A
standing warning that nobody can act on is how a real one gets scrolled past,
which is the whole reason the read is centralised rather than repeated.

Nothing here rewrites `harness_features_max_date`. That field records the
table a specific committed report ran on, and it did run on that table; what a
re-validation changes is whether the gap is actionable, not what happened.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: Where `scripts/revalidate_recipe.py` writes its verdict, relative to the
#: repository root. The read is by path rather than by import so that the
#: train scripts do not depend on the re-validation script itself.
REVALIDATION_PATH = Path("models") / "walkforward" / "recipe_revalidation.json"


def load_revalidation(root: Path) -> dict | None:
    """The committed re-validation artifact, or None when there is not one.

    A missing file is the ordinary state of a tree that has never been
    re-validated, not an error -- it is exactly the case the warning exists
    for -- so it is reported as absence rather than raised.
    """
    path = Path(root) / REVALIDATION_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def revalidation_cover(artifact: dict | None,
                       read_from: str | Path = REVALIDATION_PATH) -> dict | None:
    """What the committed re-validation covers, and whether it passed.

    `recipe_revalidation.json` is `scripts/revalidate_recipe.py`'s own output,
    which is what lets the cheap read answer its own question properly rather
    than by proxy. The question is whether the recipe's justification has been
    re-measured on the data the models train on, and a passing re-validation on
    a table that reaches the training cutoff IS that measurement.
    """
    if not artifact:
        return None
    table = artifact.get("table") or {}
    verdict_block = artifact.get("verdict") or {}
    return {
        "date": artifact.get("date"),
        "features_max_date": table.get("features_max_date"),
        "n_feature_rows": table.get("n_feature_rows"),
        "still_justified": bool(verdict_block.get("still_justified")),
        "bars_no_longer_met": list(verdict_block.get("bars_no_longer_met") or []),
        "read_from": Path(read_from).as_posix(),
    }


def covers_training_data(cover: dict | None, train_through: str) -> bool:
    """True when a re-validation both PASSED and reached the training cutoff.

    Both clauses matter, and for different reasons. A re-validation that ran on
    an older table has not seen the fights in question, so it says nothing
    about them. One that ran and *failed* has seen them and found a recorded
    bar no longer clearing -- which makes the gap more actionable, not less, so
    it must never be read as cover.
    """
    if not cover or not cover.get("still_justified"):
        return False
    reached = cover.get("features_max_date")
    if reached is None:
        return False
    return pd.Timestamp(reached) >= pd.Timestamp(train_through)


def stale_harness_warning(train_through: str, harness_features_max_date: str,
                          cover: dict | None = None) -> str | None:
    """Warning text when the refit trains on fights no harness report saw AND
    no passing re-validation covers them, else None.

    `cover` defaults to None -- "nothing has re-measured this" -- so a caller
    that does not pass one gets the plain gap check it always got. Callers that
    can read the artifact should pass `revalidation_cover(load_revalidation(root))`.

    Where it fires with a re-validation on disk it names that standing answer,
    following `scripts/check_snapshot_coverage.py`: the message is not "this is
    stale" but "this is stale; the standing answer was taken on table X, and
    here is why it does not settle this".
    """
    if pd.Timestamp(train_through) <= pd.Timestamp(harness_features_max_date):
        return None
    if covers_training_data(cover, train_through):
        return None
    text = (f"WARNING: harness evidence predates this training data (harness "
            f"features_max_date {harness_features_max_date} < train_through "
            f"{train_through}); re-run scripts/run_walkforward.py to refresh")
    if cover is None:
        return text
    if not cover["still_justified"]:
        return (
            text + f"\n  The last re-validation ({cover['date']}, table through "
            f"{cover['features_max_date']}) RAN AND FAILED: "
            f"{', '.join(cover['bars_no_longer_met']) or 'see the artifact'} "
            f"no longer clear. See {cover['read_from']}; this is a human call."
        )
    return (
        text + f"\n  The last re-validation ({cover['date']}) passed but reached only "
        f"{cover['features_max_date']}, short of {train_through}, so it does not "
        f"cover this training data. See {cover['read_from']}; re-run "
        "`python scripts/revalidate_recipe.py`."
    )
