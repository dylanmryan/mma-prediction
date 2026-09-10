"""The shared staleness read: when the harness gap is worth saying, and when not.

`mma.staleness` exists to stop five copies of one naive comparison from each
warning about a gap that a committed re-validation had already closed. The
risk in silencing a warning is silencing a real one, so the cases below pin
both directions -- and the last one pins the committed tree itself, so that
when it genuinely does go stale the suite says so instead of staying quiet.
"""
import json
from pathlib import Path

import pytest

from mma.staleness import (
    REVALIDATION_PATH,
    covers_training_data,
    load_revalidation,
    revalidation_cover,
    stale_harness_warning,
)

ROOT = Path(__file__).resolve().parents[1]

PASSED = {
    "date": "2026-09-09",
    "table": {"features_max_date": "2026-09-05", "n_feature_rows": 11290},
    "verdict": {"still_justified": True, "bars_no_longer_met": []},
}
FAILED = {
    "date": "2026-09-09",
    "table": {"features_max_date": "2026-09-05", "n_feature_rows": 11290},
    "verdict": {"still_justified": False, "bars_no_longer_met": ["sp3_joint_bar"]},
}
OLDER = {
    "date": "2026-08-20",
    "table": {"features_max_date": "2026-08-15", "n_feature_rows": 11250},
    "verdict": {"still_justified": True, "bars_no_longer_met": []},
}


# --- the plain gap, unchanged from the copies this replaced -------------------

def test_no_gap_is_no_warning():
    assert stale_harness_warning("2026-08-08", "2026-08-08") is None
    assert stale_harness_warning("2026-08-01", "2026-08-08") is None


def test_a_gap_with_nothing_re_measuring_it_warns():
    warning = stale_harness_warning("2026-09-05", "2026-08-08")
    assert warning is not None
    assert "harness evidence predates this training data" in warning
    # with no artifact to name there is no standing answer to report
    assert "re-validation" not in warning


# --- what a re-validation does, and does not, settle --------------------------

def test_a_passing_revalidation_over_the_training_data_says_nothing():
    cover = revalidation_cover(PASSED)
    assert covers_training_data(cover, "2026-09-05") is True
    assert stale_harness_warning("2026-09-05", "2026-08-08", cover) is None


def test_a_passing_revalidation_on_an_older_table_still_warns_and_names_itself():
    cover = revalidation_cover(OLDER)
    assert covers_training_data(cover, "2026-09-05") is False
    warning = stale_harness_warning("2026-09-05", "2026-08-08", cover)
    assert warning is not None
    # it has to say WHY the standing answer does not settle this
    assert "2026-08-15" in warning and "short of 2026-09-05" in warning
    assert REVALIDATION_PATH.as_posix() in warning


def test_a_failing_revalidation_never_clears_the_gap_and_says_it_failed():
    cover = revalidation_cover(FAILED)
    assert covers_training_data(cover, "2026-09-05") is False
    warning = stale_harness_warning("2026-09-05", "2026-08-08", cover)
    assert warning is not None
    assert "RAN AND FAILED" in warning
    assert "sp3_joint_bar" in warning


def test_a_failing_revalidation_is_not_read_as_cover_even_when_it_reaches_further():
    """Reaching past the cutoff is the weaker clause; passing is the binding one."""
    assert covers_training_data(revalidation_cover(FAILED), "2026-08-01") is False


def test_a_gap_beats_a_passing_revalidation_that_has_no_table_date():
    cover = revalidation_cover({"date": "2026-09-09", "table": {},
                                "verdict": {"still_justified": True}})
    assert covers_training_data(cover, "2026-09-05") is False


# --- reading the artifact off disk --------------------------------------------

def test_absent_artifact_reads_as_absence_not_an_error(tmp_path):
    assert load_revalidation(tmp_path) is None
    assert revalidation_cover(None) is None


def test_load_reads_the_artifact_the_revalidation_script_writes(tmp_path):
    path = tmp_path / REVALIDATION_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(PASSED))
    cover = revalidation_cover(load_revalidation(tmp_path))
    assert cover["date"] == "2026-09-09"
    assert cover["features_max_date"] == "2026-09-05"
    assert cover["still_justified"] is True
    assert cover["read_from"].endswith("recipe_revalidation.json")


# --- the committed tree, which is why the suite is quiet ----------------------

METRICS = [
    ROOT / "models" / "xgb_metrics_val.json",
    ROOT / "models" / "torch" / "metrics_val.json",
]


@pytest.mark.parametrize("metrics_path", METRICS, ids=lambda p: p.parent.name + "/" + p.name)
def test_the_committed_tree_is_covered_so_the_provenance_tests_are_quiet(metrics_path):
    """The provenance tests warn through this read, so silence has to be earned.

    If this fails, the tree has genuinely run ahead of its evidence and the
    answer is to re-run `python scripts/revalidate_recipe.py` -- not to relax
    the read. That is the whole point of pinning it here rather than letting
    the warning quietly stop meaning anything.
    """
    if not metrics_path.exists():
        pytest.skip(f"{metrics_path.name} not present (models not trained)")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("mode") != "refit_through":
        pytest.skip("split-mode metrics carry their own held-out numbers")
    cover = revalidation_cover(load_revalidation(ROOT))
    assert cover is not None, (
        "no re-validation artifact on disk; run scripts/revalidate_recipe.py"
    )
    assert covers_training_data(cover, metrics["train_through"]), (
        f"the deployed model trains through {metrics['train_through']} but the last "
        f"re-validation ({cover['date']}) reached {cover['features_max_date']} and "
        f"{'passed' if cover['still_justified'] else 'FAILED'}; "
        "run scripts/revalidate_recipe.py"
    )
