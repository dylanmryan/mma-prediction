from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists(),
    reason="features not built (run scripts/build_features.py)",
)


def test_target_is_balanced_after_symmetrization():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    # raw red-corner win rate is ~0.646; symmetrization must land near 0.5
    assert 0.47 < features["y_winner"].mean() < 0.53


def test_row_count_matches_decisive_fights():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert len(features) == (fights["winner"].isin(["a", "b"])).sum()


def test_no_leakage_truncation_invariance():
    """Features for old fights must not change when future fights exist."""
    from mma.features import build_features
    from mma.history import build_history

    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")

    cutoff = "2015-01-01"
    old_fights = fights[fights["date"] < cutoff]
    truncated = build_features(
        old_fights, fighters, ratings, build_history(old_fights, stats, ratings)
    )
    full = pd.read_parquet(PROCESSED / "features.parquet")
    full_old = full[full["fight_id"].isin(truncated["fight_id"])]

    merged = truncated.sort_values("fight_id").reset_index(drop=True)
    full_sorted = full_old.sort_values("fight_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, full_sorted, check_like=True, check_dtype=False)


def test_finish_round_target_classes():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    finishes = features[features["y_method"].isin(["ko_tko", "submission"])]
    assert set(finishes["y_finish_round"].dropna()) <= {"1", "2", "3", "45"}
    assert finishes["y_finish_round"].notna().all()
