from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
XGB_WINNER = ROOT / "models" / "xgb_winner.json"

pytestmark = pytest.mark.skipif(
    not XGB_WINNER.exists(), reason="xgb_winner artifact not built"
)


@pytest.fixture(scope="module")
def booster():
    from mma.explain import load_booster

    return load_booster()


@pytest.fixture(scope="module")
def matchup():
    """A real build_matchup() output for a clearly-mismatched pair, both orderings."""
    from mma.inference import build_matchup

    snapshot = pd.Series(
        {
            "career_fights": 10, "career_wins": 8.0, "career_win_rate": 0.8,
            "career_finish_rate": 0.5, "kd_pf": 0.4, "sub_att_pf": 0.5,
            "td_landed_pf": 1.5, "td_acc": 0.5, "td_def": 0.7, "sig_pm": 4.5,
            "sig_absorbed_pm": 3.0, "ctrl_share": 0.2, "streak": 3,
            "last5_win_rate": 0.8, "last5_avg_opp_elo": 1550.0,
            "elo_overall": 1600.0, "elo_striking": 1580.0,
            "elo_grappling": 1570.0, "last_date": pd.Timestamp("2025-06-01"),
        }
    )
    weaker = snapshot.copy()
    weaker["elo_overall"] = 1450.0
    weaker["career_win_rate"] = 0.4
    weaker["career_wins"] = 4.0
    weaker["streak"] = -2
    bio = pd.Series(
        {"dob": pd.Timestamp("1993-01-01"), "height_cm": 180.0,
         "reach_cm": 185.0, "stance": "Orthodox"}
    )
    matchup_ab = build_matchup(
        snapshot, weaker, bio, bio, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    )
    matchup_ba = build_matchup(
        weaker, snapshot, bio, bio, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    )
    return matchup_ab, matchup_ba


def test_additivity_holds_both_orientations(booster, matchup):
    from mma.explain import raw_contributions

    matchup_ab, matchup_ba = matchup
    for frame in (matchup_ab, matchup_ba):
        values, bias, logit = raw_contributions(booster, frame)
        assert bias + values.sum() == pytest.approx(logit, abs=1e-3)


def test_contributions_excludes_bias_and_sorted_by_magnitude(booster, matchup):
    from mma.explain import contributions

    matchup_ab, matchup_ba = matchup
    result = contributions(matchup_ab, matchup_ba, booster=booster)
    assert "bias" not in result.index
    magnitudes = result.abs().to_numpy()
    assert (magnitudes[:-1] >= magnitudes[1:]).all()


def test_symmetry_flips_sign_and_preserves_magnitude(booster, matchup):
    from mma.explain import contributions

    matchup_ab, matchup_ba = matchup
    forward = contributions(matchup_ab, matchup_ba, booster=booster)
    reverse = contributions(matchup_ba, matchup_ab, booster=booster)
    # Same features, opposite sign, identical magnitude.
    forward_sorted = forward.sort_index()
    reverse_sorted = reverse.sort_index()
    pd.testing.assert_series_equal(
        forward_sorted, -reverse_sorted, check_exact=False, atol=1e-9
    )
    pd.testing.assert_series_equal(
        forward_sorted.abs(), reverse_sorted.abs(), check_exact=False, atol=1e-9
    )


def test_the_favored_fighter_has_a_positive_net_contribution(booster, matchup):
    from mma.explain import contributions

    matchup_ab, matchup_ba = matchup
    result = contributions(matchup_ab, matchup_ba, booster=booster)
    # Fighter A (the stronger snapshot) should be net-favored by the model.
    assert result.sum() > 0


def test_humanize_coverage_matches_booster_feature_names(booster):
    from mma.explain import FEATURE_LABELS

    missing = set(booster.feature_names) - set(FEATURE_LABELS)
    assert missing == set(), f"no label for features: {missing}"


def test_humanize_returns_top_n_rows_with_expected_shape(booster, matchup):
    from mma.explain import contributions, humanize

    matchup_ab, matchup_ba = matchup
    contribs = contributions(matchup_ab, matchup_ba, booster=booster)
    rows = humanize(contribs, "Fighter A", "Fighter B", top_n=6)
    assert len(rows) == 6
    for row in rows:
        assert set(row) == {"label", "contribution", "favors", "strength"}
        assert isinstance(row["label"], str)
        assert isinstance(row["contribution"], float)
        assert row["favors"] in {"Fighter A", "Fighter B"}
        assert row["strength"] in {"strong", "moderate", "slight"}
        expected_favor = "Fighter A" if row["contribution"] >= 0 else "Fighter B"
        assert row["favors"] == expected_favor


def test_humanize_strength_thresholds():
    from mma.explain import humanize

    contribs = pd.Series(
        {"elo_diff": 0.5, "reach_diff": -0.3001, "age_diff": 0.2, "height_diff": 0.1001,
         "career_wins_diff": 0.05, "streak_diff": -0.01},
    )
    rows = humanize(contribs, "A", "B", top_n=6)
    by_label = {row["label"]: row["strength"] for row in rows}
    assert by_label["Elo rating edge"] == "strong"
    assert by_label["Reach advantage"] == "strong"
    assert by_label["Age gap"] == "moderate"
    assert by_label["Height advantage"] == "moderate"
    assert by_label["Career wins edge"] == "slight"
    assert by_label["Recent win/loss streak"] == "slight"


def test_humanize_top_n_respects_smaller_series():
    from mma.explain import humanize

    contribs = pd.Series({"elo_diff": 0.2, "reach_diff": -0.1})
    rows = humanize(contribs, "A", "B", top_n=6)
    assert len(rows) == 2
