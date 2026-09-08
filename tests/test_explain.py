from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not list((ROOT / "models").glob("xgb_winner_seed*.json")),
    reason="xgb winner seed ensemble not built (run scripts/train_xgb.py)",
)


@pytest.fixture(scope="module")
def boosters():
    """Every seed of the deployed winner head -- the model the blend averages."""
    from mma.explain import load_boosters

    return load_boosters()


@pytest.fixture(scope="module")
def booster(boosters):
    return boosters[0]


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
            # the blocks restored for SP2.1: a served snapshot carries every
            # field they read, exactly as `mma.snapshots.build_snapshots` emits
            "first_date": pd.Timestamp("2019-03-01"),
            "glicko_mu": 1610.0, "glicko_phi": 90.0, "glicko_sigma": 0.06,
            "elo_delta_3": 24.0, "elo_delta_5": 31.0,
            "elo_peak_minus_current": 12.0, "bonus_rate": 0.3,
            "sig_pm_vs_exp": 0.8, "sig_absorbed_pm_vs_exp": -0.4,
            "td_landed_pf_vs_exp": 0.5, "td_def_vs_exp": 0.1,
            "ctrl_share_vs_exp": 0.05,
            "avg_opp_elo_wins": 1520.0, "avg_opp_elo_losses": 1610.0,
        }
    )
    weaker = snapshot.copy()
    weaker["elo_overall"] = 1450.0
    weaker["career_win_rate"] = 0.4
    weaker["career_wins"] = 4.0
    weaker["streak"] = -2
    # The bio row's index label is the ufcstats id the `external` block joins
    # on, so the two corners get real ids: one the snapshot maps (A) and one it
    # does not (B), which is also the served shape of every post-snapshot
    # debutant -- NaN differentials plus the flag, not a crash.
    bio = pd.Series(
        {"dob": pd.Timestamp("1993-01-01"), "height_cm": 180.0,
         "reach_cm": 185.0, "stance": "Orthodox"},
        name="002ca196477ce572",
    )
    bio_unmapped = pd.Series(bio, name="000774e57404d8c7")
    matchup_ab = build_matchup(
        snapshot, weaker, bio, bio_unmapped, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    )
    matchup_ba = build_matchup(
        weaker, snapshot, bio_unmapped, bio, "Lightweight", False, 3,
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
    result = contributions(matchup_ab, matchup_ba, boosters=booster)
    assert "bias" not in result.index
    magnitudes = result.abs().to_numpy()
    assert (magnitudes[:-1] >= magnitudes[1:]).all()


def test_symmetry_flips_sign_and_preserves_magnitude(booster, matchup):
    from mma.explain import contributions

    matchup_ab, matchup_ba = matchup
    forward = contributions(matchup_ab, matchup_ba, boosters=booster)
    reverse = contributions(matchup_ba, matchup_ab, boosters=booster)
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
    result = contributions(matchup_ab, matchup_ba, boosters=booster)
    # Fighter A (the stronger snapshot) should be net-favored by the model.
    assert result.sum() > 0


def test_humanize_coverage_matches_booster_feature_names(boosters):
    from mma.explain import FEATURE_LABELS

    for booster in boosters:
        missing = set(booster.feature_names) - set(FEATURE_LABELS)
        assert missing == set(), f"no label for features: {missing}"


def test_load_boosters_returns_the_whole_seed_ensemble_in_seed_order(boosters):
    """The blend averages five boosters, so explaining one of them would be
    explaining a model that does not exist."""
    from mma.explain import winner_paths
    from scripts.train_xgb import SEEDS

    assert len(boosters) == len(SEEDS) == 5
    assert [int(p.stem.rsplit("seed", 1)[1]) for p in winner_paths()] == list(SEEDS)


def test_load_boosters_raises_rather_than_explaining_a_subset(tmp_path):
    from mma.explain import load_boosters

    with pytest.raises(FileNotFoundError, match="xgb_winner_seed"):
        load_boosters(tmp_path)


def test_contributions_average_over_seeds_not_over_one_member(matchup, boosters):
    """The default explanation is the mean over the ensemble; asking for one
    booster gives that booster's own, and the two differ."""
    from mma.explain import contributions

    matchup_ab, matchup_ba = matchup
    ensemble = contributions(matchup_ab, matchup_ba)
    per_seed = [contributions(matchup_ab, matchup_ba, boosters=b) for b in boosters]
    stacked = pd.concat(per_seed, axis=1)
    expected = stacked.mean(axis=1).reindex(ensemble.index)
    pd.testing.assert_series_equal(ensemble, expected, check_exact=False, atol=1e-12)
    assert not ensemble.equals(per_seed[0].reindex(ensemble.index))


def test_symmetrized_ensemble_contributions_sum_toward_the_favored_fighter(matchup):
    from mma.explain import contributions

    matchup_ab, matchup_ba = matchup
    assert contributions(matchup_ab, matchup_ba).sum() > 0


def test_humanize_returns_top_n_rows_with_expected_shape(booster, matchup):
    from mma.explain import contributions, humanize

    matchup_ab, matchup_ba = matchup
    contribs = contributions(matchup_ab, matchup_ba, boosters=booster)
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
