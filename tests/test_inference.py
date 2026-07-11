from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "torch" / "metrics_val.json").exists(),
    reason="ensemble artifacts not built",
)


@pytest.fixture(scope="module")
def ensemble():
    from mma.inference import Ensemble
    return Ensemble.load()


@pytest.fixture(scope="module")
def matchup(ensemble):
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
    return build_matchup(
        snapshot, weaker, bio, bio, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    ), build_matchup(
        weaker, snapshot, bio, bio, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    )


def test_feature_contract_complete(ensemble, matchup):
    frame, _ = matchup
    missing = set(ensemble.preprocessor.numeric_columns) - set(frame.columns)
    assert missing == set()


def test_stronger_fighter_favored_and_symmetric(ensemble, matchup):
    forward, reverse = matchup
    p_forward = ensemble.predict(forward)["winner_prob"][0]
    p_reverse = ensemble.predict(reverse)["winner_prob"][0]
    assert p_forward > 0.5
    assert p_forward + p_reverse == pytest.approx(1.0, abs=0.08)


def test_round_45_zero_for_three_round_fight(ensemble, matchup):
    frame, _ = matchup
    result = ensemble.predict(frame)
    assert result["round_probs"][0, 3] == pytest.approx(0.0, abs=1e-6)
    assert result["method_probs"][0].sum() == pytest.approx(1.0, abs=1e-5)


def test_predict_symmetrized_is_exactly_self_consistent(ensemble, matchup):
    from mma.inference import predict_symmetrized

    forward, reverse = matchup
    result_fwd = predict_symmetrized(ensemble, forward, reverse)
    result_rev = predict_symmetrized(ensemble, reverse, forward)
    assert result_fwd["winner_prob"] + result_rev["winner_prob"] == pytest.approx(1.0, abs=1e-9)
    assert result_fwd["winner_prob"] > 0.5
    assert result_fwd["winner_spread"] >= 0.0
    assert result_fwd["method_probs"].sum() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(
        result_fwd["method_probs"], result_rev["method_probs"], atol=1e-6
    )
    np.testing.assert_allclose(
        result_fwd["round_probs"], result_rev["round_probs"], atol=1e-6
    )


def test_apply_prior_correction_matches_hand_computed_example():
    from mma.inference import apply_prior_correction

    probs = {"a": 0.7, "b": 0.3}
    priors = {"a": 0.1, "b": 0.9}
    corrected = apply_prior_correction(probs, priors)
    assert corrected["a"] == pytest.approx(0.2059, abs=1e-4)
    assert corrected["b"] == pytest.approx(0.7941, abs=1e-4)
    assert sum(corrected.values()) == pytest.approx(1.0, abs=1e-9)


def test_apply_prior_correction_guards_zero_sum():
    from mma.inference import apply_prior_correction

    probs = {"a": 0.7, "b": 0.3}
    priors = {"a": 0.0, "b": 0.0}
    assert apply_prior_correction(probs, priors) == probs


def test_compute_correction_factors_hand_computed():
    from mma.inference import compute_correction_factors

    empirical = {"a": 0.2, "b": 0.8}
    mean_predicted = {"a": 0.5, "b": 0.5}
    factors = compute_correction_factors(empirical, mean_predicted)
    assert factors["a"] == pytest.approx(0.4)
    assert factors["b"] == pytest.approx(1.6)


def test_compute_correction_factors_mean_matches_aggregate():
    # The defining property: applying the factors to the model's mean
    # predicted distribution recovers the empirical prior exactly.
    from mma.inference import apply_prior_correction, compute_correction_factors

    empirical = {"1": 0.4, "2": 0.25, "3": 0.17, "45": 0.18}
    mean_predicted = {"1": 0.07, "2": 0.08, "3": 0.09, "45": 0.76}
    factors = compute_correction_factors(empirical, mean_predicted)
    recovered = apply_prior_correction(mean_predicted, factors)
    for cls, value in empirical.items():
        assert recovered[cls] == pytest.approx(value, abs=1e-9)


def test_compute_correction_factors_guards_tiny_mean_predicted():
    from mma.inference import compute_correction_factors

    empirical = {"a": 0.5, "b": 0.5, "c": 0.0}
    mean_predicted = {"a": 0.999999, "b": 1e-9, "c": 1e-9}
    factors = compute_correction_factors(empirical, mean_predicted)
    assert factors["b"] == 0.0  # capped instead of exploding
    assert factors["c"] == 0.0


def test_committed_display_priors_json_structure():
    import json

    payload = json.loads(
        (ROOT / "models" / "torch" / "display_priors.json").read_text()
    )
    assert set(payload) == {"method", "round_3", "round_5"}
    assert set(payload["method"]) == {"ko_tko", "submission", "decision"}
    for key in ("round_3", "round_5"):
        assert set(payload[key]) == {"1", "2", "3", "45"}
    assert payload["round_3"]["45"] == 0.0
    assert all(v >= 0.0 for group in payload.values() for v in group.values())


def test_compute_display_priors_synthetic_frame():
    from mma.inference import compute_display_priors

    # Two train-split rows (date < 2021-01-01) and one post-cutoff row that
    # must be excluded from the priors entirely.
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-01-01", "2019-06-01", "2020-01-01", "2020-06-01", "2022-01-01"]
            ),
            "y_method": ["ko_tko", "ko_tko", "submission", "decision", "ko_tko"],
            "y_finish_round": ["1", "2", "1", pd.NA, "45"],
            "scheduled_rounds": pd.array([3, 3, 5, 3, 5], dtype="Int64"),
        }
    )
    priors = compute_display_priors(features)

    # Method prior uses all 4 train rows (2019-2020), post-cutoff row excluded.
    assert priors["method"]["ko_tko"] == pytest.approx(0.5)
    assert priors["method"]["submission"] == pytest.approx(0.25)
    assert priors["method"]["decision"] == pytest.approx(0.25)
    assert sum(priors["method"].values()) == pytest.approx(1.0)

    # 3-round fights can never finish in rounds 4-5.
    assert priors["round_3"]["45"] == 0.0
    assert sum(priors["round_3"].values()) == pytest.approx(1.0)

    # round_5 prior computed only from the single 5-round train finish.
    assert priors["round_5"]["1"] == pytest.approx(1.0)
    assert sum(priors["round_5"].values()) == pytest.approx(1.0)


def test_mc_dropout_preserves_batchnorm(ensemble, matchup):
    frame, _ = matchup
    net = ensemble.nets[0]
    before = {
        name: buffer.clone()
        for name, buffer in net.named_buffers()
    }
    samples = ensemble.mc_dropout(frame, passes=25)
    assert samples.shape == (25, 1)
    assert samples.std() > 0.0
    for name, buffer in net.named_buffers():
        assert torch.equal(before[name], buffer), f"buffer {name} mutated"
    assert not net.training
