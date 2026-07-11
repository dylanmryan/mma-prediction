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
