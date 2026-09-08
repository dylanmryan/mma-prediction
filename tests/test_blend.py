"""The blend's arithmetic, and that the SERVED blend is the MEASURED one.

`mma.blend` exists so `mma.candidates.BlendCandidate` (which measured B1 on the
walk-forward harness) and `mma.inference.BlendedPredictor` (which serves it)
run the same three operations rather than two implementations that could
drift. These tests pin the operations, and pin the two deployment constants to
the committed harness report they were derived from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma.blend import HEADS, LOGIT_EPS, apply_temperature, blend_heads, logit, mask_round_45

ROOT = Path(__file__).resolve().parents[1]


# --- the operations ---------------------------------------------------------


def test_blend_at_weight_one_is_the_xgb_member_and_at_zero_the_torch_member():
    """The degenerate weights are the correctness check for everything else."""
    rng = np.random.default_rng(0)
    xgb_pred = {"winner": rng.uniform(size=7), "method": rng.uniform(size=(7, 3)),
                "round": rng.uniform(size=(7, 4))}
    torch_pred = {"winner": rng.uniform(size=7), "method": rng.uniform(size=(7, 3)),
                  "round": rng.uniform(size=(7, 4))}
    at_one = blend_heads(xgb_pred, torch_pred, 1.0)
    at_zero = blend_heads(xgb_pred, torch_pred, 0.0)
    for head in HEADS:
        np.testing.assert_array_equal(at_one[head], xgb_pred[head])
        np.testing.assert_array_equal(at_zero[head], torch_pred[head])


def test_blend_at_half_is_the_plain_average():
    xgb_pred = {"winner": np.array([0.2, 0.8]), "method": np.zeros((2, 3)),
                "round": np.zeros((2, 4))}
    torch_pred = {"winner": np.array([0.4, 0.4]), "method": np.ones((2, 3)),
                  "round": np.ones((2, 4))}
    out = blend_heads(xgb_pred, torch_pred, 0.5)
    np.testing.assert_allclose(out["winner"], [0.3, 0.6])
    np.testing.assert_allclose(out["method"], np.full((2, 3), 0.5))


def test_blend_rejects_a_weight_outside_the_unit_interval():
    pred = {head: np.zeros((1, 4)) for head in HEADS}
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            blend_heads(pred, pred, bad)


def test_temperature_of_one_is_the_identity():
    p = np.array([0.01, 0.3, 0.5, 0.77, 0.99])
    np.testing.assert_allclose(apply_temperature(p, 1.0), p, atol=1e-12)


def test_temperature_below_one_sharpens_and_above_one_softens():
    p = np.array([0.2, 0.5, 0.8])
    sharp = apply_temperature(p, 0.8)   # the deployed direction
    soft = apply_temperature(p, 1.25)
    assert sharp[0] < p[0] and sharp[2] > p[2]
    assert soft[0] > p[0] and soft[2] < p[2]
    np.testing.assert_allclose([sharp[1], soft[1]], [0.5, 0.5], atol=1e-12)


def test_temperature_is_monotone_and_never_returns_exactly_zero_or_one():
    p = np.array([0.0, 1e-12, 0.5, 1.0 - 1e-12, 1.0])
    out = apply_temperature(p, 0.8)
    assert (np.diff(out) >= 0).all()
    assert (out > 0).all() and (out < 1).all()
    # log-loss reads an exact 0 or 1 as infinity; LOGIT_EPS is what stops that
    assert logit(0.0) == pytest.approx(np.log(LOGIT_EPS / (1 - LOGIT_EPS)))


def test_mask_round_45_zeroes_and_renormalises_three_round_fights():
    probs = np.array([[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])
    out = mask_round_45(probs, [True, False])
    assert out[0, 3] == 0.0
    np.testing.assert_allclose(out[0].sum(), 1.0)
    np.testing.assert_allclose(out[0][:3], np.array([0.4, 0.3, 0.2]) / 0.9)
    np.testing.assert_array_equal(out[1], probs[1])  # five-round row untouched


def test_mask_round_45_leaves_an_already_zero_row_bit_identical():
    """Renormalising a row that already sums to one still moves its last bits,
    and a blend at weight 0.0 has to be the torch member exactly."""
    row = np.array([[1 / 3, 1 / 3, 1 / 3, 0.0]])
    out = mask_round_45(row, [True])
    assert out.tobytes() == row.tobytes()


def test_mask_round_45_does_not_mutate_its_input():
    probs = np.array([[0.4, 0.3, 0.2, 0.1]])
    before = probs.copy()
    mask_round_45(probs, [True])
    np.testing.assert_array_equal(probs, before)


# --- the deployment constants -----------------------------------------------


def test_the_deployed_weight_is_the_one_the_shipped_report_was_scored_with():
    from mma.inference import BLEND_REPORT, BLEND_WEIGHT

    report = json.loads(BLEND_REPORT.read_text())
    assert report["config"]["blend_weight"] == BLEND_WEIGHT == 0.5
    assert report["config"]["blend_calibrated"] is True


def test_the_deployed_temperature_is_the_median_of_the_reports_per_fold_fits():
    """Deployment has no held-out year to fit a temperature on, so it applies a
    fixed value derived from the harness's per-fold fits -- by exactly the rule
    `run_walkforward.fixed_budget_from` uses for the torch member's own
    temperature under the refit recipe. This recomputes it from the committed
    report rather than trusting the constant."""
    from scripts.run_walkforward import fixed_budget_from
    from mma.inference import BLEND_REPORT, BLEND_TEMPERATURE

    report = json.loads(BLEND_REPORT.read_text())
    assert fixed_budget_from(report)["temperature"] == BLEND_TEMPERATURE


def test_the_deployed_blend_names_the_report_the_decision_shipped():
    from mma.inference import BLEND_REPORT

    decision = json.loads(
        (ROOT / "models" / "walkforward" / "sp2_2_decision.json").read_text()
    )
    assert decision["decision"]["outcome"] == "ship B1"
    assert decision["candidates"]["B1"]["seeds_0_4"]["report"] == str(
        BLEND_REPORT.relative_to(ROOT)
    )


# --- BlendedPredictor's own arithmetic, on stub members ---------------------


class _FakeBooster:
    feature_names = None

    def get_categories(self, export_to_arrow=False):
        class _Cats:
            @staticmethod
            def to_arrow():
                return []
        return _Cats()


class _FakeXGB:
    """Returns fixed probabilities regardless of input, one per seed."""

    def __init__(self, winner, method, rounds):
        self.winner, self.method, self.rounds = winner, method, rounds

    def get_booster(self):
        return _FakeBooster()

    def predict_proba(self, x):
        n = len(x)
        if self._head == "winner":
            return np.column_stack([1 - np.resize(self.winner, n), np.resize(self.winner, n)])
        return np.tile(self.method if self._head == "method" else self.rounds, (n, 1))


class _FakeEnsemble:
    """Per-seed torch outputs, tiled to whatever row count it is asked for."""

    preprocessor = None

    def __init__(self, winners):
        self.winners = winners

    def predict_members(self, features):
        n = len(features)
        return [
            {"winner": np.full(n, w), "method": np.full((n, 3), 1 / 3),
             "round": np.tile([0.5, 0.3, 0.2, 0.0], (n, 1))}
            for w in self.winners
        ]


def _frame(n=2, scheduled=3):
    return pd.DataFrame({"scheduled_rounds": [scheduled] * n,
                         "weight_class": pd.Series(["Lightweight"] * n, dtype="string"),
                         "elo_diff": [0.0] * n})


def _fake_predictor(xgb_winners, torch_winners, weight=0.5, temperature=1.0):
    from mma.inference import BlendedPredictor

    boosters = {}
    for head in ("winner", "method", "round"):
        models = []
        for w in xgb_winners:
            model = _FakeXGB(np.array([w]), np.array([0.2, 0.3, 0.5]),
                             np.array([0.25, 0.25, 0.25, 0.25]))
            model._head = head
            models.append(model)
        boosters[head] = models
    return BlendedPredictor(_FakeEnsemble(torch_winners), boosters, weight, temperature)


def test_blended_predictor_winner_is_the_temperature_scaled_average_of_the_two_means():
    predictor = _fake_predictor([0.6, 0.8], [0.2, 0.4], weight=0.5, temperature=0.8)
    out = predictor.predict(_frame())
    expected = apply_temperature(0.5 * 0.7 + 0.5 * 0.3, 0.8)
    np.testing.assert_allclose(out["winner_prob"], np.full(2, expected))


def test_blended_predictor_spread_is_the_per_seed_disagreement_and_brackets_the_mean():
    predictor = _fake_predictor([0.6, 0.8], [0.2, 0.4], weight=0.5, temperature=0.8)
    out = predictor.predict(_frame())
    seed0 = apply_temperature(0.5 * 0.6 + 0.5 * 0.2, 0.8)
    seed1 = apply_temperature(0.5 * 0.8 + 0.5 * 0.4, 0.8)
    np.testing.assert_allclose(out["winner_spread"], np.full(2, abs(seed1 - seed0)))
    assert seed0 < out["winner_prob"][0] < seed1


def test_blended_predictor_masks_round_45_for_three_round_fights():
    """The torch head masks it, the XGB head does not, so their average puts
    mass on a round that cannot happen."""
    predictor = _fake_predictor([0.5], [0.5])
    three = predictor.predict(_frame(scheduled=3))
    five = predictor.predict(_frame(scheduled=5))
    assert (three["round_probs"][:, 3] == 0.0).all()
    np.testing.assert_allclose(three["round_probs"].sum(axis=1), 1.0)
    assert (five["round_probs"][:, 3] > 0.0).all()


def test_blended_predictor_returns_the_torch_ensembles_contract_and_symmetrizes():
    """`predict_symmetrized` is written against `Ensemble.predict`'s dict, and
    what it must symmetrize is the BLEND -- so the blend has to satisfy that
    same contract, key for key."""
    from mma.inference import predict_symmetrized

    predictor = _fake_predictor([0.6, 0.8], [0.2, 0.4], weight=0.5, temperature=0.8)
    out = predictor.predict(_frame())
    assert set(out) == {
        "winner_prob", "winner_spread", "method_probs", "round_probs",
        "method_classes", "round_classes",
    }
    result = predict_symmetrized(predictor, _frame(n=1), _frame(n=1))
    assert result["winner_prob"] + (1 - result["winner_prob"]) == pytest.approx(1.0)
    assert result["winner_prob"] == pytest.approx(0.5)  # identical corners


def test_blended_predictor_load_refuses_a_half_built_model(tmp_path):
    """Serving a torch-only 'blend' silently is exactly what the model hash
    exists to make impossible; loading must fail loudly instead."""
    from mma.inference import BlendedPredictor

    with pytest.raises(FileNotFoundError):
        BlendedPredictor.load(tmp_path)
