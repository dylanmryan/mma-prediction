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


# The XGB member of the deployed blend. Its artifacts and the torch ones are
# built by separate scripts, so the blend's tests skip independently.
blended_artifacts = pytest.mark.skipif(
    not list((ROOT / "models").glob("xgb_winner_seed*.json")),
    reason="xgb seed ensemble not built (run scripts/train_xgb.py)",
)


@pytest.fixture(scope="module")
def ensemble():
    """The TORCH MEMBER. Since SP2.2 this is half of what serves; the served
    scorer is the `blended` fixture below."""
    from mma.inference import Ensemble
    return Ensemble.load()


# The simulator's two members, written by scripts/train_hazard.py.
simulator_artifacts = pytest.mark.skipif(
    not list((ROOT / "models").glob("xgb_hazard_seed*.json"))
    or not (ROOT / "models" / "simulator.json").exists(),
    reason="simulator artifacts not built (run scripts/train_hazard.py)",
)


@pytest.fixture(scope="module")
def blended():
    from mma.inference import BlendedPredictor
    return BlendedPredictor.load()


@pytest.fixture(scope="module")
def simulator(blended):
    """The DEPLOYED scorer since SP3: the blend's winner, the simulator's shape."""
    from mma.inference import SimulatorPredictor
    return SimulatorPredictor.load(blend=blended)


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
    return build_matchup(
        snapshot, weaker, bio, bio_unmapped, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    ), build_matchup(
        weaker, snapshot, bio_unmapped, bio, "Lightweight", False, 3,
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


# --- the deployed blend ----------------------------------------------------


@blended_artifacts
def test_blend_serves_both_members_on_the_same_served_row(blended, matchup):
    """Every column both members were trained on has to be on the served row --
    the torch preprocessor's numeric list and each XGB head's own feature
    names. A missing one on the XGB side used to be someone else's problem
    (the explainer's); it is the headline probability's now."""
    from mma.models.xgb import feature_frame

    frame, _ = matchup
    assert set(blended.preprocessor.numeric_columns) - set(frame.columns) == set()
    served = set(feature_frame(frame).columns)
    for head, models in blended.boosters.items():
        for model in models:
            trained = set(model.get_booster().feature_names or ())
            assert trained - served == set(), head


@blended_artifacts
def test_blend_has_five_seeds_a_side(blended):
    assert len(blended.ensemble.nets) == 5
    for head in ("winner", "method", "round"):
        assert len(blended.boosters[head]) == 5


@blended_artifacts
def test_blend_predict_symmetrized_is_exactly_self_consistent(blended, matchup):
    """`predict_symmetrized` must corner-average THE BLEND. Blending two
    already-symmetrized members, or symmetrizing one member and then blending,
    would each be a different number."""
    from mma.inference import predict_symmetrized

    forward, reverse = matchup
    result_fwd = predict_symmetrized(blended, forward, reverse)
    result_rev = predict_symmetrized(blended, reverse, forward)
    assert result_fwd["winner_prob"] + result_rev["winner_prob"] == pytest.approx(1.0, abs=1e-9)
    assert result_fwd["winner_prob"] > 0.5  # the stronger snapshot
    assert result_fwd["winner_spread"] >= 0.0
    assert result_fwd["method_probs"].sum() == pytest.approx(1.0, abs=1e-5)


@blended_artifacts
def test_blend_is_the_calibrated_average_of_its_two_members(blended, matchup):
    """Recomputed member by member: the served probability must be
    temperature(w * xgb_mean + (1 - w) * torch_mean), not either member and not
    an uncalibrated average."""
    from mma.blend import apply_temperature

    frame, _ = matchup
    torch_mean = np.mean(
        [m["winner"] for m in blended.ensemble.predict_members(frame)], axis=0
    )
    xgb_mean = np.mean(
        [m["winner"] for m in blended._xgb_predict(frame)], axis=0
    )
    expected = apply_temperature(
        blended.weight * xgb_mean + (1 - blended.weight) * torch_mean,
        blended.temperature,
    )
    np.testing.assert_allclose(blended.predict(frame)["winner_prob"], expected, atol=1e-12)


@blended_artifacts
def test_blend_round_45_is_zero_for_a_three_round_fight(blended, matchup):
    """The XGB round head does not mask it and the torch one does, so an
    unmasked average would put mass on a round that cannot happen."""
    frame, _ = matchup
    result = blended.predict(frame)
    assert result["round_probs"][0, 3] == 0.0
    assert result["round_probs"][0].sum() == pytest.approx(1.0, abs=1e-9)
    assert result["method_probs"][0].sum() == pytest.approx(1.0, abs=1e-5)


@blended_artifacts
def test_blend_survives_a_weight_class_the_models_never_saw(blended, matchup):
    """A Wikipedia card can name a division the training table does not carry.
    The torch member maps it to its reserved unknown index; the XGB member
    would raise XGBoostError and take down the whole card's predictions, so
    `align_to_booster` turns it into a missing value instead."""
    frame, _ = matchup
    unknown = frame.copy()
    unknown["weight_class"] = pd.Series(["Superheavyweight"], dtype="string")
    result = blended.predict(unknown)
    assert 0.0 < float(result["winner_prob"][0]) < 1.0


def test_compute_correction_factors_hand_computed():
    from mma.inference import compute_correction_factors

    empirical = {"a": 0.2, "b": 0.8}
    mean_predicted = {"a": 0.5, "b": 0.5}
    factors = compute_correction_factors(empirical, mean_predicted)
    assert factors["a"] == pytest.approx(0.4)
    assert factors["b"] == pytest.approx(1.6)


def test_compute_correction_factors_mean_matches_aggregate():
    # The defining property, and the reason a factor near 1 means "already
    # calibrated in aggregate": scaling the model's mean predicted
    # distribution by the factors and renormalising recovers the base rates.
    from mma.inference import compute_correction_factors

    empirical = {"1": 0.4, "2": 0.25, "3": 0.17, "45": 0.18}
    mean_predicted = {"1": 0.07, "2": 0.08, "3": 0.09, "45": 0.76}
    factors = compute_correction_factors(empirical, mean_predicted)
    scaled = {cls: mean_predicted[cls] * factors[cls] for cls in empirical}
    total = sum(scaled.values())
    for cls, value in empirical.items():
        assert scaled[cls] / total == pytest.approx(value, abs=1e-9)


def test_compute_correction_factors_guards_tiny_mean_predicted():
    from mma.inference import compute_correction_factors

    empirical = {"a": 0.5, "b": 0.5, "c": 0.0}
    mean_predicted = {"a": 0.999999, "b": 1e-9, "c": 1e-9}
    factors = compute_correction_factors(empirical, mean_predicted)
    assert factors["b"] == 0.0  # capped instead of exploding
    assert factors["c"] == 0.0


def test_committed_display_calibration_says_the_correction_is_retired():
    """SP3 measured the deployed simulator against the base rates and dropped
    the mean-matching correction on the evidence. The committed artifact is
    that evidence, so it has to carry the verdict and the numbers behind it."""
    import json

    payload = json.loads((ROOT / "models" / "display_calibration.json").read_text())
    assert payload["correction_applied"] is False
    assert set(payload["measured_on"]) == {"rows", "of", "train_through", "mode"}
    # Deliberately NOT asserted here: that the measurement is inside its
    # tolerance. That is a property of a retrain, not of this code, and the
    # weekly Action runs the suite BEFORE it commits refreshed data -- pinning
    # a threshold that can drift would block a data refresh on a calibration
    # wobble. `scripts/check_display_calibration.py` warns on stderr instead.
    # What IS asserted is the comparison the decision rests on, which is not
    # marginal: the blend's heads are off by three to six times as much.
    for key in ("method", "round_3", "round_5"):
        block = payload[key]
        assert block["n"] > 0
        simulator = block["max_deviation_points"]["deployed_simulator"]
        blend = block["max_deviation_points"]["retired_blend_heads"]
        # the whole argument in one assertion: the simulator's marginals sit
        # closer to the base rates than the heads the correction was built for
        assert simulator < blend, key
        assert set(block["deployed_simulator"]) == {
            "empirical", "mean_predicted", "factor_needed"}
    assert payload["round_3"]["deployed_simulator"]["mean_predicted"]["45"] == 0.0


def _synthetic_features():
    # Four pre-2021 rows and one 2022 row: the split protocol trained only
    # on the former; a refit_through model trained on all five.
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-01-01", "2019-06-01", "2020-01-01", "2020-06-01", "2022-01-01"]
            ),
            "y_method": ["ko_tko", "ko_tko", "submission", "decision", "ko_tko"],
            "y_finish_round": ["1", "2", "1", pd.NA, "45"],
            "scheduled_rounds": pd.array([3, 3, 5, 3, 5], dtype="Int64"),
        }
    )


def test_deployed_training_mask_follows_the_metrics_file():
    from mma.inference import deployed_training_mask

    features = _synthetic_features()
    # split-mode metrics (no "mode" key): the original date < 2021-01-01 split
    split = deployed_training_mask(features, {"winner_ensemble": {"log_loss": 0.65}})
    assert split.tolist() == [True, True, True, True, False]
    # refit_through: every row dated <= train_through
    refit = deployed_training_mask(
        features, {"mode": "refit_through", "train_through": "2022-01-01"}
    )
    assert refit.tolist() == [True, True, True, True, True]
    partial = deployed_training_mask(
        features, {"mode": "refit_through", "train_through": "2020-01-01"}
    )
    assert partial.tolist() == [True, True, True, False, False]


def test_deployed_training_mask_reads_committed_metrics():
    """With no explicit metrics the mask is read from the committed
    models/torch/metrics_val.json; under the refit recipe that is every row
    through train_through."""
    import json

    from mma.inference import TORCH_METRICS, deployed_training_mask

    metrics = json.loads(TORCH_METRICS.read_text())
    features = _synthetic_features()
    mask = deployed_training_mask(features)
    if metrics.get("mode") == "refit_through":
        expected = (features["date"] <= pd.Timestamp(metrics["train_through"])).tolist()
    else:
        expected = [True, True, True, True, False]
    assert mask.tolist() == expected


def test_compute_display_priors_refit_uses_all_training_rows():
    from mma.inference import compute_display_priors

    priors = compute_display_priors(
        _synthetic_features(), {"mode": "refit_through", "train_through": "2022-01-01"}
    )
    # all five rows count: 3 ko_tko, 1 submission, 1 decision
    assert priors["method"]["ko_tko"] == pytest.approx(0.6)
    assert priors["method"]["submission"] == pytest.approx(0.2)
    assert priors["method"]["decision"] == pytest.approx(0.2)
    # both 5-round finishes count: one round 1, one rounds 4-5
    assert priors["round_5"]["1"] == pytest.approx(0.5)
    assert priors["round_5"]["45"] == pytest.approx(0.5)


def test_compute_display_priors_synthetic_frame():
    from mma.inference import compute_display_priors

    # Split-protocol model: two train-split rows (date < 2021-01-01) and one
    # post-cutoff row that must be excluded from the priors entirely.
    features = _synthetic_features()
    priors = compute_display_priors(features, {"winner_ensemble": {"log_loss": 0.65}})

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


# --- the deployed hybrid (SP3) ---------------------------------------------

@simulator_artifacts
def test_simulator_config_is_the_one_the_harness_measured():
    """The deployed simulation parameters must be the harness's.

    `models/simulator.json` exists so the four numbers that move a simulated
    probability are inside the model hash. That only means anything if they
    are also the numbers the hybrid was scored with -- a deployment serving
    n_runs=100 would be hashed, reproducible, and describe no report.
    """
    from mma import simulator as sim_module
    from mma.inference import SIMULATOR_PARAMETERS, load_simulator_config

    config = load_simulator_config()
    assert set(SIMULATOR_PARAMETERS) <= set(config)
    assert config["n_runs"] == sim_module.DEFAULT_N_RUNS
    assert config["alpha"] == sim_module.DEFAULT_ALPHA
    assert config["sim_seed"] == sim_module.DEFAULT_SIM_SEED
    assert config["default_rounds"] == sim_module.DEFAULT_ROUNDS


def test_load_simulator_config_refuses_to_guess(tmp_path):
    from mma.inference import load_simulator_config

    with pytest.raises(FileNotFoundError, match="no in-code fallback"):
        load_simulator_config(tmp_path / "simulator.json")
    partial = tmp_path / "partial.json"
    partial.write_text('{"n_runs": 10000, "alpha": 1.0}')
    with pytest.raises(ValueError, match="missing simulation parameter"):
        load_simulator_config(partial)


@simulator_artifacts
def test_simulator_winner_probability_is_the_blend_untouched(simulator, blended, matchup):
    """The clause the whole fallback design exists for.

    SP3 ships a joint distribution whose winner marginal IS the incumbent's,
    element for element -- `models/walkforward/sp3_decision.json` verified that
    over 4,804 pooled rows and it has to hold at serving time too, or the
    deployment moved a number it promised not to move.
    """
    from mma.inference import predict_symmetrized

    frame, reversed_frame = matchup
    hybrid = simulator.predict(frame)
    blend = blended.predict(frame)
    np.testing.assert_array_equal(hybrid["winner_prob"], blend["winner_prob"])
    np.testing.assert_array_equal(hybrid["winner_spread"], blend["winner_spread"])

    sym_hybrid = predict_symmetrized(simulator, frame, reversed_frame)
    sym_blend = predict_symmetrized(blended, frame, reversed_frame)
    assert sym_hybrid["winner_prob"] == sym_blend["winner_prob"]
    assert sym_hybrid["orientation_ab_prob"] == sym_blend["orientation_ab_prob"]
    assert sym_hybrid["mc_dropout_shift"] == sym_blend["mc_dropout_shift"]


@simulator_artifacts
def test_simulator_joint_is_a_distribution_its_marginals_are_read_off(simulator, matchup):
    from mma.joint import marginals_from_cells
    from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES

    frame, _ = matchup
    result = simulator.predict(frame)
    cells = result["joint_cells"]
    assert cells.shape == (1, 18)
    np.testing.assert_allclose(cells.sum(axis=1), 1.0)
    assert (cells >= 0).all()

    marginals = marginals_from_cells(cells, METHOD_CLASSES, ROUND_CLASSES)
    np.testing.assert_allclose(result["method_probs"], marginals["method"])
    np.testing.assert_allclose(result["round_probs"], marginals["round"])
    np.testing.assert_allclose(result["winner_prob"], marginals["winner"], atol=1e-12)
    # p_distance is the decision cells' mass, which is the method head's
    # decision class -- one distribution, not two that could disagree
    np.testing.assert_allclose(result["p_distance"], cells[:, -2:].sum(axis=1))
    np.testing.assert_allclose(result["p_distance"], result["method_probs"][:, -1])


@simulator_artifacts
def test_simulator_three_round_fight_gets_no_round_45_mass(simulator, matchup):
    frame, _ = matchup
    assert int(frame["scheduled_rounds"].iloc[0]) == 3
    result = simulator.predict(frame)
    assert result["round_probs"][0, -1] == 0.0
    # ...and no joint cell for rounds 4-5 either, in EITHER corner
    cells = result["joint_cells"][0]
    assert cells[3] == 0.0 and cells[7] == 0.0 and cells[11] == 0.0 and cells[15] == 0.0


@simulator_artifacts
def test_simulator_predict_symmetrized_reports_one_coherent_distribution(simulator, matchup):
    """Everything the symmetrized result reports must come from the joint it
    reports -- including the winner, which is the blend's."""
    from mma.inference import predict_symmetrized
    from mma.joint import marginals_from_cells

    frame, reversed_frame = matchup
    result = predict_symmetrized(simulator, frame, reversed_frame)
    cells = np.asarray(result["joint_cells"])[None, :]
    np.testing.assert_allclose(cells.sum(), 1.0)
    marginals = marginals_from_cells(cells, list(result["method_classes"]),
                                     list(result["round_classes"]))
    assert marginals["winner"][0] == pytest.approx(result["winner_prob"], abs=1e-12)
    np.testing.assert_allclose(result["method_probs"], marginals["method"][0])
    np.testing.assert_allclose(result["round_probs"], marginals["round"][0])
    assert result["p_distance"] == pytest.approx(cells[0, -2:].sum())
    assert result["winner_prob"] + (1.0 - result["winner_prob"]) == 1.0


@simulator_artifacts
def test_symmetrized_joint_keeps_the_blends_winner_exactly(simulator, blended, matchup):
    """The invariant `_symmetrize_joint` used to raise on, kept where a
    regression gets caught in CI rather than in a Monday morning traceback.

    The hybrid must not move a winner probability: whatever the corner-average
    of the two orientations' joints does to the cells, the winner marginal
    read back off them has to be the symmetrized BLEND's number. That holds by
    construction now -- each orientation's marginal is set exactly by
    `impose_winner_marginal` and the average is re-imposed after the fact --
    so this asserts it at the tightest tolerance the arithmetic supports
    rather than at the 1e-9 the old runtime check used.
    """
    from mma.inference import predict_symmetrized
    from mma.joint import marginals_from_cells

    frame, reversed_frame = matchup
    hybrid = predict_symmetrized(simulator, frame, reversed_frame)
    blend = predict_symmetrized(blended, frame, reversed_frame)
    cells = np.asarray(hybrid["joint_cells"])[None, :]
    winner_from_cells = float(
        marginals_from_cells(cells, list(hybrid["method_classes"]),
                             list(hybrid["round_classes"]))["winner"][0]
    )
    assert winner_from_cells == pytest.approx(blend["winner_prob"], abs=1e-15)
    assert hybrid["winner_prob"] == pytest.approx(blend["winner_prob"], abs=1e-15)


@simulator_artifacts
def test_simulator_predict_is_a_superset_of_the_blend_contract(simulator, blended, matchup):
    """Callers written against `Ensemble.predict` keep working: the hybrid adds
    keys, it never drops one."""
    frame, _ = matchup
    hybrid = simulator.predict(frame)
    blend = blended.predict(frame)
    assert set(blend) <= set(hybrid)
    assert set(hybrid) - set(blend) == {"joint_cells", "joint_zero_mass", "p_distance"}
    assert hybrid["method_classes"] == blend["method_classes"]
    assert hybrid["round_classes"] == blend["round_classes"]
    assert hybrid["method_probs"].shape == blend["method_probs"].shape
    assert hybrid["round_probs"].shape == blend["round_probs"].shape


@simulator_artifacts
def test_simulator_is_deterministic(simulator, matchup):
    """A Monte Carlo scorer that moved between runs would make the track
    record's model_version a lie."""
    frame, _ = matchup
    first = simulator.predict(frame)
    second = simulator.predict(frame)
    np.testing.assert_array_equal(first["joint_cells"], second["joint_cells"])


@simulator_artifacts
def test_simulator_survives_a_weight_class_the_models_never_saw(simulator, matchup):
    frame, _ = matchup
    unseen = frame.copy()
    unseen["weight_class"] = pd.array(["Catchweight 165"], dtype="string")
    result = simulator.predict(unseen)
    np.testing.assert_allclose(result["joint_cells"].sum(axis=1), 1.0)


@simulator_artifacts
def test_simulator_load_demands_both_members(blended, tmp_path):
    """A missing member is a loud error, never a silent fall back to the
    blend's own method and round heads -- that would be a scorer no report
    describes."""
    from mma.inference import SimulatorPredictor, load_simulator_config

    (tmp_path / "models").mkdir()
    config = load_simulator_config()
    with pytest.raises(FileNotFoundError, match="no XGBoost hazard models"):
        SimulatorPredictor.load(root=tmp_path, blend=blended, config=config)
    for seed in range(5):
        (tmp_path / "models" / f"xgb_hazard_seed{seed}.json").write_bytes(
            (ROOT / "models" / f"xgb_hazard_seed{seed}.json").read_bytes())
    with pytest.raises(FileNotFoundError, match="no XGBoost decision models"):
        SimulatorPredictor.load(root=tmp_path, blend=blended, config=config)
