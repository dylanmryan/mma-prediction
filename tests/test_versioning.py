"""model_version() must depend only on deployed model artifact bytes."""
import json

import pytest

from mma.versioning import MODEL_ARTIFACT_GLOBS, model_version
from scripts.migrate_model_versions import rekey_record


SIMULATOR = {"n_runs": 10000, "alpha": 1.0, "sim_seed": 0, "default_rounds": 3}


def _make_models(root, seed_bytes=b"seed0"):
    """Everything the deployed hybrid loads: the blend's two members and their
    committed weight/temperature, and the simulator's two members and its
    committed simulation parameters."""
    models = root / "models"
    torch_dir = models / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "net_seed0.pt").write_bytes(seed_bytes)
    (torch_dir / "preprocess.json").write_text(json.dumps({"medians": {}}))
    for head in ("winner", "method", "round", "hazard", "decision"):
        (models / f"xgb_{head}_seed0.json").write_text(json.dumps({"head": head}))
    (models / "blend.json").write_text(json.dumps({"weight": 0.5, "temperature": 0.8}))
    (models / "simulator.json").write_text(json.dumps(SIMULATOR))


def test_version_is_12_hex_and_stable(tmp_path):
    _make_models(tmp_path)
    first = model_version(tmp_path)
    assert len(first) == 12 and int(first, 16) >= 0
    assert model_version(tmp_path) == first


def test_version_changes_when_a_weight_file_changes(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "torch" / "net_seed0.pt").write_bytes(b"seed0-retrained")
    assert model_version(tmp_path) != before


def test_version_changes_when_an_xgboost_booster_changes(tmp_path):
    """The XGBoost seed ensemble is half of what scores since SP2.2, so a
    retrain that moves only the boosters must open a new track-record
    section -- before SP2.2 this hash would not have noticed at all."""
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "xgb_winner_seed0.json").write_text('{"head": "retrained"}')
    assert model_version(tmp_path) != before


def test_version_changes_when_an_xgboost_seed_is_added(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "xgb_winner_seed1.json").write_text('{"head": "winner"}')
    assert model_version(tmp_path) != before


def test_version_covers_every_head_of_the_xgb_member(tmp_path):
    """The method and round boosters still feed the blend, and the hazard and
    decision models are what the simulator plays a fight out with -- every one
    of the five is part of the scorer."""
    for head in ("winner", "method", "round", "hazard", "decision"):
        _make_models(tmp_path / head)
        before = model_version(tmp_path / head)
        (tmp_path / head / "models" / f"xgb_{head}_seed0.json").write_text('{"x": 1}')
        assert model_version(tmp_path / head) != before, head


def test_version_changes_when_the_blend_weight_changes(tmp_path):
    """The blend's mixing weight is part of the scorer, not display
    configuration -- changing it changes every recorded probability, so it
    must move the hash exactly as a retrained booster does."""
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "blend.json").write_text(
        json.dumps({"weight": 0.7, "temperature": 0.8})
    )
    assert model_version(tmp_path) != before


def test_version_changes_when_the_blend_temperature_changes(tmp_path):
    """Same as the weight above: the post-average temperature is part of the
    scorer, so it must move the hash too."""
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "blend.json").write_text(
        json.dumps({"weight": 0.5, "temperature": 1.0})
    )
    assert model_version(tmp_path) != before


@pytest.mark.parametrize(
    "parameter,value",
    [("n_runs", 5000), ("alpha", 0.5), ("sim_seed", 1), ("default_rounds", 5)],
)
def test_version_changes_when_a_simulation_parameter_changes(tmp_path, parameter, value):
    """SP2.2's lesson, applied to SP3's four numbers.

    None of them is learned by anything, and every one of them changes what a
    simulated fight comes out at -- so each must move the hash exactly as a
    retrained booster does, or a prediction's recorded model_version stops
    identifying the scorer that made it.
    """
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "simulator.json").write_text(
        json.dumps({**SIMULATOR, parameter: value})
    )
    assert model_version(tmp_path) != before


def test_version_changes_when_a_simulator_member_is_retrained(tmp_path):
    """The simulator supplies P(method, round | winner); a retrain of either
    member changes every joint cell."""
    for member in ("hazard", "decision"):
        root = tmp_path / member
        _make_models(root)
        before = model_version(root)
        (root / "models" / f"xgb_{member}_seed0.json").write_text('{"head": "retrained"}')
        assert model_version(root) != before, member


def test_version_ignores_non_artifact_files(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "display_calibration.json").write_text("{}")
    (tmp_path / "models" / "torch" / "metrics_val.json").write_text('{"acc": 1}')
    (tmp_path / "models" / "xgb_metrics_val.json").write_text("{}")
    (tmp_path / "models" / "hazard_metrics.json").write_text("{}")
    (tmp_path / "models" / "market_benchmark.json").write_text("{}")
    (tmp_path / "models" / "final_test_metrics.json").write_text("{}")
    assert model_version(tmp_path) == before


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_version(tmp_path)


def test_version_changes_when_preprocess_changes(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "torch" / "preprocess.json").write_text(
        json.dumps({"medians": {"age": 30}})
    )
    assert model_version(tmp_path) != before


def test_globs_match_what_the_deployed_predictor_loads():
    # mma.inference.SimulatorPredictor.load reads exactly these: the torch
    # ensemble's per-seed checkpoints and preprocessor, the XGBoost member's
    # per-seed boosters for all three heads, models/blend.json (the committed
    # mixing weight and post-average temperature the two are combined with),
    # the simulator's per-seed hazard and decision models, and
    # models/simulator.json (the committed simulation parameters). Nothing
    # else feeds the recorded probabilities, so the glob set must match them
    # exactly.
    assert MODEL_ARTIFACT_GLOBS == (
        "models/torch/net_seed*.pt",
        "models/torch/preprocess.json",
        "models/xgb_winner_seed*.json",
        "models/xgb_method_seed*.json",
        "models/xgb_round_seed*.json",
        "models/blend.json",
        "models/xgb_hazard_seed*.json",
        "models/xgb_decision_seed*.json",
        "models/simulator.json",
    )


def test_every_simulation_parameter_the_predictor_reads_is_in_the_hashed_artifact():
    """The hash covers the FILE; this covers the claim that the file is where
    the parameters live. A parameter the predictor read from anywhere else --
    a module constant, an environment variable -- would be outside the hash
    however carefully the file is hashed."""
    import fnmatch

    from mma.inference import SIMULATOR_CONFIG, SIMULATOR_PARAMETERS, load_simulator_config

    assert any(fnmatch.fnmatch("models/simulator.json", glob)
               for glob in MODEL_ARTIFACT_GLOBS)
    config = load_simulator_config(SIMULATOR_CONFIG)
    assert set(SIMULATOR_PARAMETERS) == {"n_runs", "alpha", "sim_seed", "default_rounds"}
    assert set(SIMULATOR_PARAMETERS) <= set(config)


def test_rekey_replaces_git_shas_and_leaves_hashes():
    record = {
        "model_version": "79135ef",
        "fights": [
            {"model_version": "e47f720", "p_a_wins": 0.6},
            {"model_version": "abcdef012345", "p_a_wins": 0.4},
            {"skipped": True},
        ],
    }
    assert rekey_record(record, "abcdef012345") == 2
    assert record["model_version"] == "abcdef012345"
    assert record["fights"][0]["model_version"] == "abcdef012345"
    assert record["fights"][1]["model_version"] == "abcdef012345"
    assert rekey_record(record, "abcdef012345") == 0  # idempotent


def test_rekey_replaces_old_values_only_when_passed():
    record = {
        "model_version": "f4aa6f8d9d6f",
        "fights": [
            {"model_version": "f4aa6f8d9d6f", "p_a_wins": 0.6},
            {"skipped": True},
        ],
    }
    assert rekey_record(record, "0123456789ab") == 0
    assert record["model_version"] == "f4aa6f8d9d6f"
    assert record["fights"][0]["model_version"] == "f4aa6f8d9d6f"

    assert rekey_record(record, "0123456789ab", old_values=("f4aa6f8d9d6f",)) == 2
    assert record["model_version"] == "0123456789ab"
    assert record["fights"][0]["model_version"] == "0123456789ab"
