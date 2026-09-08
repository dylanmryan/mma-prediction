"""model_version() must depend only on deployed model artifact bytes."""
import json

import pytest

from mma.versioning import MODEL_ARTIFACT_GLOBS, model_version
from scripts.migrate_model_versions import rekey_record


def _make_models(root, seed_bytes=b"seed0"):
    """Both halves of the deployed scorer: the torch ensemble and the XGBoost
    seed ensemble the blend averages with it."""
    models = root / "models"
    torch_dir = models / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "net_seed0.pt").write_bytes(seed_bytes)
    (torch_dir / "preprocess.json").write_text(json.dumps({"medians": {}}))
    for head in ("winner", "method", "round"):
        (models / f"xgb_{head}_seed0.json").write_text(json.dumps({"head": head}))


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
    """The method and round boosters feed the app's displayed method/round
    splits through the blend, so they are part of the scorer too."""
    for head in ("winner", "method", "round"):
        _make_models(tmp_path / head)
        before = model_version(tmp_path / head)
        (tmp_path / head / "models" / f"xgb_{head}_seed0.json").write_text('{"x": 1}')
        assert model_version(tmp_path / head) != before, head


def test_version_ignores_non_artifact_files(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "torch" / "display_priors.json").write_text("{}")
    (tmp_path / "models" / "torch" / "metrics_val.json").write_text('{"acc": 1}')
    (tmp_path / "models" / "xgb_metrics_val.json").write_text("{}")
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
    # mma.inference.BlendedPredictor.load reads exactly these: the torch
    # ensemble's per-seed checkpoints and preprocessor, and the XGBoost
    # member's per-seed boosters for all three heads. Nothing else feeds the
    # recorded probabilities, so the glob set must match them exactly.
    assert MODEL_ARTIFACT_GLOBS == (
        "models/torch/net_seed*.pt",
        "models/torch/preprocess.json",
        "models/xgb_winner_seed*.json",
        "models/xgb_method_seed*.json",
        "models/xgb_round_seed*.json",
    )


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
