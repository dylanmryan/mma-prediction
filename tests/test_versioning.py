"""model_version() must depend only on deployed model artifact bytes."""
import json

import pytest

from mma.versioning import MODEL_ARTIFACT_GLOBS, model_version
from scripts.migrate_model_versions import rekey_record


def _make_models(root, seed_bytes=b"seed0"):
    torch_dir = root / "models" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "net_seed0.pt").write_bytes(seed_bytes)
    (torch_dir / "preprocess.json").write_text(json.dumps({"medians": {}}))


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


def test_version_ignores_non_artifact_files(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "torch" / "display_priors.json").write_text("{}")
    (tmp_path / "models" / "torch" / "metrics_val.json").write_text('{"acc": 1}')
    (tmp_path / "models" / "xgb_winner.json").write_text("{}")
    (tmp_path / "models" / "market_benchmark.json").write_text("{}")
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


def test_globs_match_what_ensemble_loads():
    # mma.inference.Ensemble.load reads exactly net_seed*.pt and
    # preprocess.json from the torch directory -- nothing else feeds the
    # recorded probabilities, so the glob set must match those two exactly.
    assert MODEL_ARTIFACT_GLOBS == (
        "models/torch/net_seed*.pt",
        "models/torch/preprocess.json",
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
