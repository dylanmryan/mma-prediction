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
    (torch_dir / "display_priors.json").write_text("{}")
    (root / "models" / "xgb_winner.json").write_text("{}")
    (root / "models" / "xgb_method.json").write_text("{}")
    (root / "models" / "xgb_round.json").write_text("{}")


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
    (tmp_path / "models" / "torch" / "metrics_val.json").write_text('{"acc": 1}')
    (tmp_path / "models" / "market_benchmark.json").write_text("{}")
    assert model_version(tmp_path) == before


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_version(tmp_path)


def test_globs_cover_torch_and_xgb():
    assert any("net_seed" in g for g in MODEL_ARTIFACT_GLOBS)
    assert any("xgb_winner" in g for g in MODEL_ARTIFACT_GLOBS)


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
