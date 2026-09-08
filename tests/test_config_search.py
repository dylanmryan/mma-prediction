"""The SP2.1 search space: fixed, shared between the arms, and verified.

The experiment's whole comparison rests on arms C and T searching the
*identical* 25 configurations, so these tests pin the sampler (determinism,
count, the pre-registered ranges) and the reuse-and-verify behaviour that
stops a second arm from silently searching a different space. Nothing here
runs the harness; ``summarise`` is exercised against hand-written reports.
"""
from __future__ import annotations

import json

import pytest

import scripts.config_search as cs


def test_sampling_is_deterministic_given_the_seed():
    assert cs.sample_configs() == cs.sample_configs()
    assert cs.sample_configs(seed=cs.SAMPLE_SEED) == cs.sample_configs()


def test_a_different_seed_samples_a_different_space():
    assert cs.sample_configs(seed=1) != cs.sample_configs(seed=0)


def test_twenty_five_configurations():
    assert len(cs.sample_configs()) == cs.N_CONFIGS == 25


def test_a_prefix_of_a_larger_draw_is_the_same_draw():
    # The per-config draw order is what makes this true; it is also what
    # makes the stored configs.json comparable to a fresh sample.
    assert cs.sample_configs(n=3) == cs.sample_configs()[:3]


def test_every_value_lies_in_the_pre_registered_space():
    for config in cs.sample_configs():
        assert sorted(config) == [
            "dropout", "embedding_dim", "hidden", "lr",
            "method_scale", "round_scale", "weight_decay",
        ]
        assert tuple(config["hidden"]) in cs.HIDDEN_CHOICES
        assert config["embedding_dim"] in cs.EMBEDDING_CHOICES
        for key, (low, high) in [
            ("dropout", cs.DROPOUT_RANGE), ("lr", cs.LR_RANGE),
            ("weight_decay", cs.WEIGHT_DECAY_RANGE),
            ("method_scale", cs.METHOD_SCALE_RANGE),
            ("round_scale", cs.ROUND_SCALE_RANGE),
        ]:
            assert low <= config[key] <= high, (key, config[key])


def test_the_space_is_actually_explored():
    configs = cs.sample_configs()
    assert len({tuple(c["hidden"]) for c in configs}) == len(cs.HIDDEN_CHOICES)
    assert len({c["embedding_dim"] for c in configs}) == len(cs.EMBEDDING_CHOICES)


def test_configs_are_the_train_loop_config_keys():
    from mma.models.train_loop import DEFAULT_CONFIG, resolve_config

    for config in cs.sample_configs():
        assert set(config) == set(DEFAULT_CONFIG)
        resolve_config(config)  # raises on an unknown key


def test_configs_survive_a_json_round_trip():
    configs = cs.sample_configs()
    assert json.loads(json.dumps(configs)) == configs


def test_first_run_writes_the_shared_configs(tmp_path):
    path = tmp_path / "configs.json"
    written = cs.load_or_write_configs(path)
    assert path.exists()
    assert written == cs.sample_configs()


def test_second_run_reuses_the_stored_configs(tmp_path):
    path = tmp_path / "configs.json"
    first = cs.load_or_write_configs(path)
    stamp = path.stat().st_mtime_ns
    assert cs.load_or_write_configs(path) == first
    assert path.stat().st_mtime_ns == stamp  # verified, not rewritten


def test_a_disagreeing_configs_file_stops_the_run(tmp_path):
    path = tmp_path / "configs.json"
    tampered = cs.sample_configs()
    tampered[7]["dropout"] = 0.42
    path.write_text(json.dumps(tampered, indent=2))
    with pytest.raises(SystemExit, match="disagrees with the sampler"):
        cs.load_or_write_configs(path)


def test_a_short_configs_file_stops_the_run(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps(cs.sample_configs()[:5], indent=2))
    with pytest.raises(SystemExit, match="disagrees with the sampler"):
        cs.load_or_write_configs(path)


def _report(path, loss):
    path.write_text(json.dumps({
        "pooled": {"winner_log_loss": loss},
        "folds": {"2018": {"winner_log_loss": loss}},
        "config": {"runtime_sec": 1.0},
    }))


def test_summary_ranks_by_pooled_loss_and_reports_the_spread(tmp_path):
    configs = cs.sample_configs(n=3)
    for index, loss in enumerate([0.66, 0.64, 0.65]):
        _report(tmp_path / f"config_{index:02d}.json", loss)
    summary = cs.summarise("T", ("base",), "external_missing", configs, tmp_path)
    assert [row["index"] for row in summary["ranked"]] == [1, 2, 0]
    assert [row["rank"] for row in summary["ranked"]] == [1, 2, 3]
    assert summary["best"]["index"] == 1
    assert summary["best"]["config"] == configs[1]
    assert summary["distribution"] == {"min": 0.64, "median": 0.65, "max": 0.66}
    assert summary["drop_columns"] == ["external_missing"]
    assert summary["n_configs"] == 3


def test_summary_covers_only_the_configs_that_have_reports(tmp_path):
    configs = cs.sample_configs(n=3)
    _report(tmp_path / "config_01.json", 0.65)
    summary = cs.summarise("C", ("base",), None, configs, tmp_path)
    assert summary["n_configs"] == 3 and summary["n_scored"] == 1
    assert summary["best"]["index"] == 1
    assert summary["drop_columns"] == []


def test_the_wrong_feature_table_stops_the_run(monkeypatch):
    # Both directions of the mismatch, spelled with blocks that are actually
    # registered: an unregistered name is caught one level lower, by
    # `resolve_blocks`, and would test that instead of this guard.
    monkeypatch.setattr(cs, "table_blocks", lambda: ("base", "external"))
    assert cs.check_table("base,external") == ("base", "external")
    with pytest.raises(SystemExit, match="was built from"):
        cs.check_table("base")
    monkeypatch.setattr(cs, "table_blocks", lambda: ("base",))
    with pytest.raises(SystemExit, match="was built from"):
        cs.check_table("base,external")
