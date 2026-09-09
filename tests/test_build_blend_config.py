"""models/blend.json must not drift from the report it claims to come from.

The deployed blend's mixing weight and post-average temperature used to live
as hand-set module constants in `src/mma/inference.py`, invisible to
`mma.versioning.model_version`. `scripts/build_blend_config.py` now derives
both from the committed walk-forward report and writes them to
`models/blend.json`, which the version hash covers. These tests recompute the
artifact from the report and assert the committed file matches -- so a hand
edit to either one (or a report update without rerunning the script) is
caught rather than silently served.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLEND_CONFIG = ROOT / "models" / "blend.json"

pytestmark = pytest.mark.skipif(
    not BLEND_CONFIG.exists(),
    reason="models/blend.json not built (run scripts/build_blend_config.py)",
)


def test_committed_artifact_matches_a_fresh_build_from_the_report():
    from scripts.build_blend_config import BLEND_REPORT, build_config

    committed = json.loads(BLEND_CONFIG.read_text())
    fresh = build_config(BLEND_REPORT)
    assert committed == fresh


def test_weight_is_the_pre_registered_0_5_named_by_the_source_report():
    committed = json.loads(BLEND_CONFIG.read_text())
    report = json.loads((ROOT / committed["source_report"]).read_text())
    assert committed["weight"] == report["config"]["blend_weight"] == 0.5


def test_temperature_is_the_median_of_the_reports_per_fold_fits():
    """Recomputed independently of `build_config`, via the same
    `fixed_budget_from` median rule the refit recipe uses for the torch
    member's own deployment temperature."""
    from scripts.run_walkforward import fixed_budget_from

    committed = json.loads(BLEND_CONFIG.read_text())
    report = json.loads((ROOT / committed["source_report"]).read_text())
    assert committed["temperature"] == fixed_budget_from(report)["temperature"]


def test_source_report_path_is_relative_and_exists():
    committed = json.loads(BLEND_CONFIG.read_text())
    source = committed["source_report"]
    assert not Path(source).is_absolute()
    assert (ROOT / source).exists()


def test_build_config_recomputes_rather_than_trusting_the_report_weight_blindly():
    """A report scored at a different weight changes the artifact's weight
    rather than the writer silently hardcoding 0.5."""
    from scripts.build_blend_config import build_config

    report = json.loads((ROOT / "models" / "walkforward" / "blend_b1.json").read_text())
    report["config"] = dict(report["config"], blend_weight=0.7)
    tmp_report = ROOT / "models" / "walkforward" / "_test_blend_b1_w07_scratch.json"
    tmp_report.write_text(json.dumps(report))
    try:
        config = build_config(tmp_report)
        assert config["weight"] == 0.7
    finally:
        tmp_report.unlink()
