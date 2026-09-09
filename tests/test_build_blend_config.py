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


def test_temperature_is_the_walkforward_value_not_the_median_of_the_fits():
    """Recomputed independently of `build_config`, straight from the committed
    dump: fit one temperature on every out-of-fold row, which is what the
    walk-forward rule reduces to at serving time (every fold is 'before' the
    fold being served).

    The median of the per-fold fits -- the rule this file carried until
    2026-09-09 -- is asserted to be a DIFFERENT number, because that is the
    defect this test exists to keep fixed: a median is a training-budget rule
    with no calibration justification, and it re-scores to pooled ECE 0.0177
    against the walk-forward rule's 0.0108.
    """
    from scripts.derive_blend_temperature import (
        deployed_temperature, invert_per_fold, load_inputs,
    )
    from scripts.run_walkforward import fixed_budget_from

    committed = json.loads(BLEND_CONFIG.read_text())
    report = json.loads((ROOT / committed["source_report"]).read_text())
    inputs = load_inputs(ROOT / committed["source_report"],
                         ROOT / committed["source_predictions"])
    pre = invert_per_fold(inputs["p_scored"], inputs["years"], inputs["fold_temperatures"])
    assert committed["temperature"] == deployed_temperature(pre, inputs["y"])
    assert committed["temperature"] != fixed_budget_from(report)["temperature"]


def test_the_deployed_temperature_matches_the_derivation_artifact():
    """`models/blend.json` and `models/walkforward/blend_temperature.json` are
    written by different scripts from the same two inputs; if they ever
    disagree, the published metrics describe a temperature that is not served."""
    committed = json.loads(BLEND_CONFIG.read_text())
    record = json.loads((ROOT / "models" / "walkforward" / "blend_temperature.json").read_text())
    assert committed["temperature"] == record["deployed"]["temperature"]
    assert committed["source_report"] == record["source_report"]
    assert committed["source_predictions"] == record["source_predictions"]


@pytest.mark.parametrize("key", ["source_report", "source_predictions"])
def test_source_paths_are_relative_and_exist(key):
    committed = json.loads(BLEND_CONFIG.read_text())
    source = committed[key]
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
        # The dump is the real one: only the report's declared weight changes,
        # and the weight is the thing under test.
        config = build_config(tmp_report, ROOT / "models" / "walkforward" / "preds" / "blend_b1.json")
        assert config["weight"] == 0.7
    finally:
        tmp_report.unlink()
