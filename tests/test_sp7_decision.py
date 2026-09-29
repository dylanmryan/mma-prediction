"""SP7's screen: why the simulator's winner opinion never reached the harness.

The pre-registration (docs/superpowers/plans/2026-09-29-sp7-simulator-third-member.md)
gated the expensive arm behind a cheap mechanism screen, with both thresholds
fixed in writing first. The screen failed, so no walk-forward arm was ever run
-- which is the gate working, not the gate being skipped.

These tests pin the three things that make the negative trustworthy: that the
four prediction streams are joined by `fight_id` and not by position, that the
threshold the screen was judged against is re-derived from the current table
rather than trusted as a number copied out of SP6, and that the committed
artifact is reproducible from the committed dumps.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.sp7_decision import (
    REFERENCE_PAIR_CORR,
    ARTIFACT,
    correlations,
    load_streams,
    screen,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_streams_are_joined_by_fight_id_not_by_position():
    """All four dumps carry `fight_id`, so there is no excuse for a positional
    join here -- and a silent positional mismatch is the bug class that broke
    the market benchmark once. The join is only sound if the outcome column
    agrees afterwards, so `load_streams` checks it."""
    y, streams = load_streams()
    assert set(streams) == {"xgb", "torch", "simulator", "blend"}
    assert all(len(p) == len(y) for p in streams.values())


def test_a_stream_whose_outcomes_disagree_after_the_join_is_refused(tmp_path):
    y, streams = load_streams()
    bad = tmp_path / "bad.json"
    src = json.loads((ROOT / "models" / "walkforward" / "preds"
                      / "sp7_screen_hazard.json").read_text())
    bad.write_text(json.dumps(dict(src, y_winner=list(reversed(src["y_winner"])))))
    with pytest.raises(ValueError, match="outcome"):
        load_streams({"blend": "hybrid_e2_cells.json", "simulator": str(bad)})


def test_the_threshold_is_re_derived_from_the_current_table_not_trusted():
    """The screen's threshold is the incumbent pair's own correlation, which
    SP6 measured at 0.8518 on a 4,804-fight table. Judging a new candidate
    against a stale reference would be judging it against nothing, so the
    decision script re-measures it -- and this asserts the two agree."""
    _, streams = load_streams()
    live = correlations(streams)[("xgb", "torch")]
    assert live == pytest.approx(REFERENCE_PAIR_CORR, abs=0.01)


def test_the_screen_fails_and_names_the_clause_that_failed():
    report = screen()
    assert report["verdict"] == "FAIL"
    assert report["clauses"]["standalone_within_0.005_of_blend"]["passed"] is True
    assert report["clauses"]["correlation_below_reference"]["passed"] is False


def test_the_recorded_prediction_is_kept_beside_the_outcome():
    """The plan predicted 0.60-0.80 before measuring. A pre-registration whose
    wrong predictions quietly vanish is not a pre-registration."""
    report = screen()
    predicted = report["predicted_before_measuring"]
    assert predicted["range"] == [0.60, 0.80]
    assert predicted["was_correct"] is False


def test_the_committed_artifact_is_reproducible():
    if not ARTIFACT.exists():
        pytest.skip("run scripts/sp7_decision.py to write the artifact")
    committed = json.loads(ARTIFACT.read_text())
    assert screen()["correlations"] == committed["correlations"]
    assert screen()["verdict"] == committed["verdict"]
