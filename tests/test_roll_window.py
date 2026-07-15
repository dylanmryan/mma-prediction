from __future__ import annotations

import pandas as pd
import pytest

from scripts.roll_window import (
    current_data_cutoff,
    decide_promotion,
    graded_fights_since,
    promotion_protocol_text,
)


def test_current_data_cutoff_is_max_date():
    features = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2023-06-15", "2022-01-01"])})
    assert current_data_cutoff(features) == pd.Timestamp("2023-06-15")


def _event(event_date, fights):
    return {"event_name": "Test", "event_date": event_date, "fights": fights}


def test_graded_fights_since_excludes_events_before_cutoff():
    cutoff = pd.Timestamp("2026-01-01")
    events = [
        _event("2025-06-01", [{"skipped": False, "actual_winner": "x"}]),  # before cutoff
        _event("2026-06-01", [{"skipped": False, "actual_winner": "y"}]),  # after cutoff
    ]
    graded = graded_fights_since(events, cutoff)
    assert len(graded) == 1
    assert graded[0]["actual_winner"] == "y"


def test_graded_fights_since_excludes_skipped_and_ungraded():
    cutoff = pd.Timestamp("2026-01-01")
    events = [
        _event("2026-06-01", [
            {"skipped": True, "reason": "no match"},
            {"skipped": False},  # not yet graded
            {"skipped": False, "actual_winner": "z"},
        ]),
    ]
    graded = graded_fights_since(events, cutoff)
    assert len(graded) == 1
    assert graded[0]["actual_winner"] == "z"


def test_graded_fights_since_empty():
    assert graded_fights_since([], pd.Timestamp("2026-01-01")) == []


def test_promotion_protocol_text_mentions_threshold_and_margin():
    text = promotion_protocol_text(150, pd.Timestamp("2023-01-01"))
    assert "150" in text
    assert "0.002" in text
    assert "2023-01-01" in text


def test_decide_promotion_beats_margin():
    assert decide_promotion(new_log_loss=0.640, incumbent_log_loss=0.650) is True


def test_decide_promotion_within_margin_rejects():
    # 0.001 improvement -- below the 0.002 margin -> reject
    assert decide_promotion(new_log_loss=0.649, incumbent_log_loss=0.650) is False


def test_decide_promotion_worse_rejects():
    assert decide_promotion(new_log_loss=0.660, incumbent_log_loss=0.650) is False


def test_decide_promotion_custom_margin():
    assert decide_promotion(new_log_loss=0.640, incumbent_log_loss=0.650, margin=0.02) is False
