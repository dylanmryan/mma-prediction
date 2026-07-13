"""Unit tests for the pure refresh-decision function. No network, no file IO."""
import pandas as pd

from scripts.refresh_data import refresh_needed


def test_newer_date_triggers_refresh():
    raw = pd.DataFrame({"date": ["2026-06-01", "2026-07-01"]})
    processed = pd.DataFrame({"date": pd.to_datetime(["2026-05-01", "2026-06-01"])})

    needed, reason = refresh_needed(raw, processed)

    assert needed is True
    assert "2026-07-01" in reason


def test_more_rows_triggers_refresh():
    raw = pd.DataFrame({"date": ["2026-05-01", "2026-05-02", "2026-05-03"]})
    processed = pd.DataFrame({"date": pd.to_datetime(["2026-05-01", "2026-05-02"])})

    needed, reason = refresh_needed(raw, processed)

    assert needed is True
    assert "3" in reason


def test_unchanged_data_does_not_trigger_refresh():
    raw = pd.DataFrame({"date": ["2026-05-01", "2026-05-02"]})
    processed = pd.DataFrame({"date": pd.to_datetime(["2026-05-01", "2026-05-02"])})

    needed, reason = refresh_needed(raw, processed)

    assert needed is False


def test_unparseable_dates_are_safe_and_do_not_trigger_refresh():
    raw = pd.DataFrame({"date": ["not-a-date", "also-not-a-date"]})
    processed = pd.DataFrame({"date": pd.to_datetime(["2026-05-01", "2026-05-02"])})

    needed, reason = refresh_needed(raw, processed)

    assert needed is False
    assert "unparseable" in reason.lower()


def test_empty_processed_bootstrap_triggers_refresh():
    raw = pd.DataFrame({"date": ["2026-05-01", "2026-05-02"]})
    processed = pd.DataFrame({"date": pd.to_datetime([], format="mixed")})

    needed, reason = refresh_needed(raw, processed)

    assert needed is True
