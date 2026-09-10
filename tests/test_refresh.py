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


# --------------------------------------------------------------------------
# two sources, one trigger
# --------------------------------------------------------------------------


def _provenance(primary_ids, secondary_ids):
    rows = [("fights", i, "primary") for i in primary_ids]
    rows += [("fights", i, "secondary") for i in secondary_ids]
    return pd.DataFrame(rows, columns=["table", "row_id", "source"]).astype("string")


def _processed(ids, dates):
    return pd.DataFrame(
        {"fight_id": pd.array(ids, dtype="string"), "date": pd.to_datetime(dates)}
    )


def test_kaggle_is_compared_against_the_primary_rows_only():
    """The processed table now runs ahead of the Kaggle mirror by design, so
    comparing against the whole of it would never see a Kaggle refresh again."""
    raw = pd.DataFrame({"fight_id": ["f1", "f2"], "date": ["2026-08-01", "2026-08-22"]})
    processed = _processed(["f1", "f3"], ["2026-08-01", "2026-09-05"])

    needed, reason = refresh_needed(
        raw, processed, provenance=_provenance(["f1"], ["f3"])
    )

    assert needed is True
    assert "2026-08-22" in reason


def test_a_stale_kaggle_snapshot_still_does_not_trigger_a_refresh():
    raw = pd.DataFrame({"fight_id": ["f1"], "date": ["2026-08-01"]})
    processed = _processed(["f1", "f3"], ["2026-08-01", "2026-09-05"])

    needed, _reason = refresh_needed(
        raw, processed, provenance=_provenance(["f1"], ["f3"])
    )

    assert needed is False


def test_the_daily_scrape_running_ahead_triggers_a_refresh_on_its_own():
    """Without this the tables freeze: Kaggle is stale, so nothing else asks
    for a rebuild, and the fights the scrape already has never arrive."""
    raw = pd.DataFrame({"fight_id": ["f1"], "date": ["2026-08-01"]})
    processed = _processed(["f1"], ["2026-08-01"])

    needed, reason = refresh_needed(
        raw, processed, secondary_max_date=pd.Timestamp("2026-09-05")
    )

    assert needed is True
    assert "2026-09-05" in reason
    assert "scrape" in reason.lower() or "secondary" in reason.lower()


def test_a_secondary_source_that_has_nothing_new_does_not_trigger_a_refresh():
    raw = pd.DataFrame({"fight_id": ["f1"], "date": ["2026-08-01"]})
    processed = _processed(["f1", "f3"], ["2026-08-01", "2026-09-05"])

    needed, _reason = refresh_needed(
        raw, processed,
        provenance=_provenance(["f1"], ["f3"]),
        secondary_max_date=pd.Timestamp("2026-09-05"),
    )

    assert needed is False


def test_an_unreachable_secondary_source_leaves_the_kaggle_decision_alone():
    raw = pd.DataFrame({"fight_id": ["f1"], "date": ["2026-08-01"]})
    processed = _processed(["f1"], ["2026-08-01"])

    needed, _reason = refresh_needed(raw, processed, secondary_max_date=None)

    assert needed is False
