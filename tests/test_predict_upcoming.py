from __future__ import annotations

from datetime import date

from scripts.predict_upcoming import select_upcoming_events


def _events():
    return [
        {"event_name": "UFC 331", "date": "2026-09-19"},
        {"event_name": "UFC Fight Night A", "date": "2026-07-25"},
        {"event_name": "UFC Fight Night B", "date": "2026-07-18"},
        {"event_name": "UFC Past Event", "date": "2026-07-01"},
    ]


def test_select_upcoming_events_filters_to_horizon():
    today = date(2026, 7, 15)
    selected = select_upcoming_events(_events(), today, horizon_days=30)
    names = [e["event_name"] for e in selected]
    assert "UFC Past Event" not in names  # already happened
    assert "UFC Fight Night B" in names
    assert "UFC Fight Night A" in names
    assert "UFC 331" not in names  # 66 days out, beyond horizon


def test_select_upcoming_events_sorted_soonest_first():
    today = date(2026, 7, 15)
    selected = select_upcoming_events(_events(), today, horizon_days=90)
    dates = [e["date"] for e in selected]
    assert dates == sorted(dates)
    assert selected[0]["event_name"] == "UFC Fight Night B"


def test_select_upcoming_events_includes_today_and_boundary():
    today = date(2026, 7, 18)
    selected = select_upcoming_events(
        [{"event_name": "Today", "date": "2026-07-18"},
         {"event_name": "Exactly at horizon", "date": "2026-08-17"},
         {"event_name": "One day past horizon", "date": "2026-08-18"}],
        today, horizon_days=30,
    )
    names = {e["event_name"] for e in selected}
    assert names == {"Today", "Exactly at horizon"}


def test_select_upcoming_events_empty_input():
    assert select_upcoming_events([], date(2026, 7, 15), 30) == []
