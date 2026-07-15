from __future__ import annotations

from datetime import date

import pytest

from scripts.predict_upcoming import (
    ensure_scheduled_events_parsed,
    select_upcoming_events,
    warn_if_empty_fight_card,
)


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


def test_ensure_scheduled_events_parsed_exits_1_on_empty(capsys):
    with pytest.raises(SystemExit) as excinfo:
        ensure_scheduled_events_parsed([])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "page structure likely changed" in err


def test_ensure_scheduled_events_parsed_passes_with_events():
    ensure_scheduled_events_parsed([{"event_name": "UFC 331", "date": "2026-09-19"}])


def test_warn_if_empty_fight_card_fires_and_names_event(capsys):
    fired = warn_if_empty_fight_card("UFC Fight Night: Broken Page", [])
    assert fired is True
    err = capsys.readouterr().err
    assert "UFC Fight Night: Broken Page" in err
    assert "0 fights" in err


def test_warn_if_empty_fight_card_silent_when_card_parsed(capsys):
    fights = [{"fighter_a_name": "A", "fighter_b_name": "B"}]
    assert warn_if_empty_fight_card("UFC 999", fights) is False
    assert capsys.readouterr().err == ""
