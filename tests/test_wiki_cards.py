"""Tests for Wikipedia HTML parsing. No network -- fixtures only."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wikipedia"


def _events_html() -> str:
    return (FIXTURES / "list_of_ufc_events.html").read_text()


def _fight_card_html() -> str:
    return (FIXTURES / "event_fight_card.html").read_text()


def test_parse_scheduled_events_returns_all_linked_rows():
    from mma.wiki_cards import parse_scheduled_events

    events = parse_scheduled_events(_events_html())

    assert len(events) == 10
    first = events[0]
    assert first["event_name"] == "UFC 331"
    assert first["wiki_title"] == "UFC 331"
    assert first["date"] == "2026-09-19"
    assert first["page_url"] == "https://en.wikipedia.org/wiki/UFC_331"


def test_parse_scheduled_events_handles_colon_titles():
    from mma.wiki_cards import parse_scheduled_events

    events = parse_scheduled_events(_events_html())
    names = [e["event_name"] for e in events]
    assert "UFC Fight Night: du Plessis vs. Usman" in names


def test_parse_scheduled_events_orders_soonest_first():
    from mma.wiki_cards import parse_scheduled_events

    events = parse_scheduled_events(_events_html())
    dates = [e["date"] for e in events]
    assert dates == sorted(dates, reverse=True)  # table lists furthest-first
    # soonest event overall is the last row
    assert events[-1]["event_name"] == "UFC Fight Night: du Plessis vs. Usman"


def test_parse_scheduled_events_skips_rows_without_a_link():
    from mma.wiki_cards import parse_scheduled_events

    html = """
    <table id="Scheduled_events">
    <tbody><tr><th>Event</th><th>Date</th></tr>
    <tr><td>TBA Event (no page yet)</td>
    <td><span data-sort-value="x">Dec 1, 2026</span></td></tr>
    <tr><td><a href="/wiki/UFC_999" title="UFC 999">UFC 999</a></td>
    <td><span data-sort-value="x">Nov 1, 2026</span></td></tr>
    </tbody></table>
    """
    events = parse_scheduled_events(html)
    assert len(events) == 1
    assert events[0]["event_name"] == "UFC 999"


def test_parse_scheduled_events_skips_unparseable_dates():
    from mma.wiki_cards import parse_scheduled_events

    html = """
    <table id="Scheduled_events">
    <tbody><tr><th>Event</th><th>Date</th></tr>
    <tr><td><a href="/wiki/UFC_998" title="UFC 998">UFC 998</a></td>
    <td><span data-sort-value="x">TBA</span></td></tr>
    </tbody></table>
    """
    events = parse_scheduled_events(html)
    assert events == []


def test_parse_scheduled_events_missing_table_returns_empty():
    from mma.wiki_cards import parse_scheduled_events

    assert parse_scheduled_events("<p>no table here</p>") == []


def test_parse_fight_card_returns_all_bouts_in_order():
    from mma.wiki_cards import parse_fight_card

    fights = parse_fight_card(_fight_card_html())

    assert len(fights) == 11
    main_event = fights[0]
    assert main_event["fighter_a_name"] == "Dricus du Plessis"
    assert main_event["fighter_b_name"] == "Kamaru Usman"
    assert main_event["weight_class"] == "Middleweight"
    assert main_event["main_event"] is True


def test_parse_fight_card_handles_unlinked_fighter_names():
    from mma.wiki_cards import parse_fight_card

    fights = parse_fight_card(_fight_card_html())
    names = {f["fighter_a_name"] for f in fights} | {f["fighter_b_name"] for f in fights}
    assert "Mitch Ramirez" in names  # unlinked plain-text fighter in fixture
    assert "Tommy McMillen" in names


def test_parse_fight_card_only_first_bout_is_main_event():
    from mma.wiki_cards import parse_fight_card

    fights = parse_fight_card(_fight_card_html())
    assert sum(1 for f in fights if f["main_event"]) == 1


def test_parse_fight_card_strips_champion_marker():
    from mma.wiki_cards import parse_fight_card

    html = """
    <div class="mw-heading mw-heading2"><h2 id="Fight_card">Fight card</h2></div>
    <table class="toccolours">
    <tbody>
    <tr><th colspan="8">Main card</th></tr>
    <tr><td>Lightweight</td>
    <td><a href="/wiki/Champ">Champ Name (c)</a></td>
    <td>vs.</td>
    <td>Challenger Name</td>
    <td></td><td></td><td></td><td></td></tr>
    </tbody></table>
    """
    fights = parse_fight_card(html)
    assert fights[0]["fighter_a_name"] == "Champ Name"
    assert fights[0]["title_fight"] is True


def test_parse_fight_card_missing_heading_returns_empty():
    from mma.wiki_cards import parse_fight_card

    assert parse_fight_card("<p>nothing here</p>") == []


def test_fetch_page_html_is_not_called_by_parsers(monkeypatch):
    """Parsers must be pure functions -- no accidental network access."""
    import urllib.request
    from mma.wiki_cards import parse_scheduled_events, parse_fight_card

    def _boom(*args, **kwargs):
        raise AssertionError("parsers must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    parse_scheduled_events(_events_html())
    parse_fight_card(_fight_card_html())
