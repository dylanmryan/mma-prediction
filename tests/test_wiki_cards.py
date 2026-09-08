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


# --- Background prose: withdrawals, replacements, missed weight -------------
#
# Fixtures are the Background section of three real event pages, one per era
# the parser has to survive: UFC 196 (2016), UFC 302 (2024) and UFC 326
# (2026). Each is trimmed to that section and carries a comment naming its
# source URL and licence (CC BY-SA 4.0). UFC 196 doubles as the "no such
# notes" case for weigh-ins: it has withdrawals and no missed weight at all.


def _background(name: str) -> str:
    return (FIXTURES / f"{name}_background.html").read_text()


NO_BACKGROUND = "<p>An event page with no Background section at all.</p>"
EMPTY_BACKGROUND = (
    '<div class="mw-heading mw-heading2"><h2 id="Background">Background</h2></div>'
    "<p>The event was announced in March and took place as scheduled.</p>"
)


def test_background_paragraphs_strips_reference_markers():
    from mma.wiki_cards import background_paragraphs

    paragraphs = background_paragraphs(_background("ufc196"))
    assert paragraphs, "fixture should have Background prose"
    assert not any("[ 7 ]" in p or "[7]" in p for p in paragraphs)


def test_background_paragraphs_stops_at_the_next_section():
    from mma.wiki_cards import background_paragraphs

    paragraphs = background_paragraphs(_background("ufc302"))
    assert paragraphs
    assert not any("Bonus awards" in p or "See also" in p for p in paragraphs)


def test_a_page_with_no_background_section_yields_nothing():
    from mma.wiki_cards import parse_background

    assert parse_background(NO_BACKGROUND) == {"withdrawals": [], "missed_weight": []}


def test_a_background_with_no_such_notes_yields_nothing():
    from mma.wiki_cards import parse_background

    assert parse_background(EMPTY_BACKGROUND) == {"withdrawals": [], "missed_weight": []}


def test_2016_page_reports_withdrawals_and_no_missed_weight():
    """UFC 196: dos Anjos and Johnson both pulled out; nobody missed weight."""
    from mma.wiki_cards import parse_background

    parsed = parse_background(_background("ufc196"))
    assert parsed["missed_weight"] == []
    withdrew = {entry["withdrew"] for entry in parsed["withdrawals"]}
    assert {"dos Anjos", "Johnson"} <= withdrew
    assert all(entry["replacement"] is None for entry in parsed["withdrawals"])


def test_2024_page_reports_a_replacement_chain_and_a_weigh_in_miss():
    """UFC 302: Park -> Tumendemberel -> Raposo, and Lima four pounds over."""
    from mma.wiki_cards import parse_background

    parsed = parse_background(_background("ufc302"))
    replacements = [entry["replacement"] for entry in parsed["withdrawals"]]
    assert "Nyamjargal Tumendemberel" in replacements
    assert "Mitch Raposo" in replacements
    assert "Alex Morono" in replacements
    assert parsed["missed_weight"] == [{
        "fighter": "Lima", "weight_lbs": 130.0, "over_lbs": 4.0,
        "text": parsed["missed_weight"][0]["text"],
    }]
    assert "flyweight non-title fight limit" in parsed["missed_weight"][0]["text"]


def test_2026_page_reports_withdrawals_with_titled_replacements():
    """UFC 326: 'replaced by former LFA Middleweight Champion Gregory
    Rodrigues' -- the capitalised title has to come off the name."""
    from mma.wiki_cards import parse_background

    parsed = parse_background(_background("ufc326"))
    pairs = {(e["withdrew"], e["replacement"]) for e in parsed["withdrawals"]}
    assert ("Costa", "Gregory Rodrigues") in pairs
    assert ("Todorović", "Cody Brundage") in pairs
    assert ("Yoo", "Lee Jeong-yeong") in pairs
    assert parsed["missed_weight"] == []


def test_notice_days_are_returned_when_stated_and_none_when_not():
    from mma.wiki_cards import parse_background

    html = (
        '<div class="mw-heading"><h2 id="Background">Background</h2></div>'
        "<p>Smith stepped in on 11 days notice to replace Jones. "
        "Doe accepted the bout on short notice. "
        "Roe withdrew due to injury and was replaced by Ann Poe.</p>"
    )
    entries = parse_background(html)["withdrawals"]
    assert [e["notice_days"] for e in entries] == [11, None, None]
    assert [e["short_notice"] for e in entries] == [True, True, False]
    assert entries[2] == {
        "withdrew": "Roe", "replacement": "Ann Poe", "notice_days": None,
        "short_notice": False, "text": entries[2]["text"],
    }


def test_a_bare_missed_weight_mention_is_reported_without_numbers():
    from mma.wiki_cards import parse_background

    html = (
        '<div class="mw-heading"><h2 id="Background">Background</h2></div>'
        "<p>The bout was scrapped after Bonfim missed weight.</p>"
    )
    assert parse_background(html)["missed_weight"] == [{
        "fighter": "Bonfim", "weight_lbs": None, "over_lbs": None,
        "text": "The bout was scrapped after Bonfim missed weight.",
    }]


def test_two_weigh_in_misses_in_one_sentence_are_both_reported():
    from mma.wiki_cards import parse_background

    html = (
        '<div class="mw-heading"><h2 id="Background">Background</h2></div>'
        "<p>Taveras weighed in at 139.75 pounds, three and three quarters pounds "
        "over the bantamweight non-title fight limit and Gordon weighed in at "
        "127.5 pounds, one and a half pounds over the flyweight limit.</p>"
    )
    misses = parse_background(html)["missed_weight"]
    assert [(m["fighter"], m["weight_lbs"], m["over_lbs"]) for m in misses] == [
        ("Taveras", 139.75, 3.75), ("Gordon", 127.5, 1.5),
    ]


@pytest.mark.parametrize("text,expected", [
    ("4", 4.0), ("3.75", 3.75), ("one", 1.0), ("half", 0.5),
    ("one and a half", 1.5), ("three and three quarters", 3.75),
    ("two and a quarter", 2.25), ("eleven", 11.0),
    ("zebra", None), ("", None), ("one and zebra", None),
])
def test_parse_number_handles_digits_and_words(text, expected):
    from mma.wiki_cards import parse_number

    assert parse_number(text) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Mitch Raposo.", "Mitch Raposo"),
    ("Raul Rosas Jr.", "Raul Rosas Jr."),
    ("former LFA Middleweight Champion Gregory Rodrigues", "Gregory Rodrigues"),
    ("promotional newcomer Nyamjargal Tumendemberel", "Nyamjargal Tumendemberel"),
    ("dos Anjos", "dos Anjos"),
    ("Champion", None),
    (None, None),
])
def test_clean_name_strips_titles_and_sentence_periods(raw, expected):
    from mma.wiki_cards import clean_name

    assert clean_name(raw) == expected


def test_parse_background_touches_no_network(monkeypatch):
    import urllib.request
    from mma.wiki_cards import parse_background

    def _boom(*args, **kwargs):
        raise AssertionError("parsers must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    for name in ("ufc196", "ufc302", "ufc326"):
        parse_background(_background(name))
