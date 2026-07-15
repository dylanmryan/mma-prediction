"""Fetch and parse Wikipedia UFC event pages.

Two independent concerns, kept separate on purpose:

- `fetch_page_html` talks to the MediaWiki API (network I/O, untested by
  pytest -- see tests/fixtures/wikipedia for trimmed real HTML instead).
- `parse_scheduled_events` / `parse_fight_card` are pure functions over
  HTML strings, fully covered by fixture-based tests
  (tests/test_wiki_cards.py). This is the only supported way to get
  fight-card data into the pipeline; no other site is scraped.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime

from bs4 import BeautifulSoup

USER_AGENT = "mma-prediction research (github.com/dylanmryan/mma-prediction)"
API_URL = "https://en.wikipedia.org/w/api.php"
MIN_REQUEST_INTERVAL_SEC = 1.0

_last_request_monotonic: float | None = None


def _rate_limit() -> None:
    """Sleep as needed so consecutive calls stay >= 1s apart (polite scraping)."""
    global _last_request_monotonic
    if _last_request_monotonic is not None:
        elapsed = time.monotonic() - _last_request_monotonic
        if elapsed < MIN_REQUEST_INTERVAL_SEC:
            time.sleep(MIN_REQUEST_INTERVAL_SEC - elapsed)
    _last_request_monotonic = time.monotonic()


def fetch_page_html(title: str) -> str:
    """Fetch a Wikipedia page's rendered HTML via action=parse. Network I/O."""
    _rate_limit()
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return payload["parse"]["text"]


def parse_scheduled_events(html: str) -> list[dict]:
    """Parse the 'Scheduled events' table on the 'List of UFC events' page.

    Returns rows in table order (furthest-out first, matching the live
    page): {event_name, wiki_title, date (ISO 'YYYY-MM-DD'), page_url}.
    Rows for events with no Wikipedia page yet (plain text, no link) or an
    unparseable date are skipped -- there is nothing reliable to fetch a
    fight card from.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="Scheduled_events")
    if table is None:
        return []
    events = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        link = cells[0].find("a")
        if link is None or not link.get("href"):
            continue
        wiki_title = link.get("title") or link.get_text(strip=True)
        event_name = link.get_text(strip=True)
        date_span = cells[1].find("span")
        date_text = (date_span or cells[1]).get_text(strip=True)
        parsed_date = _parse_event_date(date_text)
        if parsed_date is None:
            continue
        page_title = wiki_title.replace(" ", "_")
        events.append({
            "event_name": event_name,
            "wiki_title": wiki_title,
            "date": parsed_date.isoformat(),
            "page_url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page_title),
        })
    return events


def _parse_event_date(text: str) -> date | None:
    text = text.strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_fight_card(html: str) -> list[dict]:
    """Parse the 'Fight card' table on a UFC event page.

    Returns fights in listed order (index 0 = main event):
    {fighter_a_name, fighter_b_name, weight_class, title_fight, main_event}.
    A trailing '(c)' marker on a fighter's name flags a title fight and is
    stripped from the name before it's returned.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(id="Fight_card")
    if heading is None:
        return []
    heading_container = heading.find_parent("div") or heading
    table = heading_container.find_next_sibling("table")
    if table is None:
        return []
    fights: list[dict] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        separator = cells[2].get_text(strip=True)
        if "vs" not in separator.lower():
            continue
        weight_class = cells[0].get_text(strip=True)
        fighter_a_name, title_a = _strip_champion_marker(cells[1].get_text(strip=True))
        fighter_b_name, title_b = _strip_champion_marker(cells[3].get_text(strip=True))
        if not fighter_a_name or not fighter_b_name:
            continue
        fights.append({
            "fighter_a_name": fighter_a_name,
            "fighter_b_name": fighter_b_name,
            "weight_class": weight_class,
            "title_fight": title_a or title_b,
            "main_event": len(fights) == 0,
        })
    return fights


def _strip_champion_marker(name: str) -> tuple[str, bool]:
    name = name.strip()
    if name.endswith("(c)"):
        return name[: -len("(c)")].strip(), True
    return name, False
