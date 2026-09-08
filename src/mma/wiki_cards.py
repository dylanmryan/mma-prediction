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
import re
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


# --- event Background prose: withdrawals, replacements, missed weight --------
#
# Added by SP2 Task 12 for the `notice` feature block. The block did NOT clear
# the walk-forward bar and its feature wiring was reverted (see the plan's
# results table), so nothing in the prediction path calls this today; it is
# kept because it is the only way to get short-notice and missed-weight facts
# for a FUTURE event -- the Bet MMA source behind
# `data/external/fight_notice.parquet` stops at 2024-12-14 -- and a later block
# with a live source would need exactly this.
#
# The prose is templated but not structured, so this reports what a sentence
# SAYS rather than deciding what it means. Two known limitations, both left to
# the caller:
#   * a withdrawal or a missed weight may refer to a DIFFERENT, earlier event
#     ("the pair were previously expected to meet at UFC ... but the bout was
#     scrapped after Bonfim missed weight"); every entry carries its `text` so
#     the caller can check;
#   * days of notice are usually not stated at all, and when they are, they are
#     as often qualitative ("on short notice") as numeric ("on 11 days'
#     notice"). `notice_days` is None in the first case and `short_notice` is
#     True in both, which is why the block binned rather than differenced them.

_NAME_WORD = r"[A-Z][\wÀ-ɏ'’.\-]*"
_NAME_TAIL = (
    r"(?:\s(?:" + _NAME_WORD + r"|d[aeiou]|van|von|der|dos|das|do|de|Jr\.?|Sr\.?|"
    r"I{2,3}|Neto))"
)
# A leading lowercase particle is part of the name ("dos Anjos", "van Zant");
# without it the pattern starts at the capitalised word and reports "Anjos",
# which no fighter index will match.
_PARTICLE = r"(?:(?:d[aeiou]|dos|das|van|von|der)\s)?"
NAME = _PARTICLE + _NAME_WORD + _NAME_TAIL + r"{0,3}"


# Capitalised words Wikipedia puts in FRONT of a fighter's name -- promotions,
# divisions and titles -- which a "run of capitalised words" pattern would
# otherwise swallow ("replaced by former LFA Middleweight Champion Gregory
# Rodrigues"). Listing them explicitly is duller than a heuristic and does not
# mangle a genuine three-word name.
TITLE_WORDS = frozenset("""
Champion Champions Championship Interim Undisputed Contender Series Season
Winner Newcomer Tournament Veteran Prospect Debutant Titleholder
UFC LFA Bellator PFL KSW ONE Invicta Strikeforce WEC PRIDE Titan Cage Warriors
Fighter Ultimate Road The A An DWCS
Flyweight Bantamweight Featherweight Lightweight Welterweight Middleweight
Heavyweight Strawweight Light Women Women's Catchweight
""".split())
# Lower-case words that ARE part of a name and must survive the title strip
# ("dos Anjos", "van Zant"); without this the leading-lowercase rule turns
# "dos Anjos pulled out" into "Anjos".
PARTICLES = frozenset("da de del dos das do van von der di la le ter".split())
_MAX_NAME_WORDS = 3


def clean_name(name: str | None) -> str | None:
    """Trim what the prose glues to a name: titles, articles, sentence periods.

    Three fixes, all from real fixture text:
      * a name at the end of a sentence keeps its full stop
        ("replaced by Mitch Raposo.") -- stripped, except for "Jr."/"Sr.";
      * lower-case role words precede it ("promotional newcomer Nyamjargal
        Tumendemberel") -- dropped, except the name particles in `PARTICLES`;
      * capitalised promotional titles precede it ("former LFA Middleweight
        Champion Gregory Rodrigues") -- dropped via `TITLE_WORDS`.
    Whatever survives is capped at three words and returned as found. A name
    this cannot repair is not guessed at: the caller's matcher
    (`mma.prospective.match_fighter_id`) reports it unmatched instead.
    """
    if name is None:
        return None
    words = name.strip().split()
    while words and words[0].lower() not in PARTICLES and (
        words[0][:1].islower() or words[0] in TITLE_WORDS
    ):
        words = words[1:]
    words = words[:_MAX_NAME_WORDS]
    if words and words[-1].endswith(".") and words[-1] not in ("Jr.", "Sr."):
        words[-1] = words[-1][:-1]
    while words and not words[-1]:
        words.pop()
    return " ".join(words) if words else None


_WITHDREW = re.compile(
    r"(?P<name>" + NAME + r")\s(?:was\s(?:forced\sto\s|later\s)?|had\sto\s|then\s)?"
    r"(?:withdrew|withdraw|pulled\sout|pull\sout|was\sremoved|was\sscratched|"
    r"was\sforced\sout)"
)
# Captures the raw clause after "replaced by" -- titles and all -- because the
# prefixes are themselves capitalised and cannot be excluded by the pattern;
# `clean_name` is what turns it into a name.
_REPLACED_BY = re.compile(r"replaced\sby\s(?P<name>[^.,;()]{1,80})")
_STEPPED_IN = re.compile(r"(?P<name>" + NAME + r")\s(?:stepped\sin|steps\sin)")
_NOTICE_DAYS = re.compile(r"(?:on|with|at)\s(?P<days>[\w\-\s]{1,20}?)\sdays?[’']?\s?notice")
_SHORT_NOTICE = re.compile(r"short\snotice", re.IGNORECASE)
_WEIGHED_IN = re.compile(
    r"(?P<name>" + NAME + r")\sweighed\sin\sat\s(?P<weight>\d+(?:\.\d+)?)\spounds?"
    r"(?:,\s(?P<over>[\w\s\-]{1,40}?)\spounds?\sover)?"
)
_MISSED_WEIGHT = re.compile(r"(?P<name>" + NAME + r")\smissed\sweight")

_NUMBER_WORDS = {
    "a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0,
    "nineteen": 19.0, "twenty": 20.0, "thirty": 30.0, "forty": 40.0,
}
_FRACTION_WORDS = {
    "half": 0.5, "quarter": 0.25, "quarters": 0.25, "third": 1 / 3, "thirds": 1 / 3,
}


def parse_number(text: str) -> float | None:
    """'3.75' / 'three and three quarters' / 'one and a half' -> a float.

    Wikipedia writes weigh-in overages both ways in the same paragraph, so a
    digits-only parser silently drops half of them. Returns None for anything
    it does not recognise rather than guessing a magnitude.
    """
    text = text.strip().lower().replace("-", " ")
    try:
        return float(text)
    except ValueError:
        pass
    whole, fraction, multiplier = 0.0, 0.0, 1.0
    seen = False
    for word in text.split():
        if word in ("and", "a", "an") and seen:
            multiplier = 1.0
            if word in ("a", "an"):
                multiplier = 1.0
            continue
        if word in _NUMBER_WORDS:
            if fraction or (seen and multiplier != 1.0):
                return None
            multiplier = _NUMBER_WORDS[word]
            if not seen:
                whole = _NUMBER_WORDS[word]
                seen = True
            continue
        if word in _FRACTION_WORDS:
            fraction += multiplier * _FRACTION_WORDS[word]
            multiplier = 1.0
            continue
        return None
    if not seen and not fraction:
        return None
    return whole + fraction


def background_paragraphs(html: str) -> list[str]:
    """The Background section's paragraphs as plain text, references stripped.

    Reference superscripts ('[ 15 ]') are removed before the text is taken;
    left in, they land in the middle of the sentences the patterns match.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(id="Background")
    if heading is None:
        return []
    container = heading.find_parent("div") or heading
    paragraphs = []
    for sibling in container.find_next_siblings():
        if sibling.name in ("h2", "h1") or (
            sibling.name == "div" and sibling.find(["h1", "h2"])
        ):
            break
        if sibling.name != "p":
            continue
        clone = BeautifulSoup(str(sibling), "html.parser")
        for reference in clone.select("sup"):
            reference.decompose()
        text = clone.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).replace(" ", " ")
        if text:
            paragraphs.append(text)
    return paragraphs


def _sentences(paragraph: str) -> list[str]:
    """Split on sentence end, keeping 'Jr.'/'Sr.'/initials intact."""
    protected = re.sub(r"\b((?:Jr|Sr|Mr|Ms|St|vs|No)\.)", lambda m: m.group(1).replace(".", "\0"), paragraph)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", protected)
    return [part.replace("\0", ".").strip() for part in parts if part.strip()]


def parse_background(html: str) -> dict[str, list[dict]]:
    """Withdrawals/replacements and missed weight from an event's Background.

    Returns::

        {"withdrawals": [{"withdrew", "replacement", "notice_days",
                          "short_notice", "text"}],
         "missed_weight": [{"fighter", "weight_lbs", "over_lbs", "text"}]}

    `withdrew` is the fighter the prose says pulled out and `replacement` the
    one it says came in (either may be None -- a bout is often cancelled
    outright, and a replacement is sometimes announced in its own sentence).
    `notice_days` is an int only when the prose states a number;
    `short_notice` is True whenever the phrase "short notice" appears, which is
    the far commoner form. A withdrawal entry is emitted for every sentence
    that names one, so a chain of two replacements produces two entries.

    Pure: no network, no fighter-id matching, no judgement about which event a
    sentence refers to. See the module comment above for the two limitations
    that leaves to the caller.
    """
    withdrawals: list[dict] = []
    missed: list[dict] = []
    for paragraph in background_paragraphs(html):
        for sentence in _sentences(paragraph):
            _collect_withdrawal(sentence, withdrawals)
            _collect_missed_weight(sentence, missed)
    return {"withdrawals": withdrawals, "missed_weight": missed}


_ACCEPTED = re.compile(
    r"(?P<name>" + NAME + r")\s(?:accepted|took|agreed\sto\stake|came\sin)"
)


def _collect_withdrawal(sentence: str, out: list[dict]) -> None:
    withdrew = _WITHDREW.search(sentence)
    replaced = _REPLACED_BY.search(sentence) or _STEPPED_IN.search(sentence)
    days_match = _NOTICE_DAYS.search(sentence)
    days = parse_number(days_match.group("days")) if days_match else None
    if withdrew is None and replaced is None:
        # A sentence can state the notice without naming a withdrawal at all
        # ("Doe accepted the bout on short notice"). The named fighter is the
        # one who came IN, so it is recorded as the replacement.
        if days is None and not _SHORT_NOTICE.search(sentence):
            return
        replaced = _ACCEPTED.search(sentence)
        if replaced is None:
            return
    out.append({
        "withdrew": clean_name(withdrew.group("name")) if withdrew else None,
        "replacement": clean_name(replaced.group("name")) if replaced else None,
        "notice_days": int(days) if days is not None else None,
        "short_notice": bool(_SHORT_NOTICE.search(sentence)) or days is not None,
        "text": sentence,
    })


def _collect_missed_weight(sentence: str, out: list[dict]) -> None:
    found = False
    for match in _WEIGHED_IN.finditer(sentence):
        over = match.group("over")
        out.append({
            "fighter": clean_name(match.group("name")),
            "weight_lbs": float(match.group("weight")),
            "over_lbs": parse_number(over) if over else None,
            "text": sentence,
        })
        found = True
    if found:
        return
    for match in _MISSED_WEIGHT.finditer(sentence):
        out.append({
            "fighter": clean_name(match.group("name")),
            "weight_lbs": None,
            "over_lbs": None,
            "text": sentence,
        })
