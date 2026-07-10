"""Target-label and fight-context mappers."""
from __future__ import annotations

import re

# Ordered longest/most-specific first so substrings don't shadow
# (Light Heavyweight before Heavyweight, Women's before men's).
WEIGHT_CLASSES = [
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Light Heavyweight",
    "Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Strawweight",
    "Catch Weight",
    "Catchweight",
    "Open Weight",
]


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    return text or None


def map_method(win_by) -> str | None:
    """Raw method string -> 'ko_tko' | 'submission' | 'decision' | None.

    None means the fight is excluded from method modeling (DQ, overturned...).
    """
    text = _clean(win_by)
    if text is None:
        return None
    lower = text.lower()
    if "ko/tko" in lower or "doctor" in lower or lower.startswith("tko"):
        return "ko_tko"
    if lower.startswith("submission"):
        return "submission"
    if lower.startswith("decision"):
        return "decision"
    return None


def decision_subtype(win_by) -> str | None:
    """'Decision - Split' -> 'split'; non-decisions -> None."""
    text = _clean(win_by)
    if text is None or not text.lower().startswith("decision"):
        return None
    for subtype in ("unanimous", "split", "majority"):
        if subtype in text.lower():
            return subtype
    return None


def parse_scheduled_rounds(format_str) -> int | None:
    """'3 Rnd (5-5-5)' -> 3. 'No Time Limit' -> None."""
    text = _clean(format_str)
    if text is None:
        return None
    match = re.match(r"(\d+)\s*Rnd", text)
    if not match:
        return None
    return int(match.group(1))


def parse_weight_class(fight_type) -> str | None:
    """Extract weight class from e.g. 'UFC Middleweight Title Bout'."""
    text = _clean(fight_type)
    if text is None:
        return None
    for weight_class in WEIGHT_CLASSES:
        if weight_class.lower() in text.lower():
            return weight_class
    return None


def is_title_fight(fight_type) -> bool:
    text = _clean(fight_type)
    return text is not None and "title" in text.lower()
