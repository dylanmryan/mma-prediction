"""Parsers for ufcstats-style raw string values."""
from __future__ import annotations

import re

import pandas as pd

_MISSING = {"", "--", "---", "n/a", "nan"}


def _is_missing(value) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float):  # NaN
        return value != value
    return str(value).strip().lower() in _MISSING


def parse_landed_attempted(value) -> tuple[int | None, int | None]:
    """'45 of 118' -> (45, 118)."""
    if _is_missing(value):
        return (None, None)
    match = re.fullmatch(r"(\d+)\s+of\s+(\d+)", str(value).strip())
    if not match:
        return (None, None)
    return (int(match.group(1)), int(match.group(2)))


def parse_mmss_seconds(value) -> int | None:
    """'2:35' -> 155 seconds."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+):(\d{2})", str(value).strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def parse_height_inches(value) -> float | None:
    """`5' 11"` -> 71.0 inches."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+)'\s*(\d+)\"?", str(value).strip())
    if not match:
        return None
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def parse_reach_inches(value) -> float | None:
    """'72"' -> 72.0 inches."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\"?", str(value).strip())
    if not match:
        return None
    return float(match.group(1))


def parse_percent(value) -> float | None:
    """'45%' -> 0.45."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)%?", str(value).strip())
    if not match:
        return None
    return float(match.group(1)) / 100.0
