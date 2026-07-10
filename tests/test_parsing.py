import pandas as pd

from mma.parsing import (
    parse_height_inches,
    parse_landed_attempted,
    parse_mmss_seconds,
    parse_percent,
    parse_reach_inches,
)


def test_landed_attempted_basic():
    assert parse_landed_attempted("45 of 118") == (45, 118)


def test_landed_attempted_zero():
    assert parse_landed_attempted("0 of 0") == (0, 0)


def test_landed_attempted_missing():
    assert parse_landed_attempted("--") == (None, None)
    assert parse_landed_attempted(None) == (None, None)
    assert parse_landed_attempted(float("nan")) == (None, None)


def test_mmss_basic():
    assert parse_mmss_seconds("2:35") == 155
    assert parse_mmss_seconds("0:00") == 0


def test_mmss_missing():
    assert parse_mmss_seconds("--") is None
    assert parse_mmss_seconds(None) is None


def test_height_feet_inches():
    assert parse_height_inches("5' 11\"") == 71.0
    assert parse_height_inches("6' 0\"") == 72.0


def test_height_missing():
    assert parse_height_inches("--") is None
    assert parse_height_inches(None) is None


def test_reach():
    assert parse_reach_inches('72"') == 72.0
    assert parse_reach_inches("72") == 72.0


def test_reach_missing():
    assert parse_reach_inches("--") is None


def test_percent():
    assert parse_percent("45%") == 0.45
    assert parse_percent("0%") == 0.0


def test_percent_missing():
    assert parse_percent("---") is None
    assert parse_percent(None) is None


def test_pandas_na_is_missing():
    assert parse_landed_attempted(pd.NA) == (None, None)
    assert parse_mmss_seconds(pd.NA) is None
    assert parse_percent(pd.NA) is None


def test_malformed_inputs_return_missing():
    assert parse_landed_attempted("45 of") == (None, None)
    assert parse_height_inches("5'") is None
    assert parse_mmss_seconds("2:3") is None
