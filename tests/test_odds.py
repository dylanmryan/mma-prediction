"""Unit tests for mma.odds -- pure functions only, no network."""
from __future__ import annotations

import math

import pytest

from mma.odds import (
    american_to_implied,
    consensus_odds,
    decimal_to_implied,
    devig_multiway,
    devig_pair,
    extract_fight_id,
)


def test_decimal_to_implied_evens():
    assert decimal_to_implied(2.0) == pytest.approx(0.5)


def test_decimal_to_implied_heavy_favorite():
    assert decimal_to_implied(1.25) == pytest.approx(0.8)


def test_decimal_to_implied_underdog():
    assert decimal_to_implied(4.0) == pytest.approx(0.25)


def test_american_to_implied_favorite():
    # -190 -> 190 / (190 + 100)
    assert american_to_implied(-190) == pytest.approx(190 / 290)


def test_american_to_implied_underdog():
    # +165 -> 100 / (165 + 100)
    assert american_to_implied(165) == pytest.approx(100 / 265)


def test_american_to_implied_zero_raises():
    with pytest.raises(ValueError):
        american_to_implied(0)


def test_devig_pair_sums_to_one():
    p1, p2 = devig_pair(0.6, 0.5)
    assert p1 + p2 == pytest.approx(1.0)
    # relative order preserved
    assert p1 > p2


def test_devig_pair_hand_computed_from_american_minus190_plus165():
    """-190 / +165 American -> devigged pair sums to 1, favorite ~0.63."""
    p1_raw = american_to_implied(-190)
    p2_raw = american_to_implied(165)
    p1, p2 = devig_pair(p1_raw, p2_raw)
    assert p1 + p2 == pytest.approx(1.0)
    assert p1 == pytest.approx(0.634, abs=0.01)
    assert p2 == pytest.approx(0.366, abs=0.01)


def test_devig_pair_already_fair_unchanged():
    p1, p2 = devig_pair(0.5, 0.5)
    assert p1 == pytest.approx(0.5)
    assert p2 == pytest.approx(0.5)


def test_extract_fight_id_valid_url():
    url = "http://ufcstats.com/fight-details/d215c4e6dc1346ae"
    assert extract_fight_id(url) == "d215c4e6dc1346ae"


def test_extract_fight_id_trailing_slash():
    url = "http://ufcstats.com/fight-details/d215c4e6dc1346ae/"
    assert extract_fight_id(url) == "d215c4e6dc1346ae"


def test_extract_fight_id_none_input():
    assert extract_fight_id(None) is None


def test_extract_fight_id_empty_string():
    assert extract_fight_id("") is None


def test_extract_fight_id_no_hex_id():
    assert extract_fight_id("http://ufcstats.com/fight-details/") is None


def test_extract_fight_id_rejects_short_hex():
    # only 8 hex chars -- not the 16-hex ufcstats scheme
    assert extract_fight_id("http://ufcstats.com/fight-details/d215c4e6") is None


def test_consensus_odds_single_book():
    rows = [{"odds_1": 2.0, "odds_2": 2.0}]
    p1, p2 = consensus_odds(rows)
    assert p1 == pytest.approx(0.5)
    assert p2 == pytest.approx(0.5)


def test_consensus_odds_multi_book_median_and_devig():
    rows = [
        {"odds_1": 1.5, "odds_2": 3.0},
        {"odds_1": 1.4, "odds_2": 3.2},
        {"odds_1": 1.6, "odds_2": 2.8},
    ]
    # median decimal odds: side 1 -> 1.5, side 2 -> 3.0
    expected_p1_raw = decimal_to_implied(1.5)
    expected_p2_raw = decimal_to_implied(3.0)
    expected_p1, expected_p2 = devig_pair(expected_p1_raw, expected_p2_raw)
    p1, p2 = consensus_odds(rows)
    assert p1 == pytest.approx(expected_p1)
    assert p2 == pytest.approx(expected_p2)
    assert p1 + p2 == pytest.approx(1.0)
    assert p1 > 0.5  # side 1 is the favorite here


def test_consensus_odds_ignores_nan_rows():
    rows = [
        {"odds_1": 1.5, "odds_2": 3.0},
        {"odds_1": float("nan"), "odds_2": float("nan")},
    ]
    p1, p2 = consensus_odds(rows)
    assert p1 + p2 == pytest.approx(1.0)
    expected_p1, expected_p2 = devig_pair(
        decimal_to_implied(1.5), decimal_to_implied(3.0)
    )
    assert p1 == pytest.approx(expected_p1)
    assert p2 == pytest.approx(expected_p2)


def test_consensus_odds_ignores_none_rows():
    rows = [
        {"odds_1": 1.5, "odds_2": 3.0},
        {"odds_1": None, "odds_2": None},
    ]
    p1, p2 = consensus_odds(rows)
    assert p1 + p2 == pytest.approx(1.0)


def test_consensus_odds_empty_raises():
    with pytest.raises(ValueError):
        consensus_odds([])


def test_consensus_odds_all_nan_raises():
    with pytest.raises(ValueError):
        consensus_odds([{"odds_1": float("nan"), "odds_2": float("nan")}])


# --------------------------------------------------------------------------
# multi-way devig (the six-way fighter x method prop market)
# --------------------------------------------------------------------------


def test_devig_multiway_sums_to_one():
    out = devig_multiway([0.30, 0.25, 0.20, 0.18, 0.15, 0.12])
    assert math.isclose(sum(out), 1.0, rel_tol=0, abs_tol=1e-12)


def test_devig_multiway_reduces_to_devig_pair_on_a_two_way_market():
    """The whole point of the family choice: two outcomes must give devig_pair.

    A different devig family (power, Shin) would NOT agree with devig_pair
    here, so this is the test that pins which family was used.
    """
    for p1, p2 in ((0.6552, 0.3774), (0.5, 0.5), (0.91, 0.14), (0.2, 0.85)):
        assert devig_multiway([p1, p2]) == pytest.approx(devig_pair(p1, p2))


def test_devig_multiway_preserves_ratios():
    """Proportional devig scales every outcome by the same factor."""
    raw = [0.44, 0.22, 0.11, 0.33, 0.10, 0.05]
    out = devig_multiway(raw)
    ratios = [o / r for o, r in zip(out, raw)]
    assert ratios == pytest.approx([ratios[0]] * len(raw))


def test_devig_multiway_already_fair_unchanged():
    raw = [0.5, 0.25, 0.25]
    assert devig_multiway(raw) == pytest.approx(raw)


def test_devig_multiway_hand_computed():
    raw = [0.5, 0.4, 0.3]  # overround 1.2
    assert devig_multiway(raw) == pytest.approx([0.5 / 1.2, 0.4 / 1.2, 0.3 / 1.2])


def test_devig_multiway_rejects_fewer_than_two_outcomes():
    with pytest.raises(ValueError, match="at least two"):
        devig_multiway([1.0])


def test_devig_multiway_rejects_non_positive_probability():
    with pytest.raises(ValueError, match="strictly positive"):
        devig_multiway([0.5, 0.0, 0.4])


def test_devig_multiway_rejects_non_finite():
    with pytest.raises(ValueError, match="strictly positive"):
        devig_multiway([0.5, float("nan"), 0.4])


def test_decimal_odds_to_devigged_six_way_end_to_end():
    """Decimal prop odds in, a six-way distribution out."""
    decimals = [3.0, 6.0, 8.0, 12.0, 4.0, 5.0]
    raw = [decimal_to_implied(d) for d in decimals]
    out = devig_multiway(raw)
    assert math.isclose(sum(out), 1.0, abs_tol=1e-12)
    assert out[0] > out[1] > out[2]  # ordering survives the normalisation
