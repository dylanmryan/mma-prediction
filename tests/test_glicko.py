"""Glicko-2, checked against the worked example in Glickman's own paper.

The paper ("Example of the Glicko-2 system", Mark E. Glickman) walks one
rating period for a player at 1500/200 with volatility 0.06 and tau 0.5 who
faces three opponents -- 1400/30, 1550/100, 1700/300 -- winning the first and
losing the other two, and reports the answer to two decimals: rating 1464.06,
RD 151.52. Anything that reproduces those two numbers has the whole chain
right (the scale conversion, g/E, v, delta, the volatility iteration, and the
two RD steps), which is why this is the anchor test rather than a set of
hand-rolled expectations.
"""
from __future__ import annotations

import pytest

from mma import glicko


def test_the_published_worked_example():
    player = glicko.Rating(1500.0, 200.0, 0.06)
    opponents = [
        (glicko.Rating(1400.0, 30.0), 1.0),
        (glicko.Rating(1550.0, 100.0), 0.0),
        (glicko.Rating(1700.0, 300.0), 0.0),
    ]
    updated = glicko.update(player, opponents)
    # The paper's figures are quoted to two decimals and its worked example
    # carries rounded intermediates, so the agreement is to one unit in the
    # last quoted place (we get 1464.0507 / 151.5165 exactly).
    assert updated.rating == pytest.approx(1464.06, abs=0.01)
    assert updated.rd == pytest.approx(151.52, abs=0.01)
    # The paper also reports the new volatility to five decimals.
    assert updated.volatility == pytest.approx(0.05999, abs=1e-5)


def test_a_rating_period_with_no_games_leaves_the_rating_and_grows_the_rd():
    idle = glicko.update(glicko.Rating(1500.0, 200.0, 0.06), [])
    assert idle.rating == 1500.0
    assert idle.volatility == 0.06
    assert idle.rd > 200.0


def test_rd_grows_with_inactivity_and_is_capped():
    rated = glicko.Rating(1500.0, 60.0, 0.06)
    short = glicko.decay(rated, periods=4.0)
    long = glicko.decay(rated, periods=104.0)
    assert rated.rd < short.rd < long.rd
    assert short.rating == long.rating == 1500.0
    assert glicko.decay(rated, periods=10_000_000.0).rd == pytest.approx(glicko.MAX_RD)
    assert glicko.decay(rated, periods=0.0) == rated


def test_an_upset_moves_an_uncertain_player_further_than_a_settled_one():
    """The point of carrying RD at all: the same result, two levels of doubt."""
    opponent = (glicko.Rating(1800.0, 50.0), 1.0)  # the underdog wins
    uncertain = glicko.update(glicko.Rating(1500.0, 300.0, 0.06), [opponent])
    settled = glicko.update(glicko.Rating(1500.0, 40.0, 0.06), [opponent])
    assert uncertain.rating - 1500.0 > settled.rating - 1500.0 > 0.0
    # ... and the uncertain player also learns more: a big RD drops sharply,
    # while an already-settled one is barely moved by a single result (its RD
    # even ticks up, because one game is less information than a period of
    # drift costs -- that is the volatility term, not a bug).
    assert uncertain.rd < 300.0 - 40.0
    assert abs(settled.rd - 40.0) < 2.0


def test_beating_a_stronger_opponent_gains_more_than_beating_a_weaker_one():
    player = glicko.Rating(1500.0, 100.0, 0.06)
    over_favourite = glicko.update(player, [(glicko.Rating(1800.0, 50.0), 1.0)])
    over_underdog = glicko.update(player, [(glicko.Rating(1200.0, 50.0), 1.0)])
    assert over_favourite.rating > over_underdog.rating > 1500.0


def test_a_draw_between_equals_leaves_the_rating_alone():
    updated = glicko.update(
        glicko.Rating(1500.0, 100.0, 0.06), [(glicko.Rating(1500.0, 100.0), 0.5)]
    )
    assert updated.rating == pytest.approx(1500.0, abs=1e-9)
    assert updated.rd < 100.0


def test_defaults_match_the_papers_starting_point():
    fresh = glicko.Rating()
    assert (fresh.rating, fresh.rd, fresh.volatility) == (1500.0, 350.0, 0.06)
    assert glicko.TAU == 0.5
    assert glicko.EPSILON == 1e-6
