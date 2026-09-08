"""Glicko-2: a rating that carries its own uncertainty and volatility.

Glickman's published algorithm, implemented step by step (the step numbers in
the comments are his). A `Rating` is the familiar triple -- rating on the
1500-point scale, rating deviation on the 350-point scale, and volatility --
and `update` advances one rating period given that period's opponents and
scores. `tests/test_glicko.py` pins the implementation to the worked example
in the paper (1500/200 vs 1400/30, 1550/100, 1700/300 with W/L/L ->
1464.06/151.52), which exercises every step of the chain.

Why it exists here: Elo tracks a level and nothing else, so it cannot say
whether a 1600 is a settled 1600 or a two-fight fluke. `phi` is exactly that
distinction, and `sigma` is how erratic the fighter's results have been.
SP1's elo-v1.1 rejected Glicko as a *replacement* for the Elo columns; SP2's
`trajectory` block tests it as an addition.

Two deliberate choices for MMA, both outside the paper:

* **Rating periods are event dates.** Every fight on a card is scored against
  the same pre-card ratings, which is what makes a fighter's pre-fight
  rating point-in-time correct even when two of their bouts share a date.
* **Inactivity is measured in weeks, not in periods elapsed.** The paper
  grows RD by one step per *empty rating period*, which for a per-event-date
  clock would make a fighter's RD depend on how many cards the promotion
  happened to run while they were out -- roughly four a year in 1995 and
  fifty in 2025. `decay` therefore takes a fractional number of periods and
  `PERIOD_DAYS = 7` converts elapsed days into them: one nominal period per
  week, which is both era-independent and close to the modern UFC's actual
  cadence. It is also what lets serving reproduce a training value exactly,
  from the fighter's post-fight state and the number of days since.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Glickman's constants. SCALE is the conversion between the familiar
# 1500/350 scale and the internal one the algorithm works on.
SCALE = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOLATILITY = 0.06
TAU = 0.5
EPSILON = 1e-6
# An RD of 350 is the paper's "we know nothing" value, so nothing an idle
# fighter accrues should exceed it.
MAX_RD = 350.0
# One nominal rating period, in days. See the module docstring.
PERIOD_DAYS = 7.0


@dataclass(frozen=True)
class Rating:
    """A fighter's Glicko-2 state on the familiar (1500, 350) scale."""

    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    volatility: float = DEFAULT_VOLATILITY

    @property
    def mu(self) -> float:
        """The rating on the algorithm's internal scale (step 2)."""
        return (self.rating - DEFAULT_RATING) / SCALE

    @property
    def phi(self) -> float:
        """The deviation on the algorithm's internal scale (step 2)."""
        return self.rd / SCALE


def _g(phi: float) -> float:
    """Step 3's g(phi): how much weight an opponent's result carries."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _expected(mu: float, opponent_mu: float, opponent_phi: float) -> float:
    """Step 3's E: probability of beating an opponent, damped by their RD."""
    return 1.0 / (1.0 + math.exp(-_g(opponent_phi) * (mu - opponent_mu)))


def expected_score(player: Rating, opponent: Rating) -> float:
    """P(player beats opponent) under the Glicko-2 model, for callers."""
    return _expected(player.mu, opponent.mu, opponent.phi)


def _new_volatility(phi: float, v: float, delta: float, sigma: float,
                    tau: float, epsilon: float) -> float:
    """Step 5: the illinois-variant root find for the new volatility.

    Solves f(x) = 0 for x = ln(sigma'^2). The bracket and the iteration are
    the paper's verbatim -- in particular the `k`-doubling search for the
    lower bound, which is what keeps the method safe when a period's results
    are far more surprising than the current volatility expects.
    """
    a = math.log(sigma * sigma)
    phi_sq, delta_sq = phi * phi, delta * delta

    def f(x: float) -> float:
        exp_x = math.exp(x)
        denominator = phi_sq + v + exp_x
        return (
            exp_x * (delta_sq - phi_sq - v - exp_x) / (2.0 * denominator * denominator)
            - (x - a) / (tau * tau)
        )

    upper = a
    if delta_sq > phi_sq + v:
        lower = math.log(delta_sq - phi_sq - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        lower = a - k * tau
    f_upper, f_lower = f(upper), f(lower)
    while abs(lower - upper) > epsilon:
        c = upper + (upper - lower) * f_upper / (f_lower - f_upper)
        f_c = f(c)
        if f_c * f_lower <= 0:
            upper, f_upper = lower, f_lower
        else:
            f_upper /= 2.0
        lower, f_lower = c, f_c
    return math.exp(lower / 2.0)


def decay(rating: Rating, periods: float = 1.0, max_rd: float = MAX_RD) -> Rating:
    """RD growth over `periods` idle rating periods (step 6, generalised).

    The paper's single idle period is `phi* = sqrt(phi^2 + sigma^2)`; a
    fractional or repeated count is the same random walk run for longer, so
    the variance term scales linearly with the number of periods. The result
    is capped at `max_rd`, because a rating deviation above the initial 350
    would claim to know less than nothing about the fighter.
    """
    if periods <= 0:
        return rating
    phi = math.sqrt(rating.phi ** 2 + rating.volatility ** 2 * float(periods))
    return Rating(rating.rating, min(phi * SCALE, max_rd), rating.volatility)


def decay_days(rating: Rating, days: float | None, max_rd: float = MAX_RD) -> Rating:
    """`decay` over a real elapsed time, in `PERIOD_DAYS`-long periods.

    This is the single place training and serving agree on how a lay-off
    turns into uncertainty: `scripts/build_ratings.py` calls it with the days
    since the fighter's last card, `mma.inference.build_matchup` with the
    days since their last fight as of the event being predicted.
    """
    if days is None or not (float(days) > 0):
        return rating
    return decay(rating, float(days) / PERIOD_DAYS, max_rd)


def update(rating: Rating, results, tau: float = TAU,
           epsilon: float = EPSILON, max_rd: float = MAX_RD) -> Rating:
    """One rating period. `results` is (opponent Rating, score) pairs.

    Scores are 1.0 / 0.5 / 0.0. An empty period is the paper's step 6 alone:
    the rating and volatility stand and the RD grows by one period.
    """
    results = list(results)
    if not results:
        return decay(rating, 1.0, max_rd)

    mu, phi, sigma = rating.mu, rating.phi, rating.volatility

    # Step 3: the estimated variance of the rating from the period's games.
    variance_terms = []
    outcome_terms = []
    for opponent, score in results:
        g = _g(opponent.phi)
        e = _expected(mu, opponent.mu, opponent.phi)
        variance_terms.append(g * g * e * (1.0 - e))
        outcome_terms.append(g * (float(score) - e))
    v = 1.0 / sum(variance_terms)

    # Step 4: the estimated improvement, in rating units.
    outcome = sum(outcome_terms)
    delta = v * outcome

    # Step 5-6: the new volatility, then the pre-update RD.
    sigma_prime = _new_volatility(phi, v, delta, sigma, tau, epsilon)
    phi_star = math.sqrt(phi * phi + sigma_prime * sigma_prime)

    # Step 7: the new RD and rating.
    phi_prime = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_prime = mu + phi_prime * phi_prime * outcome

    # Step 8: back to the familiar scale.
    return Rating(
        mu_prime * SCALE + DEFAULT_RATING,
        min(phi_prime * SCALE, max_rd),
        sigma_prime,
    )


# --- the MMA pass ------------------------------------------------------------

_SCORES = {"a": (1.0, 0.0), "b": (0.0, 1.0), "draw": (0.5, 0.5)}


def run_glicko(fights, tau: float = TAU) -> "pd.DataFrame":
    """Chronological Glicko-2 pass over the fights table, one row per corner.

    Shaped exactly like `mma.elo.run_elo` -- same row set, same key columns --
    so `scripts/build_ratings.py` can merge the two on (fight_id, corner,
    fighter_id) and leave the Elo columns untouched.

    A rating period is an EVENT DATE: every bout on a card is scored against
    the ratings each fighter carried into that card, and all of a card's
    results are folded in together. That is what makes the pre-fight values
    point-in-time correct even for a fighter with two bouts on one night, and
    it is the reason this cannot be a per-fight loop like the Elo pass.

    Between cards, a fighter's RD grows with the calendar rather than with the
    number of cards they sat out -- see `decay_days` and the module docstring.
    No-contests are skipped entirely, as in the Elo pass.
    """
    import pandas as pd

    states: dict[str, Rating] = {}
    last_seen: dict[str, "pd.Timestamp"] = {}
    rows = []
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for date, card in ordered.groupby("date", sort=True):
        bouts = []
        pre: dict[str, Rating] = {}
        for fight in card.itertuples(index=False):
            if fight.winner not in _SCORES:
                continue
            score_a, score_b = _SCORES[fight.winner]
            id_a, id_b = fight.fighter_a_id, fight.fighter_b_id
            if id_a == id_b:
                raise ValueError(f"fight {fight.fight_id} has both corners = {id_a}")
            bouts.append((fight.fight_id, id_a, id_b, score_a, score_b))
            for fighter in (id_a, id_b):
                if fighter not in pre:
                    rating = states.get(fighter, Rating())
                    if fighter in last_seen:
                        rating = decay_days(rating, (date - last_seen[fighter]).days)
                    pre[fighter] = rating
        if not bouts:
            continue

        period: dict[str, list] = {fighter: [] for fighter in pre}
        for _, id_a, id_b, score_a, score_b in bouts:
            period[id_a].append((pre[id_b], score_a))
            period[id_b].append((pre[id_a], score_b))
        post = {
            fighter: update(pre[fighter], results, tau)
            for fighter, results in period.items()
        }

        for fight_id, id_a, id_b, _, _ in bouts:
            for fighter, corner in ((id_a, "a"), (id_b, "b")):
                rows.append({
                    "fight_id": fight_id,
                    "corner": corner,
                    "fighter_id": fighter,
                    "pre_glicko_mu": pre[fighter].rating,
                    "pre_glicko_phi": pre[fighter].rd,
                    "pre_glicko_sigma": pre[fighter].volatility,
                    "post_glicko_mu": post[fighter].rating,
                    "post_glicko_phi": post[fighter].rd,
                    "post_glicko_sigma": post[fighter].volatility,
                })
        states.update(post)
        for fighter in pre:
            last_seen[fighter] = date

    ratings = pd.DataFrame(rows)
    for column in ("fight_id", "corner", "fighter_id"):
        ratings[column] = ratings[column].astype("string")
    return ratings
