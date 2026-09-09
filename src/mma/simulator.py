"""Monte Carlo fight simulator: one joint outcome distribution per fight.

The generative process this module plays out is the whole point of SP3.
Instead of three independent heads (winner, method, finish round) whose
product can assert things no fight could do, one process produces every
outcome at once:

    for r = 1 .. R:
        draw the round's outcome from `hazard_probs[r - 1]`, a distribution
        over `mma.hazard.HAZARD_CLASSES`
        -- a_ko / a_sub / b_ko / b_sub ends the fight in round r
        -- survive moves to round r + 1
    if every round survived, draw the winner from `decision_prob` and the
    fight is a decision.

Winner, method, round and "goes the distance" are then read off the same
simulated fights, so they cannot contradict each other.

**Pure.** numpy arrays in, an `OutcomeDistribution` out; no pandas, no
model, no I/O, and the caller's `rng` is the only source of randomness, so
a seed fixes the result exactly.

**Laplace smoothing (alpha = 1, pre-registered in the SP3 plan).** A cell
with no simulated occurrences would score `-log 0` when the realised
outcome lands in it, so every cell that is *reachable for this fight* gets
`alpha` pseudo-counts. Reachability is set by `scheduled_rounds`: round 4
of a three-round fight is not a possible outcome, so it stays exactly zero
and is kept out of the smoothing denominator rather than being handed mass
the fight could never produce. The raw counts are exposed alongside the
smoothed probabilities so the zero-mass fraction can be measured.

**Vectorised across runs.** All `n_runs x n_rounds` round outcomes are
drawn in one `rng.random` call and resolved by comparison against the
per-round cumulative distribution, so there is no Python loop over runs.
10,000 runs for one five-round fight cost well under a millisecond.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mma.hazard import HAZARD_CLASSES

#: Index of `survive` within `HAZARD_CLASSES`; 0..3 are the finishing
#: classes, ordered so that `index // 2` is the winner (0 = the feature
#: table's corner A, 1 = its corner B) and `index % 2` is the method
#: (0 = KO/TKO, 1 = submission).
_SURVIVE = HAZARD_CLASSES.index("survive")
_N_CLASSES = len(HAZARD_CLASSES)

#: Order of `OutcomeDistribution.method_probs`, matching the harness's
#: `mma.models.train_loop.METHOD_CLASSES`.
METHOD_ORDER = ("ko_tko", "submission", "decision")

_TOLERANCE = 1e-6


@dataclass(frozen=True)
class OutcomeDistribution:
    """The joint distribution over one fight's outcome cells.

    `finish_probs` is `(2, 2, R)` -- winner x method x round, with winner
    0 = the feature table's corner A and method 0 = KO/TKO -- and
    `decision_probs` is `(2,)`, the two ways the fight goes to the cards.
    Together they are the joint distribution and sum to 1. `finish_counts`
    and `decision_counts` are the same cells' raw simulated counts, before
    smoothing.

    `n_rounds` is how many rounds were reachable (`scheduled_rounds`
    bounded by the hazard rows supplied); cells beyond it are exactly zero
    in both the probabilities and the counts.
    """

    finish_probs: np.ndarray
    decision_probs: np.ndarray
    finish_counts: np.ndarray
    decision_counts: np.ndarray
    n_runs: int
    n_rounds: int
    alpha: float

    # -- joint ------------------------------------------------------------
    @property
    def joint_probs(self) -> np.ndarray:
        """Every cell as one flat vector: the 4R finish cells then the 2
        decision cells. Sums to 1."""
        return np.concatenate([self.finish_probs.ravel(), self.decision_probs])

    @property
    def valid_finish_mask(self) -> np.ndarray:
        """`(2, 2, R)` bool: cells this fight could actually reach."""
        mask = np.zeros(self.finish_probs.shape, dtype=bool)
        mask[:, :, : self.n_rounds] = True
        return mask

    @property
    def n_valid_cells(self) -> int:
        return 4 * self.n_rounds + 2

    @property
    def n_zero_mass_cells(self) -> int:
        """Reachable cells that no simulated fight landed in -- the quantity
        the plan's zero-mass-fraction check is built from."""
        finish_zero = int(
            (self.finish_counts[:, :, : self.n_rounds] == 0).sum()
        )
        return finish_zero + int((self.decision_counts == 0).sum())

    # -- marginals --------------------------------------------------------
    @property
    def p_a_wins(self) -> float:
        return float(self.finish_probs[0].sum() + self.decision_probs[0])

    @property
    def p_b_wins(self) -> float:
        return float(self.finish_probs[1].sum() + self.decision_probs[1])

    @property
    def method_probs(self) -> np.ndarray:
        """`(3,)` over `METHOD_ORDER`."""
        return np.array(
            [
                self.finish_probs[:, 0, :].sum(),
                self.finish_probs[:, 1, :].sum(),
                self.decision_probs.sum(),
            ],
            dtype=float,
        )

    @property
    def round_probs(self) -> np.ndarray:
        """`(R,)` over rounds 1..R, conditional on the fight ending in a
        finish. Zero everywhere the fight could not reach; sums to 1 unless
        the fight has no finish mass at all (only possible with alpha = 0)."""
        by_round = self.finish_probs.sum(axis=(0, 1))
        total = by_round.sum()
        if total <= 0:
            return np.zeros_like(by_round)
        return by_round / total

    @property
    def p_distance(self) -> float:
        """Mass on the decision cells: P(the fight goes the distance)."""
        return float(self.decision_probs.sum())

    @property
    def p_a_wins_standard_error(self) -> float:
        """Binomial Monte Carlo error on `p_a_wins` at this `n_runs`."""
        p = self.p_a_wins
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / self.n_runs))


def _validate(hazard_probs: np.ndarray, decision_prob: float, n_runs: int) -> np.ndarray:
    hazard = np.asarray(hazard_probs, dtype=float)
    if hazard.ndim != 2 or hazard.shape[1] != _N_CLASSES:
        raise ValueError(
            f"hazard_probs must have shape (R, {_N_CLASSES}); got {hazard.shape}"
        )
    if hazard.shape[0] < 1:
        raise ValueError("hazard_probs must cover at least one round")
    if not np.isfinite(hazard).all() or (hazard < -_TOLERANCE).any():
        raise ValueError("hazard_probs must be finite and non-negative")
    totals = hazard.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6):
        raise ValueError(
            f"hazard_probs rows must sum to 1; got {np.asarray(totals).tolist()}"
        )
    if not np.isfinite(decision_prob) or not 0.0 <= float(decision_prob) <= 1.0:
        raise ValueError(f"decision_prob must be in [0, 1]; got {decision_prob!r}")
    if int(n_runs) < 1:
        raise ValueError(f"n_runs must be at least 1; got {n_runs!r}")
    return hazard


def simulate(
    hazard_probs,
    decision_prob: float,
    scheduled_rounds,
    n_runs: int,
    rng: np.random.Generator,
    alpha: float = 1.0,
) -> OutcomeDistribution:
    """Play the fight out `n_runs` times and return its outcome distribution.

    `hazard_probs` is `(R, 5)`: round r's distribution over
    `mma.hazard.HAZARD_CLASSES`. `decision_prob` is P(the feature table's
    corner A wins on the cards). `scheduled_rounds` bounds which rounds are
    reachable -- `None` (the 45 fights with no recorded value) means every
    round the hazard supplies is reachable, and a value larger than `R`
    simulates the `R` rounds actually supplied. `alpha` is the Laplace
    pseudo-count applied to the reachable cells; the plan fixes it at 1.

    Determinism is entirely `rng`'s: two calls with equally-seeded
    generators return identical counts.
    """
    hazard = _validate(hazard_probs, decision_prob, n_runs)
    n_runs = int(n_runs)
    rounds_available = hazard.shape[0]
    if scheduled_rounds is None:
        n_rounds = rounds_available
    else:
        bound = int(scheduled_rounds)
        if bound < 1:
            raise ValueError(f"scheduled_rounds must be >= 1; got {scheduled_rounds!r}")
        n_rounds = min(rounds_available, bound)
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative; got {alpha!r}")

    # One draw per (run, round), resolved against the round's cumulative
    # distribution: `draws[i, r]` is the class index round r produced in run i.
    cumulative = np.cumsum(hazard[:n_rounds], axis=1)
    # A row's cumsum can stop a few ulps short of 1 -- `_validate` allows 1e-6,
    # and averaging several seeds' softmax rows lands there routinely. A
    # uniform falling in that gap would be counted past the last class, i.e.
    # as a sixth outcome that does not exist. Pinning the final edge at 1.0
    # gives the residual to `survive`, which is what an inverse-CDF sampler
    # should do with it.
    cumulative[:, -1] = 1.0
    uniforms = rng.random((n_runs, n_rounds))
    draws = (uniforms[:, :, None] >= cumulative[None, :, :]).sum(axis=2)

    finished = draws != _SURVIVE
    ended = finished.any(axis=1)
    first = finished.argmax(axis=1)
    outcome = draws[np.arange(n_runs), first]

    # class index in 0..3 -> (winner, method); flattened as index * n_rounds
    # + round so one bincount fills the whole (2, 2, n_rounds) block.
    flat = outcome[ended] * n_rounds + first[ended]
    block = np.bincount(flat, minlength=4 * n_rounds).reshape(2, 2, n_rounds)

    finish_counts = np.zeros((2, 2, rounds_available), dtype=np.int64)
    finish_counts[:, :, :n_rounds] = block

    # Drawn for every run, not just the ones that went the distance, so the
    # decision draw does not consume a variable number of random values.
    a_on_cards = (~ended) & (rng.random(n_runs) < float(decision_prob))
    n_distance = int((~ended).sum())
    n_a = int(a_on_cards.sum())
    decision_counts = np.array([n_a, n_distance - n_a], dtype=np.int64)

    denominator = n_runs + alpha * (4 * n_rounds + 2)
    finish_probs = np.zeros_like(finish_counts, dtype=float)
    finish_probs[:, :, :n_rounds] = (block + alpha) / denominator
    decision_probs = (decision_counts + alpha) / denominator

    for array in (finish_counts, decision_counts, finish_probs, decision_probs):
        array.flags.writeable = False
    return OutcomeDistribution(
        finish_probs=finish_probs,
        decision_probs=decision_probs,
        finish_counts=finish_counts,
        decision_counts=decision_counts,
        n_runs=n_runs,
        n_rounds=n_rounds,
        alpha=float(alpha),
    )


# --------------------------------------------------------------------------
# The ensemble -> joint-cells arithmetic, shared by the evaluated path
# (`mma.candidates.HazardCandidate`) and the served one
# (`mma.inference.SimulatorPredictor`).
#
# Everything below moves a probability, so it lives in exactly one place.
# `mma.serving` enforces the same discipline for the feature row: the model
# that serves has to compute the number the harness measured, and the only
# way to guarantee that is for both callers to run the same code rather than
# two copies of it.
# --------------------------------------------------------------------------

#: Fixed by the SP3 plan before any run: 10,000 simulated fights per member
#: per orientation, and Laplace alpha = 1 over the cells a fight can reach.
#: These are the DEFAULTS the harness ran with; the deployed values live in
#: `models/simulator.json` and are hashed, because both are part of what a
#: prediction is.
DEFAULT_N_RUNS = 10_000
DEFAULT_ALPHA = 1.0
DEFAULT_SIM_SEED = 0
#: Rounds simulated for a fight with no recorded `scheduled_rounds` (45 of
#: them in the training table). `mma.hazard` keeps those fights and leaves the
#: column NA so the model sees it as missing; the simulator still needs a
#: bound, and 3 is the project's standing default for a missing scheduled-round
#: value (`walkforward.slice_masks`, the torch round mask, `mma.blend`).
DEFAULT_ROUNDS = 3


def bucket_rounds(array: np.ndarray, n_classes: int) -> np.ndarray:
    """`(2, 2, R)` over rounds 1..R -> `(2, 2, n_classes)` over the harness's
    round classes, folding rounds 4 and up into the trailing '45' class."""
    head = n_classes - 1
    out = np.zeros(array.shape[:2] + (n_classes,), dtype=array.dtype)
    out[:, :, :min(array.shape[2], head)] = array[:, :, :head]
    if array.shape[2] > head:
        out[:, :, head] = array[:, :, head:].sum(axis=2)
    return out


def mean_over_seeds(members: list) -> np.ndarray:
    """Row-wise mean across seed members; one member is returned untouched.

    The "average the probabilities" convention every ensemble in this project
    uses (`mma.candidates._mean_over_members`, `scripts.train_xgb.mean_proba`),
    including its bit-identical single-member behaviour.
    """
    return np.asarray(members[0]) if len(members) == 1 else np.mean(members, axis=0)


def normalise_rows(probs: np.ndarray) -> np.ndarray:
    """Rescale each row to sum to 1 -- averaging softmax rows across seeds
    leaves them a few ulps off, which `simulate` rejects outright."""
    probs = np.asarray(probs, dtype=float)
    return probs / probs.sum(axis=1, keepdims=True)


def simulate_fights(
    hazard, mirrored, decision, mirror_decision, n_rounds, *,
    n_runs: int, alpha: float, sim_seed: int, n_round_classes: int,
) -> dict:
    """Simulate a batch of fights in both corner orientations -> joint cells.

    `hazard` and `mirrored` are the two orientations' per-round hazard
    distributions, stacked fight after fight: fight i owns
    ``hazard[starts[i] : starts[i] + n_rounds[i]]``, exactly the layout
    `mma.hazard.build_hazard_rows` and `mma.hazard.round_frame` produce.
    `decision` and `mirror_decision` are `(n,)` P(corner A wins on the cards).

    Each fight is played out twice -- once as given, once mirrored -- and the
    two distributions are averaged after the mirrored one is mapped back into
    this table's frame, which for a joint means exchanging its A and B blocks
    rather than flipping a scalar. Getting that backwards would invert half of
    every method and round prediction.

    **Both orientations share one RNG stream** (`default_rng([sim_seed, i])`
    per fight, re-created for each orientation), so a matchup with no corner
    asymmetry comes back at exactly 0.5 rather than 0.5 plus Monte Carlo
    noise.

    Returns the cell layout `mma.evaluate` defines -- the 2 x n_methods x
    n_round_classes finish cells flattened, then the two decision cells --
    plus, per fight, the raw zero-mass flags, the two orientations' Monte
    Carlo standard errors on P(A wins), and how many of its cells were
    reachable at all (a three-round fight can never reach the '45' class).
    """
    n_rounds = np.asarray(n_rounds, dtype=int)
    n = len(n_rounds)
    hazard = np.asarray(hazard, dtype=float)
    mirrored = np.asarray(mirrored, dtype=float)
    if int(n_rounds.sum()) != len(hazard) or len(mirrored) != len(hazard):
        raise ValueError(
            f"hazard rows ({len(hazard)}) and mirrored rows ({len(mirrored)}) must both "
            f"equal the total rounds simulated ({int(n_rounds.sum())})"
        )
    if len(decision) != n or len(mirror_decision) != n:
        raise ValueError(
            f"decision probabilities must be one per fight ({n}); got "
            f"{len(decision)} and {len(mirror_decision)}"
        )

    finish = np.zeros((n, 2, 2, n_round_classes))
    cards = np.zeros((n, 2))
    finish_zero = np.zeros((n, 2, 2, n_round_classes), dtype=bool)
    cards_zero = np.zeros((n, 2), dtype=bool)
    errors = np.zeros((n, 2))
    starts = np.cumsum(n_rounds) - n_rounds
    for i in range(n):
        rounds = int(n_rounds[i])
        window = slice(int(starts[i]), int(starts[i]) + rounds)
        runs = [
            simulate(probs, float(p_cards), rounds, n_runs,
                     np.random.default_rng([sim_seed, i]), alpha=alpha)
            for probs, p_cards in ((hazard[window], decision[i]),
                                   (mirrored[window], mirror_decision[i]))
        ]
        straight, flipped = runs
        finish[i] = 0.5 * (bucket_rounds(straight.finish_probs, n_round_classes)
                           + bucket_rounds(flipped.finish_probs, n_round_classes)[::-1])
        cards[i] = 0.5 * (straight.decision_probs + flipped.decision_probs[::-1])
        finish_zero[i] = (
            (bucket_rounds(straight.finish_counts, n_round_classes) == 0)
            & (bucket_rounds(flipped.finish_counts, n_round_classes)[::-1] == 0)
        )
        cards_zero[i] = (straight.decision_counts == 0) & (flipped.decision_counts[::-1] == 0)
        errors[i] = (straight.p_a_wins_standard_error, flipped.p_a_wins_standard_error)

    reachable = np.minimum(n_rounds, n_round_classes)
    zero_reachable = np.array(
        [int(finish_zero[i, :, :, : reachable[i]].sum()) + int(cards_zero[i].sum())
         for i in range(n)],
        dtype=int,
    )
    return {
        "cells": np.concatenate([finish.reshape(n, -1), cards], axis=1),
        "zero_mass": np.concatenate([finish_zero.reshape(n, -1), cards_zero], axis=1),
        "standard_errors": errors,
        "n_zero_mass_reachable": zero_reachable,
        "n_reachable_cells": 4 * reachable + 2,
    }
