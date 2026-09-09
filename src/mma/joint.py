"""Arithmetic on the joint outcome-cell distribution, shared by SP3's candidates.

`mma.evaluate` defines the cell layout and scores a realised cell against
it. This module is the other half: the four operations that BUILD a joint
in that layout, factored out here because SP3's fallback branch needs each
of them in more than one place.

* `compose_joint_cells` — the independence composition P(w)·P(m)·P(r|finish)
  written out as cells. Routing a marginal-head candidate's prediction
  through it and scoring with `joint_cell_log_loss` reproduces
  `joint_outcome_log_loss` exactly, which is what makes it a control:
  the cell machinery contributes nothing of its own.
* `impose_winner_marginal` — replace a joint's winner marginal with a
  given one while leaving each corner's P(method, round | that corner
  wins) untouched. This is the v3 spec's "re-weighting simulated runs"
  written as arithmetic, and it is used twice: by the calibrated simulator
  (D1), where the imposed marginal is the simulator's own temperature-scaled
  winner, and by the hybrid (D2), where it is the incumbent blend's.
* `swap_corners` — exchange a joint's two winner blocks, the cell-layout
  form of "predict the mirrored corner ordering and map it back", which
  `mma.inference.predict_symmetrized` needs to corner-average a served
  matchup's joint.
* `marginals_from_cells` — read the winner/method/round heads back off a
  joint, so a candidate that models cells still reports the three marginals
  every existing harness metric consumes, and reports them from the SAME
  distribution its joint log-loss is computed from.

Pure numpy; no pandas, no model, no I/O.
"""
from __future__ import annotations

import numpy as np

from mma.evaluate import DECISION, n_joint_cells

_EPS = 1e-12


def _split(method_classes, round_classes) -> tuple[int, int]:
    if not method_classes or method_classes[-1] != DECISION:
        raise ValueError(
            f"method_classes must end with {DECISION!r}; got {list(method_classes)}"
        )
    return len(method_classes) - 1, len(round_classes)


def corner_cells(method_classes, round_classes) -> tuple[list[int], list[int]]:
    """Flat cell indices belonging to corner A and to corner B.

    A corner owns its finishing cells and its decision cell, which together
    are everything that counts towards its winner marginal.
    """
    n_methods, n_rounds = _split(method_classes, round_classes)
    block = n_methods * n_rounds
    total = n_joint_cells(method_classes, round_classes)
    a = list(range(block)) + [total - 2]
    b = list(range(block, 2 * block)) + [total - 1]
    return a, b


def swap_corners(cells, method_classes, round_classes) -> np.ndarray:
    """The same joint with corner A's cells and corner B's exchanged.

    Mapping a joint back from the mirrored corner ordering: winner is the one
    axis of the cell layout that names a corner, so mirroring exchanges the
    two winner blocks and leaves method and round where they are. The two
    index lists `corner_cells` returns are in the same (method, round) order,
    so exchanging them elementwise is exactly that swap.

    `mma.simulator.simulate_fights` does the same thing to an
    `OutcomeDistribution` with `[::-1]` on its leading winner axis; this is
    the flat-cell form, for `mma.inference.predict_symmetrized`, which
    averages a served matchup's two orientations after the model has already
    been run on each.
    """
    a_idx, b_idx = corner_cells(method_classes, round_classes)
    cells = np.asarray(cells, dtype=float)
    out = np.array(cells, dtype=float, copy=True)
    out[:, a_idx] = cells[:, b_idx]
    out[:, b_idx] = cells[:, a_idx]
    return out


def compose_joint_cells(winner, method, round_probs, method_classes, round_classes) -> np.ndarray:
    """`(n, n_joint_cells)` from three independent marginals.

    `winner` is P(corner A wins), `method` is `(n, len(method_classes))` with
    the decision class last, and `round_probs` is `(n, len(round_classes))`
    read as P(round | the fight ends in a finish).
    """
    n_methods, n_rounds = _split(method_classes, round_classes)
    p = np.asarray(winner, dtype=float)
    method = np.asarray(method, dtype=float)
    round_probs = np.asarray(round_probs, dtype=float)
    corners = np.stack([p, 1.0 - p], axis=1)  # (n, 2)
    finish = (corners[:, :, None, None]
              * method[:, None, :n_methods, None]
              * round_probs[:, None, None, :])
    cards = corners * method[:, [-1]]
    return np.concatenate([finish.reshape(len(p), -1), cards], axis=1)


def impose_winner_marginal(cells, p_a, method_classes, round_classes) -> np.ndarray:
    """Rescale each corner's cells so the winner marginal is exactly `p_a`.

    Every cell of a corner is multiplied by one scalar, so P(method, round |
    that corner wins) is untouched -- the conditional structure is the whole
    of what the simulator contributes, and imposing a winner must not disturb
    it. A corner carrying no mass at all has no shape of its own to keep and
    borrows the other corner's, which is the only shape available that
    already respects which rounds the fight could reach; if neither corner
    has mass the cells are left as they are.
    """
    a_idx, b_idx = corner_cells(method_classes, round_classes)
    cells = np.asarray(cells, dtype=float)
    out = np.array(cells, dtype=float, copy=True)
    p_a = np.asarray(p_a, dtype=float)
    mass = {"a": cells[:, a_idx].sum(axis=1), "b": cells[:, b_idx].sum(axis=1)}

    # An empty corner borrows the other's normalised shape before scaling.
    for side, other, index, other_index in (("a", "b", a_idx, b_idx), ("b", "a", b_idx, a_idx)):
        empty = (mass[side] <= 0) & (mass[other] > 0)
        if empty.any():
            out[np.ix_(empty, index)] = (
                cells[np.ix_(empty, other_index)] / mass[other][empty][:, None]
            )
            mass[side] = np.where(empty, 1.0, mass[side])

    for index, target, side in ((a_idx, p_a, "a"), (b_idx, 1.0 - p_a, "b")):
        scale = np.where(mass[side] > 0, target / np.maximum(mass[side], _EPS), 1.0)
        out[:, index] *= scale[:, None]
    return out


def marginals_from_cells(cells, method_classes, round_classes) -> dict:
    """The winner/method/round heads implied by a joint over cells.

    `round` is conditional on a finish, matching what the harness's round
    head means; a row with no finish mass at all gets a zero row rather than
    a division by zero.
    """
    n_methods, n_rounds = _split(method_classes, round_classes)
    cells = np.asarray(cells, dtype=float)
    n = len(cells)
    block = n_methods * n_rounds
    finish = cells[:, : 2 * block].reshape(n, 2, n_methods, n_rounds)
    cards = cells[:, 2 * block:]
    a_mass = finish[:, 0].sum(axis=(1, 2)) + cards[:, 0]
    b_mass = finish[:, 1].sum(axis=(1, 2)) + cards[:, 1]
    total = a_mass + b_mass
    finish_mass = finish.sum(axis=(1, 2, 3))
    method = np.stack(
        [finish[:, :, m, :].sum(axis=(1, 2)) for m in range(n_methods)]
        + [cards.sum(axis=1)],
        axis=1,
    )
    denominator = np.where(total > 0, total, 1.0)
    return {
        "winner": a_mass / denominator,
        "method": method / denominator[:, None],
        "round": finish.sum(axis=(1, 2)) / np.where(finish_mass > 0, finish_mass, 1.0)[:, None],
    }
