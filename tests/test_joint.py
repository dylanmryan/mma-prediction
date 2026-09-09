"""Cell-layout arithmetic shared by the calibrated simulator and the hybrid.

The three operations tested here are what SP3's fallback branch is built
out of: compose a joint from independent marginals (the D3 control),
impose a winner marginal on a joint while keeping each winner's
conditional method/round shape (D1's recalibration and D2's hybrid), and
read the marginals back off a joint.
"""
import numpy as np
import pytest

from mma.evaluate import joint_cell_log_loss, joint_outcome_log_loss
from mma.joint import (
    cells_to_dict, compose_joint_cells, corner_cells, impose_winner_marginal,
    marginals_from_cells, swap_corners,
)
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES

M, R = METHOD_CLASSES, ROUND_CLASSES


def _marginals(n=40, seed=0):
    rng = np.random.default_rng(seed)
    winner = rng.uniform(0.05, 0.95, size=n)
    method = rng.dirichlet(np.ones(3), size=n)
    rounds = rng.dirichlet(np.ones(4), size=n)
    return winner, method, rounds


def test_corner_cells_partitions_the_layout():
    a, b = corner_cells(M, R)
    assert a == list(range(8)) + [16]
    assert b == list(range(8, 16)) + [17]
    assert sorted(a + b) == list(range(18))


def test_composed_cells_are_a_distribution():
    winner, method, rounds = _marginals()
    cells = compose_joint_cells(winner, method, rounds, M, R)
    assert cells.shape == (len(winner), 18)
    assert (cells >= 0).all()
    assert cells.sum(axis=1) == pytest.approx(np.ones(len(winner)))


def test_composed_cells_score_exactly_what_the_composed_path_scores():
    """The D3 control's whole point: routing independent marginals through
    the cell scorer must reproduce `joint_outcome_log_loss` to the last bit,
    so any difference the hybrid shows is the simulator's conditional
    structure and not the re-weighting arithmetic."""
    rng = np.random.default_rng(3)
    winner, method, rounds = _marginals(n=60, seed=2)
    y_w = rng.integers(0, 2, size=60).astype(float)
    y_m = np.array(rng.choice(M, size=60), dtype=object)
    y_r = np.array([None if m == "decision" else rng.choice(R) for m in y_m], dtype=object)
    composed = joint_outcome_log_loss(y_w, y_m, y_r, winner, method, rounds, M, R)
    cells = compose_joint_cells(winner, method, rounds, M, R)
    assert joint_cell_log_loss(y_w, y_m, y_r, cells, M, R) == pytest.approx(composed)


def test_marginals_round_trip_through_the_cells():
    winner, method, rounds = _marginals()
    out = marginals_from_cells(compose_joint_cells(winner, method, rounds, M, R), M, R)
    assert out["winner"] == pytest.approx(winner)
    assert out["method"] == pytest.approx(method)
    assert out["round"] == pytest.approx(rounds)


def test_imposing_a_winner_marginal_hits_it_exactly():
    winner, method, rounds = _marginals()
    cells = compose_joint_cells(winner, method, rounds, M, R)
    target = np.clip(winner + 0.1, 0.01, 0.99)
    out = impose_winner_marginal(cells, target, M, R)
    assert out.sum(axis=1) == pytest.approx(np.ones(len(target)))
    assert marginals_from_cells(out, M, R)["winner"] == pytest.approx(target)


def test_imposing_a_winner_marginal_preserves_each_corner_s_conditional_shape():
    """`P(method, round | winner)` is exactly what the simulator contributes,
    so re-weighting must scale a corner's block and nothing else."""
    rng = np.random.default_rng(7)
    cells = rng.dirichlet(np.ones(18), size=25)
    target = rng.uniform(0.05, 0.95, size=25)
    out = impose_winner_marginal(cells, target, M, R)
    a, b = corner_cells(M, R)
    for block in (a, b):
        before = cells[:, block] / cells[:, block].sum(axis=1, keepdims=True)
        after = out[:, block] / out[:, block].sum(axis=1, keepdims=True)
        assert after == pytest.approx(before)


def test_imposing_the_marginal_a_joint_already_has_changes_nothing():
    winner, method, rounds = _marginals()
    cells = compose_joint_cells(winner, method, rounds, M, R)
    out = impose_winner_marginal(cells, marginals_from_cells(cells, M, R)["winner"], M, R)
    assert out == pytest.approx(cells)


def test_imposing_survives_a_corner_with_no_mass():
    """A corner whose cells are all zero has no conditional shape of its own,
    so it borrows the other corner's -- which is the only shape on hand that
    already respects what rounds the fight could reach. The target still has
    to be met exactly rather than dividing by zero."""
    cells = np.zeros((1, 18))
    cells[0, :4] = [0.5, 0.3, 0.2, 0.0]  # corner A: KO in rounds 1-3, no round 45
    out = impose_winner_marginal(cells, np.array([0.6]), M, R)
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)
    assert marginals_from_cells(out, M, R)["winner"] == pytest.approx([0.6])
    # B's block is A's shape, so the unreachable round-45 cells stay empty.
    assert out[0, [3, 7, 11, 15]].sum() == 0.0
    assert out[0, 8:16] == pytest.approx(0.4 * np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0]))


def test_swap_corners_exchanges_the_two_winner_blocks_and_is_an_involution():
    winner, method, rounds = _marginals()
    cells = compose_joint_cells(winner, method, rounds, M, R)
    swapped = swap_corners(cells, M, R)
    # the mirrored joint is the joint of the mirrored winner, same conditionals
    expected = compose_joint_cells(1.0 - winner, method, rounds, M, R)
    np.testing.assert_allclose(swapped, expected)
    np.testing.assert_allclose(swap_corners(swapped, M, R), cells)
    np.testing.assert_allclose(
        marginals_from_cells(swapped, M, R)["winner"], 1.0 - winner)
    # method and round describe the fight, not a corner, so they are unmoved
    for head in ("method", "round"):
        np.testing.assert_allclose(marginals_from_cells(swapped, M, R)[head],
                                   marginals_from_cells(cells, M, R)[head])


def test_cells_to_dict_is_the_same_distribution_in_nested_form():
    winner, method, rounds = _marginals(n=3)
    cells = compose_joint_cells(winner, method, rounds, M, R)
    record = cells_to_dict(cells[0], M, R)
    assert list(record) == ["a", "b"]
    total = 0.0
    for corner in ("a", "b"):
        assert set(record[corner]) == set(M)
        assert record[corner]["decision"] >= 0.0
        total += record[corner]["decision"]
        for finishing in M[:-1]:
            assert list(record[corner][finishing]) == list(R)
            total += sum(record[corner][finishing].values())
    assert total == pytest.approx(1.0)
    # cell for cell, in the layout's own order
    flat = [record["a"][m][r] for m in M[:-1] for r in R]
    flat += [record["b"][m][r] for m in M[:-1] for r in R]
    flat += [record["a"]["decision"], record["b"]["decision"]]
    np.testing.assert_allclose(flat, cells[0])


def test_cells_to_dict_rejects_a_batch():
    winner, method, rounds = _marginals(n=3)
    cells = compose_joint_cells(winner, method, rounds, M, R)
    with pytest.raises(ValueError, match="one row"):
        cells_to_dict(cells, M, R)
