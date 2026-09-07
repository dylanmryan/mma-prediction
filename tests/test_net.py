import numpy as np
import pytest
import torch

from mma.models.net import MultiTaskNet, multitask_loss

BATCH, FEATURES, CLASSES = 8, 12, 5


def _net():
    torch.manual_seed(0)
    return MultiTaskNet(n_features=FEATURES, n_weight_classes=CLASSES)


def test_forward_shapes():
    net = _net()
    x = torch.randn(BATCH, FEATURES)
    wc = torch.randint(0, CLASSES, (BATCH,))
    winner, method, rounds = net(x, wc)
    assert winner.shape == (BATCH,)
    assert method.shape == (BATCH, 3)
    assert rounds.shape == (BATCH, 4)


def test_loss_masks_unknown_method_and_nonfinish():
    net = _net()
    x = torch.randn(4, FEATURES)
    wc = torch.zeros(4, dtype=torch.long)
    winner, method, rounds = net(x, wc)
    y_winner = torch.tensor([1.0, 0.0, 1.0, 0.0])
    y_method = torch.tensor([0, -1, 2, 1])     # -1 = unknown -> masked
    y_round = torch.tensor([1, -1, -1, 3])     # -1 = non-finish -> masked
    three_round = torch.tensor([True, False, True, False])
    loss = multitask_loss(
        winner, method, rounds, y_winner, y_method, y_round, three_round,
        method_weights=torch.ones(3), round_weights=torch.ones(4),
    )
    assert torch.isfinite(loss)


def test_round_45_masked_for_three_round_fights():
    net = _net()
    net.eval()
    x = torch.randn(2, FEATURES)
    wc = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        _, _, rounds = net(x, wc)
    probs = MultiTaskNet.round_probs(rounds, torch.tensor([True, False]))
    assert probs[0, 3].item() == pytest.approx(0.0, abs=1e-6)   # 3-round fight: no R4-5
    assert probs.sum(dim=1).allclose(torch.ones(2))


def test_dropout_active_only_in_train_mode():
    net = _net()
    x = torch.randn(64, FEATURES)
    wc = torch.zeros(64, dtype=torch.long)
    net.eval()
    a, _, _ = net(x, wc)
    b, _, _ = net(x, wc)
    assert torch.equal(a, b)
    net.train()
    torch.manual_seed(1)
    c, _, _ = net(x, wc)
    torch.manual_seed(2)
    d, _, _ = net(x, wc)
    assert not torch.equal(c, d)


def test_multitask_loss_sample_weight_reweights_rows():
    n = 4
    logits = torch.zeros(n)
    m = torch.zeros(n, 3)
    r = torch.zeros(n, 4)
    y_w = torch.tensor([1.0, 0.0, 1.0, 0.0])
    y_m = torch.full((n,), -1)
    y_r = torch.full((n,), -1)
    three = torch.ones(n, dtype=torch.bool)
    base = multitask_loss(logits, m, r, y_w, y_m, y_r, three, torch.ones(3), torch.ones(4))
    weighted = multitask_loss(logits, m, r, y_w, y_m, y_r, three, torch.ones(3), torch.ones(4),
                              sample_weight=torch.tensor([2.0, 0.0, 2.0, 0.0]))
    # winner BCE at logit 0 is log 2 for every row, so any weighting gives the same mean
    assert torch.isclose(base, weighted)
    skewed = multitask_loss(torch.tensor([3.0, 3.0, 3.0, 3.0]), m, r, y_w, y_m, y_r, three,
                            torch.ones(3), torch.ones(4),
                            sample_weight=torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert skewed < multitask_loss(torch.tensor([3.0] * 4), m, r, y_w, y_m, y_r, three,
                                   torch.ones(3), torch.ones(4))


def test_multitask_loss_weights_apply_to_method_and_round_terms():
    n = 4
    y_w = torch.tensor([1.0, 0.0, 1.0, 0.0])
    y_m = torch.tensor([0, 1, 2, 0])
    y_r = torch.tensor([0, 1, 2, 3])
    three = torch.zeros(n, dtype=torch.bool)
    good = torch.zeros(n, 3); good[torch.arange(n), y_m] = 5.0   # confident-correct method logits
    bad = torch.zeros(n, 3); bad[torch.arange(n), (y_m + 1) % 3] = 5.0
    rounds = torch.zeros(n, 4)
    winner = torch.zeros(n)
    # rows 0,2 have correct method logits and weight 1; rows 1,3 wrong with weight 0
    mixed = torch.stack([good[0], bad[1], good[2], bad[3]])
    w = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_w = multitask_loss(winner, mixed, rounds, y_w, y_m, y_r, three, torch.ones(3), torch.ones(4), sample_weight=w)
    loss_all_good = multitask_loss(winner, good, rounds, y_w, y_m, y_r, three, torch.ones(3), torch.ones(4))
    assert torch.isclose(loss_w, loss_all_good, atol=1e-5)
