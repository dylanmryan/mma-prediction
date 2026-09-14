"""SP5's auxiliary margin head, and the guarantee that it costs nothing when off.

The head is opt-in twice over: it is not CONSTRUCTED unless `margin_scale > 0`,
and `forward` does not return it unless asked. That is what lets the
pre-registration's `margin_scale = 0` arm be the incumbent itself rather than a
re-run of it, and it is why `mma.inference` -- the serving path -- needs no
change at all.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from mma.models.net import MultiTaskNet, multitask_loss
from mma.models.train_loop import DEFAULT_CONFIG, encode_targets

FEATURES, CLASSES, N = 7, 3, 16


def _net(margin_head=False, seed=0):
    torch.manual_seed(seed)
    return MultiTaskNet(n_features=FEATURES, n_weight_classes=CLASSES,
                        margin_head=margin_head)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, FEATURES, generator=g),
            torch.randint(0, CLASSES, (N,), generator=g))


# --- off means off ------------------------------------------------------------

def test_no_margin_head_is_constructed_by_default():
    assert _net().margin_head is None


def test_the_default_config_leaves_the_head_off():
    assert DEFAULT_CONFIG.get("margin_scale", 0.0) == 0.0


def test_a_net_without_the_head_has_the_incumbent_parameter_set():
    """Identical state_dict keys, so committed checkpoints still load."""
    keys = set(_net().state_dict())
    assert not any(k.startswith("margin_head") for k in keys)
    assert set(_net(margin_head=True).state_dict()) - keys == {
        "margin_head.weight", "margin_head.bias",
    }


def test_adding_the_head_does_not_perturb_the_other_parameters():
    """It is constructed last, so every earlier draw from the RNG is the same
    -- which is what makes `margin_scale = 0` the incumbent rather than a
    re-run that merely ought to match."""
    plain, with_head = _net(seed=3), _net(margin_head=True, seed=3)
    for key, value in plain.state_dict().items():
        assert torch.equal(value, with_head.state_dict()[key]), key


def test_forward_still_returns_three_heads_so_serving_is_untouched():
    net = _net(margin_head=True).eval()
    out = net(*_batch())
    assert len(out) == 3, "the serving path unpacks exactly three"
    winner, method, rounds = out
    assert winner.shape == (N,) and method.shape == (N, 3) and rounds.shape == (N, 4)


def test_the_margin_is_returned_only_when_asked():
    net = _net(margin_head=True).eval()
    out = net(*_batch(), return_margin=True)
    assert len(out) == 4
    assert out[3].shape == (N,)


def test_asking_a_headless_net_for_a_margin_yields_none_not_a_crash():
    net = _net().eval()
    assert net(*_batch(), return_margin=True)[3] is None


# --- the loss term ------------------------------------------------------------

def _loss_args(y_margin=None, margin_logits=None, margin_scale=0.0):
    g = torch.Generator().manual_seed(1)
    return dict(
        winner_logits=torch.randn(N, generator=g),
        method_logits=torch.randn(N, 3, generator=g),
        round_logits=torch.randn(N, 4, generator=g),
        y_winner=torch.randint(0, 2, (N,), generator=g).float(),
        y_method=torch.randint(-1, 3, (N,), generator=g),
        y_round=torch.randint(-1, 4, (N,), generator=g),
        three_round=torch.zeros(N, dtype=torch.bool),
        method_weights=torch.ones(3),
        round_weights=torch.ones(4),
        margin_logits=margin_logits,
        y_margin=y_margin,
        margin_scale=margin_scale,
    )


def test_scale_zero_reproduces_the_incumbent_loss_exactly():
    base = _loss_args()
    del base["margin_logits"], base["y_margin"], base["margin_scale"]
    incumbent = multitask_loss(**base)
    with_arg = multitask_loss(**_loss_args(
        y_margin=torch.randn(N), margin_logits=torch.randn(N), margin_scale=0.0))
    assert torch.equal(incumbent, with_arg)


def test_an_all_unlabelled_batch_contributes_nothing():
    """55% of fights are finishes and carry no card. A batch of them must not
    move the loss, or the head would be training on its own mask."""
    nan = torch.full((N,), float("nan"))
    args = _loss_args(y_margin=nan, margin_logits=torch.randn(N), margin_scale=1.0)
    off = dict(args, margin_scale=0.0)
    assert torch.allclose(multitask_loss(**args), multitask_loss(**off))


def test_a_labelled_batch_does_move_the_loss():
    y = torch.full((N,), 2.0)
    args = _loss_args(y_margin=y, margin_logits=torch.zeros(N), margin_scale=1.0)
    assert multitask_loss(**args) > multitask_loss(**dict(args, margin_scale=0.0))


def test_the_margin_term_is_masked_not_imputed():
    """Half-labelled must equal the same batch with the unlabelled rows gone,
    up to the mean over labelled rows -- i.e. NaN is skipped, not read as 0."""
    y = torch.full((N,), float("nan"))
    y[:4] = 1.5
    logits = torch.zeros(N)
    partial = multitask_loss(**_loss_args(y_margin=y, margin_logits=logits,
                                          margin_scale=1.0))
    all_same = torch.full((N,), 1.5)
    full = multitask_loss(**_loss_args(y_margin=all_same, margin_logits=logits,
                                       margin_scale=1.0))
    assert torch.allclose(partial, full), (
        "the labelled rows all carry the same target, so masking correctly "
        "must give the same mean as a fully-labelled batch of that target"
    )


def test_nan_targets_produce_no_nan_gradient():
    """The failure mode that silently destroys a run."""
    net = _net(margin_head=True)
    x, wc = _batch()
    winner, method, rounds, margin = net(x, wc, return_margin=True)
    y = torch.full((N,), float("nan"))
    y[:3] = 1.0
    loss = multitask_loss(
        winner, method, rounds,
        torch.randint(0, 2, (N,)).float(), torch.full((N,), -1),
        torch.full((N,), -1), torch.zeros(N, dtype=torch.bool),
        torch.ones(3), torch.ones(4),
        margin_logits=margin, y_margin=y, margin_scale=1.0,
    )
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), name


# --- the targets --------------------------------------------------------------

def test_encode_targets_carries_the_margin_with_its_nans_intact():
    features = pd.DataFrame({
        "y_winner": [1.0, 0.0, 1.0],
        "y_method": ["decision", "ko_tko", "decision"],
        "y_finish_round": [None, "1", None],
        "scheduled_rounds": [3, 3, 5],
        "y_margin": [2.0, float("nan"), -0.5],
    })
    targets = encode_targets(features)
    assert "y_margin" in targets
    got = targets["y_margin"]
    assert torch.isnan(got[1])
    assert got[0].item() == pytest.approx(2.0)
    assert got[2].item() == pytest.approx(-0.5)


def test_encode_targets_tolerates_a_table_without_the_column():
    """A feature table built before SP5 still trains."""
    features = pd.DataFrame({
        "y_winner": [1.0], "y_method": ["decision"],
        "y_finish_round": [None], "scheduled_rounds": [3],
    })
    got = encode_targets(features)["y_margin"]
    assert torch.isnan(got).all()
