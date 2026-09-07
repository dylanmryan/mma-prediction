"""Deterministic seeded training, temperature scaling, ensemble/MC-dropout inference.

``train_one`` early-stops on inner-validation winner log-loss by default. For the
walk-forward harness it also takes an optional ``config`` dict (architecture,
optimizer, loss scales), per-row ``sample_weight`` (recency weights), and a
``fixed_epochs`` budget mode that trains exactly N epochs with no validation set.
All defaults reproduce the original training path bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import torch

from mma.evaluate import log_loss
from mma.models.net import MultiTaskNet, multitask_loss

METHOD_CLASSES = ["ko_tko", "submission", "decision"]
ROUND_CLASSES = ["1", "2", "3", "45"]


def encode_targets(features) -> dict[str, torch.Tensor]:
    method_index = {label: i for i, label in enumerate(METHOD_CLASSES)}
    round_index = {label: i for i, label in enumerate(ROUND_CLASSES)}
    return {
        "y_winner": torch.tensor(features["y_winner"].to_numpy(dtype=np.float32)),
        "y_method": torch.tensor(
            features["y_method"].map(method_index).fillna(-1).to_numpy(dtype=np.int64)
        ),
        "y_round": torch.tensor(
            features["y_finish_round"].map(round_index).fillna(-1).to_numpy(dtype=np.int64)
        ),
        "three_round": torch.tensor(
            (features["scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        ),
    }


def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    known = y[y >= 0]
    counts = torch.bincount(known, minlength=n_classes).float().clamp(min=1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights


DEFAULT_CONFIG = {
    "hidden": (128, 64),
    "dropout": 0.3,
    "embedding_dim": 4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "method_scale": 0.5,
    "round_scale": 0.25,
}


def resolve_config(config: dict | None) -> dict:
    """DEFAULT_CONFIG overlaid with ``config``; unknown keys raise ValueError.

    ``hidden`` is coerced to a tuple so a config read back from JSON (a list)
    compares equal to DEFAULT_CONFIG."""
    config = dict(config or {})
    unknown = sorted(set(config) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(
            f"unknown train_one config keys: {unknown}; "
            f"allowed: {sorted(DEFAULT_CONFIG)}"
        )
    resolved = {**DEFAULT_CONFIG, **config}
    resolved["hidden"] = tuple(resolved["hidden"])
    return resolved


def train_one(seed, x_train, wc_train, targets_train, x_val, wc_val, targets_val,
              max_epochs: int = 200, patience: int = 20, batch_size: int = 256,
              n_weight_classes: int | None = None, config: dict | None = None,
              sample_weight=None, fixed_epochs: int | None = None):
    """Train one seeded MultiTaskNet; returns ``(net, info)``.

    Default mode: early-stop on inner-val winner log-loss, restore the best
    state; ``info`` has ``best_epoch``, ``best_val_log_loss`` and ``epochs_run``
    (epochs actually executed, including those that triggered the stop).

    ``fixed_epochs``: train exactly that many epochs with no validation pass and
    return the final weights; ``x_val``/``wc_val``/``targets_val`` are never
    touched (pass None). ``info`` is then ``{"best_epoch": fixed_epochs - 1,
    "best_val_log_loss": None, "epochs_run": fixed_epochs}``.

    ``config`` keys (all optional, see DEFAULT_CONFIG): hidden, dropout,
    embedding_dim, lr, weight_decay, method_scale, round_scale.
    ``sample_weight``: per-row training weights aligned with ``x_train``.
    """
    cfg = resolve_config(config)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if n_weight_classes is None:  # tests; production passes Preprocessor's count
        wc_max = wc_train.max()
        if fixed_epochs is None:
            wc_max = max(wc_max, wc_val.max())
        n_weight_classes = int(wc_max) + 1
    net = MultiTaskNet(
        n_features=x_train.shape[1], n_weight_classes=n_weight_classes,
        embedding_dim=cfg["embedding_dim"], hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    method_weights = class_weights(targets_train["y_method"], 3)
    round_weights = class_weights(targets_train["y_round"], 4)

    x_train_t = torch.tensor(x_train)
    wc_train_t = torch.tensor(wc_train)
    weight_t = None
    if sample_weight is not None:
        weight_t = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32))
        if weight_t.shape != (len(x_train_t),):
            raise ValueError(
                f"sample_weight has shape {tuple(weight_t.shape)}, "
                f"expected ({len(x_train_t)},)"
            )
    if fixed_epochs is None:
        x_val_t = torch.tensor(x_val)
        wc_val_t = torch.tensor(wc_val)
        n_epochs = max_epochs
    else:
        n_epochs = fixed_epochs

    best_loss, best_state, best_epoch, since_best = float("inf"), None, -1, 0
    epochs_run = 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(n_epochs):
        net.train()
        order = torch.randperm(len(x_train_t), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            winner, method, rounds = net(x_train_t[batch], wc_train_t[batch])
            loss = multitask_loss(
                winner, method, rounds,
                targets_train["y_winner"][batch], targets_train["y_method"][batch],
                targets_train["y_round"][batch], targets_train["three_round"][batch],
                method_weights, round_weights,
                method_scale=cfg["method_scale"], round_scale=cfg["round_scale"],
                sample_weight=None if weight_t is None else weight_t[batch],
            )
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        if fixed_epochs is not None:
            continue
        net.eval()
        with torch.no_grad():
            winner_logits, _, _ = net(x_val_t, wc_val_t)
            val_loss = log_loss(
                targets_val["y_winner"].numpy(),
                torch.sigmoid(winner_logits).numpy(),
            )
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, since_best = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    if fixed_epochs is not None:
        net.eval()
        return net, {
            "best_epoch": fixed_epochs - 1, "best_val_log_loss": None,
            "epochs_run": epochs_run,
        }
    net.load_state_dict(best_state)
    net.eval()
    return net, {
        "best_epoch": best_epoch, "best_val_log_loss": round(best_loss, 4),
        "epochs_run": epochs_run,
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.arange(0.5, 3.01, 0.01)
    losses = [
        log_loss(y, 1 / (1 + np.exp(-(logits / t)))) for t in grid
    ]
    return float(grid[int(np.argmin(losses))])


@torch.no_grad()
def predict(net, x, wc, temperature: float = 1.0):
    net.eval()
    winner_logits, method_logits, round_logits = net(
        torch.tensor(x), torch.tensor(wc)
    )
    return {
        "winner": torch.sigmoid(winner_logits / temperature).numpy(),
        "winner_logits": winner_logits.numpy(),
        "method": torch.softmax(method_logits, dim=1).numpy(),
        "round_logits": round_logits.numpy(),
    }


@torch.no_grad()
def mc_dropout_winner(net, x, wc, passes: int = 100, seed: int = 0,
                      temperature: float = 1.0) -> np.ndarray:
    """(passes, n) matrix of stochastic winner probabilities (dropout on)."""
    net.train()  # enables dropout; batchnorm uses batch stats -- acceptable for MC
    torch.manual_seed(seed)
    x_t, wc_t = torch.tensor(x), torch.tensor(wc)
    samples = []
    for _ in range(passes):
        winner_logits, _, _ = net(x_t, wc_t)
        samples.append(torch.sigmoid(winner_logits / temperature).numpy())
    net.eval()
    return np.stack(samples)
