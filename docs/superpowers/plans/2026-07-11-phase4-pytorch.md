# Phase 4: PyTorch Multi-Task Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A multi-task neural net (winner + method + finish-round heads) with a 5-seed ensemble, MC-dropout uncertainty, and temperature-scaled calibration — compared honestly against XGBoost (val log-loss 0.6584 to beat; matching it is an acceptable, reportable finding).

**Architecture:** `src/mma/tensors.py` (feature selection, train-fitted imputation/standardization, weight-class indexing) → `src/mma/models/net.py` (embedding + shared trunk + three heads, masked multi-task loss) → `src/mma/models/train_loop.py` (seeded deterministic training, early stopping, temperature scaling, MC dropout & ensemble inference) → `scripts/train_torch.py` (5-seed ensemble, metrics, artifacts).

**Tech Stack:** torch≥2.2 (CPU — data is 8k×~40; deterministic and fast), pandas, pytest.

**Branch:** `phase-4-pytorch`. **Machine note:** first import of a newly installed package can take minutes (cold cache) — warm imports in background; never kill runs before 10 minutes.

**Design decisions (locked):**
- **Feature set = Phase 3 table minus the four `*_missing` flags** (final review ablation: they're era proxies that *hurt* validation — acc 0.601→0.613 without them). Documented deviation from the spec's "missingness flags" line, justified by measurement. `swapped` and identifiers stay excluded.
- **Imputation/standardization fit on train rows only** (train < 2021-01-01), stored as JSON artifacts: numeric NaN → train median; then standardize (train mean/std). `weight_class` → embedding index (unknown/NaN → 0).
- **Heads & masking:** winner = 1 sigmoid logit (BCE, all rows); method = 3 softmax (CE, rows with known method, inverse-frequency class weights); finish round = 4 softmax over `1/2/3/45` (CE, finishes only, class weights; the `45` logit is masked to −1e9 whenever `scheduled_rounds == 3`, in both loss and inference).
- **Loss = BCE + 0.5·CE_method + 0.25·CE_round** (fixed weights v1; winner is the primary task).
- **Training:** AdamW(lr=1e-3, weight_decay=1e-4), batch 256, ≤200 epochs, early stop patience 20 on validation winner log-loss, restore best. Full determinism: `torch.manual_seed(seed)`, `torch.use_deterministic_algorithms(True)`, CPU.
- **Ensemble = seeds 0–4**; headline probability = mean of the 5 calibrated winner probabilities; ensemble spread = max−min. **MC dropout** = 100 stochastic forward passes of seed-0 model (dropout forced on) for the app's uncertainty visual.
- **Calibration:** per-seed temperature scaling on validation winner logits (scalar T via grid search 0.5–3.0 step 0.01 minimizing val log-loss — simple, deterministic, no LBFGS fragility). Report pre/post log-loss and 10-bin ECE.
- **2024+ is never read** for training, tuning, calibration, or reporting.
- Artifacts committed: `models/torch/net_seed{0..4}.pt`, `models/torch/preprocess.json`, `models/torch/metrics_val.json`.

---

### Task 1: Tensor preparation (`tensors.py`)

**Files:** Create `src/mma/tensors.py`, `tests/test_tensors.py`. Add `"torch>=2.2",` to pyproject dependencies; `.venv/bin/pip install -e ".[dev]"` (background; big wheel).

- [ ] **Step 1: Tests** — `tests/test_tensors.py`:

```python
import numpy as np
import pandas as pd
import pytest

from mma.tensors import DROPPED, Preprocessor


def _features():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2", "f3"],
            "date": pd.to_datetime(["2019-01-01", "2020-01-01", "2022-01-01"]),
            "swapped": [False, True, False],
            "y_winner": [1, 0, 1],
            "y_method": ["ko_tko", None, "decision"],
            "y_finish_round": ["2", None, None],
            "weight_class": pd.array(["Lightweight", None, "Heavyweight"], dtype="string"),
            "title_fight": [False, True, False],
            "scheduled_rounds": pd.array([3, 5, 3], dtype="Int64"),
            "elo_diff": [50.0, None, -20.0],
            "reach_diff": [5.0, 2.0, None],
            "reach_missing_a": [False, False, True],
            "reach_missing_b": [False, False, False],
            "dob_missing_a": [False, False, False],
            "dob_missing_b": [False, False, False],
        }
    )


def test_missing_flags_dropped_and_ids_excluded():
    prep = Preprocessor.fit(_features(), train_mask=np.array([True, True, False]))
    assert set(DROPPED) & set(prep.numeric_columns) == set()
    for column in ("fight_id", "date", "swapped", "y_winner", "weight_class"):
        assert column not in prep.numeric_columns


def test_impute_and_standardize_fit_on_train_only():
    features = _features()
    prep = Preprocessor.fit(features, train_mask=np.array([True, True, False]))
    x, wc = prep.transform(features)
    elo = features["elo_diff"]
    # train rows: [50, NaN] -> median 50, mean after impute 50, std 0 -> guarded to 1
    column = prep.numeric_columns.index("elo_diff")
    assert x[0, column] == pytest.approx(0.0)   # (50-50)/1
    assert x[1, column] == pytest.approx(0.0)   # imputed to train median 50
    assert x[2, column] == pytest.approx(-70.0) # (-20-50)/1 -- unseen val row


def test_weight_class_indexing_unknown_to_zero():
    features = _features()
    prep = Preprocessor.fit(features, train_mask=np.array([True, True, False]))
    _, wc = prep.transform(features)
    assert wc[1] == 0                      # NaN -> unknown bucket
    assert wc[0] != wc[2] and wc[0] > 0    # two known classes, distinct


def test_round_trip_json(tmp_path):
    features = _features()
    prep = Preprocessor.fit(features, train_mask=np.array([True, True, False]))
    path = tmp_path / "preprocess.json"
    prep.save(path)
    loaded = Preprocessor.load(path)
    x1, wc1 = prep.transform(features)
    x2, wc2 = loaded.transform(features)
    assert np.allclose(x1, x2) and (wc1 == wc2).all()
```

- [ ] **Step 2:** Run — FAILS. 
- [ ] **Step 3: Implement `src/mma/tensors.py`:**

```python
"""Feature-table -> tensor preparation, fit on training rows only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("y_winner", "y_method", "y_finish_round")
IDENTIFIERS = ("fight_id", "date", "swapped")
# Era-proxy flags dropped per Phase 3 final-review ablation (hurt validation).
DROPPED = ("reach_missing_a", "reach_missing_b", "dob_missing_a", "dob_missing_b")
CATEGORICAL = "weight_class"


class Preprocessor:
    def __init__(self, numeric_columns, medians, means, stds, weight_classes):
        self.numeric_columns = list(numeric_columns)
        self.medians = dict(medians)
        self.means = dict(means)
        self.stds = dict(stds)
        self.weight_classes = list(weight_classes)  # index 0 reserved for unknown

    @classmethod
    def fit(cls, features: pd.DataFrame, train_mask: np.ndarray) -> "Preprocessor":
        excluded = set(TARGETS) | set(IDENTIFIERS) | set(DROPPED) | {CATEGORICAL}
        numeric_columns = [
            column for column in features.columns if column not in excluded
        ]
        train = features.loc[train_mask, numeric_columns].astype(float)
        medians = train.median().fillna(0.0).to_dict()
        imputed = train.fillna(medians)
        means = imputed.mean().to_dict()
        stds = {
            column: (value if value and not np.isnan(value) else 1.0)
            for column, value in imputed.std(ddof=0).to_dict().items()
        }
        classes = sorted(
            features.loc[train_mask, CATEGORICAL].dropna().unique().tolist()
        )
        return cls(numeric_columns, medians, means, stds, classes)

    def transform(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = features[self.numeric_columns].astype(float)
        numeric = numeric.fillna(self.medians)
        x = np.stack(
            [
                (numeric[column].to_numpy() - self.means[column]) / self.stds[column]
                for column in self.numeric_columns
            ],
            axis=1,
        ).astype(np.float32)
        index = {name: i + 1 for i, name in enumerate(self.weight_classes)}
        wc = (
            features[CATEGORICAL]
            .map(lambda v: index.get(v, 0) if pd.notna(v) else 0)
            .to_numpy(dtype=np.int64)
        )
        return x, wc

    @property
    def n_weight_classes(self) -> int:
        return len(self.weight_classes) + 1

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "numeric_columns": self.numeric_columns,
                    "medians": self.medians,
                    "means": self.means,
                    "stds": self.stds,
                    "weight_classes": self.weight_classes,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path) -> "Preprocessor":
        payload = json.loads(Path(path).read_text())
        return cls(
            payload["numeric_columns"], payload["medians"], payload["means"],
            payload["stds"], payload["weight_classes"],
        )
```

- [ ] **Step 4:** Task tests + full suite (94 expected — 90 + 4). Note: torch not needed yet for these tests; the install can proceed in parallel.
- [ ] **Step 5:** Commit: `"Add tensor preprocessor fit on training rows"`

---

### Task 2: The network + masked multi-task loss (`net.py`)

**Files:** Create `src/mma/models/net.py`, `tests/test_net.py`.

- [ ] **Step 1: Tests** — `tests/test_net.py`:

```python
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
    _, _, rounds = net(x, wc)
    probs = MultiTaskNet.round_probs(rounds, torch.tensor([True, False]))
    assert probs[0, 3] == pytest.approx(0.0, abs=1e-6)   # 3-round fight: no R4-5
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
```

- [ ] **Step 2:** Run — FAILS.
- [ ] **Step 3: Implement `src/mma/models/net.py`:**

```python
"""Multi-task net: shared trunk, winner/method/finish-round heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MASK_VALUE = -1e9


class MultiTaskNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_weight_classes: int,
        embedding_dim: int = 4,
        hidden: tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.weight_class_embedding = nn.Embedding(n_weight_classes, embedding_dim)
        layers: list[nn.Module] = []
        width = n_features + embedding_dim
        for size in hidden:
            layers += [
                nn.Linear(width, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.winner_head = nn.Linear(width, 1)
        self.method_head = nn.Linear(width, 3)
        self.round_head = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor, weight_class: torch.Tensor):
        combined = torch.cat([x, self.weight_class_embedding(weight_class)], dim=1)
        hidden = self.trunk(combined)
        return (
            self.winner_head(hidden).squeeze(-1),
            self.method_head(hidden),
            self.round_head(hidden),
        )

    @staticmethod
    def round_probs(round_logits: torch.Tensor, three_round: torch.Tensor):
        logits = round_logits.clone()
        logits[three_round, 3] = _MASK_VALUE
        return F.softmax(logits, dim=1)


def multitask_loss(
    winner_logits, method_logits, round_logits,
    y_winner, y_method, y_round, three_round,
    method_weights, round_weights,
    method_scale: float = 0.5, round_scale: float = 0.25,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(winner_logits, y_winner)
    method_known = y_method >= 0
    if method_known.any():
        loss = loss + method_scale * F.cross_entropy(
            method_logits[method_known], y_method[method_known],
            weight=method_weights,
        )
    round_known = y_round >= 0
    if round_known.any():
        logits = round_logits.clone()
        logits[three_round, 3] = _MASK_VALUE
        loss = loss + round_scale * F.cross_entropy(
            logits[round_known], y_round[round_known], weight=round_weights,
        )
    return loss
```

- [ ] **Step 4:** Task tests + full suite (98 expected).
- [ ] **Step 5:** Commit: `"Add multi-task network and masked loss"`

---

### Task 3: Training loop + calibration + inference (`train_loop.py`)

**Files:** Create `src/mma/models/train_loop.py`, `tests/test_train_loop.py`.

- [ ] **Step 1: Implement `src/mma/models/train_loop.py`:**

```python
"""Deterministic seeded training, temperature scaling, ensemble/MC-dropout inference."""
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


def train_one(seed, x_train, wc_train, targets_train, x_val, wc_val, targets_val,
              max_epochs: int = 200, patience: int = 20, batch_size: int = 256,
              n_weight_classes: int | None = None):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if n_weight_classes is None:  # tests; production passes Preprocessor's count
        n_weight_classes = int(max(wc_train.max(), wc_val.max())) + 1
    net = MultiTaskNet(n_features=x_train.shape[1], n_weight_classes=n_weight_classes)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    method_weights = class_weights(targets_train["y_method"], 3)
    round_weights = class_weights(targets_train["y_round"], 4)

    x_train_t = torch.tensor(x_train)
    wc_train_t = torch.tensor(wc_train)
    x_val_t = torch.tensor(x_val)
    wc_val_t = torch.tensor(wc_val)

    best_loss, best_state, best_epoch, since_best = float("inf"), None, -1, 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
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
            )
            loss.backward()
            optimizer.step()
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
    net.load_state_dict(best_state)
    net.eval()
    return net, {"best_epoch": best_epoch, "best_val_log_loss": round(best_loss, 4)}


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
```

- [ ] **Step 2: Tests** — `tests/test_train_loop.py` (fast: tiny synthetic problem):

```python
import numpy as np
import pandas as pd
import pytest
import torch

from mma.models.train_loop import (
    class_weights, encode_targets, fit_temperature, mc_dropout_winner, train_one,
)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 6)).astype(np.float32)
    y = (x[:, 0] + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    features = pd.DataFrame(
        {
            "y_winner": y,
            "y_method": pd.array(["ko_tko" if v > 0 else "decision" for v in x[:, 1]], dtype="string"),
            "y_finish_round": pd.array(["1" if v > 0 else None for v in x[:, 2]], dtype="string"),
            "scheduled_rounds": pd.array([3] * n, dtype="Int64"),
        }
    )
    wc = rng.integers(0, 3, n)
    return x, wc.astype(np.int64), features


def test_encode_targets_masks():
    _, _, features = _toy()
    targets = encode_targets(features)
    assert set(targets["y_method"].unique().tolist()) <= {-1, 0, 1, 2}
    assert (targets["y_round"][features["y_finish_round"].isna().to_numpy()] == -1).all()
    assert targets["three_round"].all()


def test_class_weights_inverse_frequency():
    weights = class_weights(torch.tensor([0, 0, 0, 1, -1]), 2)
    assert weights[1] > weights[0]


def test_training_learns_and_is_seed_deterministic():
    x, wc, features = _toy()
    targets = encode_targets(features)
    split = 200
    def sliced(t, sl):
        return {k: v[sl] for k, v in t.items()}
    net1, info1 = train_one(0, x[:split], wc[:split], sliced(targets, slice(None, split)),
                            x[split:], wc[split:], sliced(targets, slice(split, None)),
                            max_epochs=30, patience=10)
    net2, info2 = train_one(0, x[:split], wc[:split], sliced(targets, slice(None, split)),
                            x[split:], wc[split:], sliced(targets, slice(split, None)),
                            max_epochs=30, patience=10)
    assert info1 == info2
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.equal(p1, p2)
    assert info1["best_val_log_loss"] < 0.65  # learned the signal


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    clean_logits = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1, 4000)
    overconfident = clean_logits * 3
    t = fit_temperature(overconfident, y)
    assert t > 1.5  # must cool the logits substantially


def test_mc_dropout_produces_spread():
    x, wc, features = _toy(n=64)
    targets = encode_targets(features)
    net, _ = train_one(0, x, wc, targets, x, wc, targets, max_epochs=3, patience=5)
    samples = mc_dropout_winner(net, x[:8], wc[:8], passes=20, seed=0)
    assert samples.shape == (20, 8)
    assert samples.std(axis=0).mean() > 0.0
```

- [ ] **Step 3:** Run tests (order: implement first here since tests import everything; the RED step is the initial ImportError run). Full suite: 103 expected.
- [ ] **Step 4:** Commit: `"Add deterministic training loop, calibration, MC dropout"`

---

### Task 4: Ensemble training script

**Files:** Create `scripts/train_torch.py`, `tests/test_processed_torch.py`.

- [ ] **Step 1: `scripts/train_torch.py`:**

```python
"""Train the 5-seed ensemble; report validation-years metrics only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mma.evaluate import accuracy, brier_score, log_loss, macro_f1
from mma.models.net import MultiTaskNet
from mma.models.train_loop import (
    METHOD_CLASSES, ROUND_CLASSES, encode_targets, fit_temperature, predict, train_one,
)
from mma.tensors import Preprocessor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "models" / "torch"
TRAIN_END = "2021-01-01"
VAL_START, VAL_END = "2021-01-01", "2023-12-31"
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    train = (features["date"] < TRAIN_END).to_numpy()
    val = (
        (features["date"] >= VAL_START) & (features["date"] <= VAL_END)
    ).to_numpy()

    prep = Preprocessor.fit(features, train_mask=train)
    x, wc = prep.transform(features)
    targets = encode_targets(features)

    def sliced(mask):
        return {key: value[torch.tensor(mask)] for key, value in targets.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    prep.save(OUT / "preprocess.json")

    y_val = targets["y_winner"][torch.tensor(val)].numpy()
    per_seed, winner_probs = [], []
    method_prob_sum = None
    for seed in SEEDS:
        net, info = train_one(
            seed, x[train], wc[train], sliced(train), x[val], wc[val], sliced(val),
            n_weight_classes=prep.n_weight_classes,
        )
        raw = predict(net, x[val], wc[val])
        temperature = fit_temperature(raw["winner_logits"], y_val)
        calibrated = predict(net, x[val], wc[val], temperature=temperature)
        per_seed.append(
            {
                "seed": seed,
                **info,
                "temperature": temperature,
                "val_log_loss_calibrated": round(
                    log_loss(y_val, calibrated["winner"]), 4
                ),
            }
        )
        winner_probs.append(calibrated["winner"])
        method_prob_sum = (
            calibrated["method"]
            if method_prob_sum is None
            else method_prob_sum + calibrated["method"]
        )
        torch.save(
            {"state_dict": net.state_dict(), "temperature": temperature,
             "n_features": x.shape[1], "n_weight_classes": prep.n_weight_classes},
            OUT / f"net_seed{seed}.pt",
        )

    ensemble = np.mean(winner_probs, axis=0)
    spread = np.max(winner_probs, axis=0) - np.min(winner_probs, axis=0)
    metrics = {
        "winner_ensemble": {
            "n_val": int(val.sum()),
            "accuracy": round(accuracy(y_val, ensemble), 4),
            "log_loss": round(log_loss(y_val, ensemble), 4),
            "brier": round(brier_score(y_val, ensemble), 4),
            "mean_seed_spread": round(float(spread.mean()), 4),
        },
        "per_seed": per_seed,
    }

    method_known = (targets["y_method"][torch.tensor(val)] >= 0).numpy()
    method_pred = [
        METHOD_CLASSES[i] for i in method_prob_sum[method_known].argmax(axis=1)
    ]
    method_true = [
        METHOD_CLASSES[i]
        for i in targets["y_method"][torch.tensor(val)][method_known].tolist()
    ]
    metrics["method_ensemble"] = {
        "n_val": int(method_known.sum()),
        "accuracy": round(
            float(np.mean([p == t for p, t in zip(method_pred, method_true)])), 4
        ),
        "macro_f1": round(macro_f1(method_true, method_pred), 4),
    }

    (OUT / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `tests/test_processed_torch.py`:**

```python
import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "models" / "torch"

pytestmark = pytest.mark.skipif(
    not (OUT / "metrics_val.json").exists(),
    reason="torch ensemble not trained (run scripts/train_torch.py)",
)


def test_five_seed_artifacts_exist():
    for seed in range(5):
        assert (OUT / f"net_seed{seed}.pt").exists()
    assert (OUT / "preprocess.json").exists()


def test_metrics_sane_and_honest():
    metrics = json.loads((OUT / "metrics_val.json").read_text())
    winner = metrics["winner_ensemble"]
    assert 0.55 < winner["accuracy"] < 0.70   # >0.70 would smell like leakage
    assert winner["log_loss"] < 0.6777        # must at least beat Elo
    assert 0 < winner["mean_seed_spread"] < 0.5
    assert len(metrics["per_seed"]) == 5
    temperatures = [seed["temperature"] for seed in metrics["per_seed"]]
    assert all(0.5 <= t <= 3.0 for t in temperatures)
```

- [ ] **Step 3:** Run `.venv/bin/python scripts/train_torch.py` (background; 5 seeds × ≤200 epochs on 6.2k rows — expect minutes, not hours). Capture FULL output. Gates: ensemble winner log-loss < 0.6777 (Elo) required; vs XGBoost 0.6584 either way is reportable — beating it is a win, matching it is the expected honest outcome for tabular data (say so in the report). Accuracy > 0.70 → suspect leakage → STOP.
- [ ] **Step 4:** Full suite (105 expected). Commit code + `models/torch/`: `"Train 5-seed multi-task ensemble with calibration"`

---

### Task 5: README + wrap-up

- [ ] Add the neural net row to the winner table (ensemble accuracy/log-loss/brier + mean seed spread), method row comparison, and an honest paragraph: multi-task learning + uncertainty + calibration are the deliverables; state plainly whether the net beat, matched, or trailed XGBoost. Note the dropped missingness flags (with the ablation rationale). Commit: `"Report neural net results in README"`

---

## Done criteria (Phase 4)

- Suite green (~105 tests) including seed-determinism and artifact-integrity tests.
- 5 seed checkpoints + preprocess.json + metrics_val.json committed; training reproducible.
- Ensemble beats Elo on val log-loss; XGBoost comparison reported honestly either way.
- 2024+ never read. README updated.
