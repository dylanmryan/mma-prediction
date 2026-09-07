# SP1: Walk-Forward Evaluation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One harness that scores any candidate model on an expanding-window walk-forward over 2018–2026 (~3,900 fights), with a measured seed noise floor and pre-registered shipping bars; then use it to decide whether the deployed model should be refit on all data through the latest event, and ship that recipe if it holds.

**Architecture:** `src/mma/walkforward.py` builds folds, scores predictions (winner, joint outcome, slices), and applies the bars. `src/mma/candidates.py` wraps Elo, XGBoost, and the torch ensemble behind one `fit_predict` call with config, training-window, recency-weight, and fixed-budget options. `scripts/run_walkforward.py` turns a candidate spec into a JSON report under `models/walkforward/`; `scripts/noise_floor.py` derives σ_seed from three seed-set reports. The train scripts gain a `--refit-through` mode. Existing training code (`models/xgb.py`, `models/train_loop.py`, `models/net.py`) is extended with sample weights, a config dict, and a fixed-epoch mode, all backward compatible.

**Tech Stack:** Python 3.11, pandas, numpy, xgboost 3.x, torch 2.x (CPU), pytest. Local-disk venv `~/.venvs/mma` (see SP0 plan); every command below is `OMP_NUM_THREADS=1 ~/.venvs/mma/bin/{python,pytest}` from the repo root. Full suite ≈ 5 s; a 5-seed torch fit ≈ 8 s; a full torch walk-forward ≈ 1–2 min.

**Commit convention:** plain messages, no attribution trailers. `git status --short` before each commit; stage exactly the listed files.

---

## Evaluation protocol (locked; from the spec §4 SP1 and §5)

- **Fold years** Y ∈ {2018, …, 2025}. For fold Y: `train` = fights dated before Y−1-01-01 (optionally ≥ `train_start`); `inner_val` = year Y−1 (early stopping / temperature only); `eval` = year Y, except the last fold, whose `eval` is every fight dated ≥ 2025-01-01 (absorbs the partial 2026).
- **Rows**: decisive fights from `features.parquet` (already winner-labelled). Joint-outcome metrics use rows with known method; round metrics use finishes.
- **Metrics**: winner log-loss (primary), accuracy, Brier, ECE (10 equal-width bins); joint-outcome log-loss over composed probabilities P(w)·P(m)·P(r|finish) with round buckets {1, 2, 3, 45}; method macro-F1; finish-round macro-F1. Pooled = computed on the concatenation of all fold evaluation rows; per-fold reported alongside.
- **Slices** (pooled, reported for every candidate): `debut` (either fighter's first UFC fight), `womens`, `five_round`, and each fold year. An `external_missing` slice is added in SP2 when such a column exists.
- **Noise floor** σ_seed: standard deviation (ddof=1) of pooled winner log-loss over three runs of the incumbent torch config with disjoint seed sets {0–4}, {5–9}, {10–14}.
- **Bars** (mechanical): winner ships iff Δ pooled winner LL < −max(0.003, 2·σ_seed) and no fold year worsens by > 0.01; simulator joint bar in SP3. A shipped configuration is re-scored with fresh seeds; that number is reported.
- **Refit strategy** (decided in this plan): (A) incumbent protocol vs (B) fixed budget on train ∪ inner_val; (B) becomes the deployment recipe iff its pooled winner LL is not worse than (A) by more than σ_seed.

## File map

| Path | Change | Responsibility |
|---|---|---|
| `src/mma/evaluate.py` | extend | `expected_calibration_error`, `joint_outcome_log_loss` |
| `src/mma/walkforward.py` | create | `Fold`, `make_folds`, `recency_weights`, `score_fold`, `pool`, `slice_masks`, `build_report`, `bar_check` |
| `src/mma/candidates.py` | create | `EloCandidate`, `XGBCandidate`, `TorchCandidate` with one `fit_predict` signature |
| `src/mma/models/xgb.py` | extend | `params` override, `sample_weight`, fixed-round mode |
| `src/mma/models/net.py` | extend | per-sample weights in `multitask_loss`; `MultiTaskNet` accepts `hidden`/`dropout`/`embedding_dim` (already) |
| `src/mma/models/train_loop.py` | extend | `train_one(config=..., sample_weight=..., fixed_epochs=...)` |
| `scripts/run_walkforward.py` | create | CLI: candidate spec → `models/walkforward/<name>.json` |
| `scripts/noise_floor.py` | create | three reports → `models/walkforward/noise_floor.json` |
| `scripts/train_xgb.py`, `scripts/train_torch.py` | extend | `--refit-through DATE --budget N [--temperature T]` mode |
| `models/walkforward/*.json` | create | baseline reports, noise floor, refit experiment |
| `tests/test_evaluate.py` | extend | ECE, joint log-loss |
| `tests/test_walkforward.py` | create | folds, weights, slices, pooling, bar arithmetic |
| `tests/test_candidates.py` | create | protocol on a tiny synthetic table |
| `tests/test_train_loop.py`, `tests/test_xgb.py` | extend | weights, config, fixed budget |
| `README.md` | modify | evaluation section: walk-forward replaces the spent-test story |

---

### Task 1: Branch and baseline

- [ ] **Step 1: Branch from main**

```bash
git checkout main && git status --short && git checkout -b sp1-walkforward
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
```
Expected: clean tree; `281 passed, 1 skipped`.

---

### Task 2: Metrics — ECE and joint-outcome log-loss

**Files:** Modify `src/mma/evaluate.py`; Modify `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_evaluate.py`)

```python
import numpy as np
import pytest

from mma.evaluate import expected_calibration_error, joint_outcome_log_loss


def test_ece_zero_when_perfectly_calibrated_bins():
    # bin [0.2,0.3): mean pred 0.25, empirical 0.25 (1 of 4); bin [0.7,0.8): 0.75, 3 of 4
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_weights_bins_by_count():
    p = np.array([0.9] * 3 + [0.1] * 1)   # bin 9: pred .9, empirical 2/3; bin 1: pred .1, empirical 0
    y = np.array([1, 1, 0, 0])
    expected = 0.75 * abs(0.9 - 2 / 3) + 0.25 * abs(0.1 - 0.0)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(expected)


def test_joint_outcome_log_loss_composes_winner_method_round():
    method_classes = ["ko_tko", "submission", "decision"]
    round_classes = ["1", "2", "3", "45"]
    p_winner = np.array([0.8, 0.4])
    method = np.array([[0.5, 0.2, 0.3], [0.1, 0.1, 0.8]])
    rounds = np.array([[0.6, 0.2, 0.1, 0.1], [0.25] * 4])
    y_winner = np.array([1, 0])
    y_method = np.array(["ko_tko", "decision"], dtype=object)
    y_round = np.array(["1", None], dtype=object)
    # row 0: A wins by KO in R1 -> 0.8 * 0.5 * 0.6 ; row 1: B wins by decision -> 0.6 * 0.8
    expected = -np.mean([np.log(0.8 * 0.5 * 0.6), np.log(0.6 * 0.8)])
    got = joint_outcome_log_loss(
        y_winner, y_method, y_round, p_winner, method, rounds, method_classes, round_classes
    )
    assert got == pytest.approx(expected)


def test_joint_outcome_log_loss_skips_unknown_method():
    got = joint_outcome_log_loss(
        np.array([1, 1]), np.array([None, "decision"], dtype=object), np.array([None, None], dtype=object),
        np.array([0.5, 0.5]), np.array([[0.2, 0.2, 0.6]] * 2), np.array([[0.25] * 4] * 2),
        ["ko_tko", "submission", "decision"], ["1", "2", "3", "45"],
    )
    assert got == pytest.approx(-np.log(0.5 * 0.6))
```

- [ ] **Step 2: Run to verify failure**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_evaluate.py -q
```
Expected: `ImportError` on the two new names.

- [ ] **Step 3: Implement** (append to `src/mma/evaluate.py`)

```python
def expected_calibration_error(y_true, p_pred, n_bins: int = 10) -> float:
    """Count-weighted mean |mean prediction − empirical rate| over equal-width bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def joint_outcome_log_loss(
    y_winner, y_method, y_round, p_winner, method_probs, round_probs,
    method_classes, round_classes,
) -> float:
    """Log-loss of the realised (winner, method, round) outcome under the
    composed distribution P(w)·P(m)·P(r | finish). Decisions have no round
    term. Rows with unknown method are skipped; a finish with unknown round
    is skipped too (it cannot be scored)."""
    y_w = np.asarray(y_winner, dtype=float)
    p_w = np.clip(np.asarray(p_winner, dtype=float), _EPS, 1 - _EPS)
    method_probs = np.asarray(method_probs, dtype=float)
    round_probs = np.asarray(round_probs, dtype=float)
    m_index = {label: i for i, label in enumerate(method_classes)}
    r_index = {label: i for i, label in enumerate(round_classes)}
    losses = []
    for i, (m, r) in enumerate(zip(y_method, y_round)):
        if m is None or m not in m_index or (isinstance(m, float) and np.isnan(m)):
            continue
        p = p_w[i] if y_w[i] == 1.0 else 1.0 - p_w[i]
        p *= method_probs[i, m_index[m]]
        if m != "decision":
            if r is None or r not in r_index:
                continue
            p *= round_probs[i, r_index[r]]
        losses.append(-np.log(max(p, _EPS)))
    return float(np.mean(losses)) if losses else float("nan")
```

- [ ] **Step 4: Run tests**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_evaluate.py -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mma/evaluate.py tests/test_evaluate.py
git commit -m "Add expected calibration error and joint-outcome log-loss metrics"
```

---

### Task 3: Fold construction and recency weights

**Files:** Create `src/mma/walkforward.py`; Create `tests/test_walkforward.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_walkforward.py`:
```python
import numpy as np
import pandas as pd
import pytest

from mma.walkforward import FOLD_YEARS, Fold, make_folds, recency_weights


def _dates():
    return pd.Series(pd.to_datetime([
        "2005-06-01", "2015-03-01", "2016-07-01", "2017-01-15", "2017-12-31",
        "2018-01-01", "2018-06-01", "2024-05-05", "2025-02-02", "2026-08-08",
    ]))


def test_fold_years_are_2018_to_2025():
    assert FOLD_YEARS == (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)


def test_fold_2018_masks():
    folds = {f.year: f for f in make_folds(_dates())}
    f = folds[2018]
    d = _dates()
    assert f.train.tolist() == (d < "2017-01-01").tolist()
    assert f.inner_val.tolist() == ((d >= "2017-01-01") & (d < "2018-01-01")).tolist()
    assert f.eval.tolist() == ((d >= "2018-01-01") & (d < "2019-01-01")).tolist()


def test_no_eval_row_in_train_or_inner_val_and_masks_disjoint():
    for f in make_folds(_dates()):
        assert not (f.train & f.eval).any()
        assert not (f.inner_val & f.eval).any()
        assert not (f.train & f.inner_val).any()


def test_last_fold_absorbs_partial_2026():
    folds = {f.year: f for f in make_folds(_dates())}
    d = _dates()
    assert folds[2025].eval.tolist() == (d >= "2025-01-01").tolist()
    assert folds[2025].eval.sum() == 2


def test_train_start_drops_old_fights_from_train_only():
    folds = {f.year: f for f in make_folds(_dates(), train_start="2010-01-01")}
    f = folds[2018]
    assert not f.train[0]  # 2005 row excluded from training
    assert f.train.sum() == 2  # 2015, 2016
    assert f.inner_val.sum() == 2


def test_recency_weights_half_life():
    dates = pd.Series(pd.to_datetime(["2018-01-01", "2014-01-01", "2010-01-01"]))
    w = recency_weights(dates, reference=pd.Timestamp("2018-01-01"), half_life_years=4.0)
    assert w == pytest.approx([1.0, 0.5, 0.25], rel=1e-2)


def test_recency_weights_none_is_uniform():
    dates = pd.Series(pd.to_datetime(["2018-01-01", "2014-01-01"]))
    assert recency_weights(dates, pd.Timestamp("2018-01-01"), None).tolist() == [1.0, 1.0]
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError: mma.walkforward`.

- [ ] **Step 3: Implement `src/mma/walkforward.py`** (first part; scoring functions are added in Task 5)

```python
"""Expanding-window walk-forward evaluation.

Fold Y trains on fights before Y-1, early-stops on year Y-1, and is scored
on year Y (the last fold absorbs everything from 2025 onward). Pooled
metrics over all fold years are the headline; per-fold and slice metrics
are reported alongside. See the v3 design spec §4 SP1 / §5 for the locked
protocol and shipping bars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FOLD_YEARS = tuple(range(2018, 2026))
_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray
    inner_val: np.ndarray
    eval: np.ndarray

    @property
    def eval_start(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=1, day=1)


def make_folds(
    dates: pd.Series, fold_years=FOLD_YEARS, train_start: str | None = None
) -> list[Fold]:
    """Boolean masks (aligned to `dates`) for every fold year."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    start = pd.Timestamp(train_start) if train_start else None
    folds = []
    last = max(fold_years)
    for year in fold_years:
        val_start = pd.Timestamp(year=year - 1, month=1, day=1)
        eval_start = pd.Timestamp(year=year, month=1, day=1)
        eval_end = pd.Timestamp(year=year + 1, month=1, day=1)
        train = (d < val_start).to_numpy()
        if start is not None:
            train &= (d >= start).to_numpy()
        inner_val = ((d >= val_start) & (d < eval_start)).to_numpy()
        eval_mask = (d >= eval_start).to_numpy()
        if year != last:
            eval_mask &= (d < eval_end).to_numpy()
        folds.append(Fold(year, train, inner_val, eval_mask))
    return folds


def recency_weights(
    dates: pd.Series, reference: pd.Timestamp, half_life_years: float | None
) -> np.ndarray:
    """0.5 ** (age_in_years / half_life); uniform ones when half_life is None."""
    if half_life_years is None:
        return np.ones(len(dates), dtype=float)
    age_years = (reference - pd.to_datetime(dates)).dt.days.to_numpy() / _DAYS_PER_YEAR
    return np.power(0.5, np.clip(age_years, 0, None) / half_life_years)
```

- [ ] **Step 4: Run tests** → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mma/walkforward.py tests/test_walkforward.py
git commit -m "Walk-forward folds and recency weights"
```

---

### Task 4: Training code extensions — weights, config, fixed budget

**Files:** Modify `src/mma/models/xgb.py`, `src/mma/models/net.py`, `src/mma/models/train_loop.py`; Modify `tests/test_xgb.py`, `tests/test_train_loop.py`, `tests/test_net.py`

Read the three existing test files first and keep every existing test passing; all new parameters default to the old behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_xgb.py`:
```python
import numpy as np
import pandas as pd

from mma.models.xgb import train_binary


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((x["a"] + 0.3 * rng.normal(size=n) > 0).astype(int))
    return x, y


def test_train_binary_accepts_params_and_sample_weight():
    x, y = _toy()
    w = np.where(x["a"] > 0, 5.0, 1.0)
    model = train_binary(x[:150], y[:150], x[150:], y[150:],
                         params={"max_depth": 2}, sample_weight=w[:150])
    assert model.get_params()["max_depth"] == 2
    assert model.predict_proba(x[150:]).shape == (50, 2)


def test_train_binary_fixed_rounds_has_no_early_stopping():
    x, y = _toy()
    model = train_binary(x, y, None, None, fixed_rounds=37)
    assert model.get_booster().num_boosted_rounds() == 37
```

Append to `tests/test_train_loop.py` (reuse that file's existing synthetic-data helpers if present; otherwise this self-contained version):
```python
import numpy as np
import torch

from mma.models.train_loop import train_one


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    wc = rng.integers(0, 3, size=n)
    y = (x[:, 0] > 0).astype(np.float32)
    targets = {
        "y_winner": torch.tensor(y),
        "y_method": torch.tensor(rng.integers(0, 3, size=n)),
        "y_round": torch.tensor(rng.integers(0, 4, size=n)),
        "three_round": torch.tensor(rng.integers(0, 2, size=n).astype(bool)),
    }
    return x, wc, targets


def _split(x, wc, t, k=200):
    a = {key: v[:k] for key, v in t.items()}
    b = {key: v[k:] for key, v in t.items()}
    return x[:k], wc[:k], a, x[k:], wc[k:], b


def test_train_one_config_changes_architecture():
    x, wc, t = _synthetic()
    net, info = train_one(0, *_split(x, wc, t), max_epochs=2,
                          config={"hidden": (16, 8), "dropout": 0.1, "embedding_dim": 2})
    assert net.weight_class_embedding.embedding_dim == 2
    assert net.trunk[0].out_features == 16


def test_train_one_fixed_epochs_runs_exactly_that_many():
    x, wc, t = _synthetic()
    net, info = train_one(0, *_split(x, wc, t), fixed_epochs=3)
    assert info["epochs_run"] == 3 and info["best_epoch"] == 2


def test_train_one_sample_weight_changes_result():
    x, wc, t = _synthetic()
    split = _split(x, wc, t)
    net_a, _ = train_one(0, *split, max_epochs=3)
    w = np.where(x[:200, 1] > 0, 5.0, 0.2)
    net_b, _ = train_one(0, *split, max_epochs=3, sample_weight=w)
    pa = net_a.state_dict()["winner_head.weight"]
    pb = net_b.state_dict()["winner_head.weight"]
    assert not torch.allclose(pa, pb)
```

Append to `tests/test_net.py`:
```python
import torch

from mma.models.net import multitask_loss


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
```

- [ ] **Step 2: Run** the three test files → new tests fail on unexpected keyword arguments.

- [ ] **Step 3: Implement**

`src/mma/models/xgb.py` — replace `train_binary` and `train_multiclass`:
```python
def _classifier(objective: str, params: dict | None, fixed_rounds: int | None, **extra):
    merged = {**BASE_PARAMS, **(params or {})}
    if fixed_rounds is not None:
        return xgb.XGBClassifier(
            **merged, n_estimators=fixed_rounds, objective=objective,
            enable_categorical=True, **extra,
        )
    return xgb.XGBClassifier(
        **merged, n_estimators=MAX_ROUNDS, objective=objective,
        early_stopping_rounds=EARLY_STOP, enable_categorical=True, **extra,
    )


def train_binary(x_train, y_train, x_val, y_val, *, params=None,
                 sample_weight=None, fixed_rounds=None) -> xgb.XGBClassifier:
    """Early-stop on (x_val, y_val); or, with fixed_rounds, train exactly that
    many rounds and ignore the validation set (pass None)."""
    model = _classifier("binary:logistic", params, fixed_rounds, eval_metric="logloss")
    fit_kwargs = {"sample_weight": sample_weight, "verbose": False}
    if fixed_rounds is None:
        fit_kwargs["eval_set"] = [(x_val, y_val)]
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def train_multiclass(x_train, y_train, x_val, y_val, classes, *, params=None,
                     sample_weight=None, fixed_rounds=None) -> xgb.XGBClassifier:
    mapping = {label: index for index, label in enumerate(classes)}
    model = _classifier("multi:softprob", params, fixed_rounds,
                        num_class=len(classes), eval_metric="mlogloss")
    fit_kwargs = {"sample_weight": sample_weight, "verbose": False}
    if fixed_rounds is None:
        fit_kwargs["eval_set"] = [(x_val, y_val.map(mapping))]
    model.fit(x_train, y_train.map(mapping), **fit_kwargs)
    return model
```

`src/mma/models/net.py` — `multitask_loss` gains `sample_weight=None` (a 1-D tensor aligned with the batch). Winner term: `F.binary_cross_entropy_with_logits(..., reduction="none")` then a weighted mean `(w * l).sum() / w.sum()` (plain mean when `sample_weight is None`). Method/round terms: `F.cross_entropy(..., weight=class_weights, reduction="none")` on the known rows, weighted mean with `sample_weight[method_known]` (or plain mean). Keep the `method_scale`/`round_scale` arguments.

`src/mma/models/train_loop.py` — `train_one` signature becomes:
```python
def train_one(seed, x_train, wc_train, targets_train, x_val, wc_val, targets_val,
              max_epochs: int = 200, patience: int = 20, batch_size: int = 256,
              n_weight_classes: int | None = None, config: dict | None = None,
              sample_weight=None, fixed_epochs: int | None = None):
```
- `config` keys (all optional, defaults = today's values): `hidden` (tuple), `dropout`, `embedding_dim`, `lr` (1e-3), `weight_decay` (1e-4), `method_scale` (0.5), `round_scale` (0.25). Build `MultiTaskNet(n_features, n_weight_classes, embedding_dim=..., hidden=..., dropout=...)` and `AdamW(lr=..., weight_decay=...)`; pass the scales and the batch's `sample_weight` slice (as a float32 tensor) to `multitask_loss`.
- `fixed_epochs`: when set, run exactly that many epochs with no validation pass and no early stopping; the returned net is the final state; `info = {"best_epoch": fixed_epochs - 1, "best_val_log_loss": None, "epochs_run": fixed_epochs}`. Otherwise unchanged behaviour plus `"epochs_run": epoch + 1`.
- Keep `torch.use_deterministic_algorithms(True)` and the seeded generator so results stay bit-reproducible.

- [ ] **Step 4: Run** `tests/test_xgb.py tests/test_train_loop.py tests/test_net.py tests/test_processed_torch.py` → all pass (existing tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/mma/models/xgb.py src/mma/models/net.py src/mma/models/train_loop.py tests/test_xgb.py tests/test_train_loop.py tests/test_net.py
git commit -m "Training code: sample weights, config dict, fixed-budget mode (backward compatible)"
```

---

### Task 5: Scoring, slices, pooling, and bars

**Files:** Modify `src/mma/walkforward.py`; Modify `tests/test_walkforward.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_walkforward.py`)

```python
from mma.walkforward import bar_check, build_report, pool, score_rows, slice_masks

METHOD = ["ko_tko", "submission", "decision"]
ROUND = ["1", "2", "3", "45"]


def _rows(n=8, seed=0):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame({
        "date": pd.to_datetime(["2018-03-01"] * 4 + ["2019-03-01"] * 4),
        "y_winner": rng.integers(0, 2, size=n),
        "y_method": ["ko_tko", "decision", "submission", "decision"] * 2,
        "y_finish_round": ["1", None, "3", None] * 2,
        "weight_class": ["Lightweight", "Women's Strawweight"] * 4,
        "scheduled_rounds": pd.array([3, 5, 3, 3] * 2, dtype="Int64"),
        "debut_a": [True, False] * 4, "debut_b": [False] * 8,
    })
    pred = {
        "winner": rng.uniform(0.2, 0.8, size=n),
        "method": np.full((n, 3), 1 / 3),
        "round": np.full((n, 4), 0.25),
    }
    return feats, pred


def test_score_rows_reports_all_metrics():
    feats, pred = _rows()
    out = score_rows(feats, pred, METHOD, ROUND)
    for key in ("n", "winner_log_loss", "accuracy", "brier", "ece",
                "joint_log_loss", "method_macro_f1", "round_macro_f1", "n_method", "n_round"):
        assert key in out
    assert out["n"] == 8 and out["n_method"] == 8 and out["n_round"] == 4


def test_score_rows_without_method_head():
    feats, pred = _rows()
    out = score_rows(feats, {"winner": pred["winner"], "method": None, "round": None}, METHOD, ROUND)
    assert out["joint_log_loss"] is None and out["method_macro_f1"] is None


def test_slice_masks():
    feats, _ = _rows()
    masks = slice_masks(feats)
    assert masks["debut"].sum() == 4
    assert masks["womens"].sum() == 4
    assert masks["five_round"].sum() == 2


def test_pool_concatenates_fold_predictions_in_row_order():
    feats, pred = _rows()
    a = np.array([True] * 4 + [False] * 4)
    b = ~a
    pooled_feats, pooled_pred = pool(feats, [(a, {k: (v[a] if v is not None else None) for k, v in pred.items()}),
                                             (b, {k: (v[b] if v is not None else None) for k, v in pred.items()})])
    assert len(pooled_feats) == 8
    assert pooled_pred["winner"].tolist() == pred["winner"].tolist()


def test_bar_check_arithmetic():
    cand = {"pooled": {"winner_log_loss": 0.640}, "folds": {"2018": {"winner_log_loss": 0.65}, "2019": {"winner_log_loss": 0.63}}}
    inc = {"pooled": {"winner_log_loss": 0.650}, "folds": {"2018": {"winner_log_loss": 0.64}, "2019": {"winner_log_loss": 0.66}}}
    out = bar_check(cand, inc, sigma_seed=0.002)
    assert out["bar"] == pytest.approx(0.004)          # max(0.003, 2*sigma)
    assert out["delta"] == pytest.approx(-0.010)
    assert out["worst_fold_delta"] == pytest.approx(0.010)  # 2018 got worse by exactly 0.01
    assert out["clears_delta"] is True and out["no_fold_regression"] is True and out["ships"] is True
    inc["folds"]["2018"]["winner_log_loss"] = 0.635
    assert bar_check(cand, inc, 0.002)["ships"] is False


def test_build_report_shape():
    feats, pred = _rows()
    a = np.array([True] * 4 + [False] * 4)
    report = build_report(
        name="toy", config={"k": 1}, features=feats,
        fold_results=[(2018, a, {k: (v[a] if v is not None else None) for k, v in pred.items()}, {"best_iteration": 10}),
                      (2019, ~a, {k: (v[~a] if v is not None else None) for k, v in pred.items()}, {"best_iteration": 20})],
        method_classes=METHOD, round_classes=ROUND,
    )
    assert set(report) >= {"name", "config", "folds", "pooled", "slices", "fit_info"}
    assert set(report["folds"]) == {"2018", "2019"}
    assert report["fit_info"]["best_iteration"] == [10, 20]
    assert "debut" in report["slices"] and "womens" in report["slices"]
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement** (append to `src/mma/walkforward.py`)

```python
from mma.evaluate import (  # noqa: E402  (placed after dataclasses on purpose)
    accuracy, brier_score, expected_calibration_error, joint_outcome_log_loss,
    log_loss, macro_f1,
)

MIN_FOLD_REGRESSION = 0.01
MIN_BAR = 0.003


def score_rows(features: pd.DataFrame, pred: dict, method_classes, round_classes) -> dict:
    """Metrics for one set of rows. `pred` has 'winner' (n,), and 'method'
    (n,3) / 'round' (n,4) or None when the candidate has no such head."""
    y = features["y_winner"].to_numpy(dtype=float)
    p = np.asarray(pred["winner"], dtype=float)
    out = {
        "n": int(len(y)),
        "winner_log_loss": round(log_loss(y, p), 4),
        "accuracy": round(accuracy(y, p), 4),
        "brier": round(brier_score(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "joint_log_loss": None, "method_macro_f1": None, "round_macro_f1": None,
        "n_method": int(features["y_method"].notna().sum()),
        "n_round": int(features["y_finish_round"].notna().sum()),
    }
    if pred.get("method") is not None and pred.get("round") is not None:
        y_m = features["y_method"].astype(object).where(features["y_method"].notna(), None).to_numpy()
        y_r = features["y_finish_round"].astype(object).where(features["y_finish_round"].notna(), None).to_numpy()
        out["joint_log_loss"] = round(joint_outcome_log_loss(
            y, y_m, y_r, p, pred["method"], pred["round"], method_classes, round_classes), 4)
        known = features["y_method"].notna().to_numpy()
        if known.any():
            m_pred = [method_classes[i] for i in np.asarray(pred["method"])[known].argmax(axis=1)]
            out["method_macro_f1"] = round(macro_f1(list(y_m[known]), m_pred), 4)
        finish = features["y_finish_round"].notna().to_numpy()
        if finish.any():
            r_pred = [round_classes[i] for i in np.asarray(pred["round"])[finish].argmax(axis=1)]
            out["round_macro_f1"] = round(macro_f1(list(y_r[finish]), r_pred), 4)
    return out


def slice_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {
        "debut": (features["debut_a"].astype(bool) | features["debut_b"].astype(bool)).to_numpy(),
        "womens": features["weight_class"].astype("string").str.startswith("Women").fillna(False).to_numpy(),
        "five_round": (features["scheduled_rounds"].fillna(3) >= 5).to_numpy(),
    }
    if "external_missing" in features.columns:  # SP2 adds this column
        masks["external_missing"] = features["external_missing"].astype(bool).to_numpy()
    return masks


def pool(features: pd.DataFrame, fold_preds: list[tuple[np.ndarray, dict]]):
    """Concatenate (mask, pred) pairs into one row-aligned frame + prediction dict."""
    frames, winner, method, rounds = [], [], [], []
    has_heads = all(p.get("method") is not None for _, p in fold_preds)
    for mask, pred in fold_preds:
        frames.append(features.loc[mask])
        winner.append(np.asarray(pred["winner"]))
        if has_heads:
            method.append(np.asarray(pred["method"]))
            rounds.append(np.asarray(pred["round"]))
    pooled = {
        "winner": np.concatenate(winner),
        "method": np.concatenate(method) if has_heads else None,
        "round": np.concatenate(rounds) if has_heads else None,
    }
    return pd.concat(frames).reset_index(drop=True), pooled


def _subset(pred: dict, mask: np.ndarray) -> dict:
    return {k: (np.asarray(v)[mask] if v is not None else None) for k, v in pred.items()}


def build_report(name, config, features, fold_results, method_classes, round_classes) -> dict:
    """fold_results: list of (year, eval_mask, pred, fit_info)."""
    folds, fit_info = {}, {}
    for year, mask, pred, info in fold_results:
        folds[str(year)] = score_rows(features.loc[mask], pred, method_classes, round_classes)
        for key, value in (info or {}).items():
            fit_info.setdefault(key, []).append(value)
    pooled_feats, pooled_pred = pool(features, [(m, p) for _, m, p, _ in fold_results])
    slices = {}
    for slice_name, mask in slice_masks(pooled_feats).items():
        if mask.any():
            slices[slice_name] = score_rows(pooled_feats.loc[mask], _subset(pooled_pred, mask),
                                            method_classes, round_classes)
    return {
        "name": name, "config": config, "fold_years": [int(y) for y, *_ in fold_results],
        "folds": folds,
        "pooled": score_rows(pooled_feats, pooled_pred, method_classes, round_classes),
        "slices": slices, "fit_info": fit_info,
    }


def bar_check(candidate: dict, incumbent: dict, sigma_seed: float) -> dict:
    """Pre-registered winner bar: pooled Δ < −max(MIN_BAR, 2σ) and no fold worse by > 0.01."""
    bar = max(MIN_BAR, 2.0 * sigma_seed)
    delta = candidate["pooled"]["winner_log_loss"] - incumbent["pooled"]["winner_log_loss"]
    fold_deltas = {
        year: candidate["folds"][year]["winner_log_loss"] - incumbent["folds"][year]["winner_log_loss"]
        for year in candidate["folds"] if year in incumbent["folds"]
    }
    worst = round(max(fold_deltas.values()), 6) if fold_deltas else 0.0  # rounding: 0.65-0.64 != 0.01 in floats
    clears = delta < -bar
    no_regression = worst <= MIN_FOLD_REGRESSION
    return {
        "bar": bar, "sigma_seed": sigma_seed, "delta": round(delta, 4),
        "fold_deltas": {k: round(v, 4) for k, v in fold_deltas.items()},
        "worst_fold_delta": round(worst, 4),
        "clears_delta": bool(clears), "no_fold_regression": bool(no_regression),
        "ships": bool(clears and no_regression),
    }
```
(Move the `evaluate` import to the top of the module with the other imports; the `noqa` comment is only there because the plan shows it as an append.)

- [ ] **Step 4: Run** `tests/test_walkforward.py` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mma/walkforward.py tests/test_walkforward.py
git commit -m "Walk-forward scoring, slices, pooling, and pre-registered bar check"
```

---

### Task 6: Candidates — one protocol for Elo, XGBoost, torch

**Files:** Create `src/mma/candidates.py`; Create `tests/test_candidates.py`

Every candidate implements:
```python
def fit_predict(self, features: pd.DataFrame, fold: Fold, sample_weight: np.ndarray | None) -> tuple[dict, dict]:
    """Returns (pred, info). pred = {"winner": (n_eval,), "method": (n_eval,3)|None, "round": (n_eval,4)|None}
    aligned with features.loc[fold.eval] in row order. info = fit diagnostics (best_iteration / best_epoch / temperature)."""
```
`sample_weight` is aligned with `features` (full length); candidates slice it with their training mask. In fixed-budget mode the training mask is `fold.train | fold.inner_val`.

- [ ] **Step 1: Write the failing tests**

`tests/test_candidates.py`:
```python
import numpy as np
import pandas as pd
import pytest

from mma.candidates import EloCandidate, TorchCandidate, XGBCandidate
from mma.walkforward import make_folds


def _table(n=600, seed=0):
    """Tiny feature table shaped like features.parquet: dates 2014-2019, a
    signal column, and the identifier/target columns the candidates need."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2014-01-01") + pd.to_timedelta(rng.integers(0, 6 * 365, size=n), unit="D")
    elo_diff = rng.normal(0, 120, size=n)
    p = 1 / (1 + 10 ** (-elo_diff / 400))
    y = (rng.uniform(size=n) < p).astype(int)
    methods = rng.choice(["ko_tko", "submission", "decision"], size=n)
    rounds = np.where(methods == "decision", None, rng.choice(["1", "2", "3"], size=n))
    return pd.DataFrame({
        "fight_id": [f"f{i}" for i in range(n)], "date": dates, "swapped": False,
        "y_winner": y, "y_method": methods, "y_finish_round": pd.array(rounds, dtype="string"),
        "weight_class": rng.choice(["Lightweight", "Women's Strawweight"], size=n),
        "title_fight": False, "scheduled_rounds": pd.array(rng.choice([3, 5], size=n), dtype="Int64"),
        "elo_diff": elo_diff, "noise": rng.normal(size=n),
        "debut_a": False, "debut_b": False,
    }).sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="module")
def table():
    return _table()


@pytest.fixture(scope="module")
def fold(table):
    return make_folds(table["date"], fold_years=(2019,))[0]


def test_elo_candidate_is_expected_score(table, fold):
    pred, info = EloCandidate().fit_predict(table, fold, None)
    diff = table.loc[fold.eval, "elo_diff"].to_numpy()
    assert pred["winner"] == pytest.approx(1 / (1 + 10 ** (-diff / 400)))
    assert pred["method"] is None and pred["round"] is None


def test_xgb_candidate_shapes_and_signal(table, fold):
    pred, info = XGBCandidate(params={"max_depth": 2}).fit_predict(table, fold, None)
    n = int(fold.eval.sum())
    assert pred["winner"].shape == (n,) and pred["method"].shape == (n, 3) and pred["round"].shape == (n, 4)
    assert "best_iteration" in info
    y = table.loc[fold.eval, "y_winner"].to_numpy()
    assert np.mean((pred["winner"] > 0.5) == (y == 1)) > 0.6


def test_xgb_candidate_fixed_budget_uses_inner_val_for_training(table, fold):
    cand = XGBCandidate(params={"max_depth": 2}, fixed_rounds=20)
    pred, info = cand.fit_predict(table, fold, None)
    assert info["best_iteration"] == 20 and info["n_train"] == int((fold.train | fold.inner_val).sum())


def test_torch_candidate_shapes_and_determinism(table, fold):
    cand = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, max_epochs=3)
    pred_a, info_a = cand.fit_predict(table, fold, None)
    pred_b, _ = cand.fit_predict(table, fold, None)
    n = int(fold.eval.sum())
    assert pred_a["winner"].shape == (n,) and pred_a["round"].shape == (n, 4)
    assert np.array_equal(pred_a["winner"], pred_b["winner"])
    assert info_a["temperature"] and len(info_a["temperature"]) == 1


def test_torch_candidate_fixed_epochs(table, fold):
    cand = TorchCandidate(seeds=(0,), config={"hidden": (16, 8)}, fixed_epochs=2, temperature=1.2)
    pred, info = cand.fit_predict(table, fold, None)
    assert info["best_epoch"] == [1] and info["temperature"] == [1.2]


def test_sample_weight_reaches_xgb(table, fold):
    w = np.where(table["noise"] > 0, 3.0, 1.0)
    a, _ = XGBCandidate(params={"max_depth": 2}).fit_predict(table, fold, None)
    b, _ = XGBCandidate(params={"max_depth": 2}).fit_predict(table, fold, w)
    assert not np.allclose(a["winner"], b["winner"])
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement `src/mma/candidates.py`**

```python
"""Candidate models behind one fit/predict protocol for the walk-forward harness."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from mma.elo import expected_score
from mma.models.train_loop import (
    METHOD_CLASSES, ROUND_CLASSES, encode_targets, fit_temperature, predict, train_one,
)
from mma.models.xgb import feature_frame, train_binary, train_multiclass
from mma.tensors import Preprocessor
from mma.walkforward import Fold


def _slice_targets(targets: dict, mask: np.ndarray) -> dict:
    index = torch.tensor(mask)
    return {key: value[index] for key, value in targets.items()}


class EloCandidate:
    """Winner-only floor: the Elo expected score from the pre-fight rating diff."""
    name = "elo"

    def fit_predict(self, features, fold: Fold, sample_weight=None):
        diff = features.loc[fold.eval, "elo_diff"].to_numpy(dtype=float)
        winner = np.array([expected_score(d, 0.0) for d in np.nan_to_num(diff)])
        return {"winner": winner, "method": None, "round": None}, {}


@dataclass
class XGBCandidate:
    name: str = "xgb"
    params: dict = field(default_factory=dict)
    fixed_rounds: int | None = None

    def fit_predict(self, features, fold: Fold, sample_weight=None):
        x = feature_frame(features)
        train = (fold.train | fold.inner_val) if self.fixed_rounds is not None else fold.train
        val = fold.inner_val
        w = None if sample_weight is None else np.asarray(sample_weight)
        common = {"params": self.params, "fixed_rounds": self.fixed_rounds}

        y = features["y_winner"]
        winner = train_binary(x[train], y[train], x[val], y[val],
                              sample_weight=None if w is None else w[train], **common)
        info = {"best_iteration": int(winner.best_iteration) if self.fixed_rounds is None else self.fixed_rounds,
                "n_train": int(train.sum())}

        known = features["y_method"].notna().to_numpy()
        ym = features["y_method"]
        method = train_multiclass(x[train & known], ym[train & known], x[val & known], ym[val & known],
                                  METHOD_CLASSES, sample_weight=None if w is None else w[train & known], **common)
        finish = features["y_finish_round"].notna().to_numpy()
        yr = features["y_finish_round"]
        rounds = train_multiclass(x[train & finish], yr[train & finish], x[val & finish], yr[val & finish],
                                  ROUND_CLASSES, sample_weight=None if w is None else w[train & finish], **common)
        ev = x[fold.eval]
        pred = {
            "winner": winner.predict_proba(ev)[:, 1],
            "method": method.predict_proba(ev),
            "round": rounds.predict_proba(ev),
        }
        return pred, info


@dataclass
class TorchCandidate:
    name: str = "torch"
    seeds: tuple = (0, 1, 2, 3, 4)
    config: dict = field(default_factory=dict)
    max_epochs: int = 200
    patience: int = 20
    fixed_epochs: int | None = None
    temperature: float | None = None  # used only in fixed-epoch mode (no inner val to fit on)

    def fit_predict(self, features, fold: Fold, sample_weight=None):
        fixed = self.fixed_epochs is not None
        train = (fold.train | fold.inner_val) if fixed else fold.train
        prep = Preprocessor.fit(features, train_mask=train)
        x, wc = prep.transform(features)
        targets = encode_targets(features)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        three_round = torch.tensor((features.loc[fold.eval, "scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool))

        winners, methods, rounds, info = [], [], [], {"best_epoch": [], "temperature": []}
        for seed in self.seeds:
            net, fit_info = train_one(
                seed, x[train], wc[train], _slice_targets(targets, train),
                x[fold.inner_val], wc[fold.inner_val], _slice_targets(targets, fold.inner_val),
                max_epochs=self.max_epochs, patience=self.patience,
                n_weight_classes=prep.n_weight_classes, config=self.config,
                sample_weight=None if w is None else w[train],
                fixed_epochs=self.fixed_epochs,
            )
            if fixed:
                temperature = float(self.temperature or 1.0)
            else:
                raw = predict(net, x[fold.inner_val], wc[fold.inner_val])
                temperature = fit_temperature(raw["winner_logits"], targets["y_winner"][torch.tensor(fold.inner_val)].numpy())
            out = predict(net, x[fold.eval], wc[fold.eval], temperature=temperature)
            winners.append(out["winner"])
            methods.append(out["method"])
            rounds.append(torch.softmax(
                torch.tensor(out["round_logits"]).masked_fill(
                    three_round[:, None] & (torch.arange(4) == 3)[None, :], -1e9), dim=1).numpy())
            info["best_epoch"].append(fit_info["best_epoch"])
            info["temperature"].append(temperature)
        pred = {"winner": np.mean(winners, axis=0), "method": np.mean(methods, axis=0),
                "round": np.mean(rounds, axis=0)}
        return pred, info
```
Notes for the implementer: `MultiTaskNet.round_probs` already masks round "45" for three-round fights — prefer calling it (`MultiTaskNet.round_probs(torch.tensor(out["round_logits"]), three_round)`) over the inline mask above if the signature fits. `expected_score(rating_a, rating_b)` lives in `mma.elo`.

- [ ] **Step 4: Run** `tests/test_candidates.py` → all pass (torch tests take a few seconds).

- [ ] **Step 5: Commit**

```bash
git add src/mma/candidates.py tests/test_candidates.py
git commit -m "Candidate protocol: Elo, XGBoost, torch ensemble with config, weights, fixed budget"
```

---

### Task 7: CLI runner and baseline reports

**Files:** Create `scripts/run_walkforward.py`; Create `models/walkforward/{elo,xgb_v1,torch_v1}.json`; Modify `.gitignore` (nothing needed — reports are committed)

- [ ] **Step 1: Write `scripts/run_walkforward.py`**

```python
"""Score one candidate on the walk-forward harness and write a JSON report.

Examples:
  python scripts/run_walkforward.py --candidate elo --name elo
  python scripts/run_walkforward.py --candidate xgb --name xgb_v1
  python scripts/run_walkforward.py --candidate torch --name torch_v1 --seeds 0,1,2,3,4
  python scripts/run_walkforward.py --candidate torch --name torch_v1_seeds5 --seeds 5,6,7,8,9
  python scripts/run_walkforward.py --candidate xgb --name xgb_hl4 --half-life 4 --train-start 2005-01-01
  python scripts/run_walkforward.py --candidate torch --name torch_refit --fixed-budget-from models/walkforward/torch_v1.json

Reports land in models/walkforward/<name>.json. Nothing here touches the
deployed artifacts under models/torch or models/xgb_*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mma.candidates import EloCandidate, TorchCandidate, XGBCandidate  # noqa: E402
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES  # noqa: E402
from mma.walkforward import build_report, make_folds, recency_weights  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models" / "walkforward"


def fixed_budget_from(report_path: Path) -> dict:
    """Median fit budget (and temperature) across the earlier folds of a reference report."""
    report = json.loads(Path(report_path).read_text())
    info = report["fit_info"]
    budget = {}
    if "best_iteration" in info:
        budget["fixed_rounds"] = int(np.median(info["best_iteration"]))
    if "best_epoch" in info:
        epochs = [np.median(e) if isinstance(e, list) else e for e in info["best_epoch"]]
        budget["fixed_epochs"] = int(np.median(epochs)) + 1
    if "temperature" in info:
        temps = [np.median(t) if isinstance(t, list) else t for t in info["temperature"]]
        budget["temperature"] = float(np.median(temps))
    return budget


def build_candidate(args) -> object:
    config = json.loads(args.config_json) if args.config_json else {}
    budget = fixed_budget_from(args.fixed_budget_from) if args.fixed_budget_from else {}
    if args.candidate == "elo":
        return EloCandidate()
    if args.candidate == "xgb":
        return XGBCandidate(name=args.name, params=config, fixed_rounds=budget.get("fixed_rounds"))
    seeds = tuple(int(s) for s in args.seeds.split(","))
    return TorchCandidate(name=args.name, seeds=seeds, config=config,
                         fixed_epochs=budget.get("fixed_epochs"), temperature=budget.get("temperature"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", choices=["elo", "xgb", "torch"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--config-json", default=None, help="JSON dict of model params / torch config")
    parser.add_argument("--train-start", default=None, help="drop training fights before this date")
    parser.add_argument("--half-life", type=float, default=None, help="recency half-life in years")
    parser.add_argument("--fixed-budget-from", type=Path, default=None,
                        help="reference report; train on train+inner_val with its median budget, no early stopping")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    features = pd.read_parquet(PROCESSED / "features.parquet").sort_values("date", kind="stable").reset_index(drop=True)
    candidate = build_candidate(args)
    fold_results = []
    for fold in make_folds(features["date"], train_start=args.train_start):
        weights = recency_weights(features["date"], fold.eval_start, args.half_life)
        pred, info = candidate.fit_predict(features, fold, weights if args.half_life else None)
        fold_results.append((fold.year, fold.eval, pred, info))
        print(f"fold {fold.year}: n_eval={int(fold.eval.sum())} info={info}")

    config = {
        "candidate": args.candidate, "seeds": args.seeds if args.candidate == "torch" else None,
        "config": json.loads(args.config_json) if args.config_json else {},
        "train_start": args.train_start, "half_life": args.half_life,
        "fixed_budget_from": str(args.fixed_budget_from) if args.fixed_budget_from else None,
        "n_features_rows": int(len(features)),
    }
    report = build_report(args.name, config, features, fold_results, METHOD_CLASSES, ROUND_CLASSES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pooled": report["pooled"], "folds": {y: f["winner_log_loss"] for y, f in report["folds"].items()}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the three baselines**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate elo --name elo
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_v1
time ~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v1
```
Expected: eight folds each; pooled n ≈ 3,900 (2018–2026 decisive fights; print the exact number); torch pooled winner log-loss in the 0.63–0.66 range, XGB slightly worse, Elo ≈ 0.68; torch run under ~3 minutes. Sanity: the 2021–2023 fold numbers should be in the neighbourhood of the committed validation metrics (they will not match exactly: different training cutoff per fold).

- [ ] **Step 3: Determinism check**

```bash
cp models/walkforward/torch_v1.json /tmp/torch_v1_first.json
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v1 >/dev/null
diff /tmp/torch_v1_first.json models/walkforward/torch_v1.json && echo IDENTICAL
```
Expected: `IDENTICAL`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_walkforward.py models/walkforward/elo.json models/walkforward/xgb_v1.json models/walkforward/torch_v1.json
git commit -m "Walk-forward runner and baseline reports for Elo, XGBoost, torch"
```

---

### Task 8: Noise floor

**Files:** Create `scripts/noise_floor.py`; Create `models/walkforward/torch_v1_seeds5.json`, `torch_v1_seeds10.json`, `noise_floor.json`; Modify `tests/test_walkforward.py`

- [ ] **Step 1: Test for the sigma arithmetic** (append to `tests/test_walkforward.py`)

```python
from scripts.noise_floor import sigma_from_reports


def test_sigma_from_reports_is_sample_std():
    reports = [{"pooled": {"winner_log_loss": v}} for v in (0.650, 0.652, 0.648)]
    out = sigma_from_reports(reports)
    assert out["sigma_seed"] == pytest.approx(np.std([0.650, 0.652, 0.648], ddof=1))
    assert out["bar"] == pytest.approx(max(0.003, 2 * out["sigma_seed"]))
    assert out["n_reports"] == 3
```

- [ ] **Step 2: Write `scripts/noise_floor.py`**

```python
"""Measure the seed noise floor of the incumbent torch configuration.

sigma_seed = sample standard deviation of pooled walk-forward winner
log-loss across runs that differ only in their seed set. The shipping bar
is max(0.003, 2 * sigma_seed). Usage:
    python scripts/noise_floor.py models/walkforward/torch_v1.json models/walkforward/torch_v1_seeds5.json models/walkforward/torch_v1_seeds10.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "walkforward" / "noise_floor.json"
MIN_BAR = 0.003


def sigma_from_reports(reports: list[dict]) -> dict:
    values = [r["pooled"]["winner_log_loss"] for r in reports]
    sigma = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {"n_reports": len(values), "pooled_winner_log_loss": values,
            "sigma_seed": sigma, "bar": max(MIN_BAR, 2.0 * sigma)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    result = sigma_from_reports(reports)
    result["reports"] = [str(p) for p in args.reports]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the two extra seed sets and the floor**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v1_seeds5 --seeds 5,6,7,8,9
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v1_seeds10 --seeds 10,11,12,13,14
~/.venvs/mma/bin/python scripts/noise_floor.py models/walkforward/torch_v1.json models/walkforward/torch_v1_seeds5.json models/walkforward/torch_v1_seeds10.json
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_walkforward.py -q
```
Expected: σ_seed on the order of 0.001–0.002 (the model-v2 session measured ~0.0014 between two 10-seed ensembles on 1,507 fights; on ~3,900 pooled fights it should be smaller). Record σ_seed and the bar in the completion notes.

- [ ] **Step 4: Commit**

```bash
git add scripts/noise_floor.py tests/test_walkforward.py models/walkforward/torch_v1_seeds5.json models/walkforward/torch_v1_seeds10.json models/walkforward/noise_floor.json
git commit -m "Measure the seed noise floor and derive the shipping bar"
```

---

### Task 9: Refit-strategy experiment

**Files:** Create `models/walkforward/xgb_refit.json`, `torch_refit.json`, `refit_decision.json`

- [ ] **Step 1: Run strategy (B) for both learners**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_refit --fixed-budget-from models/walkforward/xgb_v1.json
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_refit --fixed-budget-from models/walkforward/torch_v1.json
```
Strategy (B) trains each fold on train ∪ inner_val with the median budget from the (A) report: XGB fixed rounds = median `best_iteration`; torch fixed epochs = median best_epoch + 1 and temperature = median per-seed temperature.

- [ ] **Step 2: Decide mechanically**

```bash
~/.venvs/mma/bin/python - <<'EOF'
import json
from pathlib import Path
from mma.walkforward import bar_check
wf = Path("models/walkforward")
sigma = json.loads((wf / "noise_floor.json").read_text())["sigma_seed"]
decision = {}
for learner in ("xgb", "torch"):
    a = json.loads((wf / f"{learner}_v1.json").read_text())
    b = json.loads((wf / f"{learner}_refit.json").read_text())
    delta = b["pooled"]["winner_log_loss"] - a["pooled"]["winner_log_loss"]
    decision[learner] = {
        "A_pooled": a["pooled"]["winner_log_loss"], "B_pooled": b["pooled"]["winner_log_loss"],
        "delta_B_minus_A": round(delta, 4), "sigma_seed": sigma,
        "B_not_worse_than_A_by_sigma": bool(delta <= sigma),
        "bar_check_B_vs_A": bar_check(b, a, sigma),
        "budget": b["config"],
    }
decision["deployment_recipe"] = "refit_through_latest" if decision["torch"]["B_not_worse_than_A_by_sigma"] else "incumbent_protocol"
(wf / "refit_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps(decision, indent=2))
EOF
```
The rule (spec §4 SP1): (B) becomes the deployment recipe iff its pooled winner log-loss is not worse than (A) by more than σ_seed, judged on the torch ensemble (the deployed model). Record the numbers in the completion notes either way.

- [ ] **Step 3: Commit**

```bash
git add models/walkforward/xgb_refit.json models/walkforward/torch_refit.json models/walkforward/refit_decision.json
git commit -m "Refit-strategy experiment: fixed budget on all data vs inner-val early stopping"
```

---

### Task 10: `--refit-through` in the train scripts (ships only if Task 9 chose it)

**Files:** Modify `scripts/train_xgb.py`, `scripts/train_torch.py`; possibly regenerate `models/**`; Modify `tests/test_processed_torch.py` / `tests/test_processed_ratings.py` / `tests/test_xgb.py` only if they assert on `metrics_val.json` keys that change (read them first)

If `refit_decision.json` says `incumbent_protocol`, implement the flag (Steps 1–2) but do **not** retrain the deployed artifacts (skip Steps 3–4) and say so in the completion notes.

- [ ] **Step 1: Add the mode to both scripts**

New arguments: `--refit-through DATE` (train on every decisive fight dated ≤ DATE, default off), `--budget N` (XGB rounds / torch epochs; required with `--refit-through`), `--temperature T` (torch only; default 1.0), `--report PATH` (the harness report whose pooled metrics are copied into `metrics_val.json` as the model's validation evidence).
- XGB: `train_binary(x[train], y[train], None, None, fixed_rounds=args.budget)` and likewise for method/round; write `xgb_metrics_val.json` as `{"mode": "refit_through", "train_through": DATE, "budget": N, "n_train": ..., "harness_report": PATH, "winner": <pooled block from the report>, "method": {...}, "finish_round": {...}}` keeping the same inner keys (`n_val`, `accuracy`, `log_loss`, `brier`, `macro_f1`) that the README and tests read — populate them from the report's pooled metrics.
- Torch: `Preprocessor.fit` on the refit mask; `train_one(..., fixed_epochs=args.budget)`; per-seed temperature = `args.temperature`; save checkpoints exactly as today; `metrics_val.json` gets the same `mode`/`train_through`/`budget`/`harness_report` fields plus `winner_ensemble`/`method_ensemble` blocks copied from the report's pooled metrics.
- Default invocation (no `--refit-through`) must behave exactly as today (the weekly Action and `roll_window.py` rely on it) — until Step 3 flips the default.

- [ ] **Step 2: Test the flag on a scratch dir**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/train_xgb.py --refit-through 2026-12-31 --budget 100 --report models/walkforward/xgb_refit.json --models-dir /tmp/sp1_xgb
~/.venvs/mma/bin/python scripts/train_torch.py --refit-through 2026-12-31 --budget 12 --temperature 1.1 --report models/walkforward/torch_refit.json --out-dir /tmp/sp1_torch
ls /tmp/sp1_xgb /tmp/sp1_torch && ~/.venvs/mma/bin/python -c "import json; print(json.load(open('/tmp/sp1_torch/metrics_val.json'))['mode'])"
git status --short   # deployed artifacts untouched
```

- [ ] **Step 3 (only if the recipe was chosen): make refit the default deployment**

Set the scripts' defaults so a bare `python scripts/train_torch.py` / `train_xgb.py` (the weekly Action) runs the refit recipe: `REFIT_THROUGH = "latest"` (resolved to `features["date"].max()`), `BUDGET` = the values from `refit_decision.json["torch"]["budget"]` / `["xgb"]["budget"]`, `TEMPERATURE` likewise, `REPORT` = the committed harness report paths. Keep `--train-end/--val-start/--val-end` working (passing any of them switches back to the split protocol, which `roll_window.py --execute` uses). Then retrain the deployed artifacts:

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/train_xgb.py | tail -5
~/.venvs/mma/bin/python scripts/train_torch.py | tail -5
~/.venvs/mma/bin/python scripts/build_display_priors.py | tail -2
~/.venvs/mma/bin/python -c "from pathlib import Path; from mma.versioning import model_version; print('model v3 hash', model_version(Path('.')))"
~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
```
Expected: suite green (fix any `test_processed_*` assertion that read the old `metrics_val.json` shape, keeping the assertion's intent); new hash; `git status` shows only `models/**` changed.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_xgb.py scripts/train_torch.py tests/ models/
git status --short
git commit -m "Deploy the refit-through-latest recipe: models trained on all data through the newest event"
```
(or, if not chosen: `git add scripts/train_xgb.py scripts/train_torch.py tests/` and `"Add --refit-through mode to the train scripts (not the default: refit did not clear the bar)"`.)

---

### Task 11: `roll_window.py` compatibility and README

**Files:** Modify `scripts/roll_window.py` (docstring + `model_version` sentence only, unless Step 1 finds more); Modify `README.md`

- [ ] **Step 1: Check the promotion hook against the new world**

Read `scripts/roll_window.py`. Its docstring still says "that commit's git sha becomes the new model_version" — fix to the artifact hash. Its `--execute` path calls `train_torch.py` with explicit `--train-end/--val-start/--val-end`, which still selects the split protocol, so it keeps working; confirm by reading the subprocess call. If the refit recipe shipped, add one sentence to the docstring saying that after a promotion the human should re-run the default (refit) training so the deployed model includes the newest fights, and that the SP1 harness (not the two-year slice) is now the primary comparison — the full switch of the promotion gate to the harness is SP4 work.

- [ ] **Step 2: README**

Replace the "## Results so far" intro paragraph and add a walk-forward subsection: the protocol (expanding window, fold years 2018–2025 plus 2026, ~N pooled fights — use the real count), a table of pooled winner metrics for Elo / XGBoost / torch from the three baseline reports, σ_seed and the bar from `noise_floor.json`, and the refit-strategy result and what was deployed. Keep the old 2021–2023 table under a heading "Original validation window (2021–2023, for continuity)". State plainly that the prospective track record is the only true holdout. Update the "Development notes" bullet on model identity to mention `models/walkforward/` reports as the evidence behind each deployed model.

- [ ] **Step 3: Commit**

```bash
git add scripts/roll_window.py README.md
git commit -m "README: walk-forward evaluation replaces the spent-test story; roll_window docstring"
```

---

### Task 12: Verification and merge

- [ ] **Step 1: Full verification**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/make_dataset.py | tail -3 && git status --short data/processed
git log --oneline main..sp1-walkforward
```
Expected: green; no parquet drift; ~11 commits.

- [ ] **Step 2: Fill the completion notes below, commit, merge locally (no push)**

```bash
git add docs/superpowers/plans/2026-09-06-sp1-walkforward-harness.md
git commit -m "SP1 plan: completion notes"
git checkout main && git merge --no-ff sp1-walkforward -m "Merge sp1-walkforward: walk-forward harness, noise floor, refit-through-latest recipe"
```

---

## Completion notes (filled in during execution)

- Pooled evaluation rows: _n_ (fold years 2018–2025+2026)
- Baselines (pooled winner LL / acc / joint LL): Elo _…_ ; XGB v1 _…_ ; torch v1 _…_
- σ_seed: _…_ ; bar: _…_
- Refit experiment: XGB A _…_ vs B _…_ ; torch A _…_ vs B _…_ ; decision: _…_
- Deployed model after SP1: hash _…_ ; recipe _…_
- Deferred: `external_missing` slice (SP2); harness-driven promotion gate (SP4)
