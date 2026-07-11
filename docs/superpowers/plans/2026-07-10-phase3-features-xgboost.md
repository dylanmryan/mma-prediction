# Phase 3: Features + XGBoost Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A point-in-time-correct feature table (one row per decisive fight, corner-symmetrized) and three tuned XGBoost models (winner, method, finish round) evaluated on 2021–2023 validation — Baseline #2, the model the neural net must beat.

**Architecture:** A chronological history accumulator (`src/mma/history.py`) mirrors the Elo engine's single-pass design and emits each fighter's pre-fight career/rolling stats. A matchup assembler (`src/mma/features.py`) joins history + ratings + bio, applies hash-based corner symmetrization, and produces A-minus-B differentials plus targets. `scripts/build_features.py` writes `features.parquet`; `src/mma/models/xgb.py` + `scripts/train_xgb.py` train and persist the three models with validation metrics.

**Tech Stack:** pandas, xgboost≥2.0 (new dependency; native NaN + categorical support), pytest.

**Spec:** design doc §3 + cold-start section. Branch: `phase-3-features-xgboost`.

**Design decisions (locked here):**
- **Symmetrization**: per fight, corners are swapped when `int(md5(fight_id).hexdigest(), 16) % 2 == 1` — deterministic across runs/platforms (no RNG state, no salted `hash()`). Post-swap fighter A wins ⇔ `y_winner = 1`. This kills the 64.6% red-corner artifact.
- **Rows**: only decisive fights (`winner ∈ {a, b}`). Draws/NC still flow through the history accumulator (they're real career events) but get no feature row.
- **History semantics**: draws count in the fight denominator but not as wins; a draw resets streak to 0; NC fights are ignored entirely (consistent with Elo). Missing per-fight stats accumulate as 0 (affects mostly pre-2001 fights) — documented, not hidden.
- **Missing values stay NaN** in the feature table (XGBoost handles them natively); explicit `*_missing` flags for reach and dob so the model can distinguish "unknown" from "average". The neural net (Phase 4) will add neutral imputation on top.
- **Targets**: `y_winner` (binary), `y_method` (`ko_tko`/`submission`/`decision`, NA when method unknown), `y_finish_round` (`1`/`2`/`3`/`45`, finishes only, NA otherwise).
- **Splits (locked)**: train < 2021-01-01; validation 2021-01-01..2023-12-31 (all tuning/early stopping); test 2024+ **never read by any Phase 3 code path**.
- **XGBoost**: fixed sensible hyperparameters (max_depth 4, eta 0.05, subsample 0.8, colsample_bytree 0.8) with early stopping on validation log-loss, up to 2000 rounds — no grid search in v1 (early stopping is the tuner; honest and cheap).
- **The leakage test**: build features twice — full data vs data truncated to pre-2023 — and assert feature rows for common fights are identical. If any feature sees the future, this fails.

---

### Task 1: Add fight duration to the fights table

Per-minute rates need fight duration; `UFC.csv` has `match_time_sec` but Phase 1 didn't keep it.

**Files:** Modify `src/mma/dataset.py`, `tests/test_dataset_fights.py`; rebuild parquet.

- [ ] **Step 1:** Add to `tests/test_dataset_fights.py`:

```python
def test_match_time_sec_kept():
    raw = _raw_fights()
    raw["match_time_sec"] = [260.0, 900.0, None]
    fights = build_fights(raw)
    assert fights[fights["fight_id"] == "f2"].iloc[0]["match_time_sec"] == 260.0
    assert pd.isna(fights[fights["fight_id"] == "f3"].iloc[0]["match_time_sec"])
```

Also update `_raw_fights()` to include a `"match_time_sec": [260.0, 900.0, 452.0]` column so the other tests keep passing.

- [ ] **Step 2:** Run — FAILS (KeyError/assert). 
- [ ] **Step 3:** In `build_fights`, add to the frame dict after `title_fight`: `"match_time_sec": pd.to_numeric(raw["match_time_sec"], errors="coerce"),` and append `"match_time_sec"` to the `columns` list (at the end).
- [ ] **Step 4:** Run suite; rebuild parquet: `.venv/bin/python scripts/make_dataset.py`; rerun `scripts/build_ratings.py` (ratings unchanged by this column but parquet timestamps regenerate — verify `git diff --stat` shows only fights.parquet materially changed). Full suite green (72 expected).
- [ ] **Step 5:** Commit: `git add -u data/processed && git commit -m "Keep fight duration in fights table"`

---

### Task 2: Macro-F1 metric

**Files:** Modify `src/mma/evaluate.py`, `tests/test_evaluate.py`.

- [ ] **Step 1:** Add tests:

```python
def test_macro_f1_perfect():
    assert macro_f1(["x", "y", "x"], ["x", "y", "x"]) == 1.0


def test_macro_f1_one_class_wrong():
    # classes: x predicted perfectly (f1=1), y never predicted (f1=0) -> macro 0.5
    assert macro_f1(["x", "y"], ["x", "x"]) == pytest.approx(0.5)


def test_macro_f1_ignores_labels_missing_from_truth():
    assert macro_f1(["x", "x"], ["x", "y"]) == pytest.approx(1 / 3)
```

(add `from mma.evaluate import macro_f1` and `import pytest` to the imports)

- [ ] **Step 2:** Run — FAILS.
- [ ] **Step 3:** Append to `src/mma/evaluate.py`:

```python
def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over the union of true and predicted labels."""
    true = list(y_true)
    pred = list(y_pred)
    labels = sorted(set(true) | set(pred))
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))
```

- [ ] **Step 4:** Suite green (75 expected). **Step 5:** Commit: `"Add macro-F1 metric"`

---

### Task 3: Fighter history accumulator (`history.py`)

The point-in-time core. One chronological pass; emits each fighter's PRE-fight career stats for every rated fight, then updates state with that fight.

**Files:** Create `src/mma/history.py`, `tests/test_history.py`.

Emitted columns (per fight_id × corner): `career_fights, career_wins, career_win_rate, career_finish_rate, kd_pf, sub_att_pf, td_landed_pf, td_acc, td_def, sig_pm, sig_absorbed_pm, ctrl_share, streak, days_since_last, last5_win_rate, last5_avg_opp_elo`.

Definitions: `*_pf` = career sum / career fights; `td_acc` = own landed/attempted takedowns (NaN if 0 attempts); `td_def` = 1 − opponents' landed/attempted against them (NaN if never attacked); `sig_pm`/`sig_absorbed_pm` = career sig strikes landed/absorbed per minute of cage time (NaN if no timed minutes); `ctrl_share` = own control seconds / total timed seconds; `streak` = +n consecutive wins / −n losses, draw resets to 0; `last5_*` over the previous ≤5 rated fights (NaN when 0 prior); `last5_avg_opp_elo` = mean opponent `pre_overall` over those fights. All NaN for debuts except `career_fights=0` and `streak=0`.

- [ ] **Step 1: Write `tests/test_history.py`:**

```python
import pandas as pd
import pytest

from mma.history import build_history


def _fights():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2"],
            "date": pd.to_datetime(["2020-01-01", "2020-03-01"]),
            "fighter_a_id": ["x", "x"],
            "fighter_b_id": ["y", "z"],
            "winner": ["a", "a"],
            "method": ["ko_tko", "decision"],
            "match_time_sec": [300.0, 900.0],
        }
    )


def _stats():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "kd": [2, 0, 0, 0],
            "sig_landed": [30, 10, 50, 40],
            "td_landed": [1, 0, 2, 1],
            "td_attempted": [2, 1, 4, 2],
            "sub_att": [0, 1, 1, 0],
            "ctrl_sec": [60.0, 30.0, 300.0, 100.0],
        }
    )


def _ratings():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "fighter_id": ["x", "y", "x", "z"],
            "pre_overall": [1500.0, 1500.0, 1532.0, 1500.0],
        }
    )


def test_debut_row_is_empty_history():
    history = build_history(_fights(), _stats(), _ratings())
    x_f1 = history[(history["fight_id"] == "f1") & (history["corner"] == "a")].iloc[0]
    assert x_f1["career_fights"] == 0
    assert x_f1["streak"] == 0
    assert pd.isna(x_f1["career_win_rate"])
    assert pd.isna(x_f1["days_since_last"])
    assert pd.isna(x_f1["last5_avg_opp_elo"])


def test_second_fight_reflects_first():
    history = build_history(_fights(), _stats(), _ratings())
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_fights"] == 1
    assert x_f2["career_win_rate"] == 1.0
    assert x_f2["career_finish_rate"] == 1.0  # won by ko
    assert x_f2["streak"] == 1
    assert x_f2["days_since_last"] == 60
    assert x_f2["kd_pf"] == 2.0
    assert x_f2["td_acc"] == pytest.approx(0.5)          # 1 of 2
    assert x_f2["td_def"] == pytest.approx(1.0)          # opponent 0 of 1
    assert x_f2["sig_pm"] == pytest.approx(30 / 5.0)     # 30 landed in 5 min
    assert x_f2["sig_absorbed_pm"] == pytest.approx(10 / 5.0)
    assert x_f2["ctrl_share"] == pytest.approx(60 / 300.0)
    assert x_f2["last5_win_rate"] == 1.0
    assert x_f2["last5_avg_opp_elo"] == 1500.0


def test_loss_and_draw_semantics():
    fights = _fights()
    fights.loc[0, "winner"] = "b"     # x loses f1
    fights.loc[1, "winner"] = "draw"  # then draws f2 (still emitted to history)
    history = build_history(fights, _stats(), _ratings())
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_win_rate"] == 0.0
    assert x_f2["streak"] == -1


def test_nc_ignored():
    fights = _fights()
    fights.loc[0, "winner"] = "nc"
    history = build_history(fights, _stats(), _ratings())
    assert "f1" not in set(history["fight_id"])
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_fights"] == 0
```

- [ ] **Step 2:** Run — FAILS (no module).
- [ ] **Step 3: Implement `src/mma/history.py`:**

```python
"""Chronological per-fighter career-stat accumulator.

Mirrors the Elo engine's single-pass design: for every rated fight it
emits each fighter's PRE-fight career/rolling stats, then folds the
fight into their state. Point-in-time correct by construction.

Missing per-fight stats accumulate as zero (mostly pre-2001 fights);
rates are NaN until a fighter has the relevant denominator.
"""
from __future__ import annotations

from collections import deque

import pandas as pd

_HISTORY_WINDOW = 5


class _FighterState:
    def __init__(self) -> None:
        self.fights = 0
        self.wins = 0.0
        self.finish_wins = 0
        self.kd = 0.0
        self.sub_att = 0.0
        self.td_landed = 0.0
        self.td_attempted = 0.0
        self.opp_td_landed = 0.0
        self.opp_td_attempted = 0.0
        self.sig_landed = 0.0
        self.sig_absorbed = 0.0
        self.ctrl_sec = 0.0
        self.time_sec = 0.0
        self.streak = 0
        self.last_date: pd.Timestamp | None = None
        self.recent_results: deque[float] = deque(maxlen=_HISTORY_WINDOW)
        self.recent_opp_elo: deque[float] = deque(maxlen=_HISTORY_WINDOW)

    def snapshot(self, date: pd.Timestamp) -> dict:
        def ratio(num, den):
            return num / den if den else None

        minutes = self.time_sec / 60.0
        return {
            "career_fights": self.fights,
            "career_wins": self.wins,
            "career_win_rate": ratio(self.wins, self.fights),
            "career_finish_rate": ratio(self.finish_wins, self.wins),
            "kd_pf": ratio(self.kd, self.fights),
            "sub_att_pf": ratio(self.sub_att, self.fights),
            "td_landed_pf": ratio(self.td_landed, self.fights),
            "td_acc": ratio(self.td_landed, self.td_attempted),
            "td_def": (
                1 - self.opp_td_landed / self.opp_td_attempted
                if self.opp_td_attempted
                else None
            ),
            "sig_pm": ratio(self.sig_landed, minutes),
            "sig_absorbed_pm": ratio(self.sig_absorbed, minutes),
            "ctrl_share": ratio(self.ctrl_sec, self.time_sec),
            "streak": self.streak,
            "days_since_last": (
                (date - self.last_date).days if self.last_date is not None else None
            ),
            "last5_win_rate": (
                sum(self.recent_results) / len(self.recent_results)
                if self.recent_results
                else None
            ),
            "last5_avg_opp_elo": (
                sum(self.recent_opp_elo) / len(self.recent_opp_elo)
                if self.recent_opp_elo
                else None
            ),
        }

    def update(self, score, own, opp, method, time_sec, date, opp_elo) -> None:
        def num(mapping, key):
            value = mapping.get(key)
            return 0.0 if value is None or pd.isna(value) else float(value)

        self.fights += 1
        self.wins += score
        if score == 1.0 and method in ("ko_tko", "submission"):
            self.finish_wins += 1
        self.kd += num(own, "kd")
        self.sub_att += num(own, "sub_att")
        self.td_landed += num(own, "td_landed")
        self.td_attempted += num(own, "td_attempted")
        self.opp_td_landed += num(opp, "td_landed")
        self.opp_td_attempted += num(opp, "td_attempted")
        self.sig_landed += num(own, "sig_landed")
        self.sig_absorbed += num(opp, "sig_landed")
        self.ctrl_sec += num(own, "ctrl_sec")
        if time_sec is not None and not pd.isna(time_sec):
            self.time_sec += float(time_sec)
        if score == 1.0:
            self.streak = self.streak + 1 if self.streak > 0 else 1
        elif score == 0.0:
            self.streak = self.streak - 1 if self.streak < 0 else -1
        else:
            self.streak = 0
        self.last_date = date
        self.recent_results.append(score)
        if opp_elo is not None and not pd.isna(opp_elo):
            self.recent_opp_elo.append(float(opp_elo))


_SCORES = {"a": (1.0, 0.0), "b": (0.0, 1.0), "draw": (0.5, 0.5)}


def build_history(
    fights: pd.DataFrame, stats: pd.DataFrame, ratings: pd.DataFrame
) -> pd.DataFrame:
    """One row per fighter per rated fight with PRE-fight career stats."""
    stat_lookup = stats.set_index(["fight_id", "corner"]).to_dict("index")
    elo_lookup = ratings.set_index(["fight_id", "corner"])["pre_overall"].to_dict()

    states: dict[str, _FighterState] = {}
    rows = []
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for fight in ordered.itertuples(index=False):
        if fight.winner not in _SCORES:
            continue
        score_a, score_b = _SCORES[fight.winner]
        stats_a = stat_lookup.get((fight.fight_id, "a"), {})
        stats_b = stat_lookup.get((fight.fight_id, "b"), {})
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            state = states.setdefault(fighter_id, _FighterState())
            row = {
                "fight_id": fight.fight_id,
                "corner": corner,
                "fighter_id": fighter_id,
            }
            row.update(state.snapshot(fight.date))
            rows.append(row)
        # update AFTER both snapshots so neither side sees this fight
        method = fight.method if pd.notna(fight.method) else None
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            states[fighter_id].update(
                score, own, opp, method, fight.match_time_sec, fight.date,
                elo_lookup.get((fight.fight_id, opp_corner)),
            )

    history = pd.DataFrame(rows)
    for column in ("fight_id", "corner", "fighter_id"):
        history[column] = history[column].astype("string")
    return history
```

- [ ] **Step 4:** Task tests + full suite (79 expected).
- [ ] **Step 5: Real-data smoke** (report verbatim): build history from the three parquet files + ratings; expect 16,498 rows, and spot-check a known veteran (e.g. the fighter with most fights) for monotonically increasing `career_fights`.
- [ ] **Step 6:** Commit: `"Add fighter history accumulator"`

---

### Task 4: Matchup assembler (`features.py`)

**Files:** Create `src/mma/features.py`, `tests/test_features.py`.

- [ ] **Step 1: Tests** — `tests/test_features.py`:

```python
import hashlib

import pandas as pd

from mma.features import build_features, swap_corner


def test_swap_is_deterministic_md5_parity():
    for fight_id in ("abc", "f1", "20170729-x-vs-y"):
        expected = int(hashlib.md5(fight_id.encode()).hexdigest(), 16) % 2 == 1
        assert swap_corner(fight_id) is expected


def _tables():
    fights = pd.DataFrame(
        {
            "fight_id": ["f1"],
            "date": pd.to_datetime(["2021-06-01"]),
            "fighter_a_id": ["x"],
            "fighter_b_id": ["y"],
            "winner": ["a"],
            "method": ["ko_tko"],
            "finish_round": pd.array([2], dtype="Int64"),
            "scheduled_rounds": pd.array([3], dtype="Int64"),
            "weight_class": ["Lightweight"],
            "title_fight": [False],
            "match_time_sec": [500.0],
        }
    )
    fighters = pd.DataFrame(
        {
            "fighter_id": ["x", "y"],
            "name": ["X", "Y"],
            "height_cm": [180.0, 175.0],
            "reach_cm": [183.0, None],
            "stance": ["Southpaw", "Orthodox"],
            "dob": pd.to_datetime(["1990-06-01", None]),
        }
    )
    ratings = pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "corner": ["a", "b"],
            "fighter_id": ["x", "y"],
            "pre_overall": [1550.0, 1500.0],
            "pre_striking": [1540.0, 1500.0],
            "pre_grappling": [1510.0, 1500.0],
            "pre_fights": [3, 0],
        }
    )
    history = pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "corner": ["a", "b"],
            "fighter_id": ["x", "y"],
            "career_fights": [3, 0],
            "career_wins": [2.0, 0.0],
            "career_win_rate": [2 / 3, None],
            "career_finish_rate": [0.5, None],
            "kd_pf": [0.3, None],
            "sub_att_pf": [0.7, None],
            "td_landed_pf": [1.0, None],
            "td_acc": [0.5, None],
            "td_def": [0.8, None],
            "sig_pm": [4.0, None],
            "sig_absorbed_pm": [3.0, None],
            "ctrl_share": [0.2, None],
            "streak": [2, 0],
            "days_since_last": [120.0, None],
            "last5_win_rate": [2 / 3, None],
            "last5_avg_opp_elo": [1510.0, None],
        }
    )
    return fights, fighters, ratings, history


def test_feature_row_shape_and_targets():
    features = build_features(*_tables())
    assert len(features) == 1
    row = features.iloc[0]
    swapped = row["swapped"]
    assert row["y_winner"] == (0 if swapped else 1)
    assert row["y_method"] == "ko_tko"
    assert row["y_finish_round"] == "2"
    assert row["weight_class"] == "Lightweight"
    assert row["scheduled_rounds"] == 3


def test_diffs_flip_sign_with_swap():
    features = build_features(*_tables()).iloc[0]
    sign = -1 if features["swapped"] else 1
    assert features["elo_diff"] == sign * 50.0
    assert features["height_diff"] == sign * 5.0
    assert features["career_fights_diff"] == sign * 3


def test_missing_flags_and_debut():
    features = build_features(*_tables()).iloc[0]
    # y has no reach and no dob; x has both
    if features["swapped"]:
        assert features["reach_missing_a"] and not features["reach_missing_b"]
        assert features["debut_a"] and not features["debut_b"]
    else:
        assert features["reach_missing_b"] and not features["reach_missing_a"]
        assert features["debut_b"] and not features["debut_a"]
    assert bool(features["debut_matchup"]) is True


def test_draws_and_nc_excluded():
    fights, fighters, ratings, history = _tables()
    fights.loc[0, "winner"] = "draw"
    assert len(build_features(fights, fighters, ratings, history)) == 0
```

- [ ] **Step 2:** Run — FAILS.
- [ ] **Step 3: Implement `src/mma/features.py`:**

```python
"""Assemble the model-ready feature table.

One row per decisive fight. Corners are deterministically swapped by
md5(fight_id) parity so column order cannot encode the winner (the red
corner wins ~65% of raw fights). Numeric features enter as A-minus-B
differentials plus a few absolutes; missing values stay NaN.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_HISTORY_FEATURES = [
    "career_fights", "career_wins", "career_win_rate", "career_finish_rate",
    "kd_pf", "sub_att_pf", "td_landed_pf", "td_acc", "td_def",
    "sig_pm", "sig_absorbed_pm", "ctrl_share", "streak", "days_since_last",
    "last5_win_rate", "last5_avg_opp_elo",
]
_ELO_FEATURES = ["pre_overall", "pre_striking", "pre_grappling", "pre_fights"]
_DIFF_RENAMES = {"pre_overall": "elo", "pre_striking": "striking_elo",
                 "pre_grappling": "grappling_elo", "pre_fights": "elo_fights"}


def swap_corner(fight_id: str) -> bool:
    """Deterministic, platform-stable coin flip per fight."""
    return int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 1


def _side_frame(fights, fighters, ratings, history, corner: str) -> pd.DataFrame:
    fighter_col = f"fighter_{corner}_id"
    side = fights[["fight_id", "date", fighter_col]].rename(
        columns={fighter_col: "fighter_id"}
    )
    side = side.merge(fighters, on="fighter_id", how="left")
    side = side.merge(
        ratings[ratings["corner"] == corner][["fight_id"] + _ELO_FEATURES],
        on="fight_id", how="left",
    )
    side = side.merge(
        history[history["corner"] == corner][["fight_id"] + _HISTORY_FEATURES],
        on="fight_id", how="left",
    )
    side["age"] = (side["date"] - side["dob"]).dt.days / 365.25
    side["reach_missing"] = side["reach_cm"].isna()
    side["dob_missing"] = side["dob"].isna()
    side["southpaw"] = (side["stance"] == "Southpaw").fillna(False)
    side["debut"] = side["career_fights"].fillna(0) == 0
    return side


def build_features(fights, fighters, ratings, history) -> pd.DataFrame:
    decisive = fights[fights["winner"].isin(["a", "b"])].reset_index(drop=True)
    side_a = _side_frame(decisive, fighters, ratings, history, "a")
    side_b = _side_frame(decisive, fighters, ratings, history, "b")

    swapped = decisive["fight_id"].map(swap_corner).to_numpy(dtype=bool)
    # positional row-swap: both frames share identical columns and index
    first = side_a.copy()
    second = side_b.copy()
    first.loc[swapped] = side_b.loc[swapped].values
    second.loc[swapped] = side_a.loc[swapped].values

    features = pd.DataFrame(
        {
            "fight_id": decisive["fight_id"],
            "date": decisive["date"],
            "swapped": swapped,
            "y_winner": np.where(
                swapped,
                (decisive["winner"] == "b").astype(int),
                (decisive["winner"] == "a").astype(int),
            ),
            "y_method": decisive["method"],
            "y_finish_round": decisive["finish_round"]
            .map(lambda r: "45" if pd.notna(r) and r >= 4 else (str(int(r)) if pd.notna(r) else None))
            .astype("string"),
            "weight_class": decisive["weight_class"].astype("string"),
            "title_fight": decisive["title_fight"],
            "scheduled_rounds": decisive["scheduled_rounds"],
        }
    )

    numeric = (
        {name: name for name in _HISTORY_FEATURES}
        | {name: _DIFF_RENAMES[name] for name in _ELO_FEATURES}
        | {"height_cm": "height", "reach_cm": "reach", "age": "age"}
    )
    for source, out in numeric.items():
        features[f"{out}_diff"] = pd.to_numeric(
            first[source], errors="coerce"
        ) - pd.to_numeric(second[source], errors="coerce")

    for side, frame in (("a", first), ("b", second)):
        features[f"age_{side}"] = frame["age"]
        features[f"career_fights_{side}"] = frame["career_fights"]
        features[f"reach_missing_{side}"] = frame["reach_missing"].astype(bool)
        features[f"dob_missing_{side}"] = frame["dob_missing"].astype(bool)
        features[f"southpaw_{side}"] = frame["southpaw"].astype(bool)
        features[f"debut_{side}"] = frame["debut"].astype(bool)
    features["debut_matchup"] = features["debut_a"] ^ features["debut_b"]
    features["stance_mismatch"] = features["southpaw_a"] ^ features["southpaw_b"]
    return features
```

- [ ] **Step 4:** Task tests + full suite (84 expected).
- [ ] **Step 5:** Commit: `"Add matchup feature assembler"`

---

### Task 5: Feature pipeline + leakage test

**Files:** Create `scripts/build_features.py`, `tests/test_processed_features.py`.

- [ ] **Step 1: `scripts/build_features.py`:**

```python
"""Build the model-ready feature table from processed parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mma.features import build_features
from mma.history import build_history

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")

    history = build_history(fights, stats, ratings)
    features = build_features(fights, fighters, ratings, history)
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    print(f"{len(features)} rows, {features.shape[1]} columns")
    print("y_winner balance:", features["y_winner"].mean().round(4))
    print("swapped share:", features["swapped"].mean().round(4))
    per_year = features.groupby(features["date"].dt.year).size()
    print("rows/year (last 6):")
    print(per_year.tail(6).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `tests/test_processed_features.py`:**

```python
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists(),
    reason="features not built (run scripts/build_features.py)",
)


def test_target_is_balanced_after_symmetrization():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    # raw red-corner win rate is ~0.646; symmetrization must land near 0.5
    assert 0.47 < features["y_winner"].mean() < 0.53


def test_row_count_matches_decisive_fights():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert len(features) == (fights["winner"].isin(["a", "b"])).sum()


def test_no_leakage_truncation_invariance():
    """Features for old fights must not change when future fights exist."""
    from mma.features import build_features
    from mma.history import build_history

    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")

    cutoff = "2015-01-01"
    old_fights = fights[fights["date"] < cutoff]
    truncated = build_features(
        old_fights, fighters, ratings, build_history(old_fights, stats, ratings)
    )
    full = pd.read_parquet(PROCESSED / "features.parquet")
    full_old = full[full["fight_id"].isin(truncated["fight_id"])]

    merged = truncated.sort_values("fight_id").reset_index(drop=True)
    full_sorted = full_old.sort_values("fight_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, full_sorted, check_like=True, check_dtype=False)


def test_finish_round_target_classes():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    finishes = features[features["y_method"].isin(["ko_tko", "submission"])]
    assert set(finishes["y_finish_round"].dropna()) <= {"1", "2", "3", "45"}
    assert finishes["y_finish_round"].notna().all()
```

- [ ] **Step 3:** Run pipeline; expect ~8,190 rows (decisive fights), y_winner mean ≈ 0.5, swapped ≈ 0.5. Full suite green (88 expected). If the truncation-invariance test fails, that's a REAL LEAK — investigate, do not weaken.
- [ ] **Step 4:** Commit incl. `data/processed/features.parquet`: `"Add feature pipeline with leakage test"`

---

### Task 6: XGBoost models + training script

**Files:** Create `src/mma/models/__init__.py` (empty), `src/mma/models/xgb.py`, `scripts/train_xgb.py`, `tests/test_xgb.py`. Add `"xgboost>=2.0",` to pyproject dependencies; `.venv/bin/pip install -e ".[dev]"`.

- [ ] **Step 1: `src/mma/models/xgb.py`:**

```python
"""XGBoost baselines: winner (binary), method (3-class), finish round (4-class)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

TARGETS = ("y_winner", "y_method", "y_finish_round")
NON_FEATURES = {"fight_id", "date", "swapped", *TARGETS}

BASE_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}
MAX_ROUNDS = 2000
EARLY_STOP = 50


def feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    x = features[[c for c in features.columns if c not in NON_FEATURES]].copy()
    x["weight_class"] = x["weight_class"].astype("category")
    for column in x.columns:
        if x[column].dtype == "bool" or str(x[column].dtype) == "boolean":
            x[column] = x[column].astype(int)
        elif str(x[column].dtype) in ("Int64", "Float64"):
            x[column] = x[column].astype(float)
    return x


def train_binary(x_train, y_train, x_val, y_val) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        **BASE_PARAMS,
        n_estimators=MAX_ROUNDS,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP,
        enable_categorical=True,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def train_multiclass(x_train, y_train, x_val, y_val, classes) -> xgb.XGBClassifier:
    mapping = {label: index for index, label in enumerate(classes)}
    model = xgb.XGBClassifier(
        **BASE_PARAMS,
        n_estimators=MAX_ROUNDS,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        early_stopping_rounds=EARLY_STOP,
        enable_categorical=True,
    )
    model.fit(
        x_train, y_train.map(mapping),
        eval_set=[(x_val, y_val.map(mapping))], verbose=False,
    )
    return model
```

- [ ] **Step 2: `tests/test_xgb.py`** (unit-level, tiny synthetic data — fast):

```python
import numpy as np
import pandas as pd

from mma.models.xgb import feature_frame, train_binary


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "elo_diff": rng.normal(0, 100, n),
            "age_diff": rng.normal(0, 5, n),
            "weight_class": pd.array(["Lightweight"] * n, dtype="string"),
            "title_fight": [False] * n,
        }
    )
    y = (x["elo_diff"] + rng.normal(0, 50, n) > 0).astype(int)
    return x, y


def test_feature_frame_drops_ids_and_targets():
    features = pd.DataFrame(
        {
            "fight_id": ["f"], "date": [pd.Timestamp("2020-01-01")],
            "swapped": [True], "y_winner": [1], "y_method": ["decision"],
            "y_finish_round": [None], "elo_diff": [10.0],
            "weight_class": pd.array(["Lightweight"], dtype="string"),
        }
    )
    x = feature_frame(features)
    assert list(x.columns) == ["elo_diff", "weight_class"]
    assert str(x["weight_class"].dtype) == "category"


def test_binary_model_learns_signal():
    x, y = _synthetic()
    xf = feature_frame(pd.concat([x], axis=1).assign(fight_id="f", date=pd.Timestamp("2020-01-01"), swapped=False, y_winner=0, y_method=None, y_finish_round=None))
    model = train_binary(xf[:300], y[:300], xf[300:], y[300:])
    p = model.predict_proba(xf[300:])[:, 1]
    assert ((p >= 0.5).astype(int) == y[300:]).mean() > 0.7
```

- [ ] **Step 3: `scripts/train_xgb.py`:**

```python
"""Train XGBoost baselines; report validation-years metrics only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mma.evaluate import accuracy, brier_score, log_loss, macro_f1
from mma.models.xgb import feature_frame, train_binary, train_multiclass

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
TRAIN_END = "2021-01-01"
VAL_START, VAL_END = "2021-01-01", "2023-12-31"
METHOD_CLASSES = ["ko_tko", "submission", "decision"]
ROUND_CLASSES = ["1", "2", "3", "45"]


def main() -> None:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    x = feature_frame(features)
    train = features["date"] < TRAIN_END
    val = (features["date"] >= VAL_START) & (features["date"] <= VAL_END)
    MODELS.mkdir(exist_ok=True)
    metrics = {}

    # winner
    y = features["y_winner"]
    winner = train_binary(x[train], y[train], x[val], y[val])
    p_val = winner.predict_proba(x[val])[:, 1]
    metrics["winner"] = {
        "n_val": int(val.sum()),
        "accuracy": round(accuracy(y[val], p_val), 4),
        "log_loss": round(log_loss(y[val], p_val), 4),
        "brier": round(brier_score(y[val], p_val), 4),
        "best_iteration": int(winner.best_iteration),
    }
    winner.save_model(MODELS / "xgb_winner.json")

    # method (rows with known method)
    known = features["y_method"].notna()
    y = features["y_method"]
    method = train_multiclass(
        x[train & known], y[train & known], x[val & known], y[val & known],
        METHOD_CLASSES,
    )
    pred = [METHOD_CLASSES[i] for i in method.predict(x[val & known])]
    truth = list(y[val & known])
    majority = y[train & known].mode()[0]
    metrics["method"] = {
        "n_val": int((val & known).sum()),
        "accuracy": round(float(sum(p == t for p, t in zip(pred, truth)) / len(truth)), 4),
        "macro_f1": round(macro_f1(truth, pred), 4),
        "majority_baseline_accuracy": round(float(sum(t == majority for t in truth) / len(truth)), 4),
    }
    method.save_model(MODELS / "xgb_method.json")

    # finish round (finishes only)
    finish = features["y_finish_round"].notna()
    y = features["y_finish_round"]
    rounds = train_multiclass(
        x[train & finish], y[train & finish], x[val & finish], y[val & finish],
        ROUND_CLASSES,
    )
    pred = [ROUND_CLASSES[i] for i in rounds.predict(x[val & finish])]
    truth = list(y[val & finish])
    majority = y[train & finish].mode()[0]
    metrics["finish_round"] = {
        "n_val": int((val & finish).sum()),
        "accuracy": round(float(sum(p == t for p, t in zip(pred, truth)) / len(truth)), 4),
        "macro_f1": round(macro_f1(truth, pred), 4),
        "majority_baseline_accuracy": round(float(sum(t == majority for t in truth) / len(truth)), 4),
    }
    rounds.save_model(MODELS / "xgb_round.json")

    (MODELS / "xgb_metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    importances = pd.Series(
        winner.feature_importances_, index=x.columns
    ).sort_values(ascending=False)
    print("\ntop 15 winner-model features:")
    print(importances.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** Install dep, run unit tests, then `.venv/bin/python scripts/train_xgb.py`. Capture full output. Sanity: winner val accuracy expected 0.58–0.66 and MUST beat the committed Elo baseline log-loss (0.6777) to justify the feature work; if accuracy > 0.72, suspect leakage — STOP and report. Method accuracy should beat majority baseline; round accuracy will be close to its majority baseline (R1 dominates) — that's expected, macro-F1 is the honest lens.
- [ ] **Step 5:** Full suite green (90 expected). Commit `models/*.json`, code, pyproject: `"Add XGBoost baselines with validation metrics"`

---

### Task 7: README results update

- [ ] Add XGBoost rows to the README results table (winner accuracy/log-loss/brier next to Elo), a short "what predicts fights" note from the top feature importances, and method/finish-round validation numbers with their majority baselines. Keep the "2024+ held out" line. Commit: `"Report XGBoost baseline results in README"`

---

## Done criteria (Phase 3)

- Suite green (~90 tests) including the truncation-invariance leakage test and symmetrization balance test.
- `features.parquet`, three `models/xgb_*.json`, `xgb_metrics_val.json` committed and reproducible.
- XGBoost winner model beats Elo baseline on validation log-loss; README updated honestly.
- No code path reads 2024+ outcomes for training, tuning, or reporting.
