# Phase 5: Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An interactive matchup app (free Streamlit Community Cloud): pick two fighters → calibrated ensemble win probability with MC-dropout uncertainty, method breakdown, fight-duration distribution, tale of the tape — loading only committed artifacts (no training at runtime).

**Architecture:** `src/mma/snapshots.py` (each fighter's CURRENT career state, reusing the history accumulator semantics) → `src/mma/inference.py` (ensemble loader + matchup feature builder + BatchNorm-safe MC dropout) → `app.py` (UI) + `requirements.txt`. Tested headlessly via `streamlit.testing.v1.AppTest`.

**Branch:** `phase-5-streamlit`. Machine quirks: quote the repo path (space); OMP_NUM_THREADS=1 for any process importing torch+xgboost; first imports slow — background + poll, never conclude hung before 10 min; git index.lock → wait 15s retry ≤3; "classifier temporarily unavailable" shell errors → wait ~60s, retry.

**Inference contracts (from Phase 4 final review — binding):**
- Checkpoints `models/torch/net_seed{0..4}.pt` are dicts `{state_dict, temperature, n_features, n_weight_classes}`; load with `torch.load(path, weights_only=False)`; rebuild `MultiTaskNet(n_features=..., n_weight_classes=...)`.
- Headline probability = mean over seeds of `sigmoid(winner_logit / T_seed)`; uncertainty = per-seed spread + MC dropout.
- **MC dropout must NOT call `net.train()`** (mutates BatchNorm running stats). Instead: `net.eval()` then set ONLY `nn.Dropout` modules to train mode. A test must prove BatchNorm buffers are unchanged after MC sampling.
- Round head: apply `MultiTaskNet.round_probs(logits, three_round)` — it does not self-mask.
- The app must reproduce the exact 35 `preprocess.json:numeric_columns` (plus `weight_class`) via `Preprocessor.load(...).transform(...)`. NaN numerics impute to train medians automatically; unknown weight class → embedding 0.
- "As-of" date for age/days_since_last = max fight date in the data (document in the UI footer).

**Design decisions:**
- Snapshots computed by running the SAME chronological pass as `history.py` but emitting the FINAL state per fighter (post their last rated fight), plus current Elo trio from the ratings table (last `post_*` per fighter) and bio from fighters table.
- Matchup rows are built A-vs-B with no corner swap (training symmetrization makes the model order-agnostic; we verify: P(A beats B) + P(B beats A) ≈ 1 in a test).
- Method/round display composes: P(A by KO) = P(A) · P(ko_tko), etc. (independence assumption documented).
- `requirements.txt` for Streamlit Cloud: `-e .` + `streamlit`. Local dep added as optional group `app = ["streamlit>=1.35"]`.
- App caches all artifact loading with `@st.cache_resource`.

---

### Task 1: Current-state snapshots (`snapshots.py`)

**Files:** Create `src/mma/snapshots.py`, `tests/test_snapshots.py`.

- [ ] **Step 1: Tests** — `tests/test_snapshots.py`:

```python
import pandas as pd

from mma.snapshots import build_snapshots


def _fights():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2"],
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "fighter_a_id": ["x", "x"],
            "fighter_b_id": ["y", "z"],
            "winner": ["a", "b"],
            "method": ["ko_tko", "decision"],
            "match_time_sec": [300.0, 900.0],
        }
    )


def _stats():
    rows = []
    for fid, (sa, sb) in (("f1", (30, 10)), ("f2", (50, 40))):
        rows.append({"fight_id": fid, "corner": "a", "sig_landed": sa,
                     "td_landed": 1, "td_attempted": 2, "sub_att": 0, "ctrl_sec": 60.0, "kd": 0})
        rows.append({"fight_id": fid, "corner": "b", "sig_landed": sb,
                     "td_landed": 0, "td_attempted": 1, "sub_att": 1, "ctrl_sec": 30.0, "kd": 0})
    return pd.DataFrame(rows)


def _ratings():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "fighter_id": ["x", "y", "x", "z"],
            "pre_overall": [1500.0, 1500.0, 1520.0, 1500.0],
            "post_overall": [1520.0, 1480.0, 1502.0, 1518.0],
            "pre_striking": [1500.0] * 4,
            "post_striking": [1510.0, 1490.0, 1495.0, 1515.0],
            "pre_grappling": [1500.0] * 4,
            "post_grappling": [1505.0, 1495.0, 1500.0, 1510.0],
            "pre_fights": [0, 0, 1, 0],
        }
    )


def test_snapshot_reflects_full_career():
    snapshots = build_snapshots(_fights(), _stats(), _ratings())
    x = snapshots.loc["x"]
    assert x["career_fights"] == 2
    assert x["career_wins"] == 1.0
    assert x["streak"] == -1            # won f1, lost f2
    assert x["elo_overall"] == 1502.0   # last post_overall
    assert x["last_date"] == pd.Timestamp("2024-06-01")


def test_one_fight_fighters_present():
    snapshots = build_snapshots(_fights(), _stats(), _ratings())
    assert snapshots.loc["y"]["career_fights"] == 1
    assert snapshots.loc["z"]["elo_overall"] == 1518.0


def test_snapshot_history_columns_match_feature_names():
    from mma.features import _HISTORY_FEATURES
    snapshots = build_snapshots(_fights(), _stats(), _ratings())
    missing = set(_HISTORY_FEATURES) - {"days_since_last"} - set(snapshots.columns)
    assert missing == set()
```

- [ ] **Step 2:** RED run. **Step 3: Implement `src/mma/snapshots.py`:**

```python
"""Each fighter's CURRENT career state (after their last rated fight).

Reuses the history accumulator: replay all rated fights chronologically,
then snapshot every fighter's final state. `days_since_last` is left to
the caller (needs an as-of date); `last_date` is provided instead.
"""
from __future__ import annotations

import pandas as pd

from mma.history import _SCORES, _FighterState


def build_snapshots(
    fights: pd.DataFrame, stats: pd.DataFrame, ratings: pd.DataFrame
) -> pd.DataFrame:
    stat_lookup = stats.set_index(["fight_id", "corner"]).to_dict("index")
    elo_lookup = ratings.set_index(["fight_id", "corner"])["pre_overall"].to_dict()

    states: dict[str, _FighterState] = {}
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for fight in ordered.itertuples(index=False):
        if fight.winner not in _SCORES:
            continue
        score_a, score_b = _SCORES[fight.winner]
        stats_a = stat_lookup.get((fight.fight_id, "a"), {})
        stats_b = stat_lookup.get((fight.fight_id, "b"), {})
        method = fight.method if pd.notna(fight.method) else None
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            states.setdefault(fighter_id, _FighterState()).update(
                score, own, opp, method, fight.match_time_sec, fight.date,
                elo_lookup.get((fight.fight_id, opp_corner)),
            )

    last_elo = (
        ratings.sort_values("date", kind="stable")
        .groupby("fighter_id")[["post_overall", "post_striking", "post_grappling"]]
        .last()
        .rename(columns={
            "post_overall": "elo_overall",
            "post_striking": "elo_striking",
            "post_grappling": "elo_grappling",
        })
        if "date" in ratings.columns
        else ratings.groupby("fighter_id")[["post_overall", "post_striking", "post_grappling"]]
        .last()
        .rename(columns={
            "post_overall": "elo_overall",
            "post_striking": "elo_striking",
            "post_grappling": "elo_grappling",
        })
    )

    rows = {}
    for fighter_id, state in states.items():
        snapshot = state.snapshot(state.last_date)  # days_since_last -> 0, ignored
        snapshot["last_date"] = state.last_date
        rows[fighter_id] = snapshot
    snapshots = pd.DataFrame.from_dict(rows, orient="index")
    snapshots.index.name = "fighter_id"
    snapshots = snapshots.drop(columns=["days_since_last"])
    return snapshots.join(last_elo, how="left")
```

(If `ratings` passed by tests lacks a `date` column, the fallback branch groups in row order — the real ratings.parquet has `date`; the production path sorts by it. Note: the last-Elo join relies on ratings being per-fight rows.)

- [ ] **Step 4:** Task tests + full suite (108 expected). **Step 5:** Commit: `"Add current-state fighter snapshots"`

---

### Task 2: Ensemble inference (`inference.py`)

**Files:** Create `src/mma/inference.py`, `tests/test_inference.py`.

- [ ] **Step 1: Implement `src/mma/inference.py`:**

```python
"""Load the committed ensemble and predict hypothetical matchups."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mma.models.net import MultiTaskNet
from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
from mma.tensors import Preprocessor

ROOT = Path(__file__).resolve().parents[2]


class Ensemble:
    def __init__(self, nets, temperatures, preprocessor):
        self.nets = nets
        self.temperatures = temperatures
        self.preprocessor = preprocessor

    @classmethod
    def load(cls, directory=ROOT / "models" / "torch") -> "Ensemble":
        directory = Path(directory)
        preprocessor = Preprocessor.load(directory / "preprocess.json")
        nets, temperatures = [], []
        for path in sorted(directory.glob("net_seed*.pt")):
            payload = torch.load(path, weights_only=False)
            net = MultiTaskNet(
                n_features=payload["n_features"],
                n_weight_classes=payload["n_weight_classes"],
            )
            net.load_state_dict(payload["state_dict"])
            net.eval()
            nets.append(net)
            temperatures.append(float(payload["temperature"]))
        if not nets:
            raise FileNotFoundError(f"no checkpoints in {directory}")
        return cls(nets, temperatures, preprocessor)

    @torch.no_grad()
    def predict(self, features: pd.DataFrame) -> dict:
        x, wc = self.preprocessor.transform(features)
        x_t, wc_t = torch.tensor(x), torch.tensor(wc)
        three_round = torch.tensor(
            (features["scheduled_rounds"].fillna(3) <= 3).to_numpy(dtype=bool)
        )
        winner_probs, method_probs, round_probs = [], [], []
        for net, temperature in zip(self.nets, self.temperatures):
            winner_logits, method_logits, round_logits = net(x_t, wc_t)
            winner_probs.append(torch.sigmoid(winner_logits / temperature).numpy())
            method_probs.append(torch.softmax(method_logits, dim=1).numpy())
            round_probs.append(
                MultiTaskNet.round_probs(round_logits, three_round).numpy()
            )
        winner = np.stack(winner_probs)
        return {
            "winner_prob": winner.mean(axis=0),
            "winner_spread": winner.max(axis=0) - winner.min(axis=0),
            "method_probs": np.mean(method_probs, axis=0),
            "round_probs": np.mean(round_probs, axis=0),
            "method_classes": METHOD_CLASSES,
            "round_classes": ROUND_CLASSES,
        }

    @torch.no_grad()
    def mc_dropout(self, features: pd.DataFrame, passes: int = 100, seed: int = 0):
        """Stochastic winner probabilities from seed-0 net, dropout-only train mode.

        BatchNorm stays in eval mode (running stats untouched) -- Phase 4
        review requirement.
        """
        net, temperature = self.nets[0], self.temperatures[0]
        x, wc = self.preprocessor.transform(features)
        x_t, wc_t = torch.tensor(x), torch.tensor(wc)
        for module in net.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        torch.manual_seed(seed)
        samples = []
        for _ in range(passes):
            winner_logits, _, _ = net(x_t, wc_t)
            samples.append(torch.sigmoid(winner_logits / temperature).numpy())
        net.eval()
        return np.stack(samples)


def build_matchup(
    snapshot_a: pd.Series, snapshot_b: pd.Series,
    bio_a: pd.Series, bio_b: pd.Series,
    weight_class: str, title_fight: bool, scheduled_rounds: int,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """One feature row matching the training feature contract (A vs B, no swap)."""
    def side(snapshot, bio):
        age = (
            (as_of - bio["dob"]).days / 365.25 if pd.notna(bio["dob"]) else np.nan
        )
        days = (
            (as_of - snapshot["last_date"]).days
            if pd.notna(snapshot.get("last_date"))
            else np.nan
        )
        return {
            "age": age,
            "height_cm": bio["height_cm"],
            "reach_cm": bio["reach_cm"],
            "southpaw": bio["stance"] == "Southpaw",
            "days_since_last": days,
            "pre_overall": snapshot["elo_overall"],
            "pre_striking": snapshot["elo_striking"],
            "pre_grappling": snapshot["elo_grappling"],
            "pre_fights": snapshot["career_fights"],
            **{
                name: snapshot.get(name)
                for name in (
                    "career_fights", "career_wins", "career_win_rate",
                    "career_finish_rate", "kd_pf", "sub_att_pf", "td_landed_pf",
                    "td_acc", "td_def", "sig_pm", "sig_absorbed_pm", "ctrl_share",
                    "streak", "last5_win_rate", "last5_avg_opp_elo",
                )
            },
        }

    first, second = side(snapshot_a, bio_a), side(snapshot_b, bio_b)
    row: dict = {
        "weight_class": weight_class,
        "title_fight": title_fight,
        "scheduled_rounds": scheduled_rounds,
    }
    diff_names = {
        "pre_overall": "elo", "pre_striking": "striking_elo",
        "pre_grappling": "grappling_elo", "pre_fights": "elo_fights",
        "height_cm": "height", "reach_cm": "reach", "age": "age",
    }
    history_names = (
        "career_fights", "career_wins", "career_win_rate", "career_finish_rate",
        "kd_pf", "sub_att_pf", "td_landed_pf", "td_acc", "td_def",
        "sig_pm", "sig_absorbed_pm", "ctrl_share", "streak", "days_since_last",
        "last5_win_rate", "last5_avg_opp_elo",
    )
    for name in history_names:
        row[f"{name}_diff"] = _minus(first.get(name), second.get(name))
    for source, out in diff_names.items():
        row[f"{out}_diff"] = _minus(first.get(source), second.get(source))
    for label, data in (("a", first), ("b", second)):
        row[f"age_{label}"] = data["age"]
        row[f"career_fights_{label}"] = data["career_fights"]
        row[f"southpaw_{label}"] = bool(data["southpaw"])
        row[f"debut_{label}"] = (data["career_fights"] or 0) == 0
    row["debut_matchup"] = row["debut_a"] ^ row["debut_b"]
    row["stance_mismatch"] = row["southpaw_a"] ^ row["southpaw_b"]
    frame = pd.DataFrame([row])
    frame["weight_class"] = frame["weight_class"].astype("string")
    return frame


def _minus(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)
```

- [ ] **Step 2: Tests** — `tests/test_inference.py` (uses the real committed artifacts; skip if absent):

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "torch" / "metrics_val.json").exists(),
    reason="ensemble artifacts not built",
)


@pytest.fixture(scope="module")
def ensemble():
    from mma.inference import Ensemble
    return Ensemble.load()


@pytest.fixture(scope="module")
def matchup(ensemble):
    from mma.inference import build_matchup
    snapshot = pd.Series(
        {
            "career_fights": 10, "career_wins": 8.0, "career_win_rate": 0.8,
            "career_finish_rate": 0.5, "kd_pf": 0.4, "sub_att_pf": 0.5,
            "td_landed_pf": 1.5, "td_acc": 0.5, "td_def": 0.7, "sig_pm": 4.5,
            "sig_absorbed_pm": 3.0, "ctrl_share": 0.2, "streak": 3,
            "last5_win_rate": 0.8, "last5_avg_opp_elo": 1550.0,
            "elo_overall": 1600.0, "elo_striking": 1580.0,
            "elo_grappling": 1570.0, "last_date": pd.Timestamp("2025-06-01"),
        }
    )
    weaker = snapshot.copy()
    weaker["elo_overall"] = 1450.0
    weaker["career_win_rate"] = 0.4
    weaker["career_wins"] = 4.0
    weaker["streak"] = -2
    bio = pd.Series(
        {"dob": pd.Timestamp("1993-01-01"), "height_cm": 180.0,
         "reach_cm": 185.0, "stance": "Orthodox"}
    )
    return build_matchup(
        snapshot, weaker, bio, bio, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    ), build_matchup(
        weaker, snapshot, bio, bio, "Lightweight", False, 3,
        as_of=pd.Timestamp("2025-09-06"),
    )


def test_feature_contract_complete(ensemble, matchup):
    frame, _ = matchup
    missing = set(ensemble.preprocessor.numeric_columns) - set(frame.columns)
    assert missing == set()


def test_stronger_fighter_favored_and_symmetric(ensemble, matchup):
    forward, reverse = matchup
    p_forward = ensemble.predict(forward)["winner_prob"][0]
    p_reverse = ensemble.predict(reverse)["winner_prob"][0]
    assert p_forward > 0.5
    assert p_forward + p_reverse == pytest.approx(1.0, abs=0.08)


def test_round_45_zero_for_three_round_fight(ensemble, matchup):
    frame, _ = matchup
    result = ensemble.predict(frame)
    assert result["round_probs"][0, 3] == pytest.approx(0.0, abs=1e-6)
    assert result["method_probs"][0].sum() == pytest.approx(1.0, abs=1e-5)


def test_mc_dropout_preserves_batchnorm(ensemble, matchup):
    frame, _ = matchup
    net = ensemble.nets[0]
    before = {
        name: buffer.clone()
        for name, buffer in net.named_buffers()
    }
    samples = ensemble.mc_dropout(frame, passes=25)
    assert samples.shape == (25, 1)
    assert samples.std() > 0.0
    for name, buffer in net.named_buffers():
        assert torch.equal(before[name], buffer), f"buffer {name} mutated"
    assert not net.training
```

- [ ] **Step 3:** RED (module missing) → implement → task tests + full suite (112 expected). **Step 4:** Commit: `"Add ensemble inference and matchup builder"`

---

### Task 3: The app + deployment scaffolding

**Files:** Create `app.py`, `requirements.txt`, `tests/test_app.py`; modify `pyproject.toml` (add `app = ["streamlit>=1.35"]` optional group), README.

- [ ] **Step 1:** Add the optional dep group and `.venv/bin/pip install -e ".[dev,app]"` (background). `requirements.txt`:

```
-e .
streamlit>=1.35
```

- [ ] **Step 2: `app.py`:**

```python
"""MMA matchup predictor -- Streamlit app over the committed ensemble."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mma.inference import Ensemble, build_matchup
from mma.snapshots import build_snapshots

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="MMA Fight Predictor", page_icon="🥊", layout="wide")


@st.cache_resource
def load_everything():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    snapshots = build_snapshots(fights, stats, ratings)
    ensemble = Ensemble.load()
    as_of = fights["date"].max()
    weight_classes = sorted(fights["weight_class"].dropna().unique().tolist())
    return fights, fighters.set_index("fighter_id"), ratings, snapshots, ensemble, as_of, weight_classes


fights, fighters, ratings, snapshots, ensemble, as_of, weight_classes = load_everything()

eligible = snapshots.join(fighters[["name"]], how="inner").sort_values("name")
names = eligible["name"].tolist()
by_name = {name: fighter_id for fighter_id, name in eligible["name"].items()}

st.title("🥊 MMA Fight Predictor")
st.caption(
    f"Elo → XGBoost → multi-task neural ensemble, honestly evaluated. "
    f"Fighter stats as of {as_of:%Y-%m-%d}."
)

col_a, col_b = st.columns(2)
with col_a:
    name_a = st.selectbox("Fighter A", names, index=None, placeholder="Pick fighter A")
with col_b:
    name_b = st.selectbox("Fighter B", names, index=None, placeholder="Pick fighter B")

context_cols = st.columns(3)
with context_cols[0]:
    weight_class = st.selectbox("Weight class", weight_classes, index=None)
with context_cols[1]:
    scheduled_rounds = st.radio("Rounds", [3, 5], horizontal=True)
with context_cols[2]:
    title_fight = st.checkbox("Title fight")

if name_a and name_b and name_a != name_b:
    id_a, id_b = by_name[name_a], by_name[name_b]
    snap_a, snap_b = snapshots.loc[id_a], snapshots.loc[id_b]
    bio_a, bio_b = fighters.loc[id_a], fighters.loc[id_b]

    tape = pd.DataFrame(
        {
            name_a: [
                f"{(as_of - bio_a['dob']).days / 365.25:.1f}" if pd.notna(bio_a["dob"]) else "?",
                f"{bio_a['height_cm']:.0f} cm" if pd.notna(bio_a["height_cm"]) else "?",
                f"{bio_a['reach_cm']:.0f} cm" if pd.notna(bio_a["reach_cm"]) else "?",
                f"{snap_a['career_wins']:.0f}-{snap_a['career_fights'] - snap_a['career_wins']:.0f}",
                f"{snap_a['elo_overall']:.0f}",
                f"{snap_a['elo_striking']:.0f} / {snap_a['elo_grappling']:.0f}",
            ],
            "": ["Age", "Height", "Reach", "UFC record", "Elo", "Striking / Grappling Elo"],
            name_b: [
                f"{(as_of - bio_b['dob']).days / 365.25:.1f}" if pd.notna(bio_b["dob"]) else "?",
                f"{bio_b['height_cm']:.0f} cm" if pd.notna(bio_b["height_cm"]) else "?",
                f"{bio_b['reach_cm']:.0f} cm" if pd.notna(bio_b["reach_cm"]) else "?",
                f"{snap_b['career_wins']:.0f}-{snap_b['career_fights'] - snap_b['career_wins']:.0f}",
                f"{snap_b['elo_overall']:.0f}",
                f"{snap_b['elo_striking']:.0f} / {snap_b['elo_grappling']:.0f}",
            ],
        }
    )
    st.table(tape.set_index(""))

    matchup = build_matchup(
        snap_a, snap_b, bio_a, bio_b,
        weight_class or "Lightweight", title_fight, scheduled_rounds, as_of,
    )
    result = ensemble.predict(matchup)
    p_a = float(result["winner_prob"][0])
    spread = float(result["winner_spread"][0])

    st.subheader("Prediction")
    st.progress(p_a, text=f"{name_a}: {p_a:.0%}  ·  {name_b}: {1 - p_a:.0%}")
    st.caption(f"5-seed ensemble; seeds range ±{spread / 2:.1%} around the mean.")

    samples = ensemble.mc_dropout(matchup, passes=100)[:, 0]
    histogram = np.histogram(samples, bins=20, range=(0.0, 1.0))[0]
    st.bar_chart(
        pd.DataFrame({"MC dropout samples": histogram},
                     index=[f"{edge / 20:.2f}" for edge in range(20)]),
        height=160,
    )

    method = result["method_probs"][0]
    rounds = result["round_probs"][0]
    labels = {"ko_tko": "KO/TKO", "submission": "Submission", "decision": "Decision"}
    outcome_rows = []
    for fighter, p_fighter in ((name_a, p_a), (name_b, 1 - p_a)):
        for index, method_name in enumerate(result["method_classes"]):
            outcome_rows.append(
                {
                    "Winner": fighter,
                    "Method": labels[method_name],
                    "Probability": f"{p_fighter * method[index]:.1%}",
                }
            )
    st.subheader("How it ends")
    st.dataframe(pd.DataFrame(outcome_rows), hide_index=True)
    round_labels = ["Round 1", "Round 2", "Round 3", "Rounds 4-5"]
    finish_share = 1 - method[result["method_classes"].index("decision")]
    st.caption(
        "If it doesn't go the distance ("
        + f"{finish_share:.0%} chance), the finish comes in: "
        + "  ·  ".join(
            f"{label} {p:.0%}" for label, p in zip(round_labels, rounds) if p > 0.001
        )
    )

    elo_a = ratings[ratings["fighter_id"] == id_a][["date", "post_overall"]]
    elo_b = ratings[ratings["fighter_id"] == id_b][["date", "post_overall"]]
    trajectory = pd.concat(
        [
            elo_a.assign(fighter=name_a),
            elo_b.assign(fighter=name_b),
        ]
    ).pivot_table(index="date", columns="fighter", values="post_overall")
    st.subheader("Elo trajectories")
    st.line_chart(trajectory, height=240)

st.divider()
st.caption(
    "Model card: multi-task net (winner/method/finish-round), 5-seed ensemble, "
    "temperature-calibrated. Validation (2021-2023): accuracy 0.601, log-loss 0.656, "
    "Brier 0.232 — vs XGBoost 0.601/0.658/0.233 and Elo 0.576/0.678/0.242. "
    "Test years (2024+) held out. Method and round probabilities assume "
    "independence from the winner. "
    "[Source](https://github.com/dylanmryan/mma-prediction)"
)
```

- [ ] **Step 3: `tests/test_app.py`** (headless smoke via AppTest):

```python
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "torch" / "metrics_val.json").exists(),
    reason="ensemble artifacts not built",
)


def test_app_boots_and_predicts():
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app_test.run()
    assert not app_test.exception
    names = app_test.selectbox[0].options
    assert len(names) > 1000
    app_test.selectbox[0].select(names[0])
    app_test.selectbox[1].select(names[1])
    app_test.run()
    assert not app_test.exception
    assert app_test.subheader[0].value == "Prediction"
```

- [ ] **Step 4:** Run the app test (background; first run slow). Full suite — 113 expected. Manually verify boot: `OMP_NUM_THREADS=1 .venv/bin/streamlit run app.py --server.headless true` briefly, then kill it (confirm "You can now view your Streamlit app" in output).
- [ ] **Step 5:** README: add an "Interactive app" section — one paragraph + run-locally command + Community Cloud deploy steps (push repo public → share.streamlit.io → New app → pick repo/branch/app.py). Commit everything: `"Add Streamlit matchup app"`

---

### Task 4: Final review + merge

- [ ] Opus final review (leakage N/A here; focus: inference correctness vs contracts, app UX sanity, BatchNorm preservation test, no training-time code paths in the app, README accuracy). Fix findings, merge to main, push.

## Done criteria (Phase 5)

- Suite green (~113) incl. BatchNorm-preservation and app smoke tests.
- `streamlit run app.py` works locally against committed artifacts only.
- README documents local run + free deployment steps (deployment itself needs the user's Streamlit account — flag to user at wrap-up).
