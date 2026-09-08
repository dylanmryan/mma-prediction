# SP2: Data Expansion and Features v3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add new point-in-time signal to the feature table in independently evaluated blocks, keep only the blocks that clear the SP1 bar, and deploy the union of what cleared — without breaking serving parity or point-in-time discipline.

**Architecture:** A feature *block* is a named group of columns produced by one builder and switchable by name. `src/mma/feature_blocks.py` owns the registry; `src/mma/history.py` gains the per-fighter accumulators; `src/mma/features.py` assembles the table from the enabled blocks; `src/mma/serving.py` (new) becomes the single source of truth for turning a pair of fighter states into a feature row, used by both the training-table builder and `build_matchup`, so a new feature cannot exist in training but be missing at prediction time. Each block is judged by `scripts/run_walkforward.py` against the committed baselines using `mma.walkforward.bar_check`.

**Tech Stack:** Python 3.11, pandas, numpy, xgboost, torch (CPU), pytest. Local-disk venv `~/.venvs/mma`; every command is `OMP_NUM_THREADS=1 ~/.venvs/mma/bin/{python,pytest}` from the repo root. A full XGB walk-forward is ~15 s, torch ~55 s, the suite ~5 s.

**Commit convention:** end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. `git status --short` before each commit; stage exactly the listed files.

---

## Locked evaluation rules for this sub-project

From the v3 spec §4 SP1/§5 and `models/walkforward/noise_floor.json`:

- **Incumbent reports** (committed, do not regenerate): `models/walkforward/xgb_v1.json` (pooled winner LL 0.6537), `models/walkforward/torch_v1.json` (0.6510). σ_seed = 0.000346, so the bar is **0.003**.
- **A block ships** iff, with the block enabled on top of everything already cleared, pooled winner log-loss improves by more than 0.003 **and** no fold year worsens by more than 0.01 — i.e. `bar_check(candidate, incumbent, sigma_seed)["ships"] is True`.
- **Screening vs deciding.** Screen each block with XGBoost (fast). A block that clears on XGB is then run on torch; the **torch** result decides, because torch is the deployed scorer. A block that fails on XGB by more than 0.002 is dropped without a torch run (record it).
- **Fresh-seed re-scoring.** The final shipped feature set is re-run on torch with seeds 5–9 and that number is reported (as SP1 did for the refit recipe).
- **Negative results are deliverables.** Every block gets a row in the results table in this plan's Completion notes, cleared or not, with its pooled numbers. Code for a block that does not ship is reverted (the elo-v1.1 / model-v2 precedent), but its measurement stays.
- **Point-in-time.** Every new column must pass `tests/test_processed_features.py::test_no_leakage_truncation_invariance`, which rebuilds the table from pre-2015-truncated fights and compares every column. External joins must filter on a date strictly before the fight.
- **Missingness.** Every externally-sourced column carries a `*_missing` boolean, and the row-level `external_missing` flag feeds the SP1 slice of the same name. *(Amended during Task 11, which measured the reason: when membership of the external source is itself determined by the future — as it is for `ehan03/jds-mma-data`'s fighter mapping, whose composition tracks how long a fighter's UFC career turned out to be — a **per-corner** missingness flag is a look-ahead feature and must be dropped. The row-level flag is symmetric in the corners and stays. See the Task 11 completion note.)*

## Block evaluation procedure (invoked by every block task)

Given a block name `B` already implemented and registered:

1. Rebuild the table with the block enabled on top of the cleared set:
   `OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py --blocks <cleared+B>` (writes `data/processed/features.parquet`).
2. Screen on XGB: `OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_<B>` .
3. Compare: `OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/block_decision.py --candidate models/walkforward/xgb_<B>.json --incumbent models/walkforward/xgb_v1.json` (prints the `bar_check` block and writes nothing).
4. If XGB pooled LL is worse than the incumbent by more than 0.002 → the block is rejected; record the numbers, revert the block's code (keep the report), move on.
   Otherwise run torch: `--candidate torch --name torch_<B>`, then `block_decision.py --candidate models/walkforward/torch_<B>.json --incumbent models/walkforward/torch_v1.json`.
5. The block ships iff the torch `bar_check` says `ships: true`. Append the row to the results table; commit the report(s) either way.
6. Restore the table to the cleared set before starting the next block:
   `scripts/build_features.py --blocks <cleared>`.

**Incumbent bookkeeping:** once a block ships, the "incumbent" for subsequent blocks becomes that block's report (`xgb_<B>.json` / `torch_<B>.json`), not `*_v1.json`. Record which report is the current incumbent in the results table after every block.

---

## File map

| Path | Change | Responsibility |
|---|---|---|
| `src/mma/serving.py` | create | `FighterState`-to-feature-row assembly shared by training and inference |
| `src/mma/feature_blocks.py` | create | block registry: name → (accumulator fields, differential/absolute column specs) |
| `src/mma/history.py` | extend | new per-fighter accumulators, one section per block |
| `src/mma/features.py` | modify | assemble from enabled blocks via `serving.py` |
| `src/mma/inference.py` | modify | `build_matchup` delegates to `serving.py` |
| `src/mma/snapshots.py` | modify | carry the new accumulator fields |
| `src/mma/external.py` | create | load/join the derived external tables (snapshot, rankings, notice/weigh-in) |
| `scripts/build_features.py` | modify | `--blocks` argument; records the block set in the parquet's metadata sidecar |
| `scripts/block_decision.py` | create | thin CLI over `mma.walkforward.bar_check` for block decisions |
| `scripts/build_external.py` | create | download the open snapshot → derived per-fighter/per-fight parquet under `data/external/` |
| `src/mma/wiki_cards.py` | extend | parse withdrawals/replacements and missed weight from event Background prose |
| `scripts/refresh_secondary.py` | create | daily ufcstats scrape adapter filling the gap between Kaggle versions |
| `tests/test_serving_parity.py` | create | value-level parity between the training row and the served row |
| `tests/test_feature_blocks.py` | create | registry, block toggling, missingness flags |
| `tests/test_history.py`, `tests/test_features.py` | extend | per-block accumulator unit tests |
| `tests/test_external.py`, `tests/test_wiki_background.py`, `tests/test_refresh_secondary.py` | create | external loaders and parsers |
| `models/walkforward/{xgb,torch}_<block>.json` | create | one report per evaluated block |
| `data/external/*.parquet` | create | derived, committed (a few MB), regenerated by `build_external.py` |
| `README.md` | modify | feature documentation and the block results table |

---

### Task 1: Branch and baseline

**Files:** none (verification only)

- [ ] **Step 1: Branch from main and confirm the starting state**

```bash
git checkout main && git status --short && git checkout -b sp2-features
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python -c "import pandas as pd; f = pd.read_parquet('data/processed/features.parquet'); print(f.shape); print(sorted(f.columns))"
```
Expected: clean tree; `357 passed, 1 skipped`; 11,238 rows × 46 columns.

---

### Task 2: Serving parity — one feature-row builder for training and inference

The feature contract is currently written twice: `src/mma/features.py` builds the training table from `history`/`ratings`/`fighters`, and `src/mma/inference.py::build_matchup` re-derives the same columns by hand from snapshots (`inference.py:270-350`). Adding ~40 columns across eight blocks by editing both would drift silently. This task collapses them onto one builder and pins it with a value-level test, *before* any new feature exists.

**Files:**
- Create: `src/mma/serving.py`
- Create: `tests/test_serving_parity.py`
- Modify: `src/mma/features.py`, `src/mma/inference.py`

- [ ] **Step 1: Write the failing parity test**

`tests/test_serving_parity.py`:
```python
"""The row served for a future fight must equal the row trained on.

`build_features` (training) and `build_matchup` (serving) must produce
identical feature values for the same matchup as of the same date. This is
value-level, not just column-level: it replays history to just before a
real historical fight, builds the served row from the resulting snapshots,
and compares it against that fight's row in the training table.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma.features import build_features
from mma.history import build_history
from mma.inference import build_matchup
from mma.snapshots import build_snapshots

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists(),
    reason="processed data not built (run scripts/make_dataset.py)",
)


@pytest.fixture(scope="module")
def tables():
    return {
        name: pd.read_parquet(PROCESSED / f"{name}.parquet")
        for name in ("fights", "fight_stats", "fighters", "ratings")
    }


def _target_fight(fights: pd.DataFrame) -> pd.Series:
    """A 2024 fight where both fighters already have several UFC bouts."""
    counts = pd.concat([fights["fighter_a_id"], fights["fighter_b_id"]]).value_counts()
    veterans = set(counts[counts >= 5].index)
    candidates = fights[
        (fights["date"] >= "2024-01-01")
        & (fights["winner"].isin(["a", "b"]))
        & fights["fighter_a_id"].isin(veterans)
        & fights["fighter_b_id"].isin(veterans)
    ]
    assert len(candidates), "no suitable veteran-vs-veteran fight found"
    return candidates.sort_values("date").iloc[0]


def test_served_row_equals_training_row(tables):
    fights, stats, fighters, ratings = (
        tables["fights"], tables["fight_stats"], tables["fighters"], tables["ratings"]
    )
    target = _target_fight(fights)

    # Rebuild every table from fights strictly BEFORE the target fight, exactly
    # as the weekly prospective run sees the world on the morning of the event.
    past = fights[fights["date"] < target["date"]]
    past_ids = set(past["fight_id"])
    snapshots = build_snapshots(
        past, stats[stats["fight_id"].isin(past_ids)], ratings[ratings["fight_id"].isin(past_ids)]
    )
    served = build_matchup(
        snapshots.loc[target["fighter_a_id"]], snapshots.loc[target["fighter_b_id"]],
        fighters.set_index("fighter_id").loc[target["fighter_a_id"]],
        fighters.set_index("fighter_id").loc[target["fighter_b_id"]],
        target["weight_class"], bool(target["title_fight"]),
        int(target["scheduled_rounds"]), target["date"],
    )

    history = build_history(fights, stats, ratings)
    trained = build_features(fights, fighters, ratings, history)
    row = trained[trained["fight_id"] == target["fight_id"]]
    assert len(row) == 1
    row = row.iloc[0]

    # The training table may have swapped corners for this fight; the served
    # row is always A-vs-B, so compare against the unswapped orientation.
    sign = -1.0 if bool(row["swapped"]) else 1.0
    compared = 0
    for column in served.columns:
        if column in ("weight_class", "title_fight", "scheduled_rounds"):
            continue
        assert column in row.index, f"served column {column} missing from the training table"
        served_value = served.iloc[0][column]
        trained_value = row[column]
        if isinstance(served_value, (bool, np.bool_)) or column.endswith(("_a", "_b")):
            continue  # corner-labelled absolutes flip with the swap; covered below
        if pd.isna(served_value) and pd.isna(trained_value):
            continue
        assert float(served_value) == pytest.approx(sign * float(trained_value), abs=1e-6), column
        compared += 1
    assert compared >= 20, f"only {compared} differential columns compared"


def test_training_table_and_served_row_have_the_same_columns(tables):
    fights, stats, fighters, ratings = (
        tables["fights"], tables["fight_stats"], tables["fighters"], tables["ratings"]
    )
    target = _target_fight(fights)
    past = fights[fights["date"] < target["date"]]
    past_ids = set(past["fight_id"])
    snapshots = build_snapshots(
        past, stats[stats["fight_id"].isin(past_ids)], ratings[ratings["fight_id"].isin(past_ids)]
    )
    served = build_matchup(
        snapshots.loc[target["fighter_a_id"]], snapshots.loc[target["fighter_b_id"]],
        fighters.set_index("fighter_id").loc[target["fighter_a_id"]],
        fighters.set_index("fighter_id").loc[target["fighter_b_id"]],
        target["weight_class"], bool(target["title_fight"]),
        int(target["scheduled_rounds"]), target["date"],
    )
    trained = pd.read_parquet(PROCESSED / "features.parquet")
    identifiers = {"fight_id", "date", "swapped", "y_winner", "y_method", "y_finish_round"}
    assert set(served.columns) == set(trained.columns) - identifiers
```

- [ ] **Step 2: Run to see the current state**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_serving_parity.py -q
```
Expected: it may already pass or fail on a few columns. **Record the exact output in the Completion notes.** If a column mismatches, that is a pre-existing serving bug worth its own line in the notes — investigate before refactoring (the known intentional case is `career_fights` feeding both `elo_fights_diff` and `career_fights_diff`, documented at `inference.py:314-320`; a mismatch there is expected to be tiny, and if it is non-zero the test's tolerance must not be loosened without saying why).

- [ ] **Step 3: Extract the shared builder into `src/mma/serving.py`**

```python
"""The one place a (fighter A, fighter B, context) pair becomes a feature row.

Both paths use it:
  * training  — `mma.features.build_features` feeds it per-fight PRE-fight
    accumulator snapshots from `mma.history`;
  * serving   — `mma.inference.build_matchup` feeds it current-state
    snapshots from `mma.snapshots`.
Keeping one implementation is what stops a feature from existing in the
training table but silently missing (or differing) at prediction time; the
value-level guard is `tests/test_serving_parity.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (column-in-state, output stem) pairs entering as A-minus-B differentials.
DIFFERENTIALS: tuple[tuple[str, str], ...] = (
    ("career_fights", "career_fights"), ("career_wins", "career_wins"),
    ("career_win_rate", "career_win_rate"), ("career_finish_rate", "career_finish_rate"),
    ("kd_pf", "kd_pf"), ("sub_att_pf", "sub_att_pf"), ("td_landed_pf", "td_landed_pf"),
    ("td_acc", "td_acc"), ("td_def", "td_def"), ("sig_pm", "sig_pm"),
    ("sig_absorbed_pm", "sig_absorbed_pm"), ("ctrl_share", "ctrl_share"),
    ("streak", "streak"), ("days_since_last", "days_since_last"),
    ("last5_win_rate", "last5_win_rate"), ("last5_avg_opp_elo", "last5_avg_opp_elo"),
    ("pre_overall", "elo"), ("pre_striking", "striking_elo"),
    ("pre_grappling", "grappling_elo"), ("pre_fights", "elo_fights"),
    ("height_cm", "height"), ("reach_cm", "reach"), ("age", "age"),
)
# Absolutes emitted per corner as `<stem>_a` / `<stem>_b`.
ABSOLUTES: tuple[str, ...] = ("age", "career_fights")
BOOLEANS: tuple[str, ...] = ("reach_missing", "dob_missing", "southpaw", "debut")


def minus(a, b) -> float:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def feature_row(state_a: dict, state_b: dict, context: dict,
                differentials=DIFFERENTIALS, absolutes=ABSOLUTES,
                booleans=BOOLEANS) -> dict:
    """One feature row for A-vs-B. `state_*` are plain dicts of the values
    named in `differentials`/`absolutes`/`booleans`; `context` carries
    weight_class, title_fight, scheduled_rounds and any fight-level columns."""
    row = dict(context)
    for source, stem in differentials:
        row[f"{stem}_diff"] = minus(state_a.get(source), state_b.get(source))
    for stem in absolutes:
        row[f"{stem}_a"] = state_a.get(stem)
        row[f"{stem}_b"] = state_b.get(stem)
    for stem in booleans:
        row[f"{stem}_a"] = bool(state_a.get(stem))
        row[f"{stem}_b"] = bool(state_b.get(stem))
    row["debut_matchup"] = row["debut_a"] ^ row["debut_b"]
    row["stance_mismatch"] = row["southpaw_a"] ^ row["southpaw_b"]
    return row
```
Then rewrite `build_matchup` in `inference.py` so its `side()` helper produces the state dict and the row comes from `serving.feature_row`, and rewrite the numeric section of `features.py::build_features` to call `serving.feature_row` per fight (vectorised construction is fine as long as the column set and values come from the shared spec — if a row-wise loop over 11,238 fights is too slow, keep the vectorised implementation but derive the column lists from `serving.DIFFERENTIALS`/`ABSOLUTES`/`BOOLEANS` so the two cannot diverge, and say so in the docstring).

- [ ] **Step 4: Verify parity and no table change**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_serving_parity.py tests/test_features.py tests/test_inference.py tests/test_prospective.py tests/test_explain.py -q
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py | tail -3
git status --short data/processed
```
Expected: all pass; `features.parquet` **byte-identical** (no `git status` output). If the table changes, the refactor altered a value — stop and find out which column before continuing.

- [ ] **Step 5: Full suite and commit**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
git add src/mma/serving.py src/mma/features.py src/mma/inference.py tests/test_serving_parity.py
git status --short
git commit -m "$(cat <<'EOF'
One feature-row builder for training and serving, pinned by a parity test

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Feature-block registry and `--blocks`

**Files:**
- Create: `src/mma/feature_blocks.py`, `tests/test_feature_blocks.py`
- Modify: `src/mma/features.py`, `scripts/build_features.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_feature_blocks.py`:
```python
import pandas as pd
import pytest

from mma.feature_blocks import BASE_BLOCK, BLOCKS, columns_for, resolve_blocks


def test_base_block_is_always_present():
    assert resolve_blocks([]) == (BASE_BLOCK,)
    assert resolve_blocks([BASE_BLOCK]) == (BASE_BLOCK,)
    assert resolve_blocks(["in_fight"])[0] == BASE_BLOCK


def test_unknown_block_rejected():
    with pytest.raises(ValueError, match="unknown feature block"):
        resolve_blocks(["not_a_block"])


def test_blocks_are_ordered_deterministically():
    assert resolve_blocks(["context", "in_fight"]) == resolve_blocks(["in_fight", "context"])


def test_every_block_declares_columns_and_they_are_unique():
    seen = set()
    for name in BLOCKS:
        cols = columns_for([name])
        assert cols, f"block {name} declares no columns"
        overlap = seen & set(cols)
        assert not overlap, f"block {name} redeclares {overlap}"
        seen |= set(cols)


def test_columns_for_is_cumulative():
    base = set(columns_for([BASE_BLOCK]))
    with_block = set(columns_for([BASE_BLOCK, "in_fight"]))
    assert base < with_block
```
(Registry entries for blocks not yet implemented are added by their own task; at this point only `BASE_BLOCK` and a stub for `in_fight` need to exist — the test iterates whatever is registered.)

- [ ] **Step 2: Implement `src/mma/feature_blocks.py`**

```python
"""Feature blocks: named, independently evaluable groups of columns.

SP2 adds signal in blocks so each can be judged separately against the
walk-forward bar (see docs/superpowers/plans/2026-09-07-sp2-features-v3.md).
`BASE_BLOCK` is the v1 feature set and is always enabled. A block declares
the accumulator fields it needs and the output columns it produces; the
registry is what `scripts/build_features.py --blocks` and the serving path
both read, so a block cannot be half-enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASE_BLOCK = "base"


@dataclass(frozen=True)
class Block:
    name: str
    differentials: tuple[tuple[str, str], ...] = ()
    absolutes: tuple[str, ...] = ()
    booleans: tuple[str, ...] = ()
    fight_level: tuple[str, ...] = ()   # columns that describe the fight, not a corner
    external: bool = False              # contributes to the external_missing slice


BLOCKS: dict[str, Block] = {}


def register(block: Block) -> Block:
    if block.name in BLOCKS:
        raise ValueError(f"duplicate feature block {block.name}")
    BLOCKS[block.name] = block
    return block


def resolve_blocks(names) -> tuple[str, ...]:
    """Normalise a requested block list: base first, then registry order."""
    requested = set(names) | {BASE_BLOCK}
    unknown = sorted(requested - set(BLOCKS))
    if unknown:
        raise ValueError(f"unknown feature block(s): {unknown}; known: {sorted(BLOCKS)}")
    return tuple([BASE_BLOCK] + [n for n in BLOCKS if n != BASE_BLOCK and n in requested])


def columns_for(names) -> tuple[str, ...]:
    columns: list[str] = []
    for name in resolve_blocks(names):
        block = BLOCKS[name]
        columns += [f"{stem}_diff" for _, stem in block.differentials]
        columns += [f"{stem}_{side}" for stem in block.absolutes for side in ("a", "b")]
        columns += [f"{stem}_{side}" for stem in block.booleans for side in ("a", "b")]
        columns += list(block.fight_level)
    return tuple(columns)
```
Register `BASE_BLOCK` from the tuples now living in `serving.py` (move them here and have `serving.py` import them, so there is one definition), and have `serving.feature_row` take a block list instead of three tuples.

- [ ] **Step 3: `--blocks` in `scripts/build_features.py`**

Add `--blocks` (comma-separated, default `base`) and `--out` (default `data/processed/features.parquet`). Write a sidecar `data/processed/features_blocks.json` recording `{"blocks": [...], "n_rows": N, "n_columns": M}` so a report can be traced to the table it was computed on. (No timestamp: the sidecar is committed alongside the table and records nothing time-varying, which is how the byte-identity check on `features.parquet` stays meaningful.) Print the block list and column count.

- [ ] **Step 4: Verify the default is unchanged**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_feature_blocks.py -q
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py | tail -4
git status --short data/processed
```
Expected: tests pass; only `features_blocks.json` is new; `features.parquet` unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/mma/feature_blocks.py src/mma/serving.py src/mma/features.py scripts/build_features.py tests/test_feature_blocks.py data/processed/features_blocks.json
git status --short
git commit -m "$(cat <<'EOF'
Feature-block registry and --blocks selection

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `scripts/block_decision.py`

**Files:** Create `scripts/block_decision.py`, extend `tests/test_walkforward.py`

- [ ] **Step 1: Write the script**

```python
"""Apply the pre-registered bar to one block's walk-forward report.

    python scripts/block_decision.py --candidate models/walkforward/xgb_in_fight.json \
        --incumbent models/walkforward/xgb_v1.json

Prints the bar_check block (bar, delta, per-fold deltas, ships) and the
slice table so a regression concentrated in one slice is visible. Writes
nothing: the decision is recorded by a human in the SP2 plan's results
table, and the block's code is reverted when it does not ship.
"""
```
It reads `models/walkforward/noise_floor.json` for σ_seed (override with `--sigma`), calls `mma.walkforward.bar_check`, prints `json.dumps(..., indent=2)`, and additionally prints a two-column slice comparison (candidate vs incumbent winner LL for `debut`, `womens`, `five_round`, and `external_missing` when present). Exit code 0 always (it is a report, not a gate).

- [ ] **Step 2: Test the slice-comparison helper**

Factor the slice table as a pure function `slice_comparison(candidate, incumbent) -> list[dict]` and test it in `tests/test_walkforward.py` with two synthetic reports (one slice present in both, one only in the candidate → reported as `None` for the incumbent).

- [ ] **Step 3: Verify against the committed reports and commit**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/block_decision.py --candidate models/walkforward/torch_refit.json --incumbent models/walkforward/torch_v1.json
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_walkforward.py -q
git add scripts/block_decision.py tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
Block-decision CLI over the pre-registered walk-forward bar

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```
Expected: the refit-vs-v1 output reproduces `refit_decision.json`'s `bar_check_B_vs_A` (delta +0.0002, ships false).

---

### Task 5: Block `in_fight` — per-round and strike-target profile

The richest un-mined source: `data/processed/round_stats.parquet` (50,262 rows) plus the target/position columns already on `fight_stats`.

**Files:** Modify `src/mma/history.py`, `src/mma/feature_blocks.py`, `src/mma/snapshots.py`; extend `tests/test_history.py`; create the reports.

- [ ] **Step 1: Write the failing accumulator tests** (append to `tests/test_history.py`)

Cover, with a hand-built two-fight fixture whose numbers you can verify by hand:
- `head_share`, `body_share`, `leg_share` — career significant strikes landed to each target divided by total significant strikes landed; NaN before any landed strike.
- `distance_share`, `clinch_share`, `ground_share` — same over position.
- `kd_absorbed_pf` — knockdowns suffered per fight.
- `r1_output_share` — round-1 significant strikes landed divided by career significant strikes landed (needs `round_stats`).
- `late_round_fade` — mean over fights of (last-round sig landed − round-1 sig landed) for fights reaching round ≥ 2; NaN otherwise.
- `finish_rate_by_round` as three columns `finish_r1_rate`, `finish_r2_rate`, `finish_r3plus_rate` — share of career wins finished in that bucket.
- `been_finished_rate`, `ko_losses`, `sub_losses` — durability; `ko_loss_recency_days` (days since the most recent KO loss, NaN if none).
- `median_finish_sec` — median duration of the fighter's finish wins.

Each must be **pre-fight**: the accumulator updates after both corners are snapshotted, exactly as the existing fields do (`history.py:130-150`).

- [ ] **Step 2: Implement in `_FighterState`**

Extend `_FighterState.__init__`, `.snapshot()` and `.update()`. `build_history` gains a `round_stats` argument (default `None` so the base block still builds without it) and passes each fight's round rows to `update`. Register the block in `feature_blocks.py`:
```python
register(Block(
    name="in_fight",
    differentials=(
        ("head_share", "head_share"), ("body_share", "body_share"), ("leg_share", "leg_share"),
        ("distance_share", "distance_share"), ("clinch_share", "clinch_share"),
        ("ground_share", "ground_share"), ("kd_absorbed_pf", "kd_absorbed_pf"),
        ("r1_output_share", "r1_output_share"), ("late_round_fade", "late_round_fade"),
        ("finish_r1_rate", "finish_r1_rate"), ("finish_r2_rate", "finish_r2_rate"),
        ("finish_r3plus_rate", "finish_r3plus_rate"), ("been_finished_rate", "been_finished_rate"),
        ("ko_losses", "ko_losses"), ("sub_losses", "sub_losses"),
        ("ko_loss_recency_days", "ko_loss_recency_days"), ("median_finish_sec", "median_finish_sec"),
    ),
))
```
Mirror the new fields in `snapshots.py` so serving has them (the parity test from Task 2 will fail loudly if you forget).

- [ ] **Step 3: Leakage and parity gates**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py --blocks in_fight | tail -4
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_serving_parity.py tests/test_history.py -q
```
Expected: table gains 17 columns; truncation-invariance and parity both pass. **Both are blocking** — a failure here means the block is leaky or serving-incomplete, not that the test is wrong.

- [ ] **Step 4: Evaluate** — run the Block evaluation procedure with `B = in_fight`, cleared set = `base`.

- [ ] **Step 5: Record and commit**

Append the results row to the Completion notes. Commit the code and reports if it ships; if it does not, `git revert`/reset the code changes and commit only the reports plus the notes row:
```bash
git add -A && git status --short
git commit -m "$(cat <<'EOF'
Block in_fight: per-round and strike-target profile features

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Block `opponent_adjusted` — versus-expectation rates

A fighter's raw rates conflate skill with opposition. For each core rate, subtract what that fighter's opponents typically allowed *before* the fight.

**Files:** Modify `src/mma/history.py`, `src/mma/feature_blocks.py`, `src/mma/snapshots.py`; extend `tests/test_history.py`.

- [ ] **Step 1: Write the failing tests**

The accumulator needs two passes' worth of information but must stay single-pass and point-in-time. Implementation contract to test:
- Maintain, per fighter, running "allowed" aggregates: `sig_allowed_pm`, `td_allowed_pf`, `ctrl_allowed_share` (what opponents achieved against them).
- When fighter X fights opponent O, record for X the pair (X's own value this fight, O's pre-fight allowed value). The *feature* is the career mean of (own − opponent's allowed), i.e. output above expectation.
- Columns: `sig_pm_vs_exp`, `sig_absorbed_pm_vs_exp`, `td_landed_pf_vs_exp`, `td_def_vs_exp`, `ctrl_share_vs_exp`.
- Plus `avg_opp_elo_wins` and `avg_opp_elo_losses` (mean pre-fight Elo of beaten / losing opponents; NaN with no such fight).
Tests: a debutant has NaN everywhere; a fighter who outperforms a "leaky" opponent gets a positive `sig_pm_vs_exp`; the opponent's allowed value used is the one *before* the shared fight (construct three chronological fights and assert the exact number).

- [ ] **Step 2: Implement and register** the block with those seven differentials.

- [ ] **Step 3: Leakage and parity gates** (same two commands as Task 5, Step 3).

- [ ] **Step 4: Evaluate** — Block evaluation procedure, `B = opponent_adjusted`, cleared set = base + whatever cleared so far.

- [ ] **Step 5: Record and commit** (same shape as Task 5, Step 5).

---

### Task 7: Block `trajectory` — Glicko-2 and rating dynamics

Elo v1.1 rejected Glicko as a *replacement*; here it is tested as *additional* columns, together with rating dynamics that a level-only Elo cannot express.

**Files:** Create `src/mma/glicko.py`; modify `src/mma/history.py`, `src/mma/feature_blocks.py`, `src/mma/snapshots.py`, `scripts/build_ratings.py`; create `tests/test_glicko.py`.

- [ ] **Step 1: Write the failing Glicko-2 tests**

Implement Glicko-2 (Glickman's published algorithm: rating μ, deviation φ, volatility σ, system constant τ = 0.5, convergence ε = 1e-6) and test against the worked example in the official paper: a player rated 1500 with RD 200 facing three opponents (1400/30, 1550/100, 1700/300) with results W/L/L ends at rating ≈ 1464.06, RD ≈ 151.52. Assert to 2 decimals. Also test: RD grows with inactivity; a single upset moves a high-RD player more than a low-RD one.

- [ ] **Step 2: Add the ratings pass and the block**

`build_ratings.py` gains Glicko-2 columns to `ratings.parquet` (`pre_glicko_mu`, `pre_glicko_phi`, `pre_glicko_sigma`), computed in the same chronological pass as Elo (rating periods = one per event date). Block columns:
- `glicko_mu` (diff), `glicko_phi` (diff — uncertainty), `glicko_sigma` (diff)
- `elo_delta_3`, `elo_delta_5` — sum of the last 3 / 5 post-fight Elo changes (momentum)
- `elo_peak_minus_current` — career peak Elo minus current
- `years_since_ufc_debut`
- `age_squared` (absolute per corner) and `age_x_fights` (diff)
Tests: momentum is zero for a debutant, equals the sum of the last three deltas for a veteran; peak-minus-current is ≥ 0.

- [ ] **Step 3: Rebuild ratings, then gates**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_ratings.py | tail -6
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py --blocks trajectory | tail -4
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_processed_ratings.py tests/test_serving_parity.py tests/test_glicko.py -q
```
Note `ratings.parquet` changes here even if the block is rejected (new columns); that is acceptable — the Elo columns must be **byte-identical**, verify by comparing the pre-existing columns against `git show main:data/processed/ratings.parquet`.

- [ ] **Step 4: Evaluate** — Block evaluation procedure, `B = trajectory`.

- [ ] **Step 5: Record and commit.**

---

### Task 8: Block `context` — referee, home advantage, bonuses

**Files:** Modify `src/mma/history.py`, `src/mma/features.py`, `src/mma/feature_blocks.py`; create `src/mma/context.py`; extend tests.

Point-in-time care: referee and location are known before the fight (they are on the card), but the *statistics* about them must be computed from prior fights only.

- [ ] **Step 1: Write the failing tests**

- `referee_finish_rate`, `referee_decision_rate` (fight-level): the referee's rate over their prior fights only; NaN for a referee's first fight; `referee_missing` flag. A chronological three-fight fixture pins the exact values.
- `home_country_a` / `home_country_b` (boolean): fighter nationality equals the event country. Nationality comes from the external snapshot (Task 11) — until that block exists, derive the event country from `fights.location` (last comma-separated component, normalised) and leave the nationality side NaN with `nationality_missing`. If the external block is rejected later, this feature degrades to "unknown" for everyone and must be dropped from the block; note that dependency in the block docstring.
- `bonus_rate` (diff): career performance-bonus wins per fight, from `bonuses.parquet`, prior fights only.

- [ ] **Step 2: Implement**, registering `context` with `fight_level=("referee_finish_rate", "referee_decision_rate", "referee_missing")` and the rest as differentials/booleans.

- [ ] **Step 3: Gates** (leakage + parity, as before). The referee columns are fight-level, so `serving.feature_row` must accept them through `context` — and `build_matchup`'s caller (`prospective.predict_event`) must supply the referee when known and `None` otherwise; make the missing path explicit and tested, because Wikipedia cards do not carry referees, so **at prediction time these are always missing**. That asymmetry is a real risk: a feature that is always present in training and always missing in serving will hurt. State in the block docstring that `context` ships only if it clears the bar *and* a re-run with the referee columns forced missing on the evaluation rows does not lose more than σ_seed — run that ablation as part of Step 4 and record both numbers.

- [ ] **Step 4: Evaluate** — Block evaluation procedure plus the forced-missing ablation described above.

- [ ] **Step 5: Record and commit.**

---

### Task 9: Block `recency` — training window and sample weights

Not new columns: the SP1 harness capability, searched as a block.

**Files:** none in `src/`; reports only, plus the Completion notes.

- [x] **Step 1: Grid over training window and half-life**

```bash
export OMP_NUM_THREADS=1
for start in "" 2000-01-01 2005-01-01 2010-01-01; do
  for hl in "" 8 4 2; do
    name="xgb_rec_${start:-none}_${hl:-inf}"
    ~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name "$name" \
      ${start:+--train-start $start} ${hl:+--half-life $hl} | tail -2
  done
done
```
16 XGB runs ≈ 4 minutes. Record every pooled number in the Completion notes as a 4×4 table.

- [x] **Step 2: Carry the best two configurations to torch**

Run the two best XGB cells on torch (`--candidate torch --name torch_rec_<...>`), then `block_decision.py` against `torch_v1.json`.

- [x] **Step 3: Temperature/budget from recent folds** (the SP1 follow-up)

Per `models/walkforward/refit_decision.json`'s notes, per-fold optimal temperature drifts from ≈1.4 (2018–2022) to ≈0.96 (2023–2025) while the deployed recipe uses 1.1. Derive a budget and temperature from the **last four folds only** and score that refit variant:
```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python - <<'PY'
import json, numpy as np
from pathlib import Path
r = json.loads(Path("models/walkforward/torch_v1.json").read_text())
years = r["fold_years"][-4:]
idx = [r["fold_years"].index(y) for y in years]
ep = [float(np.median(r["fit_info"]["best_epoch"][i])) for i in idx]
tp = [float(np.median(r["fit_info"]["temperature"][i])) for i in idx]
print("recent-fold budget:", int(np.median(ep)) + 1, "temperature:", round(float(np.median(tp)), 2))
PY
```
Then run a torch walk-forward with those values (add a `--fixed-epochs` / `--temperature` pass-through to `run_walkforward.py` if `--fixed-budget-from` cannot express it — a small CLI addition, tested) and compare against `torch_refit.json` with `block_decision.py`. This decides whether SP4 should deploy the recent-fold temperature instead of 1.1.

- [x] **Step 4: Record and commit** the reports and the notes; if a recency configuration clears the bar, record it as a **deployment setting** (it changes how the train scripts are invoked, not the feature table) and update the train-script defaults in Task 13.

---

### Task 10: Secondary daily data source

**Files:** Create `scripts/refresh_secondary.py`, `tests/test_refresh_secondary.py`; modify `.github/workflows/refresh-data.yml`.

Goal: fill the gap between irregular Kaggle refreshes using the daily-refreshed ufcstats scrape (`Greco1899/scrape_ufc_stats`, same ufcstats ids, GPL-3.0 — used as a *data* source, not vendored code, and cited in the README).

- [ ] **Step 1: Write the failing tests** for a pure `merge_secondary(primary_fights, secondary_fights) -> (merged, report)`: rows whose `fight_id` is already in primary are dropped (primary wins); genuinely new rows are appended and counted; rows with ids absent from the fighters table are rejected with a named error; the report carries `n_added`, `max_date_before`, `max_date_after`.

- [ ] **Step 2: Implement** the fetch (HTTP GET of the repo's raw CSVs, no auth), the schema adaptation to our `fights`/`fight_stats` shape, and the merge. Gate the whole thing behind `--enable`; default off, so a fetch failure can never break the weekly Action.

- [ ] **Step 3: Verify on real data**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/refresh_secondary.py --enable --dry-run | tail -10
```
Expected: reports how many fights the secondary source has beyond 2026-08-08 (today's Kaggle cutoff). Record the number.

- [ ] **Step 4: Wire into the Action** as an optional step that runs *after* the Kaggle refresh and *before* `make_dataset.py`, with `continue-on-error: true`, then commit.

---

### Task 11: Block `external` — pre-UFC career, gym, nationality

**Files:** Create `scripts/build_external.py`, `src/mma/external.py`, `tests/test_external.py`; modify `src/mma/features.py`, `feature_blocks.py`; create `data/external/*.parquet`.

Source: `ehan03/jds-mma-data` (MIT), specifically `data/clean/fighter_mapping.csv` (ufcstats id ↔ other site ids), Sherdog `fighter_histories.csv` / `fighters.csv`, Tapology `fighters.csv` / `fighter_gyms.csv`. Coverage ends ≈ 2024-08; every column carries a missingness flag and feeds the `external_missing` slice.

- [ ] **Step 1: `scripts/build_external.py`** downloads the snapshot (git clone or raw file fetch into the scratch dir — never commit the raw 80 MB) and derives, keyed by ufcstats `fighter_id`:
`pre_ufc_wins`, `pre_ufc_losses`, `pre_ufc_finish_rate`, `pre_ufc_finish_loss_rate`, `days_since_pro_debut` (as of a given date, so store the debut date), `pre_ufc_avg_opp_wins`, `nationality`, `gym_id`. Write `data/external/fighter_external.parquet` (a few MB) plus a `data/external/README.md` recording source, licence, snapshot commit, and coverage dates.
- [ ] **Step 2: `src/mma/external.py`** joins it to a fight row **strictly by fighter id** (never by name), computing `days_since_pro_debut` relative to the fight date and emitting `external_missing` per corner and a row-level `external_missing`. Tests: unmatched fighter → all NaN plus the flag; a fighter whose UFC debut precedes their recorded pro debut → flagged and NaN (data error, not negative days).
- [ ] **Step 3: Gates.** Leakage: pre-UFC career values are as-of-snapshot and therefore constant — that is safe *only* because they describe the period before the fighter's UFC career; assert in a test that no external column changes when the fights table is truncated (this is exactly what the truncation-invariance test checks). Parity: serving must join the same table.
- [ ] **Step 4: Evaluate**, and additionally report the `external_missing` slice explicitly — a block that helps overall but hurts the post-snapshot debutant slice materially is recorded as such.
- [ ] **Step 5: Record and commit** (including the derived parquet and its README).

---

### Task 12: Blocks `notice` and `rankings`

Both are smaller and share the "join an external table by id, flag missingness" shape of Task 11.

- [ ] **Step 1: `notice`** — `late_replacements.csv` / `missed_weights.csv` from the same snapshot (through 2024-08) give `notice_days` (binned: ≤7 / ≤30 / full camp / unknown, as one-hot `short_notice_7`, `short_notice_30`, `notice_unknown`) and `missed_weight_lbs` (+ `missed_weight` boolean). Extend `src/mma/wiki_cards.py` with a `parse_background(html) -> {withdrawals: [...], missed_weight: [...]}` parser for the templated prose ("X was expected to face Y ... replaced by Z", "weighed in at N pounds, M pounds over") to extend coverage past 2024-08 and to serve future fights; unit-test it against three saved fixture pages under `tests/fixtures/wikipedia/` (add them). **Serving asymmetry check**, as in Task 8: score the block with these columns forced missing on the evaluation rows and record both numbers, since the Wikipedia parser will not always find them.
- [ ] **Step 2: `rankings`** — weekly official rankings history (2013→now; CC0). Fighter matching is by name, so reuse the prospective matcher's never-guess policy: exact then accent-folded, and anything else is unmatched (flagged). Columns: `rank` (diff, unranked = NaN + flag), `is_champion_a/b`, `is_ranked_a/b`, `rank_missing`, plus `ranking_regime_post_2026_06` (fight-level flag for the Elo-based ranking switch).
- [ ] **Step 3:** For each, run the Block evaluation procedure; record; commit.

---

### Task 13: Assemble, redeploy, document, merge

- [ ] **Step 1: Build the shipped table**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/build_features.py --blocks <comma-separated cleared blocks> | tail -4
~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_serving_parity.py -q
```

- [ ] **Step 2: Fresh-seed re-score of the shipped set**

```bash
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v3_seeds5 --seeds 5,6,7,8,9
~/.venvs/mma/bin/python scripts/block_decision.py --candidate models/walkforward/torch_v3_seeds5.json --incumbent models/walkforward/torch_v1_seeds5.json
```
The fresh-seed delta is the number reported publicly. If it does not clear the bar, the shipped set is reduced to the blocks that survive fresh seeds — say so in the notes.

- [ ] **Step 3: Re-derive the deployment budget and redeploy**

The budget in the train-script defaults was derived from the v1 feature table; a new table needs a new budget:
```bash
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_v3
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v3
~/.venvs/mma/bin/python scripts/refit_decision.py --out models/walkforward/refit_decision_v3.json
```
Update the train scripts' default `BUDGET`/`TEMPERATURE`/`REPORT` from the v3 reports (and the recency setting if Task 9 cleared one), retrain (`train_xgb.py`, `train_torch.py`, `build_display_priors.py`), record the new model hash, and confirm `Ensemble.load()` works and the app boots headless.

- [ ] **Step 4: Documentation**

README: a "Features" section listing the shipped blocks and what each contributes, the block results table (including rejected blocks — the honest-negative-results story this project already tells twice), the new pooled numbers, and the data-source credits with licences. Plan Completion notes: the full results table, the fresh-seed number, the new hash, and the follow-ups. Spec: mark SP2 done and note anything deferred to SP3/SP4.

- [ ] **Step 5: Verify and merge**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -p no:cacheprovider 2>&1 | tail -1
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/make_dataset.py | tail -2 && git status --short data/processed
git log --oneline main..sp2-features
git checkout main && git merge --no-ff sp2-features -m "$(cat <<'EOF'
Merge sp2-features: evaluated feature blocks, external data, secondary source

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Completion notes (filled in during execution)

**Serving parity (Task 2):** _pre-existing mismatches found: …_

**Block results** (pooled winner log-loss; incumbent in parentheses):

| Block | XGB | torch | Δ vs incumbent | Ships? | Notes |
|---|---|---|---|---|---|
| in_fight | 0.6540 (0.6537) | 0.6528 (0.6510) | +0.0018 vs torch_v1 | **no** | worse on both scorers; torch slices: womens +0.0062, debut +0.0034, five_round −0.0035. Incumbent stays `{xgb,torch}_v1`. |
| opponent_adjusted | 0.6519 (0.6537) | 0.6532 (0.6510) | +0.0022 vs torch_v1 | **no** | XGB liked it (-0.0018), torch did not; every fold worse on torch. Slices: debut +0.0030, womens +0.0020, five_round -0.0018. Coverage 0.63/0.51 of rows. Incumbent stays `{xgb,torch}_v1`. |
| trajectory | _…_ | _…_ | _…_ | _…_ | |
| context | _…_ | _…_ | _…_ | _…_ | forced-missing ablation: _…_ |
| recency | 0.6525 (0.6537) | 0.6499 (0.6510) | -0.0011 vs torch_v1 | **no** | best window/half-life: half-life 8y (window barely matters); improves on both scorers but by a third of the 0.003 bar. Incumbent stays `{xgb,torch}_v1`. |
| external | 0.6499 (0.6537) | 0.6472 (0.6510) | -0.0038 vs torch_v1 | **yes** | leak-free form only (fight-level missingness, no per-corner flag). external_missing slice: n=779, LL 0.6452 (vs pooled 0.6472); no incumbent slice exists to diff against. Coverage 0.799 of rows. New incumbent: `{xgb,torch}_external_noleak`. |
| notice | _…_ | _…_ | _…_ | _…_ | forced-missing ablation: _…_ |
| rankings | _…_ | _…_ | _…_ | _…_ | match rate: _…_ |

**Block `in_fight` (Task 5), measured and rejected.** 17 differentials from the
per-round table and the target/position columns: `head/body/leg_share`,
`distance/clinch/ground_share`, `kd_absorbed_pf`, `r1_output_share`,
`late_round_fade`, `finish_r1/r2/r3plus_rate`, `been_finished_rate`,
`ko_losses`, `sub_losses`, `ko_loss_recency_days`, `median_finish_sec`.

- XGB screen: pooled winner LL **0.6540** vs incumbent 0.6537 (delta **+0.0003**,
  i.e. very slightly worse). Inside the 0.002 drop threshold, so torch was run.
- torch decision: pooled winner LL **0.6528** vs incumbent 0.6510
  (delta **+0.0018**, worse). `bar_check` -> `clears_delta: false`,
  `no_fold_regression: true`, **`ships: false`**. Worst fold +0.0041 (2018);
  2021 and 2022 improved (-0.0031, -0.0020) but no fold cleared the bar.
- torch slices vs `torch_v1`: womens +0.0062, debut +0.0034, five_round -0.0035.
  The regression is concentrated in the womens slice, where the strike-profile
  columns are thinnest.
- Coverage of the new columns (11,238 rows): the six share columns and
  `r1_output_share` are populated on 7,997 rows, `late_round_fade` 6,930,
  the three finish-rate columns 6,959, `ko_loss_recency_days` only 1,873
  (both corners must have been KO'd). That much missingness on top of 17
  correlated columns is the most likely reason it costs rather than pays.
- Gates before evaluation, both passing on the in_fight table:
  `test_no_leakage_truncation_invariance` and both serving-parity tests.
- Reports kept: `models/walkforward/xgb_in_fight.json`,
  `models/walkforward/torch_in_fight.json`. Block code reverted.

**Block `opponent_adjusted` (Task 6), measured and rejected.** Seven
differentials that price a fighter's rates against the opposition they were
produced against: `sig_pm_vs_exp`, `sig_absorbed_pm_vs_exp`,
`td_landed_pf_vs_exp`, `td_def_vs_exp`, `ctrl_share_vs_exp` (career mean of
own-minus-the-opponent's-pre-fight-allowed for each rate), plus
`avg_opp_elo_wins` / `avg_opp_elo_losses` (mean pre-fight Elo of the opponents
beaten and lost to).

- XGB screen: pooled winner LL **0.6519** vs incumbent 0.6537 (delta
  **-0.0018**, i.e. better, though short of the 0.003 bar). Five of eight folds
  improved; 2021 was the worst at +0.0086, so `no_fold_regression` held but
  `clears_delta` did not.
- torch decision: pooled winner LL **0.6532** vs incumbent 0.6510 (delta
  **+0.0022**, worse). `bar_check` -> `clears_delta: false`,
  `no_fold_regression: true`, **`ships: false`**. The regression is uniform
  rather than concentrated: seven of eight folds worsened (2019 flat at
  -0.0001), worst +0.0059 (2021).
- The scorers disagree in sign, which is the interesting part of this result.
  The seven columns are strong but heavily collinear with the base rates they
  are derived from (`sig_pm`, `td_landed_pf`, `td_def`, `ctrl_share` are all
  already in the table); a tree ensemble can pick the better-conditioned of a
  correlated pair per split, while the MLP has to spend capacity on all of
  them. Same lesson as `in_fight`: correlated additions cost the deployed
  scorer even when they help the screen.
- torch slices vs `torch_v1`: debut +0.0030, womens +0.0020,
  five_round -0.0018.
- Coverage of the new differential columns (11,238 rows, both corners needed):
  `sig_pm_vs_exp`, `sig_absorbed_pm_vs_exp`, `td_landed_pf_vs_exp` and
  `ctrl_share_vs_exp` on 7,046 rows (0.627); `avg_opp_elo_wins` 6,931 (0.617);
  `td_def_vs_exp` 5,775 (0.514, both fighters need a prior fight in which the
  opponent shot a takedown); `avg_opp_elo_losses` 5,573 (0.496, both fighters
  need a prior loss). Better than `in_fight`'s thinnest columns but still about
  a third to a half of the table missing.
- Gates before evaluation, both passing on the opponent_adjusted table:
  `test_no_leakage_truncation_invariance` and both serving-parity tests. The
  base-only rebuild stayed byte-identical.
- Reports kept: `models/walkforward/xgb_opponent_adjusted.json`,
  `models/walkforward/torch_opponent_adjusted.json`. Block code reverted.
- Kept from this task regardless of the decision: the two table-level gates now
  build with the block set recorded in `features_blocks.json`
  (`tests/conftest.py::table_blocks`) instead of hard-coding `base`. Without
  it, `test_no_leakage_truncation_invariance` compares a base-only rebuild
  against a block-enabled table and fails on shape, and the serving-parity
  test never exercises a new block's snapshot fields at all -- so the two
  "blocking gates" could not actually gate a block.

**Block `recency` (Task 9), measured and rejected.** No feature columns: the
SP1 harness's `--train-start` / `--half-life` capability searched as a block.
The feature table stayed base-only (11,238 x 46) throughout.

16 XGB runs, pooled winner log-loss (rows = training-window start, columns =
recency half-life in years):

| train_start \ half-life | inf | 8 | 4 | 2 |
|---|---|---|---|---|
| none (1993-) | **0.6537** | **0.6525** | 0.6541 | 0.6540 |
| 2000-01-01 | 0.6535 | **0.6525** | 0.6531 | 0.6560 |
| 2005-01-01 | 0.6534 | **0.6525** | 0.6527 | 0.6529 |
| 2010-01-01 | 0.6539 | 0.6542 | 0.6556 | 0.6544 |

- Harness sanity check: the `none`/`inf` cell reproduced `xgb_v1.json`'s
  0.6537 exactly, so the grid is comparable to the incumbent reports.
- The signal is the half-life, not the window. An 8-year half-life is the best
  cell in three of the four rows; truncating hard (2010) is the worst row of
  the grid, at or above baseline in every column -- it throws away 2,741 of
  the 9,710 training rows available to the 2025 fold for no gain (the 2005
  window drops 885 and is roughly neutral). Half-lives
  shorter than 8y overshoot: 2y is worse than baseline in three rows.
- Best cells: a three-way tie at 0.6525 (`none`/8, `2000-01-01`/8,
  `2005-01-01`/8) at the reports' 4-dp storage precision. Tie broken on mean
  per-fold log-loss (2005/8 0.65426 < none/8 0.65461 < 2000/8 0.65470), so
  `2005-01-01`/8 and `none`/8 were carried to torch.
- torch decision, `torch_rec_2005-01-01_8`: pooled **0.6499** vs incumbent
  0.6510 (delta **-0.0011**, better). `bar_check` -> `clears_delta: false`,
  `no_fold_regression: true`, **`ships: false`**. Best folds 2025 -0.0052 and
  2024 -0.0034; worst 2019 +0.0032. Slices: debut -0.0022, womens -0.0019,
  five_round +0.0015.
- torch decision, `torch_rec_none_8`: pooled **0.6501** (delta **-0.0009**),
  same verdict; worst fold 2019 +0.0023. Slices: debut -0.0027,
  womens +0.0007, five_round 0.0000.
- Both configurations agree in sign and in shape -- down-weighting old fights
  helps, and helps most on the newest folds, which is the era-drift story the
  block was testing -- but the effect is about a third of the pre-registered
  0.003 bar and roughly 3 sigma_seed, so **no recency setting is adopted**.
  Task 13's train scripts keep uniform weights over the full history.
- Reports kept: the 16 `models/walkforward/xgb_rec_*.json` plus
  `torch_rec_2005-01-01_8.json` and `torch_rec_none_8.json`. Nothing in
  `src/` changed, so there is no block code to revert.

**Recent-fold temperature experiment (Task 9 Step 3, the SP1 follow-up).**
`refit_decision.json` recorded that the per-fold optimal temperature drifts
from ~1.3-1.65 (2018-2022) to ~0.9-1.0 (2023-2025) while the deployed refit
recipe applies the all-fold median 1.1, and that the refit variant lost to the
incumbent protocol on exactly the recent folds. Derived from `torch_v1.json`'s
last four folds (2022-2025): per-fold median epochs [19, 25, 10, 16] ->
**budget 18**; per-fold median temperatures [1.34, 0.96, 1.00, 0.90] ->
**temperature 0.98** (against the deployed all-fold 14 / 1.1).

- Scored as `torch_refit_recent` (pooled **0.6514**) through the new
  `--fixed-epochs` / `--temperature` pass-through on `run_walkforward.py`
  (mutually exclusive with `--fixed-budget-from`; `scripts/run_walkforward.py::resolve_budget`,
  covered by `tests/test_run_walkforward.py`).
- vs `torch_refit` (0.6512): delta **+0.0002**, worst fold +0.0029,
  **`ships: false`**. The fold split is the finding: 2018-2021 all worse
  (+0.0021 to +0.0029), 2022-2025 all better (-0.0022, -0.0005, -0.0014,
  -0.0018). The recent-fold recipe pays exactly where the drift predicted and
  costs where it did not.
- vs `torch_v1` (0.6510): delta **+0.0004**, worst fold +0.0037 (2019),
  **`ships: false`**.
- Decision for SP4: **keep the deployed 14 / 1.1**. The headline metric pools
  all eight folds, and on that metric the recent-fold recipe is a hair worse;
  the ~0.0015 mean gain over the four newest folds is real in sign but the
  same order as seed noise, and adopting it would mean overriding the locked
  pooled rule on four folds of evidence. Follow-up if this is revisited: the
  honest test is a recency-weighted headline metric (or more fold years), not
  a tighter read of the same eight numbers.

**Block `external` (Task 11), measured and SHIPPED in a corrected form; the
as-specified form was measured and rejected as leaky.** Six differentials over
pre-UFC career from the MIT `ehan03/jds-mma-data` snapshot (commit
`ec77f537`, UFC coverage to 2024-12-14), plus two fight-level flags:
`pre_ufc_wins`, `pre_ufc_losses`, `pre_ufc_finish_rate`,
`pre_ufc_finish_loss_rate`, `pre_ufc_avg_opp_wins`, `days_since_pro_debut`,
`external_missing` (row level) and `same_country`. This is the first block
whose information is not a recombination of the fight table.

*Source paths actually found* (the plan's guesses were close but not exact):
`data/clean/fighter_mapping.csv` (2,553 rows, not ~3,500),
`data/clean/Sherdog/fighter_histories.csv` (652k bouts, all promotions, from
1980), `data/clean/Sherdog/fighters.csv` (nationality plus a `pro_debut_date`
column that agrees with min(history date) for 100% of matched fighters), and
`data/clean/Tapology/fighter_gyms.csv` (which does exist, keyed by
(fighter, bout)). `pre_ufc_avg_opp_wins` was kept: the histories table carries
both sides of every bout, so an opponent's wins as of the bout date are
directly countable, and the opponent is present for 99.98% of pre-UFC bouts.

*Gym data is DATED, and is still not used.* `fighter_gyms.csv` is per
(fighter, bout), 97.7% of its rows map to a ufcstats bout, and 451 of 2,253
fighters change gym over their career -- so it is a genuine affiliation
history, not an as-of-scrape snapshot, and a gym win-rate would not be leaky.
It is excluded for a different reason: it is accumulated per-fight state rather
than fighter-static data, and the snapshot's UFC coverage stops at 2024-12-14,
so the affiliation is unknown for every future fight the deployed model
actually serves. `gym_id` (the gym at the fighter's earliest UFC bout) is
recorded in the derived table for a later block with a live source.

*Coverage.* 2,553 of our 4,581 fighters are in the mapping (0.5573); 8,976 of
the 11,238 rows have both corners matched (0.7987). Per column over the 11,238
rows: `pre_ufc_wins_diff`, `pre_ufc_losses_diff` and `days_since_pro_debut_diff`
0.799; `pre_ufc_avg_opp_wins_diff` 0.732; `pre_ufc_finish_rate_diff` 0.729;
`pre_ufc_finish_loss_rate_diff` only 0.302 (both corners need a pre-UFC loss).
`external_missing` is 0.201 of rows overall but 0.359 of 2025 and 0.534 of 2026
-- the snapshot ageing, which is exactly what that slice is for.

**The leak, and how it was found.** The block as specified in this plan carries
`external_missing` per corner as well as at fight level. Measured that way it
looked spectacular -- XGB 0.6475 (delta -0.0062), torch **0.6367** (delta
**-0.0143**), debut slice -0.0846 -- and it was wrong. Two diagnostics:

- Where exactly one corner is unmapped (n=1,445), that corner loses 76% of the
  time; in the debut slice (n=752) it loses 90% of the time. Meanwhile the
  actual differentials carry almost nothing on their own: with
  `pre_ufc_wins_diff > +2`, P(A wins) = 0.475, and with `< -2`, 0.519 -- if
  anything backwards.
- Membership of the source's cross-source fighter mapping is a function of how
  long a fighter's UFC career turned out to be. Among fighters debuting
  2013-2022 (well inside coverage), the mapped share is 0.241 for those with
  one UFC bout ever, 0.813 at two, 0.940 at three, 0.989 at four-to-five and
  1.000 at eleven or more; mean UFC bouts 7.41 mapped vs 1.25 unmapped.

So `external_missing_a`/`_b` is a look-ahead feature meaning "this fighter went
on to have a career". The walk-forward signature confirms it: folds 2018-2023
improved by -0.026 to -0.046 while **2025 cost +0.0487**, because in 2025
missingness stops meaning "short career" and starts meaning "debuted after the
snapshot". `bar_check` -> `clears_delta: true`, `no_fold_regression: false`,
**`ships: false`**. Reports kept: `xgb_external.json`, `torch_external.json`.

**The corrected block.** Missingness is fight-level only -- the symmetric
either-corner OR, which keeps the `external_missing` slice that
`walkforward.slice_masks` reads but cannot say which corner wins. The
differentials are NaN whenever either corner is unmapped, so no per-corner
channel survives. Guarded by
`tests/test_external.py::test_missingness_is_fight_level_only_never_per_corner`.

- XGB screen, `xgb_external_noleak`: pooled **0.6499** vs 0.6537 (delta
  **-0.0038**), worst fold +0.0031 (2021), `ships: true`.
- torch decision, `torch_external_noleak`: pooled **0.6472** vs 0.6510 (delta
  **-0.0038**), fold deltas 2018 -0.0077, 2019 -0.0073, 2020 -0.0069,
  2021 -0.0059, 2022 -0.0026, 2023 -0.0050, 2024 +0.0009, 2025 +0.0004;
  worst fold +0.0009. `clears_delta: true`, `no_fold_regression: true`,
  **`ships: true`**. ECE also improves, 0.0121 vs 0.0139.
- torch slices vs `torch_v1`: five_round -0.0088, debut -0.0015,
  womens -0.0011. Nothing regresses.
- **`external_missing` slice**: n=779, candidate LL **0.6452**, accuracy 0.6367
  -- marginally better than the candidate's own pooled 0.6472, so the block
  does not hurt the post-snapshot rows. A slice *delta* is not computable:
  `torch_v1.json` predates the column and the locked rules forbid regenerating
  incumbent reports. The fold deltas carry the same information more honestly:
  the two folds with the most missing rows (2024, 2025) are the two flat ones
  (+0.0009, +0.0004), while every well-covered fold improves. The contrast with
  the leaky variant is stark -- there the same slice read 0.5842 with accuracy
  0.6906, which was the leak showing.
- **Ablation, `{xgb,torch}_external_diffsonly`**: the six differentials with
  both fight-level flags withheld. XGB 0.6506 (delta -0.0031), torch **0.6476**
  (delta **-0.0034**), worst fold +0.0015, `ships: true`. So essentially all of
  the -0.0038 is the pre-UFC data itself; the symmetric flags add ~-0.0004.
  This is the check that the corrected block is not just a subtler encoding of
  the same selection effect.
- Gates on the shipped table, all passing:
  `test_no_leakage_truncation_invariance`, both serving-parity tests, and the
  base-only rebuild stayed byte-identical.
- **Caveat for SP4:** the benefit decays as the snapshot ages. It is already
  ~0 on 2024-2025 and `external_missing` reaches 0.534 of 2026 rows. The
  snapshot is static (last commit December 2025); if this block is to keep
  paying, `scripts/build_external.py` needs a refreshable source, or the
  pre-UFC record needs to come from a live scrape.

**Incumbent after this block: `models/walkforward/xgb_external_noleak.json`
(0.6499) and `models/walkforward/torch_external_noleak.json` (0.6472).** The
cleared feature set is now `base,external`.

**Secondary source (Task 10):** _fights available beyond the Kaggle cutoff: …_
**Shipped set:** _…_ ; fresh-seed re-score: _…_ ; deployed model hash: _…_
**Follow-ups:** _…_
