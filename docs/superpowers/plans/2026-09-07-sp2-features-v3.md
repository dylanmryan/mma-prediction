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
- **Missingness.** Every externally-sourced column carries a `*_missing` boolean, and the row-level `external_missing` flag feeds the SP1 slice of the same name. *(Amended during Task 11, which measured the reason: when membership of the external source is itself determined by the future — as it is for `ehan03/jds-mma-data`'s fighter mapping, whose composition tracks how long a fighter's UFC career turned out to be — a **per-corner** missingness flag is a look-ahead feature and must be dropped. The row-level flag stays in the TABLE, where it is what makes the `external_missing` slice reportable, but the shipped variant keeps it out of both model matrices too. The property to check is not the flag's name: over the rows where exactly one corner is unmapped, the block's columns must take exactly one distinct value tuple. See the Task 11 completion note.)*

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

- [x] **Step 1: Branch from main and confirm the starting state**

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

- [x] **Step 1: Write the failing parity test**

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

- [x] **Step 2: Run to see the current state**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_serving_parity.py -q
```
Expected: it may already pass or fail on a few columns. **Record the exact output in the Completion notes.** If a column mismatches, that is a pre-existing serving bug worth its own line in the notes — investigate before refactoring (the known intentional case is `career_fights` feeding both `elo_fights_diff` and `career_fights_diff`, documented at `inference.py:314-320`; a mismatch there is expected to be tiny, and if it is non-zero the test's tolerance must not be loosened without saying why).

- [x] **Step 3: Extract the shared builder into `src/mma/serving.py`**

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

- [x] **Step 4: Verify parity and no table change**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_serving_parity.py tests/test_features.py tests/test_inference.py tests/test_prospective.py tests/test_explain.py -q
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py | tail -3
git status --short data/processed
```
Expected: all pass; `features.parquet` **byte-identical** (no `git status` output). If the table changes, the refactor altered a value — stop and find out which column before continuing.

- [x] **Step 5: Full suite and commit**

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

- [x] **Step 1: Write the failing tests**

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

- [x] **Step 2: Implement `src/mma/feature_blocks.py`**

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

- [x] **Step 3: `--blocks` in `scripts/build_features.py`**

Add `--blocks` (comma-separated, default `base`) and `--out` (default `data/processed/features.parquet`). Write a sidecar `data/processed/features_blocks.json` recording `{"blocks": [...], "n_rows": N, "n_columns": M}` so a report can be traced to the table it was computed on. (No timestamp: the sidecar is committed alongside the table and records nothing time-varying, which is how the byte-identity check on `features.parquet` stays meaningful.) Print the block list and column count.

- [x] **Step 4: Verify the default is unchanged**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_feature_blocks.py -q
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py | tail -4
git status --short data/processed
```
Expected: tests pass; only `features_blocks.json` is new; `features.parquet` unchanged.

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write the script**

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

- [x] **Step 2: Test the slice-comparison helper**

Factor the slice table as a pure function `slice_comparison(candidate, incumbent) -> list[dict]` and test it in `tests/test_walkforward.py` with two synthetic reports (one slice present in both, one only in the candidate → reported as `None` for the incumbent).

- [x] **Step 3: Verify against the committed reports and commit**

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

- [x] **Step 1: Write the failing accumulator tests** (append to `tests/test_history.py`)

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

- [x] **Step 2: Implement in `_FighterState`**

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

- [x] **Step 3: Leakage and parity gates**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py --blocks in_fight | tail -4
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_serving_parity.py tests/test_history.py -q
```
Expected: table gains 17 columns; truncation-invariance and parity both pass. **Both are blocking** — a failure here means the block is leaky or serving-incomplete, not that the test is wrong.

- [x] **Step 4: Evaluate** — run the Block evaluation procedure with `B = in_fight`, cleared set = `base`.

- [x] **Step 5: Record and commit**

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

- [x] **Step 1: Write the failing tests**

The accumulator needs two passes' worth of information but must stay single-pass and point-in-time. Implementation contract to test:
- Maintain, per fighter, running "allowed" aggregates: `sig_allowed_pm`, `td_allowed_pf`, `ctrl_allowed_share` (what opponents achieved against them).
- When fighter X fights opponent O, record for X the pair (X's own value this fight, O's pre-fight allowed value). The *feature* is the career mean of (own − opponent's allowed), i.e. output above expectation.
- Columns: `sig_pm_vs_exp`, `sig_absorbed_pm_vs_exp`, `td_landed_pf_vs_exp`, `td_def_vs_exp`, `ctrl_share_vs_exp`.
- Plus `avg_opp_elo_wins` and `avg_opp_elo_losses` (mean pre-fight Elo of beaten / losing opponents; NaN with no such fight).
Tests: a debutant has NaN everywhere; a fighter who outperforms a "leaky" opponent gets a positive `sig_pm_vs_exp`; the opponent's allowed value used is the one *before* the shared fight (construct three chronological fights and assert the exact number).

- [x] **Step 2: Implement and register** the block with those seven differentials.

- [x] **Step 3: Leakage and parity gates** (same two commands as Task 5, Step 3).

- [x] **Step 4: Evaluate** — Block evaluation procedure, `B = opponent_adjusted`, cleared set = base + whatever cleared so far.

- [x] **Step 5: Record and commit** (same shape as Task 5, Step 5).

---

### Task 7: Block `trajectory` — Glicko-2 and rating dynamics

Elo v1.1 rejected Glicko as a *replacement*; here it is tested as *additional* columns, together with rating dynamics that a level-only Elo cannot express.

**Files:** Create `src/mma/glicko.py`; modify `src/mma/history.py`, `src/mma/feature_blocks.py`, `src/mma/snapshots.py`, `scripts/build_ratings.py`; create `tests/test_glicko.py`.

- [x] **Step 1: Write the failing Glicko-2 tests**

Implement Glicko-2 (Glickman's published algorithm: rating μ, deviation φ, volatility σ, system constant τ = 0.5, convergence ε = 1e-6) and test against the worked example in the official paper: a player rated 1500 with RD 200 facing three opponents (1400/30, 1550/100, 1700/300) with results W/L/L ends at rating ≈ 1464.06, RD ≈ 151.52. Assert to 2 decimals. Also test: RD grows with inactivity; a single upset moves a high-RD player more than a low-RD one.

- [x] **Step 2: Add the ratings pass and the block**

`build_ratings.py` gains Glicko-2 columns to `ratings.parquet` (`pre_glicko_mu`, `pre_glicko_phi`, `pre_glicko_sigma`), computed in the same chronological pass as Elo (rating periods = one per event date). Block columns:
- `glicko_mu` (diff), `glicko_phi` (diff — uncertainty), `glicko_sigma` (diff)
- `elo_delta_3`, `elo_delta_5` — sum of the last 3 / 5 post-fight Elo changes (momentum)
- `elo_peak_minus_current` — career peak Elo minus current
- `years_since_ufc_debut`
- `age_squared` (absolute per corner) and `age_x_fights` (diff)
Tests: momentum is zero for a debutant, equals the sum of the last three deltas for a veteran; peak-minus-current is ≥ 0.

- [x] **Step 3: Rebuild ratings, then gates**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_ratings.py | tail -6
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/build_features.py --blocks trajectory | tail -4
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_processed_ratings.py tests/test_serving_parity.py tests/test_glicko.py -q
```
Note `ratings.parquet` changes here even if the block is rejected (new columns); that is acceptable — the Elo columns must be **byte-identical**, verify by comparing the pre-existing columns against `git show main:data/processed/ratings.parquet`.

- [x] **Step 4: Evaluate** — Block evaluation procedure, `B = trajectory`.

- [x] **Step 5: Record and commit.**

---

### Task 8: Block `context` — referee, home advantage, bonuses

**Files:** Modify `src/mma/history.py`, `src/mma/features.py`, `src/mma/feature_blocks.py`; create `src/mma/context.py`; extend tests.

Point-in-time care: referee and location are known before the fight (they are on the card), but the *statistics* about them must be computed from prior fights only.

- [x] **Step 1: Write the failing tests**

- `referee_finish_rate`, `referee_decision_rate` (fight-level): the referee's rate over their prior fights only; NaN for a referee's first fight; `referee_missing` flag. A chronological three-fight fixture pins the exact values.
- `home_country_a` / `home_country_b` (boolean): fighter nationality equals the event country. Nationality comes from the external snapshot (Task 11) — until that block exists, derive the event country from `fights.location` (last comma-separated component, normalised) and leave the nationality side NaN with `nationality_missing`. If the external block is rejected later, this feature degrades to "unknown" for everyone and must be dropped from the block; note that dependency in the block docstring.
- `bonus_rate` (diff): career performance-bonus wins per fight, from `bonuses.parquet`, prior fights only.

- [x] **Step 2: Implement**, registering `context` with `fight_level=("referee_finish_rate", "referee_decision_rate", "referee_missing")` and the rest as differentials/booleans.

- [x] **Step 3: Gates** (leakage + parity, as before). The referee columns are fight-level, so `serving.feature_row` must accept them through `context` — and `build_matchup`'s caller (`prospective.predict_event`) must supply the referee when known and `None` otherwise; make the missing path explicit and tested, because Wikipedia cards do not carry referees, so **at prediction time these are always missing**. That asymmetry is a real risk: a feature that is always present in training and always missing in serving will hurt. State in the block docstring that `context` ships only if it clears the bar *and* a re-run with the referee columns forced missing on the evaluation rows does not lose more than σ_seed — run that ablation as part of Step 4 and record both numbers.

- [x] **Step 4: Evaluate** — Block evaluation procedure plus the forced-missing ablation described above.

- [x] **Step 5: Record and commit.**

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

- [x] **Step 1: Write the failing tests** for a pure `merge_secondary(primary_fights, secondary_fights) -> (merged, report)`: rows whose `fight_id` is already in primary are dropped (primary wins); genuinely new rows are appended and counted; rows with ids absent from the fighters table are rejected with a named error; the report carries `n_added`, `max_date_before`, `max_date_after`.

- [x] **Step 2: Implement** the fetch (HTTP GET of the repo's raw CSVs, no auth), the schema adaptation to our `fights`/`fight_stats` shape, and the merge. Gate the whole thing behind `--enable`; default off, so a fetch failure can never break the weekly Action.

- [x] **Step 3: Verify on real data**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/refresh_secondary.py --enable --dry-run | tail -10
```
Expected: reports how many fights the secondary source has beyond 2026-08-08 (today's Kaggle cutoff). Record the number.

- [x] **Step 4: Wire into the Action** as an optional step that runs *after* the Kaggle refresh and *before* `make_dataset.py`, with `continue-on-error: true`, then commit.

---

### Task 11: Block `external` — pre-UFC career, gym, nationality

**Files:** Create `scripts/build_external.py`, `src/mma/external.py`, `tests/test_external.py`; modify `src/mma/features.py`, `feature_blocks.py`; create `data/external/*.parquet`.

Source: `ehan03/jds-mma-data` (MIT), specifically `data/clean/fighter_mapping.csv` (ufcstats id ↔ other site ids), Sherdog `fighter_histories.csv` / `fighters.csv`, Tapology `fighters.csv` / `fighter_gyms.csv`. Coverage ends ≈ 2024-08; every column carries a missingness flag and feeds the `external_missing` slice.

- [x] **Step 1: `scripts/build_external.py`** downloads the snapshot (git clone or raw file fetch into the scratch dir — never commit the raw 80 MB) and derives, keyed by ufcstats `fighter_id`:
`pre_ufc_wins`, `pre_ufc_losses`, `pre_ufc_finish_rate`, `pre_ufc_finish_loss_rate`, `days_since_pro_debut` (as of a given date, so store the debut date), `pre_ufc_avg_opp_wins`, `nationality`, `gym_id`. Write `data/external/fighter_external.parquet` (a few MB) plus a `data/external/README.md` recording source, licence, snapshot commit, and coverage dates.
- [x] **Step 2: `src/mma/external.py`** joins it to a fight row **strictly by fighter id** (never by name), computing `days_since_pro_debut` relative to the fight date and emitting `external_missing` per corner and a row-level `external_missing`. Tests: unmatched fighter → all NaN plus the flag; a fighter whose UFC debut precedes their recorded pro debut → flagged and NaN (data error, not negative days).
- [x] **Step 3: Gates.** Leakage: pre-UFC career values are as-of-snapshot and therefore constant — that is safe *only* because they describe the period before the fighter's UFC career; assert in a test that no external column changes when the fights table is truncated (this is exactly what the truncation-invariance test checks). Parity: serving must join the same table.
- [x] **Step 4: Evaluate**, and additionally report the `external_missing` slice explicitly — a block that helps overall but hurts the post-snapshot debutant slice materially is recorded as such.
- [x] **Step 5: Record and commit** (including the derived parquet and its README).

---

### Task 12: Blocks `notice` and `rankings`

Both are smaller and share the "join an external table by id, flag missingness" shape of Task 11.

- [x] **Step 1: `notice`** — `late_replacements.csv` / `missed_weights.csv` from the same snapshot (through 2024-08) give `notice_days` (binned: ≤7 / ≤30 / full camp / unknown, as one-hot `short_notice_7`, `short_notice_30`, `notice_unknown`) and `missed_weight_lbs` (+ `missed_weight` boolean). Extend `src/mma/wiki_cards.py` with a `parse_background(html) -> {withdrawals: [...], missed_weight: [...]}` parser for the templated prose ("X was expected to face Y ... replaced by Z", "weighed in at N pounds, M pounds over") to extend coverage past 2024-08 and to serve future fights; unit-test it against three saved fixture pages under `tests/fixtures/wikipedia/` (add them). **Serving asymmetry check**, as in Task 8: score the block with these columns forced missing on the evaluation rows and record both numbers, since the Wikipedia parser will not always find them.
- [x] **Step 2: `rankings`** — weekly official rankings history (2013→now; CC0). Fighter matching is by name, so reuse the prospective matcher's never-guess policy: exact then accent-folded, and anything else is unmatched (flagged). Columns: `rank` (diff, unranked = NaN + flag), `is_champion_a/b`, `is_ranked_a/b`, `rank_missing`, plus `ranking_regime_post_2026_06` (fight-level flag for the Elo-based ranking switch).
- [x] **Step 3:** For each, run the Block evaluation procedure; record; commit.

---

### Task 13: Assemble, redeploy, document, merge

- [x] **Step 1: Build the shipped table**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/build_features.py --blocks <comma-separated cleared blocks> | tail -4
~/.venvs/mma/bin/pytest tests/test_processed_features.py tests/test_serving_parity.py -q
```

- [x] **Step 2: Fresh-seed re-score of the shipped set**

```bash
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v3_seeds5 --seeds 5,6,7,8,9
~/.venvs/mma/bin/python scripts/block_decision.py --candidate models/walkforward/torch_v3_seeds5.json --incumbent models/walkforward/torch_v1_seeds5.json
```
The fresh-seed delta is the number reported publicly. If it does not clear the bar, the shipped set is reduced to the blocks that survive fresh seeds — say so in the notes.

- [x] **Step 3: Re-derive the deployment budget and redeploy**

The budget in the train-script defaults was derived from the v1 feature table; a new table needs a new budget:
```bash
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate xgb --name xgb_v3
~/.venvs/mma/bin/python scripts/run_walkforward.py --candidate torch --name torch_v3
~/.venvs/mma/bin/python scripts/refit_decision.py --out models/walkforward/refit_decision_v3.json
```
Update the train scripts' default `BUDGET`/`TEMPERATURE`/`REPORT` from the v3 reports (and the recency setting if Task 9 cleared one), retrain (`train_xgb.py`, `train_torch.py`, `build_display_priors.py`), record the new model hash, and confirm `Ensemble.load()` works and the app boots headless.

- [x] **Step 4: Documentation**

README: a "Features" section listing the shipped blocks and what each contributes, the block results table (including rejected blocks — the honest-negative-results story this project already tells twice), the new pooled numbers, and the data-source credits with licences. Plan Completion notes: the full results table, the fresh-seed number, the new hash, and the follow-ups. Spec: mark SP2 done and note anything deferred to SP3/SP4.

- [x] **Step 5: Verify and merge**

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

**Serving parity (Task 2):** the commit for that task did not record the first run's output, so this line is filled from what is verifiable now rather than reconstructed: both parity tests pass on every table built during this sub-project, including the shipped one, and the single known intentional divergence is the documented one -- `career_fights` feeds both `elo_fights_diff` and `career_fights_diff` at serving time, where training has two separate counters (`inference.py`, `build_matchup`'s closing NOTE). It is within the test's 1e-6 tolerance on the compared fights and the tolerance was never loosened.

**Block results** (pooled winner log-loss; incumbent in parentheses):

| Block | XGB | torch | Δ vs incumbent | Ships? | Notes |
|---|---|---|---|---|---|
| in_fight | 0.6540 (0.6537) | 0.6528 (0.6510) | +0.0018 vs torch_v1 | **no** | worse on both scorers; torch slices: womens +0.0062, debut +0.0034, five_round −0.0035. Incumbent stays `{xgb,torch}_v1`. |
| opponent_adjusted | 0.6519 (0.6537) | 0.6532 (0.6510) | +0.0022 vs torch_v1 | **no** | XGB liked it (-0.0018), torch did not; every fold worse on torch. Slices: debut +0.0030, womens +0.0020, five_round -0.0018. Coverage 0.63/0.51 of rows. Incumbent stays `{xgb,torch}_v1`. |
| trajectory | 0.6486 (0.6506) | 0.6472 (0.6476) | -0.0004 vs torch_external_diffsonly_extslice | **no** | XGB liked it (-0.0020), torch barely moved. Holding the most Elo-collinear column (`glicko_mu_diff`, r=0.88) out of the model lands in the same place (-0.0005), so collinearity is not the whole story. Kept regardless: `mma.glicko` and the `run_glicko` pass, so `ratings.parquet` carries the Glicko-2 triple. Incumbent stays `{xgb,torch}_external_diffsonly_extslice`. |
| context | 0.6499 (0.6506) | 0.6474 (0.6476) | -0.0002 vs torch_external_diffsonly_extslice | **no** | best of three variants is `context_nohome` (per-corner `home_country` held out of the model as a measured coverage channel); a fifteenth of the bar. Referee held-out ablation: 0.6481, **+0.0005** -- worse than the incumbent, so the serving-asymmetry gate fails too. Modelling `home_country_a`/`_b` is the worst variant (+0.0007). Incumbent stays `{xgb,torch}_external_diffsonly_extslice`. |
| recency | 0.6525 (0.6537) | 0.6499 (0.6510) | -0.0011 vs torch_v1 | **no** | best window/half-life: half-life 8y (window barely matters); improves on both scorers but by a third of the 0.003 bar. Incumbent stays `{xgb,torch}_v1`. |
| external | 0.6506 (0.6537) | 0.6476 (0.6510) | -0.0034 vs torch_v1 | **yes** | shipped as `external_diffsonly`: six pre-UFC differentials, the two fight-level flags kept in the table but excluded from both model matrices. Three variants measured and all three clear the bar; this one is the only one that does not degrade on 2024-2025 (row-weighted -0.00064). external_missing slice n=779, 0.6429 -> 0.6447, +0.0018, against the paired incumbent `torch_v1_extslice`. Coverage 0.799 of rows. New incumbent: `{xgb,torch}_external_diffsonly_extslice`. |
| notice | 0.6501 (0.6506) | 0.6472 (0.6476) | -0.0004 vs torch_external_diffsonly_extslice | **no** | best variant `notice_noflag` (coverage flag in the table, out of the model); a seventh of the bar. Coverage 0.498 of rows and **0.000 of 2025-2026**, so the forced-unknown ablation was moot -- serving is the unknown state for every future fight by construction. Incumbent stays `{xgb,torch}_external_diffsonly_extslice`. |
| rankings | 0.6499 (0.6506) | 0.6487 (0.6476) | +0.0011 vs torch_external_diffsonly_extslice | **no** | match rate 631/656 names (0.9786 of source rows). All three variants worse on torch (+0.0011 to +0.0017). Real signal on 13.8% of rows, correlated -0.40 with `elo_diff` there. Incumbent stays `{xgb,torch}_external_diffsonly_extslice`. |

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

**Block `trajectory` (Task 7), measured and rejected.** Ten columns of rating
DYNAMICS on top of the base block's rating LEVELS: `glicko_mu_diff`,
`glicko_phi_diff`, `glicko_sigma_diff` (Glicko-2 rating, deviation and
volatility), `elo_delta_3_diff` / `elo_delta_5_diff` (the sum of the last
three and five post-fight Elo movements), `elo_peak_minus_current_diff`,
`years_since_ufc_debut_diff`, `age_x_fights_diff` and `age_squared_a`/`_b`.

- **Glicko-2 is verified against the published worked example**, which is the
  part of this task worth keeping whatever the block did. Glickman's paper
  walks a 1500/200/0.06 player at tau 0.5 through one rating period against
  1400/30, 1550/100 and 1700/300 with W/L/L and reports 1464.06 / 151.52 /
  0.05999; `mma.glicko.update` returns 1464.0507 / 151.5165 / 0.0599960 --
  agreement to one unit in the paper's last quoted decimal, which is what its
  own rounded intermediates allow. `tests/test_glicko.py` (7 tests) also pins
  RD growth under inactivity and its 350 cap, the larger move a high-RD player
  takes from the same upset, and the empty rating period.
- **Two MMA-specific choices, both documented in `mma.glicko`.** A rating
  period is an EVENT DATE, so every bout on a card is scored against the
  ratings its fighters carried into that card (which is what makes the
  pre-fight value correct for a fighter with two bouts in one night). Between
  cards, RD grows with the CALENDAR rather than with the number of cards
  missed -- `PERIOD_DAYS = 7`, one nominal period per week -- because a
  per-event-date clock would make a lay-off cost four RD steps a year in 1995
  and fifty in 2025. The weekly clock is also what lets serving reproduce a
  trained value exactly: `build_matchup` grows the fighter's post-fight
  deviation over the days since, and the serving-parity test passed on the
  Glicko columns unchanged.
- **The Elo columns are byte-identical.** `run_glicko` produces the same row
  set and keys as `run_elo`, and `build_ratings.py` joins on
  (fight_id, corner, fighter_id) with `validate="one_to_one"`, so the six new
  columns are appended and nothing existing moves. Checked, not assumed:
  `assert_frame_equal(git show HEAD:ratings.parquet, new[old.columns],
  check_exact=True, check_dtype=True)` passes over all 22,646 x 11.
- XGB screen: pooled winner LL **0.6486** vs incumbent 0.6506 (delta
  **-0.0020**, better but two-thirds of the bar). Six of eight folds improved,
  worst +0.0029 (2021); best -0.0086 (2023).
- torch decision: pooled winner LL **0.6472** vs incumbent 0.6476 (delta
  **-0.0004**). `bar_check` -> `clears_delta: false`,
  `no_fold_regression: true`, **`ships: false`**. Fold deltas 2018 -0.0004,
  2019 -0.0015, 2020 -0.0012, 2021 -0.0021, 2022 +0.0024, 2023 -0.0024,
  2024 +0.0024, 2025 -0.0004; six of eight better, none by anything like the
  bar.
- torch slices vs `torch_external_diffsonly_extslice`: debut -0.0008,
  womens -0.0008, five_round -0.0016, external_missing +0.0009.
- **The collinearity hypothesis was tested and is not the explanation.**
  `glicko_mu_diff` correlates 0.88 with `elo_diff`, and the three previous
  rejected blocks all failed with the "correlated addition costs the MLP"
  signature, so a second torch run held it out of the model
  (`torch_trajectory_nomu`): pooled 0.6471, delta **-0.0005** -- the same
  place. The block is a recombination of rating history the base block already
  carries, and on this table the deployed scorer already has it.
- Coverage (11,238 rows, both corners needed): the three Glicko columns and
  the two momentum sums are 1.000 (Glicko has a defined prior for everyone and
  momentum is 0.0 for a debutant by design); `elo_peak_minus_current_diff` and
  `years_since_ufc_debut_diff` 0.726 (both corners must have a prior bout);
  `age_x_fights_diff` 0.969; `age_squared_a`/`_b` 0.980/0.981 (dob coverage).
- Gates before evaluation, all passing on the trajectory table (11,238 x 64):
  `test_no_leakage_truncation_invariance`, both serving-parity tests, and
  `tests/test_processed_ratings.py`.
- **Kept after the revert**, because it is reusable and correct: `src/mma/glicko.py`
  (the engine plus the `run_glicko` chronological pass), `tests/test_glicko.py`,
  and the `run_glicko` step in `scripts/build_ratings.py` -- so
  `data/processed/ratings.parquet` carries `{pre,post}_glicko_{mu,phi,sigma}`
  (22,646 x 17) whether or not any block models them, exactly as the plan's
  Task 7 Step 3 anticipated. Reports kept: `models/walkforward/xgb_trajectory.json`,
  `torch_trajectory.json`, `torch_trajectory_nomu.json`. The block's feature
  wiring is reverted; the commented-out registration in
  `mma.feature_blocks` records what restoring it would take.

**Block `context` (Task 8), measured and rejected.** Seven columns describing
the fight's SETTING rather than either fighter's record: `bonus_rate_diff`,
`home_country_a`/`_b`, and the four fight-level columns
`referee_finish_rate`, `referee_decision_rate`, `referee_missing`,
`home_country_unknown`. `mma.context` owns the derivations.

- **The constant-vector leak check FAILS for `home_country_a`/`_b`, and that
  is the block's first real finding.** A per-corner boolean is False when the
  corner's nationality is unknown, and nationality only exists for fighters
  the `external` snapshot mapped -- so over the **1,448** feature rows where
  exactly ONE corner is mapped, the pair takes **three** distinct value tuples
  ((True, False), (False, True), (False, False)) rather than one, and those
  rows are exactly where the mapped corner wins **0.745** of the time. That is
  the `external` block's measured selection effect in a second channel. Per
  the amended missingness rule the two columns stay in the TABLE and are held
  out of both model matrices; the variant that models them is the worst of the
  three measured (+0.0007 on torch), which is the outcome that rule predicts.
  Asserted one level lower, on `mma.context` over the real fights and external
  tables, in `tests/test_context.py::test_a_per_corner_home_country_pair_is_a_coverage_channel`,
  so it survives the block being reverted.
- **The home-advantage effect itself is real but small.** Over the 8,973 rows
  where BOTH nationalities are known, the corner fighting in its own country
  wins 0.516 (n=1,885 with only A home) against 0.490 (n=1,859 with only B
  home) -- about a 2.6-point edge, on 33% of rows.
- XGB screen, three variants, all inside the 0.002 threshold and all slightly
  better than the incumbent 0.6506: `xgb_context` (everything modelled)
  **0.6501** (-0.0005), `xgb_context_nohome` (the two `home_country` columns
  held out) **0.6499** (-0.0007), `xgb_context_nohome_noref` (also the three
  referee columns held out) **0.6498** (-0.0008). Note the ordering: on the
  screen, each thing removed makes it slightly BETTER.
- torch decision, against the incumbent 0.6476:
  `torch_context` **0.6483** (delta **+0.0007**, worst fold +0.0021),
  `torch_context_nohome` **0.6474** (delta **-0.0002**, worst fold +0.0017),
  `torch_context_nohome_noref` **0.6481** (delta **+0.0005**, worst fold
  +0.0047). `bar_check` on all three -> `clears_delta: false`,
  `no_fold_regression: true`, **`ships: false`**. The best is a fifteenth of
  the bar.
- `torch_context_nohome` fold deltas: 2018 0.0000, 2019 +0.0007, 2020 +0.0004,
  2021 0.0000, 2022 +0.0017, 2023 -0.0042, 2024 -0.0002, 2025 +0.0003 -- the
  whole of its pooled gain is one fold (2023).
- torch slices vs `torch_external_diffsonly_extslice` (`context_nohome`):
  debut -0.0014, womens -0.0026, five_round -0.0037,
  external_missing +0.0006.
- **The serving-asymmetry gate, run as required and failed independently.**
  Wikipedia cards do not name a referee, so the three referee columns are
  present on 97.7% of training rows and on **0%** of the rows the deployed
  model will ever score. The gate is "ship only if the block clears the bar
  AND the referee-held-out variant does not lose more than sigma_seed (0.000346)
  against the incumbent". The held-out variant is **+0.0005** against the
  incumbent -- worse, by more than sigma_seed -- so it fails the second clause
  as well as the first. The columns are also worth nothing before the gate is
  applied: removing them moves torch from 0.6474 to 0.6481 and XGB from 0.6499
  to 0.6498, i.e. within seed noise on the screen and a small LOSS on the
  scorer that decides. There is no variant of this block that ships.
- Coverage over the 11,238 rows: `bonus_rate_diff` 0.726 (both corners need a
  prior bout); `home_country_a`/`_b` True on 0.395/0.393;
  `home_country_unknown` True on 0.259 (nationality unknown for ~20% of
  corners, `fights.location` absent for ~23% of fights);
  `referee_finish_rate` / `referee_decision_rate` populated on 0.951 and
  `referee_missing` True on 0.023, concentrated in the pre-2017 rows -- every
  fold year from 2019 on has full referee coverage in TRAINING, which is
  exactly what makes the serving asymmetry sharp rather than academic.
- Gates before evaluation, all passing on the context table (11,238 x 61):
  `test_no_leakage_truncation_invariance` and both serving-parity tests. The
  parity test had to be given the two card-level facts a real caller supplies
  (`referee` + `mma.context.referee_rates(past)`, and
  `mma.context.event_country(location)`); with them the served row matched the
  trained row exactly, including `home_country_a`/`_b`.
- **Kept after the revert**, because it is reusable and correct:
  `src/mma/context.py` (the prior-fights-only referee pass, the event-country
  parser with its nationality/location normalisation, the home-advantage
  comparison and the bonus fight-id loader) and `tests/test_context.py` (25
  tests). Reports kept: `models/walkforward/{xgb,torch}_context.json`,
  `..._context_nohome.json`, `..._context_nohome_noref.json`. The block's
  feature wiring is reverted; the commented-out registration in
  `mma.feature_blocks` records what restoring it would take.
- *Definition care taken along the way.* `bonuses.parquet` records bonuses per
  FIGHT, not per fighter, so "Performance of the Night" cannot be attributed
  to a corner from the data; `bonus_rate` credits the WINNER of a bonus fight,
  which is stated in `mma.context`'s docstring rather than left implicit, and
  means a shared "Fight of the Night" counts for the winner only.

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
channel survives.

*The evidence that no channel survives is the constant-vector result, not the
`diffsonly` ablation.* On the 1,445 feature rows where exactly one corner is
unmapped -- precisely the rows the selection effect could be read off -- the
block's eight columns take **exactly one distinct value tuple**:
`(NaN x 6, external_missing=True, same_country=False)`. No function of those
columns can separate a fight whose unmapped corner is A from one whose
unmapped corner is B, so the selection effect is *unreachable*, not merely
unmodelled. That is what
`tests/test_external.py::test_a_half_matched_fight_carries_exactly_one_value_tuple`
asserts, on the real table, over those rows -- a data-level property rather
than a rule about column names, so it also catches a per-corner channel
smuggled in under a name that does not end in `_a`/`_b`.

*Correction to the first write-up:* the `{xgb,torch}_external_diffsonly`
ablation was presented as "the check that the corrected block is not just a
subtler encoding of the same selection effect". It cannot be that check.
`diffsonly` withholds the flags from the model but leaves the differentials
NaN exactly where a corner is unmapped, so missingness is fully recoverable
from the NaN pattern -- an ablation in which the leak is still present in the
input cannot demonstrate its absence. What `diffsonly` actually measures is
the flags' marginal value (~-0.0004 pooled), which is a useful number and is
what the variant comparison below uses it for.

- XGB screen, `xgb_external_noleak`: pooled **0.6499** vs 0.6537 (delta
  **-0.0038**), worst fold +0.0031 (2021), `ships: true`.
- torch decision, `torch_external_noleak`: pooled **0.6472** vs 0.6510 (delta
  **-0.0038**), fold deltas 2018 -0.0077, 2019 -0.0073, 2020 -0.0069,
  2021 -0.0059, 2022 -0.0026, 2023 -0.0050, 2024 +0.0009, 2025 +0.0004;
  worst fold +0.0009. `clears_delta: true`, `no_fold_regression: true`,
  **`ships: true`**. ECE also improves, 0.0121 vs 0.0139.
- torch slices vs `torch_v1`: five_round -0.0088, debut -0.0015,
  womens -0.0011. Nothing regresses.
- **`external_missing` slice: the block makes those rows WORSE, by +0.0023.**
  n=779, candidate LL **0.6452** (ECE 0.0515) against a paired incumbent's
  **0.6429** (ECE 0.0461). Both accuracy and Brier are flat (0.6367 vs 0.6354,
  0.2271 vs 0.2264); the cost is calibration, which is what a log-loss slice
  delta is for.

  *This corrects an error in the first write-up of this task, which reported
  the slice as "not computable" and then compared 0.6452 to the candidate's own
  pooled 0.6472 -- concluding the block "does not hurt the post-snapshot rows".
  That comparison measures how easy those rows are relative to the rest of the
  table, not what the block did to them, and it is exactly the failure the
  harness exists to prevent. The two numbers happen to point opposite ways: the
  slice is easier than average AND the block degrades it.*

  The delta is computable, and cheaply. `torch_v1.json` does predate the
  column, but re-running the v1 *recipe* on the current 54-column table with
  the eight external columns held out of the model matrix
  (`scripts/run_walkforward.py --drop-columns`, added in this task) reproduces
  `torch_v1.json` **bit-exactly** -- pooled, all eight folds, all three shared
  slices and every `fit_info` entry -- while still seeing `external_missing` in
  the table, so `walkforward.slice_masks` reports the slice. That equality is
  what makes it a valid paired stand-in rather than a regenerated incumbent:
  the locked rule forbids *changing* the incumbent numbers, and these are the
  same numbers. Report: `models/walkforward/torch_v1_extslice.json`.

  So the block's -0.0038 pooled gain is bought entirely on rows the snapshot
  covers, and paid for slightly on the rows it does not -- which is the same
  story the fold deltas tell (below), read on the axis that names the cause.
  The contrast with the leaky variant is still stark: there the same slice read
  0.5842 with accuracy 0.6906, i.e. *better* than the well-covered rows, which
  was the leak showing.
**The pooled gain is entirely historical.** The eight fold deltas above split
cleanly in two: 2018-2023 average **-0.0058** row-weighted (n=3,276) while
2024-2025 average **+0.0006** (n=1,528). Over the same span `external_missing`
climbs 0.086 (2018) -> 0.138 (2024) -> 0.359 (2025) -> 0.534 (2026). The block
pays where the snapshot has seen the fighters and does nothing -- very
slightly worse than nothing -- where it has not, and the share of rows it has
not seen only grows.

**Three shipping variants, and the rule that chose between them.** The six
pre-UFC differentials are the same in all three; they differ only in which of
the two fight-level flags the MODEL sees. All three keep both flags in the
TABLE, so `walkforward.slice_masks` reports the `external_missing` slice for
each. Torch decides; every number below is against `torch_v1` (pooled/folds)
and against the paired `torch_v1_extslice` (slice).

| variant | flags modelled | XGB | torch | Δ pooled | worst fold | ships? | 2018-23 rw | **2024-25 rw** | `external_missing` slice |
|---|---|---|---|---|---|---|---|---|---|
| `external_noleak` | both | 0.6499 | 0.6472 | **-0.0038** | +0.0009 | yes | -0.0058 | **+0.00059** | 0.6452 (+0.0023), ECE 0.0515 |
| `external_diffsonly` | neither | 0.6506 | 0.6476 | -0.0034 | +0.0015 | yes | -0.0047 | **-0.00064** | 0.6447 (+0.0018), ECE 0.0457 |
| `external_nomissflag` | `same_country` only | 0.6513 | 0.6478 | -0.0032 | +0.0032 | yes | -0.0052 | **+0.00106** | 0.6469 (+0.0040), ECE 0.0581 |

Per-fold torch deltas vs `torch_v1`:

| variant | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| `external_noleak` | -0.0077 | -0.0073 | -0.0069 | -0.0059 | -0.0026 | -0.0050 | +0.0009 | +0.0004 |
| `external_diffsonly` | -0.0071 | -0.0097 | -0.0052 | -0.0015 | -0.0045 | -0.0005 | +0.0015 | **-0.0020** |
| `external_nomissflag` | -0.0086 | -0.0090 | -0.0055 | -0.0033 | -0.0038 | -0.0012 | +0.0032 | -0.0003 |

`external_nomissflag` is the new measurement (XGB delta -0.0024, inside the
0.002 screen threshold, so torch was run; torch `bar_check` -> `clears_delta:
true`, `no_fold_regression: true`, **`ships: true`**). It answers whether
`same_country` alone is the useful flag: it is not -- keeping it costs on 2024
(+0.0032, the worst recent fold of the three) and degrades the
`external_missing` slice most (+0.0040). `same_country` is False whenever
either nationality is unknown, so it carries the coverage artifact in a second
channel rather than avoiding it.

**The shipping rule (pre-registered here, applied below):**

> The pre-registered bar decides **whether** the block ships. Among variants
> that clear it, prefer the one that does not degrade on the most recent
> folds, because coverage decay is a known, mechanistic, forward-looking risk
> rather than a post-hoc preference. Record all variants and their numbers.

**Applied: `external_diffsonly` ships** -- the six differentials, neither flag
modelled. It is the only one of the three whose 2024-2025 row-weighted delta
is negative (-0.00064 against +0.00059 and +0.00106), it has the best
`external_missing` slice of the three (+0.0018 against +0.0023 and +0.0040)
and the best pooled ECE (0.0088 against 0.0121 and 0.0115). It gives up
0.0004 of pooled log-loss to `external_noleak` -- an eighth of the bar, and
roughly one sigma_seed -- to buy the fold profile that does not decay. The
flags add ~-0.0004 pooled, and they buy it entirely on folds whose coverage we
will never have again.

- Reports for the shipped variant: `models/walkforward/xgb_external_diffsonly_extslice.json`
  and `models/walkforward/torch_external_diffsonly_extslice.json`. These are
  the `--drop-columns external_missing,same_country` runs on the full
  54-column table; both reproduce the earlier `{xgb,torch}_external_diffsonly`
  reports (built on a table that lacked the flag columns) **bit-exactly** on
  pooled, all folds and all three shared slices, and additionally carry the
  `external_missing` slice, which the old pair could not report.
- **Shipping mechanism.** The columns stay in the table and in the served row;
  the two learners exclude them -- `mma.tensors.DROPPED` for torch (which
  already existed for exactly this, holding the four era-proxy flags dropped
  after the Phase 3 ablation) and `mma.models.xgb.MODEL_EXCLUDED` /
  `NON_FEATURES` for XGB. Both carry the reason and a pointer back here.
  Guarded by `tests/test_external.py::test_the_fight_level_flags_are_produced_but_never_modelled`
  and `::test_the_flags_survive_into_the_table_and_out_of_both_model_matrices`.
  This is why the flags could not simply be deleted from the block: without
  `external_missing` in the table there is no `external_missing` slice, and
  the coverage decay this whole section is about becomes unmeasurable.
- **What `external_diffsonly` measures**, restated: the flags' marginal value.
  The six differentials carry essentially all of the gain (-0.0034 of the
  -0.0038) and the two flags add ~-0.0004 pooled -- which, per the shipping
  rule, is bought entirely on folds whose coverage we will never have again.
  It is *not* leak evidence; see the correction above.
- Gates on the shipped table, all passing:
  `test_no_leakage_truncation_invariance`, both serving-parity tests, and the
  base-only rebuild stayed byte-identical.
- **Caveat for SP4:** the benefit decays as the snapshot ages. Even in the
  shipped variant it is only -0.0006 row-weighted on 2024-2025 against -0.0047
  on 2018-2023, and `external_missing` reaches 0.534 of 2026 rows. The
  snapshot is static (last commit December 2025); if this block is to keep
  paying, `scripts/build_external.py` needs a refreshable source, or the
  pre-UFC record needs to come from a live scrape.

  *Followed up 2026-09-09, and the caveat does not hold on the deployed
  scorer.* Coverage decayed as predicted -- `external_missing` is now 0.510
  over the trailing twelve months and the 2025 FOLD (which spans 2025-2026)
  runs at 0.4225 -- so the question was re-asked under a pre-registered
  removal rule (`docs/superpowers/plans/2026-09-09-external-decay-decision.md`).
  Dropping the six differentials from the **hybrid** costs **+0.0059** pooled
  joint log-loss and **+0.0041** row-weighted on 2024-2025, with all eight
  folds worse; at fresh seeds 5-9, +0.0061 and +0.0026. The "-0.0006 on
  2024-2025" above is a torch-only number, and SP2.2 had already found these
  blocks alive through the XGBoost member the deployed blend carries. The
  block stays; nothing was redeployed. `models/walkforward/external_decay_decision.json`.

**Incumbent after this block (and still the incumbent after Task 12, since
neither `notice` nor `rankings` shipped):
`models/walkforward/xgb_external_diffsonly_extslice.json`
(0.6506) and `models/walkforward/torch_external_diffsonly_extslice.json`
(0.6476)** -- the SHIPPED variant's reports, not the best-pooled one's. The
cleared feature set is `base,external`, with `external_missing` and
`same_country` present in the table and excluded from both model matrices.
Later blocks are judged against these.

**Block `notice` (Task 12 Step 1), measured and rejected.** Two hinged
differentials and three per-corner flags over short-notice replacements and
missed weight, plus a fight-level coverage flag:
`notice_shortfall_days_diff` (max(0, 30 - days of notice), 0 for an observed
full camp), `missed_weight_over_lbs_diff` (pounds over the divisional limit, 0
if the fighter made weight), `short_notice_7_a/b`, `short_notice_30_a/b`,
`missed_weight_a/b`, and `notice_unknown`.

*Source paths actually found* (the plan's row counts were high):
`data/clean/Bet MMA/late_replacements.csv` (**586** rows, not ~1,000;
`fighter_id, bout_id, notice_time_days`, 1-46 days, median 9),
`data/clean/Bet MMA/missed_weights.csv` (**271** rows;
`fighter_id, bout_id, weight_lbs` -- the WEIGH-IN weight, not the overage),
`data/clean/Bet MMA/bouts.csv` (13,163 bouts over 1,685 events, 2013-04-20 to
2024-12-14), `data/clean/bout_mapping.csv` (5,674 of our fights carry a
`betmma_id`) and `data/clean/Bet MMA/fighters.csv` (a `ufcstats_id` column,
which with `fighter_mapping.csv`'s `betmma_id` resolves 1,874 Bet MMA fighter
ids). Same MIT snapshot and commit as the `external` block (`ec77f537`).

**The observability question the task posed, and the answer.** "Not a late
replacement" IS observable, but not from `late_replacements.csv`, which lists
only fighters it happened to. What makes it observable is the source's own
BOUT list: Bet MMA covers its events bout by bout, so inside that set the
absence of a replacement row is an observation and outside it is ignorance.
The derivation therefore emits one row per CORNER of every covered bout and
nothing at all for the rest, which makes membership of
`data/external/fight_notice.parquet` the three-state boundary (row + NaN days =
observed full camp; row + n days = replaced on n days' notice; no row =
unknown). So the block encodes a genuine three-state, not
`short_notice in {True, unknown}`.

*Corner assignment never uses names, and needs no elimination.* All 5,674
mapped bouts have BOTH Bet MMA fighter ids resolvable to ufcstats ids, and in
all 5,674 the resulting pair is exactly the pair our own fights table records
-- checked, not assumed, and asserted in the derivation. 11,348 corner rows
over 5,674 fights.

*Coverage.* 5,597 of the 11,238 feature rows are observed (0.498). Within the
source's 2013-04-20 .. 2024-12-14 window the rate is 0.917; outside it, zero.
**Every 2025 and 2026 row is unknown**, which is the finding that matters:
`notice_unknown` runs 0.10-0.14 across the 2017-2024 folds and 1.000 for 2025
and 2026. Of the observed rows, 484 have exactly one corner on <= 30 days'
notice, 159 on <= 7 days, and 216 have exactly one corner missing weight.

*The raw signal is real and large -- and still not enough.* Over the observed
rows where exactly one corner was a late replacement, that corner wins
**0.351** of the time at <= 30 days and **0.264** at <= 7 days; the corner that
missed weight wins 0.458. That is a much stronger marginal than anything the
`external` differentials carry. It reaches 4.3% of the table.

*The constant-vector leak check passes.* Over the 5,641 feature rows the source
has not seen, the block's nine columns take **exactly one distinct value
tuple** -- `(NaN, NaN, False x 6, True)` -- so no function of them can say
which corner the source is missing. Because the source's unit of coverage is
the BOUT, this holds by construction rather than by correction: there is no
per-corner missingness channel to leak in the first place. Now asserted one
level lower, on `mma.notice.attach` over the real fights table, in
`tests/test_notice.py::test_an_unknown_fight_carries_exactly_one_state_tuple`,
so it survives the block being reverted.

- XGB screen, `xgb_notice` (coverage flag modelled): pooled **0.6509** vs
  incumbent 0.6506 (delta **+0.0003**). `xgb_notice_noflag` (flag in the table,
  held out of the model with `--drop-columns`): **0.6501** (delta **-0.0005**).
  Both inside the 0.002 screen threshold, so both went to torch.
- torch decision, `torch_notice`: pooled **0.6482** vs 0.6476 (delta
  **+0.0006**), worst fold +0.0063 (2018). `clears_delta: false`,
  `no_fold_regression: true`, **`ships: false`**.
- torch decision, `torch_notice_noflag`: pooled **0.6472** (delta **-0.0004**),
  fold deltas 2018 +0.0080, 2019 -0.0014, 2020 -0.0014, 2021 -0.0011,
  2022 +0.0003, 2023 -0.0057, 2024 +0.0024, 2025 -0.0023; worst fold +0.0080.
  `clears_delta: false`, `no_fold_regression: true`, **`ships: false`**.
  The better of the two, and still an eighth of the bar.
- torch slices vs `torch_external_diffsonly_extslice` (`notice_noflag`):
  debut -0.0052, womens +0.0001, five_round +0.0009,
  external_missing -0.0004.
- **The forced-unknown (serving-asymmetry) ablation was not run, and the reason
  is stronger than the ablation.** It exists to ask what happens when the
  Wikipedia parser finds nothing at prediction time. Here the answer is not a
  probability but a certainty: the source stops at 2024-12-14, so the served
  state is the unknown state for *every* future fight, and the eight columns
  are constant on 100% of the rows the deployed model will ever see. The gate
  is "ship only if the block clears the bar AND the forced-unknown variant does
  not lose more than sigma_seed"; the block fails the first clause by 0.0026,
  so the second was not reached. A forced-unknown run could only have moved the
  result toward the incumbent it already fails to beat.
- *Source data care taken along the way.* `missed_weight` is a separate boolean
  column from `missed_weight_over_lbs` because three of the 223 covered misses
  have no believable magnitude: two are `Catch Weight` bouts with no limit, and
  one records a 273 lb weigh-in at a bantamweight bout, which entered the first
  build of the feature table as **138 pounds over**. `MAX_MISSED_WEIGHT_OVER_LBS`
  = 25 now bounds it; the flag keeps the fact, the magnitude goes to 0 rather
  than being invented. All four reports were regenerated after that fix.
- Gates before evaluation, all passing on the notice table (11,238 x 63):
  `test_no_leakage_truncation_invariance` and both serving-parity tests. The
  base+external rebuild afterwards stayed byte-identical.
- Reports kept: `models/walkforward/{xgb,torch}_notice.json` and
  `{xgb,torch}_notice_noflag.json`. Block registration reverted (left
  commented in `mma.feature_blocks` with the numbers), together with the
  `features.py` / `inference.py` / parity-test wiring.
- **Kept and reusable, because the derivation is correct and the coverage is
  the only thing wrong with it:** `scripts/build_external.py::derive_notice`,
  the committed `data/external/fight_notice.parquet` (11,348 rows), the loader
  `src/mma/notice.py` with its tests, and
  `mma.wiki_cards.parse_background(html) -> {"withdrawals": [...],
  "missed_weight": [...]}` -- the only route to these facts for a FUTURE event,
  and the thing a live source would need. The parser is fixture-tested against
  the trimmed Background sections of three real pages, one per era
  (`tests/fixtures/wikipedia/ufc{196,302,326}_background.html`, 2016 / 2024 /
  2026, CC BY-SA 4.0, source URL in each file): it reads withdrawals with their
  replacements through a two-step chain, strips capitalised promotional titles
  off names ("replaced by former LFA Middleweight Champion Gregory Rodrigues"),
  parses overages written as words as well as digits ("three and three quarters
  pounds over" -> 3.75), and returns days of notice only when the prose states
  them, which it rarely does -- UFC 196 is the "no such notes" case for
  weigh-ins, with two withdrawals and no missed weight. It is NOT wired into
  `prospective.predict_event`, per the task's condition that it be wired only
  if the block ships.

**Block `rankings` (Task 12 Step 2), measured and rejected.** The UFC's own
weekly divisional rankings as `rank_diff` (unranked = NaN), `is_champion_a/b`,
`is_ranked_a/b`, and the fight-level `rank_missing` and
`ranking_regime_post_2026_06`.

*Source and licence.* Kaggle `jerzyszocik/ufc-rankings-history` ("UFC Rankings
History (2013-ongoing)"), version 53, **CC0: Public Domain** -- confirmed from
the dataset metadata, not assumed. `date, weightclass, fighter, rank`, champion
= rank 0, 533 weekly publications from 2013-02-04 to 2026-09-03. Downloaded
with `kagglehub`, already a dependency; the raw CSV is not committed. The
GitHub alternative `martj42/ufc_rankings_history` was not used: it carries no
LICENSE file at all, so redistributing anything derived from it would be a
guess. Derived table: `data/external/rankings.parquet` (86,850 rows over 12
divisions), documented in `data/external/RANKINGS.md`.

*Pound-for-pound rows are dropped.* They are not a division a bout happens in,
they re-list fighters already ranked in their own division, and the source
spells them four different ways across the years.

*Match rate.* The source has names and nothing else, so matching used the
prospective pipeline's never-guess matcher unchanged (exact unicode-normalised,
then accent-folded, ambiguous = unmatched): **631 of 656 distinct names**
matched, 620 exact and 11 accent-folded, covering **0.9786 of the divisional
source rows**; 630 distinct fighter ids. The 25 misses are ring names the UFC
uses and ufcstats does not (`Rampage Jackson`, `Cris Cyborg`, `Mirko Cro Cop`,
`Minotauro Nogueira`, `Michael Venom Page`), spellings (`Georges St. Pierre`,
`Costas Philippou`, `Seohee Ham` vs `Seo Hee Ham`), four names shared by two of
our fighters (ambiguous by construction), and three MECHANICAL near-misses --
`Jan Błachowicz`/`Blachowicz` and `Klaudia Syguła`/`Sygula` (a stroked Latin
letter, which NFKD does not decompose the way it decomposes an accent) and
`Lone'er Kavanagh` (curly vs straight apostrophe). Those three would close by
extending `mma.prospective.fold_accents` with a stroked-letter map and
apostrophe normalisation, without weakening the never-guess rule; left as a
follow-up because `fold_accents` is on the live prediction path.
An unmatched name is dropped, so that fighter reads as *unranked* rather than
*unknown* -- the honest cost of a name-keyed source, recorded rather than
patched over with a fuzzy match.

*Point-in-time is a JOIN RULE here, not a cut.* Every other external table in
this project is safe because its window closed before the fights; a ranking
moves every week and moves *because of results*, so the safety is
`merge_asof(..., allow_exact_matches=False)` -- the last publication STRICTLY
before the fight date. A same-day list is inadmissible even though it is not
literally after the fight: the UFC updates on Tuesdays after the weekend's
cards. Asserted on a fixture whose rank changes on the fight day
(`test_the_ranking_used_is_the_last_one_published_before_the_fight`) and end to
end on the real tables
(`test_no_fight_reads_a_ranking_published_on_or_after_its_own_date`).

**Leak check: this block's per-corner flags are NOT the `external` failure.**
Three separate readings, all pointing the same way:
- *No coverage decay.* `rank_diff` is populated on 0.19-0.27 of rows in every
  year from 2015 to 2026 with no trend -- the source is refreshed weekly, so
  unlike the 2024-12 jds snapshot it does not age away from the fights served.
- *The per-corner channel is weak, not decisive.* Over the 1,300 rows where
  exactly one corner is ranked, that corner wins **0.545** of the time, flat
  across years (0.44-0.64, n~100/yr). Compare `external`, where the unmapped
  corner lost 76% of the time overall and 90% in the debut slice. Being ranked
  in the week before a fight is determined by results already in our own fights
  table; it is contemporaneous merit, not a look-ahead into career length.
- The constant-vector check therefore does NOT apply as a pass/fail here and
  was not expected to: over those rows the block's seven columns take six
  distinct value tuples, by design, because `is_ranked_a/b` is legitimately
  per-corner. Recording it explicitly so the difference from Task 11 is on the
  record: there the property was the evidence of absence of a leak, here the
  evidence is the coverage profile and the 0.545.
So the flags were kept in the model for the headline variant, and a variant
with them held out was measured anyway (below) rather than assumed.

*The signal is real, and better than Elo where it exists.* Among the 1,502
fights where both corners were ranked at different ranks, the better-ranked
corner wins **0.563** of the time, while the higher-`elo_diff` corner wins only
**0.533** on the same rows; the champion wins **0.673** of the 199
champion-vs-challenger bouts. But `rank_diff` reaches 1,546 of 11,238 rows
(**0.138**) and where it does it correlates -0.40 with `elo_diff` and -0.54
with `last5_avg_opp_elo_diff`.

- XGB screen (all three inside the 0.002 threshold, so all three went to
  torch): `xgb_rankings` **0.6505** (delta -0.0001), `xgb_rankings_noflags`
  (the two fight-level flags held out) **0.6499** (delta **-0.0007**),
  `xgb_rankings_diffonly` (flags and per-corner booleans held out, `rank_diff`
  alone) **0.6509** (delta +0.0003).
- torch decision, all three WORSE than the incumbent 0.6476:

  | variant | modelled | torch | Δ pooled | worst fold | ships? | slices (debut / womens / five_round / external_missing) |
  |---|---|---|---|---|---|---|
  | `rankings` | everything | 0.6493 | **+0.0017** | +0.0037 | no | -0.0019 / +0.0009 / +0.0032 / +0.0009 |
  | `rankings_noflags` | no fight-level flags | 0.6491 | **+0.0015** | +0.0041 | no | -0.0013 / +0.0025 / +0.0009 / +0.0018 |
  | `rankings_diffonly` | `rank_diff` only | 0.6487 | **+0.0011** | +0.0035 | no | +0.0011 / +0.0004 / +0.0058 / +0.0040 |

  `bar_check` on every one: `clears_delta: false`, `no_fold_regression: true`,
  **`ships: false`**. Per-fold for the best variant (`diffonly`): 2018 +0.0018,
  2019 +0.0035, 2020 +0.0009, 2021 -0.0018, 2022 +0.0030, 2023 -0.0025,
  2024 -0.0001, 2025 +0.0029.
- **The shape of the result is the third repetition of the same lesson.** XGB
  is roughly neutral (-0.0007 at best) and torch is uniformly worse, exactly as
  for `in_fight` and `opponent_adjusted`: a tree ensemble can ignore a
  correlated column that is NaN on 86% of rows, while the MLP has to impute a
  median for it on every one of those rows and spend capacity on the result.
  Notably the ordering *within* the rankings variants is the reverse of
  `external`'s -- the fewer rankings columns the model sees, the less it loses
  -- which is what "the columns are correlated with what is already there"
  looks like from the other side.
- Gates before evaluation, all passing on the rankings table (11,238 x 60):
  `test_no_leakage_truncation_invariance` and both serving-parity tests; the
  base+external rebuild afterwards stayed byte-identical.
- Reports kept: `models/walkforward/{xgb,torch}_rankings.json`,
  `..._rankings_noflags.json`, `..._rankings_diffonly.json`. Block registration
  reverted (left commented in `mma.feature_blocks` with the numbers) along with
  the `features.py` / `inference.py` wiring.
- **Kept and reusable:** `scripts/build_rankings.py`, the committed
  `data/external/rankings.parquet` and `RANKINGS.md`, and `src/mma/rankings.py`
  with `tests/test_rankings.py` -- including the serving path, which needs only
  the two ids, the division and the date and so works for a future card. A
  later block that can widen the coverage (a continuous divisional standing for
  everyone rather than a top-15 flag) has the join and the point-in-time rule
  already written and tested.

**Gate improvement kept from this task (applies to every future block).**
`tests/test_serving_parity.py::_target_fight` now also requires the target
fight's committed row to have NO NaN in any enabled block's declared columns
(`_fully_populated_fights`). The value comparison skips a column that is NaN on
both sides, so a partially-covered block -- `external` populates
`pre_ufc_finish_loss_rate_diff` on 30% of rows, `rankings` populates
`rank_diff` on 14% -- made the `compared == expected` assertion fail on a count
rather than naming the column, and would previously have let a genuinely
unpopulated column pass unnoticed on a luckier draw.

**Secondary source (Task 10).** `scripts/refresh_secondary.py` adapts the
daily-refreshed ufcstats scrape at https://github.com/Greco1899/scrape_ufc_stats
(GPL-3.0, used as a **data** source only -- the published CSVs are fetched over
plain HTTPS; no GPL code is vendored). Files read: `ufc_event_details.csv`
(EVENT/URL/DATE/LOCATION), `ufc_fight_results.csv`
(EVENT/BOUT/OUTCOME/WEIGHTCLASS/METHOD/ROUND/TIME/TIME FORMAT/REFEREE/DETAILS/URL),
`ufc_fight_stats.csv` (per round, per fighter: KD, SIG.STR. "45 of 118", TD,
CTRL "4:20", HEAD/BODY/LEG/DISTANCE/CLINCH/GROUND), `ufc_fighter_details.csv`
and `ufc_fighter_tott.csv` (name -> fighter URL). Stable ufcstats ids are the
16-hex tail of every URL. The adaptation reshapes into the Kaggle `master.csv`
column names and then calls `mma.dataset.build_fights` / `build_fight_stats`,
so winner codes, method mapping and `duration_sec` are the tested ones by
construction rather than a second implementation that could drift.

Live dry run, 2026-09-08: **8,899 source rows -> 8,820 fights through
2026-09-05** (25 duplicate fight ids from co-branded cards listed under two
event names; 54 rows dropped because the bout string's name is ambiguous or
unspelt in the source's own fighter files). Overlap with our table 8,768
fights; reconciliation via `scripts/reconcile_sources.py::reconcile` gives
**winner agreement 0.9999**, method 0.9999, date 0.9998, and 1.0000 on
`fighter_a_id`/`fighter_b_id`/`finish_round`/`scheduled_rounds`/`weight_class`
-- corner order matches ours, so no swap is needed. Below a 0.99 winner-agreement
floor the script refuses to merge and exits non-zero.

**52 fights exist beyond our 2026-08-08 Kaggle cutoff** (4 events: 2026-08-15,
-08-22, -08-29, -09-05). 44 merge cleanly; **8 are rejected** because they
involve one of **9 UFC debutants absent from `fighters.parquet`** -- we cannot
build features for a fighter with no biographical row or history. Merging would
move the table's max date from 2026-08-08 to 2026-09-05, i.e. **28 days fresher**.

Two surprises worth recording. (1) The premise "same ids, so no name matching"
only half holds: the fight tables key on the bout string, not on fighter URLs,
so resolving a corner to an id needs a *within-source* name lookup against
ufcstats' own two fighter files -- which themselves disagree on a few spellings
("Zach Reese"/"Zachary Reese"). Both files are indexed and any name mapping to
more than one id is dropped rather than guessed. (2) The scrape covers 787
events to our 1,259: it is *fresher* but not a superset, so it can only ever be
an append-on-top source, never a replacement.

**Writes are off by default and report-only in CI** (`--enable` required;
`.github/workflows/refresh-data.yml` runs `--dry-run` with
`continue-on-error: true` after the Kaggle refresh and before `make_dataset.py`).

> **Superseded 2026-09-09.** All four prerequisites below are resolved and the
> source is now a standing stage of every `make_dataset.py` rebuild rather than
> a write to enable; `refresh_secondary.py` no longer writes at all. The one
> that mattered was (b): the collision was *dissolved* rather than patched, by
> merging on every rebuild so that no rebuild can drop what a previous one
> added. Resolving it that way settled the other three — see the SP4 section of
> `docs/superpowers/specs/2026-09-06-predictor-v3-simulator-design.md`. Two
> notes for the record: (a) closed for all 9 debutants, so **52 fights merged
> rather than 44**; and (d) turned out not to be optional after all, because 44
> fights without a per-round record would have taken `make_dataset.py`'s
> modern round-coverage check from 100% to 99.36%, below its 99.5% bar. The
> survey (d) asked for is confirmed: nothing shipped reads `round_stats`.

**To enable writes (SP4)** needed four things. (a) An adaptation of
`ufc_fighter_tott.csv` into `fighters.parquet` so debutants stop being
rejected -- otherwise the freshest card is exactly the one that merges worst.
(b) A resolution of the collision with `make_dataset.py`: it rebuilds
`fights.parquet` wholesale from Kaggle and its regression guard fails when the
new table *drops* any fight the committed one has, so the very next weekly run
after a secondary write would go red on `n_dropped_by_new: 44` until Kaggle
catches up. Either the write path re-runs after every rebuild, or the appended
rows live in their own table that `make_dataset.py` knows to re-apply.
(c) A provenance column on the fights table, so a secondary-sourced row is
distinguishable in the track record and in any post-hoc audit.
(d) The per-round table: this script adapts `fights` and `fight_stats` only, so
`round_stats.parquet` would lag `fights.parquet` for the gap-filled fights --
harmless today (no shipped block reads it) but a silent hole for any future
per-round feature.
**Task 13 (assemble, redeploy, document, merge).**

**Shipped set: `base,external`** -- `data/processed/features.parquet` is
11,238 x 54 with the sidecar `{"blocks": ["base","external"]}`. Six pre-UFC
differentials are modelled; `external_missing` and `same_country` live in the
table and are excluded from both model matrices (`mma.tensors.DROPPED`,
`mma.models.xgb.MODEL_EXCLUDED`).

**Fresh-seed re-score (the locked rule's number): -0.0043.** The paired
incumbent was built first, exactly as `torch_v1_extslice` was: the v1 recipe
on the CURRENT 54-column table with all eight external columns held out of
the model, seeds 5-9, as `torch_v1_seeds5_extslice`. It reproduces
`torch_v1_seeds5.json` **bit-exactly** -- pooled, all eight folds, all three
shared slices and every `fit_info` entry -- differing only in `name` and in
carrying the extra `external_missing` slice, which is what makes it a valid
paired stand-in rather than a regenerated incumbent. Against it,
`torch_external_diffsonly_seeds5` scores pooled **0.6473** vs **0.6516**,
delta **-0.0043**, worst fold +0.0006 (2025); `clears_delta: true`,
`no_fold_regression: true`, **`ships: true`**. Fold deltas: 2018 -0.0056,
2019 -0.0128, 2020 -0.0094, 2021 -0.0033, 2022 -0.0065, 2023 -0.0014,
2024 -0.0006, 2025 +0.0006. Slices: debut -0.0068, womens -0.0018,
five_round +0.0011, external_missing +0.0002. The block clears the bar on
seeds it was never chosen on, by MORE than it did on seeds 0-4 (-0.0034),
and the `external_missing` slice cost is smaller on fresh seeds (+0.0002
against +0.0018) -- i.e. that cost is at the edge of seed noise.

**Deployment budget, re-derived on the shipped table.** The train scripts'
`BUDGET`/`TEMPERATURE`/`REPORT` were derived from reports computed on the v1
46-column table, and a budget belongs to a feature table, not to a recipe. So
protocol A and protocol B were both re-run on the shipped table with the
shipped `--drop-columns external_missing,same_country`:

| candidate | A (early stopping) | B (fixed budget, all data) | delta | gate |
|---|---|---|---|---|
| xgb | `xgb_v3` 0.6506 | `xgb_v3_refit` 0.6507 | +0.0001 | passes |
| torch | `torch_v3` 0.6476 | `torch_v3_refit` **0.6470** | **-0.0006** | passes |

Both inside sigma_seed, so the pre-registered rule ships B again:
`deployment_recipe: refit_through_latest`. Fresh-seed re-scoring of the same
B-vs-A pair (seeds 5-9: `torch_external_diffsonly_seeds5` vs
`torch_v3_refit_seeds5`) gives +0.0002, `verdict: confirmed`. New budgets:
**XGB {winner 105, method 61, round 75} trees; torch 10 epochs at temperature
1.07** (against the v1 table's 82/80/76 and 14 epochs at 1.1). Written to
`models/walkforward/refit_decision_v3.json`.

`scripts/refit_decision.py` hard-coded the v1 report paths, so it grew a
`--reports` argument over a `REPORT_SETS` registry: each set names its four
A/B reports plus the fresh-seed pair, records the feature table it was
computed on, and writes its own decision file. `refit_decision.json`
regenerates byte-for-byte apart from the new `feature_table` field.
`deployment_recipe` is now COMPUTED from the torch gate instead of asserted,
so the file cannot record a recipe its own numbers do not support.
`tests/test_refit_decision.py` (6 tests) pins the registry, the
reproducibility of both committed decision files, and both branches of the
recipe rule.

**Redeploy.** `train_xgb.py` / `train_torch.py` refit-mode defaults updated to
the v3 budget and reports; `build_display_priors.py` rerun. New model hash
**`b617b96dae45`** (was `40df77ec43c7`). `Ensemble.load()` returns 5 nets at
temperature 1.07 each; the deployed `models/torch/preprocess.json` now carries
41 numeric columns including the six `pre_ufc_*`/`days_since_pro_debut`
differentials and NOT `external_missing`/`same_country`. Determinism checked
rather than assumed: retraining both learners into a scratch directory
reproduces all eleven artifacts byte-for-byte. `tests/test_roll_window.py`
green (the split-protocol path is untouched); suite **532 passed, 1 skipped**
(526 before, +6 from `test_refit_decision.py`).

**A real serving bug this step exposed.** `inference.build_matchup` defaulted
to `blocks=(BASE_BLOCK,)`, and no serving caller overrode it -- so the app,
`prospective.predict_fight` and the explainer would all have built rows
missing the six external columns the newly deployed preprocessor asks for.
The suite failed loudly (12 tests) the moment the artifacts were retrained,
which is the parity discipline working, but the fix is the point: the default
is now `feature_blocks.table_blocks()`, which reads the sidecar
`scripts/build_features.py` writes next to the table. The serving contract is
therefore the table on disk, not a tuple each caller has to remember, and
`tests/conftest.py::table_blocks` becomes a re-export of it rather than a
second implementation. Fixtures that stood in for bio rows now carry real
ufcstats ids, because the external block joins on the bio row's index label.

**End-to-end exercise, actually run rather than reasoned about.** Through the
same code path `prospective.predict_fight` uses, with the real committed
ensemble and `as_of` a future date:
- mapped vs UNmapped corner (Alex Perez `ab2b4ff41d6ebe0f` vs Josh Hokit
  `955da1675ad58a50`): the six differentials come back NaN with
  `external_missing=True`, `same_country=False`, no crash, and
  `p_a_wins = 0.2606` with a full method distribution;
- mapped vs mapped (Perez vs Adam Fugitt): real values flow through
  (`pre_ufc_wins_diff` 9.0, `days_since_pro_debut_diff` 1982.0,
  `external_missing=False`), `p_a_wins = 0.6952`.
The app boots headless (`/_stcore/health` -> 200) on the redeployed
artifacts.

**Shipped set:** `base,external` (11,238 x 54; six pre-UFC differentials
modelled, two coverage flags in the table and out of both model matrices) ;
fresh-seed re-score: **-0.0043** pooled (0.6516 -> 0.6473, seeds 5-9, worst
fold +0.0006, `ships: true`) ; deployed model hash: **`b617b96dae45`**.

> **FORWARD POINTER, added 2026-09-08 (SP2.2).** Everything above is SP2's
> record and is left exactly as it was written. Two statements in it are no
> longer descriptions of the deployed system:
>
> * **The shipped feature table is no longer `base,external` (11,238 x 54).**
>   It is `base,external,trajectory,notice,context,opponent_adjusted`
>   (11,238 x 87). SP2.2's blend candidate B1 was scored on that table and
>   shipped, so the four blocks SP2 rejected and SP2.1 rejected again are
>   registered permanently -- **not** because any of them cleared a bar
>   (none did, and those verdicts stand unedited), but because the blend that
>   cleared was scored on the table containing them. B1 against the same
>   blend on `base,external` is -0.0016, which clears nothing on its own.
>   Three columns joined the two coverage flags in the "in the table, out of
>   both model matrices" set: `notice_unknown`, `home_country_a`,
>   `home_country_b`.
> * **The deployed scorer is no longer the torch ensemble, and the hash is
>   no longer `b617b96dae45`.** It is a 0.5/0.5 blend of a 5-seed XGBoost
>   ensemble and the 5-seed torch ensemble, temperature-scaled after
>   averaging (T = 0.85), hash **`b863389f1760`** (`5aa33460ef40` when this
>   note was written; a pre-merge fix on the same branch moved the 0.5/0.80
>   above from module constants into a committed, hashed `models/blend.json`
>   -- same numbers, same predictions, `6207d19d615b` -- and a second one
>   replaced that 0.80 with the walk-forward temperature 0.85, which does
>   change every prediction) -- which now covers the XGBoost
>   artifacts as well. The deployment budgets in this record
>   (105/61/75 trees, 10 epochs at T 1.07) belong to the `base,external`
>   table and were re-derived on the new one (6 epochs at T 1.15;
>   109/73/71 trees per seed, `models/walkforward/refit_decision_b1.json`).
>
> See `docs/superpowers/plans/2026-09-08-sp2-2-blend-experiment.md` and
> `models/walkforward/sp2_2_decision.json`. Follow-ups 1, 6 and 7 below are
> all still live and are carried forward there.

**Follow-ups (carried into SP3/SP4):**

1. **The external snapshot is static and decaying.** `ehan03/jds-mma-data`
   covers UFC events to 2024-12-14; `external_missing` is 0.201 of all rows
   but 0.359 of 2025 and 0.534 of 2026, and the block's gain follows
   (-0.0047 row-weighted on 2018-2023 against -0.0006 on 2024-2025). Without
   a refreshable source -- or a live scrape of the pre-UFC record --
   `scripts/build_external.py` will drift the only shipped block to neutral.
2. **Enabling secondary-source writes needs four things** (Task 10):
   (a) an adaptation of `ufc_fighter_tott.csv` into `fighters.parquet`, or
   the freshest card is the one that merges worst (8 of 52 gap fights are
   rejected today for 9 unknown debutants); (b) a resolution of the collision
   with `make_dataset.py`, which rebuilds `fights.parquet` wholesale from
   Kaggle and whose regression guard fails when the rebuild drops rows a
   secondary write added; (c) a provenance column on the fights table;
   (d) `round_stats.parquet`, which this adapter does not fill.
3. **`mma.wiki_cards.parse_background` is built and fixture-tested but not
   wired into `prospective.predict_event`.** It is the only route to
   withdrawals and missed weight for a FUTURE event; it stayed unwired
   because the `notice` block did not ship. A live source for those facts
   would change the block's verdict, since its raw marginal is strong (a
   corner on <=7 days' notice wins 0.264) and its problem is purely coverage.
4. **Three rankings name-match misses are mechanical.** `Jan Błachowicz`,
   `Klaudia Syguła` (stroked Latin letters, which NFKD does not decompose the
   way it decomposes an accent) and `Lone'er Kavanagh` (curly vs straight
   apostrophe) would close by extending `mma.prospective.fold_accents` with a
   stroked-letter map and apostrophe normalisation, without weakening the
   never-guess rule. Left alone here because `fold_accents` is on the live
   prediction path and the `rankings` block did not ship.
5. **One row carries a `dob` data error.** Fight `92961925688cd2d6`
   (2003-05-16) has `age_b` = 4.61. Harmless for the base block's linear age
   differential -- it is one row of 11,238, pre-dating every fold year -- but
   it would matter to anything quadratic in age (the `trajectory` block's
   `age_squared`, or an SP3 hazard model with an age term), so a sanity bound
   on `dob` belongs in `scripts/make_dataset.py` before such a feature ships.
6. **Recency and temperature drift** (Task 9 Step 3), unchanged: the
   recent-fold budget/temperature pays on 2022-2025 and costs on 2018-2021,
   netting +0.0002 pooled. The honest test is a recency-weighted headline
   metric or more fold years, not a tighter read of the same eight numbers.
7. **`roll_window.py`'s promotion gate** still scores a newest-2-years slice
   that is in-sample for a refit-through-latest incumbent, so `--execute`
   aborts. SP4 moves it onto the walk-forward harness. (2026-09-08: SP2.2
   added a *second*, earlier abort -- the gate compares a torch-only
   candidate against a torch-only incumbent, which is half the served model
   now -- so the promotion path is fully inert until SP4 does this.)
