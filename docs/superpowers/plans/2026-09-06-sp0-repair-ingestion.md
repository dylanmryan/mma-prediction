# SP0: Repair and Foundations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pipeline ingest the rebuilt Kaggle dataset (new file layout, +3,104 fights, per-round stats), make the prospective track record key on model artifacts instead of git HEAD, rebuild every artifact on the fuller data, and grade the 78 pending predictions.

**Architecture:** `src/mma/dataset.py` keeps its three output tables (`fighters`, `fights`, `fight_stats`) with identical columns and dtypes so every downstream module (Elo, history, features, models, app) is untouched, and gains two new tables (`round_stats`, `bonuses`). Scripts that read raw CSVs (`download_data.py`, `refresh_data.py`, `make_dataset.py`) switch to the new files. A new `src/mma/versioning.py` hashes the deployed model artifacts; `predict_upcoming.py` uses it and a one-time migration re-keys existing prediction records.

**Tech Stack:** Python 3.11, pandas, pyarrow, pytest, kagglehub, xgboost, torch (CPU). All commands run from a **local-disk venv** at `~/.venvs/mma` (the repo's own `.venv` lives under iCloud and stalls on libtorch page-ins). Create it once with:

```bash
python3 -m venv ~/.venvs/mma && ~/.venvs/mma/bin/pip install -e ".[dev,app]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Every `pytest`/`python` below means `~/.venvs/mma/bin/pytest` / `~/.venvs/mma/bin/python`, run from the repo root with `OMP_NUM_THREADS=1` exported (torch+xgboost OpenMP clash; `tests/conftest.py` pins it for tests, scripts need it in the environment).

**Commit convention:** plain messages, no attribution trailers (user preference). Before every commit run `git status` and inspect the staged list; an empty or mass-deletion index after an iCloud lock cleanup has happened before.

---

## Verified facts about the new source (2026-09-06)

Downloaded snapshot of `neelagiriaditya/ufc-datasets-1994-2025` (Kaggle `lastUpdated` 2026-08-11) contains seven CSVs:

| File | Rows | Use |
|---|---|---|
| `master.csv` | 11,441 × 93 | one row per fight, joined with event + both fighters' profiles + **fight-total stats** (`r_total_*`, `b_total_*`, `r_total_ctrl_seconds`). Closest analog of the old `UFC.csv`. Source for `fights` and `fight_stats`. |
| `fighter.csv` | 4,581 × 16 | `fighter_id, fighter_name, fighter_nick_name, height ("5' 10\""), weight_lbs, reach_inches, stance, dob (ISO), slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg` (career cols are as-of-scrape → dropped). Source for `fighters`. |
| `round.csv` | 25,131 × 48 | one row per fight per round: `fight_id, round_no, r_id, b_id`, then per corner `kd, sig_landed, sig_atmp, total_str_landed, total_str_atmp, td_success, td_atmp, sub_att, rev, ctrl ("m:ss"), sig_str_landed_{head,body,leg,distance,clinch,ground}, sig_str_atmp_{...}`. Source for `round_stats`. |
| `fighter_bonus.csv` | 2,409 × 2 | `fight_id, bonus_type` (Performance/Fight/Knockout/Submission of the Night). Source for `bonuses`. |
| `fight.csv`, `event.csv` | 11,441 / 1,259 | normalized versions of what `master.csv` already joins; not read. |
| `scrape_error.csv` | 330 | upstream scraper log; not read. |

`master.csv` columns used: `fight_id, event_id, event_date, event_location, weight_class, title_fight (0/1), winner_id, result_status (win/draw/no_contest), method, finish_round (last round fought), finish_time ("m:ss"), time_format ("3 Rnd (5-5-5)", "No Time Limit"), referee, r_fighter_id, b_fighter_id, r_total_*, b_total_*, rounds_fought`.

Reconciliation against the old `UFC.csv` (8,337 fights): **all 8,337 old fight ids exist in the new file; winner_id, corner ids, and last round fought agree on 100% of them; method strings agree on 100%.** The new file adds 3,104 fights: 2025-09 → 2026-08-08 (585) plus **50–80 fights per year from 2012 onward that the old scrape had silently dropped**. 330 fights (all ≤2013) have no round rows; one fight has fewer round rows than rounds fought. 41 fights are "No Time Limit" (→ `scheduled_rounds` NA, as before).

---

## File map

| Path | Change | Responsibility |
|---|---|---|
| `src/mma/versioning.py` | create | `model_version(root)`: sha256 of deployed model artifacts → 12-hex string |
| `tests/test_versioning.py` | create | hash is stable, changes with artifact bytes, ignores non-artifacts |
| `scripts/predict_upcoming.py` | modify | use `model_version()` instead of git HEAD sha |
| `scripts/migrate_model_versions.py` | create | one-time re-key of `predictions/*.json` to an artifact hash |
| `scripts/download_data.py` | modify | new file list + `verify_raw_files` guard |
| `tests/test_download_guard.py` | create | guard raises listing missing files |
| `src/mma/dataset.py` | rewrite builders | `build_fighters(fighter.csv)`, `build_fights(master.csv)`, `build_fight_stats(master.csv)`, `build_round_stats(round.csv)`, `build_bonuses(fighter_bonus.csv)` |
| `tests/test_dataset_fighters.py`, `tests/test_dataset_fights.py`, `tests/test_dataset_stats.py` | rewrite fixtures | new raw schema |
| `tests/test_dataset_rounds.py` | create | round_stats + bonuses builders |
| `scripts/make_dataset.py` | modify | read new files, write 5 parquet tables, integrity checks |
| `scripts/refresh_data.py` | modify | compare on `master.csv` |
| `scripts/reconcile_sources.py` | create | old-vs-new processed fights report (provenance) |
| `tests/test_processed_data.py` | modify | volume/date expectations |
| `tests/test_processed_rounds.py` | create | round_stats parquet sanity |
| `data/processed/*.parquet`, `models/**` | regenerate | rebuilt on the fuller data |
| `predictions/*.json`, `predictions/track_record.json` | regenerate | re-keyed versions + first graded fights |
| `README.md` | modify | data section, results numbers, development section |
| `models/market_benchmark 2.json` + 4 other `* 2.*` files | delete | iCloud sync duplicates (verified byte-identical) |

---

### Task 1: Branch, housekeeping, baseline

**Files:**
- Delete: `models/market_benchmark 2.json`, `scripts/build_odds_benchmark 2.py`, `src/mma/odds 2.py`, `tests/test_market_benchmark 2.py`, `tests/test_odds 2.py`

- [ ] **Step 1: Create the branch from up-to-date main**

```bash
git checkout main && git status --short && git checkout -b sp0-repair
```
Expected: only the five `?? ... 2.*` untracked files listed; branch `sp0-repair` created.

- [ ] **Step 2: Confirm the duplicates are byte-identical, then delete them**

```bash
cmp "models/market_benchmark 2.json" models/market_benchmark.json && cmp "scripts/build_odds_benchmark 2.py" scripts/build_odds_benchmark.py && cmp "src/mma/odds 2.py" src/mma/odds.py && cmp "tests/test_market_benchmark 2.py" tests/test_market_benchmark.py && cmp "tests/test_odds 2.py" tests/test_odds.py && echo IDENTICAL
rm "models/market_benchmark 2.json" "scripts/build_odds_benchmark 2.py" "src/mma/odds 2.py" "tests/test_market_benchmark 2.py" "tests/test_odds 2.py"
```
Expected: `IDENTICAL`, then `git status --short` prints nothing. (They were untracked, so nothing to commit.)

- [ ] **Step 3: Run the baseline suite from the local venv**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q -x 2>&1 | tail -3
```
Expected: `250 passed, 1 skipped` (or close; record the exact numbers in this plan's completion notes). If a first-import stall occurs, wait; it is the iCloud page-in, not a failure.

---

### Task 2: Model version from artifact hash

**Files:**
- Create: `src/mma/versioning.py`
- Create: `tests/test_versioning.py`
- Modify: `scripts/predict_upcoming.py:78-83` and `:106`

- [ ] **Step 1: Write the failing tests**

`tests/test_versioning.py`:
```python
"""model_version() must depend only on deployed model artifact bytes."""
import json

import pytest

from mma.versioning import MODEL_ARTIFACT_GLOBS, model_version


def _make_models(root, seed_bytes=b"seed0"):
    torch_dir = root / "models" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "net_seed0.pt").write_bytes(seed_bytes)
    (torch_dir / "preprocess.json").write_text(json.dumps({"medians": {}}))
    (torch_dir / "display_priors.json").write_text("{}")
    (root / "models" / "xgb_winner.json").write_text("{}")
    (root / "models" / "xgb_method.json").write_text("{}")
    (root / "models" / "xgb_round.json").write_text("{}")


def test_version_is_12_hex_and_stable(tmp_path):
    _make_models(tmp_path)
    first = model_version(tmp_path)
    assert len(first) == 12 and int(first, 16) >= 0
    assert model_version(tmp_path) == first


def test_version_changes_when_a_weight_file_changes(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "torch" / "net_seed0.pt").write_bytes(b"seed0-retrained")
    assert model_version(tmp_path) != before


def test_version_ignores_non_artifact_files(tmp_path):
    _make_models(tmp_path)
    before = model_version(tmp_path)
    (tmp_path / "models" / "torch" / "metrics_val.json").write_text('{"acc": 1}')
    (tmp_path / "models" / "market_benchmark.json").write_text("{}")
    assert model_version(tmp_path) == before


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_version(tmp_path)


def test_globs_cover_torch_and_xgb():
    assert any("net_seed" in g for g in MODEL_ARTIFACT_GLOBS)
    assert any("xgb_winner" in g for g in MODEL_ARTIFACT_GLOBS)
```

- [ ] **Step 2: Run to verify failure**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_versioning.py -q
```
Expected: `ModuleNotFoundError: No module named 'mma.versioning'`.

- [ ] **Step 3: Implement `src/mma/versioning.py`**

```python
"""Deployed-model identity for the prospective track record.

The track record must split by *model*, not by git commit: the weekly
Action commits predictions every week, so a HEAD sha changes weekly while
the model does not. Hashing the artifact bytes that inference actually
loads gives a version that changes exactly when the model changes
(retrain, promotion, display-prior regeneration).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ARTIFACT_GLOBS = (
    "models/torch/net_seed*.pt",
    "models/torch/preprocess.json",
    "models/torch/display_priors.json",
    "models/xgb_winner.json",
    "models/xgb_method.json",
    "models/xgb_round.json",
)


def model_version(root: Path) -> str:
    """12-hex sha256 prefix over the sorted (path, bytes) of every artifact."""
    root = Path(root)
    paths = sorted(
        {path for pattern in MODEL_ARTIFACT_GLOBS for path in root.glob(pattern)}
    )
    if not paths:
        raise FileNotFoundError(f"no model artifacts under {root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_versioning.py -q
```
Expected: `5 passed`.

- [ ] **Step 5: Use it in `scripts/predict_upcoming.py`**

Delete the `_git_short_sha` function (lines 78-83) and the now-unused `import subprocess` (line 16). Replace line 106 `model_version = _git_short_sha()` with:

```python
    model_version = artifact_model_version(ROOT)
```
and add, next to the other `mma` imports inside `main()` (line 86 area, where `from mma.inference import Ensemble` lives):

```python
    from mma.versioning import model_version as artifact_model_version
```

Update the module docstring sentence "predicts each matched fight with the committed ensemble" to add: "Each record is stamped with `model_version`, a hash of the deployed artifacts (see `mma.versioning`), so the track record splits by model, not by commit."

- [ ] **Step 6: Verify the script still imports and the suite is green**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python -c "import runpy; m = runpy.run_path('scripts/predict_upcoming.py', run_name='not_main'); print('ok', 'select_upcoming_events' in m)"
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_predict_upcoming.py tests/test_versioning.py -q
```
Expected: `ok True`; all passed.

- [ ] **Step 7: Commit**

```bash
git add src/mma/versioning.py tests/test_versioning.py scripts/predict_upcoming.py
git status --short
git commit -m "Stamp prospective predictions with a model-artifact hash, not the git HEAD sha"
```

---

### Task 3: Re-key existing prediction records (before any retrain)

The 78 committed predictions were all produced by the artifacts currently at HEAD (`models/` last changed 2026-07-13, first prediction 2026-07-15). This task must run **before Task 8 retrains**, so the hash still describes the model that made those predictions.

**Files:**
- Create: `scripts/migrate_model_versions.py`
- Modify: `predictions/*.json`, `predictions/track_record.json`

- [ ] **Step 1: Write the migration script**

`scripts/migrate_model_versions.py`:
```python
"""One-time migration: re-key prediction records to an artifact-hash model_version.

Before this migration, `model_version` was the git HEAD short sha at
prediction time, which changed with every weekly commit even though the
model never did. Every record carrying one of the old commit shas was
produced by the same artifacts, so they all collapse to that single
artifact hash. Idempotent: values already equal to `--to` are left alone.

Usage:
    python scripts/migrate_model_versions.py --to <12-hex hash> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "predictions"
_GIT_SHORT_SHA = re.compile(r"^[0-9a-f]{7}$")


def rekey_record(record: dict, new_version: str) -> int:
    """Replace every git-sha-shaped model_version in place; return count changed."""
    changed = 0
    if _GIT_SHORT_SHA.match(str(record.get("model_version", ""))):
        record["model_version"] = new_version
        changed += 1
    for fight in record.get("fights", []):
        if _GIT_SHORT_SHA.match(str(fight.get("model_version", ""))):
            fight["model_version"] = new_version
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="artifact hash from mma.versioning")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{12}", args.to):
        raise SystemExit(f"--to must be a 12-hex artifact hash, got {args.to!r}")

    total = 0
    for path in sorted(args.predictions_dir.glob("*.json")):
        if path.name == "track_record.json":
            continue
        record = json.loads(path.read_text())
        changed = rekey_record(record, args.to)
        total += changed
        print(f"{path.name}: {changed} value(s) re-keyed")
        if changed and not args.dry_run:
            path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"total re-keyed: {total}{' (dry run)' if args.dry_run else ''}")
    print("now run scripts/grade_predictions.py to regenerate track_record.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Unit-test `rekey_record`**

Append to `tests/test_versioning.py`:
```python
from scripts.migrate_model_versions import rekey_record


def test_rekey_replaces_git_shas_and_leaves_hashes():
    record = {
        "model_version": "79135ef",
        "fights": [
            {"model_version": "e47f720", "p_a_wins": 0.6},
            {"model_version": "abcdef012345", "p_a_wins": 0.4},
            {"skipped": True},
        ],
    }
    assert rekey_record(record, "abcdef012345") == 2
    assert record["model_version"] == "abcdef012345"
    assert record["fights"][0]["model_version"] == "abcdef012345"
    assert record["fights"][1]["model_version"] == "abcdef012345"
    assert rekey_record(record, "abcdef012345") == 0  # idempotent
```
(`tests/conftest.py` already puts the repo root on `sys.path`, which is how `tests/test_refresh.py` imports `scripts.refresh_data`.)

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_versioning.py -q
```
Expected: `6 passed`.

- [ ] **Step 3: Compute the current artifact hash and migrate**

```bash
HASH=$(OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python -c "from pathlib import Path; from mma.versioning import model_version; print(model_version(Path('.')))"); echo "$HASH"
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/migrate_model_versions.py --to "$HASH" --dry-run
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/migrate_model_versions.py --to "$HASH"
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/grade_predictions.py
```
Expected: the hash printed; dry run and real run report the same non-zero total; `grade_predictions.py` regenerates `track_record.json` with **exactly one** entry under `model_versions` (78 predicted, 0 graded, because the processed data still ends 2025-09-06 at this point). Record the hash in the completion notes: it is "model v1".

- [ ] **Step 4: Verify and commit**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python -c "import json; t = json.load(open('predictions/track_record.json')); print(list(t['model_versions']), t['overall'])"
git add scripts/migrate_model_versions.py tests/test_versioning.py predictions/
git status --short
git commit -m "Re-key existing prediction records to the v1 artifact hash"
```
Expected: one key; `n_predicted` 78.

---

### Task 4: Download script for the new file layout

**Files:**
- Modify: `scripts/download_data.py`
- Create: `tests/test_download_guard.py`

- [ ] **Step 1: Write the failing test**

`tests/test_download_guard.py`:
```python
import pytest

from scripts.download_data import REQUIRED_FILES, verify_raw_files


def test_all_required_files_present_passes(tmp_path):
    for name in REQUIRED_FILES:
        (tmp_path / name).write_text("x")
    verify_raw_files(tmp_path)  # no raise


def test_missing_file_raises_with_names(tmp_path):
    (tmp_path / "master.csv").write_text("x")
    with pytest.raises(FileNotFoundError) as excinfo:
        verify_raw_files(tmp_path)
    assert "round.csv" in str(excinfo.value)
    assert "master.csv" not in str(excinfo.value).split("missing")[-1]


def test_required_files_are_the_new_layout():
    assert set(REQUIRED_FILES) == {"master.csv", "fighter.csv", "round.csv", "fighter_bonus.csv"}
```

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_download_guard.py -q
```
Expected: `ImportError: cannot import name 'REQUIRED_FILES'`.

- [ ] **Step 2: Rewrite `scripts/download_data.py`**

```python
"""Download the Kaggle UFC dataset into data/raw/ and report its schema.

Dataset: https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025
(a ufcstats.com scrape; rebuilt 2026-08-11 with a new layout). Files used
downstream: master.csv (one row per fight with fight-total stats),
fighter.csv, round.csv (per-round stats), fighter_bonus.csv. The other
files in the snapshot (fight.csv, event.csv, scrape_error.csv) are copied
for reference but not read.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "neelagiriaditya/ufc-datasets-1994-2025"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REQUIRED_FILES = ("master.csv", "fighter.csv", "round.csv", "fighter_bonus.csv")


def verify_raw_files(raw_dir: Path) -> None:
    """Fail loudly if the snapshot lacks any file the pipeline reads."""
    missing = [name for name in REQUIRED_FILES if not (Path(raw_dir) / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Kaggle snapshot layout changed: missing {missing} under {raw_dir}"
        )


def download(raw_dir: Path) -> None:
    """Download the Kaggle dataset snapshot and copy its CSVs into raw_dir."""
    cache_path = Path(kagglehub.dataset_download(DATASET))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for src in cache_path.rglob("*.csv"):
        dest = raw_dir / src.name
        shutil.copy2(src, dest)
        print(f"copied {src.name}")
    verify_raw_files(raw_dir)


def main() -> None:
    download(RAW_DIR)

    print("\n=== SCHEMA REPORT ===")
    for csv in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(csv, nrows=5, sep=None, engine="python")
        print(f"\n{csv.name}  ({len(df.columns)} cols)")
        print("  columns:", list(df.columns))
        print(df.head(2).to_string(max_colwidth=25))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests, then download for real**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_download_guard.py -q
rm -f data/raw/UFC.csv data/raw/fight_details.csv data/raw/event_details.csv data/raw/fighter_details.csv
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/download_data.py 2>&1 | grep -E "^copied|cols\)"
ls -la data/raw
```
Expected: `3 passed`; seven `copied ...` lines; `master.csv` (~6.3 MB), `round.csv` (~4.0 MB) present. (`data/raw/*.csv` is gitignored.)

- [ ] **Step 4: Commit**

```bash
git add scripts/download_data.py tests/test_download_guard.py
git status --short
git commit -m "Download the rebuilt Kaggle layout and guard on required files"
```

---

### Task 5: Fighters builder on `fighter.csv`

**Files:**
- Modify: `src/mma/dataset.py` (`build_fighters`)
- Rewrite: `tests/test_dataset_fighters.py`

- [ ] **Step 1: Rewrite the fixture and tests**

Replace `tests/test_dataset_fighters.py` entirely:
```python
import pandas as pd
import pytest

from mma.dataset import build_fighters


def _raw():
    return pd.DataFrame(
        {
            "fighter_id": ["jj", "cm", "xx"],
            "fighter_name": ["Jon Jones", " Conor McGregor ", "No Data"],
            "fighter_nick_name": ["Bones", "Notorious", None],
            "height": ["6' 4\"", "5' 9\"", None],
            "weight_lbs": [205.0, 155.0, None],
            "reach_inches": [84.5, 74.0, None],
            "stance": ["Orthodox", "Southpaw", None],
            "dob": ["1987-07-19", "1988-07-14", None],
            "slpm": [4.3, 5.3, 0.0],
            "str_acc": [57, 49, 0],
            "sapm": [2.2, 4.0, 0.0],
            "str_def": [64, 54, 0],
            "td_avg": [1.9, 0.7, 0.0],
            "td_acc": [45, 55, 0],
            "td_def": [95, 67, 0],
            "sub_avg": [0.5, 0.2, 0.0],
        }
    )


def test_schema_and_values():
    fighters = build_fighters(_raw())
    assert list(fighters.columns) == [
        "fighter_id", "name", "height_cm", "reach_cm", "stance", "dob",
    ]
    jj = fighters[fighters["fighter_id"] == "jj"].iloc[0]
    assert jj["name"] == "Jon Jones"
    assert jj["height_cm"] == pytest.approx(76 * 2.54)
    assert jj["reach_cm"] == pytest.approx(84.5 * 2.54)
    assert jj["stance"] == "Orthodox"
    assert jj["dob"] == pd.Timestamp("1987-07-19")
    cm = fighters[fighters["fighter_id"] == "cm"].iloc[0]
    assert cm["name"] == "Conor McGregor"  # stripped


def test_missing_stance_dob_height_stay_missing():
    fighters = build_fighters(_raw())
    xx = fighters[fighters["fighter_id"] == "xx"].iloc[0]
    assert pd.isna(xx["stance"]) and pd.isna(xx["dob"])
    assert pd.isna(xx["height_cm"]) and pd.isna(xx["reach_cm"])


def test_unparseable_height_is_missing():
    raw = _raw()
    raw.loc[0, "height"] = "--"
    fighters = build_fighters(raw)
    assert pd.isna(fighters[fighters["fighter_id"] == "jj"].iloc[0]["height_cm"])


def test_duplicate_ids_rejected():
    raw = _raw()
    raw.loc[1, "fighter_id"] = "jj"
    with pytest.raises(ValueError, match="duplicate fighter ids"):
        build_fighters(raw)


def test_missing_id_rejected():
    raw = _raw()
    raw.loc[1, "fighter_id"] = None
    with pytest.raises(ValueError, match="missing ids"):
        build_fighters(raw)


def test_leaky_career_columns_dropped():
    fighters = build_fighters(_raw())
    for leaky in ("slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc",
                  "td_def", "sub_avg", "weight_lbs", "fighter_nick_name"):
        assert leaky not in fighters.columns


def test_sorted_by_id():
    fighters = build_fighters(_raw())
    assert list(fighters["fighter_id"]) == ["cm", "jj", "xx"]
```

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_dataset_fighters.py -q
```
Expected: failures with `KeyError: 'id'` (old builder reads the old columns).

- [ ] **Step 2: Replace `build_fighters` in `src/mma/dataset.py`**

Update the module docstring's first lines to:
```python
"""Builders that turn the raw Kaggle UFC CSVs into clean tables.

Source schema (rebuilt 2026-08-11): neelagiriaditya/ufc-datasets-1994-2025 —
master.csv (one row per fight incl. fight-total stats), fighter.csv,
round.csv (per-round stats), fighter_bonus.csv. Stable ufcstats hex ids
throughout. See docs/superpowers/plans/2026-09-06-sp0-repair-ingestion.md.
"""
```
Add to the imports:
```python
from mma.labels import decision_subtype, map_method, parse_scheduled_rounds, parse_weight_class
from mma.parsing import parse_height_inches, parse_mmss_seconds
```
Replace the whole `build_fighters` function:
```python
_INCH_CM = 2.54


def build_fighters(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter: stable id + biographical fields only.

    Career-aggregate columns (slpm, td_avg, ...) and weight_lbs are dropped
    on purpose: they are as-of-scrape values and would leak the future if
    joined to historical fights. Height arrives as `5' 10"` text and reach
    as inches; both are converted to centimetres to keep the processed
    schema identical to the pre-2026 one.
    """
    ids = raw["fighter_id"].astype("string").str.strip()
    if ids.isna().any():
        raise ValueError(f"{int(ids.isna().sum())} fighter rows have missing ids")
    height_in = raw["height"].map(parse_height_inches)
    fighters = pd.DataFrame(
        {
            "fighter_id": ids,
            "name": raw["fighter_name"].astype("string").str.strip(),
            "height_cm": pd.to_numeric(height_in, errors="coerce") * _INCH_CM,
            "reach_cm": pd.to_numeric(raw["reach_inches"], errors="coerce") * _INCH_CM,
            "stance": raw["stance"].astype("string").str.strip(),
            "dob": pd.to_datetime(raw["dob"], format="mixed", errors="coerce"),
        }
    )
    if not fighters["fighter_id"].is_unique:
        duplicated = fighters.loc[fighters["fighter_id"].duplicated(), "fighter_id"]
        raise ValueError(f"duplicate fighter ids: {sorted(set(duplicated))[:5]}")
    return fighters.sort_values("fighter_id").reset_index(drop=True)
```

- [ ] **Step 3: Run tests**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_dataset_fighters.py tests/test_parsing.py -q
```
Expected: all passed (the `tests/test_dataset_fights.py` / `_stats.py` failures come next).

- [ ] **Step 4: Commit**

```bash
git add src/mma/dataset.py tests/test_dataset_fighters.py
git status --short
git commit -m "Build fighters from the new fighter.csv layout"
```

---

### Task 6: Fights builder on `master.csv`

**Files:**
- Modify: `src/mma/dataset.py` (`_winner_code`, `build_fights`)
- Rewrite: `tests/test_dataset_fights.py`

Output columns stay the original 13 **plus three appended context columns** `referee`, `event_id`, `location` (needed by SP2; harmless to every current consumer, which selects columns by name).

- [ ] **Step 1: Rewrite the fixture and tests**

Replace `tests/test_dataset_fights.py` entirely:
```python
import pandas as pd
import pytest

from mma.dataset import build_fights


def _raw_master():
    return pd.DataFrame(
        {
            "fight_id": ["f2", "f1", "f3", "f4"],
            "event_id": ["e2", "e1", "e3", "e3"],
            "event_date": ["2017-07-29", "2016-11-12", "2019-03-02", "2019-03-02"],
            "event_location": ["Anaheim, California, USA"] * 4,
            "weight_class": ["Light Heavyweight", "Lightweight", "Bout", "Women's Strawweight"],
            "title_fight": [1, 0, 0, 0],
            "r_fighter_id": ["jj", "cm", "aa", "ww"],
            "b_fighter_id": ["dc", "ed", "bb", "vv"],
            "winner_id": ["jj", None, None, "vv"],
            "result_status": ["win", "draw", "no_contest", "win"],
            "method": ["KO/TKO", "Decision - Majority", "Overturned", "Submission"],
            "finish_round": [3, 3, 2, 1],
            "finish_time": ["4:20", "5:00", "2:32", "0:45"],
            "time_format": ["5 Rnd (5-5-5-5-5)", "3 Rnd (5-5-5)", "No Time Limit", "3 Rnd (5-5-5)"],
            "referee": ["Herb Dean", "Marc Goddard", None, "Jason Herzog"],
        }
    )


def test_schema_order_and_sorting():
    fights = build_fights(_raw_master())
    assert list(fights.columns) == [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight", "duration_sec",
        "referee", "event_id", "location",
    ]
    assert list(fights["fight_id"]) == ["f1", "f2", "f3", "f4"]  # date-sorted, stable


def test_finish_fight_values():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f2"].iloc[0]
    assert row["fighter_a_id"] == "jj" and row["fighter_b_id"] == "dc"
    assert row["winner"] == "a"
    assert row["method"] == "ko_tko" and row["method_raw"] == "KO/TKO"
    assert row["finish_round"] == 3
    assert row["scheduled_rounds"] == 5
    assert row["weight_class"] == "Light Heavyweight"
    assert row["title_fight"] is True or row["title_fight"] == True  # noqa: E712
    assert row["duration_sec"] == (3 - 1) * 300 + 260
    assert row["referee"] == "Herb Dean"
    assert row["event_id"] == "e2"
    assert row["location"] == "Anaheim, California, USA"


def test_draw_and_decision_have_no_finish_round():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f1"].iloc[0]
    assert row["winner"] == "draw"
    assert row["method"] == "decision" and row["decision_subtype"] == "majority"
    assert pd.isna(row["finish_round"])
    assert row["duration_sec"] == 3 * 300  # went the distance


def test_no_contest_no_time_limit_and_noise_weight_class():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f3"].iloc[0]
    assert row["winner"] == "nc"
    assert pd.isna(row["method"])
    assert pd.isna(row["scheduled_rounds"])
    assert pd.isna(row["weight_class"])  # "Bout" carries no class
    assert row["duration_sec"] == (2 - 1) * 300 + 152  # raw-round fallback
    assert pd.isna(row["referee"])


def test_winner_b():
    fights = build_fights(_raw_master())
    row = fights[fights["fight_id"] == "f4"].iloc[0]
    assert row["winner"] == "b"
    assert row["method"] == "submission"
    assert row["finish_round"] == 1
    assert row["duration_sec"] == 45
    assert row["weight_class"] == "Women's Strawweight"


def test_win_status_with_unmatched_winner_id_is_nc():
    raw = _raw_master()
    raw.loc[0, "winner_id"] = "someone-else"
    fights = build_fights(raw)
    assert fights[fights["fight_id"] == "f2"].iloc[0]["winner"] == "nc"


def test_duplicate_fight_ids_rejected():
    raw = _raw_master()
    raw.loc[1, "fight_id"] = "f2"
    with pytest.raises(ValueError, match="fight_id"):
        build_fights(raw)


def test_missing_corner_id_rejected():
    raw = _raw_master()
    raw.loc[0, "b_fighter_id"] = None
    with pytest.raises(ValueError, match="missing corner"):
        build_fights(raw)


def test_string_dtypes():
    fights = build_fights(_raw_master())
    for column in ("winner", "method", "method_raw", "decision_subtype",
                   "weight_class", "referee", "event_id", "location"):
        assert fights[column].dtype == "string"
```

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_dataset_fights.py -q
```
Expected: failures with `KeyError: 'r_id'`.

- [ ] **Step 2: Replace `_winner_code` and `build_fights` in `src/mma/dataset.py`**

Delete `_NO_CONTEST_METHODS` and the old `_winner_code`; replace with:
```python
def _winner_code(status, winner_id, id_a: str, id_b: str) -> str:
    """'a'/'b' from the winning corner; 'draw'/'nc' from result_status.

    result_status is authoritative for no-winner fights ('draw',
    'no_contest'); a 'win' whose winner_id matches neither corner is
    treated as a no-contest rather than guessed.
    """
    text = "" if pd.isna(status) else str(status).strip().lower()
    if text == "draw":
        return "draw"
    if text == "no_contest":
        return "nc"
    winner = None if pd.isna(winner_id) else str(winner_id).strip()
    if winner == id_a:
        return "a"
    if winner == id_b:
        return "b"
    return "nc"
```
Replace `build_fights` entirely:
```python
def build_fights(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fight from master.csv: ids, date, winner code, targets, context."""
    ids_a = raw["r_fighter_id"].astype("string").str.strip()
    ids_b = raw["b_fighter_id"].astype("string").str.strip()
    fight_ids = raw["fight_id"].astype("string").str.strip()
    if fight_ids.isna().any() or not fight_ids.is_unique:
        raise ValueError("fight_id must be present and unique")
    if ids_a.isna().any() or ids_b.isna().any():
        raise ValueError("fights with missing corner fighter ids")
    method = raw["method"].map(map_method)
    fights = pd.DataFrame(
        {
            "fight_id": fight_ids,
            "date": pd.to_datetime(raw["event_date"], format="mixed", errors="coerce"),
            "fighter_a_id": ids_a,
            "fighter_b_id": ids_b,
            "winner": [
                _winner_code(status, winner_id, id_a, id_b)
                for status, winner_id, id_a, id_b in zip(
                    raw["result_status"], raw["winner_id"], ids_a, ids_b
                )
            ],
            "method": method,
            "method_raw": raw["method"],
            "decision_subtype": raw["method"].map(decision_subtype),
            "scheduled_rounds": pd.Series(
                raw["time_format"].map(parse_scheduled_rounds), dtype="Int64"
            ),
            "weight_class": raw["weight_class"].map(parse_weight_class),
            "title_fight": pd.to_numeric(raw["title_fight"], errors="coerce")
            .fillna(0)
            .astype(bool),
            "referee": raw["referee"],
            "event_id": raw["event_id"],
            "location": raw["event_location"],
        }
    )
    # finish_round only for finishes: decisions go the distance by definition,
    # and the raw column stores the last round fought for every fight.
    last_round = pd.to_numeric(raw["finish_round"], errors="coerce").astype("Int64")
    is_finish = fights["method"].isin(["ko_tko", "submission"])
    fights["finish_round"] = last_round.where(is_finish)

    # finish_time is the clock WITHIN the final round fought ("m:ss"), not
    # total duration. Derive elapsed duration_sec with 5-minute rounds:
    #   - finish: (finish_round - 1) * 300 + final-round clock
    #   - true decision: scheduled_rounds * 300 (went the distance)
    #   - everything else (DQ, Overturned, Could Not Continue, "Other",
    #     no-time-limit era): most end early, so fall back to the raw last
    #     round fought + clock.
    last_round_sec = pd.to_numeric(
        raw["finish_time"].map(parse_mmss_seconds), errors="coerce"
    )
    raw_last_round = last_round.astype("Float64")
    duration_sec = pd.Series(pd.NA, index=fights.index, dtype="Float64")
    duration_sec = duration_sec.where(
        fights["finish_round"].isna(),
        (fights["finish_round"].astype("Float64") - 1) * 300 + last_round_sec,
    )
    distance_mask = duration_sec.isna() & (fights["method"] == "decision")
    duration_sec = duration_sec.where(
        ~distance_mask, fights["scheduled_rounds"].astype("Float64") * 300
    )
    fallback_mask = duration_sec.isna()
    fallback = (raw_last_round - 1) * 300 + last_round_sec
    duration_sec = duration_sec.where(~fallback_mask, fallback)
    fights["duration_sec"] = duration_sec

    for column in (
        "winner", "method", "method_raw", "decision_subtype", "weight_class",
        "referee", "event_id", "location",
    ):
        fights[column] = fights[column].astype("string")

    columns = [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight", "duration_sec",
        "referee", "event_id", "location",
    ]
    return (
        fights[columns]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
```

- [ ] **Step 3: Run tests**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_dataset_fights.py tests/test_labels.py -q
```
Expected: all passed. If `test_string_dtypes` fails on `referee` because pandas infers `object` for an all-None column, the `.astype("string")` loop above handles it; if `scheduled_rounds` raises on `None` values, wrap as `pd.array([...], dtype="Int64")` instead of `pd.Series(..., dtype="Int64")`.

- [ ] **Step 4: Commit**

```bash
git add src/mma/dataset.py tests/test_dataset_fights.py
git status --short
git commit -m "Build fights from master.csv: result_status winners, referee/event/location context"
```

---

### Task 7: Fight-total stats, per-round stats, and bonuses builders

**Files:**
- Modify: `src/mma/dataset.py` (`_STAT_COLUMNS` → `_STAT_SUFFIXES`, `build_fight_stats`, new `build_round_stats`, `build_bonuses`)
- Rewrite: `tests/test_dataset_stats.py`
- Create: `tests/test_dataset_rounds.py`

`fight_stats` keeps its nine original columns (`kd, sig_landed, sig_attempted, total_landed, total_attempted, td_landed, td_attempted, sub_att, ctrl_sec`) in the same order and appends `rev` plus the six target/position pairs. `history.py` reads stats by key with `.get`, so extra columns are inert until SP2.

- [ ] **Step 1: Rewrite `tests/test_dataset_stats.py`**

```python
import pandas as pd
import pytest

from mma.dataset import build_fight_stats

_SUFFIXES = {
    "kd": "kd", "sig_landed": "sig_landed", "sig_atmp": "sig_atmp",
    "total_str_landed": "total_str_landed", "total_str_atmp": "total_str_atmp",
    "td_success": "td_success", "td_atmp": "td_atmp", "sub_att": "sub_att",
    "rev": "rev", "ctrl_seconds": "ctrl_seconds",
}
_TARGETS = ("head", "body", "leg", "distance", "clinch", "ground")


def _raw_master():
    rows = {
        "fight_id": ["f2", "f1"],
        "r_fighter_id": ["jj", "cm"],
        "b_fighter_id": ["dc", "ed"],
    }
    base = {"r": 10, "b": 20}
    for corner in ("r", "b"):
        for i, suffix in enumerate(_SUFFIXES.values()):
            rows[f"{corner}_total_{suffix}"] = [base[corner] + i, base[corner] + i + 100]
        for j, target in enumerate(_TARGETS):
            rows[f"{corner}_total_sig_str_landed_{target}"] = [base[corner] * 10 + j, None]
            rows[f"{corner}_total_sig_str_atmp_{target}"] = [base[corner] * 10 + j + 1, None]
    return pd.DataFrame(rows)


def test_two_rows_per_fight_schema_and_order():
    stats = build_fight_stats(_raw_master())
    assert list(stats.columns[:12]) == [
        "fight_id", "fighter_id", "corner", "kd", "sig_landed", "sig_attempted",
        "total_landed", "total_attempted", "td_landed", "td_attempted",
        "sub_att", "ctrl_sec",
    ]
    assert list(stats.columns[12:]) == ["rev"] + [
        f"{target}_{kind}" for target in _TARGETS for kind in ("landed", "attempted")
    ]
    assert list(zip(stats["fight_id"], stats["corner"])) == [
        ("f1", "a"), ("f1", "b"), ("f2", "a"), ("f2", "b"),
    ]


def test_values_unpivoted_to_correct_corner():
    stats = build_fight_stats(_raw_master()).set_index(["fight_id", "corner"])
    a = stats.loc[("f2", "a")]
    b = stats.loc[("f2", "b")]
    assert a["fighter_id"] == "jj" and b["fighter_id"] == "dc"
    assert a["kd"] == 10 and b["kd"] == 20
    assert a["td_landed"] == 15 and a["td_attempted"] == 16  # td_success, td_atmp
    assert a["ctrl_sec"] == 19 and b["ctrl_sec"] == 29
    assert a["rev"] == 18
    assert a["head_landed"] == 100 and a["head_attempted"] == 101
    assert b["ground_landed"] == 205


def test_missing_stat_stays_missing():
    raw = _raw_master()
    raw.loc[0, "r_total_kd"] = None
    stats = build_fight_stats(raw).set_index(["fight_id", "corner"])
    assert pd.isna(stats.loc[("f2", "a"), "kd"])
    assert pd.isna(stats.loc[("f1", "a"), "head_landed"])


def test_duplicate_fight_ids_rejected():
    raw = _raw_master()
    raw.loc[1, "fight_id"] = "f2"
    with pytest.raises(ValueError, match="fight_id"):
        build_fight_stats(raw)
```

- [ ] **Step 2: Create `tests/test_dataset_rounds.py`**

```python
import pandas as pd
import pytest

from mma.dataset import build_bonuses, build_round_stats

_TARGETS = ("head", "body", "leg", "distance", "clinch", "ground")


def _raw_round():
    rows = {
        "fight_id": ["f1", "f1", "f2"],
        "round_no": [2, 1, 1],
        "r_id": ["cm", "cm", "jj"],
        "b_id": ["ed", "ed", "dc"],
    }
    for corner, base in (("r", 10), ("b", 20)):
        rows[f"{corner}_kd"] = [base, base + 1, base + 2]
        rows[f"{corner}_sig_landed"] = [base + 3, base + 4, base + 5]
        rows[f"{corner}_sig_atmp"] = [base + 6, base + 7, base + 8]
        rows[f"{corner}_total_str_landed"] = [1, 2, 3]
        rows[f"{corner}_total_str_atmp"] = [4, 5, 6]
        rows[f"{corner}_td_success"] = [0, 1, 2]
        rows[f"{corner}_td_atmp"] = [1, 2, 3]
        rows[f"{corner}_sub_att"] = [0, 0, 1]
        rows[f"{corner}_rev"] = [0, 1, 0]
        rows[f"{corner}_ctrl"] = ["1:05", "0:00", "--"]
        for j, target in enumerate(_TARGETS):
            rows[f"{corner}_sig_str_landed_{target}"] = [base * 10 + j] * 3
            rows[f"{corner}_sig_str_atmp_{target}"] = [base * 10 + j + 1] * 3
    return pd.DataFrame(rows)


def test_round_stats_schema_and_order():
    rounds = build_round_stats(_raw_round())
    assert list(rounds.columns[:5]) == ["fight_id", "round_no", "corner", "fighter_id", "kd"]
    assert list(rounds.columns[5:14]) == [
        "sig_landed", "sig_attempted", "total_landed", "total_attempted",
        "td_landed", "td_attempted", "sub_att", "ctrl_sec", "rev",
    ]
    assert list(rounds.columns[14:]) == [
        f"{target}_{kind}" for target in _TARGETS for kind in ("landed", "attempted")
    ]
    assert list(zip(rounds["fight_id"], rounds["round_no"], rounds["corner"])) == [
        ("f1", 1, "a"), ("f1", 1, "b"), ("f1", 2, "a"), ("f1", 2, "b"), ("f2", 1, "a"), ("f2", 1, "b"),
    ]


def test_round_values_and_control_parsing():
    rounds = build_round_stats(_raw_round()).set_index(["fight_id", "round_no", "corner"])
    a2 = rounds.loc[("f1", 2, "a")]
    assert a2["fighter_id"] == "cm" and a2["kd"] == 10 and a2["sig_landed"] == 13
    assert a2["ctrl_sec"] == 65
    assert rounds.loc[("f1", 1, "b"), "ctrl_sec"] == 0
    assert pd.isna(rounds.loc[("f2", 1, "a"), "ctrl_sec"])  # "--" stays missing
    assert rounds.loc[("f2", 1, "b"), "head_landed"] == 200
    assert rounds.loc[("f1", 1, "a"), "td_landed"] == 1


def test_round_dtypes():
    rounds = build_round_stats(_raw_round())
    assert rounds["fight_id"].dtype == "string" and rounds["fighter_id"].dtype == "string"
    assert rounds["corner"].dtype == "string"
    assert rounds["round_no"].dtype == "int64"


def test_duplicate_fight_round_rejected():
    raw = _raw_round()
    raw.loc[1, "round_no"] = 2
    with pytest.raises(ValueError, match="fight_id, round_no"):
        build_round_stats(raw)


def test_bonuses():
    raw = pd.DataFrame({
        "fight_id": ["f2", "f1", "f1"],
        "bonus_type": ["Performance of the Night", "Fight of the Night", "Fight of the Night"],
    })
    bonuses = build_bonuses(raw)
    assert list(bonuses.columns) == ["fight_id", "bonus_type"]
    assert len(bonuses) == 2  # exact duplicate row dropped
    assert list(bonuses["fight_id"]) == ["f1", "f2"]
    assert bonuses["fight_id"].dtype == "string" and bonuses["bonus_type"].dtype == "string"
```

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_dataset_stats.py tests/test_dataset_rounds.py -q
```
Expected: `ImportError` for `build_round_stats` and `KeyError` in the stats tests.

- [ ] **Step 3: Implement in `src/mma/dataset.py`**

Replace `_STAT_COLUMNS` and `build_fight_stats` with:
```python
# output name -> raw column suffix. master.csv fight totals are
# `{r|b}_total_{suffix}` (control is `{r|b}_total_ctrl_seconds`, numeric);
# round.csv per-round values are `{r|b}_{suffix}` (control is `{r|b}_ctrl`,
# "m:ss" text).
_CORE_STAT_SUFFIXES = {
    "kd": "kd",
    "sig_landed": "sig_landed",
    "sig_attempted": "sig_atmp",
    "total_landed": "total_str_landed",
    "total_attempted": "total_str_atmp",
    "td_landed": "td_success",
    "td_attempted": "td_atmp",
    "sub_att": "sub_att",
}
_TARGETS = ("head", "body", "leg", "distance", "clinch", "ground")
_TARGET_STAT_SUFFIXES = {
    f"{target}_{kind}": f"sig_str_{raw_kind}_{target}"
    for target in _TARGETS
    for kind, raw_kind in (("landed", "landed"), ("attempted", "atmp"))
}


def _require_unique_fight_ids(raw: pd.DataFrame) -> pd.Series:
    fight_ids = raw["fight_id"].astype("string").str.strip()
    if fight_ids.isna().any() or not fight_ids.is_unique:
        raise ValueError("fight_id must be present and unique")
    return fight_ids


def build_fight_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Two rows per fight (one per fighter) with fight-total performance stats."""
    fight_ids = _require_unique_fight_ids(raw)
    frames = []
    for corner, prefix, id_column in (
        ("a", "r_total_", "r_fighter_id"), ("b", "b_total_", "b_fighter_id"),
    ):
        frame = pd.DataFrame(
            {
                "fight_id": fight_ids,
                "fighter_id": raw[id_column].astype("string").str.strip(),
                "corner": pd.Series(corner, index=raw.index, dtype="string"),
            }
        )
        for out_name, suffix in _CORE_STAT_SUFFIXES.items():
            frame[out_name] = pd.to_numeric(raw[prefix + suffix], errors="coerce")
        frame["ctrl_sec"] = pd.to_numeric(raw[prefix + "ctrl_seconds"], errors="coerce")
        frame["rev"] = pd.to_numeric(raw[prefix + "rev"], errors="coerce")
        for out_name, suffix in _TARGET_STAT_SUFFIXES.items():
            frame[out_name] = pd.to_numeric(raw[prefix + suffix], errors="coerce")
        frames.append(frame)
    stats = pd.concat(frames, ignore_index=True)
    return stats.sort_values(["fight_id", "corner"]).reset_index(drop=True)


def build_round_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Two rows per fight per round with that round's performance stats."""
    fight_ids = raw["fight_id"].astype("string").str.strip()
    round_no = pd.to_numeric(raw["round_no"], errors="coerce").astype("int64")
    if fight_ids.isna().any() or pd.DataFrame({"f": fight_ids, "r": round_no}).duplicated().any():
        raise ValueError("(fight_id, round_no) must be present and unique")
    frames = []
    for corner, prefix, id_column in (("a", "r_", "r_id"), ("b", "b_", "b_id")):
        frame = pd.DataFrame(
            {
                "fight_id": fight_ids,
                "round_no": round_no,
                "corner": pd.Series(corner, index=raw.index, dtype="string"),
                "fighter_id": raw[id_column].astype("string").str.strip(),
            }
        )
        for out_name, suffix in _CORE_STAT_SUFFIXES.items():
            frame[out_name] = pd.to_numeric(raw[prefix + suffix], errors="coerce")
        frame["ctrl_sec"] = pd.to_numeric(
            raw[prefix + "ctrl"].map(parse_mmss_seconds), errors="coerce"
        )
        frame["rev"] = pd.to_numeric(raw[prefix + "rev"], errors="coerce")
        for out_name, suffix in _TARGET_STAT_SUFFIXES.items():
            frame[out_name] = pd.to_numeric(raw[prefix + suffix], errors="coerce")
        frames.append(frame)
    rounds = pd.concat(frames, ignore_index=True)
    return rounds.sort_values(["fight_id", "round_no", "corner"]).reset_index(drop=True)


def build_bonuses(raw: pd.DataFrame) -> pd.DataFrame:
    """Post-fight bonus awards, one row per (fight, bonus type)."""
    bonuses = pd.DataFrame(
        {
            "fight_id": raw["fight_id"].astype("string").str.strip(),
            "bonus_type": raw["bonus_type"].astype("string").str.strip(),
        }
    )
    return (
        bonuses.drop_duplicates()
        .sort_values(["fight_id", "bonus_type"])
        .reset_index(drop=True)
    )
```
Also make `build_fights` use `_require_unique_fight_ids(raw)` in place of its inline check (same message, so the test regex still matches).

- [ ] **Step 4: Run tests**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_dataset_stats.py tests/test_dataset_rounds.py tests/test_dataset_fights.py -q
```
Expected: all passed. Note the duplicate-round error message must contain the literal `fight_id, round_no` for the regex.

- [ ] **Step 5: Commit**

```bash
git add src/mma/dataset.py tests/test_dataset_stats.py tests/test_dataset_rounds.py
git status --short
git commit -m "Build fight totals from master.csv, add per-round stats and bonuses tables"
```

---

### Task 8: `make_dataset.py`, `refresh_data.py`, and the first rebuild of processed data

**Files:**
- Modify: `scripts/make_dataset.py`, `scripts/refresh_data.py:83`
- Modify: `tests/test_processed_data.py:14-19`
- Create: `tests/test_processed_rounds.py`
- Regenerate: `data/processed/fighters.parquet`, `fights.parquet`, `fight_stats.parquet`; create `round_stats.parquet`, `bonuses.parquet`

- [ ] **Step 1: Rewrite `scripts/make_dataset.py`**

```python
"""Build processed parquet tables from raw CSVs. Reproducible end to end."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mma.dataset import (
    build_bonuses, build_fight_stats, build_fighters, build_fights, build_round_stats,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"integrity check failed: {message}")


def main() -> None:
    raw_master = pd.read_csv(RAW / "master.csv")
    raw_fighters = pd.read_csv(RAW / "fighter.csv")
    raw_rounds = pd.read_csv(RAW / "round.csv")
    raw_bonuses = pd.read_csv(RAW / "fighter_bonus.csv")

    fighters = build_fighters(raw_fighters)
    fights = build_fights(raw_master)
    stats = build_fight_stats(raw_master)
    rounds = build_round_stats(raw_rounds)
    bonuses = build_bonuses(raw_bonuses)

    check(fights["date"].notna().all(), "unparseable fight dates")
    check(len(stats) == 2 * len(fights), "stats rows != 2x fights")
    known = set(fighters["fighter_id"])
    in_fights = set(fights["fighter_a_id"]) | set(fights["fighter_b_id"])
    orphans = in_fights - known
    check(
        len(orphans) < 0.02 * len(in_fights),
        f"{len(orphans)} fight participants missing from fighters table",
    )
    method_rate = fights["method"].notna().mean()
    check(method_rate > 0.95, f"method mapped for only {method_rate:.1%} of fights")
    weight_rate = fights["weight_class"].notna().mean()
    check(weight_rate > 0.95, f"weight class for only {weight_rate:.1%} of fights")
    round_fights = set(rounds["fight_id"])
    check(round_fights <= set(fights["fight_id"]), "round_stats has unknown fight ids")
    per_fight = rounds.groupby("fight_id").size()
    check((per_fight % 2 == 0).all(), "round_stats must have both corners per round")
    round_cover = len(round_fights) / len(fights)
    check(round_cover > 0.90, f"round stats cover only {round_cover:.1%} of fights")
    check(set(bonuses["fight_id"]) <= set(fights["fight_id"]), "bonuses has unknown fight ids")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    fighters.to_parquet(PROCESSED / "fighters.parquet", index=False)
    fights.to_parquet(PROCESSED / "fights.parquet", index=False)
    stats.to_parquet(PROCESSED / "fight_stats.parquet", index=False)
    rounds.to_parquet(PROCESSED / "round_stats.parquet", index=False)
    bonuses.to_parquet(PROCESSED / "bonuses.parquet", index=False)

    print(f"fighters:    {len(fighters)} rows")
    print(
        f"fights:      {len(fights)} rows, "
        f"{fights['date'].min():%Y-%m-%d} .. {fights['date'].max():%Y-%m-%d}"
    )
    print(f"stats:       {len(stats)} rows")
    print(f"round_stats: {len(rounds)} rows covering {len(round_fights)} fights")
    print(f"bonuses:     {len(bonuses)} rows")
    print("\nwinner distribution:")
    print(fights["winner"].value_counts().to_string())
    print("\nmethod distribution:")
    print(fights["method"].value_counts(dropna=False).to_string())
    if orphans:
        print(f"\nwarning: {len(orphans)} orphan fighter ids (sample): {sorted(orphans)[:5]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Point `refresh_data.py` at `master.csv`**

Replace line 83 `raw_fights = pd.read_csv(RAW_DIR / "UFC.csv")` with:
```python
    raw_fights = pd.read_csv(
        RAW_DIR / "master.csv", usecols=["fight_id", "event_date"]
    ).rename(columns={"event_date": "date"})
```
`refresh_needed` and its tests are unchanged (they only see a `date` column).

- [ ] **Step 3: Update processed-data expectations**

In `tests/test_processed_data.py` change `test_fights_volume_and_range` to:
```python
def test_fights_volume_and_range():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert len(fights) > 11000
    assert fights["date"].min().year <= 1995
    assert fights["date"].max().year >= 2026
```
Create `tests/test_processed_rounds.py`:
```python
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "round_stats.parquet").exists(),
    reason="processed data not built (run scripts/make_dataset.py)",
)


def test_round_stats_volume_and_shape():
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    assert len(rounds) > 44000  # 2 corners x ~22k+ rounds
    assert set(rounds["corner"]) == {"a", "b"}
    assert rounds["round_no"].between(1, 6).all()
    assert not rounds.duplicated(["fight_id", "round_no", "corner"]).any()


def test_round_stats_join_to_fights():
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert set(rounds["fight_id"]) <= set(fights["fight_id"])
    joined = rounds.merge(fights[["fight_id", "fighter_a_id", "fighter_b_id"]], on="fight_id")
    a = joined[joined["corner"] == "a"]
    assert (a["fighter_id"] == a["fighter_a_id"]).all()


def test_round_totals_match_fight_totals_for_recent_fights():
    """Fight-total sig strikes must equal the sum over rounds where both exist."""
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    recent = fights[fights["date"] >= "2015-01-01"]["fight_id"]
    summed = (
        rounds[rounds["fight_id"].isin(recent)]
        .groupby(["fight_id", "corner"])["sig_landed"].sum()
    )
    totals = stats.set_index(["fight_id", "corner"]).loc[summed.index, "sig_landed"]
    agree = (summed.values == totals.values).mean()
    assert agree > 0.99


def test_control_seconds_plausible():
    rounds = pd.read_parquet(PROCESSED / "round_stats.parquet")
    assert rounds["ctrl_sec"].dropna().between(0, 720).all()  # 10-min rounds in early eras


def test_bonuses_table():
    bonuses = pd.read_parquet(PROCESSED / "bonuses.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert len(bonuses) > 2000
    assert set(bonuses["fight_id"]) <= set(fights["fight_id"])
    assert set(bonuses["bonus_type"]) == {
        "Performance of the Night", "Fight of the Night",
        "Knockout of the Night", "Submission of the Night",
    }
```

- [ ] **Step 4: Save the old fights table for reconciliation, then rebuild**

```bash
mkdir -p /tmp/sp0 && git show main:data/processed/fights.parquet > /tmp/sp0/old_fights.parquet
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/make_dataset.py
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/refresh_data.py --help >/dev/null && echo refresh-imports-ok
```
Expected: `fights: 11441 rows, 1993-11-12 .. 2026-08-08`; `stats: 22882 rows`; `round_stats: 50262 rows covering 11111 fights`; `bonuses` ≈ 2,409 rows; all integrity checks pass; method distribution dominated by decision/ko_tko/submission.

- [ ] **Step 5: Run the processed-data and dataset tests**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_processed_data.py tests/test_processed_rounds.py tests/test_dataset_*.py tests/test_refresh.py -q
```
Expected: all passed. (`test_processed_features.py`, `test_processed_ratings.py`, `test_processed_torch.py` will be stale until Task 10 rebuilds their artifacts — do not run the full suite yet.)

- [ ] **Step 6: Commit**

```bash
git add scripts/make_dataset.py scripts/refresh_data.py tests/test_processed_data.py tests/test_processed_rounds.py data/processed/fighters.parquet data/processed/fights.parquet data/processed/fight_stats.parquet data/processed/round_stats.parquet data/processed/bonuses.parquet
git status --short
git commit -m "Rebuild processed tables from the new Kaggle layout; add round_stats and bonuses"
```

---

### Task 9: Source reconciliation report (provenance)

**Files:**
- Create: `scripts/reconcile_sources.py`

- [ ] **Step 1: Write the script**

```python
"""Compare a previous processed fights table against the current one.

Used once when switching Kaggle layouts (2026-09), kept so any future
source change can be audited the same way. Reports overlap, label
agreement on overlapping fights, and what the new source adds or drops.

Usage:
    python scripts/reconcile_sources.py --old /path/old_fights.parquet [--new data/processed/fights.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW_DEFAULT = ROOT / "data" / "processed" / "fights.parquet"
_COMPARE = ("winner", "method", "finish_round", "scheduled_rounds", "weight_class",
            "fighter_a_id", "fighter_b_id", "date")


def reconcile(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Pure comparison; returns a dict of counts and agreement rates."""
    old_ids, new_ids = set(old["fight_id"]), set(new["fight_id"])
    overlap = old.merge(new, on="fight_id", suffixes=("_old", "_new"))
    agreement = {}
    for column in _COMPARE:
        a, b = overlap[f"{column}_old"], overlap[f"{column}_new"]
        same = (a == b) | (a.isna() & b.isna())
        agreement[column] = float(same.mean()) if len(overlap) else None
    new_only = new[~new["fight_id"].isin(old_ids)]
    return {
        "n_old": len(old),
        "n_new": len(new),
        "n_overlap": len(overlap),
        "n_dropped_by_new": len(old_ids - new_ids),
        "n_added_by_new": len(new_only),
        "added_by_year": new_only.groupby(new_only["date"].dt.year).size().to_dict(),
        "agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, default=NEW_DEFAULT)
    args = parser.parse_args()
    report = reconcile(pd.read_parquet(args.old), pd.read_parquet(args.new))
    print(f"old fights: {report['n_old']}   new fights: {report['n_new']}")
    print(f"overlap: {report['n_overlap']}   dropped by new: {report['n_dropped_by_new']}   added by new: {report['n_added_by_new']}")
    print("agreement on overlap:")
    for column, rate in report["agreement"].items():
        print(f"  {column:<18} {rate:.4f}" if rate is not None else f"  {column:<18} n/a")
    print("added by year:", {int(k): int(v) for k, v in report["added_by_year"].items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Unit-test `reconcile`**

Append to `tests/test_dataset_rounds.py`:
```python
from scripts.reconcile_sources import reconcile


def test_reconcile_counts_and_agreement():
    old = pd.DataFrame({
        "fight_id": ["f1", "f2"], "winner": ["a", "b"], "method": ["ko_tko", "decision"],
        "finish_round": [1, None], "scheduled_rounds": [3, 3], "weight_class": ["Lightweight"] * 2,
        "fighter_a_id": ["x", "y"], "fighter_b_id": ["p", "q"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
    })
    new = pd.concat([old, pd.DataFrame({
        "fight_id": ["f3"], "winner": ["a"], "method": ["submission"], "finish_round": [2],
        "scheduled_rounds": [3], "weight_class": ["Lightweight"], "fighter_a_id": ["z"],
        "fighter_b_id": ["r"], "date": pd.to_datetime(["2021-03-01"]),
    })], ignore_index=True)
    new.loc[1, "winner"] = "a"  # one disagreement
    report = reconcile(old, new)
    assert report["n_overlap"] == 2 and report["n_added_by_new"] == 1 and report["n_dropped_by_new"] == 0
    assert report["agreement"]["winner"] == 0.5
    assert report["agreement"]["finish_round"] == 1.0  # NaN == NaN counts as agreement
    assert report["added_by_year"] == {2021: 1}
```

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest tests/test_dataset_rounds.py -q
```
Expected: all passed.

- [ ] **Step 3: Run the report and record it**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/reconcile_sources.py --old /tmp/sp0/old_fights.parquet
```
Expected: overlap 8337, dropped 0, added 3104; agreement ≥ 0.99 on every column (winner/method/fighter ids/date should be exactly 1.0; `scheduled_rounds` may differ slightly where the old `total_rounds` and the new `time_format` disagree — anything below 0.99 on any column is a stop-and-investigate signal, not something to paper over). Paste the printed report into the "Completion notes" section at the bottom of this plan.

- [ ] **Step 4: Commit**

```bash
git add scripts/reconcile_sources.py tests/test_dataset_rounds.py docs/superpowers/plans/2026-09-06-sp0-repair-ingestion.md
git status --short
git commit -m "Add source reconciliation report; record old-vs-new Kaggle layout agreement"
```

---

### Task 10: Rebuild ratings, features, models, and display priors on the fuller data

This is a data refresh, exactly what the weekly Action does, run locally so the merged branch is self-consistent. It produces **model v2** (new artifact hash) trained on ~37% more fights.

**Files:**
- Regenerate: `data/processed/elo_params.json`, `ratings.parquet`, `features.parquet`, `models/xgb_*.json`, `models/xgb_metrics_val.json`, `models/torch/*`

- [ ] **Step 1: Run the chain**

```bash
export OMP_NUM_THREADS=1
~/.venvs/mma/bin/python scripts/build_ratings.py 2>&1 | tail -5
~/.venvs/mma/bin/python scripts/build_features.py 2>&1 | tail -8
~/.venvs/mma/bin/python scripts/train_xgb.py 2>&1 | tail -8
~/.venvs/mma/bin/python scripts/train_torch.py 2>&1 | tail -10
~/.venvs/mma/bin/python scripts/build_display_priors.py 2>&1 | tail -5
```
Expected: features ≈ 11,200 rows × 46 columns (decisive fights only), `y_winner` balance within 0.47–0.53; XGB and torch print validation metrics for the 2021–2023 window (now ≈ 1,700 fights). Torch trains 5 seeds; if a seed stalls for more than a few minutes, the venv is not on local disk — stop and fix that rather than waiting.

- [ ] **Step 2: Full suite**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q 2>&1 | tail -3
```
Expected: all passed except the known pandas-3 AppTest env-skip. Any failure in `test_processed_features.py::test_no_leakage_truncation_invariance` is a real leakage regression in the new builders and blocks the task.

- [ ] **Step 3: Record the new metrics and the v2 hash**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python - <<'EOF'
import json
from pathlib import Path
from mma.versioning import model_version
print("xgb:", json.load(open("models/xgb_metrics_val.json"))["winner"])
print("torch:", json.load(open("models/torch/metrics_val.json")))
print("model v2 hash:", model_version(Path(".")))
EOF
```
Write both metric blocks and the hash into the "Completion notes" below.

- [ ] **Step 4: Commit the artifacts**

```bash
git add data/processed models/xgb_winner.json models/xgb_method.json models/xgb_round.json models/xgb_metrics_val.json models/torch
git status --short
git commit -m "Retrain Elo, XGBoost, and torch ensemble on the fuller 11.4k-fight dataset"
```
Check the staged list before committing: it must be parquet/json/pt files only, and `models/final_test_metrics.json` and `models/market_benchmark.json` must **not** be touched (they are one-time artifacts).

---

### Task 11: Grade pending predictions

**Files:**
- Regenerate: `predictions/*.json` (grading fields added), `predictions/track_record.json`

- [ ] **Step 1: Grade**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/grade_predictions.py 2>&1 | tail -15
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python -c "import json; t = json.load(open('predictions/track_record.json')); print(json.dumps({'overall': t['overall'], 'versions': {k: v['n_graded'] for k, v in t['model_versions'].items()}}, indent=1))"
```
Expected: the events dated on or before 2026-08-08 (du Plessis vs Usman 07-18, Ankalaev vs Guskov 07-25, Medi vs Rodriguez 08-01, Gamrot vs Salkilld 08-08) are graded; `n_graded` > 0 with a real accuracy/log-loss/Brier; still one model version (v1 hash). Record the numbers below.

- [ ] **Step 2: Commit**

```bash
git add predictions/
git status --short
git commit -m "Grade the first prospective predictions against refreshed results"
```

---

### Task 12: README

**Files:**
- Modify: `README.md` (data paragraph ~lines 47-55, results block lines 56-79, add a Development section before "## Interactive app")

- [ ] **Step 1: Data paragraph**

Replace the sentence beginning "Data bootstraps from the [Kaggle UFC dataset]" through "runs this refresh weekly and commits any rebuilt artifacts automatically." with:

```markdown
Data bootstraps from the [Kaggle UFC dataset](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
via `kagglehub` and auto-refreshes weekly from the same maintained mirror
(`scripts/refresh_data.py`). The dataset was rebuilt upstream in August 2026
with a new layout (`master.csv`, `fighter.csv`, `round.csv`, `fighter_bonus.csv`)
that also carries per-round stats, strike-target and position splits,
referees, and bonuses; the ingestion switched over in September 2026
(`scripts/reconcile_sources.py` audits the swap). The rebuilt source turned
out to include roughly 50–80 fights per year from 2012 onward that the
earlier scrape had silently dropped, so every artifact was retrained on
11,441 fights instead of 8,337. A direct ufcstats.com scraper was planned
but dropped — the site gates automated clients behind an anti-bot challenge;
`src/mma/parsing.py` supplies the string parsers the new layout needs. The
`.github/workflows/refresh-data.yml` Action runs this refresh weekly and
commits any rebuilt artifacts automatically.
```

- [ ] **Step 2: Results block**

In "## Results so far", change `(1,507 validation fights)` to the new validation count from `models/xgb_metrics_val.json["winner"]["n_val"]`, and replace the four numeric cells for XGBoost and the neural net with the values recorded in Task 10 (three decimals). Update the method/finish-round sentence's numbers from `models/xgb_metrics_val.json["method"]`, `["finish_round"]`, and `models/torch/metrics_val.json`. Add one sentence at the end of the intro paragraph:

```markdown
Numbers below are from the September 2026 retrain on the fuller dataset;
the [Final held-out test results](#final-held-out-test-results-2024) and the
market benchmark are one-time artifacts from July 2026 and are left as
recorded.
```

- [ ] **Step 3: Development section**

Insert before `## Interactive app`:

```markdown
## Development notes

- **Local-disk virtualenv.** If the repo lives in an iCloud-synced folder,
  create the venv elsewhere (`python3 -m venv ~/.venvs/mma && ~/.venvs/mma/bin/pip install -e ".[dev,app]"`)
  — torch's shared libraries stall for minutes when paged in from iCloud.
- `OMP_NUM_THREADS=1` for any script that imports both torch and xgboost.
- **Model identity.** Prospective predictions are stamped with
  `mma.versioning.model_version()`, a hash of the deployed artifacts, so the
  track record splits by model, not by commit. Retraining (weekly refresh or
  walk-forward promotion) starts a new section automatically.
```

- [ ] **Step 4: Verify links and commit**

```bash
grep -n "final-held-out-test-results" README.md | head -3
git add README.md
git status --short
git commit -m "README: new data layout, retrained results, development notes"
```

---

### Task 13: Merge

- [ ] **Step 1: Final verification on the branch**

```bash
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/pytest -q 2>&1 | tail -2
OMP_NUM_THREADS=1 ~/.venvs/mma/bin/python scripts/refresh_data.py 2>&1 | tail -4
git log --oneline main..sp0-repair
```
Expected: suite green; `REFRESH_NEEDED=false` (processed data matches the snapshot); roughly 11 commits.

- [ ] **Step 2: Merge locally (no push in this task)**

```bash
git checkout main && git status --short && git merge --no-ff sp0-repair -m "Merge sp0-repair: new Kaggle layout, per-round data, artifact-hash model versions, first graded predictions"
git log --oneline -3
```
Pushing to the public repo is a separate, user-confirmed step.

---

## Completion notes (filled in during execution)

- Baseline suite: _n passed / n skipped_
- Model v1 artifact hash: _…_
- Reconciliation report: _…_
- Model v2 artifact hash: _…_ ; XGB val metrics: _…_ ; torch val metrics: _…_
- Graded prospective fights: _n graded, accuracy, log-loss, Brier_
- Deferred to SP2: the truncation-invariance test only gains round-derived columns once such features exist (none in SP0).
