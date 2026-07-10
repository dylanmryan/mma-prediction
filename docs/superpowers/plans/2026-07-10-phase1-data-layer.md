# Phase 1: Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project scaffolding plus a reproducible pipeline that turns a raw Kaggle UFC dataset into three clean parquet tables (`fighters`, `fights`, `fight_stats`).

**Architecture:** Pure functions in `src/mma/` (string parsers → label mappers → table builders), orchestrated by `scripts/make_dataset.py`. All dataset-specific column names live in one `COLUMN_MAP` dict so the Phase 6 scraper can reuse every builder unchanged. Raw data is gitignored; processed parquet is committed.

**Tech Stack:** Python ≥3.10, pandas, pyarrow, kagglehub, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-mma-prediction-design.md`

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/mma/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `README.md`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mma"
version = "0.1.0"
description = "UFC fight outcome prediction: Elo -> XGBoost -> PyTorch"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "kagglehub>=0.3",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.ipynb_checkpoints/
data/raw/
.DS_Store
```

- [ ] **Step 3: Create package and smoke test**

`src/mma/__init__.py`:

```python
"""UFC fight outcome prediction."""
```

`tests/__init__.py`: empty file.

`tests/test_smoke.py`:

```python
import mma


def test_package_imports():
    assert mma is not None
```

`README.md`:

```markdown
# MMA Fight Prediction

Predicting UFC fight winners, method of victory, and finish round.
Elo baseline -> XGBoost -> PyTorch multi-task net, honestly evaluated.

Work in progress. Design: `docs/superpowers/specs/2026-07-10-mma-prediction-design.md`
```

- [ ] **Step 4: Create venv, install, run smoke test**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pytest -v
```
Expected: `test_package_imports PASSED`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests README.md
git commit -m "Scaffold project: package layout, pytest, gitignore"
```

---

### Task 2: String parsers (`parsing.py`)

ufcstats-style raw values: strikes as `"45 of 118"`, control time as `"2:35"`, height as `5' 11"`, reach as `72"`, percentages as `45%`. Missing values appear as `"--"`, `"---"`, empty string, or NaN.

**Files:**
- Create: `src/mma/parsing.py`
- Test: `tests/test_parsing.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_parsing.py`:

```python
from mma.parsing import (
    parse_height_inches,
    parse_landed_attempted,
    parse_mmss_seconds,
    parse_percent,
    parse_reach_inches,
)


def test_landed_attempted_basic():
    assert parse_landed_attempted("45 of 118") == (45, 118)


def test_landed_attempted_zero():
    assert parse_landed_attempted("0 of 0") == (0, 0)


def test_landed_attempted_missing():
    assert parse_landed_attempted("--") == (None, None)
    assert parse_landed_attempted(None) == (None, None)
    assert parse_landed_attempted(float("nan")) == (None, None)


def test_mmss_basic():
    assert parse_mmss_seconds("2:35") == 155
    assert parse_mmss_seconds("0:00") == 0


def test_mmss_missing():
    assert parse_mmss_seconds("--") is None
    assert parse_mmss_seconds(None) is None


def test_height_feet_inches():
    assert parse_height_inches("5' 11\"") == 71.0
    assert parse_height_inches("6' 0\"") == 72.0


def test_height_missing():
    assert parse_height_inches("--") is None
    assert parse_height_inches(None) is None


def test_reach():
    assert parse_reach_inches('72"') == 72.0
    assert parse_reach_inches("72") == 72.0


def test_reach_missing():
    assert parse_reach_inches("--") is None


def test_percent():
    assert parse_percent("45%") == 0.45
    assert parse_percent("0%") == 0.0


def test_percent_missing():
    assert parse_percent("---") is None
    assert parse_percent(None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_parsing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mma.parsing'`

- [ ] **Step 3: Implement `src/mma/parsing.py`**

```python
"""Parsers for ufcstats-style raw string values."""
from __future__ import annotations

import re

_MISSING = {"", "--", "---", "n/a", "nan"}


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):  # NaN
        return value != value
    return str(value).strip().lower() in _MISSING


def parse_landed_attempted(value) -> tuple[int | None, int | None]:
    """'45 of 118' -> (45, 118)."""
    if _is_missing(value):
        return (None, None)
    match = re.fullmatch(r"(\d+)\s+of\s+(\d+)", str(value).strip())
    if not match:
        return (None, None)
    return (int(match.group(1)), int(match.group(2)))


def parse_mmss_seconds(value) -> int | None:
    """'2:35' -> 155 seconds."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+):(\d{2})", str(value).strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def parse_height_inches(value) -> float | None:
    """`5' 11"` -> 71.0 inches."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+)'\s*(\d+)\"?", str(value).strip())
    if not match:
        return None
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def parse_reach_inches(value) -> float | None:
    """'72"' -> 72.0 inches."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\"?", str(value).strip())
    if not match:
        return None
    return float(match.group(1))


def parse_percent(value) -> float | None:
    """'45%' -> 0.45."""
    if _is_missing(value):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)%?", str(value).strip())
    if not match:
        return None
    return float(match.group(1)) / 100.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_parsing.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mma/parsing.py tests/test_parsing.py
git commit -m "Add ufcstats string parsers"
```

---

### Task 3: Label mappers (`labels.py`)

Maps raw method strings to the 3-class target, extracts decision sub-type for display, parses scheduled rounds and weight class.

**Files:**
- Create: `src/mma/labels.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_labels.py`:

```python
from mma.labels import (
    decision_subtype,
    map_method,
    parse_scheduled_rounds,
    parse_weight_class,
)


def test_method_ko_tko():
    assert map_method("KO/TKO") == "ko_tko"
    assert map_method("TKO - Doctor's Stoppage") == "ko_tko"


def test_method_submission():
    assert map_method("Submission") == "submission"


def test_method_decision():
    assert map_method("Decision - Unanimous") == "decision"
    assert map_method("Decision - Split") == "decision"
    assert map_method("Decision - Majority") == "decision"


def test_method_excluded():
    assert map_method("DQ") is None
    assert map_method("Overturned") is None
    assert map_method("Could Not Continue") is None
    assert map_method(None) is None


def test_decision_subtype():
    assert decision_subtype("Decision - Unanimous") == "unanimous"
    assert decision_subtype("Decision - Split") == "split"
    assert decision_subtype("Decision - Majority") == "majority"
    assert decision_subtype("KO/TKO") is None


def test_scheduled_rounds():
    assert parse_scheduled_rounds("3 Rnd (5-5-5)") == 3
    assert parse_scheduled_rounds("5 Rnd (5-5-5-5-5)") == 5
    assert parse_scheduled_rounds("1 Rnd + OT (15-3)") == 1
    assert parse_scheduled_rounds("No Time Limit") is None
    assert parse_scheduled_rounds(None) is None


def test_weight_class_ordering_pitfalls():
    # 'Light Heavyweight' must not match 'Heavyweight'
    assert parse_weight_class("UFC Light Heavyweight Title Bout") == "Light Heavyweight"
    assert parse_weight_class("Heavyweight Bout") == "Heavyweight"
    # Women's divisions must not match the men's substring
    assert parse_weight_class("Women's Strawweight Bout") == "Women's Strawweight"
    assert parse_weight_class("Lightweight Bout") == "Lightweight"
    assert parse_weight_class("Catch Weight Bout") == "Catch Weight"
    assert parse_weight_class("Some Unknown Bout") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mma.labels'`

- [ ] **Step 3: Implement `src/mma/labels.py`**

```python
"""Target-label and fight-context mappers."""
from __future__ import annotations

import re

# Ordered longest/most-specific first so substrings don't shadow
# (Light Heavyweight before Heavyweight, Women's before men's).
WEIGHT_CLASSES = [
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Light Heavyweight",
    "Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Strawweight",
    "Catch Weight",
    "Catchweight",
    "Open Weight",
]


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    return text or None


def map_method(win_by) -> str | None:
    """Raw method string -> 'ko_tko' | 'submission' | 'decision' | None.

    None means the fight is excluded from method modeling (DQ, overturned...).
    """
    text = _clean(win_by)
    if text is None:
        return None
    lower = text.lower()
    if "ko/tko" in lower or "doctor" in lower or lower.startswith("tko"):
        return "ko_tko"
    if lower.startswith("submission"):
        return "submission"
    if lower.startswith("decision"):
        return "decision"
    return None


def decision_subtype(win_by) -> str | None:
    """'Decision - Split' -> 'split'; non-decisions -> None."""
    text = _clean(win_by)
    if text is None or not text.lower().startswith("decision"):
        return None
    for subtype in ("unanimous", "split", "majority"):
        if subtype in text.lower():
            return subtype
    return None


def parse_scheduled_rounds(format_str) -> int | None:
    """'3 Rnd (5-5-5)' -> 3. 'No Time Limit' -> None."""
    text = _clean(format_str)
    if text is None:
        return None
    match = re.match(r"(\d+)\s*Rnd", text)
    if not match:
        return None
    return int(match.group(1))


def parse_weight_class(fight_type) -> str | None:
    """Extract weight class from e.g. 'UFC Middleweight Title Bout'."""
    text = _clean(fight_type)
    if text is None:
        return None
    for weight_class in WEIGHT_CLASSES:
        if weight_class.lower() in text.lower():
            return weight_class
    return None


def is_title_fight(fight_type) -> bool:
    text = _clean(fight_type)
    return text is not None and "title" in text.lower()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mma/labels.py tests/test_labels.py
git commit -m "Add method/weight-class/rounds label mappers"
```

---

### Task 4: Dataset download + schema inspection

**Files:**
- Create: `scripts/download_data.py`
- Create: `data/raw/.gitkeep` (directory placeholder; contents gitignored)

- [ ] **Step 1: Write `scripts/download_data.py`**

```python
"""Download the Kaggle UFC dataset into data/raw/ and report its schema.

Dataset: https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025
(a ufcstats.com scrape, 1994 - mid-2025).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "neelagiriaditya/ufc-datasets-1994-2025"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    cache_path = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for src in cache_path.rglob("*.csv"):
        dest = RAW_DIR / src.name
        shutil.copy2(src, dest)
        print(f"copied {src.name}")

    print("\n=== SCHEMA REPORT ===")
    for csv in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(csv, nrows=5, sep=None, engine="python")
        print(f"\n{csv.name}  ({len(df.columns)} cols)")
        print("  columns:", list(df.columns))
        print(df.head(2).to_string(max_colwidth=25))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and review the schema report**

Run: `.venv/bin/python scripts/download_data.py`
Expected: CSVs copied to `data/raw/`, schema report printed.

**DECISION POINT — read the report carefully and record findings.** The builders in Tasks 5–7 assume a `COLUMN_MAP` (defined in Task 5) targeting standard ufcstats-scrape names (`R_fighter`, `B_fighter`, `win_by`, `last_round`, `Format`, `date`, `Fight_type`, `Winner`, `R_KD`, `R_SIG_STR.`, `R_TD`, `R_SUB_ATT`, `R_CTRL`, and a fighter-details file with `fighter_name`, `Height`, `Reach`, `Stance`, `DOB`). If the actual names differ, update ONLY the `COLUMN_MAP` values (and file names in `make_dataset.py`) — not the builder logic. If the dataset structure is fundamentally different (e.g., no per-fight stats file), STOP and flag to the user; fallback dataset is `rajeevw/ufcdata` (same conventions, ends 2021, split years would shift).

If kagglehub asks for credentials: create a Kaggle API token (kaggle.com → Settings → API) and save to `~/.kaggle/kaggle.json`, then re-run.

- [ ] **Step 3: Commit**

```bash
mkdir -p data/raw && touch data/raw/.gitkeep
git add scripts/download_data.py data/raw/.gitkeep
git commit -m "Add Kaggle dataset download + schema report script"
```

---

### Task 5: Fighters table builder

**Files:**
- Create: `src/mma/dataset.py`
- Test: `tests/test_dataset_fighters.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset_fighters.py`:

```python
import pandas as pd

from mma.dataset import build_fighters, slugify


def _raw_fighters():
    return pd.DataFrame(
        {
            "fighter_name": ["Jon Jones", "Amanda Nunes", "Jon Jones"],
            "Height": ["6' 4\"", "5' 8\"", "6' 4\""],
            "Reach": ['84.5"', '69"', "--"],
            "Stance": ["Orthodox", "Orthodox", None],
            "DOB": ["Jul 19, 1987", "May 30, 1988", "Jul 19, 1987"],
        }
    )


def test_build_fighters_basic():
    fighters = build_fighters(_raw_fighters())
    assert list(fighters.columns) == [
        "fighter_id", "name", "height_in", "reach_in", "stance", "dob",
    ]
    jones = fighters[fighters["name"] == "Jon Jones"].iloc[0]
    assert jones["fighter_id"] == "jon-jones"
    assert jones["height_in"] == 76.0
    assert jones["reach_in"] == 84.5
    assert jones["stance"] == "Orthodox"
    assert pd.Timestamp(jones["dob"]) == pd.Timestamp("1987-07-19")


def test_build_fighters_dedupes_exact_names():
    fighters = build_fighters(_raw_fighters())
    assert len(fighters) == 2
    assert fighters["fighter_id"].is_unique


def test_build_fighters_missing_reach_is_nan():
    raw = _raw_fighters().iloc[[1]].assign(Reach="--")
    fighters = build_fighters(raw)
    assert pd.isna(fighters.iloc[0]["reach_in"])


def test_slugify():
    assert slugify("Jon Jones") == "jon-jones"
    assert slugify("José Aldo") == "jose-aldo"
    assert slugify("Khabib  Nurmagomedov") == "khabib-nurmagomedov"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dataset_fighters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mma.dataset'`

- [ ] **Step 3: Implement `build_fighters` in `src/mma/dataset.py`**

```python
"""Builders that turn raw ufcstats-scrape CSVs into clean tables.

All dataset-specific column names live in COLUMN_MAP. If the Kaggle
dataset's schema report (scripts/download_data.py) shows different
names, edit COLUMN_MAP only.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from mma.labels import (
    decision_subtype,
    is_title_fight,
    map_method,
    parse_scheduled_rounds,
    parse_weight_class,
)
from mma.parsing import (
    parse_height_inches,
    parse_landed_attempted,
    parse_mmss_seconds,
    parse_reach_inches,
)

COLUMN_MAP = {
    # fighter details file
    "fighter_name": "fighter_name",
    "height": "Height",
    "reach": "Reach",
    "stance": "Stance",
    "dob": "DOB",
    # fights file
    "r_fighter": "R_fighter",
    "b_fighter": "B_fighter",
    "winner": "Winner",
    "win_by": "win_by",
    "last_round": "last_round",
    "format": "Format",
    "date": "date",
    "fight_type": "Fight_type",
    # per-corner stats (R_ prefix shown; B_ mirrors it)
    "kd": "KD",
    "sig_str": "SIG_STR.",
    "total_str": "TOTAL_STR.",
    "td": "TD",
    "sub_att": "SUB_ATT",
    "ctrl": "CTRL",
}


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text


def build_fighters(raw: pd.DataFrame) -> pd.DataFrame:
    c = COLUMN_MAP
    fighters = pd.DataFrame(
        {
            "name": raw[c["fighter_name"]].astype(str).str.strip(),
            "height_in": raw[c["height"]].map(parse_height_inches),
            "reach_in": raw[c["reach"]].map(parse_reach_inches),
            "stance": raw[c["stance"]].where(raw[c["stance"]].notna(), None),
            "dob": pd.to_datetime(raw[c["dob"]], format="mixed", errors="coerce"),
        }
    )
    fighters["fighter_id"] = fighters["name"].map(slugify)
    # Exact duplicate rows (same name) -> keep the one with the most data.
    fighters["_completeness"] = fighters.notna().sum(axis=1)
    fighters = (
        fighters.sort_values("_completeness", ascending=False)
        .drop_duplicates(subset="fighter_id", keep="first")
        .drop(columns="_completeness")
        .sort_values("fighter_id")
        .reset_index(drop=True)
    )
    return fighters[["fighter_id", "name", "height_in", "reach_in", "stance", "dob"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dataset_fighters.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mma/dataset.py tests/test_dataset_fighters.py
git commit -m "Add fighters table builder"
```

---

### Task 6: Fights table builder

One row per fight. Red corner = fighter A, blue corner = fighter B (raw corner order preserved here; anti-leakage symmetrization happens in the Phase 3 feature builder, per spec).

**Files:**
- Modify: `src/mma/dataset.py` (append `build_fights`)
- Test: `tests/test_dataset_fights.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset_fights.py`:

```python
import pandas as pd

from mma.dataset import build_fights


def _raw_fights():
    return pd.DataFrame(
        {
            "R_fighter": ["Jon Jones", "Amanda Nunes"],
            "B_fighter": ["Daniel Cormier", "Valentina Shevchenko"],
            "Winner": ["Jon Jones", "Amanda Nunes"],
            "win_by": ["KO/TKO", "Decision - Split"],
            "last_round": [3, 5],
            "last_round_time": ["4:20", "5:00"],
            "Format": ["5 Rnd (5-5-5-5-5)", "5 Rnd (5-5-5-5-5)"],
            "date": ["July 29, 2017", "September 09, 2017"],
            "Fight_type": [
                "UFC Light Heavyweight Title Bout",
                "UFC Women's Bantamweight Title Bout",
            ],
        }
    )


def test_build_fights_schema_and_values():
    fights = build_fights(_raw_fights())
    row = fights.iloc[0]
    assert row["fight_id"] == "20170729-jon-jones-vs-daniel-cormier"
    assert row["fighter_a_id"] == "jon-jones"
    assert row["fighter_b_id"] == "daniel-cormier"
    assert row["winner"] == "a"
    assert row["method"] == "ko_tko"
    assert row["method_raw"] == "KO/TKO"
    assert row["decision_subtype"] is None or pd.isna(row["decision_subtype"])
    assert row["finish_round"] == 3
    assert row["scheduled_rounds"] == 5
    assert row["weight_class"] == "Light Heavyweight"
    assert bool(row["title_fight"]) is True
    assert pd.Timestamp(row["date"]) == pd.Timestamp("2017-07-29")


def test_decision_has_no_finish_round():
    fights = build_fights(_raw_fights())
    row = fights.iloc[1]
    assert row["method"] == "decision"
    assert row["decision_subtype"] == "split"
    assert pd.isna(row["finish_round"])


def test_winner_b_and_draw():
    raw = _raw_fights()
    raw.loc[0, "Winner"] = "Daniel Cormier"
    raw.loc[1, "Winner"] = "Draw"
    fights = build_fights(raw)
    assert fights.iloc[0]["winner"] == "b"
    assert fights.iloc[1]["winner"] == "draw"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dataset_fights.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_fights'`

- [ ] **Step 3: Append `build_fights` to `src/mma/dataset.py`**

```python
def _winner_code(winner_raw, name_a: str, name_b: str) -> str:
    if winner_raw is None or (isinstance(winner_raw, float) and winner_raw != winner_raw):
        return "nc"
    text = str(winner_raw).strip()
    if text == name_a:
        return "a"
    if text == name_b:
        return "b"
    if text.lower() == "draw":
        return "draw"
    return "nc"


def build_fights(raw: pd.DataFrame) -> pd.DataFrame:
    c = COLUMN_MAP
    name_a = raw[c["r_fighter"]].astype(str).str.strip()
    name_b = raw[c["b_fighter"]].astype(str).str.strip()
    date = pd.to_datetime(raw[c["date"]], format="mixed", errors="coerce")

    fights = pd.DataFrame(
        {
            "date": date,
            "fighter_a_id": name_a.map(slugify),
            "fighter_b_id": name_b.map(slugify),
            "winner": [
                _winner_code(w, a, b)
                for w, a, b in zip(raw[c["winner"]], name_a, name_b)
            ],
            "method_raw": raw[c["win_by"]],
            "method": raw[c["win_by"]].map(map_method),
            "decision_subtype": raw[c["win_by"]].map(decision_subtype),
            "scheduled_rounds": raw[c["format"]].map(parse_scheduled_rounds),
            "weight_class": raw[c["fight_type"]].map(parse_weight_class),
            "title_fight": raw[c["fight_type"]].map(is_title_fight),
        }
    )
    fights["fight_id"] = (
        date.dt.strftime("%Y%m%d")
        + "-" + fights["fighter_a_id"]
        + "-vs-" + fights["fighter_b_id"]
    )
    # finish_round only defined for finishes; decisions/DQ get NA
    last_round = pd.to_numeric(raw[c["last_round"]], errors="coerce")
    is_finish = fights["method"].isin(["ko_tko", "submission"])
    fights["finish_round"] = last_round.where(is_finish)

    columns = [
        "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
        "method", "method_raw", "decision_subtype", "finish_round",
        "scheduled_rounds", "weight_class", "title_fight",
    ]
    return fights[columns].sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dataset_fights.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mma/dataset.py tests/test_dataset_fights.py
git commit -m "Add fights table builder"
```

---

### Task 7: Fight-stats table builder

Two rows per fight (one per fighter), unpivoted from the R_/B_ column pairs.

**Files:**
- Modify: `src/mma/dataset.py` (append `build_fight_stats`)
- Test: `tests/test_dataset_stats.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset_stats.py`:

```python
import pandas as pd

from mma.dataset import build_fight_stats


def _raw_fights():
    return pd.DataFrame(
        {
            "R_fighter": ["Jon Jones"],
            "B_fighter": ["Daniel Cormier"],
            "date": ["July 29, 2017"],
            "R_KD": [1], "B_KD": [0],
            "R_SIG_STR.": ["58 of 92"], "B_SIG_STR.": ["44 of 96"],
            "R_TOTAL_STR.": ["70 of 105"], "B_TOTAL_STR.": ["61 of 115"],
            "R_TD": ["1 of 2"], "B_TD": ["0 of 1"],
            "R_SUB_ATT": [0], "B_SUB_ATT": [1],
            "R_CTRL": ["2:10"], "B_CTRL": ["--"],
        }
    )


def test_two_rows_per_fight():
    stats = build_fight_stats(_raw_fights())
    assert len(stats) == 2
    assert set(stats["corner"]) == {"a", "b"}


def test_stat_values_unpivoted():
    stats = build_fight_stats(_raw_fights())
    a = stats[stats["corner"] == "a"].iloc[0]
    b = stats[stats["corner"] == "b"].iloc[0]
    assert a["fight_id"] == "20170729-jon-jones-vs-daniel-cormier"
    assert a["fighter_id"] == "jon-jones"
    assert a["kd"] == 1
    assert a["sig_landed"] == 58 and a["sig_attempted"] == 92
    assert a["td_landed"] == 1 and a["td_attempted"] == 2
    assert a["ctrl_sec"] == 130
    assert b["sub_att"] == 1
    assert pd.isna(b["ctrl_sec"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dataset_stats.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_fight_stats'`

- [ ] **Step 3: Append `build_fight_stats` to `src/mma/dataset.py`**

```python
def build_fight_stats(raw: pd.DataFrame) -> pd.DataFrame:
    c = COLUMN_MAP
    date = pd.to_datetime(raw[c["date"]], format="mixed", errors="coerce")
    slug_a = raw[c["r_fighter"]].astype(str).str.strip().map(slugify)
    slug_b = raw[c["b_fighter"]].astype(str).str.strip().map(slugify)
    fight_id = date.dt.strftime("%Y%m%d") + "-" + slug_a + "-vs-" + slug_b

    rows = []
    for corner, prefix, fighter_id in (("a", "R_", slug_a), ("b", "B_", slug_b)):
        sig = raw[prefix + c["sig_str"]].map(parse_landed_attempted)
        total = raw[prefix + c["total_str"]].map(parse_landed_attempted)
        td = raw[prefix + c["td"]].map(parse_landed_attempted)
        rows.append(
            pd.DataFrame(
                {
                    "fight_id": fight_id,
                    "fighter_id": fighter_id,
                    "corner": corner,
                    "kd": pd.to_numeric(raw[prefix + c["kd"]], errors="coerce"),
                    "sig_landed": [pair[0] for pair in sig],
                    "sig_attempted": [pair[1] for pair in sig],
                    "total_landed": [pair[0] for pair in total],
                    "total_attempted": [pair[1] for pair in total],
                    "td_landed": [pair[0] for pair in td],
                    "td_attempted": [pair[1] for pair in td],
                    "sub_att": pd.to_numeric(raw[prefix + c["sub_att"]], errors="coerce"),
                    "ctrl_sec": raw[prefix + c["ctrl"]].map(parse_mmss_seconds),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["fight_id", "corner"]
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dataset_stats.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mma/dataset.py tests/test_dataset_stats.py
git commit -m "Add fight-stats table builder"
```

---

### Task 8: `make_dataset.py` orchestrator + integrity checks

**Files:**
- Create: `scripts/make_dataset.py`
- Create: `tests/test_processed_data.py` (runs only when processed parquet exists)

- [ ] **Step 1: Write `scripts/make_dataset.py`**

File names below are placeholders confirmed/adjusted at the Task 4 decision point — adjust `FIGHTS_CSV` / `FIGHTERS_CSV` to the actual names from the schema report.

```python
"""Build processed parquet tables from raw CSVs. Reproducible end to end."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mma.dataset import build_fight_stats, build_fighters, build_fights

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FIGHTS_CSV = RAW / "raw_total_fight_data.csv"     # adjust per schema report
FIGHTERS_CSV = RAW / "raw_fighter_details.csv"    # adjust per schema report


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"integrity check failed: {message}")


def main() -> None:
    raw_fights = pd.read_csv(FIGHTS_CSV, sep=None, engine="python")
    raw_fighters = pd.read_csv(FIGHTERS_CSV, sep=None, engine="python")

    fighters = build_fighters(raw_fighters)
    fights = build_fights(raw_fights)
    stats = build_fight_stats(raw_fights)

    check(fights["fight_id"].is_unique, "duplicate fight_ids")
    check(fighters["fighter_id"].is_unique, "duplicate fighter_ids")
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

    PROCESSED.mkdir(parents=True, exist_ok=True)
    fighters.to_parquet(PROCESSED / "fighters.parquet", index=False)
    fights.to_parquet(PROCESSED / "fights.parquet", index=False)
    stats.to_parquet(PROCESSED / "fight_stats.parquet", index=False)

    print(f"fighters: {len(fighters)} rows")
    print(f"fights:   {len(fights)} rows, {fights['date'].min():%Y-%m-%d} .. {fights['date'].max():%Y-%m-%d}")
    print(f"stats:    {len(stats)} rows")
    print("\nmethod distribution:")
    print(fights["method"].value_counts(dropna=False).to_string())
    if orphans:
        print(f"\nwarning: {len(orphans)} orphan fighter ids (sample): {sorted(orphans)[:5]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/test_processed_data.py`**

```python
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "fights.parquet").exists(),
    reason="processed data not built (run scripts/make_dataset.py)",
)


def test_fights_reasonable_volume_and_range():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    assert len(fights) > 6000
    assert fights["date"].min().year <= 1995
    assert fights["date"].max().year >= 2024


def test_method_distribution_plausible():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    shares = fights["method"].value_counts(normalize=True)
    # UFC-wide base rates: decisions and ko/tko each roughly a third or more
    assert shares["decision"] > 0.30
    assert shares["ko_tko"] > 0.25
    assert shares["submission"] > 0.10


def test_finishes_have_round_decisions_do_not():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    finishes = fights[fights["method"].isin(["ko_tko", "submission"])]
    decisions = fights[fights["method"] == "decision"]
    assert finishes["finish_round"].notna().all()
    assert decisions["finish_round"].isna().all()


def test_stats_join_to_fights():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    assert set(stats["fight_id"]) == set(fights["fight_id"])
```

- [ ] **Step 3: Run the pipeline**

Run: `.venv/bin/python scripts/make_dataset.py`
Expected: row counts printed, all integrity checks pass, three parquet files in `data/processed/`. If a check fails, inspect the offending raw values and fix the relevant parser/mapper (with a new unit test reproducing the raw value) — do not weaken the check.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest -v`
Expected: all PASS including the four processed-data tests (no skips).

- [ ] **Step 5: Commit (including processed parquet)**

```bash
git add scripts/make_dataset.py tests/test_processed_data.py data/processed/*.parquet
git commit -m "Add dataset pipeline; build processed parquet tables"
git push
```

---

## Done criteria (Phase 1)

- `pytest` fully green, including processed-data integrity tests.
- `data/processed/{fighters,fights,fight_stats}.parquet` committed, covering 1994 → mid-2025.
- Fresh clone can reproduce with: `pip install -e ".[dev]"` → `python scripts/download_data.py` → `python scripts/make_dataset.py`.
