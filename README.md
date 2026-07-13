# MMA Fight Prediction

![Weekly data refresh](https://github.com/dylanmryan/mma-prediction/actions/workflows/refresh-data.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Predicting UFC fight winners, method of victory, and finish round —
an Elo rating system, gradient boosting, and a calibrated multi-task
neural-network ensemble, each evaluated honestly on strict time splits,
plus an interactive Streamlit matchup explorer.

**Highlights**

- **Point-in-time discipline, machine-verified**: every feature is built only
  from data available before each fight; a truncation-invariance test proves
  no feature can see the future.
- **Baseline ladder**: coin flip → Elo (0.576 acc) → XGBoost (0.595) →
  5-seed calibrated neural ensemble (0.606, best log-loss) on 1,507
  never-tuned-on validation fights.
- **Uncertainty done properly**: deep-ensemble spread + MC dropout, per-seed
  temperature scaling, display probabilities recalibrated to historical base rates.
- **Self-updating**: a weekly GitHub Action refreshes the dataset and rebuilds
  every artifact; the entire pipeline reproduces byte-for-byte.

Design doc: `docs/superpowers/specs/2026-07-10-mma-prediction-design.md` ·
Phase plans: `docs/superpowers/plans/`

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py   # Kaggle UFC dataset -> data/raw/
.venv/bin/python scripts/make_dataset.py    # clean parquet -> data/processed/
.venv/bin/python scripts/build_ratings.py   # tune + build Elo ratings
.venv/bin/pytest
```

Interactive app: `streamlit run app.py` (pick any two fighters, get win
probability with uncertainty, method and finish-round tendencies).

Data bootstraps from the [Kaggle UFC dataset](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
via `kagglehub` and auto-refreshes weekly from the same maintained mirror
(`scripts/refresh_data.py`). A direct ufcstats.com scraper was planned but
dropped — the site gates automated clients behind an anti-bot challenge;
`src/mma/parsing.py` is retained in case a future raw-string data source
needs its parsing helpers. The `.github/workflows/refresh-data.yml` Action
runs this refresh weekly and commits any rebuilt artifacts automatically.

## Results so far

All models are evaluated on a strict time split: trained on pre-2021 fights,
reported on 2021–2023 validation fights. **Test years (2024+) are held out
until the final model comparison.**

**Winner prediction** (1,507 validation fights):

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Coin flip | 0.500 | 0.693 | 0.250 |
| Higher-Elo-wins dummy | 0.573 | — | — |
| Elo baseline | 0.576 | 0.678 | 0.242 |
| XGBoost (46 features) | 0.595 | 0.658 | 0.233 |
| **Neural net** (5-seed ensemble, calibrated) | **0.606** | **0.654** | **0.231** |

**Method of victory** (3 classes): XGBoost 0.489 accuracy vs 0.485 majority-class
baseline, macro-F1 0.281. The neural net's class-weighted method head makes a
different trade: 0.407 accuracy but **macro-F1 0.364** — it actually identifies
submissions and KOs instead of defaulting to "decision". **Finish round**
(R1/R2/R3/R4-5, finishes only): XGBoost 0.477 accuracy vs 0.474 majority
baseline, macro-F1 0.167. Predicting *how* fights end is genuinely hard; these
numbers are reported honestly rather than hidden.

The neural net is a multi-task network (shared trunk; winner, method, and
finish-round heads) trained as a deterministic 5-seed ensemble with per-seed
temperature scaling — fitted temperatures all land near 1.0, i.e. the raw
model was already well calibrated. Uncertainty comes from ensemble spread
(mean 0.087) and MC dropout. Per the Phase 3 ablation, the era-proxy
`*_missing` flags are excluded from its inputs.

What predicts the winner? Reach and age differentials, Elo differential, and
opponent-quality-adjusted activity rates top the feature importances. A caveat
worth knowing: reach-*missingness* indicators rank highly, which likely proxies
for era and fighter obscurity rather than physiology.

The Elo system: fighters start at 1500; K = 64 for a fighter's first 5 UFC
fights then 48 (tuned by grid search on pre-2020 log-loss); KO/sub wins get a
1.4× update bonus; separate striking and grappling Elos update in proportion
to how striking-dominated each fight was.

All-time peak Elo (through 2025-09):

| # | Fighter | Peak Elo |
|---|---|---|
| 1 | Jon Jones | 1985 |
| 2 | Islam Makhachev | 1938 |
| 3 | Georges St-Pierre | 1916 |
| 4 | Anderson Silva | 1912 |
| 5 | Kamaru Usman | 1907 |
| 6 | Charles Oliveira | 1877 |
| 7 | Max Holloway | 1872 |
| 8 | Francis Ngannou | 1870 |
| 9 | Khabib Nurmagomedov | 1867 |
| 10 | Tony Ferguson | 1867 |

## Interactive app

`app.py` is a Streamlit front end over the committed ensemble: pick two
fighters, a weight class, round count, and title-fight flag, and it renders
the win probability, ensemble spread, MC-dropout uncertainty histogram,
method-of-victory / finish-round breakdown, and both fighters' Elo
trajectories — all from the artifacts already checked into `models/torch/`,
no training required. Predictions are symmetrized across both fighter
orderings (`mma.inference.predict_symmetrized`) so the reported probability
is always self-consistent.

Run it locally:

```bash
.venv/bin/pip install -e ".[dev,app]"
.venv/bin/streamlit run app.py
```

Deploy to Streamlit Community Cloud: push this repo to a public GitHub
remote, go to [share.streamlit.io](https://share.streamlit.io), click
**New app**, and point it at this repo/branch with `app.py` as the entry
point.
