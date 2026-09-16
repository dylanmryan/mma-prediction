# MMA Fight Prediction

![Weekly data refresh](https://github.com/dylanmryan/mma-prediction/actions/workflows/refresh-data.yml/badge.svg)
![Tests](https://github.com/dylanmryan/mma-prediction/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**[▶ Try the live app](https://mma-prediction-lxqvmgheqvzccdpyord3q9.streamlit.app/)** — pick any two UFC fighters, get calibrated win probabilities with uncertainty.

A UFC fight predictor that says who wins, how, and in which round — as one
coherent probability distribution — built under the rule that **nothing ships
without clearing a bar fixed before the number was measured**.

It does not beat the betting market. That is stated up front because it is the
honest headline, it is measured rather than assumed, and the measuring is most
of what this project is.

---

## The engine, in one page

Predictions come from a **hybrid**: one model decides *who wins*, a second
decides *how and when*, and the two compose into a joint distribution over
every (winner × method × round) outcome.

```
                   ┌─ 5-seed XGBoost ensemble ─┐
 87 features ──────┤                           ├── average ── temperature ──►  P(A wins)
 (6 blocks)        └─ 5-seed neural ensemble ──┘        (fitted on held-out data)
                                                                   │
                   ┌─ per-round hazard model (KO / sub / survive) ─┐│
 fight context ────┤                                              ├┴──►  P(method, round │ winner)
                   └─ 40,000 Monte Carlo simulated fights ────────┘
                                                                   │
                                      joint = P(winner) × P(method, round │ winner)
```

- **The winner half is a blend** because two model families are wrong in
  different places. Alone they score 0.6487 (net) and 0.6488 (trees); averaged
  and re-calibrated they score **0.6459**. Their predictions correlate 0.85 and
  they disagree on the winner in 14.5% of fights — that disagreement is the
  entire gain.
- **The method/round half is a simulator.** A discrete-time hazard model gives
  per-round finish probabilities; 40,000 simulated fights per matchup turn
  those into a distribution. It improved the joint by 5.8× its bar.
- **One feature builder serves both training and inference** (`mma.serving`),
  so a feature cannot exist at training time and be missing or different at
  prediction time.

Deployed model version `18d1a70777ea`. Every artifact rebuilds byte-for-byte.

## How well it works

Expanding-window walk-forward, **4,856 fights over fold years 2018–2025**, each
scored by a model that never saw its year:

| Model | Accuracy | Log-loss |
|---|---|---|
| Coin flip | 0.500 | 0.693 |
| Elo baseline | — | 0.6827 |
| 5-seed XGBoost ensemble | 0.6153 | 0.6488 |
| 5-seed neural ensemble | 0.6209 | 0.6487 |
| **The deployed blend** | **0.6199** | **0.6459** |

On the 3,310 fights with odds, out of fold: **model 0.6463, closing line
0.6099.** The market wins by 0.036, and beats us on seven of eight
pre-registered subsets. ROI is negative at every threshold tested.

**Prospective**: 86 fights predicted before the event and committed to git,
57 graded — 0.5965 accuracy, 0.6602 log-loss, against a coin flip at 0.5088
and a higher-Elo-wins dummy at 0.4386. Every one belongs to the *July* model;
the hybrid has not yet made a graded prediction, and at ~13 graded fights a
month the confidence interval ([0.458, 0.724]) will not resolve a small
improvement for years.

## What was tried, and what it cost

Almost everything failed. That is the finding, and each row links to the
numbers:

| Experiment | Verdict |
|---|---|
| **8 feature blocks** measured individually | 1 cleared its own bar (`external`, −0.0034) |
| 4 more blocks, dead to the net | **alive to the trees** — they ship inside the blend |
| `rankings`, `in_fight` | neutral-to-negative on both members |
| **25-configuration capacity search** | winner cleared at −0.0039, **fresh seeds re-scored it to −0.0024** |
| **Two model families blended** | **shipped** — the project's one real win |
| **Monte Carlo fight simulator** | **shipped** — joint log-loss 2.143 vs 2.201 |
| Calibration remedies (temperature, Platt, isotonic) | 0 of 7 bands miscalibrated; best oracle repair 0.0004 vs a 0.003 bar |
| **Scorecard margin** as a training label | fails; the stated mechanism was *backwards* |
| A third, structurally different blend member | fails; 2 → 3 members is worth nothing |
| **Accumulated damage** (incl. knockdowns suffered, which the model never had) | no signal, at a floor that *could* have seen one |
| Ten pre-registered looks for any edge over the market | ten losses |

**→ [The full record, with every number](docs/EXPERIMENTS.md)**

Three things make those verdicts trustworthy: every bar was fixed in a
committed document *before* measuring; every noise floor was measured rather
than assumed (σ ranges from 0.0001 for the blend to 0.00125 for a single
XGBoost fit); and a search of *N* arms is judged against the σ√(2 ln N) a
best-of-*N* produces from nothing.

## Honest limitations

- **It loses to the market by 0.036 log-loss**, and no subset flips that.
- **The odds-free constraint is self-imposed** and costs roughly what the
  literature says devigged opening odds are worth. Public models that beat this
  one do it by using odds as a feature.
- **Three of six shipped feature blocks are decaying.** Their source snapshot
  froze at 2024-12-14: `external` is 53% unknown on 2026 fights, and `notice`
  is **100%** unknown — it contributes nothing to any prediction made today.
- **The prospective sample is too small to conclude anything** (n=57), and it
  measures a superseded model.
- Method and round are genuinely hard. The simulator improved them; they are
  still the weakest part.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py   # Kaggle UFC dataset -> data/raw/
.venv/bin/python scripts/make_dataset.py    # clean parquet -> data/processed/
.venv/bin/python scripts/build_ratings.py   # tune + build Elo ratings
.venv/bin/pytest
```

Interactive app: `streamlit run app.py` — win probability with uncertainty,
method and finish-round tendencies, and a "Why this prediction?" breakdown.

**What runs weekly.** `.github/workflows/refresh-data.yml` refreshes the
dataset, rebuilds every artifact, predicts upcoming events *before* they
happen, grades past ones, and commits the result. `scripts/` holds twelve
operational scripts it invokes; the rest are the experiments above.

## Repository map

| Path | What it is |
|---|---|
| `src/mma/` | the library: features, ratings, models, simulator, serving, inference |
| `scripts/` | operational pipeline (12 run weekly) plus one script per recorded experiment |
| `tests/` | 1,270 tests, including point-in-time and leak guards |
| `models/` | deployed artifacts, walk-forward reports, decision artifacts |
| `predictions/` | timestamped pre-event predictions and the graded track record |
| `docs/EXPERIMENTS.md` | the full experimental record |
| `docs/superpowers/plans/` | pre-registrations, written before each measurement |

## Data sources and licences

| Source | Used for | Licence |
|---|---|---|
| [Kaggle UFC dataset](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025) (`neelagiriaditya`) | the primary fights / fighters / round-stats tables | per the dataset page |
| [`ehan03/jds-mma-data`](https://github.com/ehan03/jds-mma-data) | pre-UFC career, nationality, notice and weigh-in tables (`data/external/`) | **MIT** (notice vendored as `data/external/LICENSE-jds-mma-data`) |
| [`jerzyszocik/ufc-rankings-history`](https://www.kaggle.com/datasets/jerzyszocik/ufc-rankings-history) | the weekly divisional rankings (`data/external/rankings.parquet`) | **CC0** |
| [`jerzyszocik/ufc-betting-odds-daily-dataset`](https://www.kaggle.com/datasets/jerzyszocik/ufc-betting-odds-daily-dataset) | the market benchmark only — never a model feature | **CC0** |
| [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats) | the secondary daily source, merged on top of the primary by every rebuild (`scripts/refresh_secondary.py`, driven by `scripts/make_dataset.py`) — fights, fight totals, per-round stats and debutant fighter rows | **GPL-3.0** — used as a *data* source only: its published CSVs are fetched over HTTPS and no GPL code is vendored. The primary wins on any overlapping fight; every row's origin is recorded in `data/processed/provenance.parquet` |
| Wikipedia event pages | upcoming fight cards, and the withdrawal / missed-weight parser | CC BY-SA 4.0 |

Only *derived*, compact tables are committed (a few MB); the raw snapshots
are never vendored and are regenerated by `scripts/build_external.py` and
`scripts/build_rankings.py`.

## Development notes

- **Local-disk virtualenv.** If the repo lives in an iCloud-synced folder,
  create the venv elsewhere (`python3 -m venv ~/.venvs/mma && ~/.venvs/mma/bin/pip install -e ".[dev,app]"`)
  — torch's shared libraries stall for minutes when paged in from iCloud.
  Better still, keep the repo itself outside iCloud (or exclude `.git` from
  sync): sync has corrupted `.git` metadata here more than once.
- `OMP_NUM_THREADS=1` for any script that imports both torch and xgboost.
- **CI.** `.github/workflows/tests.yml` runs the suite on every pull request
  and every push to `main`. It needs no secrets: everything the tests read —
  the processed tables, the deployed artifacts, the walk-forward reports — is
  committed. The weekly refresh runs pytest too, but only when new fights
  arrived, so it is not a gate.
- **Model identity and evidence.** Prospective predictions are stamped with
  `mma.versioning.model_version()`, a hash of **everything that scores** —
  the torch weights and preprocessing statistics, the twenty-five per-seed
  XGBoost boosters (winner/method/round, plus SP3's hazard and decision), and
  the two config artifacts whose hand-set numbers are as load-bearing as any
  trained one, `models/blend.json` (weight, temperature) and
  `models/simulator.json` (`n_runs`, `alpha`, `sim_seed`, `default_rounds`) —
  so the track record splits by model, not by commit.
  Before SP2.2 the hash covered only the torch half, which was correct while
  XGBoost was an explainer and would have covered half of what serves now;
  SP3 needed the same expansion again for the simulator. It is verified
  sensitive to a mutation on any member. `models/display_calibration.json`
  stays excluded: since SP3 nothing recalibrates a displayed number, so that
  file is a measurement of the model rather than an input to a prediction, and
  hashing it would open a new track-record section for byte-identical
  predictions whenever the weekly measurement moved a decimal.
  Retraining (the weekly refresh, or a deliberate refit by hand) starts a new
  section automatically; retraining is deterministic, so an unchanged
  dataset yields an unchanged version. The evidence behind each deployed
  model lives in `models/walkforward/` (the harness reports, noise floors,
  and refit decision), and the metrics files quote it. After a weekly data
  refresh the refit reuses the committed budget (6 epochs / T 1.15;
  109/73/71 trees on each of five seeds) on the newer data, but the harness
  reports are *not* re-run by the Action — run `scripts/revalidate_recipe.py`
  by hand to re-measure every recorded bar on the current table (it re-runs
  the harness for the deployed configuration and its paired comparisons, and
  writes `models/walkforward/recipe_revalidation.json`). The train scripts
  warn when the harness's data is older than the training cutoff, and the
  weekly Action reports the same gap through
  `scripts/revalidate_recipe.py --check-staleness`, which is the signal that
  it is worth running the full thing.

  All of them ask that question through one read, `mma.staleness`, and the
  answer is not the date comparison alone: a *passing* re-validation whose
  table reaches the training cutoff **is** the missing measurement, so the
  warning goes quiet even though each member's own harness report still
  records the older table (it did run on that table; nothing rewrites that).
  It comes back the moment the table grows past what the re-validation
  reached, and a re-validation that ran and *failed* never counts as cover —
  that gap is more actionable, not less, and says so. Before this was
  centralised the comparison existed five times over, in the three train
  scripts and two provenance tests, and every copy was still warning about a
  gap the 2026-09-09 re-validation had already closed. A standing warning
  nobody can act on is how a real one gets scrolled past, which is why
  `tests/test_staleness.py` pins the committed tree as covered: when it
  genuinely runs ahead, the suite says so rather than staying quiet.

## Interactive app

`app.py` is a Streamlit front end over the committed **blend**: pick two
fighters, a weight class, round count, and title-fight flag, and it renders
the win probability, the spread across the five per-seed blends, an
MC-dropout uncertainty histogram, method-of-victory / finish-round
breakdown, both fighters' Elo trajectories, and a "Why this prediction?"
panel breaking down the top factors driving the call — all from the
artifacts already checked into `models/torch/` and
`models/xgb_<head>_seed*.json`, no training required. Predictions are
symmetrized across both fighter orderings
(`mma.inference.predict_symmetrized`), and **the thing being symmetrized is
the blend**: each corner ordering is blended and calibrated first, then the
two blended probabilities are averaged, so the reported probability is
always self-consistent. The "Why this prediction?" panel averages TreeSHAP
over all five boosters and both orientations — the XGBoost winner head is
now half the deployed scorer rather than a companion model that merely
agreed with it — and says plainly what it still is not: an attribution of
the whole blend, since the neural half is not decomposed and the
post-average temperature rescales the blended logit.

Run it locally:

```bash
.venv/bin/pip install -e ".[dev,app]"
.venv/bin/streamlit run app.py
```

Deploy to Streamlit Community Cloud: push this repo to a public GitHub
remote, go to [share.streamlit.io](https://share.streamlit.io), click
**New app**, and point it at this repo/branch with `app.py` as the entry
point.
