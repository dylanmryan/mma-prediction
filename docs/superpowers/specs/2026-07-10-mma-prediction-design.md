# MMA Fight Prediction — Design

**Date:** 2026-07-10
**Status:** Approved
**Goal:** A portfolio-grade ML project that predicts UFC fight winners and method of victory, with a rigorous baseline progression (Elo → XGBoost → PyTorch), uncertainty quantification, and a free interactive Streamlit app.

## Summary

The project builds a staged pipeline where each stage is shippable and each model must beat the last under an honest, time-based evaluation:

1. **Data layer** — Kaggle UFC dataset normalized into three parquet tables (scraper swaps in later, same schema).
2. **Elo system** — MMA-tuned Elo; Baseline #1 and a feature generator.
3. **Features + XGBoost** — point-in-time features; Baseline #2 (the model to beat).
4. **PyTorch multi-output net** — shared trunk, winner + method heads, ensembles + MC dropout for uncertainty, temperature-scaled calibration.
5. **Streamlit app** — free hosting on Community Cloud; matchup picker, win probability with uncertainty visual, method breakdown.
6. **Scraper** — ufcstats.com, incremental, weekly GitHub Action keeps data and app current.

The README leads with the story: "Elo → GBM → deep learning, honestly evaluated," plus method-of-victory prediction and calibrated, uncertainty-aware probabilities as differentiators.

## Prediction targets

- **Winner**: binary (fighter A wins).
- **Method**: 3 classes — KO/TKO (incl. doctor stoppage), Submission, Decision (all types). Raw method strings preserved for display. Combined outputs like "A by KO" = P(A wins) × P(KO), validated against a 6-class variant in an ablation.
- **Finish round**: 4 classes — R1, R2, R3, R4–5 (grouped) — defined only for finishes; "Decision" in the method head already covers going the distance. Rounds 4–5 masked for 3-round fights. Full outcomes compose by multiplication: P(A by KO in R1) = P(A) × P(KO) × P(R1 | finish). Framed as a calibrated distribution (noisiest target; base rates are strong).

## 1. Data layer

Source: Kaggle "UFC complete" style dataset (~7k+ fights, 1993–present) now; own scraper later.

Three parquet tables produced by a single reproducible `make_dataset.py`:

- **`fighters`** — one row per fighter: name, height, reach, stance, DOB.
- **`fights`** — one row per fight: date, event, weight class, fighter A/B ids, winner, method (mapped to 3 classes + raw string), rounds scheduled/ended.
- **`fight_stats`** — one row per fighter per fight: sig. strikes landed/attempted, takedowns, control time, knockdowns, submission attempts.

Cleaning: name deduplication, missing reach/height imputation with flags, method mapping.

**Point-in-time rule (non-negotiable):** any feature for a fight uses only data dated strictly before that fight. This is the #1 leakage source in public UFC models and is enforced by construction and by tests.

## 2. Elo system

- Start 1500; update `K * (actual − expected)`, standard logistic expected score; strict chronological processing.
- **K schedule:** ~40 for a fighter's first 5 fights, ~24 after.
- **Finish bonus:** ~1.2× K multiplier for KO/sub wins.
- **No inactivity decay in v1** (future experiment; returning-fighter fairness issue).
- **Three ratings per fighter:** overall, striking, grappling — striking/grappling updates weighted by fight-stats dominance of each phase.
- **Tuning:** K and bonuses maximize predictive log-loss on pre-2020 data only.
- **Output:** `ratings` table — every fighter's ratings as of each fight date. Elo as baseline predicts via expected-score formula; as features it contributes levels, diffs, and `avg_opponent_elo_last_5`.

Note on opponent strength: Elo already adjusts ratings for opponent quality; `avg_opponent_elo_last_5` exists to contextualize the *rolling stats* (which are not opponent-adjusted). Correlated features are acceptable — learned models split credit; this is not double counting.

## 3. Features + XGBoost baseline

One feature-builder module → one model-ready table, one row per fight, A-minus-B differentials plus selected absolutes:

- **Physical/bio:** reach diff, height diff, age diff, ages, stance matchup flag.
- **Elo block:** overall/striking/grappling Elo diffs, fight-count (uncertainty proxy), `avg_opponent_elo_last_5` diff.
- **Rolling/career stats (pre-fight only; last 3/5 and career):** win rate, finish rate, sig. strikes landed & absorbed per min, TD accuracy/defense, control time share, knockdowns per fight, streak.
- **Context:** weight class, title fight flag, scheduled rounds, days since last fight.
- **Symmetry:** random A/B assignment (or both orders) so column order can't encode the winner.

XGBoost trains winner (binary), method (3-class), and finish-round (4-class, finishes only) models. Feature importances feed the README analysis.

### Cold start: UFC debutants

Fighters arriving from other organizations have no UFC history. Handling:

- **Elo:** start at 1500 with high early-career K (~40) for fast convergence; `ufc_fight_count` serves as a rating-confidence feature; the expected-score formula already discounts wins over unknown 1500-rated debutants.
- **Missing rolling stats are left missing, not faked:** XGBoost handles NaN natively; the neural net gets neutral imputation plus missingness-indicator flags.
- **Explicit `ufc_debut` flag** (and debut-vs-veteran matchup flag) so the model can learn debut effects directly.
- **Pre-UFC career record** (W-L, win rate, finish rate at fight time) used as debut features where the dataset provides it.
- **Evaluation reported separately** for fights involving debutants vs established-vs-established, so model weakness on debuts is visible, not averaged away.

Out of scope for v1: fight-level pre-UFC data (Sherdog/Tapology) and org-based Elo entry priors (e.g., ex-Bellator champions seeded above 1500) — listed as a v2 experiment.

## Evaluation protocol (locked, shared by all models)

- **Time-based split:** train < ~2021; validation 2021–2023 (all tuning); test 2024+ (untouched until final).
- **Winner metrics:** accuracy, log-loss, Brier, calibration curve.
- **Method metrics:** accuracy, macro-F1.
- **Finish-round metrics:** accuracy, macro-F1 (evaluated on finishes only), vs base-rate dummy.
- **Baselines in every results table:** higher-Elo-wins dummy, Elo expected-score, XGBoost, neural net.
- **No random splits anywhere.**

## 4. PyTorch multi-output network

- **Inputs:** same feature table as XGBoost (fair comparison); numerics standardized; weight class via learned embedding.
- **Trunk:** 2–3 FC layers (~128 → 64), ReLU, batch norm, dropout ~0.3.
- **Heads:** winner (sigmoid) + method (3-way softmax) + finish round (4-way softmax, trained on finishes, R4–5 masked for 3-round fights). Loss = weighted BCE + CE + CE (multi-task).
- **Training:** AdamW, early stopping on val log-loss, honest hyperparameter search on validation years only.
- **Uncertainty:** 5-seed deep ensemble (headline = mean, spread = confidence range); MC dropout (100 passes) for distribution visuals.
- **Calibration:** temperature scaling fit on validation; before/after curves in README.
- **Honest framing:** NN may only match XGBoost on 7k rows — the comparison is the finding; multi-task + uncertainty is the differentiator regardless.

## 5. Streamlit app

Free on Streamlit Community Cloud, deployed from this repo (`app.py`). Loads pre-exported artifacts (parquet + model weights); no training or DB at runtime.

- **Matchup picker:** two searchable fighter dropdowns, weight-class filter; tale-of-the-tape with Elo trajectory sparkline.
- **Prediction panel:** ensemble win-probability bar, MC-dropout probability distribution strip, method breakdown grid ("A by KO 28% …"), and a fight-duration distribution (finish round probabilities + expected rounds) using latest ratings/stats.
- **Model card footer:** test metrics vs baselines, repo link.

## 6. Scraper

`scrape_ufcstats.py`: requests + BeautifulSoup, ~1 req/sec, raw HTML cached, incremental mode (fetch only events newer than latest stored). Writes the same three tables as the Kaggle loader. Weekly GitHub Action refreshes data and commits, keeping the app current.

## Repo structure

```
mma_prediction/
├── data/            # raw/ (gitignored), processed/ parquet
├── src/mma/         # dataset.py, elo.py, features.py, models/ (xgb.py, net.py),
│                    # evaluate.py, scrape_ufcstats.py
├── scripts/         # make_dataset.py, train.py, predict_event.py
├── app.py           # Streamlit
├── tests/           # pytest: elo math, point-in-time correctness, feature builder
├── notebooks/       # EDA + results analysis
└── README.md        # story, results table, app link
```

## Build order

Phases 1–6 in sequence: data → Elo → features/XGBoost → PyTorch → app → scraper. Each phase ends committed and demonstrable. Tests prioritize Elo update math and time-leakage prevention.

## Out of scope (v1)

- Non-UFC organizations (incl. fight-level pre-UFC data and org-based Elo entry priors); betting odds comparison; inactivity decay in Elo; round-by-round fight simulation; decision sub-type prediction (unanimous vs split) beyond display.
