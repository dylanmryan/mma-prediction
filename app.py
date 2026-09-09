"""MMA matchup predictor -- Streamlit app over the committed blended scorer."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

# Streamlit Cloud installs from requirements.txt only (no `pip install -e .`),
# so the `mma` package under src/ isn't on sys.path unless we put it there.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from mma.explain import contributions, humanize, load_boosters
from mma.inference import (
    BLEND_REPORT,
    BlendedPredictor,
    apply_prior_correction,
    build_matchup,
    predict_symmetrized,
)
from mma.snapshots import build_snapshots

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DISPLAY_PRIORS = ROOT / "models" / "torch" / "display_priors.json"
TORCH_METRICS = ROOT / "models" / "torch" / "metrics_val.json"
XGB_METRICS = ROOT / "models" / "xgb_metrics_val.json"
ELO_WALKFORWARD = ROOT / "models" / "walkforward" / "elo.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _fold_span(fold_years: list) -> str:
    """'2018-2025 (+2026)' from the harness fold years; the last fold absorbs
    every fight after its year, so the newest partial year rides along."""
    if not fold_years:
        return "2018-2025 (+2026)"
    return f"{min(fold_years)}-{max(fold_years)} (+{max(fold_years) + 1})"


def model_card_text(weight: float, temperature: float) -> str:
    """Model-card caption built from the committed metrics files and the
    deployed blend's own harness report.

    `weight`/`temperature` are the deployed scorer's own committed
    configuration (`models/blend.json`, via `mma.inference.load_blend_config`
    -- the caller passes `predictor.weight`/`predictor.temperature` so the
    card names the numbers actually serving rather than a re-read that could
    disagree with them). The deployed scorer is a BLEND, so the card's
    headline numbers are the blend's -- models/walkforward/blend_b1.json, the
    report SP2.2's decision shipped -- and the two members' own pooled
    numbers are shown beside it, as the comparison that makes the case for
    blending them. Under the refit_through recipe the metrics files carry no
    held-out slice of their own; they carry the harness evidence behind each
    member's budget, and the training cutoff. Falls back to a numberless
    card if a file is missing or unreadable.
    """
    torch_m = _read_json(TORCH_METRICS)
    xgb_m = _read_json(XGB_METRICS)
    elo_m = _read_json(ELO_WALKFORWARD).get("pooled", {})
    blend = _read_json(BLEND_REPORT)
    head = (
        f"Model card: a blend — {weight:g}/{1 - weight:g} average of a "
        "5-seed XGBoost ensemble and a 5-seed multi-task net "
        "(winner/method/finish-round), temperature-calibrated after averaging "
        f"(T={temperature:g}). ")
    tail = ("The prospective track record (predictions/track_record.json) is the "
            "only true holdout. Method and round probabilities assume independence "
            "from the winner, and predictions are symmetrized across both fighter "
            "orderings (see code comments). "
            "[Source](https://github.com/dylanmryan/mma-prediction)")

    def _triple(block: dict) -> str | None:
        # metrics files say "log_loss"; the harness's pooled blocks say
        # "winner_log_loss"
        log_loss = block.get("log_loss", block.get("winner_log_loss"))
        try:
            return (f"{block['accuracy']:.3f} accuracy, {log_loss:.3f} "
                    f"log-loss, {block['brier']:.3f} Brier")
        except (KeyError, TypeError, ValueError):
            return None

    blend_line = _triple(blend.get("pooled", {}))
    torch_line = _triple(torch_m.get("winner_ensemble", {}))
    if blend_line and torch_m.get("mode") == "refit_through":
        n = blend.get("pooled", {}).get("n")
        n_text = f"n={n:,} pooled fights" if isinstance(n, int) else "pooled"
        ece = blend.get("pooled", {}).get("ece")
        ece_text = f", ECE {ece:.4f}" if isinstance(ece, (int, float)) else ""
        rivals = []
        if torch_line:
            rivals.append(f"net alone {torch_line}")
        xgb_line = _triple(xgb_m.get("winner", {}))
        if xgb_m.get("mode") == "refit_through" and xgb_line:
            rivals.append(f"XGBoost alone {xgb_line}")
        elo_line = _triple(elo_m)
        if elo_line:
            rivals.append(f"Elo {elo_line}")
        rival_text = f" — vs {'; '.join(rivals)}" if rivals else ""
        n_train = torch_m.get("n_train")
        n_train_text = f"{n_train:,}" if isinstance(n_train, int) else "?"
        return (
            head
            + f"Evaluated by expanding-window walk-forward "
            f"{_fold_span(blend.get('fold_years', []))}, {n_text}: "
            f"{blend_line}{ece_text}{rival_text}. Both members are refit on every "
            f"decisive fight through {torch_m.get('train_through', '?')} "
            f"({n_train_text} fights), so no historical year is "
            "held out from them. " + tail
        )
    if torch_line:
        n = torch_m["winner_ensemble"].get("n_val")
        n_text = f"{n:,}" if isinstance(n, int) else "?"
        return (head + f"Held-out validation ({n_text} fights), neural member only: "
                f"{torch_line}. " + tail)
    return head + "Metrics files not found. " + tail

st.set_page_config(page_title="MMA Fight Predictor", page_icon="🥊", layout="wide")


@st.cache_resource
def load_xgb_boosters():
    """The winner head's five seed boosters -- half of the deployed blend, and
    the half TreeSHAP can decompose."""
    return load_boosters()


@st.cache_resource
def load_everything():
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    stats = pd.read_parquet(PROCESSED / "fight_stats.parquet")
    fighters = pd.read_parquet(PROCESSED / "fighters.parquet")
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    snapshots = build_snapshots(fights, stats, ratings)
    # The deployed scorer: both members, plus the deployed weight and
    # post-average temperature it was loaded with (models/blend.json, via
    # mma.inference.load_blend_config -- see predictor.weight/.temperature).
    predictor = BlendedPredictor.load()
    as_of = fights["date"].max()
    weight_classes = sorted(fights["weight_class"].dropna().unique().tolist())
    # Mean-matching correction factors precomputed by
    # scripts/build_display_priors.py: empirical base rates over the rows the
    # deployed members trained on (every fight through their train_through
    # under the refit recipe) divided by the BLEND's mean prediction there.
    display_factors = json.loads(DISPLAY_PRIORS.read_text())
    return (
        fights, fighters.set_index("fighter_id"), ratings, snapshots, predictor,
        as_of, weight_classes, display_factors,
    )


fights, fighters, ratings, snapshots, predictor, as_of, weight_classes, display_factors = load_everything()

eligible = snapshots.join(fighters[["name"]], how="inner").sort_values("name")
names = eligible["name"].tolist()
by_name = {name: fighter_id for fighter_id, name in eligible["name"].items()}

st.title("🥊 MMA Fight Predictor")
st.caption(
    f"Elo → XGBoost → neural ensemble → calibrated blend, honestly evaluated. "
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

    # Documented deviation from the Phase 5 plan: the model is not exactly
    # symmetric under fighter order (P(A beats B) + P(B beats A) can miss 1.0
    # by ~15pp -- see docs/superpowers/plans/2026-07-11-phase5-streamlit.md
    # Task 3). We predict both orientations (A-vs-B and B-vs-A) and average
    # via `predict_symmetrized`, which guarantees the reported probability is
    # exactly self-consistent (p_A + p_B == 1) instead of merely approximate.
    matchup_ab = build_matchup(
        snap_a, snap_b, bio_a, bio_b,
        weight_class or "Lightweight", title_fight, scheduled_rounds, as_of,
    )
    matchup_ba = build_matchup(
        snap_b, snap_a, bio_b, bio_a,
        weight_class or "Lightweight", title_fight, scheduled_rounds, as_of,
    )
    # The BLEND is what gets symmetrized: each orientation is blended and
    # temperature-scaled first, then the two are corner-averaged.
    result = predict_symmetrized(predictor, matchup_ab, matchup_ba)
    p_a = result["winner_prob"]
    spread = result["winner_spread"]

    st.subheader("Prediction")
    st.progress(p_a, text=f"{name_a}: {p_a:.0%}  ·  {name_b}: {1 - p_a:.0%}")
    st.caption(
        f"Ten models (5 XGBoost seeds + 5 net seeds); the per-seed blends range "
        f"±{spread / 2:.1%} around the mean."
    )

    # MC dropout runs on the (cheap) A-vs-B orientation only, then shifts the
    # samples by the same delta that symmetrization applied to the headline
    # probability, so the displayed distribution is centered on `p_a` above
    # rather than on the single-orientation `orientation_ab_prob`.
    # MC dropout exists only in the neural member; the samples are re-centred
    # on the blend's headline probability below, so read the spread as the
    # net's parameter uncertainty drawn around the blend's mean.
    samples = predictor.mc_dropout(matchup_ab, passes=100)[:, 0]
    samples = np.clip(samples + result["mc_dropout_shift"], 0.0, 1.0)
    histogram = np.histogram(samples, bins=20, range=(0.0, 1.0))[0]
    st.bar_chart(
        pd.DataFrame({"MC dropout samples": histogram},
                     index=[f"{edge / 20:.2f}" for edge in range(20)]),
        height=160,
    )

    # Explanations are native TreeSHAP (exact per-prediction attribution) from
    # the XGBoost winner seed ensemble -- which since SP2.2 is HALF OF THE
    # MODEL ABOVE, not a companion that merely agrees with it. Averaged over
    # the same five seeds the blend averages, and over both orientations. See
    # mma.explain for what this does and does not attribute.
    with st.expander("Why this prediction?"):
        boosters = load_xgb_boosters()
        factors = humanize(
            contributions(matchup_ab, matchup_ba, boosters=boosters),
            name_a, name_b, top_n=6,
        )
        chart = pd.DataFrame(
            {"Log-odds contribution": [row["contribution"] for row in factors]},
            index=[f"{row['label']} (favors {row['favors']})" for row in factors],
        )
        st.bar_chart(chart, horizontal=True, height=260)
        st.caption(
            "Factor attributions from the XGBoost half of the blend (TreeSHAP, "
            f"exact, averaged over its 5 seeds) — {predictor.weight:.0%} of the "
            "probability above. The neural half is not decomposed, so read "
            "these as what the tree half saw rather than the whole story."
        )

    # Final review finding: the raw model heads are trained with class-weighted
    # loss, so their softmax outputs overstate rare classes (e.g. predicted
    # P(rounds 4-5) for 5-round fights was ~3.8x the empirical rate). We
    # recalibrate before display with mean-matching factors
    # (empirical prior / mean model prediction on the training split, see
    # mma.inference.compute_correction_factors) and never show raw numbers.
    method_raw = dict(zip(result["method_classes"], result["method_probs"]))
    round_raw = dict(zip(result["round_classes"], result["round_probs"]))
    round_key = "round_3" if scheduled_rounds <= 3 else "round_5"
    method = apply_prior_correction(method_raw, display_factors["method"])
    rounds = apply_prior_correction(round_raw, display_factors[round_key])

    labels = {"ko_tko": "KO/TKO", "submission": "Submission", "decision": "Decision"}
    outcome_rows = []
    for fighter, p_fighter in ((name_a, p_a), (name_b, 1 - p_a)):
        for method_name in result["method_classes"]:
            outcome_rows.append(
                {
                    "Winner": fighter,
                    "Method": labels[method_name],
                    "Probability": f"{p_fighter * method[method_name]:.1%}",
                }
            )
    st.subheader("How it ends")
    st.caption(
        "Method and round splits are recalibrated to historical base rates; "
        "treat as tendencies, not betting odds."
    )
    st.dataframe(pd.DataFrame(outcome_rows), hide_index=True)
    round_labels = ["Round 1", "Round 2", "Round 3", "Rounds 4-5"]
    finish_share = 1 - method["decision"]
    st.caption(
        "If it doesn't go the distance ("
        + f"{finish_share:.0%} chance), the finish comes in: "
        + "  ·  ".join(
            f"{label} {rounds[cls]:.0%}"
            for label, cls in zip(round_labels, result["round_classes"])
            if rounds[cls] > 0.001
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

# Prospective track record: public, timestamped predictions for real
# upcoming events (predictions/, written weekly by scripts/predict_upcoming.py)
# graded automatically after each event (scripts/grade_predictions.py). This
# is the strongest evaluation in the project -- committed before results are
# known -- so it's surfaced here too, not just in the README. Silently
# skipped if track_record.json doesn't exist yet or has nothing graded, so
# the app never shows a broken or all-null section.
TRACK_RECORD = ROOT / "predictions" / "track_record.json"
if TRACK_RECORD.exists():
    track_record = json.loads(TRACK_RECORD.read_text())
    if track_record.get("overall", {}).get("n_graded", 0) > 0:
        st.divider()
        st.subheader("Prospective track record")
        st.caption(
            "Real upcoming UFC events, predicted and committed to git before "
            "they happen, graded automatically afterward. See "
            "[predictions/](https://github.com/dylanmryan/mma-prediction/tree/main/predictions) "
            "and [track_record.json](https://github.com/dylanmryan/mma-prediction/blob/main/predictions/track_record.json)."
        )
        overall = track_record["overall"]
        cols = st.columns(4)
        cols[0].metric("Fights predicted", overall["n_predicted"])
        cols[1].metric("Fights graded", overall["n_graded"])
        cols[2].metric("Accuracy", f"{overall['accuracy']:.1%}" if overall["accuracy"] is not None else "—")
        cols[3].metric("Log-loss", f"{overall['log_loss']:.3f}" if overall["log_loss"] is not None else "—")
        baselines = track_record.get("baselines", {})
        coin_flip = baselines.get("coin_flip", {})
        elo_dummy = baselines.get("higher_elo_dummy", {})
        if coin_flip.get("accuracy") is not None:
            st.caption(
                f"vs. coin flip {coin_flip['accuracy']:.1%} accuracy / "
                f"{coin_flip['log_loss']:.3f} log-loss, higher-Elo dummy "
                f"{elo_dummy.get('accuracy', float('nan')):.1%} accuracy."
            )

# Model vs. the betting market: the honest "does it beat Vegas?" answer, from
# scripts/build_odds_benchmark.py comparing the ensemble against devigged
# sportsbook closing lines on out-of-sample (2021+) fights. Betting odds are
# an evaluation comparator only, never a model feature. Skipped silently if
# the benchmark artifact isn't present.
MARKET_BENCHMARK = ROOT / "models" / "market_benchmark.json"
if MARKET_BENCHMARK.exists():
    benchmark = json.loads(MARKET_BENCHMARK.read_text())
    head = benchmark["headline_2021_plus"]
    st.divider()
    st.subheader("Model vs. the betting market")
    st.caption(
        f"On {head['n_fights']:,} fights from 2021 onward, the model's win "
        "probabilities vs. devigged sportsbook closing lines — computed once "
        f"({benchmark.get('computed_once_on', 'July 2026')}) against the "
        "pre-refit model, for which those years were out of sample; the "
        "currently deployed model trains through the latest event, so this "
        "comparison is not recomputed. Betting odds are an evaluation "
        "yardstick here, never a model input."
    )
    compare = pd.DataFrame(
        {
            "Model": [
                f"{head['model']['accuracy']:.3f}",
                f"{head['model']['log_loss']:.3f}",
                f"{head['model']['brier']:.3f}",
            ],
            "Market (Vegas)": [
                f"{head['market']['accuracy']:.3f}",
                f"{head['market']['log_loss']:.3f}",
                f"{head['market']['brier']:.3f}",
            ],
        },
        index=["Accuracy", "Log-loss", "Brier"],
    )
    st.table(compare)
    st.caption(
        "The market is sharper — closing lines are near the sharpest public "
        "signal in MMA, and the model lands close but doesn't beat them "
        f"(log-loss {head['model']['log_loss']:.3f} vs "
        f"{head['market']['log_loss']:.3f}). That's the honest, expected result."
    )

    def _bin_rates(rows):
        return {
            round((i + 0.5) / 10, 2): row["empirical_rate"]
            for i, row in enumerate(rows)
            if row["empirical_rate"] is not None
        }

    midpoints = [round((i + 0.5) / 10, 2) for i in range(10)]
    calibration = pd.DataFrame(
        {
            "Ideal": {m: m for m in midpoints},
            "Model": _bin_rates(head["calibration"]["model"]),
            "Market": _bin_rates(head["calibration"]["market"]),
        }
    ).sort_index()
    st.caption(
        "Calibration — predicted win probability (x) vs. actual win rate (y), "
        "by decile; closer to the *Ideal* diagonal is better:"
    )
    st.line_chart(calibration, height=240)
    st.caption(
        "Flat-stake backtest: betting the model's disagreements with the market "
        "loses money at every edge threshold "
        f"({head['roi']['0.00']['favorite_edge_on_a']['roi_pct']:.1f}% to "
        f"{head['roi']['0.10']['favorite_edge_on_a']['roi_pct']:.1f}% ROI on "
        "favorite edges) — the vig plus a sharp market leave no exploitable gap. "
        "[Details](https://github.com/dylanmryan/mma-prediction/blob/main/models/market_benchmark.json)."
    )

st.divider()
st.caption(model_card_text(predictor.weight, predictor.temperature))
