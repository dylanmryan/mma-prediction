"""MMA matchup predictor -- Streamlit app over the committed hybrid scorer.

Since SP3 the deployed scorer is `mma.inference.SimulatorPredictor`: the blend
supplies P(A wins) and a Monte Carlo fight simulator supplies
P(method, round | winner). The app's payoff is the outcome table below --
every way the fight can end, with one probability each, read straight off the
one joint distribution the model produces. They sum to 1 because they are one
distribution, not three heads multiplied together.
"""
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
    SimulatorPredictor,
    build_matchup,
    predict_symmetrized,
)
from mma.joint import cells_to_dict
from mma.snapshots import build_snapshots

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
TORCH_METRICS = ROOT / "models" / "torch" / "metrics_val.json"
XGB_METRICS = ROOT / "models" / "xgb_metrics_val.json"
ELO_WALKFORWARD = ROOT / "models" / "walkforward" / "elo.json"
# The DEPLOYED blend's own pooled metrics. `BLEND_REPORT` scores the HARNESS
# form, which fits one temperature per fold on that fold's inner-validation
# year; this scorer applies one fixed temperature and cannot fit per fold, so
# the card's numbers come from here instead
# (`scripts/derive_blend_temperature.py`, which re-scores the committed
# per-row dump under the deployed rule).
BLEND_TEMPERATURE = ROOT / "models" / "walkforward" / "blend_temperature.json"


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

    The deployed scorer is the SP3 hybrid, but its WIN PROBABILITY is the
    blend's, unchanged -- so the accuracy/log-loss/Brier/ECE quoted here are
    still the blend's numbers and still describe what serves. The simulator's
    own evidence is the joint-outcome log-loss, which is quoted in the tail
    rather than mixed in with winner metrics it is not comparable to.

    `weight`/`temperature` are the deployed scorer's own committed
    configuration (`models/blend.json`, via `mma.inference.load_blend_config`
    -- the caller passes `predictor.weight`/`predictor.temperature` so the
    card names the numbers actually serving rather than a re-read that could
    disagree with them). The deployed scorer is a BLEND, so the card's
    headline numbers are the blend's -- and they are the DEPLOYED form's
    (`models/walkforward/blend_temperature.json`), not the harness report's.
    `blend_b1.json` scores a scorer that fits one temperature per fold on that
    fold's inner-validation year; this one applies a single fixed temperature
    to every prediction it makes and cannot fit per fold, so its calibration
    is a different number and this card is about this one. The harness form's
    ECE is quoted beside it, labelled, so the card cannot be read as
    contradicting the report. The two members' own pooled numbers are shown
    beside it, as the comparison that makes the case for blending them. Under the refit_through recipe the metrics files carry no
    held-out slice of their own; they carry the harness evidence behind each
    member's budget, and the training cutoff. Falls back to a numberless
    card if a file is missing or unreadable.
    """
    torch_m = _read_json(TORCH_METRICS)
    xgb_m = _read_json(XGB_METRICS)
    elo_m = _read_json(ELO_WALKFORWARD).get("pooled", {})
    blend = _read_json(BLEND_REPORT)
    deployed = _read_json(BLEND_TEMPERATURE).get("deployed", {}).get("honest_pooled", {})
    head = (
        f"Model card: a hybrid — the win probability is a blend, a "
        f"{weight:g}/{1 - weight:g} average of a 5-seed XGBoost ensemble and a "
        "5-seed multi-task net, temperature-calibrated after averaging "
        f"(T={temperature:g}); the outcome table is a Monte Carlo fight "
        "simulator (a 5-seed per-round hazard model and a 5-seed decision "
        "model, 10,000 simulated fights per matchup) conditioned on that win "
        "probability. ")
    tail = ("The prospective track record (predictions/track_record.json) is the "
            "only true holdout. The headline numbers below are the win "
            "probability's; the simulator was shipped on a joint-outcome "
            "log-loss of 2.1432 against the previous composition's 2.2028 "
            "(models/walkforward/sp3_decision.json), and it leaves the win "
            "probability itself unchanged. Predictions are symmetrized across "
            "both fighter orderings (see code comments). "
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

    # The deployed form first. The harness report is a fallback that carries no
    # calibration claim at all (see the ECE below), rather than one describing a
    # scorer other than this one.
    blend_line = _triple(deployed) or _triple(blend.get("pooled", {}))
    torch_line = _triple(torch_m.get("winner_ensemble", {}))
    if blend_line and torch_m.get("mode") == "refit_through":
        n = blend.get("pooled", {}).get("n")
        n_text = f"n={n:,} pooled fights" if isinstance(n, int) else "pooled"
        ece = deployed.get("ece", {}).get("10")
        harness_ece = blend.get("pooled", {}).get("ece")
        ece_text = ""
        if isinstance(ece, (int, float)):
            ece_text = f", ECE {ece:.4f}"
            if isinstance(harness_ece, (int, float)):
                ece_text += (" (this fixed-temperature form; the harness's "
                             f"per-fold-fitted form reads {harness_ece:.4f})")
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

METHOD_LABELS = {"ko_tko": "KO/TKO", "submission": "Submission"}
ROUND_LABELS = {"1": "round 1", "2": "round 2", "3": "round 3", "45": "rounds 4-5"}
#: A cell below this is not shown as its own row. Rounds the bout cannot reach
#: carry exactly zero in both corners, and a "0.0%" row for rounds 4-5 of a
#: three-round fight is noise rather than information.
OUTCOME_FLOOR = 0.0005


def outcome_rows(joint: dict, method_classes, round_classes) -> list[tuple]:
    """`(label, P(corner A wins that way), P(corner B wins that way))` per row.

    `joint` is `mma.joint.cells_to_dict` output -- the whole outcome
    distribution for one fight. Rows are in reading order: each finishing
    method by round, then the decision. Every cell of the distribution appears
    exactly once, except cells below `OUTCOME_FLOOR` in both corners, so the
    rows shown account for the fight.
    """
    rows = []
    for method_name in (m for m in method_classes if m != "decision"):
        for round_class in round_classes:
            values = [joint[corner][method_name][round_class] for corner in ("a", "b")]
            if max(values) > OUTCOME_FLOOR:
                label = f"{METHOD_LABELS[method_name]} in {ROUND_LABELS[round_class]}"
                rows.append((label, *values))
    rows.append(("Decision", joint["a"]["decision"], joint["b"]["decision"]))
    return rows


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
    # The deployed scorer: the blend (both members, plus the weight and
    # post-average temperature from models/blend.json -- see
    # predictor.weight/.temperature) wrapped in the simulator that turns its
    # winner probability into a joint distribution over outcomes.
    #
    # Nothing recalibrates the displayed method/round splits any more. They
    # used to be multiplied by mean-matching factors because the class-weighted
    # heads overstated rare classes by up to 2.5x; the simulator's marginals
    # land within a few points of the base rates on their own
    # (models/display_calibration.json records the measurement), and
    # correcting two marginals separately would pull them off the joint they
    # are read from -- reintroducing exactly the contradiction SP3 removed.
    predictor = SimulatorPredictor.load()
    as_of = fights["date"].max()
    weight_classes = sorted(fights["weight_class"].dropna().unique().tolist())
    return (
        fights, fighters.set_index("fighter_id"), ratings, snapshots, predictor,
        as_of, weight_classes,
    )


fights, fighters, ratings, snapshots, predictor, as_of, weight_classes = load_everything()

eligible = snapshots.join(fighters[["name"]], how="inner").sort_values("name")
names = eligible["name"].tolist()
by_name = {name: fighter_id for fighter_id, name in eligible["name"].items()}

st.title("🥊 MMA Fight Predictor")
st.caption(
    f"Elo → XGBoost → neural ensemble → calibrated blend → fight simulator, "
    f"honestly evaluated. "
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

    # The payoff of the whole simulator sub-project: every way this fight can
    # end, with one probability each, read straight off the joint distribution
    # the model produced. These are not three heads multiplied together -- the
    # simulator played the fight out round by round 10,000 times and this is
    # where those fights ended, so the cells sum to 1 and cannot assert
    # something no fight could do (a round-3 finish in a two-round bout, a
    # knockout and a decision at once).
    joint = cells_to_dict(
        result["joint_cells"], result["method_classes"], result["round_classes"])
    outcomes = outcome_rows(joint, result["method_classes"], result["round_classes"])

    st.subheader("How it ends")
    distance = result["p_distance"]
    best = max(outcomes, key=lambda row: max(row[1], row[2]))
    best_corner = name_a if best[1] >= best[2] else name_b
    metrics = st.columns(2)
    metrics[0].metric("Goes the distance", f"{distance:.0%}")
    metrics[1].metric("Ends inside the distance", f"{1 - distance:.0%}")
    st.caption(f"Most likely single outcome: **{best_corner} by {best[0].lower()}** "
               f"at {max(best[1], best[2]):.0%}.")
    st.dataframe(
        pd.DataFrame(
            {name_a: [f"{row[1]:.1%}" for row in outcomes],
             name_b: [f"{row[2]:.1%}" for row in outcomes]},
            index=[row[0] for row in outcomes],
        ),
    )
    st.caption(
        f"Each cell is the probability that fighter wins that exact way, and "
        "together they account for the whole fight. They come from one simulated "
        "process rather than three separate heads, so each column adds up to that "
        "fighter's win probability above, exactly. Tendencies, not betting odds."
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
