"""MMA matchup predictor -- Streamlit app over the committed ensemble."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mma.inference import Ensemble, build_matchup, predict_symmetrized
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

    # Documented deviation from the Phase 5 plan: the ensemble is not exactly
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
    result = predict_symmetrized(ensemble, matchup_ab, matchup_ba)
    p_a = result["winner_prob"]
    spread = result["winner_spread"]

    st.subheader("Prediction")
    st.progress(p_a, text=f"{name_a}: {p_a:.0%}  ·  {name_b}: {1 - p_a:.0%}")
    st.caption(f"5-seed ensemble; seeds range ±{spread / 2:.1%} around the mean.")

    # MC dropout runs on the (cheap) A-vs-B orientation only, then shifts the
    # samples by the same delta that symmetrization applied to the headline
    # probability, so the displayed distribution is centered on `p_a` above
    # rather than on the single-orientation `orientation_ab_prob`.
    samples = ensemble.mc_dropout(matchup_ab, passes=100)[:, 0]
    samples = np.clip(samples + result["mc_dropout_shift"], 0.0, 1.0)
    histogram = np.histogram(samples, bins=20, range=(0.0, 1.0))[0]
    st.bar_chart(
        pd.DataFrame({"MC dropout samples": histogram},
                     index=[f"{edge / 20:.2f}" for edge in range(20)]),
        height=160,
    )

    method = result["method_probs"]
    rounds = result["round_probs"]
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
    "temperature-calibrated. Validation (2021-2023): accuracy 0.606, log-loss 0.654, "
    "Brier 0.231 — vs XGBoost 0.595/0.658/0.233 and Elo 0.576/0.678/0.242. "
    "Test years (2024+) held out. Method and round probabilities assume "
    "independence from the winner, and predictions are symmetrized across "
    "both fighter orderings (see code comments). "
    "[Source](https://github.com/dylanmryan/mma-prediction)"
)
