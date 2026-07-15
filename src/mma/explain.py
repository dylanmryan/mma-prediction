"""Per-fight "why this prediction?" explanations from the companion XGBoost model.

The headline probability shown in the app comes from the 5-seed torch
ensemble (mma.inference.Ensemble). That ensemble has no native per-prediction
attribution. The committed XGBoost winner model (models/xgb_winner.json,
val log-loss 0.6578 vs the ensemble's 0.6537 -- close) does: XGBoost's
TreeSHAP implementation (`Booster.predict(..., pred_contribs=True)`) gives
exact, per-feature contributions to the model's logit for a single
prediction, with no sampling or approximation and no new dependencies
(xgboost is already a training-time dependency; this module just uses it
at inference time too).

Contributions are computed for both fighter orderings (A-vs-B and B-vs-A)
and averaged after flipping the sign of the B-vs-A orientation, mirroring
`mma.inference.predict_symmetrized`'s handling of the torch ensemble's
order-sensitivity.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from mma.models.xgb import feature_frame

ROOT = Path(__file__).resolve().parents[2]
XGB_WINNER_PATH = ROOT / "models" / "xgb_winner.json"


def load_booster(path: Path = XGB_WINNER_PATH) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def raw_contributions(booster: xgb.Booster, matchup: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """TreeSHAP contributions for one fighter ordering.

    Returns `(per_feature_contribs, bias, raw_logit)`. TreeSHAP's additivity
    property guarantees `bias + per_feature_contribs.sum() == raw_logit`
    (see test_explain.py::test_additivity_holds_both_orientations).
    """
    x = feature_frame(matchup)
    dmatrix = xgb.DMatrix(x, enable_categorical=True)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    values, bias = contribs[:-1], float(contribs[-1])
    logit = float(booster.predict(dmatrix, output_margin=True)[0])
    return pd.Series(values, index=list(x.columns)), bias, logit


def contributions(
    matchup_ab: pd.DataFrame, matchup_ba: pd.DataFrame, booster: xgb.Booster | None = None
) -> pd.Series:
    """Symmetrized per-feature log-odds contributions toward fighter A winning.

    Runs native TreeSHAP for both orderings, flips the sign of the B-vs-A
    orientation (so both are expressed "toward A winning"), and averages.
    Positive values push the prediction toward fighter A; negative toward
    fighter B. The bias term is excluded. Sorted by |value| descending.
    """
    booster = booster or load_booster()
    values_ab, _, _ = raw_contributions(booster, matchup_ab)
    values_ba, _, _ = raw_contributions(booster, matchup_ba)
    averaged = 0.5 * (values_ab - values_ba)
    order = averaged.abs().sort_values(ascending=False).index
    return averaged.reindex(order)


# Plain-English labels for all 40 features in the committed xgb_winner model
# (see models/xgb_winner.json's booster.feature_names). "_a"/"_b" suffixed
# features describe a single fighter's absolute value (not a differential);
# everything else is fighter-A-minus-fighter-B.
FEATURE_LABELS: dict[str, str] = {
    "weight_class": "Weight class context",
    "title_fight": "Title-fight stakes",
    "scheduled_rounds": "Scheduled fight length",
    "career_fights_diff": "Experience edge (career fights)",
    "career_wins_diff": "Career wins edge",
    "career_win_rate_diff": "Career win rate edge",
    "career_finish_rate_diff": "Finishing rate edge",
    "kd_pf_diff": "Knockdown power (per fight)",
    "sub_att_pf_diff": "Submission attempts (per fight)",
    "td_landed_pf_diff": "Takedowns landed (per fight)",
    "td_acc_diff": "Takedown accuracy edge",
    "td_def_diff": "Takedown defense edge",
    "sig_pm_diff": "Striking output (sig. strikes/min)",
    "sig_absorbed_pm_diff": "Striking defense (absorbed/min)",
    "ctrl_share_diff": "Grappling control-time share",
    "streak_diff": "Recent win/loss streak",
    "days_since_last_diff": "Layoff/recency",
    "last5_win_rate_diff": "Recent form (last 5 fights)",
    "last5_avg_opp_elo_diff": "Recent strength of schedule",
    "elo_diff": "Elo rating edge",
    "striking_elo_diff": "Striking Elo edge",
    "grappling_elo_diff": "Grappling Elo edge",
    "elo_fights_diff": "Elo-tracked experience edge",
    "height_diff": "Height advantage",
    "reach_diff": "Reach advantage",
    "age_diff": "Age gap",
    "age_a": "Fighter A's age",
    "career_fights_a": "Fighter A's career fight count",
    "reach_missing_a": "Fighter A's reach is unlisted",
    "dob_missing_a": "Fighter A's birthdate is unlisted",
    "southpaw_a": "Fighter A's stance (southpaw)",
    "debut_a": "Fighter A is a UFC debutant",
    "age_b": "Fighter B's age",
    "career_fights_b": "Fighter B's career fight count",
    "reach_missing_b": "Fighter B's reach is unlisted",
    "dob_missing_b": "Fighter B's birthdate is unlisted",
    "southpaw_b": "Fighter B's stance (southpaw)",
    "debut_b": "Fighter B is a UFC debutant",
    "debut_matchup": "Debut-fight dynamics",
    "stance_mismatch": "Stance mismatch (orthodox vs. southpaw)",
}


def _strength(magnitude: float) -> str:
    if magnitude > 0.3:
        return "strong"
    if magnitude > 0.1:
        return "moderate"
    return "slight"


def humanize(contribs: pd.Series, name_a: str, name_b: str, top_n: int = 6) -> list[dict]:
    """Top `top_n` contributions as plain-English rows for display.

    Each row: {label, contribution (log-odds float, signed toward `name_a`),
    favors (fighter name), strength ("strong"/"moderate"/"slight")}.
    """
    rows = []
    for feature, value in contribs.head(top_n).items():
        value = float(value)
        rows.append(
            {
                "label": FEATURE_LABELS.get(feature, feature),
                "contribution": value,
                "favors": name_a if value >= 0 else name_b,
                "strength": _strength(abs(value)),
            }
        )
    return rows
