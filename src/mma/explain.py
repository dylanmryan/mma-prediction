"""Per-fight "why this prediction?" explanations, from inside the served model.

Until SP2.2 this was an explainer for a *companion* model: the headline
probability came from the torch ensemble, which has no native per-prediction
attribution, so the app showed TreeSHAP from a separate XGBoost winner model
and had to caveat that the two merely agreed closely. That gap is gone. The
deployed scorer is a blend (`mma.inference.BlendedPredictor`) and the XGBoost
winner seed ensemble IS half of it, so these contributions come from a model
that actually casts half the vote in the number displayed above them.

What they still are not is an attribution of the whole blend: the torch half's
contribution is not decomposed, and the post-average temperature rescales the
blended logit. Read them as "what the XGBoost half of the model saw", which is
a stronger claim than this module could make before and a weaker one than
"this is why the model said 0.63".

XGBoost's TreeSHAP implementation (`Booster.predict(..., pred_contribs=True)`)
gives exact, per-feature contributions to one model's logit for a single
prediction, with no sampling and no approximation. Contributions are averaged
over the five seed boosters -- the same five the blend averages -- and over
both fighter orderings (A-vs-B and B-vs-A, the latter sign-flipped), mirroring
`mma.inference.predict_symmetrized`'s handling of order-sensitivity.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from mma.models.xgb import align_to_booster, feature_frame

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
XGB_WINNER_GLOB = "xgb_winner_seed*.json"


def winner_paths(models_dir: Path = MODELS) -> list[Path]:
    """The deployed winner boosters, in seed order."""
    return sorted(
        Path(models_dir).glob(XGB_WINNER_GLOB),
        key=lambda path: int(path.stem.rsplit("seed", 1)[1]),
    )


def load_boosters(models_dir: Path = MODELS) -> list[xgb.Booster]:
    """Every seed of the deployed XGBoost winner head.

    All of them, not one: the blend averages five boosters, so explaining one
    of them would be explaining a model that does not exist. Raises rather than
    quietly explaining a subset.
    """
    paths = winner_paths(models_dir)
    if not paths:
        raise FileNotFoundError(
            f"no XGBoost winner boosters under {models_dir} (expected "
            f"{XGB_WINNER_GLOB}); run scripts/train_xgb.py"
        )
    boosters = []
    for path in paths:
        booster = xgb.Booster()
        booster.load_model(str(path))
        boosters.append(booster)
    return boosters


def raw_contributions(booster: xgb.Booster, matchup: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """TreeSHAP contributions for one fighter ordering.

    Returns `(per_feature_contribs, bias, raw_logit)`. TreeSHAP's additivity
    property guarantees `bias + per_feature_contribs.sum() == raw_logit`
    (see test_explain.py::test_additivity_holds_both_orientations).
    """
    # `align_to_booster` reshapes the served row to this booster's own trained
    # columns and re-categorises `weight_class` with its own category list --
    # the same alignment the served blend uses, so the explanation is computed
    # on the row the model scored.
    x = align_to_booster(feature_frame(matchup), booster)
    dmatrix = xgb.DMatrix(x, enable_categorical=True)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    values, bias = contribs[:-1], float(contribs[-1])
    logit = float(booster.predict(dmatrix, output_margin=True)[0])
    return pd.Series(values, index=list(x.columns)), bias, logit


def contributions(
    matchup_ab: pd.DataFrame, matchup_ba: pd.DataFrame, boosters=None
) -> pd.Series:
    """Symmetrized per-feature log-odds contributions toward fighter A winning.

    Runs native TreeSHAP for both orderings on every seed booster, flips the
    sign of the B-vs-A orientation (so both are expressed "toward A winning"),
    and averages over orientations and seeds. Averaging over seeds is what
    makes this an explanation of the deployed XGB member -- the mean of the
    five models -- rather than of one arbitrary member of it, and TreeSHAP's
    additivity survives the mean, so the averaged contributions still sum to
    the ensemble's mean logit minus its mean bias.

    Positive values push the prediction toward fighter A; negative toward
    fighter B. The bias term is excluded. Sorted by |value| descending.
    `boosters` accepts a list, or a single booster for convenience.
    """
    if boosters is None:
        boosters = load_boosters()
    elif isinstance(boosters, xgb.Booster):
        boosters = [boosters]
    per_booster = []
    for booster in boosters:
        values_ab, _, _ = raw_contributions(booster, matchup_ab)
        values_ba, _, _ = raw_contributions(booster, matchup_ba)
        per_booster.append(0.5 * (values_ab - values_ba))
    averaged = sum(per_booster) / len(per_booster)
    order = averaged.abs().sort_values(ascending=False).index
    return averaged.reindex(order)


# Plain-English labels for every feature in the committed winner boosters (see
# any models/xgb_winner_seed*.json's booster.feature_names). "_a"/"_b" suffixed
# features describe a single fighter's absolute value (not a differential);
# everything else is fighter-A-minus-fighter-B. A feature with no label falls
# back to its raw column name, and `test_explain.py` fails if any exists -- an
# unlabelled column in a "why this prediction?" panel is not an explanation.
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
    # SP2's `external` block: the record each fighter brought INTO the UFC,
    # from the jds-mma-data snapshot. NaN (and so absent from the explanation)
    # whenever either corner is unmapped.
    "pre_ufc_wins_diff": "Pre-UFC wins edge",
    "pre_ufc_losses_diff": "Pre-UFC losses edge",
    "pre_ufc_finish_rate_diff": "Pre-UFC finishing rate edge",
    "pre_ufc_finish_loss_rate_diff": "Pre-UFC rate of being finished",
    "pre_ufc_avg_opp_wins_diff": "Pre-UFC strength of schedule",
    "days_since_pro_debut_diff": "Time as a professional",
    # SP2.2's S1 table: the four blocks that ship as part of the blend.
    # `trajectory` -- Glicko state, Elo momentum, and age/experience curvature.
    "glicko_mu_diff": "Glicko rating edge",
    "glicko_phi_diff": "Rating uncertainty (Glicko deviation)",
    "glicko_sigma_diff": "Rating volatility (Glicko sigma)",
    "elo_delta_3_diff": "Elo momentum over the last 3 fights",
    "elo_delta_5_diff": "Elo momentum over the last 5 fights",
    "elo_peak_minus_current_diff": "Distance below career-peak Elo",
    "years_since_ufc_debut_diff": "Years in the UFC",
    "age_x_fights_diff": "Age against mileage (age x fights)",
    "age_squared_a": "Fighter A's age curve (age squared)",
    "age_squared_b": "Fighter B's age curve (age squared)",
    # `notice` -- short-notice bookings and missed weight.
    "notice_shortfall_days_diff": "Short-notice shortfall (days)",
    "missed_weight_over_lbs_diff": "Pounds over the weight limit",
    "short_notice_7_a": "Fighter A took the fight inside 7 days",
    "short_notice_30_a": "Fighter A took the fight inside 30 days",
    "missed_weight_a": "Fighter A missed weight",
    "short_notice_7_b": "Fighter B took the fight inside 7 days",
    "short_notice_30_b": "Fighter B took the fight inside 30 days",
    "missed_weight_b": "Fighter B missed weight",
    # `context` -- the referee and the fight's location.
    "bonus_rate_diff": "Performance-bonus rate edge",
    "referee_finish_rate": "Referee's historical finish rate",
    "referee_decision_rate": "Referee's historical decision rate",
    "referee_missing": "Referee unknown for this bout",
    "home_country_unknown": "Home advantage unknown (nationality unlisted)",
    # `opponent_adjusted` -- each rate priced against the opposition it came
    # against, and the quality of who a fighter has beaten and lost to.
    "sig_pm_vs_exp_diff": "Striking output vs. what opponents allow",
    "sig_absorbed_pm_vs_exp_diff": "Strikes absorbed vs. what opponents land",
    "td_landed_pf_vs_exp_diff": "Takedowns vs. what opponents concede",
    "td_def_vs_exp_diff": "Takedown defense vs. what opponents achieve",
    "ctrl_share_vs_exp_diff": "Control time vs. what opponents allow",
    "avg_opp_elo_wins_diff": "Quality of opponents beaten",
    "avg_opp_elo_losses_diff": "Quality of opponents lost to",
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
