"""Ridge-penalised Bradley-Terry latent skill, fitted by MAP.

SP6's structurally different blend member
(`docs/superpowers/plans/2026-09-15-sp6-third-blend-member.md`). Every other
scorer in this project is feature-discriminative: it reads 87 columns about
two fighters and maps them to a probability. This one reads **only who fought
whom and who won**, gives every fighter one latent skill, and predicts

    P(a beats b) = sigmoid(s_a - s_b)

That is the whole model. It is a deliberately weak learner; the blend does not
need a third strong member, it needs a differently-wrong one, and a member
that never sees the feature matrix is the least correlated thing available.

**Why ridge, and why toward zero rather than a per-division mean.** The
pre-registration said "partial pooling toward a per-division mean". That turns
out not to be identified: Bradley-Terry skills are fixed only up to an additive
constant *within each connected component* of the comparison graph, and
divisions are very nearly disconnected -- fighters rarely cross weight classes
-- so each division's mean is arbitrary and a free parameter for it would
wander. Shrinking every skill toward a single zero is the identified version of
the same idea, and it costs nothing in practice because every matchup this
model is ever asked about is within a division, where the constant cancels.
The deviation is recorded in the plan's completion notes.

The penalty is what makes thin records behave. A fighter with one win has a
likelihood that would push their skill to infinity; `alpha` pulls them back
toward the population mean in proportion to how little is known about them,
which is the partial pooling the plan asked for. 20.5% of the table has a
debutant in one corner, so this is most of what the model does.

**Point-in-time by construction.** `fit` is given training fights only. A
fighter with no training fight gets skill 0 -- the prior mean, not an estimate
-- so a debut can never be scored on evidence from the fight being predicted.
`tests/test_bradley_terry.py` pins that erasing the evaluation rows changes no
evaluation prediction.
"""
from __future__ import annotations

import numpy as np

#: Ridge strengths the fold's inner-validation year picks between. Fitting one
#: scalar on held-out rows is fitting, not searching: no evaluation row is
#: involved, and the grid is fixed here rather than per run.
ALPHA_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)


def _negative_log_posterior(skills, idx_win, idx_lose, alpha, weights):
    """NLL of the observed winners plus the ridge penalty, and its gradient."""
    margin = skills[idx_win] - skills[idx_lose]
    # log(1 + exp(-margin)), stable
    loss = float(np.sum(weights * np.logaddexp(0.0, -margin)))
    loss += alpha * float(skills @ skills)

    # d/dmargin of log(1+exp(-m)) is -sigmoid(-m)
    g = -weights / (1.0 + np.exp(margin))
    grad = np.zeros_like(skills)
    np.add.at(grad, idx_win, g)
    np.add.at(grad, idx_lose, -g)
    grad += 2.0 * alpha * skills
    return loss, grad


def fit_skills(idx_win, idx_lose, n_fighters: int, alpha: float,
               weights=None) -> np.ndarray:
    """MAP latent skills for `n_fighters`, from won/lost index pairs.

    `idx_win[k]` beat `idx_lose[k]`. Fighters with no fight keep skill 0,
    which the penalty makes the exact optimum for them -- absence of evidence
    reads as the population mean rather than as a fitted value.
    """
    from scipy.optimize import minimize

    idx_win = np.asarray(idx_win, dtype=np.intp)
    idx_lose = np.asarray(idx_lose, dtype=np.intp)
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if len(idx_win) != len(idx_lose):
        raise ValueError("idx_win and idx_lose must be the same length")
    w = (np.ones(len(idx_win), dtype=float) if weights is None
         else np.asarray(weights, dtype=float))
    if not len(idx_win):
        return np.zeros(n_fighters, dtype=float)

    result = minimize(
        _negative_log_posterior, np.zeros(n_fighters, dtype=float),
        args=(idx_win, idx_lose, alpha, w), jac=True, method="L-BFGS-B",
        options={"maxiter": 500, "maxfun": 500},
    )
    return np.asarray(result.x, dtype=float)


def predict(skills: np.ndarray, idx_a, idx_b) -> np.ndarray:
    """P(a beats b). Symmetric by construction: swapping the corners returns
    exactly 1 - p, because the model depends on the skills only through their
    difference."""
    idx_a = np.asarray(idx_a, dtype=np.intp)
    idx_b = np.asarray(idx_b, dtype=np.intp)
    return 1.0 / (1.0 + np.exp(-(skills[idx_a] - skills[idx_b])))


class BradleyTerry:
    """Fit on a set of decided fights; score any pair of the same fighters.

    `fighter_index` maps every id the fitter has ever been told about to a row
    of `skills`. An id absent from it scores 0 against 0 -- an even matchup --
    which is the correct statement about two fighters this model knows nothing
    about.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.fighter_index: dict = {}
        self.skills = np.zeros(0, dtype=float)

    def fit(self, winner_ids, loser_ids, weights=None) -> "BradleyTerry":
        ids = list(dict.fromkeys(list(winner_ids) + list(loser_ids)))
        self.fighter_index = {fid: i for i, fid in enumerate(ids)}
        idx_win = [self.fighter_index[f] for f in winner_ids]
        idx_lose = [self.fighter_index[f] for f in loser_ids]
        self.skills = fit_skills(idx_win, idx_lose, len(ids), self.alpha,
                                 weights=weights)
        return self

    def skill_of(self, fighter_id) -> float:
        i = self.fighter_index.get(fighter_id)
        return 0.0 if i is None else float(self.skills[i])

    def predict_pairs(self, a_ids, b_ids) -> np.ndarray:
        sa = np.array([self.skill_of(f) for f in a_ids], dtype=float)
        sb = np.array([self.skill_of(f) for f in b_ids], dtype=float)
        return 1.0 / (1.0 + np.exp(-(sa - sb)))


def select_alpha(winner_ids, loser_ids, val_a, val_b, val_y,
                 grid=ALPHA_GRID) -> tuple[float, dict]:
    """The ridge strength with the best log-loss on held-out rows.

    Fitted on the fold's inner-validation year, which no evaluation row is in.
    Returns the chosen alpha and every grid point's score, so a run can show
    the curve was not flat-with-a-lucky-winner.
    """
    scores = {}
    for alpha in grid:
        model = BradleyTerry(alpha=alpha).fit(winner_ids, loser_ids)
        p = np.clip(model.predict_pairs(val_a, val_b), 1e-9, 1 - 1e-9)
        y = np.asarray(val_y, dtype=float)
        scores[alpha] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    best = min(scores, key=scores.get)
    return best, scores
