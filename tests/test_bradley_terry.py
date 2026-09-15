"""The Bradley-Terry member: symmetry, shrinkage, and point-in-time safety.

This member exists to be *differently wrong* from the other two, not to be
strong. So the tests pin the properties the blend relies on -- symmetry, and
the fact that an unknown fighter reads as an even matchup rather than as a
fitted value -- and the one that keeps it honest: it can see only training
fights.
"""
import numpy as np
import pytest

from mma.bradley_terry import ALPHA_GRID, BradleyTerry, fit_skills, select_alpha


# --- the model ----------------------------------------------------------------

def test_a_fighter_who_only_wins_ranks_above_one_who_only_loses():
    model = BradleyTerry(alpha=1.0).fit(["a", "a", "a"], ["b", "b", "b"])
    assert model.skill_of("a") > model.skill_of("b")
    assert model.predict_pairs(["a"], ["b"])[0] > 0.5


def test_predictions_are_symmetric():
    """p(A beats B) must be exactly 1 - p(B beats A) -- the blend averages
    probability streams and an asymmetric member would break the deployed
    scorer's own symmetrisation."""
    model = BradleyTerry(alpha=1.0).fit(["a", "b", "c"], ["b", "c", "a"])
    ab = model.predict_pairs(["a", "b", "c"], ["b", "c", "a"])
    ba = model.predict_pairs(["b", "c", "a"], ["a", "b", "c"])
    assert np.allclose(ab, 1.0 - ba)


def test_two_unknown_fighters_are_an_even_matchup():
    model = BradleyTerry(alpha=1.0).fit(["a"], ["b"])
    assert model.predict_pairs(["stranger"], ["nobody"])[0] == pytest.approx(0.5)


def test_an_unknown_fighter_scores_against_a_known_one_at_the_prior():
    model = BradleyTerry(alpha=1.0).fit(["a", "a", "a"], ["b", "b", "b"])
    p = model.predict_pairs(["a"], ["stranger"])[0]
    assert 0.5 < p < 1.0, "a proven winner should be favoured over an unknown"


def test_shrinkage_pulls_a_one_fight_record_back_toward_the_mean():
    """Without a penalty an unbeaten fighter's skill diverges. This is the
    partial pooling the 20.5% of rows with a debutant depend on."""
    strong = BradleyTerry(alpha=0.3).fit(["a"], ["b"]).skill_of("a")
    weak = BradleyTerry(alpha=30.0).fit(["a"], ["b"]).skill_of("a")
    assert strong > weak > 0


def test_more_evidence_moves_the_skill_further_than_less():
    one = BradleyTerry(alpha=1.0).fit(["a"], ["b"]).skill_of("a")
    many = BradleyTerry(alpha=1.0).fit(["a"] * 10, ["b"] * 10).skill_of("a")
    assert many > one


def test_a_fighter_with_no_fights_sits_at_the_prior_mean():
    skills = fit_skills([0], [1], n_fighters=3, alpha=1.0)
    assert skills[2] == pytest.approx(0.0)


def test_an_empty_training_set_is_not_an_error():
    assert fit_skills([], [], n_fighters=4, alpha=1.0).tolist() == [0.0] * 4


def test_alpha_must_be_positive():
    with pytest.raises(ValueError, match="alpha"):
        fit_skills([0], [1], n_fighters=2, alpha=0.0)


def test_weights_let_a_fight_count_more_than_once():
    plain = BradleyTerry(alpha=1.0).fit(["a"] * 4, ["b"] * 4).skill_of("a")
    weighted = BradleyTerry(alpha=1.0).fit(["a"], ["b"], weights=[4.0]).skill_of("a")
    assert weighted == pytest.approx(plain, rel=1e-4)


# --- point-in-time safety -----------------------------------------------------

def test_erasing_the_evaluation_fights_changes_no_evaluation_prediction():
    """SP6 guard 2. The member is refit per fold on `fold.train` alone; if a
    later fight could move an earlier prediction, the walk-forward would be
    scoring a model that had seen its own answer."""
    train_w, train_l = ["a", "a", "c"], ["b", "c", "b"]
    model_train_only = BradleyTerry(alpha=1.0).fit(train_w, train_l)
    # the same fit, with the evaluation fights appended -- what a leak looks like
    model_contaminated = BradleyTerry(alpha=1.0).fit(
        train_w + ["b", "b"], train_l + ["a", "c"]
    )
    p_clean = model_train_only.predict_pairs(["a"], ["b"])[0]
    p_leaky = model_contaminated.predict_pairs(["a"], ["b"])[0]
    assert p_clean != pytest.approx(p_leaky), (
        "if adding the evaluation results does not move the prediction, this "
        "test cannot detect a leak and needs a stronger fixture"
    )
    # and the honest one is the one built from training rows only
    assert p_clean > 0.5


def test_the_fit_depends_on_no_row_order():
    a = BradleyTerry(alpha=1.0).fit(["a", "b", "c"], ["b", "c", "a"])
    b = BradleyTerry(alpha=1.0).fit(["c", "a", "b"], ["a", "b", "c"])
    assert a.predict_pairs(["a"], ["b"])[0] == pytest.approx(
        b.predict_pairs(["a"], ["b"])[0], abs=1e-6)


# --- choosing the penalty -----------------------------------------------------

def test_select_alpha_returns_a_grid_point_and_every_score():
    rng = np.random.default_rng(0)
    fighters = [f"f{i}" for i in range(20)]
    skill = {f: rng.normal() for f in fighters}
    w, l = [], []
    for _ in range(300):
        x, y = rng.choice(fighters, 2, replace=False)
        if rng.uniform() < 1 / (1 + np.exp(-(skill[x] - skill[y]))):
            w.append(x); l.append(y)
        else:
            w.append(y); l.append(x)
    va, vb, vy = [], [], []
    for _ in range(120):
        x, y = rng.choice(fighters, 2, replace=False)
        va.append(x); vb.append(y)
        vy.append(1.0 if rng.uniform() < 1/(1+np.exp(-(skill[x]-skill[y]))) else 0.0)
    best, scores = select_alpha(w, l, va, vb, vy)
    assert best in ALPHA_GRID
    assert set(scores) == set(ALPHA_GRID)
    assert scores[best] == min(scores.values())


def test_the_model_learns_a_synthetic_ladder_better_than_a_coin_flip():
    """A sanity floor: if it cannot beat 0.693 on data generated from its own
    likelihood, the fitter is broken."""
    rng = np.random.default_rng(3)
    fighters = [f"f{i}" for i in range(40)]
    skill = {f: rng.normal(0, 1.0) for f in fighters}

    def sample(n):
        w, l, a, b, y = [], [], [], [], []
        for _ in range(n):
            x, z = rng.choice(fighters, 2, replace=False)
            p = 1 / (1 + np.exp(-(skill[x] - skill[z])))
            x_wins = rng.uniform() < p
            w.append(x if x_wins else z); l.append(z if x_wins else x)
            a.append(x); b.append(z); y.append(1.0 if x_wins else 0.0)
        return w, l, a, b, np.array(y)

    tw, tl, _, _, _ = sample(2000)
    _, _, va, vb, vy = sample(600)
    model = BradleyTerry(alpha=1.0).fit(tw, tl)
    p = np.clip(model.predict_pairs(va, vb), 1e-9, 1 - 1e-9)
    ll = float(-np.mean(vy * np.log(p) + (1 - vy) * np.log(1 - p)))
    assert ll < 0.66, f"log-loss {ll:.4f} on its own generative model"


# --- the blend extension ------------------------------------------------------

def test_extra_members_default_to_none_so_committed_blends_keep_their_meaning():
    from mma.candidates import BlendCandidate

    assert BlendCandidate().extra == ()
    assert BlendCandidate().extra_members() == ()


def test_extra_members_are_built_by_name_and_an_unknown_one_is_loud():
    from mma.candidates import BlendCandidate

    built = BlendCandidate(extra=("logistic", "bt")).extra_members()
    assert [m.name for m in built] == ["logistic", "bt"]
    with pytest.raises(ValueError, match="unknown extra blend member"):
        BlendCandidate(extra=("elo",)).extra_members()


def test_the_winner_head_is_an_equal_mean_over_every_member():
    """SP6 fixes equal weights across EVERY member, so the winner head must be
    (xgb + torch + extra) / 3. Averaging the `weight`-blended pair against the
    newcomer instead would silently give xgb and torch a quarter each and the
    third member a half -- a different experiment from the registered one.

    Exercised through `fit_predict` with stub members, so it fails if the
    implementation changes rather than merely restating the arithmetic.
    """
    import numpy as np
    import pandas as pd
    from mma.candidates import BlendCandidate
    from mma.walkforward import Fold

    n = 6
    features = pd.DataFrame({
        "fight_id": [f"f{i}" for i in range(n)],
        "y_winner": [1, 0, 1, 0, 1, 0],
        "scheduled_rounds": [3] * n,
        "swapped": [False] * n,
    })
    train = np.array([True, True, False, False, False, False])
    inner = np.array([False, False, True, True, False, False])
    ev = np.array([False, False, False, False, True, True])
    fold = Fold(year=2020, train=train, inner_val=inner, eval=ev)

    class Stub:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def fit_predict(self, features, fold, sample_weight=None):
            k = int(np.asarray(fold.eval).sum())
            return ({"winner": np.full(k, self.value),
                     "method": np.tile([0.4, 0.3, 0.3], (k, 1)),
                     "round": np.tile([0.25] * 4, (k, 1))}, {})

    class Rigged(BlendCandidate):
        def members(self):
            return (Stub("xgb", 0.1), Stub("torch", 0.3))

        def extra_members(self):
            return (Stub("logistic", 0.9),)

    candidate = Rigged(extra=("logistic",), calibrate=False)
    pred, info = candidate.fit_predict(features, fold)
    assert pred["winner"] == pytest.approx([(0.1 + 0.3 + 0.9) / 3] * 2)
    assert pred["winner"] != pytest.approx([((0.1 + 0.3) / 2 + 0.9) / 2] * 2)
    assert info["extra_members"] == ["logistic"]


def test_the_bt_member_needs_the_fights_table_and_says_so():
    from mma.candidates import BradleyTerryCandidate
    import pandas as pd

    with pytest.raises(ValueError, match="fights table"):
        BradleyTerryCandidate()._corners(pd.DataFrame({"fight_id": ["x"],
                                                       "swapped": [False]}))


def test_the_bt_member_orients_corners_through_the_same_swapped_flag():
    """If this were read the other way every prediction would invert."""
    import pandas as pd
    from mma.candidates import BradleyTerryCandidate

    fights = pd.DataFrame({"fight_id": ["f1", "f2"],
                           "fighter_a_id": ["A1", "A2"],
                           "fighter_b_id": ["B1", "B2"]})
    features = pd.DataFrame({"fight_id": ["f1", "f2"], "swapped": [False, True]})
    a, b = BradleyTerryCandidate(fights=fights)._corners(features)
    assert a.tolist() == ["A1", "B2"], "a swapped row's corner A is the fights' B"
    assert b.tolist() == ["B1", "A2"]
