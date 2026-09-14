"""Judges' scorecards, parsed frame-invariantly.

The one hazard in this table is orientation. `data/raw/fight.csv`'s `details`
lists two scores per judge and the order is NOT keyed to `r_id`/`b_id` --
tested against the recorded winner it agrees on 37% of unanimous decisions,
which is neither "right" nor "flipped" but "inconsistent". This repo has
already mis-paired out-of-fold predictions by position twice, so the module
does not try to resolve that ordering at all.

Instead every quantity it exposes is **frame-invariant**: flipping which score
comes first negates every judge's margin, so `|mean margin|` and the card
shape are unchanged. The SIGN comes from `y_winner`, which the feature table
already owns. There is therefore no orientation to get wrong.
"""
import numpy as np
import pandas as pd
import pytest

from mma.scorecards import (
    abs_mean_margin,
    build_scorecards,
    card_shape,
    parse_cards,
    subtype_of,
)

UNANIMOUS = "Andy Roberts 30 - 27. Doug Crosby 30 - 27. Chris Lee 29 - 28."
SPLIT = "Sal D'amato 29 - 28. Chris Lee 28 - 29. Derek Cleary 29 - 28."
MAJORITY = "Mike Bell 29 - 28. Derek Cleary 29 - 29. Rick Winter 29 - 28."


# --- parsing -----------------------------------------------------------------

def test_parses_three_judges_with_names_and_scores():
    assert parse_cards(UNANIMOUS) == (
        ("Andy Roberts", 30, 27), ("Doug Crosby", 30, 27), ("Chris Lee", 29, 28),
    )


def test_parses_names_with_apostrophes_and_accents():
    assert parse_cards("Sal D'amato 29 - 28.")[0][0] == "Sal D'amato"
    assert parse_cards("Junichiro Kamijo 30 - 27.")[0][0] == "Junichiro Kamijo"


def test_a_finish_carries_no_cards():
    assert parse_cards("Armbar From Bottom Guard") == ()
    assert parse_cards("Punches to Head From Mount") == ()
    assert parse_cards(None) == ()
    assert parse_cards(float("nan")) == ()


# --- frame invariance, which is the whole design -----------------------------

def _flip(cards):
    """The same card read with the two columns swapped."""
    return tuple((n, b, a) for n, a, b in cards)


@pytest.mark.parametrize("text", [UNANIMOUS, SPLIT, MAJORITY])
def test_abs_mean_margin_is_unchanged_by_reading_the_columns_the_other_way(text):
    cards = parse_cards(text)
    assert abs_mean_margin(cards) == pytest.approx(abs_mean_margin(_flip(cards)))


@pytest.mark.parametrize("text", [UNANIMOUS, SPLIT, MAJORITY])
def test_card_shape_is_unchanged_by_reading_the_columns_the_other_way(text):
    cards = parse_cards(text)
    assert card_shape(cards) == card_shape(_flip(cards))


def test_a_judge_needs_a_real_name_not_an_initial():
    """The name pattern deliberately will not match a single letter: `details`
    is free text and a looser pattern invites false positives out of it."""
    assert parse_cards("A 30 - 27. B 30 - 27. C 30 - 27.") == ()
    assert len(parse_cards("Al Wong 30 - 27.")) == 1


def test_the_magnitude_separates_a_sweep_from_a_squeaker():
    """The entire point: these two are the same training example today."""
    sweep = abs_mean_margin(parse_cards(
        "Andy Roberts 30 - 27. Doug Crosby 30 - 27. Chris Lee 30 - 27."))
    squeaker = abs_mean_margin(parse_cards(SPLIT))
    assert sweep == pytest.approx(3.0)
    assert squeaker == pytest.approx(1 / 3)
    assert sweep > 8 * squeaker


# --- shape agrees with what the promotion recorded ---------------------------

def test_shape_reads_unanimous_split_and_majority():
    assert card_shape(parse_cards(UNANIMOUS)) == (3, 0, 0)
    assert card_shape(parse_cards(SPLIT)) == (2, 1, 0)
    assert card_shape(parse_cards(MAJORITY)) == (2, 0, 1)


def test_subtype_is_read_from_the_method_string():
    assert subtype_of("Decision - Unanimous") == "unanimous"
    assert subtype_of("Decision - Split") == "split"
    assert subtype_of("Decision - Majority") == "majority"
    assert subtype_of("KO/TKO") is None


# --- the table ---------------------------------------------------------------

def _raw(rows):
    return pd.DataFrame(rows, columns=["fight_id", "method", "details"])


def test_build_keeps_only_fights_whose_cards_agree_with_the_recorded_subtype():
    """A parse that contradicts the promotion's own label is dropped, not
    trusted: 0.2% of decisions do this and guessing which side is wrong would
    be inventing data."""
    raw = _raw([
        ("ok", "Decision - Unanimous", UNANIMOUS),
        ("bad", "Decision - Unanimous", SPLIT),      # says unanimous, cards split
        ("split_ok", "Decision - Split", SPLIT),
    ])
    out = build_scorecards(raw)
    assert set(out["fight_id"]) == {"ok", "split_ok"}
    assert out.set_index("fight_id").loc["ok", "n_judges"] == 3


def test_build_skips_finishes_and_unparseable_rows_without_raising():
    raw = _raw([
        ("ko", "KO/TKO", "Punches to Head From Mount"),
        ("missing", "Decision - Unanimous", None),
        ("partial", "Decision - Unanimous", "Andy Roberts 30 - 27."),
        ("ok", "Decision - Unanimous", UNANIMOUS),
    ])
    out = build_scorecards(raw)
    assert out["fight_id"].tolist() == ["ok"]


def test_build_records_whether_the_judges_disagreed():
    raw = _raw([("u", "Decision - Unanimous", UNANIMOUS),
                ("s", "Decision - Split", SPLIT)])
    out = build_scorecards(raw).set_index("fight_id")
    assert bool(out.loc["u", "judges_disagreed"]) is False
    assert bool(out.loc["s", "judges_disagreed"]) is True


def test_build_carries_no_column_that_could_name_a_corner():
    """If nothing in the table names a side, nothing downstream can orient it
    wrongly. The sign is `y_winner`'s job and only `y_winner`'s job."""
    raw = _raw([("u", "Decision - Unanimous", UNANIMOUS)])
    out = build_scorecards(raw)
    forbidden = {"r_id", "b_id", "winner_id", "score_a", "score_b",
                 "fighter_a_id", "fighter_b_id", "mean_margin"}
    assert forbidden.isdisjoint(out.columns), (
        f"frame-dependent column leaked into the table: "
        f"{forbidden & set(out.columns)}"
    )


def test_the_magnitude_is_never_negative():
    raw = _raw([("u", "Decision - Unanimous", UNANIMOUS),
                ("s", "Decision - Split", SPLIT)])
    assert (build_scorecards(raw)["abs_mean_margin"] >= 0).all()


# --- against the real table --------------------------------------------------

def test_the_real_scrape_parses_at_the_rate_the_preregistration_claims():
    """SP5's pre-registration fixed 4,461 of 4,995 decisions as parseable
    before anything was built. If the scrape grows this may rise, but a
    COLLAPSE means the parser broke against a format change."""
    raw = pd.read_csv("data/raw/fight.csv")
    decisions = raw[raw["method"].astype(str).str.contains("ecision", na=False)]
    out = build_scorecards(raw)
    assert len(decisions) >= 4995
    assert len(out) >= 4400, (
        f"only {len(out)} of {len(decisions)} decisions parsed; the "
        "pre-registration measured 4,461 -- check for a format change"
    )


def test_split_decisions_really_are_closer_than_unanimous_ones():
    """The mechanism, on the real data: if this ever stopped holding the
    label would be carrying no information worth adding a head for."""
    raw = pd.read_csv("data/raw/fight.csv")
    out = build_scorecards(raw)
    split = out.loc[out["judges_disagreed"], "abs_mean_margin"].mean()
    unanimous = out.loc[~out["judges_disagreed"], "abs_mean_margin"].mean()
    assert unanimous > 3 * split, f"unanimous {unanimous:.3f} vs split {split:.3f}"


# --- the label must never become a feature (SP5 pre-registration, guard 2) ---

def test_y_margin_is_excluded_from_both_members_feature_matrices():
    """`y_margin`'s SIGN IS `y_winner`. A model given it as an input would
    read the answer off its own input row, so both members' exclusion sets
    have to carry it -- and they key off their own TARGETS tuples, which is
    where it was missing when this was first wired: four guard tests caught
    it, and this is the direct assertion they were standing in for.
    """
    from mma.models.xgb import NON_FEATURES, TARGETS as XGB_TARGETS
    from mma.tensors import TARGETS as TORCH_TARGETS

    assert "y_margin" in XGB_TARGETS
    assert "y_margin" in TORCH_TARGETS
    assert "y_margin" in NON_FEATURES


def test_y_margin_is_absent_from_the_built_feature_matrices():
    """The exclusion, exercised rather than asserted about."""
    import numpy as np
    import pandas as pd
    from mma.models.xgb import feature_frame
    from mma.tensors import Preprocessor

    features = pd.read_parquet("data/processed/features.parquet")
    if "y_margin" not in features.columns:
        pytest.skip("feature table predates SP5")
    assert "y_margin" not in feature_frame(features).columns
    prep = Preprocessor.fit(features,
                            train_mask=np.ones(len(features), dtype=bool))
    assert "y_margin" not in prep.numeric_columns


def test_y_margin_is_in_no_registered_feature_block():
    from mma.feature_blocks import BLOCKS, columns_of

    for name, block in BLOCKS.items():
        assert "y_margin" not in columns_of(block), f"block {name!r} claims it"


# --- the xgb arm's soft label ------------------------------------------------

def test_the_soft_label_never_crosses_against_the_winner():
    """`y_margin`'s sign IS `y_winner`, so softening can say a win was narrow
    but can never claim the loser won."""
    from mma.scorecards import soft_winner_label

    y = np.array([1.0, 1.0, 0.0, 0.0])
    margin = np.array([3.0, 1 / 3, -1 / 3, -3.0])
    for temperature in (0.5, 2.0, 4.0, 20.0):
        soft = soft_winner_label(y, margin, temperature)
        assert ((soft > 0.5) == (y == 1)).all(), temperature


def test_a_fight_with_no_card_keeps_its_hard_label():
    """Finishes are not close fights we failed to measure; they are fights
    that ended. Softening them would be imputing the wrong thing."""
    from mma.scorecards import soft_winner_label

    soft = soft_winner_label([1.0, 0.0], [float("nan")] * 2, 2.0)
    assert soft.tolist() == [1.0, 0.0]


def test_a_narrow_win_softens_further_than_a_sweep():
    from mma.scorecards import soft_winner_label

    sweep, narrow = soft_winner_label([1.0, 1.0], [3.0, 1 / 3], 2.0)
    assert 0.5 < narrow < sweep < 1.0


def test_soft_temperature_must_be_positive():
    from mma.scorecards import soft_winner_label

    with pytest.raises(ValueError, match="positive"):
        soft_winner_label([1.0], [1.0], 0.0)


def test_the_expansion_reproduces_the_soft_loss_exactly():
    """The expansion is the whole reason the xgb arm is runnable: log-loss is
    linear in the weights, so p*loss(1) + (1-p)*loss(0) IS the soft-label
    loss. If that ever stopped holding, X1/X2 would be measuring something
    other than what they claim to."""
    from mma.scorecards import expand_soft_labels

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 50)
    pred = rng.uniform(0.05, 0.95, 50)
    x = pd.DataFrame({"pred": pred})

    def logloss(y, q):
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))

    direct = logloss(p, pred).mean()
    xs, ys, ws = expand_soft_labels(x, p)
    expanded = (ws * logloss(ys, xs["pred"].to_numpy())).sum() / ws.sum()
    assert expanded == pytest.approx(direct)


def test_the_expansion_conserves_total_weight():
    from mma.scorecards import expand_soft_labels

    x = pd.DataFrame({"f": range(6)})
    p = np.array([1.0, 0.0, 0.7, 0.3, 0.55, 0.9])
    _, _, ws = expand_soft_labels(x, p, sample_weight=np.full(6, 2.0))
    assert ws.sum() == pytest.approx(12.0)


def test_the_expansion_leaves_hard_rows_unduplicated():
    from mma.scorecards import expand_soft_labels

    x = pd.DataFrame({"f": [1, 2, 3]})
    xs, ys, ws = expand_soft_labels(x, np.array([1.0, 0.0, 0.6]))
    assert len(xs) == 4, "two hard rows stay single, the soft one becomes two"
    assert sorted(ys.tolist()) == [0, 0, 1, 1]
