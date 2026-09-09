"""The row served for a future fight must equal the row trained on.

`build_features` (training) and `build_matchup` (serving) must produce
identical feature values for the same matchup as of the same date. This is
value-level, not just column-level: it replays history to just before a
real historical fight, builds the served row from the resulting snapshots,
and compares it against that fight's row in the training table.
"""
import bisect
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mma import context, notice
from mma.feature_blocks import columns_for, spec_for
from mma.features import build_features
from mma.history import build_history
from mma.inference import build_matchup
from mma.snapshots import build_snapshots

from tests.conftest import table_blocks

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists(),
    reason="processed data not built (run scripts/make_dataset.py)",
)


@pytest.fixture(scope="module")
def tables():
    return {
        name: pd.read_parquet(PROCESSED / f"{name}.parquet")
        for name in ("fights", "fight_stats", "fighters", "ratings")
    }


def _card_facts(target: pd.Series, past: pd.DataFrame) -> dict:
    """The card-level facts a real caller supplies to `build_matchup`.

    A snapshot describes a fighter; these describe the BOUT, and without them
    the served row cannot match the trained one on the blocks that read them:
    `context` needs the referee (plus their rates over the fights known so
    far) and the event's country, `notice` needs each corner's camp. Passing
    them here is not cheating on the asymmetry -- serving a FUTURE card really
    does have the venue, and really does not have a referee or a Bet MMA row,
    which is why those paths default to "unknown" in `build_matchup` and are
    exercised by the blocks' own tests. What this test pins is the other
    half: given the same known facts, the two paths agree value for value.
    """
    return {
        "referee": target["referee"],
        "referee_rates": context.referee_rates(past),
        "event_country": context.event_country(target["location"]),
        "notice_a": notice.state_for(target["fight_id"], target["fighter_a_id"]),
        "notice_b": notice.state_for(target["fight_id"], target["fighter_b_id"]),
    }


def _prior_bouts(fights: pd.DataFrame) -> "callable":
    """(fighter_id, date) -> number of that fighter's bouts strictly before it."""
    ordered = fights.sort_values("date", kind="stable")
    dates: dict[str, list] = defaultdict(list)
    for fight in ordered.itertuples(index=False):
        dates[fight.fighter_a_id].append(fight.date)
        dates[fight.fighter_b_id].append(fight.date)
    return lambda fighter_id, date: bisect.bisect_left(dates[fighter_id], date)


def _fully_populated_fights() -> set:
    """Fights whose committed row has no NaN in any enabled block's columns.

    The value comparison skips a column that is NaN on BOTH sides, so a target
    fight that happens to be missing one block's data compares fewer columns
    than the spec declares -- and the `compared == expected` assertion then
    fails on a count instead of naming the column. Every block whose coverage
    is partial hits this: `external` populates `pre_ufc_finish_loss_rate_diff`
    on only 30% of rows, and `rankings` populates `rank_diff` only when both
    corners were ranked the week before. Requiring the target to be complete
    keeps the assertion meaning "every declared column was actually checked".
    """
    trained = pd.read_parquet(PROCESSED / "features.parquet")
    declared = [c for c in columns_for(table_blocks()) if c in trained.columns]
    complete = trained[declared].notna().all(axis=1)
    return set(trained.loc[complete, "fight_id"])


def _target_fight(fights: pd.DataFrame) -> pd.Series:
    """The 2024+ fight whose two fighters have the deepest records so far.

    "Veteran" has to mean bouts BEFORE the target date: counting career
    totals (which this did originally) let a fighter qualify on bouts that
    had not happened yet, and picked a matchup with 2 and 1 prior bouts --
    most per-corner columns were then NaN on both sides and silently skipped
    by the value comparison. Maximising min(prior_a, prior_b) picks a pair
    with real history on both sides. `_fully_populated_fights` adds the
    second constraint the blocks need: every declared column populated.
    """
    prior = _prior_bouts(fights)
    candidates = fights[
        (fights["date"] >= "2024-01-01") & fights["winner"].isin(["a", "b"])
    ].sort_values(["date", "fight_id"], kind="stable")
    candidates = candidates[candidates["fight_id"].isin(_fully_populated_fights())]
    assert len(candidates), "no 2024+ decisive fight found"
    depth = candidates.apply(
        lambda f: min(prior(f["fighter_a_id"], f["date"]),
                      prior(f["fighter_b_id"], f["date"])),
        axis=1,
    )
    return candidates.loc[depth.idxmax()]


def test_the_target_matchup_has_real_history_on_both_sides(tables):
    """A shallow matchup leaves most columns NaN on both sides, which the
    value comparison skips -- the test would then pass while comparing
    almost nothing."""
    fights = tables["fights"]
    target = _target_fight(fights)
    prior = _prior_bouts(fights)
    for corner in ("a", "b"):
        bouts = prior(target[f"fighter_{corner}_id"], target["date"])
        assert bouts >= 5, f"corner {corner} has only {bouts} prior bouts"


def test_served_row_equals_training_row(tables):
    fights, stats, fighters, ratings = (
        tables["fights"], tables["fight_stats"], tables["fighters"], tables["ratings"]
    )
    target = _target_fight(fights)

    past = fights[fights["date"] < target["date"]]
    past_ids = set(past["fight_id"])
    snapshots = build_snapshots(
        past, stats[stats["fight_id"].isin(past_ids)], ratings[ratings["fight_id"].isin(past_ids)]
    )
    indexed = fighters.set_index("fighter_id")
    served = build_matchup(
        snapshots.loc[target["fighter_a_id"]], snapshots.loc[target["fighter_b_id"]],
        indexed.loc[target["fighter_a_id"]], indexed.loc[target["fighter_b_id"]],
        target["weight_class"], bool(target["title_fight"]),
        int(target["scheduled_rounds"]), target["date"], blocks=table_blocks(),
        **_card_facts(target, past),
    )

    history = build_history(fights, stats, ratings)
    trained = build_features(fights, fighters, ratings, history,
                             blocks=table_blocks())
    row = trained[trained["fight_id"] == target["fight_id"]]
    assert len(row) == 1
    row = row.iloc[0]

    # The training table may have swapped corners for this fight; the served
    # row is always A-vs-B. Differentials flip sign, per-corner columns swap
    # which corner they came from, and the derived flags are symmetric.
    sign = -1.0 if bool(row["swapped"]) else 1.0
    sa, sb = ("b", "a") if bool(row["swapped"]) else ("a", "b")

    spec = spec_for(table_blocks())
    diff_columns = [f"{stem}_diff" for _, stem in spec.differentials]
    corner_pairs = [
        (f"{stem}_a", f"{stem}_{sa}") for stem in spec.absolutes + spec.booleans
    ] + [
        (f"{stem}_b", f"{stem}_{sb}") for stem in spec.absolutes + spec.booleans
    ]
    flag_columns = [name for name, _ in spec.derived_booleans]
    boolean_stems = set(spec.booleans)

    for column in served.columns:
        if column in ("weight_class", "title_fight", "scheduled_rounds"):
            continue
        assert column in row.index, f"served column {column} missing from the training table"

    mismatches = []
    compared = 0

    def compare(served_column, trained_column, expected):
        """Count a column as compared unless it is NaN on BOTH sides."""
        nonlocal compared
        got, want = served.iloc[0][served_column], row[trained_column]
        if pd.isna(got) and pd.isna(want):
            return
        compared += 1
        if got != expected(want):
            mismatches.append((served_column, got, expected(want)))

    for column in diff_columns:
        compare(column, column,
                lambda v: pytest.approx(sign * float(v), abs=1e-6))
    def _expect_bool(v):
        # `compare` already skips the case where both sides are NaN; if we
        # get here with a NaN `want`, `got` is a real bool and must not
        # match it -- `bool(nan)` is True, which let a NaN-vs-True served
        # value silently "match" before this fix.
        return float("nan") if pd.isna(v) else bool(v)

    for served_column, trained_column in corner_pairs:
        stem = served_column.rsplit("_", 1)[0]
        if stem in boolean_stems:
            compare(served_column, trained_column, _expect_bool)
        else:
            compare(served_column, trained_column,
                    lambda v: pytest.approx(float(v), abs=1e-6))
    for column in flag_columns:
        compare(column, column, lambda v: bool(v))

    # A column that collapses to NaN on both sides is skipped above, which is
    # exactly how a silently-unpopulated feature would hide here: require
    # every one of the spec's columns to have actually been compared. The
    # target matchup is picked by test_the_target_matchup_has_real_history_
    # on_both_sides (>=5 prior bouts per corner) precisely so this holds
    # without slack.
    expected_columns = len(diff_columns) + len(corner_pairs) + len(flag_columns)
    assert compared == expected_columns, (
        f"only {compared} of {expected_columns} spec columns compared"
    )
    assert not mismatches, f"served != trained for: {mismatches}"


def test_training_table_and_served_row_have_the_same_columns(tables):
    fights, stats, fighters, ratings = (
        tables["fights"], tables["fight_stats"], tables["fighters"], tables["ratings"]
    )
    target = _target_fight(fights)
    past = fights[fights["date"] < target["date"]]
    past_ids = set(past["fight_id"])
    snapshots = build_snapshots(
        past, stats[stats["fight_id"].isin(past_ids)], ratings[ratings["fight_id"].isin(past_ids)]
    )
    indexed = fighters.set_index("fighter_id")
    served = build_matchup(
        snapshots.loc[target["fighter_a_id"]], snapshots.loc[target["fighter_b_id"]],
        indexed.loc[target["fighter_a_id"]], indexed.loc[target["fighter_b_id"]],
        target["weight_class"], bool(target["title_fight"]),
        int(target["scheduled_rounds"]), target["date"], blocks=table_blocks(),
        **_card_facts(target, past),
    )
    trained = pd.read_parquet(PROCESSED / "features.parquet")
    identifiers = {"fight_id", "date", "swapped", "y_winner", "y_method", "y_finish_round"}
    assert set(served.columns) == set(trained.columns) - identifiers
    # Column SETS matching is not enough -- a registry reorder would
    # silently change the committed parquet's physical column order with a
    # green suite. Pin the order too.
    non_identifier_columns = [c for c in trained.columns if c not in identifiers]
    assert non_identifier_columns == list(served.columns)


# --------------------------------------------------------------------------
# The PREDICTION served must equal the prediction the harness measured (SP3)
# --------------------------------------------------------------------------
# Everything above pins the feature ROW: the values a future fight is served
# are the values the model trained on. The hybrid needs the same discipline one
# level up, on the prediction itself. `mma.candidates.HybridCandidate` is what
# the walk-forward harness scored and `mma.inference.SimulatorPredictor` is
# what serves, and they are two different code paths over the same models --
# one builds four frames at once from a fold's masks, the other aligns a
# served frame to each booster's own columns and categories.
#
# So this hands the SERVED path the very models a harness fold fitted, on the
# very fights that fold evaluated, and requires the joint distribution to come
# back identical. Refitting a lookalike would compare two fits; keeping the
# fitted members (`HazardCandidate.fitted_members`) compares two code paths,
# which is the thing that can silently drift.


class _StubBlend:
    """A blend that returns a fixed winner probability.

    `SimulatorPredictor` composes P_blend(winner) with the simulator's
    conditional; which winner it is does not matter to the parity claim, and
    fixing it keeps this test off the torch artifacts (and out of a two-minute
    blend fit) while still exercising the exact composition that ships.
    """

    ensemble = preprocessor = None
    weight = temperature = 0.0

    def __init__(self, winner):
        self.winner = np.asarray(winner, dtype=float)

    def predict(self, features):
        from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
        n = len(features)
        return {
            "winner_prob": self.winner[:n],
            "winner_spread": np.zeros(n),
            "method_probs": np.full((n, len(METHOD_CLASSES)), 1 / len(METHOD_CLASSES)),
            "round_probs": np.full((n, len(ROUND_CLASSES)), 1 / len(ROUND_CLASSES)),
            "method_classes": METHOD_CLASSES,
            "round_classes": ROUND_CLASSES,
        }


@pytest.fixture(scope="module")
def harness_hybrid(tables, tmp_path_factory):
    """One real walk-forward fold, fitted once: its joint cells, the models
    that produced them (written out as deployment artifacts), and the
    evaluation fights they were produced for."""
    from mma.candidates import HazardCandidate
    from mma.walkforward import make_folds

    features = pd.read_parquet(PROCESSED / "features.parquet")
    # The most recent fights only: a full-table fold is a minute of fitting for
    # a claim that a few thousand rows establish just as well.
    features = features.tail(2500).sort_values("date", kind="stable").reset_index(drop=True)
    fold = make_folds(features["date"], fold_years=(2025,))[0]
    candidate = HazardCandidate(fights=tables["fights"], seeds=(0, 1),
                                params={"max_depth": 3}, n_runs=2000)
    pred, _ = candidate.fit_predict(features, fold, None)

    models_dir = tmp_path_factory.mktemp("served") / "models"
    models_dir.mkdir()
    for member, models in candidate.fitted_members.items():
        for seed, model in zip(candidate.seeds, models):
            model.save_model(models_dir / f"xgb_{member}_seed{seed}.json")

    eval_feats = features.loc[fold.eval].reset_index(drop=True)
    # A blend-shaped winner probability: deterministic, varied, and never 0 or 1.
    winner = 0.5 + 0.35 * np.sin(np.arange(len(eval_feats), dtype=float))
    return {"cells": pred["joint_cells"], "eval": eval_feats, "winner": winner,
            "root": models_dir.parent, "candidate": candidate}


def test_served_hybrid_reproduces_the_harness_joint(harness_hybrid):
    """The single claim this whole deployment rests on.

    Served through `SimulatorPredictor`, the models a fold fitted must produce
    the fold's own joint distribution -- cell for cell -- once the blend's
    winner is imposed on it, which is exactly `HybridCandidate`'s composition.
    Anything less and the numbers in `models/walkforward/sp3_decision.json`
    describe a scorer other than the one serving.
    """
    from mma.inference import SimulatorPredictor
    from mma.joint import impose_winner_marginal, marginals_from_cells
    from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
    from mma.simulator import DEFAULT_ALPHA, DEFAULT_ROUNDS, DEFAULT_SIM_SEED

    served = SimulatorPredictor.load(
        root=harness_hybrid["root"], blend=_StubBlend(harness_hybrid["winner"]),
        config={"n_runs": harness_hybrid["candidate"].n_runs, "alpha": DEFAULT_ALPHA,
                "sim_seed": DEFAULT_SIM_SEED, "default_rounds": DEFAULT_ROUNDS},
    ).predict(harness_hybrid["eval"])

    expected = impose_winner_marginal(
        harness_hybrid["cells"], harness_hybrid["winner"], METHOD_CLASSES, ROUND_CLASSES)
    np.testing.assert_array_equal(served["joint_cells"], expected)

    marginals = marginals_from_cells(expected, METHOD_CLASSES, ROUND_CLASSES)
    np.testing.assert_array_equal(served["method_probs"], marginals["method"])
    np.testing.assert_array_equal(served["round_probs"], marginals["round"])
    # and the winner marginal of what is served is the blend's, exactly
    np.testing.assert_allclose(marginals["winner"], harness_hybrid["winner"], atol=1e-12)


def test_served_hybrid_reproduces_the_harness_simulator_before_any_imposition(harness_hybrid):
    """The simulator half on its own, with no winner imposed.

    Imposing a joint's OWN winner marginal is an exact no-op, so serving with
    a stub blend that returns the simulator's winner has to return the
    simulator's cells. That isolates the simulation and the seed ensembling
    from the hybrid composition: if this passes and the test above fails, the
    composition drifted; if this fails, the served simulation did.
    """
    from mma.inference import SimulatorPredictor
    from mma.joint import marginals_from_cells
    from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES
    from mma.simulator import DEFAULT_ALPHA, DEFAULT_ROUNDS, DEFAULT_SIM_SEED

    own_winner = marginals_from_cells(
        harness_hybrid["cells"], METHOD_CLASSES, ROUND_CLASSES)["winner"]
    served = SimulatorPredictor.load(
        root=harness_hybrid["root"], blend=_StubBlend(own_winner),
        config={"n_runs": harness_hybrid["candidate"].n_runs, "alpha": DEFAULT_ALPHA,
                "sim_seed": DEFAULT_SIM_SEED, "default_rounds": DEFAULT_ROUNDS},
    ).predict(harness_hybrid["eval"])
    np.testing.assert_allclose(served["joint_cells"], harness_hybrid["cells"], atol=1e-12)
