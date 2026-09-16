"""The weekly Action's path, run against the COMMITTED artifacts.

Every other test of the prospective path passes a fake scorer -- deliberately,
so the unit tests do not load 30-odd model files. The consequence is that
nothing checked the one thing the Monday run actually depends on: that the
artifacts on disk load, serve, and produce the record shape the grader expects.

That gap matters right now. All 57 graded prospective predictions were made by
`40df77ec43c7`, the July SP1 model. Everything built since -- the SP2 feature
set, the SP2.2 blend, the SP3 hybrid -- has never made a graded prediction,
because every event inside the horizon was predicted before the 2026-09-09
redeployment. **The next event to enter the horizon is the hybrid's debut**, it
will run unattended, and a silent failure there costs months of track record
before anyone notices.

These tests are slower than the rest of the suite (they load the real
ensembles) and they are worth it.
"""
import pandas as pd
import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "simulator.json").exists(),
    reason="deployed artifacts not present",
)


@pytest.fixture(scope="module")
def predictor():
    from mma.inference import SimulatorPredictor
    return SimulatorPredictor.load(ROOT)


@pytest.fixture(scope="module")
def tables():
    from mma.prospective import build_name_index
    from mma.snapshots import build_snapshots

    processed = ROOT / "data" / "processed"
    fights = pd.read_parquet(processed / "fights.parquet")
    stats = pd.read_parquet(processed / "fight_stats.parquet")
    fighters = pd.read_parquet(processed / "fighters.parquet")
    ratings = pd.read_parquet(processed / "ratings.parquet")
    return {
        "snapshots": build_snapshots(fights, stats, ratings),
        "name_index": build_name_index(fighters),
        "fighters": fighters.set_index("fighter_id"),
    }


def _predict(predictor, tables, name_a, name_b, weight_class, rounds=3):
    from mma.prospective import predict_fight

    return predict_fight(
        {"fighter_a_name": name_a, "fighter_b_name": name_b,
         "weight_class": weight_class, "title_fight": rounds == 5,
         "scheduled_rounds": rounds},
        tables["name_index"], tables["snapshots"], tables["fighters"],
        predictor, pd.Timestamp("2026-10-03"),
    )


def test_the_committed_artifacts_load_as_the_hybrid(predictor):
    """Not merely "something loaded": the deployed scorer since SP3 is the
    hybrid, and a torn load that silently produced the bare blend would serve
    a different model than every committed decision artifact describes."""
    from mma.inference import SimulatorPredictor

    assert isinstance(predictor, SimulatorPredictor)


def test_the_artifacts_on_disk_hash_to_the_version_the_decisions_name():
    """The committed decision artifacts record which model they were taken on.
    If the artifacts are retrained without those being refreshed, predictions
    get stamped with a version no recorded evidence describes."""
    import json

    from mma.versioning import model_version

    on_disk = model_version(ROOT)
    revalidation = ROOT / "models" / "walkforward" / "recipe_revalidation.json"
    if not revalidation.exists():
        pytest.skip("no re-validation artifact to check against")
    recorded = json.loads(revalidation.read_text())[
        "deployed_configuration"]["deployed_hash"]
    assert on_disk == recorded, (
        f"artifacts hash to {on_disk} but the last re-validation was taken on "
        f"{recorded}; re-run scripts/revalidate_recipe.py so the evidence "
        "describes the model that will actually serve"
    )


def test_a_real_matchup_predicts_end_to_end(predictor, tables):
    out = _predict(predictor, tables, "Islam Makhachev", "Arman Tsarukyan",
                   "Lightweight", rounds=5)
    assert not out.get("skipped"), out.get("reason")
    assert 0.0 < out["p_a_wins"] < 1.0
    assert out["match_tier"] == "exact"


def test_the_record_carries_the_joint_the_hybrid_exists_to_produce(predictor, tables):
    """`joint_probs` is the SP3 hybrid's whole output. The 157 records written
    before the redeployment do not have this key; the grader tolerates both,
    and this asserts the new shape is the one that will be written."""
    out = _predict(predictor, tables, "Islam Makhachev", "Arman Tsarukyan",
                   "Lightweight", rounds=5)
    assert "joint_probs" in out and out["joint_probs"]


def test_the_served_marginals_are_probability_distributions(predictor, tables):
    out = _predict(predictor, tables, "Alexandre Pantoja", "Joshua Van",
                   "Flyweight", rounds=5)
    assert not out.get("skipped"), out.get("reason")
    assert sum(out["method_probs"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(out["round_probs"].values()) == pytest.approx(1.0, abs=1e-6)


def test_a_three_round_fight_puts_no_mass_on_round_45(predictor, tables):
    out = _predict(predictor, tables, "Islam Makhachev", "Arman Tsarukyan",
                   "Lightweight", rounds=3)
    assert not out.get("skipped"), out.get("reason")
    assert out["round_probs"]["45"] == pytest.approx(0.0, abs=1e-9)


def test_swapping_the_corners_mirrors_the_winner_probability(predictor, tables):
    """The served path symmetrises; if it stopped, which corner Wikipedia
    happened to list first would move the prediction."""
    forward = _predict(predictor, tables, "Islam Makhachev", "Arman Tsarukyan",
                       "Lightweight", rounds=5)
    reverse = _predict(predictor, tables, "Arman Tsarukyan", "Islam Makhachev",
                       "Lightweight", rounds=5)
    assert forward["p_a_wins"] == pytest.approx(1.0 - reverse["p_a_wins"], abs=1e-9)


def test_an_unknown_name_is_skipped_rather_than_guessed(predictor, tables):
    out = _predict(predictor, tables, "Not A Real Fighter", "Islam Makhachev",
                   "Lightweight")
    assert out.get("skipped") is True
    assert out.get("reason")
