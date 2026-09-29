"""The market probe: does our odds-free signal add anything on top of the line?

Every other residual probe pins the DEPLOYED model's logit and asks whether new
columns add on top of it. The market probe inverts that -- it pins the devigged
closing line and asks whether the model's own logit adds on top of *that* --
which turns "we lose to the market by 0.036" into a number: what our odds-free
signal is worth to someone who already has the line. That number is also the
ceiling for any future betting layer, so it decides whether one is worth
building.

Inverting the instrument means the offset is no longer a constant of the
script, so these tests pin that the pinned side is really the side named, that
the default has not moved for the groups already registered, and that the market
column reaches the probe by a `fight_id` join rather than by row position.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.residual_probe import assess_subgroup, log_loss, logit


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 900
    skill = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-skill))).astype(float)
    # Two informative but different views of the same fights.
    model = 1 / (1 + np.exp(-(0.8 * skill + rng.normal(scale=0.5, size=n))))
    market = 1 / (1 + np.exp(-(1.2 * skill + rng.normal(scale=0.3, size=n))))
    return pd.DataFrame({
        "y": y, "p": model, "market_p": market,
        "model_logit": logit(model), "market_logit": logit(market),
    })


def test_the_baseline_is_the_log_loss_of_whichever_side_is_pinned(frame):
    """The instrument's whole claim is that the offset is FIXED, so the reported
    baseline must be the pinned side's own log-loss and nothing else. If these
    two came back equal, the offset argument would not be doing anything."""
    rng = np.random.default_rng(0)
    model_pinned = assess_subgroup(frame, ["market_logit"], rng, offset="p")
    market_pinned = assess_subgroup(frame, ["model_logit"], rng, offset="market_p")

    assert model_pinned["baseline_log_loss"] == pytest.approx(
        round(log_loss(frame["y"], frame["p"]), 4))
    assert market_pinned["baseline_log_loss"] == pytest.approx(
        round(log_loss(frame["y"], frame["market_p"]), 4))
    assert model_pinned["baseline_log_loss"] != market_pinned["baseline_log_loss"]


def test_the_pinned_side_is_named_in_the_result(frame):
    """A gain of 0.01 means opposite things depending on what was pinned, so the
    artifact has to say. Reading a market-pinned number as a model-pinned one
    would invert the conclusion."""
    rng = np.random.default_rng(0)
    out = assess_subgroup(frame, ["model_logit"], rng, offset="market_p")
    assert out["offset"] == "market_p"


def test_the_default_offset_is_still_the_deployed_model(frame):
    """Regression guard: `accumulated_damage` and every future column group is
    a question about what the DEPLOYED model already knows. If the default
    moved, those results would silently change meaning."""
    rng = np.random.default_rng(0)
    assert assess_subgroup(frame, ["market_logit"], rng)["offset"] == "p"


def test_a_stronger_pinned_side_leaves_less_for_the_other_to_add(frame):
    """A sanity check on the arithmetic of an offset probe, using a fixture
    where the market view is built to be the better one: pinning the stronger
    side must leave the weaker side less to contribute than the reverse."""
    rng = np.random.default_rng(0)
    on_top_of_market = assess_subgroup(frame, ["model_logit"], rng,
                                       offset="market_p")["linear"]["gain"]
    on_top_of_model = assess_subgroup(frame, ["market_logit"], rng,
                                      offset="p")["linear"]["gain"]
    assert on_top_of_market < on_top_of_model


# --- the committed per-fight market table -------------------------------------

def test_the_market_table_is_committed_and_keyed_by_fight_id():
    """`build_odds_benchmark.py` downloads the odds and writes only aggregates,
    so before this there was no way to re-derive a per-fight market number
    offline -- which would have made the market probe depend on a network call
    and unreproducible in CI. The table is committed for the same reason
    `external`, `notice` and `rankings` are."""
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    table = pd.read_parquet(root / "data" / "external" / "market_odds.parquet")
    assert "fight_id" in table.columns
    assert not table["fight_id"].duplicated().any()
    assert table["market_implied_a"].between(0.0, 1.0).all()

    features = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    assert set(table["fight_id"]) <= set(features["fight_id"])


def test_every_market_row_is_oriented_to_the_feature_tables_corner():
    """The table stores corner A in features.parquet's frame, not
    fights.parquet's. If the orientation were dropped, roughly half the rows
    would be inverted and the probe would read the line backwards -- which
    would look like the market being anti-predictive rather than like a bug."""
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    table = pd.read_parquet(root / "data" / "external" / "market_odds.parquet")
    features = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    merged = table.merge(features[["fight_id", "y_winner"]], on="fight_id")
    # The devigged favourite should win more often than not in the feature
    # table's own frame. Read with the orientation dropped this lands below 0.5.
    favoured_won = ((merged["market_implied_a"] > 0.5) ==
                    (merged["y_winner"] > 0.5)).mean()
    assert favoured_won > 0.60


def test_the_screen_is_null_and_the_diagnostic_is_not():
    """The two directions must be read separately, and the artifact says so.

    The screen -- does our odds-free signal add on top of the line -- came back
    null, and the diagnostic -- does the line add on top of us -- came back
    overwhelmingly positive. Folding them into one flag would either hide the
    null or make the instrument look like it had found a shippable feature."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    block = json.loads(
        (root / "models" / "residual_probes.json").read_text())["groups"]["market"]
    assert block["any_signal"] is False
    assert block["diagnostic_signal"] is True


def test_the_null_came_from_a_procedure_that_could_have_seen_the_effect():
    """The one check that makes the null worth anything. A detection floor above
    the bar would mean the probe simply could not resolve what it was looking
    for, and the negative would say nothing at all."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    block = json.loads(
        (root / "models" / "residual_probes.json").read_text())["groups"]["market"]
    screen = block["subgroups"]["our signal on top of the closing line"]
    assert screen["role"] == "screen"
    assert screen["linear"]["floor_resolves_the_bar"] is True
    assert screen["linear"]["signal"] is False
