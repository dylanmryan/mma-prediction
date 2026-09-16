"""The reusable "does this add anything?" instrument.

Three things could make this tool lie, and each has a test: an offset that is
not really pinned (the fit then earns a gain by re-calibrating the deployed
model and credits it to the new columns), an orientation that does not follow
the feature table's corner swap, and a wear accumulator that can see a fight's
own outcome. The fourth test is the one that matters most in practice -- that
the detection floor is reported honestly, because a null result from a
procedure that cannot resolve the bar says nothing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.residual_probe import (
    EXPLORE_YEARS,
    GROUPS,
    HELD_BACK,
    MIN_ROWS,
    PROJECT_BAR,
    Wear,
    assess_subgroup,
    build_wear,
    linear_gain,
    log_loss,
    logit,
    orient,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "residual_probes.json"


# --- the offset really is pinned ----------------------------------------------

def test_a_useless_column_earns_nothing_even_when_the_offset_is_miscalibrated():
    """The trap the pinned offset exists to avoid.

    The offset here is deliberately over-confident, so there IS a gain
    available from shrinking it. A free coefficient would take that gain and
    attribute it to the noise column. Pinned, the column earns nothing.
    """
    rng = np.random.default_rng(0)
    n = 2000
    true_p = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < true_p).astype(float)
    overconfident = logit(true_p) * 1.8          # miscalibrated on purpose
    noise = rng.normal(size=(n, 1))
    gain = np.mean([linear_gain(noise, y, overconfident, seed=s) for s in range(3)])
    assert gain < 0.002, (
        f"a pure-noise column earned {gain:+.5f}; the offset is not pinned and "
        "the fit is being paid for re-calibrating the deployed model"
    )


def test_a_genuinely_informative_column_is_found():
    """The instrument has to be able to say yes, or its noes are worthless."""
    rng = np.random.default_rng(1)
    n = 2000
    hidden = rng.normal(size=n)
    base = rng.normal(size=n) * 0.5
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(base + 1.2 * hidden)))).astype(float)
    gain = np.mean([linear_gain(hidden.reshape(-1, 1), y, base, seed=s)
                    for s in range(3)])
    assert gain > 0.05, f"missed a strong signal (gain {gain:+.5f})"


# --- the floor is reported, and it is what makes a null meaningful ------------

def test_a_null_result_reports_whether_it_could_have_seen_the_bar():
    rng = np.random.default_rng(2)
    n = 1200
    frame = pd.DataFrame({
        "p": rng.uniform(0.3, 0.7, n),
        "noise_diff": rng.normal(size=n),
    })
    frame["y"] = (rng.uniform(size=n) < frame["p"]).astype(float)
    out = assess_subgroup(frame, ["noise_diff"], np.random.default_rng(3))
    assert out["linear"]["signal"] is False
    assert "detection_floor" in out["linear"]
    assert isinstance(out["linear"]["floor_resolves_the_bar"], bool)
    assert out["linear"]["floor_resolves_the_bar"] == (
        out["linear"]["detection_floor"] <= PROJECT_BAR)


def test_too_few_rows_is_reported_rather_than_scored():
    frame = pd.DataFrame({"p": [0.5] * 10, "y": [1.0] * 10,
                          "x_diff": list(range(10))})
    out = assess_subgroup(frame, ["x_diff"], np.random.default_rng(0))
    assert out["skipped"]
    assert out["n"] < MIN_ROWS


# --- orientation, the bug class this repo has hit twice -----------------------

def test_orient_follows_the_feature_tables_swap_flag():
    per_fighter = pd.DataFrame({
        "fight_id": ["f1", "f1", "f2", "f2"],
        "fighter_id": ["A1", "B1", "A2", "B2"],
        "kd_suffered": [10.0, 1.0, 10.0, 1.0],
    })
    fights = pd.DataFrame({"fight_id": ["f1", "f2"],
                           "fighter_a_id": ["A1", "A2"],
                           "fighter_b_id": ["B1", "B2"]})
    features = pd.DataFrame({"fight_id": ["f1", "f2"], "swapped": [False, True]})
    out = orient(per_fighter, fights, features, ("kd_suffered",))
    # f1 unswapped: A1 - B1 = +9. f2 swapped: corner A is B2, so B2 - A2 = -9.
    assert out["kd_suffered_diff"].tolist() == [9.0, -9.0]


def test_orient_joins_on_ids_not_on_row_position():
    per_fighter = pd.DataFrame({
        "fight_id": ["f2", "f2", "f1", "f1"],       # deliberately out of order
        "fighter_id": ["A2", "B2", "A1", "B1"],
        "kd_suffered": [5.0, 0.0, 10.0, 1.0],
    })
    fights = pd.DataFrame({"fight_id": ["f1", "f2"],
                           "fighter_a_id": ["A1", "A2"],
                           "fighter_b_id": ["B1", "B2"]})
    features = pd.DataFrame({"fight_id": ["f1", "f2"], "swapped": [False, False]})
    out = orient(per_fighter, fights, features, ("kd_suffered",))
    assert out["kd_suffered_diff"].tolist() == [9.0, 5.0]


# --- the wear accumulator is point-in-time ------------------------------------

def _two_fight_fixture():
    fights = pd.DataFrame({
        "fight_id": ["f1", "f2"],
        "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "duration_sec": [900.0, 900.0],
        "fighter_a_id": ["A", "A"],
        "fighter_b_id": ["B", "C"],
    })
    stats = pd.DataFrame({
        "fight_id": ["f1", "f1", "f2", "f2"],
        "fighter_id": ["A", "B", "A", "C"],
        "kd": [0.0, 2.0, 0.0, 3.0],
        "head_landed": [10.0, 40.0, 10.0, 60.0],
        "sig_landed": [20.0, 50.0, 20.0, 70.0],
    })
    fighters = pd.DataFrame({"fighter_id": ["A", "B", "C"],
                             "dob": pd.to_datetime(["1990-01-01"] * 3)})
    return fights, stats, fighters


def test_wear_is_emitted_before_the_fight_is_folded_in():
    """A's first row must be an empty record, not one containing f1."""
    wear = build_wear(*_two_fight_fixture()).set_index(["fight_id", "fighter_id"])
    assert wear.loc[("f1", "A"), "kd_suffered"] == 0.0
    assert wear.loc[("f1", "A"), "head_absorbed"] == 0.0
    # by f2, A has absorbed exactly what B did to them in f1
    assert wear.loc[("f2", "A"), "kd_suffered"] == 2.0
    assert wear.loc[("f2", "A"), "head_absorbed"] == 40.0


def test_a_later_fight_never_moves_an_earlier_row():
    fights, stats, fighters = _two_fight_fixture()
    full = build_wear(fights, stats, fighters)
    truncated = build_wear(fights.iloc[:1], stats[stats["fight_id"] == "f1"],
                           fighters)
    key = ["fight_id", "fighter_id"]
    merged = full.merge(truncated, on=key, suffixes=("_full", "_trunc"))
    for field in Wear.FIELDS:
        if field == "age_x_head_absorbed":
            continue
        a, b = merged[f"{field}_full"], merged[f"{field}_trunc"]
        # compared as VALUES, not dtypes: a rate with no denominator yet comes
        # back as None from one build and NaN from the other purely because
        # pandas infers an all-missing column differently. Both mean "no
        # fights yet", and the property under test is that the numbers agree.
        both_missing = a.isna() & b.isna()
        assert (both_missing | (a.fillna(0) == b.fillna(0))).all(), field


def test_the_recent_window_forgets_damage_older_than_a_year():
    wear = Wear()
    wear.fold({"kd": 0, "head_landed": 100, "sig_landed": 100}, 900,
              pd.Timestamp("2020-01-01"))
    fresh = wear.snapshot(pd.Timestamp("2020-06-01"), age=30.0)
    stale = wear.snapshot(pd.Timestamp("2023-01-01"), age=33.0)
    assert fresh["head_absorbed_365d"] == 100
    assert stale["head_absorbed_365d"] == 0
    assert stale["head_absorbed"] == 100, "the career total never forgets"


# --- the committed result -----------------------------------------------------

def test_the_windows_do_not_overlap():
    assert not set(EXPLORE_YEARS) & set(HELD_BACK)


def test_every_registered_group_names_real_fields():
    for name, (_, fields, subgroups) in GROUPS.items():
        for label, cols in subgroups.items():
            assert set(cols) <= set(fields), f"{name}/{label}"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="probe not run")
def test_the_committed_probe_found_no_signal():
    """If this fails a group has started showing something, and it needs a
    pre-registered harness run rather than a louder screen."""
    report = json.loads(ARTIFACT.read_text())
    for name, block in report["groups"].items():
        assert block["any_signal"] is False, name
