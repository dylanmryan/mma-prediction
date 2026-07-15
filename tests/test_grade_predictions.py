"""TDD grading math on synthetic records -- no network, no real data files."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.grade_predictions import (
    aggregate_track_record,
    find_result,
    grade_fight,
)


def _fights_df():
    return pd.DataFrame(
        {
            "fighter_a_id": ["x", "y", "m"],
            "fighter_b_id": ["y", "x", "n"],
            "date": pd.to_datetime(["2026-07-18", "2026-08-01", "2026-08-10"]),
            "winner": ["a", "b", "draw"],
        }
    )


# --- find_result -----------------------------------------------------------


def test_find_result_matches_unordered_ids_within_window():
    fight = {"fighter_a_id": "y", "fighter_b_id": "x"}  # swapped order vs. row 1
    result = find_result(fight, _fights_df(), "2026-07-19")  # 1 day off
    assert result is not None
    assert result["winner"] == "a"


def test_find_result_returns_none_when_ids_dont_match():
    fight = {"fighter_a_id": "p", "fighter_b_id": "q"}
    assert find_result(fight, _fights_df(), "2026-07-18") is None


def test_find_result_respects_date_window():
    fight = {"fighter_a_id": "x", "fighter_b_id": "y"}
    # 10 days off -- outside the +/-3 day grading window
    assert find_result(fight, _fights_df(), "2026-07-28") is None


def test_find_result_within_window_boundary_matches():
    fight = {"fighter_a_id": "x", "fighter_b_id": "y"}
    result = find_result(fight, _fights_df(), "2026-07-21")  # exactly 3 days off
    assert result is not None


def test_find_result_missing_ids_returns_none():
    assert find_result({"fighter_a_id": None, "fighter_b_id": "y"}, _fights_df(), "2026-07-18") is None


def test_find_result_empty_fights_returns_none():
    assert find_result({"fighter_a_id": "x", "fighter_b_id": "y"}, pd.DataFrame(), "2026-07-18") is None


# --- grade_fight -------------------------------------------------------


def test_grade_fight_correct_confident_win():
    fight = {"fighter_a_id": "x", "fighter_b_id": "y", "p_a_wins": 0.9,
             "elo_a": 1600.0, "elo_b": 1500.0}
    result = pd.Series({"fighter_a_id": "x", "fighter_b_id": "y", "winner": "a"})
    graded = grade_fight(fight, result)
    assert graded["actual_winner"] == "x"
    assert graded["correct"] is True
    assert graded["log_loss_contribution"] == pytest.approx(-math.log(0.9))
    assert graded["brier_contribution"] == pytest.approx((0.9 - 1.0) ** 2)
    assert graded["coin_flip_correct"] is True  # actual winner was corner A
    assert graded["elo_dummy_correct"] is True  # higher elo (x) actually won


def test_grade_fight_incorrect_prediction():
    fight = {"fighter_a_id": "x", "fighter_b_id": "y", "p_a_wins": 0.9,
             "elo_a": 1600.0, "elo_b": 1500.0}
    result = pd.Series({"fighter_a_id": "x", "fighter_b_id": "y", "winner": "b"})
    graded = grade_fight(fight, result)
    assert graded["actual_winner"] == "y"
    assert graded["correct"] is False
    assert graded["log_loss_contribution"] == pytest.approx(-math.log(0.1))
    assert graded["coin_flip_correct"] is False
    assert graded["elo_dummy_correct"] is False  # higher elo (x) did NOT win


def test_grade_fight_handles_swapped_corners_in_result_row():
    # Result row lists the SAME pair but corners flipped vs. the prediction.
    fight = {"fighter_a_id": "x", "fighter_b_id": "y", "p_a_wins": 0.7,
             "elo_a": 1500.0, "elo_b": 1600.0}
    result = pd.Series({"fighter_a_id": "y", "fighter_b_id": "x", "winner": "b"})  # x won
    graded = grade_fight(fight, result)
    assert graded["actual_winner"] == "x"
    assert graded["correct"] is True  # predicted A (x) with p=0.7, and x won


def test_grade_fight_draw_excludes_from_accuracy():
    fight = {"fighter_a_id": "x", "fighter_b_id": "y", "p_a_wins": 0.7,
             "elo_a": 1500.0, "elo_b": 1600.0}
    result = pd.Series({"fighter_a_id": "x", "fighter_b_id": "y", "winner": "draw"})
    graded = grade_fight(fight, result)
    assert graded["actual_winner"] == "draw"
    assert graded["correct"] is None
    assert graded["coin_flip_correct"] is None
    assert graded["elo_dummy_correct"] is None
    # log-loss/Brier still scored against y=0.5
    assert graded["log_loss_contribution"] == pytest.approx(
        -(0.5 * math.log(0.7) + 0.5 * math.log(0.3))
    )
    assert graded["brier_contribution"] == pytest.approx((0.7 - 0.5) ** 2)


def test_grade_fight_elo_tie_is_undefined():
    fight = {"fighter_a_id": "x", "fighter_b_id": "y", "p_a_wins": 0.6,
             "elo_a": 1500.0, "elo_b": 1500.0}
    result = pd.Series({"fighter_a_id": "x", "fighter_b_id": "y", "winner": "a"})
    graded = grade_fight(fight, result)
    assert graded["elo_dummy_correct"] is None


def test_grade_fight_coin_flip_constants():
    fight = {"fighter_a_id": "x", "fighter_b_id": "y", "p_a_wins": 0.5,
             "elo_a": 1500.0, "elo_b": 1500.0}
    result = pd.Series({"fighter_a_id": "x", "fighter_b_id": "y", "winner": "a"})
    graded = grade_fight(fight, result)
    assert graded["coin_flip_log_loss_contribution"] == pytest.approx(math.log(2.0))
    assert graded["coin_flip_brier_contribution"] == 0.25


# --- aggregate_track_record ----------------------------------------------


def _graded_event(model_version, fights):
    return {"event_name": "Test Event", "event_date": "2026-07-18", "fights": fights}


def test_aggregate_track_record_hand_computed_accuracy_log_loss():
    fights = [
        {
            "fighter_a_id": "x", "fighter_b_id": "y", "p_a_wins": 0.8, "skipped": False,
            "model_version": "abc1234", "predicted_at_utc": "2026-07-10T00:00:00Z",
            "actual_winner": "x", "correct": True,
            "log_loss_contribution": -math.log(0.8), "brier_contribution": (0.8 - 1.0) ** 2,
            "coin_flip_correct": True, "coin_flip_log_loss_contribution": math.log(2.0),
            "coin_flip_brier_contribution": 0.25, "elo_dummy_correct": True,
        },
        {
            "fighter_a_id": "m", "fighter_b_id": "n", "p_a_wins": 0.3, "skipped": False,
            "model_version": "abc1234", "predicted_at_utc": "2026-07-11T00:00:00Z",
            "actual_winner": "n", "correct": True,
            "log_loss_contribution": -math.log(0.7), "brier_contribution": (0.3 - 0.0) ** 2,
            "coin_flip_correct": False, "coin_flip_log_loss_contribution": math.log(2.0),
            "coin_flip_brier_contribution": 0.25, "elo_dummy_correct": False,
        },
    ]
    track = aggregate_track_record([_graded_event("abc1234", fights)], "2026-07-20T00:00:00Z")

    overall = track["overall"]
    assert overall["n_predicted"] == 2
    assert overall["n_graded"] == 2
    assert overall["accuracy"] == pytest.approx(1.0)
    assert overall["log_loss"] == pytest.approx((-math.log(0.8) - math.log(0.7)) / 2)
    assert overall["brier"] == pytest.approx(((0.2) ** 2 + (0.3) ** 2) / 2)

    assert track["model_versions"]["abc1234"]["n_graded"] == 2
    assert track["model_versions"]["abc1234"]["first_prediction_at"] == "2026-07-10T00:00:00Z"

    assert track["baselines"]["coin_flip"]["accuracy"] == pytest.approx(0.5)
    assert track["baselines"]["coin_flip"]["log_loss"] == pytest.approx(math.log(2.0))
    assert track["baselines"]["higher_elo_dummy"]["accuracy"] == pytest.approx(0.5)


def test_aggregate_track_record_separates_by_model_version():
    fight_v1 = {
        "p_a_wins": 0.6, "skipped": False, "model_version": "v1",
        "predicted_at_utc": "2026-01-01T00:00:00Z",
        "actual_winner": "x", "correct": True,
        "log_loss_contribution": 0.5, "brier_contribution": 0.1,
        "coin_flip_correct": True, "coin_flip_log_loss_contribution": math.log(2.0),
        "coin_flip_brier_contribution": 0.25, "elo_dummy_correct": None,
    }
    fight_v2 = {
        "p_a_wins": 0.6, "skipped": False, "model_version": "v2",
        "predicted_at_utc": "2026-02-01T00:00:00Z",
        "actual_winner": "x", "correct": False,
        "log_loss_contribution": 1.5, "brier_contribution": 0.4,
        "coin_flip_correct": True, "coin_flip_log_loss_contribution": math.log(2.0),
        "coin_flip_brier_contribution": 0.25, "elo_dummy_correct": None,
    }
    track = aggregate_track_record(
        [_graded_event(None, [fight_v1, fight_v2])], "2026-07-20T00:00:00Z"
    )
    assert set(track["model_versions"]) == {"v1", "v2"}
    assert track["model_versions"]["v1"]["accuracy"] == pytest.approx(1.0)
    assert track["model_versions"]["v2"]["accuracy"] == pytest.approx(0.0)
    assert track["overall"]["n_graded"] == 2


def test_aggregate_track_record_ignores_skipped_and_ungraded_fights():
    fights = [
        {"skipped": True, "reason": "no match", "fighter_a_name": "A", "fighter_b_name": "B"},
        {
            "p_a_wins": 0.5, "skipped": False, "model_version": "v1",
            "predicted_at_utc": "2026-01-01T00:00:00Z",
            # no actual_winner yet -- still pending
        },
    ]
    track = aggregate_track_record([_graded_event("v1", fights)], "2026-07-20T00:00:00Z")
    assert track["overall"]["n_predicted"] == 1  # skipped fight excluded entirely
    assert track["overall"]["n_graded"] == 0
    assert track["overall"]["accuracy"] is None
    assert track["overall"]["log_loss"] is None


def test_aggregate_track_record_empty_input():
    track = aggregate_track_record([], "2026-07-20T00:00:00Z")
    assert track["overall"] == {
        "n_predicted": 0, "n_graded": 0, "accuracy": None, "log_loss": None, "brier": None,
    }
    assert track["model_versions"] == {}
    assert track["baselines"]["coin_flip"]["n_graded"] == 0
