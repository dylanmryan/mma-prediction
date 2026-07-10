import pandas as pd
import pytest

from mma.elo import EloParams, expected_score, run_elo


def _fights():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2", "f3"],
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-01"]),
            "fighter_a_id": ["x", "x", "y"],
            "fighter_b_id": ["y", "z", "z"],
            "winner": ["a", "a", "draw"],
            "method": ["decision", "ko_tko", "decision"],
        }
    )


def _stats():
    rows = []
    for fid in ("f1", "f2", "f3"):
        for corner in ("a", "b"):
            rows.append(
                {
                    "fight_id": fid, "corner": corner,
                    "sig_landed": 50, "td_landed": 0, "sub_att": 0, "ctrl_sec": 0,
                }
            )
    return pd.DataFrame(rows)


PARAMS = EloParams(k_early=40, k_late=24, early_fights=5, finish_bonus=1.5)


def test_first_fight_starts_at_1500():
    ratings = run_elo(_fights(), _stats(), PARAMS)
    f1 = ratings[ratings["fight_id"] == "f1"]
    assert (f1["pre_overall"] == 1500.0).all()
    assert (f1["pre_fights"] == 0).all()


def test_decision_win_updates_symmetrically():
    ratings = run_elo(_fights(), _stats(), PARAMS)
    f1 = ratings[ratings["fight_id"] == "f1"].set_index("corner")
    # equal ratings, K=40, decision (no bonus): delta = 40 * (1 - 0.5) = 20
    assert f1.loc["a", "post_overall"] == pytest.approx(1520.0)
    assert f1.loc["b", "post_overall"] == pytest.approx(1480.0)


def test_finish_bonus_scales_delta():
    ratings = run_elo(_fights(), _stats(), PARAMS)
    f2 = ratings[ratings["fight_id"] == "f2"].set_index("corner")
    # x is 1520 pre; z is 1500. expected = E(1520,1500); ko bonus 1.5
    expected = expected_score(1520.0, 1500.0)
    assert f2.loc["a", "pre_overall"] == pytest.approx(1520.0)
    assert f2.loc["a", "post_overall"] == pytest.approx(
        1520.0 + 40 * 1.5 * (1 - expected)
    )


def test_draw_zero_sum_and_direction():
    ratings = run_elo(_fights(), _stats(), PARAMS)
    f3 = ratings[ratings["fight_id"] == "f3"].set_index("corner")
    # y (corner a) is rated higher than z pre-fight; a draw must cost y points
    assert f3.loc["a", "pre_overall"] > f3.loc["b", "pre_overall"]
    assert f3.loc["a", "post_overall"] < f3.loc["a", "pre_overall"]
    # both fighters under early_fights -> equal K -> zero-sum
    assert f3["post_overall"].sum() == pytest.approx(f3["pre_overall"].sum())


def test_nc_fights_skipped():
    fights = _fights()
    fights.loc[1, "winner"] = "nc"
    ratings = run_elo(fights, _stats(), PARAMS)
    assert "f2" not in set(ratings["fight_id"])
    f3 = ratings[ratings["fight_id"] == "f3"]
    assert len(f3) == 2
    # z never got a rated fight before f3
    assert f3.set_index("corner").loc["b", "pre_fights"] == 0


def test_striking_grappling_split():
    stats = _stats()
    # make f1 purely grappling-dominated
    mask = stats["fight_id"] == "f1"
    stats.loc[mask, "sig_landed"] = 0
    stats.loc[mask, "td_landed"] = 4
    ratings = run_elo(_fights(), stats, PARAMS)
    f1 = ratings[ratings["fight_id"] == "f1"].set_index("corner")
    assert f1.loc["a", "post_striking"] == pytest.approx(1500.0)  # share 0
    assert f1.loc["a", "post_grappling"] == pytest.approx(1520.0)


def test_chronological_regardless_of_input_order():
    ratings_sorted = run_elo(_fights(), _stats(), PARAMS)
    shuffled = _fights().sample(frac=1, random_state=7)
    ratings_shuffled = run_elo(shuffled, _stats(), PARAMS)
    merged = ratings_sorted.merge(
        ratings_shuffled, on=["fight_id", "corner"], suffixes=("_1", "_2")
    )
    assert len(merged) == len(ratings_sorted)
    assert (merged["post_overall_1"] == merged["post_overall_2"]).all()
