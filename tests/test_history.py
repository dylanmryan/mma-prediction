import pandas as pd
import pytest

from mma.history import build_history


def _fights():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f2"],
            "date": pd.to_datetime(["2020-01-01", "2020-03-01"]),
            "fighter_a_id": ["x", "x"],
            "fighter_b_id": ["y", "z"],
            "winner": ["a", "a"],
            "method": ["ko_tko", "decision"],
            "duration_sec": [300.0, 900.0],
        }
    )


def _stats():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "kd": [2, 0, 0, 0],
            "sig_landed": [30, 10, 50, 40],
            "td_landed": [1, 0, 2, 1],
            "td_attempted": [2, 1, 4, 2],
            "sub_att": [0, 1, 1, 0],
            "ctrl_sec": [60.0, 30.0, 300.0, 100.0],
        }
    )


def _ratings():
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1", "f2", "f2"],
            "corner": ["a", "b", "a", "b"],
            "fighter_id": ["x", "y", "x", "z"],
            "pre_overall": [1500.0, 1500.0, 1532.0, 1500.0],
        }
    )


def test_debut_row_is_empty_history():
    history = build_history(_fights(), _stats(), _ratings())
    x_f1 = history[(history["fight_id"] == "f1") & (history["corner"] == "a")].iloc[0]
    assert x_f1["career_fights"] == 0
    assert x_f1["streak"] == 0
    assert pd.isna(x_f1["career_win_rate"])
    assert pd.isna(x_f1["days_since_last"])
    assert pd.isna(x_f1["last5_avg_opp_elo"])


def test_second_fight_reflects_first():
    history = build_history(_fights(), _stats(), _ratings())
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_fights"] == 1
    assert x_f2["career_win_rate"] == 1.0
    assert x_f2["career_finish_rate"] == 1.0  # won by ko
    assert x_f2["streak"] == 1
    assert x_f2["days_since_last"] == 60
    assert x_f2["kd_pf"] == 2.0
    assert x_f2["td_acc"] == pytest.approx(0.5)          # 1 of 2
    assert x_f2["td_def"] == pytest.approx(1.0)          # opponent 0 of 1
    assert x_f2["sig_pm"] == pytest.approx(30 / 5.0)     # 30 landed in 5 min
    assert x_f2["sig_absorbed_pm"] == pytest.approx(10 / 5.0)
    assert x_f2["ctrl_share"] == pytest.approx(60 / 300.0)
    assert x_f2["last5_win_rate"] == 1.0
    assert x_f2["last5_avg_opp_elo"] == 1500.0


def test_loss_and_draw_semantics():
    fights = _fights()
    fights.loc[0, "winner"] = "b"     # x loses f1
    fights.loc[1, "winner"] = "draw"  # then draws f2 (still emitted to history)
    history = build_history(fights, _stats(), _ratings())
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_win_rate"] == 0.0
    assert x_f2["streak"] == -1


def test_nc_ignored():
    fights = _fights()
    fights.loc[0, "winner"] = "nc"
    history = build_history(fights, _stats(), _ratings())
    assert "f1" not in set(history["fight_id"])
    x_f2 = history[(history["fight_id"] == "f2") & (history["corner"] == "a")].iloc[0]
    assert x_f2["career_fights"] == 0


# --- block `trajectory`: rating dynamics --------------------------------------


def _momentum_fights(n: int = 5) -> pd.DataFrame:
    """`n` chronological wins for x over fresh opponents, one a month."""
    return pd.DataFrame({
        "fight_id": [f"m{i}" for i in range(1, n + 1)],
        "date": pd.to_datetime([f"2020-{i:02d}-01" for i in range(1, n + 1)]),
        "fighter_a_id": ["x"] * n,
        "fighter_b_id": [f"o{i}" for i in range(1, n + 1)],
        "winner": ["a"] * n,
        "method": ["decision"] * n,
        "duration_sec": [900.0] * n,
    })


def _momentum_stats(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "fight_id": [f"m{i}" for i in range(1, n + 1) for _ in ("a", "b")],
        "corner": ["a", "b"] * n,
        "kd": [0] * 2 * n,
        "sig_landed": [30, 10] * n,
        "td_landed": [1, 0] * n,
        "td_attempted": [2, 1] * n,
        "sub_att": [0, 0] * n,
        "ctrl_sec": [60.0, 30.0] * n,
    })


def _momentum_ratings(deltas, n: int = 5) -> pd.DataFrame:
    """x's pre/post Elo walks by `deltas`; every opponent stays at 1500."""
    pre = 1500.0
    rows = []
    for i, delta in enumerate(deltas, start=1):
        rows.append({"fight_id": f"m{i}", "corner": "a", "fighter_id": "x",
                     "pre_overall": pre, "post_overall": pre + delta})
        rows.append({"fight_id": f"m{i}", "corner": "b", "fighter_id": f"o{i}",
                     "pre_overall": 1500.0, "post_overall": 1500.0 - delta})
        pre += delta
    return pd.DataFrame(rows)


def test_momentum_is_zero_for_a_debutant_and_sums_the_last_three_moves():
    """`elo_delta_3` is a SUM of movements, so a fighter with none has 0.0.

    NaN would be wrong here: "has not moved" is a fact about a debutant, not
    a gap in the data, and the block's coverage (1.000) depends on it.
    """
    deltas = [10.0, -4.0, 6.0, -2.0, 8.0]
    history = build_history(
        _momentum_fights(), _momentum_stats(), _momentum_ratings(deltas)
    ).set_index(["fight_id", "corner"])
    assert history.loc[("m1", "a"), "elo_delta_3"] == 0.0
    assert history.loc[("m1", "a"), "elo_delta_5"] == 0.0
    # at m5, x has moved four times: the last three are -4, +6, -2
    assert history.loc[("m5", "a"), "elo_delta_3"] == pytest.approx(0.0)
    assert history.loc[("m5", "a"), "elo_delta_5"] == pytest.approx(10.0)
    # at m4 the last three are +10, -4, +6 -- a genuinely different window
    assert history.loc[("m4", "a"), "elo_delta_3"] == pytest.approx(12.0)


def test_peak_minus_current_is_never_negative_and_measures_the_drawdown():
    deltas = [10.0, -4.0, 6.0, -2.0, 8.0]
    history = build_history(
        _momentum_fights(), _momentum_stats(), _momentum_ratings(deltas)
    ).set_index(["fight_id", "corner"])
    x = history.xs("a", level="corner")["elo_peak_minus_current"]
    assert pd.isna(x.loc["m1"])                      # no rated fight yet
    assert x.loc["m2"] == pytest.approx(0.0)         # 1510 is both peak and current
    assert x.loc["m3"] == pytest.approx(4.0)         # peak 1510, current 1506
    assert (x.dropna() >= 0).all()


def test_years_since_ufc_debut_is_nan_on_the_debut_and_grows_after():
    history = build_history(
        _momentum_fights(), _momentum_stats(),
        _momentum_ratings([1.0] * 5),
    ).set_index(["fight_id", "corner"])
    x = history.xs("a", level="corner")["years_since_ufc_debut"]
    assert pd.isna(x.loc["m1"])
    assert x.loc["m2"] == pytest.approx(31 / 365.25)
    assert x.loc["m5"] > x.loc["m3"] > x.loc["m2"]


# --- block `context`: the bonus record ----------------------------------------


def test_bonus_rate_counts_bonus_WINS_over_prior_bouts():
    """A bonus is recorded per FIGHT, so crediting the winner is the rule --
    a fighter who lost a Fight-of-the-Night gets nothing from it."""
    bonuses = pd.DataFrame({"fight_id": ["f1"]})
    history = build_history(
        _fights(), _stats(), _ratings(), bonuses
    ).set_index(["fight_id", "corner"])
    # x won f1 and f1 carried a bonus, so at f2 x is 1 for 1
    assert history.loc[("f2", "a"), "bonus_rate"] == pytest.approx(1.0)
    assert pd.isna(history.loc[("f1", "a"), "bonus_rate"])   # no prior bouts
    # the same fight LOST gives the corner nothing, even though the bonus is
    # recorded against the fight and both corners were in it
    lost = _fights()
    lost.loc[0, "winner"] = "b"
    losing = build_history(lost, _stats(), _ratings(), bonuses)
    assert losing.set_index(["fight_id", "corner"]).loc[("f2", "a"), "bonus_rate"] == 0.0


# --- block `opponent_adjusted`: versus-expectation rates -----------------------


def _vs_exp_fights() -> pd.DataFrame:
    """Three chronological fights: o1 builds a record, then x meets o1, then x
    fights again so the mean computed from that single pair is observable."""
    return pd.DataFrame({
        "fight_id": ["f1", "f2", "f3"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "fighter_a_id": ["o1", "x", "x"],
        "fighter_b_id": ["o2", "o1", "o2"],
        "winner": ["a", "a", "a"],
        "method": ["decision", "decision", "decision"],
        "duration_sec": [600.0, 300.0, 300.0],
    })


def _vs_exp_stats() -> pd.DataFrame:
    return pd.DataFrame({
        "fight_id": ["f1", "f1", "f2", "f2", "f3", "f3"],
        "corner": ["a", "b", "a", "b", "a", "b"],
        "kd": [0, 0, 0, 0, 0, 0],
        #        o1  o2   x  o1   x  o2
        "sig_landed": [20, 50, 40, 10, 5, 5],
        "td_landed": [1, 3, 2, 1, 0, 0],
        "td_attempted": [2, 6, 4, 2, 0, 0],
        "sub_att": [0, 0, 0, 0, 0, 0],
        "ctrl_sec": [60.0, 120.0, 90.0, 30.0, 0.0, 0.0],
    })


def _vs_exp_ratings() -> pd.DataFrame:
    return pd.DataFrame({
        "fight_id": ["f1", "f1", "f2", "f2", "f3", "f3"],
        "corner": ["a", "b", "a", "b", "a", "b"],
        "fighter_id": ["o1", "o2", "x", "o1", "x", "o2"],
        "pre_overall": [1500.0, 1500.0, 1500.0, 1520.0, 1520.0, 1480.0],
        "post_overall": [1520.0, 1480.0, 1520.0, 1500.0, 1540.0, 1460.0],
    })


def test_a_debutant_has_no_versus_expectation_values():
    history = build_history(
        _vs_exp_fights(), _vs_exp_stats(), _vs_exp_ratings()
    ).set_index(["fight_id", "corner"])
    row = history.loc[("f2", "a")]  # x's debut
    for column in ("sig_pm_vs_exp", "sig_absorbed_pm_vs_exp", "td_landed_pf_vs_exp",
                   "td_def_vs_exp", "ctrl_share_vs_exp",
                   "avg_opp_elo_wins", "avg_opp_elo_losses"):
        assert pd.isna(row[column]), column


def test_an_opponent_with_no_history_contributes_no_pair():
    """o1's f1 opponent (o2) was a debutant, so o1 has no expectation to be
    priced against and stays NaN at f2 despite having fought."""
    history = build_history(
        _vs_exp_fights(), _vs_exp_stats(), _vs_exp_ratings()
    ).set_index(["fight_id", "corner"])
    assert history.loc[("f2", "a"), "career_fights"] == 0      # x
    assert history.loc[("f2", "b"), "career_fights"] == 1      # o1 has fought
    assert pd.isna(history.loc[("f2", "b"), "sig_pm_vs_exp"])


def test_versus_expectation_uses_the_opponents_value_from_before_the_fight():
    """Every number here is hand-checked off the fixture.

    After f1, o1 has absorbed 50 significant strikes in 10 minutes, landed 20,
    been taken down 3 times in one fight, hit 1 of 2 takedowns, and been
    controlled for 120 of 600 seconds. Those are the values x is priced
    against in f2 -- NOT o1's values including f2, which would make
    `sig_pm_vs_exp` 8.0 - 6.0 = 2.0 instead of the 3.0 asserted below.
    """
    history = build_history(
        _vs_exp_fights(), _vs_exp_stats(), _vs_exp_ratings()
    ).set_index(["fight_id", "corner"])
    x = history.loc[("f3", "a")]
    # x landed 40 in 5 min = 8.0/min; o1 allowed 50 in 10 min = 5.0/min
    assert x["sig_pm_vs_exp"] == pytest.approx(3.0)
    # x absorbed 10 in 5 min = 2.0/min; o1 lands 20 in 10 min = 2.0/min
    assert x["sig_absorbed_pm_vs_exp"] == pytest.approx(0.0)
    # x landed 2 takedowns; o1 had allowed 3 per fight
    assert x["td_landed_pf_vs_exp"] == pytest.approx(-1.0)
    # x stuffed 1 of o1's 2 shots (td_def 0.5); o1 lands 0.5 of what it shoots
    assert x["td_def_vs_exp"] == pytest.approx(0.0)
    # x controlled 90 of 300 s = 0.30; o1 had allowed 120 of 600 = 0.20
    assert x["ctrl_share_vs_exp"] == pytest.approx(0.10)


def test_a_fighter_who_outperforms_a_leaky_opponent_scores_positive():
    """The direction of the whole block: more output than the opponent
    usually gives up is a positive number."""
    stats = _vs_exp_stats()
    stats.loc[2, "sig_landed"] = 100  # x lands far more than o1 usually allows
    history = build_history(
        _vs_exp_fights(), stats, _vs_exp_ratings()
    ).set_index(["fight_id", "corner"])
    assert history.loc[("f3", "a"), "sig_pm_vs_exp"] > 0


def test_average_opponent_elo_splits_wins_from_losses():
    fights = _vs_exp_fights()
    fights.loc[1, "winner"] = "b"          # x loses f2 to o1 (pre-fight Elo 1520)
    history = build_history(
        fights, _vs_exp_stats(), _vs_exp_ratings()
    ).set_index(["fight_id", "corner"])
    x = history.loc[("f3", "a")]
    assert pd.isna(x["avg_opp_elo_wins"])
    assert x["avg_opp_elo_losses"] == pytest.approx(1520.0)


def test_a_fight_with_no_recorded_statistics_contributes_no_pair():
    """Missing per-fight stats accumulate as ZERO in the career counters (the
    module's documented rule), but a versus-expectation pair built on them
    would say the fighter landed nothing and controlled nothing -- a
    measurement of the source, not of the fighter. The pair is skipped."""
    stats = _vs_exp_stats()
    stats.loc[[2, 3], ["sig_landed", "td_landed", "td_attempted", "ctrl_sec"]] = None
    history = build_history(
        _vs_exp_fights(), stats, _vs_exp_ratings()
    ).set_index(["fight_id", "corner"])
    assert pd.isna(history.loc[("f3", "a"), "sig_pm_vs_exp"])
