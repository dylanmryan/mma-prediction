"""Chronological per-fighter career-stat accumulator.

Mirrors the Elo engine's single-pass design: for every rated fight it
emits each fighter's PRE-fight career/rolling stats, then folds the
fight into their state. Point-in-time correct by construction.

Missing per-fight stats accumulate as zero (mostly pre-2001 fights);
rates are NaN until a fighter has the relevant denominator.
"""
from __future__ import annotations

from collections import deque

import pandas as pd

_HISTORY_WINDOW = 5


class _FighterState:
    def __init__(self) -> None:
        self.fights = 0
        self.wins = 0.0
        self.finish_wins = 0
        self.kd = 0.0
        self.sub_att = 0.0
        self.td_landed = 0.0
        self.td_attempted = 0.0
        self.opp_td_landed = 0.0
        self.opp_td_attempted = 0.0
        self.sig_landed = 0.0
        self.sig_absorbed = 0.0
        self.ctrl_sec = 0.0
        self.time_sec = 0.0
        self.streak = 0
        self.last_date: pd.Timestamp | None = None
        self.recent_results: deque[float] = deque(maxlen=_HISTORY_WINDOW)
        self.recent_opp_elo: deque[float] = deque(maxlen=_HISTORY_WINDOW)

    def snapshot(self, date: pd.Timestamp) -> dict:
        def ratio(num, den):
            return num / den if den else None

        minutes = self.time_sec / 60.0
        return {
            "career_fights": self.fights,
            "career_wins": self.wins,
            "career_win_rate": ratio(self.wins, self.fights),
            "career_finish_rate": ratio(self.finish_wins, self.wins),
            "kd_pf": ratio(self.kd, self.fights),
            "sub_att_pf": ratio(self.sub_att, self.fights),
            "td_landed_pf": ratio(self.td_landed, self.fights),
            "td_acc": ratio(self.td_landed, self.td_attempted),
            "td_def": (
                1 - self.opp_td_landed / self.opp_td_attempted
                if self.opp_td_attempted
                else None
            ),
            "sig_pm": ratio(self.sig_landed, minutes),
            "sig_absorbed_pm": ratio(self.sig_absorbed, minutes),
            "ctrl_share": ratio(self.ctrl_sec, self.time_sec),
            "streak": self.streak,
            "days_since_last": (
                (date - self.last_date).days if self.last_date is not None else None
            ),
            "last5_win_rate": (
                sum(self.recent_results) / len(self.recent_results)
                if self.recent_results
                else None
            ),
            "last5_avg_opp_elo": (
                sum(self.recent_opp_elo) / len(self.recent_opp_elo)
                if self.recent_opp_elo
                else None
            ),
        }

    def update(self, score, own, opp, method, time_sec, date, opp_elo) -> None:
        def num(mapping, key):
            value = mapping.get(key)
            return 0.0 if value is None or pd.isna(value) else float(value)

        self.fights += 1
        self.wins += score
        if score == 1.0 and method in ("ko_tko", "submission"):
            self.finish_wins += 1
        self.kd += num(own, "kd")
        self.sub_att += num(own, "sub_att")
        self.td_landed += num(own, "td_landed")
        self.td_attempted += num(own, "td_attempted")
        self.opp_td_landed += num(opp, "td_landed")
        self.opp_td_attempted += num(opp, "td_attempted")
        self.sig_landed += num(own, "sig_landed")
        self.sig_absorbed += num(opp, "sig_landed")
        self.ctrl_sec += num(own, "ctrl_sec")
        if time_sec is not None and not pd.isna(time_sec):
            self.time_sec += float(time_sec)
        if score == 1.0:
            self.streak = self.streak + 1 if self.streak > 0 else 1
        elif score == 0.0:
            self.streak = self.streak - 1 if self.streak < 0 else -1
        else:
            self.streak = 0
        self.last_date = date
        self.recent_results.append(score)
        if opp_elo is not None and not pd.isna(opp_elo):
            self.recent_opp_elo.append(float(opp_elo))


_SCORES = {"a": (1.0, 0.0), "b": (0.0, 1.0), "draw": (0.5, 0.5)}


def build_history(
    fights: pd.DataFrame, stats: pd.DataFrame, ratings: pd.DataFrame
) -> pd.DataFrame:
    """One row per fighter per rated fight with PRE-fight career stats."""
    stat_lookup = stats.set_index(["fight_id", "corner"]).to_dict("index")
    elo_lookup = ratings.set_index(["fight_id", "corner"])["pre_overall"].to_dict()

    states: dict[str, _FighterState] = {}
    rows = []
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for fight in ordered.itertuples(index=False):
        if fight.winner not in _SCORES:
            continue
        score_a, score_b = _SCORES[fight.winner]
        stats_a = stat_lookup.get((fight.fight_id, "a"), {})
        stats_b = stat_lookup.get((fight.fight_id, "b"), {})
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            state = states.setdefault(fighter_id, _FighterState())
            row = {
                "fight_id": fight.fight_id,
                "corner": corner,
                "fighter_id": fighter_id,
            }
            row.update(state.snapshot(fight.date))
            rows.append(row)
        # update AFTER both snapshots so neither side sees this fight
        method = fight.method if pd.notna(fight.method) else None
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            states[fighter_id].update(
                score, own, opp, method, fight.match_time_sec, fight.date,
                elo_lookup.get((fight.fight_id, opp_corner)),
            )

    history = pd.DataFrame(rows)
    for column in ("fight_id", "corner", "fighter_id"):
        history[column] = history[column].astype("string")
    return history
