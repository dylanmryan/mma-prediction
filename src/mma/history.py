"""Chronological per-fighter career-stat accumulator.

Mirrors the Elo engine's single-pass design: for every rated fight it
emits each fighter's PRE-fight career/rolling stats, then folds the
fight into their state. Point-in-time correct by construction.

Missing per-fight stats accumulate as zero (mostly pre-2001 fights);
rates are NaN until a fighter has the relevant denominator.

Three groups of fields exist for feature blocks beyond `base` and are
computed unconditionally, the way every base rate is: nothing here is
block-aware, and a block that is not enabled simply never reads them.

  * `bonus_rate` (block `context`) -- the share of a fighter's prior bouts
    that they won AND that carried a post-fight bonus.
  * `elo_delta_3` / `elo_delta_5` / `elo_peak_minus_current` /
    `years_since_ufc_debut` (block `trajectory`) -- rating DYNAMICS, which a
    level-only Elo column cannot express.
  * the five `*_vs_exp` means plus `avg_opp_elo_wins` / `avg_opp_elo_losses`
    (block `opponent_adjusted`) -- each core rate minus what that fight's
    opponent had historically ALLOWED before it, averaged over a career.

The `*_vs_exp` group is the only one that needs a second fighter's state, and
it is what `allowed` exists for: when X fights O, the pair recorded for X is
(X's own value in this fight, O's PRE-fight allowed value). Both corners'
`allowed` snapshots are therefore taken before either corner is updated --
otherwise the second corner would price itself against an opponent who had
already absorbed the shared fight.
"""
from __future__ import annotations

from collections import deque

import pandas as pd

from mma import context

_HISTORY_WINDOW = 5
# `elo_delta_3` and `elo_delta_5` are suffix sums of the same deque.
_MOMENTUM_WINDOWS = (3, 5)

# The five rates the `opponent_adjusted` block prices against expectation.
# Each maps to the mirror-image rate of the opponent: what a fighter LANDS is
# measured against what the opponent has ALLOWED to be landed on them, what a
# fighter ABSORBS against what the opponent has historically landed, and so
# on. `_FighterState.allowed` is the one place that mirroring is written down.
VS_EXPECTATION_RATES = (
    "sig_pm", "sig_absorbed_pm", "td_landed_pf", "td_def", "ctrl_share",
)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


class _FighterState:
    def __init__(self) -> None:
        self.fights = 0
        self.wins = 0.0
        self.finish_wins = 0
        self.bonus_wins = 0
        self.kd = 0.0
        self.sub_att = 0.0
        self.td_landed = 0.0
        self.td_attempted = 0.0
        self.opp_td_landed = 0.0
        self.opp_td_attempted = 0.0
        self.sig_landed = 0.0
        self.sig_absorbed = 0.0
        self.ctrl_sec = 0.0
        self.opp_ctrl_sec = 0.0
        self.time_sec = 0.0
        self.streak = 0
        self.last_date: pd.Timestamp | None = None
        self.first_date: pd.Timestamp | None = None
        self.elo_current: float | None = None
        self.elo_peak: float | None = None
        self.recent_results: deque[float] = deque(maxlen=_HISTORY_WINDOW)
        self.recent_opp_elo: deque[float] = deque(maxlen=_HISTORY_WINDOW)
        self.recent_elo_deltas: deque[float] = deque(maxlen=max(_MOMENTUM_WINDOWS))
        self.vs_exp_total = {rate: 0.0 for rate in VS_EXPECTATION_RATES}
        self.vs_exp_count = {rate: 0 for rate in VS_EXPECTATION_RATES}
        self.opp_elo_beaten = 0.0
        self.n_beaten = 0
        self.opp_elo_lost_to = 0.0
        self.n_lost_to = 0

    def allowed(self) -> dict:
        """What this fighter's opponents have historically achieved on them.

        The expectation side of the `opponent_adjusted` pairs, one entry per
        rate in `VS_EXPECTATION_RATES` and keyed by the rate it prices. Each
        is the mirror of that rate: `sig_pm`'s expectation is how many
        significant strikes per minute this fighter lets through
        (`sig_absorbed_pm`), `sig_absorbed_pm`'s is how many they land,
        `td_def`'s is how often their own takedowns get through (1 - td_acc),
        and so on. A fighter with no prior fight has None everywhere, which is
        what makes a debutant opponent contribute no pair at all rather than a
        pair against a made-up baseline.
        """
        minutes = self.time_sec / 60.0
        return {
            "sig_pm": _ratio(self.sig_absorbed, minutes),
            "sig_absorbed_pm": _ratio(self.sig_landed, minutes),
            "td_landed_pf": _ratio(self.opp_td_landed, self.fights),
            # `td_def`'s mirror is the opponent's takedown ACCURACY, not
            # `1 - accuracy`: the pair asks how much of the takedown exchange
            # a fighter won against an opponent who lands `td_acc` of what
            # they shoot, so a fighter who stuffs 0.8 against a 0.5 shooter
            # scores +0.3. (Pinned by the SP2 measurement --
            # `models/walkforward/xgb_opponent_adjusted.json` reproduces
            # fold for fold on this definition and on no other tried.)
            "td_def": _ratio(self.td_landed, self.td_attempted),
            "ctrl_share": _ratio(self.opp_ctrl_sec, self.time_sec),
        }

    def snapshot(self, date: pd.Timestamp) -> dict:
        ratio = _ratio
        minutes = self.time_sec / 60.0
        deltas = list(self.recent_elo_deltas)
        state = {
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
            # --- block `context` ---
            "bonus_rate": ratio(self.bonus_wins, self.fights),
            # --- block `trajectory` ---
            # Momentum is a SUM of movements, so a debutant's is 0.0 rather
            # than NaN: no movement is a fact about them, not a gap.
            "elo_delta_3": float(sum(deltas[-_MOMENTUM_WINDOWS[0]:])),
            "elo_delta_5": float(sum(deltas[-_MOMENTUM_WINDOWS[1]:])),
            "elo_peak_minus_current": (
                self.elo_peak - self.elo_current
                if self.elo_peak is not None and self.elo_current is not None
                else None
            ),
            "years_since_ufc_debut": (
                (date - self.first_date).days / 365.25
                if self.first_date is not None and date is not None
                else None
            ),
            # --- block `opponent_adjusted` ---
            "avg_opp_elo_wins": ratio(self.opp_elo_beaten, self.n_beaten),
            "avg_opp_elo_losses": ratio(self.opp_elo_lost_to, self.n_lost_to),
        }
        for rate in VS_EXPECTATION_RATES:
            state[f"{rate}_vs_exp"] = ratio(
                self.vs_exp_total[rate], self.vs_exp_count[rate]
            )
        return state

    def update(self, score, own, opp, method, time_sec, date, opp_elo,
               opp_allowed=None, pre_elo=None, post_elo=None,
               bonus: bool = False) -> None:
        def num(mapping, key):
            value = mapping.get(key)
            return 0.0 if value is None or pd.isna(value) else float(value)

        def raw(mapping, key):
            """A per-fight stat as a float, or None when it was not recorded.

            The career counters take the `num` path, where a missing stat
            accumulates as zero; the versus-expectation pairs must not. A
            fight with no recorded statistics (660 corner rows, all pre-2001)
            would otherwise enter every mean as "landed nothing, controlled
            nothing", which is a measurement of the source rather than of the
            fighter -- so the pair is skipped instead."""
            value = mapping.get(key)
            return None if value is None or pd.isna(value) else float(value)

        seconds = (
            float(time_sec)
            if time_sec is not None and not pd.isna(time_sec) and float(time_sec) > 0
            else None
        )
        # This fight's own value for each priced rate, computed before any
        # counter moves so it describes the fight and not the career.
        opp_td_attempted = raw(opp, "td_attempted")
        own_sig, opp_sig = raw(own, "sig_landed"), raw(opp, "sig_landed")
        own_ctrl = raw(own, "ctrl_sec")
        own_rates = {
            "sig_pm": own_sig / (seconds / 60.0) if seconds and own_sig is not None else None,
            "sig_absorbed_pm": opp_sig / (seconds / 60.0) if seconds and opp_sig is not None else None,
            "td_landed_pf": raw(own, "td_landed"),
            "td_def": (
                1 - raw(opp, "td_landed") / opp_td_attempted
                if opp_td_attempted and raw(opp, "td_landed") is not None
                else None
            ),
            "ctrl_share": own_ctrl / seconds if seconds and own_ctrl is not None else None,
        }
        for rate in VS_EXPECTATION_RATES:
            expected = (opp_allowed or {}).get(rate)
            actual = own_rates[rate]
            if expected is None or actual is None:
                continue
            self.vs_exp_total[rate] += actual - expected
            self.vs_exp_count[rate] += 1

        self.fights += 1
        self.wins += score
        if score == 1.0 and method in ("ko_tko", "submission"):
            self.finish_wins += 1
        if score == 1.0 and bonus:
            self.bonus_wins += 1
        self.kd += num(own, "kd")
        self.sub_att += num(own, "sub_att")
        self.td_landed += num(own, "td_landed")
        self.td_attempted += num(own, "td_attempted")
        self.opp_td_landed += num(opp, "td_landed")
        self.opp_td_attempted += num(opp, "td_attempted")
        self.sig_landed += num(own, "sig_landed")
        self.sig_absorbed += num(opp, "sig_landed")
        self.ctrl_sec += num(own, "ctrl_sec")
        self.opp_ctrl_sec += num(opp, "ctrl_sec")
        if time_sec is not None and not pd.isna(time_sec):
            self.time_sec += float(time_sec)
        if score == 1.0:
            self.streak = self.streak + 1 if self.streak > 0 else 1
        elif score == 0.0:
            self.streak = self.streak - 1 if self.streak < 0 else -1
        else:
            self.streak = 0
        if self.first_date is None:
            self.first_date = date
        self.last_date = date
        self.recent_results.append(score)
        if opp_elo is not None and not pd.isna(opp_elo):
            self.recent_opp_elo.append(float(opp_elo))
            if score == 1.0:
                self.opp_elo_beaten += float(opp_elo)
                self.n_beaten += 1
            elif score == 0.0:
                self.opp_elo_lost_to += float(opp_elo)
                self.n_lost_to += 1
        if (
            pre_elo is not None and post_elo is not None
            and not pd.isna(pre_elo) and not pd.isna(post_elo)
        ):
            self.recent_elo_deltas.append(float(post_elo) - float(pre_elo))
        if post_elo is not None and not pd.isna(post_elo):
            self.elo_current = float(post_elo)
            self.elo_peak = (
                self.elo_current if self.elo_peak is None
                else max(self.elo_peak, self.elo_current)
            )


_SCORES = {"a": (1.0, 0.0), "b": (0.0, 1.0), "draw": (0.5, 0.5)}


def elo_lookups(ratings: pd.DataFrame) -> tuple[dict, dict]:
    """(pre, post) overall-Elo lookups keyed by (fight_id, corner).

    `post_overall` is optional so a fixture that only carries the pre-fight
    column still builds: without it the momentum and peak fields stay empty
    rather than raising, exactly as a fighter with no rated fight does.
    """
    keyed = ratings.set_index(["fight_id", "corner"])
    pre = keyed["pre_overall"].to_dict()
    post = keyed["post_overall"].to_dict() if "post_overall" in keyed.columns else {}
    return pre, post


def build_history(
    fights: pd.DataFrame, stats: pd.DataFrame, ratings: pd.DataFrame,
    bonuses: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per fighter per rated fight with PRE-fight career stats."""
    stat_lookup = stats.set_index(["fight_id", "corner"]).to_dict("index")
    elo_lookup, post_lookup = elo_lookups(ratings)
    bonus_ids = context.bonus_fights(bonuses)

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
        bonus = str(fight.fight_id) in bonus_ids
        # Both corners' expectation state is read before either is updated.
        allowed = {
            corner: states[fighter_id].allowed()
            for corner, fighter_id in (("a", fight.fighter_a_id),
                                       ("b", fight.fighter_b_id))
        }
        for corner, fighter_id, score, own, opp, opp_corner in (
            ("a", fight.fighter_a_id, score_a, stats_a, stats_b, "b"),
            ("b", fight.fighter_b_id, score_b, stats_b, stats_a, "a"),
        ):
            states[fighter_id].update(
                score, own, opp, method, fight.duration_sec, fight.date,
                elo_lookup.get((fight.fight_id, opp_corner)),
                opp_allowed=allowed[opp_corner],
                pre_elo=elo_lookup.get((fight.fight_id, corner)),
                post_elo=post_lookup.get((fight.fight_id, corner)),
                bonus=bonus,
            )

    history = pd.DataFrame(rows)
    for column in ("fight_id", "corner", "fighter_id"):
        history[column] = history[column].astype("string")
    return history
