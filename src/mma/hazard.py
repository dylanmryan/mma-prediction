"""Training rows for the round-by-round fight simulator (SP3).

Two builders turn the committed feature table into the two frames the
simulator's members are fitted on:

* `build_hazard_rows` — one row per (fight, round actually fought), carrying
  the fight's feature columns plus `round_no` and a 5-class `hazard_label`.
  Every round the fight survived is a `survive` row; the round it ended in
  (if it ended by KO/TKO or submission) carries the finishing label. Rounds
  that were never reached are absent, which is exactly what makes the
  censoring correct: a first-round knockout is one observation of "round 1
  ended in a KO", not three observations of a three-round fight.
* `build_decision_rows` — one row per fight that went to the cards, labelled
  by which corner won it.

**Corner orientation.** `mma.features` swaps the two corners by
`md5(fight_id)` parity, so the feature table's corner A is the fights
table's corner *b* whenever `swapped` is True. Both builders label in the
feature table's frame -- `a_ko` means "the corner whose columns are the
`*_a` / positive side of every differential won by KO" -- and derive that
from the feature row's own `y_winner`, which `mma.features` already
expresses in the swapped frame. Labelling from `fights["winner"]` instead
would silently invert half the method predictions.

**Rounds fought is the authority for how many rows a fight emits.** The
number of rows comes from `finish_round` for a finish and from
`scheduled_rounds` for everything that reached the end of the bout;
`scheduled_rounds` is otherwise only a feature and a bound for the
simulator. So the 45 fights with no recorded `scheduled_rounds` are *kept*
whenever the rounds they actually fought are known (all 45 are finishes,
so they are), with `scheduled_rounds` left NA for the model to see as
missing; and the two fights recorded as ending in round 6 emit six rows
rather than being clamped to five. Clamping would delete a real
observation and, worse, relabel it: round 5 of those fights genuinely
survived. Only a fight whose rounds fought cannot be determined at all
(no `finish_round` *and* no `scheduled_rounds`) is dropped; there are none
in the current data.

Both functions are pure: frames in, a new frame out, inputs untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Fixed order of the 5-class hazard target. `mma.simulator` indexes the
#: per-round probability vector with exactly this order, so it is part of
#: the contract between the two modules.
HAZARD_CLASSES = ("a_ko", "a_sub", "b_ko", "b_sub", "survive")

SURVIVE = "survive"

_FINISH_METHODS = ("ko_tko", "submission")
_SUFFIX = {"ko_tko": "ko", "submission": "sub"}


def _rounds_fought(features: pd.DataFrame, fights: pd.DataFrame) -> pd.Series:
    """Rounds actually contested, per feature row: finish round, else scheduled.

    `mma.dataset.build_fights` sets `finish_round` only for finishes, so a
    non-finish (decision, DQ, no-contest that still has a winner) reached the
    end of the bout and contributed `scheduled_rounds` rounds.
    """
    lookup = fights.drop_duplicates("fight_id").set_index("fight_id")
    keys = features["fight_id"]
    finish = pd.to_numeric(
        lookup["finish_round"].reindex(keys).to_numpy(), errors="coerce"
    )
    if "scheduled_rounds" in features.columns:
        scheduled = pd.to_numeric(features["scheduled_rounds"], errors="coerce")
    else:
        scheduled = pd.Series(
            pd.to_numeric(
                lookup["scheduled_rounds"].reindex(keys).to_numpy(), errors="coerce"
            )
        )
    rounds = pd.Series(np.asarray(finish, dtype=float), index=features.index)
    rounds = rounds.where(rounds.notna(), np.asarray(scheduled, dtype=float))
    return rounds


def build_hazard_rows(features: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """One row per (fight, round fought) with a 5-class `hazard_label`.

    The output carries every column of `features` plus `round_no` (1-based)
    and `hazard_label` in `HAZARD_CLASSES`. `scheduled_rounds` is carried
    through from `features` when present and joined from `fights` otherwise,
    so it is always available as the simulator's per-fight round bound.
    """
    rounds = _rounds_fought(features, fights)
    known = rounds.notna() & (rounds >= 1)
    base = features.loc[known.to_numpy()].reset_index(drop=True)
    counts = rounds.loc[known.to_numpy()].to_numpy().astype(int)

    if "scheduled_rounds" not in base.columns:
        lookup = fights.drop_duplicates("fight_id").set_index("fight_id")
        base = base.assign(
            scheduled_rounds=lookup["scheduled_rounds"]
            .reindex(base["fight_id"])
            .to_numpy()
        )

    repeat = np.repeat(np.arange(len(base)), counts)
    out = base.iloc[repeat].reset_index(drop=True)
    # 1..count for each fight, without a Python loop
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    out["round_no"] = np.arange(len(out), dtype=int) - starts + 1

    method = base["y_method"].astype("string").fillna("").to_numpy(dtype=object)
    is_finish = base["y_method"].isin(_FINISH_METHODS).to_numpy(dtype=bool)
    corner = np.where(base["y_winner"].to_numpy() == 1, "a", "b")
    finish_label = np.array(
        [
            f"{c}_{_SUFFIX[m]}" if f else SURVIVE
            for c, m, f in zip(corner, method, is_finish)
        ],
        dtype=object,
    )
    last_round = np.repeat(counts, counts)
    label = np.where(
        out["round_no"].to_numpy() == last_round,
        np.repeat(finish_label, counts),
        SURVIVE,
    )
    out["hazard_label"] = pd.Series(label, dtype="string")
    return out


def build_decision_rows(features: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """One row per fight that went to the cards.

    `decision_label` is 1 when the *feature table's* corner A won on the
    scorecards and 0 when its corner B did -- the same frame and the same
    convention as `y_winner`. `fights` is accepted, and unused, so the two
    builders share one signature at the call site.
    """
    keep = features["y_method"].astype("string").eq("decision").fillna(False)
    rows = features.loc[keep.to_numpy(dtype=bool)].reset_index(drop=True)
    return rows.assign(
        decision_label=rows["y_winner"].to_numpy().astype(int)
    )
