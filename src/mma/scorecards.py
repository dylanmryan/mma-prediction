"""The judges' scorecards, read out of the scrape's free-text `details` field.

`data/raw/fight.csv` has carried these since the first scrape and nothing has
ever read them. A decision row's `details` looks like::

    David Therien 28 - 29. Greg Jackson 29 - 28. Nelson Hamilton 27 - 30.

Three named judges and their totals. 4,461 of 4,995 decisions parse cleanly,
across 573 distinct judges. This module is the only place that field is read.

**Why every number here is frame-invariant, and what that buys.** The two
scores per judge are *not* ordered by `r_id`/`b_id`. Tested against the
recorded winner, "the first score is the red corner" holds on 37.4% of
unanimous decisions -- which is not "right" and not "flipped" but
*inconsistent*, so no fixed rule recovers the side. This repo has twice
mis-paired out-of-fold predictions by row position, so rather than guess an
ordering, nothing here names a side at all:

* flipping which score is read first negates every judge's margin, so
  ``|mean margin|`` is unchanged;
* the card SHAPE -- how many judges favoured the majority side, the minority
  side, and how many scored it level -- is likewise unchanged.

The sign belongs to ``y_winner``, which `mma.features` already owns and
already orients through the md5-parity corner swap. So the label is built as
``(2 * y_winner - 1) * abs_mean_margin`` at the point where ``y_winner``
exists, and there is no orientation in this file to get wrong.
``tests/test_scorecards.py`` asserts no frame-dependent column can escape.

**Validation is against the promotion's own label.** UFC records each decision
as unanimous, split or majority, and the frame-invariant card shape has to
agree: unanimous is 3-0-0, split is 2-1-0 (or 1-1-1 when a judge scores a
draw), majority is 2-0-1. On the real scrape it agrees on 99.8% of rows. The
0.2% that disagree are DROPPED -- a parse that contradicts the promotion is
not evidence about which of the two is wrong, and keeping it would be
inventing data.

**What the magnitude carries.** Unanimous decisions average ``|mean margin|``
2.28; split decisions average 0.54. Today both are the same training example,
``y_winner = 1``. See `docs/superpowers/plans/2026-09-13-sp5-scorecard-label.md`.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: A judge's line: a name, then two totals. The name is non-greedy so it stops
#: at the first digit run, and the class covers the accented and apostrophed
#: names in the source ("Sal D'amato", "Junichiro Kamijo").
CARD = re.compile(r"([A-Z][A-Za-z.'\-À-ɏ ]+?)\s+(\d{1,2})\s*-\s*(\d{1,2})")

#: Card shapes the promotion's own subtype allows, as
#: (majority side, minority side, level). A split where one judge scores a
#: draw reads 1-1-1, which is still a split.
VALID_SHAPES = {
    "unanimous": {(3, 0, 0)},
    "split": {(2, 1, 0), (1, 1, 1)},
    "majority": {(2, 0, 1), (1, 0, 2)},
}

N_JUDGES = 3


def parse_cards(details) -> tuple[tuple[str, int, int], ...]:
    """Every ``(judge, score, score)`` in a `details` string, in source order.

    The two scores are deliberately NOT named: which is which varies by row.
    A finish, an empty field or a NaN yields ``()``.
    """
    if details is None or (isinstance(details, float) and np.isnan(details)):
        return ()
    return tuple(
        (name.strip(), int(one), int(two))
        for name, one, two in CARD.findall(str(details))
    )


def margins(cards) -> list[int]:
    """Each judge's margin in the source's own (unknown) direction."""
    return [one - two for _, one, two in cards]


def abs_mean_margin(cards) -> float:
    """|mean per-judge margin|. Frame-invariant: flipping the read negates
    every element, and the absolute value of the mean is unchanged."""
    if not cards:
        return float("nan")
    return float(abs(np.mean(margins(cards))))


def card_shape(cards) -> tuple[int, int, int]:
    """(majority side, minority side, level), frame-invariant.

    Which side is "majority" is not named -- only how the three judges split,
    which is what the promotion's unanimous/split/majority label describes.
    """
    m = margins(cards)
    one_way = sum(x > 0 for x in m)
    other_way = sum(x < 0 for x in m)
    return (max(one_way, other_way), min(one_way, other_way),
            sum(x == 0 for x in m))


def subtype_of(method) -> str | None:
    """'unanimous' / 'split' / 'majority' from the scrape's method string."""
    match = re.search(r"Decision\s*-\s*(\w+)", str(method))
    return match.group(1).lower() if match else None


def shape_agrees_with_subtype(cards, method) -> bool:
    subtype = subtype_of(method)
    if subtype is None or subtype not in VALID_SHAPES:
        return False
    return card_shape(cards) in VALID_SHAPES[subtype]


def build_scorecards(raw_fights: pd.DataFrame) -> pd.DataFrame:
    """One row per decision whose three cards parse AND agree with the
    promotion's recorded subtype.

    Columns: ``fight_id``, ``n_judges``, ``abs_mean_margin``,
    ``judges_disagreed``, ``decision_subtype``. Nothing that names a corner.

    Everything else -- finishes, unparseable rows, partial cards, and the
    0.2% whose shape contradicts the subtype -- is simply absent. Absence is
    the mask: `mma.features` leaves those fights' label NaN and the auxiliary
    head skips them.
    """
    rows = []
    for fight_id, method, details in raw_fights[
        ["fight_id", "method", "details"]
    ].itertuples(index=False):
        cards = parse_cards(details)
        if len(cards) != N_JUDGES:
            continue
        if not shape_agrees_with_subtype(cards, method):
            continue
        shape = card_shape(cards)
        rows.append({
            "fight_id": fight_id,
            "n_judges": len(cards),
            "abs_mean_margin": abs_mean_margin(cards),
            # a split or majority card -- ground truth that the fight was close
            "judges_disagreed": shape[1] > 0 or shape[2] > 0,
            "decision_subtype": subtype_of(method),
        })
    frame = pd.DataFrame(
        rows,
        columns=["fight_id", "n_judges", "abs_mean_margin",
                 "judges_disagreed", "decision_subtype"],
    )
    if not frame.empty:
        frame["judges_disagreed"] = frame["judges_disagreed"].astype(bool)
    return frame


def signed_margin(y_winner, abs_mean_margin_values):
    """The SP5 label: the magnitude, signed by whoever actually won.

    ``y_winner`` is the feature table's own post-swap label, so the sign is
    correct in the feature table's frame by construction -- this function is
    the only place the two are combined, and it cannot be called with a
    corner id because it does not take one.

    A fight with no usable card keeps NaN, which is the auxiliary head's mask.
    """
    y = np.asarray(y_winner, dtype=float)
    magnitude = np.asarray(abs_mean_margin_values, dtype=float)
    return (2.0 * y - 1.0) * magnitude
