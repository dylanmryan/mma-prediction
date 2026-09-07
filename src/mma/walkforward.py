"""Expanding-window walk-forward evaluation.

Fold Y trains on fights before Y-1, early-stops on year Y-1, and is scored
on year Y (the last fold absorbs everything from 2025 onward). Pooled
metrics over all fold years are the headline; per-fold and slice metrics
are reported alongside. See the v3 design spec §4 SP1 / §5 for the locked
protocol and shipping bars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FOLD_YEARS = tuple(range(2018, 2026))
_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray
    inner_val: np.ndarray
    eval: np.ndarray

    @property
    def eval_start(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=1, day=1)


def make_folds(
    dates: pd.Series, fold_years=FOLD_YEARS, train_start: str | None = None
) -> list[Fold]:
    """Boolean masks (aligned to `dates`) for every fold year.

    The largest year in `fold_years` is unbounded above (it absorbs every
    later fight), which is how the default 2025 fold takes in the partial
    2026."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    start = pd.Timestamp(train_start) if train_start else None
    folds = []
    last = max(fold_years)
    for year in fold_years:
        val_start = pd.Timestamp(year=year - 1, month=1, day=1)
        eval_start = pd.Timestamp(year=year, month=1, day=1)
        eval_end = pd.Timestamp(year=year + 1, month=1, day=1)
        train = (d < val_start).to_numpy()
        if start is not None:
            train = train & (d >= start).to_numpy()
        inner_val = ((d >= val_start) & (d < eval_start)).to_numpy()
        eval_mask = (d >= eval_start).to_numpy()
        if year != last:
            eval_mask = eval_mask & (d < eval_end).to_numpy()
        folds.append(Fold(year, train, inner_val, eval_mask))
    return folds


def recency_weights(
    dates: pd.Series, reference: pd.Timestamp, half_life_years: float | None
) -> np.ndarray:
    """0.5 ** (age_in_years / half_life); uniform ones when half_life is None."""
    if half_life_years is None:
        return np.ones(len(dates), dtype=float)
    age_years = (reference - pd.to_datetime(dates)).dt.days.to_numpy() / _DAYS_PER_YEAR
    return np.power(0.5, np.clip(age_years, 0, None) / half_life_years)
