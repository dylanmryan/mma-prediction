"""Feature-table -> tensor preparation, fit on training rows only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("y_winner", "y_method", "y_finish_round")
IDENTIFIERS = ("fight_id", "date", "swapped")
# Era-proxy flags dropped per Phase 3 final-review ablation (hurt validation).
DROPPED = ("reach_missing_a", "reach_missing_b", "dob_missing_a", "dob_missing_b")
CATEGORICAL = "weight_class"


class Preprocessor:
    def __init__(self, numeric_columns, medians, means, stds, weight_classes):
        self.numeric_columns = list(numeric_columns)
        self.medians = dict(medians)
        self.means = dict(means)
        self.stds = dict(stds)
        self.weight_classes = list(weight_classes)  # index 0 reserved for unknown

    @classmethod
    def fit(cls, features: pd.DataFrame, train_mask: np.ndarray) -> "Preprocessor":
        excluded = set(TARGETS) | set(IDENTIFIERS) | set(DROPPED) | {CATEGORICAL}
        numeric_columns = [
            column for column in features.columns if column not in excluded
        ]
        train = features.loc[train_mask, numeric_columns].astype(float)
        medians = train.median().fillna(0.0).to_dict()
        imputed = train.fillna(medians)
        means = imputed.mean().to_dict()
        stds = {
            column: (value if value and not np.isnan(value) else 1.0)
            for column, value in imputed.std(ddof=0).to_dict().items()
        }
        classes = sorted(
            features.loc[train_mask, CATEGORICAL].dropna().unique().tolist()
        )
        return cls(numeric_columns, medians, means, stds, classes)

    def transform(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = features[self.numeric_columns].astype(float)
        numeric = numeric.fillna(self.medians)
        x = np.stack(
            [
                (numeric[column].to_numpy() - self.means[column]) / self.stds[column]
                for column in self.numeric_columns
            ],
            axis=1,
        ).astype(np.float32)
        index = {name: i + 1 for i, name in enumerate(self.weight_classes)}
        wc = (
            features[CATEGORICAL]
            .map(lambda v: index.get(v, 0) if pd.notna(v) else 0)
            .to_numpy(dtype=np.int64)
        )
        return x, wc

    @property
    def n_weight_classes(self) -> int:
        return len(self.weight_classes) + 1

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "numeric_columns": self.numeric_columns,
                    "medians": self.medians,
                    "means": self.means,
                    "stds": self.stds,
                    "weight_classes": self.weight_classes,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path) -> "Preprocessor":
        payload = json.loads(Path(path).read_text())
        return cls(
            payload["numeric_columns"], payload["medians"], payload["means"],
            payload["stds"], payload["weight_classes"],
        )
