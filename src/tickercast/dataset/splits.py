"""Time-based train / validation / test split with an embargo.

    |------ train ------|xx|---- val ----|xx|------- test -------|
                         ^^ embargo: last N trading days before each boundary are dropped,
                            because their targets (up to N days ahead) fall in the next split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def assign_splits(dates: pd.Series, train_end: str, val_end: str, embargo_days: int) -> pd.Series:
    """Return 'train' / 'val' / 'test' / 'embargo' for each row's date."""
    d = pd.to_datetime(dates)
    split = np.where(d <= pd.Timestamp(train_end), "train", np.where(d <= pd.Timestamp(val_end), "val", "test"))
    split = pd.Series(split, index=dates.index)

    if embargo_days > 0:
        calendar = pd.DatetimeIndex(sorted(d.unique()))
        for boundary in (train_end, val_end):
            before = calendar[calendar <= pd.Timestamp(boundary)]
            if len(before):
                cutoff = before[-embargo_days:]
                split[d.isin(cutoff)] = "embargo"
    return split
