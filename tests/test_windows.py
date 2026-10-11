from __future__ import annotations

import numpy as np
import pandas as pd

from tickercast.dataset.windows import valid_window_ends


def test_windows_never_cross_tickers():
    df = pd.DataFrame({
        "ticker": ["A"] * 10 + ["B"] * 4,
        "split": ["train"] * 14,
        "y": np.r_[np.ones(9), np.nan, np.ones(4)],
    })
    ends = valid_window_ends(df, window=5, target="y", split="train")
    # A: rows 4..8 valid (row 9 has no target); B has only 4 rows < window
    assert ends.tolist() == [4, 5, 6, 7, 8]
    for t in ends:
        assert df["ticker"].iloc[t - 4 : t + 1].nunique() == 1
