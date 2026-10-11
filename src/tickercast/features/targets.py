"""What the model should predict. These are the ONLY columns allowed to look into the future.

fwd_ret_{h}d        log return from close t to close t+h
label_up_{h}d       1 if fwd_ret_{h}d > 0 else 0
fwd_excess_{h}d     fwd_ret minus SPY's forward return (how the stock did vs the market)
label_outperform_{h}d  1 if fwd_excess_{h}d > 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_PREFIXES = ("fwd_", "label_")


def is_target(col: str) -> bool:
    return col.startswith(TARGET_PREFIXES)


def make_targets(prices: pd.DataFrame, spy_close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    log_close = np.log(prices["close"])
    g = log_close.groupby(prices["ticker"], sort=False)
    spy_log = np.log(spy_close)
    for h in horizons:
        fwd = g.shift(-h) - log_close
        spy_fwd = (spy_log.shift(-h) - spy_log).reindex(prices["date"]).to_numpy()
        out[f"fwd_ret_{h}d"] = fwd
        out[f"label_up_{h}d"] = (fwd > 0).astype(float).where(fwd.notna())
        out[f"fwd_excess_{h}d"] = fwd - spy_fwd
        out[f"label_outperform_{h}d"] = (out[f"fwd_excess_{h}d"] > 0).astype(float).where(out[f"fwd_excess_{h}d"].notna())
    return out
