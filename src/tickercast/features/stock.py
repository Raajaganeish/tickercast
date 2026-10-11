"""Features from each stock's own price and volume history.

Every feature on row (ticker, date t) uses only data up to and including the close of t.
All rolling windows look backwards. Nothing here may use shift(-k).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    # Wilder's RSI, rescaled from [0, 100] to [-0.5, 0.5]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(loss != 0, 100.0)  # no down days in window -> RSI 100
    return rsi / 100 - 0.5


def _one_ticker(g: pd.DataFrame, fcfg: dict) -> pd.DataFrame:
    close, high, low, open_, vol = g["close"], g["high"], g["low"], g["open"], g["volume"]
    logret = np.log(close).diff()
    f = pd.DataFrame(index=g.index)

    for w in fcfg["return_windows"]:
        f[f"ret_{w}d"] = logret.rolling(w, min_periods=w).sum()
    for w in fcfg["vol_windows"]:
        f[f"vol_{w}d"] = logret.rolling(w, min_periods=w).std()

    vw = fcfg["volume_window"]
    f["volume_ratio"] = np.log1p(vol) - np.log1p(vol.rolling(vw, min_periods=vw).mean())
    f["rsi"] = _rsi(close, fcfg["rsi_window"])
    for w in fcfg["ma_windows"]:
        f[f"ma_dist_{w}"] = close / close.rolling(w, min_periods=w).mean() - 1
    f["hl_range"] = (high - low) / close
    f["gap"] = np.log(open_ / close.shift(1))
    f["drawdown_252"] = close / close.rolling(252, min_periods=60).max() - 1
    return f


def stock_features(prices: pd.DataFrame, fcfg: dict) -> pd.DataFrame:
    """prices: long OHLCV sorted by (ticker, date). Returns features aligned to prices.index."""
    parts = [_one_ticker(g, fcfg) for _, g in prices.groupby("ticker", sort=False)]
    return pd.concat(parts).loc[prices.index]


def cross_sectional_features(df: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """Compare each stock with all other stocks on the same day.

    - cs_rank_*: percentile rank among all stocks that day, centered to [-0.5, 0.5]
    - sector_ret_*: average return of stocks in the same GICS sector that day
    - rel_sector_*: stock return minus its sector average
    - breadth: share of stocks that went up that day (a market-wide feature)
    """
    out = pd.DataFrame(index=df.index)
    by_date = df.groupby("date")
    for col in ["ret_5d", "ret_20d", "ret_60d", "vol_20d", "volume_ratio"]:
        out[f"cs_rank_{col}"] = by_date[col].rank(pct=True) - 0.5

    key = [df["date"], sector]
    for col in ["ret_1d", "ret_5d"]:
        sec = df.groupby(key)[col].transform("mean")
        out[f"sector_{col}"] = sec
        out[f"rel_sector_{col}"] = df[col] - sec

    out["breadth"] = by_date["ret_1d"].transform(lambda s: (s > 0).mean()) - 0.5
    return out
