"""Market-wide features: one value per trading day, shared by every stock.

Includes benchmark returns (SPY, QQQ, XLK, semis, bonds, gold, oil, dollar), the VIX
fear index, and market stress measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_features(market: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """market: long OHLCV for benchmark tickers. Returns a frame indexed by date."""
    close = market.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close.rename(columns=names)
    # Yahoo sometimes has VIX quotes on US market holidays. Keep only days SPY traded,
    # otherwise those phantom days break every other benchmark's returns.
    if "spy" in close:
        close = close[close["spy"].notna()]
    f = pd.DataFrame(index=close.index)

    for name in close.columns:
        if name == "vix":
            continue
        lr = np.log(close[name]).diff()
        f[f"{name}_ret_1d"] = lr
        f[f"{name}_ret_5d"] = lr.rolling(5, min_periods=5).sum()

    if "spy" in close:
        spy_lr = np.log(close["spy"]).diff()
        f["spy_vol_20d"] = spy_lr.rolling(20, min_periods=20).std()
        f["spy_drawdown_252"] = close["spy"] / close["spy"].rolling(252, min_periods=60).max() - 1

    if "vix" in close:
        vix = close["vix"]
        f["vix_log"] = np.log(vix)
        f["vix_chg_1d"] = np.log(vix).diff()
        mean, std = vix.rolling(60, min_periods=20).mean(), vix.rolling(60, min_periods=20).std()
        f["vix_z_60"] = (vix - mean) / std

    return f


def beta_to_market(stock_ret: pd.Series, spy_ret: pd.Series, tickers: pd.Series, window: int = 60) -> pd.Series:
    """Rolling beta of each stock's daily return to SPY's daily return (past `window` days)."""
    df = pd.DataFrame({"r": stock_ret, "m": spy_ret, "t": tickers})

    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["r"].rolling(window, min_periods=window).cov(g["m"])
        var = g["m"].rolling(window, min_periods=window).var()
        return cov / var

    return df.groupby("t", sort=False, group_keys=False)[["r", "m"]].apply(_beta).loc[df.index]
