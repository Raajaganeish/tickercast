"""Cleaning rules applied to raw prices before features are built.

Illiquid history trim:
    Some stocks have early Yahoo history from over-the-counter or foreign trading before a
    proper US listing (e.g. SW, FERG, AMCR). Those years show flat prices and a handful of
    shares traded per day, which is not real price discovery. For each stock we find the
    last day its 20-day median dollar volume (close x volume) was below the threshold and
    drop everything up to that day. The stock enters the dataset once it is liquid for good.
"""

from __future__ import annotations

import pandas as pd


def illiquid_cutoffs(prices: pd.DataFrame, min_dollar_volume: float, window: int = 20) -> pd.Series:
    """Last illiquid date per ticker (only tickers that have one)."""
    p = prices.sort_values(["ticker", "date"])
    dollar_vol = (p["close"] * p["volume"]).groupby(p["ticker"]).transform(
        lambda s: s.rolling(window, min_periods=1).median()
    )
    return p.loc[dollar_vol < min_dollar_volume].groupby("ticker")["date"].max()


def trim_illiquid_history(prices: pd.DataFrame, min_dollar_volume: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned prices, summary of what was trimmed per ticker)."""
    cut = illiquid_cutoffs(prices, min_dollar_volume)
    cutoff = prices["ticker"].map(cut)
    drop = cutoff.notna() & (prices["date"] <= cutoff)
    summary = (
        prices[drop].groupby("ticker")["date"].agg(first_dropped="min", last_dropped="max", rows_dropped="count")
        .sort_values("rows_dropped", ascending=False).reset_index()
    )
    return prices[~drop].reset_index(drop=True), summary
