"""Synthetic market data so tests run offline and fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tickercast.config import load_config


def _ohlcv(ticker: str, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
    open_ = close * np.exp(rng.normal(0, 0.003, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(dates)))
    vol = rng.integers(1_000_000, 5_000_000, len(dates)).astype(float)
    return pd.DataFrame({"date": dates, "ticker": ticker, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synthetic():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", periods=700)
    tickers = [f"T{i:02d}" for i in range(12)]
    sectors = ["Tech", "Health", "Energy"]
    prices = pd.concat([_ohlcv(t, dates, rng) for t in tickers], ignore_index=True)
    # One stock lists later, so warm-up and windows are exercised on a short history.
    prices = prices[~((prices["ticker"] == "T11") & (prices["date"] < dates[300]))].reset_index(drop=True)
    universe = pd.DataFrame({"ticker": tickers, "sector": [sectors[i % 3] for i in range(len(tickers))]})
    bench = ["SPY", "QQQ", "^VIX", "TLT", "GLD", "USO", "UUP", "SOXX", "XLK"]
    market = pd.concat([_ohlcv(t, dates, rng) for t in bench], ignore_index=True)
    market.loc[market["ticker"] == "^VIX", "close"] = 15 + 5 * rng.random((market["ticker"] == "^VIX").sum())
    return prices, universe, market
