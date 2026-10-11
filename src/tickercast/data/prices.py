"""Daily OHLCV prices from Yahoo Finance (via yfinance).

Prices are split- and dividend-adjusted (auto_adjust=True), so a 2-for-1 split does
not look like a -50% crash, and returns include dividends.

Output format ("long"): one row per (date, ticker)
    date | ticker | open | high | low | close | volume
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from tickercast.config import Config

COLUMNS = ["open", "high", "low", "close", "volume"]


def download_ohlcv(tickers: list[str], start: str, end: str, chunk_size: int = 50) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        for attempt in range(3):
            try:
                df = yf.download(
                    chunk, start=start, end=end, auto_adjust=True, group_by="ticker",
                    threads=True, progress=False,
                )
                break
            except Exception as e:  # network hiccups / rate limits
                print(f"  retry {attempt + 1} for chunk {i // chunk_size}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed to download chunk starting at {chunk[0]}")
        if df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):  # single ticker
            df.columns = pd.MultiIndex.from_product([[chunk[0]], df.columns])
        long = df.stack(level=0, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
        long.columns = [str(c).lower() for c in long.columns]
        frames.append(long[["date", "ticker", *COLUMNS]])
        print(f"  downloaded {min(i + chunk_size, len(tickers))}/{len(tickers)}")

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    # Yahoo returns NaN rows for dates before a ticker existed; drop them.
    prices = prices.dropna(subset=["close"])
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_stock_prices(cfg: Config, universe: pd.DataFrame) -> pd.DataFrame:
    tickers = universe["ticker"].tolist()
    print(f"prices: downloading {len(tickers)} stocks {cfg.start} -> {cfg.end}")
    prices = download_ohlcv(tickers, cfg.start, cfg.end)
    missing = sorted(set(tickers) - set(prices["ticker"]))
    if missing:
        print(f"  WARNING: no data for {len(missing)} tickers: {missing}")
    out = cfg.path("raw") / "prices.parquet"
    prices.to_parquet(out, index=False)
    print(f"prices: {len(prices):,} rows, {prices['ticker'].nunique()} tickers -> {out}")
    return prices


def download_market(cfg: Config) -> pd.DataFrame:
    tickers = list(cfg["market"]["benchmarks"].keys())
    print(f"market: downloading {tickers}")
    market = download_ohlcv(tickers, cfg.start, cfg.end, chunk_size=len(tickers))
    out = cfg.path("raw") / "market.parquet"
    market.to_parquet(out, index=False)
    print(f"market: {len(market):,} rows -> {out}")
    return market
