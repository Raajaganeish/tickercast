"""Join every feature group into one table: one row per (date, ticker).

Output: data/interim/features.parquet
    date | ticker | sector | <feature columns...> | <target columns...>
"""

from __future__ import annotations

import pandas as pd

from tickercast.config import Config
from tickercast.data.clean import trim_illiquid_history
from tickercast.features.calendar import calendar_features
from tickercast.features.macro import macro_features
from tickercast.features.market import beta_to_market, market_features
from tickercast.features.stock import cross_sectional_features, stock_features
from tickercast.features.targets import is_target, make_targets

ID_COLS = ["date", "ticker", "sector"]


def build_features(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    macro: pd.DataFrame | None,
    cfg: Config,
) -> pd.DataFrame:
    fcfg = cfg["features"]
    names = cfg["market"]["benchmarks"]

    # Trading calendar = days SPY traded. Drop stock rows on any other date.
    spy_close = market[market["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    prices = prices[prices["date"].isin(spy_close.index)]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    df = prices[["date", "ticker"]].copy()
    df["sector"] = df["ticker"].map(universe.set_index("ticker")["sector"]).fillna("Unknown")

    df = df.join(stock_features(prices, fcfg))
    df = df.join(cross_sectional_features(df, df["sector"]))

    mkt = market_features(market, names)
    df = df.join(mkt, on="date")
    df["rel_spy_ret_1d"] = df["ret_1d"] - df["spy_ret_1d"]
    df["beta_spy"] = beta_to_market(df["ret_1d"], df["spy_ret_1d"], df["ticker"], fcfg["beta_window"])

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    df = df.join(macro_features(macro, dates, cfg["macro"]["series"]), on="date")
    df = df.join(calendar_features(dates), on="date")

    df = df.join(make_targets(prices, spy_close, cfg["targets"]["horizons"]))

    # Warm-up: long rolling windows (200-day average etc.) are empty for a stock's first rows.
    row_num = df.groupby("ticker").cumcount()
    df = df[row_num >= fcfg["warmup_days"]]
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and not is_target(c)]


def target_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_target(c)]


def run_build(cfg: Config) -> pd.DataFrame:
    raw = cfg.path("raw")
    prices = pd.read_parquet(raw / "prices.parquet")
    universe = pd.read_parquet(raw / "universe.parquet")
    market = pd.read_parquet(raw / "market.parquet")
    macro_path = raw / "macro.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else None

    prices, trimmed = trim_illiquid_history(prices, cfg["universe"]["min_dollar_volume"])
    print(f"features: trimmed illiquid early history of {len(trimmed)} tickers "
          f"({trimmed['rows_dropped'].sum():,} rows), see reports/01_raw_validation.md")

    counts = prices.groupby("ticker").size()
    keep = counts[counts >= cfg["universe"]["min_history_days"]].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"features: dropping {len(dropped)} tickers with < {cfg['universe']['min_history_days']} days: {dropped}")
    prices = prices[prices["ticker"].isin(keep)]

    df = build_features(prices, universe, market, macro, cfg)
    float_cols = feature_columns(df) + target_columns(df)
    df[float_cols] = df[float_cols].astype("float32")
    out = cfg.path("interim") / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"features: {len(df):,} rows x {len(feature_columns(df))} features, "
          f"{df['ticker'].nunique()} tickers -> {out}")
    return df
