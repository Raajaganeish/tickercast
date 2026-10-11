"""Checks on downloaded raw data, before any features are built.

Goal: catch broken data early (bad prices, missing days, wrong split adjustments,
macro values dated before they were published).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tickercast.config import Config
from tickercast.data.clean import trim_illiquid_history
from tickercast.validation.report import Report

EXTREME_MOVE = 0.40      # |daily log return| above this is suspicious (~ +49% / -33%)
STALE_RUN = 5            # same close for this many days in a row
GAP_DAYS = 5             # consecutive missing trading days that count as a gap


def check_universe(r: Report, uni: pd.DataFrame, prices: pd.DataFrame) -> None:
    n = len(uni)
    r.add("Universe size", "PASS" if 480 <= n <= 520 else "WARN", f"{n} tickers (S&P 500 has ~503 share classes)")
    dup = uni["ticker"].duplicated().sum()
    r.add("Universe duplicates", "PASS" if dup == 0 else "FAIL", f"{dup} duplicated tickers")
    no_sector = uni["sector"].isna().sum()
    r.add("Sector assigned", "PASS" if no_sector == 0 else "WARN", f"{no_sector} tickers without a sector")
    missing = sorted(set(uni["ticker"]) - set(prices["ticker"]))
    r.add("Prices downloaded for every ticker", "PASS" if not missing else "WARN",
          f"{len(missing)} missing" + (f": {', '.join(missing[:15])}" if missing else ""))
    r.add("Sector mix", "INFO", f"{uni['sector'].nunique()} sectors",
          uni["sector"].value_counts().rename_axis("sector").reset_index(name="tickers"))


def check_prices(r: Report, prices: pd.DataFrame, calendar: pd.DatetimeIndex, label: str = "Prices",
                 min_dollar_volume: float = 1e6) -> None:
    p = prices.sort_values(["ticker", "date"])
    dup = p.duplicated(["ticker", "date"]).sum()
    r.add(f"{label}: duplicate rows", "PASS" if dup == 0 else "FAIL", f"{dup} duplicated (ticker, date)")

    ohlc = p[["open", "high", "low", "close"]]
    nan_rows = ohlc.isna().any(axis=1).sum()
    r.add(f"{label}: missing OHLC values", "PASS" if nan_rows == 0 else "WARN", f"{nan_rows:,} rows with NaN")

    nonpos = (ohlc <= 0).any(axis=1).sum()
    r.add(f"{label}: non-positive prices", "PASS" if nonpos == 0 else "FAIL", f"{nonpos:,} rows with price <= 0")

    tol = 1e-6 * p["close"].abs() + 1e-3
    bad = ((p["low"] > p[["open", "close"]].min(axis=1) + tol) | (p["high"] < p[["open", "close"]].max(axis=1) - tol))
    share = bad.mean()
    r.add(f"{label}: OHLC consistency (low <= open,close <= high)",
          "PASS" if share < 1e-4 else "WARN", f"{bad.sum():,} inconsistent rows ({share:.3%})")

    negvol = (p["volume"] < 0).sum()
    zerovol = (p["volume"] == 0).mean()
    r.add(f"{label}: volume", "FAIL" if negvol else ("PASS" if zerovol < 0.01 else "WARN"),
          f"{negvol} negative, {zerovol:.2%} zero-volume days")

    if label != "Prices":
        return

    logret = np.log(p["close"]).groupby(p["ticker"]).diff()
    ext = p.assign(log_return=logret)[logret.abs() > EXTREME_MOVE]
    r.add("Prices: extreme daily moves", "PASS" if len(ext) < 0.0001 * len(p) else "WARN",
          f"{len(ext)} days with |log return| > {EXTREME_MOVE} (real crashes or bad split data; review list)",
          ext.reindex(ext["log_return"].abs().sort_values(ascending=False).index)
             [["ticker", "date", "close", "log_return"]].head(25).reset_index(drop=True))

    _, trimmed = trim_illiquid_history(p, min_dollar_volume)
    r.add("Prices: illiquid early history (trimmed in feature step)", "INFO",
          f"{len(trimmed)} tickers have early history trading < ${min_dollar_volume:,.0f}/day; "
          f"{trimmed['rows_dropped'].sum():,} rows ({trimmed['rows_dropped'].sum() / len(p):.1%}) will be dropped",
          trimmed.head(25).assign(first_dropped=lambda d: d["first_dropped"].dt.date,
                                  last_dropped=lambda d: d["last_dropped"].dt.date))
    p = p.merge(trimmed[["ticker", "last_dropped"]], on="ticker", how="left")
    p = p[p["last_dropped"].isna() | (p["date"] > p["last_dropped"])].drop(columns="last_dropped").reset_index(drop=True)

    # Remaining checks look at the data that will actually be used.
    same = p["close"].eq(p.groupby("ticker")["close"].shift(1))
    run_id = (~same).groupby(p["ticker"]).cumsum()
    runs = same.groupby([p["ticker"], run_id]).sum()
    stale = runs[runs >= STALE_RUN - 1]
    r.add("Prices: stale prices", "PASS" if stale.empty else "WARN",
          f"{len(stale)} runs of >= {STALE_RUN} identical closes")

    # Missing trading days inside each stock's active period (vs the SPY calendar).
    rows = []
    cal_pos = pd.Series(np.arange(len(calendar)), index=calendar)
    for t, g in p.groupby("ticker"):
        pos = cal_pos.reindex(g["date"]).dropna().to_numpy()
        if len(pos) < 2:
            continue
        expected = int(pos[-1] - pos[0] + 1)
        steps = np.diff(pos) - 1
        rows.append({"ticker": t, "first": g["date"].iloc[0].date(), "last": g["date"].iloc[-1].date(),
                     "days": len(g), "missing_days": expected - len(pos), "longest_gap": int(steps.max(initial=0))})
    cov = pd.DataFrame(rows)
    gaps = cov[cov["longest_gap"] >= GAP_DAYS].sort_values("longest_gap", ascending=False)
    total_missing = cov["missing_days"].sum()
    r.add("Prices: missing trading days", "PASS" if gaps.empty else "WARN",
          f"{total_missing:,} missing days in total; {len(gaps)} tickers with a gap >= {GAP_DAYS} days",
          gaps.head(25).reset_index(drop=True))

    off_cal = (~p["date"].isin(calendar)).sum()
    r.add("Prices: dates outside trading calendar", "PASS" if off_cal == 0 else "WARN",
          f"{off_cal:,} rows on days SPY did not trade (dropped during feature build)")

    short = cov[cov["days"] < 252 * 3]
    r.add("Prices: history length", "INFO",
          f"median {cov['days'].median():.0f} days; {len(short)} tickers with < 3 years",
          cov["days"].describe().to_frame("trading_days").T.round(0))

    per_year = p.groupby(p["date"].dt.year)["ticker"].nunique().rename_axis("year").reset_index(name="tickers")
    r.add("Prices: survivorship check (tickers per year)", "INFO",
          f"{per_year['tickers'].iloc[0]} tickers in {per_year['year'].iloc[0]} -> "
          f"{per_year['tickers'].iloc[-1]} in {per_year['year'].iloc[-1]}. Fewer early tickers = today's "
          f"members that were not public yet; failed companies are missing entirely (survivorship bias).",
          per_year)


def check_market(r: Report, market: pd.DataFrame, cfg: Config) -> None:
    expected = set(cfg["market"]["benchmarks"])
    got = set(market["ticker"])
    r.add("Market: all benchmarks downloaded", "PASS" if expected <= got else "FAIL",
          f"missing: {sorted(expected - got) or 'none'}")
    spans = market.groupby("ticker")["date"].agg(["min", "max", "count"]).reset_index()
    late = spans[spans["min"] > pd.Timestamp(cfg.start) + pd.Timedelta(days=30)]
    r.add("Market: history starts at config start", "PASS" if late.empty else "WARN",
          f"{len(late)} benchmarks start later: " + ", ".join(f"{t} ({d.date()})" for t, d in zip(late['ticker'], late['min'])),
          spans)
    wide = market.pivot(index="date", columns="ticker", values="close")
    if "SPY" in wide:
        phantom = wide.index[wide["SPY"].isna() & wide.notna().any(axis=1)]
        r.add("Market: quotes on non-trading days", "PASS" if len(phantom) == 0 else "WARN",
              f"{len(phantom)} dates with quotes but no SPY trading (dropped in feature step): "
              + ", ".join(str(d.date()) for d in phantom[-10:]))

    # The VIX is an index, not a traded fund: it has no volume, so leave it out of the OHLCV checks.
    tradable = market[market["ticker"] != "^VIX"]
    check_prices(r, tradable.assign(volume=tradable["volume"].fillna(0)), pd.DatetimeIndex([]), label="Market")


def check_macro(r: Report, macro: pd.DataFrame | None, cfg: Config) -> None:
    if macro is None:
        r.add("Macro: data present", "WARN", "no macro.parquet (FRED_API_KEY not set). Macro features will be absent.")
        return
    early = (macro["available_date"] < macro["obs_date"]).sum()
    r.add("Macro: available_date >= observation date", "PASS" if early == 0 else "FAIL",
          f"{early} values available before the period they describe")
    rows = []
    for sid, g in macro.groupby("series"):
        lag = (g["available_date"] - g["obs_date"]).dt.days
        rows.append({"series": sid, "name": g["name"].iloc[0], "obs": len(g),
                     "first": g["obs_date"].min().date(), "last": g["obs_date"].max().date(),
                     "median_lag_days": lag.median(), "max_lag_days": lag.max()})
    tbl = pd.DataFrame(rows)
    r.add("Macro: publication lags", "INFO", "days between the period a value describes and when it became public", tbl)
    stale = tbl[pd.to_datetime(tbl["last"]) < pd.Timestamp(cfg.end) - pd.Timedelta(days=75)]
    r.add("Macro: series are up to date", "PASS" if stale.empty else "WARN",
          f"{len(stale)} series older than 75 days: {', '.join(stale['series']) or 'none'}")


def run_raw_checks(cfg: Config) -> Report:
    raw = cfg.path("raw")
    uni = pd.read_parquet(raw / "universe.parquet")
    prices = pd.read_parquet(raw / "prices.parquet")
    market = pd.read_parquet(raw / "market.parquet")
    macro_path = raw / "macro.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else None
    calendar = pd.DatetimeIndex(sorted(market.loc[market["ticker"] == "SPY", "date"].unique()))

    r = Report("Raw data validation", "Checks on downloaded data before features are built. "
               "See docs/05_validation.md for what each check means.")
    check_universe(r, uni, prices)
    check_prices(r, prices, calendar, min_dollar_volume=cfg["universe"]["min_dollar_volume"])
    check_market(r, market, cfg)
    check_macro(r, macro, cfg)
    r.save(cfg.path("reports") / "01_raw_validation.md")
    return r
