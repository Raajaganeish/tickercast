"""Checks on the feature table: quality, target alignment, and above all, no leakage.

The most important check is the point-in-time (truncation) test:
    build features with all data  vs  build features with data cut off at date T.
    For every row on or before T, the two must be identical. If a feature changes when
    future data is removed, it was using the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tickercast.config import Config
from tickercast.dataset.splits import SPLITS, assign_splits
from tickercast.features.build import build_features, feature_columns, target_columns
from tickercast.validation.report import Report

TRUNCATION_CUTOFFS = ["2012-06-29", "2020-03-16"]
TRUNCATION_TICKERS = 40


def check_basic(r: Report, df: pd.DataFrame, feats: list[str]) -> None:
    r.add("Shape", "INFO", f"{len(df):,} rows · {len(feats)} features · {df['ticker'].nunique()} tickers · "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    dup = df.duplicated(["ticker", "date"]).sum()
    r.add("Unique (ticker, date)", "PASS" if dup == 0 else "FAIL", f"{dup} duplicates")

    vals = df[feats].to_numpy(dtype="float64")
    n_inf = np.isinf(vals).sum()
    r.add("No infinite values", "PASS" if n_inf == 0 else "FAIL", f"{n_inf} inf values",
          pd.Series(np.isinf(vals).sum(0), index=feats).loc[lambda s: s > 0].rename("inf").to_frame())

    std = df[feats].std()
    const = std[std < 1e-12].index.tolist()
    r.add("No constant features", "PASS" if not const else "FAIL", f"{len(const)} constant: {const}")


def check_missing(r: Report, df: pd.DataFrame, feats: list[str], split: pd.Series) -> None:
    rates = df[feats].isna().groupby(split).mean().T
    rates = rates[[s for s in SPLITS if s in rates.columns]]
    rates["overall"] = df[feats].isna().mean()
    bad = rates[rates["overall"] > 0.05].sort_values("overall", ascending=False)
    r.add("Missing values per feature", "PASS" if bad.empty else "WARN",
          f"{len(bad)} features with > 5% missing (filled with 0 + scaled later; review causes)",
          rates[rates["overall"] > 0].sort_values("overall", ascending=False))


def check_targets(r: Report, df: pd.DataFrame, split: pd.Series) -> None:
    # fwd_ret_1d on day t must equal ret_1d on the stock's next trading day.
    nxt = df.groupby("ticker")["ret_1d"].shift(-1)
    both = df["fwd_ret_1d"].notna() & nxt.notna()
    err = (df.loc[both, "fwd_ret_1d"] - nxt[both]).abs()
    bad = (err > 1e-4).sum()
    r.add("Target alignment (fwd_ret_1d == next day's ret_1d)", "PASS" if bad == 0 else "FAIL",
          f"{bad:,} mismatches out of {both.sum():,}")

    nan_targets = df["fwd_ret_1d"].isna().sum()
    last_rows = df.groupby("ticker").tail(1)["fwd_ret_1d"].isna().sum()
    r.add("Missing targets only at series end", "PASS" if nan_targets <= last_rows + df["ticker"].nunique() * 5 else "WARN",
          f"{nan_targets:,} rows without a 1-day target ({last_rows} are each stock's last day)")

    bal = df.groupby(split)[[c for c in target_columns(df) if c.startswith("label_")]].mean()
    bal = bal.reindex([s for s in SPLITS if s in bal.index])
    r.add("Label balance (share of 1s)", "INFO",
          "close to 0.5 is good; 'always up' accuracy in each split = label_up share", bal)


def _daily_rank_ic(df: pd.DataFrame, feats: list[str], target: str) -> pd.Series:
    """Mean over days of the Spearman correlation between each feature and the target."""
    d = df[["date", target, *feats]].dropna(subset=[target])
    g = d.groupby("date")
    ranks = g[[target, *feats]].rank(pct=True)
    ranks = ranks - ranks.groupby(d["date"]).transform("mean")
    y = ranks[target]
    out = {}
    for f in feats:
        x = ranks[f]
        ok = x.notna()
        num = (x[ok] * y[ok]).groupby(d.loc[ok, "date"]).sum()
        den = np.sqrt((x[ok] ** 2).groupby(d.loc[ok, "date"]).sum() * (y[ok] ** 2).groupby(d.loc[ok, "date"]).sum())
        ic = (num / den.replace(0, np.nan)).dropna()
        out[f] = ic.mean() if len(ic) else np.nan
    return pd.Series(out)


def check_leakage_scan(r: Report, df: pd.DataFrame, feats: list[str]) -> None:
    # Cross-sectional features only make sense here; market-wide features are constant per day.
    varying = [f for f in feats if df.groupby("date")[f].nunique().median() > 1]
    ic = _daily_rank_ic(df, varying, "fwd_ret_1d").sort_values(key=np.abs, ascending=False)
    top = ic.abs().max()
    status = "FAIL" if top > 0.2 else ("WARN" if top > 0.08 else "PASS")
    r.add("Leakage scan: feature vs next-day return correlation", status,
          f"max |daily rank IC| = {top:.3f}. Real signals in daily stock data are ~0.01-0.05; "
          f"> 0.2 almost always means the feature sees the future",
          ic.rename("mean_daily_rank_IC").to_frame().head(15))


def check_truncation(r: Report, cfg: Config) -> None:
    raw = cfg.path("raw")
    prices = pd.read_parquet(raw / "prices.parquet")
    universe = pd.read_parquet(raw / "universe.parquet")
    market = pd.read_parquet(raw / "market.parquet")
    macro_path = raw / "macro.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else None

    rng = np.random.default_rng(0)
    tickers = sorted(prices["ticker"].unique())
    sample = set(rng.choice(tickers, size=min(TRUNCATION_TICKERS, len(tickers)), replace=False)) | {"AAPL"}
    prices = prices[prices["ticker"].isin(sample)]

    full = build_features(prices, universe, market, macro, cfg)
    feats = feature_columns(full)
    rows = []
    for cutoff in TRUNCATION_CUTOFFS:
        c = pd.Timestamp(cutoff)
        cut = build_features(
            prices[prices["date"] <= c], universe, market[market["date"] <= c],
            None if macro is None else macro[macro["available_date"] <= c], cfg,
        )
        a = full[full["date"] <= c].set_index(["ticker", "date"])[feats].sort_index()
        b = cut.set_index(["ticker", "date"])[feats].sort_index()
        a, b = a.align(b, join="inner")
        diff = ~np.isclose(a.to_numpy("float64"), b.to_numpy("float64"), rtol=1e-5, atol=1e-6, equal_nan=True)
        per_feat = pd.Series(diff.sum(0), index=feats)
        rows.append({"cutoff": cutoff, "rows_compared": len(a), "cells_differ": int(diff.sum()),
                     "features_differ": ", ".join(per_feat[per_feat > 0].index) or "none"})
    tbl = pd.DataFrame(rows)
    bad = tbl["cells_differ"].sum()
    r.add("Point-in-time test (truncation)", "PASS" if bad == 0 else "FAIL",
          f"features rebuilt with data cut at {', '.join(TRUNCATION_CUTOFFS)} on {len(sample)} tickers: "
          f"{bad} differing values", tbl)


def check_drift(r: Report, df: pd.DataFrame, feats: list[str], split: pd.Series) -> None:
    tr = df.loc[split == "train", feats]
    mu, sd = tr.mean(), tr.std().replace(0, np.nan)
    rows = {}
    for s in ("val", "test"):
        part = df.loc[split == s, feats]
        if len(part):
            rows[f"{s}_shift_in_std"] = (part.mean() - mu) / sd
    shift = pd.DataFrame(rows)
    worst = shift.abs().max(axis=1).sort_values(ascending=False)
    big = worst[worst > 1.0]
    r.add("Distribution drift vs train", "PASS" if big.empty else "WARN",
          f"{len(big)} features whose val/test mean moved > 1 train std (expected for rates/inflation; "
          f"a reason to prefer changes and ranks over raw levels)",
          shift.loc[worst.index].head(15))

    z = ((df[feats] - mu) / sd).abs()
    extreme = (z > 10).mean().sort_values(ascending=False)
    r.add("Extreme values (> 10 train std)", "PASS" if extreme.max() < 0.001 else "WARN",
          f"worst feature: {extreme.index[0]} ({extreme.iloc[0]:.3%} of rows). Clipped to ±5 std after scaling",
          extreme[extreme > 0].head(10).rename("share_of_rows").to_frame())


def run_feature_checks(cfg: Config, truncation: bool = True) -> Report:
    df = pd.read_parquet(cfg.path("interim") / "features.parquet")
    feats = feature_columns(df)
    sp = cfg["splits"]
    split = assign_splits(df["date"], sp["train_end"], sp["val_end"], sp["embargo_days"])

    r = Report("Feature validation", "Checks on data/interim/features.parquet. "
               "See docs/05_validation.md for what each check means.")
    check_basic(r, df, feats)
    check_missing(r, df, feats, split)
    check_targets(r, df, split)
    check_leakage_scan(r, df, feats)
    if truncation:
        check_truncation(r, cfg)
    check_drift(r, df, feats, split)
    r.save(cfg.path("reports") / "02_feature_validation.md")
    return r
