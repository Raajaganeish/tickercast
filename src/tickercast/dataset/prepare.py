"""Turn the feature table into a model-ready dataset.

1. Assign train / val / test / embargo by date.
2. Fit scaling (mean, std) on TRAIN rows only, apply everywhere, clip to ±5 std.
3. Fill remaining gaps with 0 (= the training average after scaling).
4. Encode ticker and sector as integer IDs (for embedding layers).

Outputs:
    data/processed/dataset.parquet   one row per (ticker, date), scaled features + targets
    data/processed/meta.json         feature list, scaler stats, ID vocabularies, split dates
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tickercast.config import Config
from tickercast.dataset.splits import SPLITS, assign_splits
from tickercast.features.build import feature_columns, target_columns
from tickercast.validation.report import Report

CLIP = 5.0


def prepare_dataset(cfg: Config) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(cfg.path("interim") / "features.parquet")
    feats, targets = feature_columns(df), target_columns(df)
    sp = cfg["splits"]

    df["split"] = assign_splits(df["date"], sp["train_end"], sp["val_end"], sp["embargo_days"])
    train = df["split"] == "train"

    mean = df.loc[train, feats].mean()
    std = df.loc[train, feats].std().replace(0, 1.0)
    df[feats] = ((df[feats] - mean) / std).clip(-CLIP, CLIP).fillna(0.0).astype("float32")

    tickers = sorted(df["ticker"].unique())
    sectors = sorted(df["sector"].unique())
    df["ticker_id"] = df["ticker"].map({t: i for i, t in enumerate(tickers)}).astype("int32")
    df["sector_id"] = df["sector"].map({s: i for i, s in enumerate(sectors)}).astype("int32")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    meta = {
        "features": feats,
        "targets": targets,
        "scaler": {"mean": mean.astype(float).round(8).to_dict(), "std": std.astype(float).round(8).to_dict(), "clip": CLIP},
        "tickers": tickers,
        "sectors": sectors,
        "splits": sp,
        "window": cfg["dataset"]["window"],
        "default_target": cfg["dataset"]["target"],
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
    }
    out = cfg.path("processed")
    df.to_parquet(out / "dataset.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"dataset: {len(df):,} rows, {len(feats)} features -> {out / 'dataset.parquet'}")
    return df, meta


def check_dataset(cfg: Config, df: pd.DataFrame, meta: dict) -> Report:
    from tickercast.dataset.windows import valid_window_ends

    feats, target = meta["features"], meta["default_target"]
    r = Report("Model-ready dataset validation", "Checks on data/processed/dataset.parquet.")

    spans = df[df["split"].isin(SPLITS)].groupby("split")["date"].agg(["min", "max", "count"]).reindex(SPLITS)
    order_ok = spans["max"]["train"] < spans["min"]["val"] and spans["max"]["val"] < spans["min"]["test"]
    r.add("Splits are in time order, no overlap", "PASS" if order_ok else "FAIL", "train < val < test", spans.reset_index())

    calendar = pd.DatetimeIndex(sorted(df["date"].unique()))
    horizon = max(cfg["targets"]["horizons"])
    leaks = []
    for a, b in (("train", "val"), ("val", "test")):
        last_a = spans["max"][a]
        future_pos = calendar.get_loc(last_a) + horizon
        target_date = calendar[min(future_pos, len(calendar) - 1)]
        if target_date >= spans["min"][b]:
            leaks.append(f"{a}->{b}")
    r.add("Embargo: targets never reach into the next split", "PASS" if not leaks else "FAIL",
          f"longest target = {horizon} days, embargo = {cfg['splits']['embargo_days']} days; violations: {leaks or 'none'}")

    tr = df.loc[df["split"] == "train", feats]
    m, s = tr.mean().abs().max(), (tr.std() - 1).abs().max()
    r.add("Scaler fitted on train only", "PASS" if m < 0.1 and s < 0.25 else "WARN",
          f"train features: max |mean| = {m:.3f} (≈0), max |std - 1| = {s:.3f} (≈0, clipping shrinks it slightly)")

    vals = df[feats].to_numpy()
    bad = int((~np.isfinite(vals)).sum())
    r.add("No NaN / inf in model inputs", "PASS" if bad == 0 else "FAIL", f"{bad} non-finite values")

    rows = []
    for s in SPLITS:
        ends = valid_window_ends(df, meta["window"], target, s)
        y = df[target].to_numpy()[ends]
        rows.append({"split": s, "windows": len(ends), "tickers": df.iloc[ends]["ticker"].nunique(), "share_up": y.mean()})
    tbl = pd.DataFrame(rows)
    r.add("Training windows per split", "PASS" if (tbl["windows"] > 0).all() else "FAIL",
          f"{meta['window']}-day windows: " + ", ".join(f"{a}={b:,}" for a, b in zip(tbl['split'], tbl['windows'])), tbl)
    r.save(cfg.path("reports") / "03_dataset_validation.md")
    return r
