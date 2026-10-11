"""Serve (128 days x features) windows to PyTorch.

A sample ending on row t of a stock:
    x          = that stock's features for rows t-127 ... t      shape (window, n_features)
    y          = target on row t (e.g. did the price go up on day t+1?)
    ticker_id, sector_id = integers for embedding layers

The window may start in an earlier split. That is fine: those rows are past data the
model would have had anyway. Only the row the label belongs to decides the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def valid_window_ends(df: pd.DataFrame, window: int, target: str, split: str) -> np.ndarray:
    """Row positions t where: t is in `split`, the target exists, and the stock has >= window rows up to t."""
    pos_in_ticker = df.groupby("ticker").cumcount().to_numpy()
    ok = (df["split"].to_numpy() == split) & df[target].notna().to_numpy() & (pos_in_ticker >= window - 1)
    return np.flatnonzero(ok)


class WindowDataset:
    """torch.utils.data.Dataset-compatible (no torch import needed to build it)."""

    def __init__(self, processed_dir: str | Path, split: str, target: str | None = None, window: int | None = None,
                 features: list[str] | None = None):
        processed_dir = Path(processed_dir)
        self.meta = json.loads((processed_dir / "meta.json").read_text())
        df = pd.read_parquet(processed_dir / "dataset.parquet")
        self.target = target or self.meta["default_target"]
        self.window = window or self.meta["window"]
        self.features = features or self.meta["features"]   # optional subset, e.g. stock-specific only

        self.X = df[self.features].to_numpy(dtype=np.float32, copy=True)  # writable, so torch can wrap it
        self.y = df[self.target].to_numpy(dtype=np.float32)
        self.ticker_id = df["ticker_id"].to_numpy(dtype=np.int64)
        self.sector_id = df["sector_id"].to_numpy(dtype=np.int64)
        self.dates = df["date"].to_numpy()
        # real 20-day volatility on each row (log-return std), floored so calm days don't divide by ~0
        self.vol = np.maximum(df["vol_20d_raw"].to_numpy(dtype=np.float32), 0.005) if "vol_20d_raw" in df else None
        self.ends = valid_window_ends(df, self.window, self.target, split)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int):
        t = self.ends[i]
        x = self.X[t - self.window + 1 : t + 1]
        return x, self.y[t], self.ticker_id[t], self.sector_id[t]

    def batch(self, idx: np.ndarray):
        """Many samples at once (fast path for training loops): x has shape (len(idx), window, n_features)."""
        t = self.ends[idx]
        rows = t[:, None] + np.arange(-self.window + 1, 1)[None, :]
        return self.X[rows], self.y[t], self.ticker_id[t], self.sector_id[t]
