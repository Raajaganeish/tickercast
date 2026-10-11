"""The most important tests: features must never use future data.

We also prove the detector works by planting a deliberate leak and checking it is caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tickercast.features.build as build_mod
from tickercast.features.build import build_features, feature_columns


def _truncation_diffs(prices, universe, market, cfg, cutoff) -> pd.Series:
    c = pd.Timestamp(cutoff)
    full = build_features(prices, universe, market, None, cfg)
    cut = build_features(prices[prices["date"] <= c], universe, market[market["date"] <= c], None, cfg)
    feats = feature_columns(full)
    a = full[full["date"] <= c].set_index(["ticker", "date"])[feats].sort_index()
    b = cut.set_index(["ticker", "date"])[feats].sort_index()
    a, b = a.align(b, join="inner")
    assert len(a) > 1000
    diff = ~np.isclose(a.to_numpy("float64"), b.to_numpy("float64"), rtol=1e-6, atol=1e-9, equal_nan=True)
    return pd.Series(diff.sum(0), index=feats)


def test_features_are_point_in_time(synthetic, cfg):
    prices, universe, market = synthetic
    diffs = _truncation_diffs(prices, universe, market, cfg, "2016-06-30")
    assert diffs.sum() == 0, f"features changed when future data was removed: {diffs[diffs > 0].to_dict()}"


def test_truncation_test_catches_a_planted_leak(synthetic, cfg, monkeypatch):
    prices, universe, market = synthetic
    original = build_mod.stock_features

    def leaky(p, fcfg):
        f = original(p, fcfg)
        # tomorrow's return disguised as a feature
        f["leak"] = f["ret_1d"].groupby(p["ticker"]).shift(-1)
        return f

    monkeypatch.setattr(build_mod, "stock_features", leaky)
    diffs = _truncation_diffs(prices, universe, market, cfg, "2016-06-30")
    assert diffs["leak"] > 0
    assert diffs.drop("leak").sum() == 0


def test_leakage_scan_flags_a_planted_leak(synthetic, cfg):
    from tickercast.validation.feature_checks import _daily_rank_ic

    prices, universe, market = synthetic
    df = build_features(prices, universe, market, None, cfg)
    df["leak"] = df["fwd_ret_1d"] + np.random.default_rng(0).normal(0, 0.005, len(df))
    ic = _daily_rank_ic(df, ["ret_5d", "leak"], "fwd_ret_1d")
    assert ic["leak"] > 0.5
    assert abs(ic["ret_5d"]) < 0.2
