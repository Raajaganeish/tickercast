from __future__ import annotations

import numpy as np
import pandas as pd

from tickercast.dataset.splits import assign_splits
from tickercast.features.build import build_features, feature_columns
from tickercast.features.calendar import calendar_features
from tickercast.features.macro import macro_features


def test_targets_line_up_with_next_day_return(synthetic, cfg):
    prices, universe, market = synthetic
    df = build_features(prices, universe, market, None, cfg)
    nxt = df.groupby("ticker")["ret_1d"].shift(-1)
    ok = nxt.notna() & df["fwd_ret_1d"].notna()
    np.testing.assert_allclose(df.loc[ok, "fwd_ret_1d"], nxt[ok], atol=1e-10)
    assert set(df["label_up_1d"].dropna().unique()) <= {0.0, 1.0}


def test_warmup_rows_dropped_and_no_inf(synthetic, cfg):
    prices, universe, market = synthetic
    df = build_features(prices, universe, market, None, cfg)
    first = df.groupby("ticker")["date"].min()
    raw_first = prices.groupby("ticker")["date"].min()
    assert (first > raw_first).all()
    assert np.isfinite(df[feature_columns(df)].to_numpy(dtype="float64")[~df[feature_columns(df)].isna().to_numpy()]).all()
    assert df["rsi"].dropna().between(-0.5, 0.5).all()


def test_macro_value_appears_only_from_release_date():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    macro = pd.DataFrame({
        "series": "UNRATE", "name": "unemployment",
        "obs_date": pd.to_datetime(["2019-12-01", "2020-01-01", "2020-02-01"]),
        "available_date": pd.to_datetime(["2020-01-10", "2020-02-07", "2020-03-06"]),
        "value": [3.5, 3.6, 3.5],
    })
    f = macro_features(macro, dates, {"UNRATE": {"name": "unemployment", "kind": "release"}})
    assert np.isnan(f.loc["2020-01-09", "unemployment"])
    assert f.loc["2020-01-10", "unemployment"] == 3.5
    assert f.loc["2020-02-06", "unemployment"] == 3.5      # January's number not out yet
    assert f.loc["2020-02-07", "unemployment"] == 3.6


def test_calendar_cyclical_encoding():
    f = calendar_features(pd.DatetimeIndex(["2024-01-01", "2024-01-05", "2024-12-31"]))
    assert np.isclose(f["dow_sin"].iloc[0], 0) and np.isclose(f["dow_cos"].iloc[0], 1)   # Monday
    assert f["is_q4"].tolist() == [0, 0, 1]


def test_splits_have_embargo_gap():
    dates = pd.Series(pd.bdate_range("2019-12-01", "2020-02-28"))
    split = assign_splits(dates, "2019-12-31", "2020-01-31", embargo_days=5)
    assert (split[dates <= "2019-12-24"] == "train").all()
    assert (split[(dates > "2019-12-24") & (dates <= "2019-12-31")] == "embargo").all()
    assert split[dates == "2020-01-02"].item() == "val"
    assert split[dates == "2020-02-03"].item() == "test"
