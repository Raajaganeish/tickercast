"""Economic data from FRED (US Federal Reserve) with point-in-time availability dates.

Each value gets an `available_date`: the first trading day the market could know it.

- daily series (interest rates, spreads): the value for day D is published after the
  close or the next morning, so we make it available on D + 1 business day.
- release series (CPI, unemployment): monthly numbers are published weeks later and
  revised afterwards. We use ALFRED's first-release vintage, so the value and its date
  are exactly what the market saw on release day (no revisions, no peeking).

Output format ("long"):
    series | name | obs_date | available_date | value
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay

from tickercast.config import Config


def _daily(fred, sid: str, start: str) -> pd.DataFrame:
    s = fred.get_series(sid, observation_start=start).dropna()
    df = s.rename("value").rename_axis("obs_date").reset_index()
    df["available_date"] = df["obs_date"] + BDay(1)
    return df


def _first_release(fred, sid: str, start: str) -> pd.DataFrame:
    # One row per (observation date, vintage); keep the earliest vintage per observation.
    allr = fred.get_series_all_releases(sid)
    allr = allr.rename(columns={"date": "obs_date", "realtime_start": "available_date"})
    allr["obs_date"] = pd.to_datetime(allr["obs_date"])
    allr["available_date"] = pd.to_datetime(allr["available_date"])
    allr["value"] = pd.to_numeric(allr["value"], errors="coerce")
    first = allr.dropna(subset=["value"]).sort_values("available_date").groupby("obs_date", as_index=False).first()
    # Keep one extra year before `start` so year-over-year changes exist from day one.
    lookback = pd.Timestamp(start) - pd.DateOffset(years=1)
    return first[first["obs_date"] >= lookback][["obs_date", "available_date", "value"]]


def download_macro(cfg: Config) -> pd.DataFrame | None:
    if not cfg.fred_api_key:
        print("macro: FRED_API_KEY not set in .env -> skipping macro data (pipeline continues without it)")
        return None
    from fredapi import Fred

    fred = Fred(api_key=cfg.fred_api_key)
    frames = []
    for sid, spec in cfg["macro"]["series"].items():
        fetch = _daily if spec["kind"] == "daily" else _first_release
        df = fetch(fred, sid, cfg.start)
        df.insert(0, "series", sid)
        df.insert(1, "name", spec["name"])
        frames.append(df)
        print(f"  {sid:14s} -> {spec['name']:14s} {len(df):6,} obs ({spec['kind']})")
    macro = pd.concat(frames, ignore_index=True)
    out = cfg.path("raw") / "macro.parquet"
    macro.to_parquet(out, index=False)
    print(f"macro: {len(macro):,} rows -> {out}")
    return macro
