"""Economy features from FRED, aligned to trading days without peeking.

For every trading day t we take the latest value whose `available_date` <= t
(an "as-of" join), so a CPI number released on April 10 first appears on April 10's row.
"""

from __future__ import annotations

import pandas as pd


def _asof(dates: pd.DatetimeIndex, s: pd.DataFrame) -> pd.Series:
    # merge_asof needs identical datetime resolution on both sides (parquet may load ms vs us).
    s = s.assign(available_date=s["available_date"].astype("datetime64[ns]")).sort_values("available_date")
    left = pd.DataFrame({"date": dates.astype("datetime64[ns]")})
    merged = pd.merge_asof(left, s[["available_date", "value"]], left_on="date", right_on="available_date")
    return pd.Series(merged["value"].to_numpy(), index=dates)


def macro_features(macro: pd.DataFrame | None, dates: pd.DatetimeIndex, specs: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)
    if macro is None or macro.empty:
        return f

    for sid, spec in specs.items():
        s = macro[macro["series"] == sid]
        if s.empty:
            continue
        name = spec["name"]
        if spec["kind"] == "daily":
            level = _asof(dates, s)
            f[name] = level
            f[f"{name}_chg_20d"] = level - level.shift(20)
        else:
            # Year-over-year change computed on first-release values, made available on
            # the release date of the newer observation.
            s = s.sort_values("obs_date").copy()
            prev = s.set_index("obs_date")["value"]
            s["yoy"] = s["value"].to_numpy() / prev.reindex(s["obs_date"] - pd.DateOffset(years=1)).to_numpy() - 1
            s["chg_3m"] = s["value"].to_numpy() - prev.reindex(s["obs_date"] - pd.DateOffset(months=3)).to_numpy()
            if name == "cpi":
                f["cpi_yoy"] = _asof(dates, s.assign(value=s["yoy"]))
            else:
                f[name] = _asof(dates, s)
                f[f"{name}_chg_3m"] = _asof(dates, s.assign(value=s["chg_3m"]))
    return f
