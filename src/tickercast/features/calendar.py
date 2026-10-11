"""Calendar features. They depend only on the date, so they are known in advance.

Cyclical values use sin/cos so the model sees that Friday is next to Monday and
December is next to January.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _third_friday(d: pd.Timestamp) -> pd.Timestamp:
    first = d.replace(day=1)
    offset = (4 - first.weekday()) % 7  # Friday = 4
    return first + pd.Timedelta(days=offset + 14)


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    f["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    quarter_end = dates + pd.offsets.QuarterEnd(0)
    f["days_to_quarter_end"] = ((quarter_end - dates).days.to_numpy()) / 92
    f["is_q4"] = (month >= 10).astype(float)

    # Monthly options expiry (third Friday): the week ending on it often trades differently.
    third_fri = pd.DatetimeIndex([_third_friday(d) for d in dates])
    days_to_opex = (third_fri - dates).days.to_numpy()
    f["opex_week"] = ((days_to_opex >= 0) & (days_to_opex <= 4)).astype(float)
    return f
