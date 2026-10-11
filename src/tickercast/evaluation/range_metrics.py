"""Scoring for range forecasts: for each stock-day, a low / mid / high guess of tomorrow's change.

low  = 10th percentile  (only a 10% chance the real move is below this)
mid  = 50th percentile  (best single guess)
high = 90th percentile  (only a 10% chance the real move is above this)

| Metric             | Plain meaning                                                         | Ideal          |
|--------------------|-----------------------------------------------------------------------|----------------|
| pinball_loss       | Overall error for the three guesses (lower is better)                 | as low as possible |
| coverage           | Share of days where the real move landed between low and high         | 0.80           |
| below_low/above_high | Share of real moves under low / over high                           | 0.10 each      |
| avg_width_pct      | Average (high - low) in %                                             | narrow, if coverage holds |
| width_vs_move_corr | Is the range wide when big moves happen? corr(high - low, |real move|) | higher = better |
| mid_corr           | corr(mid, real move): does the middle guess say anything about direction? | > 0       |
| mid_sign_accuracy  | % of days where mid had the right sign (up/down)                      | > 50%          |
| mid_daily_ic       | Each day: rank correlation of mid with real moves across stocks       | > 0            |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def range_score(low, mid, high, y, dates=None) -> dict:
    low, mid, high, y = (np.asarray(a, dtype="float64") for a in (low, mid, high, y))
    width = high - low
    out = {
        "rows": len(y),
        "pinball_loss": np.mean([pinball(y, p, q) for p, q in zip((low, mid, high), QUANTILES)]),
        "coverage": float(((y >= low) & (y <= high)).mean()),
        "below_low": float((y < low).mean()),
        "above_high": float((y > high).mean()),
        "avg_width_pct": float(width.mean()),
        "width_vs_move_corr": float(np.corrcoef(width, np.abs(y))[0, 1]),
        "mid_corr": float(np.corrcoef(mid, y)[0, 1]) if np.ptp(mid) > 0 else float("nan"),
        "mid_sign_accuracy": float(((mid > 0) == (y > 0)).mean()),
    }
    if dates is not None and np.ptp(mid) > 0:
        df = pd.DataFrame({"date": dates, "mid": mid, "y": y})
        r = df.groupby("date")[["mid", "y"]].rank(pct=True)
        r = r - r.groupby(df["date"]).transform("mean")
        num = (r["mid"] * r["y"]).groupby(df["date"]).sum()
        den = np.sqrt((r["mid"] ** 2).groupby(df["date"]).sum() * (r["y"] ** 2).groupby(df["date"]).sum())
        out["mid_daily_ic"] = float((num / den.replace(0, np.nan)).dropna().mean())
    return out
