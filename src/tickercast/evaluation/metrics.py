"""How we score a model's predictions.

Every model outputs, for each (stock, day), a probability that the stock goes up the next day.

| Metric          | Plain meaning                                                          | Random guess |
|-----------------|------------------------------------------------------------------------|--------------|
| accuracy        | % of days where "prob > 0.5" matched what happened                     | ~50%         |
| auc             | How well it ranks up-days above down-days (ignores the 0.5 cut-off)    | 0.50         |
| logloss         | Penalty for confident wrong answers (lower is better)                  | ~0.693       |
| daily_ic        | Each day: rank correlation between prediction and next-day return      | 0.00         |
| top_minus_bottom| Each day: avg next-day return of the 10% highest-ranked stocks minus   | 0.00%        |
|                 | the 10% lowest-ranked (what a simple "buy top, avoid bottom" would see)|              |
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def _daily_ic(df: pd.DataFrame) -> float:
    r = df.groupby("date")[["prob", "fwd_ret"]].rank(pct=True)
    r = r - r.groupby(df["date"]).transform("mean")
    num = (r["prob"] * r["fwd_ret"]).groupby(df["date"]).sum()
    den = np.sqrt((r["prob"] ** 2).groupby(df["date"]).sum() * (r["fwd_ret"] ** 2).groupby(df["date"]).sum())
    ic = (num / den.replace(0, np.nan)).dropna()
    return float(ic.mean()) if len(ic) else float("nan")


def _top_minus_bottom(df: pd.DataFrame, q: float = 0.1) -> float:
    pct = df.groupby("date")["prob"].rank(pct=True, method="first")
    top = df.loc[pct > 1 - q].groupby("date")["fwd_ret"].mean()
    bot = df.loc[pct <= q].groupby("date")["fwd_ret"].mean()
    spread = (top - bot).dropna()
    return float(spread.mean()) if len(spread) else float("nan")


def score(prob: np.ndarray, label: np.ndarray, fwd_ret: np.ndarray, dates: np.ndarray) -> dict:
    prob = np.asarray(prob, dtype="float64")
    label = np.asarray(label, dtype="float64")
    constant = np.ptp(prob) < 1e-12
    df = pd.DataFrame({"date": dates, "prob": prob, "fwd_ret": fwd_ret})
    return {
        "rows": len(prob),
        "accuracy": float(((prob > 0.5) == (label == 1)).mean()),
        "auc": 0.5 if constant else float(roc_auc_score(label, prob)),
        "logloss": float(log_loss(label, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "daily_ic": float("nan") if constant else _daily_ic(df),
        "top_minus_bottom": float("nan") if constant else _top_minus_bottom(df),
    }


def score_by_year(prob: np.ndarray, label: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"year": pd.DatetimeIndex(dates).year, "prob": prob, "label": label})
    rows = []
    for y, g in df.groupby("year"):
        constant = np.ptp(g["prob"]) < 1e-12
        rows.append({
            "year": y,
            "accuracy": ((g["prob"] > 0.5) == (g["label"] == 1)).mean(),
            "share_up": g["label"].mean(),
            "auc": 0.5 if constant else roc_auc_score(g["label"], g["prob"]),
        })
    return pd.DataFrame(rows).set_index("year")
