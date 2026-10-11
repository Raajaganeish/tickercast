"""Build reports/prediction_viewer.html: a self-contained page comparing predicted ranges with real moves.

Data comes from `tickercast evaluate` (data/results/test_predictions.parquet, test years 2022+).
Each prediction is made at the close of day t for the move to the close of day t+1; the page
plots it on day t+1, the day the move happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tickercast.config import Config

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "ORCL", "CRM",
           "ADBE", "INTC", "CSCO", "IBM", "QCOM", "JPM", "BAC", "GS", "V", "MA", "UNH", "LLY", "JNJ", "MRK",
           "ABBV", "PFE", "XOM", "CVX", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "DIS", "BA",
           "CAT", "T"]
MODELS = {
    "patchtst_stock_only_vol_scaled_small": "Transformer (PatchTST)",
    "lightgbm_q_stock_only": "LightGBM",
    "vol_band": "Volatility rule",
}
TEMPLATE = Path(__file__).with_name("viewer_template.html")


def build_viewer(cfg: Config) -> Path:
    results = cfg.path("raw").parent / "results"
    tp = pd.read_parquet(results / "test_predictions.parquet")
    tp = tp[tp["ticker"].isin(TICKERS)]
    prices = pd.read_parquet(cfg.path("raw") / "prices.parquet", columns=["date", "ticker", "close"])
    prices = prices[prices["ticker"].isin(TICKERS)].sort_values(["ticker", "date"])
    prices["move_date"] = prices.groupby("ticker")["date"].shift(-1)
    prices["move_close"] = prices.groupby("ticker")["close"].shift(-1)
    tp = tp.merge(prices[["date", "ticker", "move_date", "move_close"]], on=["date", "ticker"], how="left")
    tp = tp.dropna(subset=["move_date"]).sort_values(["ticker", "move_date"])

    calendar = sorted(tp["move_date"].unique())
    pos = {d: i for i, d in enumerate(calendar)}
    names = pd.read_parquet(cfg.path("raw") / "universe.parquet").set_index("ticker")["name"]

    stocks = {}
    for t, g in tp.groupby("ticker"):
        rec = {"name": str(names.get(t, t)),
               "d": [pos[d] for d in g["move_date"]],
               "close": np.round(g["move_close"].to_numpy(), 2).tolist(),
               "actual": np.round(g["actual"].to_numpy() * 100).astype(int).tolist()}   # basis points
        for key in MODELS:
            rec[key] = [np.round(g[f"{key}_{q}"].to_numpy() * 100).astype(int).tolist() for q in ("low", "mid", "high")]
        stocks[t] = rec

    scores = pd.read_csv(results / "final_range_scores.csv")
    scores = scores[(scores["split"] == "test") & scores["model"].isin(MODELS)].set_index("model")
    overall = {m: {k: round(float(scores.loc[m, k]), 4) for k in
                   ("pinball_loss", "coverage", "avg_width_pct", "width_vs_move_corr", "mid_sign_accuracy")}
               for m in MODELS if m in scores.index}

    data = {"calendar": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in calendar], "models": MODELS,
            "overall": overall, "stocks": stocks, "n_all_stocks": int(pd.read_parquet(
                results / "test_predictions.parquet", columns=["ticker"])["ticker"].nunique())}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = cfg.path("reports") / "prediction_viewer.html"
    out.write_text(html)
    print(f"viewer: {len(stocks)} stocks, {len(calendar)} days -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
