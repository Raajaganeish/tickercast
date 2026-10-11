"""Final comparison on validation (2020-2021) AND the locked test years (2022+).

Models compared, all predicting tomorrow's low / mid / high change in %:
    zero_band      no learning: the same band for every stock and day (the training data's
                   10th / 50th / 90th percentile moves)
    vol_band       mid = 0, low/high = ±1.28 x the stock's own 20-day volatility
                   (the classic "use recent jumpiness" rule)
    lightgbm_q     three LightGBM quantile models on today's 66 features
    lightgbm_q_stock_only  same, on the 24 stock-specific features only
    patchtst_*     best PatchTST size of each training run (chosen on validation), 128-day windows

The test set is scored here once. Nothing is tuned on it.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tickercast.config import Config
from tickercast.dataset.windows import WindowDataset
from tickercast.evaluation.range_metrics import QUANTILES, range_score
from tickercast.models.patchtst import PatchTST, PatchTSTConfig, best_device
from tickercast.training.train import TARGET, TARGET_SCALE, predict

Z90 = 1.2816  # 90th percentile of a normal distribution
LGB_Q = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=5000, feature_fraction=0.7,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)


def rows_frame(ds: WindowDataset, all_feats: pd.DataFrame) -> pd.DataFrame:
    return all_feats.iloc[ds.ends]


def run_evaluation(cfg: Config, runs: list[str]) -> pd.DataFrame:
    """runs: training run names like "all_vol_scaled" (features_targetmode). The first is the main one."""
    processed = cfg.path("processed")
    results_dir = cfg.path("raw").parent / "results"
    ckpt_dir = cfg.path("raw").parent.parent / "checkpoints"
    winners = {r: json.loads((results_dir / f"patchtst_winner_{r}.json").read_text()) for r in runs}
    meta = json.loads((processed / "meta.json").read_text())
    feats = meta["features"]
    full = pd.read_parquet(processed / "dataset.parquet",
                           columns=["date", "ticker", "sector_id", *meta["features"], TARGET])

    dsets = {s: WindowDataset(processed, s, target=TARGET, features=feats) for s in ("train", "val", "test")}
    frames = {s: full.iloc[d.ends] for s, d in dsets.items()}
    y = {s: f[TARGET].to_numpy() * TARGET_SCALE for s, f in frames.items()}
    preds: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}

    # zero_band: fixed band from training data
    lo, hi = np.quantile(y["train"], [0.1, 0.9])
    med = np.median(y["train"])
    for s in preds:
        n = len(y[s])
        preds[s]["zero_band"] = np.column_stack([np.full(n, lo), np.full(n, med), np.full(n, hi)])

    # vol_band: each stock's recent (real, unclipped) volatility
    for s in preds:
        vol = dsets[s].vol[dsets[s].ends] * TARGET_SCALE
        preds[s]["vol_band"] = np.column_stack([-Z90 * vol, np.zeros_like(vol), Z90 * vol])

    # lightgbm_q: one model per quantile, early-stopped on validation (all features, then stock-specific only)
    stock_feats = json.loads((results_dir / "stock_specific_features.json").read_text())
    for name, fcols in (("lightgbm_q", meta["features"]), ("lightgbm_q_stock_only", stock_feats)):
        cols = [*fcols, "sector_id"]
        q_out = {s: [] for s in preds}
        for q in QUANTILES:
            dtr = lgb.Dataset(frames["train"][cols], y["train"], categorical_feature=["sector_id"])
            dva = lgb.Dataset(frames["val"][cols], y["val"], reference=dtr, categorical_feature=["sector_id"])
            m = lgb.train(dict(LGB_Q, objective="quantile", alpha=q), dtr, 2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            for s in preds:
                q_out[s].append(m.predict(frames[s][cols], num_iteration=m.best_iteration))
            print(f"  {name} quantile {q}: {m.best_iteration} trees")
        for s in preds:
            # separate models can cross; sort to keep low <= mid <= high
            preds[s][name] = np.sort(np.column_stack(q_out[s]), axis=1)

    # patchtst: best size of each run
    device = best_device()
    for run, w in winners.items():
        ckpt = torch.load(ckpt_dir / f"patchtst_{run}_{w['size']}.pt", map_location="cpu", weights_only=False)
        model = PatchTST(PatchTSTConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["state_dict"])
        for s in preds:
            ds = WindowDataset(processed, s, target=TARGET, features=ckpt["features"])
            preds[s][f"patchtst_{run}_{w['size']}"] = predict(model, ds, device, w.get("target_mode", "raw"))

    rows = []
    for s in preds:
        for name, p in preds[s].items():
            rows.append({"split": s, "model": name,
                         **range_score(p[:, 0], p[:, 1], p[:, 2], y[s], frames[s]["date"].to_numpy())})
    table = pd.DataFrame(rows).set_index(["split", "model"])
    table.to_csv(results_dir / "final_range_scores.csv")

    test_pred = frames["test"][["date", "ticker"]].copy()
    test_pred["actual"] = y["test"]
    for name, p in preds["test"].items():
        test_pred[[f"{name}_low", f"{name}_mid", f"{name}_high"]] = p
    test_pred.to_parquet(results_dir / "test_predictions.parquet", index=False)
    main = runs[0]
    write_report(cfg, table, main, winners[main]["size"], results_dir, test_pred, runs)
    return table


def write_report(cfg, table, run, winner, results_dir, test_pred, runs):
    reports = cfg.path("reports")
    figs = reports / "figures"
    figs.mkdir(exist_ok=True)
    hist = pd.read_csv(results_dir / f"patchtst_history_{run}.csv")
    best = pd.read_csv(results_dir / f"patchtst_best_{run}.csv", index_col=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for size, g in hist.groupby("size", sort=False):
        axes[0].plot(g["epoch"], g["train_loss"], ls="--", label=f"{size} train")
        axes[0].plot(g["epoch"], g["val_loss"], label=f"{size} val")
        axes[1].plot(g["epoch"], g["coverage"], label=size)
        axes[2].plot(g["epoch"], g["width_vs_move_corr"], label=size)
    axes[0].set_title("Pinball loss (lower = better)"); axes[0].legend(fontsize=7)
    axes[1].axhline(0.8, color="k", lw=0.8, ls=":"); axes[1].set_title("Validation coverage (target 0.80)")
    axes[2].set_title("Validation: wide range when big moves happen? (corr)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(figs / "05_training_curves.png", dpi=120); plt.close(fig)

    p = f"patchtst_{run}_{winner}"
    a = test_pred[(test_pred["ticker"] == "AAPL") & (test_pred["date"] >= "2024-01-01") & (test_pred["date"] < "2024-07-01")]
    if len(a):
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.fill_between(a["date"], a[f"{p}_low"], a[f"{p}_high"], alpha=0.3, label="predicted range (low–high)")
        ax.plot(a["date"], a[f"{p}_mid"], lw=1, label="predicted mid")
        inside = (a["actual"] >= a[f"{p}_low"]) & (a["actual"] <= a[f"{p}_high"])
        ax.scatter(a["date"][inside], a["actual"][inside], s=10, color="#54A24B", label="actual (inside)")
        ax.scatter(a["date"][~inside], a["actual"][~inside], s=14, color="#E45756", label="actual (outside)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("next-day change (%)"); ax.legend(fontsize=8, ncol=4)
        ax.set_title(f"AAPL, first half of 2024 (test period): predicted range vs what happened ({p})")
        plt.tight_layout(); fig.savefig(figs / "05_aapl_test_ranges.png", dpi=120); plt.close(fig)

    def fmt(t):
        f = pd.DataFrame(index=t.index)
        f["pinball loss"] = t["pinball_loss"].map("{:.4f}".format)
        f["coverage"] = t["coverage"].map("{:.1%}".format)
        f["below low / above high"] = [f"{a:.1%} / {b:.1%}" for a, b in zip(t["below_low"], t["above_high"])]
        f["avg width"] = t["avg_width_pct"].map("{:.2f}%".format)
        f["width~move corr"] = t["width_vs_move_corr"].map("{:.3f}".format)
        f["mid corr"] = t["mid_corr"].map(lambda v: "—" if pd.isna(v) else f"{v:+.4f}")
        f["mid daily IC"] = t["mid_daily_ic"].map(lambda v: "—" if pd.isna(v) else f"{v:+.4f}")
        f["mid sign acc"] = t["mid_sign_accuracy"].map("{:.1%}".format)
        return f

    bfmt = best[["params", "epoch", "train_loss", "val_loss", "coverage", "width_vs_move_corr", "mid_daily_ic"]].copy()
    lines = [
        "# PatchTST range forecast: results",
        "",
        f"Task: forecast tomorrow's change as a range (low = 10th, mid = 50th, high = 90th percentile), in %. "
        f"Main run: `{run}` (inputs_target). Runs compared: {', '.join(f'`{r}`' for r in runs)}. 128-day windows. Plain-language guide: "
        "[docs/09_training.md](../docs/09_training.md).",
        "",
        f"## Training of `{run}` (validation = 2020–2021)",
        "",
        "![curves](figures/05_training_curves.png)",
        "",
        "Best epoch per size:",
        "",
        bfmt.to_markdown(floatfmt=".4f"),
        "",
        f"**Chosen size: `{winner}`** (lowest validation loss).",
        "",
        "## Validation 2020–2021",
        "",
        fmt(table.loc["val"]).to_markdown(),
        "",
        "## Test 2022 → today (scored once, never used for any decision)",
        "",
        fmt(table.loc["test"]).to_markdown(),
        "",
        "![aapl](figures/05_aapl_test_ranges.png)",
        "",
        "**How to read:** *coverage* should be ~80% with ~10% below and ~10% above. *avg width* smaller is "
        "better only if coverage holds. *width~move corr*: does the range widen when big moves happen? "
        "*mid* columns: does the middle guess say anything about direction (0 = nothing)?",
        "",
    ]
    (reports / "05_patchtst.md").write_text("\n".join(lines))
    print(f"report -> {reports / '05_patchtst.md'}")
