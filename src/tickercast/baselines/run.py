"""Train and score the baselines on the validation years (2020-2021).

The test years (2022+) are NOT touched here. They are scored once, at the end, for all
models together, so no decision is ever tuned on them.

Two targets are compared:
    label_up_1d          will the stock close higher tomorrow?
    label_outperform_1d  will the stock beat the S&P 500 (SPY) tomorrow?

and two input sets:
    all         all 66 features
    stock_only  only features that differ between stocks on the same day (24 features).
                Market-wide columns (SPY, VIX, oil, rates, calendar...) are identical for all
                ~500 stocks on a day. Together they act like a fingerprint for each date, which
                lets a model memorize "the market went up on this date" instead of learning
                patterns. See docs/08_baselines.md.

All models use exactly the rows PatchTST will predict (stock-days with >= 128 days of
history), so the comparison is fair.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tickercast.config import Config
from tickercast.dataset.windows import valid_window_ends
from tickercast.evaluation.metrics import score, score_by_year

TARGETS = {"label_up_1d": "fwd_ret_1d", "label_outperform_1d": "fwd_excess_1d"}
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 5000,      # stock data is noisy: force each leaf to cover many examples
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "seed": 0,
}
MACRO_PREFIXES = ("yield_", "curve_", "credit_", "fed_", "cpi_", "unemployment")


def stock_specific_features(df: pd.DataFrame, feats: list[str]) -> list[str]:
    """Features that take different values for different stocks on the same day."""
    sample = df[df["split"] == "train"].sample(min(200_000, len(df)), random_state=0)
    varies = sample.groupby("date")[feats].nunique().median()
    return [f for f in feats if varies[f] > 1]


def load(cfg: Config):
    processed = cfg.path("processed")
    meta = json.loads((processed / "meta.json").read_text())
    feats = meta["features"]
    df = pd.read_parquet(processed / "dataset.parquet",
                         columns=["date", "ticker", "split", "sector_id", *feats, *TARGETS, *TARGETS.values()])
    return df, meta


def unscale(part: pd.DataFrame, meta: dict, col: str) -> np.ndarray:
    s = meta["scaler"]
    return part[col].to_numpy() * s["std"][col] + s["mean"][col]


def fit_lightgbm(train, val, cols, target):
    cols = [*cols, "sector_id"]
    dtrain = lgb.Dataset(train[cols], train[target], categorical_feature=["sector_id"], free_raw_data=False)
    dval = lgb.Dataset(val[cols], val[target], reference=dtrain, categorical_feature=["sector_id"])
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
    n = model.best_iteration
    return model, model.predict(train[cols], num_iteration=n), model.predict(val[cols], num_iteration=n)


def run_target(df, meta, target, stock_feats):
    ret_col = TARGETS[target]
    feats = meta["features"]
    train = df.iloc[valid_window_ends(df, meta["window"], target, "train")]
    val = df.iloc[valid_window_ends(df, meta["window"], target, "val")]
    y_tr, y_va = train[target].to_numpy(), val[target].to_numpy()

    preds, train_auc, timing, notes = {}, {}, {}, {}

    # Rule baselines. For "beat the market", "today" means today's return minus SPY's.
    today = unscale(val, meta, "ret_1d" if target == "label_up_1d" else "rel_spy_ret_1d")
    majority = float(y_tr.mean() > 0.5)
    preds["majority_class"] = np.full(len(val), 0.5 + (1e-6 if majority else -1e-6))
    notes["majority_class"] = "always up" if majority else "always 'does not beat market'"
    preds["same_as_today"] = (today > 0).astype(float)
    notes["same_as_today"] = "momentum rule"
    preds["opposite_of_today"] = (today <= 0).astype(float)
    notes["opposite_of_today"] = "bounce-back rule"
    for k in preds:
        timing[k] = 0.0

    for fs_name, cols in (("all", feats), ("stock_only", stock_feats)):
        name = f"logistic_{fs_name}"
        t = time.time()
        lr = LogisticRegression(max_iter=300, C=0.1).fit(train[cols].to_numpy(), y_tr)
        preds[name] = lr.predict_proba(val[cols].to_numpy())[:, 1]
        train_auc[name] = roc_auc_score(y_tr, lr.predict_proba(train[cols].to_numpy())[:, 1])
        timing[name], notes[name] = time.time() - t, f"{len(cols)} features"

    variants = {
        "lightgbm_all": feats,
        "lightgbm_no_macro": [f for f in feats if not f.startswith(MACRO_PREFIXES)],
        "lightgbm_stock_only": stock_feats,
    }
    importance = {}
    for name, cols in variants.items():
        t = time.time()
        model, p_tr, p_va = fit_lightgbm(train, val, cols, target)
        preds[name] = p_va
        train_auc[name] = roc_auc_score(y_tr, p_tr)
        timing[name] = time.time() - t
        notes[name] = f"{len(cols)} features + sector, {model.best_iteration} trees"
        imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
        importance[name] = (imp / imp.sum()).sort_values(ascending=False)
        print(f"  {target} {name}: {timing[name]:.0f}s, {model.best_iteration} trees")

    r, d = val[ret_col].to_numpy(), val["date"].to_numpy()
    res = pd.DataFrame({k: score(p, y_va, r, d) for k, p in preds.items()}).T
    res.insert(1, "train_auc", pd.Series(train_auc))
    res["train_seconds"] = pd.Series(timing)
    res["notes"] = pd.Series(notes)
    by_year = {k: score_by_year(p, y_va, d) for k, p in preds.items()}
    out_preds = pd.DataFrame({"date": d, "ticker": val["ticker"].to_numpy(), "label": y_va, ret_col: r, **preds})
    return res, by_year, importance, out_preds, len(train), len(val)


def run_baselines(cfg: Config) -> dict[str, pd.DataFrame]:
    df, meta = load(cfg)
    stock_feats = stock_specific_features(df, meta["features"])
    print(f"baselines: {len(stock_feats)} stock-specific features, "
          f"{len(meta['features']) - len(stock_feats)} market-wide")

    out = cfg.path("raw").parent / "results"
    out.mkdir(exist_ok=True)
    all_results = {}
    sections = []
    for target in TARGETS:
        res, by_year, importance, preds, n_tr, n_va = run_target(df, meta, target, stock_feats)
        preds.to_parquet(out / f"baselines_val_{target}.parquet", index=False)
        res.to_csv(out / f"baselines_val_scores_{target}.csv")
        all_results[target] = res
        sections.append((target, res, by_year, importance, n_tr, n_va))
    (out / "stock_specific_features.json").write_text(json.dumps(stock_feats, indent=2))
    write_report(cfg, sections, stock_feats, meta["features"])
    return all_results


def _fmt(res: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=res.index)
    f["val accuracy"] = (res["accuracy"] * 100).map("{:.2f}%".format)
    f["val AUC"] = res["auc"].map("{:.4f}".format)
    f["train AUC"] = res["train_auc"].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}")
    f["daily IC"] = res["daily_ic"].map(lambda v: "—" if pd.isna(v) else f"{v:+.4f}")
    f["top − bottom 10% / day"] = (res["top_minus_bottom"] * 100).map(lambda v: "—" if pd.isna(v) else f"{v:+.3f}%")
    f["seconds"] = res["train_seconds"].map("{:.0f}".format)
    f["notes"] = res["notes"]
    return f


def write_report(cfg, sections, stock_feats, feats):
    reports = cfg.path("reports")
    figs = reports / "figures"
    figs.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, (target, res, *_rest) in zip(axes, sections):
        ic = res["daily_ic"].fillna(0).sort_values()
        ic.plot.barh(ax=ax, color=["#E45756" if v <= 0 else "#4C78A8" for v in ic])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{target}: daily IC on validation\n(stock-picking skill; 0 = none)")
    plt.tight_layout(); fig.savefig(figs / "04_baselines_ic.png", dpi=120); plt.close(fig)

    best_target, best_res, _, best_imp, *_ = sections[1]
    best = best_res.loc[list(best_imp), "daily_ic"].idxmax()  # best tree model (only those have importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    (best_imp[best].head(20)[::-1] * 100).plot.barh(ax=ax, color="#54A24B")
    ax.set_title(f"{best} ({best_target}): top 20 features, % of gain")
    plt.tight_layout(); fig.savefig(figs / "04_best_importance.png", dpi=120); plt.close(fig)

    shared = [f for f in feats if f not in stock_feats]
    lines = [
        "# Baselines (validation years 2020–2021)",
        "",
        "Trained on 2005–2019, scored on 2020–2021. Test years (2022+) untouched. "
        "Plain-language guide: [docs/08_baselines.md](../docs/08_baselines.md).",
        "",
        "**How to read:** *val AUC* 0.50 = random. *train AUC* far above val AUC = memorizing. "
        "*daily IC* = each day, do the stocks the model ranks highest actually do better? (0 = no skill; "
        "0.02–0.05 is good for daily data). *top − bottom 10%* = average next-day return of the 10% of stocks the "
        "model likes most minus the 10% it likes least, before trading costs.",
        "",
        "![ic](figures/04_baselines_ic.png)",
        "",
    ]
    for target, res, by_year, importance, n_tr, n_va in sections:
        title = "Will it go up tomorrow?" if target == "label_up_1d" else "Will it beat the market tomorrow?"
        yr = pd.DataFrame({k: v["auc"] for k, v in by_year.items()}).map("{:.4f}".format)
        lines += [
            f"## Target `{target}`: {title}",
            "",
            f"{n_tr:,} training rows · {n_va:,} validation rows · share of 1s in validation: "
            f"{by_year['majority_class']['share_up'].mean():.2%}",
            "",
            _fmt(res.sort_values("daily_ic", ascending=False, na_position="last")).to_markdown(),
            "",
            "AUC by year:",
            "",
            yr.to_markdown(),
            "",
        ]
    lines += [
        f"## What the best model relied on (`{best}`, `{best_target}`)",
        "",
        "![importance](figures/04_best_importance.png)",
        "",
        (best_imp[best].head(20) * 100).round(2).rename("% of gain").to_frame().to_markdown(),
        "",
        f"## Feature groups used",
        "",
        f"**Stock-specific ({len(stock_feats)}):** " + ", ".join(f"`{f}`" for f in stock_feats),
        "",
        f"**Market-wide ({len(shared)}), same value for every stock on a day:** " + ", ".join(f"`{f}`" for f in shared),
        "",
    ]
    (reports / "04_baselines.md").write_text("\n".join(lines))
    print(f"report -> {reports / '04_baselines.md'}")
