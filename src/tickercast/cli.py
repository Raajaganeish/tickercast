"""Command line entry point.

    tickercast download           # step 1: universe, prices, market, macro -> data/raw/
    tickercast validate-raw       #         checks -> reports/01_raw_validation.md
    tickercast features           # step 2: feature table -> data/interim/features.parquet
    tickercast validate-features  #         checks -> reports/02_feature_validation.md
    tickercast dataset            # step 3: split + scale -> data/processed/ (+ reports/03_...)
    tickercast all                # everything above in order
    tickercast sample --ticker AAPL --stage features   # one stock -> data/samples/*.csv to browse
"""

from __future__ import annotations

import argparse
import sys

from tickercast.config import load_config


def cmd_download(cfg, args):
    from tickercast.data.macro import download_macro
    from tickercast.data.prices import download_market, download_stock_prices
    from tickercast.data.universe import build_universe

    uni = build_universe(cfg)
    if args.limit:
        uni = uni[uni["ticker"].isin(["AAPL"]) | (uni.index < args.limit)]
        uni.to_parquet(cfg.path("raw") / "universe.parquet", index=False)
        print(f"  --limit: keeping {len(uni)} tickers")
    download_stock_prices(cfg, uni)
    download_market(cfg)
    download_macro(cfg)


def cmd_validate_raw(cfg, args):
    from tickercast.validation.raw_checks import run_raw_checks

    return run_raw_checks(cfg).n_fail


def cmd_features(cfg, args):
    from tickercast.features.build import run_build

    run_build(cfg)


def cmd_validate_features(cfg, args):
    from tickercast.validation.feature_checks import run_feature_checks

    return run_feature_checks(cfg, truncation=not args.skip_truncation).n_fail


def cmd_dataset(cfg, args):
    from tickercast.dataset.prepare import check_dataset, prepare_dataset

    df, meta = prepare_dataset(cfg)
    return check_dataset(cfg, df, meta).n_fail


def cmd_baselines(cfg, args):
    from tickercast.baselines.run import run_baselines

    for target, res in run_baselines(cfg).items():
        print(f"\n{target}")
        print(res[["accuracy", "train_auc", "auc", "daily_ic", "top_minus_bottom", "train_seconds"]].round(4).to_string())


def cmd_benchmark(cfg, args):
    from tickercast.models.benchmark import run_benchmark

    res = run_benchmark(cfg)
    out = cfg.path("raw").parent / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "patchtst_benchmark.csv")
    print(res.round(2).to_string())


def cmd_train(cfg, args):
    from tickercast.training.train import TrainConfig, run_training

    tcfg = TrainConfig(sizes=args.sizes.split(","), features=args.features, max_epochs=args.max_epochs,
                       target_mode=args.target_mode)
    print(run_training(cfg, tcfg)[["params", "epoch", "val_loss", "coverage", "width_vs_move_corr"]].round(4).to_string())


def cmd_evaluate(cfg, args):
    from tickercast.training.evaluate import run_evaluation

    print(run_evaluation(cfg, args.runs.split(","))[["pinball_loss", "coverage", "avg_width_pct", "width_vs_move_corr",
                                              "mid_daily_ic"]].round(4).to_string())


def cmd_viewer(cfg, args):
    from tickercast.viz.viewer import build_viewer

    build_viewer(cfg)


def cmd_sample(cfg, args):
    """Export one stock's rows to CSV so they can be opened in Excel / an IDE table viewer."""
    import pandas as pd

    stages = {"raw": cfg.path("raw") / "prices.parquet",
              "features": cfg.path("interim") / "features.parquet",
              "dataset": cfg.path("processed") / "dataset.parquet"}
    df = pd.read_parquet(stages[args.stage], filters=[("ticker", "==", args.ticker)])
    if args.start:
        df = df[df["date"] >= args.start]
    if args.end:
        df = df[df["date"] <= args.end]
    out_dir = cfg.path("raw").parent / "samples"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.ticker}_{args.stage}.csv"
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"{len(df):,} rows x {df.shape[1]} columns -> {out}")


def cmd_all(cfg, args):
    failures = 0
    for step in (cmd_download, cmd_validate_raw, cmd_features, cmd_validate_features, cmd_dataset):
        failures += step(cfg, args) or 0
    return failures


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tickercast", description="tickercast data pipeline")
    p.add_argument("--config", default=None, help="path to config YAML (default: configs/data.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("download", cmd_download), ("validate-raw", cmd_validate_raw), ("features", cmd_features),
                     ("validate-features", cmd_validate_features), ("dataset", cmd_dataset), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if name in ("download", "all"):
            sp.add_argument("--limit", type=int, default=0, help="only download the first N tickers (quick test)")
        if name in ("validate-features", "all"):
            sp.add_argument("--skip-truncation", action="store_true", help="skip the slower point-in-time test")
    sub.add_parser("baselines", help="train + score baselines on validation years").set_defaults(fn=cmd_baselines)
    sub.add_parser("benchmark", help="measure PatchTST training speed for several sizes").set_defaults(fn=cmd_benchmark)
    sp = sub.add_parser("train", help="train PatchTST range models (one per size)")
    sp.set_defaults(fn=cmd_train)
    sp.add_argument("--sizes", default="small,medium,large,xlarge", help="comma-separated: small,medium,large,xlarge")
    sp.add_argument("--features", choices=["all", "stock_only"], default="all")
    sp.add_argument("--max-epochs", type=int, default=10)
    sp.add_argument("--target-mode", choices=["vol_scaled", "raw"], default="vol_scaled",
                    help="vol_scaled: forecast in units of each stock's recent volatility (default); raw: in %%")
    sp = sub.add_parser("evaluate", help="final comparison on validation + locked test years")
    sp.set_defaults(fn=cmd_evaluate)
    sp.add_argument("--runs", default="all_vol_scaled,all_raw",
                    help="comma-separated training runs to compare (first = main), e.g. all_vol_scaled,all_raw")
    sub.add_parser("viewer", help="build reports/prediction_viewer.html (predicted range vs actual)").set_defaults(fn=cmd_viewer)
    sp = sub.add_parser("sample", help="export one stock to data/samples/<TICKER>_<stage>.csv")
    sp.set_defaults(fn=cmd_sample)
    sp.add_argument("--ticker", default="AAPL")
    sp.add_argument("--stage", choices=["raw", "features", "dataset"], default="features")
    sp.add_argument("--start", default=None, help="YYYY-MM-DD")
    sp.add_argument("--end", default=None, help="YYYY-MM-DD")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    failures = args.fn(cfg, args) or 0
    if failures:
        print(f"\n{failures} validation check(s) FAILED - see reports/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
