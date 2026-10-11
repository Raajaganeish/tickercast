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
