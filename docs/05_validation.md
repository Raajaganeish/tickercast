# 05 · Validation: every check explained

Each pipeline step writes a Markdown report in `reports/`. Status meanings:

| Status | Meaning | Action |
|---|---|---|
| ✅ PASS | As expected | None |
| ⚠️ WARN | Unusual but not necessarily wrong | Read the detail table and decide |
| ❌ FAIL | Data is broken or leaking | **Don't train.** Fix first. The CLI exits with code 1 |
| ℹ️ INFO | Statistics for context | None |

## Report 1: Raw data (`tickercast validate-raw`)

Code: [validation/raw_checks.py](../src/tickercast/validation/raw_checks.py)

| Check | What it catches | Threshold |
|---|---|---|
| Universe size / duplicates / sectors | Broken Wikipedia scrape | 480–520 tickers, 0 duplicates |
| Prices downloaded for every ticker | Delisted / renamed symbols | WARN if any missing |
| Duplicate rows | Download bugs | FAIL if > 0 |
| Missing OHLC values | Partial rows | WARN if > 0 |
| Non-positive prices | Corrupt data | FAIL if > 0 |
| OHLC consistency | `low` above `close` etc. | WARN if > 0.01% of rows |
| Volume | Negative or zero volume | FAIL if negative |
| Illiquid early history | OTC / pre-listing data | INFO: lists what the feature step trims |
| Extreme daily moves | Bad split adjustment, or real crashes | Lists \|log return\| > 0.40 for review |
| Stale prices | Same close 5+ days running = no real trading | WARN if any |
| Missing trading days | Holes inside a stock's history | WARN for gaps ≥ 5 days |
| Dates outside the trading calendar | Rows on weekends/holidays | WARN (dropped later) |
| History length / tickers per year | Short histories, survivorship | INFO |
| Market: quotes on non-trading days | e.g. VIX quoted on US holidays | WARN (dropped later) |
| Macro: available ≥ observation date | Value dated before the period it describes | FAIL if any |
| Macro: publication lags / freshness | Wrong release dates, stale series | INFO / WARN |

## Report 2: Features (`tickercast validate-features`)

Code: [validation/feature_checks.py](../src/tickercast/validation/feature_checks.py)

| Check | What it catches | Threshold |
|---|---|---|
| Unique (ticker, date) | Join bugs that duplicate rows | FAIL if > 0 |
| No infinite values | Division by zero (e.g. zero volume) | FAIL if > 0 |
| No constant features | Broken feature (all same value) | FAIL |
| Missing values per feature (by split) | Late-starting data, bugs | WARN if > 5% |
| Target alignment | Off-by-one errors in targets | FAIL on any mismatch |
| Missing targets only at series end | Gaps in price history | WARN |
| Label balance by split | Context: the "always up" baseline | INFO |
| **Leakage scan** | Feature correlates with tomorrow too strongly | WARN > 0.08, FAIL > 0.2 |
| **Point-in-time truncation test** | Any feature that uses future data | FAIL on any difference |
| Distribution drift vs train | Features whose level shifts over time (rates, prices) | WARN if mean moves > 1 train std |
| Extreme values | Outliers that would dominate training | WARN; clipped to ±5 std in step 3 |

## Report 3: Model-ready dataset (`tickercast dataset`)

Code: [dataset/prepare.py](../src/tickercast/dataset/prepare.py)

| Check | What it catches |
|---|---|
| Splits in time order, no overlap | Wrong split dates |
| Embargo | Targets from the last train days reaching into validation |
| Scaler fitted on train only | Train features should be mean ≈ 0, std ≈ 1 |
| No NaN / inf in model inputs | Values that would crash training |
| Training windows per split | Enough 128-day samples in each split |

## Latest results (full run, 499 stocks, with FRED macro)

| Report | Pass | Warn | Fail |
|---|---|---|---|
| 01 Raw | 20 | 3 | 0 |
| 02 Features | 7 | 3 | 0 |
| 03 Dataset | 5 | 0 | 0 |

Remaining warnings, all understood:
- **USO / UUP start in 2006 / 2007**: their features are missing for early rows (filled with 0 = "average" after scaling).
- **1 stale-price run** after cleaning: negligible.
- **2 VIX quotes on US holidays** (Memorial Day and Labor Day 2026): dropped in the feature step.
- **`gap` has 0.11% extreme values**: real overnight jumps (earnings, news). Clipped to ±5 std.
- **Macro levels drift** (`yield_10y`, `fed_funds`, `cpi_yoy`, …): rates and inflation in 2022+ are far above the 2005–2019 average, so the test set sits outside the range the model trained on. Expected. Phase 2 will compare models with and without raw levels (the `_chg_20d` / `_chg_3m` versions are more stable).
- **`unemployment_chg_3m` extreme values**: the COVID jobs shock in 2020 (+10 points in 3 months). Clipped to ±5 std.
- **`unemployment_chg_3m` missing for 3 weeks in Feb 2026**: the October 2025 jobs report was never published (US government shutdown), so there is no 3-month-earlier value to compare against.

## Bugs these checks already caught

| Found by | Problem | Fix |
|---|---|---|
| Stale-price check | 527 runs of identical prices, mostly SW / FERG / AMCR OTC history | Illiquid-history trim ([02_universe.md](02_universe.md)) |
| FRED coverage | ICE high-yield spread (`BAMLH0A0HYM2`) returned only ~3 years of history (FRED licensing limit) | Replaced with Moody's Baa spread (`BAA10Y`), daily since 1986 |
| Missing values by split | SPY/QQQ/… returns missing on 2 days in the test period; `beta_spy` missing for 60 days after each | Yahoo had VIX quotes on US holidays → phantom dates. Market data is now restricted to days SPY traded |
