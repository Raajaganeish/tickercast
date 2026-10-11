# Raw data validation

_Generated 2026-10-11 14:02_

Checks on downloaded data before features are built. See docs/05_validation.md for what each check means.

**Summary:** 18 passed · 4 warnings · 0 failed · 4 info

| Check | Status | Result |
|---|---|---|
| Universe size | ✅ PASS | 503 tickers (S&P 500 has ~503 share classes) |
| Universe duplicates | ✅ PASS | 0 duplicated tickers |
| Sector assigned | ✅ PASS | 0 tickers without a sector |
| Prices downloaded for every ticker | ✅ PASS | 0 missing |
| Sector mix | ℹ️ INFO | 11 sectors |
| Prices: duplicate rows | ✅ PASS | 0 duplicated (ticker, date) |
| Prices: missing OHLC values | ✅ PASS | 0 rows with NaN |
| Prices: non-positive prices | ✅ PASS | 0 rows with price <= 0 |
| Prices: OHLC consistency (low <= open,close <= high) | ✅ PASS | 1 inconsistent rows (0.000%) |
| Prices: volume | ✅ PASS | 0 negative, 0.25% zero-volume days |
| Prices: extreme daily moves | ✅ PASS | 86 days with |log return| > 0.4 (real crashes or bad split data; review list) |
| Prices: illiquid early history (trimmed in feature step) | ℹ️ INFO | 37 tickers have early history trading < $1,000,000/day; 30,766 rows (1.2%) will be dropped |
| Prices: stale prices | ⚠️ WARN | 1 runs of >= 5 identical closes |
| Prices: missing trading days | ✅ PASS | 1 missing days in total; 0 tickers with a gap >= 5 days |
| Prices: dates outside trading calendar | ✅ PASS | 0 rows on days SPY did not trade (dropped during feature build) |
| Prices: history length | ℹ️ INFO | median 5477 days; 9 tickers with < 3 years |
| Prices: survivorship check (tickers per year) | ℹ️ INFO | 380 tickers in 2005 -> 503 in 2026. Fewer early tickers = today's members that were not public yet; failed companies are missing entirely (survivorship bias). |
| Market: all benchmarks downloaded | ✅ PASS | missing: none |
| Market: history starts at config start | ⚠️ WARN | 2 benchmarks start later: USO (2006-04-10), UUP (2007-03-01) |
| Market: quotes on non-trading days | ⚠️ WARN | 2 dates with quotes but no SPY trading (dropped in feature step): 2026-05-25, 2026-09-07 |
| Market: duplicate rows | ✅ PASS | 0 duplicated (ticker, date) |
| Market: missing OHLC values | ✅ PASS | 0 rows with NaN |
| Market: non-positive prices | ✅ PASS | 0 rows with price <= 0 |
| Market: OHLC consistency (low <= open,close <= high) | ✅ PASS | 0 inconsistent rows (0.000%) |
| Market: volume | ✅ PASS | 0 negative, 0.00% zero-volume days |
| Macro: data present | ⚠️ WARN | no macro.parquet (FRED_API_KEY not set). Macro features will be absent. |

## Sector mix

|    | sector                 |   tickers |
|---:|:-----------------------|----------:|
|  0 | Industrials            |        83 |
|  1 | Financials             |        76 |
|  2 | Information Technology |        75 |
|  3 | Health Care            |        60 |
|  4 | Consumer Discretionary |        47 |
|  5 | Consumer Staples       |        34 |
|  6 | Utilities              |        31 |
|  7 | Real Estate            |        30 |
|  8 | Materials              |        24 |
|  9 | Communication Services |        22 |
| 10 | Energy                 |        21 |


## Prices: extreme daily moves

|    | ticker   | date                |   close |   log_return |
|---:|:---------|:--------------------|--------:|-------------:|
|  0 | MRNA     | 2026-08-19 00:00:00 | 174.4   |       1.019  |
|  1 | AIG      | 2008-09-15 00:00:00 |  58.7   |      -0.9363 |
|  2 | STT      | 2009-01-20 00:00:00 |  10.02  |      -0.8925 |
|  3 | APA      | 2020-03-09 00:00:00 |   8.113 |      -0.7736 |
|  4 | JCI      | 2007-07-02 00:00:00 |  16.45  |      -0.7646 |
|  5 | GL       | 2024-04-11 00:00:00 |  48.11  |      -0.758  |
|  6 | TRGP     | 2020-03-09 00:00:00 |  11.47  |      -0.7531 |
|  7 | PCG      | 2019-01-14 00:00:00 |   8.202 |      -0.7415 |
|  8 | OXY      | 2020-03-09 00:00:00 |  11.6   |      -0.7343 |
|  9 | HIG      | 2008-10-30 00:00:00 |   6.516 |      -0.7249 |
| 10 | HIG      | 2008-12-05 00:00:00 |  10.35  |       0.7049 |
| 11 | MS       | 2008-10-13 00:00:00 |  12.19  |       0.6259 |
| 12 | AIG      | 2008-09-17 00:00:00 |  25.28  |      -0.6039 |
| 13 | FANG     | 2020-03-09 00:00:00 |  20.78  |      -0.5914 |
| 14 | LVS      | 2008-10-29 00:00:00 |   5.576 |       0.5878 |
| 15 | FISV     | 2025-10-29 00:00:00 |  70.6   |      -0.5806 |
| 16 | FITB     | 2008-09-29 00:00:00 |   5.474 |      -0.5732 |
| 17 | TKO      | 2014-05-16 00:00:00 |   9.121 |      -0.5701 |
| 18 | CVNA     | 2022-12-07 00:00:00 |   0.766 |      -0.5607 |
| 19 | PCG      | 2019-01-24 00:00:00 |  13.65  |       0.5573 |
| 20 | BIIB     | 2005-02-28 00:00:00 |  38.65  |      -0.5543 |
| 21 | BE       | 2019-08-13 00:00:00 |   4.6   |      -0.5534 |
| 22 | PNC      | 2009-01-20 00:00:00 |  13.67  |      -0.5344 |
| 23 | MOS      | 2008-10-02 00:00:00 |  28.33  |      -0.5322 |
| 24 | ECHO     | 2025-08-26 00:00:00 |  50.87  |       0.5321 |


## Prices: illiquid early history (trimmed in feature step)

|    | ticker   | first_dropped   | last_dropped   |   rows_dropped |
|---:|:---------|:----------------|:---------------|---------------:|
|  0 | SW       | 2008-06-17      | 2024-07-16     |           4047 |
|  1 | SMCI     | 2007-03-29      | 2019-08-14     |           3117 |
|  2 | TPL      | 2005-01-03      | 2016-07-29     |           2914 |
|  3 | FERG     | 2010-01-05      | 2021-03-22     |           2822 |
|  4 | ERIE     | 2005-01-03      | 2013-12-04     |           2247 |
|  5 | TKO      | 2005-01-03      | 2013-10-04     |           2205 |
|  6 | FIX      | 2005-01-03      | 2013-05-21     |           2110 |
|  7 | AMCR     | 2012-05-15      | 2019-06-21     |           1787 |
|  8 | AXON     | 2005-01-03      | 2011-08-08     |           1662 |
|  9 | CRH      | 2005-01-03      | 2010-01-25     |           1274 |
| 10 | DXCM     | 2005-04-14      | 2009-06-03     |           1042 |
| 11 | ARES     | 2014-05-02      | 2017-12-13     |            913 |
| 12 | MPWR     | 2005-01-03      | 2007-04-20     |            578 |
| 13 | PODD     | 2007-05-15      | 2009-04-27     |            492 |
| 14 | EXR      | 2005-01-03      | 2006-07-03     |            378 |
| 15 | APO      | 2011-03-30      | 2012-09-24     |            376 |
| 16 | ULTA     | 2007-10-25      | 2009-04-08     |            366 |
| 17 | VRT      | 2018-08-02      | 2019-12-16     |            346 |
| 18 | GNRC     | 2010-02-11      | 2011-06-10     |            336 |
| 19 | TDG      | 2006-03-15      | 2007-05-17     |            296 |
| 20 | TYL      | 2005-01-03      | 2006-02-13     |            281 |
| 21 | DLR      | 2005-01-03      | 2005-12-06     |            235 |
| 22 | COHR     | 2005-01-03      | 2005-12-05     |            234 |
| 23 | IVZ      | 2005-01-03      | 2005-11-01     |            211 |
| 24 | CHTR     | 2010-01-05      | 2010-10-14     |            197 |


## Prices: history length

|              |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|:-------------|--------:|-------:|------:|------:|------:|------:|------:|------:|
| trading_days |     503 |   4887 |  1236 |     7 |  5244 |  5477 |  5477 |  5477 |


## Prices: survivorship check (tickers per year)

|    |   year |   tickers |
|---:|-------:|----------:|
|  0 |   2005 |       380 |
|  1 |   2006 |       387 |
|  2 |   2007 |       397 |
|  3 |   2008 |       402 |
|  4 |   2009 |       409 |
|  5 |   2010 |       419 |
|  6 |   2011 |       427 |
|  7 |   2012 |       434 |
|  8 |   2013 |       447 |
|  9 |   2014 |       451 |
| 10 |   2015 |       458 |
| 11 |   2016 |       463 |
| 12 |   2017 |       468 |
| 13 |   2018 |       471 |
| 14 |   2019 |       480 |
| 15 |   2020 |       485 |
| 16 |   2021 |       490 |
| 17 |   2022 |       492 |
| 18 |   2023 |       494 |
| 19 |   2024 |       498 |
| 20 |   2025 |       500 |
| 21 |   2026 |       503 |


## Market: history starts at config start

|    | ticker   | min                 | max                 |   count |
|---:|:---------|:--------------------|:--------------------|--------:|
|  0 | GLD      | 2005-01-03 00:00:00 | 2026-10-09 00:00:00 |    5477 |
|  1 | QQQ      | 2005-01-03 00:00:00 | 2026-10-09 00:00:00 |    5477 |
|  2 | SOXX     | 2005-01-03 00:00:00 | 2026-10-09 00:00:00 |    5477 |
|  3 | SPY      | 2005-01-03 00:00:00 | 2026-10-09 00:00:00 |    5477 |
|  4 | TLT      | 2005-01-03 00:00:00 | 2026-10-09 00:00:00 |    5477 |
|  5 | USO      | 2006-04-10 00:00:00 | 2026-10-09 00:00:00 |    5158 |
|  6 | UUP      | 2007-03-01 00:00:00 | 2026-10-09 00:00:00 |    4935 |
|  7 | XLK      | 2005-01-03 00:00:00 | 2026-10-09 00:00:00 |    5477 |
|  8 | ^VIX     | 2005-01-03 00:00:00 | 2026-10-09 00:00:00 |    5479 |

