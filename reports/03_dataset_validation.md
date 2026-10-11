# Model-ready dataset validation

_Generated 2026-10-11 14:02_

Checks on data/processed/dataset.parquet.

**Summary:** 5 passed · 0 warnings · 0 failed · 0 info

| Check | Status | Result |
|---|---|---|
| Splits are in time order, no overlap | ✅ PASS | train < val < test |
| Embargo: targets never reach into the next split | ✅ PASS | longest target = 5 days, embargo = 5 days; violations: none |
| Scaler fitted on train only | ✅ PASS | train features: max |mean| = 0.013 (≈0), max |std - 1| = 0.147 (≈0, clipping shrinks it slightly) |
| No NaN / inf in model inputs | ✅ PASS | 0 non-finite values |
| Training windows per split | ✅ PASS | 128-day windows: train=1,461,974, val=238,444, test=588,876 |

## Splits are in time order, no overlap

|    | split   | min                 | max                 |   count |
|---:|:--------|:--------------------|:--------------------|--------:|
|  0 | train   | 2005-10-18 00:00:00 | 2019-12-23 00:00:00 | 1521727 |
|  1 | val     | 2020-01-02 00:00:00 | 2021-12-23 00:00:00 |  240150 |
|  2 | test    | 2022-01-03 00:00:00 | 2026-10-09 00:00:00 |  591256 |


## Training windows per split

|    | split   |   windows |   tickers |   share_up |
|---:|:--------|----------:|----------:|-----------:|
|  0 | train   |   1461974 |       470 |     0.5196 |
|  1 | val     |    238444 |       482 |     0.5261 |
|  2 | test    |    588876 |       499 |     0.514  |

