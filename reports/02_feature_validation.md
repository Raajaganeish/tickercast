# Feature validation

_Generated 2026-10-11 14:02_

Checks on data/interim/features.parquet. See docs/05_validation.md for what each check means.

**Summary:** 8 passed · 2 warnings · 0 failed · 2 info

| Check | Status | Result |
|---|---|---|
| Shape | ℹ️ INFO | 2,357,926 rows · 53 features · 499 tickers · 2005-10-18 -> 2026-10-09 |
| Unique (ticker, date) | ✅ PASS | 0 duplicates |
| No infinite values | ✅ PASS | 0 inf values |
| No constant features | ✅ PASS | 0 constant: [] |
| Missing values per feature | ⚠️ WARN | 2 features with > 5% missing (filled with 0 + scaled later; review causes) |
| Target alignment (fwd_ret_1d == next day's ret_1d) | ✅ PASS | 0 mismatches out of 2,357,427 |
| Missing targets only at series end | ✅ PASS | 499 rows without a 1-day target (499 are each stock's last day) |
| Label balance (share of 1s) | ℹ️ INFO | close to 0.5 is good; 'always up' accuracy in each split = label_up share |
| Leakage scan: feature vs next-day return correlation | ✅ PASS | max |daily rank IC| = 0.017. Real signals in daily stock data are ~0.01-0.05; > 0.2 almost always means the feature sees the future |
| Point-in-time test (truncation) | ✅ PASS | features rebuilt with data cut at 2012-06-29, 2020-03-16 on 40 tickers: 0 differing values |
| Distribution drift vs train | ✅ PASS | 0 features whose val/test mean moved > 1 train std (expected for rates/inflation; a reason to prefer changes and ranks over raw levels) |
| Extreme values (> 10 train std) | ⚠️ WARN | worst feature: gap (0.111% of rows). Clipped to ±5 std after scaling |

## Missing values per feature

|            |   train |   val |   test |   overall |
|:-----------|--------:|------:|-------:|----------:|
| usd_ret_5d | 0.08549 |     0 |      0 |   0.05518 |
| usd_ret_1d | 0.08449 |     0 |      0 |   0.05453 |
| oil_ret_5d | 0.03015 |     0 |      0 |   0.01945 |
| oil_ret_1d | 0.02917 |     0 |      0 |   0.01883 |


## Label balance (share of 1s)

|       |   label_up_1d |   label_outperform_1d |   label_up_5d |   label_outperform_5d |
|:------|--------------:|----------------------:|--------------:|----------------------:|
| train |        0.5196 |                0.4999 |        0.5498 |                0.5078 |
| val   |        0.5261 |                0.4934 |        0.5597 |                0.4912 |
| test  |        0.5139 |                0.495  |        0.5278 |                0.4894 |


## Leakage scan: feature vs next-day return correlation

|                      |   mean_daily_rank_IC |
|:---------------------|---------------------:|
| rel_spy_ret_1d       |            -0.01745  |
| ret_1d               |            -0.01745  |
| rel_sector_ret_1d    |            -0.01547  |
| cs_rank_ret_5d       |            -0.01375  |
| ret_5d               |            -0.01375  |
| rel_sector_ret_5d    |            -0.01185  |
| ret_20d              |            -0.008031 |
| cs_rank_ret_20d      |            -0.008031 |
| cs_rank_volume_ratio |             0.007884 |
| volume_ratio         |             0.007884 |
| rsi                  |            -0.00786  |
| sector_ret_1d        |            -0.007283 |
| ma_dist_50           |            -0.006447 |
| gap                  |            -0.006395 |
| sector_ret_5d        |            -0.004856 |


## Point-in-time test (truncation)

|    | cutoff     |   rows_compared |   cells_differ | features_differ   |
|---:|:-----------|----------------:|---------------:|:------------------|
|  0 | 2012-06-29 |           54824 |              0 | none              |
|  1 | 2020-03-16 |          128963 |              0 | none              |


## Distribution drift vs train

|                  |   val_shift_in_std |   test_shift_in_std |
|:-----------------|-------------------:|--------------------:|
| vix_log          |           0.7937   |            0.1984   |
| vol_60d          |           0.4793   |            0.1782   |
| vol_20d          |           0.4053   |            0.1604   |
| spy_vol_20d      |           0.3837   |            0.04779  |
| beta_spy         |          -0.07483  |           -0.3474   |
| hl_range         |           0.2709   |            0.08705  |
| ma_dist_200      |           0.2301   |           -0.06558  |
| drawdown_252     |          -0.04264  |           -0.2229   |
| tlt_ret_5d       |          -0.002944 |           -0.1725   |
| spy_drawdown_252 |           0.1564   |            0.02174  |
| tech_ret_5d      |           0.1508   |            0.04371  |
| semis_ret_5d     |           0.1418   |            0.07024  |
| oil_ret_5d       |          -0.09351  |            0.1413   |
| qqq_ret_5d       |           0.1323   |            0.005467 |
| ret_60d          |           0.127    |           -0.04422  |


## Extreme values (> 10 train std)

|                   |   share_of_rows |
|:------------------|----------------:|
| gap               |       0.001112  |
| oil_ret_5d        |       0.001011  |
| rel_sector_ret_1d |       0.0006485 |
| rel_spy_ret_1d    |       0.0005988 |
| hl_range          |       0.0005607 |
| vol_20d           |       0.0004546 |
| ret_1d            |       0.0004339 |
| oil_ret_1d        |       0.0004037 |
| vol_60d           |       0.0003808 |
| rel_sector_ret_5d |       0.0003753 |

