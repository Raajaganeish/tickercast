# Data dictionary: every column in plain words

This explains every column you see in `features.parquet` or `tickercast sample` CSV files.

Every example value is from a real row: **Apple (AAPL) on Friday 3 May 2024**. The evening
before, Apple reported earnings and announced a record $110B share buyback.

## First: 4 tricks to read the numbers

**1. Returns are small decimals, roughly percentages.**
`0.058` ≈ **+5.8%** (a bit more exactly, +6.0%). `-0.009` ≈ **−0.9%**.
They're stored as *log returns*, which add up neatly over days. For small values,
just read them as percentages.

**2. Ranks go from −0.5 (worst) to +0.5 (best).**
They compare a stock with all ~500 stocks on the same day.
`+0.46` = near the very top. `0` = middle. `−0.23` = in the bottom third.
How to convert: add 0.5 → share of stocks it beat. `+0.46 + 0.5 = 0.96` → beat 96% of stocks.

**3. sin / cos pairs are calendar positions on a circle.**
You don't need to read them. They just tell the model "which weekday" or "which month"
in a way where Friday is next to Monday and December is next to January.

**4. Columns starting with `fwd_` or `label_` are the answers (the future).**
Everything else is what was known that evening. The model sees the inputs and learns to
guess the answers.

---

## 1. Who and when (3 columns)

| Column | Meaning | AAPL value |
|---|---|---|
| `date` | Trading day | 2024-05-03 |
| `ticker` | Stock symbol | AAPL |
| `sector` | Industry group (11 groups) | Information Technology |

## 2. How the stock itself behaved (13 columns)

| Column | Plain meaning | AAPL value | How to read it |
|---|---|---|---|
| `ret_1d` | Price change today | 0.058 | Up ~6% today (earnings reaction) |
| `ret_5d` | Price change over the last week | 0.080 | Up ~8% this week |
| `ret_20d` | …last month (20 trading days) | 0.078 | Up ~8% this month |
| `ret_60d` | …last 3 months | −0.031 | Still down ~3% over 3 months |
| `vol_20d` | How jumpy: typical daily move, last month | 0.021 | Usually moves ~2% a day |
| `vol_60d` | Same, last 3 months | 0.016 | ~1.6% a day |
| `volume_ratio` | Trading busier than usual? (0 = normal) | 0.91 | ~2.5× normal volume. Very busy |
| `rsi` | "Overbought / oversold" meter (−0.5 to +0.5; 0 = neutral) | 0.17 | RSI 67: strong recent buying |
| `ma_dist_50` | Price vs its 50-day average | 0.064 | 6.4% above its 2-month average |
| `ma_dist_200` | Price vs its 200-day average | 0.014 | 1.4% above its ~10-month average |
| `hl_range` | Gap between today's high and low | 0.024 | Moved within a 2.4% range today |
| `gap` | Jump at the open vs yesterday's close | 0.076 | Opened ~8% higher (news overnight) |
| `drawdown_252` | How far below its 1-year high (0 = at the high) | −0.073 | 7.3% below its 52-week high |

## 3. Compared with the other ~500 stocks (10 columns)

| Column | Plain meaning | AAPL value | How to read it |
|---|---|---|---|
| `cs_rank_ret_5d` | Rank of this week's return | 0.46 | Beat 96% of stocks this week |
| `cs_rank_ret_20d` | Rank of this month's return | 0.43 | Beat 93% |
| `cs_rank_ret_60d` | Rank of 3-month return | −0.23 | Beat only 27% |
| `cs_rank_vol_20d` | Rank of jumpiness | 0.24 | Jumpier than 74% of stocks |
| `cs_rank_volume_ratio` | Rank of unusual volume | 0.48 | Busier than 98% of stocks today |
| `sector_ret_1d` | Average move of its sector today | 0.014 | Tech stocks up ~1.4% on average |
| `rel_sector_ret_1d` | Stock minus its sector, today | 0.044 | Beat other tech stocks by ~4.4% (Apple-specific news) |
| `sector_ret_5d` | Sector's average move this week | −0.003 | Tech roughly flat this week |
| `rel_sector_ret_5d` | Stock minus sector, this week | 0.082 | Beat tech by ~8% this week |
| `breadth` | Share of all stocks up today (−0.5 to +0.5) | 0.24 | 74% of stocks rose: good day overall |

## 4. The overall market (23 columns)

Each benchmark has a 1-day (`_ret_1d`) and 1-week (`_ret_5d`) change.

| Column | What it tracks | AAPL row value (1d / 5d) | How to read it |
|---|---|---|---|
| `spy_ret_1d / 5d` | Whole US stock market (S&P 500) | 0.012 / 0.006 | Market up 1.2% today |
| `qqq_ret_1d / 5d` | Big tech companies (Nasdaq-100) | 0.020 / 0.010 | Up 2% today |
| `tech_ret_1d / 5d` | Technology sector fund (XLK) | 0.027 / 0.014 | Up 2.7% (Apple is a big part of it) |
| `semis_ret_1d / 5d` | Chip makers (SOXX) | 0.022 / −0.003 | Up 2.2% today |
| `tlt_ret_1d / 5d` | Long-term government bonds | 0.010 / 0.021 | Bonds up = interest rates fell |
| `gld_ret_1d / 5d` | Gold | −0.001 / −0.017 | Flat today |
| `oil_ret_1d / 5d` | Crude oil | −0.011 / −0.068 | Oil down ~7% this week |
| `usd_ret_1d / 5d` | US dollar | −0.003 / −0.008 | Dollar slightly weaker |
| `spy_vol_20d` | How jumpy the market has been | 0.009 | Calm: ~0.9% typical daily move |
| `spy_drawdown_252` | Market's distance below its 1-year high | −0.023 | Market only 2.3% off its high |
| `vix_log` | Fear index (VIX), log scale | 2.60 | VIX ≈ 13.5 → calm (panic is 30+) |
| `vix_chg_1d` | Change in fear today | −0.085 | Fear dropped ~8% today |
| `vix_z_60` | Fear vs the last 3 months (0 = normal) | −0.82 | Calmer than usual |
| `rel_spy_ret_1d` | Stock minus market today | 0.046 | Beat the market by ~4.6% |
| `beta_spy` | How closely the stock follows the market (1 = same) | 0.98 | Moves almost exactly with the market |

## 5. The economy (13 columns, from FRED)

These are the same for every stock on a given day.

| Column | Plain meaning | AAPL row value | How to read it |
|---|---|---|---|
| `yield_10y` | 10-year US government interest rate (%) | 4.58 | 4.58%: high by 2010s standards |
| `yield_10y_chg_20d` | Change over the last month (% points) | 0.27 | Rates rose 0.27 pts this month |
| `yield_2y` | 2-year interest rate (%) | 4.87 | |
| `yield_2y_chg_20d` | Its 1-month change | 0.22 | |
| `curve_10y2y` | 10-year rate minus 2-year rate | −0.29 | Negative ("inverted") has often come before recessions |
| `curve_10y2y_chg_20d` | Its 1-month change | 0.05 | |
| `credit_spread` | Extra interest risky companies pay vs the government | 1.47 | Low = lenders aren't worried |
| `credit_spread_chg_20d` | Its 1-month change | −0.04 | |
| `fed_funds` | The Federal Reserve's policy rate (%) | 5.33 | Highest in ~20 years |
| `fed_funds_chg_20d` | Its 1-month change | 0.00 | Fed didn't change rates |
| `cpi_yoy` | Inflation: prices vs a year ago | 0.0345 | 3.45% inflation |
| `unemployment` | Unemployment rate (%) | 3.9 | Low: strong job market |
| `unemployment_chg_3m` | Change over the last 3 months (% points) | 0.2 | Slightly rising |

## 6. Calendar (7 columns)

| Column | Plain meaning | AAPL value | How to read it |
|---|---|---|---|
| `dow_sin`, `dow_cos` | Day of the week (circle position) | −0.95, 0.31 | = Friday |
| `month_sin`, `month_cos` | Month (circle position) | 0.87, −0.50 | = May |
| `days_to_quarter_end` | Time until quarter end (0 = today, ~1 = 3 months) | 0.63 | 58 days until June 30 |
| `is_q4` | Is it October–December? (1 = yes) | 0 | No |
| `opex_week` | Monthly options-expiry week? (1 = yes) | 0 | No (that was May 13–17) |

## 7. The answers: targets (8 columns)

The model sees everything above and learns to predict these. They're **in the future**
relative to the row's date.

| Column | Plain meaning | AAPL value | How to read it |
|---|---|---|---|
| `fwd_ret_1d` | Price change on the **next** trading day | −0.009 | Fell ~0.9% on Monday May 6 |
| **`label_up_1d`** | Did it go up the next day? (1 = yes, 0 = no) | **0** | **No. This is the main target** |
| `fwd_excess_1d` | Next-day change minus the market's | −0.019 | Lagged the market by ~1.9% |
| `label_outperform_1d` | Did it beat the market the next day? | 0 | No |
| `fwd_ret_5d` | Change over the next week | −0.000 | Flat over the next week |
| `label_up_5d` | Up over the next week? | 0 | No (barely) |
| `fwd_excess_5d` | Next-week change minus the market's | −0.019 | Lagged the market by ~1.9% |
| `label_outperform_5d` | Beat the market over the next week? | 0 | No |

## The story this row tells

> "Apple jumped 6% on huge volume after news overnight, beating its sector and the
> market by over 4%. The market was calm (fear index low) and most stocks rose.
> Interest rates were high and inflation 3.5%. It was a Friday in May."
>
> **Answer:** the next trading day, Apple fell 0.9% → `label_up_1d = 0`.

The model's job: look at ~128 days of rows like this one and learn patterns
(e.g. "after a huge one-day jump, the next day is often slightly down").

## In the model-ready file (`dataset.parquet`)

The same columns, but every input number is **rescaled** so that 0 = "typical" and
±1 = "one usual step away from typical". Example: `volume_ratio` 0.91 becomes about +2.5
("much busier than usual"). Three columns are added:

| Column | Meaning |
|---|---|
| `split` | `train` (2005–2019), `val` (2020–2021), `test` (2022+), or `embargo` (gap days) |
| `ticker_id` | The stock as a number, alphabetical (A = 0, AAPL = 1) |
| `sector_id` | The sector as a number |
