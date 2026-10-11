# 03 · Features: turning the real world into numbers

A model can't read "Apple beat earnings." It only reads numbers. So for **every stock on
every trading day**, we answer a fixed set of questions with numbers. Each answer is a
**feature** (a column). Each stock-day is a **row**.

**The one rule:** a row for day *t* may only use information known by the **close of
day *t***. Details in [04_point_in_time.md](04_point_in_time.md).

## A real example: Apple on 3 May 2024

What happened: Apple reported earnings the evening before and announced a record $110B
share buyback. The stock jumped ~6%, while the market rose ~1%.

Here is part of that row in our dataset (`data/interim/features.parquet`):

| Question | Feature | Value | Plain meaning |
|---|---|---|---|
| How much did it move today? | `ret_1d` | +0.058 | ≈ +6.0% (stored as a log return) |
| …over the last week? | `ret_5d` | +0.080 | ≈ +8.3% in 5 days |
| Did it open with a jump? | `gap` | +0.076 | Opened ≈ 7.9% above yesterday's close (earnings news overnight) |
| Was trading unusually busy? | `volume_ratio` | +0.91 | ~2.5× its normal volume |
| How did it do vs its sector? | `rel_sector_ret_1d` | +0.044 | Beat other tech stocks by ~4.4% |
| How did the market do? | `spy_ret_1d` | +0.012 | S&P 500 up 1.2% |
| How scared is the market? | `vix_log`, `vix_z_60` | 2.60, -0.82 | VIX ≈ 13.5, calmer than usual |
| How many stocks rose today? | `breadth` | +0.24 | ~74% of stocks were up |
| How far from its 1-year high? | `drawdown_252` | -0.073 | 7.3% below its 52-week high |
| What day/month is it? | `dow_sin`, `month_sin` | -0.95, 0.87 | A Friday in May |
| **Did it go up the next day?** | `label_up_1d` | **0** | **No: -0.9% on Monday** |

The last row is the **target** (what the model learns to predict). Everything above it is
**input**.

## All features (53 without macro, 66 with)

### 1. The stock's own behavior ([features/stock.py](../src/tickercast/features/stock.py))

| Feature | Formula | Why it might matter |
|---|---|---|
| `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d` | Sum of daily log returns over 1/5/20/60 days | Momentum and short-term reversal |
| `vol_20d`, `vol_60d` | Std of daily log returns | How jumpy the stock is |
| `volume_ratio` | log(volume) − log(20-day avg volume) | Unusual activity often follows news |
| `rsi` | Wilder RSI(14), rescaled to [-0.5, 0.5] | "Overbought/oversold" indicator |
| `ma_dist_50`, `ma_dist_200` | close / moving average − 1 | Trend: above or below its average |
| `hl_range` | (high − low) / close | Intraday uncertainty |
| `gap` | log(open / yesterday's close) | Overnight news reaction |
| `drawdown_252` | close / 52-week high − 1 | How beaten-down it is |

Log returns are used because they add up over time and treat +10% and −10% more evenly.

### 2. Compared with other stocks ([features/stock.py](../src/tickercast/features/stock.py))

| Feature | Meaning |
|---|---|
| `cs_rank_ret_5d`, `cs_rank_ret_20d`, `cs_rank_ret_60d` | Where this stock ranks among all stocks that day (−0.5 = worst, +0.5 = best) |
| `cs_rank_vol_20d`, `cs_rank_volume_ratio` | Rank of volatility / unusual volume |
| `sector_ret_1d`, `sector_ret_5d` | Average return of all stocks in the same sector that day |
| `rel_sector_ret_1d`, `rel_sector_ret_5d` | Stock return minus its sector's (company-specific news) |
| `breadth` | Share of all stocks that rose today, minus 0.5 (market mood) |

Ranks are useful because they mean the same thing in calm 2017 and wild 2020.

Sector returns are computed from our own 500 stocks, not sector ETFs: two of the sector
ETFs (XLRE, XLC) only start in 2015/2018 and would leave gaps.

### 3. Market and sectors ([features/market.py](../src/tickercast/features/market.py))

| Feature | Meaning |
|---|---|
| `spy_ret_1d/5d`, `qqq_…`, `tech_…`, `semis_…` | Market, large tech, tech sector, chips |
| `tlt_…`, `gld_…`, `oil_…`, `usd_…` | Bonds, gold, oil, dollar |
| `spy_vol_20d` | How volatile the market has been |
| `spy_drawdown_252` | How far the market is below its 1-year high (panic measure) |
| `vix_log`, `vix_chg_1d`, `vix_z_60` | Fear level, today's change, and how unusual it is vs the last 60 days |
| `rel_spy_ret_1d` | Stock return minus market return |
| `beta_spy` | How strongly the stock follows the market (60-day rolling) |

These cover your list's *technology sector* (tech, semis), *overall market* (SPY, bonds,
dollar) and *unexpected events* (VIX spikes, drawdowns, breadth collapses). We can't
predict a shock, but the model can see the market's state while it's happening.

### 4. The economy ([features/macro.py](../src/tickercast/features/macro.py), needs FRED key)

| Feature | Meaning |
|---|---|
| `yield_10y`, `yield_2y`, `fed_funds` (+ `_chg_20d`) | Interest rates and their 1-month change |
| `curve_10y2y` (+ `_chg_20d`) | Yield curve; negative has historically preceded recessions |
| `hy_spread` (+ `_chg_20d`) | Extra yield on risky bonds = credit stress |
| `cpi_yoy` | Inflation, year over year, as first published |
| `unemployment`, `unemployment_chg_3m` | Jobs market |

Monthly numbers are carried forward from their **release date** until the next release.

### 5. Calendar / seasonality ([features/calendar.py](../src/tickercast/features/calendar.py))

| Feature | Meaning |
|---|---|
| `dow_sin`, `dow_cos` | Day of week as a circle (Friday sits next to Monday) |
| `month_sin`, `month_cos` | Month as a circle (December sits next to January) |
| `days_to_quarter_end` | Funds rebalance near quarter end |
| `is_q4` | Holiday quarter |
| `opex_week` | Week of monthly options expiry (3rd Friday) |

Why sin/cos: if we used month = 1…12, the model would think December (12) and January (1)
are far apart. On a circle they're neighbors:
```
month_sin = sin(2π × (month − 1) / 12)
month_cos = cos(2π × (month − 1) / 12)
```

### 6. Identity (used as embeddings in the model)

`ticker_id` and `sector_id` are integers added in the dataset step. The model learns a
small vector for each, so it can learn that utilities behave differently from chip makers.

## Targets ([features/targets.py](../src/tickercast/features/targets.py))

The **only** columns allowed to look into the future. They start with `fwd_` or `label_`.

| Column | Meaning |
|---|---|
| `fwd_ret_1d`, `fwd_ret_5d` | Log return from today's close to the close 1 / 5 days later |
| `label_up_1d`, `label_up_5d` | 1 if that return > 0 |
| `fwd_excess_1d`, `fwd_excess_5d` | Return minus SPY's return over the same period |
| `label_outperform_1d`, `label_outperform_5d` | 1 if the stock beat the market |

Default target: `label_up_1d`. `label_outperform_1d` is a good second experiment: it
removes the "market went up" effect and is balanced at ~50%.

## Warm-up

A 200-day moving average needs 200 days of history. The first 200 rows of each stock
are dropped (`features.warmup_days`) so every row has its long-window features filled in.

## Your original list → where it's covered

| Factor | Phase 1 (now) | Phase 4 (planned) |
|---|---|---|
| Financial performance | — (reaction visible via `gap`, `volume_ratio`) | EDGAR revenue/EPS/margins, earnings surprise |
| News | Indirectly via price/volume reaction | 8-K events, FinBERT headline sentiment |
| Technology sector | `tech_`, `semis_`, `qqq_`, sector averages | AI news volume |
| Overall market | SPY, VIX, bonds, dollar, oil, FRED rates & inflation | — |
| Seasonal effects | Calendar features | Days to earnings |
| Unexpected events | VIX spikes, drawdowns, breadth | Geopolitical risk index |
