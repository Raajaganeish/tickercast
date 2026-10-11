# 02 · Universe: which stocks we use

## Selection

1. Download the current S&P 500 list from Wikipedia (~503 share classes, 11 GICS sectors).
2. Convert symbols to Yahoo format (`BRK.B` → `BRK-B`).
3. Download 2005 → today for each.
4. **Trim illiquid early history** (see below).
5. Drop stocks with fewer than 252 trading days (1 year) left. Recent IPOs/spin-offs.

Result in the latest run: **499 stocks** used.

## Known limitation: survivorship bias

We use **today's** members. Companies that went bankrupt or were removed from the index
(Lehman Brothers, Enron-style failures, acquired companies) are missing entirely.

Effect: the training data is tilted toward companies that *survived and grew*, so results
can look better than they would have in real time.

| Year | Stocks with data |
|---|---|
| 2005 | ~380 |
| 2026 | 503 |

The early-year count is lower because some of today's members weren't public yet.

**Why accept it for now:** free point-in-time membership data is hard to get. It's fine for
a learning prototype, as long as we remember it. A fix (historical constituents) is
listed in [07_future_data.md](07_future_data.md).

## Cleaning: illiquid early history

Some stocks have early Yahoo history from over-the-counter or foreign trading before a
proper US listing. Example from our raw data:

| Ticker | Period | What the data looks like |
|---|---|---|
| FERG | 2012-03 → 2012-07 | Same price for 73 days in a row, ~4 shares/day |
| SW | 2009-05 → 2009-08 | Same price for 69 days, ~6 shares/day |
| AMCR | 2016-09 → 2016-12 | Same price for 66 days |

That isn't real trading, so it would teach the model nonsense.

**Rule** ([data/clean.py](../src/tickercast/data/clean.py)): for each stock, find the last
day its 20-day median **dollar volume** (price × shares) was below **$1M**, and drop
everything up to that day. Threshold: `universe.min_dollar_volume` in the config.

Latest run: 37 stocks affected, 30,766 rows dropped (1.2% of data). Stale-price runs
dropped from 527 to 1. The list is in `reports/01_raw_validation.md`.
