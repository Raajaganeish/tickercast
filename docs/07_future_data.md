# 07 · Future data (Phase 4 plan)

Each addition follows the same recipe:

1. Find a source and the **date the information became public**.
2. Turn it into numbers.
3. As-of join onto trading days (carry forward until the next update).
4. Add checks to the validation reports and a truncation test.
5. Retrain and compare the test score with and without it. Keep it only if it helps.

## A. Company fundamentals: SEC EDGAR (free)

- API: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`. Every 10-Q/10-K
  value with its **filing date** (`filed`). Requires a descriptive User-Agent header.
- Ticker → CIK map: `https://www.sec.gov/files/company_tickers.json`.

| Feature | Formula |
|---|---|
| `rev_growth_yoy` | Revenue / revenue 4 quarters earlier − 1 |
| `gross_margin`, `gross_margin_chg` | Gross profit / revenue, change vs last quarter |
| `eps_growth_yoy` | EPS vs 4 quarters earlier |
| `pe_rank` | Price / trailing EPS, ranked across stocks each day |
| `days_since_filing` | Trading days since the last 10-Q/10-K |

**Point-in-time:** use `filed`, never the period-end date. Available from the next trading day.

## B. Earnings events

- `yfinance` `Ticker.earnings_dates`: EPS estimate, reported EPS, date (recent years).
- `eps_surprise = (actual − estimate) / |estimate|`, available from the day after the report.
- `days_to_next_earnings`: scheduled dates are public weeks ahead, so this is a legitimate "known future" input.

## C. Company events: SEC 8-K filings (free, 20+ years)

| Item | Event | Feature |
|---|---|---|
| 2.02 | Earnings release | `ev_earnings` |
| 5.02 | Executive departure / appointment | `ev_exec_change` |
| 1.01 | Material agreement | `ev_deal` |
| 2.01 | Acquisition / disposal completed | `ev_acquisition` |
| 8.01 | Other events (often legal / product news) | `ev_other` |

Events fade, so each becomes a decaying value: `exp(−days_since_event / 10)`.

## D. News sentiment

| Source | Coverage |
|---|---|
| FNSPID (Hugging Face) | ~15M headlines for S&P 500 stocks, 1999–2023 |
| GDELT | Global news since 2015 with tone scores |
| Finnhub / Alpha Vantage | Recent years on free tiers |

Pipeline: headline → **FinBERT** (`ProsusAI/finbert`, local on MPS) → score in [−1, 1] →
per stock-day: `news_sent_mean`, `news_count`, `news_neg_count`. Only headlines timestamped
before 16:00 ET count for that day; later ones go to the next day.

Check license terms of each dataset before use.

## E. Risk indices

- Geopolitical Risk Index (Caldara & Iacoviello), daily: `gpr`, `gpr_chg_5d`.
- Economic Policy Uncertainty Index, daily: `epu_z`.

## F. Fixing survivorship bias

Historical S&P 500 membership (who was in the index on each date) would let us include
companies that later failed. Free options exist on GitHub (reconstructed from Wikipedia's
change history), but prices for delisted tickers are often missing from Yahoo. Worth
revisiting if results look promising.
