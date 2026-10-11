# 01 · Data sources

Everything is free and downloaded by scripts. Nothing is typed by hand.

## Used now (Phase 1)

| Data | Source | Library | Code | History |
|---|---|---|---|---|
| S&P 500 member list + GICS sector | Wikipedia "List of S&P 500 companies" | `requests` + `pandas.read_html` | [data/universe.py](../src/tickercast/data/universe.py) | Today's list |
| Daily stock prices (OHLCV) | Yahoo Finance | `yfinance` | [data/prices.py](../src/tickercast/data/prices.py) | 2005 → today |
| Market & sector benchmarks | Yahoo Finance | `yfinance` | same | 2005 → today |
| Interest rates, credit spread, inflation, unemployment | FRED / ALFRED (St. Louis Fed) | `fredapi` | [data/macro.py](../src/tickercast/data/macro.py) | 2004 → today |

### Benchmarks downloaded

| Ticker | Feature prefix | What it represents |
|---|---|---|
| SPY | `spy_` | Whole US stock market (S&P 500) |
| QQQ | `qqq_` | Large tech (Nasdaq-100) |
| XLK | `tech_` | Technology sector |
| SOXX | `semis_` | Semiconductors (chip supply, AI hardware) |
| ^VIX | `vix_` | "Fear index": expected market volatility |
| TLT | `tlt_` | Long-term US government bonds (safe haven, rate moves) |
| GLD | `gld_` | Gold (safe haven) |
| USO | `oil_` | Crude oil (starts April 2006) |
| UUP | `usd_` | US dollar (starts March 2007) |

### FRED series (needs a free key in `.env`)

| Series | Feature | Frequency | How it's dated |
|---|---|---|---|
| DGS10 | `yield_10y` | Daily | Available next business day |
| DGS2 | `yield_2y` | Daily | Available next business day |
| T10Y2Y | `curve_10y2y` | Daily | Available next business day |
| BAMLH0A0HYM2 | `hy_spread` | Daily | Available next business day |
| DFF | `fed_funds` | Daily | Available next business day |
| CPIAUCSL | `cpi_yoy` | Monthly | **First-release date** from ALFRED |
| UNRATE | `unemployment` | Monthly | **First-release date** from ALFRED |

Why ALFRED: inflation and jobs numbers get revised months later. FRED shows the revised
number; ALFRED keeps what was published on release day. We use the first release so the
model only sees what the market actually saw.

Get a key (1 minute): https://fred.stlouisfed.org/docs/api/api_key.html, then put it in `.env`:
```
FRED_API_KEY=your_key_here
```
Without a key the pipeline still runs; macro features are simply left out.

## Price adjustment

Prices are downloaded with `auto_adjust=True`: split- and dividend-adjusted. A 4-for-1
split doesn't show up as a -75% crash, and returns include dividends. Adjusted history is
recomputed by Yahoo after each new split/dividend, which changes old price *levels* but not
*returns*. This is why our features use returns and ratios, not raw prices.

## Planned (Phase 4)

| Data | Source | Notes |
|---|---|---|
| Revenue, EPS, margins | SEC EDGAR XBRL API | Free, with filing dates (point-in-time) |
| Earnings surprises | Yahoo Finance earnings dates | Recent years only |
| Company events | SEC 8-K filings (item codes) | Executive changes, deals, lawsuits |
| News headlines | FNSPID (Hugging Face), GDELT, Finnhub | Scored with FinBERT locally |
| Geopolitical risk | Caldara & Iacoviello GPR index | Daily, free |

Details in [07_future_data.md](07_future_data.md).
