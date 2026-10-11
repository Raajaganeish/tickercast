# Step 2 · Features (`tickercast features`)

| File | Features |
|---|---|
| `stock.py` | Returns, volatility, volume, RSI, moving averages, gaps, drawdown; cross-stock ranks, sector averages, breadth |
| `market.py` | SPY/QQQ/XLK/SOXX/TLT/GLD/USO/UUP returns, VIX, market drawdown, beta |
| `macro.py` | Rates, yield curve, credit spread, inflation, unemployment (as-of release date) |
| `calendar.py` | Day-of-week / month cycles, quarter end, Q4, options expiry week |
| `targets.py` | `fwd_*` and `label_*`: the only forward-looking columns |
| `build.py` | Joins everything, drops warm-up rows → `data/interim/features.parquet` |

Rule: a row for day *t* uses only data known at the close of *t*.

Docs: [03_features](../../../docs/03_features.md) · [04_point_in_time](../../../docs/04_point_in_time.md)
