# Step 1 · Data download (`tickercast download`)

| File | Does | Output |
|---|---|---|
| `universe.py` | Current S&P 500 list + GICS sectors from Wikipedia | `data/raw/universe.parquet` |
| `prices.py` | Adjusted daily OHLCV from Yahoo for stocks and benchmarks | `data/raw/prices.parquet`, `data/raw/market.parquet` |
| `macro.py` | FRED series with point-in-time `available_date` (needs `FRED_API_KEY`) | `data/raw/macro.parquet` |
| `clean.py` | Trims illiquid OTC/pre-listing history (applied in step 2) | — |

Raw files are never edited by later steps. Rebuild everything with `tickercast all`.

Docs: [01_data_sources](../../../docs/01_data_sources.md) · [02_universe](../../../docs/02_universe.md)
