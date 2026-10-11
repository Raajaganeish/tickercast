# tickercast

Predicting next-day stock direction for S&P 500 companies with a **PatchTST Transformer**,
trained locally on a Mac. This is a learning project: build a clean dataset, train a
model, test it honestly, and measure whether it beats simple baselines.

> ⚠️ Educational project, not investment advice. Daily stock moves are mostly noise;
> 52–55% direction accuracy would already be a good result.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | **Dataset**: prices, market, sector, calendar, macro (FRED) → validated, model-ready | ✅ Done |
| 2 | Model choice review + baselines (naive, LightGBM) | ⏭️ Next |
| 3 | PatchTST training (~5M params, 45–60 min on M-series Mac) | Planned |
| 4 | Extra data: fundamentals (SEC EDGAR), company events (8-K), news sentiment (FinBERT) | Planned |

Current dataset: **2.36M rows** (499 stocks × ~20 years) · **66 features** (incl. FRED macro) · **0 failed validation checks**.

## How it works

```
 STEP 1: DOWNLOAD           STEP 2: FEATURES               STEP 3: DATASET
 ─────────────────          ──────────────────             ─────────────────────
 S&P 500 list (Wikipedia)   prices  → returns, volatility  split by time
 stock prices (Yahoo)   →   market  → SPY/VIX/sector   →   (train / val / test)
 market + VIX (Yahoo)       macro   → rates, inflation     scale (train stats only)
 macro (FRED)               calendar→ day/month cycles     128-day windows
                            targets → next-day up/down     → PyTorch Dataset
        │                          │                              │
        ▼                          ▼                              ▼
 reports/01_raw_validation  reports/02_feature_validation  reports/03_dataset_validation
```

Every step writes a validation report. The pipeline stops being trustworthy the moment a
feature sees the future, so the checks focus on that (see [docs/04](docs/04_point_in_time.md)).

## Quick start

```bash
git clone https://github.com/Raajaganeish/tickercast.git
cd tickercast
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model]"

cp .env.example .env          # optional: add a free FRED API key for macro features

tickercast all                # download → validate → features → validate → dataset (~2 min)
pytest                        # offline tests, including leakage tests
```

Individual steps:

| Command | Does | Output |
|---|---|---|
| `tickercast download` | S&P 500 list, prices, market data, macro | `data/raw/*.parquet` |
| `tickercast validate-raw` | Checks raw data | `reports/01_raw_validation.md` |
| `tickercast features` | Builds the feature table | `data/interim/features.parquet` |
| `tickercast validate-features` | Quality + leakage checks | `reports/02_feature_validation.md` |
| `tickercast dataset` | Split, scale, window, check | `data/processed/` + `reports/03_dataset_validation.md` |

Add `--limit 20` to `download` for a quick test with 20 stocks.

## Viewing the data

The data files are Parquet (compressed, not human-readable). Three ways to look at them:

| Way | How |
|---|---|
| **Guided notebook** | Open [`notebooks/01_explore_dataset.ipynb`](notebooks/01_explore_dataset.ipynb): tables and charts from raw prices to the exact 128-day window the model sees |
| **Column meanings** | [docs/data_dictionary.md](docs/data_dictionary.md): what every column means and how to read its numbers |
| **Spreadsheet** | `tickercast sample --ticker AAPL --stage features` → `data/samples/AAPL_features.csv` (also `--stage raw` / `dataset`, `--start`, `--end`) |
| **Python** | `pd.read_parquet("data/interim/features.parquet", filters=[("ticker", "==", "AAPL")])` |

## Using the dataset in PyTorch

```python
from torch.utils.data import DataLoader
from tickercast.dataset.windows import WindowDataset

train = WindowDataset("data/processed", split="train")       # target: label_up_1d
x, y, ticker_id, sector_id = train[0]                        # x: (128, 66)
loader = DataLoader(train, batch_size=512, shuffle=True)     # shuffling windows is fine;
                                                             # the split itself is by time
```

## Documentation

Read in order:

| Doc | Explains |
|---|---|
| [00 Plan](docs/00_plan.md) | Goals, decisions, roadmap |
| [01 Data sources](docs/01_data_sources.md) | Where every number comes from |
| [02 Universe](docs/02_universe.md) | Which stocks, survivorship bias, liquidity cleaning |
| [03 Features](docs/03_features.md) | How real-world information becomes numbers (with a real Apple example) |
| [Data dictionary](docs/data_dictionary.md) | **Every column in plain words**, with a real Apple value and how to read it |
| [04 Point-in-time](docs/04_point_in_time.md) | The no-peeking rule and how it's enforced |
| [05 Validation](docs/05_validation.md) | Every check, why it exists, what to do when it fails |
| [06 Model input](docs/06_model_input.md) | Splits, scaling, windows, and how PatchTST will read them |
| [07 Future data](docs/07_future_data.md) | Plan for fundamentals, events, news sentiment |

## Repository layout

```
configs/data.yaml        all settings (dates, tickers, windows, splits)
src/tickercast/
  data/                  step 1: download + cleaning
  features/              step 2: feature engineering + targets
  validation/            checks + Markdown reports
  dataset/               step 3: splits, scaling, PyTorch windows
  cli.py                 `tickercast` command
notebooks/               exploration notebooks
tests/                   offline tests on synthetic data
docs/                    documentation
reports/                 latest validation reports (committed so the team can see them)
data/                    downloaded + generated data (git-ignored, rebuild with `tickercast all`)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
