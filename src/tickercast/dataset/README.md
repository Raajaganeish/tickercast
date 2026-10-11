# Step 3 · Model-ready dataset (`tickercast dataset`)

| File | Does |
|---|---|
| `splits.py` | Train / val / test by date, with a 5-day embargo at each boundary |
| `prepare.py` | Scales with train-only stats, clips ±5, fills gaps with 0, adds ticker/sector IDs, runs checks |
| `windows.py` | `WindowDataset`: serves `(128, n_features)` windows + label + IDs to PyTorch |

Outputs: `data/processed/dataset.parquet`, `data/processed/meta.json`, `reports/03_dataset_validation.md`.

Docs: [06_model_input](../../../docs/06_model_input.md)
