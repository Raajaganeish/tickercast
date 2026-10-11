# Validation (`tickercast validate-raw`, `tickercast validate-features`)

| File | Checks | Report |
|---|---|---|
| `raw_checks.py` | Universe, prices, market, macro quality | `reports/01_raw_validation.md` |
| `feature_checks.py` | Missing/inf/constant values, target alignment, leakage scan, **truncation test**, drift | `reports/02_feature_validation.md` |
| `report.py` | PASS / WARN / FAIL / INFO collection → Markdown | — |

Dataset checks (splits, embargo, scaling, windows) live in `dataset/prepare.py` → `reports/03_dataset_validation.md`.

Any FAIL makes the CLI exit with code 1.

Docs: [05_validation](../../../docs/05_validation.md)
