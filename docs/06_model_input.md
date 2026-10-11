# 06 · Model input: from table to Transformer

## 1. Split by time

```
2005 ─────────── train ──────────── 2019 │ 2020 ── val ── 2021 │ 2022 ──── test ──── today
                                    ▲ embargo                ▲ embargo
                                    (last 5 trading days dropped at each boundary)
```

| Split | Dates | Rows | 128-day windows | Share of up days |
|---|---|---|---|---|
| train | 2005-10 → 2019-12 | 1,521,727 | 1,461,974 | 52.0% |
| val | 2020-01 → 2021-12 | 240,150 | 238,444 | 52.6% |
| test | 2022-01 → today | 591,256 | 588,876 | 51.4% |
| embargo | 5 days before each boundary | 4,793 | — | — |

- **Train**: the model learns from this.
- **Val**: used during training to pick settings and decide when to stop (early stopping).
- **Test**: used **once**, at the very end. Looking at it repeatedly while tuning turns it
  into a second validation set and makes results look better than they are.

The validation period (2020–2021) includes the COVID crash and recovery; the test period
includes the 2022 bear market. Results are deliberately judged on very different markets.

**Embargo:** the 5-day target of the last train day would land in January, inside
validation. Dropping the last 5 trading days before each boundary prevents that overlap.

## 2. Scale (train statistics only)

Each feature is converted to a z-score using the **training** mean and std:

```
x_scaled = clip((x − train_mean) / train_std, −5, +5)
```

- Neural networks train better when inputs are on similar scales.
- Using train-only statistics means no information from 2020+ leaks into the inputs.
- Clipping at ±5 stops rare extreme days from dominating.
- Missing values become **0** after scaling, which means "average".

Scaler statistics are saved in `data/processed/meta.json` so new data can be scaled the
same way later.

## 3. Windows

The model sees the **last 128 trading days** (~6 months) of one stock:

```
x : (128, 53)   ← 128 days × 53 features, oldest first
y : 0 or 1      ← label on the LAST day of the window (did day t+1 close higher?)
ticker_id, sector_id : integers
```

A window belongs to the split of its **last day**. It may start in an earlier split. That's
fine, because those rows are past information the model would have had anyway.

Windows never span two stocks ([dataset/windows.py](../src/tickercast/dataset/windows.py),
tested in `tests/test_windows.py`).

```python
from tickercast.dataset.windows import WindowDataset
ds = WindowDataset("data/processed", split="train", target="label_up_1d")
x, y, ticker_id, sector_id = ds[0]
```

The dataset keeps all features in one float32 array (~0.5 GB) and slices windows on
demand, so it doesn't store 1.5M copies of overlapping windows.

## 4. How PatchTST will read a window (Phase 3)

```
x: 128 days × 53 features
        │  cut into 16 patches of 8 days
        ▼
16 patches × (8 × 53 values)
        │  linear layer → 256-dim vector per patch
        │  + positional encoding (which patch is which)
        │  + ticker embedding + sector embedding
        ▼
16 tokens × 256
        │  Transformer encoder: 6 layers, 8 attention heads
        ▼
        │  flatten → linear
        ▼
P(up tomorrow)   → binary cross-entropy loss vs y
```

Patching makes each "token" a short chunk of time rather than one day: 16 tokens instead
of 128 makes attention ~64× cheaper and each token carries more information.

Model size will be re-checked in Phase 2 now that the real dataset size is known
(~1.5M training windows).
