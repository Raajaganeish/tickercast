# 09 · Training PatchTST to forecast tomorrow's range

Full numbers: [reports/05_patchtst.md](../reports/05_patchtst.md).

## The task

For each stock, each day, look at the last **128 trading days** and predict **three numbers
about tomorrow's change**:

| Output | Meaning | Example |
|---|---|---|
| **low** | Only a 10% chance it falls further than this | −2.1% |
| **mid** | Best single guess | +0.1% |
| **high** | Only a 10% chance it rises more than this | +2.3% |

Why a range and not "up or down": direction is barely predictable, but **how big a move
is likely** is fairly predictable (calm stays calm, jumpy stays jumpy). See the Phase 2
discussion in [08_baselines.md](08_baselines.md).

## How one training example looks

```
input:  last 128 days of one stock × N numbers per day     shape [128, N]
        + which stock (ticker id) + which sector (sector id)
answer: that stock's change on the next day
```

Sliding the 128-day block one day at a time over ~500 stocks × ~20 years gives 1.46M
training examples. A batch is 512 of them: shape `[512, 128, N]`.

## How the Transformer reads 128 days

1. **Patches:** cut 128 days into 16 chunks of 8 days. Each chunk becomes one "token".
2. **Position + identity:** each token is tagged with its position (oldest → newest) plus the
   stock's and sector's learned vectors.
3. **Attention:** each chunk looks at all the others to decide which past weeks matter now
   (e.g. "volatility has risen for 4 weeks in a row").
4. **Output:** the 16 chunks are combined into 3 raw numbers, turned into an ordered range:
   `low = mid − positive`, `high = mid + positive`, so low < mid < high always holds.

## How it learns

| Step | What happens |
|---|---|
| Predict | For 512 examples, guess (low, mid, high) |
| Score | **Pinball loss**: if the real move lands above "high", the penalty is 9× bigger than if "high" was a bit too wide. This pushes "high" to where only ~10% of moves exceed it (same for "low"). For "mid" it is plain absolute error |
| Adjust | **AdamW** nudges the weights to lower the penalty. Gradient clipping stops rare huge days from causing wild jumps |
| Repeat | 2,855 batches = 1 epoch (all 1.46M examples once), in a new random order each epoch |
| Check | After each epoch, score 2020–2021 (never trained on) |
| Stop | Keep the best epoch on 2020–2021; stop after 2 epochs without improvement (**early stopping**) |

Learning rate: short warm-up, then cosine decay (big steps early, small steps at the end).

## What we tried (all decided on 2020–2021 only)

| Run | Inputs | Range expressed in | Best size | Val penalty | Coverage | Lesson |
|---|---|---|---|---|---|---|
| 1 | all 66 | fixed % | small | 0.602 | 66% | Ranges too narrow in the 2020 crash: learned "normal" % sizes from calmer years, and inputs capped at ±5 std hid how extreme 2020 was |
| 2 | all 66 | units of the stock's 20-day volatility | small | 0.588 | 73% | Adapts to calm vs wild periods. Still memorizing after 1 epoch |
| 3 | **24 stock-specific** | **volatility units** | **small** | **0.565** | **79%** | No date fingerprint to memorize: training and validation stay close. Best run |

"Volatility units" means the model says "1.3 × this stock's normal daily move" instead of
"2.1%", and we multiply back. The simple volatility-band rule works the same way with a
fixed multiplier; the Transformer learns when to widen or narrow it.

**Bigger was never better.** In every run the smallest model (0.12–0.14M parameters) won, and
larger models (up to 5M) memorized faster. With noisy daily stock data, ~1.5M examples
support only a small model.

## Final results

Models compared (all forecast low / mid / high):

| Model | How |
|---|---|
| Fixed band | Same range for every stock and day |
| Volatility band | ±1.28 × the stock's 20-day volatility |
| LightGBM | 3 quantile models on today's row (no 128-day history) |
| LightGBM, stock-only | Same, 24 stock-specific inputs |
| **PatchTST** | Run 3 above (128-day history, 24 stock-specific inputs) |

| Model | Val 2020–21 penalty | **Test 2022+ penalty** | Test coverage (aim 80%) | Test width~move |
|---|---|---|---|---|
| Fixed band | 0.621 | 0.502 | 76.3% | 0.00 |
| Volatility band | 0.581 | 0.492 | 81.0% | 0.30 |
| LightGBM | 0.573 | 0.483 | 78.9% | 0.36 |
| **LightGBM, stock-only** | **0.565** | **0.481** | 79.2% | **0.38** |
| **PatchTST** | **0.565** | 0.486 | **80.7%** | 0.37 |

(Penalties are lower in the test years because 2022+ was calmer than 2020's crash.)

### What this means

1. ✅ **The Transformer learned something real.** It clearly beats the fixed band and the
   classic volatility-band rule on years it never saw, and its ranges are the best
   calibrated (80.7% inside, close to the 80% target).
2. ⚠️ **It did not beat LightGBM.** LightGBM sees only today's row and matched it on
   validation and edged it on test. The 128-day history didn't add information beyond what
   the summary columns (`vol_20d`, `vol_60d`, `ret_20d`, …) already contain.
3. ❌ **Direction is still close to unpredictable.** The middle guess's stock-picking skill
   (mid daily IC) was +0.024 on validation but only +0.006 on test.
4. The range **widens when big moves are likely**, but nobody can predict a surprise like an
   earnings jump the day before.

This is a common, honest outcome in financial ML: a well-engineered single-row model is
very hard to beat, and the Transformer's extra capacity mostly goes into memorizing noise.

## Ideas for next time

| Idea | Why it might help |
|---|---|
| Add earnings dates (`days_to_next_earnings`) | The biggest surprise moves happen on earnings days. Knowing one is tomorrow should widen the range |
| Stronger regularization (more dropout, no ticker embedding) | Every model peaked after 1–2 epochs |
| Train on more stocks (e.g. Russell 1000) | More data supports a bigger model |
| Ensemble: average PatchTST + LightGBM ranges | They make different mistakes |
| Forecast 5 days ahead | Longer horizons are less noisy |

## Reproduce

```bash
tickercast train --features stock_only --target-mode vol_scaled --sizes small,medium
tickercast train --features all --target-mode vol_scaled --sizes small,medium,large
tickercast train --features all --target-mode raw            # first attempt, 4 sizes (~36 min)
tickercast evaluate --runs stock_only_vol_scaled,all_vol_scaled,all_raw
```
