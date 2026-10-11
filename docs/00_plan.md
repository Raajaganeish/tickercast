# 00 · Plan

## Goal

Practice the full machine-learning loop on a real, hard problem:

1. **Prepare data** carefully (this phase).
2. **Train** a Transformer ourselves.
3. **Test** on years the model never saw.
4. **Evaluate** honestly against simple baselines.

The question we answer: *Can a Transformer trained on 20 years of S&P 500 data predict
next-day direction better than "always up" or "same as today"?*

## Decisions so far

| Decision | Choice | Why |
|---|---|---|
| Task | Next-day direction (up/down), binary classification | Easy to judge; regression on returns comes later |
| Stocks | ~500 current S&P 500 members | AAPL alone (~4,000 days) is far too little for a Transformer |
| History | 2005 → today, daily | ~2.4M rows; enough for a ~5M-parameter model |
| Model | PatchTST, ~5M params (6 layers, d_model 256, 8 heads) | Simple, strong time-series Transformer; same patching idea as Chronos-Bolt |
| Hardware | Apple M5 Pro, 64 GB, PyTorch MPS | Training budget 45–60 minutes |
| Input | Last 128 trading days × 53 features + stock/sector IDs | ~6 months of context |
| Split | Train 2005–2019 · Val 2020–2021 · Test 2022 → today | Strictly by time with a 5-day embargo |

## Phases

### Phase 1: Dataset ✅
Price, market, sector, calendar and macro features; three validation reports; leakage tests.

### Phase 2: Model check + baselines ✅
- Re-check PatchTST vs alternatives (LSTM, TFT, LightGBM) with the real dataset size.
- Baselines: "always up", "same direction as today", logistic regression, LightGBM.
- These set the score PatchTST must beat.
- **Result** ([08_baselines.md](08_baselines.md)): with ~500 stocks sharing the same market-wide values each day, models memorized dates when asked "will it go up?". "Will it beat the market?" (`label_outperform_1d`) with stock-specific inputs gives a real signal: daily IC ≈ 0.027. PatchTST trains far faster than estimated (5M params: 3.6 min per pass), so Phase 3 trains four sizes and keeps the best on validation.

### Phase 3: PatchTST training ✅
- Task changed (user decision): forecast tomorrow's **range** (low / mid / high) instead of up/down.
- Result ([09_training.md](09_training.md)): best = small model (0.12M params), 24 stock-specific inputs,
  range in units of recent volatility. Test 2022+: beats fixed and volatility-band rules, best calibration
  (80.7%), slightly behind LightGBM on today's row. Bigger models (up to 5M) memorized faster.

### Phase 4: Richer data
See [07_future_data.md](07_future_data.md): fundamentals, 8-K events, news sentiment.
After each addition: retrain and check whether the test score actually improves.

## What "success" means

| Result | Interpretation |
|---|---|
| Test accuracy ≈ share of up days (~51–53%) | Model learned nothing beyond "mostly up" |
| A steady 1–3 points above that, across years | Real (small) signal |
| 60%+ | Almost certainly a bug or leak. Investigate before celebrating |
