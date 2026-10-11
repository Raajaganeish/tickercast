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

### Phase 2: Model check + baselines (next)
- Re-check PatchTST vs alternatives (LSTM, TFT, LightGBM) with the real dataset size.
- Baselines: "always up", "same direction as today", logistic regression, LightGBM.
- These set the score PatchTST must beat.

### Phase 3: PatchTST training
- Plain PyTorch implementation, training loop with early stopping on validation loss.
- Evaluate: accuracy vs baselines, ROC-AUC, per-year results, simple backtest with costs.
- Experiment: 500k vs 5M vs 20M parameters to see overfitting.

### Phase 4: Richer data
See [07_future_data.md](07_future_data.md): fundamentals, 8-K events, news sentiment.
After each addition: retrain and check whether the test score actually improves.

## What "success" means

| Result | Interpretation |
|---|---|
| Test accuracy ≈ share of up days (~51–53%) | Model learned nothing beyond "mostly up" |
| A steady 1–3 points above that, across years | Real (small) signal |
| 60%+ | Almost certainly a bug or leak. Investigate before celebrating |
