# 04 · Point-in-time: the no-peeking rule

## The rule

> A row for day *t* may only contain information that was **public by the close of day *t***.
> Only target columns (`fwd_*`, `label_*`) look forward.

Breaking this rule is called **look-ahead bias** or **leakage**. A leaky model looks
brilliant in testing and fails in real use. It's the most common bug in stock prediction.

## Typical ways to leak, and how we prevent each

| Leak | Example | Prevention in this repo |
|---|---|---|
| Rolling window includes tomorrow | `rolling(..., center=True)`, `shift(-1)` in a feature | All features use backward windows only; enforced by the truncation test |
| Using quarter-end date instead of release date | Q1 earnings (Mar 31) used in April before the May release | Macro uses the official release date (ALFRED); fundamentals will use SEC filing dates |
| Using revised data | CPI revised months later | ALFRED first-release values |
| Daily data dated too early | 10-year yield for day D used on day D | Daily FRED series shifted to D + 1 business day |
| Scaling with full-data statistics | Mean/std computed on 2005–2026, including the test years | Scaler fitted on train rows only ([dataset/prepare.py](../src/tickercast/dataset/prepare.py)) |
| Targets crossing splits | Last train day's 5-day target lands in validation | 5-day embargo at each boundary |
| Shuffling across time | Random train/test split | Split strictly by date |

## How it's tested

### 1. Truncation test (the strongest check)

Build all features twice:
- **A**: with the full history
- **B**: with all data **deleted after date T**

For every row on or before T, A and B must be identical. If a value changes when the
future is removed, that feature was using the future.

Run in `tickercast validate-features` (cutoffs 2012-06-29 and 2020-03-16, 40 random stocks
+ AAPL) and in `tests/test_leakage.py`.

The test is itself tested: `test_truncation_test_catches_a_planted_leak` adds a fake
feature equal to tomorrow's return and asserts the test flags exactly that column.

### 2. Leakage scan

For each feature, compute the average daily rank correlation (Spearman "IC") with the
next-day return. Real signals in daily stock data are tiny (|IC| ≈ 0.01–0.05). Anything
above 0.2 is almost certainly leakage.

Latest run: max |IC| = 0.017 (`ret_1d`, negative: a small short-term reversal effect,
a known market pattern).

### 3. Target alignment

`fwd_ret_1d` on day *t* must equal `ret_1d` on day *t+1* for the same stock.
Latest run: 0 mismatches out of 2.36M.

## Example: a monthly number

```
Unemployment for January 2020 (obs_date 2020-01-01) was published 2020-02-07.

Row date      unemployment
2020-02-06    3.5   ← still December's number
2020-02-07    3.6   ← January's number appears on release day
```
Tested in `tests/test_features.py::test_macro_value_appears_only_from_release_date`.
