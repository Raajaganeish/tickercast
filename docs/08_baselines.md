# 08 · Baselines: the scores to beat

Full numbers: [reports/04_baselines.md](../reports/04_baselines.md). Run: `tickercast baselines` (~1 minute).

## Why baselines

A score on its own means nothing. If 52.6% of days are "up" days, a model that **always says
"up"** scores 52.6% without learning anything. Baselines tell us what "good" means before
we train the Transformer.

All models here are trained on **2005–2019** and scored on **2020–2021** (validation).
The test years (**2022+**) are kept locked until the very end.

## The baselines

| Baseline | How it guesses |
|---|---|
| **Majority class** | Always gives the most common answer ("up", or "doesn't beat the market") |
| **Same as today** | If the stock went up today, guess up tomorrow |
| **Opposite of today** | If the stock went up today, guess down tomorrow |
| **Logistic regression** | A weighted sum of today's numbers (simplest real ML model, ~1 second) |
| **LightGBM** | Hundreds of small decision trees on today's numbers (~10–20 seconds) |

## How we score

| Score | Simple meaning | "No skill" value |
|---|---|---|
| **Accuracy** | % of guesses that were right | Majority-class score (~51–53%) |
| **AUC** | Are "likely up" stocks really more often up? | 0.50 |
| **Train AUC vs val AUC** | If train is much higher, the model memorized instead of learning | Close together = healthy |
| **Daily IC** | Each day: do the stocks ranked highest actually do better than those ranked lowest? | 0.00 |
| **Top − bottom 10%** | Each day: average next-day return of the model's 10 favorite % of stocks minus its 10 least favorite % | 0.00% |

**Daily IC** is the most useful for us: it measures *stock-picking skill*, which is what the
model can actually learn. For daily stock data, 0.02–0.05 counts as good.

## Discovery 1: the model memorized dates ⚠️

The first LightGBM run looked strange: it stopped after **3 trees**, and its predictions
were **identical for every stock on the same day**.

Cause:

- 42 of our 66 columns are **market-wide** (S&P 500, VIX, oil, gold, interest rates,
  calendar). They have the **same value for all ~500 stocks on a given day**.
- Together, those 42 numbers are like a **fingerprint for each date**.
- The target "did the stock go up?" is mostly decided by **whether the whole market went up**
  that day (on a good day, most stocks rise together).
- So the model learned "on the day with *this* fingerprint, everything went up". That's
  memorizing answers by question number, not learning.

Evidence: **training AUC 0.62–0.77, validation AUC ~0.52**. Huge gap = memorizing.

Analogy: a student memorizes "question 7's answer is B". It works perfectly on the
practice exam and fails on a new exam.

## Discovery 2: "beat the market" is the learnable question ✅

We tried a second target: **"will this stock do better than the S&P 500 tomorrow?"**
(`label_outperform_1d`). This removes the "whole market went up" part, so there's no date
to memorize. What's left is *which stocks do better than others*.

Results on 2020–2021 (best of each family):

| Target | Best model | Daily IC | Top − bottom 10% / day |
|---|---|---|---|
| Will it go up? | Logistic regression | 0.018 | +0.04% |
| **Will it beat the market?** | **LightGBM, stock-specific inputs** | **0.027** | **+0.12%** |
| **Will it beat the market?** | **Logistic regression, all inputs** | **0.029** | **+0.10%** |
| Will it beat the market? | "Opposite of today" rule | 0.019 | +0.01% |

What "+0.12% per day" means: each day, the 10% of stocks the model liked most beat the 10%
it liked least by 0.12% the next day, on average. That's before trading costs, which would
eat a large part of it. It's a real but small edge, and it appeared on 55% of days.

### Does a higher score really mean a better stock?

Average next-day return vs the market, by model score (LightGBM, stock-specific inputs,
2020–2021). Group 1 = the 10% of stocks the model liked least each day, 10 = liked most:

| Group | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Next-day vs market | −0.09% | −0.03% | −0.01% | −0.02% | −0.00% | −0.02% | +0.01% | +0.01% | +0.04% | +0.02% |

Mostly rising from left to right: higher score → better next day. The clearest signal is
in **avoiding the bottom group**.

## Discovery 3: what about the economy (FRED) data?

| Inputs (target: beat the market) | Daily IC |
|---|---|
| All 66 columns | 0.020 |
| Without the 13 FRED columns | 0.025 |
| Only the 24 stock-specific columns | **0.027** |

For **comparing stocks with each other on the same day**, the FRED columns don't help. Every
stock gets the same values, so they can't tell Apple from Microsoft, and they add more
"date fingerprint" for the model to memorize. They might still be useful for other
questions (e.g. predicting the whole market, or longer horizons), so they stay in the
dataset.

## What the best LightGBM model relied on

Top inputs: **sector** and **sector returns** (today's and this week's), then **overnight
gap**, **distance from the 50-day average**, **today's return vs the market**, **beta**, **today's
return**, **volatility**, **distance from the 1-year high**, **volume**.

In plain words: "how did my sector do, did the stock jump or drop today, is it stretched
above or below its average, and how jumpy is it". These are known short-term patterns:
stocks that jump a lot in one day tend to give a little back.

## Conclusions for the Transformer (Phase 3)

1. **Recommended target: `label_outperform_1d`** (beat the market tomorrow). "Up tomorrow"
   mostly means guessing the whole market's direction, which can't be learned from ~3,500
   training days.
2. **Inputs:** start with the **24 stock-specific columns**. Try all 66 as an experiment,
   but expect memorization: a 128-day window of market-wide columns is an even stronger
   date fingerprint.
3. **The bar to beat:** daily IC **≈ 0.027–0.029** and top − bottom ≈ **+0.10–0.12%/day** on 2020–2021.
   If the Transformer can't beat logistic regression, the 128 days of history aren't helping.
4. **Watch train vs validation** every epoch and stop early. Memorizing is the main danger.

## Model size check (measured on this Mac, PyTorch MPS)

| Size | Parameters | Speed | One full pass over 1.46M windows | Passes in 50 min |
|---|---|---|---|---|
| small | 0.14M | 65,000 / s | 0.4 min | ~130 |
| medium | 0.67M | 30,000 / s | 0.8 min | ~60 |
| large | 1.6M | 17,000 / s | 1.4 min | ~35 |
| xlarge | 5.0M | 6,800 / s | 3.6 min | ~14 |

Training is much faster than first estimated. Even the 5M model fits 14 passes in 50
minutes. Given how quickly the baselines started memorizing, **bigger is not better here**.

Recommended plan for the 45–60 minute budget: **train all four sizes** (≈10 passes each with
early stopping, ~60 min total) and keep the one with the best validation IC. That turns
the size question into a measured answer.
