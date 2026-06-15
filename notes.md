# Interview Notes — Aquatic Capital

## Setup & goal

**Problem**: time series forecasting — predict the next hourly temperature at a weather station given its history. 3 stations, 1 year, ~8k observations each.

**What they evaluate**: methodology over performance. Baseline first, honest evaluation, clean code.

---

## Data traps (mention these in the first 5 minutes)

- **Missing values are implicit** — absent rows, not NaN. `df.isna().sum() == 0` but hundreds of hours are missing. `resample("1h")` makes them visible.
- **`shift(k)` fails silently on irregular grids** — without a fixed hourly grid, `lag_1` might silently point 13h back. Resample first.
- **`bfill` = future, `ffill` = past**. Never bfill features. Interpolated values can be features but never targets (look-ahead bias if used as targets).
- **File may be sorted reverse-chronologically** (GitHub version). Always `sort_values(["station", "ts"])`.

---

## Stationarity

**Weak stationarity**: E[X_t], Var(X_t), and Cov(X_t, X_{t−k}) depend only on lag k, not on t.  
**Strong stationarity**: all joint distributions are invariant under time translation.

AR/ARMA/ACF assume stationarity — otherwise you average different regimes.

**Our series**: non-stationary type (a) — deterministic seasonality.  
X_t = μ(t) + ε_t, where μ(t) is the seasonal mean (fixed function of calendar) and ε_t is stationary.

**Remedy**: encode μ(t) inside the model with sin/cos features. Do NOT difference (no unit root, process mean-reverts).

**How to check**: rolling 30-day mean and variance, ADF test (H0 = unit root), KPSS test (H0 = stationary). Use both — opposite nulls.

**The trap**: raw ACF at lag 168h = 0.806. After removing μ(t): 0.101. 80% was just "July looks like July", not true persistence. Same trap for inter-station correlation: 0.995 in levels → 0.98 on deseasonalised residuals → 0.53–0.71 on hourly diffs.

---

## Autocorrelation

ACF(k) = Cov(X_t, X_{t−k}) / Var(X_t) under stationarity (= Pearson correlation between X_t and its lag-k copy).

**`statsmodels.acf`**: subtracts global mean, divides by global variance — assumes one mean, one variance across the whole series. Mathematically consistent (positive semi-definite).  
**`pandas.autocorr(k)`**: Pearson on overlapping pairs, with their own local mean/std. Negligible difference on 8k points, visible on short series.

**PACF** (partial autocorrelation): contribution of lag k *given* lags 1..k-1. The right tool to decide how many lags to include.  
Example on this data: ACF of lag_3 = 0.985 (looks important), PACF of lag_3 = −0.05 (adds nothing once lag_1 and lag_2 are included).

---

## Feature engineering rationale

| Feature | Why |
|---|---|
| lag_1, lag_2 | ACF = 0.996 at lag 1 — the single most informative feature |
| dtemp = lag_1 − lag_2 | Discrete derivative — air mass inertia (2nd strongest coefficient) |
| hour_sin/cos | Daily cycle. Raw integer hour: cliff at midnight, can't represent a bump. (sin, cos) = circle coordinates, no cliff, linear model learns any phase with 2 coefficients |
| doy_sin/cos | Annual cycle. Same argument — cliff at Dec 31 with raw integers |
| Station dummies | Shared dynamics, station-specific intercept. Coefficients ~0.02°C here (stations on the same lake) |

**Dropped**: `rmean_24` (low-pass filter, systematically late on the daily cycle, MAE ~2.3°C), `lag_24` (sin/cos already captures the daily cycle).

**Ablation results**: with sin/cos MAE=0.434, without MAE=0.448, with raw integer hour/day MAE=0.698 (worse than nothing — forces a line onto a cycle).

**Anti-leakage rule**: every feature at row t uses only information ≤ t. Hence `s.shift(1).rolling(w).mean()` not `s.rolling(w).mean()`.

---

## Model choice: Ridge over Lasso

Features are highly correlated (lags correlated at 0.99+, rmean partially redundant). Lasso picks arbitrarily among correlated features — unstable (resample and it picks a different one). Ridge distributes weight across the group and is stable.

Sparsity (Lasso's strength) is useful when p is large and most features are noise. Here p=8, and feature selection was already done by hand via ablation.

With p=8 and n≈15k, regularisation is nearly inactive: α=0.01 to α=10 all give MAE≈0.576 in CV. α must be tuned via `GridSearchCV` with `TimeSeriesSplit` inside the train only.

---

## Evaluation

**Why not K-fold**: autocorrelation = 0.996. If t is in the test, t−1 and t+1 in the train are near-perfect copies. Score would be fictional.

**Correct approach**: `TimeSeriesSplit` — expanding window, train on past, test on next block. Never shuffle.

**Two fits only**:
1. Grid search on `X_train` (finds alpha via TimeSeriesSplit internally)
2. Refit on full `X_train` with best alpha (automatic via `refit=True`) → `grid.predict(X_test)`

`X_test` is never seen during fitting or alpha selection.

---

## Why MAE and not MSE?

Ridge minimises MSE (L2 loss) → RMSE is the consistent metric.  
MAE is reported in addition: robust to outliers (spring cold fronts), directly readable ("0.42°C average error"), and its minimiser is the conditional median.  
If RMSE >> MAE: signature of heavy tails (a few large errors). Here RMSE/MAE ≈ 1.5 — spring fronts.

If the official criterion were MAE: use quantile regression (τ=0.5, L1 loss) or Huber loss.

---

## Pooled model vs per-station

| | Per-station | Pooled no dummy | Pooled with dummy |
|---|---|---|---|
| Bias | Low | High if levels differ | Low (dummy absorbs level) |
| Variance | High (1/3 data) | Low | Low |
| Verdict here | OK | Risky | Best |

Deseasonalised residual correlation = 0.98 → same dynamics. Dummy coefficients = 0.02°C → same levels. Pooling is fully justified. If a station had different micro-climate: per-station or interaction terms.

---

## Outliers

`df.isna().sum() == 0` even with outliers — they are valid numbers, just extreme.

Detection layers:
1. Physical bounds (−35 to +45°C for Chicago)
2. `describe()`, IQR rule — but on raw series this is misleading due to seasonality
3. **Z-score on deseasonalised residuals** — the right approach
4. Temporal coherence: |Δ| between consecutive hours
5. Cross-station validation: spike on one station = sensor error; spike on all three = real weather event

On this dataset: no sensor outliers. Drops of 15–16°C/h in spring, but synchronised across all 3 stations → real cold fronts from Lake Michigan. Do not remove.

---

## Rolling mean — why always bad

1. **Phase lag**: rmean_24 at 8am reflects mostly the previous night → systematically too low in the afternoon, too high at night → constant bias → MSE explodes quadratically.
2. **Low-pass filter**: destroys exactly the high-frequency info that predicts the next hour (ACF 0.996 at lag 1 = the instantaneous value dominates).
3. Rolling mean answers "what is the thermal level of the day?" — a different question from "what will the next hour be?". At long horizons (predict 3 days ahead), it would be competitive.
