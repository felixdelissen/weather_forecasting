# Aquatic Capital — Weather Station Temperature Forecasting

Interview exercise: predict the next hourly temperature at Chicago beach weather stations.

**Dataset**: Chicago Open Data — 3 stations (Foster, 63rd Street, Oak Street), hourly, 2016.  
**Task**: Given all observations up to time t, predict temperature at t+1.  
**Model**: Pooled Ridge regression with station intercept dummies.  
**Result**: MAE 0.425°C vs 0.470°C for the persistence baseline — **~10% gain**.

---

## Structure

```
aquatic_interview_project/
├── temperature.csv      # raw data (3 stations, 8784 hourly slots, 2016)
├── model.py             # full pipeline: EDA → features → CV → evaluation
├── notes.md             # interview prep notes
└── README.md
```

---

## Key design decisions

### Missing values
The missing data is **implicit** (absent rows, not NaN). `resample("1h")` materialises gaps as explicit NaN. Small gaps (≤3h) are filled with linear interpolation — for features only. Imputed values are **never used as targets** (mask on `observed`).

### Features
| Feature | Justification |
|---|---|
| lag_1, lag_2 | ACF = 0.996 at lag 1 — dominant signal |
| dtemp = lag_1 − lag_2 | Momentum: air mass keeps moving |
| hour_sin/cos | Daily cycle, continuous encoding (no cliff at midnight) |
| doy_sin/cos | Annual cycle, continuous encoding (no cliff at Dec 31) |
| Station dummies | Shared slopes, separate intercepts (coefficients ~0.02°C — near zero) |

Rolling mean dropped: low-pass filter, systematically late on the daily cycle (MAE ~2.3°C).

### Model
Ridge over Lasso: features are highly collinear (lags correlated at 0.99+). Lasso picks arbitrarily among correlated features and is unstable; Ridge distributes weight across the group. With p=8 features and n≈15k, regularisation is nearly inactive (α=1 ≈ α=0.01 in validation).

### Evaluation
- No K-fold: autocorrelation of 0.996 means t-1 and t+1 in train would leak t in test.
- `TimeSeriesSplit` expanding window CV — train on past, predict next block.
- Alpha tuned inside the train, test set touched once at the end.

### Stationarity
Series is **non-stationary type (a)**: X_t = μ(t) + ε_t with μ(t) a deterministic seasonal mean. Remedy: encode μ(t) via sin/cos features inside the model. Differencing would be wrong (no unit root — process mean-reverts to the seasonal baseline).

Effect: raw ACF at lag 168h = 0.806. After removing μ(t): **0.101**. 80% of long-lag autocorrelation was the shared summer/winter level, not true persistence.

### Station correlation
- Levels: 0.995 (inflated by shared seasonality)
- Deseasonalised residuals: **0.98** (true co-movement of weather anomalies)
- Hourly differences: 0.53–0.71 (fine-grained timing differs between stations)

Pooling slopes across stations is justified (0.98 residual correlation, dummy coefficients ~0.02°C). But station histories must never be mixed: each prediction uses its own station's lags.

---

## Results

| Fold | Persistence MAE | Ridge MAE | Ridge RMSE |
|---|---|---|---|
| 0 | 0.495 | 0.514 | 0.882 |
| 1 | 0.636 | 0.630 | 1.261 |
| 2 | 0.693 | 0.652 | 1.133 |
| 3 | 0.591 | 0.546 | 0.916 |
| 4 | 0.582 | 0.499 | 0.820 |
| **TEST** | **0.470** | **0.425** | **0.630** |

Fold 0 underperforms: annual sin/cos features require at least one full cycle to be useful. Folds 2–4 and the final test are reliable. Residual errors concentrated on spring cold fronts (15°C/h drops, simultaneous across stations — true meteorological events, not outliers).

---

## Run

```bash
pip install numpy pandas scikit-learn
python model.py
```
