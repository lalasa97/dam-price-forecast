# DAM Price Forecasting

## Problem
Forecast 15-minute electricity prices (Day-Ahead Market) using historical data.

---

## Data
- Frequency: 15 minutes (96 blocks/day)
- Period: Apr–Sep (train), Oct (test)

---

## Approach

### Feature Engineering
- Lag features: 96, 192, 672
- Time features: hour, block, day-of-week
- Volatility features
- Buy/sell imbalance

---

### Models
- Random Forest
- LightGBM
- CatBoost
- XGBoost

---

### Baselines
- Naive (price_lag_96)
- Statistical models (ARIMA, ETS)

---

### Ensemble
- Mean of ML models
- Blended with naive baseline

---

## Results

| Model | MAE | RMSE | MAPE |
|------|-----|------|------|
| Naive | ~875 | ... | ... |
| RF | ... | ... | ... |
| Ensemble | ... | ... | ... |

---

## Pipeline

```bash
python src/build_dataset.py
python src/tune.py
python src/train.py
python src/evaluate.py
python src/predict.py