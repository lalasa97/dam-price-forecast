# DAM Price Forecasting

## Problem
Forecast 15-minute electricity prices (Day-Ahead Market) using historical data.

---

## Data
- Frequency: 15 minutes (96 blocks/day)
- Period: Apr–Sep (train), Oct (test)

---

## Approach

### EDA
- understand the data through visualizations
- To get better insight for feature engineering.

### Feature Engineering
- Lag features: 96, 192, 672 etc
- Time features: hour, block, day-of-week
- Volatility features
- Buy/sell imbalance

---

### Models
- Random Forest
- LightGBM
- CatBoost
- XGBoost

### Hyperparameter tuning

---

### Baselines
- Naive (price_lag_96)
- Statistical models (ARIMA, SARIMA, Prophet)

---

### Ensemble
- Blended with naive baseline with ML models

---

## Results

===== Individual Model Performance =====

RF     | MAE: 902.09 | RMSE: 1395.88 | MAPE: 16.08%
XGB    | MAE: 947.41 | RMSE: 1422.72 | MAPE: 16.48%
LGB    | MAE: 946.74 | RMSE: 1407.79 | MAPE: 16.32%
CAT    | MAE: 978.52 | RMSE: 1419.23 | MAPE: 17.34%
NAIVE  | MAE: 875.76 | RMSE: 1685.43 | MAPE: 16.94%

===== Ensemble + Naive Blend =====
MAE: 869.58 | RMSE: 1454.30 | MAPE: 15.96%

## Pipeline

```bash
python src/build_dataset.py
python src/tune.py
python src/train.py
python src/evaluate.py
python src/predict.py