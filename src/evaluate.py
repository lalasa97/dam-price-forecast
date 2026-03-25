import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# Paths
# =========================
PRED_PATH = "outputs/predictions/all_model_preds.csv"


# =========================
# Metrics
# =========================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return mae, rmse, mape


# =========================
# Main
# =========================
def main():

    df = pd.read_csv(PRED_PATH)

    y_true = df['price']

    model_cols = ['rf', 'xgb', 'lgb', 'cat', 'naive']

    print("\n===== Individual Model Performance =====\n")

    results = {}

    for col in model_cols:
        mae, rmse, mape = compute_metrics(y_true, df[col])
        results[col] = (mae, rmse, mape)

        print(f"{col.upper():6} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")


    df['ensemble_naive'] = ( 0.5 * df['naive']+
                            0.3 * df['rf']+ 0.1 * df['cat']+
                            0.1 * df['lgb'])
    

    mae, rmse, mape = compute_metrics(y_true, df['ensemble_naive'])

    print("\n===== Ensemble + Naive Blend =====")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # =========================
    # Save final predictions
    # =========================

    df[['datetime', 'ensemble_naive']].rename(
        columns={'ensemble_naive': 'y_pred'}
    ).to_csv("outputs/final_predictions.csv", index=False)

    print("\nFinal predictions saved.")


if __name__ == "__main__":
    main()