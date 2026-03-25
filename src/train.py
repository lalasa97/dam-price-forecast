import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from config import RF_PARAMS, XGB_PARAMS, LGB_PARAMS, CAT_PARAMS


# =========================
# Paths
# =========================
DATA_PATH = "data/processed/full_features.parquet"
OUTPUT_PATH = "outputs/predictions/all_model_preds.csv"

# =========================
# Models
# =========================
models = {
    "rf": RandomForestRegressor(**RF_PARAMS),
    "xgb": XGBRegressor(**XGB_PARAMS),
    "lgb": LGBMRegressor(**LGB_PARAMS),
    "cat": CatBoostRegressor(**CAT_PARAMS),
}


# =========================
# Main
# =========================
def main():

    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    # =========================
    # Split
    # =========================
    TEST_START_DATE = "2023-10-01"

    train_df = df[df["datetime"] < TEST_START_DATE].copy()
    test_df  = df[df["datetime"] >= TEST_START_DATE].copy()

    feature_cols = [c for c in df.columns if c not in ["datetime", "price"]]

    X_train = train_df[feature_cols]
    y_train = train_df["price"]

    X_test = test_df[feature_cols]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # =========================
    # Prediction DF
    # =========================
    pred_df = test_df[["datetime", "price"]].copy()

    # =========================
    # Train + Predict
    # =========================
    for name, model in models.items():

        print(f"\nTraining {name.upper()}...")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        pred_df[name] = preds

        # optional: save model
        joblib.dump(model, f"outputs/models/{name}.pkl")

    # =========================
    # Naive baseline
    # =========================
    pred_df["naive"] = test_df["price_lag_96"]

    # =========================
    # Save
    # =========================
    pred_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved predictions to {OUTPUT_PATH}")
    print(pred_df.head())


if __name__ == "__main__":
    main()