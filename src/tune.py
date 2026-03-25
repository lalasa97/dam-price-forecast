import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# =========================
# Paths
# =========================
DATA_PATH = "data/processed/full_features.parquet"


# =========================
# Models
# =========================
class_models = {
    'rf': RandomForestRegressor,
    'xgb': XGBRegressor,
    'lgb': LGBMRegressor,
    'cat': CatBoostRegressor
}

MODELS = ['rf', 'xgb', 'lgb', 'cat']


# =========================
# Base Params (from config)
# =========================
fixed_params = {
    'rf' : {'n_jobs': -1, 'random_state': 42},
    'xgb': {'subsample': 0.6, 'colsample_bytree': 0.7, 'random_state': 42, 'n_jobs' : -1},
    'lgb': {'n_jobs': -1, 'random_state': 42, 'verbose': -2},
    'cat': {'random_state': 42, 'logging_level':'Silent'}
}


# =========================
# Param Grids (your version)
# =========================
grids = {
    'rf': [
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 300, 'max_depth': 12},
        {'n_estimators': 500, 'max_depth': 15},
    ],
    'xgb': [
        {'n_estimators':300, 'max_depth':6, 'learning_rate':0.05},
        {'n_estimators':500, 'max_depth':8, 'learning_rate':0.03},
        {'n_estimators':700, 'max_depth':10, 'learning_rate':0.02},
    ],
    'lgb': [
        {'n_estimators':800, 'learning_rate':0.03, 'num_leaves':63, 'max_depth':-1, 'min_child_samples':50, 'subsample':0.8, 'colsample_bytree':0.8},
        {'n_estimators':1200, 'learning_rate':0.02, 'num_leaves':63, 'max_depth':-1, 'min_child_samples':50, 'subsample':0.8, 'colsample_bytree':0.8},
        {'n_estimators':1400, 'learning_rate':0.01, 'num_leaves':127, 'max_depth':10, 'min_child_samples':100, 'subsample':0.7, 'colsample_bytree':0.7}
    ],
    'cat': [
        {'iterations':500, 'depth':6, 'learning_rate':0.05},
        {'iterations':800, 'depth':8, 'learning_rate':0.03},
        {'iterations':1000, 'depth':10, 'learning_rate':0.02}
    ]
}


# =========================
# Tuning function
# =========================
def tune_model(model_class, param_grid, X_train, y_train, X_val, y_val, fixed_params=None):

    best_params = None
    best_score = float('inf')

    if fixed_params is None:
        fixed_params = {}

    for params in param_grid:

        all_params = {**params, **fixed_params}

        model = model_class(**all_params)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        score = mean_absolute_error(y_val, pred)

        print("Params:", params, "| MAE:", round(score, 2))

        if score < best_score:
            best_score = score
            best_params = all_params

    return best_params, best_score


# =========================
# Main
# =========================
def main():

    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    # ---------- Time split ----------
    VAL_START_DATE = "2023-09-01"
    TEST_START_DATE = "2023-10-01"

    train_df = df[df['datetime'] < VAL_START_DATE].copy()
    val_df   = df[(df['datetime'] >= VAL_START_DATE) & (df['datetime'] < TEST_START_DATE)].copy()

    feature_cols = [c for c in df.columns if c not in ['datetime', 'price']]

    X_train = train_df[feature_cols]
    y_train = train_df['price']

    X_val = val_df[feature_cols]
    y_val = val_df['price']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # ---------- Tuning ----------
    best_params = {}
    best_scores = {}

    for model_name in MODELS:

        print(f"\n===== {model_name.upper()} =====")

        params, score = tune_model(
            class_models[model_name],
            grids[model_name],
            X_train, y_train,
            X_val, y_val,
            fixed_params[model_name]
        )

        best_params[model_name] = params
        best_scores[model_name] = score

        print("Best Params:", params)
        print("Best MAE:", round(score, 2))

    print("\nFinal Results:")
    for k in best_params:
        print(k, best_params[k], "| MAE:", round(best_scores[k], 2))


if __name__ == "__main__":
    main()