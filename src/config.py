# =========================
# Base Params (used in tuning)
# =========================

RF_BASE_PARAMS = {
    "random_state": 42,
    "n_jobs": -1
}

XGB_BASE_PARAMS = {
    'subsample': 0.6, 
    'colsample_bytree': 0.7, 
    'random_state': 42, 'n_jobs' : -1}

LGB_BASE_PARAMS = {
    'n_jobs': -1, 
    'random_state': 42, 
    'verbose': -2
}

CAT_BASE_PARAMS = {
    'logging_level':'Silent',
    "random_state": 42
}

# =========================
# Final Params (empty for now)
# =========================

RF_PARAMS = {
    'n_estimators': 200, 
    'max_depth': 10, 
    'n_jobs': -1, 
    'random_state': 42}
XGB_PARAMS = {
    'n_estimators': 500, 
    'max_depth': 8, 
    'learning_rate': 0.03, 
    'subsample': 0.6, 
    'colsample_bytree': 0.7, 
    'random_state': 42, 
    'n_jobs': -1
}
LGB_PARAMS = {
    'n_estimators': 800, 'learning_rate': 0.03, 
    'num_leaves': 63, 'max_depth': -1, 
    'min_child_samples': 50, 'subsample': 0.8, 
    'colsample_bytree': 0.8, 'n_jobs': -1, 
    'random_state': 42, 'verbose': -2
}
CAT_PARAMS = {
    'iterations': 1000, 'depth': 10, 
    'learning_rate': 0.02, 'random_state': 42, 
    'logging_level': 'Silent'
}