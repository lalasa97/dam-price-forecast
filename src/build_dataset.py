import pandas as pd

from load_data import load_dam
from preprocess import preprocess
from feature_engineering import build_features


# =========================
# Config
# =========================
TRAIN_FILES = [
    'DAM_April_2023.csv', 'DAM_May_2023.csv', 'DAM_June_2023.csv',
    'DAM_July_2023.csv', 'DAM_August_2023.csv', 'DAM_September_2023.csv'
]

TEST_FILE = 'DAM_October_2023.csv'
DATA_DIR = 'data/'

OUTPUT_PATH = "data/processed/full_features.parquet"


def main():

    print("Loading data...")

    train_raw = load_dam(TRAIN_FILES, DATA_DIR)
    test_raw  = load_dam([TEST_FILE], DATA_DIR)

    # =========================
    # Preprocess
    # =========================
    print("Preprocessing...")

    train_df = preprocess(train_raw)
    test_df  = preprocess(test_raw)

    # =========================
    # Combine (CRITICAL)
    # =========================
    print("Combining train + test for feature consistency...")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values('datetime').reset_index(drop=True)

    # =========================
    # Feature Engineering
    # =========================
    print("Building features...")

    full_df = build_features(full_df)

    # =========================
    # Final sanity checks
    # =========================
    print("Running checks...")

    assert full_df['datetime'].is_monotonic_increasing
    assert full_df['datetime'].is_unique

    print(f"Final dataset shape: {full_df.shape}")

    # =========================
    # Save
    # =========================
    print(f"Saving to {OUTPUT_PATH}...")

    full_df.to_parquet(OUTPUT_PATH, index=False)

    print("Done.")


if __name__ == "__main__":
    main()