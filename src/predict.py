import pandas as pd
from pathlib import Path


# =========================
# Paths
# =========================

INPUT_PATH =  "outputs/final_predictions.csv"
OUTPUT_PATH = "outputs/submission.csv"


# =========================
# Helper: create time_period
# =========================
def create_time_period(dt_series):
    start = dt_series.dt.strftime("%H:%M")
    end = (dt_series + pd.Timedelta(minutes=15)).dt.strftime("%H:%M")
    return start + "-" + end


# =========================
# Main
# =========================
def main():

    print("Loading predictions...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])

    # =========================
    # Create required columns
    # =========================
    df["Delivery Date"] = df["datetime"].dt.strftime("%d/%m/%Y")
    df["Time Period"] = create_time_period(df["datetime"])

    # rename prediction column
    df = df.rename(columns={"y_pred": "Predicted Price"})

    # =========================
    # Final format
    # =========================
    submission = df[["Delivery Date", "Time Period", "Predicted Price"]].copy()

    # sort (important)
    submission = submission.sort_values(["Delivery Date", "Time Period"])

    # =========================
    # Save
    # =========================
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Submission saved to {OUTPUT_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()