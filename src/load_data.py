import pandas as pd
import os


def load_dam(files, base_dir):
    """
    Load raw DAM CSV files.
    """
    frames = []

    for f in files:
        path = os.path.join(base_dir, f)
        df = pd.read_csv(path)
        print(f"Loaded {f}: {df.shape}")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)