# data_loader.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FIXED_ROWS = 470
MIN_ROWS = 200   # <----- skip CSVs shorter than this


def infer_zone(fname):
    import re
    m = re.search(r"(^\d+)|_(\d+)_|zone(\d+)", fname.lower())
    if not m:
        return None
    return m.group(1) or m.group(2) or m.group(3)


def clean_df(df):
    """Convert dataframe to numeric-only and fill missing values."""
    # drop timestamp if exists
    if df.columns[0].lower() in ["timestamp","time","ts"]:
        df = df.drop(df.columns[0], axis=1)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().fillna(0)
    return df


def pad_or_trim(arr, target=FIXED_ROWS):
    rows, cols = arr.shape

    if rows == target:
        return arr
    elif rows > target:
        return arr[:target]                      # trim
    else:
        pad = np.zeros((target - rows, cols))    # pad
        return np.vstack([arr, pad])


def load_dataset(data_dir):
    X_list = []
    y_list = []
    file_paths = []
    column_set = None

    # FIRST PASS: determine consistent column set
    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue

        zone = infer_zone(fname)
        if zone is None:
            continue

        path = os.path.join(data_dir, fname)
        df = clean_df(pd.read_csv(path))

        # Skip extremely short files
        if len(df) < MIN_ROWS:
            print(f"Skipping {fname}: only {len(df)} rows (< {MIN_ROWS})")
            continue

        # Track consistent columns
        if column_set is None:
            column_set = list(df.columns)

        df = df.reindex(columns=column_set, fill_value=0)
        file_paths.append(path)
        y_list.append(int(zone))

    # SECOND PASS: fit scaler to all rows (numpy only)
    print("Fitting scaler on all rows…")
    all_rows = []

    for path in file_paths:
        df = clean_df(pd.read_csv(path))
        df = df.reindex(columns=column_set, fill_value=0)

        arr = df.to_numpy(float)
        arr = pad_or_trim(arr, FIXED_ROWS)

        all_rows.append(arr)

    all_rows = np.vstack(all_rows)  # (N * 470, C)
    scaler = StandardScaler().fit(all_rows)

    # THIRD PASS: build final dataset
    for path in file_paths:
        df = clean_df(pd.read_csv(path))
        df = df.reindex(columns=column_set, fill_value=0)

        arr = df.to_numpy(float)
        arr = pad_or_trim(arr, FIXED_ROWS)
        arr = scaler.transform(arr)

        X_list.append(arr)

    X = np.stack(X_list)   # (N, 470, C)
    y = np.array(y_list)

    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} rows, {X.shape[2]} channels.")
    return X, y, scaler
