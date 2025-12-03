import os, re, glob
import numpy as np
import pandas as pd
from config import DATA_DIR, SENSOR_COLS

def extract_label_from_filename(path):
    """Example: '3_20251201_123000.csv' -> 3"""
    fname = os.path.basename(path)
    m = re.match(r"(\d+)_", fname)
    if not m:
        raise ValueError(f"Cannot extract label from {fname}")
    return int(m.group(1))

def extract_features_from_df(df):
    df = df.dropna().reset_index(drop=True)
    feats = {}

    for col in SENSOR_COLS:
        s = df[col].astype(float)
        feats[f"{col}_mean"]   = s.mean()
        feats[f"{col}_std"]    = s.std()
        feats[f"{col}_min"]    = s.min()
        feats[f"{col}_max"]    = s.max()
        feats[f"{col}_p25"]    = s.quantile(0.25)
        feats[f"{col}_p75"]    = s.quantile(0.75)
        feats[f"{col}_energy"] = np.mean(s**2)
        feats[f"{col}_first"]  = s.iloc[0]
        feats[f"{col}_last"]   = s.iloc[-1]

    return feats

def build_dataset():
    feature_rows = []
    labels = []

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print("Found", len(files), "CSV files")

    for path in files:
        df = pd.read_csv(path)

        if len(df) < 300:
            print("Skipping short file:", path)
            continue

        feats = extract_features_from_df(df)
        label = extract_label_from_filename(path)

        feature_rows.append(feats)
        labels.append(label)

    X = pd.DataFrame(feature_rows)
    y = np.array(labels)

    return X, y
