# debug_shapes.py
import os
import pandas as pd

DATA_DIR = "./data"

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)

    # drop timestamp if exists
    if df.columns[0].lower() in ["timestamp", "time", "ts"]:
        df = df.drop(df.columns[0], axis=1)

    # convert numeric
    df = df.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)

    rows, cols = df.shape

    if rows < 470:
        print("❌ Too SHORT:", fname, rows, "rows")
    if cols != df.shape[1]:
        print("❌ Column mismatch:", fname, cols)
    if rows >= 470:
        df = df.iloc[:470]
        print("OK:", fname, df.shape)
