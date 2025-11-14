# feature_utils.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def _channel_basic_stats(arr):
    # arr: 1D np.array
    return [
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        np.median(arr),
        skew(arr) if arr.size>2 else 0.0,
        kurtosis(arr) if arr.size>3 else 0.0,
    ]


def extract_features_from_df(df, time_col=None, sample_rate=None, pre_seconds=1.0):
    """
    df: pandas DataFrame
    time_col: optional name of timestamp column
    sample_rate: if timestamp missing, provide samples/sec
    returns: 1D numpy array of features
    """
    # Determine numeric data columns (exclude time column)
    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors='coerce')
        if t.notna().all():
            times = (t - t.iloc[0]).dt.total_seconds().to_numpy()
        else:
            times = np.arange(len(df)) / (sample_rate if sample_rate else 1.0)
    else:
        times = np.arange(len(df)) / (sample_rate if sample_rate else 1.0)

    # Which columns are signals?
    sig_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(sig_cols) == 0:
        # if all non-numeric, try converting all to numeric
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        sig_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]

    pre_mask = times <= pre_seconds
    post_mask = times > pre_seconds
    features = []

    # For each signal channel, compute stats on pre and post sections
    for c in sig_cols:
        col = df[c].to_numpy(dtype=float)
        pre = col[pre_mask]
        post = col[post_mask]
        # If empty, fill with zeros
        if pre.size == 0:
            pre = np.zeros(1)
        if post.size == 0:
            post = np.zeros(1)
        features.extend(_channel_basic_stats(pre))
        features.extend(_channel_basic_stats(post))
        # delta stats
        features.append(np.mean(post) - np.mean(pre))
        features.append(np.std(post) - np.std(pre))

    # global features
    flattened = df[sig_cols].to_numpy(dtype=float).ravel()
    features.append(np.nanmean(flattened))
    features.append(np.nanstd(flattened))
    return np.array(features, dtype=float)
