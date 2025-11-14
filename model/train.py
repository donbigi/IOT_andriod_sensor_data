# train.py
import re
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from feature_utils import extract_features_from_df


ZONE_RE = re.compile(r"zone[_-]?(\d+)|ZONE(\d+)|_(\d{1,2})_")


def infer_zone_from_filename(fname):
    m = ZONE_RE.search(fname.lower())
    if not m:
        # fallback: try to find any digit
        digits = re.findall(r"(\d+)", fname)
        return digits[0] if digits else None
    for g in m.groups():
        if g:
            return g
    return None


def load_all_features(data_dir, time_col=None, sample_rate=None):
    X = []
    y = []
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    for f in files:
        zone = infer_zone_from_filename(os.path.basename(f))
        if zone is None:
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"skipping {f}: {e}")
            continue
        feats = extract_features_from_df(df, time_col=time_col, sample_rate=sample_rate)
        X.append(feats)
        y.append(zone)
    X = np.vstack(X)
    return X, np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', default='models/model_bundle.joblib')
    p.add_argument('--time-col', default=None)
    p.add_argument('--sample-rate', type=float, default=None)
    args = p.parse_args()

    X, y = load_all_features(args.data_dir, time_col=args.time_col, sample_rate=args.sample_rate)
    print('Loaded', X.shape, 'samples')

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    # quick cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring='accuracy')
    print('CV accuracy:', scores, 'mean', scores.mean())

    # final fit on all data
    pipeline.fit(X, y_enc)

    # Save everything
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'label_encoder': le}, args.out)
    print('Saved model to', args.out)

    # optional test split report
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
    preds = pipeline.predict(Xte)
    print(classification_report(yte, preds, target_names=le.inverse_transform(sorted(set(yte)))))

if __name__ == '__main__':
    main()
