from features import build_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

print("Loading dataset...")
X, y = build_dataset()

print("Label distribution:")
print(pd.Series(y).value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

print("Training...")
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("Unique predictions:", np.unique(y_pred))
