import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------

DATA_DIR = "data"
TARGET_LEN = 500     # Pad/truncate all sequences to this
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001

SENSOR_COLS = [
    "accX","accY","accZ",
    "gyroX","gyroY","gyroZ",
    "rotX","rotY","rotZ",
    "magX","magY","magZ"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ------------------------------
# HELPERS
# ------------------------------

def extract_label(path):
    fname = os.path.basename(path)
    return int(fname.split("_")[0])

def load_sequence(path):
    df = pd.read_csv(path)
    df = df.dropna()
    seq = df[SENSOR_COLS].values.astype(np.float32)
    return seq

def pad_or_trim(seq, target_len=TARGET_LEN):
    L = seq.shape[0]
    if L > target_len:
        return seq[:target_len]
    if L < target_len:
        pad = np.zeros((target_len - L, seq.shape[1]), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)
    return seq

# ------------------------------
# DATASET CLASS
# ------------------------------

class SensorDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = extract_label(path)
        seq = load_sequence(path)
        seq = pad_or_trim(seq)

        # PyTorch expects: channels × length
        seq = torch.tensor(seq).T    # shape: 12 × 500

        return seq, label

# ------------------------------
# MODEL: 1D CNN
# ------------------------------

class CNN1D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * (TARGET_LEN // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

# ------------------------------
# LOAD DATA FILES
# ------------------------------

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print("Found", len(files), "files")

# Optional: filter very short/long files
clean_files = []
for f in files:
    df = pd.read_csv(f)
    L = len(df)
    if 350 <= L <= 9000:   # adjust as needed
        clean_files.append(f)

print("Using", len(clean_files), "clean files")

train_files, test_files = train_test_split(
    clean_files, test_size=0.2, random_state=42, stratify=[extract_label(f) for f in clean_files]
)

train_ds = SensorDataset(train_files)
test_ds  = SensorDataset(test_files)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ------------------------------
# TRAINING
# ------------------------------

model = CNN1D().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Starting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), torch.tensor(y).to(DEVICE)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {total_loss/len(train_loader):.4f}")

print("\nTraining complete!")

# ------------------------------
# EVALUATION
# ------------------------------

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEVICE)
        out = model(X)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds))

print("=== Confusion Matrix ===")
print(confusion_matrix(all_labels, all_preds))

# ------------------------------
# SAVE MODEL
# ------------------------------

torch.save(model.state_dict(), "cnn_model.pt")
print("Model saved to cnn_model.pt")
