import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DATA_DIR = "data"
TARGET_LEN = 500          # BEST for your current dataset
BATCH_SIZE = 32
EPOCHS = 40               # we will early-stop so this is fine
LR = 0.001

SENSOR_COLS = [
    "accX","accY","accZ",
    "gyroX","gyroY","gyroZ",
    "rotX","rotY","rotZ",
    "magX","magY","magZ"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def extract_label(path):
    fname = os.path.basename(path)
    return int(fname.split("_")[0])

def normalize_sequence(seq):
    """
    Column-wise normalization (per sensor dimension)
    (x - mean) / std
    """
    mean = seq.mean(axis=0)
    std  = seq.std(axis=0) + 1e-8
    return (seq - mean) / std

def load_sequence(path):
    df = pd.read_csv(path).dropna()
    seq = df[SENSOR_COLS].values.astype(np.float32)
    seq = normalize_sequence(seq)
    return seq

def pad_or_trim(seq, target_len=TARGET_LEN):
    L = seq.shape[0]
    if L > target_len:
        return seq[:target_len]
    if L < target_len:
        pad = np.zeros((target_len - L, seq.shape[1]), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)
    return seq

# ---------------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------------

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
        seq = torch.tensor(seq).T   # (12, T)
        return seq, label

# ---------------------------------------------------------
# MODEL: CNN + BiLSTM hybrid
# ---------------------------------------------------------

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(2),     # T → T/2

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.MaxPool1d(2),     # T → T/4
        )

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128*2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, 12, T)
        x = self.cnn(x)           # (B, 256, T/4)
        x = x.transpose(1, 2)     # (B, T/4, 256)

        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # last timestep only

        return self.fc(last)

# ---------------------------------------------------------
# LOAD DATA FILES
# ---------------------------------------------------------

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print("Found", len(files), "files")

# Filter extremes
clean_files = []
for f in files:
    L = len(pd.read_csv(f))
    if 300 <= L <= 9000:
        clean_files.append(f)

print("Using", len(clean_files), "clean files")

train_files, test_files = train_test_split(
    clean_files,
    test_size=0.2,
    random_state=42,
    stratify=[extract_label(f) for f in clean_files]
)

train_ds = SensorDataset(train_files)
test_ds  = SensorDataset(test_files)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------------------------------------------------
# TRAINING (with early stopping)
# ---------------------------------------------------------

model = CNN_LSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3
)

best_loss = float("inf")
patience = 7
patience_counter = 0

print("\nStarting training...\n")

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    scheduler.step(epoch_loss)

    print(f"Epoch {epoch}/{EPOCHS}, Loss = {epoch_loss:.4f}")

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), "cnn_lstm_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

print("\nTraining complete!")

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

model.load_state_dict(torch.load("cnn_lstm_best.pt"))
model.eval()

all_preds, all_labels = [], []

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

# ---------------------------------------------------------
# SAVE FINAL MODEL
# ---------------------------------------------------------

torch.save(model.state_dict(), "cnn_lstm_final.pt")
print("Final model saved to cnn_lstm_final.pt")
