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
TARGET_LEN = 500          # Pad/truncate all sequences to this
BATCH_SIZE = 32
EPOCHS = 40               # We'll use early stopping
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

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Column-wise normalization (per sensor dimension):
        x_norm = (x - mean) / std
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
        seq = pad_or_trim(seq)   # (T, 12)
        seq = torch.tensor(seq).T    # (12, T) for Conv1d
        return seq, label

# ---------------------------------------------------------
# POSITIONAL ENCODING FOR TRANSFORMER
# ---------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding.
    Assumes input shape: (B, T, D).
    """
    def __init__(self, d_model, max_len=5120):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ---------------------------------------------------------
# MODEL: CNN + Transformer
# ---------------------------------------------------------

class CNN_Transformer(nn.Module):
    def __init__(self, num_classes=10, seq_len=TARGET_LEN):
        super().__init__()

        # CNN front-end: feature extractor over time
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(2),     # T -> T/2

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.MaxPool1d(2),     # T -> T/4
        )

        # After two MaxPool1d(2): effective length = seq_len // 4
        self.reduced_len = seq_len // 4
        d_model = 256

        # Positional encoding + Transformer encoder
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.reduced_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True  # input shape: (B, T, D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, 12, T)
        x = self.cnn(x)          # (B, 256, T')
        x = x.transpose(1, 2)    # (B, T', 256)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, T', 256)

        # Transformer encoder
        x = self.transformer(x)  # (B, T', 256)

        # Global average pooling over time instead of just last token
        x = x.mean(dim=1)        # (B, 256)

        # Classification
        return self.fc(x)

# ---------------------------------------------------------
# LOAD DATA FILES
# ---------------------------------------------------------

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print("Found", len(files), "files")

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

model = CNN_Transformer(num_classes=10, seq_len=TARGET_LEN).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3
)

best_loss = float("inf")
patience = 7
patience_counter = 0

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)

    # LR scheduler step
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(epoch_loss)
    new_lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch}/{EPOCHS}, Loss = {epoch_loss:.4f} (LR: {old_lr:.6f} -> {new_lr:.6f})")

    # Early stopping tracking
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), "cnn_transformer_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

print("\nTraining complete!")

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

model.load_state_dict(torch.load("cnn_transformer_best.pt"))
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

torch.save(model.state_dict(), "cnn_transformer_final.pt")
print("Final model saved to cnn_transformer_final.pt")
