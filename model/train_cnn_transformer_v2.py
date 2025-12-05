import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import deque, Counter

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DATA_DIR   = "data"
TARGET_LEN = 500
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 1e-3

SENSOR_COLS = [
    "accX","accY","accZ",
    "gyroX","gyroY","gyroZ",
    "rotX","rotY","rotZ",
    "magX","magY","magZ"
]

N_INPUT_CHANNELS = 24   # 12 raw + 12 delta channels
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def extract_label(path: str) -> int:
    return int(os.path.basename(path).split("_")[0])

def normalize_sequence(seq):
    mean = seq.mean(axis=0, keepdims=True)
    std  = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std

def add_deltas(seq_norm):
    deltas = np.diff(seq_norm, axis=0, prepend=seq_norm[:1])
    return np.concatenate([seq_norm, deltas], axis=1)

def load_sequence(path):
    df = pd.read_csv(path).dropna()
    seq = df[SENSOR_COLS].values.astype(np.float32)
    seq_norm = normalize_sequence(seq)
    seq_full = add_deltas(seq_norm)
    return seq_full

def pad_or_trim(seq, target_len=TARGET_LEN):
    L = seq.shape[0]
    if L > target_len:
        return seq[:target_len]
    if L < target_len:
        pad = np.zeros((target_len - L, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])
    return seq

# ---------------------------------------------------------
# MAJORITY-VOTE SMOOTHING
# ---------------------------------------------------------

def majority_vote(pred_list, window=5):
    dq = deque(maxlen=window)
    out = []
    for p in pred_list:
        dq.append(p)
        counts = Counter(dq)
        out.append(counts.most_common(1)[0][0])
    return out

# ---------------------------------------------------------
# ZONE CLUSTERING
# ---------------------------------------------------------

CLUSTERS = {
    0: 0,
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3
}

def map_to_cluster(z):
    # safe fallback
    return CLUSTERS.get(z, z)

def cluster_accuracy(true_labels, pred_labels):
    true_c = [map_to_cluster(x) for x in true_labels]
    pred_c = [map_to_cluster(x) for x in pred_labels]
    return sum(1 for a, b in zip(true_c, pred_c) if a == b) / len(true_labels)

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

class SensorDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        y = extract_label(path)

        seq = load_sequence(path)
        seq = pad_or_trim(seq, TARGET_LEN)

        seq = torch.tensor(seq, dtype=torch.float32).T  # (24, T)
        return seq, y

# ---------------------------------------------------------
# MODEL: CNN + TRANSFORMER
# ---------------------------------------------------------

class CNNTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        d_model = 256
        self.seq_len_after_cnn = TARGET_LEN // 4

        self.cnn = nn.Sequential(
            nn.Conv1d(N_INPUT_CHANNELS, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(128, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),

            nn.MaxPool1d(2)
        )

        self.pos_embedding = nn.Embedding(self.seq_len_after_cnn, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)          # (B, 256, T')
        x = x.transpose(1, 2)    # (B, T', 256)

        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ---------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(files)} files")

    clean_files = []
    for f in files:
        try:
            L = len(pd.read_csv(f))
            if 300 <= L <= 9000:
                clean_files.append(f)
        except:
            continue

    print(f"Using {len(clean_files)} clean files")

    labels = [extract_label(f) for f in clean_files]
    train_files, test_files = train_test_split(
        clean_files,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    def show_dist(name, fs):
        labs = [extract_label(f) for f in fs]
        cnt = np.bincount(labs, minlength=10)
        print(f"\n{name} distribution:")
        for i, c in enumerate(cnt):
            print(f"  Zone {i}: {c}")

    show_dist("Train", train_files)
    show_dist("Test", test_files)

    train_loader = DataLoader(SensorDataset(train_files), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(SensorDataset(test_files),  batch_size=BATCH_SIZE)

    model = CNNTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print("\nTraining...\n")
    for epoch in range(1, EPOCHS + 1):
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

        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS} | Loss {total_loss/len(train_loader):.4f} | LR {scheduler.get_last_lr()[0]:.6f}")

    print("\nTraining complete.\n")

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    model.eval()
    raw_preds, raw_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(1).cpu().numpy()
            raw_preds.extend(preds)
            raw_labels.extend(y.numpy())

    # RAW METRICS
    print("\n=== RAW CLASSIFICATION REPORT ===")
    print(classification_report(raw_labels, raw_preds, digits=3))
    print("\n=== RAW CONFUSION MATRIX ===")
    print(confusion_matrix(raw_labels, raw_preds))

    # CLUSTER ACCURACY
    print("\n=== CLUSTER ACCURACY ===")
    print(f"Raw cluster accuracy     : {cluster_accuracy(raw_labels, raw_preds):.3f}")

    torch.save(model.state_dict(), "cnn_transformer_v3.pt")
    print("\nModel saved to cnn_transformer_v3.pt\n")


if __name__ == "__main__":
    main()
