import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =========================================================
# CONFIG — must match training parameters exactly
# =========================================================

TARGET_LEN = 500
SENSOR_COLS = [
    "accX","accY","accZ",
    "gyroX","gyroY","gyroZ",
    "rotX","rotY","rotZ",
    "magX","magY","magZ"
]
N_INPUT_CHANNELS = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cluster mapping (optional)
CLUSTERS = {
    0: 0,
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3
}

def map_to_cluster(z):
    return CLUSTERS.get(z, z)

# =========================================================
# PREPROCESSING HELPERS (same as training)
# =========================================================

def normalize_sequence(seq):
    mean = seq.mean(axis=0, keepdims=True)
    std  = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std

def add_deltas(seq_norm):
    deltas = np.diff(seq_norm, axis=0, prepend=seq_norm[:1])
    return np.concatenate([seq_norm, deltas], axis=1)

def pad_or_trim(seq, target_len=TARGET_LEN):
    L = seq.shape[0]
    if L > target_len:
        return seq[:target_len]
    if L < target_len:
        pad = np.zeros((target_len - L, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])
    return seq


# =========================================================
# MODEL — must exactly match training architecture
# =========================================================

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
        x = self.cnn(x)           # (B, 256, T')
        x = x.transpose(1, 2)     # (B, T', 256)

        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(csv_path, model_path="cnn_transformer_v3.pt"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).dropna()
    seq = df[SENSOR_COLS].values.astype(np.float32)

    # preprocessing
    seq_norm = normalize_sequence(seq)
    seq_full = add_deltas(seq_norm)
    seq_full = pad_or_trim(seq_full)

    tensor = torch.tensor(seq_full, dtype=torch.float32).T.unsqueeze(0)  # (1, 24, T)

    # load model
    model = CNNTransformer().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        pred = logits.argmax(dim=1).item()

    cluster = map_to_cluster(pred)

    return pred, cluster


# =========================================================
# CLI USAGE
# =========================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_zone.py path/to/file.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    pred, cluster = predict(csv_path)

    print("\n======================")
    print("Prediction Results")
    print("======================")
    print(f"Predicted Zone       : {pred}")
    print(f"Predicted Cluster    : {cluster}")
    print("======================\n")
