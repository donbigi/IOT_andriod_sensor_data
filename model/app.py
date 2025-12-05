import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid libomp issue on macOS

import io
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# CONFIG – must match training
# =========================================================

TARGET_LEN = 500
SENSOR_COLS = [
    "accX","accY","accZ",
    "gyroX","gyroY","gyroZ",
    "rotX","rotY","rotZ",
    "magX","magY","magZ",
]
N_INPUT_CHANNELS = 24  # 12 raw + 12 deltas
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "cnn_transformer_v3.pt"

CLUSTERS = {
    0: 0,
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3,
}


def map_to_cluster(z: int) -> int:
    return CLUSTERS.get(z, z)


# =========================================================
# PREPROCESS – exactly as in training
# =========================================================

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    mean = seq.mean(axis=0, keepdims=True)
    std  = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std


def add_deltas(seq_norm: np.ndarray) -> np.ndarray:
    deltas = np.diff(seq_norm, axis=0, prepend=seq_norm[:1])
    return np.concatenate([seq_norm, deltas], axis=1)


def pad_or_trim(seq: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    L = seq.shape[0]
    if L > target_len:
        return seq[:target_len]
    if L < target_len:
        pad = np.zeros((target_len - L, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])
    return seq


def preprocess_csv_bytes(csv_bytes: bytes) -> torch.Tensor:
    df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8"))).dropna()
    seq = df[SENSOR_COLS].values.astype(np.float32)  # (T, 12)
    seq_norm = normalize_sequence(seq)
    seq_full = add_deltas(seq_norm)                  # (T, 24)
    seq_full = pad_or_trim(seq_full, TARGET_LEN)
    tensor = torch.tensor(seq_full, dtype=torch.float32).T.unsqueeze(0)  # (1, 24, T)
    return tensor


# =========================================================
# MODEL – same as training
# =========================================================

class CNNTransformer(nn.Module):
    def __init__(self, num_classes: int = 10):
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

            nn.MaxPool1d(2),
        )

        self.pos_embedding = nn.Embedding(self.seq_len_after_cnn, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 24, T)
        x = self.cnn(x)         # (B, 256, T')
        x = x.transpose(1, 2)   # (B, T', 256)

        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


# =========================================================
# SESSION STATE – NO TIMEOUT HERE
# =========================================================

class SessionState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.digits: List[int] = []
        self.finalized: bool = False
        self.finalized_pin: Optional[str] = None

    def add_digit(self, d: int) -> str:
        # If last session was finalized, start a new one
        if self.finalized:
            self.reset()
        self.digits.append(d)
        pin_str = "".join(str(x) for x in self.digits)
        print(f"[SESSION] New digit: {d} → PIN so far: {pin_str}")
        return pin_str

    def finalize(self) -> Optional[str]:
        if not self.digits:
            return None
        self.finalized = True
        self.finalized_pin = "".join(str(x) for x in self.digits)
        print("\n====================================")
        print(f"FINAL PIN (manual finalize): {self.finalized_pin}")
        print("====================================\n")
        return self.finalized_pin


session = SessionState()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="PIN Prediction Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = CNNTransformer().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("Model loaded on", DEVICE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-tap")
async def predict_tap(file: UploadFile = File(...)):
    """
    One tap = one CSV.
    We predict a *digit* (zone), append to PIN, and return pin_so_far + confidence.
    """
    csv_bytes = await file.read()
    x = preprocess_csv_bytes(csv_bytes).to(DEVICE)  # (1, 24, T)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        zone = int(probs.argmax())
        confidence = float(probs[zone])

    pin_so_far = session.add_digit(zone)
    cluster = map_to_cluster(zone)

    return {
        "zone": zone,
        "cluster": cluster,
        "confidence": confidence,
        "pin_so_far": pin_so_far,
        "num_taps": len(session.digits),
        "finalized": session.finalized,
        "finalized_pin": session.finalized_pin,
    }


@app.post("/finalize-session")
def finalize_session():
    """
    Called by Android (e.g., after 10s of inactivity or when TA taps a button).
    """
    pin = session.finalize()
    return {
        "status": "ok" if pin is not None else "empty",
        "finalized": session.finalized,
        "finalized_pin": session.finalized_pin,
        "num_taps": len(session.digits),
    }


@app.post("/reset-session")
def reset_session():
    """
    Use between attempt 1 and attempt 2 for the same secret PIN.
    """
    session.reset()
    print("[SESSION] Manual reset.")
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
