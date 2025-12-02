# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from data_loader import load_dataset
from model import SensorCNN
import joblib
import argparse
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="models/model_bundle.joblib")
    args = p.parse_args()

    print("Loading dataset...")
    X, y, scaler = load_dataset(args.data_dir)

    num_samples, seq_len, num_channels = X.shape
    print("Dataset:", X.shape)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = SensorCNN(num_channels=num_channels, num_classes=len(set(y)))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15

    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0

        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()

        print(f"Epoch {epoch}/{epochs}  loss={total_loss/len(dl):.4f}  acc={correct/len(ds):.3f}")

    # Save model + scaler
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    torch.save(model.state_dict(), "models/cnn_weights.pt")
    joblib.dump({"scaler": scaler}, "models/scaler.joblib")

    print("Saved CNN model + scaler.")

if __name__ == "__main__":
    main()
