# model.py
import torch
import torch.nn as nn

class SensorCNN(nn.Module):
    def __init__(self, num_channels, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.MaxPool1d(2),  # 470 -> 235

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.MaxPool1d(2),  # 235 -> 117
        )

        # Final length is 117, NOT 118
        final_len = 117

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * final_len, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # input x: (batch, seq=470, channels)
        x = x.transpose(1, 2)  # → (batch, channels, seq)
        x = self.net(x)
        x = self.classifier(x)
        return x
