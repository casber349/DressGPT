import torch
import torch.nn as nn

class DressGPT(nn.Module):
    def __init__(self, input_dim=527):
        super(DressGPT, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止過擬合 (Fold 2 崩盤的主因)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)