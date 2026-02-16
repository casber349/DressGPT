import torch
import torch.nn as nn

class DressGPT(nn.Module):
    def __init__(self, input_dim=527):
        super(DressGPT, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)