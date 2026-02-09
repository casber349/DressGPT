import torch
import torch.nn as nn

class DressGPT(nn.Module):
    def __init__(self):
        super(DressGPT, self).__init__()
        # 這裡就是你唯一的修改中心
        self.net = nn.Sequential(
            nn.Linear(527, 256), 
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)