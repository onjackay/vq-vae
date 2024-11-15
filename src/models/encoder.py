# encoder.py
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dims, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.encoder(x)