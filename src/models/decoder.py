
import torch.nn as nn
# decoder.py
class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dims=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)