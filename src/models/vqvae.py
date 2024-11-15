import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
from .encoder import Encoder
from .decoder import Decoder

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=256,
        codebook_size=512,
        decay=0.8,
        commitment_weight=1.0
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims)
        self.decoder = Decoder(in_channels, hidden_dims)
        
        self.vq = VectorQuantize(
            dim=hidden_dims,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight
        )
        
        self.hidden_dims = hidden_dims
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Reshape for VQ
        z = z.permute(0, 2, 3, 1)  # [B, H, W, C]
        shape = z.shape
        z = z.reshape(-1, shape[-1])  # [B*H*W, C]
        
        # Vector Quantization
        quantized, indices, commit_loss = self.vq(z)
        
        # Reshape back
        quantized = quantized.reshape(shape)  # [B, H, W, C]
        quantized = quantized.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Decode
        x_recon = self.decoder(quantized)
        
        return x_recon, commit_loss, indices.reshape(shape[0], shape[1], shape[2])

    def encode(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1)
        shape = z.shape
        z = z.reshape(-1, shape[-1])
        _, indices, _ = self.vq(z)
        return indices.reshape(shape[0], shape[1], shape[2])

    def decode(self, indices):
        shape = indices.shape
        indices = indices.reshape(-1)
        quantized = self.vq.codebook[indices]
        quantized = quantized.reshape(shape[0], shape[1], shape[2], -1)
        quantized = quantized.permute(0, 3, 1, 2)
        return self.decoder(quantized)