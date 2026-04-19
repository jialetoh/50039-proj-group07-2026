import torch
import torch.nn as nn

class Autoencoder3Block(nn.Module):
    """
    Baseline 3-block Convolutional Autoencoder. Refer to proposal for full diagram.
    Input: [B, 3, 256, 256] -> Bottleneck: [B, 64, 32, 32] -> Output: [B, 3, 256, 256]
    Compression: Bottleneck is ~33% of the original spatial dimensions.

    Encoder
    Conv1  : Conv2d(3→16,  3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 16,  128, 128]
    Conv2  : Conv2d(16→32, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 32,   64,  64]
    Conv3  : Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 64,   32,  32]
    Decoder
    DeConv3: ConvTranspose2d(64→32,  3x3, stride=2, pad=1, out_pad=1) → BN → ReLU  →  [B, 32, 64,  64]
    DeConv2: ConvTranspose2d(32→16, 3x3, stride=2, pad=1, out_pad=1) → BN → ReLU  →  [B, 16, 128, 128]
    DeConv1: ConvTranspose2d(16→3,  3x3, stride=2, pad=1, out_pad=1) → Sigmoid    →  [B,  3, 256, 256]
    """
    def __init__(self):
        super().__init__()

        # ==========================================
        # Encoder
        # ==========================================
        self.encoder = nn.Sequential(
            # Conv1: 3x256x256 → 16x128x128
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 16x128x128 → 32x64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3: 32x64x64 → 64x32x32  (bottleneck)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # ==========================================
        # Decoder
        # ==========================================
        self.decoder = nn.Sequential(
            # DeConv3: 64x32x32 -> Output: 32x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # DeConv2: 32x64x64 -> Output: 16x128x128
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # DeConv1: 16x128x128 -> Output: 3x256x256
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class Autoencoder4Block(nn.Module):
    pass

class Autoencoder5Block(nn.Module):
    pass
