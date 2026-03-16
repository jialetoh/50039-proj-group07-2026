import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for unsupervised cable anomaly detection.

    Architecture (from proposal):
      Input  : [B, 3, 256, 256]

      Encoder
        Conv1  : Conv2d(3→16,  3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 16,  128, 128]
        Conv2  : Conv2d(16→32, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 32,   64,  64]
        Conv3  : Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 64,   32,  32]  ← bottleneck

      Decoder
        DeConv3: ConvTranspose2d(64→32, 3x3, stride=2, pad=1, out_pad=1) → BN → ReLU  →  [B, 32, 64,  64]
        DeConv2: ConvTranspose2d(32→16, 3x3, stride=2, pad=1, out_pad=1) → BN → ReLU  →  [B, 16, 128, 128]
        DeConv1: ConvTranspose2d(16→3,  3x3, stride=2, pad=1, out_pad=1) → Sigmoid    →  [B,  3, 256, 256]

      Output : [B, 3, 256, 256]  (same shape as input)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Conv1: 256x256x3 → 128x128x16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2: 128x128x16 → 64x64x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3: 64x64x32 → 32x32x64  (bottleneck)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            # DeConv3: 32x32x64 → 64x64x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # DeConv2: 64x64x32 → 128x128x16
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # DeConv1: 128x128x16 → 256x256x3
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
