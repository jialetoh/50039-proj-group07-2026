import torch
import torch.nn as nn
from torchvision import models


class ResNetAutoencoder(nn.Module):
    """
    Pretrained ResNet18 encoder autoencoder for 256x256 RGB inputs.

    Input shape:  [B, 3, 256, 256]
    Encoder out:  [B, 256, 16, 16]
    Output shape: [B, 3, 256, 256]
    """

    def __init__(self, bottleneck_width: int = 256, freeze_encoder: bool = False):
        super().__init__()

        if bottleneck_width < 16:
            raise ValueError("bottleneck_width must be >= 16")

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-3])

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, bottleneck_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_width),
            nn.LeakyReLU(inplace=True),
        )

        ch1 = bottleneck_width
        ch2 = max(16, bottleneck_width // 2)
        ch3 = max(16, bottleneck_width // 4)
        ch4 = max(16, bottleneck_width // 8)

        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(ch1, ch2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch2),
            nn.LeakyReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(ch2, ch3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch3),
            nn.LeakyReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(ch3, ch4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch4),
            nn.LeakyReLU(inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(ch4, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            # Channel projection at full resolution
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# Backward compatibility alias for older imports.
resNetAutoencoder = ResNetAutoencoder
