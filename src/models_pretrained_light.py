import torch
import torch.nn as nn
from torchvision import models


class ResNetAutoencoderLight(nn.Module):
    """
    Lightweight autoencoder with a pretrained ResNet18 encoder.

    Expected input shape: [B, 3, 256, 256]
    Encoder output shape: [B, 512, 8, 8]
    Decoder output shape: [B, 3, 256, 256]
    """

    def __init__(self, bottleneck_width: int = 64, freeze_encoder: bool = False):
        super().__init__()

        if bottleneck_width < 8:
            raise ValueError("bottleneck_width must be >= 8")

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, bottleneck_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_width),
            nn.ReLU(inplace=True),
        )

        ch1 = bottleneck_width
        ch2 = max(8, bottleneck_width // 2)
        ch3 = max(8, bottleneck_width // 4)
        ch4 = max(8, bottleneck_width // 8)

        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(ch1, ch1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(ch1, ch2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(ch2, ch3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(ch3, ch4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch4),
            nn.ReLU(inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(ch4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


def shape_smoke_test(batch_size: int = 2, image_size: int = 256, bottleneck_width: int = 64):
    """Quick shape sanity check for the lightweight autoencoder."""
    model = ResNetAutoencoderLight(bottleneck_width=bottleneck_width)
    x = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        y = model(x)
    return tuple(x.shape), tuple(y.shape)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return trainable parameter count for quick model-size comparison."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Backward compatibility alias for older imports.
resNetAutoencoderLight = ResNetAutoencoderLight
