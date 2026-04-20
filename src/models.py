import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Normalize

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
            # DeConv3: 64x32x32 -> 32x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # DeConv2: 32x64x64 -> 16x128x128
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # DeConv1: 16x128x128 -> 3x256x256
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Autoencoder4Block(nn.Module):
    """
    4-Block Convolutional Autoencoder.
    Same as baseline 3-block but an additional encoder and decoder block.
    Input: [B, 3, 256, 256] -> Bottleneck: [B, 128, 16, 16] -> Output: [B, 3, 256, 256]
    Compression: Bottleneck is ~16.7% of the original spatial dimensions.

    Encoder
    Conv1  : Conv2d(3→16,  3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 16,  128, 128]
    Conv2  : Conv2d(16→32, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 32,   64,  64]
    Conv3  : Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 64,   32,  32]
    Conv4  : Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 128,   16,  16]
    Decoder
    DeConv4: ConvTranspose2d(128→64,  3x3, stride=2, pad=1, out_pad=1) → BN → ReLU  →  [B, 64, 32,  32]
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
            
            # Conv3: 32x64x64 → 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4: 64x32x32 → 128x16x16  (bottleneck)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        
        # ==========================================
        # Decoder
        # ==========================================
        self.decoder = nn.Sequential(
            # DeConv4: 128x16x16 -> 64x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # DeConv3: 64x32x32 -> 32x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # DeConv2: 32x64x64 -> 16x128x128
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # DeConv1: 16x128x128 -> 3x256x256
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Autoencoder5Block(nn.Module):
    """
    5-Block Convolutional Autoencoder.
    Input: [B, 3, 256, 256] -> Bottleneck: [B, 256, 8, 8] -> Output: [B, 3, 256, 256]
    Compression: Bottleneck is ~8.33% of the original spatial dimensions.

    Encoder
    Conv1  : Conv2d(3→16,  3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 16,  128, 128]
    Conv2  : Conv2d(16→32, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 32,   64,  64]
    Conv3  : Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 64,   32,  32]
    Conv4  : Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 128,   16,  16]
    Conv5  : Conv2d(128→256, 3x3, pad=1) → BN → ReLU → MaxPool2d(2,2)  →  [B, 256,   8,  8]
    Decoder
    DeConv4: ConvTranspose2d(256→128,  3x3, stride=2, pad=1, out_pad=1) → BN → ReLU  →  [B, 128, 16,  16]
    DeConv4: ConvTranspose2d(128→64,  3x3, stride=2, pad=1, out_pad=1) → BN → ReLU  →  [B, 64, 32,  32]
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
            
            # Conv3: 32x64x64 → 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4: 64x32x32 → 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv5: 128x16x16 → 256x8x8  (bottleneck)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)      
        )
        
        # ==========================================
        # Decoder
        # ==========================================
        self.decoder = nn.Sequential(
            # DeConv5: 256x8x8 -> 128x16x16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # DeConv4: 128x16x16 -> 64x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # DeConv3: 64x32x32 -> 32x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # DeConv2: 32x64x64 -> 16x128x128
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # DeConv1: 16x128x128 -> 3x256x256
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Autoencoder6Block(nn.Module):
    """
    6-Block Convolutional Autoencoder.
    Input: [B, 3, 256, 256] -> Bottleneck: [B, 512, 4, 4] -> Output: [B, 3, 256, 256]
    Compression: Bottleneck is ~4.167% of the original spatial dimensions.
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
            
            # Conv3: 32x64x64 → 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4: 64x32x32 → 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv5: 128x16x16 → 256x8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv6: 256x8x8 → 512x4x4  (bottleneck)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)     
        )
        
        # ==========================================
        # Decoder
        # ==========================================
        self.decoder = nn.Sequential(
            # DeConv5: 512x4x4 -> 256x8x8
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # DeConv5: 256x8x8 -> 128x16x16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # DeConv4: 128x16x16 -> 64x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # DeConv3: 64x32x32 -> 32x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # DeConv2: 32x64x64 -> 16x128x128
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # DeConv1: 16x128x128 -> 3x256x256
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderResNet18(nn.Module):
    def __init__(self, freeze_encoder=True):
        """
        Autoencoder with pre-trained ResNet18 backbone.
        Encoder weights frozen by default with `freeze_encoder=True`.
        """
        super().__init__()
        
        # ImageNet Normalization so the model accepts standard [0, 1] tensors
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # ==========================================
        # Pre-trained Encoder (ResNet18)
        # ==========================================
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Extract all convolutional layers only so our output is a feature map
        # Last two layers are avgpool and fc, which we don't want
        # Output feature map is [B, 512, 8, 8] for a 256x256 input
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # ==========================================
        # Decoder
        # ==========================================
        self.decoder = nn.Sequential(
            # Block 1: 512x8x8 -> 256x16x16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Block 2: 256x16x16 -> 128x32x32
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Block 3: 128x32x32 -> 64x64x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Block 4: 64x64x64 -> 32x128x128
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Block 5: 32x128x128 -> 3x256x256
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm_x = self.normalize(x)  # Normalize [0, 1] image tensor from the dataloader
        features = self.encoder(norm_x)  
        reconstructed = self.decoder(features) 
        return reconstructed

    def unfreeze_top_encoder_layers(self):
        """
        Unfreeze the last stage of the ResNet18 encoder (layer4)
        to allow for fine-tuning for this dataset.
        """
        # 0 = conv1, 1 = bn1, 2 = relu, 3 = maxpool,
        # 4 = layer1, 5 = layer2, 6 = layer3, 7 = layer4
        top_layer = self.encoder[7] 
        
        for param in top_layer.parameters():
            param.requires_grad = True # Unfreeze
            
        print("ResNet18 'layer4' has been unfrozen for fine-tuning.")

