import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class UNet(nn.Module):
    """
    U-Net with DenseNet as the feature extractor
    """

    def __init__(self, in_channels=1, out_channels=1, pretrained=True):
        super().__init__()

        # Load DenseNet as encoder
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Replace the first convolutional layer to accept the correct input channels
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Extract DenseNet's layers
        self.encoder1 = nn.Sequential(*list(self.densenet.features.children())[:6])  # Up to first dense block
        self.encoder2 = nn.Sequential(*list(self.densenet.features.children())[6:8])  # First dense block
        self.encoder3 = nn.Sequential(*list(self.densenet.features.children())[8:10])  # Second dense block
        self.encoder4 = nn.Sequential(*list(self.densenet.features.children())[10:])  # Third dense block

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),  # Simple 1x1 convolution
            nn.ReLU(inplace=True)  # Activation after the bottleneck
        )

        # Decoder
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Block consisting of two convolutional layers and ReLU activation.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        """
        Upsampling block with ReLU activation.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the U-Net model.
        """
        # Encoder
        enc1 = self.encoder1(x)  # Output shape: (N, 64, H/2, W/2)
        enc2 = self.encoder2(enc1)  # Output shape: (N, 128, H/4, W/4)
        enc3 = self.encoder3(enc2)  # Output shape: (N, 256, H/8, W/8)
        enc4 = self.encoder4(enc3)  # Output shape: (N, 1024, H/16, W/16)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # Apply bottleneck here

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final Convolution
        return self.final_conv(dec1)
