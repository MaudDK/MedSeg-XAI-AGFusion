import timm
import torch
import torch.nn as nn
from VisionTransformer import ViTSegmentationDecoder, DecoderBlock
from PretrainedModels import DecoderBlock as DecoderBlockConcat
import torch.nn.functional as F

class TimSegmentationModel(nn.Module):
    def __init__(self, 
                 img_size: int = 384,
                 num_classes: int = 1,
                 pretrained: bool = False,
                 model_name: str = None
    ):
        super(TimSegmentationModel, self).__init__()
        self.pretrained = pretrained
        if model_name is not None:
            self.encoder = timm.create_model(model_name, pretrained=pretrained, img_size=img_size)
            self.encoder.head = torch.nn.Identity()
            self.encoder.head_dist = torch.nn.Identity()
            embed_dim = self.encoder.num_features

        self.decoder = ViTSegmentationDecoder(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward_features(x)
        x = x[:, 1:, :]  # remove CLS token) 
        x = self.decoder(x)
        return x

class ViTSwinSegmentationDecoder(nn.Module):
    def __init__(self, embed_dim: int = 768, num_classes: int = 1):
        super(ViTSwinSegmentationDecoder, self).__init__()
        self.embed_dim = embed_dim

        self.projection = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.up1 = DecoderBlock(512, 256)
        self.up2 = DecoderBlock(256, 128)
        self.up3 = DecoderBlock(128, 64)
        self.up4 = DecoderBlock(64, 32)
        self.up5 = DecoderBlock(32, 32)

        self.prediction = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.projection(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.prediction(x)
        return x
  
class ViTSwinSegmentationModel(nn.Module):
    def __init__(self, 
                 img_size: int = 384, 
                 num_classes: int = 1,
                 pretrained: bool = False,
    ):
        super(ViTSwinSegmentationModel, self).__init__()
        self.encoder = timm.create_model(
                'swin_base_patch4_window12_384', 
                pretrained=pretrained,
                features_only=True,
                img_size=img_size,
                )
        self.decoder = ViTSwinSegmentationDecoder(embed_dim=1024, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1, enc2, enc3, enc4 = self.encoder(x)
        B, H, W, C = enc4.shape
        x = enc4.reshape(B, H*W, C)
        x = self.decoder(x)
        return x

class ViTSwinSkipSegmentationModel(nn.Module):
    def __init__(self, 
                 img_size: int = 384, 
                 num_classes: int = 1,
                 pretrained: bool = False,
    ):
        super(ViTSwinSkipSegmentationModel, self).__init__()
        self.encoder = timm.create_model(
                'swin_base_patch4_window12_384', 
                pretrained=pretrained,
                features_only=True,
                img_size=img_size,
            )
            
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up1 = DecoderBlockConcat(1024, 512, 512)
        self.up2 = DecoderBlockConcat(512, 256, 256)
        self.up3 = DecoderBlockConcat(256, 128, 128)
        self.up4 = DecoderBlockConcat(128, 64, 64)

        self.prediction = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_layer = self.input_layer(x)
        enc1, enc2, enc3, enc4 = self.encoder(x)
        # #Convert from B, H, W, C to B, C, H, W
        enc4 = enc4.transpose(1,3).transpose(2,3)
        enc3 = enc3.transpose(1,3).transpose(2,3)
        enc2 = enc2.transpose(1,3).transpose(2,3)
        enc1 = enc1.transpose(1,3).transpose(2,3)

        enc3 = self.up1(enc4, enc3)
        enc2 = self.up2(enc3, enc2)
        enc1 = self.up3(enc2, enc1)
        x = self.up4(enc1, input_layer)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.prediction(x)
        return x

if __name__ == "__main__":
    model = ViTSwinSkipSegmentationModel(num_classes=1, pretrained=True)
    sample_input = torch.randn(1, 3, 384, 384)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")

