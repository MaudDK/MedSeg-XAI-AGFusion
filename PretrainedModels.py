import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet34
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import vit_b_16
from torchvision.models import ResNet34_Weights, ViT_B_16_Weights

## Utilities ##
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class DecoderNoSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderNoSkipBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.double_conv(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, mode=None):
        super(Upsample, self).__init__()
        if mode:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.upsample(x)
        return x

## Encoders ##
class ResNetEncoder(nn.Module):
    def __init__(self, resnet, weights):
        super(ResNetEncoder, self).__init__()
        resnet = resnet(weights=weights)
        
        self.input_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        input_layer = self.input_layer(x)
        enc1 = self.layer1(input_layer)
        enc2 = self.layer2(enc1)
        enc3 = self.layer3(enc2)
        enc4 = self.layer4(enc3)
        return input_layer, enc1, enc2, enc3, enc4

class DeepLabV3ResNet50Encoder(nn.Module):
    def __init__(self, weights):
        super(DeepLabV3ResNet50Encoder, self).__init__()
        self.model = deeplabv3_resnet50(weights=weights)

        self.input_layer = nn.Sequential(
            self.model.backbone.conv1,
            self.model.backbone.bn1,
            self.model.backbone.relu,
            self.model.backbone.maxpool
        )

        self.layer1 = self.model.backbone.layer1
        self.layer2 = self.model.backbone.layer2
        self.layer3 = self.model.backbone.layer3
        self.layer4 = self.model.backbone.layer4

    def forward(self, x):
        input_layer = self.input_layer(x)
        enc1 = self.layer1(input_layer)
        enc2 = self.layer2(enc1)
        enc3 = self.layer3(enc2)
        enc4 = self.layer4(enc3)
        return input_layer, enc1, enc2, enc3, enc4

## Decoders ##
class NoSkipDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super(NoSkipDecoder, self).__init__()
        self.up1 = DecoderNoSkipBlock(in_channels, in_channels // 2)
        self.up2 = DecoderNoSkipBlock(in_channels // 2, in_channels // 4)
        self.up3 = DecoderNoSkipBlock(in_channels // 4, in_channels // 8)
        self.up4 = DecoderNoSkipBlock(in_channels // 8, in_channels // 16)
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x
    
#Skip Decoders
class ResNet34SkipDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super(ResNet34SkipDecoder, self).__init__()
        self.up1 = DecoderBlock(in_channels, 256, in_channels // 2)
        self.up2 = DecoderBlock(in_channels // 2, 128, in_channels // 4)
        self.up3 = DecoderBlock(in_channels // 4, 64, in_channels // 8)
        self.up4 = DecoderBlock(in_channels // 8, 64, in_channels // 16)
        
    def forward(self, input_layer, enc1, enc2, enc3, enc4):
        up1 = self.up1(enc4, enc3)
        up2 = self.up2(up1, enc2)
        up3 = self.up3(up2, enc1)
        up4 = self.up4(up3, input_layer)
        return up4

class DeepLabV3SkipDecoder(nn.Module):
    def __init__(self, in_channels=2048):
        super(DeepLabV3SkipDecoder, self).__init__()
        self.up1 = DecoderBlock(in_channels, 1024, in_channels // 2)
        self.up2 = DecoderBlock(in_channels // 2, 512, in_channels // 4)
        self.up3 = DecoderBlock(in_channels // 4, 256, in_channels // 8)
        self.up4 = DecoderBlock(in_channels // 8, 64, in_channels // 16)
        
    def forward(self, input_layer, enc1, enc2, enc3, enc4):
        up1 = self.up1(enc4, enc3)
        up2 = self.up2(up1, enc2)
        up3 = self.up3(up2, enc1)
        up4 = self.up4(up3, input_layer)
        return up4
    
## Models ##
class Res34UNet(nn.Module):
    def __init__(self, weights, out_channels=1):
        super(Res34UNet, self).__init__()
        self.encoder = ResNetEncoder(resnet=resnet34, weights=weights)
        self.decoder = ResNet34SkipDecoder(in_channels=512)

        self.prediction = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        input_layer, enc1, enc2, enc3, enc4 = self.encoder(x)
        x = self.decoder(input_layer, enc1, enc2, enc3, enc4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.prediction(x)
        return x

class Res34UNetNoSkip(nn.Module):
    def __init__(self, weights, out_channels=1):
        super(Res34UNetNoSkip, self).__init__()
        self.encoder = ResNetEncoder(resnet=resnet34, weights=weights)
        self.decoder = NoSkipDecoder(in_channels=512)

        self.prediction = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        _, _, _, _, enc4 = self.encoder(x)
        x = self.decoder(enc4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.prediction(x)
        return x
    
class DeepLabV3Res50UNet(nn.Module):
    def __init__(self, weights, out_channels=1):
        super(DeepLabV3Res50UNet, self).__init__()
        self.encoder = DeepLabV3ResNet50Encoder(weights=weights)
        self.decoder = DeepLabV3SkipDecoder(in_channels=2048)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.prediction = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        input_layer, enc1, enc2, enc3, enc4 = self.encoder(x)
        x = self.decoder(input_layer, enc1, enc2, enc3, enc4)
        x = self.project(x)
        x = self.prediction(x)
        return x
    
class DeepLabV3Res50UNetNoSkip(nn.Module):
    def __init__(self, weights, out_channels=1):
        super(DeepLabV3Res50UNetNoSkip, self).__init__()
        self.encoder = DeepLabV3ResNet50Encoder(weights=weights)
        self.decoder = NoSkipDecoder(in_channels=2048)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.prediction = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        _, _, _, _, enc4 = self.encoder(x)
        x = self.decoder(enc4)
        x = self.project(x)
        x = self.prediction(x)
        return x
    
if __name__ == "__main__":
    res34unet = Res34UNetNoSkip(weights=None, out_channels=1)
    print(res34unet)
    # deeplab = DeepLabV3Res50UNetNoSkip(weights=None, out_channels=1)

    # dummy_input = torch.randn(1, 3, 384, 384)
    # res34unet_output = res34unet(dummy_input)
    # print("Res34UNet Output Shape:", res34unet_output.shape)
    
    
    # deeplab_output = deeplab(dummy_input)
    # # print("DeepLabV3Res50UNet Output Shape:", deeplab_output.shape)

