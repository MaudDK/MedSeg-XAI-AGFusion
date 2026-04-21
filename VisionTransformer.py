import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


CONFIGS = {
    'ViT_Tiny': {
        'img_size': 384,
        'patch_size': 16,
        'in_channels': 3,
        'embed_dim': 192,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'depth': 12,
        'dropout': 0.1,
    },
    'ViT_Small': {
        'img_size': 384,
        'patch_size': 16,
        'in_channels': 3,
        'embed_dim': 384,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'depth': 12,
        'dropout': 0.1,
    },
    'ViT_Base': {
        'img_size': 384,
        'patch_size': 16,
        'in_channels': 3,
        'embed_dim': 768,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'depth': 12,
        'dropout': 0.1,
    },
    'ViT_Large': {
        'img_size': 384,
        'patch_size': 16,
        'in_channels': 3,
        'embed_dim': 1024,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'depth': 24,
        'dropout': 0.1,
    },
    'ViT_Huge': {
        'img_size': 384,
        'patch_size': 14,
        'in_channels': 3,
        'embed_dim': 1280,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'depth': 32,
        'dropout': 0.1,
    }
}

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
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
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) 
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, num_patches: int):
        super(PositionalEncoding, self).__init__()
        self.grid_size = int(num_patches ** 0.5)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size, self.grid_size, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = self.grid_size
        x = x.reshape(B, H, W, C) + self.pos_embed
        x = x.reshape(B, N, C)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size: int = 384, 
                 patch_size: int = 16, 
                 in_channels: int = 3, 
                 embed_dim: int = 768, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0, 
                 depth: int = 12, 
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 norm_first: bool = True
    ):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        num_patches = self.patch_embed.num_patches
        self.pos_embed = PositionalEncoding(embed_dim, num_patches)

        dim_feedforward = int(embed_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=batch_first,
            norm_first=norm_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.transformer(x)
        return x

class ViTSegmentationDecoder(nn.Module):
    def __init__(self, embed_dim: int = 768, patch_size: int = 16, num_classes: int = 1):
        super(ViTSegmentationDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

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
        x = self.prediction(x)
        return x

class ViTSegmentationModel(nn.Module):
    def __init__(self, 
                 img_size: int = 384, 
                 patch_size: int = 16, 
                 in_channels: int = 3, 
                 embed_dim: int = 768, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0, 
                 depth: int = 12, 
                 dropout: float = 0.0,
                 num_classes: int = 1,
                 batch_first: bool = True,
                 norm_first: bool = True,
                 pretrained: bool = False,
                 model_name: str = None
    ):
        super(ViTSegmentationModel, self).__init__()
        self.pretrained = pretrained
        if pretrained and not model_name:
            raise ValueError("Model name must be provided when using pretrained weights.")
        if pretrained:
            self.encoder = timm.create_model('deit_base_patch16_384', pretrained=True)
            self.encoder.head = torch.nn.Identity()
            self.encoder.head_dist = torch.nn.Identity()  # important for DeiT
        else:
            self.encoder = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                dropout=dropout,
                batch_first=batch_first,
                norm_first=norm_first
            )
        self.decoder = ViTSegmentationDecoder(embed_dim=embed_dim, patch_size=patch_size, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = self.encoder.forward_features(x)
            x = x[:, 1:, :]  # remove CLS token
        else:
            x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    vit_model = ViTSegmentationModel(num_classes=1, pretrained=True,**CONFIGS["ViT_Base"])
    dummy_input = torch.randn(1, 3, 384, 384)
    vit_output = vit_model(dummy_input)
    print("ViTSegmentationModel Output Shape:", vit_output.shape)
