import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from PretrainedModels import ResNetEncoder, ResNet34SkipDecoder, Res34UNet
from TimModels import ViTSwinSkipSegmentationModel

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

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        
        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Store for visualization
        self.attention_weights = psi.detach()
        fused = psi * x + (1 - psi) * g
        
        return fused

class DualEncoderFusionBlock(nn.Module):
    """Swin-guided attention fusion for dual encoders"""
    def __init__(self, in_channels, out_channels):
        super(DualEncoderFusionBlock, self).__init__()
        
        # Swin guides ResNet (Swin as gate, ResNet as input)
        self.attn_swin_to_res = AttentionBlock(
            F_g=in_channels,  
            F_l=in_channels,  
            F_int=out_channels
        )
        
    def forward(self, res_feat, swin_feat):
        # Swin guides what to attend in ResNet features
        res_attended = self.attn_swin_to_res(g=swin_feat, x=res_feat)
        return res_attended

class AttentionDualEncoderSwin(nn.Module):
    def __init__(self, out_channels=1, freeze_encoders=True):
        super(AttentionDualEncoderSwin, self).__init__()
        
        # Load pretrained models
        Res34Unet = Res34UNet(weights=None, out_channels=1)
        SwinBaseModel = ViTSwinSkipSegmentationModel(
            img_size=384,
            num_classes=1,
            pretrained=False,
        )

        checkpoint_path_res = (
            "checkpoints/Res34Unet/20260202/Pretrained/"
            "4ebb5a3a_epoch_198_metric_0.2078_Res34Unet.pth"
        )
        checkpoint_path_swin = (
            "checkpoints/Swin_Base_Skip/20260203/d699bbf1/"
            "d699bbf1_epoch_71_metric_0.2019_Swin_Base_Skip.pth"
        )

        state_dict_res = torch.load(checkpoint_path_res)
        state_dict_swin = torch.load(checkpoint_path_swin)

        Res34Unet.load_state_dict(state_dict_res)
        SwinBaseModel.load_state_dict(state_dict_swin)

        # Extract encoders
        self.ResEncoder = Res34Unet.encoder
        self.SwinEncoder = SwinBaseModel.encoder

        # Freeze encoders
        for param in self.ResEncoder.parameters():
            param.requires_grad = freeze_encoders
        
        for param in self.SwinEncoder.parameters():
            param.requires_grad = freeze_encoders

        # Swin projection layers (reduce channels to match ResNet)
        self.swin_proj1 = nn.Conv2d(1024, 512, kernel_size=1)  # Level 4: 1024→512
        self.swin_proj2 = nn.Conv2d(512, 256, kernel_size=1)   # Level 3: 512→256
        self.swin_proj3 = nn.Conv2d(256, 128, kernel_size=1)   # Level 2: 256→128
        # Level 1: 128 (no projection needed, already matches)

        # Attention-based fusion blocks
        self.fusion_attn4 = DualEncoderFusionBlock(512, 512)  # 12×12
        self.fusion_attn3 = DualEncoderFusionBlock(256, 256)  # 24×24
        self.fusion_attn2 = DualEncoderFusionBlock(128, 128)  # 48×48
        self.fusion_attn1 = DualEncoderFusionBlock(128, 128)  # 96×96

        # Decoder blocks (in_channels, skip_channels, out_channels)
        self.up1 = DecoderBlock(512, 256, 256)  # 12×12 → 24×24
        self.up2 = DecoderBlock(256, 128, 128)   # 24×24 → 48×48
        self.up3 = DecoderBlock(128, 128, 128)   # 48×48 → 96×96
        
        # Final upsampling to original resolution (96×96 → 384×384)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 96→192
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 192→384
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final 1×1 convolution to output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def get_attention_weights(self):
        """Return attention weights from all fusion blocks for visualization"""
        return {
            "fusion_attn4": self.fusion_attn4.attn_swin_to_res.attention_weights,
            "fusion_attn3": self.fusion_attn3.attn_swin_to_res.attention_weights,
            "fusion_attn2": self.fusion_attn2.attn_swin_to_res.attention_weights,
            "fusion_attn1": self.fusion_attn1.attn_swin_to_res.attention_weights,
        }
    
    def extract_visual_tokens(self, x):
        input_layer, res_enc1, res_enc2, res_enc3, res_enc4 = self.ResEncoder(x)
        res_enc1 = torch.cat([input_layer, res_enc1], dim=1)  # Combine input with first encoder output
        swin_enc1, swin_enc2, swin_enc3, swin_enc4 = self.SwinEncoder(x)

        swin_enc1 = swin_enc1.permute(0,3,1,2).contiguous()
        swin_enc2 = self.swin_proj3(swin_enc2.permute(0,3,1,2).contiguous())
        swin_enc3 = self.swin_proj2(swin_enc3.permute(0,3,1,2).contiguous())
        swin_enc4 = self.swin_proj1(swin_enc4.permute(0,3,1,2).contiguous())

        concat_4 = self.fusion_attn4(res_enc4, swin_enc4)
        return concat_4
    
    def forward(self, x):
        # Extract features from ResNet encoder
        input_layer, res_enc1, res_enc2, res_enc3, res_enc4 = self.ResEncoder(x)
        res_enc1 = torch.cat([input_layer, res_enc1], dim=1)  # Combine input with first encoder output
        
        # Extract features from Swin encoder
        swin_enc1, swin_enc2, swin_enc3, swin_enc4 = self.SwinEncoder(x)

        # Permute Swin features from (B, H, W, C) to (B, C, H, W) and project
        swin_enc1 = swin_enc1.permute(0, 3, 1, 2).contiguous()  # 128 channels (no projection)
        
        swin_enc2 = swin_enc2.permute(0, 3, 1, 2).contiguous()  # 256 channels
        swin_enc2 = self.swin_proj3(swin_enc2)                  # 256 → 128
        
        swin_enc3 = swin_enc3.permute(0, 3, 1, 2).contiguous()  # 512 channels
        swin_enc3 = self.swin_proj2(swin_enc3)                  # 512 → 256
        
        swin_enc4 = swin_enc4.permute(0, 3, 1, 2).contiguous()  # 1024 channels
        swin_enc4 = self.swin_proj1(swin_enc4)                  # 1024 → 512
        
        # Fuse encoder features with attention blocks
        concat_4 = self.fusion_attn4(res_enc4, swin_enc4)  # 512+512 = 1024 @ 12×12
        concat_3 = self.fusion_attn3(res_enc3, swin_enc3)  # 256+256 = 512  @ 24×24
        concat_2 = self.fusion_attn2(res_enc2, swin_enc2)  # 128+128 = 256  @ 48×48
        concat_1 = self.fusion_attn1(res_enc1, swin_enc1)  # 128+128 = 256  @ 96×96

        # Decoder pathway
        up1_out = self.up1(concat_4, concat_3)  # 1024 @ 12×12 → 512 @ 24×24
        up2_out = self.up2(up1_out, concat_2)   # 512  @ 24×24 → 256 @ 48×48
        up3_out = self.up3(up2_out, concat_1)   # 256  @ 48×48 → 128 @ 96×96
        up4_out = self.up4(up3_out)             # 128  @ 96×96 → 64  @ 384×384
        
        # Final segmentation output
        output = self.final_conv(up4_out)       # 64 @ 384×384 → 1 @ 384×384
        
        return output
    
if __name__ == "__main__":
    model = AttentionDualEncoderSwin(out_channels=1)
    input_tensor = torch.randn(1, 3, 384, 384)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")