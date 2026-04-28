import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from config import Config

class RadianceGuidedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Attention map = Sigmoid(Conv(Concat(F, M)))
        self.conv = nn.Conv2d(channels + 1, 1, kernel_size=3, padding=1)
        
    def forward(self, x, darkness_map):
        # interpolate darkness map to feature size
        m = F.interpolate(darkness_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([x, m], dim=1)
        attn = torch.sigmoid(self.conv(combined))
        return x * (1.0 + attn)

class IllumiSegUNet(nn.Module):
    def __init__(self, use_rgsa=True):
        super().__init__()
        self.base_model = smp.Unet(
            encoder_name=Config.ENCODER_NAME,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.IN_CHANNELS,
            classes=Config.CLASSES,
        )
        self.use_rgsa = use_rgsa
        # Bottleneck channels for resnet34/resnet50 is typically 512
        self.rgsa = RadianceGuidedAttention(channels=512)
        
    def forward(self, x, darkness_map=None, return_features=False):
        # Extract multiscale features
        features = self.base_model.encoder(x)
        
        # Save pre-attention bottleneck for Feature Consistency
        cic_features = features[-1]
        
        if self.use_rgsa and darkness_map is not None:
            # Reverted to v1: Bottleneck only RGSA
            features[-1] = self.rgsa(features[-1], darkness_map)
            
        decoder_output = self.base_model.decoder(features)
        masks = self.base_model.segmentation_head(decoder_output)
        
        if return_features:
            return masks, cic_features
            
        return masks

class RetinexDisentanglementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.reflectance_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # Deepwise representation for illumination scale
        self.illumination_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        shared_feat = self.shared_conv(x)
        f_reflect = self.reflectance_branch(shared_feat)
        f_illum = self.illumination_branch(shared_feat)
        return f_reflect, f_illum

class UncertaintyHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.head(x)

class EnhancementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class EnhancementDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 5 upsamples progressively halving channels (with minimum of 16 to preserve capacity)
        ch = in_channels
        ch1 = max(ch // 2, 16)
        ch2 = max(ch1 // 2, 16)
        ch3 = max(ch2 // 2, 16)
        ch4 = max(ch3 // 2, 16)
        ch5 = max(ch4 // 2, 16)
        
        self.blocks = nn.Sequential(
            EnhancementBlock(ch, ch1),
            EnhancementBlock(ch1, ch2),
            EnhancementBlock(ch2, ch3),
            EnhancementBlock(ch3, ch4),
            EnhancementBlock(ch4, ch5)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(ch5, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.final_conv(x)

class FullRFDMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = smp.Unet(
            encoder_name=Config.ENCODER_NAME,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.IN_CHANNELS,
            classes=Config.CLASSES,
        )
        encoder_channels = self.base_model.encoder.out_channels[-1]
        
        self.retinex = RetinexDisentanglementBlock(encoder_channels)
        self.uncertainty_head = UncertaintyHead(encoder_channels)
        self.enhancement_decoder = EnhancementDecoder(encoder_channels)
        
    def forward(self, x, return_aux=False):
        # Extract encoder features
        features = self.base_model.encoder(x)
        
        # Take bottleneck feature
        bottleneck = features[-1]
        
        # Disentangle
        f_reflect, f_illum = self.retinex(bottleneck)
        
        # Uncertainty map
        uncertainty_map = self.uncertainty_head(f_illum)
        
        # Replace bottleneck with reflectance feature
        # Since features is a list/tuple, rebuild it properly
        if isinstance(features, tuple):
            features_list = list(features)
            features_list[-1] = f_reflect
            features = tuple(features_list)
        else:
            features[-1] = f_reflect
            
        # Segment using decoder
        decoder_output = self.base_model.decoder(features)
        masks = self.base_model.segmentation_head(decoder_output)
        
        if return_aux:
            recon_image = self.enhancement_decoder(f_illum)
            return masks, {
                "F_reflect": f_reflect,
                "F_illum": f_illum,
                "uncertainty": uncertainty_map,
                "recon": recon_image
            }
        
        return masks


def build_model(use_illumiseg=False, use_full_rfdm=False):
    """
    Builds the baseline segmentation model, the advanced IllumiSeg variant, or Full RFDM.
    """
    if use_full_rfdm:
        return FullRFDMUNet()
    elif use_illumiseg:
        return IllumiSegUNet(use_rgsa=Config.USE_RGSA)
    else:
        return smp.Unet(
            encoder_name=Config.ENCODER_NAME,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.IN_CHANNELS,
            classes=Config.CLASSES,
        )
