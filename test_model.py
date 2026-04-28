import torch
from config import Config
from model import build_model, FullRFDMUNet

def test_full_rfdm():
    # Force ResNet backbone for consistency in this test or let it use config default if set
    Config.ENCODER_NAME = "resnet34" 
    
    print("Testing FullRFDMUNet forward pass...")
    model = build_model(use_full_rfdm=True)
    
    H, W = 256, 256
    x = torch.randn(2, 3, H, W)
    masks, aux = model(x, return_aux=True)
    
    print("Masks shape:", masks.shape)
    assert masks.shape == (2, Config.CLASSES, H, W), "Mask shape mismatch"
    
    # 512 is the expected bottleneck for resnet34 out_channels[-1]
    bottleneck_channels = model.base_model.encoder.out_channels[-1]
    
    print("F_reflect shape:", aux["F_reflect"].shape)
    print("F_illum shape:", aux["F_illum"].shape)
    
    assert aux["F_reflect"].shape == (2, bottleneck_channels, H // 32, W // 32)
    assert aux["F_illum"].shape == (2, bottleneck_channels, H // 32, W // 32)
    
    print("Uncertainty map shape:", aux["uncertainty"].shape)
    assert aux["uncertainty"].shape == (2, 1, H // 32, W // 32)
    
    print("Test passed! All shapes are correct.")

if __name__ == "__main__":
    test_full_rfdm()
