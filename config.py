import os
import torch

class Config:
    # Environment
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "Kvasir-SEG")
    IMG_DIR = os.path.join(DATA_DIR, "images")
    MASK_DIR = os.path.join(DATA_DIR, "masks")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    
    # Model & Train params
    MODEL_NAME = "Unet"
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    
    IMG_SIZE = (256, 256)  # format: (H, W)
    IN_CHANNELS = 3
    CLASSES = 1
    
    BATCH_SIZE = 4
    ACCUM_STEPS = 4
    NUM_WORKERS = 4  # adjust based on CPU cores
    LOG_WANDB = False
    WANDB_PROJECT = "IllumiSeg"
    
    EPOCHS = 10
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-5
    
    # Phase 3 Enhancements
    USE_RGSA = True
    USE_CIC = True
    CIC_WEIGHT = 0.5
    
    # Phase 7 Upgrade
    INIT_WEIGHTS = "current_best_full_rfdm.pth"
    USE_FULL_RFDM = True
    TV_WEIGHT = 0.01
    RECON_WEIGHT = 0.20
    
    # Early stopping
    PATIENCE = 10
    
    # Evaluation
    SEVERITY_LEVELS = ["clean", "mild", "medium", "severe"]
