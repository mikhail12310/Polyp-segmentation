import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import KvasirDataset, get_train_val_test_splits
from augmentations import get_validation_augmentation
from model import build_model
from utils import set_seed, calculate_metrics, MetricTracker

def evaluate_threshold_sweep(model, dataloader, device, severity, threshold):
    tracker = MetricTracker()
    
    with torch.no_grad():
        for images_clean, images_dark, masks, darkness_maps in dataloader:
            images_dark = images_dark.to(device)
            masks = masks.to(device)
            
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                preds = model(images_dark)
                
            preds_prob = torch.sigmoid(preds)
            
            metrics = calculate_metrics(preds_prob, masks, threshold=threshold)
            tracker.update(0, metrics, images_dark.size(0))
            
    return tracker.get_avg()

def main():
    set_seed(Config.SEED)
    
    model_path = os.path.join(Config.CHECKPOINT_DIR, "current_best_full_rfdm.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
        
    model = build_model(use_full_rfdm=True).to(Config.DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    except Exception:
        state_dict = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    print("Loaded current_best_full_rfdm successfully.\n")
    
    splits = get_train_val_test_splits(Config.IMG_DIR, Config.MASK_DIR)
    test_img_paths, test_mask_paths = splits["test"]
    
    thresholds = [0.45]
    severities = ["clean", "mild", "medium", "severe"]
    
    for thresh in thresholds:
        print(f"Threshold: {thresh:.2f}")
        print(f"{'Severity':<10} {'Dice':<8} {'Precision':<11} {'Recall':<9} {'IoU':<8}")
        
        for sev in severities:
            test_dataset = KvasirDataset(
                test_img_paths, test_mask_paths, 
                augmentations=get_validation_augmentation(Config.IMG_SIZE), 
                severity=sev
            )
            loader = DataLoader(
                test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                num_workers=Config.NUM_WORKERS, pin_memory=True
            )
            
            avg_metrics = evaluate_threshold_sweep(model, loader, Config.DEVICE, sev, thresh)
            print(f"{sev:<10} {avg_metrics['dice']:.3f}    {avg_metrics['precision']:.3f}       {avg_metrics['recall']:.3f}     {avg_metrics['iou']:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
