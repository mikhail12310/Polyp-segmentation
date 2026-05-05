import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import KvasirDataset, get_train_val_test_splits
from model import build_model
from augmentations import get_training_augmentation, get_validation_augmentation
from losses import UncertaintyWeightedLoss, TVLoss, ReconstructionLoss
from utils import (
    set_seed, calculate_metrics, MetricTracker, 
    plot_multi_curve
)

def train_epoch(model, loader, optimizer, criterion, scaler, device, tv_criterion=None, recon_criterion=None):
    model.train()
    tracker = MetricTracker()
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for step, (images_clean, images_dark, masks, darkness_maps) in enumerate(pbar):
        images_clean = images_clean.to(device)
        images_dark = images_dark.to(device)
        masks = masks.to(device)
        
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
            preds, aux = model(images_dark, return_aux=True)
            
            # Interpolate uncertainty map to match mask size
            uncertainty_map = F.interpolate(aux["uncertainty"], size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            # Loss returns (combined_seg_loss, bce, dice)
            loss_seg, _, _ = criterion(preds, masks, uncertainty_map)
            
            combined_loss = loss_seg
            if tv_criterion:
                combined_loss += Config.TV_WEIGHT * tv_criterion(aux["F_illum"])
            if recon_criterion:
                combined_loss += Config.RECON_WEIGHT * recon_criterion(aux["recon"], images_clean)
                
            loss_accum = combined_loss / Config.ACCUM_STEPS

        scaler.scale(loss_accum).backward()
        
        if (step + 1) % Config.ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        with torch.no_grad():
            metrics = calculate_metrics(torch.sigmoid(preds), masks, threshold=Config.THRESHOLD)
        tracker.update(combined_loss.item(), metrics, images_dark.size(0))
        pbar.set_postfix({"Loss": f"{combined_loss.item():.4f}", "Dice": f"{metrics['dice']:.4f}"})
        
    return tracker.get_avg()

def val_epoch(model, loader, device):
    model.eval()
    tracker = MetricTracker()
    criterion = nn.BCEWithLogitsLoss() # Simple val criterion
    
    with torch.no_grad():
        for _, images_dark, masks, _ in loader:
            images_dark = images_dark.to(device)
            masks = masks.to(device)
            preds = model(images_dark)
            loss = criterion(preds, masks)
            metrics = calculate_metrics(torch.sigmoid(preds), masks, threshold=Config.THRESHOLD)
            tracker.update(loss.item(), metrics, images_dark.size(0))
            
    return tracker.get_avg()

def evaluate_models(models, dataloader, device):
    for m in models.values(): m.eval()
    trackers = {k: MetricTracker() for k in models.keys()}
    
    with torch.no_grad():
        for _, images_dark, masks, _ in dataloader:
            images_dark = images_dark.to(device)
            masks = masks.to(device)
            
            for name, model in models.items():
                p = torch.sigmoid(model(images_dark))
                m = calculate_metrics(p, masks, threshold=Config.THRESHOLD)
                trackers[name].update(0, m, images_dark.size(0))
                
    return {k: v.get_avg() for k, v in trackers.items()}

def main():
    set_seed(Config.SEED)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    splits = get_train_val_test_splits(Config.IMG_DIR, Config.MASK_DIR)
    
    # 1. Training Phase (P4)
    print("\n--- Phase 1: Training P4 (Bootstrapped from P2 Dark-Trained) ---")
    model_p4 = build_model(use_full_rfdm=True).to(Config.DEVICE)
    
    p2_path = os.path.join(Config.CHECKPOINT_DIR, "best_dark_model.pth")
    if os.path.exists(p2_path):
        state_dict = torch.load(p2_path, map_location=Config.DEVICE)
        mapped_dict = {f"base_model.{k}": v for k, v in state_dict.items() if not k.startswith("base_model.")}
        if not mapped_dict: mapped_dict = state_dict
        model_p4.load_state_dict(mapped_dict, strict=False)
        print(f"Loaded weights from {p2_path}")
    
    train_loader = DataLoader(KvasirDataset(splits["train"][0], splits["train"][1], get_training_augmentation(Config.IMG_SIZE), "random_mix"), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(KvasirDataset(splits["val"][0], splits["val"][1], get_validation_augmentation(Config.IMG_SIZE), "clean"), batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    criterion = UncertaintyWeightedLoss()
    tv_criterion = TVLoss()
    recon_criterion = ReconstructionLoss()
    optimizer = optim.AdamW(model_p4.parameters(), lr=Config.LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    best_dice = 0.0
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"Epoch {epoch}/{Config.EPOCHS}")
        train_epoch(model_p4, train_loader, optimizer, criterion, scaler, Config.DEVICE, tv_criterion, recon_criterion)
        val_stats = val_epoch(model_p4, val_loader, Config.DEVICE)
        print(f"Val Dice: {val_stats['dice']:.4f}")
        if val_stats['dice'] > best_dice:
            best_dice = val_stats['dice']
            torch.save(model_p4.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_illumiseg_model.pth"))
            print("--> Saved P4 Model (Final)")

    # 2. Evaluation Phase (P1, P2, P3, P4)
    print("\n--- Phase 2: Comparative Evaluation (P1, P2, P3, P4) ---")
    
    models = {}
    # P1: Baseline
    models["P1 Baseline"] = build_model(use_illumiseg=False).to(Config.DEVICE)
    models["P1 Baseline"].load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), map_location=Config.DEVICE))
    
    # P2: Dark-Trained
    models["P2 Dark-Trained"] = build_model(use_illumiseg=False).to(Config.DEVICE)
    models["P2 Dark-Trained"].load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_dark_model.pth"), map_location=Config.DEVICE))
    
    # P3: Original RFDM
    models["P3 Original RFDM"] = build_model(use_full_rfdm=True).to(Config.DEVICE)
    models["P3 Original RFDM"].load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "current_best_full_rfdm.pth"), map_location=Config.DEVICE), strict=False)
    
    # P4: Final RFDM (Last Trained)
    models["P4 Final RFDM"] = build_model(use_full_rfdm=True).to(Config.DEVICE)
    models["P4 Final RFDM"].load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_illumiseg_model.pth"), map_location=Config.DEVICE))
    
    results = []
    plot_data = {name: [] for name in models.keys()}
    
    for severity in Config.SEVERITY_LEVELS:
        loader = DataLoader(KvasirDataset(splits["test"][0], splits["test"][1], get_validation_augmentation(Config.IMG_SIZE), severity), batch_size=Config.BATCH_SIZE, shuffle=False)
        metrics = evaluate_models(models, loader, Config.DEVICE)
        
        row = {"Severity": severity}
        for name, m in metrics.items():
            row[name] = m["dice"]
            plot_data[name].append(m["dice"])
        results.append(row)
        
    df = pd.DataFrame(results)
    print("\n====== FINAL COMPARISON ======")
    print(df.to_string(index=False))
    print("==============================\n")
    
    plot_multi_curve(Config.SEVERITY_LEVELS, plot_data, "Final Robustness Comparison", "Severity", "Dice", os.path.join(Config.OUTPUT_DIR, "p1_p2_p3_p4_comparison.png"))

if __name__ == "__main__":
    main()
