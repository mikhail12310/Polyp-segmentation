import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from config import Config
from dataset import KvasirDataset, get_train_val_test_splits
from augmentations import get_training_augmentation, get_validation_augmentation
from model import build_model
from losses import DarknessWeightedLoss, TVLoss, UncertaintyWeightedLoss, ReconstructionLoss
from utils import set_seed, calculate_metrics, MetricTracker
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, criterion, scaler, device, tv_criterion=None, recon_criterion=None):
    model.train()
    tracker = MetricTracker()
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for step, (images_clean, images_dark, masks, darkness_maps) in enumerate(pbar):
        images_clean = images_clean.to(device)
        images_dark = images_dark.to(device)
        masks = masks.to(device)
        darkness_maps = darkness_maps.to(device)
        
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
            
            if hasattr(Config, "USE_FULL_RFDM") and Config.USE_FULL_RFDM:
                preds_dark, aux = model(images_dark, return_aux=True)
                
                # Resize uncertainty map to match mask shape
                uncertainty_map = F.interpolate(aux["uncertainty"], size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # Losses
                loss_seg, bce_dark, dice_dark = criterion(preds_dark, masks, uncertainty_map)
                tv_loss_val = tv_criterion(aux["F_illum"])
                
                loss_recon = recon_criterion(aux["recon"], images_clean)
                
                combined_loss = loss_seg + Config.TV_WEIGHT * tv_loss_val + Config.RECON_WEIGHT * loss_recon
                
                cic_loss = torch.tensor(0.0).to(device)
            else:
                # Forward Clean (Anchor)
                if Config.USE_CIC:
                    preds_clean, features_clean = model(images_clean, darkness_map=torch.zeros_like(darkness_maps), return_features=True)
                else:
                    preds_clean = model(images_clean, darkness_map=torch.zeros_like(darkness_maps))
                
                # Forward Dark (Target)
                if Config.USE_CIC:
                    preds_dark, features_dark = model(images_dark, darkness_map=darkness_maps, return_features=True)
                    # Siamese MSE Consistency: v1 style
                    cic_loss = F.mse_loss(features_clean, features_dark)
                else:
                    preds_dark = model(images_dark, darkness_map=darkness_maps)
                    cic_loss = torch.tensor(0.0).to(device)
                    
                # Losses
                loss_clean, bce_clean, dice_clean = criterion(preds_clean, masks, torch.zeros_like(darkness_maps))
                loss_dark, bce_dark, dice_dark = criterion(preds_dark, masks, darkness_maps)
                
                combined_loss = loss_clean + loss_dark + Config.CIC_WEIGHT * cic_loss
                tv_loss_val = torch.tensor(0.0).to(device)
                
            loss_accum = combined_loss / Config.ACCUM_STEPS
            
        scaler.scale(loss_accum).backward()
        
        if (step + 1) % Config.ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        with torch.no_grad():
            preds_prob = torch.sigmoid(preds_dark)
            metrics = calculate_metrics(preds_prob, masks)
            if hasattr(Config, "USE_FULL_RFDM") and Config.USE_FULL_RFDM:
                metrics["bce_loss"] = bce_dark.item()
                metrics["dice_loss"] = dice_dark.item()
                metrics["recon_loss"] = loss_recon.item()
            else:
                metrics["bce_loss"] = (bce_clean + bce_dark).item() / 2.0
                metrics["dice_loss"] = (dice_clean + dice_dark).item() / 2.0
            metrics["cic_loss"] = cic_loss.item()
            
        tracker.update(combined_loss.item(), metrics, images_dark.size(0))
        pbar.set_postfix({"Loss": f"{tracker.get_avg()['loss']:.4f}", "CIC": f"{tracker.get_avg()['cic_loss']:.4f}", "Dice": f"{tracker.get_avg()['dice']:.4f}"})
        
    return tracker.get_avg()

def val_epoch(model, loader, criterion, device):
    model.eval()
    tracker = MetricTracker()
    
    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images_clean, images_dark, masks, darkness_maps in pbar:
            images_dark = images_dark.to(device)
            masks = masks.to(device)
            darkness_maps = darkness_maps.to(device)
            
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                if hasattr(Config, "USE_FULL_RFDM") and Config.USE_FULL_RFDM:
                    preds, aux = model(images_dark, return_aux=True)
                    uncertainty_map = F.interpolate(aux["uncertainty"], size=masks.shape[2:], mode='bilinear', align_corners=False)
                    loss, bce_loss, dice_loss = criterion(preds, masks, uncertainty_map)
                else:
                    preds = model(images_dark, darkness_map=darkness_maps)
                    loss, bce_loss, dice_loss = criterion(preds, masks, darkness_maps)
                
            preds_prob = torch.sigmoid(preds)
            metrics = calculate_metrics(preds_prob, masks)
            metrics["bce_loss"] = bce_loss.item()
            metrics["dice_loss"] = dice_loss.item()
            metrics["cic_loss"] = 0.0
            
            tracker.update(loss.item(), metrics, images_dark.size(0))
            pbar.set_postfix({"Loss": f"{tracker.get_avg()['loss']:.4f}", "Dice": f"{tracker.get_avg()['dice']:.4f}"})
            
    return tracker.get_avg()

def main():
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    if Config.LOG_WANDB:
        wandb.init(project=Config.WANDB_PROJECT, config={
            "epochs": Config.EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "model": Config.MODEL_NAME,
            "encoder": Config.ENCODER_NAME
        })
        
    splits = get_train_val_test_splits(Config.IMG_DIR, Config.MASK_DIR)
    
    train_dataset = KvasirDataset(
        splits["train"][0], splits["train"][1], 
        augmentations=get_training_augmentation(Config.IMG_SIZE), severity="random_mix"
    )
    val_dataset = KvasirDataset(
        splits["val"][0], splits["val"][1], 
        augmentations=get_validation_augmentation(Config.IMG_SIZE), severity="clean"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    
    model = build_model(use_illumiseg=True, use_full_rfdm=getattr(Config, "USE_FULL_RFDM", False)).to(Config.DEVICE)
    
    if hasattr(Config, "INIT_WEIGHTS") and Config.INIT_WEIGHTS:
        init_path = os.path.join(Config.CHECKPOINT_DIR, Config.INIT_WEIGHTS)
        if os.path.exists(init_path):
            try:
                state_dict = torch.load(init_path, map_location=Config.DEVICE)
                mapped_dict = {f"base_model.{k}": v for k, v in state_dict.items() if not k.startswith("base_model.")}
                if len(mapped_dict) == 0: mapped_dict = state_dict 
                model.load_state_dict(mapped_dict, strict=False)
                print(f"--> Bootstrapped weights from {Config.INIT_WEIGHTS}!")
            except Exception as e:
                print(f"--> Bootstrapping failed: {e}")
                
    if getattr(Config, "USE_FULL_RFDM", False):
        criterion = UncertaintyWeightedLoss(alpha=2.0)
        tv_criterion = TVLoss()
        recon_criterion = ReconstructionLoss(alpha=0.8, beta=0.2)
    else:
        criterion = DarknessWeightedLoss(alpha=2.0)
        tv_criterion = None
        recon_criterion = None
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # scheduler args (verbose -> deprecated in latest PyTorch, skipping log for cleanliness)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_dice = 0.0
    patience_counter = 0
    
    train_losses, val_losses, val_dices = [], [], []
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")
        
        train_stats = train_epoch(model, train_loader, optimizer, criterion, scaler, Config.DEVICE, tv_criterion, recon_criterion)
        val_stats = val_epoch(model, val_loader, criterion, Config.DEVICE)
        
        train_losses.append(train_stats["loss"])
        val_losses.append(val_stats["loss"])
        val_dices.append(val_stats["dice"])
        
        scheduler.step(val_stats["dice"])
        
        if Config.LOG_WANDB:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "train/dice": train_stats["dice"],
                "val/loss": val_stats["loss"],
                "val/dice": val_stats["dice"],
                "lr": optimizer.param_groups[0]['lr']
            })
            
        if val_stats["dice"] > best_dice:
            best_dice = val_stats["dice"]
            patience_counter = 0
            best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_illumiseg_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved best model with Val Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            print(f"Early stop counter: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered!")
                break
                
    # Save curves offline
    from utils import plot_multi_curve
    plot_multi_curve(range(1, len(train_losses)+1), {"Train": train_losses, "Validation": val_losses},
                    "Loss Curve", "Epoch", "Loss", os.path.join(Config.OUTPUT_DIR, "loss_curve.png"))
    plot_multi_curve(range(1, len(val_dices)+1), {"Validation Dice": val_dices},
                    "Validation Dice Curve", "Epoch", "Dice Score", os.path.join(Config.OUTPUT_DIR, "dice_curve.png"))
                    
    if Config.LOG_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()
