import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import KvasirDataset, get_train_val_test_splits
from augmentations import get_validation_augmentation
from model import build_model
from utils import set_seed, calculate_metrics, MetricTracker, plot_multi_curve

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
    
    print("\n--- Phase 2: Comparative Evaluation (P1, P2, P3) ---")
    
    models = {}
    
    # P1: Baseline
    try:
        models["P1 Baseline"] = build_model(use_illumiseg=False).to(Config.DEVICE)
        models["P1 Baseline"].load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), map_location=Config.DEVICE))
    except FileNotFoundError:
        print("Warning: P1 Baseline checkpoint not found.")
        
    # P2: Dark-Trained
    try:
        models["P2 Dark-Trained"] = build_model(use_illumiseg=False).to(Config.DEVICE)
        models["P2 Dark-Trained"].load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_dark_model.pth"), map_location=Config.DEVICE))
    except FileNotFoundError:
        print("Warning: P2 Dark-Trained checkpoint not found.")
    
    # P3 logic removed
    
    # P3: Final RFDM (Last Trained)
    try:
        models["P3 Final RFDM"] = build_model(use_full_rfdm=True).to(Config.DEVICE)
        models["P3 Final RFDM"].load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_illumiseg_model.pth"), map_location=Config.DEVICE))
    except FileNotFoundError:
        print("Warning: P3 Final RFDM checkpoint not found.")
    
    if not models:
        raise ValueError("No models found. Please train models first.")
        
    print(f"Loaded {len(models)} models successfully.")
    
    splits = get_train_val_test_splits(Config.IMG_DIR, Config.MASK_DIR)
    
    results = []
    plot_data = {name: [] for name in models.keys()}
    
    for severity in Config.SEVERITY_LEVELS:
        loader = DataLoader(
            KvasirDataset(splits["test"][0], splits["test"][1], get_validation_augmentation(Config.IMG_SIZE), severity),
            batch_size=Config.BATCH_SIZE, shuffle=False
        )
        metrics = evaluate_models(models, loader, Config.DEVICE)
        
        row = {"Severity": severity}
        for name, m in metrics.items():
            row[name] = m["dice"]
            plot_data[name].append(m["dice"])
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    results_csv_path = os.path.join(Config.OUTPUT_DIR, "robustness_metrics.csv")
    df.to_csv(results_csv_path, index=False)
    
    print("\n====== FINAL COMPARISON ======")
    print(df.to_string(index=False))
    print("==============================\n")
    
    plot_multi_curve(
        Config.SEVERITY_LEVELS, 
        plot_data, 
        "Final Robustness Comparison", 
        "Severity", 
        "Dice", 
        os.path.join(Config.OUTPUT_DIR, "p1_p2_p3_comparison.png")
    )
    print("Saved comparative robustness evaluation assets to outputs/ directory.")

if __name__ == "__main__":
    main()
