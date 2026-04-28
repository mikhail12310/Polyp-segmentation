import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import KvasirDataset, get_train_val_test_splits
from augmentations import get_validation_augmentation
from model import build_model
from utils import set_seed, calculate_metrics, MetricTracker, save_prediction_grid, save_triple_prediction_grid, plot_robustness_curve, plot_multi_curve

def evaluate_severity_triple(base_model, dark_model, rfdm_model, dataloader, device, severity, save_dir, num_samples_to_save=12):
    base_model.eval()
    dark_model.eval()
    rfdm_model.eval()
    
    samples_saved = 0
    base_tracker = MetricTracker()
    dark_tracker = MetricTracker()
    rfdm_tracker = MetricTracker()
    
    os.makedirs(save_dir, exist_ok=True)
    
    pbar = tqdm(dataloader, desc=f"Eval [{severity}]")
    with torch.no_grad():
        for i, (images_clean, images_dark, masks, darkness_maps) in enumerate(pbar):
            images_dark = images_dark.to(device)
            masks = masks.to(device)
            darkness_maps = darkness_maps.to(device)
            
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                preds_base = base_model(images_dark)
                preds_dark = dark_model(images_dark)
                preds_rfdm = rfdm_model(images_dark)
                
            preds_prob_base = torch.sigmoid(preds_base)
            preds_prob_dark = torch.sigmoid(preds_dark)
            preds_prob_rfdm = torch.sigmoid(preds_rfdm)
            
            metrics_base = calculate_metrics(preds_prob_base, masks)
            metrics_dark = calculate_metrics(preds_prob_dark, masks)
            metrics_rfdm = calculate_metrics(preds_prob_rfdm, masks)
            
            base_tracker.update(0, metrics_base, images_dark.size(0))
            dark_tracker.update(0, metrics_dark, images_dark.size(0))
            rfdm_tracker.update(0, metrics_rfdm, images_dark.size(0))
            
            # Save predictions across batches until limit is reached
            if samples_saved < num_samples_to_save:
                for j in range(min(images_dark.size(0), num_samples_to_save - samples_saved)):
                    save_path = os.path.join(save_dir, f"sample_{samples_saved}.png")
                    save_triple_prediction_grid(
                        images_dark[j], masks[j], 
                        preds_prob_base[j], preds_prob_dark[j], preds_prob_rfdm[j], 
                        save_path
                    )
                    samples_saved += 1
                    
    return base_tracker.get_avg(), dark_tracker.get_avg(), rfdm_tracker.get_avg()

def main():
    set_seed(Config.SEED)
    
    model_base_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_base_path):
        raise FileNotFoundError(f"Baseline model not found at {model_base_path}.")
        
    model_dark_path = os.path.join(Config.CHECKPOINT_DIR, "best_dark_model.pth")
    if not os.path.exists(model_dark_path):
        raise FileNotFoundError(f"Dark-trained model not found at {model_dark_path}.")
        
    model_rfdm_path = os.path.join(Config.CHECKPOINT_DIR, "best_illumiseg_model.pth")
    if not os.path.exists(model_rfdm_path):
        raise FileNotFoundError(f"RFDM model not found at {model_rfdm_path}.")
        
    model_base = build_model(use_illumiseg=False).to(Config.DEVICE)
    model_base.load_state_dict(torch.load(model_base_path, map_location=Config.DEVICE))
    
    model_dark = build_model(use_illumiseg=False).to(Config.DEVICE)
    model_dark.load_state_dict(torch.load(model_dark_path, map_location=Config.DEVICE))
    
    # Evaluate with the fine-tuned Full RFDM model
    model_rfdm = build_model(use_full_rfdm=True).to(Config.DEVICE)
    try:
        model_rfdm.load_state_dict(torch.load(model_rfdm_path, map_location=Config.DEVICE))
    except Exception:
        # Graceful fallback mapping if needed for base model keys
        state_dict = torch.load(model_rfdm_path, map_location=Config.DEVICE)
        model_rfdm.load_state_dict(state_dict, strict=False)
    
    print("Loaded all three models successfully.")
    
    splits = get_train_val_test_splits(Config.IMG_DIR, Config.MASK_DIR)
    test_img_paths, test_mask_paths = splits["test"]
    
    results = []
    dices_base = []
    dices_dark = []
    dices_rfdm = []
    
    print("\nStarting Comparative Robustness Evaluation Pipeline...")
    for severity in Config.SEVERITY_LEVELS:
        test_dataset = KvasirDataset(
            test_img_paths, test_mask_paths,
            augmentations=get_validation_augmentation(Config.IMG_SIZE),
            severity=severity
        )
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        
        save_dir = os.path.join(Config.OUTPUT_DIR, f"eval_{severity}")
        metrics_base, metrics_dark, metrics_rfdm = evaluate_severity_triple(model_base, model_dark, model_rfdm, test_loader, Config.DEVICE, severity, save_dir)
        
        results.append({
            "Severity": severity,
            "P1 Baseline": metrics_base["dice"],
            "P2 Dark-Trained": metrics_dark["dice"],
            "P3 Full RFDM": metrics_rfdm["dice"]
        })
        dices_base.append(metrics_base["dice"])
        dices_dark.append(metrics_dark["dice"])
        dices_rfdm.append(metrics_rfdm["dice"])
        
    # Stats Dataframe
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(Config.OUTPUT_DIR, "robustness_metrics.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print("\n====== STRESS TEST RESULTS ======")
    print(results_df.to_string(index=False))
    print("=================================\n")
    
    # Degradation plot
    plot_multi_curve(
        Config.SEVERITY_LEVELS, 
        {"Phase 1 Baseline": dices_base, "Phase 2 Dark-Trained": dices_dark, "Phase 3 Full RFDM": dices_rfdm},
        "Comparative Robustness Curve", "Severity Level", "Dice Score",
        os.path.join(Config.OUTPUT_DIR, "robustness_comparative_curve.png")
    )
    print("Saved comparative robustness evaluation assets to outputs/ directory.")

if __name__ == "__main__":
    main()
