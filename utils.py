import os
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(preds, targets, smooth=1e-6):
    """
    Calculates Dice, IoU, Precision, Recall.
    preds: post-sigmoid probabilities [N, 1, H, W]
    targets: binary masks [N, 1, H, W]
    """
    preds = (preds > 0.5).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    tp = (preds_flat * targets_flat).sum()
    fp = ((1 - targets_flat) * preds_flat).sum()
    fn = (targets_flat * (1 - preds_flat)).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    intersection = tp
    union = (preds_flat.sum() + targets_flat.sum() - intersection)
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item()
    }

class MetricTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
        self.count = 0
        self.loss = 0
        
    def update(self, loss, metrics_dict, n=1):
        self.loss += loss * n
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = 0
            self.metrics[k] += v * n
        self.count += n
        
    def get_avg(self):
        return {
            "loss": self.loss / self.count if self.count > 0 else 0,
            **{k: v / self.count if self.count > 0 else 0 for k, v in self.metrics.items()}
        }

def denormalize_image(tensor):
    """Converts normalized CHW tensor back to HWC uint8 numpy array"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if len(tensor.shape) == 3 and tensor.shape[0] in [1, 3]:
        tensor = np.transpose(tensor, (1, 2, 0))
    tensor = np.squeeze(tensor)
    
    mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
    std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
    tensor = tensor * std + mean
    tensor = np.clip(tensor, 0, 1)
    return (tensor * 255).astype(np.uint8)

def save_prediction_grid(dark_img_tensor, true_mask_tensor, pred_mask_tensor, save_path):
    """
    Saves a comparison grid of 3 images.
    """
    img_np = denormalize_image(dark_img_tensor)
    
    true_mask_np = true_mask_tensor.detach().cpu().numpy().squeeze()
    pred_mask_np = pred_mask_tensor.detach().cpu().numpy().squeeze()
    pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask_np, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_triple_prediction_grid(img_tensor, true_mask_tensor, pred_base, pred_dark, pred_illum, save_path):
    """
    Saves a comparison grid of 5 images for Phase 3 comparison.
    """
    img_np = denormalize_image(img_tensor)
    
    true_mask_np = true_mask_tensor.detach().cpu().numpy().squeeze()
    
    pred_base_np = pred_base.detach().cpu().numpy().squeeze()
    pred_base_np = (pred_base_np > 0.5).astype(np.float32)
    
    pred_dark_np = pred_dark.detach().cpu().numpy().squeeze()
    pred_dark_np = (pred_dark_np > 0.5).astype(np.float32)
    
    pred_illum_np = pred_illum.detach().cpu().numpy().squeeze()
    pred_illum_np = (pred_illum_np > 0.5).astype(np.float32)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title("Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred_base_np, cmap='gray')
    axes[2].set_title("Phase 1 Baseline")
    axes[2].axis("off")
    
    axes[3].imshow(pred_dark_np, cmap='gray')
    axes[3].set_title("Phase 2 Dark-Trained")
    axes[3].axis("off")
    
    axes[4].imshow(pred_illum_np, cmap='gray')
    axes[4].set_title("Phase 3 IllumiSeg")
    axes[4].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_multi_curve(x_values, y_dict, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    for label, y_vals in y_dict.items():
         plt.plot(x_values, y_vals, marker='o', linewidth=2, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_robustness_curve(severities, metric_values, save_path, metric_name="Dice Score"):
    plt.figure(figsize=(8, 6))
    plt.plot(severities, metric_values, marker='o', linewidth=2, markersize=8, color='crimson')
    plt.title(f"Robustness Curve: {metric_name} vs Degradation Severity")
    plt.xlabel("Severity Level")
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    plt.savefig(save_path)
    plt.close()
