"""
eval_all_checkpoints.py
=======================
Evaluates EVERY .pth checkpoint in outputs/checkpoints/ across all four
severity levels (clean / mild / medium / severe).

Architecture is inferred from the filename:
  *illumiseg*  →  IllumiSegUNet  (use_illumiseg=True)
  *rfdm*       →  FullRFDMUNet   (use_full_rfdm=True)
  anything else→  plain UNet baseline

Results go to:  outputs/all_checkpoints_eval/
  ├── metrics_summary.csv          ← full table (model × severity × metric)
  ├── dice_heatmap.png             ← Dice scores as colour-coded heatmap
  ├── robustness_curves.png        ← Dice vs severity for every model
  ├── iou_curves.png
  └── per_model/<name>/
        eval_<severity>/sample_XXXX.png  ← 3-panel comparison grids

Usage
-----
python eval_all_checkpoints.py
python eval_all_checkpoints.py --severity severe        # single severity
python eval_all_checkpoints.py --num_save 4             # fewer grids per model
"""

import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import KvasirDataset, get_train_val_test_splits
from augmentations import get_validation_augmentation
from model import build_model
from utils import set_seed, calculate_metrics, MetricTracker, denormalize_image


# ── Architecture inference ────────────────────────────────────────────────────

def infer_arch(ckpt_name: str) -> dict:
    """Return kwargs for build_model() based on the checkpoint filename."""
    n = ckpt_name.lower()
    if "illumiseg" in n:
        return dict(use_illumiseg=True, use_full_rfdm=False)
    if "rfdm" in n:
        return dict(use_illumiseg=False, use_full_rfdm=True)
    return dict(use_illumiseg=False, use_full_rfdm=False)


def friendly_name(ckpt_name: str) -> str:
    """Turn a filename into a short readable label."""
    stem = os.path.splitext(ckpt_name)[0]
    return stem.replace("_", " ").title()


# ── Model loader ──────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device) -> torch.nn.Module:
    """Build model + load weights; falls back to strict=False on mismatch."""
    ckpt_name = os.path.basename(ckpt_path)
    arch = infer_arch(ckpt_name)
    model = build_model(**arch).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  ✓  {ckpt_name}  [{arch}]")
    return model


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_sample_grid(dark_tensor, mask_tensor, pred_tensor, save_path: str, dice: float):
    """3-panel: dark input | ground truth | prediction."""
    img_np  = denormalize_image(dark_tensor.cpu())
    gt_np   = mask_tensor.cpu().numpy().squeeze()
    pred_np = (pred_tensor.cpu().numpy().squeeze() > 0.5).astype(np.float32)

    # contour overlay on the dark image
    overlay = img_np.copy()
    contours, _ = cv2.findContours(
        (pred_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 80), 2)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.patch.set_facecolor("#0d0d0d")
    panels = [
        (img_np,  "Dark Input",              None),
        (gt_np,   "Ground Truth",            "gray"),
        (pred_np, f"Prediction  Dice={dice:.3f}", "hot"),
        (overlay, "Overlay",                 None),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")
    plt.tight_layout(pad=0.4)
    plt.savefig(save_path, bbox_inches="tight", dpi=110, facecolor=fig.get_facecolor())
    plt.close()


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model, dataloader, device, out_dir: str, num_save: int):
    """Run one model on one severity. Returns avg metrics dict."""
    os.makedirs(out_dir, exist_ok=True)
    tracker = MetricTracker()
    saved   = 0

    with torch.no_grad():
        for images_clean, images_dark, masks, darkness_maps in dataloader:
            images_dark   = images_dark.to(device)
            masks         = masks.to(device)

            device_type = "cuda" if "cuda" in str(device) else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.float16,
                                enabled=(device_type == "cuda")):
                preds = model(images_dark)

            probs = torch.sigmoid(preds)
            m     = calculate_metrics(probs, masks)
            tracker.update(0, m, images_dark.size(0))

            if saved < num_save:
                for j in range(min(images_dark.size(0), num_save - saved)):
                    dice_j = calculate_metrics(probs[j:j+1], masks[j:j+1])["dice"]
                    sp = os.path.join(out_dir, f"sample_{saved:04d}.png")
                    save_sample_grid(images_dark[j], masks[j], probs[j], sp, dice_j)
                    saved += 1

    return tracker.get_avg()


# ── Summary plots ─────────────────────────────────────────────────────────────

PALETTE = [
    "#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f87171",
    "#fb923c", "#60a5fa", "#e879f9", "#4ade80", "#fbbf24",
]

def _style_ax(ax, title):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines[:].set_color("#444")
    ax.grid(True, linestyle="--", alpha=0.25, color="#888")


def plot_robustness_curves(pivot: pd.DataFrame, metric: str, title: str, save_path: str):
    """Line chart: metric vs severity for each model."""
    severities = Config.SEVERITY_LEVELS
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#111111")
    _style_ax(ax, title)

    models = [c for c in pivot.columns if c != "Severity"]
    for i, m in enumerate(models):
        col = PALETTE[i % len(PALETTE)]
        vals = [pivot.loc[pivot["Severity"] == s, m].values[0] for s in severities]
        ax.plot(severities, vals, "o-", color=col, linewidth=2.2,
                markersize=7, label=m, zorder=3)
        for x, y in zip(severities, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", color=col, fontsize=7.5)

    ax.set_xlabel("Severity Level", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dice_heatmap(pivot: pd.DataFrame, save_path: str):
    """Heatmap: models × severities, colour = Dice."""
    models     = [c for c in pivot.columns if c != "Severity"]
    severities = Config.SEVERITY_LEVELS
    data = np.array([
        [pivot.loc[pivot["Severity"] == s, m].values[0] for s in severities]
        for m in models
    ])  # shape: (n_models, n_severities)

    fig, ax = plt.subplots(figsize=(max(7, len(severities) * 2),
                                    max(4, len(models) * 0.8 + 1)))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(data, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([s.capitalize() for s in severities], color="white", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, color="white", fontsize=9)
    ax.tick_params(length=0)
    ax.set_title("Dice Score — All Models × All Severities", color="white",
                 fontsize=12, pad=10)

    for i in range(len(models)):
        for j in range(len(severities)):
            val  = data[i, j]
            tc   = "black" if val > 0.6 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=tc, fontsize=10, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.set_label("Dice Score", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bar_chart(summary_df: pd.DataFrame, save_path: str):
    """Grouped bar chart: each severity group, bars = models."""
    models     = summary_df["Model"].unique().tolist()
    severities = Config.SEVERITY_LEVELS

    x      = np.arange(len(severities))
    n      = len(models)
    width  = 0.8 / n

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#111111")
    _style_ax(ax, "Dice Score by Model & Severity")

    for i, m in enumerate(models):
        vals = [
            summary_df.loc[(summary_df["Model"] == m) &
                           (summary_df["Severity"] == s), "Dice"].values[0]
            for s in severities
        ]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=PALETTE[i % len(PALETTE)], label=m,
                      alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", color="white",
                    fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in severities])
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.1)
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints across all severity levels."
    )
    parser.add_argument("--severity", type=str, default=None,
                        choices=["clean", "mild", "medium", "severe"],
                        help="Evaluate a single severity only.")
    parser.add_argument("--num_save", type=int, default=6,
                        help="Sample grids to save per model per severity.")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(Config.OUTPUT_DIR, "all_checkpoints_eval"))
    args = parser.parse_args()

    set_seed(Config.SEED)
    device     = Config.DEVICE
    severities = [args.severity] if args.severity else Config.SEVERITY_LEVELS

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Discover checkpoints ────────────────────────────────────────────────
    ckpt_dir   = Config.CHECKPOINT_DIR
    ckpt_files = sorted([
        f for f in os.listdir(ckpt_dir) if f.endswith(".pth")
    ])
    if not ckpt_files:
        raise RuntimeError(f"No .pth files found in {ckpt_dir}")

    print(f"\nFound {len(ckpt_files)} checkpoints:")
    for f in ckpt_files:
        size_mb = os.path.getsize(os.path.join(ckpt_dir, f)) / 1e6
        print(f"  {f:45s}  {size_mb:.0f} MB")

    # ── Load all models ────────────────────────────────────────────────────
    print(f"\nLoading models onto: {device}")
    models = {}
    for fname in ckpt_files:
        try:
            models[fname] = load_checkpoint(os.path.join(ckpt_dir, fname), device)
        except Exception as e:
            print(f"  ✗  {fname}  FAILED: {e}")

    # ── Data splits ────────────────────────────────────────────────────────
    splits = get_train_val_test_splits(Config.IMG_DIR, Config.MASK_DIR)
    test_img_paths, test_mask_paths = splits["test"]

    # ── Evaluation loop ────────────────────────────────────────────────────
    print(f"\nEvaluating {len(models)} models × {len(severities)} severities …\n")
    all_rows = []

    for sev in severities:
        test_dataset = KvasirDataset(
            test_img_paths, test_mask_paths,
            augmentations=get_validation_augmentation(Config.IMG_SIZE),
            severity=sev,
        )
        loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(str(device) != "cpu"),
        )

        print(f"── Severity: {sev.upper()} ──────────────────────────────────")
        for fname, model in models.items():
            label   = friendly_name(fname)
            out_dir = os.path.join(args.out_dir, "per_model",
                                   os.path.splitext(fname)[0], f"eval_{sev}")
            pbar = tqdm(loader, desc=f"  {label[:35]:35s}", leave=False)

            tracker = MetricTracker()
            saved   = 0

            with torch.no_grad():
                for images_clean, images_dark, masks, darkness_maps in pbar:
                    images_dark = images_dark.to(device)
                    masks       = masks.to(device)

                    device_type = "cuda" if "cuda" in str(device) else "cpu"
                    with torch.autocast(device_type=device_type, dtype=torch.float16,
                                        enabled=(device_type == "cuda")):
                        preds = model(images_dark)

                    probs = torch.sigmoid(preds)
                    m     = calculate_metrics(probs, masks)
                    tracker.update(0, m, images_dark.size(0))
                    pbar.set_postfix(dice=f"{m['dice']:.3f}")

                    if saved < args.num_save:
                        os.makedirs(out_dir, exist_ok=True)
                        for j in range(min(images_dark.size(0),
                                          args.num_save - saved)):
                            dice_j = calculate_metrics(
                                probs[j:j+1], masks[j:j+1])["dice"]
                            sp = os.path.join(out_dir, f"sample_{saved:04d}.png")
                            save_sample_grid(
                                images_dark[j], masks[j], probs[j], sp, dice_j)
                            saved += 1

            avg = tracker.get_avg()
            all_rows.append({
                "Model":     label,
                "Severity":  sev,
                "Dice":      round(avg["dice"],      4),
                "IoU":       round(avg["iou"],       4),
                "Precision": round(avg["precision"], 4),
                "Recall":    round(avg["recall"],    4),
            })
            print(f"  {label[:40]:40s}  "
                  f"Dice={avg['dice']:.4f}  IoU={avg['iou']:.4f}  "
                  f"Prec={avg['precision']:.4f}  Rec={avg['recall']:.4f}")

        print()

    # ── Save CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out_dir, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)

    # ── Print final table ───────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  ALL CHECKPOINTS — FULL RESULTS")
    print("═" * 80)
    pivot = df.pivot_table(index="Severity", columns="Model",
                           values="Dice").reset_index()
    pivot.columns.name = None
    print(df.to_string(index=False))
    print("═" * 80 + "\n")

    # Best model per severity
    print("  Best model per severity (Dice):")
    for sev in severities:
        sub  = df[df["Severity"] == sev]
        best = sub.loc[sub["Dice"].idxmax()]
        print(f"    {sev:8s}  →  {best['Model']}  (Dice={best['Dice']:.4f})")

    # Best model overall
    mean_dice = df.groupby("Model")["Dice"].mean().sort_values(ascending=False)
    print(f"\n  Best overall (mean Dice across severities):")
    for name, val in mean_dice.items():
        print(f"    {name:45s}  {val:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    if len(severities) > 1:
        # Build pivot for plotting
        plot_pivot = df[["Severity", "Model", "Dice"]].pivot(
            index="Severity", columns="Model", values="Dice").reset_index()
        plot_pivot.columns.name = None

        plot_robustness_curves(
            plot_pivot, "Dice Score",
            "Robustness: Dice vs Severity — All Checkpoints",
            os.path.join(args.out_dir, "dice_robustness_curves.png"),
        )

        iou_pivot = df[["Severity", "Model", "IoU"]].pivot(
            index="Severity", columns="Model", values="IoU").reset_index()
        iou_pivot.columns.name = None
        plot_robustness_curves(
            iou_pivot, "IoU Score",
            "Robustness: IoU vs Severity — All Checkpoints",
            os.path.join(args.out_dir, "iou_robustness_curves.png"),
        )

        plot_dice_heatmap(plot_pivot, os.path.join(args.out_dir, "dice_heatmap.png"))
        plot_bar_chart(df, os.path.join(args.out_dir, "dice_bar_chart.png"))

    print(f"\nAll outputs written to: {args.out_dir}\n")


if __name__ == "__main__":
    main()
