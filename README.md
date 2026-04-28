# PolySeg: Robust Polyp Segmentation Under Complex Illumination

This repository contains **IllumiSeg**, an advanced, deep-learning-based medical image segmentation framework engineered to maintain state-of-the-art accuracy in heavily degraded, low-light, and corrupted illumination environments (typically encountered in endoscopic procedures).

At the core of this pipeline is the **Full Radiance-Guided Feature Disentanglement Module (RFDM)**. Unlike traditional approaches which either brute-force augmentations or rely entirely on heavy preprocessing logic, PolySeg enforces architectural priors dictating how the network routes, learns, and separates invariant anatomy from degradation representations. 

---

## 🏗 System Architecture 

Built on top of `segmentation_models_pytorch` utilizing a `ResNet34` UNet backbone, the codebase implements several custom feature-routing strategies in `model.py` and dynamic physics-inspired losses in `losses.py`.

### 1. Retinex Feature Disentanglement Block
The central bottleneck features ($F$) from the ResNet encoder are explicitly split into two decoupled representations based on classical Retinex theory:
* **$F_{reflect}$ (Reflectance Branch):** Encodes structural, illumination-invariant organ morphology. This tensor directly replaces the standard bottleneck routing and feeds the principal segmentation decoder.
* **$F_{illum}$ (Illumination Branch):** Encodes the ambient mapping and intensity degradation profiles. Rather than being discarded, this tensor serves as the foundation for the dual auxiliary heads.

### 2. Enhancement Decoder Auxiliary Branch
A heavily constrained, lightweight 5-stage bilinear-upsampling convolution block projects $F_{illum}$ back to raw RGB visual space. 
- During robust paired training (`train.py`), the network aims to reconstruct an artificially darkened image back to its parallel *clean* target. 
- This establishes pure representational understanding of spatial brightness corruptions by fusing an **L1 Distance Target** with a differentiable **Gaussian 11x11 SSIM Loss** (`ReconstructionLoss`).

### 3. Spatial Uncertainty Formulation 
Since standard BCE + Dice functions fail heavily when regions lack distinguishable local gradients, an **Uncertainty Head** computes a local spatial probability map ($U$) derived solely from the $F_{illum}$ branch. 
- Using `UncertaintyWeightedLoss`, BCE pixel penalties scale dynamically according to estimated uncertainty ($Loss = \text{mean}((1 + \alpha \cdot U) \cdot \text{BCE}) $). 
- Network gradients are thereby deliberately shifted to prioritize difficult, underexposed lesion boundaries.

### 4. Mathematical Regularization
- **Total Variation (TV) Loss**: Directly constrains $F_{illum}$ mapping boundaries to eliminate high-frequency semantic noise. 
- **Siamese Radiance-Consistency (CIC)**: Early Phase mechanisms used paired Euclidean loss maps bounding a clean anchor to a darkened target to map consistency. 

---

## 📊 Pipeline Stress Evaluation

The evaluation suite (`evaluate.py`) explicitly cascades images through Kvasir-SEG validation datasets modified by localized dynamic lighting filters simulating hardware limitations. 

The metrics confirm that while naive models collapse under heavy sensor-degradations, the custom **Full RFDM** architecture exhibits phenomenal recovery in severe bounds without relying on generative hallucination parameters.

| Testing Severity | Phase 1 (Clean Baseline) | Phase 2 (Naive Dark-Training) | Phase 3 (Full RFDM) |
| :---: | :---: | :---: | :---: |
| **Clean** | `0.858` | `0.827` | `0.792` |
| **Mild** | `0.829` | `0.831` | `0.801` |
| **Medium** | `0.628` | `0.785` | `0.770` |
| **Severe** | **`0.014` *(Collapse)*** | **`0.432`** | **`0.617` *(+42x Gain vs Zero-shot)*** |

<br>

**(Note: Output diagnostic curves detailing precision breakpoints across all thresholds are automatically written to `outputs/robustness_comparative_curve.png`)*.*

---

## 🚦 Executing the Pipeline

### 1. Requirements Configurations
```bash
pip install -r requirements.txt
```
*(Core dependnecies: `torch`, `segmentation_models_pytorch`, `albumentations`, `pandas`, `tqdm`)*

### 2. Runtime Flow (`config.py`)
Ensure your Kvasir-SEG (or identical format) dataset is properly mapped into `BASE_DIR`. You can quickly configure the network parameters from `config.py`:
- `USE_FULL_RFDM`: Toggles between the standard Baseline or the RFDM Enhancement logic.
- Target check-pointing bounds, early stopping mechanisms, AdamW LR constraints, and WandB active-logging links are modular.

### 3. Training & Validation
```bash
python train.py
```
This triggers an automatic subset-augmentation routine dynamically casting synthetic low-light masks dynamically parallelized across the `DataLoader`. Outputs, including scaled checkpoints, are tracked directly in `/outputs`. 

### 4. Stress Assessment 
```bash
python evaluate.py
```
Invokes the multi-stage triple inference layout—generating both numeric metric curves and plotting visually overlaid prediction sequences mapped directly over the source pathology masks (`sample_0.png`, etc.) for visual physician audits.