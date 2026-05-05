# IllumiSeg: Robust Medical Image Segmentation

This project implements **FullRFDMUNet**, an advanced architecture for segmenting medical images (such as polyps) that is highly robust to severe illumination changes and extreme low-light conditions.

## Reproducing Results with Docker

This project is fully containerized. To ensure reproducibility, the Docker container automatically pulls the latest source code directly from this GitHub repository. You do not need to install Python, PyTorch, or configure any virtual environments to reproduce the findings.

### Prerequisites
1. **Docker**: Ensure Docker is installed and running on your machine.
   * [Download Docker Desktop (Windows/Mac)](https://www.docker.com/products/docker-desktop)
   * [Download Docker Engine (Linux)](https://docs.docker.com/engine/install/)

### Step-by-Step Execution

**Step 1: Download the Dataset and Weights**
Because of their size, the `Kvasir-SEG` dataset and the `.pth` model weights are not hosted on GitHub.
1. Download the `IllumiSeg_Data_and_Weights.zip` package from the provided Google Drive link.
2. Extract the ZIP file into an empty folder on your computer. You should now see two folders:
   * `Kvasir-SEG/` (contains `images/` and `masks/`)
   * `outputs/` (contains `checkpoints/` with the model weights)

**Step 2: Build the Docker Image**
Open your terminal, navigate to the folder where you extracted the files, and run:
```bash
# This command pulls the official code from GitHub and sets up the environment
docker build -t illumiseg-project https://github.com/mikhail12310/Polyp-segmentation.git
```

**Step 3: Run the Evaluation Pipeline**
Since the code runs inside an isolated container, you must "mount" your local dataset and weights so the code can access them. Run the following command:

On **Linux/macOS**:
```bash
docker run --rm \
  -v "$(pwd)/Kvasir-SEG:/app/Kvasir-SEG" \
  -v "$(pwd)/outputs:/app/outputs" \
  illumiseg-project python evaluate.py
```

On **Windows (Command Prompt / PowerShell)**:
```cmd
docker run --rm -v "%cd%\Kvasir-SEG:/app/Kvasir-SEG" -v "%cd%\outputs:/app/outputs" illumiseg-project python evaluate.py
```

### What happens next?
The container will boot up, load the three primary model checkpoints (Baseline, Dark-Trained, and Final RFDM), and evaluate them across all illumination severities (`clean`, `mild`, `medium`, `severe`). 

Once finished, the comparative results will be printed directly to your console, matching the benchmarks of the final model.

---

### Advanced: Running Training
If you wish to run the training pipeline from scratch (which bootstraps from the `best_dark_model.pth` and trains the `EnhancementDecoder`), run the following command instead of the evaluation script:

**Linux/macOS:**
```bash
docker run --rm \
  -v "$(pwd)/Kvasir-SEG:/app/Kvasir-SEG" \
  -v "$(pwd)/outputs:/app/outputs" \
  illumiseg-project python train.py
```
