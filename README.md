# IllumiSeg: Robust Medical Image Segmentation

This project implements **FullRFDMUNet**, an advanced architecture for segmenting medical images (such as polyps) that is highly robust to severe illumination changes and extreme low-light conditions.

## Reproducing Results with Docker

This project is fully containerized. You do not need to install Python, PyTorch, or configure any virtual environments to reproduce the findings. 

### Prerequisites
1. **Docker**: Ensure Docker is installed and running on your machine.
   * [Download Docker Desktop (Windows/Mac)](https://www.docker.com/products/docker-desktop)
   * [Download Docker Engine (Linux)](https://docs.docker.com/engine/install/)
2. **Git**: To clone the repository.

### Step-by-Step Execution

**Step 1: Clone the Repository**
Open your terminal and clone this project:
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

**Step 2: Provide the Dataset**
Ensure the `Kvasir-SEG` dataset folder is located in the root of the project directory. The container expects the images and masks to be in:
* `Kvasir-SEG/images/`
* `Kvasir-SEG/masks/`

**Step 3: Build and Run with a Single Command**
Run the provided shell script. This script will automatically build the Docker image with all necessary dependencies and execute the evaluation pipeline to reproduce the benchmark results.

On Linux/macOS:
```bash
chmod +x run_docker.sh
./run_docker.sh
```

On Windows (Command Prompt / PowerShell):
```cmd
docker build -t illumiseg-project .
docker run --rm illumiseg-project python evaluate.py
```

### What happens next?
The container will boot up, load the four primary model checkpoints (Baseline, Dark-Trained, Original RFDM, Final RFDM), and evaluate them across all illumination severities (`clean`, `mild`, `medium`, `severe`). 

Once finished, the comparative results will be printed directly to your console, matching the benchmarks of the final model.

---

### Advanced: Running Training
If you wish to run the training pipeline from scratch (which bootstraps from the `best_dark_model.pth` and trains the `EnhancementDecoder`), run the following command instead of the evaluation script:

```bash
docker run --rm illumiseg-project python train.py
```
