# Use an official PyTorch runtime as a parent image (CUDA enabled for GPU acceleration if available)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables to prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Clone the GitHub repository directly
RUN git clone https://github.com/mikhail12310/Polyp-segmentation .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# The default command runs the evaluation script to reproduce the final results.
# (If you want to train from scratch, you can override this with: docker run ... python train.py)
CMD ["python", "evaluate.py"]
