# Use NVIDIA CUDA base image with Ubuntu 22.04 and CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install Python, pip, and system dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set the entry point to run the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "app:app"]