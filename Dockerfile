# Use FastAPI + Uvicorn base image with Python 3.11
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements first for better Docker caching
COPY ./requirements.txt /app/requirements.txt

# Update apt packages and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install PyTorch CPU wheels first (fixed versions)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the Python dependencies
RUN pip install --default-timeout=10000 --retries 10 --no-cache-dir \
        -i https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        -r /app/requirements.txt

# Verify OpenCV installation (optional, for debugging)
RUN python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# List app files (optional, for debugging)
RUN ls -la /app

# Expose the FastAPI port
EXPOSE 5000

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
