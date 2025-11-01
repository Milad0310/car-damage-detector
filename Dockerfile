FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system libs required for YOLO / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY ./requirements.txt /app/requirements.txt

# Install PyTorch CPU
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.3.1 \
        torchvision==0.18.1 \
        torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

EXPOSE 8080

# Required for Render (use $PORT instead of fixed)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "$PORT"]
