# Use slim Python 3.11 image
FROM python:3.11-slim

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system libraries required for YOLO / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install PyTorch CPU version
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.3.1 \
        torchvision==0.18.1 \
        torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

# Expose port (Render will use $PORT env variable)
EXPOSE 8080

# Run the app (shell form so $PORT is expanded)
CMD uvicorn server:app --host 0.0.0.0 --port $PORT
