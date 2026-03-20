# Dockerfile - QuickDraw CNN with MediaPipe
FROM python:3.10-slim

# Install system dependencies for OpenCV + display
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY *.py ./

# Default command: run training
CMD ["python", "train_Quick_Draw.py"]