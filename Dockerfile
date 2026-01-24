# SURAPP - Kaplan-Meier Curve Extractor
# Docker image for cross-platform compatibility

# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set metadata
LABEL maintainer="SURAPP Team"
LABEL description="Kaplan-Meier survival curve data extractor"
LABEL version="1.0"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY lib/ ./lib/
COPY extract_km.py .
COPY step1_preview_image.py .
COPY step2_calibrate_axes.py .
COPY step3_extract_curves.py .

# Create directories for input/output
RUN mkdir -p /data/input /data/output

# Set the data directory as the working directory for processing
WORKDIR /data

# Default command - show help
CMD ["python", "/app/extract_km.py", "--help"]
