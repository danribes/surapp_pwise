# SURAPP Docker Implementation Guide

This document explains the Docker implementation for SURAPP, enabling cross-platform compatibility across Mac, Linux, and Windows.

## Overview

Docker containerization ensures SURAPP runs identically on any system with Docker installed, eliminating "works on my machine" issues caused by:
- Different Python versions
- Missing system libraries (especially OpenCV dependencies)
- Conflicting package versions
- Operating system differences

All Docker files are located in the `docker/` folder to keep the project root clean.

## Docker Files

### docker/Dockerfile

The `Dockerfile` defines the container image:

```dockerfile
FROM python:3.11-slim
```

**Why Python 3.11-slim?**
- `python:3.11` - Modern Python with good performance
- `slim` variant - Smaller image size (~150MB vs ~900MB for full image)
- Debian-based - Easy to install system dependencies

**System Dependencies:**

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

These libraries are required by OpenCV:
| Library | Purpose |
|---------|---------|
| `libgl1-mesa-glx` | OpenGL support for image rendering |
| `libglib2.0-0` | GLib library for GTK |
| `libsm6` | X11 Session Management |
| `libxext6` | X11 extensions |
| `libxrender-dev` | X Rendering Extension |
| `libgomp1` | OpenMP for parallel processing |

**Application Structure:**

```dockerfile
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lib/ ./lib/
COPY src/extract_km.py .
COPY src/step1_preview_image.py .
COPY src/step2_calibrate_axes.py .
COPY src/step3_extract_curves.py .
```

Files are copied in order of change frequency:
1. `requirements.txt` first (rarely changes) - enables layer caching
2. Application code last (changes often)

This means rebuilding after code changes reuses the cached dependency layer.

**Data Directories:**

```dockerfile
RUN mkdir -p /data/input /data/output
WORKDIR /data
```

Standard mount points for input images and output results.

### docker/docker-compose.yml

Docker Compose simplifies running the container:

```yaml
version: '3.8'

services:
  surapp:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: surapp:latest
    volumes:
      - ../input:/data/input:ro   # Read-only input
      - ../output:/data/output    # Writable output
    working_dir: /data/input
    entrypoint: ["python", "/app/extract_km.py"]
```

**Volume Mounts:**
- `./input:/data/input:ro` - Input folder mounted read-only (`:ro`)
- `./output:/data/output` - Output folder mounted read-write

**Why read-only input?**
- Prevents accidental modification of source images
- Security best practice

### .dockerignore

Excludes unnecessary files from the Docker build context:

```
.git
docs/
*.png
*.jpg
results/
__pycache__/
venv/
```

**Benefits:**
- Faster builds (less data to transfer)
- Smaller build context
- Avoids copying test images into container

### Helper Scripts

**docker/run.sh (Linux/Mac):**
```bash
./docker/run.sh my_plot.png --time-max 24
```

**docker/run.bat (Windows):**
```batch
docker\run.bat my_plot.png --time-max 24
```

Both scripts:
1. Check if Docker is installed
2. Validate the input image exists
3. Create output directory if needed
4. Build the Docker image (first run only)
5. Run extraction with proper volume mounts
6. Display results location

## Usage

### Method 1: Unified Entry Point (Recommended)

The easiest way is to use the unified `surapp.sh` script:

**Linux/Mac:**
```bash
# Make executable (first time only)
chmod +x src/surapp.sh

# Run with Docker mode
./src/surapp.sh --mode docker path/to/your/km_plot.png

# Or interactive mode selection
./src/surapp.sh km_plot.png
```

**Windows:**
```batch
src\surapp.bat --mode docker path\to\your\km_plot.png
```

### Method 2: Docker Helper Scripts

**Linux/Mac:**
```bash
# Make script executable (first time only)
chmod +x docker/run.sh

# Run extraction
./docker/run.sh path/to/your/km_plot.png

# With options
./docker/run.sh km_plot.png --time-max 36 --curves 2
```

**Windows:**
```batch
docker\run.bat path\to\your\km_plot.png
docker\run.bat km_plot.png --time-max 36 --curves 2
```

Results are saved to a `results/` folder next to the input image.

### Method 3: Docker Compose

```bash
# From the docker/ folder
cd docker

# Place images in ../input folder
mkdir -p ../input ../output
cp your_plot.png ../input/

# Run extraction
docker-compose run surapp your_plot.png --time-max 24

# Results appear in ../output folder
```

### Method 4: Direct Docker Commands

```bash
# Build image (from project root)
docker build -t surapp:latest -f docker/Dockerfile .

# Run extraction
docker run --rm \
    -v "$(pwd)/input:/data/input:ro" \
    -v "$(pwd)/output:/data/output" \
    surapp:latest \
    python /app/extract_km.py /data/input/your_plot.png -o /data/output
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Host System                              │
│  ┌─────────────┐                      ┌─────────────┐       │
│  │ input/      │                      │ output/     │       │
│  │ ├─plot1.png │                      │ ├─curves.csv│       │
│  │ └─plot2.png │                      │ └─debug.png │       │
│  └──────┬──────┘                      └──────▲──────┘       │
│         │ :ro (read-only)                    │ :rw          │
├─────────┼────────────────────────────────────┼──────────────┤
│         ▼              Docker Container      │              │
│  ┌─────────────┐                      ┌──────┴──────┐       │
│  │/data/input/ │ ──► extract_km.py ──►│/data/output/│       │
│  └─────────────┘                      └─────────────┘       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ /app/                                                │   │
│  │ ├── extract_km.py                                    │   │
│  │ ├── step1_preview_image.py                           │   │
│  │ ├── step2_calibrate_axes.py                          │   │
│  │ ├── step3_extract_curves.py                          │   │
│  │ └── lib/                                             │   │
│  │     ├── detector.py                                  │   │
│  │     ├── calibrator.py                                │   │
│  │     └── color_detector.py                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Docker Not Found

**Error:** `Error: Docker is not installed.`

**Solution:** Install Docker:
- **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux:** `sudo apt install docker.io` (Ubuntu/Debian)

### Permission Denied (Linux)

**Error:** `Got permission denied while trying to connect to the Docker daemon socket`

**Solution:** Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Image Build Fails

**Error:** Network issues during `apt-get` or `pip install`

**Solution:** Retry with no-cache:
```bash
docker build --no-cache -t surapp:latest .
```

### Volume Mount Issues (Windows)

**Error:** Files not appearing in output folder

**Solution:** Ensure Docker Desktop has access to the drive:
1. Open Docker Desktop Settings
2. Go to Resources > File Sharing
3. Add the drive containing your files

### Container Runs But No Output

**Check:** Verify paths are correct
```bash
# Debug: Run interactive shell
docker run -it --rm \
    -v "$(pwd)/input:/data/input:ro" \
    -v "$(pwd)/output:/data/output" \
    surapp:latest \
    /bin/bash

# Inside container:
ls /data/input/    # Should show your images
ls /app/           # Should show Python files
```

## Performance Considerations

### First Run
The first run takes longer because it:
1. Downloads the Python 3.11-slim base image (~150MB)
2. Installs system dependencies
3. Installs Python packages

Subsequent runs are fast (seconds) because layers are cached.

### Image Size

| Component | Size |
|-----------|------|
| Base image (python:3.11-slim) | ~150MB |
| System libraries | ~50MB |
| Python packages | ~200MB |
| Application code | ~100KB |
| **Total** | **~400MB** |

### Rebuilding After Code Changes

Only the application layer is rebuilt:
```bash
docker build -t surapp:latest .
# Output: "Using cache" for dependency layers
# Only final COPY layer rebuilds
```

## Security

The Docker setup follows security best practices:

1. **Read-only input mount** - Cannot modify source images
2. **Non-root user** - Could be added for production
3. **Minimal base image** - Reduced attack surface
4. **No network required** - Container runs offline
5. **Temporary containers** - `--rm` flag removes container after use

## Comparison: Docker vs Native Installation

| Aspect | Docker | Native Python |
|--------|--------|---------------|
| Setup time | ~5 min (first run) | ~2 min |
| Reproducibility | Guaranteed | Depends on environment |
| Disk space | ~400MB | ~200MB |
| Startup time | ~2 sec | Instant |
| System libraries | Included | Must install manually |
| Python version | Fixed (3.11) | System dependent |
| Isolation | Complete | Shares system packages |

**Use Docker when:**
- You need guaranteed reproducibility
- System has conflicting Python packages
- Deploying to multiple machines
- You prefer not to install system dependencies

**Use native Python when:**
- You want faster startup
- Disk space is limited
- You're already familiar with Python environments
- You need to modify/debug the code frequently
