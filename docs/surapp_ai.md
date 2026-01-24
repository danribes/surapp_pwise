# SURAPP AI-Enhanced Extraction Guide

This document explains how to use AI-powered validation for Kaplan-Meier curve extraction using Ollama with llama3.2-vision.

## Overview

The AI-enhanced extraction adds a validation step that:
1. Compares the original image with the extracted curves overlay
2. Identifies potential extraction issues
3. Suggests parameter adjustments if needed
4. Provides confidence scores for extraction quality

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI-Enhanced Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Input   │───►│  Extraction  │───►│  Extracted Curves    │  │
│  │  Image   │    │  (existing)  │    │  (CSV + overlay PNG) │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                  │               │
│                                                  ▼               │
│                                      ┌──────────────────────┐   │
│                                      │   AI Validator       │   │
│                                      │   (llama3.2-vision)  │   │
│                                      │                      │   │
│                                      │  "Do the extracted   │   │
│                                      │   curves match the   │   │
│                                      │   original image?"   │   │
│                                      └──────────┬───────────┘   │
│                                                  │               │
│                         ┌────────────────────────┼────────┐     │
│                         ▼                        ▼        ▼     │
│                    ┌─────────┐            ┌─────────┐ ┌──────┐  │
│                    │  PASS   │            │  FAIL   │ │REVIEW│  │
│                    │ Output  │            │ Issues  │ │ User │  │
│                    │ results │            │ flagged │ │ check│  │
│                    └─────────┘            └─────────┘ └──────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware Requirements

| Model | RAM (CPU mode) | VRAM (GPU mode) | Performance |
|-------|----------------|-----------------|-------------|
| llama3.2-vision (11B) | 16GB+ | 8GB+ | Good quality, slower |

**Note:** CPU-only mode works but is slower. GPU acceleration recommended for frequent use.

### Software Requirements

- Docker and Docker Compose
- ~4GB disk space for the model

## Quick Start

### Option 1: Unified Entry Point (Recommended)

```bash
# Make executable (first time only)
chmod +x src/surapp.sh

# Run with AI mode (starts services automatically)
./src/surapp.sh --mode ai my_km_plot.png

# Or use interactive selection
./src/surapp.sh my_km_plot.png
# Then select [3] Docker (AI)
```

### Option 2: Docker Helper Scripts

```bash
# 1. Start AI services (first time downloads ~4GB model)
./docker/run-ai.sh --start

# 2. Run extraction with AI validation
./docker/run-ai.sh my_km_plot.png --validate

# 3. Check results in ./results/ folder
```

**Windows:**
```batch
docker\run-ai.bat --start
docker\run-ai.bat my_km_plot.png --validate
```

### Option 2: Native Python with Local Ollama

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the vision model
ollama pull llama3.2-vision

# 3. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements.ai.txt

# 4. Run extraction with validation
python src/extract_km_ai.py my_km_plot.png --validate
```

## Usage

### Basic Extraction (No AI)

```bash
# Standard extraction without AI validation
python src/extract_km_ai.py my_km_plot.png

# Or with Docker
./docker/run-ai.sh my_km_plot.png
```

### With AI Validation

```bash
# Enable AI validation
python src/extract_km_ai.py my_km_plot.png --validate

# With additional options
python src/extract_km_ai.py my_km_plot.png --time-max 36 --curves 2 --validate
```

### Check AI Service Status

```bash
# Python
python src/extract_km_ai.py --status

# Docker
./docker/run-ai.sh --status
```

### Manage AI Services (Docker)

```bash
# Start Ollama service
./docker/run-ai.sh --start
# Or: ./src/surapp.sh --start

# Stop Ollama service
./docker/run-ai.sh --stop
# Or: ./src/surapp.sh --stop

# Check status
./docker/run-ai.sh --status
# Or: ./src/surapp.sh --status
```

## Output Files

When AI validation is enabled, additional files are generated:

| File | Description |
|------|-------------|
| `ai_validation_report.txt` | Detailed AI validation report |
| `comparison_overlay.png` | Image used for AI comparison |

### Validation Report Format

```
AI Validation Report
========================================

Model: llama3.2-vision
Image: my_km_plot.png
Timestamp: 2024-01-15T10:30:00

Validation Result: VALID
  Match: YES
  Confidence: 92.0%
  Issues:
    - None
  Suggestions:
    - None

Raw Response:
----------------------------------------
MATCH: YES
CONFIDENCE: 0.92
ISSUES: None
SUGGESTIONS: None
```

## Understanding AI Validation Results

### Match Status

| Status | Meaning | Action |
|--------|---------|--------|
| **YES** | Curves match well | Results are reliable |
| **PARTIAL** | Some discrepancies | Review flagged issues |
| **NO** | Significant mismatch | Manual review needed |

### Confidence Score

- **> 0.8**: High confidence, results likely accurate
- **0.5 - 0.8**: Medium confidence, review recommended
- **< 0.5**: Low confidence, manual verification needed

### Common Issues Detected

1. **Curve deviation**: Extracted line doesn't follow original
2. **Missing sections**: Parts of curve not captured
3. **Wrong endpoints**: Start/end points incorrect
4. **Merged curves**: Multiple curves detected as one

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `AI_MODEL` | `llama3.2-vision` | Vision model to use |
| `AI_ENABLED` | `true` | Enable/disable AI |
| `AI_MAX_RETRIES` | `3` | Max retry attempts |
| `AI_TIMEOUT` | `120` | Request timeout (seconds) |
| `AI_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence |

### Docker Compose Configuration

Edit `docker-compose.ai.yml` to customize:

```yaml
services:
  ollama:
    # Enable GPU support (NVIDIA)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Architecture

### File Structure

```
surapp_pwise/
├── requirements.ai.txt     # AI Python dependencies
│
├── src/                    # Source scripts
│   ├── surapp.sh           # Unified entry point (Linux/Mac)
│   ├── surapp.bat          # Unified entry point (Windows)
│   └── extract_km_ai.py    # AI-enhanced extraction script
│
├── docker/                 # Docker configuration
│   ├── Dockerfile.ai       # AI-enabled container image
│   ├── docker-compose.ai.yml # AI services configuration
│   ├── run-ai.sh           # Linux/Mac helper script
│   └── run-ai.bat          # Windows helper script
│
└── lib/
    ├── ai_validator.py     # AI validation logic
    └── ai_config.py        # Configuration classes
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                     Host System                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────┐      ┌───────────────────────────┐  │
│  │   surapp-ai       │      │      ollama               │  │
│  │   container       │      │      container            │  │
│  │                   │ HTTP │                           │  │
│  │ extract_km_ai.py  │◄────►│  llama3.2-vision model   │  │
│  │ ai_validator.py   │:11434│                           │  │
│  │                   │      │  ~4GB model weights       │  │
│  └─────────┬─────────┘      └───────────────────────────┘  │
│            │                                                 │
│  ┌─────────┴─────────┐      ┌───────────────────────────┐  │
│  │   /data/input     │      │   ollama_data volume      │  │
│  │   (your images)   │      │   (persistent models)     │  │
│  └───────────────────┘      └───────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### AI Service Not Starting

**Symptom:** `Ollama Container: Not running`

**Solutions:**
1. Check Docker is running: `docker ps`
2. Check for port conflicts: `lsof -i :11434`
3. View logs: `docker logs surapp-ollama`

### Model Download Fails

**Symptom:** `Error pulling llama3.2-vision`

**Solutions:**
1. Check internet connection
2. Ensure sufficient disk space (~4GB)
3. Retry: `docker exec surapp-ollama ollama pull llama3.2-vision`

### Validation Timeout

**Symptom:** `Validation error: timeout`

**Solutions:**
1. Increase timeout: `export AI_TIMEOUT=300`
2. Use GPU acceleration for faster inference
3. Try a smaller model: `export AI_MODEL=moondream`

### Out of Memory

**Symptom:** Container crashes during validation

**Solutions:**
1. Close other applications to free RAM
2. Use a smaller model
3. Enable GPU offloading if available

### Slow Performance (CPU-only)

**Expected:** 30-60 seconds per validation on CPU

**Improvements:**
1. Enable GPU acceleration (see Configuration)
2. Use a smaller model like `moondream`
3. Skip validation for batch processing, validate samples

## Alternative Models

While llama3.2-vision is recommended, other models work:

| Model | Size | Quality | Speed | Command |
|-------|------|---------|-------|---------|
| llama3.2-vision | 11B | Best | Slow | `ollama pull llama3.2-vision` |
| llava | 7B | Good | Medium | `ollama pull llava` |
| moondream | 1.8B | Basic | Fast | `ollama pull moondream` |

To use an alternative:
```bash
export AI_MODEL=llava
python src/extract_km_ai.py my_plot.png --validate
```

## Best Practices

1. **Start with validation disabled** for initial parameter tuning
2. **Enable validation** for final extraction runs
3. **Review PARTIAL matches** manually - AI may flag minor issues
4. **Keep Ollama running** if processing multiple images
5. **Use GPU** for production workloads

## API Reference

### AIValidator Class

```python
from lib.ai_validator import AIValidator
from lib.ai_config import AIConfig

# Initialize
config = AIConfig.from_environment()
validator = AIValidator(config)

# Check availability
if validator.is_available:
    # Validate extraction
    result = validator.validate(
        original_image="original.png",
        overlay_image="overlay.png"
    )

    print(f"Match: {result.match}")
    print(f"Confidence: {result.confidence}")
    print(f"Valid: {result.is_valid}")
```

### ValidationResult Properties

| Property | Type | Description |
|----------|------|-------------|
| `match` | str | "YES", "NO", or "PARTIAL" |
| `confidence` | float | 0.0 to 1.0 |
| `is_valid` | bool | True if acceptable |
| `needs_retry` | bool | True if should retry |
| `issues` | list[str] | Detected problems |
| `suggestions` | list[str] | Parameter adjustments |
