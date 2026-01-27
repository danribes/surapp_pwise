# SURAPP - Kaplan-Meier Curve Extractor

Extract survival data coordinates from Kaplan-Meier plot images.

This tool detects curves in KM plot images (both solid and dashed line styles) and exports the extracted time-survival coordinates to CSV files for further analysis.

## Features

- Automatic axis detection and calibration
- Solid and dashed curve differentiation
- Multi-curve extraction with color detection
- CSV export with time and survival values
- Debug visualization for verification
- Cross-platform support (Windows, Mac, Linux)
- **AI-powered validation** using llama3.2-vision (optional)

## Quick Start

SURAPP provides a **unified entry point** that lets you choose how to run:

**Linux/Mac:**
```bash
# Make executable (first time only)
chmod +x src/surapp.sh

# Run - will prompt for execution mode
./src/surapp.sh my_km_plot.png
```

**Windows:**
```batch
src\surapp.bat my_km_plot.png
```

You'll see a menu to select the execution mode:

```
╔═══════════════════════════════════════════════════════════╗
║           SURAPP - Kaplan-Meier Curve Extractor           ║
╚═══════════════════════════════════════════════════════════╝

Select execution mode:

  [1] Python (Native)     - Fastest startup           [Ready]
  [2] Docker (Standard)   - No Python setup needed    [Ready]
  [3] Docker (AI)         - With AI validation        [Ready]

  [0] Cancel

Select mode [1-3]:
```

Or specify the mode directly:

```bash
./src/surapp.sh --mode python my_plot.png      # Native Python
./src/surapp.sh --mode docker my_plot.png      # Docker container
./src/surapp.sh --mode ai my_plot.png          # AI-enhanced (Docker)
```

Check system status:
```bash
./src/surapp.sh --status
```

---

## Execution Modes

| Mode | Command | Requirements | Best For |
|------|---------|--------------|----------|
| **Python** | `--mode python` | Python + packages | Fast iteration, development |
| **Docker** | `--mode docker` | Docker only | Guaranteed compatibility |
| **AI** | `--mode ai` | Docker + ~4GB model | Validation, quality assurance |

---

## Installation Options

### Option 1: Docker Installation (Recommended for most users)

Docker ensures SURAPP works identically on any system, with all dependencies pre-configured.

### Prerequisites

Install Docker:
- **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux:** `sudo apt install docker.io` (Ubuntu/Debian)

### Quick Start with Docker

**Linux/Mac:**
```bash
# Make script executable (first time only)
chmod +x docker/run.sh

# Run extraction
./docker/run.sh your_km_plot.png

# With options
./docker/run.sh your_km_plot.png --time-max 36 --curves 2
```

**Windows:**
```batch
docker\run.bat your_km_plot.png
docker\run.bat your_km_plot.png --time-max 36 --curves 2
```

Results are saved to a `results/` folder next to your input image.

### Alternative: Docker Compose

```bash
# Create input/output folders
mkdir -p input output

# Copy your image to input folder
cp your_km_plot.png input/

# Run extraction
docker-compose run surapp your_km_plot.png --time-max 24

# Results appear in ./output folder
```

For detailed Docker documentation, see [docs/surapp_docker.md](docs/surapp_docker.md).

---

## Option 2: Native Python Installation

Install and run directly with Python for faster startup and easier development.

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python numpy pandas matplotlib pillow
```

**Note for Linux users:** You may need system libraries for OpenCV:
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libglib2.0-0

# Fedora/RHEL
sudo dnf install mesa-libGL glib2
```

**Note for Mac users:** If you encounter issues:
```bash
# Install with Homebrew
brew install python@3.11
pip3 install opencv-python numpy pandas matplotlib pillow
```

### Step 2: Run Extraction

**Interactive mode** (recommended) - select image from list:

```bash
python src/extract_km.py
```

Output:
```
Found 2 image(s):

  [1] my_km_plot.png  (45.2 KB)
  [2] another_plot.jpg  (32.1 KB)

  [0] Cancel

Select image number: 1
```

**Direct mode** - specify image path:

```bash
python src/extract_km.py your_km_plot.png --time-max 40 --curves 2
```

Results are saved to `results/<image_name>_<timestamp>/`.

---

## Step-by-Step Extraction

For more control over the process, use the sequential scripts.
All scripts support interactive image selection (just run without arguments).

### Step 1: Preview Image

```bash
python src/step1_preview_image.py              # Interactive
python src/step1_preview_image.py image.png    # Direct
```

Shows image dimensions, color type, and basic properties.

### Step 2: Calibrate Axes

```bash
python src/step2_calibrate_axes.py                        # Interactive
python src/step2_calibrate_axes.py image.png --time-max 50
```

Detects plot area boundaries and axis ranges.

### Step 3: Extract Curves

```bash
python src/step3_extract_curves.py                                  # Interactive
python src/step3_extract_curves.py image.png --time-max 40 --curves 2
```

Detects curves and exports coordinates to CSV files.

---

## Output Files

| File | Description |
|------|-------------|
| `all_curves.csv` | Combined data from all curves |
| `curve_solid_1.csv` | Data for the solid curve |
| `curve_dashed_1.csv` | Data for the dashed curve |
| `extracted_curves.png` | Visualization of extracted curves |
| `debug_*.png` | Debug images showing detection steps |

### CSV Format

```csv
Time,Survival
0.00,1.0000
0.50,0.9800
1.00,0.9500
...
```

---

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--time-max TIME` | Maximum time value on X-axis |
| `--curves N` | Expected number of curves (default: 2) |
| `-o, --output DIR` | Output directory |
| `-q, --quiet` | Suppress progress messages |

## Examples

```bash
# Interactive mode - select image from list
python src/extract_km.py

# Direct mode - specify image
python src/extract_km.py km_plot.png

# With known time range (0-60 months)
python src/extract_km.py km_plot.png --time-max 60

# Single curve only
python src/extract_km.py km_plot.png --curves 1

# Custom output directory
python src/extract_km.py km_plot.png -o my_results/
```

---

## Project Structure

```
surapp_pwise/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── requirements.ai.txt     # AI-specific dependencies
├── .dockerignore           # Docker build exclusions
│
├── src/                    # Source scripts
│   ├── surapp.sh           # Unified entry point (Linux/Mac)
│   ├── surapp.bat          # Unified entry point (Windows)
│   ├── extract_km.py       # Main extraction script (all-in-one)
│   ├── extract_km_ai.py    # AI-enhanced extraction script
│   ├── document_extraction_ai.py  # Step-by-step documentation with AI
│   ├── step1_preview_image.py  # Step 1: Preview image
│   ├── step2_calibrate_axes.py # Step 2: Calibrate axes
│   └── step3_extract_curves.py # Step 3: Extract curves
│
├── lib/                    # Core detection modules
│   ├── __init__.py
│   ├── detector.py         # Curve detection (solid/dashed)
│   ├── calibrator.py       # Axis calibration
│   ├── color_detector.py   # Color-based curve separation
│   ├── ai_validator.py     # AI validation logic
│   └── ai_config.py        # AI configuration
│
├── input/                  # Input images folder
│
├── docs/                   # Documentation
│   ├── surapp_code.md      # Technical code documentation
│   ├── surapp_learn.md     # Educational guide
│   ├── surapp_plan.md      # Development plan
│   ├── surapp_docker.md    # Docker implementation guide
│   └── surapp_ai.md        # AI-enhanced extraction guide
│
├── docker/                 # Docker configuration
│   ├── Dockerfile          # Standard Docker image
│   ├── Dockerfile.ai       # AI-enhanced Docker image
│   ├── docker-compose.yml  # Docker Compose configuration
│   ├── docker-compose.ai.yml # AI services configuration
│   ├── run.sh              # Docker helper (Linux/Mac)
│   ├── run.bat             # Docker helper (Windows)
│   ├── run-ai.sh           # AI Docker helper (Linux/Mac)
│   └── run-ai.bat          # AI Docker helper (Windows)
│
└── results/                # Output directory (auto-created)
```

---

## Tips for Best Results

### Image Quality
- Use high-resolution images (at least 500x400 pixels)
- Ensure curves are clearly visible against background
- Avoid heavily compressed images (JPEG artifacts)

### Troubleshooting

**No curves detected:**
- Check `debug_binary.png` to see preprocessed image
- Ensure curves are darker than background
- Try with different `--curves` value

**Wrong time range:**
- Use `--time-max` to manually specify maximum time
- Check `step2_calibration.txt` for auto-detected values

**Curves merged together:**
- Increase image resolution if possible
- Try `--curves 1` if curves heavily overlap

**Docker issues:**
- See [docs/surapp_docker.md](docs/surapp_docker.md) for troubleshooting

---

## AI-Enhanced Extraction (Optional)

SURAPP can use AI (llama3.2-vision via Ollama) to validate extraction results:

```bash
# Start AI services (first time downloads ~4GB model)
./docker/run-ai.sh --start

# Run extraction with AI validation
./docker/run-ai.sh my_km_plot.png --validate

# Check AI service status
./docker/run-ai.sh --status
```

The AI compares the original image with extracted curves and:
- Validates extraction accuracy
- Identifies potential issues
- Provides confidence scores

**Requirements:** Docker + ~4GB disk space for the model

For detailed AI setup, see [docs/surapp_ai.md](docs/surapp_ai.md).

---

## AI-Assisted Extraction with Step-by-Step Documentation

Generate detailed step-by-step documentation of the extraction process, producing visualizations at each stage and CSV files with the extracted curve data.

### Quick Start (Recommended)

```bash
# Fast mode - recommended for most users
python -m src.document_extraction_ai -s input/your_image.png -r results/output_folder --fast

# With verbose output to see extraction details
python -m src.document_extraction_ai -s input/your_image.png -r results/output_folder --fast -v
```

### Prerequisites

1. **Calibration file**: The results folder must contain a `calibration.json` file with axis boundaries:
   ```json
   {
     "x_0_pixel": 88,
     "x_max_pixel": 421,
     "y_0_pixel": 237,
     "y_100_pixel": 2,
     "time_max": 18.0
   }
   ```

2. **Generate calibration** (if not available):
   ```bash
   python src/step2_calibrate_axes.py input/your_image.png --time-max 18
   ```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-s, --source` | Source image path (required) |
| `-r, --results` | Results output directory (required) |
| `-c, --calibration` | Path to calibration JSON file (optional, auto-detects from results folder) |
| `--fast`, `-f` | Fast mode - skips AI, uses optimized defaults (recommended) |
| `--no-ai` | Disable AI refinement |
| `-m, --max-iterations` | Maximum AI refinement iterations (default: 2) |
| `-v, --verbose` | Show detailed progress output |

### Output Files

The extraction generates 8 steps plus CSV files:

| Step | File | Description |
|------|------|-------------|
| 1 | `step1_original.png` | Original source image |
| 2 | `step2_with_axes_labels.png` | Image with calibrated axis boundaries |
| 3 | `step3_plot_area.png` | Cropped plot area (curves + grid lines) |
| 4 | `step4_curves_only.png` | Extracted curves (raw detection) |
| 5 | `step5_skeleton.png` | Skeleton curves (1-pixel width, gaps filled) |
| 6 | `step6_final_points.png` | Final extracted points (clean, monotonic) |
| 7 | `step7_overlay.png` | Original with extracted curves overlaid |
| 8 | CSV files | Curve data files |

**CSV Files:**

| File | Description |
|------|-------------|
| `curve_<color>.csv` | Individual curve data (e.g., `curve_cyan.csv`, `curve_purple.csv`) |
| `all_curves.csv` | Combined data from all detected curves |

### CSV Format

All curves start at (time=0, survival=100%) and maintain monotonic decreasing survival:

```csv
Time,Survival
0.0,1.0
0.81,0.928
1.08,0.905
...
17.89,0.311
```

### Example Workflow

```bash
# 1. Place your KM plot image in the input folder
cp my_km_plot.png input/

# 2. Create calibration (adjust time-max to match your plot's x-axis)
python src/step2_calibrate_axes.py input/my_km_plot.png --time-max 24

# 3. Run the extraction
python -m src.document_extraction_ai -s input/my_km_plot.png -r results/my_extraction --fast -v

# 4. Check the results
ls results/my_extraction/
# Output: step1_original.png, step2_with_axes_labels.png, ..., curve_*.csv, all_curves.csv
```

### AI Refinement (Optional)

For AI-assisted boundary refinement (requires Ollama with llama3.2-vision):

```bash
# Start Ollama
ollama serve

# Pull the vision model (first time only, ~4GB)
ollama pull llama3.2-vision

# Run with AI refinement
python -m src.document_extraction_ai -s input/your_image.png -r results/output_folder -v
```

**Note:** AI processing on CPU can be slow. Use `--fast` mode for most cases.

---

## Requirements

### For Native Python Installation
- Python 3.10+
- opencv-python
- numpy
- pandas
- matplotlib
- pillow

### For Docker Installation
- Docker (Docker Desktop on Windows/Mac)

---

## Documentation

- [Technical Code Documentation](docs/surapp_code.md) - Detailed explanation of each module
- [Educational Guide](docs/surapp_learn.md) - Learn image processing concepts
- [Development Plan](docs/surapp_plan.md) - How the tool was built
- [Docker Guide](docs/surapp_docker.md) - Docker implementation details
- [AI Guide](docs/surapp_ai.md) - AI-enhanced extraction with llama3.2-vision

---

## License

MIT License
