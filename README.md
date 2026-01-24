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

## Installation Options

SURAPP can be installed in two ways:

| Method | Best For | Setup Time |
|--------|----------|------------|
| **Docker** | Guaranteed compatibility, no Python setup | ~5 min first run |
| **Native Python** | Faster startup, development, customization | ~2 min |

---

## Option 1: Docker Installation (Recommended for most users)

Docker ensures SURAPP works identically on any system, with all dependencies pre-configured.

### Prerequisites

Install Docker:
- **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux:** `sudo apt install docker.io` (Ubuntu/Debian)

### Quick Start with Docker

**Linux/Mac:**
```bash
# Make script executable (first time only)
chmod +x docker-run.sh

# Run extraction
./docker-run.sh your_km_plot.png

# With options
./docker-run.sh your_km_plot.png --time-max 36 --curves 2
```

**Windows:**
```batch
docker-run.bat your_km_plot.png
docker-run.bat your_km_plot.png --time-max 36 --curves 2
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
python extract_km.py
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
python extract_km.py your_km_plot.png --time-max 40 --curves 2
```

Results are saved to `results/<image_name>_<timestamp>/`.

---

## Step-by-Step Extraction

For more control over the process, use the sequential scripts.
All scripts support interactive image selection (just run without arguments).

### Step 1: Preview Image

```bash
python step1_preview_image.py              # Interactive
python step1_preview_image.py image.png    # Direct
```

Shows image dimensions, color type, and basic properties.

### Step 2: Calibrate Axes

```bash
python step2_calibrate_axes.py                        # Interactive
python step2_calibrate_axes.py image.png --time-max 50
```

Detects plot area boundaries and axis ranges.

### Step 3: Extract Curves

```bash
python step3_extract_curves.py                                  # Interactive
python step3_extract_curves.py image.png --time-max 40 --curves 2
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
python extract_km.py

# Direct mode - specify image
python extract_km.py km_plot.png

# With known time range (0-60 months)
python extract_km.py km_plot.png --time-max 60

# Single curve only
python extract_km.py km_plot.png --curves 1

# Custom output directory
python extract_km.py km_plot.png -o my_results/
```

---

## Project Structure

```
surapp_pwise/
├── extract_km.py           # Main extraction script (all-in-one)
├── step1_preview_image.py  # Step 1: Preview image
├── step2_calibrate_axes.py # Step 2: Calibrate axes
├── step3_extract_curves.py # Step 3: Extract curves
├── README.md               # This file
├── requirements.txt        # Python dependencies
│
├── lib/                    # Core detection modules
│   ├── __init__.py
│   ├── detector.py         # Curve detection (solid/dashed)
│   ├── calibrator.py       # Axis calibration
│   └── color_detector.py   # Color-based curve separation
│
├── docs/                   # Documentation
│   ├── surapp_code.md      # Technical code documentation
│   ├── surapp_learn.md     # Educational guide
│   ├── surapp_plan.md      # Development plan
│   └── surapp_docker.md    # Docker implementation guide
│
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── docker-run.sh           # Docker helper script (Linux/Mac)
├── docker-run.bat          # Docker helper script (Windows)
├── .dockerignore           # Docker build exclusions
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

---

## License

MIT License
