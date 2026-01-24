# SURAPP Code Documentation

## Overview

SURAPP (Survival Analysis Plot Processor) is a Python-based tool for extracting numerical data from Kaplan-Meier survival curve images. It converts visual plot images into CSV data files that can be used for statistical analysis.

The codebase is organized into two main areas:
- **Root folder**: User-facing scripts for running the extraction pipeline
- **lib folder**: Core library modules with the detection and calibration algorithms

---

## Root Folder Files

### `extract_km.py`

**Purpose**: The main all-in-one script for extracting Kaplan-Meier curves from images.

**Rationale**: Users need a single entry point that handles the complete extraction workflow without requiring knowledge of the underlying modules. This script orchestrates all the steps: loading, calibration, detection, extraction, and export.

**How it achieves its purpose**:
1. **Automatic image type detection**: Uses `is_color_image()` to determine whether to use color-based or line-style-based detection
2. **Axis calibration**: Invokes `AxisCalibrator` to detect plot boundaries and coordinate ranges
3. **Curve detection**: Uses either `ColorCurveDetector` or `LineStyleDetector` based on image type
4. **Data cleaning**: Applies `_clean_curve_data()` to enforce KM curve properties:
   - Curves start at (0, 1.0) - all subjects alive at time 0
   - Survival values are monotonically decreasing
   - Duplicate times are merged using median survival
5. **Export**: Generates CSV files and visualization plots

**Key functions**:
- `extract_km_curves()`: Main extraction function
- `_clean_curve_data()`: Enforces KM curve mathematical properties
- `_plot_curves()`: Creates matplotlib visualization
- `_plot_comparison_overlay()`: Overlays extracted curves on original image

---

### `step1_preview_image.py`

**Purpose**: Preview an image and display its basic properties before extraction.

**Rationale**: Before running extraction, users need to verify they have the correct image and understand its characteristics (size, color/grayscale, aspect ratio). This helps identify multi-panel images that may need cropping.

**How it achieves its purpose**:
1. Loads the image using OpenCV
2. Analyzes image dimensions and channel count
3. Detects whether the image is grayscale or color using saturation analysis
4. Calculates aspect ratio to identify potential multi-panel images
5. Saves an annotated preview showing image boundaries

---

### `step2_calibrate_axes.py`

**Purpose**: Detect and calibrate the plot axes to establish the coordinate system.

**Rationale**: Before extracting curves, we need to know where the plot area is and what data ranges the axes represent. This step isolates the calibration logic for debugging and verification.

**How it achieves its purpose**:
1. Uses `AxisCalibrator` to detect X and Y axis lines via Hough transform
2. Determines the plot rectangle (bounding box of the data area)
3. Calculates pixel-to-coordinate conversion factors
4. Saves a visualization showing detected axes and plot bounds
5. Exports calibration data to a text file

---

### `step3_extract_curves.py`

**Purpose**: Detect and extract curve data after calibration is complete.

**Rationale**: Separating curve extraction from calibration allows users to verify axes are correct before proceeding. This modular approach aids debugging.

**How it achieves its purpose**:
1. Loads calibration from step 2 or performs auto-calibration
2. Uses `LineStyleDetector` to find and trace curves
3. Converts pixel coordinates to data coordinates using calibration
4. Applies data cleaning (monotonicity, rescaling)
5. Exports individual and combined CSV files

---

## Library Folder (`lib/`)

### `lib/__init__.py`

**Purpose**: Package initialization and public API definition.

**Rationale**: Provides a clean import interface so users can write `from lib import AxisCalibrator` instead of `from lib.calibrator import AxisCalibrator`.

**How it achieves its purpose**:
- Imports key classes from submodules
- Defines `__all__` to specify the public API
- Groups exports by functionality (grayscale detection, color detection, calibration)

---

### `lib/calibrator.py`

**Purpose**: Detect axis lines and provide coordinate calibration for pixel-to-data conversion.

**Rationale**: Accurate data extraction requires knowing the exact mapping between pixel positions and data values. The axes define this mapping, so detecting them precisely is critical.

**How it achieves its purpose**:

1. **Edge Detection**: Uses Canny edge detection to find edges in the image
   ```python
   edges = cv2.Canny(blurred, 30, 100)
   ```

2. **Line Detection**: Uses Hough Line Transform to find straight lines
   ```python
   lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, ...)
   ```

3. **Axis Identification**:
   - X-axis: Horizontal line in bottom 60% of image, spanning >30% width
   - Y-axis: Vertical line in left 40% of image, spanning >50% height

4. **Origin Detection**: Finds intersection point of X and Y axes

5. **Data Range Detection**: Analyzes tick labels using OCR or estimates based on typical KM plot ranges (0-1 for survival, 0-N for time)

6. **Curve-based Refinement**: `refine_plot_bounds_from_curves()` uses actual curve positions to fine-tune boundaries:
   - `detect_y100_from_curves()`: Finds topmost curve pixels (survival=100%)
   - `detect_x0_from_curves()`: Finds leftmost curve pixels (time=0)

**Key classes**:
- `DetectedAxisLine`: Represents a detected axis with endpoints and confidence
- `AxisCalibrationResult`: Complete calibration data including plot bounds and stretching factors
- `AxisCalibrator`: Main calibration class

---

### `lib/detector.py`

**Purpose**: Detect and separate multiple curves in grayscale images based on line style (solid, dashed, dotted).

**Rationale**: Many published KM plots use grayscale with different line styles to distinguish treatment groups. Color-based detection won't work for these images, so we need pattern-based detection.

**How it achieves its purpose**:

1. **Pixel Extraction**: Adaptive thresholding to extract dark curve pixels
   ```python
   binary = cv2.adaptiveThreshold(plot_region, 255,
       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
       blockSize=15, C=10)
   ```

2. **Reference Line Filtering**: Removes horizontal/vertical grid lines that would interfere with curve detection

3. **Dash Mask Creation**: Identifies gaps in curves to distinguish dashed from solid lines
   ```python
   # Dilate to fill small gaps, then XOR to find only the gaps
   dilated = cv2.dilate(binary, kernel)
   dash_gaps = cv2.bitwise_xor(dilated, binary)
   ```

4. **Curve Tracing**: Follows curves across X positions by tracking Y positions
   - For each X column, finds curve pixels
   - Uses Hungarian algorithm for optimal assignment when multiple curves exist
   - Handles curve crossings by swapping assignments

5. **Style Classification**: Analyzes pixel patterns to classify line style
   - Solid: Continuous pixels with few gaps
   - Dashed: Regular gaps of 5-15 pixels
   - Dotted: Very short segments with frequent gaps

**Key classes**:
- `LineStyle`: Enum of detectable styles (SOLID, DASHED, DOTTED, DASH_DOT)
- `DetectedCurve`: Complete curve with segments, style, and data points
- `LineStyleDetector`: Main detection class

---

### `lib/color_detector.py`

**Purpose**: Detect and separate multiple curves based on color (cyan, magenta, orange, etc.).

**Rationale**: Modern digital KM plots often use color to distinguish curves. Color-based detection is more reliable than line-style detection when colors are distinct.

**How it achieves its purpose**:

1. **Color Space Conversion**: Converts BGR to HSV for easier color analysis
   ```python
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   ```

2. **Color Detection**:
   - Identifies colored pixels (saturation > threshold, not too dark/bright)
   - Builds histogram of hue values
   - Finds peaks representing dominant colors

3. **Color Classification**: Maps hue values to color names
   ```python
   def hue_to_name(h):
       if h < 8 or h >= 165: return 'red'
       elif h < 25: return 'orange'
       elif h < 100: return 'cyan'
       # etc.
   ```

4. **Mask Creation**: Creates binary mask for each detected color
   ```python
   mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
   ```

5. **Overlap Handling**: `extract_curves_with_overlap_handling()` addresses cases where curves cross or touch:
   - Detects when curves are within 3 pixels (true overlap)
   - Assigns shared pixels to both curves at overlap points
   - Does NOT extrapolate - only uses actually detected pixels

6. **Point Extraction**: For each X position, finds the median Y of colored pixels

**Key classes**:
- `ColorCurve`: Represents a curve with name, RGB color, mask, and points
- `ColorCurveDetector`: Main detection class with overlap-aware extraction

**Key functions**:
- `is_color_image()`: Determines if image has colored curves
- `detect_curve_colors()`: Finds dominant colors in the image
- `extract_curves_with_overlap_handling()`: Extracts curves with crossing support

---

## Data Flow

```
Image File
    │
    ▼
┌─────────────────────┐
│  extract_km.py      │  (or step1 → step2 → step3)
└─────────────────────┘
    │
    ├──► AxisCalibrator (lib/calibrator.py)
    │        │
    │        └──► Plot bounds, coordinate mapping
    │
    ├──► is_color_image() (lib/color_detector.py)
    │        │
    │        ├──► ColorCurveDetector (if color)
    │        │
    │        └──► LineStyleDetector (if grayscale)
    │
    ▼
┌─────────────────────┐
│  Raw pixel points   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  _clean_curve_data  │  Enforce KM properties
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  CSV files          │  all_curves.csv, curve_*.csv
│  Visualizations     │  extracted_curves.png, comparison_overlay.png
└─────────────────────┘
```

---

## Key Design Decisions

1. **Dual Detection Modes**: Supporting both color and grayscale images covers the majority of published KM plots.

2. **Conservative Overlap Handling**: Only assuming true overlap when curves are within 3 pixels prevents incorrect curve merging.

3. **No Point Extrapolation**: Curves end where detected pixels end, ensuring output reflects actual image data.

4. **Enforced KM Properties**: All curves start at (0, 1.0) and are monotonically decreasing, matching the mathematical definition of survival curves.

5. **Modular Architecture**: Separating calibration, detection, and cleaning allows each component to be tested and debugged independently.
