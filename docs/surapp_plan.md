# SURAPP Development Plan

## Project Goal

Build a tool that extracts numerical survival data from Kaplan-Meier curve images, converting visual plots into CSV files suitable for statistical analysis.

---

## Phase 1: Project Setup and Image Loading

### Objective
Establish the foundation for image processing.

### Functions to Build

#### `load_image(path) -> np.ndarray`
Load an image file using OpenCV.

```python
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img
```

#### `is_grayscale_image(img) -> bool`
Determine if the image is grayscale or color.

```python
def is_grayscale_image(img):
    if len(img.shape) == 2:
        return True
    b, g, r = cv2.split(img)
    return np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)
```

#### `is_color_image(img) -> bool`
Check if image has colored curves (more reliable than just checking channels).

```python
def is_color_image(img, threshold=0.005):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = (saturation > 50) & (value > 30) & (value < 250)
    ratio = np.sum(colored) / (img.shape[0] * img.shape[1])
    return ratio > threshold
```

### Performance Tip
Convert to grayscale once at load time and cache it. Many operations need grayscale, so avoid repeated conversions.

---

## Phase 2: Axis Detection and Calibration

### Objective
Find the plot area boundaries and establish the coordinate mapping between pixels and data values.

### Functions to Build

#### `detect_axis_lines(gray_img) -> (horizontal_lines, vertical_lines)`
Use Canny edge detection and Hough transform to find potential axis lines.

```python
def detect_axis_lines(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=50, minLineLength=30, maxLineGap=20
    )

    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if abs(angle) < 10:
            horizontal.append(line)
        elif abs(angle) > 80:
            vertical.append(line)

    return horizontal, vertical
```

#### `find_x_axis(horizontal_lines, img_height, img_width) -> AxisLine`
Select the X-axis from candidates (bottom-most long horizontal line).

**Selection criteria:**
- Located in bottom 60% of image
- Spans at least 30% of image width
- Score by: length × position (prefer longer lines closer to bottom)

#### `find_y_axis(vertical_lines, img_height, img_width) -> AxisLine`
Select the Y-axis from candidates (left-most long vertical line).

**Selection criteria:**
- Located in left 40% of image
- Spans at least 50% of image height
- Score by: length × position (prefer longer lines closer to left)

#### `find_origin(x_axis, y_axis) -> (x, y)`
Calculate the intersection point of the two axes.

#### `detect_data_ranges(img, x_axis, y_axis) -> ((x_min, x_max), (y_min, y_max))`
Analyze tick labels to determine data ranges. For KM curves:
- Y-axis is typically 0 to 1.0 (or 0% to 100%)
- X-axis varies (time in months, years, etc.)

**Fallback:** If OCR fails, use defaults (0-1 for Y, estimate X from axis length).

#### `create_pixel_to_coord(plot_bounds, data_ranges) -> function`
Return a function that converts pixel coordinates to data coordinates.

```python
def create_pixel_to_coord(plot_bounds, data_ranges):
    px, py, pw, ph = plot_bounds
    (x_min, x_max), (y_min, y_max) = data_ranges

    def pixel_to_coord(pixel_x, pixel_y):
        time = x_min + (pixel_x - px) / pw * (x_max - x_min)
        survival = y_max - (pixel_y - py) / ph * (y_max - y_min)
        return time, survival

    return pixel_to_coord
```

### Performance Tip
Cache the calibration result. Don't recalculate for every curve - calibrate once, use many times.

---

## Phase 3: Curve Pixel Extraction

### Objective
Extract the pixels that belong to curves (not background, axes, or grid lines).

### Functions to Build

#### `extract_curve_pixels(gray_img, plot_bounds) -> binary_mask`
Use adaptive thresholding to find dark pixels (curves).

```python
def extract_curve_pixels(gray_img, plot_bounds):
    px, py, pw, ph = plot_bounds
    plot_region = gray_img[py:py+ph, px:px+pw]

    binary = cv2.adaptiveThreshold(
        plot_region, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=10
    )

    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary
```

#### `filter_reference_lines(binary_mask) -> cleaned_mask`
Remove horizontal and vertical grid/reference lines that would interfere with curve detection.

**Strategy:**
1. Detect long horizontal runs of pixels (grid lines)
2. Detect long vertical runs of pixels (grid lines)
3. Remove these while preserving curve pixels at intersections

```python
def filter_reference_lines(binary):
    # Horizontal line detection
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Vertical line detection
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Remove lines but preserve intersections with curves
    lines_mask = cv2.bitwise_or(h_lines, v_lines)
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(lines_mask))

    return cleaned
```

### Performance Tip
Use morphological operations instead of pixel-by-pixel loops. OpenCV's morphological functions are highly optimized in C++.

---

## Phase 4: Curve Detection Strategy Selection

### Objective
Choose the appropriate detection method based on image characteristics.

### Decision Tree

```
Is the image colored?
├── YES → Use Color-Based Detection (Phase 5A)
└── NO  → Use Line-Style Detection (Phase 5B)
```

### Function to Build

#### `select_detection_strategy(img) -> str`

```python
def select_detection_strategy(img):
    if is_color_image(img):
        return "color"
    else:
        return "line_style"
```

---

## Phase 5A: Color-Based Curve Detection

### Objective
Separate curves by their color (cyan, magenta, orange, etc.).

### Functions to Build

#### `detect_curve_colors(img, max_colors=4) -> List[ColorInfo]`
Find the dominant colors in the image using hue histogram analysis.

```python
def detect_curve_colors(img, max_colors=4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask for colored pixels
    sat, val = hsv[:,:,1], hsv[:,:,2]
    color_mask = (sat > 40) & (val > 40) & (val < 250)

    # Histogram of hues
    hues = hsv[color_mask][:, 0]
    hist, bins = np.histogram(hues, bins=36, range=(0, 180))

    # Find and merge peaks
    peaks = find_histogram_peaks(hist, bins)
    merged = merge_nearby_peaks(peaks, threshold=15)

    return merged[:max_colors]
```

#### `hue_to_color_name(hue) -> str`
Map hue values to human-readable color names.

#### `create_color_mask(img, color_info) -> binary_mask`
Create a mask for pixels matching a specific color.

```python
def create_color_mask(img, color_info):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_lower = np.array([color_info.hue - 12, 15, 40])
    hsv_upper = np.array([color_info.hue + 12, 255, 255])

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Clean up
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
```

#### `extract_curve_from_mask(mask, plot_bounds) -> List[(x, y)]`
Extract curve points from a color mask.

```python
def extract_curve_from_mask(mask, plot_bounds):
    px, py, pw, ph = plot_bounds
    points = []

    for x in range(pw):
        col = mask[py:py+ph, px+x]
        y_positions = np.where(col > 0)[0]

        if len(y_positions) > 0:
            y = int(np.median(y_positions))  # Use median for thick lines
            points.append((px + x, py + y))

    return points
```

#### `handle_curve_overlaps(curves_data, colors) -> List[CurvePoints]`
Handle regions where curves cross or touch.

**Strategy:**
1. Only assume TRUE overlap when curves are within 3 pixels
2. At overlap points, assign the shared position to both curves
3. Do NOT extrapolate - if a curve isn't detected, it ends there

### Performance Tip
Process all colors in a single pass through the HSV image rather than converting multiple times.

---

## Phase 5B: Line-Style Based Curve Detection

### Objective
Separate curves by their visual pattern (solid, dashed, dotted).

### Functions to Build

#### `create_dash_mask(binary) -> mask`
Identify gaps in curves to distinguish dashed from solid lines.

```python
def create_dash_mask(binary):
    # Dilate to fill small gaps
    kernel = np.ones((1, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # XOR to find only the gaps
    dash_gaps = cv2.bitwise_xor(dilated, binary)

    return dash_gaps
```

#### `trace_curves_by_position(binary, expected_count) -> List[TracedCurve]`
Follow curves across the image by tracking Y positions at each X.

**Algorithm:**
1. For each X column, find all Y positions with curve pixels
2. Use Hungarian algorithm to assign Y positions to curve trackers
3. Handle gaps by allowing trackers to "coast" for a few pixels
4. Detect and handle curve crossings

#### `analyze_line_style(curve_pixels, dash_mask) -> LineStyle`
Classify the line style based on gap patterns.

```python
def analyze_line_style(curve_pixels, dash_mask):
    # Count gaps along the curve
    gap_count = 0
    segment_lengths = []
    current_segment = 0

    for x, y in curve_pixels:
        if dash_mask[y, x] > 0:  # Gap detected
            if current_segment > 0:
                segment_lengths.append(current_segment)
            current_segment = 0
            gap_count += 1
        else:
            current_segment += 1

    # Classify based on gap ratio
    gap_ratio = gap_count / len(curve_pixels)

    if gap_ratio < 0.05:
        return LineStyle.SOLID
    elif gap_ratio < 0.20:
        return LineStyle.DASHED
    else:
        return LineStyle.DOTTED
```

#### `separate_curves_by_style(traced_curves) -> Dict[LineStyle, CurveData]`
Group traced curves by their detected line style.

### Performance Tip
Use NumPy vectorized operations for column analysis instead of Python loops where possible.

---

## Phase 6: Coordinate Conversion

### Objective
Convert pixel coordinates to data coordinates (time, survival).

### Functions to Build

#### `pixels_to_data(pixel_points, pixel_to_coord) -> List[(time, survival)]`

```python
def pixels_to_data(pixel_points, pixel_to_coord):
    data_points = []
    for px, py in pixel_points:
        time, survival = pixel_to_coord(px, py)
        # Clamp survival to valid range
        survival = max(0.0, min(1.0, survival))
        data_points.append((time, survival))
    return data_points
```

#### `refine_calibration_from_curves(calibrator, curves) -> refined_bounds`
Use curve positions to improve calibration accuracy.

```python
def refine_calibration_from_curves(calibrator, curves):
    # KM curves start at time=0, survival=1.0
    # Find topmost curve pixels (survival=100%)
    all_y = [y for curve in curves for x, y in curve.pixels]
    y_100 = min(all_y)  # Topmost = highest survival

    # Find leftmost curve pixels (time=0)
    all_x = [x for curve in curves for x, y in curve.pixels]
    x_0 = min(all_x)  # Leftmost = time 0

    # Adjust bounds
    return (x_0, y_100, original_width, original_height - y_100)
```

---

## Phase 7: Data Cleaning and Validation

### Objective
Ensure extracted data conforms to KM curve mathematical properties.

### Functions to Build

#### `clean_curve_data(points) -> cleaned_points`
Apply all cleaning steps in sequence.

```python
def clean_curve_data(points):
    points = sort_by_time(points)
    points = clip_negative_times(points)
    points = deduplicate_times(points)
    points = rescale_if_needed(points)
    points = ensure_starts_at_origin(points)
    points = enforce_monotonicity(points)
    points = remove_near_duplicates(points)
    return points
```

#### `sort_by_time(points) -> sorted_points`
Sort points by time value (required for all other operations).

#### `clip_negative_times(points) -> clipped_points`
Ensure no points have negative time values.

```python
def clip_negative_times(points):
    return [(max(0.0, t), s) for t, s in points]
```

#### `deduplicate_times(points) -> deduplicated_points`
Merge points with the same time value using median survival.

#### `rescale_if_needed(points) -> rescaled_points`
If max survival is significantly below 1.0, rescale all values.

```python
def rescale_if_needed(points):
    max_survival = max(s for t, s in points)

    if max_survival < 0.95 and max_survival > 0.1:
        scale = 1.0 / max_survival
        return [(t, min(1.0, s * scale)) for t, s in points]

    return points
```

#### `ensure_starts_at_origin(points) -> fixed_points`
Ensure the curve starts at (0, 1.0).

```python
def ensure_starts_at_origin(points):
    first_t, first_s = points[0]

    if first_t > 0.01:
        points.insert(0, (0.0, 1.0))
    elif first_s < 0.999:
        points[0] = (0.0, 1.0)

    return points
```

#### `enforce_monotonicity(points) -> monotonic_points`
Ensure survival never increases (can only decrease or stay flat).

```python
def enforce_monotonicity(points):
    result = []
    max_survival = 1.0

    for t, s in points:
        s = min(s, max_survival)
        result.append((t, s))
        max_survival = s

    return result
```

### Performance Tip
Process all cleaning steps in a single pass where possible to avoid multiple list iterations.

---

## Phase 8: Export and Visualization

### Objective
Save extracted data and create visualizations for verification.

### Functions to Build

#### `export_to_csv(curves, output_dir)`
Save each curve to individual CSV and combined CSV.

```python
def export_to_csv(curves, output_dir):
    # Individual files
    for curve in curves:
        df = pd.DataFrame(curve.points, columns=['Time', 'Survival'])
        df.to_csv(output_dir / f"curve_{curve.name}.csv", index=False)

    # Combined file
    rows = []
    for curve in curves:
        for t, s in curve.points:
            rows.append({'Curve': curve.name, 'Time': t, 'Survival': s})

    pd.DataFrame(rows).to_csv(output_dir / "all_curves.csv", index=False)
```

#### `plot_extracted_curves(curves, output_path)`
Create a matplotlib visualization of extracted curves.

```python
def plot_extracted_curves(curves, output_path):
    plt.figure(figsize=(10, 6))

    for curve in curves:
        times = [p[0] for p in curve.points]
        survivals = [p[1] for p in curve.points]
        plt.step(times, survivals, where='post', label=curve.name)

    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    plt.savefig(output_path, dpi=150)
    plt.close()
```

#### `plot_comparison_overlay(original_img, curves, plot_bounds, output_path)`
Overlay extracted curves on the original image for verification.

---

## Implementation Order

For best results, implement in this order:

### Stage 1: Minimal Viable Product
1. Image loading (`load_image`)
2. Basic calibration (`detect_axis_lines`, `find_x_axis`, `find_y_axis`)
3. Pixel extraction (`extract_curve_pixels`)
4. Single curve tracing (no separation)
5. Basic export (`export_to_csv`)

### Stage 2: Multi-Curve Support
6. Color detection (`is_color_image`, `detect_curve_colors`)
7. Color-based separation (`create_color_mask`, `extract_curve_from_mask`)
8. Line-style detection (`create_dash_mask`, `analyze_line_style`)
9. Curve tracing with separation (`trace_curves_by_position`)

### Stage 3: Accuracy Improvements
10. Reference line filtering (`filter_reference_lines`)
11. Overlap handling (`handle_curve_overlaps`)
12. Calibration refinement (`refine_calibration_from_curves`)
13. Full data cleaning pipeline

### Stage 4: Polish
14. Visualization (`plot_extracted_curves`, `plot_comparison_overlay`)
15. Debug output options
16. Error handling and user feedback

---

## Critical Functions for Performance

These functions have the most impact on extraction accuracy:

| Function | Impact | Why Critical |
|----------|--------|--------------|
| `create_pixel_to_coord` | High | Incorrect mapping = wrong data values |
| `extract_curve_pixels` | High | Missing pixels = incomplete curves |
| `handle_curve_overlaps` | High | Wrong handling = merged/crossed curves |
| `enforce_monotonicity` | Medium | Violations indicate extraction errors |
| `filter_reference_lines` | Medium | Grid lines confuse curve tracing |
| `refine_calibration_from_curves` | Medium | Improves accuracy by 5-10% |

---

## Testing Strategy

### Unit Tests
- Test each cleaning function with known inputs
- Test color detection with synthetic colored images
- Test line detection with simple geometric shapes

### Integration Tests
- Test full pipeline on reference images with known data
- Compare extracted values against manually digitized values
- Acceptable error: ±2% for survival values, ±1% for time values

### Visual Tests
- Generate comparison overlays for every extraction
- Human review of overlay accuracy
- Flag extractions where curves don't match original

---

## Common Pitfalls to Avoid

1. **Don't extrapolate missing data** - Only output what's actually detected
2. **Don't assume axis ranges** - Always try to detect, use defaults only as fallback
3. **Don't over-process the image** - Minimal morphological operations preserve curve details
4. **Don't ignore curve crossings** - They're common in KM plots and must be handled
5. **Don't forget Y-axis inversion** - Pixel Y increases downward, survival increases upward
