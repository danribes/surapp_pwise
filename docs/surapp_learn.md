# SURAPP Learning Guide

## Introduction

This guide teaches the concepts and techniques used in SURAPP for extracting data from Kaplan-Meier survival curve images. Each section explains *why* certain approaches were chosen and presents alternatives that were considered.

---

## Lesson 1: Image Color Spaces

### The Problem
We need to distinguish between colored and grayscale KM plots because they require different detection strategies.

### The Solution: HSV Color Space

Images are typically stored in BGR (Blue-Green-Red) format, but this makes color analysis difficult. Instead, we convert to HSV (Hue-Saturation-Value):

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

**Why HSV?**

| Color Space | Pros | Cons |
|------------|------|------|
| **BGR/RGB** | Native format, no conversion | Color detection requires complex conditions |
| **HSV** | Hue directly encodes color, easy to detect "colorfulness" | Requires conversion |
| **LAB** | Perceptually uniform | More complex, overkill for this task |

In HSV:
- **Hue** (0-180 in OpenCV): The color itself (red, blue, green, etc.)
- **Saturation** (0-255): How "colorful" vs "gray" the pixel is
- **Value** (0-255): How bright the pixel is

### Example: Detecting Colored Pixels

```python
def is_color_image(img, threshold=0.005):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]  # Extract saturation channel
    value = hsv[:, :, 2]       # Extract value channel

    # Colored pixels have high saturation and medium brightness
    colored = (saturation > 50) & (value > 30) & (value < 250)
    ratio = np.sum(colored) / (img.shape[0] * img.shape[1])

    return ratio > threshold  # More than 0.5% colored pixels
```

**Why these thresholds?**
- `saturation > 50`: Excludes gray pixels (which have low saturation regardless of RGB values)
- `value > 30`: Excludes very dark pixels (black appears "colored" in noisy images)
- `value < 250`: Excludes very bright pixels (white can have noise in saturation)

### Alternative Considered: RGB Distance

```python
# Alternative: Check if R, G, B channels are similar (grayscale)
def is_grayscale_rgb(img):
    b, g, r = cv2.split(img)
    return np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)
```

This approach checks if all three channels are approximately equal. **We didn't use this** because:
1. It's slower (checks every pixel in three channels)
2. It doesn't distinguish between "slightly tinted grayscale" and "truly colored"

---

## Lesson 2: Edge and Line Detection

### The Problem
We need to find the X and Y axes to know where the plot area is.

### The Solution: Canny + Hough Transform

**Step 1: Canny Edge Detection**

Canny finds edges by detecting rapid intensity changes:

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise first
edges = cv2.Canny(blurred, 30, 100)          # Find edges
```

**Why blur first?** Noise creates false edges. Gaussian blur smooths the image while preserving true edges.

**Why these thresholds (30, 100)?**
- Low threshold (30): Pixels with gradient above this MIGHT be edges
- High threshold (100): Pixels with gradient above this ARE definitely edges
- Pixels between thresholds are edges only if connected to definite edges

**Step 2: Hough Line Transform**

Hough transform finds lines in the edge image:

```python
lines = cv2.HoughLinesP(
    edges,
    rho=1,              # Distance resolution (1 pixel)
    theta=np.pi/180,    # Angle resolution (1 degree)
    threshold=50,       # Minimum votes for a line
    minLineLength=30,   # Minimum line length
    maxLineGap=20       # Maximum gap between line segments
)
```

**Why HoughLinesP instead of HoughLines?**
- `HoughLines`: Returns infinite lines (rho, theta) - need extra processing
- `HoughLinesP`: Returns line segments with endpoints - directly usable

### Example: Classifying Lines as Axes

```python
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    if abs(angle) < 10:           # Nearly horizontal
        horizontal_lines.append(line)
    elif abs(angle) > 80:         # Nearly vertical
        vertical_lines.append(line)
    # Lines at other angles are ignored (not axes)
```

### Alternative Considered: Template Matching

```python
# Alternative: Match against a template of typical axis appearance
result = cv2.matchTemplate(gray, axis_template, cv2.TM_CCOEFF_NORMED)
```

**We didn't use this** because:
1. Axis appearance varies widely (thickness, color, tick marks)
2. Would need many templates for different plot styles
3. Hough transform is more generalizable

---

## Lesson 3: Adaptive Thresholding for Curve Extraction

### The Problem
We need to extract curve pixels from the background, but lighting and paper color vary across the image.

### The Solution: Adaptive Thresholding

```python
binary = cv2.adaptiveThreshold(
    gray,
    255,                          # Output value for foreground
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Method
    cv2.THRESH_BINARY_INV,        # Invert (curves are dark)
    blockSize=15,                 # Neighborhood size
    C=10                          # Constant subtracted from mean
)
```

**Why adaptive instead of global thresholding?**

| Method | How it works | Best for |
|--------|--------------|----------|
| **Global** | Single threshold for entire image | Uniform lighting |
| **Adaptive** | Different threshold for each region | Varying lighting, scanned documents |

**Example of the difference:**

```python
# Global thresholding - fails with uneven lighting
_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive thresholding - handles uneven lighting
adaptive_thresh = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
```

**Why GAUSSIAN_C instead of MEAN_C?**
- `MEAN_C`: Uses simple average of neighborhood
- `GAUSSIAN_C`: Uses weighted average (center pixels matter more)

Gaussian gives smoother results and better handles gradual lighting changes.

**Why blockSize=15?**

The block size should be larger than the curve thickness but smaller than background variations. For typical KM plots:
- Curves are 1-3 pixels thick
- Background variations happen over 50+ pixels
- 15 pixels is a good middle ground

---

## Lesson 4: Morphological Operations

### The Problem
After thresholding, we have noise (isolated pixels) and broken curves (gaps in dashed lines).

### The Solution: Morphological Operations

```python
kernel = np.ones((2, 2), np.uint8)

# Opening: Remove small noise
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing: Fill small gaps
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

**What these operations do:**

| Operation | Definition | Effect |
|-----------|------------|--------|
| **Erosion** | Shrink white regions | Removes small white spots |
| **Dilation** | Expand white regions | Fills small holes |
| **Opening** | Erosion then Dilation | Removes noise while preserving shape |
| **Closing** | Dilation then Erosion | Fills gaps while preserving shape |

### Example: Why Order Matters

```python
# BAD: Closing then Opening can lose thin curves
bad = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
                       cv2.MORPH_OPEN, kernel)

# GOOD: Opening then Closing (noise removal first)
good = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
                        cv2.MORPH_CLOSE, kernel)
```

**Why a 2x2 kernel?**

Larger kernels have stronger effects:
- 2x2: Removes 1-pixel noise, fills 1-pixel gaps (minimal impact on curves)
- 3x3: Removes 2-pixel noise, but may merge nearby curves
- 5x5: Too aggressive, destroys fine details

---

## Lesson 5: Color-Based Curve Separation

### The Problem
In color images, we need to identify which pixels belong to which curve based on their color.

### The Solution: Hue Histogram Analysis

```python
def detect_curve_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Find colored pixels
    color_mask = (hsv[:,:,1] > 40) & (hsv[:,:,2] > 40) & (hsv[:,:,2] < 250)
    colored_pixels = hsv[color_mask]

    # Build histogram of hues
    hues = colored_pixels[:, 0]
    hist, bin_edges = np.histogram(hues, bins=36, range=(0, 180))

    # Find peaks (dominant colors)
    peaks = find_local_maxima(hist)

    return peaks
```

**Why histogram peaks?**

Each curve has a dominant hue. The histogram shows how many pixels exist at each hue value. Peaks correspond to curve colors.

### Example: Hue to Color Name Mapping

```python
def hue_to_name(h):
    # OpenCV uses 0-180 for hue (not 0-360)
    if h < 8 or h >= 165:
        return 'red'      # Red wraps around 0/180
    elif h < 25:
        return 'orange'
    elif h < 35:
        return 'yellow'
    elif h < 85:
        return 'green'
    elif h < 100:
        return 'cyan'
    elif h < 130:
        return 'blue'
    elif h < 150:
        return 'purple'
    else:
        return 'magenta'
```

**Why does red wrap around?**

Hue is circular (like degrees on a compass). Red exists at both ends:
- Hue 0-8: Red-orange
- Hue 165-180: Red-magenta

### Creating Color Masks

```python
# Create mask for a specific color
hsv_lower = np.array([hue - 12, 15, 40])  # Allow some variation
hsv_upper = np.array([hue + 12, 255, 255])
mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
```

**Why saturation minimum of 15 (not 40)?**

Curve tails often fade due to:
1. Image compression artifacts
2. Anti-aliasing at curve edges
3. Lower saturation in printed/scanned images

Using 15 instead of 40 captures these faded pixels without including gray background.

---

## Lesson 6: Handling Curve Overlaps

### The Problem
When two curves cross or run very close together, their pixels may blend or one curve may "hide" the other.

### The Solution: Conservative Overlap Detection

```python
def extract_curves_with_overlap_handling(img, colors, plot_bounds):
    # For each x position
    for x in range(width):
        detected_curves = find_colored_pixels_at_x(x)

        # For curves NOT detected at this x
        for curve_idx in undetected_curves:
            last_x, last_y = curve_points[curve_idx][-1]
            gap = x - last_x

            # Only assume overlap if gap is tiny AND curves are touching
            if gap <= 3:
                for detected_y in detected_curves:
                    if abs(detected_y - last_y) <= 3:
                        # TRUE overlap - assign this point to both curves
                        curve_points[curve_idx].append((x, detected_y))
                        break
            # If not overlapping, curve simply ends here (no extrapolation)
```

**Why 3 pixels for overlap detection?**

| Threshold | Effect |
|-----------|--------|
| 1 pixel | Too strict - misses anti-aliased overlaps |
| 3 pixels | Catches real overlaps, avoids false positives |
| 10 pixels | Too loose - merges curves that are just "close" |

### What We Don't Do: Extrapolation

```python
# BAD: Extrapolating missing points
slope = (y2 - y1) / (x2 - x1)
extrapolated_y = last_y + slope * gap
curve_points.append((x, extrapolated_y))  # INVENTED DATA!
```

**We don't extrapolate because:**
1. It invents data that doesn't exist in the image
2. The user expects output to reflect actual image content
3. A curve "ending early" is useful information (may indicate image quality issues)

---

## Lesson 7: Enforcing Kaplan-Meier Properties

### The Problem
Extracted data may have noise, duplicates, or violate KM curve mathematical properties.

### The Solution: Data Cleaning Pipeline

**Property 1: Curves start at (0, 1.0)**

All subjects are alive at time 0. If our extraction misses this:

```python
first_t, first_s = points[0]
if first_t > 0.01:
    points.insert(0, (0.0, 1.0))  # Add starting point
elif first_s < 0.999:
    points[0] = (0.0, 1.0)        # Fix incorrect starting survival
```

**Property 2: Survival is monotonically decreasing**

Subjects can die but not come back to life:

```python
monotonic = []
max_survival = 1.0
for t, s in points:
    s = min(s, max_survival)  # Can't exceed previous maximum
    monotonic.append((t, s))
    max_survival = s          # Update maximum for next iteration
```

**Property 3: Handle duplicate times**

Multiple pixels may map to the same time value:

```python
# Group by time, take median survival
time_groups = {}
for t, s in points:
    t_rounded = round(t, 3)
    if t_rounded not in time_groups:
        time_groups[t_rounded] = []
    time_groups[t_rounded].append(s)

deduplicated = []
for t in sorted(time_groups.keys()):
    survivals = time_groups[t]
    median_s = sorted(survivals)[len(survivals) // 2]
    deduplicated.append((t, median_s))
```

**Why median instead of mean?**

Median is more robust to outliers. If one pixel is misclassified, mean would be affected but median would not.

---

## Lesson 8: Coordinate System Calibration

### The Problem
We have pixel coordinates but need data coordinates (time, survival).

### The Solution: Linear Mapping

```python
def pixel_to_coord(px_x, px_y):
    # Plot bounds: (plot_x, plot_y, plot_width, plot_height)
    # Data ranges: time 0 to time_max, survival 0 to 1.0

    # X: left edge = time 0, right edge = time_max
    time = (px_x - plot_x) / plot_width * time_max

    # Y: top edge = survival 1.0, bottom edge = survival 0
    # Note: pixel Y increases downward, but survival increases upward
    survival = 1.0 - (px_y - plot_y) / plot_height

    return time, survival
```

**Why is Y inverted?**

In image coordinates, Y=0 is at the TOP. In plot coordinates, Y=0 (survival=0) is at the BOTTOM. We must invert:

```
Image:              Plot:
Y=0  ────────       survival=1.0  ────────
     │                            │
     │                            │
Y=max ────────       survival=0   ────────
```

### Refining Calibration with Curve Data

Initial calibration uses axis lines, but curves provide more accurate bounds:

```python
def refine_plot_bounds_from_curves(self):
    # Find topmost curve pixels (these are at survival=100%)
    y_100_pixel = self.detect_y100_from_curves()

    # Find leftmost curve pixels (these are at time=0)
    x_0_pixel = self.detect_x0_from_curves()

    # Adjust plot bounds to match actual curve positions
    refined_py = y_100_pixel
    refined_px = x_0_pixel
    # ... recalculate width and height
```

**Why refine?**

Axis lines may not perfectly align with where curves actually start. Curves always start at (0, 1.0), so their leftmost/topmost pixels give us ground truth.

---

## Summary of Key Techniques

| Problem | Technique | Why This Approach |
|---------|-----------|-------------------|
| Color vs grayscale | HSV saturation analysis | Direct measure of "colorfulness" |
| Find axes | Canny + Hough transform | Robust line detection |
| Extract curves | Adaptive thresholding | Handles uneven lighting |
| Remove noise | Morphological opening | Preserves curve shape |
| Separate colors | Hue histogram peaks | Each curve = one peak |
| Handle overlaps | 3-pixel tolerance | Conservative, no invented data |
| Clean data | Monotonicity enforcement | Matches KM definition |
| Calibrate | Linear mapping + refinement | Simple and accurate |

---

## Exercises

1. **Modify the saturation threshold**: Change `sat_min = 15` to `sat_min = 40` in `color_detector.py` and test on a faded image. What happens to the curve tails?

2. **Change adaptive threshold block size**: Try `blockSize=5` and `blockSize=31`. How does this affect noise vs. curve detection?

3. **Disable monotonicity enforcement**: Comment out the monotonicity code in `_clean_curve_data()`. What artifacts appear in the output?

4. **Add a new color**: Extend `hue_to_name()` to detect "teal" (hue 80-90). Test with an image containing teal curves.
