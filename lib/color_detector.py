"""
Color-based curve detection for KM plots with colored curves.

Detects curves by their color (e.g., orange, blue, red, green) rather than
line style (solid/dashed).

AI-assisted curve reconstruction available for compressed images.
"""

import cv2
import numpy as np
import base64
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from scipy import interpolate as scipy_interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Import AI config if available
try:
    from .ai_config import AIConfig
    HAS_AI_CONFIG = True
except ImportError:
    HAS_AI_CONFIG = False


@dataclass
class ColorCurve:
    """Represents a curve detected by color."""
    name: str
    color_rgb: Tuple[int, int, int]
    mask: np.ndarray
    points: List[Tuple[int, int]]  # (x, y) pixel coordinates
    confidence: float = 0.8


def is_color_image(img: np.ndarray, threshold: float = 0.005) -> bool:
    """
    Check if image has significant color content (colored curves).

    For KM plots, curves are thin lines so even 0.5% colored pixels
    indicates a color image.

    Args:
        img: BGR image
        threshold: Minimum ratio of colored pixels (default 0.5%)

    Returns:
        True if image has colored curves
    """
    if len(img.shape) < 3:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Colored pixels: saturation > 50 and not too dark/bright
    colored = (saturation > 50) & (value > 30) & (value < 250)
    ratio = np.sum(colored) / (img.shape[0] * img.shape[1])

    # Also check if we can detect distinct curve colors
    if ratio > threshold:
        return True

    # Even with low ratio, check if distinct colors are present
    if ratio > 0.001:  # At least 0.1% colored
        colors = detect_curve_colors(img, min_pixels=50, max_colors=4)
        if len(colors) >= 2:
            return True

    return False


def detect_curve_colors(img: np.ndarray, min_pixels: int = 100, max_colors: int = 4) -> List[dict]:
    """
    Detect distinct curve colors in a KM plot.

    Args:
        img: BGR image
        min_pixels: Minimum pixels for a valid color
        max_colors: Maximum number of colors to detect

    Returns:
        List of color descriptors with HSV ranges
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]

    # Mask for colored pixels (exclude white, black, gray)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = (saturation > 40) & (value > 40) & (value < 250)

    colored_pixels = hsv[color_mask]

    if len(colored_pixels) < min_pixels:
        return []

    # Histogram of hues
    hues = colored_pixels[:, 0]
    hist, bin_edges = np.histogram(hues, bins=36, range=(0, 180))

    # Find peaks (dominant hues)
    peaks = []
    for i in range(len(hist)):
        left = hist[i-1] if i > 0 else 0
        right = hist[i+1] if i < len(hist)-1 else 0

        if hist[i] > min_pixels and hist[i] >= left and hist[i] >= right:
            center_hue = (bin_edges[i] + bin_edges[i+1]) / 2
            peaks.append((hist[i], center_hue))

    peaks.sort(reverse=True)

    # Merge nearby peaks
    merged = []
    for count, hue in peaks:
        is_close = False
        for i, (ex_count, ex_hue) in enumerate(merged):
            hue_diff = min(abs(hue - ex_hue), 180 - abs(hue - ex_hue))
            if hue_diff < 15:
                is_close = True
                if count > ex_count:
                    merged[i] = (count, hue)
                break
        if not is_close:
            merged.append((count, hue))

    merged = merged[:max_colors]

    # Build color descriptors
    colors = []

    def hue_to_name(h):
        # Hue ranges in OpenCV HSV: 0-180
        # Red: 0-8 and 165-180 (wraps around)
        # Orange: 8-25
        if h < 8 or h >= 165:
            return 'red'
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
        return 'magenta'

    for count, hue in merged:
        hue = float(hue)

        # Use TIGHTER hue ranges to avoid color bleeding between nearby curves
        # KM curve lines can have very low saturation near the Y-axis
        # Different ranges for different color regions
        if 85 <= hue <= 100:  # Cyan/teal region
            hue_range = 10  # Tighter range to avoid magenta bleeding
            sat_min = 30   # Higher saturation threshold to avoid faint/mixed pixels
            val_min = 60   # Higher value to focus on clear cyan pixels
        elif 145 <= hue <= 170:  # Purple/magenta region
            hue_range = 12  # Tighter range for purple
            sat_min = 30   # Higher saturation threshold
            val_min = 50   # Higher value threshold
        else:
            hue_range = 15  # Default - moderate
            sat_min = 20   # Moderate saturation
            val_min = 50

        # Create HSV bounds
        if hue < hue_range:
            hsv_lower = np.array([0, sat_min, val_min], dtype=np.uint8)
            hsv_upper = np.array([int(hue + hue_range), 255, 255], dtype=np.uint8)
        elif hue > 180 - hue_range:
            hsv_lower = np.array([int(hue - hue_range), sat_min, val_min], dtype=np.uint8)
            hsv_upper = np.array([180, 255, 255], dtype=np.uint8)
        else:
            hsv_lower = np.array([int(max(0, hue - hue_range)), sat_min, val_min], dtype=np.uint8)
            hsv_upper = np.array([int(min(180, hue + hue_range)), 255, 255], dtype=np.uint8)

        # Get average RGB
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        mean_bgr = cv2.mean(img, mask=mask)[:3]
        rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))

        colors.append({
            'name': hue_to_name(hue),
            'hue': hue,
            'hsv_lower': hsv_lower,
            'hsv_upper': hsv_upper,
            'rgb': rgb,
            'pixel_count': count
        })

    # Also check for gray curves (low saturation, medium value)
    # Gray curves are common in KM plots and are filtered out by the saturation threshold above
    # Add gray detection if:
    # 1. We haven't found max_colors yet, OR
    # 2. One of the detected colors has very few pixels (likely noise)
    has_weak_color = any(c['pixel_count'] < min_pixels * 10 for c in colors)
    if len(colors) < max_colors or has_weak_color:
        # Use moderate saturation threshold to catch gray curves
        gray_mask = (saturation <= 40) & (value > 80) & (value < 180)
        gray_pixel_count = np.sum(gray_mask)

        if gray_pixel_count >= min_pixels * 2:
            # Create the gray HSV bounds
            gray_hsv_lower = np.array([0, 0, 80], dtype=np.uint8)
            gray_hsv_upper = np.array([180, 40, 180], dtype=np.uint8)

            # Get average RGB for gray pixels
            gray_full_mask = cv2.inRange(hsv, gray_hsv_lower, gray_hsv_upper)
            if np.sum(gray_full_mask) > 0:
                mean_bgr = cv2.mean(img, mask=gray_full_mask)[:3]
                gray_rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))

                # Only add gray if it's a distinct medium gray
                avg_value = np.mean(hsv[:, :, 2][gray_mask])
                if 100 <= avg_value <= 170:
                    # If we have a weak color, replace it with gray
                    if has_weak_color:
                        colors = [c for c in colors if c['pixel_count'] >= min_pixels * 10]

                    colors.append({
                        'name': 'gray',
                        'hue': 0,
                        'hsv_lower': gray_hsv_lower,
                        'hsv_upper': gray_hsv_upper,
                        'rgb': gray_rgb,
                        'pixel_count': gray_pixel_count
                    })

    return colors


def extract_color_curve(
    img: np.ndarray,
    color_info: dict,
    plot_bounds: Tuple[int, int, int, int],
    other_curves_masks: List[np.ndarray] = None
) -> List[Tuple[int, int]]:
    """
    Extract curve points for a specific color.

    Args:
        img: BGR image
        color_info: Color descriptor from detect_curve_colors
        plot_bounds: (x, y, width, height) of plot area
        other_curves_masks: Masks of other detected curves (for overlap handling)

    Returns:
        List of (x, y) pixel coordinates
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]

    # Create color mask with slightly wider tolerance for edge pixels
    hsv_lower = color_info['hsv_lower'].copy()
    hsv_upper = color_info['hsv_upper'].copy()

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Extend plot bounds slightly to capture curves near edges
    px, py, pw, ph = plot_bounds
    margin = 10  # pixels
    px_ext = max(0, px - margin)
    py_ext = max(0, py - margin)
    pw_ext = min(width - px_ext, pw + 2 * margin)
    ph_ext = min(height - py_ext, ph + 2 * margin)

    mask_roi = mask[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext]

    # Clean up mask
    kernel = np.ones((2, 2), np.uint8)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)

    # Extract points - for each x, find the curve y position
    points = []

    for x in range(pw_ext):
        col = mask_roi[:, x]
        y_positions = np.where(col > 0)[0]

        if len(y_positions) > 0:
            # Use median y for this x (handles thick lines)
            y = int(np.median(y_positions))
            # Convert back to image coordinates
            points.append((px_ext + x, py_ext + y))

    return points


def _detect_text_regions(img: np.ndarray, plot_bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Detect ALL text regions in the plot area using OCR.

    Returns a mask where text regions are marked with 255.
    Masks ALL detected text to prevent legend text from connecting to curves
    during connected component analysis.

    Args:
        img: BGR image
        plot_bounds: (x, y, width, height) of plot area

    Returns:
        Binary mask with text regions marked
    """
    if not HAS_TESSERACT:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    px, py, pw, ph = plot_bounds
    plot_roi = img[py:py+ph, px:px+pw]

    # Create text mask
    text_mask = np.zeros((ph, pw), dtype=np.uint8)

    try:
        # Use pytesseract to get bounding boxes of text
        # Convert to RGB for tesseract
        rgb = cv2.cvtColor(plot_roi, cv2.COLOR_BGR2RGB)

        # Resize image if too large to speed up OCR
        # Note: Use larger threshold (1500) to avoid resizing typical KM plots
        # Smaller images cause OCR to miss small text annotations like "55%"
        max_dim = 1500
        h, w = rgb.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        # Get text data with bounding boxes (with timeout to prevent hanging)
        data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, timeout=10)

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

            # Mask ALL text with reasonable confidence (not just annotations)
            # This prevents legend text from connecting to curves
            if conf > 30 and len(text) > 0:
                # Scale coordinates back to original size if image was resized
                x = int(data['left'][i] / scale)
                y = int(data['top'][i] / scale)
                tw = int(data['width'][i] / scale)
                th = int(data['height'][i] / scale)

                # Skip very large boxes (might be misdetection covering plot area)
                if tw > pw * 0.5 or th > ph * 0.2:
                    continue

                # Skip very small text (might be noise or single pixels)
                if tw < 5 or th < 5:
                    continue

                # Add padding around text to ensure complete removal
                # This breaks any connection between text and curves
                padding = 3
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(pw, x + tw + padding)
                y2 = min(ph, y + th + padding)

                # Mark text region with padding
                text_mask[y1:y2, x1:x2] = 255

    except Exception:
        # If OCR fails, return empty mask
        pass

    return text_mask


def _isolate_main_curve_component(mask: np.ndarray, plot_height: int) -> np.ndarray:
    """
    Isolate curve components from a mask, removing noise.

    For survival curves:
    - The main curve should span significant horizontal extent
    - Small isolated blobs are likely noise

    This function keeps ALL components that look like curve segments,
    not just the single best one, to handle curves that may be split
    into multiple disconnected components.

    Args:
        mask: Binary mask of detected color pixels
        plot_height: Height of the plot area

    Returns:
        Cleaned mask with curve components (noise removed)
    """
    if np.sum(mask) == 0:
        return mask

    # Apply dilation to connect nearby curve segments
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Find connected components on the dilated mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    if num_labels <= 1:
        return mask

    # Minimum width to be considered a curve segment (at least 2% of plot width)
    min_width = max(8, mask.shape[1] * 0.02)

    # Keep all components that look like curve segments
    curve_labels = []

    for label in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]

        # Skip tiny components (likely noise) - use lower threshold
        if area < 20:
            continue

        # Skip very narrow components (likely text or tick marks)
        if w < min_width:
            continue

        # Skip components that span almost full height AND are narrow (likely axis lines)
        height_ratio = h / plot_height
        width_ratio = w / mask.shape[1]
        if height_ratio > 0.9 and width_ratio < 0.2:
            continue

        # Skip components entirely in the bottom 15% with small width
        # (curves might drop there, but small blobs are noise)
        if cy > plot_height * 0.85 and w < mask.shape[1] * 0.1:
            continue

        # Skip text-like components: non-trivial bounding box but sparse fill
        # Text annotations have gaps between characters; curves are continuous lines
        # Only apply to SMALL-to-MEDIUM components (text annotations are typically < 200px wide)
        # Large components spanning the plot are likely the actual curve (thin line = low fill)
        # Text "14% (7-28)" might have w~80, h~15, area~300
        max_text_width = min(300, mask.shape[1] * 0.25)  # Text annotations < 25% of plot width
        max_text_height = min(50, plot_height * 0.1)  # Text annotations < 10% of plot height
        if w < max_text_width and h > 12 and h < max_text_height:
            fill_ratio = area / (w * h) if w * h > 0 else 1.0
            # Text: sparse (fill_ratio < 0.35) due to character gaps
            if fill_ratio < 0.35:
                continue

        # This component looks like a curve segment - keep it
        curve_labels.append(label)

    if not curve_labels:
        # No good components found - fall back to keeping the largest
        areas = [(stats[label][4], label) for label in range(1, num_labels)]
        if areas:
            areas.sort(reverse=True)
            curve_labels = [areas[0][1]]
        else:
            return mask

    # Create a mask of the dilated regions we want to keep
    dilated_keep = np.zeros_like(dilated)
    for label in curve_labels:
        dilated_keep[labels == label] = 255

    # Return the original mask pixels that fall within the kept dilated regions
    # This preserves the original pixel positions while using the dilated connectivity
    result = cv2.bitwise_and(mask, dilated_keep)

    return result


def _bridge_curve_gaps(
    points: List[Tuple[int, int]],
    max_gap_pixels: int = 100
) -> List[Tuple[int, int]]:
    """
    Bridge gaps in extracted curve points caused by text annotations.

    When text overlays a curve, there may be gaps in the extracted x positions.
    For Kaplan-Meier curves, gaps should be filled with horizontal segments
    (maintaining the previous y value until the next detected point).

    This function also handles the case where text pixels are detected at wrong
    y positions by checking for sudden jumps followed by returns to the original level.

    Args:
        points: List of (x, y) pixel coordinates
        max_gap_pixels: Maximum gap size to bridge (default 100 pixels)

    Returns:
        List of (x, y) points with gaps bridged
    """
    if len(points) < 10:
        return points

    # Sort points by x
    sorted_points = sorted(points, key=lambda p: p[0])

    # First pass: detect and fix text-induced anomalies
    # Text annotations can cause:
    # 1. Sudden jumps UP (text above curve picked instead of curve)
    # 2. Sudden jumps DOWN (text below curve picked instead of curve)
    # 3. Gaps in x coverage (text completely masks curve)

    # Detect anomalous regions where y jumps significantly then returns
    cleaned_points = []
    i = 0
    while i < len(sorted_points):
        x, y = sorted_points[i]

        # Look for sudden anomalies: y jumps by > 15 pixels for a short stretch
        # then returns to near the original level
        if i > 0 and i < len(sorted_points) - 1:
            prev_x, prev_y = sorted_points[i - 1]

            # Check if this point is an anomaly
            y_diff = abs(y - prev_y)
            x_diff = x - prev_x

            # Anomaly: large y jump with small x change (not a real KM step)
            if y_diff > 15 and x_diff < 5:
                # Look ahead to see if curve returns to near prev_y
                look_ahead = min(50, len(sorted_points) - i)
                recovery_found = False
                recovery_idx = i

                for j in range(i + 1, i + look_ahead):
                    future_x, future_y = sorted_points[j]
                    # Check if we're back near the original level
                    if abs(future_y - prev_y) < 10:
                        recovery_found = True
                        recovery_idx = j
                        break

                if recovery_found:
                    # Skip anomalous points and bridge with horizontal line
                    # Add horizontal points from prev_x to recovery_x at prev_y
                    for fill_x in range(prev_x + 1, sorted_points[recovery_idx][0]):
                        cleaned_points.append((fill_x, prev_y))
                    i = recovery_idx
                    continue

        cleaned_points.append((x, y))
        i += 1

    if len(cleaned_points) < 5:
        cleaned_points = sorted_points

    # Second pass: bridge gaps in x coverage
    # Sort again after cleaning
    cleaned_points = sorted(cleaned_points, key=lambda p: p[0])

    bridged_points = []
    for i, (x, y) in enumerate(cleaned_points):
        bridged_points.append((x, y))

        if i < len(cleaned_points) - 1:
            next_x, next_y = cleaned_points[i + 1]
            gap = next_x - x

            # If there's a gap, fill it with horizontal line at current y
            # (KM curves are step functions - horizontal until next event)
            if gap > 3 and gap <= max_gap_pixels:
                # Check if this looks like a real gap (not a step transition)
                # A step transition: y changes significantly at the gap
                # A text gap: y stays similar before and after

                y_change = abs(next_y - y)

                # For KM curves, if the next y is LOWER (survival dropped),
                # we should still fill the gap horizontally at current y
                # then let the step happen at the transition point

                # Fill the gap with horizontal line at current y level
                for fill_x in range(x + 1, next_x):
                    bridged_points.append((fill_x, y))

    return bridged_points


def correct_curve_with_at_risk_interpolation(
    curve_points: List[Tuple[float, float]],
    at_risk_data: Optional[Dict] = None,
    annotation_points: Optional[List[Tuple[float, float]]] = None,
    time_max: float = 24.0,
    curve_name: str = ""
) -> List[Tuple[float, float]]:
    """
    Correct or fill gaps in a curve using at-risk table data and/or annotations.

    This is particularly useful when image quality causes curve detection to fail
    in certain regions. The at-risk numbers and text annotations can provide
    anchor points for interpolation.

    Args:
        curve_points: List of (time, survival) tuples from detection
        at_risk_data: Dict with 'times' and 'counts' lists from at-risk table
        annotation_points: List of (time, survival) tuples from image annotations
        time_max: Maximum time value for the x-axis
        curve_name: Name of the curve (for logging)

    Returns:
        Corrected list of (time, survival) tuples
    """
    if not curve_points:
        return curve_points

    # Sort by time
    sorted_points = sorted(curve_points, key=lambda p: p[0])

    # If no reference data provided, return as-is
    if not at_risk_data and not annotation_points:
        return sorted_points

    # Analyze the detected curve for problems
    times = [p[0] for p in sorted_points]
    survivals = [p[1] for p in sorted_points]

    max_detected_time = max(times) if times else 0
    min_detected_survival = min(survivals) if survivals else 1.0

    # Problem detection:
    # 1. Curve ends too early (< 80% of time_max)
    # 2. Curve drops to near-zero too early
    # 3. Large gaps in time coverage

    needs_correction = False

    # Check if curve ends too early
    if max_detected_time < time_max * 0.8:
        needs_correction = True

    # Check if survival drops unrealistically (to <5% when at-risk suggests higher)
    if at_risk_data and min_detected_survival < 0.05:
        # Get expected minimum survival from at-risk
        if 'counts' in at_risk_data and len(at_risk_data['counts']) > 0:
            initial_count = at_risk_data['counts'][0]
            final_count = at_risk_data['counts'][-1]
            expected_min_survival = final_count / initial_count if initial_count > 0 else 0
            if min_detected_survival < expected_min_survival * 0.5:
                needs_correction = True

    if not needs_correction:
        return sorted_points

    # Build reference points for interpolation
    reference_points = []

    # Add annotation points if provided
    if annotation_points:
        reference_points.extend(annotation_points)

    # Add at-risk derived constraints
    if at_risk_data and 'times' in at_risk_data and 'counts' in at_risk_data:
        ar_times = at_risk_data['times']
        ar_counts = at_risk_data['counts']
        if len(ar_counts) > 0:
            initial_count = ar_counts[0]
            for t, count in zip(ar_times, ar_counts):
                # Maximum possible survival at this time
                max_survival = count / initial_count if initial_count > 0 else 1.0
                # Use 90% of max as estimate (accounting for events between observations)
                estimated_survival = max_survival * 0.9
                reference_points.append((t, estimated_survival))

    if not reference_points:
        return sorted_points

    # Sort reference points by time
    reference_points = sorted(set(reference_points), key=lambda p: p[0])

    # Find where detected curve is reliable vs unreliable
    # Reliable: survival values are reasonable (not near 0 when shouldn't be)
    reliable_points = []
    unreliable_start_time = None

    for t, s in sorted_points:
        # Check if this point seems reliable
        # Find nearest reference point
        nearest_ref = None
        min_dist = float('inf')
        for ref_t, ref_s in reference_points:
            dist = abs(t - ref_t)
            if dist < min_dist:
                min_dist = dist
                nearest_ref = (ref_t, ref_s)

        # Point is unreliable if survival is way off from reference
        if nearest_ref and min_dist < 3:  # Within 3 time units
            ref_t, ref_s = nearest_ref
            if s < ref_s * 0.3:  # Detected survival is <30% of expected
                if unreliable_start_time is None:
                    unreliable_start_time = t
                continue

        reliable_points.append((t, s))

    if not reliable_points:
        # All points unreliable - use reference points directly
        if HAS_SCIPY:
            ref_times = [p[0] for p in reference_points]
            ref_survivals = [p[1] for p in reference_points]
            f = scipy_interpolate.interp1d(ref_times, ref_survivals, kind='linear',
                                           bounds_error=False, fill_value='extrapolate')

            # Generate dense points
            result = []
            t = 0.0
            dt = time_max / 1000
            while t <= time_max:
                s = float(f(t))
                s = max(0.0, min(1.0, s))  # Clamp to valid range
                result.append((t, s))
                t += dt
            return result
        else:
            return sorted_points

    # Blend reliable detected points with reference interpolation
    if unreliable_start_time is not None and HAS_SCIPY:
        # Use detected points up to unreliable region
        # Then use reference-based interpolation for the rest

        # Get reference points for interpolation
        ref_times = [p[0] for p in reference_points if p[0] >= unreliable_start_time - 2]
        ref_survivals = [p[1] for p in reference_points if p[0] >= unreliable_start_time - 2]

        if len(ref_times) >= 2:
            f = scipy_interpolate.interp1d(ref_times, ref_survivals, kind='linear',
                                           bounds_error=False, fill_value='extrapolate')

            # Build result: reliable points + interpolated points
            result = list(reliable_points)

            # Find last reliable time
            last_reliable_time = max(p[0] for p in reliable_points)

            # Add interpolated points from last reliable time to end
            t = last_reliable_time + (time_max / 1000)
            dt = time_max / 1000
            while t <= time_max:
                s = float(f(t))
                s = max(0.0, min(1.0, s))
                result.append((t, s))
                t += dt

            return sorted(result, key=lambda p: p[0])

    return sorted_points


def _remove_all_text_from_image(
    img: np.ndarray,
    use_ai: bool = False,
    protect_curves: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove ALL text from the image using OCR detection and inpainting.

    Args:
        img: BGR image
        use_ai: Whether to use AI-enhanced text removal (requires Ollama)
        protect_curves: Whether to protect curve pixels during inpainting

    Returns:
        Tuple of (cleaned_image, text_mask)
    """
    # Try AI-enhanced text removal first
    if use_ai:
        try:
            from .ai_text_remover import AITextRemover
            remover = AITextRemover()
            if remover.is_available:
                print("  Using AI-enhanced text removal...")
                cleaned, text_mask, report = remover.remove_text_with_ai_guidance(img)
                if report.get('success', True):
                    return cleaned, text_mask
                print(f"  AI text removal issues: {report.get('issues', [])}")
        except ImportError:
            pass
        except Exception as e:
            print(f"  AI text removal failed: {e}")

    # Fallback to enhanced OCR-based text removal
    if not HAS_TESSERACT:
        return img.copy(), np.zeros(img.shape[:2], dtype=np.uint8)

    result = img.copy()
    h, w = img.shape[:2]

    # Create mask for all text regions
    text_mask = np.zeros((h, w), dtype=np.uint8)

    # Detect curve pixels FIRST to protect them
    curve_mask = _detect_curve_pixels_for_protection(img) if protect_curves else None

    # Multiple OCR passes with different preprocessing for better detection
    text_regions = []

    # Pass 1: Standard OCR
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, timeout=30)
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
            if conf > 20 and len(text) > 0:
                text_regions.append({
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'text': text,
                    'conf': conf
                })
    except Exception:
        pass

    # Pass 2: Enhanced contrast OCR for small/faint text
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        # Binarize with Otsu
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if background is dark
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, timeout=30)
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
            if conf > 15 and len(text) > 0:  # Lower threshold for enhanced
                # Check if this region is not already detected
                x, y, tw, th = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                is_new = True
                for r in text_regions:
                    # Check overlap
                    if abs(r['x'] - x) < 10 and abs(r['y'] - y) < 10:
                        is_new = False
                        break
                if is_new:
                    text_regions.append({
                        'x': x, 'y': y, 'w': tw, 'h': th,
                        'text': text, 'conf': conf
                    })
    except Exception:
        pass

    print(f"  Found {len(text_regions)} text regions")

    # Create text mask from detected regions
    for r in text_regions:
        x, y, tw, th = r['x'], r['y'], r['w'], r['h']
        padding = 3
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + tw + padding)
        y2 = min(h, y + th + padding)
        text_mask[y1:y2, x1:x2] = 255

    # Dilate text mask slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask_dilated = cv2.dilate(text_mask, kernel, iterations=1)

    # Create inpaint mask, protecting curves
    if curve_mask is not None:
        inpaint_mask = text_mask_dilated.copy()
        # Remove curve pixels from inpaint mask
        inpaint_mask[curve_mask > 0] = 0
    else:
        inpaint_mask = text_mask_dilated

    # Inpaint to remove text
    result = cv2.inpaint(result, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return result, text_mask


def _detect_curve_pixels_for_protection(img: np.ndarray) -> np.ndarray:
    """
    Detect curve pixels to protect during text removal inpainting.

    This creates a mask of pixels that are likely part of curves
    and should not be removed even if they're near text.

    Args:
        img: BGR image

    Returns:
        Binary mask of curve pixels
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]

    # Curves are dark pixels - use adaptive threshold for varying backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    # Remove very small noise
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    curve_mask = np.zeros((h, w), dtype=np.uint8)

    # Find the largest components (likely main curves)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    if areas:
        sorted_areas = sorted(areas, reverse=True)
        # Threshold: components larger than 10% of second largest are likely curves
        curve_threshold = sorted_areas[1] * 0.1 if len(sorted_areas) >= 2 else 200

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        if area < 20:  # Skip tiny noise
            continue

        # Calculate characteristics
        aspect = max(bw, bh) / max(1, min(bw, bh))
        fill_ratio = area / (bw * bh) if bw * bh > 0 else 0

        # Curves are typically:
        # - Large connected components
        # - OR elongated (high aspect ratio)
        # - OR span significant vertical range (for survival curves)
        # - OR thin with low fill ratio (unlike solid text)

        is_curve = False

        # Large components are likely curves
        if area > curve_threshold:
            is_curve = True

        # Elongated components (aspect > 3) are likely curve segments
        if aspect > 3:
            is_curve = True

        # Components spanning significant height are likely curves
        if bh > h * 0.15:
            is_curve = True

        # Thin, wide segments with low fill are likely dashed curves
        if bw > 30 and bh < 20 and fill_ratio < 0.5:
            is_curve = True

        # Components in the middle Y-range that are wide enough
        in_curve_region = h * 0.1 < cy < h * 0.9
        if in_curve_region and bw > 50:
            is_curve = True

        if is_curve:
            curve_mask[labels == i] = 255

    # Dilate to protect curve edges and bridge small gaps
    kernel = np.ones((3, 3), np.uint8)
    curve_mask = cv2.dilate(curve_mask, kernel, iterations=2)

    return curve_mask


def extract_curves_with_skeleton(
    img: np.ndarray,
    colors: List[dict],
    plot_bounds: Tuple[int, int, int, int]
) -> List[Tuple[List[Tuple[int, int]], dict]]:
    """
    Extract curves using text removal and skeletonization for clean extraction.

    This approach:
    1. Removes ALL text from the image using OCR + inpainting
    2. Creates color masks for each curve
    3. Skeletonizes to get single-pixel-wide curves
    4. Extracts points from the skeleton

    Args:
        img: BGR image
        colors: List of color descriptors
        plot_bounds: (x, y, width, height) of plot area

    Returns:
        List of (points, color_info) tuples for each curve
    """
    if not HAS_SKIMAGE:
        print("Warning: skimage not available, falling back to standard extraction")
        return extract_curves_with_overlap_handling(img, colors, plot_bounds)

    # Step 1: Remove all text from image
    cleaned_img, text_mask = _remove_all_text_from_image(img)

    # Step 2: Create HSV for color detection
    hsv = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2HSV)

    px, py, pw, ph = plot_bounds
    margin = 5
    px_ext = max(0, px - margin)
    py_ext = max(0, py - margin)
    pw_ext = min(img.shape[1] - px_ext, pw + 2*margin)
    ph_ext = min(img.shape[0] - py_ext, ph + 2*margin)

    results = []

    for color_info in colors:
        name = color_info.get('name', 'unknown')

        # Create color mask
        mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])

        # Crop to plot area
        mask_roi = mask[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext]

        # Clean up mask - remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_clean = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)

        # Remove small connected components (noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)
        min_area = 30  # Minimum pixels for a valid curve segment

        mask_filtered = np.zeros_like(mask_clean)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask_filtered[labels == label] = 255

        # Skeletonize to get single-pixel-wide curves
        skeleton = skeletonize(mask_filtered > 0).astype(np.uint8) * 255

        # Extract points from skeleton
        points = []
        ys, xs = np.where(skeleton > 0)

        for x, y in zip(xs, ys):
            # Convert to full image coordinates
            full_x = px_ext + x
            full_y = py_ext + y
            points.append((full_x, full_y))

        results.append((points, color_info))

    return results


def _fix_curve_continuity(
    results: List[Tuple[List[Tuple[int, int]], dict]],
    binary: np.ndarray,
    h: int,
    w: int,
    px_ext: int,
    py_ext: int
) -> List[Tuple[List[Tuple[int, int]], dict]]:
    """
    Fix curve continuity issues when curves are close together.

    When the LineStyleDetector swaps curve segments (e.g., at crossing points),
    this function detects sudden Y-jumps and swaps points between curves to
    maintain spatial continuity.

    Args:
        results: List of (points, curve_info) tuples
        binary: Binary image of curves
        h, w: Height and width of ROI
        px_ext, py_ext: ROI offset in full image

    Returns:
        Corrected results with swapped points where necessary
    """
    if len(results) != 2:
        return results

    points0, info0 = results[0]
    points1, info1 = results[1]

    # Build column-by-column view: for each x, get the Y value for each curve
    curve0_by_x = {}  # x -> y (in ROI coords)
    curve1_by_x = {}

    for full_x, full_y in points0:
        roi_x = full_x - px_ext
        roi_y = full_y - py_ext
        if roi_x not in curve0_by_x or roi_y < curve0_by_x[roi_x]:
            curve0_by_x[roi_x] = roi_y

    for full_x, full_y in points1:
        roi_x = full_x - px_ext
        roi_y = full_y - py_ext
        if roi_x not in curve1_by_x or roi_y > curve1_by_x[roi_x]:
            curve1_by_x[roi_x] = roi_y

    # Find common x positions
    common_x = sorted(set(curve0_by_x.keys()) & set(curve1_by_x.keys()))

    if len(common_x) < 20:
        return results

    # Detect and fix jumps using sliding window
    window_size = 10
    max_jump = 15  # Maximum allowed Y-jump between consecutive x positions

    # First pass: identify regions where curves might be swapped
    swap_regions = []  # List of (start_x, end_x) tuples

    i = 0
    while i < len(common_x) - window_size:
        x = common_x[i]
        y0 = curve0_by_x[x]
        y1 = curve1_by_x[x]

        # Look ahead to check for sudden jumps
        next_x = common_x[i + 1]
        next_y0 = curve0_by_x[next_x]
        next_y1 = curve1_by_x[next_x]

        jump0 = abs(next_y0 - y0)
        jump1 = abs(next_y1 - y1)

        # If curve0 jumps toward where curve1 was (and vice versa), curves likely swapped
        sep = abs(y0 - y1)
        if sep < 30 and (jump0 > max_jump or jump1 > max_jump):
            # Check if swapping would reduce the jump
            swap_jump0 = abs(next_y1 - y0)
            swap_jump1 = abs(next_y0 - y1)

            if swap_jump0 < jump0 and swap_jump1 < jump1:
                # Find extent of this swap region
                region_start = i + 1
                region_end = region_start

                # Extend region while swapped values are more continuous
                for j in range(i + 1, len(common_x) - 1):
                    curr_x = common_x[j]
                    next_x = common_x[j + 1]
                    curr_y0 = curve0_by_x[curr_x]
                    curr_y1 = curve1_by_x[curr_x]
                    next_y0 = curve0_by_x[next_x]
                    next_y1 = curve1_by_x[next_x]

                    # Check if still swapped
                    normal_jump0 = abs(next_y0 - curr_y0)
                    normal_jump1 = abs(next_y1 - curr_y1)
                    swapped_jump0 = abs(next_y1 - curr_y0)
                    swapped_jump1 = abs(next_y0 - curr_y1)

                    if swapped_jump0 < normal_jump0 and swapped_jump1 < normal_jump1:
                        region_end = j + 1
                    elif normal_jump0 <= max_jump and normal_jump1 <= max_jump:
                        # Curves back to normal
                        break

                if region_end > region_start:
                    swap_regions.append((common_x[region_start], common_x[region_end]))
                    i = region_end + 1
                    continue

        i += 1

    if not swap_regions:
        return results

    # Apply swaps
    new_points0 = []
    new_points1 = []

    for full_x, full_y in points0:
        in_swap_region = any(start <= (full_x - px_ext) <= end for start, end in swap_regions)
        if in_swap_region:
            new_points1.append((full_x, full_y))
        else:
            new_points0.append((full_x, full_y))

    for full_x, full_y in points1:
        in_swap_region = any(start <= (full_x - px_ext) <= end for start, end in swap_regions)
        if in_swap_region:
            new_points0.append((full_x, full_y))
        else:
            new_points1.append((full_x, full_y))

    print(f"    Continuity fix: swapped {len(swap_regions)} region(s)")
    for start, end in swap_regions:
        print(f"      Region: x={start} to {end}")

    return [(new_points0, info0), (new_points1, info1)]


def extract_curves_grayscale_skeleton(
    img: np.ndarray,
    plot_bounds: Tuple[int, int, int, int],
    expected_curves: int = 2,
    detected_curves: list = None
) -> List[Tuple[List[Tuple[int, int]], dict]]:
    """
    Extract curves from grayscale images using text removal and skeletonization.

    This hybrid approach:
    1. Removes ALL text from the image using OCR + inpainting
    2. Uses pre-detected curve data (from LineStyleDetector) to create masks
    3. Skeletonizes each curve for clean extraction

    If detected_curves is provided, uses those for curve separation.
    Otherwise falls back to simple Y-position based separation.

    Args:
        img: BGR image (grayscale content)
        plot_bounds: (x, y, width, height) of plot area
        expected_curves: Number of curves to detect (default: 2)
        detected_curves: Optional list of DetectedCurve objects from LineStyleDetector

    Returns:
        List of (points, curve_info) tuples for each curve
    """
    if not HAS_SKIMAGE:
        print("Warning: skimage not available for skeleton extraction")
        return []

    # Step 1: Remove all text from image
    cleaned_img, text_mask = _remove_all_text_from_image(img)

    # Step 2: Convert to grayscale and create binary mask
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)

    px, py, pw, ph = plot_bounds
    margin = 5
    px_ext = max(0, px - margin)
    py_ext = max(0, py - margin)
    pw_ext = min(img.shape[1] - px_ext, pw + 2*margin)
    ph_ext = min(img.shape[0] - py_ext, ph + 2*margin)

    # Crop to plot area
    gray_roi = gray[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext]

    # Adaptive thresholding to get dark pixels (curves)
    binary = cv2.adaptiveThreshold(
        gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    h, w = binary.shape
    results = []

    # Step 3: Create masks for each curve
    if detected_curves and len(detected_curves) >= 2:
        # Use pre-detected curve data to create masks
        for curve in detected_curves:
            mask = np.zeros((h, w), dtype=np.uint8)

            # Get curve style name
            style_name = curve.style.value if hasattr(curve.style, 'value') else str(curve.style)

            # Get pixels from all segments
            for segment in curve.segments:
                for px_coord, py_coord in segment.pixels:
                    # Convert to ROI coordinates
                    roi_x = px_coord + px - px_ext
                    roi_y = py_coord + py - py_ext
                    if 0 <= roi_x < w and 0 <= roi_y < h:
                        # Draw a small region around each pixel
                        cv2.circle(mask, (roi_x, roi_y), 2, 255, -1)

            # Skeletonize
            if np.sum(mask) > 0:
                skeleton = skeletonize(mask > 0).astype(np.uint8) * 255

                # Extract points
                points = []
                ys, xs = np.where(skeleton > 0)

                for x, y in zip(xs, ys):
                    full_x = px_ext + x
                    full_y = py_ext + y
                    points.append((full_x, full_y))

                curve_info = {
                    'name': style_name,
                    'style': style_name,
                }
                results.append((points, curve_info))

        # Step 4: Apply continuity correction when curves are close together
        # This fixes cases where the LineStyleDetector swapped curve segments
        if len(results) == 2:
            results = _fix_curve_continuity(results, binary, h, w, px_ext, py_ext)

    else:
        # Fallback: Separate by Y position (upper vs lower curve)
        # At each x, assign upper pixels to one curve and lower to another
        upper_mask = np.zeros_like(binary)
        lower_mask = np.zeros_like(binary)

        for x in range(w):
            col = binary[:, x]
            dark_pixels = np.where(col > 0)[0]

            if len(dark_pixels) < 2:
                # Single pixel or none - assign to upper curve
                for y in dark_pixels:
                    upper_mask[y, x] = 255
                continue

            # Find the center of mass
            center_y = np.mean(dark_pixels)

            # Split pixels into upper and lower based on center
            for y in dark_pixels:
                if y < center_y:
                    upper_mask[y, x] = 255
                else:
                    lower_mask[y, x] = 255

        # Skeletonize each mask
        for mask, style in [(upper_mask, 'solid'), (lower_mask, 'dashed')]:
            if np.sum(mask) == 0:
                continue

            skeleton = skeletonize(mask > 0).astype(np.uint8) * 255

            # Extract points
            points = []
            ys, xs = np.where(skeleton > 0)

            for x, y in zip(xs, ys):
                full_x = px_ext + x
                full_y = py_ext + y
                points.append((full_x, full_y))

            curve_info = {
                'name': style,
                'style': style,
            }
            results.append((points, curve_info))

    return results


def extract_curves_with_overlap_handling(
    img: np.ndarray,
    colors: List[dict],
    plot_bounds: Tuple[int, int, int, int]
) -> List[Tuple[List[Tuple[int, int]], dict]]:
    """
    Extract multiple curves with overlap handling.

    When curves overlap (same x, similar y), pixels may only show one color.
    This function detects overlaps and assigns points to both curves.

    Args:
        img: BGR image
        colors: List of color descriptors
        plot_bounds: (x, y, width, height) of plot area

    Returns:
        List of (points, color_info) tuples for each curve
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]

    px, py, pw, ph = plot_bounds

    # IMPORTANT: Only look for colored pixels WITHIN the plot bounds
    # This excludes legend text, axis labels, and other colored elements outside the plot
    # Use a small margin to catch curves near the edges
    margin = 5
    plot_x_min = max(0, px - margin)
    plot_x_max = min(width, px + pw + margin)
    plot_y_min = max(0, py - margin)
    plot_y_max = min(height, py + ph + margin)

    # Use strict plot bounds - don't expand based on color detection
    # This prevents legend text from being included
    px_ext = plot_x_min
    py_ext = plot_y_min
    pw_ext = plot_x_max - plot_x_min
    ph_ext = plot_y_max - plot_y_min

    # Detect text regions using OCR to exclude text annotations from curve detection
    # This helps when text annotations have the same color as curves
    text_mask = _detect_text_regions(img, (px_ext, py_ext, pw_ext, ph_ext))

    # Create masks for each color with improved filtering
    masks = []
    for color_info in colors:
        mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])
        mask_roi = mask[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext]
        original_mask_roi = mask_roi.copy()

        is_gray_curve = color_info.get('name', '').lower() == 'gray'

        # For GRAY curves: use adaptive thresholding for better detection
        # Gray curves can have very thin lines that get lost with global thresholding
        if is_gray_curve:
            # Get grayscale ROI
            gray_roi = cv2.cvtColor(img[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext], cv2.COLOR_BGR2GRAY)

            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_roi)

            # Adaptive thresholding on enhanced image
            adapt_thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Combine with color mask: keep adaptive-detected pixels that are also gray-colored
            adaptive_gray = cv2.bitwise_and(adapt_thresh, mask_roi)

            # If adaptive detection found meaningful content, use it
            # Otherwise fall back to color-only detection
            adaptive_pixel_count = np.sum(adaptive_gray > 0)
            color_pixel_count = np.sum(mask_roi > 0)

            if adaptive_pixel_count > color_pixel_count * 0.3:
                # Adaptive found good content - use the union of both masks
                # This helps fill gaps where either method alone might miss pixels
                mask_roi = cv2.bitwise_or(adaptive_gray, mask_roi)

            # Remove obvious horizontal axis lines (rows with >80% filled)
            for row_idx in range(ph_ext):
                row = mask_roi[row_idx, :]
                if np.sum(row > 0) > pw_ext * 0.8:
                    mask_roi[row_idx, :] = 0

        # IMPORTANT: Apply text and legend removal AFTER all other processing
        # This ensures they are not re-added by adaptive thresholding

        # Remove text regions from the mask
        # This prevents text annotations and legend from being detected as curve
        if text_mask is not None and text_mask.shape == mask_roi.shape and np.any(text_mask):
            mask_roi = cv2.bitwise_and(mask_roi, cv2.bitwise_not(text_mask))

        # Remove legend region carefully to avoid removing actual curve data
        # The legend is typically in the top-right area of the plot
        # Curves start at t=0 (left side) with high survival (top of plot)

        # Strategy: Remove in zones that don't overlap with typical curve paths
        # Zone 1: Top 20% but only after first 15% of x (preserve t=0 to ~t=3.6 area)
        legend_zone1_height = int(ph_ext * 0.20)
        legend_zone1_start_x = int(pw_ext * 0.15)
        if legend_zone1_height > 0:
            mask_roi[:legend_zone1_height, legend_zone1_start_x:] = 0

        # Zone 2: Top-right quadrant where legend text is most common
        # This is more aggressive for the right side where curves have lower survival
        upper_right_y = int(ph_ext * 0.35)  # Top 35%
        upper_right_x = int(pw_ext * 0.35)  # Right 65%
        if upper_right_y > 0 and upper_right_x < pw_ext:
            mask_roi[:upper_right_y, upper_right_x:] = 0

        # Use connected components to find the main curve
        # This function dilates for connectivity but returns original pixels
        mask_roi = _isolate_main_curve_component(mask_roi, ph_ext)

        masks.append(mask_roi)

    # Extract points for each curve
    all_curve_points = [[] for _ in colors]

    # Determine which curves are gray (need special handling)
    gray_curve_indices = set()
    for i, color_info in enumerate(colors):
        if color_info.get('name', '').lower() == 'gray':
            gray_curve_indices.add(i)

    # For each x position, find y positions for each color
    for x in range(pw_ext):
        curve_ys = []  # (curve_idx, y_position) for each detected curve at this x

        for i, mask_roi in enumerate(masks):
            col = mask_roi[:, x]
            y_positions = np.where(col > 0)[0]

            if len(y_positions) > 0:
                # For KM survival curves with STEP function shape:
                # - At most x positions: single y value (horizontal line segment)
                # - At step transitions: multiple y values (vertical drop)
                #
                # To preserve step shape, we need to capture BOTH top and bottom
                # of vertical segments, not just the median.
                #
                # IMPORTANT: When there could be color bleeding from nearby curves,
                # prefer values that maintain continuity with previous points.

                if len(y_positions) == 1:
                    y = y_positions[0]
                    curve_ys.append((i, y, 'single'))
                else:
                    # Check if positions form a compact cluster or a vertical step
                    y_range = y_positions.max() - y_positions.min()

                    if y_range < 5:  # Compact cluster (curve thickness ~2-4 pixels)
                        # Use the TOPMOST pixel (minimum y = maximum survival)
                        # This is more accurate than median because:
                        # - The true curve position is at the top of the line
                        # - The bottom pixels are typically anti-aliasing artifacts
                        y = int(np.min(y_positions))
                        curve_ys.append((i, y, 'cluster'))
                    elif i in gray_curve_indices:
                        # SPECIAL HANDLING FOR GRAY CURVES:
                        # Gray masks often pick up text annotations, axis lines, and other
                        # gray elements BELOW the actual curve. Always prefer the topmost
                        # pixel cluster to get the actual curve, not the noise below it.
                        #
                        # Exception: if the topmost pixel is at very high survival (>95%),
                        # it might be axis/legend noise at the top of the plot.
                        sorted_y = np.sort(y_positions)
                        y_topmost = int(sorted_y[0])
                        top_survival = 1.0 - y_topmost / ph_ext

                        if top_survival > 0.98:
                            # Very top of plot - might be noise, look for next cluster
                            gaps = np.diff(sorted_y)
                            large_gaps = np.where(gaps > 5)[0]
                            if len(large_gaps) > 0:
                                # Use first cluster below the top
                                next_cluster_start = large_gaps[0] + 1
                                y = int(sorted_y[next_cluster_start])
                                curve_ys.append((i, y, 'gray_skip_top_noise'))
                            else:
                                # No clear gap, use topmost
                                curve_ys.append((i, y_topmost, 'gray_topmost'))
                        else:
                            # Normal case - use topmost pixel
                            curve_ys.append((i, y_topmost, 'gray_topmost'))
                    elif y_range >= 5:
                        # This could be a STEP in the KM curve!
                        # Capture BOTH the top (before drop) and bottom (after drop)
                        sorted_y = np.sort(y_positions)

                        # Check for gaps that might indicate separate segments
                        gaps = np.diff(sorted_y)
                        large_gaps = np.where(gaps > 3)[0]

                        if len(large_gaps) > 0:
                            # Multiple clusters with gaps - likely curve line + cross/tick mark
                            # or curve + noise at plot boundaries
                            first_gap = large_gaps[0]
                            top_cluster = sorted_y[:first_gap + 1]
                            bottom_cluster = sorted_y[first_gap + 1:]

                            top_size = len(top_cluster)
                            bottom_size = len(bottom_cluster)
                            gap_size = sorted_y[first_gap + 1] - sorted_y[first_gap]

                            # Use topmost pixel of each cluster
                            y_top = int(np.min(top_cluster))
                            y_bottom = int(np.min(bottom_cluster))

                            # Special case: huge gap indicates separate objects (curve vs noise)
                            # If gap is > 50% of plot height, one is likely noise
                            if gap_size > ph_ext * 0.5:
                                # Large gap - check which cluster is in reasonable curve position
                                # KM curves typically have survival > 0.3 (y < 70% of plot height)
                                top_survival = 1.0 - y_top / ph_ext
                                bottom_survival = 1.0 - y_bottom / ph_ext

                                # Use cluster size to help distinguish curve from noise
                                # Text annotations tend to be sparse, curves are denser
                                top_is_sparse = top_size <= 3
                                bottom_is_sparse = bottom_size <= 3

                                if top_survival > 0.9 and bottom_survival > 0.05 and bottom_survival < 0.5:
                                    # Top is at very top (>90%, likely text), bottom is reasonable curve range
                                    # This handles text annotations at top with curve at lower survival
                                    curve_ys.append((i, y_bottom, 'noise_filtered_top_text'))
                                elif top_survival > 0.9 and bottom_survival <= 0.05:
                                    # Top is at very top, bottom is at axis (both could be noise)
                                    # Prefer top if it's dense (likely actual curve at high survival)
                                    if not top_is_sparse:
                                        curve_ys.append((i, y_top, 'high_survival_curve'))
                                    elif not bottom_is_sparse:
                                        curve_ys.append((i, y_bottom, 'low_survival_curve'))
                                    else:
                                        # Both sparse - prefer top (higher survival, safer for KM)
                                        curve_ys.append((i, y_top, 'gap_top_both_sparse'))
                                elif top_survival > 0.5 and top_survival <= 0.9 and bottom_survival < 0.3:
                                    # Top cluster is in reasonable mid-range, bottom is noise
                                    curve_ys.append((i, y_top, 'noise_filtered_bottom'))
                                elif bottom_survival > 0.5 and top_survival > 0.95:
                                    # Top is at very top (possibly axis), bottom is curve
                                    curve_ys.append((i, y_bottom, 'noise_filtered_top'))
                                else:
                                    # Use top cluster (higher survival, safer for KM)
                                    curve_ys.append((i, y_top, 'gap_top'))
                            # Crosses extend only a few pixels, not huge gaps
                            elif top_size <= 2 and bottom_size >= 4:
                                # Top cluster is small (cross top), bottom is curve
                                curve_ys.append((i, y_bottom, 'cross_above'))
                            elif bottom_size <= 2 and top_size >= 4:
                                # Bottom cluster is small (cross bottom), top is curve
                                curve_ys.append((i, y_top, 'cross_below'))
                            elif gap_size > 20:
                                # Significant gap between clusters (> 20 pixels)
                                # This could be curve + text annotation or curve + cross marker
                                #
                                # Simple strategy: use continuity with previous points
                                # The cluster closer to the previous y is more likely the curve

                                top_survival = 1.0 - y_top / ph_ext

                                if top_survival > 0.85:
                                    # Top cluster is at very high survival (>85%) - likely text/legend
                                    curve_ys.append((i, y_bottom, 'gap_skip_text_top'))
                                elif all_curve_points[i]:
                                    # Use continuity - pick cluster closest to previous y
                                    _, prev_y = all_curve_points[i][-1]
                                    prev_y_local = prev_y - py_ext

                                    dist_to_top = abs(y_top - prev_y_local)
                                    dist_to_bottom = abs(y_bottom - prev_y_local)

                                    if dist_to_top <= dist_to_bottom:
                                        curve_ys.append((i, y_top, 'gap_continuity_top'))
                                    else:
                                        curve_ys.append((i, y_bottom, 'gap_continuity_bottom'))
                                else:
                                    # No previous point - prefer top (higher survival, typical for KM start)
                                    curve_ys.append((i, y_top, 'gap_prefer_top'))
                            elif all_curve_points[i]:
                                # Small gap - use continuity with previous point
                                _, prev_y = all_curve_points[i][-1]
                                prev_y_local = prev_y - py_ext

                                dist_to_top = abs(y_top - prev_y_local)
                                dist_to_bottom = abs(y_bottom - prev_y_local)

                                if dist_to_top <= dist_to_bottom:
                                    curve_ys.append((i, y_top, 'continuity_top'))
                                else:
                                    curve_ys.append((i, y_bottom, 'continuity_bottom'))
                            else:
                                # No previous point - prefer top cluster (higher survival)
                                # This is safer for KM curves which start at survival=1.0
                                curve_ys.append((i, y_top, 'default_top'))
                        else:
                            # Continuous vertical segment without gaps
                            # For KM curves, always use the TOPMOST pixel because:
                            # 1. The actual curve line is at the top
                            # 2. Lower pixels could be anti-aliasing, line thickness, or crosses
                            # 3. KM curves are step functions - horizontal segments with drops
                            #
                            # The step shape will be captured naturally as we move across x:
                            # - Before step: top pixel at high survival
                            # - At step: top pixel transitions to lower survival
                            # - After step: top pixel at low survival
                            y = int(np.min(y_positions))
                            curve_ys.append((i, y, 'topmost'))

        # Add detected points - each curve only gets points where its color was detected
        # NO overlap handling that assigns one curve's y to another curve
        # This prevents curves from "pulling" toward each other when close
        if len(curve_ys) >= 1:
            # Add all detected points (including both top and bottom of steps)
            for entry in curve_ys:
                curve_idx = entry[0]
                y = entry[1]
                # entry[2] is the type (single, cluster, step_top, step_bottom, etc.)

                img_x = px_ext + x
                img_y = py_ext + y
                all_curve_points[curve_idx].append((img_x, img_y))

            # NOTE: We deliberately do NOT add points for curves that weren't detected
            # at this x position. The old overlap handling was incorrectly assigning
            # one curve's y-value to another when curves were close, causing the
            # upper curve (cyan) to be pulled down toward the lower curve (magenta).

    # Handle curves that start late due to overlap at the beginning
    # Find the leftmost x where ANY curve is detected
    min_x = float('inf')
    min_y_at_start = {}  # curve_idx -> y at the start

    for curve_idx, points in enumerate(all_curve_points):
        if points:
            first_x, first_y = points[0]
            if first_x < min_x:
                min_x = first_x
            min_y_at_start[curve_idx] = first_y

    # For curves that start later, check if they should start at the same position
    # as curves that are detected earlier (at the top of the plot for KM curves)
    if min_x < float('inf'):
        # Find the topmost y at the start (highest survival = lowest y pixel)
        top_y = min(min_y_at_start.values()) if min_y_at_start else py_ext

        for curve_idx, points in enumerate(all_curve_points):
            if points:
                first_x, first_y = points[0]
                # If this curve starts significantly after min_x
                if first_x - min_x > 20:
                    # And if the first detected point is near the top (high survival)
                    # Then the curve likely starts at the top from the beginning
                    if first_y - top_y < 15:  # Within 15 pixels of top
                        # Prepend points from min_x to first_x at the top_y
                        prepend_points = []
                        for x in range(int(min_x), int(first_x)):
                            prepend_points.append((x, top_y))
                        all_curve_points[curve_idx] = prepend_points + points

    # Bridge gaps caused by text annotations overlaying curves
    # For each curve, detect gaps in x coverage and fill with horizontal segments
    for curve_idx, points in enumerate(all_curve_points):
        if len(points) > 10:
            bridged = _bridge_curve_gaps(points, max_gap_pixels=200)
            all_curve_points[curve_idx] = bridged

    return [(points, colors[i]) for i, points in enumerate(all_curve_points)]


def trace_color_curves(
    img: np.ndarray,
    plot_bounds: Tuple[int, int, int, int],
    pixel_to_coord: Callable,
    expected_curves: int = 2
) -> List[dict]:
    """
    Detect and trace all colored curves in an image.

    Args:
        img: BGR image
        plot_bounds: (x, y, width, height) of plot area
        pixel_to_coord: Function to convert (px, py) to (time, survival)
        expected_curves: Expected number of curves

    Returns:
        List of curve data dictionaries
    """
    # Detect colors
    colors = detect_curve_colors(img, max_colors=expected_curves)

    if not colors:
        return []

    curves = []

    for i, color_info in enumerate(colors):
        # Extract pixel points
        pixel_points = extract_color_curve(img, color_info, plot_bounds)

        if len(pixel_points) < 10:
            continue

        # Convert to coordinates
        coord_points = []
        for px, py in pixel_points:
            t, s = pixel_to_coord(px, py)
            # Clamp survival to valid range
            s = max(0.0, min(1.0, s))
            coord_points.append((t, s))

        # Sort by time
        coord_points.sort(key=lambda p: p[0])

        curves.append({
            'name': f"{color_info['name']}_{i+1}",
            'style': color_info['name'],
            'color_rgb': color_info['rgb'],
            'raw_points': coord_points,
            'clean_points': coord_points,  # Will be cleaned later
            'confidence': 0.85
        })

    return curves


class ColorCurveDetector:
    """Detects and extracts curves based on color."""

    def __init__(self, img: np.ndarray, plot_bounds: Tuple[int, int, int, int]):
        """
        Initialize detector.

        Args:
            img: BGR image
            plot_bounds: (x, y, width, height) of plot area
        """
        self.img = img
        self.plot_bounds = plot_bounds
        self.colors = []
        self.curves = []
        self.at_risk_data = {}  # Dict mapping curve name -> at-risk data
        self.annotation_points = {}  # Dict mapping curve name -> annotation points
        self.time_max = 24.0  # Maximum time for the x-axis

    def set_at_risk_data(self, curve_name: str, times: List[float], counts: List[int]):
        """
        Set at-risk table data for a curve.

        This data is used to correct/interpolate curves when detection fails.

        Args:
            curve_name: Name of the curve (e.g., 'gray', 'Pre-ICI')
            times: List of time points
            counts: List of at-risk counts at each time point
        """
        self.at_risk_data[curve_name.lower()] = {
            'times': times,
            'counts': counts
        }

    def set_annotation_points(self, curve_name: str, points: List[Tuple[float, float]]):
        """
        Set annotation-based survival points for a curve.

        These are survival values read from text annotations in the image
        (e.g., "34% (26-44)" at t=6 means survival=0.34 at time 6).

        Args:
            curve_name: Name of the curve
            points: List of (time, survival) tuples from annotations
        """
        self.annotation_points[curve_name.lower()] = points

    def set_time_max(self, time_max: float):
        """Set the maximum time value for the x-axis."""
        self.time_max = time_max

    def detect_all_curves(self, expected_count: int = 2, debug_dir: str = None) -> List[ColorCurve]:
        """
        Detect all colored curves with overlap handling.

        Args:
            expected_count: Expected number of curves
            debug_dir: Directory to save debug images

        Returns:
            List of ColorCurve objects
        """
        self.colors = detect_curve_colors(self.img, max_colors=expected_count)

        if debug_dir:
            self._save_debug_images(debug_dir)

        if not self.colors:
            return []

        # Use overlap-aware extraction for multiple curves
        curves_data = extract_curves_with_overlap_handling(
            self.img, self.colors, self.plot_bounds
        )

        self.curves = []
        for i, (pixel_points, color_info) in enumerate(curves_data):
            if len(pixel_points) < 10:
                continue

            # Create mask for this color
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])

            curve = ColorCurve(
                name=f"{color_info['name']}_{i+1}",
                color_rgb=color_info['rgb'],
                mask=mask,
                points=pixel_points,
                confidence=0.85
            )
            self.curves.append(curve)

        # If we detected more curves than expected, filter to keep best ones
        if len(self.curves) > expected_count and expected_count > 0:
            self.curves = self._filter_best_curves(self.curves, expected_count)

        return self.curves

    def _filter_best_curves(self, curves: List[ColorCurve], keep_count: int) -> List[ColorCurve]:
        """
        Filter to keep only the best curves based on curve-like characteristics.

        Scoring criteria:
        - Number of points (more is better for a real curve)
        - X coverage (should span most of plot width)
        - Y variation (real curves show survival decline, not flat lines)

        Args:
            curves: List of detected curves
            keep_count: Number of curves to keep

        Returns:
            Filtered list of best curves
        """
        if len(curves) <= keep_count:
            return curves

        px, py, pw, ph = self.plot_bounds
        scored_curves = []

        for curve in curves:
            if not curve.points:
                scored_curves.append((0, curve))
                continue

            # Get x and y ranges
            xs = [p[0] for p in curve.points]
            ys = [p[1] for p in curve.points]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Score components
            point_count = len(curve.points)
            x_coverage = (x_max - x_min) / pw if pw > 0 else 0  # 0-1, higher is better
            y_variation = (y_max - y_min) / ph if ph > 0 else 0  # 0-1, higher is better for KM curves

            # Combined score: favor curves with many points, good x coverage, and y variation
            # Real KM curves should have lots of points and show survival decline
            score = point_count * 0.4 + x_coverage * 1000 * 0.3 + y_variation * 1000 * 0.3

            scored_curves.append((score, curve))

        # Sort by score descending and keep top ones
        scored_curves.sort(key=lambda x: x[0], reverse=True)
        best_curves = [curve for _, curve in scored_curves[:keep_count]]

        # Log which curves were kept/removed
        kept_names = [c.name for c in best_curves]
        removed = [c.name for _, c in scored_curves[keep_count:]]
        if removed:
            print(f"    Filtered curves: keeping {kept_names}, removed {removed}")

        return best_curves

    def extract_curve_points(
        self,
        curve: ColorCurve,
        pixel_to_coord: Callable
    ) -> List[Tuple[float, float]]:
        """
        Convert curve pixel points to coordinates.

        IMPORTANT: The curve.points were already carefully extracted and filtered
        by extract_curves_with_overlap_handling(), which uses _isolate_main_curve_component
        and sophisticated overlap detection. We should NOT apply additional filtering
        here as it can remove valid topmost pixels.

        Args:
            curve: ColorCurve object
            pixel_to_coord: Conversion function

        Returns:
            List of (time, survival) tuples
        """
        # Use curve.points directly - they were already filtered during extraction
        # by extract_curves_with_overlap_handling() which applied:
        # 1. Plot bounds filtering
        # 2. _isolate_main_curve_component for connected component filtering
        # 3. Overlap handling to properly assign points when curves are close
        #
        # Applying _filter_curve_points here would use a DIFFERENT unfiltered mask
        # and could incorrectly remove valid topmost pixels.

        # Just apply plot bounds check for safety
        px, py, pw, ph = self.plot_bounds
        margin = 5

        coord_points = []
        for x, y in curve.points:
            # Skip points outside plot bounds
            if x < px - margin or x > px + pw + margin:
                continue
            if y < py - margin or y > py + ph + margin:
                continue

            t, s = pixel_to_coord(x, y)
            s = max(0.0, min(1.0, s))
            coord_points.append((t, s))

        coord_points.sort(key=lambda p: p[0])

        # For gray curves, apply at-risk based correction if data is available
        curve_name_lower = curve.name.lower()
        is_gray = 'gray' in curve_name_lower

        if is_gray:
            # Check if we have at-risk data or annotations for this curve
            at_risk = None
            annotations = None

            # Look for matching at-risk data
            for key in self.at_risk_data:
                if key in curve_name_lower or curve_name_lower in key:
                    at_risk = self.at_risk_data[key]
                    break
            # Also check for 'pre-ici' which is the gray curve in NSQ OS
            if not at_risk and 'pre-ici' in self.at_risk_data:
                at_risk = self.at_risk_data['pre-ici']

            # Look for matching annotations
            for key in self.annotation_points:
                if key in curve_name_lower or curve_name_lower in key:
                    annotations = self.annotation_points[key]
                    break
            if not annotations and 'pre-ici' in self.annotation_points:
                annotations = self.annotation_points['pre-ici']

            if at_risk or annotations:
                coord_points = correct_curve_with_at_risk_interpolation(
                    coord_points,
                    at_risk_data=at_risk,
                    annotation_points=annotations,
                    time_max=self.time_max,
                    curve_name=curve.name
                )

        return coord_points

    def _filter_curve_points(self, curve: ColorCurve) -> List[Tuple[int, int]]:
        """
        Filter curve points to remove noise from text, legend, and axis.

        Uses connected component analysis to identify the main curve region.
        IMPORTANT: Only keeps points within the plot bounds to exclude legend/labels.

        Args:
            curve: ColorCurve object

        Returns:
            Filtered list of (x, y) pixel coordinates
        """
        if not curve.points:
            return []

        px, py, pw, ph = self.plot_bounds
        margin = 5  # Small margin for edge detection

        # STRICT bounds - only include points within the plot area
        # This prevents legend text and axis labels from being included
        px_start = max(0, px - margin)
        px_end = min(curve.mask.shape[1], px + pw + margin)

        # Create a mask for connected component analysis
        mask = curve.mask.copy()

        # Restrict to plot area (y) and curve extent (x) + margin
        plot_mask = np.zeros_like(mask)
        py_start = max(0, py - margin)
        py_end = min(mask.shape[0], py + ph + margin)
        plot_mask[py_start:py_end, px_start:px_end] = mask[py_start:py_end, px_start:px_end]

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            plot_mask, connectivity=8
        )

        if num_labels <= 1:
            return curve.points

        # Keep all components that look like curve segments (not just the best one)
        # This handles curves that may be broken into multiple disconnected components
        actual_width = px_end - px_start
        min_component_width = max(10, actual_width * 0.02)  # At least 2% of plot width

        curve_components = []

        for label in range(1, num_labels):  # Skip background (0)
            x_start, y_start, w, h, area = stats[label]
            cx, cy = centroids[label]

            # Skip tiny components (likely noise)
            if area < 30:
                continue

            # Skip very narrow components (likely text or tick marks)
            if w < min_component_width:
                continue

            # Skip components that span almost full height AND are narrow
            # (these are likely axis lines, not curves)
            # Survival curves can span full height but should have significant width
            height_ratio = h / ph
            width_ratio = w / actual_width if actual_width > 0 else 0
            if height_ratio > 0.9 and width_ratio < 0.3:
                continue

            # Skip components in the bottom 20% unless they have significant width
            if cy > py + ph * 0.8 and w < actual_width * 0.1:
                continue

            # This component looks like a curve segment - keep it
            curve_components.append(label)

        if not curve_components:
            # No good components - fall back to keeping largest
            areas = [(stats[label][4], label) for label in range(1, num_labels)]
            if areas:
                areas.sort(reverse=True)
                curve_components = [areas[0][1]]
            else:
                return curve.points

        # Filter points to include those in ANY curve component
        # AND that are within the strict plot bounds (excludes legend/labels)
        filtered_points = []
        for x, y in curve.points:
            # STRICT: Only include points within plot bounds
            if x < px - margin or x > px + pw + margin:
                continue  # Outside x bounds - likely legend or labels
            if y < py - margin or y > py + ph + margin:
                continue  # Outside y bounds

            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                if labels[y, x] in curve_components:
                    filtered_points.append((x, y))

        # If filtering removed too many points, fall back to bounds-filtered original
        if len(filtered_points) < 10:
            # Apply strict bounds to original points
            bounded_points = [(x, y) for x, y in curve.points
                            if px - margin <= x <= px + pw + margin
                            and py - margin <= y <= py + ph + margin]
            return self._filter_by_survival_pattern(bounded_points)

        return filtered_points

    def _filter_by_survival_pattern(
        self,
        points: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Filter points based on expected survival curve pattern.

        Survival curves should:
        - Start at the top (high y pixel = low survival - wait, reversed!)
        - Actually: low y pixel = high survival (top of image)
        - Decrease monotonically (y should increase or stay same)

        Args:
            points: List of (x, y) pixel coordinates

        Returns:
            Filtered points
        """
        if not points:
            return []

        px, py, pw, ph = self.plot_bounds

        # Sort by x
        sorted_points = sorted(points, key=lambda p: p[0])

        # Find the expected y range for a survival curve
        # At the start (low x), y should be near py (top)
        # Throughout, y should not jump dramatically

        filtered = []
        prev_y = None
        max_y_jump = ph * 0.3  # Max 30% jump in one step

        for x, y in sorted_points:
            # Only include points within reasonable plot area
            if y < py - 10 or y > py + ph + 10:
                continue

            if prev_y is not None:
                # Check for unreasonable jumps
                if y - prev_y > max_y_jump:  # Sudden drop in survival
                    # This might be a jump to the axis - skip
                    continue

            filtered.append((x, y))
            prev_y = y

        return filtered

    def get_debug_image(self) -> np.ndarray:
        """Get visualization of detected curves."""
        vis = self.img.copy()

        colors_bgr = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
        ]

        for i, curve in enumerate(self.curves):
            color = colors_bgr[i % len(colors_bgr)]

            # Draw points
            for j, (x, y) in enumerate(curve.points):
                cv2.circle(vis, (x, y), 2, color, -1)
                if j > 0:
                    prev_x, prev_y = curve.points[j-1]
                    cv2.line(vis, (prev_x, prev_y), (x, y), color, 1)

            # Label
            if curve.points:
                px, py = curve.points[0]
                cv2.putText(vis, curve.name, (px, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis

    def _save_debug_images(self, debug_dir: str):
        """Save debug visualizations."""
        import os
        os.makedirs(debug_dir, exist_ok=True)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        for i, color_info in enumerate(self.colors):
            mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])
            cv2.imwrite(f"{debug_dir}/debug_color_{color_info['name']}_mask.png", mask)

    def ai_reconstruct_curve(
        self,
        curve_points: List[Tuple[float, float]],
        curve_color: str,
        time_max: float = 24.0,
        ai_config=None,
        quiet: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Use AI to suggest curve path in missing regions.

        When detected curve coverage is low (<80% of expected range),
        this method uses the vision model to trace the likely curve path.

        Args:
            curve_points: List of (time, survival) tuples from detection
            curve_color: Name of the curve color (e.g., 'gray', 'orange')
            time_max: Maximum time value for the x-axis
            ai_config: Optional AIConfig for AI detection
            quiet: Suppress progress messages

        Returns:
            Enhanced list of (time, survival) tuples with AI-suggested points
        """
        if not HAS_OLLAMA or not HAS_AI_CONFIG:
            return curve_points

        if not curve_points:
            return curve_points

        # Sort points by time
        sorted_points = sorted(curve_points, key=lambda p: p[0])
        times = [p[0] for p in sorted_points]
        survivals = [p[1] for p in sorted_points]

        max_detected_time = max(times) if times else 0
        coverage = max_detected_time / time_max if time_max > 0 else 1.0

        # Only use AI if coverage is low
        if coverage >= 0.8:
            if not quiet:
                print(f"  Curve coverage good ({coverage:.0%}), skipping AI reconstruction")
            return curve_points

        if not quiet:
            print(f"  Low curve coverage ({coverage:.0%}), attempting AI reconstruction...")

        # Create annotated image for AI
        annotated = self._create_reconstruction_image(curve_points, curve_color)
        if annotated is None:
            return curve_points

        # Get AI suggestions
        config = ai_config if ai_config else AIConfig.from_environment()
        waypoints = self._get_ai_curve_waypoints(
            annotated, curve_color, max_detected_time, time_max, config, quiet
        )

        if not waypoints:
            return curve_points

        # Merge detected points with AI waypoints
        return self._merge_with_waypoints(curve_points, waypoints, max_detected_time, time_max)

    def _create_reconstruction_image(
        self,
        curve_points: List[Tuple[float, float]],
        curve_color: str
    ) -> Optional[np.ndarray]:
        """
        Create annotated image for AI reconstruction.

        Highlights the detected curve segments and marks missing regions.

        Args:
            curve_points: Detected points
            curve_color: Color name

        Returns:
            Annotated image or None
        """
        try:
            px, py, pw, ph = self.plot_bounds
            annotated = self.img.copy()

            # Find the last detected pixel x position
            # We need to convert curve coordinates back to pixels for visualization
            # This is approximate - just for visualization

            if curve_points:
                times = [p[0] for p in curve_points]
                max_t = max(times) if times else 0

                # Mark the detected region with a green line at the bottom
                # (just for context, AI will see the actual curves)
                cv2.putText(
                    annotated,
                    f"Detected {curve_color} curve up to t={max_t:.1f}",
                    (px + 10, py + ph - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )

            return annotated

        except Exception:
            return None

    def _get_ai_curve_waypoints(
        self,
        annotated_image: np.ndarray,
        curve_color: str,
        last_detected_time: float,
        time_max: float,
        config,
        quiet: bool
    ) -> List[Tuple[float, float]]:
        """
        Ask AI to trace curve path in missing regions.

        Args:
            annotated_image: Image with annotations
            curve_color: Color of the curve
            last_detected_time: Last time point detected
            time_max: Maximum time for x-axis
            config: AIConfig
            quiet: Suppress messages

        Returns:
            List of (time, survival) waypoints from AI
        """
        if not HAS_OLLAMA:
            return []

        try:
            client = ollama.Client(host=config.host)

            # Calculate time points we need
            step = (time_max - last_detected_time) / 4
            time_points = []
            t = last_detected_time + step
            while t <= time_max:
                time_points.append(round(t, 1))
                t += step

            if not time_points:
                return []

            prompt = f"""Analyze this Kaplan-Meier survival plot.
The {curve_color} curve is partially visible but may be faded or missing after t={last_detected_time:.0f}.

Based on the visible trajectory of the {curve_color} curve, estimate the survival probability at these time points:
{', '.join([f't={t}' for t in time_points])}

Respond in this exact format (one line per time point):
t=<time>: survival=<value between 0 and 1>

Example:
t=18: survival=0.35
t=21: survival=0.28
t=24: survival=0.22
"""

            # Save temp image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                cv2.imwrite(f.name, annotated_image)
                image_path = f.name

            response = client.chat(
                model=config.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                ],
                options={'temperature': 0.1}
            )

            response_text = response['message']['content']

            if not quiet:
                print(f"    AI response: {response_text[:200]}...")

            # Parse waypoints from response
            waypoints = []
            for line in response_text.split('\n'):
                match = re.search(r't\s*=\s*([\d.]+).*?survival\s*=\s*([\d.]+)', line, re.IGNORECASE)
                if match:
                    t = float(match.group(1))
                    s = float(match.group(2))
                    # Normalize if percentage
                    if s > 1:
                        s /= 100
                    waypoints.append((t, max(0, min(1, s))))

            if not quiet:
                print(f"    AI suggested {len(waypoints)} waypoints: {waypoints}")

            return waypoints

        except Exception as e:
            if not quiet:
                print(f"    AI reconstruction error: {e}")
            return []

    def _merge_with_waypoints(
        self,
        curve_points: List[Tuple[float, float]],
        waypoints: List[Tuple[float, float]],
        last_detected_time: float,
        time_max: float
    ) -> List[Tuple[float, float]]:
        """
        Merge detected curve points with AI waypoints using interpolation.

        Args:
            curve_points: Original detected points
            waypoints: AI-suggested waypoints
            last_detected_time: Last detected time
            time_max: Maximum time

        Returns:
            Merged list of points
        """
        if not waypoints or not HAS_SCIPY:
            return curve_points

        # Get the survival value at the last detected time
        sorted_points = sorted(curve_points, key=lambda p: p[0])
        times = [p[0] for p in sorted_points]
        survivals = [p[1] for p in sorted_points]

        # Find survival at last detected time
        last_survival = None
        for t, s in sorted_points:
            if t <= last_detected_time:
                last_survival = s

        if last_survival is None and sorted_points:
            last_survival = sorted_points[-1][1]

        # Build interpolation from last detected point through waypoints
        interp_times = [last_detected_time] if last_survival else []
        interp_survivals = [last_survival] if last_survival else []

        for t, s in waypoints:
            if t > last_detected_time:
                interp_times.append(t)
                interp_survivals.append(s)

        if len(interp_times) < 2:
            return curve_points

        try:
            f = scipy_interpolate.interp1d(
                interp_times, interp_survivals,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )

            # Generate dense points in the missing region
            result = list(curve_points)
            dt = time_max / 1000
            t = last_detected_time + dt
            while t <= time_max:
                s = float(f(t))
                s = max(0.0, min(1.0, s))
                result.append((t, s))
                t += dt

            return sorted(result, key=lambda p: p[0])

        except Exception:
            return curve_points


def ai_reconstruct_all_curves(
    detector: ColorCurveDetector,
    curves: List[ColorCurve],
    pixel_to_coord: Callable,
    time_max: float = 24.0,
    ai_config=None,
    quiet: bool = True
) -> List[List[Tuple[float, float]]]:
    """
    Convenience function to apply AI reconstruction to all curves.

    Args:
        detector: ColorCurveDetector instance
        curves: List of ColorCurve objects
        pixel_to_coord: Coordinate conversion function
        time_max: Maximum time value
        ai_config: Optional AIConfig
        quiet: Suppress messages

    Returns:
        List of enhanced coordinate lists for each curve
    """
    results = []

    for curve in curves:
        # Convert to coordinates
        coord_points = detector.extract_curve_points(curve, pixel_to_coord)

        # Check coverage and apply AI reconstruction if needed
        enhanced = detector.ai_reconstruct_curve(
            coord_points, curve.name, time_max, ai_config, quiet
        )

        results.append(enhanced)

    return results
