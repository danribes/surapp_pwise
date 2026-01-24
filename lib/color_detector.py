"""
Color-based curve detection for KM plots with colored curves.

Detects curves by their color (e.g., orange, blue, red, green) rather than
line style (solid/dashed).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable


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
        hue_range = 12

        # Create HSV bounds
        # Use lower saturation threshold (15) to capture faded colors in tails
        sat_min = 15
        val_min = 40
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

    # Extend plot bounds
    px, py, pw, ph = plot_bounds
    margin = 10
    px_ext = max(0, px - margin)
    py_ext = max(0, py - margin)
    pw_ext = min(width - px_ext, pw + 2 * margin)
    ph_ext = min(height - py_ext, ph + 2 * margin)

    # Create masks for each color
    masks = []
    for color_info in colors:
        mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])
        mask_roi = mask[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext]

        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
        masks.append(mask_roi)

    # Extract points for each curve
    all_curve_points = [[] for _ in colors]

    # For each x position, find y positions for each color
    for x in range(pw_ext):
        curve_ys = []  # (curve_idx, y_position) for each detected curve at this x

        for i, mask_roi in enumerate(masks):
            col = mask_roi[:, x]
            y_positions = np.where(col > 0)[0]

            if len(y_positions) > 0:
                y = int(np.median(y_positions))
                curve_ys.append((i, y))

        # Check for overlaps - if one curve is detected and others are not,
        # but a previously detected curve was at a similar y, it might be overlapping
        if len(curve_ys) >= 1:
            detected_y = curve_ys[0][1] if curve_ys else None

            for i, (curve_idx, y) in enumerate(curve_ys):
                img_x = px_ext + x
                img_y = py_ext + y
                all_curve_points[curve_idx].append((img_x, img_y))

            # For curves not detected at this x, check if they might be truly overlapping
            # ONLY add points when curves are actually touching (within 3 pixels)
            # Do NOT extrapolate or invent points - only use detected pixels
            detected_indices = {idx for idx, _ in curve_ys}
            for curve_idx in range(len(colors)):
                if curve_idx not in detected_indices:
                    # Check if this curve was recently detected and might be overlapping
                    if all_curve_points[curve_idx]:
                        last_x, last_y = all_curve_points[curve_idx][-1]
                        gap = px_ext + x - last_x

                        # Only consider overlap if gap is very small (consecutive pixels)
                        if gap <= 3:
                            # Only assume TRUE overlap if curves are within 3 pixels vertically
                            for _, detected_y in curve_ys:
                                actual_detected_y = py_ext + detected_y
                                if abs(actual_detected_y - last_y) <= 3:
                                    # Curves are truly overlapping - use detected y
                                    all_curve_points[curve_idx].append(
                                        (px_ext + x, actual_detected_y)
                                    )
                                    break
                            # If not overlapping, don't add any point - curve simply ends here

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

        return self.curves

    def extract_curve_points(
        self,
        curve: ColorCurve,
        pixel_to_coord: Callable
    ) -> List[Tuple[float, float]]:
        """
        Convert curve pixel points to coordinates.

        Args:
            curve: ColorCurve object
            pixel_to_coord: Conversion function

        Returns:
            List of (time, survival) tuples
        """
        coord_points = []
        for px, py in curve.points:
            t, s = pixel_to_coord(px, py)
            s = max(0.0, min(1.0, s))
            coord_points.append((t, s))

        coord_points.sort(key=lambda p: p[0])
        return coord_points

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
