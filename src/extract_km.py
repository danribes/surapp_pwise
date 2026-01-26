#!/usr/bin/env python3
"""
Kaplan-Meier Curve Extractor - Simple All-in-One Script

This script extracts survival curve data from Kaplan-Meier plot images.
It detects solid and dashed curves, calibrates the axes, and exports
the extracted coordinates to CSV files.

Usage:
    python extract_km.py <image_path> [--time-max TIME] [--curves N]

Example:
    python extract_km.py my_km_plot.png --time-max 40 --curves 2

Output:
    Creates a 'results/' directory with:
    - all_curves.csv: Combined data from all curves
    - curve_*.csv: Individual curve data files
    - extracted_curves.png: Visualization of extracted curves
    - debug_*.png: Debug images showing detection steps
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies
def check_dependencies():
    """Check that required packages are installed."""
    missing = []

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import local modules
try:
    from lib import (
        LineStyleDetector, AxisCalibrator, is_grayscale_image,
        ColorCurveDetector, is_color_image
    )
    # Skeleton-based extraction (text removal + skeletonization)
    try:
        from lib.color_detector import (
            detect_curve_colors,
            extract_curves_with_skeleton,
            extract_curves_grayscale_skeleton
        )
        from lib.text_remover import remove_all_text
        _SKELETON_AVAILABLE = True
    except ImportError:
        _SKELETON_AVAILABLE = False
    # At-risk table extraction (optional - requires easyocr or pytesseract)
    try:
        from lib import AtRiskExtractor, AtRiskData
        _AT_RISK_AVAILABLE = True
    except ImportError:
        _AT_RISK_AVAILABLE = False
    # AI service for enhanced detection (optional - requires ollama)
    try:
        from lib.ai_service import get_ai_service
        _AI_SERVICE_AVAILABLE = True
    except ImportError:
        _AI_SERVICE_AVAILABLE = False
    # Validation module
    try:
        from lib.validator import validate_extraction, ValidationStatus, run_dense_validation
        _VALIDATION_AVAILABLE = True
        _DENSE_VALIDATION_AVAILABLE = True
    except ImportError:
        _VALIDATION_AVAILABLE = False
        _DENSE_VALIDATION_AVAILABLE = False
except ImportError as e:
    print(f"ERROR: Could not import local modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def _detect_time_max_from_axis(img: np.ndarray, plot_x: int, plot_w: int, plot_y: int, plot_h: int, show_progress: bool = False) -> float:
    """
    Detect the maximum time value from X-axis labels.

    Looks for numeric labels along the X-axis and returns the maximum value.
    Uses multiple OCR passes with different preprocessing to maximize detection.

    Args:
        img: BGR image
        plot_x, plot_w: X-axis bounds
        plot_y, plot_h: Y-axis bounds (used to find label region)
        show_progress: Print debug info

    Returns:
        Detected time_max or None if detection fails
    """
    try:
        import pytesseract
    except ImportError:
        return None

    import re

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # X-axis labels are typically just below the plot area
    label_y_start = plot_y + plot_h
    label_y_end = min(height, label_y_start + 30)

    if label_y_end <= label_y_start:
        return None

    # Extract the X-axis label region
    label_region = gray[label_y_start:label_y_end, plot_x:plot_x + plot_w]

    if label_region.size == 0:
        return None

    # Collect all detected numbers from multiple passes
    all_numbers = set()

    # Preprocess for OCR - try multiple thresholds
    for threshold_method in [cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV]:
        try:
            if threshold_method == cv2.THRESH_BINARY_INV:
                _, binary = cv2.threshold(label_region, 127, 255, threshold_method)
            else:
                _, binary = cv2.threshold(label_region, 0, 255, threshold_method)

            # Try multiple PSM modes
            for psm in [6, 11, 12]:
                try:
                    text = pytesseract.image_to_string(
                        binary,
                        config=f'--psm {psm}'
                    )
                    # Extract all numbers (including multi-digit)
                    numbers = re.findall(r'\d+', text)
                    for n in numbers:
                        try:
                            val = int(n)
                            if 0 < val <= 100:  # Reasonable range for time in months
                                all_numbers.add(val)
                        except ValueError:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    # Also try the right portion for better max detection
    right_region = label_region[:, -int(plot_w * 0.2):]  # Last 20%
    if right_region.size > 0:
        try:
            _, right_binary = cv2.threshold(right_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                right_binary,
                config='--psm 7'
            )
            numbers = re.findall(r'\d+', text)
            for n in numbers:
                try:
                    val = int(n)
                    if 0 < val <= 100:
                        all_numbers.add(val)
                except ValueError:
                    pass
        except Exception:
            pass

    if all_numbers:
        time_max = max(all_numbers)

        # Check if detected numbers form a reasonable sequence (sanity check)
        # For KM plots, X-axis typically goes 0, 3, 6, 9... or 0, 5, 10, 15...
        # If we have multiple numbers, validate the max makes sense
        if len(all_numbers) >= 3:
            sorted_nums = sorted(all_numbers)
            # The max should be at least 2x the median for typical KM plots
            median_num = sorted_nums[len(sorted_nums) // 2]
            if time_max < median_num * 1.5:
                # Max seems too low - might have missed higher numbers
                # Try to extrapolate from the pattern
                if len(sorted_nums) >= 2:
                    step = sorted_nums[1] - sorted_nums[0]
                    if step > 0:
                        # Estimate max based on step size and plot width
                        estimated_max = step * (plot_w // 30)  # Rough estimate
                        if estimated_max > time_max and estimated_max <= 100:
                            time_max = estimated_max

        if show_progress:
            print(f"  Detected time_max from axis labels: {time_max}")
        return float(time_max)

    return None


def _find_panel_for_color_curves(img: np.ndarray, initial_plot_bounds: tuple, show_progress: bool = False) -> tuple:
    """
    For multi-panel images, find the correct panel containing the colored curves.

    This function detects if colored curves are in a different panel than the
    initially calibrated one, and returns the correct plot bounds AND time range.

    IMPORTANT: The Y-axis LINE and the t=0 position can be different!
    - The Y-axis line is a dark vertical line
    - The t=0 position is where the "0" label is on the X-axis
    - There's often a gap between them (axis line vs plot area)

    Args:
        img: BGR image
        initial_plot_bounds: (x, y, w, h) from initial calibration
        show_progress: Print debug info

    Returns:
        Tuple of (plot_x, plot_y, plot_w, plot_h, time_max) or
        (plot_x, plot_y, plot_w, plot_h) if time_max couldn't be determined
    """
    height, width = img.shape[:2]
    px, py, pw, ph = initial_plot_bounds

    # Check if this could be a multi-panel image (wide aspect ratio)
    aspect_ratio = width / height
    if aspect_ratio < 1.5:
        # Not likely a side-by-side two-panel image
        return initial_plot_bounds

    mid_x = width // 2

    # If the calibrated plot extends past the midpoint, it's probably correct
    if px + pw > mid_x + 50:
        return initial_plot_bounds

    # The calibrated plot is in the left half - check if there are curves in the right half
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if show_progress:
        print(f"  Multi-panel check: aspect_ratio={aspect_ratio:.2f}")

    # Find where colored curves actually start (this is the key!)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Create mask for colored pixels in the PLOT area only (not legends/labels)
    # Focus on the upper portion of the plot where survival is high
    plot_top = py
    plot_bottom = py + int(ph * 0.6)  # Top 60% of plot (where curves start at ~100%)

    colored_mask = (sat > 40) & (val > 40) & (val < 250)
    colored_mask[:plot_top, :] = 0  # Ignore above plot
    colored_mask[plot_bottom:, :] = 0  # Ignore bottom portion and legends

    # Search for curves starting from the middle of the image (right panel only)
    # The right panel's curves should start around x = mid_x + 100 to mid_x + 150
    search_start = mid_x + 50  # Well into the right panel

    # Find the leftmost x where there's a significant vertical run of colored pixels
    # Real curves have continuous vertical segments, not scattered noise
    curve_start_x = None
    for x in range(search_start, width):
        col = colored_mask[plot_top:plot_bottom, x]
        colored_rows = np.where(col > 0)[0]

        if len(colored_rows) >= 3:
            # Check if pixels are clustered (not scattered noise)
            if len(colored_rows) > 1:
                y_range = colored_rows[-1] - colored_rows[0]
                if y_range < 30:  # Clustered within 30 pixels vertically
                    curve_start_x = x
                    break
            else:
                curve_start_x = x
                break

    if curve_start_x is None:
        if show_progress:
            print(f"  No colored curves found in right half")
        return initial_plot_bounds

    if show_progress:
        print(f"  Colored curves start at x={curve_start_x}")

    # The t=0 position should be very close to where the curves start
    # (curves start at t~0 or slightly after)
    # Look for the X-axis "0" label near the curve start

    # Search for the first X-axis label (darkest region in the label area)
    label_y_start = py + ph - 5  # Just below the plot
    label_y_end = min(height, py + ph + 25)  # Label region

    # Look in the region around the curve start for axis labels
    search_start = max(0, curve_start_x - 30)
    search_end = min(width, curve_start_x + 20)

    if label_y_end > label_y_start and search_end > search_start:
        label_region = gray[label_y_start:label_y_end, search_start:search_end]
        if label_region.size > 0:
            # Find the darkest column (center of first label, likely "0")
            col_means = np.mean(label_region, axis=0)
            darkest_col = np.argmin(col_means)
            first_label_x = search_start + darkest_col

            if show_progress:
                print(f"  First X-axis label (t=0) at approximately x={first_label_x}")

            # Use the first label position as plot_x (where t=0 is)
            # This is more accurate than using the Y-axis line position
            new_plot_x = first_label_x

            # Calculate plot width to the right edge
            right_margin = int(width * 0.02)
            new_plot_w = width - right_margin - new_plot_x

            # Validate: the curve start should be at or slightly after plot_x
            if curve_start_x >= new_plot_x - 5:
                if show_progress:
                    print(f"  Using t=0 at x={new_plot_x}")

                # Detect time_max from X-axis labels
                time_max_detected = _detect_time_max_from_axis(img, new_plot_x, new_plot_w, py, ph, show_progress)
                if time_max_detected:
                    return (new_plot_x, py, new_plot_w, ph, time_max_detected)
                return (new_plot_x, py, new_plot_w, ph)

    # Fallback: use the curve start position as t=0
    # (curves should start very close to t=0 for KM plots)
    new_plot_x = curve_start_x - 2  # Small offset for t=0
    right_margin = int(width * 0.02)
    new_plot_w = width - right_margin - new_plot_x

    if show_progress:
        print(f"  Fallback: using curve start x={new_plot_x} as t=0")

    # Detect time_max from X-axis labels
    time_max_detected = _detect_time_max_from_axis(img, new_plot_x, new_plot_w, py, ph, show_progress)
    if time_max_detected:
        return (new_plot_x, py, new_plot_w, ph, time_max_detected)
    return (new_plot_x, py, new_plot_w, ph)


def _detect_curve_extent(detected_curves, calibration, show_progress: bool = False) -> float:
    """
    Detect the actual maximum X extent of the curves in data coordinates.

    This function analyzes the detected curve pixels to find where the curves
    actually end, rather than relying on axis labels which may extend beyond
    the actual data.

    Args:
        detected_curves: List of DetectedCurve objects with pixel points
        calibration: AxisCalibrationResult with coordinate mapping info
        show_progress: Print debug info

    Returns:
        The detected maximum time value where curves end, or None if detection fails
    """
    if not detected_curves or calibration is None:
        return None

    # Get the calibration parameters
    origin_x = calibration.origin[0]
    x_axis_end_x = calibration.x_axis_end[0]
    time_min, time_max_axis = calibration.x_data_range

    # Calculate pixels per time unit
    pixel_range = x_axis_end_x - origin_x
    time_range = time_max_axis - time_min
    if pixel_range <= 0 or time_range <= 0:
        return None

    # Find the maximum X pixel position across all curves
    max_x_pixel = origin_x  # Start at origin

    for curve in detected_curves:
        # ColorCurve has points directly
        if hasattr(curve, 'points') and curve.points:
            for px, py in curve.points:
                if px > max_x_pixel:
                    max_x_pixel = px

        # DetectedCurve has segments, each segment has pixels
        if hasattr(curve, 'segments') and curve.segments:
            for segment in curve.segments:
                if hasattr(segment, 'pixels') and segment.pixels:
                    for px, py in segment.pixels:
                        if px > max_x_pixel:
                            max_x_pixel = px
                # Also check x_range as a backup
                if hasattr(segment, 'x_range') and segment.x_range:
                    if segment.x_range[1] > max_x_pixel:
                        max_x_pixel = segment.x_range[1]

    # Convert max pixel to time coordinate
    # time = time_min + (pixel - origin_x) / pixel_range * time_range
    detected_time_max = time_min + (max_x_pixel - origin_x) / pixel_range * time_range

    # Round to a reasonable precision and add small margin
    # Round up to nearest integer or 0.5 to avoid cutting off data
    detected_time_max = np.ceil(detected_time_max * 2) / 2  # Round up to nearest 0.5

    # Ensure we don't exceed the axis range
    detected_time_max = min(detected_time_max, time_max_axis)

    if show_progress:
        print(f"  Curve extent detection: max pixel x={max_x_pixel}, time={detected_time_max:.1f}")

    return detected_time_max


def extract_km_curves(
    image_path: str,
    output_dir: str = None,
    time_max: float = None,
    expected_curves: int = 2,
    show_progress: bool = True,
    use_ai: bool = False,
    use_skeleton: bool = True
):
    """
    Extract Kaplan-Meier curves from an image.

    Args:
        image_path: Path to the KM plot image
        output_dir: Output directory (default: results/<image_name>_<timestamp>)
        time_max: Maximum time value on X-axis (auto-detected if not specified)
        expected_curves: Expected number of curves (default: 2)
        show_progress: Print progress messages
        use_ai: Use AI assistance for validation (default: False for faster extraction)
        use_skeleton: Use skeleton-based extraction for color images (default: True)
            This removes text via OCR and uses skeletonization for clean extraction.

    Returns:
        Dictionary with extraction results
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{image_path.stem}_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print("=" * 60)
        print("KAPLAN-MEIER CURVE EXTRACTOR")
        print("=" * 60)
        print(f"\nInput image: {image_path}")
        print(f"Output directory: {output_dir}")

    # Step 1: Load image
    if show_progress:
        print("\n[Step 1/4] Loading image...")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]
    if show_progress:
        print(f"  Image size: {width} x {height} pixels")
        grayscale = is_grayscale_image(img)
        print(f"  Image type: {'Grayscale' if grayscale else 'Color'}")

    # Save original image copy
    cv2.imwrite(str(output_dir / "original.png"), img)

    # Step 2: Calibrate axes using calibrator_v2
    # Two-step process: 1) Detect axis lines and labels, 2) Calculate calibration
    if show_progress:
        print("\n[Step 2/4] Calibrating axes...")

    # Check if AI service is available for enhanced detection
    ai_service = None
    if use_ai and _AI_SERVICE_AVAILABLE:
        ai_service = get_ai_service()
        if ai_service and show_progress:
            print("  AI assistance available (note: may be slow on CPU)")
    elif show_progress and not use_ai:
        print("  AI assistance disabled")

    # Use the new calibrator_v2 with clean 2-step architecture
    calibrator = AxisCalibrator(img)

    # Calibrate with time_max hint (or None for auto-detection)
    # The calibrator handles OCR, error correction, and extrapolation automatically
    calibration = calibrator.calibrate(time_max=time_max or 100.0, verbose=show_progress)

    # Use AI to validate/improve axis detection if available
    ai_axis_result = None
    if ai_service and calibration is not None:
        try:
            ai_axis_result = ai_service.detect_axes(str(image_path))
            if ai_axis_result and ai_axis_result.is_valid:
                if show_progress:
                    print(f"  AI axis validation: X={ai_axis_result.x_range}, Y={ai_axis_result.y_range} "
                          f"(confidence: {ai_axis_result.confidence:.0%})")

                # Update calibration with AI results if confidence is high
                if ai_axis_result.confidence >= 0.8:
                    # Validate/correct X range
                    if ai_axis_result.x_range:
                        ai_x_min, ai_x_max = ai_axis_result.x_range
                        ocr_x_min, ocr_x_max = calibration.x_data_range
                        # If AI detected a significantly different range, use it
                        if abs(ai_x_max - ocr_x_max) / max(ocr_x_max, 1) > 0.1:
                            if show_progress:
                                print(f"  AI correcting X range: {ocr_x_max} -> {ai_x_max}")
                            calibration.x_data_range = (ai_x_min, ai_x_max)

                    # Validate/correct Y range
                    if ai_axis_result.y_range:
                        ai_y_min, ai_y_max = ai_axis_result.y_range
                        ocr_y_min, ocr_y_max = calibration.y_data_range
                        # Check for scale mismatch (0-1 vs 0-100)
                        if (ai_y_max <= 1.5 and ocr_y_max > 1.5) or (ai_y_max > 1.5 and ocr_y_max <= 1.5):
                            if show_progress:
                                print(f"  AI correcting Y scale: {ocr_y_max} -> {ai_y_max}")
                            calibration.y_data_range = (ai_y_min, ai_y_max)
        except Exception as e:
            if show_progress:
                print(f"  AI axis detection skipped: {e}")

    if calibration is None:
        print("  WARNING: Automatic axis calibration failed.")
        print("  Using estimated plot bounds.")
        plot_bounds = (
            int(width * 0.1),
            int(height * 0.1),
            int(width * 0.8),
            int(height * 0.8)
        )
        calibration = None
    else:
        plot_bounds = calibration.plot_rectangle
        if show_progress:
            print(f"  Plot area: x={plot_bounds[0]}, y={plot_bounds[1]}, "
                  f"w={plot_bounds[2]}, h={plot_bounds[3]}")
            print(f"  X-axis range: {calibration.x_data_range[0]} - {calibration.x_data_range[1]}")
            print(f"  Y-axis range: {calibration.y_data_range[0]} - {calibration.y_data_range[1]}")

    # Note: time_max is already passed to calibrator, but if the detected range differs
    # and user specified a different time_max, we may need to adjust
    # The calibrator_v2 already handles this in calibrate(), so typically no override needed
    if time_max is not None and calibration is not None:
        current_time_max = calibration.x_data_range[1]
        if abs(current_time_max - time_max) > 0.5:
            # User specified a different time_max than detected
            calibration.x_data_range = (calibration.x_data_range[0], time_max)
            if show_progress:
                print(f"  X-axis range (override): 0 - {time_max}")

    # Step 3: Detect and trace curves
    if show_progress:
        print("\n[Step 3/4] Detecting curves...")

    # Choose detection method based on image type
    use_color_detection = is_color_image(img)

    # Determine if we should use skeleton extraction
    # Now works for both color AND grayscale images
    use_skeleton_extraction = (
        use_skeleton and
        _SKELETON_AVAILABLE
    )

    if use_skeleton_extraction and show_progress:
        if use_color_detection:
            print("  Using SKELETON-based extraction (color + text removal + skeletonization)...")
        else:
            print("  Using SKELETON-based extraction (grayscale + text removal + skeletonization)...")

    # For color images, check if curves are in a different panel than calibrated
    detected_time_max = None
    if use_color_detection:
        adjusted_result = _find_panel_for_color_curves(img, plot_bounds, show_progress)

        # Handle both 4-tuple and 5-tuple returns (with/without time_max)
        if len(adjusted_result) == 5:
            adjusted_bounds = adjusted_result[:4]
            detected_time_max = adjusted_result[4]
        else:
            adjusted_bounds = adjusted_result

        if adjusted_bounds != plot_bounds:
            plot_bounds = adjusted_bounds
            if show_progress:
                print(f"  Plot area (panel adjusted): x={plot_bounds[0]}, y={plot_bounds[1]}, "
                      f"w={plot_bounds[2]}, h={plot_bounds[3]}")

            # Update calibration's time range if we detected it from the new panel
            if detected_time_max and calibration is not None:
                calibration.x_data_range = (0.0, detected_time_max)
                if show_progress:
                    print(f"  X-axis range (from panel): 0.0 - {detected_time_max}")

    if use_color_detection:
        if show_progress:
            print("  Using COLOR-based detection...")
        detector = ColorCurveDetector(img, plot_bounds)
        detected_curves = detector.detect_all_curves(
            expected_count=expected_curves,
            debug_dir=str(output_dir)
        )

        # Check if color curves extend beyond detected plot bounds
        # This can happen with wide images or multi-panel images
        if detected_curves:
            all_x_coords = []
            for curve in detected_curves:
                if curve.points:
                    all_x_coords.extend([p[0] for p in curve.points])

            if all_x_coords:
                min_x = min(all_x_coords)
                max_x = max(all_x_coords)
                detected_x_range = max_x - min_x

                plot_x, plot_y, plot_w, plot_h = plot_bounds

                # If curves extend significantly beyond plot bounds, expand them
                curves_outside_bounds = max_x > plot_x + plot_w + 20
                curves_span_ratio = detected_x_range / plot_w if plot_w > 0 else 1

                if curves_outside_bounds or curves_span_ratio > 1.5:
                    # Expand plot bounds to cover all detected curves
                    new_plot_x = max(0, min_x - 10)
                    new_plot_w = max_x - new_plot_x + 10

                    if show_progress:
                        print(f"  NOTE: Color curves extend beyond detected plot area")
                        print(f"    Curves span x: {min_x}-{max_x} (detected plot: {plot_x}-{plot_x+plot_w})")
                        print(f"    Expanding plot bounds: x={new_plot_x}, w={new_plot_w}")

                    plot_bounds = (new_plot_x, plot_y, new_plot_w, plot_h)

                    # Also need to adjust the x_data_range proportionally
                    if calibration is not None:
                        old_time_range = calibration.x_data_range[1] - calibration.x_data_range[0]
                        scale_factor = new_plot_w / plot_w
                        new_time_max = calibration.x_data_range[0] + old_time_range * scale_factor
                        calibration.x_data_range = (calibration.x_data_range[0], new_time_max)
                        if show_progress:
                            print(f"    Adjusted X-axis range: 0 - {new_time_max:.1f}")
    else:
        if show_progress:
            print("  Using LINE-STYLE detection (solid/dashed)...")
        # Use calibration-based bounds for detection to capture full curve region
        # The refined bounds may cut off the start of curves (where survival=1.0)
        if calibration is not None:
            # Full calibration region from origin to axis ends
            cal_x = calibration.origin[0]
            cal_y = calibration.y_axis_end[1]
            cal_w = calibration.x_axis_end[0] - cal_x
            cal_h = calibration.origin[1] - cal_y
            detection_bounds = (cal_x, cal_y, cal_w, cal_h)
            if show_progress:
                print(f"  Detection area (calibration): x={cal_x}, y={cal_y}, w={cal_w}, h={cal_h}")
        else:
            detection_bounds = plot_bounds

        # Optional AI-enhanced preprocessing to remove text before detection
        # This can help when text overlaps with curves
        detection_img = img
        if use_ai:
            try:
                from lib.ai_text_remover import AITextRemover
                remover = AITextRemover()
                if remover.is_available:
                    if show_progress:
                        print("  AI preprocessing: removing text before detection...")
                    cleaned_img, text_mask, report = remover.remove_text_with_ai_guidance(img, detection_bounds)
                    if report.get('success', False):
                        detection_img = cleaned_img
                        cv2.imwrite(str(output_dir / "debug_ai_cleaned.png"), cleaned_img)
                        cv2.imwrite(str(output_dir / "debug_ai_text_mask.png"), text_mask)
                        if show_progress:
                            print(f"    AI text removal: {report.get('ai_assessment', 'completed')}")
                    else:
                        if show_progress:
                            print(f"    AI text removal issues: {report.get('issues', [])}")
            except ImportError:
                pass
            except Exception as e:
                if show_progress:
                    print(f"    AI preprocessing failed: {e}")

        detector = LineStyleDetector(
            detection_img, detection_bounds,
            filter_reference_lines=True,
            preprocess_text_removal=True  # Remove text before curve detection
        )
        detected_curves = detector.detect_all_curves(
            expected_count=expected_curves,
            debug_dir=str(output_dir)
        )

    if show_progress:
        print(f"  Found {len(detected_curves)} curves:")
        for i, curve in enumerate(detected_curves):
            if use_color_detection:
                print(f"    Curve {i+1}: {curve.name} "
                      f"(confidence: {curve.confidence:.2f})")
            else:
                print(f"    Curve {i+1}: {curve.style.value} "
                      f"(confidence: {curve.confidence:.2f})")

    if not detected_curves:
        print("\n  ERROR: No curves detected!")
        print("  Tips:")
        print("    - Ensure the image shows clear KM curves")
        print("    - Check that curves are darker than the background")
        print("    - Try with --curves N to specify expected number")
        return None

    # Save debug images
    debug_img = detector.get_debug_image()
    cv2.imwrite(str(output_dir / "debug_detection.png"), debug_img)

    if not use_color_detection:
        binary_mask = detector.get_binary_mask()
        if binary_mask is not None:
            cv2.imwrite(str(output_dir / "debug_binary.png"), binary_mask)

    # Auto-detect curve extent if time_max not specified
    # This finds where the curves actually end rather than using full axis range
    auto_detected_time_max = None
    if time_max is None and calibration is not None:
        auto_detected_time_max = _detect_curve_extent(detected_curves, calibration, show_progress)
        if auto_detected_time_max is not None:
            # Store the original axis max for reference
            original_axis_max = calibration.x_data_range[1]

            # Only use auto-detected max if it's significantly less than axis max
            # (within 95% means curves extend to near the axis end, so use axis max)
            if auto_detected_time_max < original_axis_max * 0.95:
                if show_progress:
                    print(f"  Auto-detected curve extent: time 0 to {auto_detected_time_max:.1f}")
                    print(f"    (axis range is 0 to {original_axis_max:.1f})")

                # Update calibration to use detected extent
                calibration.x_data_range = (calibration.x_data_range[0], auto_detected_time_max)

                # Adjust x_axis_end to match
                scale_factor = auto_detected_time_max / original_axis_max
                new_x_end = calibration.origin[0] + (calibration.x_axis_end[0] - calibration.origin[0]) * scale_factor
                calibration.x_axis_end = (int(new_x_end), calibration.x_axis_end[1])
            else:
                if show_progress:
                    print(f"  Curves extend to near axis end, using full range: 0 to {original_axis_max:.1f}")
                auto_detected_time_max = None  # Reset - not using auto-detection

    # Step 3.5: Extract at-risk table (optional)
    at_risk_data = None
    if _AT_RISK_AVAILABLE:
        if show_progress:
            print("\n[Step 3.5/4] Checking for at-risk table...")

        try:
            at_risk_extractor = AtRiskExtractor(
                img, plot_bounds, calibration, debug=show_progress
            )
            at_risk_data = at_risk_extractor.extract()

            if at_risk_data and at_risk_data.groups:
                if show_progress:
                    print(f"  Found at-risk table with {len(at_risk_data.groups)} group(s)")
                    for group_name, time_data in at_risk_data.groups.items():
                        print(f"    {group_name}: {len(time_data)} time points")

                # Save debug image
                at_risk_extractor.save_debug_image(
                    str(output_dir / "debug_at_risk_detection.png"),
                    at_risk_data
                )
            else:
                if show_progress:
                    print("  No at-risk table detected (continuing without)")

            # Use AI to validate/improve table reading if available
            if ai_service:
                try:
                    # Extract table region for AI reading
                    h, w = img.shape[:2]
                    table_y = plot_bounds[1] + plot_bounds[3]  # Below plot
                    table_region = img[table_y:, :]
                    if table_region.size > 0:
                        ai_table_result = ai_service.read_at_risk_table(table_region)
                        if ai_table_result and ai_table_result.is_valid:
                            if show_progress:
                                print(f"  AI table reading: {len(ai_table_result.groups)} groups, "
                                      f"{len(ai_table_result.time_points)} time points "
                                      f"(confidence: {ai_table_result.confidence:.0%})")

                            # If OCR missed groups or AI has higher confidence, use AI results
                            if not at_risk_data or not at_risk_data.groups:
                                # Create AtRiskData from AI result
                                from lib import AtRiskData
                                at_risk_data = AtRiskData(groups=ai_table_result.groups)
                                if show_progress:
                                    print("  Using AI-detected at-risk table")
                            elif ai_table_result.confidence > 0.8:
                                # Validate OCR results against AI
                                for ai_group, ai_counts in ai_table_result.groups.items():
                                    # Find matching OCR group
                                    matched = False
                                    for ocr_group in at_risk_data.groups:
                                        if ai_group.lower() in ocr_group.lower() or ocr_group.lower() in ai_group.lower():
                                            matched = True
                                            # AI can correct individual values if needed
                                            break
                                    if not matched and ai_table_result.confidence > 0.85:
                                        # Add AI-detected group
                                        at_risk_data.groups[ai_group] = ai_counts
                                        if show_progress:
                                            print(f"  AI added missing group: {ai_group}")
                except Exception as e:
                    if show_progress:
                        print(f"  AI table reading skipped: {e}")

        except Exception as e:
            if show_progress:
                print(f"  At-risk extraction skipped: {e}")
            at_risk_data = None

    # Set up coordinate conversion
    plot_x, plot_y, plot_w, plot_h = plot_bounds

    if calibration is not None:
        time_min, time_max_cal = calibration.x_data_range
        survival_min, survival_max = calibration.y_data_range
        # Use calibration coordinates for accurate mapping
        # X-axis: origin[0] is time=0, x_axis_end[0] is time=max
        # NOTE: x_axis_end has already been adjusted if time_max was overridden
        origin_x = calibration.origin[0]
        x_axis_end_x = calibration.x_axis_end[0]
        actual_plot_w = x_axis_end_x - origin_x

        if show_progress:
            print(f"  X-axis mapping: pixel {origin_x}-{int(x_axis_end_x)} = time 0-{time_max_cal}")
        # Y-axis: y_axis_end[1] is survival=1.0 (top), origin[1] is survival=0 (bottom)
        # Y-axis: y_axis_end[1] is survival=1.0 (top)
        y_top = calibration.y_axis_end[1]
        y_bottom = calibration.origin[1]   # pixel Y for survival=0.0
        actual_plot_h = y_bottom - y_top
    else:
        time_min = 0.0
        time_max_cal = time_max if time_max else 10.0
        survival_min, survival_max = 0.0, 1.0
        origin_x = plot_x
        actual_plot_w = plot_w
        y_top = plot_y
        y_bottom = plot_y + plot_h
        actual_plot_h = plot_h

    # Create pixel_to_coord function using calibrated coordinates for both axes
    def pixel_to_coord(px_x, px_y):
        # X-axis: origin_x is time=0, x_axis_end is time=max
        t = time_min + (px_x - origin_x) / actual_plot_w * (time_max_cal - time_min)
        # Y-axis: y_top is survival=1.0, y_bottom is survival=0.0
        s = survival_max - (px_y - y_top) / actual_plot_h * (survival_max - survival_min)
        # Normalize survival to 0-1 range if Y-axis uses percentage scale (0-100)
        if survival_max > 1.5:
            s = s / survival_max
        return t, s

    # Pass at-risk data to the detector for curve correction (helps with compression artifacts)
    if use_color_detection and at_risk_data and at_risk_data.groups:
        # Set time_max for the detector
        effective_time_max = time_max_cal if calibration else (time_max if time_max else 24.0)
        detector.set_time_max(effective_time_max)

        # Fix for OCR mis-parsing: if a group named '0 1', '0 ', '0', or similar exists,
        # it's likely the time header row being misread. Use its values as time points.
        correct_time_points = None
        header_group = None
        for group_name in at_risk_data.groups:
            if group_name.strip() in ['0', '0 1', '01', '0 ', ' 0']:
                time_counts = at_risk_data.groups[group_name]
                sorted_items = sorted(time_counts.items(), key=lambda x: x[0])
                # The "counts" from this row are actually the time points
                header_counts = [c for t, c in sorted_items]
                # Check if these look like time points (small integers starting from 0 or 3)
                if len(header_counts) >= 3 and header_counts[0] < 10 and all(c < 100 for c in header_counts):
                    # These are time points: 0, 3, 6, 12, 18, 24
                    # But the first one might be missing if OCR misread it as '0 1'
                    if header_counts[0] > 0:
                        correct_time_points = [0] + list(header_counts)
                    else:
                        correct_time_points = list(header_counts)
                    header_group = group_name
                    if show_progress:
                        print(f"  Detected time header in '{group_name}': {correct_time_points}")
                break

        # Pass at-risk data for each group
        for group_name, time_counts in at_risk_data.groups.items():
            # Skip the header row group
            if group_name == header_group:
                continue

            # Convert Dict[float, int] to lists
            sorted_items = sorted(time_counts.items(), key=lambda x: x[0])
            counts = [c for t, c in sorted_items]

            # Use correct time points if available, otherwise use the parsed times
            if correct_time_points and len(correct_time_points) >= len(counts):
                times = correct_time_points[:len(counts)]
                if show_progress:
                    print(f"  Using corrected time points for '{group_name}': {times}")
            else:
                times = [t for t, c in sorted_items]

            detector.set_at_risk_data(group_name, times, counts)

            # Also generate annotation points from at-risk counts
            # The at-risk count divided by initial count gives a rough survival upper bound
            if len(counts) > 0:
                initial_count = counts[0]
                if initial_count > 0:
                    annotation_points = []
                    for t, c in zip(times, counts):
                        # At-risk count gives an upper bound on survival
                        # Use 95% of the ratio to account for events before this time
                        survival = c / initial_count * 0.95
                        annotation_points.append((t, survival))
                    detector.set_annotation_points(group_name, annotation_points)
                    if show_progress:
                        print(f"    Generated {len(annotation_points)} annotation points: {[(round(t,1), round(s,2)) for t,s in annotation_points]}")

            if show_progress:
                print(f"  At-risk data for '{group_name}': {len(times)} time points")

    # Step 4: Extract and export data
    if show_progress:
        print("\n[Step 4/4] Extracting coordinates...")

    # First pass: extract raw points for all curves
    all_curves_data = []

    # Use skeleton extraction (removes text, cleaner extraction)
    if use_skeleton_extraction:
        if show_progress:
            print("  Using skeleton-based extraction...")

        # Get effective time_max for coordinate conversion
        effective_time_max = time_max if time_max is not None else (
            calibration.x_data_range[1] if calibration else 24.0
        )

        # For skeleton extraction, use expanded bounds to capture all curve pixels
        # The calibration may detect too narrow bounds; we expand based on image structure
        skeleton_bounds = plot_bounds
        plot_x, plot_y, plot_w, plot_h = plot_bounds
        h, w = img.shape[:2]

        # Expand bounds to capture full curve area (typical KM plot structure)
        # Y-axis: extend down to capture full survival range (0-100%)
        # X-axis: extend right if time_max suggests curves go further
        expanded_plot_h = int(h * 0.62)  # Typical KM plot uses ~62% of height for curves
        if plot_h < expanded_plot_h * 0.8:  # If detected height is too small
            # Recalculate bounds based on image proportions
            skeleton_plot_y = int(h * 0.03)
            skeleton_plot_h = expanded_plot_h
            skeleton_plot_x = plot_x
            skeleton_plot_w = int(w * 0.76)  # Typical plot width
            skeleton_bounds = (skeleton_plot_x, skeleton_plot_y, skeleton_plot_w, skeleton_plot_h)
            if show_progress:
                print(f"    Expanded skeleton bounds: x={skeleton_plot_x}, y={skeleton_plot_y}, "
                      f"w={skeleton_plot_w}, h={skeleton_plot_h}")

        # Remove text and extract using skeletonization
        cleaned_img, text_mask = remove_all_text(img)

        # Save debug images
        cv2.imwrite(str(output_dir / "debug_text_mask.png"), text_mask)
        cv2.imwrite(str(output_dir / "debug_cleaned_image.png"), cleaned_img)

        # Branch based on color vs grayscale image
        if use_color_detection:
            # Color image: detect curve colors and extract each
            colors = detect_curve_colors(img, max_colors=expected_curves)
            if show_progress:
                for c in colors:
                    print(f"    Detected color: {c['name']} (hue={c['hue']:.1f})")

            # Extract curves using color-based skeleton method
            skeleton_results = extract_curves_with_skeleton(img, colors, skeleton_bounds)
        else:
            # Grayscale image: use line-style detected curves with skeleton cleanup
            if show_progress:
                print("    Using line-style detection with skeleton cleanup...")

            # Extract curves using grayscale skeleton method
            # Pass detected_curves to help with curve separation
            skeleton_results = extract_curves_grayscale_skeleton(
                img, skeleton_bounds, expected_curves=expected_curves,
                detected_curves=detected_curves if 'detected_curves' in dir() else None
            )

        # Create direct coordinate conversion for skeleton extraction
        # This uses the skeleton bounds and time_max
        plot_x, plot_y, plot_w, plot_h = skeleton_bounds

        def skeleton_pixel_to_coord(px_x, px_y):
            """Direct coordinate conversion using plot bounds and time_max."""
            t = ((px_x - plot_x) / plot_w) * effective_time_max
            s = 1.0 - ((px_y - plot_y) / plot_h)
            return t, max(0.0, min(1.0, s))

        for points, color_info in skeleton_results:
            name = color_info.get('name', 'unknown')

            # Convert pixel coordinates to data coordinates using direct conversion
            raw_points = []
            for px_x, px_y in points:
                t, s = skeleton_pixel_to_coord(px_x, px_y)
                if 0 <= t <= effective_time_max + 0.5 and 0 <= s <= 1.0:
                    raw_points.append((t, s))

            # Sort by time
            raw_points.sort(key=lambda p: p[0])

            if not raw_points:
                if show_progress:
                    print(f"  WARNING: No points extracted for {name}")
                continue

            curve_data = {
                'name': name,
                'style': name.split('_')[0] if '_' in name else name,
                'raw_points': raw_points,
                'confidence': 0.9,  # Skeleton extraction is generally reliable
            }
            all_curves_data.append(curve_data)

            if show_progress:
                print(f"    {name}: {len(raw_points)} points extracted")

    else:
        # Traditional extraction using detector
        for i, curve in enumerate(detected_curves):
            # Extract points
            raw_points = detector.extract_curve_points(curve, pixel_to_coord)

            if not raw_points:
                if show_progress:
                    print(f"  WARNING: No points extracted for curve {i+1}")
                continue

            # Filter out points beyond time_max (if specified or auto-detected)
            # Use time_max_cal which contains the effective maximum (user-specified, auto-detected, or axis max)
            effective_time_max = time_max if time_max is not None else (auto_detected_time_max if auto_detected_time_max is not None else None)
            if effective_time_max is not None:
                raw_points = [(t, s) for t, s in raw_points if t <= effective_time_max + 0.5]

            # Get curve name and style based on detection method
            if use_color_detection:
                curve_name = curve.name
                curve_style = curve.name.split('_')[0]  # e.g., "orange" from "orange_1"
            else:
                curve_name = f"{curve.style.value}_{i+1}"
                curve_style = curve.style.value

            curve_data = {
                'name': curve_name,
                'style': curve_style,
                'raw_points': raw_points,
                'confidence': curve.confidence,
            }
            all_curves_data.append(curve_data)

    # Merge overlapping regions for KM curves (both start at same point)
    if len(all_curves_data) == 2 and not use_color_detection:
        all_curves_data = _merge_overlapping_curves(all_curves_data, show_progress=show_progress)

    # Verify and correct curve labels using dash_mask
    # This ensures the curve with more dashed pixels gets the "dashed" label
    if len(all_curves_data) == 2 and not use_color_detection:
        all_curves_data = _verify_curve_labels_with_dash_mask(
            all_curves_data, detector, detected_curves, show_progress=show_progress
        )

    # Fix curve ordering so dashed curve always has higher survival than solid
    # This corrects any tracking errors where curves got swapped at certain points
    if len(all_curves_data) == 2:
        all_curves_data = _fix_curve_ordering(all_curves_data, show_progress=show_progress)

    # Second pass: clean data and validate
    for curve_data in all_curves_data:
        raw_points = curve_data['raw_points']
        curve_name = curve_data['name']

        # Check raw data quality before cleaning
        if raw_points and show_progress:
            raw_max_s = max(s for _, s in raw_points)
            raw_min_t = min(t for t, _ in raw_points)
            if raw_max_s < 0.95:
                print(f"  NOTE: {curve_name} max survival={raw_max_s:.2f} (rescaling to 1.0)")
            if raw_min_t > 1.0:
                print(f"  NOTE: {curve_name} starts at time={raw_min_t:.1f} (adding t=0 point)")

        # Clean the data (remove duplicates, ensure monotonicity)
        clean_points = _clean_curve_data(raw_points)

        # Validate curve quality
        is_valid, quality_issues = _validate_curve_quality(curve_name, clean_points)

        curve_data['clean_points'] = clean_points
        curve_data['is_valid'] = is_valid
        curve_data['quality_issues'] = quality_issues

    # Print summary for each curve
    if show_progress:
        for curve_data in all_curves_data:
            curve_name = curve_data['name']
            clean_points = curve_data.get('clean_points', [])
            is_valid = curve_data.get('is_valid', False)
            quality_issues = curve_data.get('quality_issues', [])
            if clean_points:
                t_range = f"{clean_points[0][0]:.2f} - {clean_points[-1][0]:.2f}"
                s_range = f"{min(p[1] for p in clean_points):.2f} - {max(p[1] for p in clean_points):.2f}"
                status = "OK" if is_valid else "ISSUES"
                print(f"  {curve_name}: {len(clean_points)} points, "
                      f"time: {t_range}, survival: {s_range} [{status}]")
                if not is_valid:
                    for issue in quality_issues:
                        print(f"    WARNING: {issue}")
            else:
                print(f"  {curve_name}: No valid points")

    # Save CSV files
    for curve_data in all_curves_data:
        df = pd.DataFrame(curve_data['clean_points'], columns=['Time', 'Survival'])
        df.to_csv(output_dir / f"curve_{curve_data['name']}.csv", index=False)

    # Save combined CSV
    combined_rows = []
    for curve_data in all_curves_data:
        for time, survival in curve_data['clean_points']:
            combined_rows.append({
                'Curve': curve_data['name'],
                'Style': curve_data['style'],
                'Time': time,
                'Survival': survival
            })

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(output_dir / "all_curves.csv", index=False)

    # Save at-risk table if extracted
    if at_risk_data and at_risk_data.groups:
        at_risk_df = at_risk_data.add_events_column()
        at_risk_df.to_csv(output_dir / "at_risk_table.csv", index=False)

    # Generate visualization
    _plot_curves(all_curves_data, output_dir / "extracted_curves.png")

    # Generate comparison overlay (extracted curves on original image)
    _plot_comparison_overlay(
        img, all_curves_data, plot_bounds, calibration,
        time_max if time_max else (calibration.x_data_range[1] if calibration else 12.0),
        output_dir / "comparison_overlay.png"
    )

    # Ask user to verify curve labeling
    if show_progress and len(all_curves_data) == 2:
        all_curves_data = _check_and_fix_curve_swap(
            all_curves_data, output_dir, img, plot_bounds, calibration,
            time_max if time_max else (calibration.x_data_range[1] if calibration else 12.0)
        )

    # Run validation checks
    validation_report = None
    dense_report = None
    if _VALIDATION_AVAILABLE:
        if show_progress:
            print("\n[Step 5/5] Validating extraction...")
        validation_report = validate_extraction(
            img, calibration, plot_bounds, all_curves_data, verbose=show_progress
        )

        # Save validation report to file
        with open(output_dir / "validation_report.txt", 'w') as f:
            f.write(validation_report.summary())
            f.write("\n\nDetailed Sample Point Errors:\n")
            for error in validation_report.sample_point_errors:
                f.write(f"  {error['curve']} @ t={error['sample_time']:.1f}: "
                       f"s={error['extracted_survival']:.3f}, pixel_err={error['pixel_error']:.1f}px\n")

            # Add early region details if available
            f.write("\n\nEarly Region Analysis (t=0 to 20% of range):\n")
            for result in validation_report.results:
                if "Early Region" in result.name and result.details:
                    f.write(f"\n  {result.name}:\n")
                    if 'pixel_errors' in result.details:
                        for pe in result.details['pixel_errors']:
                            f.write(f"    t={pe['t']:.2f}: s={pe['s']:.3f}, pixel_err={pe['err']:.1f}px\n")
                    if 'avg_error' in result.details:
                        f.write(f"    Summary: avg={result.details['avg_error']:.1f}px, max={result.details['max_error']:.1f}px\n")

    # Run dense validation for detailed quality control
    if _DENSE_VALIDATION_AVAILABLE:
        if show_progress:
            print("\n[Step 5.5/5] Running dense validation...")

        # Get color masks if available
        color_masks = {}
        for curve_data in all_curves_data:
            if 'mask' in curve_data:
                color_masks[curve_data['name']] = curve_data['mask']

        dense_report = run_dense_validation(
            img, calibration, plot_bounds, all_curves_data,
            color_masks=color_masks if color_masks else None,
            verbose=False
        )

        # Save dense validation report
        with open(output_dir / "dense_validation_report.txt", 'w') as f:
            f.write(dense_report.summary())

            # Add detailed offset analysis
            f.write("\n\nDETAILED OFFSET ANALYSIS\n")
            f.write("-" * 60 + "\n")

            for curve_name, report in dense_report.curve_reports.items():
                f.write(f"\n{curve_name}:\n")

                # Offset distribution
                all_offsets = report.get('all_offsets', [])
                if all_offsets:
                    f.write(f"  Offset statistics:\n")
                    f.write(f"    Mean: {np.mean(all_offsets):.2f} pixels\n")
                    f.write(f"    Std:  {np.std(all_offsets):.2f} pixels\n")
                    f.write(f"    Min:  {min(all_offsets):.2f} pixels\n")
                    f.write(f"    Max:  {max(all_offsets):.2f} pixels\n")

                    # Offset direction analysis
                    positive = sum(1 for o in all_offsets if o > 2)
                    negative = sum(1 for o in all_offsets if o < -2)
                    neutral = len(all_offsets) - positive - negative
                    f.write(f"\n  Offset direction:\n")
                    f.write(f"    Extracted above original: {negative} points ({negative/len(all_offsets)*100:.1f}%)\n")
                    f.write(f"    Within tolerance: {neutral} points ({neutral/len(all_offsets)*100:.1f}%)\n")
                    f.write(f"    Extracted below original: {positive} points ({positive/len(all_offsets)*100:.1f}%)\n")

                # Step analysis
                f.write(f"\n  Step analysis:\n")
                f.write(f"    Steps detected in extracted: {report.get('steps_detected', 'N/A')}\n")
                f.write(f"    Steps detected in original:  {report.get('steps_in_original', 'N/A')}\n")

                # Missing step regions
                missing = report.get('missing_step_regions', [])
                if missing:
                    f.write(f"\n  Potential missing step regions ({len(missing)}):\n")
                    for region in missing[:10]:
                        f.write(f"    t={region['time_start']:.2f} to t={region['time_end']:.2f}: "
                               f"gap={region['survival_gap']:.4f}\n")

                # Significant offset regions
                offsets = report.get('offset_regions', [])
                if offsets:
                    f.write(f"\n  Significant offset regions ({len(offsets)}):\n")
                    for region in offsets[:10]:
                        f.write(f"    t={region['time']:.2f}: offset={region['offset']:.1f}px "
                               f"({region['direction']})\n")

        if show_progress:
            print(f"  Overall accuracy: {dense_report.overall_accuracy*100:.1f}%")
            for curve_name, report in dense_report.curve_reports.items():
                print(f"  {curve_name}: {report['match_rate']*100:.1f}% matched, "
                      f"avg offset={report['avg_offset']:.1f}px, "
                      f"steps={report['steps_detected']}/{report['steps_in_original']}")

    if show_progress:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nOutput files saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  - all_curves.csv        : Combined data from all curves")
        for curve_data in all_curves_data:
            print(f"  - curve_{curve_data['name']}.csv : {curve_data['style']} curve data")
        if at_risk_data and at_risk_data.groups:
            print("  - at_risk_table.csv     : At-risk numbers (Guyot-compatible)")
        print("  - extracted_curves.png  : Visualization plot")
        print("  - comparison_overlay.png: Original with extracted curves overlay")
        if validation_report:
            print("  - validation_report.txt : Validation results")
        if dense_report:
            print("  - dense_validation_report.txt : Dense QC report")
        print("  - debug_*.png           : Debug images")

        # Print validation summary
        if validation_report:
            print(f"\n  Validation: {validation_report.overall_status.value}")
            if validation_report.overall_status != ValidationStatus.PASS:
                print("  Check validation_report.txt for details")

    return {
        'curves': all_curves_data,
        'output_dir': str(output_dir),
        'calibration': calibration,
        'at_risk_data': at_risk_data,
        'validation_report': validation_report
    }


def _clean_curve_data(points, tolerance=0.005):
    """Clean curve data by removing noise and enforcing STRICT monotonicity.

    KM curves MUST:
    1. Start at survival=1.0 at time=0 (all subjects alive initially)
    2. Be strictly monotonically NON-INCREASING (survival can only decrease or stay flat)
    3. Have strictly increasing time values

    If max detected survival is significantly below 1.0, rescale all values.
    """
    if not points:
        return []

    # Sort by time
    sorted_points = sorted(points, key=lambda p: p[0])

    # Clip negative times to 0 (KM curves must start at time=0)
    sorted_points = [(max(0.0, t), s) for t, s in sorted_points]

    # Remove duplicate times using continuity-based selection
    # KM curves should be monotonically non-increasing, but we need to avoid
    # selecting outliers caused by color bleeding from nearby curves
    time_groups = {}
    for t, s in sorted_points:
        t_rounded = round(t, 3)
        if t_rounded not in time_groups:
            time_groups[t_rounded] = []
        time_groups[t_rounded].append(s)

    deduplicated = []
    prev_survival = 1.0  # Start at max survival

    for t in sorted(time_groups.keys()):
        survivals = time_groups[t]

        if len(survivals) == 1:
            selected_s = survivals[0]
        else:
            # Multiple values at this time - use continuity-based selection
            # Filter out values that are too far from previous (likely noise)
            survivals_arr = np.array(survivals)

            # Values should be <= previous (KM can only go down) and not too far below
            # Allow for genuine drops but filter extreme outliers
            valid_mask = (survivals_arr <= prev_survival + 0.01) & (survivals_arr >= prev_survival - 0.15)
            valid_survivals = survivals_arr[valid_mask]

            if len(valid_survivals) > 0:
                # Use MAXIMUM valid survival (topmost pixel = most accurate position)
                # The true curve line is at the top; lower values are from anti-aliasing/thickness
                selected_s = float(np.max(valid_survivals))
            else:
                # No valid values - use the one closest to previous
                closest_idx = np.argmin(np.abs(survivals_arr - prev_survival))
                selected_s = survivals[closest_idx]

        deduplicated.append((t, selected_s))
        prev_survival = selected_s

    if not deduplicated:
        return []

    # Find max survival in the data (should be ~1.0 for KM curves)
    max_detected = max(s for _, s in deduplicated)

    # If max survival is significantly below 1.0, rescale
    # This handles calibration errors where plot bounds don't match axis range
    if max_detected < 0.95 and max_detected > 0.1:
        scale_factor = 1.0 / max_detected
        deduplicated = [(t, min(1.0, s * scale_factor)) for t, s in deduplicated]

    # KM curves MUST start at (0, 1.0) - all subjects alive at time 0
    first_t, first_s = deduplicated[0]

    # Ensure curve starts at (0, 1.0) and connects smoothly to detected points
    if first_t > 0.01:
        deduplicated.insert(0, (0.0, 1.0))
    elif first_s < 0.999:
        deduplicated[0] = (0.0, 1.0)

    # STRICT monotonicity enforcement - survival can ONLY decrease or stay flat
    # This is the key constraint for Kaplan-Meier curves
    monotonic = []
    current_max_survival = 1.0

    for t, s in deduplicated:
        # Survival cannot exceed the current running maximum
        s = min(s, current_max_survival)
        monotonic.append((t, s))
        # Update running maximum (which can only decrease)
        current_max_survival = s

    # Remove consecutive duplicates (within tolerance) - keep step function shape
    cleaned = [monotonic[0]] if monotonic else []
    for i in range(1, len(monotonic)):
        t, s = monotonic[i]
        prev_t, prev_s = cleaned[-1]

        # Keep if time changed significantly or survival dropped
        if t - prev_t > tolerance or prev_s - s > tolerance:
            cleaned.append((t, s))

    # Final validation: ensure strict monotonicity
    validated = _validate_monotonicity(cleaned)

    # Remove spurious tail drops: if curve is stable then suddenly drops to ~0,
    # those final points are likely from text annotations or axis lines, not the curve
    validated = _filter_spurious_tail_drops(validated)

    return validated


def _filter_spurious_tail_drops(points, stable_threshold=0.05, drop_threshold=0.1):
    """Remove spurious drops at the curve's tail.

    If the curve is relatively stable in the tail region and then suddenly
    drops to very low survival (especially to 0), remove those artifact points.

    Args:
        points: List of (time, survival) tuples
        stable_threshold: Max variation in tail to be considered "stable"
        drop_threshold: Min drop size to be considered "spurious"

    Returns:
        Filtered list of points
    """
    if len(points) < 10:
        return points

    # Look at the last 20% of time points
    tail_start_idx = int(len(points) * 0.8)
    tail_points = points[tail_start_idx:]

    if len(tail_points) < 3:
        return points

    # Find the survival values in the tail
    tail_survivals = [s for _, s in tail_points]

    # Check if there's a sudden drop at the very end
    # Look at the last few points vs the rest of the tail
    stable_region = tail_survivals[:-3] if len(tail_survivals) > 5 else tail_survivals[:-1]
    end_region = tail_survivals[-3:] if len(tail_survivals) > 5 else tail_survivals[-1:]

    if not stable_region or not end_region:
        return points

    stable_mean = np.mean(stable_region)
    stable_std = np.std(stable_region)
    end_min = min(end_region)

    # If the stable region is relatively flat (std < threshold)
    # and the end drops significantly below the stable mean
    # then the end points are likely artifacts
    if stable_std < stable_threshold and stable_mean - end_min > drop_threshold:
        # Check if end points are near zero (common artifact)
        if end_min < 0.05:
            # Find where the spurious drop starts
            drop_idx = len(points)
            for i in range(len(points) - 1, tail_start_idx, -1):
                if points[i][1] >= stable_mean - stable_threshold:
                    break
                drop_idx = i

            # Remove points from the spurious drop
            return points[:drop_idx]

    return points


def _validate_monotonicity(points):
    """Validate and enforce strict monotonicity on cleaned curve data.

    This is a final pass to ensure no survival increases exist.
    """
    if not points:
        return []

    result = [points[0]]
    running_max = points[0][1]

    for i in range(1, len(points)):
        t, s = points[i]

        # Strict check: survival must not increase
        if s > running_max:
            # This should not happen after cleaning, but fix it anyway
            s = running_max

        result.append((t, s))
        running_max = s

    return result


def _verify_curve_labels_with_dash_mask(curves_data, detector, detected_curves, show_progress=False):
    """
    Verify and correct curve labels using the dash_mask.

    The curve with more pixels overlapping the dash_mask should be labeled as "dashed".
    This corrects any mislabeling that may have occurred during curve tracking.

    Args:
        curves_data: List of curve data dictionaries with 'name', 'style', 'raw_points'
        detector: LineStyleDetector instance with dash_mask
        detected_curves: List of DetectedCurve objects
        show_progress: Whether to print progress messages

    Returns:
        Corrected curves_data list (labels may be swapped)
    """
    if len(curves_data) != 2 or len(detected_curves) != 2:
        return curves_data

    # Get dash_mask from detector
    if not hasattr(detector, 'dash_mask') or detector.dash_mask is None:
        return curves_data

    dash_mask = detector.dash_mask

    # Count dash pixels for each curve using traced_curves
    dash_counts = []
    total_counts = []

    for i, traced_curve in enumerate(detector.traced_curves):
        dash_count = 0
        total_count = 0

        for x, y in traced_curve.y_positions.items():
            y_int = int(y)
            if 0 <= y_int < dash_mask.shape[0] and 0 <= x < dash_mask.shape[1]:
                total_count += 1
                if dash_mask[y_int, x] > 0:
                    dash_count += 1

        dash_counts.append(dash_count)
        total_counts.append(total_count)

    # Calculate dash ratios
    if total_counts[0] == 0 or total_counts[1] == 0:
        return curves_data

    dash_ratio_0 = dash_counts[0] / total_counts[0]
    dash_ratio_1 = dash_counts[1] / total_counts[1]

    if show_progress:
        print(f"  Dash verification: curve0 ratio={dash_ratio_0:.2%}, curve1 ratio={dash_ratio_1:.2%}")

    # Determine which curve should be dashed based on dash_mask overlap
    # Higher ratio = more dashed pixels = should be labeled "dashed"
    curve0_is_dashed = dash_ratio_0 > dash_ratio_1

    # Check current labels
    curve0_labeled_dashed = 'dashed' in curves_data[0]['style'].lower()
    curve1_labeled_dashed = 'dashed' in curves_data[1]['style'].lower()

    # Check if labels need to be swapped
    needs_swap = False
    if curve0_is_dashed and not curve0_labeled_dashed:
        needs_swap = True
    elif not curve0_is_dashed and curve0_labeled_dashed:
        needs_swap = True

    if needs_swap:
        if show_progress:
            print(f"  Correcting curve labels (dash_mask verification)")

        # Swap the labels only (keep data with original curves)
        old_name_0 = curves_data[0]['name']
        old_style_0 = curves_data[0]['style']
        old_name_1 = curves_data[1]['name']
        old_style_1 = curves_data[1]['style']

        curves_data[0]['name'] = old_name_1
        curves_data[0]['style'] = old_style_1
        curves_data[1]['name'] = old_name_0
        curves_data[1]['style'] = old_style_0

    return curves_data


def _fix_curve_ordering(curves_data, show_progress=False):
    """
    Fix curve ordering so dashed curve always has higher survival than solid.

    For KM curves comparing treatments, the dashed curve (better treatment) should
    have survival >= solid curve at all time points after they diverge.
    This function swaps data points where the ordering is incorrect.

    Args:
        curves_data: List of two curve data dicts with 'raw_points', 'clean_points'
        show_progress: Print debug info

    Returns:
        Fixed curves_data with correct ordering
    """
    if len(curves_data) != 2:
        return curves_data

    # Identify which curve is dashed (should have higher survival)
    dashed_idx = None
    solid_idx = None
    for i, curve in enumerate(curves_data):
        if 'dashed' in curve.get('style', '').lower():
            dashed_idx = i
        else:
            solid_idx = i

    if dashed_idx is None or solid_idx is None:
        return curves_data

    dashed_curve = curves_data[dashed_idx]
    solid_curve = curves_data[solid_idx]

    # Get the data points
    dashed_points = dashed_curve.get('clean_points') or dashed_curve.get('raw_points', [])
    solid_points = solid_curve.get('clean_points') or solid_curve.get('raw_points', [])

    if not dashed_points or not solid_points:
        return curves_data

    # Create time-indexed dictionaries for both curves
    dashed_by_time = {p[0]: p[1] for p in dashed_points}
    solid_by_time = {p[0]: p[1] for p in solid_points}

    # Find common time points
    common_times = sorted(set(dashed_by_time.keys()) & set(solid_by_time.keys()))

    if len(common_times) < 5:
        return curves_data

    # Find where curves first diverge (separation > threshold)
    diverge_time = None
    diverge_threshold = 0.01  # 1% survival difference

    for t in common_times:
        d_surv = dashed_by_time[t]
        s_surv = solid_by_time[t]
        if abs(d_surv - s_surv) > diverge_threshold:
            diverge_time = t
            break

    if diverge_time is None:
        return curves_data

    # After divergence, dashed should have higher survival than solid
    # Count how many points are correctly ordered vs swapped
    correct_count = 0
    swapped_count = 0

    for t in common_times:
        if t < diverge_time:
            continue
        d_surv = dashed_by_time[t]
        s_surv = solid_by_time[t]

        if d_surv >= s_surv - 0.001:  # Allow tiny tolerance
            correct_count += 1
        else:
            swapped_count += 1

    if show_progress:
        print(f"  Curve ordering check: {correct_count} correct, {swapped_count} swapped points after t={diverge_time:.2f}")

    # If mostly correct, fix the swapped points
    # If mostly swapped, the labels themselves are wrong (shouldn't happen after dash_mask verification)
    if correct_count >= swapped_count:
        # Fix individual swapped points
        fixes_made = 0
        new_dashed_points = []
        new_solid_points = []

        for t in common_times:
            d_surv = dashed_by_time[t]
            s_surv = solid_by_time[t]

            if t >= diverge_time and d_surv < s_surv - 0.001:
                # This point is swapped - fix it
                new_dashed_points.append((t, s_surv))
                new_solid_points.append((t, d_surv))
                fixes_made += 1
            else:
                # Keep original
                new_dashed_points.append((t, d_surv))
                new_solid_points.append((t, s_surv))

        if fixes_made > 0:
            if show_progress:
                print(f"  Fixed {fixes_made} swapped data points")

            # Update curve data
            curves_data[dashed_idx]['clean_points'] = new_dashed_points
            curves_data[dashed_idx]['raw_points'] = new_dashed_points
            curves_data[solid_idx]['clean_points'] = new_solid_points
            curves_data[solid_idx]['raw_points'] = new_solid_points

    return curves_data


def _merge_overlapping_curves(curves_data, divergence_threshold=0.02, show_progress=False):
    """Merge overlapping regions of KM curves.

    In Kaplan-Meier plots, both treatment groups typically start together at
    (0, 1.0) and follow the same path until they diverge. This function detects
    the overlapping region and uses shared coordinates for both curves.

    Args:
        curves_data: List of curve data dicts with 'raw_points' key
        divergence_threshold: Minimum survival difference to consider curves diverged
        show_progress: Print debug info

    Returns:
        Modified curves_data with merged overlapping regions
    """
    if len(curves_data) != 2:
        return curves_data

    # Get raw points from both curves
    points1 = curves_data[0].get('raw_points', [])
    points2 = curves_data[1].get('raw_points', [])

    if not points1 or not points2:
        return curves_data

    # Create time-indexed dictionaries for quick lookup
    dict1 = {round(t, 3): s for t, s in points1}
    dict2 = {round(t, 3): s for t, s in points2}

    # Find common time points
    common_times = sorted(set(dict1.keys()) & set(dict2.keys()))

    if not common_times:
        return curves_data

    # Find the divergence point - where survival values differ significantly
    divergence_time = None
    for t in common_times:
        s1, s2 = dict1[t], dict2[t]
        if abs(s1 - s2) > divergence_threshold:
            divergence_time = t
            break

    if divergence_time is None:
        # Curves never diverge significantly - use average for all common points
        divergence_time = max(common_times) + 1

    if show_progress:
        print(f"  Curves diverge at time={divergence_time:.2f}")

    # For times before divergence, use averaged survival values for both curves
    merged_points1 = []
    merged_points2 = []

    # Process points before divergence - use shared values
    for t, s in points1:
        t_rounded = round(t, 3)
        if t_rounded < divergence_time and t_rounded in dict2:
            # Use average of both curves
            avg_s = (dict1[t_rounded] + dict2[t_rounded]) / 2
            merged_points1.append((t, avg_s))
        else:
            merged_points1.append((t, s))

    for t, s in points2:
        t_rounded = round(t, 3)
        if t_rounded < divergence_time and t_rounded in dict1:
            # Use average of both curves
            avg_s = (dict1[t_rounded] + dict2[t_rounded]) / 2
            merged_points2.append((t, avg_s))
        else:
            merged_points2.append((t, s))

    # Also add any points that only exist in one curve before divergence
    # (ensures both curves have the same early points)
    all_early_times = set()
    for t, _ in points1:
        if round(t, 3) < divergence_time:
            all_early_times.add(round(t, 3))
    for t, _ in points2:
        if round(t, 3) < divergence_time:
            all_early_times.add(round(t, 3))

    # Add missing early points to each curve
    for t in sorted(all_early_times):
        s1 = dict1.get(t)
        s2 = dict2.get(t)

        if s1 is not None and s2 is not None:
            avg_s = (s1 + s2) / 2
        elif s1 is not None:
            avg_s = s1
        elif s2 is not None:
            avg_s = s2
        else:
            continue

        # Check if this time is already in merged points
        times1 = {round(pt[0], 3) for pt in merged_points1}
        times2 = {round(pt[0], 3) for pt in merged_points2}

        if t not in times1:
            merged_points1.append((t, avg_s))
        if t not in times2:
            merged_points2.append((t, avg_s))

    # Sort by time
    merged_points1.sort(key=lambda p: p[0])
    merged_points2.sort(key=lambda p: p[0])

    # Update curves data
    curves_data[0]['raw_points'] = merged_points1
    curves_data[1]['raw_points'] = merged_points2

    return curves_data


def _validate_curve_quality(curve_name, points):
    """Validate extracted curve data for Kaplan-Meier quality requirements.

    Checks:
    1. Curve starts at time=0, survival=1.0
    2. Curve is strictly monotonically non-increasing
    3. Time values are strictly increasing
    4. Survival values are in valid range [0, 1]

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if not points:
        return False, ["No points extracted"]

    # Check 1: Starts at (0, 1.0)
    first_t, first_s = points[0]
    if abs(first_t) > 0.01:
        issues.append(f"Does not start at time=0 (starts at t={first_t:.2f})")
    if abs(first_s - 1.0) > 0.01:
        issues.append(f"Does not start at survival=1.0 (starts at s={first_s:.3f})")

    # Check 2: Monotonically non-increasing
    prev_s = first_s
    violations = 0
    for t, s in points[1:]:
        if s > prev_s + 0.001:  # Allow tiny floating point tolerance
            violations += 1
        prev_s = s

    if violations > 0:
        issues.append(f"Monotonicity violation: {violations} point(s) where survival increases")

    # Check 3: Strictly increasing time
    prev_t = first_t
    time_violations = 0
    for t, s in points[1:]:
        if t <= prev_t:
            time_violations += 1
        prev_t = t

    if time_violations > 0:
        issues.append(f"Time ordering violation: {time_violations} point(s) with non-increasing time")

    # Check 4: Valid survival range
    min_s = min(s for _, s in points)
    max_s = max(s for _, s in points)
    if min_s < 0:
        issues.append(f"Invalid survival: minimum is {min_s:.3f} (should be >= 0)")
    if max_s > 1.0 + 0.001:
        issues.append(f"Invalid survival: maximum is {max_s:.3f} (should be <= 1.0)")

    # Summary statistics
    last_t, last_s = points[-1]
    time_range = f"0 - {last_t:.1f}"
    survival_range = f"{last_s:.2f} - 1.00"

    is_valid = len(issues) == 0
    return is_valid, issues


def _plot_curves(curves_data, output_path):
    """Generate a plot of extracted curves."""
    plt.figure(figsize=(10, 6))

    # Colors for line-style detection
    style_colors = {'solid': 'blue', 'dashed': 'red', 'dotted': 'green'}
    linestyles = {'solid': '-', 'dashed': '--', 'dotted': ':'}

    # Colors for color detection (use actual color names)
    color_map = {
        'red': 'red', 'orange': 'orange', 'yellow': 'gold',
        'green': 'green', 'cyan': 'cyan', 'blue': 'blue',
        'purple': 'purple', 'magenta': 'magenta'
    }

    for curve_data in curves_data:
        points = curve_data['clean_points']
        if not points:
            continue

        times = [p[0] for p in points]
        survivals = [p[1] for p in points]
        style = curve_data['style']

        # Determine color and linestyle
        if style in color_map:
            plot_color = color_map[style]
            plot_linestyle = '-'
        else:
            plot_color = style_colors.get(style, 'black')
            plot_linestyle = linestyles.get(style, '-')

        plt.step(times, survivals,
                where='post',
                color=plot_color,
                linestyle=plot_linestyle,
                linewidth=2,
                label=curve_data['name'])

    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Extracted Kaplan-Meier Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_comparison_overlay(original_img, curves_data, plot_bounds, calibration, time_max, output_path):
    """Generate a comparison overlay showing extracted curves on the original image."""
    import cv2

    # Convert BGR to RGB for matplotlib
    if len(original_img.shape) == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    # Extract plot bounds
    plot_x, plot_y, plot_w, plot_h = plot_bounds

    # Calculate coordinate mapping
    # Must match extraction: use calibration coordinates for both X and Y axes
    # NOTE: x_axis_end has already been adjusted if time_max was overridden
    if calibration is not None:
        # X-axis: origin[0] is time=0, x_axis_end[0] is time=max
        x_0_pixel = calibration.origin[0]
        x_max_pixel = calibration.x_axis_end[0]

        # Y-axis: y_axis_end[1] is survival=1.0 (top), origin[1] is survival=0 (bottom)
        # NOTE: Do NOT apply the +4 offset here - that offset is for extraction only
        # (to account for where detected pixels appear due to line thickness)
        # For overlay, we want to draw at the actual data positions
        y_100_pixel = calibration.y_axis_end[1]  # survival=1.0 at top
        y_0_pixel = calibration.origin[1]  # survival=0.0 at bottom
    else:
        # Fallback to plot bounds
        y_100_pixel = plot_y
        y_0_pixel = plot_y + plot_h
        x_0_pixel = plot_x
        x_max_pixel = plot_x + plot_w

    def data_to_pixel(time, survival):
        pixel_x = x_0_pixel + (time / time_max) * (x_max_pixel - x_0_pixel)
        pixel_y = y_100_pixel + (1.0 - survival) * (y_0_pixel - y_100_pixel)
        return pixel_x, pixel_y

    # Color mapping for curves
    color_map = {
        'red': '#FF0000', 'orange': '#FF4500', 'yellow': '#FFD700',
        'green': '#00FF00', 'cyan': '#00CED1', 'blue': '#0000FF',
        'purple': '#800080', 'magenta': '#FF00FF',
        'solid': '#0000FF', 'dashed': '#FF0000', 'dotted': '#00AA00'
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Show original image
    ax.imshow(img_rgb, extent=[0, img_rgb.shape[1], img_rgb.shape[0], 0])

    # Plot extracted curves
    for curve_data in curves_data:
        points = curve_data['clean_points']
        if not points:
            continue

        style = curve_data['style']
        color = color_map.get(style, '#FF4500')

        # Convert to pixel coordinates using STEP function plotting
        # KM curves are step functions - horizontal until event, then vertical drop
        # We need to create the step shape manually for pixel coordinate plotting
        times = [p[0] for p in points]
        survivals = [p[1] for p in points]

        # Create step function coordinates
        step_times = []
        step_survivals = []
        for i, (t, s) in enumerate(points):
            if i > 0:
                # Add horizontal line at previous survival up to current time
                step_times.append(t)
                step_survivals.append(survivals[i-1])
            # Add the current point (creates vertical drop)
            step_times.append(t)
            step_survivals.append(s)

        # Convert step coordinates to pixels
        pixel_coords = [data_to_pixel(t, s) for t, s in zip(step_times, step_survivals)]
        px_x = [p[0] for p in pixel_coords]
        px_y = [p[1] for p in pixel_coords]

        ax.plot(px_x, px_y, '-', color=color, linewidth=2.5,
                alpha=0.85, label=f"Extracted: {curve_data['name']}")

    ax.set_title('Comparison: Extracted Curves Overlaid on Original', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def find_images(directory="."):
    """Find image files in the specified directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    images = []

    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in image_extensions:
            images.append(path)

    # Sort by modification time (most recent first)
    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return images


def select_image_interactive():
    """Let user select an image from the current directory."""
    print("=" * 60)
    print("KAPLAN-MEIER CURVE EXTRACTOR")
    print("=" * 60)
    print("\nSearching for image files...")

    images = find_images()

    if not images:
        print("\nNo image files found in current directory.")
        print("Supported formats: PNG, JPG, JPEG, BMP, TIFF, GIF")
        print("\nUsage: python extract_km.py <image_path>")
        return None

    print(f"\nFound {len(images)} image(s):\n")

    for i, img in enumerate(images, 1):
        size = img.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  [{i}] {img.name}  ({size_str})")

    print(f"\n  [0] Cancel")

    while True:
        try:
            choice = input("\nSelect image number: ").strip()

            if choice == '0' or choice.lower() == 'q':
                print("Cancelled.")
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return images[idx]
            else:
                print(f"Please enter a number between 1 and {len(images)}")

        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def _check_and_fix_curve_swap(all_curves_data, output_dir, img, plot_bounds, calibration, time_max):
    """
    Ask the user to verify if the extracted curves are correctly labeled.
    If curves are swapped, fix them and regenerate output files.

    Args:
        all_curves_data: List of curve data dictionaries
        output_dir: Path to output directory
        img: Original image
        plot_bounds: Plot region bounds
        calibration: Axis calibration data
        time_max: Maximum time value

    Returns:
        Updated all_curves_data (swapped if needed)
    """
    print("\n" + "-" * 60)
    print("CURVE VERIFICATION")
    print("-" * 60)
    print(f"\nPlease check the comparison overlay image:")
    print(f"  {output_dir / 'comparison_overlay.png'}")
    print("\nVerify that the extracted curves (colored lines) correctly")
    print("overlay the original curves in the image.")
    print("\nCurrent curve labels:")
    for i, curve in enumerate(all_curves_data):
        print(f"  {i+1}. {curve['name']} ({curve['style']})")

    print("\nAre the curve labels SWAPPED (solid/dashed reversed)?")

    while True:
        try:
            response = input("Enter 'y' if swapped, 'n' if correct, or 'q' to quit: ").strip().lower()

            if response == 'q':
                print("Continuing without changes.")
                return all_curves_data
            elif response == 'n':
                print("Curves confirmed as correct.")
                return all_curves_data
            elif response == 'y':
                print("\nSwapping curve labels...")
                all_curves_data = _swap_curves(all_curves_data)

                # Regenerate CSV files with swapped labels
                print("Regenerating output files...")
                for curve_data in all_curves_data:
                    df = pd.DataFrame(curve_data['clean_points'], columns=['Time', 'Survival'])
                    df.to_csv(output_dir / f"curve_{curve_data['name']}.csv", index=False)

                # Regenerate combined CSV
                combined_rows = []
                for curve_data in all_curves_data:
                    for time, survival in curve_data['clean_points']:
                        combined_rows.append({
                            'Curve': curve_data['name'],
                            'Style': curve_data['style'],
                            'Time': time,
                            'Survival': survival
                        })
                combined_df = pd.DataFrame(combined_rows)
                combined_df.to_csv(output_dir / "all_curves.csv", index=False)

                # Regenerate visualization
                _plot_curves(all_curves_data, output_dir / "extracted_curves.png")

                # Regenerate comparison overlay
                _plot_comparison_overlay(
                    img, all_curves_data, plot_bounds, calibration,
                    time_max, output_dir / "comparison_overlay.png"
                )

                print("Curves swapped successfully!")
                print("\nNew curve labels:")
                for i, curve in enumerate(all_curves_data):
                    print(f"  {i+1}. {curve['name']} ({curve['style']})")

                return all_curves_data
            else:
                print("Please enter 'y', 'n', or 'q'")

        except KeyboardInterrupt:
            print("\nContinuing without changes.")
            return all_curves_data
        except EOFError:
            # Non-interactive mode - skip the check
            return all_curves_data


def _swap_curves(all_curves_data):
    """
    Swap the labels (names and styles) between two curves while keeping their data.

    Args:
        all_curves_data: List of two curve data dictionaries

    Returns:
        List with swapped curve labels
    """
    if len(all_curves_data) != 2:
        return all_curves_data

    # Swap the names and styles between the two curves
    curve1, curve2 = all_curves_data[0], all_curves_data[1]

    # Create new curve data with swapped labels but same data
    # curve1 gets curve2's name/style, curve2 gets curve1's name/style
    swapped_curves = [
        {
            'name': curve2['name'],
            'style': curve2['style'],
            'raw_points': curve1.get('raw_points', []),
            'clean_points': curve1.get('clean_points', []),
            'confidence': curve1.get('confidence', 0.0)
        },
        {
            'name': curve1['name'],
            'style': curve1['style'],
            'raw_points': curve2.get('raw_points', []),
            'clean_points': curve2.get('clean_points', []),
            'confidence': curve2.get('confidence', 0.0)
        }
    ]

    return swapped_curves


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Kaplan-Meier survival curves from images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_km.py                     # Interactive mode
  python extract_km.py my_plot.png         # Direct mode
  python extract_km.py my_plot.png --time-max 50
  python extract_km.py my_plot.png --curves 2 --time-max 40

Output:
  Results are saved to results/<image_name>_<timestamp>/
  - all_curves.csv: Combined curve data
  - curve_*.csv: Individual curve files
  - extracted_curves.png: Visualization
        """
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to the Kaplan-Meier plot image (optional - interactive selection if omitted)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: results/<image>_<timestamp>)"
    )

    parser.add_argument(
        "--time-max",
        type=float,
        help="Maximum time value on X-axis (auto-detected if not specified)"
    )

    parser.add_argument(
        "--curves",
        type=int,
        default=2,
        help="Expected number of curves (default: 2)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    parser.add_argument(
        "--ai",
        action="store_true",
        help="Enable AI-assisted detection (slower but may improve accuracy)"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="(Deprecated) AI is now disabled by default"
    )

    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Disable skeleton-based extraction for color images (uses traditional method)"
    )

    args = parser.parse_args()

    # If no image provided, use interactive selection
    if args.image is None:
        selected = select_image_interactive()
        if selected is None:
            sys.exit(0)
        image_path = str(selected)
    else:
        image_path = args.image

    try:
        result = extract_km_curves(
            image_path,
            output_dir=args.output,
            time_max=args.time_max,
            expected_curves=args.curves,
            show_progress=not args.quiet,
            use_ai=args.ai,  # AI disabled by default for faster extraction
            use_skeleton=not args.no_skeleton
        )

        if result is None:
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
