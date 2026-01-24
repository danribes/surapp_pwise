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
from pathlib import Path
from datetime import datetime

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
except ImportError as e:
    print(f"ERROR: Could not import local modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def extract_km_curves(
    image_path: str,
    output_dir: str = None,
    time_max: float = None,
    expected_curves: int = 2,
    show_progress: bool = True
):
    """
    Extract Kaplan-Meier curves from an image.

    Args:
        image_path: Path to the KM plot image
        output_dir: Output directory (default: results/<image_name>_<timestamp>)
        time_max: Maximum time value on X-axis (auto-detected if not specified)
        expected_curves: Expected number of curves (default: 2)
        show_progress: Print progress messages

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

    # Step 2: Calibrate axes
    if show_progress:
        print("\n[Step 2/4] Calibrating axes...")

    calibrator = AxisCalibrator(img)
    calibration = calibrator.calibrate()

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

        # Refine Y-axis bounds by detecting where curves start (at survival=100%)
        refined_bounds = calibrator.refine_plot_bounds_from_curves()
        if refined_bounds is not None:
            plot_bounds = refined_bounds
            if show_progress:
                print(f"  Plot area (refined): x={plot_bounds[0]}, y={plot_bounds[1]}, "
                      f"w={plot_bounds[2]}, h={plot_bounds[3]}")

    # Override time_max if specified
    if time_max is not None and calibration is not None:
        calibration.x_data_range = (calibration.x_data_range[0], time_max)
        if show_progress:
            print(f"  X-axis range (override): 0 - {time_max}")

    # Step 3: Detect and trace curves
    if show_progress:
        print("\n[Step 3/4] Detecting curves...")

    # Choose detection method based on image type
    use_color_detection = is_color_image(img)

    if use_color_detection:
        if show_progress:
            print("  Using COLOR-based detection...")
        detector = ColorCurveDetector(img, plot_bounds)
        detected_curves = detector.detect_all_curves(
            expected_count=expected_curves,
            debug_dir=str(output_dir)
        )
    else:
        if show_progress:
            print("  Using LINE-STYLE detection (solid/dashed)...")
        detector = LineStyleDetector(img, plot_bounds, filter_reference_lines=True)
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

    # Set up coordinate conversion
    plot_x, plot_y, plot_w, plot_h = plot_bounds

    if calibration is not None:
        time_min, time_max_cal = calibration.x_data_range
        survival_min, survival_max = calibration.y_data_range
        # Use overridden time_max if specified
        if time_max is not None:
            time_max_cal = time_max
    else:
        time_min = 0.0
        time_max_cal = time_max if time_max else 10.0
        survival_min, survival_max = 0.0, 1.0

    # Create pixel_to_coord function using our (possibly overridden) ranges
    def pixel_to_coord(px_x, px_y):
        t = time_min + (px_x - plot_x) / plot_w * (time_max_cal - time_min)
        s = survival_max - (px_y - plot_y) / plot_h * (survival_max - survival_min)
        return t, s

    # Step 4: Extract and export data
    if show_progress:
        print("\n[Step 4/4] Extracting coordinates...")

    all_curves_data = []

    for i, curve in enumerate(detected_curves):
        # Extract points
        raw_points = detector.extract_curve_points(curve, pixel_to_coord)

        if not raw_points:
            if show_progress:
                print(f"  WARNING: No points extracted for curve {i+1}")
            continue

        # Get curve name and style based on detection method
        if use_color_detection:
            curve_name = curve.name
            curve_style = curve.name.split('_')[0]  # e.g., "orange" from "orange_1"
        else:
            curve_name = f"{curve.style.value}_{i+1}"
            curve_style = curve.style.value

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

        curve_data = {
            'name': curve_name,
            'style': curve_style,
            'raw_points': raw_points,
            'clean_points': clean_points,
            'confidence': curve.confidence
        }
        all_curves_data.append(curve_data)

        if show_progress:
            if clean_points:
                t_range = f"{clean_points[0][0]:.2f} - {clean_points[-1][0]:.2f}"
                s_range = f"{min(p[1] for p in clean_points):.2f} - {max(p[1] for p in clean_points):.2f}"
                print(f"  {curve_name}: {len(clean_points)} points, "
                      f"time: {t_range}, survival: {s_range}")
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

    # Generate visualization
    _plot_curves(all_curves_data, output_dir / "extracted_curves.png")

    # Generate comparison overlay (extracted curves on original image)
    _plot_comparison_overlay(
        img, all_curves_data, plot_bounds,
        time_max if time_max else (calibration.x_data_range[1] if calibration else 12.0),
        output_dir / "comparison_overlay.png"
    )

    if show_progress:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nOutput files saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  - all_curves.csv        : Combined data from all curves")
        for curve_data in all_curves_data:
            print(f"  - curve_{curve_data['name']}.csv : {curve_data['style']} curve data")
        print("  - extracted_curves.png  : Visualization plot")
        print("  - comparison_overlay.png: Original with extracted curves overlay")
        print("  - debug_*.png           : Debug images")

    return {
        'curves': all_curves_data,
        'output_dir': str(output_dir),
        'calibration': calibration
    }


def _clean_curve_data(points, tolerance=0.005):
    """Clean curve data by removing noise and enforcing monotonicity.

    KM curves MUST start at survival=1.0 at time=0 (all subjects alive initially).
    If max detected survival is significantly below 1.0, rescale all values.
    """
    if not points:
        return []

    # Sort by time
    sorted_points = sorted(points, key=lambda p: p[0])

    # Remove duplicate times (keep point with median survival)
    time_groups = {}
    for t, s in sorted_points:
        t_rounded = round(t, 3)
        if t_rounded not in time_groups:
            time_groups[t_rounded] = []
        time_groups[t_rounded].append(s)

    deduplicated = []
    for t in sorted(time_groups.keys()):
        survivals = time_groups[t]
        median_s = sorted(survivals)[len(survivals) // 2]
        deduplicated.append((t, median_s))

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
    # If first point is not at time 0, add (0, 1.0)
    if first_t > 0.01:
        deduplicated.insert(0, (0.0, 1.0))
    # If first point is at time ~0 but survival != 1.0, fix it
    elif first_s < 0.99:
        deduplicated[0] = (0.0, 1.0)

    # Enforce monotonicity (survival can only decrease or stay flat)
    monotonic = []
    max_survival = 1.0

    for t, s in deduplicated:
        s = min(s, max_survival)  # Cap at previous maximum
        monotonic.append((t, s))
        max_survival = s

    # Remove consecutive duplicates (within tolerance)
    cleaned = [monotonic[0]] if monotonic else []
    for i in range(1, len(monotonic)):
        t, s = monotonic[i]
        prev_t, prev_s = cleaned[-1]

        # Keep if time changed significantly or survival dropped
        if t - prev_t > tolerance or prev_s - s > tolerance:
            cleaned.append((t, s))

    return cleaned


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


def _plot_comparison_overlay(original_img, curves_data, plot_bounds, time_max, output_path):
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
    # y=plot_y corresponds to survival=1.0, y=plot_y+plot_h corresponds to survival=0
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

        # Convert to pixel coordinates
        pixel_coords = [data_to_pixel(t, s) for t, s in points]
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
            show_progress=not args.quiet
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
