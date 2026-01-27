#!/usr/bin/env python3
"""
Step 3: Detect and Extract Curve Data

This script detects Kaplan-Meier curves (solid and dashed) and extracts
their coordinates to CSV files.

Usage:
    python step3_extract_curves.py <image_path> [--time-max TIME] [--curves N]

Output:
    - Displays detected curves and their properties
    - Saves CSV files with extracted coordinates
    - Saves visualization and debug images
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies
try:
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install opencv-python numpy pandas matplotlib")
    sys.exit(1)

try:
    from lib import LineStyleDetector, AxisCalibrator, is_grayscale_image
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import detection modules: {e}")
    MODULES_AVAILABLE = False


def extract_curves(image_path, time_max=None, expected_curves=2):
    """Detect and extract KM curves from image."""
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None

    if not MODULES_AVAILABLE:
        print("ERROR: Required modules not available.")
        return None

    print("=" * 60)
    print("STEP 3: CURVE EXTRACTION")
    print("=" * 60)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None

    height, width = img.shape[:2]
    print(f"\nImage: {image_path.name} ({width}x{height})")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"{image_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save original
    cv2.imwrite(str(output_dir / "original.png"), img)

    # Calibrate axes
    print("\n[1/3] Calibrating axes...")
    calibrator = AxisCalibrator(img)
    calibration = calibrator.calibrate()

    if calibration is None:
        print("  WARNING: Auto-calibration failed, using estimates.")
        plot_bounds = (
            int(width * 0.10),
            int(height * 0.10),
            int(width * 0.80),
            int(height * 0.80)
        )
        x_min, x_max = 0.0, time_max if time_max else 10.0
        y_min, y_max = 0.0, 1.0
    else:
        plot_bounds = calibration.plot_rectangle
        x_min, x_max = calibration.x_data_range
        y_min, y_max = calibration.y_data_range

        if time_max is not None:
            x_max = time_max

        print(f"  Plot bounds: {plot_bounds}")
        print(f"  X range: {x_min} - {x_max}")
        print(f"  Y range: {y_min} - {y_max}")

    # Create pixel_to_coord function using our (possibly overridden) ranges
    px, py, pw, ph = plot_bounds

    def pixel_to_coord(px_x, px_y):
        t = x_min + (px_x - px) / pw * (x_max - x_min)
        s = y_max - (px_y - py) / ph * (y_max - y_min)
        return t, s

    # Detect curves
    print("\n[2/3] Detecting curves...")
    detector = LineStyleDetector(img, plot_bounds, filter_reference_lines=True)
    detected_curves = detector.detect_all_curves(
        expected_count=expected_curves,
        debug_dir=str(output_dir)
    )

    print(f"\n  Found {len(detected_curves)} curves:")
    for i, curve in enumerate(detected_curves):
        print(f"    {i+1}. {curve.style.value.upper()} curve "
              f"(confidence: {curve.confidence:.2f})")

    if not detected_curves:
        print("\n  ERROR: No curves detected!")
        print("  Check the debug images for preprocessing issues.")
        return None

    # Save debug images
    debug_img = detector.get_debug_image()
    cv2.imwrite(str(output_dir / "debug_detection.png"), debug_img)

    binary_mask = detector.get_binary_mask()
    if binary_mask is not None:
        cv2.imwrite(str(output_dir / "debug_binary.png"), binary_mask)

    # Extract curve data
    print("\n[3/3] Extracting coordinates...")

    all_curves = []

    for i, curve in enumerate(detected_curves):
        curve_name = f"{curve.style.value}_{i+1}"
        print(f"\n  Processing {curve_name}...")

        # Get raw points
        raw_points = detector.extract_curve_points(curve, pixel_to_coord)

        if not raw_points:
            print(f"    WARNING: No points extracted")
            continue

        print(f"    Raw points: {len(raw_points)}")

        # Clean the data
        clean_points = _clean_data(raw_points)
        print(f"    Clean points: {len(clean_points)}")

        if clean_points:
            t_min = min(p[0] for p in clean_points)
            t_max = max(p[0] for p in clean_points)
            s_min = min(p[1] for p in clean_points)
            s_max = max(p[1] for p in clean_points)
            print(f"    Time range: {t_min:.2f} - {t_max:.2f}")
            print(f"    Survival range: {s_min:.2f} - {s_max:.2f}")

        curve_data = {
            'name': curve_name,
            'style': curve.style.value,
            'raw_points': raw_points,
            'clean_points': clean_points,
            'confidence': curve.confidence
        }
        all_curves.append(curve_data)

        # Save individual CSV
        df = pd.DataFrame(clean_points, columns=['Time', 'Survival'])
        csv_path = output_dir / f"curve_{curve_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path.name}")

    # Save combined CSV
    print("\n  Saving combined data...")
    combined_rows = []
    for curve_data in all_curves:
        for t, s in curve_data['clean_points']:
            combined_rows.append({
                'Curve': curve_data['name'],
                'Style': curve_data['style'],
                'Time': t,
                'Survival': s
            })

    combined_df = pd.DataFrame(combined_rows)
    combined_path = output_dir / "all_curves.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"    Saved: {combined_path.name}")

    # Generate plot
    print("\n  Generating visualization...")
    _create_plot(all_curves, output_dir / "extracted_curves.png")
    print(f"    Saved: extracted_curves.png")

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"\nResults saved to: {output_dir}/")
    print("\nExtracted Data:")
    for curve_data in all_curves:
        pts = curve_data['clean_points']
        if pts:
            print(f"\n  {curve_data['name']} ({curve_data['style']}):")
            print(f"    Points: {len(pts)}")
            print(f"    Time:   {pts[0][0]:.2f} - {pts[-1][0]:.2f}")
            print(f"    Start survival: {pts[0][1]:.4f}")
            print(f"    End survival:   {pts[-1][1]:.4f}")

    print("\nFiles created:")
    print(f"  - all_curves.csv          (combined data)")
    for curve_data in all_curves:
        print(f"  - curve_{curve_data['name']}.csv")
    print(f"  - extracted_curves.png    (visualization)")
    print(f"  - debug_*.png             (debug images)")

    print("\n" + "=" * 60)
    print("You can now use the CSV files in your analysis.")
    print("=" * 60)

    return {
        'curves': all_curves,
        'output_dir': output_dir
    }


def _clean_data(points, tolerance=0.005):
    """Clean curve data.

    KM curves MUST start at survival=1.0 at time=0 (all subjects alive initially).
    If max detected survival is significantly below 1.0, rescale all values.

    This function also removes points that violate monotonicity (likely from
    censoring tick marks) rather than just capping them.
    """
    if not points:
        return []

    # Sort by time
    sorted_pts = sorted(points, key=lambda p: p[0])

    # Remove duplicate times - use the MINIMUM survival for each time
    # (tick marks create higher values, so min gives the true curve)
    time_groups = {}
    for t, s in sorted_pts:
        t_rounded = round(t, 3)
        if t_rounded not in time_groups:
            time_groups[t_rounded] = []
        time_groups[t_rounded].append(s)

    deduplicated = []
    for t in sorted(time_groups.keys()):
        survivals = time_groups[t]
        # Use minimum survival instead of median to filter out tick marks
        # Tick marks extend ABOVE the curve, so min gives the true curve value
        min_s = min(survivals)
        deduplicated.append((t, min_s))

    if not deduplicated:
        return []

    # Find max survival in the data (should be ~1.0 for KM curves)
    max_detected = max(s for _, s in deduplicated)

    # If max survival is significantly below 1.0, rescale
    if max_detected < 0.95 and max_detected > 0.1:
        scale_factor = 1.0 / max_detected
        deduplicated = [(t, min(1.0, s * scale_factor)) for t, s in deduplicated]

    # KM curves MUST start at (0, 1.0) - all subjects alive at time 0
    first_t, first_s = deduplicated[0]
    if first_t > 0.01:
        deduplicated.insert(0, (0.0, 1.0))
    elif first_s < 0.99:
        deduplicated[0] = (0.0, 1.0)

    # Enforce monotonicity by REMOVING violating points (not just capping)
    # This filters out tick mark artifacts that temporarily go up
    monotonic = []
    max_s = 1.0
    for t, s in deduplicated:
        # Only keep points that don't go up (within tolerance)
        if s <= max_s + tolerance:
            s = min(s, max_s)  # Cap small noise
            monotonic.append((t, s))
            max_s = s
        # else: skip this point (tick mark artifact)

    # Remove consecutive near-duplicates (keeps clean step function)
    cleaned = [monotonic[0]] if monotonic else []
    for i in range(1, len(monotonic)):
        t, s = monotonic[i]
        prev_t, prev_s = cleaned[-1]
        if t - prev_t > tolerance or prev_s - s > tolerance:
            cleaned.append((t, s))

    return cleaned


def _create_plot(curves_data, output_path):
    """Create visualization plot."""
    plt.figure(figsize=(10, 6))

    colors = {'solid': 'blue', 'dashed': 'red', 'dotted': 'green'}
    linestyles = {'solid': '-', 'dashed': '--', 'dotted': ':'}

    for curve_data in curves_data:
        pts = curve_data['clean_points']
        if not pts:
            continue

        times = [p[0] for p in pts]
        survivals = [p[1] for p in pts]
        style = curve_data['style']

        plt.step(times, survivals,
                 where='post',
                 color=colors.get(style, 'black'),
                 linestyle=linestyles.get(style, '-'),
                 linewidth=2,
                 label=curve_data['name'])

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.title('Extracted Kaplan-Meier Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_images(directory="."):
    """Find image files in the specified directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    images = []
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in image_extensions:
            images.append(path)
    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return images


def select_image_interactive():
    """Let user select an image from the current directory."""
    print("Searching for image files...\n")
    images = find_images()

    if not images:
        print("No image files found in current directory.")
        return None

    print(f"Found {len(images)} image(s):\n")
    for i, img in enumerate(images, 1):
        size = img.stat().st_size
        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"
        print(f"  [{i}] {img.name}  ({size_str})")
    print(f"\n  [0] Cancel")

    while True:
        try:
            choice = input("\nSelect image number: ").strip()
            if choice == '0':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return images[idx]
            print(f"Please enter 1-{len(images)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None


def main():
    time_max = None
    expected_curves = 2
    image_path = None

    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--time-max" and i + 1 < len(args):
            try:
                time_max = float(args[i + 1])
            except ValueError:
                print(f"ERROR: Invalid time-max: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--curves" and i + 1 < len(args):
            try:
                expected_curves = int(args[i + 1])
            except ValueError:
                print(f"ERROR: Invalid curves count: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif not args[i].startswith("--") and image_path is None:
            image_path = args[i]
            i += 1
        else:
            print(f"WARNING: Unknown argument: {args[i]}")
            i += 1

    # If no image provided, use interactive selection
    if image_path is None:
        selected = select_image_interactive()
        if selected is None:
            sys.exit(0)
        image_path = str(selected)

    extract_curves(image_path, time_max, expected_curves)


if __name__ == "__main__":
    main()
