#!/usr/bin/env python3
"""
Step 2: Calibrate Axes and Detect Plot Area

This script detects the axes in a Kaplan-Meier plot and determines
the coordinate system for extracting curve data.

Usage:
    python step2_calibrate_axes.py <image_path> [--time-max TIME]

Output:
    - Displays detected axis information
    - Saves visualization to results/step2_axes.png
    - Saves calibration data to results/step2_calibration.txt
"""

import sys
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies
try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)

try:
    from lib import AxisCalibrator
    CALIBRATOR_AVAILABLE = True
except ImportError:
    CALIBRATOR_AVAILABLE = False


def calibrate_axes(image_path, time_max=None):
    """Detect and calibrate plot axes."""
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None

    print("=" * 60)
    print("STEP 2: AXIS CALIBRATION")
    print("=" * 60)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None

    height, width = img.shape[:2]
    print(f"\nImage: {image_path.name} ({width}x{height})")

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if not CALIBRATOR_AVAILABLE:
        print("\nWARNING: Axis calibrator module not available.")
        print("Using estimated plot bounds (10% margins).")

        plot_x = int(width * 0.10)
        plot_y = int(height * 0.10)
        plot_w = int(width * 0.80)
        plot_h = int(height * 0.80)

        x_min, x_max = 0.0, time_max if time_max else 10.0
        y_min, y_max = 0.0, 1.0

        calibration = {
            'plot_bounds': (plot_x, plot_y, plot_w, plot_h),
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'method': 'estimated'
        }
    else:
        print("\nDetecting axes...")

        calibrator = AxisCalibrator(img)
        result = calibrator.calibrate()

        if result is None:
            print("  WARNING: Automatic calibration failed.")
            print("  Using estimated plot bounds.")

            plot_x = int(width * 0.10)
            plot_y = int(height * 0.10)
            plot_w = int(width * 0.80)
            plot_h = int(height * 0.80)

            x_min, x_max = 0.0, time_max if time_max else 10.0
            y_min, y_max = 0.0, 1.0

            calibration = {
                'plot_bounds': (plot_x, plot_y, plot_w, plot_h),
                'x_range': (x_min, x_max),
                'y_range': (y_min, y_max),
                'method': 'estimated',
                'calibrator': calibrator
            }
        else:
            plot_x, plot_y, plot_w, plot_h = result.plot_rectangle
            x_min, x_max = result.x_data_range
            y_min, y_max = result.y_data_range

            # Override time_max if specified
            if time_max is not None:
                x_max = time_max

            calibration = {
                'plot_bounds': (plot_x, plot_y, plot_w, plot_h),
                'x_range': (x_min, x_max),
                'y_range': (y_min, y_max),
                'method': 'automatic',
                'result': result,
                'calibrator': calibrator
            }

    # Display results
    plot_x, plot_y, plot_w, plot_h = calibration['plot_bounds']
    x_min, x_max = calibration['x_range']
    y_min, y_max = calibration['y_range']

    print(f"\nCalibration Results ({calibration['method']}):")
    print(f"\n  Plot Area (pixels):")
    print(f"    X: {plot_x} to {plot_x + plot_w}")
    print(f"    Y: {plot_y} to {plot_y + plot_h}")
    print(f"    Width:  {plot_w} pixels")
    print(f"    Height: {plot_h} pixels")

    print(f"\n  Data Range:")
    print(f"    X-axis (Time):     {x_min} to {x_max}")
    print(f"    Y-axis (Survival): {y_min} to {y_max}")

    print(f"\n  Pixel-to-Coordinate Conversion:")
    x_scale = (x_max - x_min) / plot_w if plot_w > 0 else 0
    y_scale = (y_max - y_min) / plot_h if plot_h > 0 else 0
    print(f"    X scale: {x_scale:.4f} units/pixel")
    print(f"    Y scale: {y_scale:.4f} units/pixel")

    # Create visualization
    vis = img.copy()

    # Draw plot area rectangle
    cv2.rectangle(vis,
                  (plot_x, plot_y),
                  (plot_x + plot_w, plot_y + plot_h),
                  (0, 255, 0), 2)

    # Draw axes
    cv2.line(vis, (plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h),
             (255, 0, 0), 2)  # X-axis (blue)
    cv2.line(vis, (plot_x, plot_y), (plot_x, plot_y + plot_h),
             (0, 0, 255), 2)  # Y-axis (red)

    # Add labels
    cv2.putText(vis, f"X: {x_min}-{x_max}", (plot_x + 10, plot_y + plot_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(vis, f"Y: {y_min}-{y_max}", (plot_x - 80, plot_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw corner markers
    corners = [
        (plot_x, plot_y, "TL"),
        (plot_x + plot_w, plot_y, "TR"),
        (plot_x, plot_y + plot_h, "BL (Origin)"),
        (plot_x + plot_w, plot_y + plot_h, "BR")
    ]
    for x, y, label in corners:
        cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)

    # Save visualization
    vis_path = output_dir / "step2_axes.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"\nVisualization saved to: {vis_path}")

    # Save calibration data
    cal_path = output_dir / "step2_calibration.txt"
    with open(cal_path, 'w') as f:
        f.write("Axis Calibration Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Image: {image_path.name}\n")
        f.write(f"Method: {calibration['method']}\n\n")
        f.write("Plot Area (pixels):\n")
        f.write(f"  x: {plot_x}\n")
        f.write(f"  y: {plot_y}\n")
        f.write(f"  width: {plot_w}\n")
        f.write(f"  height: {plot_h}\n\n")
        f.write("Data Range:\n")
        f.write(f"  x_min: {x_min}\n")
        f.write(f"  x_max: {x_max}\n")
        f.write(f"  y_min: {y_min}\n")
        f.write(f"  y_max: {y_max}\n")

    print(f"Calibration data saved to: {cal_path}")

    print("\n" + "=" * 60)
    print("NEXT STEP: Run step3_extract_curves.py to detect and extract curves")
    print("=" * 60)

    return calibration


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

    # Parse --time-max argument first
    if "--time-max" in sys.argv:
        idx = sys.argv.index("--time-max")
        if idx + 1 < len(sys.argv):
            try:
                time_max = float(sys.argv[idx + 1])
            except ValueError:
                print(f"ERROR: Invalid time-max value: {sys.argv[idx + 1]}")
                sys.exit(1)

    # Get image path - either from argument or interactive selection
    # Filter out --time-max and its value from argv
    args = [a for i, a in enumerate(sys.argv[1:], 1)
            if a != "--time-max" and (i == 1 or sys.argv[i-1] != "--time-max")]

    if args:
        image_path = args[0]
    else:
        selected = select_image_interactive()
        if selected is None:
            sys.exit(0)
        image_path = str(selected)

    calibrate_axes(image_path, time_max)


if __name__ == "__main__":
    main()
