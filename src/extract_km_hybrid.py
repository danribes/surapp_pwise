#!/usr/bin/env python3
"""
Hybrid AI + Pixel-based Kaplan-Meier Curve Extraction.

This script combines:
1. AI for curve IDENTIFICATION (colors, styles, count)
2. Color-based ISOLATION (separate image per curve)
3. Pixel-based EXTRACTION (accurate coordinates)
4. AI for VALIDATION (quality check)

Usage:
    python src/extract_km_hybrid.py input/image.png --time-max 24

    # With manual calibration
    python src/extract_km_hybrid.py input/image.png --time-max 24 \\
        --x0 100 --xmax 800 --y0 500 --y100 50

    # With auto-calibration
    python src/extract_km_hybrid.py input/image.png --time-max 24 --auto-calibrate
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.hybrid_extractor import HybridExtractor, hybrid_extract


def auto_calibrate(image_path: str, time_max: float, quiet: bool = False) -> dict:
    """
    Attempt automatic axis calibration.

    Returns:
        Calibration dictionary
    """
    from lib.calibrator import AxisCalibrator

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    calibrator = AxisCalibrator(img)
    result = calibrator.calibrate(time_max_hint=time_max)

    if result is None:
        raise ValueError("Auto-calibration failed. Please provide manual calibration.")

    # Extract pixel coordinates from AxisCalibrationResult
    # origin is (x, y) at data coordinates (0, 0) - bottom left of plot
    # x_axis_end is right end of x-axis
    # y_axis_end is top end of y-axis

    return {
        'x_0_pixel': int(result.origin[0]),
        'x_max_pixel': int(result.x_axis_end[0]),
        'y_0_pixel': int(result.origin[1]),  # y for survival=0 (bottom)
        'y_100_pixel': int(result.y_axis_end[1]),  # y for survival=100% (top)
        'time_max': float(time_max)
    }


def interactive_calibrate(image_path: str, time_max: float) -> dict:
    """
    Interactive calibration by clicking on the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    points = []
    point_names = [
        "origin (0, 0%)",
        "X-axis max (time_max, 0%)",
        "Y-axis max (0, 100%)"
    ]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Calibration', img)

    print("\nInteractive Calibration")
    print("=" * 40)
    print("Click on 3 points in this order:")
    for i, name in enumerate(point_names):
        print(f"  {i+1}. {name}")
    print("\nPress 'r' to reset, 'q' to cancel")

    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Calibration', mouse_callback)
    cv2.imshow('Calibration', img)

    while len(points) < 3:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('r'):
            points.clear()
            img = cv2.imread(image_path)
            cv2.imshow('Calibration', img)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise ValueError("Calibration cancelled")

    cv2.destroyAllWindows()

    origin = points[0]
    x_max_point = points[1]
    y_max_point = points[2]

    return {
        'x_0_pixel': origin[0],
        'x_max_pixel': x_max_point[0],
        'y_0_pixel': origin[1],  # y=0% (bottom of curve area)
        'y_100_pixel': y_max_point[1],  # y=100% (top of curve area)
        'time_max': time_max
    }


def create_comparison_plot(image_path: str,
                            result,
                            calibration: dict,
                            output_dir: str):
    """
    Create a comparison plot showing original and extracted curves.
    """
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Extracted curves
    colors = {
        'green': 'green',
        'red': 'red',
        'blue': 'blue',
        'cyan': 'cyan',
        'magenta': 'magenta',
        'yellow': 'gold',
        'orange': 'orange',
        'black': 'black',
        'gray': 'gray',
        'white': 'lightgray',
    }

    for curve in result.curves:
        df = curve.dataframe
        color = colors.get(curve.identification.color, 'blue')
        label = curve.identification.legend_label or curve.identification.color
        axes[1].plot(df['Time'], df['Survival'], color=color, linewidth=2, label=label)

    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Survival Probability')
    axes[1].set_title('Extracted Curves')
    axes[1].set_xlim(0, calibration['time_max'])
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_overlay_plot(image_path: str,
                         result,
                         calibration: dict,
                         output_dir: str):
    """
    Create an overlay showing extracted curves on original image.
    """
    original = cv2.imread(image_path)

    # Define colors (BGR)
    colors_bgr = {
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'yellow': (0, 255, 255),
        'orange': (0, 165, 255),
        'black': (128, 128, 128),  # Use gray for visibility
        'gray': (200, 200, 200),
        'white': (255, 255, 255),
    }

    x_0 = calibration['x_0_pixel']
    x_max = calibration['x_max_pixel']
    y_0 = calibration['y_0_pixel']
    y_100 = calibration['y_100_pixel']
    time_max = calibration['time_max']

    for curve in result.curves:
        df = curve.dataframe
        color = colors_bgr.get(curve.identification.color, (0, 255, 0))

        # Convert time/survival to pixel coordinates
        prev_point = None
        for _, row in df.iterrows():
            t, s = row['Time'], row['Survival']
            x = int(x_0 + (t / time_max) * (x_max - x_0))
            y = int(y_0 - s * (y_0 - y_100))

            if prev_point:
                cv2.line(original, prev_point, (x, y), color, 2)
            prev_point = (x, y)

    cv2.imwrite(os.path.join(output_dir, 'overlay.png'), original)


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid AI + Pixel-based KM curve extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-calibration mode
    python src/extract_km_hybrid.py input/image.png --time-max 24 --auto-calibrate

    # Manual calibration
    python src/extract_km_hybrid.py input/image.png --time-max 24 \\
        --x0 100 --xmax 800 --y0 500 --y100 50

    # Interactive calibration (click to calibrate)
    python src/extract_km_hybrid.py input/image.png --time-max 24 --interactive
        """
    )

    parser.add_argument('image', help='Path to KM plot image')
    parser.add_argument('--time-max', '-t', type=float, required=True,
                        help='Maximum time value on X-axis')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: results/hybrid_<basename>)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress messages')
    parser.add_argument('--no-ai', action='store_true',
                        help='Disable AI identification, use color detection only')

    # Calibration options
    calib_group = parser.add_argument_group('Calibration')
    calib_group.add_argument('--auto-calibrate', '-a', action='store_true',
                             help='Attempt automatic calibration')
    calib_group.add_argument('--interactive', '-i', action='store_true',
                             help='Interactive calibration (click on image)')
    calib_group.add_argument('--x0', type=int, help='X pixel for time=0')
    calib_group.add_argument('--xmax', type=int, help='X pixel for time=max')
    calib_group.add_argument('--y0', type=int, help='Y pixel for survival=0%')
    calib_group.add_argument('--y100', type=int, help='Y pixel for survival=100%')

    # Import options
    import_group = parser.add_argument_group('Import existing curves')
    import_group.add_argument('--import-black', type=str, metavar='CSV',
                              help='Import black curve from existing CSV file')

    args = parser.parse_args()

    # Validate image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        basename = Path(args.image).stem
        output_dir = f"results/hybrid_{basename}"

    os.makedirs(output_dir, exist_ok=True)

    # Get calibration
    if args.interactive:
        calibration = interactive_calibrate(args.image, args.time_max)
    elif args.auto_calibrate:
        print("Attempting auto-calibration...")
        try:
            calibration = auto_calibrate(args.image, args.time_max, args.quiet)
            print(f"Auto-calibration successful:")
            print(f"  X: {calibration['x_0_pixel']} - {calibration['x_max_pixel']}")
            print(f"  Y: {calibration['y_100_pixel']} - {calibration['y_0_pixel']}")
        except Exception as e:
            print(f"Auto-calibration failed: {e}")
            print("Please provide manual calibration or use --interactive")
            sys.exit(1)
    elif all([args.x0, args.xmax, args.y0, args.y100]):
        calibration = {
            'x_0_pixel': args.x0,
            'x_max_pixel': args.xmax,
            'y_0_pixel': args.y0,
            'y_100_pixel': args.y100,
            'time_max': args.time_max
        }
    else:
        print("Error: Must provide calibration. Use one of:")
        print("  --auto-calibrate")
        print("  --interactive")
        print("  --x0 X --xmax X --y0 Y --y100 Y")
        sys.exit(1)

    # Save calibration
    with open(os.path.join(output_dir, 'calibration.json'), 'w') as f:
        json.dump(calibration, f, indent=2)

    # Run hybrid extraction
    try:
        extractor = HybridExtractor(quiet=args.quiet, use_ai=not args.no_ai)
        result = extractor.extract(
            args.image,
            calibration,
            time_max=args.time_max,
            output_dir=output_dir
        )

        # Import black curve if specified
        if args.import_black:
            if not args.quiet:
                print(f"\nImporting black curve from: {args.import_black}")
            import_df = pd.read_csv(args.import_black)

            # Create an ExtractedCurveResult for the imported curve
            from lib.hybrid_extractor import ExtractedCurveResult, CurveIdentification
            imported_curve = ExtractedCurveResult(
                identification=CurveIdentification(
                    color='black',
                    style='solid',
                    description='black curve (imported)',
                    position='middle'
                ),
                dataframe=import_df,
                points_count=len(import_df),
                time_range=(import_df['Time'].min(), import_df['Time'].max()),
                survival_range=(import_df['Survival'].min(), import_df['Survival'].max()),
                confidence=1.0,
                extraction_method='imported'
            )

            # Replace any existing black curve
            result.curves = [c for c in result.curves if c.identification.color != 'black']
            result.curves.append(imported_curve)

            # Update combined_df
            combined_dfs = []
            for curve in result.curves:
                df = curve.dataframe.copy()
                df['Curve'] = curve.identification.color
                combined_dfs.append(df)
            result.combined_df = pd.concat(combined_dfs, ignore_index=True)

            # Save updated files
            import_df.to_csv(os.path.join(output_dir, 'curve_black.csv'), index=False)
            result.combined_df.to_csv(os.path.join(output_dir, 'all_curves.csv'), index=False)

        # Create visualizations
        if not args.quiet:
            print("\nCreating visualizations...")

        create_comparison_plot(args.image, result, calibration, output_dir)
        create_overlay_plot(args.image, result, calibration, output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Image: {args.image}")
        print(f"Output: {output_dir}")
        print(f"\nCurves extracted: {len(result.curves)}")

        for curve in result.curves:
            print(f"\n  {curve.identification.color.upper()}:")
            print(f"    Points: {curve.points_count}")
            print(f"    Time range: {curve.time_range[0]:.1f} - {curve.time_range[1]:.1f}")
            print(f"    Survival: {curve.survival_range[0]:.1%} - {curve.survival_range[1]:.1%}")

        if result.validation_result:
            print(f"\nAI Validation:")
            print(f"  Valid: {result.validation_result.get('is_valid', 'N/A')}")
            print(f"  Confidence: {result.validation_result.get('confidence', 'N/A')}")

        print("\nOutput files:")
        for f in os.listdir(output_dir):
            print(f"  {f}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
