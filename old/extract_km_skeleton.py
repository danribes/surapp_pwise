#!/usr/bin/env python3
"""
Skeleton-based Kaplan-Meier curve extraction.

This script uses OCR text removal + skeletonization for clean curve extraction.
It avoids contamination from legend text and annotations that can cause
curves to connect to text regions.
"""

import argparse
import cv2
import csv
import json
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skimage.morphology import skeletonize
from lib.color_detector import detect_curve_colors
from lib.text_remover import remove_all_text
from lib.calibrator import AxisCalibrator


def extract_curves_skeleton(img, colors, plot_bounds, time_max):
    """
    Extract curves using text removal and skeletonization.

    Args:
        img: BGR image
        colors: List of color info dicts
        plot_bounds: (x, y, w, h) of plot area
        time_max: Maximum time value for x-axis

    Returns:
        List of curve data dicts with 'name', 'points', 'color_info'
    """
    print("\n[Skeleton Extraction]")

    # Step 1: Remove all text
    print("  Removing all text...")
    cleaned_img, text_mask = remove_all_text(img)

    # Step 2: Create HSV for color detection
    hsv = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2HSV)

    plot_x, plot_y, plot_w, plot_h = plot_bounds
    margin = 5
    px = max(0, plot_x - margin)
    py = max(0, plot_y - margin)
    pw = min(img.shape[1] - px, plot_w + 2*margin)
    ph = min(img.shape[0] - py, plot_h + 2*margin)

    curves_data = []

    for idx, color_info in enumerate(colors):
        name = color_info.get('name', f'curve_{idx+1}')
        print(f"  Processing {name}...")

        # Create color mask
        mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])

        # Crop to plot area with margin
        mask_roi = mask[py:py+ph, px:px+pw]

        # Clean up mask - remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_clean = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)

        # Remove small connected components (noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)
        min_area = 50  # Minimum pixels for a valid curve segment

        mask_filtered = np.zeros_like(mask_clean)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask_filtered[labels == label] = 255

        # Skeletonize
        skeleton = skeletonize(mask_filtered > 0).astype(np.uint8) * 255

        # Extract points from skeleton
        ys, xs = np.where(skeleton > 0)
        print(f"    Found {len(xs)} skeleton points")

        # Convert to coordinates
        points = []
        for x, y in zip(xs, ys):
            full_x = px + x
            full_y = py + y

            # Convert pixel to data coordinates
            t = ((full_x - plot_x) / plot_w) * time_max
            s = 1.0 - ((full_y - plot_y) / plot_h)
            s = max(0.0, min(1.0, s))

            if 0 <= t <= time_max:
                points.append((t, s))

        # Sort by time
        points.sort(key=lambda p: p[0])

        # Remove duplicates and get one survival value per time point
        clean_points = _clean_curve_points(points)

        curves_data.append({
            'name': name,
            'style': name.split('_')[0] if '_' in name else name,
            'raw_points': points,
            'clean_points': clean_points,
            'color_info': color_info,
        })

        # Print sample values
        print(f"    Sample values:")
        for t_target in [0, 6, 12, 18, 24]:
            if t_target <= time_max:
                matching = [p for p in clean_points if abs(p[0] - t_target) < 0.5]
                if matching:
                    avg_s = np.mean([p[1] for p in matching])
                    print(f"      t={t_target:2d}: survival={avg_s:.2f} ({avg_s*100:.0f}%)")

    return curves_data, cleaned_img, text_mask


def _clean_curve_points(points):
    """Clean curve points by binning and averaging."""
    if not points:
        return []

    # Group by integer time values
    time_bins = {}
    for t, s in points:
        t_bin = round(t * 2) / 2  # Round to nearest 0.5
        if t_bin not in time_bins:
            time_bins[t_bin] = []
        time_bins[t_bin].append(s)

    # Average each bin
    clean_points = []
    for t_bin in sorted(time_bins.keys()):
        s_values = time_bins[t_bin]
        s_avg = np.median(s_values)
        clean_points.append((t_bin, s_avg))

    # Enforce monotonic decreasing (survival can't increase)
    monotonic_points = []
    max_s = 1.0
    for t, s in clean_points:
        s = min(s, max_s)
        max_s = s
        monotonic_points.append((t, s))

    return monotonic_points


def create_visualization(img, curves_data, plot_bounds, output_path):
    """Create comparison overlay visualization."""
    vis = img.copy()

    # Colors for visualization (BGR)
    colors_bgr = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    for idx, curve_data in enumerate(curves_data):
        color = colors_bgr[idx % len(colors_bgr)]
        points = curve_data['raw_points']

        # Draw points
        plot_x, plot_y, plot_w, plot_h = plot_bounds
        for t, s in points:
            x = int(plot_x + (t / max(p[0] for p in points if p[0] > 0) if points else 1) * plot_w)
            y = int(plot_y + (1.0 - s) * plot_h)
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 1, color, -1)

    cv2.imwrite(str(output_path), vis)


def export_csv(curves_data, output_dir):
    """Export curves to CSV files."""
    for curve_data in curves_data:
        name = curve_data['name']
        points = curve_data['clean_points']

        csv_path = output_dir / f"{name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Survival'])
            for t, s in points:
                writer.writerow([f'{t:.2f}', f'{s:.4f}'])

        print(f"  Saved: {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Skeleton-based KM curve extraction')
    parser.add_argument('image_path', help='Path to the KM plot image')
    parser.add_argument('-o', '--output-dir', help='Output directory')
    parser.add_argument('--time-max', type=float, default=24.0, help='Maximum time value (default: 24)')
    parser.add_argument('--plot-bounds', help='Plot bounds as x,y,w,h (auto-detected if not specified)')
    parser.add_argument('--max-colors', type=int, default=2, help='Maximum number of curves to detect')
    args = parser.parse_args()

    # Load image
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Failed to load image: {img_path}")
        sys.exit(1)

    print(f"Image: {img_path.name}")
    print(f"Shape: {img.shape}")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('results') / img_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect or parse plot bounds
    if args.plot_bounds:
        plot_x, plot_y, plot_w, plot_h = map(int, args.plot_bounds.split(','))
    else:
        # Auto-detect using calibrator
        print("\n[Step 1] Auto-detecting plot bounds...")
        calibrator = AxisCalibrator(img)
        try:
            # Try to calibrate and get plot rectangle
            result = calibrator.calibrate()
            if result and result.plot_rectangle:
                plot_x, plot_y, plot_w, plot_h = result.plot_rectangle
                print(f"  Detected bounds: x={plot_x}, y={plot_y}, w={plot_w}, h={plot_h}")
            else:
                raise ValueError("No plot rectangle detected")
        except Exception as e:
            # Use reasonable defaults based on typical KM plot structure
            h, w = img.shape[:2]
            plot_x = int(w * 0.20)
            plot_y = int(h * 0.03)
            plot_w = int(w * 0.76)
            plot_h = int(h * 0.62)
            print(f"  Using default bounds: x={plot_x}, y={plot_y}, w={plot_w}, h={plot_h}")

    plot_bounds = (plot_x, plot_y, plot_w, plot_h)

    # Detect curve colors
    print("\n[Step 2] Detecting curve colors...")
    colors = detect_curve_colors(img, max_colors=args.max_colors)
    for c in colors:
        print(f"  {c['name']}: hue={c['hue']:.1f}")

    if not colors:
        print("Error: No curves detected")
        sys.exit(1)

    # Extract curves using skeleton method
    print("\n[Step 3] Extracting curves with skeleton method...")
    curves_data, cleaned_img, text_mask = extract_curves_skeleton(
        img, colors, plot_bounds, args.time_max
    )

    # Save intermediate outputs
    print("\n[Step 4] Saving outputs...")
    cv2.imwrite(str(output_dir / 'text_mask.png'), text_mask)
    cv2.imwrite(str(output_dir / 'cleaned_image.png'), cleaned_img)

    # Create comparison overlay
    vis = img.copy()
    colors_bgr = [(0, 255, 0), (255, 0, 0)]

    for idx, curve_data in enumerate(curves_data):
        color = colors_bgr[idx % len(colors_bgr)]
        points = curve_data['raw_points']

        # Convert data coordinates back to pixels for visualization
        for t, s in points:
            x = int(plot_x + (t / args.time_max) * plot_w)
            y = int(plot_y + (1.0 - s) * plot_h)
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 1, color, -1)

    cv2.imwrite(str(output_dir / 'comparison_overlay.png'), vis)

    # Export CSV files
    print("\n[Step 5] Exporting CSV files...")
    export_csv(curves_data, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)

    for curve_data in curves_data:
        name = curve_data['name']
        points = curve_data['clean_points']
        print(f"\n{name}:")
        print(f"  Points: {len(points)}")
        if points:
            print(f"  Time range: {points[0][0]:.1f} - {points[-1][0]:.1f}")
            print(f"  Survival range: {points[-1][1]:.2f} - {points[0][1]:.2f}")

    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
