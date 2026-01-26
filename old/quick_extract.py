#!/usr/bin/env python3
"""Quick extraction without validation (faster)."""

import cv2
import numpy as np
import sys
import csv
sys.path.insert(0, '/home/dan/surapp_pwise')

from lib.color_detector import (
    detect_curve_colors,
    extract_curves_with_overlap_handling,
)

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')
print(f"Image shape: {img.shape}")

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630
time_max = 24.0

# Detect colors
print("\nDetecting colors...")
colors = detect_curve_colors(img, max_colors=2)
for c in colors:
    print(f"  {c['name']}: hue={c['hue']:.1f}")

# Extract curves
print("\nExtracting curves...")
curves_data = extract_curves_with_overlap_handling(img, colors, (plot_x, plot_y, plot_w, plot_h))

# Convert to coordinates and save
import os
output_dir = '/home/dan/surapp_pwise/results/final_fixed'
os.makedirs(output_dir, exist_ok=True)

all_curves = []
for i, (pixel_points, color_info) in enumerate(curves_data):
    name = f"{color_info['name']}_{i+1}"
    print(f"\n{name}: {len(pixel_points)} points")

    if not pixel_points:
        continue

    # Convert to time/survival coordinates
    coord_points = []
    for px, py in pixel_points:
        t = ((px - plot_x) / plot_w) * time_max
        s = 1.0 - ((py - plot_y) / plot_h)
        s = max(0.0, min(1.0, s))
        coord_points.append((t, s))

    # Sort by time
    coord_points.sort(key=lambda p: p[0])

    # Save individual curve CSV
    csv_path = f"{output_dir}/curve_{name}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'survival'])
        for t, s in coord_points:
            writer.writerow([f"{t:.3f}", f"{s:.4f}"])
    print(f"  Saved: {csv_path}")

    # Store for all_curves
    all_curves.append((name, coord_points))

    # Show values at key time points
    print("  Values at key time points:")
    for t_target in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
        closest = min(coord_points, key=lambda p: abs(p[0] - t_target))
        if abs(closest[0] - t_target) < 1.0:
            print(f"    t={t_target:2d}: survival={closest[1]:.2f} ({closest[1]*100:.0f}%)")

# Save all curves combined
csv_path = f"{output_dir}/all_curves.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['curve', 'time', 'survival'])
    for name, points in all_curves:
        for t, s in points:
            writer.writerow([name, f"{t:.3f}", f"{s:.4f}"])
print(f"\nSaved combined: {csv_path}")

# Create comparison overlay
print("\nCreating comparison overlay...")
vis = img.copy()
colors_bgr = [(0, 255, 0), (255, 0, 0)]  # Green, Blue

for i, (pixel_points, color_info) in enumerate(curves_data):
    color = colors_bgr[i % len(colors_bgr)]
    for j, (x, y) in enumerate(pixel_points):
        cv2.circle(vis, (x, y), 1, color, -1)

cv2.imwrite(f"{output_dir}/comparison_overlay.png", vis)
print(f"Saved: {output_dir}/comparison_overlay.png")

print("\n=== DONE ===")
