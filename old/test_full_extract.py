#!/usr/bin/env python3
"""Test full extraction with debug output."""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, '/home/dan/surapp_pwise')

from lib.color_detector import (
    detect_curve_colors,
    extract_curves_with_overlap_handling,
    _detect_text_regions
)

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')
print(f"Image shape: {img.shape}")

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630
time_max = 24.0

# Extended bounds (same as in extract_curves_with_overlap_handling)
margin = 5
px_ext = max(0, plot_x - margin)
py_ext = max(0, plot_y - margin)
pw_ext = min(img.shape[1] - px_ext, plot_w + 2 * margin)
ph_ext = min(img.shape[0] - py_ext, plot_h + 2 * margin)

# Test text detection
print("\nTesting text detection...")
text_mask = _detect_text_regions(img, (px_ext, py_ext, pw_ext, ph_ext))
print(f"Text mask pixels: {np.sum(text_mask > 0)}")

# Save text mask visualization
vis_text = img.copy()
full_text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
full_text_mask[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext] = text_mask
vis_text[full_text_mask > 0] = [0, 0, 255]  # Red for text
cv2.imwrite('/home/dan/surapp_pwise/results/final_fixed/debug_text_mask.png', vis_text)
print("Saved: debug_text_mask.png")

# Detect colors
print("\nDetecting colors...")
colors = detect_curve_colors(img, max_colors=2)
for c in colors:
    print(f"  {c['name']}: hue={c['hue']:.1f}")

# Extract curves
print("\nExtracting curves...")
curves_data = extract_curves_with_overlap_handling(img, colors, (plot_x, plot_y, plot_w, plot_h))

# Create debug detection image (like the problematic one)
vis = img.copy()
color_list = [(0, 255, 0), (255, 0, 0)]  # Green, Blue

for idx, (pixel_points, color_info) in enumerate(curves_data):
    name = f"{color_info['name']}_{idx+1}"
    color = color_list[idx % len(color_list)]

    print(f"\n{name}: {len(pixel_points)} points")

    # Draw curve name at starting position
    if pixel_points:
        start_x, start_y = pixel_points[0]
        cv2.putText(vis, name, (start_x + 5, start_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw all points
    for x, y in pixel_points:
        cv2.circle(vis, (x, y), 1, color, -1)

    # Connect consecutive points with lines to show connectivity
    sorted_points = sorted(pixel_points, key=lambda p: p[0])
    for i in range(1, len(sorted_points)):
        x1, y1 = sorted_points[i-1]
        x2, y2 = sorted_points[i]
        # Only draw line if points are close (part of same curve segment)
        if abs(x2 - x1) < 20 and abs(y2 - y1) < 50:
            cv2.line(vis, (x1, y1), (x2, y2), color, 1)

cv2.imwrite('/home/dan/surapp_pwise/results/final_fixed/debug_detection.png', vis)
print("\nSaved: debug_detection.png")

# Check for any points in the legend region (y < 150 in full image)
print("\n\nChecking for legend contamination...")
for idx, (pixel_points, color_info) in enumerate(curves_data):
    name = f"{color_info['name']}_{idx+1}"
    legend_points = [(x, y) for x, y in pixel_points if y < 150]
    if legend_points:
        print(f"  WARNING: {name} has {len(legend_points)} points in legend region!")
        for x, y in legend_points[:5]:
            t = ((x - plot_x) / plot_w) * time_max
            s = 1.0 - ((y - plot_y) / plot_h)
            print(f"    ({x}, {y}) -> t={t:.1f}, s={s:.2f}")
    else:
        print(f"  OK: {name} has no points in legend region")

print("\n=== DONE ===")
