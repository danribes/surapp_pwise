#!/usr/bin/env python3
"""Debug script to visualize cyan curve detection."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/dan/surapp_pwise')

from lib.color_detector import detect_curve_colors, extract_curves_with_overlap_handling

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')
print(f"Image shape: {img.shape}")

# Get calibration from the original extraction results
# From earlier analysis: plot_x=322, plot_w=1211 (corrected)
# Let's re-determine the plot bounds
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Detect colors
colors = detect_curve_colors(img, max_colors=4)
print("\nDetected colors:")
for c in colors:
    print(f"  {c['name']}: hue={c['hue']:.1f}, HSV range={c['hsv_lower']} to {c['hsv_upper']}")
    print(f"           RGB={c['rgb']}, pixel_count={c['pixel_count']}")

# Find cyan color
cyan_color = None
for c in colors:
    if c['name'] == 'cyan':
        cyan_color = c
        break

if not cyan_color:
    print("\nNo cyan color detected!")
    sys.exit(1)

# Create cyan mask
cyan_mask = cv2.inRange(hsv, cyan_color['hsv_lower'], cyan_color['hsv_upper'])
print(f"\nCyan mask total pixels: {np.sum(cyan_mask > 0)}")

# Find the bounding box of all cyan pixels
cyan_ys, cyan_xs = np.where(cyan_mask > 0)
if len(cyan_xs) > 0:
    print(f"\nCyan pixel bounds:")
    print(f"  X: {cyan_xs.min()} to {cyan_xs.max()}")
    print(f"  Y: {cyan_ys.min()} to {cyan_ys.max()}")

# Create visualization
vis = img.copy()

# Overlay cyan mask in bright green
vis[cyan_mask > 0] = [0, 255, 0]

# Save the visualization
cv2.imwrite('/home/dan/surapp_pwise/results/debug_cyan_full_mask.png', vis)
print("\nSaved: results/debug_cyan_full_mask.png")

# Now let's look at what's being extracted as the "curve"
# Use a typical plot bounds estimate
# Looking at image dimensions, estimate plot area
h, w = img.shape[:2]

# Find where the axis lines are
# The plot typically has white axes - find them
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find rows/cols that are mostly white (axis lines)
# For now, use approximate plot bounds from earlier: x=322, w=1211, y=28, h=630
# But let's verify by looking at the image structure

# Let's segment the cyan detections by location
# Group detections into regions

# Create a mask showing only pixels in the expected plot area
# vs pixels outside (legend, axis labels, etc.)
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630

# Mask for in-plot vs out-of-plot
in_plot = np.zeros_like(cyan_mask)
in_plot[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w] = cyan_mask[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w]

out_of_plot = cyan_mask.copy()
out_of_plot[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w] = 0

print(f"\nCyan pixels in plot area: {np.sum(in_plot > 0)}")
print(f"Cyan pixels outside plot area: {np.sum(out_of_plot > 0)}")

# Create another visualization showing in-plot (green) vs out-of-plot (red)
vis2 = img.copy()
vis2[in_plot > 0] = [0, 255, 0]  # Green for in-plot
vis2[out_of_plot > 0] = [0, 0, 255]  # Red for outside plot

# Draw plot bounds rectangle
cv2.rectangle(vis2, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (255, 0, 255), 2)

cv2.imwrite('/home/dan/surapp_pwise/results/debug_cyan_in_out_plot.png', vis2)
print("Saved: results/debug_cyan_in_out_plot.png")

# Now let's see where the in-plot cyan pixels are located
# For each x in the plot area, find cyan y positions
print("\nCyan detections at specific x positions (in plot area):")
time_max = 24.0
for t_target in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
    # Convert time to x pixel
    x = int(plot_x + (t_target / time_max) * plot_w)
    if x < 0 or x >= w:
        continue

    col = in_plot[:, x]
    y_positions = np.where(col > 0)[0]

    if len(y_positions) > 0:
        y_min, y_max = y_positions.min(), y_positions.max()
        # Convert y to survival
        s_at_min_y = 1.0 - (y_min - plot_y) / plot_h
        s_at_max_y = 1.0 - (y_max - plot_y) / plot_h
        print(f"  t={t_target:5.1f} (x={x:4d}): y={y_min:3d}-{y_max:3d} (survival={s_at_min_y:.2f}-{s_at_max_y:.2f})")
    else:
        print(f"  t={t_target:5.1f} (x={x:4d}): NO CYAN DETECTED")

# Also check what's happening around t=4.3 where the 55% was reportedly detected
print("\nDetailed look at t=4-5 region:")
for t_target in np.arange(4.0, 5.5, 0.1):
    x = int(plot_x + (t_target / time_max) * plot_w)
    col = in_plot[:, x]
    y_positions = np.where(col > 0)[0]

    if len(y_positions) > 0:
        y_med = int(np.median(y_positions))
        s_med = 1.0 - (y_med - plot_y) / plot_h
        print(f"  t={t_target:.1f} (x={x:4d}): median_y={y_med:3d} -> survival={s_med:.2f}")

# Check if there's cyan in the legend area (right side, top)
legend_region = cyan_mask[:200, w-300:]
print(f"\nCyan pixels in legend area (top-right): {np.sum(legend_region > 0)}")

# Look for text annotation regions
# Text annotations like "55%" are often near the curve
# Let's find dense clusters of cyan pixels that might be text

# Sample some specific y values to understand what we're seeing
print("\nCyan row coverage (how many columns have cyan at each row):")
row_counts = np.sum(in_plot > 0, axis=1)
for y in range(plot_y, plot_y + plot_h, 50):
    count = row_counts[y]
    s = 1.0 - (y - plot_y) / plot_h
    print(f"  y={y:3d} (survival={s:.2f}): {count} columns with cyan")
