#!/usr/bin/env python3
"""Debug the t=15 spike issue."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/dan/surapp_pwise')

from lib.color_detector import _detect_text_regions

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630

# Extended bounds
margin = 5
px_ext = max(0, plot_x - margin)
py_ext = max(0, plot_y - margin)
pw_ext = min(img.shape[1] - px_ext, plot_w + 2 * margin)
ph_ext = min(img.shape[0] - py_ext, plot_h + 2 * margin)

print(f"Extended bounds: x={px_ext}, y={py_ext}, w={pw_ext}, h={ph_ext}")

# Get text mask
text_mask = _detect_text_regions(img, (px_ext, py_ext, pw_ext, ph_ext))
print(f"Text mask shape: {text_mask.shape}")
print(f"Text mask pixels: {np.sum(text_mask > 0)}")

# Create cyan mask
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cyan_lower = np.array([87, 30, 60], dtype=np.uint8)
cyan_upper = np.array([107, 255, 255], dtype=np.uint8)
cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

# Crop to extended region
cyan_roi = cyan_mask[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext]
print(f"Cyan ROI shape: {cyan_roi.shape}")
print(f"Cyan ROI pixels before text removal: {np.sum(cyan_roi > 0)}")

# Apply text mask
cyan_after_text = cv2.bitwise_and(cyan_roi, cv2.bitwise_not(text_mask))
print(f"Cyan ROI pixels after text removal: {np.sum(cyan_after_text > 0)}")

# Focus on t=15 region (x=1078 in full image, x=1078-px_ext in ROI)
time_max = 24.0
t = 15.0
x_pixel = int(plot_x + (t / time_max) * plot_w)
x_roi = x_pixel - px_ext
print(f"\nt=15 corresponds to x_pixel={x_pixel}, x_roi={x_roi}")

# What cyan pixels are at this x?
col_before = cyan_roi[:, x_roi]
col_after = cyan_after_text[:, x_roi]

y_before = np.where(col_before > 0)[0]
y_after = np.where(col_after > 0)[0]

print(f"\nCyan at x={x_roi} (t=15) BEFORE text removal: {len(y_before)} pixels")
if len(y_before) > 0:
    print(f"  y range: {y_before.min()} to {y_before.max()}")
    # Convert to survival
    for y in [y_before.min(), y_before.max()]:
        s = 1.0 - ((y + py_ext - plot_y) / plot_h)
        print(f"    y={y}: survival={s:.2f} ({s*100:.0f}%)")

print(f"\nCyan at x={x_roi} (t=15) AFTER text removal: {len(y_after)} pixels")
if len(y_after) > 0:
    print(f"  y range: {y_after.min()} to {y_after.max()}")
    for y in [y_after.min(), y_after.max()]:
        s = 1.0 - ((y + py_ext - plot_y) / plot_h)
        print(f"    y={y}: survival={s:.2f} ({s*100:.0f}%)")

# Check what text mask looks like at this x
text_col = text_mask[:, x_roi]
text_y = np.where(text_col > 0)[0]
print(f"\nText mask at x={x_roi}: {len(text_y)} pixels")
if len(text_y) > 0:
    print(f"  y range: {text_y.min()} to {text_y.max()}")

# Let's also check nearby columns
print("\nCyan pixels per column around t=15:")
for t_check in [14.5, 14.75, 15.0, 15.25, 15.5]:
    x_check = int(plot_x + (t_check / time_max) * plot_w) - px_ext
    if 0 <= x_check < pw_ext:
        col = cyan_after_text[:, x_check]
        y_pos = np.where(col > 0)[0]
        if len(y_pos) > 0:
            y_med = int(np.median(y_pos))
            s = 1.0 - ((y_med + py_ext - plot_y) / plot_h)
            print(f"  t={t_check:.2f} (x={x_check}): median_y={y_med} -> survival={s:.2f} ({s*100:.0f}%)")
        else:
            print(f"  t={t_check:.2f} (x={x_check}): NO CYAN")
