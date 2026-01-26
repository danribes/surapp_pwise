#!/usr/bin/env python3
"""Debug the legend text leaking into curve detection."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/dan/surapp_pwise')

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

# Create cyan mask
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cyan_lower = np.array([87, 30, 60], dtype=np.uint8)
cyan_upper = np.array([107, 255, 255], dtype=np.uint8)
cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

# Crop to extended region
cyan_roi = cyan_mask[py_ext:py_ext+ph_ext, px_ext:px_ext+pw_ext]

# Find cyan pixels in the "top" region (y < 120 in ROI, which is high survival)
top_region_y_max = 120
cyan_top = cyan_roi[:top_region_y_max, :]

print(f"Cyan pixels in top region (y < {top_region_y_max}): {np.sum(cyan_top > 0)}")

# Find which columns have cyan in the top region
cols_with_top_cyan = np.any(cyan_top > 0, axis=0)
x_positions = np.where(cols_with_top_cyan)[0]
if len(x_positions) > 0:
    print(f"Columns with cyan in top region: x = {x_positions.min()} to {x_positions.max()}")
    print(f"  These correspond to t = {((x_positions.min() + px_ext - plot_x) / plot_w) * 24:.1f} to {((x_positions.max() + px_ext - plot_x) / plot_w) * 24:.1f}")

# Check if these are in the legend area
# The legend is typically in the upper-right portion of the plot
# Let's find the exact bounding box of cyan pixels in the top region
ys, xs = np.where(cyan_top > 0)
if len(xs) > 0:
    print(f"\nCyan pixels in top region bounding box:")
    print(f"  x: {xs.min()} to {xs.max()} (ROI coords)")
    print(f"  y: {ys.min()} to {ys.max()} (ROI coords)")

    # Convert to full image coords
    full_x_min = xs.min() + px_ext
    full_x_max = xs.max() + px_ext
    full_y_min = ys.min() + py_ext
    full_y_max = ys.max() + py_ext
    print(f"  Full image: x={full_x_min}-{full_x_max}, y={full_y_min}-{full_y_max}")

# Create visualization showing cyan pixels in top region
vis = img.copy()
# Mark top cyan pixels in bright green
full_top_mask = np.zeros(img.shape[:2], dtype=np.uint8)
full_top_mask[py_ext:py_ext+top_region_y_max, px_ext:px_ext+pw_ext] = cyan_top
vis[full_top_mask > 0] = [0, 255, 0]

# Draw rectangle showing plot area and top region
cv2.rectangle(vis, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (255, 0, 255), 2)
cv2.rectangle(vis, (px_ext, py_ext), (px_ext + pw_ext, py_ext + top_region_y_max), (0, 255, 255), 2)

cv2.imwrite('/home/dan/surapp_pwise/results/debug_legend_leak.png', vis)
print("\nSaved: results/debug_legend_leak.png")

# Now let's check what specific text is in this region
print("\n\nOCR on the top region to identify legend text...")
import pytesseract

# Get a larger region to capture full legend
legend_region = img[py_ext:py_ext+150, px_ext:px_ext+pw_ext]
rgb = cv2.cvtColor(legend_region, cv2.COLOR_BGR2RGB)

data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, timeout=10)

print("Text in legend region:")
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
    if conf > 30 and len(text) > 0:
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        print(f"  '{text}' at ({x}, {y}) size {w}x{h}, conf={conf}")
