#!/usr/bin/env python3
"""Full OCR debug without any filtering."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/dan/surapp_pwise')

import pytesseract

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630

# Get plot ROI - WITHOUT resize
plot_roi = img[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w]
rgb = cv2.cvtColor(plot_roi, cv2.COLOR_BGR2RGB)

print(f"Plot ROI size: {rgb.shape[1]}x{rgb.shape[0]}")

# Get OCR data
data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, timeout=10)

# Create visualization
vis = img.copy()

print("\nAll OCR detections (conf > 20):")
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

    if conf > 20 and len(text) > 0:
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        # Convert to full image coordinates
        full_x = plot_x + x
        full_y = plot_y + y

        print(f"  '{text}' conf={conf} at ({full_x}, {full_y})")

        # Draw box - color based on whether text has %
        if '%' in text:
            color = (0, 255, 0)  # Green for percentages
        else:
            color = (200, 200, 200)  # Gray for other text

        cv2.rectangle(vis, (full_x, full_y), (full_x + w, full_y + h), color, 2)

cv2.imwrite('/home/dan/surapp_pwise/results/debug_ocr_full.png', vis)
print("\nSaved: results/debug_ocr_full.png")

# Now let's look at the actual pixel values at the expected location of "55%" text
# The "55% (45-66)" annotation appears to be around t=12, which is x ≈ 927
# But looking at the image, the text is to the right of the curve around t=10-11
print("\n\nLooking for text-like patterns in cyan pixel regions...")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cyan_lower = np.array([87, 30, 60], dtype=np.uint8)
cyan_upper = np.array([107, 255, 255], dtype=np.uint8)
cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

# Crop to plot area
cyan_plot = cyan_mask[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w]

# Find connected components in cyan mask
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cyan_plot)
print(f"\nConnected components in cyan mask: {num_labels - 1}")  # -1 for background

# Analyze each component
for label in range(1, num_labels):
    x, y, w, h, area = stats[label]
    cx, cy = centroids[label]

    # Filter for text-like components: small to medium area, not spanning full width
    if area > 20 and area < 5000 and w < plot_w * 0.2:
        # Check aspect ratio - text tends to be wider than tall
        aspect = w / h if h > 0 else 0

        # Check fill ratio - text is sparse, curve is dense
        fill_ratio = area / (w * h) if w * h > 0 else 0

        if aspect > 1.5 or (aspect > 0.5 and fill_ratio < 0.4):
            full_x = plot_x + x
            full_y = plot_y + y
            print(f"  Component at ({full_x}, {full_y}) size {w}x{h}: area={area}, aspect={aspect:.2f}, fill={fill_ratio:.2f}")

# Let's also extract just the region where "55%" should be and check its cyan content
print("\n\nChecking specific region around 55% annotation:")
# Based on visual inspection, "55%" is around x=952-1030 (full image), y=320-360
region_x1, region_y1 = 952, 320
region_x2, region_y2 = 1030, 360

# Check cyan in this region
region_cyan = cyan_mask[region_y1:region_y2, region_x1:region_x2]
print(f"Cyan pixels in 55% region: {np.sum(region_cyan > 0)}")

# Check what OCR detects in just this region
region_img = img[region_y1:region_y2, region_x1:region_x2]
region_rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
try:
    region_text = pytesseract.image_to_string(region_rgb, timeout=5).strip()
    print(f"OCR on 55% region: '{region_text}'")
except Exception as e:
    print(f"OCR failed: {e}")
