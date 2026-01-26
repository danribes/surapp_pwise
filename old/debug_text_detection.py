#!/usr/bin/env python3
"""Debug text detection in cyan curve extraction."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/dan/surapp_pwise')

# Import the text detection function
from lib.color_detector import _detect_text_regions

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')
print(f"Image shape: {img.shape}")

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630

# Test the text detection
text_mask = _detect_text_regions(img, (plot_x, plot_y, plot_w, plot_h))

print(f"\nText mask shape: {text_mask.shape}")
print(f"Text mask pixels detected: {np.sum(text_mask > 0)}")

# Save the text mask visualization
vis = img.copy()
# Expand text_mask to full image coordinates for overlay
full_text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
full_text_mask[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w] = text_mask

# Overlay text regions in red
vis[full_text_mask > 0] = [0, 0, 255]

cv2.imwrite('/home/dan/surapp_pwise/results/debug_text_mask.png', vis)
print("Saved: results/debug_text_mask.png")

# Also try OCR directly on the plot area to see what text is detected
try:
    import pytesseract

    plot_roi = img[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w]
    rgb = cv2.cvtColor(plot_roi, cv2.COLOR_BGR2RGB)

    # Get text data with bounding boxes
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, timeout=10)

    print("\nOCR detected text:")
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

        if conf > 30 and len(text) > 0:
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            print(f"  '{text}' at ({x}, {y}) size {w}x{h}, conf={conf}")

except Exception as e:
    print(f"OCR error: {e}")

# Now let's see if we can distinguish curve pixels from text pixels
# by looking at the pattern of cyan pixels
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cyan_lower = np.array([87, 30, 60], dtype=np.uint8)
cyan_upper = np.array([107, 255, 255], dtype=np.uint8)
cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

# Crop to plot area
cyan_plot = cyan_mask[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w]

# For each column, count how many cyan pixels and their y distribution
print("\nAnalyzing cyan pixel patterns column by column (sample):")
print("Looking for text-like patterns (multiple y clusters in same column):")

text_like_columns = []
for x in range(0, plot_w, 10):  # Sample every 10th column
    col = cyan_plot[:, x]
    y_positions = np.where(col > 0)[0]

    if len(y_positions) > 5:
        # Check if there are multiple clusters (indicating text + curve)
        y_sorted = np.sort(y_positions)
        gaps = np.diff(y_sorted)
        large_gaps = np.sum(gaps > 30)  # Gaps > 30 pixels

        if large_gaps > 0:
            text_like_columns.append(x)
            y_min, y_max = y_sorted[0], y_sorted[-1]
            print(f"  x={x}: {len(y_positions)} pixels, y={y_min}-{y_max}, {large_gaps} large gaps")

print(f"\nColumns with potential text contamination: {len(text_like_columns)}")
print(f"Columns: {text_like_columns[:20]}...")  # Show first 20
