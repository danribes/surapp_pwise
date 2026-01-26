#!/usr/bin/env python3
"""Debug OCR text detection and coordinate scaling."""

import cv2
import numpy as np
import re
import sys
sys.path.insert(0, '/home/dan/surapp_pwise')

import pytesseract

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630

# Get plot ROI
plot_roi = img[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w]
rgb = cv2.cvtColor(plot_roi, cv2.COLOR_BGR2RGB)

# Check if resize happens
h, w = rgb.shape[:2]
max_dim = 800
scale = 1.0
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    rgb_scaled = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    print(f"Image resized from {w}x{h} to {int(w*scale)}x{int(h*scale)}, scale={scale:.3f}")
else:
    rgb_scaled = rgb
    print(f"No resize needed: {w}x{h}")

# Get OCR data
data = pytesseract.image_to_data(rgb_scaled, output_type=pytesseract.Output.DICT, timeout=10)

# Create visualization
vis = img.copy()

# Draw all OCR boxes
annotation_pattern = re.compile(r'^\d+%?$|^\(\d+[-–]\d+\)$|^\d+%?\s*\(\d+[-–]\d+\)$')

print("\nOCR detections with boxes:")
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

    if conf > 40 and len(text) > 0:
        # Get scaled coordinates
        x_scaled = data['left'][i]
        y_scaled = data['top'][i]
        w_scaled = data['width'][i]
        h_scaled = data['height'][i]

        # Scale back to original ROI coordinates
        x = int(x_scaled / scale)
        y = int(y_scaled / scale)
        w = int(w_scaled / scale)
        h = int(h_scaled / scale)

        # Check if annotation
        is_annotation = (
            annotation_pattern.match(text) or
            '%' in text or
            (text.startswith('(') and text.endswith(')'))
        )

        # Convert to full image coordinates
        full_x = plot_x + x
        full_y = plot_y + y

        # Color: Green if annotation, Blue if not
        color = (0, 255, 0) if is_annotation else (255, 0, 0)

        print(f"  '{text}' conf={conf} -> ROI({x},{y},{w},{h}) -> Full({full_x},{full_y})")
        print(f"       is_annotation={is_annotation}")

        # Draw box on full image
        cv2.rectangle(vis, (full_x, full_y), (full_x + w, full_y + h), color, 2)
        cv2.putText(vis, text, (full_x, full_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite('/home/dan/surapp_pwise/results/debug_ocr_boxes.png', vis)
print("\nSaved: results/debug_ocr_boxes.png")

# Now focus specifically on the cyan text annotations
print("\n\nFocusing on cyan text annotations:")
print("Looking for '55%', '30%', and their ranges...")

cyan_annotations = []
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

    if conf > 40 and len(text) > 0:
        if text in ['55%', '30%', '55', '30'] or text.startswith('(4') or text.startswith('(2'):
            x = int(data['left'][i] / scale)
            y = int(data['top'][i] / scale)
            w = int(data['width'][i] / scale)
            h = int(data['height'][i] / scale)

            full_x = plot_x + x
            full_y = plot_y + y

            print(f"  '{text}' at full coords ({full_x}, {full_y}) size {w}x{h}")

            # Check if we need to mask this
            # Skip text at edges
            if y < 15 or y + h > plot_h - 15:
                print(f"    SKIPPED: at vertical edge (y={y}, h={h}, plot_h={plot_h})")
                continue
            if x < 15 or x + w > plot_w - 15:
                print(f"    SKIPPED: at horizontal edge (x={x}, w={w}, plot_w={plot_w})")
                continue

            print(f"    SHOULD BE MASKED")
            cyan_annotations.append((full_x, full_y, w, h, text))

print(f"\nTotal cyan annotations to mask: {len(cyan_annotations)}")
