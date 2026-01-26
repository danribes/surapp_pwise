#!/usr/bin/env python3
"""
Test extraction using text removal and skeletonization.
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, '/home/dan/surapp_pwise')

from skimage.morphology import skeletonize
from lib.color_detector import detect_curve_colors
from lib.text_remover import remove_all_text

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')
print(f"Image shape: {img.shape}")

# Output directory
output_dir = '/home/dan/surapp_pwise/results/skeleton_test'
os.makedirs(output_dir, exist_ok=True)

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630
time_max = 24.0

# Step 1: Remove ALL text from the image
print("\n[Step 1] Removing all text from image...")
cleaned_img, text_mask = remove_all_text(img)
cv2.imwrite(f'{output_dir}/01_text_mask.png', text_mask)
cv2.imwrite(f'{output_dir}/02_cleaned_image.png', cleaned_img)
print(f"  Saved: 01_text_mask.png, 02_cleaned_image.png")

# Step 2: Detect curve colors
print("\n[Step 2] Detecting curve colors...")
colors = detect_curve_colors(cleaned_img, max_colors=2)
for c in colors:
    print(f"  {c['name']}: hue={c['hue']:.1f}")

# Step 3: Create color masks and skeletonize
print("\n[Step 3] Creating color masks and skeletonizing...")

hsv = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2HSV)
all_curves = []

for idx, color_info in enumerate(colors):
    name = color_info['name']
    print(f"\n  Processing {name}...")

    # Create color mask
    mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])

    # Crop to plot area with margin
    margin = 5
    px = max(0, plot_x - margin)
    py = max(0, plot_y - margin)
    pw = min(img.shape[1] - px, plot_w + 2*margin)
    ph = min(img.shape[0] - py, plot_h + 2*margin)

    mask_roi = mask[py:py+ph, px:px+pw]
    cv2.imwrite(f'{output_dir}/03_{name}_mask_raw.png', mask_roi)

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

    cv2.imwrite(f'{output_dir}/04_{name}_mask_filtered.png', mask_filtered)

    # Skeletonize
    print(f"    Skeletonizing...")
    skeleton = skeletonize(mask_filtered > 0).astype(np.uint8) * 255
    cv2.imwrite(f'{output_dir}/05_{name}_skeleton.png', skeleton)

    # Extract points from skeleton
    ys, xs = np.where(skeleton > 0)
    print(f"    Found {len(xs)} skeleton points")

    # Convert to coordinates
    points = []
    for x, y in zip(xs, ys):
        full_x = px + x
        full_y = py + y
        t = ((full_x - plot_x) / plot_w) * time_max
        s = 1.0 - ((full_y - plot_y) / plot_h)
        s = max(0.0, min(1.0, s))
        if 0 <= t <= time_max:
            points.append((t, s, full_x, full_y))

    # Sort by time
    points.sort(key=lambda p: p[0])
    all_curves.append((name, points))

    # Print sample values
    print(f"    Sample values:")
    for t_target in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
        matching = [p for p in points if abs(p[0] - t_target) < 0.5]
        if matching:
            avg_s = np.mean([p[1] for p in matching])
            print(f"      t={t_target:2d}: survival={avg_s:.2f} ({avg_s*100:.0f}%)")

# Step 4: Create visualization
print("\n[Step 4] Creating visualization...")
vis = img.copy()
colors_bgr = [(0, 255, 0), (255, 0, 0)]  # Green for cyan, Blue for gray

for idx, (name, points) in enumerate(all_curves):
    color = colors_bgr[idx % len(colors_bgr)]
    for t, s, x, y in points:
        cv2.circle(vis, (x, y), 1, color, -1)

cv2.imwrite(f'{output_dir}/06_skeleton_overlay.png', vis)
print(f"  Saved: 06_skeleton_overlay.png")

# Step 5: Create comparison with original
print("\n[Step 5] Creating comparison overlay...")
comparison = img.copy()

for idx, (name, points) in enumerate(all_curves):
    color = colors_bgr[idx % len(colors_bgr)]
    # Draw thicker line for visibility
    sorted_points = [(x, y) for t, s, x, y in points]
    for i in range(1, len(sorted_points)):
        x1, y1 = sorted_points[i-1]
        x2, y2 = sorted_points[i]
        # Only connect close points
        if abs(x2 - x1) < 30 and abs(y2 - y1) < 50:
            cv2.line(comparison, (x1, y1), (x2, y2), color, 2)

cv2.imwrite(f'{output_dir}/07_comparison.png', comparison)
print(f"  Saved: 07_comparison.png")

print("\n=== DONE ===")
print(f"Output files in: {output_dir}")
