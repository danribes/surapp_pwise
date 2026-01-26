#!/usr/bin/env python3
"""Test cyan curve extraction with the text mask fix."""

import cv2
import numpy as np
import sys
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

# Test text detection with the fix
print("\n1. Testing text detection with the fix...")
text_mask = _detect_text_regions(img, (plot_x, plot_y, plot_w, plot_h))
print(f"   Text mask pixels: {np.sum(text_mask > 0)}")

# Detect colors
print("\n2. Detecting curve colors...")
colors = detect_curve_colors(img, max_colors=2)
for c in colors:
    print(f"   {c['name']}: hue={c['hue']:.1f}")

# Extract curves with overlap handling
print("\n3. Extracting curves with overlap handling...")
curves_data = extract_curves_with_overlap_handling(img, colors, (plot_x, plot_y, plot_w, plot_h))

# Analyze the extracted curves
time_max = 24.0
for i, (pixel_points, color_info) in enumerate(curves_data):
    name = color_info['name']
    print(f"\n   {name} curve: {len(pixel_points)} points")

    if not pixel_points:
        continue

    # Convert to time/survival coordinates
    coord_points = []
    for px, py in pixel_points:
        t = ((px - plot_x) / plot_w) * time_max
        s = 1.0 - ((py - plot_y) / plot_h)
        s = max(0.0, min(1.0, s))
        coord_points.append((t, s))

    coord_points.sort(key=lambda p: p[0])

    # Show values at key time points
    print(f"   Values at key time points:")
    for t_target in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
        # Find closest point
        closest = min(coord_points, key=lambda p: abs(p[0] - t_target))
        if abs(closest[0] - t_target) < 1.0:
            print(f"      t={t_target:2d}: survival={closest[1]:.2f} ({closest[1]*100:.0f}%)")
        else:
            print(f"      t={t_target:2d}: NO DATA (closest at t={closest[0]:.1f})")

# Verify: the cyan curve should show ~55% at t=12
print("\n4. VERIFICATION:")
cyan_data = None
for pixel_points, color_info in curves_data:
    if color_info['name'] == 'cyan':
        cyan_data = pixel_points
        break

if cyan_data:
    # Find cyan value at t=12
    coord_points = []
    for px, py in cyan_data:
        t = ((px - plot_x) / plot_w) * time_max
        s = 1.0 - ((py - plot_y) / plot_h)
        coord_points.append((t, s))

    coord_points.sort(key=lambda p: p[0])

    # Get value around t=12
    t12_points = [p for p in coord_points if 11.5 <= p[0] <= 12.5]
    if t12_points:
        avg_survival = np.mean([p[1] for p in t12_points])
        print(f"   Cyan at t=12: {avg_survival:.2f} ({avg_survival*100:.0f}%)")
        if 0.50 <= avg_survival <= 0.60:
            print(f"   ✓ MATCHES expected 55% (within tolerance)")
        else:
            print(f"   ✗ EXPECTED ~55%, got {avg_survival*100:.0f}%")

    # Get value around t=18
    t18_points = [p for p in coord_points if 17.5 <= p[0] <= 18.5]
    if t18_points:
        avg_survival = np.mean([p[1] for p in t18_points])
        print(f"   Cyan at t=18: {avg_survival:.2f} ({avg_survival*100:.0f}%)")
        if 0.25 <= avg_survival <= 0.35:
            print(f"   ✓ MATCHES expected 30% (within tolerance)")
        else:
            print(f"   ✗ EXPECTED ~30%, got {avg_survival*100:.0f}%")
