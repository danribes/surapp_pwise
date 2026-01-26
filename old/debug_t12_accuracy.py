#!/usr/bin/env python3
"""Debug the t=12 accuracy - is 47% or 55% correct?"""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/dan/surapp_pwise')

# Load image
img = cv2.imread('/home/dan/surapp_pwise/input/Kaplan-Meier NSQ OS.webp')

# Plot bounds
plot_x, plot_y, plot_w, plot_h = 322, 28, 1211, 630

# The "55% (45-66)" annotation is visually positioned near the cyan curve at/before t=12
# Let me check the EXACT pixel position of the curve at the t=12 mark

time_max = 24.0

# The vertical dashed line at t=12 is at:
x_t12 = int(plot_x + (12.0 / time_max) * plot_w)
print(f"t=12 vertical line is at x={x_t12}")

# Create cyan mask
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cyan_lower = np.array([87, 30, 60], dtype=np.uint8)
cyan_upper = np.array([107, 255, 255], dtype=np.uint8)
cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

# Look at cyan pixels around t=12 (without any filtering)
print("\nCyan pixels at t=12 (x={}) - RAW mask:".format(x_t12))
col = cyan_mask[:, x_t12]
y_positions = np.where(col > 0)[0]
if len(y_positions) > 0:
    y_min, y_max = y_positions.min(), y_positions.max()
    print(f"  y range: {y_min} to {y_max}")
    for y in [y_min, y_max]:
        s = 1.0 - ((y - plot_y) / plot_h)
        print(f"    y={y}: survival={s:.2f} ({s*100:.0f}%)")

# The "55%" annotation text is at (952, 327) in full image coords
# Let me check what the actual cyan curve position is just BEFORE this text
print("\n\nLooking for actual curve position near the 55% annotation...")
annotation_x = 952  # Full image x of the "55%" text
annotation_y = 327  # Full image y of the "55%" text

# The annotation might mark the curve's value at a specific time
# Let me find where the cyan curve is JUST BEFORE the annotation text
# The curve should be to the LEFT of the text annotation

for x in range(annotation_x - 50, annotation_x, 5):
    col = cyan_mask[:, x]
    y_positions = np.where(col > 0)[0]
    # Filter to reasonable curve region (not legend, not axis)
    y_positions = y_positions[(y_positions > plot_y + 50) & (y_positions < plot_y + plot_h - 50)]
    if len(y_positions) > 0:
        y_med = int(np.median(y_positions))
        s = 1.0 - ((y_med - plot_y) / plot_h)
        t = ((x - plot_x) / plot_w) * time_max
        print(f"  x={x} (t={t:.1f}): y={y_med} -> survival={s:.2f} ({s*100:.0f}%)")

# Let's also check where 55% survival would be in pixel coordinates
survival_55 = 0.55
y_55 = int(plot_y + (1.0 - survival_55) * plot_h)
print(f"\n55% survival would be at y={y_55}")

# Check if there are cyan pixels at y=y_55
row = cyan_mask[y_55, :]
x_positions = np.where(row > 0)[0]
# Filter to plot region
x_positions = x_positions[(x_positions > plot_x) & (x_positions < plot_x + plot_w)]
if len(x_positions) > 0:
    print(f"Cyan pixels at y={y_55} (55% survival): x={x_positions.min()}-{x_positions.max()}")
    t_first = ((x_positions.min() - plot_x) / plot_w) * time_max
    t_last = ((x_positions.max() - plot_x) / plot_w) * time_max
    print(f"  This corresponds to t={t_first:.1f} to t={t_last:.1f}")
else:
    print(f"No cyan pixels at y={y_55} (55% survival)")

# Create a visualization
vis = img.copy()
# Draw vertical line at t=12
cv2.line(vis, (x_t12, plot_y), (x_t12, plot_y + plot_h), (255, 0, 255), 2)
# Draw horizontal line at 55% survival
cv2.line(vis, (plot_x, y_55), (plot_x + plot_w, y_55), (255, 255, 0), 2)
# Mark the annotation position
cv2.circle(vis, (annotation_x, annotation_y), 10, (0, 0, 255), 2)
# Mark where we think the curve actually is at t=12
col = cyan_mask[:, x_t12]
y_positions = np.where(col > 0)[0]
y_positions = y_positions[(y_positions > plot_y + 100) & (y_positions < plot_y + plot_h - 50)]
if len(y_positions) > 0:
    y_curve = int(np.median(y_positions))
    cv2.circle(vis, (x_t12, y_curve), 10, (0, 255, 0), 3)
    print(f"\nActual cyan curve at t=12 appears to be at y={y_curve}")
    s = 1.0 - ((y_curve - plot_y) / plot_h)
    print(f"  This is survival = {s:.2f} ({s*100:.0f}%)")

cv2.imwrite('/home/dan/surapp_pwise/results/debug_t12_accuracy.png', vis)
print("\nSaved: results/debug_t12_accuracy.png")
