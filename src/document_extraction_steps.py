#!/usr/bin/env python3
"""
Document the extraction steps with visualization images.

Creates step-by-step images showing:
1. Original image
2. Extracted rectangle with axes and labels
3. Plot area only (curves region)
4. Cleaned curves only (no text, grid lines, artifacts)
"""

import cv2
import numpy as np
import shutil
import json
from pathlib import Path


def create_documentation_images(
    source_image: str,
    results_dir: str,
    calibration: dict
):
    """Create step-by-step documentation images."""

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load original image
    img = cv2.imread(source_image)
    if img is None:
        raise ValueError(f"Could not load image: {source_image}")

    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")

    # Extract calibration values
    x_0 = calibration['x_0_pixel']
    x_max = calibration['x_max_pixel']
    y_0 = calibration['y_0_pixel']      # Bottom (survival=0)
    y_100 = calibration['y_100_pixel']  # Top (survival=100%)

    print(f"Calibration: x=[{x_0}, {x_max}], y=[{y_100}, {y_0}]")

    # Step 1: Copy original image
    step1_path = results_path / "step1_original.png"
    shutil.copy(source_image, step1_path)
    print(f"Step 1: Original image saved to {step1_path}")

    # Step 2: Extract rectangle with axes and labels (add margin for labels)
    margin_left = 70   # Space for Y-axis label
    margin_bottom = 50  # Space for X-axis label
    margin_top = 20     # Space for title
    margin_right = 20   # Small right margin

    x1 = max(0, x_0 - margin_left)
    y1 = max(0, y_100 - margin_top)
    x2 = min(w, x_max + margin_right)
    y2 = min(h, y_0 + margin_bottom)

    step2_img = img[y1:y2, x1:x2].copy()
    step2_path = results_path / "step2_with_axes_labels.png"
    cv2.imwrite(str(step2_path), step2_img)
    print(f"Step 2: Rectangle with axes/labels saved to {step2_path}")

    # Step 3: Plot area only (just the curves region)
    step3_img = img[y_100:y_0, x_0:x_max].copy()
    step3_path = results_path / "step3_plot_area.png"
    cv2.imwrite(str(step3_path), step3_img)
    print(f"Step 3: Plot area saved to {step3_path}")

    # Step 4: Clean curves only - remove text, grid lines, artifacts
    step4_img = clean_plot_area(step3_img.copy())
    step4_path = results_path / "step4_curves_only.png"
    cv2.imwrite(str(step4_path), step4_img)
    print(f"Step 4: Cleaned curves saved to {step4_path}")

    # Create a summary visualization
    create_summary_image(img, calibration, results_path)

    return {
        'step1': str(step1_path),
        'step2': str(step2_path),
        'step3': str(step3_path),
        'step4': str(step4_path)
    }


def clean_plot_area(img: np.ndarray) -> np.ndarray:
    """
    Clean the plot area to show only the curves.

    Removes:
    - Background (make white)
    - Grid lines (thin gray lines)
    - Text annotations
    - Legend boxes
    - Axis tick marks
    """
    h, w = img.shape[:2]

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create output image (white background)
    output = np.ones_like(img) * 255

    # Define color ranges for the curves
    # Green curve: H=35-85, S>50, V>50
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Red curve: H=0-10 or H=170-180, S>50, V>50
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # Black pixels (dark areas)
    black_mask = (gray < 80).astype(np.uint8) * 255

    # === Remove text and annotation boxes ===

    # Use edge detection to find text-dense regions
    edges = cv2.Canny(gray, 50, 150)

    # Create a density map of edges using a sliding window
    kernel_size = 20
    edge_density = cv2.blur(edges.astype(np.float32), (kernel_size, kernel_size))

    # Text regions have high edge density
    text_region_mask = (edge_density > 30).astype(np.uint8) * 255

    # Find contours of text regions
    contours, _ = cv2.findContours(text_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_box_mask = np.zeros((h, w), dtype=np.uint8)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch

        # Text boxes are rectangular with text-like dimensions
        if area > 300 and area < 30000:
            aspect = cw / max(ch, 1)
            # Text boxes tend to be wider than tall
            if aspect > 1.2 and aspect < 15:
                # This is likely a text annotation box
                margin = 5
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + cw + margin)
                y2 = min(h, y + ch + margin)
                cv2.rectangle(text_box_mask, (x1, y1), (x2, y2), 255, -1)

    # Also mask the top region where P-value text typically appears
    # Look for any significant dark content in the top 10% of the image
    top_height = int(h * 0.12)
    top_black = black_mask[:top_height, :]
    if np.sum(top_black) > 1000:  # If there's significant black content at top
        # Find the extent of the text
        cols_with_content = np.any(top_black > 0, axis=0)
        if np.any(cols_with_content):
            left = np.argmax(cols_with_content)
            right = w - np.argmax(cols_with_content[::-1])
            cv2.rectangle(text_box_mask, (max(0, left-10), 0),
                         (min(w, right+10), top_height+10), 255, -1)

    # 2. Detect text-like connected components in black mask
    # Remove horizontal and vertical lines first
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    horizontal_lines = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, vertical_kernel)

    # Lines to remove
    lines_mask = horizontal_lines | vertical_lines

    # Black without lines
    black_no_lines = black_mask.copy()
    black_no_lines[lines_mask > 0] = 0

    # 3. Analyze connected components to separate curves from text
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_no_lines, connectivity=8)

    curve_black_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cx, cy = centroids[i]

        # Skip very small components (noise)
        if area < 3:
            continue

        # Skip components inside text box regions
        if text_box_mask[int(cy), int(cx)] > 0:
            continue

        # Skip components at the top (likely the p-value text)
        if y < h * 0.15 and cw > 20:
            continue

        # Skip components that are compact and small (text characters)
        aspect = cw / max(ch, 1)
        compactness = area / (cw * ch) if cw * ch > 0 else 0

        # Curve segments: either elongated (horizontal/vertical steps) or larger connected regions
        is_curve_like = (
            (aspect > 3 or aspect < 0.33) or  # Elongated
            (area > 50 and compactness > 0.3) or  # Larger filled regions
            (ch > 20 and cw < 5) or  # Vertical drops
            (cw > 20 and ch < 5)  # Horizontal steps
        )

        # Also accept + shaped censoring marks (small crosses)
        is_censor_mark = (area < 30 and 0.5 < aspect < 2 and cw < 15 and ch < 15)

        if is_curve_like or is_censor_mark:
            curve_black_mask[labels == i] = 255

    # 4. Clean colored curves too - remove any in text box regions
    green_clean = green_mask.copy()
    red_clean = red_mask.copy()

    # Dilate text box mask more aggressively
    text_box_dilated = cv2.dilate(text_box_mask, np.ones((10, 10), np.uint8), iterations=2)
    green_clean[text_box_dilated > 0] = 0
    red_clean[text_box_dilated > 0] = 0
    curve_black_mask[text_box_dilated > 0] = 0

    # Remove isolated small green/red blobs (likely text or artifacts)
    for mask in [green_clean, red_clean]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            y = stats[i, cv2.CC_STAT_TOP]
            x = stats[i, cv2.CC_STAT_LEFT]

            # Remove small isolated blobs (likely text fragments)
            if area < 30:
                mask[labels == i] = 0
            # Remove blobs in the right portion that look like text (label area)
            elif x > w * 0.7 and area < 200 and cw / max(ch, 1) > 1.5:
                mask[labels == i] = 0

    # Apply colors to output
    output[green_clean > 0] = [0, 200, 0]  # Green
    output[red_clean > 0] = [0, 0, 200]  # Red (BGR)
    output[curve_black_mask > 0] = [0, 0, 0]  # Black

    return output


def create_summary_image(img: np.ndarray, calibration: dict, results_path: Path):
    """Create a summary image showing all extraction regions."""

    output = img.copy()
    h, w = img.shape[:2]

    x_0 = calibration['x_0_pixel']
    x_max = calibration['x_max_pixel']
    y_0 = calibration['y_0_pixel']
    y_100 = calibration['y_100_pixel']

    # Draw plot area rectangle (blue)
    cv2.rectangle(output, (x_0, y_100), (x_max, y_0), (255, 0, 0), 2)

    # Draw axes lines (green)
    cv2.line(output, (x_0, y_0), (x_max, y_0), (0, 255, 0), 2)  # X-axis
    cv2.line(output, (x_0, y_100), (x_0, y_0), (0, 255, 0), 2)  # Y-axis

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, "Plot Area", (x_0 + 10, y_100 + 25), font, 0.6, (255, 0, 0), 2)
    cv2.putText(output, f"({x_0}, {y_100})", (x_0 - 60, y_100 - 5), font, 0.4, (255, 0, 0), 1)
    cv2.putText(output, f"({x_max}, {y_0})", (x_max + 5, y_0 + 15), font, 0.4, (255, 0, 0), 1)

    summary_path = results_path / "extraction_regions.png"
    cv2.imwrite(str(summary_path), output)
    print(f"Summary: Extraction regions saved to {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Document extraction steps')
    parser.add_argument('--source', '-s', required=True, help='Source image path')
    parser.add_argument('--results', '-r', required=True, help='Results directory')
    parser.add_argument('--calibration', '-c', help='Calibration JSON file (optional)')
    args = parser.parse_args()

    # Load or detect calibration
    if args.calibration:
        with open(args.calibration, 'r') as f:
            calibration = json.load(f)
    else:
        # Try to load from results directory
        cal_path = Path(args.results) / 'calibration.json'
        if cal_path.exists():
            with open(cal_path, 'r') as f:
                calibration = json.load(f)
        else:
            raise ValueError("No calibration file found. Please provide --calibration")

    # Create documentation images
    paths = create_documentation_images(args.source, args.results, calibration)

    print("\nDocumentation images created:")
    for step, path in paths.items():
        print(f"  {step}: {path}")


if __name__ == '__main__':
    main()
