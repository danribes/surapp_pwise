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
    # Use isolated curve images if available for better accuracy
    step4_img = clean_plot_area(step3_img.copy(), results_dir=results_path)
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


def clean_plot_area(img: np.ndarray, results_dir: Path = None) -> np.ndarray:
    """
    Clean the plot area to show only the curves.

    Extracts colored curves (green, red) using HSV color detection,
    and black curves using careful filtering to avoid text.

    Removes:
    - Background (make white)
    - Text annotations and boxes
    - Legend artifacts
    - Axis labels and tick marks
    """
    h, w = img.shape[:2]

    # Always extract directly from the plot area for consistent results
    return extract_curves_clean(img)


def combine_isolated_curves(
    green_path: str,
    red_path: str,
    gray_path: str,
    target_size: tuple
) -> np.ndarray:
    """Combine isolated curve images into a single clean output."""

    h, w = target_size

    # Create white background
    output = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Load and process green curve
    green_img = cv2.imread(green_path)
    if green_img is not None:
        green_img = crop_to_plot_area(green_img)
        if green_img.shape[:2] != (h, w):
            green_img = cv2.resize(green_img, (w, h))

        # Extract green pixels
        hsv = cv2.cvtColor(green_img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

        # Remove legend area (top-right corner)
        green_mask = remove_legend_region(green_mask)

        output[green_mask > 0] = [0, 180, 0]  # Green

    # Load and process red curve
    red_img = cv2.imread(red_path)
    if red_img is not None:
        red_img = crop_to_plot_area(red_img)
        if red_img.shape[:2] != (h, w):
            red_img = cv2.resize(red_img, (w, h))

        # Extract red pixels
        hsv = cv2.cvtColor(red_img, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = red_mask1 | red_mask2

        # Remove legend area
        red_mask = remove_legend_region(red_mask)

        output[red_mask > 0] = [0, 0, 180]  # Red (BGR)

    # Load and process black/gray curve
    if gray_path:
        gray_img = cv2.imread(gray_path)
        if gray_img is not None:
            gray_img = crop_to_plot_area(gray_img)
            if gray_img.shape[:2] != (h, w):
                gray_img = cv2.resize(gray_img, (w, h))

            # The gray image has all content - need to extract just the black curve
            black_mask = extract_black_curve_from_gray(gray_img)

            # Remove legend area
            black_mask = remove_legend_region(black_mask)

            output[black_mask > 0] = [0, 0, 0]  # Black

    return output


def crop_to_plot_area(img: np.ndarray) -> np.ndarray:
    """Crop image to remove axes and labels, keeping just the plot area."""
    h, w = img.shape[:2]

    # Find the plot area by looking for where content starts/ends
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # For isolated images, they usually have white background
    # Find the bounding box of non-white content
    non_white = gray < 250

    rows_with_content = np.any(non_white, axis=1)
    cols_with_content = np.any(non_white, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return img

    y1 = np.argmax(rows_with_content)
    y2 = h - np.argmax(rows_with_content[::-1])
    x1 = np.argmax(cols_with_content)
    x2 = w - np.argmax(cols_with_content[::-1])

    # Add small margin
    margin = 5
    y1 = max(0, y1 - margin)
    y2 = min(h, y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(w, x2 + margin)

    return img[y1:y2, x1:x2]


def remove_legend_region(mask: np.ndarray) -> np.ndarray:
    """Remove legend artifacts from top-right corner."""
    h, w = mask.shape

    # Legend is typically in top-right area
    legend_region = mask.copy()

    # Clear top-right quadrant of small isolated components
    top_right_y = int(h * 0.35)
    top_right_x = int(w * 0.65)

    # Find and remove small isolated components in the legend area
    legend_area = mask[:top_right_y, top_right_x:].copy()
    if np.sum(legend_area) > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(legend_area, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Remove small components (legend symbols)
            if area < 500:
                legend_area[labels == i] = 0
        mask[:top_right_y, top_right_x:] = legend_area

    return mask


def extract_black_curve_from_gray(img: np.ndarray) -> np.ndarray:
    """Extract just the black curve from a grayscale/inverted image."""
    h, w = img.shape[:2]

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Check if image is inverted (dark background)
    mean_val = np.mean(gray)
    if mean_val < 128:
        # Dark background - invert
        gray = 255 - gray

    # Find black pixels (the curve)
    black_mask = (gray < 100).astype(np.uint8) * 255

    # Remove text and annotation boxes using connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)

    curve_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        # Skip very small noise
        if area < 5:
            continue

        # Skip very large components (likely text boxes or axes)
        if area > 5000:
            continue

        # Text boxes characteristics: rectangular, wider than tall, moderate size
        aspect = cw / max(ch, 1)
        is_text_box = (
            (200 < area < 3000) and
            (1.5 < aspect < 10) and
            (ch < 30)
        )

        # Text characters: small-medium, roughly square
        is_text_char = (
            (10 < area < 200) and
            (0.3 < aspect < 3) and
            (cw < 25) and (ch < 25)
        )

        # Curve segments: elongated horizontal or vertical lines, or connected curve pieces
        is_curve = (
            (aspect > 5 or aspect < 0.2) or  # Very elongated
            (area > 100 and (cw > 30 or ch > 30)) or  # Large connected region
            (area < 50 and (cw < 8 or ch < 8))  # Small line segments or + marks
        )

        # Censoring marks: small + shapes
        is_censor = (area < 40 and 0.5 < aspect < 2 and cw < 12 and ch < 12)

        if (is_curve or is_censor) and not is_text_box and not is_text_char:
            curve_mask[labels == i] = 255

    return curve_mask


def extract_curves_clean(img: np.ndarray) -> np.ndarray:
    """Extract curves from plot area with improved text removal."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output = np.ones_like(img) * 255

    # === Step 1: Identify text/annotation regions to exclude ===

    text_box_regions = np.zeros((h, w), dtype=np.uint8)
    text_box_list = []  # Store box coordinates for later reference

    # Method 1: Find white/light gray rectangular regions (annotation box backgrounds)
    light_mask = (gray > 215).astype(np.uint8) * 255

    # Find contours of light regions
    contours_light, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_light:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        # Text boxes: rectangular, white-filled, moderate size
        if 100 < area < 25000:
            aspect = cw / max(ch, 1)
            if aspect > 0.8:  # At least roughly square
                # Expand significantly to include border AND nearby text
                margin = 15
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + cw + margin)
                y2 = min(h, y + ch + margin)
                cv2.rectangle(text_box_regions, (x1, y1), (x2, y2), 255, -1)
                text_box_list.append((x1, y1, x2-x1, y2-y1))

    # Method 2: Detect rectangular boxes by finding closed rectangular contours
    # in the edge image
    edges = cv2.Canny(gray, 50, 150)
    contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_edges:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Look for rectangles (4 vertices)
        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            area = cw * ch
            aspect = cw / max(ch, 1)

            # Text annotation boxes: rectangular, wider than tall, moderate size
            if 300 < area < 15000 and 1.5 < aspect < 12 and 10 < ch < 40:
                margin = 5
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + cw + margin)
                y2 = min(h, y + ch + margin)
                cv2.rectangle(text_box_regions, (x1, y1), (x2, y2), 255, -1)
                text_box_list.append((x1, y1, x2-x1, y2-y1))

    # Also mark the P-value region at the top
    # Look for text-dense region in top 12%
    top_region_h = int(h * 0.12)
    top_dark = (gray[:top_region_h, :] < 200).astype(np.uint8) * 255
    if np.sum(top_dark) > 500:
        cols_with_dark = np.any(top_dark > 0, axis=0)
        if np.any(cols_with_dark):
            left = np.argmax(cols_with_dark)
            right = w - np.argmax(cols_with_dark[::-1])
            cv2.rectangle(text_box_regions, (max(0, left-20), 0),
                         (min(w, right+20), top_region_h + 15), 255, -1)
            text_box_list.append((max(0, left-20), 0, right-left+40, top_region_h+15))

    # Method 3: Look for any remaining text boxes by finding regions with
    # text-like content (small dark blobs clustered together)
    # Focus on the right side of the image where labels often appear
    right_region_x = int(w * 0.55)
    right_region = gray[:, right_region_x:]
    right_dark = (right_region < 150).astype(np.uint8) * 255

    # Find connected components in right region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(right_dark, connectivity=8)

    # Group nearby components that might form a text box
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        rx = stats[i, cv2.CC_STAT_LEFT] + right_region_x
        ry = stats[i, cv2.CC_STAT_TOP]
        rcw = stats[i, cv2.CC_STAT_WIDTH]
        rch = stats[i, cv2.CC_STAT_HEIGHT]

        # Look for rectangular outlines (box borders)
        aspect = rcw / max(rch, 1)
        if 200 < area < 8000 and 1.5 < aspect < 10 and rch < 35:
            margin = 10
            x1 = max(0, rx - margin)
            y1 = max(0, ry - margin)
            x2 = min(w, rx + rcw + margin)
            y2 = min(h, ry + rch + margin)
            cv2.rectangle(text_box_regions, (x1, y1), (x2, y2), 255, -1)
            text_box_list.append((x1, y1, x2-x1, y2-y1))

    # === Step 2: Extract green curve ===
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    # Remove pixels in text box regions
    green_mask[text_box_regions > 0] = 0

    # === Step 3: Extract red curve ===
    red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    # Remove pixels in text box regions
    red_mask[text_box_regions > 0] = 0

    # === Step 4: Extract black curve (most challenging) ===
    # Black pixels: low value in all channels
    b, g, r = cv2.split(img)
    black_mask = ((b < 100) & (g < 100) & (r < 100)).astype(np.uint8) * 255

    # Remove pixels in text box regions (expanded)
    text_box_expanded = cv2.dilate(text_box_regions, np.ones((5, 5), np.uint8), iterations=1)
    black_mask[text_box_expanded > 0] = 0

    # Analyze connected components to filter text vs curves
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)

    curve_black = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        # Skip tiny noise
        if area < 3:
            continue

        # Skip components at very bottom edge (likely axis remnants)
        if y > h - 10:
            continue

        aspect = cw / max(ch, 1)
        fill_ratio = area / max(cw * ch, 1)

        # Check if inside or near a text box
        in_text_box = False
        for tx, ty, tcw, tch in text_box_list:
            if (tx - 5 <= x <= tx + tcw + 5) and (ty - 5 <= y <= ty + tch + 5):
                in_text_box = True
                break

        if in_text_box:
            continue

        # Identify curve-like components:
        # 1. Horizontal line segments (steps): wide, thin
        is_h_line = (cw > 10 and ch < 8)
        # 2. Vertical line segments (drops): tall, thin
        is_v_line = (ch > 10 and cw < 8)
        # 3. Censoring marks (+): small, roughly square
        is_censor = (area < 60 and 0.3 < aspect < 3 and cw < 18 and ch < 18)
        # 4. Connected curve segments: elongated shapes
        is_curve_segment = (aspect > 2.5 or aspect < 0.4)
        # 5. Large connected region (likely part of curve)
        is_large_segment = (area > 50)

        # Accept if it looks like part of a curve
        if is_h_line or is_v_line or is_censor or is_curve_segment or is_large_segment:
            curve_black[labels == i] = 255

    # === Step 5: Final cleanup ===
    # Remove small isolated noise
    kernel = np.ones((2, 2), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Apply colors to output
    output[green_mask > 0] = [0, 180, 0]  # Green
    output[red_mask > 0] = [0, 0, 180]    # Red (BGR)
    output[curve_black > 0] = [0, 0, 0]   # Black

    return output


def extract_curves_from_plot(img: np.ndarray) -> np.ndarray:
    """Legacy function - redirects to extract_curves_clean."""
    return extract_curves_clean(img)


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
