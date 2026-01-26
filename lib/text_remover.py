#!/usr/bin/env python3
"""
Remove all text from KM plot images using OCR detection and inpainting.
"""

import cv2
import numpy as np

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def remove_all_text(img: np.ndarray, plot_bounds: tuple = None) -> np.ndarray:
    """
    Remove ALL text from the image using OCR detection and inpainting.

    Args:
        img: BGR image
        plot_bounds: Optional (x, y, w, h) to focus text removal on plot area

    Returns:
        Image with text removed (inpainted with surrounding colors)
    """
    if not HAS_TESSERACT:
        print("Warning: Tesseract not available, returning original image")
        return img.copy()

    result = img.copy()
    h, w = img.shape[:2]

    # Create mask for all text regions
    text_mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to RGB for tesseract
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get text data with bounding boxes
    print("  Detecting text regions with OCR...")
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, timeout=30)

    text_count = 0
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

        # Include all detected text with reasonable confidence
        if conf > 20 and len(text) > 0:
            x = data['left'][i]
            y = data['top'][i]
            tw = data['width'][i]
            th = data['height'][i]

            # Add padding around text
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + tw + padding)
            y2 = min(h, y + th + padding)

            text_mask[y1:y2, x1:x2] = 255
            text_count += 1

    print(f"  Found {text_count} text regions")

    # Also detect text in different orientations/sizes by preprocessing
    # Use multiple threshold levels to catch different text styles
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try to detect legend box area (usually top-right with white/light background)
    # and mark it for removal

    # Dilate the text mask to ensure complete coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)

    # Inpaint to remove text
    print("  Inpainting to remove text...")
    result = cv2.inpaint(result, text_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return result, text_mask


def remove_legend_region(img: np.ndarray, plot_bounds: tuple) -> np.ndarray:
    """
    Remove the legend region from the image by detecting and inpainting it.

    Args:
        img: BGR image
        plot_bounds: (x, y, w, h) of plot area

    Returns:
        Image with legend region removed
    """
    result = img.copy()
    px, py, pw, ph = plot_bounds

    # Legend is typically in the top-right of the plot area
    # Create a mask for the legend region
    legend_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Mark the top portion of the plot as potential legend area
    legend_height = int(ph * 0.25)
    legend_start_x = px + int(pw * 0.3)  # Start from 30% of plot width

    # Fill the legend region
    legend_mask[py:py+legend_height, legend_start_x:px+pw] = 255

    # Inpaint
    result = cv2.inpaint(result, legend_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    return result


def extract_curves_skeleton(img: np.ndarray, colors: list, plot_bounds: tuple) -> list:
    """
    Extract curves using skeletonization for clean single-pixel-wide lines.

    Args:
        img: BGR image (should have text removed)
        colors: List of color info dicts with 'hsv_lower', 'hsv_upper', 'name'
        plot_bounds: (x, y, w, h) of plot area

    Returns:
        List of (points, color_info) tuples
    """
    from skimage.morphology import skeletonize

    px, py, pw, ph = plot_bounds
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    results = []

    for color_info in colors:
        name = color_info.get('name', 'unknown')
        print(f"  Processing {name} curve with skeletonization...")

        # Create color mask
        mask = cv2.inRange(hsv, color_info['hsv_lower'], color_info['hsv_upper'])

        # Crop to plot area
        mask_roi = mask[py:py+ph, px:px+pw]

        # Clean up the mask
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)

        # Skeletonize to get single-pixel-wide curves
        skeleton = skeletonize(mask_roi > 0).astype(np.uint8) * 255

        # Extract points from skeleton
        points = []
        ys, xs = np.where(skeleton > 0)

        for x, y in zip(xs, ys):
            # Convert to full image coordinates
            full_x = px + x
            full_y = py + y
            points.append((full_x, full_y))

        print(f"    Found {len(points)} skeleton points")
        results.append((points, color_info))

    return results
