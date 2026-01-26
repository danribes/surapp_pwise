"""
Image Preprocessor for KM Curve Extraction.

Cleans up KM plot images by removing confusing elements:
- Vertical tick marks (censoring indicators)
- Text labels and annotations
- Gridlines
- Legend boxes

This preprocessing improves pixel-based curve detection accuracy.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


class ImagePreprocessor:
    """Clean KM plot images for better curve extraction."""

    def __init__(self, image: np.ndarray):
        """
        Initialize preprocessor with an image.

        Args:
            image: BGR image (OpenCV format)
        """
        self.original = image.copy()
        self.image = image.copy()
        self.height, self.width = image.shape[:2]
        self.masks = {}  # Store masks for debugging

    def remove_vertical_lines(self, min_length: int = 10, max_width: int = 3) -> 'ImagePreprocessor':
        """
        Remove thin vertical lines (tick marks, censoring indicators).

        Args:
            min_length: Minimum line length to remove (pixels)
            max_width: Maximum line width to consider (pixels)

        Returns:
            self for chaining
        """
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()

        # Create vertical kernel for morphological operations
        # This detects vertical structures
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))

        # Detect vertical lines
        # First, threshold to get binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Detect vertical lines using morphological opening
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # Dilate slightly to ensure we cover the full width
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_width, 1))
        vertical_lines = cv2.dilate(vertical_lines, dilate_kernel)

        # Store mask for debugging
        self.masks['vertical_lines'] = vertical_lines

        # Inpaint the vertical lines
        if len(self.image.shape) == 3:
            self.image = cv2.inpaint(self.image, vertical_lines, 3, cv2.INPAINT_TELEA)
        else:
            self.image = cv2.inpaint(
                cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR),
                vertical_lines, 3, cv2.INPAINT_TELEA
            )
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        return self

    def remove_text_regions(self, padding: int = 5) -> 'ImagePreprocessor':
        """
        Detect and remove text regions using OCR.

        Args:
            padding: Extra pixels to mask around detected text

        Returns:
            self for chaining
        """
        if not HAS_TESSERACT:
            print("Warning: pytesseract not available, skipping text removal")
            return self

        # Convert to grayscale for OCR
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        # Get text bounding boxes
        try:
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"Warning: OCR failed: {e}")
            return self

        # Create mask for text regions
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            # Skip empty detections
            if not data['text'][i].strip():
                continue

            # Get bounding box
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(self.width, x + w + padding)
            y2 = min(self.height, y + h + padding)

            # Fill mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Store mask for debugging
        self.masks['text_regions'] = mask

        # Inpaint text regions
        if mask.any():
            if len(self.image.shape) == 3:
                self.image = cv2.inpaint(self.image, mask, 5, cv2.INPAINT_TELEA)
            else:
                self.image = cv2.inpaint(
                    cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR),
                    mask, 5, cv2.INPAINT_TELEA
                )
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        return self

    def remove_gridlines(self, threshold: int = 240) -> 'ImagePreprocessor':
        """
        Remove light gridlines from the plot.

        Args:
            threshold: Brightness threshold for gridlines (0-255)

        Returns:
            self for chaining
        """
        # Convert to grayscale
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()

        # Find very light pixels (gridlines are usually light gray)
        _, light_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Use morphological operations to connect gridlines
        # Horizontal kernel
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        h_lines = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, h_kernel)

        # Vertical kernel
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        v_lines = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, v_kernel)

        # Combine
        gridlines = cv2.bitwise_or(h_lines, v_lines)

        # Store mask
        self.masks['gridlines'] = gridlines

        # Replace gridlines with white
        if len(self.image.shape) == 3:
            self.image[gridlines > 0] = [255, 255, 255]
        else:
            self.image[gridlines > 0] = 255

        return self

    def remove_legend_box(self, search_region: Tuple[float, float, float, float] = (0.5, 0.0, 1.0, 0.5)) -> 'ImagePreprocessor':
        """
        Detect and remove legend box.

        Args:
            search_region: (x_start, y_start, x_end, y_end) as fractions of image size
                          Default searches top-right quadrant

        Returns:
            self for chaining
        """
        # Define search region
        x1 = int(self.width * search_region[0])
        y1 = int(self.height * search_region[1])
        x2 = int(self.width * search_region[2])
        y2 = int(self.height * search_region[3])

        # Extract region
        region = self.image[y1:y2, x1:x2].copy()

        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Detect rectangles (legend boxes have rectangular borders)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find rectangular contours that could be legend boxes
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Legend boxes are typically small and have reasonable aspect ratio
                if 20 < w < 200 and 20 < h < 150:
                    # Adjust coordinates to full image
                    cv2.rectangle(mask, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), 255, -1)

        # Store mask
        self.masks['legend_box'] = mask

        # Fill legend area with white (or inpaint)
        if mask.any():
            if len(self.image.shape) == 3:
                self.image[mask > 0] = [255, 255, 255]
            else:
                self.image[mask > 0] = 255

        return self

    def crop_to_plot_area(self, bounds: Tuple[int, int, int, int]) -> 'ImagePreprocessor':
        """
        Crop image to just the plot area.

        Args:
            bounds: (x, y, width, height) of plot area

        Returns:
            self for chaining
        """
        x, y, w, h = bounds
        self.image = self.image[y:y+h, x:x+w]
        self.height, self.width = self.image.shape[:2]
        return self

    def enhance_curves(self, saturation_boost: float = 1.5) -> 'ImagePreprocessor':
        """
        Enhance curve colors for better detection.

        Args:
            saturation_boost: Factor to boost color saturation

        Returns:
            self for chaining
        """
        if len(self.image.shape) != 3:
            return self

        # Convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Boost saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)

        # Convert back
        self.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return self

    def get_result(self) -> np.ndarray:
        """Get the preprocessed image."""
        return self.image

    def get_debug_image(self) -> np.ndarray:
        """Get debug image showing all masks overlaid."""
        debug = self.original.copy()
        if len(debug.shape) == 2:
            debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

        colors = {
            'vertical_lines': (0, 0, 255),    # Red
            'text_regions': (0, 255, 0),      # Green
            'gridlines': (255, 0, 0),         # Blue
            'legend_box': (255, 255, 0),      # Cyan
        }

        for name, mask in self.masks.items():
            if mask is not None and mask.any():
                color = colors.get(name, (255, 255, 255))
                debug[mask > 0] = color

        return debug

    def save_debug_images(self, output_dir: Path):
        """Save all debug images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save original
        cv2.imwrite(str(output_dir / "preprocess_original.png"), self.original)

        # Save result
        cv2.imwrite(str(output_dir / "preprocess_cleaned.png"), self.image)

        # Save debug overlay
        cv2.imwrite(str(output_dir / "preprocess_debug.png"), self.get_debug_image())

        # Save individual masks
        for name, mask in self.masks.items():
            if mask is not None:
                cv2.imwrite(str(output_dir / f"preprocess_mask_{name}.png"), mask)


def preprocess_km_image(
    image: np.ndarray,
    remove_ticks: bool = True,
    remove_text: bool = True,
    remove_grid: bool = False,
    remove_legend: bool = False,
    enhance_colors: bool = False,
    debug_dir: Optional[Path] = None
) -> np.ndarray:
    """
    Preprocess a KM plot image for better curve extraction.

    Args:
        image: BGR image (OpenCV format)
        remove_ticks: Remove vertical tick marks (censoring indicators)
        remove_text: Remove text labels and annotations
        remove_grid: Remove gridlines
        remove_legend: Remove legend box
        enhance_colors: Enhance curve colors
        debug_dir: Optional directory to save debug images

    Returns:
        Preprocessed image
    """
    preprocessor = ImagePreprocessor(image)

    if remove_ticks:
        preprocessor.remove_vertical_lines()

    if remove_text:
        preprocessor.remove_text_regions()

    if remove_grid:
        preprocessor.remove_gridlines()

    if remove_legend:
        preprocessor.remove_legend_box()

    if enhance_colors:
        preprocessor.enhance_curves()

    if debug_dir:
        preprocessor.save_debug_images(debug_dir)

    return preprocessor.get_result()
