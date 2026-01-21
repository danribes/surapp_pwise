"""Axis detection and OCR module."""

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

from ..utils.config import config, get_tesseract_path
from ..utils.image_utils import (
    pil_to_cv2,
    preprocess_for_ocr,
    crop_region,
)


@dataclass
class AxisInfo:
    """Information about a detected axis."""
    start_point: tuple  # (x, y) pixel coordinates
    end_point: tuple    # (x, y) pixel coordinates
    is_horizontal: bool
    labels: list[tuple]  # List of (position, value) pairs
    title: Optional[str] = None
    min_value: float = 0.0
    max_value: float = 100.0


@dataclass
class AxisDetectionResult:
    """Result of axis detection."""
    x_axis: Optional[AxisInfo]
    y_axis: Optional[AxisInfo]
    origin: Optional[tuple]
    plot_area: Optional[tuple]  # (x, y, width, height)


class AxisExtractor:
    """Extracts axis information from graph images."""

    def __init__(self, image: Image.Image | np.ndarray):
        """Initialize axis extractor.

        Args:
            image: Input image (PIL Image or OpenCV array)
        """
        if isinstance(image, Image.Image):
            self.image = pil_to_cv2(image)
        else:
            self.image = image.copy()

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape[:2]

        # Configure Tesseract if available
        if pytesseract:
            tesseract_path = get_tesseract_path()
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def detect_axes(self) -> AxisDetectionResult:
        """Detect X and Y axes in the image.

        Returns:
            AxisDetectionResult with detected axis information
        """
        # Detect edges
        edges = cv2.Canny(self.gray, 50, 150)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return AxisDetectionResult(None, None, None, None)

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            if angle < 10:
                h_lines.append((x1, y1, x2, y2))
            elif angle > 80:
                v_lines.append((x1, y1, x2, y2))

        # Find the most likely X axis (longest horizontal line near bottom)
        x_axis = self._find_x_axis(h_lines)

        # Find the most likely Y axis (longest vertical line near left)
        y_axis = self._find_y_axis(v_lines)

        # Determine origin
        origin = None
        if x_axis and y_axis:
            # Origin is intersection of axes
            origin = (y_axis.start_point[0], x_axis.start_point[1])

        # Determine plot area
        plot_area = None
        if x_axis and y_axis:
            x = y_axis.start_point[0]
            y = y_axis.end_point[1]
            width = x_axis.end_point[0] - x_axis.start_point[0]
            height = x_axis.start_point[1] - y_axis.end_point[1]
            plot_area = (x, y, width, height)

        return AxisDetectionResult(x_axis, y_axis, origin, plot_area)

    def _find_x_axis(self, h_lines: list) -> Optional[AxisInfo]:
        """Find the X axis from horizontal lines.

        Args:
            h_lines: List of horizontal line tuples

        Returns:
            AxisInfo for X axis or None
        """
        if not h_lines:
            return None

        # Score lines based on length and position (prefer bottom of image)
        scored_lines = []
        for x1, y1, x2, y2 in h_lines:
            length = abs(x2 - x1)
            # Prefer lines in bottom 60% of image
            position_score = y1 / self.height if y1 > self.height * 0.4 else 0
            score = length * (1 + position_score)
            scored_lines.append((score, x1, y1, x2, y2))

        # Get best line
        scored_lines.sort(reverse=True)
        _, x1, y1, x2, y2 = scored_lines[0]

        # Ensure start is left of end
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        return AxisInfo(
            start_point=(x1, y1),
            end_point=(x2, y2),
            is_horizontal=True,
            labels=[]
        )

    def _find_y_axis(self, v_lines: list) -> Optional[AxisInfo]:
        """Find the Y axis from vertical lines.

        Args:
            v_lines: List of vertical line tuples

        Returns:
            AxisInfo for Y axis or None
        """
        if not v_lines:
            return None

        # Score lines based on length and position (prefer left of image)
        scored_lines = []
        for x1, y1, x2, y2 in v_lines:
            length = abs(y2 - y1)
            # Prefer lines in left 60% of image
            position_score = 1 - (x1 / self.width) if x1 < self.width * 0.6 else 0
            score = length * (1 + position_score)
            scored_lines.append((score, x1, y1, x2, y2))

        # Get best line
        scored_lines.sort(reverse=True)
        _, x1, y1, x2, y2 = scored_lines[0]

        # Ensure start is top and end is bottom
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        return AxisInfo(
            start_point=(x1, y2),  # Bottom point
            end_point=(x1, y1),    # Top point
            is_horizontal=False,
            labels=[]
        )

    def extract_axis_labels(self, axis: AxisInfo) -> list[tuple]:
        """Extract numeric labels from axis using OCR.

        Args:
            axis: AxisInfo object

        Returns:
            List of (pixel_position, numeric_value) tuples
        """
        if pytesseract is None:
            return []

        labels = []

        if axis.is_horizontal:
            # Look for labels below the X axis
            roi_y = axis.start_point[1]
            roi_height = min(50, self.height - roi_y)
            roi_x = axis.start_point[0]
            roi_width = axis.end_point[0] - axis.start_point[0]
        else:
            # Look for labels to the left of the Y axis
            roi_x = max(0, axis.start_point[0] - 60)
            roi_width = 60
            roi_y = axis.end_point[1]  # Top of axis
            roi_height = axis.start_point[1] - axis.end_point[1]

        if roi_width <= 0 or roi_height <= 0:
            return []

        # Crop ROI
        roi = crop_region(self.image, roi_x, roi_y, roi_width, roi_height)

        # Preprocess for OCR
        roi_processed = preprocess_for_ocr(roi)

        # Run OCR
        try:
            ocr_result = pytesseract.image_to_data(
                roi_processed,
                config=config.OCR_CONFIG,
                output_type=pytesseract.Output.DICT
            )
        except Exception:
            return []

        # Parse results
        for i, text in enumerate(ocr_result['text']):
            if not text.strip():
                continue

            # Try to parse as number
            try:
                value = self._parse_number(text)
                if value is not None:
                    # Get position
                    x = ocr_result['left'][i] + ocr_result['width'][i] // 2
                    y = ocr_result['top'][i] + ocr_result['height'][i] // 2

                    # Convert back to image coordinates
                    img_x = roi_x + x
                    img_y = roi_y + y

                    if axis.is_horizontal:
                        labels.append((img_x, value))
                    else:
                        labels.append((img_y, value))

            except (ValueError, TypeError):
                continue

        return labels

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from OCR text.

        Args:
            text: OCR text

        Returns:
            Parsed number or None
        """
        # Clean text
        text = text.strip().replace(',', '').replace(' ', '')

        # Handle percentage
        if '%' in text:
            text = text.replace('%', '')

        # Try to parse
        try:
            # Handle scientific notation
            if 'e' in text.lower():
                return float(text)

            # Handle decimal
            if '.' in text:
                return float(text)

            # Integer
            return float(int(text))

        except (ValueError, TypeError):
            return None

    def detect_scale(self, axis: AxisInfo) -> tuple:
        """Detect the scale of an axis (min, max values).

        Args:
            axis: AxisInfo with extracted labels

        Returns:
            (min_value, max_value) tuple
        """
        if not axis.labels:
            # Default scales for KM curves
            if axis.is_horizontal:
                return (0, 100)  # Time scale
            else:
                return (0, 100)  # Survival percentage

        values = [label[1] for label in axis.labels]

        return (min(values), max(values))

    def auto_detect_y_scale(self) -> str:
        """Detect if Y axis is 0-100 (percentage) or 0-1 (proportion).

        Returns:
            'percentage' or 'proportion'
        """
        # Look for '1.0' or '100' patterns in the left side of image
        roi = crop_region(self.image, 0, 0, self.width // 4, self.height)

        if pytesseract is None:
            return 'percentage'

        try:
            text = pytesseract.image_to_string(roi, config=config.OCR_CONFIG)
        except Exception:
            return 'percentage'

        # Check for proportion indicators
        if re.search(r'\b1\.0\b|\b0\.[0-9]\b', text):
            return 'proportion'

        return 'percentage'


def detect_axis_info(image: Image.Image | np.ndarray) -> AxisDetectionResult:
    """Convenience function to detect axis information.

    Args:
        image: Input image

    Returns:
        AxisDetectionResult with detected axes
    """
    extractor = AxisExtractor(image)
    return extractor.detect_axes()


def extract_calibration_from_axes(
    image: Image.Image | np.ndarray
) -> Optional[dict]:
    """Extract calibration data from detected axes.

    Args:
        image: Input image

    Returns:
        Dictionary with calibration points or None
    """
    extractor = AxisExtractor(image)
    result = extractor.detect_axes()

    if not result.origin or not result.x_axis or not result.y_axis:
        return None

    # Extract labels
    x_labels = extractor.extract_axis_labels(result.x_axis)
    y_labels = extractor.extract_axis_labels(result.y_axis)

    # Determine scales
    x_min, x_max = extractor.detect_scale(result.x_axis) if x_labels else (0, 100)
    y_min, y_max = extractor.detect_scale(result.y_axis) if y_labels else (0, 100)

    return {
        'origin': result.origin,
        'x_max_pixel': result.x_axis.end_point,
        'y_max_pixel': result.y_axis.end_point,
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'plot_area': result.plot_area
    }
