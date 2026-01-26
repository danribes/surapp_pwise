"""Axis detection and coordinate calibration for KM curve extraction.

This module provides robust axis detection using Canny edge detection and
Hough line transform to find actual X and Y axis lines, calculate stretching
factors, and accurately map pixel coordinates to data values.

Enhanced with OCR-based axis label detection for precise calibration based
on actual axis tick mark labels (finding x=0, y=1.0 positions).

AI-assisted fallback available when traditional methods have low confidence.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2
import numpy as np
import re

# Optional AI support
try:
    from .ai_axis_detector import AIAxisDetector, AIAxisResult, get_ai_axis_detector
    AI_AXIS_AVAILABLE = True
except ImportError:
    AI_AXIS_AVAILABLE = False


@dataclass
class DetectedAxisLine:
    """Represents a detected axis line in the image."""
    start_point: tuple[int, int]  # (x, y) pixel coordinates
    end_point: tuple[int, int]
    is_x_axis: bool
    confidence: float
    length_pixels: int


@dataclass
class AxisCalibrationResult:
    """Complete calibration information for coordinate mapping.

    Key rectangles:
    - plot_rectangle: The actual data area where curves are drawn (extraction area)
    - x_label_rectangle: Region containing X-axis labels (below plot)
    - y_label_rectangle: Region containing Y-axis labels (left of plot)

    Key coordinates:
    - origin: Pixel position of (x=0, y=0) in data coordinates
    - data_origin: Data coordinates at origin (usually (0.0, 0.0))
    - opposite_corner: Pixel position of (x_max, y_max) in data coordinates
    """
    x_axis: DetectedAxisLine
    y_axis: DetectedAxisLine
    origin: tuple[int, int]              # Pixel coords of data origin (x=0, y=0)
    x_axis_end: tuple[int, int]          # Right end of X-axis (x=max, y=0)
    y_axis_end: tuple[int, int]          # Top end of Y-axis (x=0, y=max)
    plot_rectangle: tuple[int, int, int, int]  # (x, y, w, h) - extraction area
    x_data_range: tuple[float, float]    # (0, time_max)
    y_data_range: tuple[float, float]    # (0.0, 1.0) or (0.0, 100.0)
    x_stretching_factor: float           # data_units / pixel
    y_stretching_factor: float           # data_units / pixel
    # New fields for text regions
    x_label_rectangle: Optional[tuple[int, int, int, int]] = None  # X-axis label area
    y_label_rectangle: Optional[tuple[int, int, int, int]] = None  # Y-axis label area
    detected_x_labels: Optional[List[Tuple[float, int]]] = None  # (value, x_pixel) pairs
    detected_y_labels: Optional[List[Tuple[float, int]]] = None  # (value, y_pixel) pairs


class AxisCalibrator:
    """Detects axis lines and provides coordinate calibration for KM plots."""

    def __init__(self, image: np.ndarray):
        """
        Initialize the calibrator with an image.

        Args:
            image: BGR image (OpenCV format)
        """
        self.image = image
        self.height, self.width = image.shape[:2]
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.calibration: Optional[AxisCalibrationResult] = None

    def detect_axis_lines(self) -> tuple[list, list]:
        """
        Detect lines in the image using Canny edge detection and Hough transform.

        Returns:
            Tuple of (horizontal_lines, vertical_lines) where each line is
            represented as ((x1, y1), (x2, y2), length, angle)
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Canny edge detection with lower thresholds for better line detection
        edges = cv2.Canny(blurred, 30, 100)

        # Hough Line Transform with more lenient parameters
        # Lower threshold and longer maxLineGap to connect broken axis lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=20
        )

        if lines is None:
            return [], []

        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Normalize angle to -90 to 90 range
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            line_info = ((x1, y1), (x2, y2), length, angle)

            # Classify by angle: horizontal (< 10 degrees), vertical (> 80 degrees)
            if abs(angle) < 10:
                horizontal_lines.append(line_info)
            elif abs(angle) > 80:
                vertical_lines.append(line_info)

        return horizontal_lines, vertical_lines

    def find_x_axis(self, horizontal_lines: list) -> Optional[DetectedAxisLine]:
        """
        Find the X-axis line (bottom horizontal line of the plot area).

        Looks for a horizontal line in the bottom 60% of the image that
        spans at least 30% of the image width.

        Args:
            horizontal_lines: List of detected horizontal lines

        Returns:
            DetectedAxisLine for the X-axis, or None if not found
        """
        min_length = self.width * 0.3  # Lower threshold
        search_region_top = self.height * 0.4  # Bottom 60% of image

        candidates = []

        for (x1, y1), (x2, y2), length, angle in horizontal_lines:
            # Check if line is in the bottom region
            y_mid = (y1 + y2) / 2
            if y_mid < search_region_top:
                continue

            # Check minimum length
            if length < min_length:
                continue

            # Calculate confidence based on length and position
            # Longer lines and lines closer to the bottom get higher confidence
            length_score = length / self.width
            position_score = y_mid / self.height  # Higher y = closer to bottom

            confidence = (length_score + position_score) / 2

            # Ensure consistent left-to-right ordering
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            candidates.append(DetectedAxisLine(
                start_point=(x1, y1),
                end_point=(x2, y2),
                is_x_axis=True,
                confidence=confidence,
                length_pixels=int(length)
            ))

        if not candidates:
            return None

        # Return the candidate with highest confidence
        return max(candidates, key=lambda c: c.confidence)

    def find_y_axis(self, vertical_lines: list) -> Optional[DetectedAxisLine]:
        """
        Find the Y-axis line (left vertical line of the plot area).

        Looks for a vertical line in the left 40% of the image that
        spans at least 50% of the image height.

        Args:
            vertical_lines: List of detected vertical lines

        Returns:
            DetectedAxisLine for the Y-axis, or None if not found
        """
        min_length = self.height * 0.5
        search_region_right = self.width * 0.4  # Left 40% of image

        candidates = []

        for (x1, y1), (x2, y2), length, angle in vertical_lines:
            # Check if line is in the left region
            x_mid = (x1 + x2) / 2
            if x_mid > search_region_right:
                continue

            # Check minimum length
            if length < min_length:
                continue

            # Calculate confidence based on length and position
            # Longer lines and lines closer to the left get higher confidence
            length_score = length / self.height
            position_score = 1.0 - (x_mid / self.width)  # Lower x = closer to left

            confidence = (length_score + position_score) / 2

            # Ensure consistent top-to-bottom ordering (y increases downward)
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            candidates.append(DetectedAxisLine(
                start_point=(x1, y1),  # Top of Y-axis
                end_point=(x2, y2),    # Bottom of Y-axis
                is_x_axis=False,
                confidence=confidence,
                length_pixels=int(length)
            ))

        if not candidates:
            return None

        # Return the candidate with highest confidence
        return max(candidates, key=lambda c: c.confidence)

    def find_origin(
        self,
        x_axis: DetectedAxisLine,
        y_axis: DetectedAxisLine
    ) -> tuple[int, int]:
        """
        Calculate the intersection point of the X and Y axes.

        The Y-axis is more reliably detected (unique vertical line on left),
        so we primarily use its bottom endpoint to determine the origin y-position.

        Args:
            x_axis: Detected X-axis line
            y_axis: Detected Y-axis line

        Returns:
            (x, y) pixel coordinates of the origin
        """
        # Origin X: use the Y-axis x-position (average of start/end)
        y_axis_x = (y_axis.start_point[0] + y_axis.end_point[0]) / 2
        origin_x = int(y_axis_x)

        # Origin Y: Use the BOTTOM of the Y-axis (higher y value)
        # This is more reliable than the x-axis detection because the y-axis
        # is a unique vertical line, while there may be multiple horizontal lines
        # (e.g., in at-risk tables below the plot)
        y_axis_bottom = max(y_axis.start_point[1], y_axis.end_point[1])
        x_axis_y = (x_axis.start_point[1] + x_axis.end_point[1]) / 2

        # Use the Y-axis bottom if it's close to the detected x-axis,
        # otherwise use the x-axis position (they should be within ~10 pixels)
        if abs(y_axis_bottom - x_axis_y) < 20:
            origin_y = int(x_axis_y)
        else:
            # Y-axis and x-axis don't match - prefer y-axis bottom
            # This handles cases where the x-axis detection picked a wrong line
            origin_y = int(y_axis_bottom)

        return (origin_x, origin_y)

    def find_axis_endpoints(
        self,
        x_axis: DetectedAxisLine,
        y_axis: DetectedAxisLine,
        origin: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Determine the end points of each axis (opposite from origin).

        Args:
            x_axis: Detected X-axis line
            y_axis: Detected Y-axis line
            origin: The origin point (axis intersection)

        Returns:
            Tuple of (x_axis_end, y_axis_end) where:
            - x_axis_end is the right end of the X-axis
            - y_axis_end is the top end of the Y-axis
        """
        # X-axis end: rightmost point of X-axis
        x_axis_end = x_axis.end_point
        if x_axis.start_point[0] > x_axis.end_point[0]:
            x_axis_end = x_axis.start_point

        # Y-axis end: topmost point of Y-axis (smallest y value)
        y_axis_end = y_axis.start_point
        if y_axis.end_point[1] < y_axis.start_point[1]:
            y_axis_end = y_axis.end_point

        return (x_axis_end, y_axis_end)

    def detect_y_scale(self) -> tuple[float, float]:
        """
        Detect if the Y-axis uses 0-1 or 0-100 scale.

        For KM curves, the Y-axis is always survival probability:
        - Most commonly: 0.0 to 1.0
        - Sometimes: 0 to 100 (percentage)

        Returns:
            (y_min, y_max) data range
        """
        # For KM curves, we assume standard 0.0 to 1.0 range
        # Future enhancement: OCR detection of axis labels
        return (0.0, 1.0)

    def detect_x_scale(self, time_max_hint: Optional[float] = None) -> tuple[float, float]:
        """
        Detect the X-axis (time) scale.

        Args:
            time_max_hint: If provided, use this as the time maximum

        Returns:
            (x_min, x_max) data range
        """
        # X-axis always starts at 0 for KM curves
        x_min = 0.0

        if time_max_hint is not None:
            x_max = time_max_hint
        else:
            # Default estimate based on typical KM curve time ranges
            x_max = 12.0  # Common default (months)

        return (x_min, x_max)

    def detect_tick_marks(
        self,
        axis: DetectedAxisLine,
        search_distance: int = 20
    ) -> list[int]:
        """
        Find tick marks along an axis.

        Args:
            axis: The axis line to analyze
            search_distance: Distance in pixels to search for tick marks

        Returns:
            List of pixel positions along the axis where tick marks are found
        """
        tick_positions = []

        if axis.is_x_axis:
            # For X-axis, look for vertical tick marks below the line
            y_base = int((axis.start_point[1] + axis.end_point[1]) / 2)
            x_start = min(axis.start_point[0], axis.end_point[0])
            x_end = max(axis.start_point[0], axis.end_point[0])

            for x in range(x_start, x_end, 5):
                # Check for dark pixels below the axis
                region = self.gray[y_base:y_base + search_distance, x:x + 3]
                if region.size > 0:
                    dark_pixels = np.sum(region < 100)
                    if dark_pixels > search_distance * 0.5:
                        tick_positions.append(x)
        else:
            # For Y-axis, look for horizontal tick marks to the left of the line
            x_base = int((axis.start_point[0] + axis.end_point[0]) / 2)
            y_start = min(axis.start_point[1], axis.end_point[1])
            y_end = max(axis.start_point[1], axis.end_point[1])

            for y in range(y_start, y_end, 5):
                # Check for dark pixels to the left of the axis
                region = self.gray[y:y + 3, max(0, x_base - search_distance):x_base]
                if region.size > 0:
                    dark_pixels = np.sum(region < 100)
                    if dark_pixels > search_distance * 0.3:
                        tick_positions.append(y)

        # Merge nearby positions (within 10 pixels)
        merged = []
        for pos in sorted(tick_positions):
            if not merged or pos - merged[-1] > 10:
                merged.append(pos)

        return merged

    def calculate_stretching_factors(
        self,
        origin: tuple[int, int],
        x_axis_end: tuple[int, int],
        y_axis_end: tuple[int, int],
        x_data_range: tuple[float, float],
        y_data_range: tuple[float, float]
    ) -> tuple[float, float]:
        """
        Compute data units per pixel for each axis.

        X-axis: pixels increase left-to-right, data increases left-to-right
        Y-axis: pixels increase top-to-bottom, data increases bottom-to-top (inverted)

        Args:
            origin: Origin point (axis intersection)
            x_axis_end: Right end of X-axis
            y_axis_end: Top end of Y-axis
            x_data_range: (x_min, x_max) in data units
            y_data_range: (y_min, y_max) in data units

        Returns:
            Tuple of (x_stretching_factor, y_stretching_factor) in data_units/pixel
        """
        # X-axis pixel range
        x_pixel_range = x_axis_end[0] - origin[0]
        if x_pixel_range <= 0:
            x_pixel_range = 1  # Avoid division by zero

        # Y-axis pixel range (origin_y > y_axis_end_y because y increases downward)
        y_pixel_range = origin[1] - y_axis_end[1]
        if y_pixel_range <= 0:
            y_pixel_range = 1  # Avoid division by zero

        # Calculate stretching factors
        x_data_span = x_data_range[1] - x_data_range[0]
        y_data_span = y_data_range[1] - y_data_range[0]

        x_stretching = x_data_span / x_pixel_range
        y_stretching = y_data_span / y_pixel_range

        return (x_stretching, y_stretching)

    def pixel_to_coord(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        """
        Map pixel coordinates to data coordinates.

        Args:
            pixel_x: X pixel position
            pixel_y: Y pixel position

        Returns:
            (time, survival) data coordinates where survival is ALWAYS
            normalized to 0-1 range (even if the axis shows 0-100%)
        """
        if self.calibration is None:
            raise ValueError("Calibration not performed. Call calibrate() first.")

        cal = self.calibration
        origin = cal.origin
        x_min, _ = cal.x_data_range
        y_min, y_max = cal.y_data_range

        # X: linear mapping, both increase in same direction
        time = (pixel_x - origin[0]) * cal.x_stretching_factor + x_min

        # Y: inverted mapping (pixels increase downward, data increases upward)
        survival = (origin[1] - pixel_y) * cal.y_stretching_factor + y_min

        # Normalize survival to 0-1 range if Y-axis uses percentage (0-100)
        if y_max > 1.5:  # Y-axis is in percentage (0-100 scale)
            survival = survival / y_max

        return (time, survival)

    def coord_to_pixel(self, time: float, survival: float) -> tuple[int, int]:
        """
        Map data coordinates to pixel coordinates.

        Args:
            time: Time value (X-axis)
            survival: Survival probability (Y-axis)

        Returns:
            (pixel_x, pixel_y) coordinates
        """
        if self.calibration is None:
            raise ValueError("Calibration not performed. Call calibrate() first.")

        cal = self.calibration
        origin = cal.origin
        x_min, _ = cal.x_data_range
        y_min, _ = cal.y_data_range

        # X: linear mapping
        pixel_x = int((time - x_min) / cal.x_stretching_factor + origin[0])

        # Y: inverted mapping
        pixel_y = int(origin[1] - (survival - y_min) / cal.y_stretching_factor)

        return (pixel_x, pixel_y)

    def is_in_plot_area(self, pixel_x: int, pixel_y: int) -> bool:
        """
        Check if a pixel position is within the plot rectangle.

        Args:
            pixel_x: X pixel position
            pixel_y: Y pixel position

        Returns:
            True if the pixel is within the plot area
        """
        if self.calibration is None:
            return False

        x, y, w, h = self.calibration.plot_rectangle
        return x <= pixel_x <= x + w and y <= pixel_y <= y + h

    def calibrate(
        self,
        time_max_hint: Optional[float] = None
    ) -> Optional[AxisCalibrationResult]:
        """
        Perform full axis detection and calibration.

        Args:
            time_max_hint: Optional hint for the maximum time value

        Returns:
            AxisCalibrationResult if successful, None otherwise
        """
        # Detect lines
        horizontal_lines, vertical_lines = self.detect_axis_lines()

        if not horizontal_lines or not vertical_lines:
            return None

        # Find X and Y axes
        x_axis = self.find_x_axis(horizontal_lines)
        y_axis = self.find_y_axis(vertical_lines)

        if x_axis is None or y_axis is None:
            return None

        # Find origin and endpoints
        origin = self.find_origin(x_axis, y_axis)
        x_axis_end, y_axis_end = self.find_axis_endpoints(x_axis, y_axis, origin)

        # Detect scales
        x_data_range = self.detect_x_scale(time_max_hint)
        y_data_range = self.detect_y_scale()

        # Calculate stretching factors
        x_stretching, y_stretching = self.calculate_stretching_factors(
            origin, x_axis_end, y_axis_end, x_data_range, y_data_range
        )

        # Calculate plot rectangle
        plot_x = origin[0]
        plot_y = y_axis_end[1]
        plot_w = x_axis_end[0] - origin[0]
        plot_h = origin[1] - y_axis_end[1]
        plot_rectangle = (plot_x, plot_y, plot_w, plot_h)

        # Store and return result
        self.calibration = AxisCalibrationResult(
            x_axis=x_axis,
            y_axis=y_axis,
            origin=origin,
            x_axis_end=x_axis_end,
            y_axis_end=y_axis_end,
            plot_rectangle=plot_rectangle,
            x_data_range=x_data_range,
            y_data_range=y_data_range,
            x_stretching_factor=x_stretching,
            y_stretching_factor=y_stretching
        )

        return self.calibration

    def calibrate_with_fallback(
        self,
        time_max_hint: Optional[float] = None,
        curve_bounds: Optional[tuple[int, int, int, int]] = None
    ) -> Optional[AxisCalibrationResult]:
        """
        Attempt calibration with fallback to curve-based estimation.

        If axis line detection fails, uses the provided curve bounds
        to estimate the axis positions.

        Args:
            time_max_hint: Optional hint for the maximum time value
            curve_bounds: Optional (x_min, x_max, y_min, y_max) from detected curves

        Returns:
            AxisCalibrationResult if successful, None otherwise
        """
        # First try standard calibration
        result = self.calibrate(time_max_hint)

        if result is not None:
            return result

        # Fallback: use curve bounds if provided
        if curve_bounds is None:
            return None

        x_min_px, x_max_px, y_min_px, y_max_px = curve_bounds

        # Estimate axis positions from curve extent
        # Add small margins to account for axis lines being outside curve area
        margin_x = int((x_max_px - x_min_px) * 0.02)
        margin_y = int((y_max_px - y_min_px) * 0.05)

        estimated_origin_x = x_min_px - margin_x
        estimated_origin_y = y_max_px + margin_y

        # Create synthetic axis lines
        x_axis = DetectedAxisLine(
            start_point=(estimated_origin_x, estimated_origin_y),
            end_point=(x_max_px + margin_x, estimated_origin_y),
            is_x_axis=True,
            confidence=0.5,  # Lower confidence for estimated axis
            length_pixels=x_max_px - estimated_origin_x + margin_x
        )

        y_axis = DetectedAxisLine(
            start_point=(estimated_origin_x, y_min_px - margin_y),
            end_point=(estimated_origin_x, estimated_origin_y),
            is_x_axis=False,
            confidence=0.5,
            length_pixels=estimated_origin_y - y_min_px + margin_y
        )

        origin = (estimated_origin_x, estimated_origin_y)
        x_axis_end = x_axis.end_point
        y_axis_end = y_axis.start_point

        # Detect scales
        x_data_range = self.detect_x_scale(time_max_hint)
        y_data_range = self.detect_y_scale()

        # Calculate stretching factors
        x_stretching, y_stretching = self.calculate_stretching_factors(
            origin, x_axis_end, y_axis_end, x_data_range, y_data_range
        )

        # Calculate plot rectangle
        plot_x = origin[0]
        plot_y = y_axis_end[1]
        plot_w = x_axis_end[0] - origin[0]
        plot_h = origin[1] - y_axis_end[1]
        plot_rectangle = (plot_x, plot_y, plot_w, plot_h)

        # Store and return result
        self.calibration = AxisCalibrationResult(
            x_axis=x_axis,
            y_axis=y_axis,
            origin=origin,
            x_axis_end=x_axis_end,
            y_axis_end=y_axis_end,
            plot_rectangle=plot_rectangle,
            x_data_range=x_data_range,
            y_data_range=y_data_range,
            x_stretching_factor=x_stretching,
            y_stretching_factor=y_stretching
        )

        return self.calibration

    def detect_y100_from_curves(self) -> Optional[int]:
        """
        Detect the Y pixel position where survival=100% (or 1.0) by finding
        where curves start.

        KM curves always start at survival=1.0 (100%), so the topmost curve
        pixels indicate where y=100 is in the plot.

        Works for both color and grayscale images.

        Returns:
            Y pixel position for survival=100, or None if not detected
        """
        # Try color detection first (for colored curve images)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        colored_mask = (saturation > 80) & (value > 50) & (value < 250)
        colored_rows = np.where(np.any(colored_mask, axis=1))[0]

        if len(colored_rows) > 10:
            # Color image - use colored pixels
            top_rows = colored_rows[:max(1, len(colored_rows) // 20)]
            return int(np.median(top_rows))

        # Grayscale image - detect dark curve pixels
        # Focus on the plot area (exclude margins with text/labels)
        if self.calibration is not None:
            px, py, pw, ph = self.calibration.plot_rectangle
            origin_y = self.calibration.origin[1]
        else:
            # Estimate plot area
            px, py = int(self.width * 0.1), int(self.height * 0.1)
            pw, ph = int(self.width * 0.8), int(self.height * 0.7)
            origin_y = py + ph

        # Look for dark pixels in the plot region (curve lines are dark)
        plot_region = self.gray[py:origin_y, px:px+pw]

        # Threshold to find dark pixels (curves)
        _, binary = cv2.threshold(plot_region, 180, 255, cv2.THRESH_BINARY_INV)

        # Find rows with dark pixels
        dark_rows = np.where(np.sum(binary, axis=1) > pw * 0.02)[0]  # At least 2% of width

        if len(dark_rows) == 0:
            return None

        # The topmost dark row in the plot region (offset by py)
        # Filter out noise by looking at top rows with significant content
        top_rows = dark_rows[:max(1, len(dark_rows) // 20)]
        y_100_pixel = int(np.median(top_rows)) + py

        return y_100_pixel

    def detect_x0_from_curves(self) -> Optional[int]:
        """
        Detect the X pixel position where time=0 by finding where curves start.

        KM curves always start at time=0, so the leftmost curve pixels
        indicate where x=0 is in the plot.

        Works for both color and grayscale images.

        Returns:
            X pixel position for time=0, or None if not detected
        """
        # Try color detection first (for colored curve images)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        colored_mask = (saturation > 80) & (value > 50) & (value < 250)
        colored_cols = np.where(np.any(colored_mask, axis=0))[0]

        if len(colored_cols) > 10:
            # Color image - use leftmost colored pixels
            left_cols = colored_cols[:max(1, len(colored_cols) // 20)]
            return int(np.median(left_cols))

        # Grayscale image - detect dark curve pixels
        if self.calibration is not None:
            px, py, pw, ph = self.calibration.plot_rectangle
            origin_x = self.calibration.origin[0]
        else:
            px, py = int(self.width * 0.1), int(self.height * 0.1)
            pw, ph = int(self.width * 0.8), int(self.height * 0.7)
            origin_x = px

        # Look for dark pixels in the plot region
        plot_region = self.gray[py:py+ph, px:px+pw]
        _, binary = cv2.threshold(plot_region, 180, 255, cv2.THRESH_BINARY_INV)

        # Find columns with dark pixels
        dark_cols = np.where(np.sum(binary, axis=0) > ph * 0.02)[0]

        if len(dark_cols) == 0:
            return None

        # The leftmost dark column (offset by px)
        left_cols = dark_cols[:max(1, len(dark_cols) // 20)]
        x_0_pixel = int(np.median(left_cols)) + px

        return x_0_pixel

    def refine_plot_bounds_from_curves(self) -> Optional[tuple[int, int, int, int]]:
        """
        Refine the plot rectangle by detecting actual curve positions.

        This is more accurate than axis-line-based detection because:
        1. KM curves always start at time=0 and survival=1.0 (100%)
        2. The leftmost curve pixel indicates the true x=0 position
        3. The topmost curve pixel indicates the true y=100 position

        Returns:
            Refined (x, y, w, h) plot rectangle, or None if unable to refine
        """
        if self.calibration is None:
            return None

        # Get current plot bounds
        px, py, pw, ph = self.calibration.plot_rectangle
        origin_x = self.calibration.origin[0]  # X position of y-axis (time=0)
        origin_y = self.calibration.origin[1]  # Y position of x-axis (survival=0)

        # Detect Y=100 position from curves
        y_100_pixel = self.detect_y100_from_curves()

        # Detect X=0 position from curves
        x_0_pixel = self.detect_x0_from_curves()

        # Start with current values
        refined_px = px
        refined_py = py
        refined_pw = pw
        refined_ph = ph

        # Refine Y-axis if detected
        if y_100_pixel is not None and y_100_pixel < origin_y and y_100_pixel >= 0:
            refined_py = y_100_pixel
            refined_ph = origin_y - y_100_pixel

        # Refine X-axis if detected
        if x_0_pixel is not None and x_0_pixel < px + pw and x_0_pixel >= 0:
            # Adjust width to maintain right edge
            right_edge = px + pw
            refined_px = x_0_pixel
            refined_pw = right_edge - x_0_pixel

        # Sanity checks
        if refined_ph < 50 or refined_pw < 50:
            return None

        return (refined_px, refined_py, refined_pw, refined_ph)

    # ==========================================================================
    # OCR-based axis label detection methods
    # ==========================================================================

    def _get_ocr_reader(self):
        """
        Get or initialize the OCR reader (lazy loading).

        Returns:
            OCR reader instance and type ('easyocr' or 'pytesseract')
        """
        if not hasattr(self, '_ocr_reader') or self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(['en'], verbose=False)
                self._ocr_type = 'easyocr'
            except ImportError:
                try:
                    import pytesseract
                    self._ocr_reader = pytesseract
                    self._ocr_type = 'pytesseract'
                except ImportError:
                    self._ocr_reader = None
                    self._ocr_type = None

        return self._ocr_reader, getattr(self, '_ocr_type', None)

    def _ocr_region(self, region: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Run OCR on an image region.

        Args:
            region: Image region (grayscale or BGR)

        Returns:
            List of (text, bbox, confidence) tuples where bbox is (x, y, w, h)
        """
        reader, ocr_type = self._get_ocr_reader()

        if reader is None:
            return []

        results = []

        if ocr_type == 'easyocr':
            # EasyOCR returns: [[bbox_points, text, confidence], ...]
            ocr_results = reader.readtext(region)

            for detection in ocr_results:
                bbox_points, text, confidence = detection
                # Convert polygon to bounding box
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - x), int(max(ys) - y)
                results.append((text, (x, y, w, h), confidence))

        elif ocr_type == 'pytesseract':
            # Use pytesseract with bounding box data
            import pytesseract
            from PIL import Image

            if len(region.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(region)

            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

            for i, text in enumerate(data['text']):
                if text.strip():
                    conf = int(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.5
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    results.append((text.strip(), (x, y, w, h), conf))

        return results

    def _parse_axis_number(self, text: str) -> Optional[float]:
        """
        Parse a numeric value from axis label text.

        Handles various formats:
        - "0", "1.0", "100", "0.5"
        - With % sign: "100%", "50%"
        - With commas: "1,000"

        Rejects text that appears to be labels/titles (contains multiple letters).

        Args:
            text: Raw OCR text

        Returns:
            Parsed numeric value, or None if not parseable
        """
        # Clean the text
        text = text.strip()

        # Reject text that contains too many letters (likely a label, not a number)
        # Count letters vs digits in original text
        letter_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())

        # If there are more letters than digits, it's probably not a number
        if letter_count > digit_count + 1:
            return None

        # If text is too long and has letters, it's probably a label
        if len(text) > 5 and letter_count > 1:
            return None

        # Remove common non-numeric characters
        text = text.replace('%', '').replace(',', '').replace(' ', '')

        # Only apply OCR corrections if text is short (1-3 chars)
        if len(text) <= 3:
            # Handle common OCR mistakes for short numbers
            text = text.replace('O', '0').replace('o', '0')
            text = text.replace('l', '1').replace('I', '1')
            # Don't replace S->5 as it causes too many false positives

        # Try to extract a number
        # First, handle ".X" patterns (like ".8" which should be "0.8")
        if text.startswith('.') and len(text) >= 2 and text[1].isdigit():
            text = '0' + text

        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None

    def detect_x_axis_labels(self) -> List[Tuple[float, int]]:
        """
        Detect numeric labels along the X-axis using OCR.

        Scans the region immediately below the X-axis line for numeric labels
        (tick mark values). Uses 3x scaling for better OCR accuracy on small text.

        Returns:
            List of (value, x_pixel_center) tuples sorted by x position
        """
        # Determine the region to scan (below x-axis)
        if self.calibration is not None:
            x_axis = self.calibration.x_axis
            # Use the actual X-axis line position, not the estimated origin
            x_axis_y = max(x_axis.start_point[1], x_axis.end_point[1])
            x_start = min(x_axis.start_point[0], x_axis.end_point[0]) - 30
            x_end = max(x_axis.start_point[0], x_axis.end_point[0]) + 50
            # Scan region below x-axis for tick mark numbers
            # Skip tick marks (first ~8 pixels) and capture numbers (~8-35 pixels below axis)
            y_start = x_axis_y + 8  # Skip tick marks
            y_end = min(self.height, x_axis_y + 35)  # Capture full number height
        else:
            # Estimate x-axis region
            x_start = int(self.width * 0.08)
            x_end = int(self.width * 0.98)
            y_start = int(self.height * 0.85)  # Lower region for x-axis labels
            y_end = int(self.height * 0.92)

        # Ensure valid bounds
        x_start = max(0, x_start)
        x_end = min(self.width, x_end)
        y_start = max(0, y_start)
        y_end = min(self.height, y_end)

        if x_end <= x_start or y_end <= y_start:
            return []

        # Extract the region (use color for better OCR)
        label_region = self.image[y_start:y_end, x_start:x_end]

        # Scale up 3x for better OCR on very small axis numbers
        scaled_region = cv2.resize(label_region, None, fx=3, fy=3,
                                   interpolation=cv2.INTER_CUBIC)

        # Run OCR on scaled region
        ocr_results = self._ocr_region(scaled_region)

        # If no results, try with binary thresholding preprocessing
        if not ocr_results or all(self._parse_axis_number(t) is None for t, _, _ in ocr_results):
            # Convert to grayscale if needed
            if len(scaled_region.shape) == 3:
                gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = scaled_region
            # Apply adaptive thresholding to enhance text
            binary_region = cv2.adaptiveThreshold(
                gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            # Convert back to 3-channel for OCR
            binary_rgb = cv2.cvtColor(binary_region, cv2.COLOR_GRAY2BGR)
            ocr_results_binary = self._ocr_region(binary_rgb)
            if ocr_results_binary:
                ocr_results = ocr_results_binary

        # If still no numeric results, try pytesseract with sparse text mode
        if not ocr_results or all(self._parse_axis_number(t) is None for t, _, _ in ocr_results):
            try:
                import pytesseract
                import re
                # Convert to grayscale
                if len(scaled_region.shape) == 3:
                    gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
                else:
                    gray_region = scaled_region
                # Binary threshold
                _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Try PSM 11 (sparse text) - better for widely spaced numbers
                try:
                    data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT,
                                                     config='--psm 11 -c tessedit_char_whitelist=0123456789')
                    ocr_results = []
                    for i, text in enumerate(data['text']):
                        if text.strip() and data['conf'][i] > 20:
                            value = self._parse_axis_number(text.strip())
                            if value is not None and 0 <= value <= 200:
                                x_pos = data['left'][i] + data['width'][i] // 2
                                ocr_results.append((text.strip(), (x_pos, 0, data['width'][i], data['height'][i]),
                                                   data['conf'][i] / 100.0))
                except Exception:
                    pass

                # If PSM 11 didn't work, try PSM 6 with morphological preprocessing
                if not ocr_results:
                    # Dilate slightly to connect broken digits
                    kernel = np.ones((2, 2), np.uint8)
                    dilated = cv2.dilate(binary, kernel, iterations=1)

                    try:
                        data = pytesseract.image_to_data(dilated, output_type=pytesseract.Output.DICT,
                                                         config='--psm 6 -c tessedit_char_whitelist=0123456789')
                        for i, text in enumerate(data['text']):
                            if text.strip() and data['conf'][i] > 20:
                                value = self._parse_axis_number(text.strip())
                                if value is not None and 0 <= value <= 200:
                                    x_pos = data['left'][i] + data['width'][i] // 2
                                    ocr_results.append((text.strip(), (x_pos, 0, data['width'][i], data['height'][i]),
                                                       data['conf'][i] / 100.0))
                    except Exception:
                        pass

                # Final fallback: simple string extraction
                if not ocr_results:
                    text = pytesseract.image_to_string(binary, config='--psm 11 -c tessedit_char_whitelist=0123456789')
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        region_width = scaled_region.shape[1]
                        spacing = region_width // (len(numbers) + 1)
                        for i, num_str in enumerate(numbers):
                            x_pos = spacing * (i + 1)
                            ocr_results.append((num_str, (x_pos, 0, 20, 20), 0.8))

            except ImportError:
                pass  # pytesseract not available

        labels = []
        for text, (x, y, w, h), confidence in ocr_results:
            value = self._parse_axis_number(text)
            if value is not None and confidence > 0.2:
                # Calculate center x position in original image coordinates
                # Divide by 3 to account for 3x scaling
                x_center = x_start + (x + w // 2) // 3
                labels.append((value, x_center))

        # Sort by x position
        labels.sort(key=lambda item: item[1])

        # Filter out non-time values (time axis should have values like 0, 3, 6, etc.)
        # Keep only reasonable time values (0-200 range for months/years)
        labels = [(v, x) for v, x in labels if 0 <= v <= 200]

        return labels

    def detect_y_axis_labels(self) -> List[Tuple[float, int]]:
        """
        Detect numeric labels along the Y-axis using OCR.

        Scans the region to the left of the Y-axis line for numeric labels
        (tick mark values). Uses 3x scaling for better OCR accuracy on small text.

        Returns:
            List of (value, y_pixel_center) tuples sorted by y position (top to bottom)
        """
        # Determine the region to scan (left of y-axis)
        if self.calibration is not None:
            origin = self.calibration.origin
            y_axis_end = self.calibration.y_axis_end
            # Scan 60 pixels to the left of y-axis
            x_start = max(0, origin[0] - 60)
            x_end = origin[0] + 5
            y_start = max(0, y_axis_end[1] - 10)
            # Scan within the plot area
            y_end = origin[1] + 5
        else:
            # Estimate y-axis region
            x_start = int(self.width * 0.02)
            x_end = int(self.width * 0.15)
            y_start = int(self.height * 0.05)
            y_end = int(self.height * 0.65)

        # Ensure valid bounds
        x_start = max(0, x_start)
        x_end = min(self.width, x_end)
        y_start = max(0, y_start)
        y_end = min(self.height, y_end)

        if x_end <= x_start or y_end <= y_start:
            return []

        # Extract the region (use color for better OCR)
        label_region = self.image[y_start:y_end, x_start:x_end]

        # Scale up 3x for better OCR on small Y-axis text (often smaller than X-axis)
        scaled_region = cv2.resize(label_region, None, fx=3, fy=3,
                                   interpolation=cv2.INTER_CUBIC)

        # Run OCR on scaled region
        ocr_results = self._ocr_region(scaled_region)

        # Try with preprocessing if no good results
        if not ocr_results or all(self._parse_axis_number(t) is None for t, _, _ in ocr_results):
            # Convert to grayscale if needed
            if len(scaled_region.shape) == 3:
                gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = scaled_region
            # Apply adaptive thresholding
            binary_region = cv2.adaptiveThreshold(
                gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            binary_rgb = cv2.cvtColor(binary_region, cv2.COLOR_GRAY2BGR)
            ocr_results_binary = self._ocr_region(binary_rgb)
            if ocr_results_binary:
                ocr_results = ocr_results_binary

        # Try pytesseract as fallback for Y-axis (better for small 2-3 digit numbers)
        if not ocr_results or all(self._parse_axis_number(t) is None for t, _, _ in ocr_results):
            try:
                import pytesseract
                # Convert to grayscale
                if len(scaled_region.shape) == 3:
                    gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
                else:
                    gray_region = scaled_region
                # Binary threshold
                _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Use image_to_data for bounding boxes
                try:
                    data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT,
                                                     config='--psm 6 -c tessedit_char_whitelist=0123456789.')
                    ocr_results = []
                    for i, text in enumerate(data['text']):
                        if text.strip() and data['conf'][i] > 0:
                            value = self._parse_axis_number(text.strip())
                            if value is not None and (0 <= value <= 1.5 or 0 <= value <= 150):
                                y_pos = data['top'][i] + data['height'][i] // 2
                                ocr_results.append((text.strip(), (0, y_pos, data['width'][i], data['height'][i]),
                                                   data['conf'][i] / 100.0))
                except Exception:
                    pass
            except ImportError:
                pass

        labels = []
        for text, (x, y, w, h), confidence in ocr_results:
            value = self._parse_axis_number(text)
            if value is not None and confidence > 0.3:
                # Filter for valid survival axis values (0-1 or 0-100 range)
                if not (0 <= value <= 1.5 or 0 <= value <= 150):
                    continue
                # Calculate center y position in original image coordinates
                # Divide by 3 to account for 3x scaling
                y_center = y_start + (y + h // 2) // 3
                labels.append((value, y_center))

        # Sort by y position (top to bottom)
        labels.sort(key=lambda item: item[1])

        return labels

    def find_x0_label_position(self) -> Optional[int]:
        """
        Find the x-pixel position corresponding to the "0" label on the X-axis.

        This is the true origin for the time axis. If the "0" label is not
        directly detected, extrapolates from other labels.

        Returns:
            X pixel position where time=0, or None if not found
        """
        labels = self.detect_x_axis_labels()

        if not labels:
            return None

        # Look for label with value 0 (or close to 0)
        for value, x_pixel in labels:
            if abs(value) < 0.001:  # Essentially zero
                return x_pixel

        # If no exact 0, try to extrapolate from other labels
        if len(labels) >= 2:
            # Sort by value
            sorted_by_value = sorted(labels, key=lambda item: item[0])

            # Use the two smallest values to extrapolate back to 0
            val1, x1 = sorted_by_value[0]
            val2, x2 = sorted_by_value[1]

            if val1 > 0 and abs(val2 - val1) > 0.001:
                # Linear extrapolation
                pixels_per_unit = (x2 - x1) / (val2 - val1)
                x0 = x1 - val1 * pixels_per_unit

                # Sanity check: x0 should be to the left of x1
                if x0 < x1 and x0 >= 0:
                    return int(x0)

        return None

    def _correct_y_axis_ocr_errors(
        self, labels: List[Tuple[float, int]]
    ) -> List[Tuple[float, int]]:
        """
        Detect and correct common OCR errors in Y-axis labels.

        Common OCR errors:
        - "25" read as "5" (missing leading "2")
        - "75" read as "5" (missing leading "7")
        - "100" read as "00", "10", or not detected at all

        This method uses SPACING between labels to infer the correct values,
        rather than assuming the topmost detected label is 100%.

        Args:
            labels: List of (value, y_pixel) tuples

        Returns:
            Corrected list of (value, y_pixel) tuples
        """
        if len(labels) < 2:
            return labels

        # Sort by y position (top to bottom, smaller y = higher on screen)
        sorted_labels = sorted(labels, key=lambda x: x[1])
        values = [v for v, _ in sorted_labels]

        # Find if we have a "0" label (this is usually reliable)
        zero_idx = None
        zero_y = None
        for i, (v, y) in enumerate(sorted_labels):
            if abs(v) < 0.001:
                zero_idx = i
                zero_y = y
                break

        if zero_y is None:
            # No zero detected, can't correct reliably
            return labels

        # Check for suspicious patterns - multiple identical small values
        value_counts = {}
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1

        duplicates = [(v, c) for v, c in value_counts.items() if c > 1 and 2 <= v <= 9]

        if not duplicates:
            # No obvious OCR errors detected
            return labels

        # Use spacing-based correction
        # Calculate the spacing between detected labels
        y_positions = [y for _, y in sorted_labels]

        # Calculate average spacing between consecutive labels
        spacings = []
        for i in range(len(y_positions) - 1):
            spacings.append(y_positions[i + 1] - y_positions[i])

        if not spacings:
            return labels

        # The spacing should correspond to 25% intervals (most common)
        avg_spacing = sum(spacings) / len(spacings)

        # Now infer the correct values based on distance from y=0
        corrected = []
        for orig_v, orig_y in sorted_labels:
            if abs(orig_v) < 0.001:
                # This is the 0 label, keep it
                corrected.append((0.0, orig_y))
            else:
                # Calculate how many "steps" above 0 this label is
                distance_from_zero = zero_y - orig_y  # positive = above zero
                steps_from_zero = round(distance_from_zero / avg_spacing)

                # Each step is 25%
                inferred_value = steps_from_zero * 25.0
                inferred_value = max(0, min(100, inferred_value))

                corrected.append((inferred_value, orig_y))

        # Sort by y position again
        corrected.sort(key=lambda x: x[1])

        return corrected

    def find_y100_label_position(self) -> Optional[int]:
        """
        Find the y-pixel position corresponding to the "1.0" or "100" label on Y-axis.

        This is the top of the survival axis (100% survival).
        If the 100% label isn't directly detected, extrapolates from other labels.

        Returns:
            Y pixel position where survival=100% (1.0), or None if not found
        """
        labels = self.detect_y_axis_labels()

        if not labels:
            return None

        # Try to correct common OCR errors (e.g., "25" read as "5")
        corrected_labels = self._correct_y_axis_ocr_errors(labels)

        # Look for label with value 1.0 or 100 (representing 100% survival)
        for value, y_pixel in corrected_labels:
            # Check for 1.0 (decimal) or 100 (percentage)
            if abs(value - 1.0) < 0.01 or abs(value - 100) < 1:
                return y_pixel

        # Also check original labels (in case correction was wrong)
        for value, y_pixel in labels:
            # Check for 1.0 (decimal) or 100 (percentage)
            if abs(value - 1.0) < 0.01 or abs(value - 100) < 1:
                return y_pixel

        # If using 0-1 scale, look for value closest to 1.0 at the top
        decimal_labels = [(v, y) for v, y in corrected_labels if 0 <= v <= 1.5]
        if decimal_labels:
            # The topmost label close to 1.0 should be the 100% mark
            for value, y_pixel in decimal_labels:
                if value >= 0.9:  # Close to 1.0
                    return y_pixel

        # If using 0-100 scale
        percent_labels = [(v, y) for v, y in corrected_labels if 0 <= v <= 150]
        if percent_labels:
            for value, y_pixel in percent_labels:
                if value >= 90:  # Close to 100%
                    return y_pixel

        # EXTRAPOLATION: If 100% label not found, extrapolate from corrected labels
        # This handles cases where OCR misses the 100 label entirely
        if len(corrected_labels) >= 2:
            # Sort by y position (top to bottom)
            sorted_by_y = sorted(corrected_labels, key=lambda item: item[1])

            # Find the 0 label position
            zero_y = None
            for v, y in sorted_by_y:
                if abs(v) < 0.1:
                    zero_y = y
                    break

            if zero_y is not None:
                # Calculate spacing per 25% from corrected labels
                # Find highest corrected value and its position
                sorted_by_value = sorted(corrected_labels, key=lambda x: x[0], reverse=True)
                highest_val, highest_y = sorted_by_value[0]

                if highest_val > 0:
                    # Calculate pixels per percentage point
                    # zero_y is at 0%, highest_y is at highest_val%
                    pixels_per_percent = (zero_y - highest_y) / highest_val

                    # Extrapolate to 100%
                    y_100 = zero_y - (100 * pixels_per_percent)

                    if y_100 >= 0:
                        return int(y_100)

        # Fallback: use original labels for extrapolation
        if len(labels) >= 2:
            # Sort by value (descending - highest value first)
            sorted_by_value = sorted(labels, key=lambda item: item[0], reverse=True)

            # Check if this looks like percentage scale (values like 0, 25, 50, 75)
            # or decimal scale (0.0, 0.25, 0.5, 0.75)
            max_value = sorted_by_value[0][0]
            is_percentage = max_value > 1.5

            if is_percentage:
                # Extrapolate to 100%
                # Find two labels to calculate pixel spacing per unit
                val1, y1 = sorted_by_value[0]  # Highest value detected
                val2, y2 = sorted_by_value[1]  # Second highest

                if val1 != val2:
                    # Pixels per percentage point (Y increases downward, value increases upward)
                    pixels_per_unit = (y2 - y1) / (val1 - val2)
                    # Extrapolate to 100%
                    y_100 = y1 - (100 - val1) * pixels_per_unit
                    if y_100 >= 0:
                        return int(y_100)
            else:
                # Extrapolate to 1.0
                val1, y1 = sorted_by_value[0]
                val2, y2 = sorted_by_value[1]

                if val1 != val2:
                    pixels_per_unit = (y2 - y1) / (val1 - val2)
                    y_1 = y1 - (1.0 - val1) * pixels_per_unit
                    if y_1 >= 0:
                        return int(y_1)

        return None

    def find_x_max_label_position(self) -> Optional[Tuple[float, int]]:
        """
        Find the maximum time value and its x-pixel position on the X-axis.

        Returns:
            Tuple of (max_time_value, x_pixel) or None if not found
        """
        labels = self.detect_x_axis_labels()

        if not labels:
            return None

        # Find the rightmost label (should be max time)
        # Sort by x position and take the last one
        rightmost = max(labels, key=lambda item: item[1])

        return rightmost

    def find_y_min_label_position(self) -> Optional[Tuple[float, int]]:
        """
        Find the minimum survival value (usually 0) and its y-pixel position.

        Returns:
            Tuple of (min_survival_value, y_pixel) or None if not found
        """
        labels = self.detect_y_axis_labels()

        if not labels:
            return None

        # Look for label with value 0
        for value, y_pixel in labels:
            if abs(value) < 0.001:
                return (value, y_pixel)

        # Otherwise, return the bottommost label (highest y value)
        bottommost = max(labels, key=lambda item: item[1])

        return bottommost

    def calibrate_from_axis_labels(
        self,
        time_max_hint: Optional[float] = None,
        verbose: bool = False
    ) -> Optional[AxisCalibrationResult]:
        """
        Perform calibration using OCR-detected axis labels.

        This method:
        1. Uses basic axis line detection for initial plot geometry
        2. Defines text rectangles for X and Y axis labels
        3. Uses OCR to read labels from text rectangles
        4. Calculates precise origin (x=0, y=0) and opposite corner positions
        5. Defines the extraction rectangle based on detected label positions

        Key outputs:
        - extraction_rectangle (plot_rectangle): Where curves are drawn
        - x_label_rectangle: Region containing X-axis labels
        - y_label_rectangle: Region containing Y-axis labels
        - origin: Pixel position of (time=0, survival=0)
        - y_axis_end: Pixel position of (time=0, survival=max)

        Args:
            time_max_hint: Optional hint for the maximum time value
            verbose: If True, print calibration details

        Returns:
            AxisCalibrationResult if successful, None otherwise
        """
        # First, do basic axis line detection to get plot geometry
        basic_calibration = self.calibrate(time_max_hint)

        if basic_calibration is None:
            return None

        # Define text label rectangles based on axis positions
        # X-axis labels: below the x-axis line
        x_label_rect = (
            max(0, basic_calibration.origin[0] - 20),  # x: start slightly left of origin
            basic_calibration.origin[1] + 2,           # y: just below x-axis
            basic_calibration.x_axis_end[0] - basic_calibration.origin[0] + 40,  # width
            min(40, self.height - basic_calibration.origin[1] - 2)  # height: up to 40px
        )

        # Y-axis labels: left of the y-axis line
        y_label_rect = (
            max(0, basic_calibration.origin[0] - 70),  # x: up to 70px left of y-axis
            max(0, basic_calibration.y_axis_end[1] - 10),  # y: start above top of y-axis
            70,  # width
            basic_calibration.origin[1] - basic_calibration.y_axis_end[1] + 20  # height
        )

        # Now detect labels using OCR
        x_labels = self.detect_x_axis_labels()
        y_labels = self.detect_y_axis_labels()

        # Try to correct common Y-axis OCR errors
        y_labels_corrected = self._correct_y_axis_ocr_errors(y_labels)

        if verbose:
            print(f"  OCR detected X-axis labels: {x_labels}")
            print(f"  OCR detected Y-axis labels: {y_labels}")
            if y_labels_corrected != y_labels:
                print(f"  Y-axis labels corrected to: {y_labels_corrected}")

        # Find key positions from labels
        x0_pixel = self.find_x0_label_position()
        y100_pixel = self.find_y100_label_position()
        x_max_result = self.find_x_max_label_position()
        y_min_result = self.find_y_min_label_position()

        # Determine origin_x: use x=0 label position if available
        if x0_pixel is not None:
            origin_x = x0_pixel
            if verbose:
                print(f"  Origin X from label: {origin_x}")
        else:
            origin_x = basic_calibration.origin[0]
            if verbose:
                print(f"  Origin X from axis line: {origin_x}")

        # origin_y: use y=0 label position if detected, otherwise axis line
        if y_min_result is not None and abs(y_min_result[0]) < 0.001:
            origin_y = y_min_result[1]
            if verbose:
                print(f"  Origin Y from y=0 label: {origin_y}")
        else:
            origin_y = basic_calibration.origin[1]
            if verbose:
                print(f"  Origin Y from axis line: {origin_y}")

        # y_top: use y=100% (or 1.0) label position if available
        if y100_pixel is not None:
            y_top = y100_pixel
            if verbose:
                print(f"  Y-top from label: {y_top}")
        else:
            y_top = basic_calibration.y_axis_end[1]
            if verbose:
                print(f"  Y-top from axis line: {y_top}")

        # x_axis_end: use max label position if available
        if x_max_result is not None:
            x_max_value, x_max_pixel = x_max_result
            x_axis_end = (x_max_pixel, origin_y)
            x_data_max = time_max_hint if time_max_hint is not None else x_max_value
            if verbose:
                print(f"  X-max from label: pixel={x_max_pixel}, value={x_max_value}")
        else:
            x_axis_end = basic_calibration.x_axis_end
            x_data_max = basic_calibration.x_data_range[1]

        # Create the refined calibration
        origin = (origin_x, origin_y)
        y_axis_end = (origin_x, y_top)

        # Determine data ranges
        x_data_range = (0.0, x_data_max)

        # Detect if Y-axis uses 0-1 or 0-100 scale from labels
        # Use corrected labels if available for better detection
        # Be smarter about this: OCR often misreads "0.8" as "8", "0.6" as "6", etc.
        # If we see a mix of decimal values (0.0-1.0) and integer-like values (2-9),
        # it's likely a 0-1 scale with OCR errors, not a 0-100 scale
        uses_percentage = False
        labels_to_check = y_labels_corrected if y_labels_corrected else y_labels
        if labels_to_check:
            values = [v for v, _ in labels_to_check]
            # Check for actual 0-100 percentage values (like 20, 40, 60, 80, 100)
            large_values = [v for v in values if v > 1.5]
            small_values = [v for v in values if v <= 1.5]

            # If we have values like 1.0, 0.9, 0.1, 0.0 (proper decimals)
            # AND values like 8, 6, 5, 4, 3, 2 (likely misread decimals)
            # Then it's probably a 0-1 scale
            has_proper_decimals = any(0 < v < 1 for v in small_values)
            has_endpoints = (0.0 in values or any(abs(v) < 0.01 for v in values)) and \
                           (1.0 in values or any(abs(v - 1.0) < 0.01 for v in values))

            # Suspicious values that look like misread decimals (2-9 as single digits)
            suspicious_misreads = [v for v in large_values if 2 <= v <= 9]

            if has_proper_decimals or has_endpoints:
                # We have evidence of a 0-1 scale
                if suspicious_misreads and max(large_values) < 10:
                    # These are likely misread decimals (0.8 -> 8, etc.)
                    if verbose:
                        print(f"  Y-axis scale: Detected 0-1 scale (OCR misreads corrected: {suspicious_misreads})")
                    uses_percentage = False
                elif max(values) > 10:
                    # Actual percentage scale (values like 20, 40, 60, etc.)
                    uses_percentage = True
            elif large_values and max(large_values) > 10:
                # Clear percentage scale
                uses_percentage = True

        if uses_percentage:
            y_data_range = (0.0, 100.0)
        else:
            y_data_range = (0.0, 1.0)

        if verbose:
            print(f"  Data ranges: X={x_data_range}, Y={y_data_range}")

        # Calculate stretching factors
        x_stretching, y_stretching = self.calculate_stretching_factors(
            origin, x_axis_end, y_axis_end, x_data_range, y_data_range
        )

        # Calculate extraction rectangle (plot area for curves)
        # This is the area bounded by:
        # - Left: x=0 position (origin_x)
        # - Right: x=max position (x_axis_end[0])
        # - Top: y=max position (y_top)
        # - Bottom: y=0 position (origin_y)
        plot_x = origin_x
        plot_y = y_top
        plot_w = x_axis_end[0] - origin_x
        plot_h = origin_y - y_top

        # Sanity checks
        if plot_w < 50 or plot_h < 50:
            if verbose:
                print(f"  WARNING: Plot rectangle too small ({plot_w}x{plot_h}), falling back")
            return basic_calibration  # Fall back to basic calibration

        plot_rectangle = (plot_x, plot_y, plot_w, plot_h)

        if verbose:
            print(f"  Extraction rectangle: x={plot_x}, y={plot_y}, w={plot_w}, h={plot_h}")
            print(f"  Origin (x=0, y=0): pixel ({origin_x}, {origin_y})")
            print(f"  Opposite corner (x=max, y=max): pixel ({x_axis_end[0]}, {y_top})")

        # Store and return result with all rectangles
        self.calibration = AxisCalibrationResult(
            x_axis=basic_calibration.x_axis,
            y_axis=basic_calibration.y_axis,
            origin=origin,
            x_axis_end=x_axis_end,
            y_axis_end=y_axis_end,
            plot_rectangle=plot_rectangle,
            x_data_range=x_data_range,
            y_data_range=y_data_range,
            x_stretching_factor=x_stretching,
            y_stretching_factor=y_stretching,
            x_label_rectangle=x_label_rect,
            y_label_rectangle=y_label_rect,
            detected_x_labels=x_labels if x_labels else None,
            detected_y_labels=y_labels_corrected if y_labels_corrected else y_labels
        )

        return self.calibration

    def calibrate_with_ai_fallback(
        self,
        time_max_hint: Optional[float] = None,
        use_ai: bool = True,
        ai_config=None,
        verbose: bool = False
    ) -> Optional[AxisCalibrationResult]:
        """
        Perform calibration with AI fallback for low-confidence cases.

        This method:
        1. First attempts OCR-based axis label detection
        2. If OCR fails or has low confidence, uses AI vision model
        3. Validates OCR results against AI when both are available

        Args:
            time_max_hint: Optional hint for the maximum time value
            use_ai: Whether to use AI fallback (default True)
            ai_config: Optional AIConfig for AI detection
            verbose: If True, print calibration details

        Returns:
            AxisCalibrationResult if successful, None otherwise
        """
        # First try OCR-based calibration
        ocr_result = self.calibrate_from_axis_labels(time_max_hint, verbose=verbose)

        # Evaluate OCR confidence
        ocr_confidence = 0.0
        if ocr_result is not None:
            # Calculate confidence based on detected labels
            has_x_labels = ocr_result.detected_x_labels and len(ocr_result.detected_x_labels) >= 2
            has_y_labels = ocr_result.detected_y_labels and len(ocr_result.detected_y_labels) >= 2

            if has_x_labels and has_y_labels:
                ocr_confidence = 0.9
            elif has_x_labels or has_y_labels:
                ocr_confidence = 0.6
            else:
                ocr_confidence = 0.4

        if verbose:
            print(f"  OCR calibration confidence: {ocr_confidence:.1%}")

        # Use AI fallback if available and needed
        if use_ai and AI_AXIS_AVAILABLE and (ocr_confidence < 0.7 or ocr_result is None):
            ai_detector = get_ai_axis_detector(ai_config)

            if ai_detector is not None and ai_detector.is_available:
                if verbose:
                    print("  Using AI fallback for axis detection...")

                ai_result = ai_detector.detect_axes(self.image, quiet=not verbose)

                if ai_result is not None and ai_result.is_valid:
                    # Use AI result to improve calibration
                    if ocr_result is None:
                        # No OCR result - create calibration from AI
                        return self._calibrate_from_ai_result(ai_result, time_max_hint, verbose)
                    else:
                        # Validate/correct OCR with AI
                        return self._merge_ocr_and_ai_calibration(
                            ocr_result, ai_result, time_max_hint, verbose
                        )

        return ocr_result

    def _calibrate_from_ai_result(
        self,
        ai_result,
        time_max_hint: Optional[float] = None,
        verbose: bool = False
    ) -> Optional[AxisCalibrationResult]:
        """
        Create calibration using only AI-detected axis ranges.

        This is used when OCR completely fails.

        Args:
            ai_result: AIAxisResult from AI detection
            time_max_hint: Optional time max override
            verbose: Print details

        Returns:
            AxisCalibrationResult or None
        """
        # First get basic axis line detection
        basic = self.calibrate(time_max_hint)

        if basic is None:
            if verbose:
                print("  Cannot create AI calibration - no axis lines detected")
            return None

        # Override data ranges with AI-detected values
        if ai_result.x_range:
            x_min, x_max = ai_result.x_range
            if time_max_hint is not None:
                x_max = time_max_hint
            x_data_range = (x_min, x_max)
        else:
            x_data_range = basic.x_data_range

        if ai_result.y_range:
            y_data_range = ai_result.y_range
        else:
            y_data_range = basic.y_data_range

        if verbose:
            print(f"  AI calibration: X={x_data_range}, Y={y_data_range}")

        # Recalculate stretching factors
        x_stretching, y_stretching = self.calculate_stretching_factors(
            basic.origin, basic.x_axis_end, basic.y_axis_end,
            x_data_range, y_data_range
        )

        # Create new calibration with AI-corrected ranges
        self.calibration = AxisCalibrationResult(
            x_axis=basic.x_axis,
            y_axis=basic.y_axis,
            origin=basic.origin,
            x_axis_end=basic.x_axis_end,
            y_axis_end=basic.y_axis_end,
            plot_rectangle=basic.plot_rectangle,
            x_data_range=x_data_range,
            y_data_range=y_data_range,
            x_stretching_factor=x_stretching,
            y_stretching_factor=y_stretching,
            x_label_rectangle=basic.x_label_rectangle,
            y_label_rectangle=basic.y_label_rectangle,
            detected_x_labels=None,  # AI doesn't provide individual labels
            detected_y_labels=None
        )

        return self.calibration

    def _merge_ocr_and_ai_calibration(
        self,
        ocr_result: AxisCalibrationResult,
        ai_result,
        time_max_hint: Optional[float] = None,
        verbose: bool = False
    ) -> AxisCalibrationResult:
        """
        Merge OCR and AI calibration results, preferring higher confidence source.

        Args:
            ocr_result: Calibration from OCR
            ai_result: AIAxisResult from AI detection
            time_max_hint: Optional time max override
            verbose: Print details

        Returns:
            Merged AxisCalibrationResult
        """
        # Start with OCR result
        x_data_range = ocr_result.x_data_range
        y_data_range = ocr_result.y_data_range

        # Check if AI disagrees significantly and should override
        if ai_result.x_range and ai_result.confidence > 0.7:
            ai_x_max = ai_result.x_range[1]
            ocr_x_max = ocr_result.x_data_range[1]

            # If AI detected a very different time range, consider using it
            if abs(ai_x_max - ocr_x_max) / max(ai_x_max, 1) > 0.2:
                if verbose:
                    print(f"  AI override X range: {ocr_result.x_data_range} -> {ai_result.x_range}")
                x_data_range = ai_result.x_range

        if ai_result.y_range and ai_result.confidence > 0.7:
            # Check for scale mismatch (0-1 vs 0-100)
            ai_is_percent = ai_result.y_range[1] > 1.5
            ocr_is_percent = y_data_range[1] > 1.5

            if ai_is_percent != ocr_is_percent:
                if verbose:
                    print(f"  AI override Y scale: {y_data_range} -> {ai_result.y_range}")
                y_data_range = ai_result.y_range

        # Apply time_max_hint override
        if time_max_hint is not None:
            x_data_range = (x_data_range[0], time_max_hint)

        # Recalculate stretching if ranges changed
        if x_data_range != ocr_result.x_data_range or y_data_range != ocr_result.y_data_range:
            x_stretching, y_stretching = self.calculate_stretching_factors(
                ocr_result.origin, ocr_result.x_axis_end, ocr_result.y_axis_end,
                x_data_range, y_data_range
            )

            self.calibration = AxisCalibrationResult(
                x_axis=ocr_result.x_axis,
                y_axis=ocr_result.y_axis,
                origin=ocr_result.origin,
                x_axis_end=ocr_result.x_axis_end,
                y_axis_end=ocr_result.y_axis_end,
                plot_rectangle=ocr_result.plot_rectangle,
                x_data_range=x_data_range,
                y_data_range=y_data_range,
                x_stretching_factor=x_stretching,
                y_stretching_factor=y_stretching,
                x_label_rectangle=ocr_result.x_label_rectangle,
                y_label_rectangle=ocr_result.y_label_rectangle,
                detected_x_labels=ocr_result.detected_x_labels,
                detected_y_labels=ocr_result.detected_y_labels
            )

            return self.calibration

        return ocr_result
