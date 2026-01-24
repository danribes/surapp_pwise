"""Axis detection and coordinate calibration for KM curve extraction.

This module provides robust axis detection using Canny edge detection and
Hough line transform to find actual X and Y axis lines, calculate stretching
factors, and accurately map pixel coordinates to data values.
"""

from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


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
    """Complete calibration information for coordinate mapping."""
    x_axis: DetectedAxisLine
    y_axis: DetectedAxisLine
    origin: tuple[int, int]              # Axis intersection point
    x_axis_end: tuple[int, int]          # Right end of X-axis
    y_axis_end: tuple[int, int]          # Top end of Y-axis
    plot_rectangle: tuple[int, int, int, int]  # (x, y, w, h)
    x_data_range: tuple[float, float]    # (0, time_max)
    y_data_range: tuple[float, float]    # (0.0, 1.0)
    x_stretching_factor: float           # data_units / pixel
    y_stretching_factor: float           # data_units / pixel


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

        Args:
            x_axis: Detected X-axis line
            y_axis: Detected Y-axis line

        Returns:
            (x, y) pixel coordinates of the origin
        """
        # X-axis is horizontal, so the origin Y is the Y-axis endpoint (bottom)
        # Y-axis is vertical, so the origin X is the Y-axis X position

        # Use the bottom point of Y-axis (higher y value) for origin y
        origin_y = y_axis.end_point[1]

        # Use the X-axis starting X position (left side) for origin x
        origin_x = x_axis.start_point[0]

        # Refine: use the average of nearby positions for better accuracy
        y_axis_x = (y_axis.start_point[0] + y_axis.end_point[0]) / 2
        x_axis_y = (x_axis.start_point[1] + x_axis.end_point[1]) / 2

        # The origin should be near the intersection
        origin_x = int(y_axis_x)
        origin_y = int(x_axis_y)

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
            (time, survival) data coordinates
        """
        if self.calibration is None:
            raise ValueError("Calibration not performed. Call calibrate() first.")

        cal = self.calibration
        origin = cal.origin
        x_min, _ = cal.x_data_range
        y_min, _ = cal.y_data_range

        # X: linear mapping, both increase in same direction
        time = (pixel_x - origin[0]) * cal.x_stretching_factor + x_min

        # Y: inverted mapping (pixels increase downward, data increases upward)
        survival = (origin[1] - pixel_y) * cal.y_stretching_factor + y_min

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
