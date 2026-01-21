"""Kaplan-Meier curve tracing module."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..utils.config import config
from ..utils.image_utils import pil_to_cv2, preprocess_for_curve_detection, detect_edges


@dataclass
class LineSegment:
    """Represents a line segment."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def is_horizontal(self) -> bool:
        """Check if line is approximately horizontal."""
        angle = np.degrees(np.arctan2(abs(self.y2 - self.y1), abs(self.x2 - self.x1)))
        return angle < config.LINE_ANGLE_TOLERANCE

    @property
    def is_vertical(self) -> bool:
        """Check if line is approximately vertical."""
        angle = np.degrees(np.arctan2(abs(self.y2 - self.y1), abs(self.x2 - self.x1)))
        return angle > (90 - config.LINE_ANGLE_TOLERANCE)

    @property
    def length(self) -> float:
        """Calculate line length."""
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def start_point(self) -> tuple:
        """Get start point (leftmost for horizontal, topmost for vertical)."""
        if self.is_horizontal:
            return (min(self.x1, self.x2), self.y1) if self.x1 < self.x2 else (min(self.x1, self.x2), self.y2)
        return (self.x1, min(self.y1, self.y2)) if self.y1 < self.y2 else (self.x2, min(self.y1, self.y2))

    def end_point(self) -> tuple:
        """Get end point (rightmost for horizontal, bottommost for vertical)."""
        if self.is_horizontal:
            return (max(self.x1, self.x2), self.y1) if self.x1 > self.x2 else (max(self.x1, self.x2), self.y2)
        return (self.x1, max(self.y1, self.y2)) if self.y1 > self.y2 else (self.x2, max(self.y1, self.y2))


@dataclass
class CalibrationData:
    """Stores calibration information for coordinate transformation."""
    origin_pixel: tuple  # (x, y) pixel coordinates of origin
    x_max_pixel: tuple   # (x, y) pixel coordinates of max X point
    y_max_pixel: tuple   # (x, y) pixel coordinates of max Y point (100% survival)
    x_max_value: float   # Maximum X value (time)
    y_max_value: float   # Maximum Y value (1.0 or 100)

    def pixel_to_coord(self, pixel_x: int, pixel_y: int) -> tuple:
        """Convert pixel coordinates to graph coordinates.

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate

        Returns:
            (x_value, y_value) in graph coordinates
        """
        # Calculate pixel ranges
        x_pixel_range = self.x_max_pixel[0] - self.origin_pixel[0]
        y_pixel_range = self.origin_pixel[1] - self.y_max_pixel[1]  # Note: Y is inverted in images

        # Handle division by zero
        if x_pixel_range == 0 or y_pixel_range == 0:
            return (0, 0)

        # Calculate scaling factors
        x_scale = self.x_max_value / x_pixel_range
        y_scale = self.y_max_value / y_pixel_range

        # Convert coordinates
        x_value = (pixel_x - self.origin_pixel[0]) * x_scale
        y_value = (self.origin_pixel[1] - pixel_y) * y_scale

        return (x_value, y_value)

    def coord_to_pixel(self, x_value: float, y_value: float) -> tuple:
        """Convert graph coordinates to pixel coordinates.

        Args:
            x_value: X value in graph coordinates
            y_value: Y value in graph coordinates

        Returns:
            (pixel_x, pixel_y)
        """
        # Calculate pixel ranges
        x_pixel_range = self.x_max_pixel[0] - self.origin_pixel[0]
        y_pixel_range = self.origin_pixel[1] - self.y_max_pixel[1]

        # Handle division by zero
        if self.x_max_value == 0 or self.y_max_value == 0:
            return self.origin_pixel

        # Calculate scaling factors
        x_scale = x_pixel_range / self.x_max_value
        y_scale = y_pixel_range / self.y_max_value

        # Convert coordinates
        pixel_x = int(self.origin_pixel[0] + x_value * x_scale)
        pixel_y = int(self.origin_pixel[1] - y_value * y_scale)

        return (pixel_x, pixel_y)


class CurveTracer:
    """Traces Kaplan-Meier step curves from images."""

    def __init__(self, image: Image.Image | np.ndarray):
        """Initialize curve tracer.

        Args:
            image: Input image (PIL Image or OpenCV array)
        """
        if isinstance(image, Image.Image):
            self.image = pil_to_cv2(image)
        else:
            self.image = image.copy()

        self.gray = None
        self.edges = None
        self.lines = []
        self.horizontal_lines = []
        self.vertical_lines = []
        self.step_curve_points = []

    def preprocess(self):
        """Preprocess image for line detection."""
        self.gray = preprocess_for_curve_detection(self.image)
        self.edges = detect_edges(
            self.gray,
            config.CANNY_THRESHOLD1,
            config.CANNY_THRESHOLD2
        )

    def detect_lines(self) -> list[LineSegment]:
        """Detect line segments using Probabilistic Hough Transform.

        Returns:
            List of detected LineSegment objects
        """
        if self.edges is None:
            self.preprocess()

        # Apply Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            self.edges,
            rho=1,
            theta=np.pi / 180,
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=config.HOUGH_MAX_LINE_GAP
        )

        self.lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                segment = LineSegment(x1, y1, x2, y2)
                self.lines.append(segment)

                # Categorize lines
                if segment.is_horizontal:
                    self.horizontal_lines.append(segment)
                elif segment.is_vertical:
                    self.vertical_lines.append(segment)

        return self.lines

    def find_step_curve(self, roi: tuple = None) -> list[tuple]:
        """Find step curve pattern (alternating H-V-H-V).

        Args:
            roi: Optional region of interest (x, y, width, height)

        Returns:
            List of (x, y) points forming the step curve
        """
        if not self.lines:
            self.detect_lines()

        # Filter lines by ROI if specified
        h_lines = self.horizontal_lines
        v_lines = self.vertical_lines

        if roi:
            x, y, w, h = roi
            h_lines = [l for l in h_lines if self._line_in_roi(l, roi)]
            v_lines = [l for l in v_lines if self._line_in_roi(l, roi)]

        # Sort horizontal lines by Y coordinate (top to bottom)
        h_lines = sorted(h_lines, key=lambda l: min(l.y1, l.y2))

        # Sort vertical lines by X coordinate (left to right)
        v_lines = sorted(v_lines, key=lambda l: min(l.x1, l.x2))

        # Build step curve by connecting segments
        self.step_curve_points = self._connect_step_segments(h_lines, v_lines)

        return self.step_curve_points

    def _line_in_roi(self, line: LineSegment, roi: tuple) -> bool:
        """Check if line is within ROI."""
        x, y, w, h = roi
        mid_x = (line.x1 + line.x2) / 2
        mid_y = (line.y1 + line.y2) / 2
        return x <= mid_x <= x + w and y <= mid_y <= y + h

    def _connect_step_segments(
        self, h_lines: list[LineSegment], v_lines: list[LineSegment]
    ) -> list[tuple]:
        """Connect horizontal and vertical segments to form step curve.

        Args:
            h_lines: List of horizontal line segments
            v_lines: List of vertical line segments

        Returns:
            List of (x, y) points forming the curve
        """
        points = []
        tolerance = config.ENDPOINT_TOLERANCE

        # Simple approach: sample all detected horizontal segments
        # and create points at their endpoints
        for h_line in h_lines:
            start = h_line.start_point()
            end = h_line.end_point()

            # Add both endpoints
            points.append(start)
            points.append(end)

        # Remove duplicates and sort by X
        unique_points = list(set(points))
        unique_points.sort(key=lambda p: (p[0], p[1]))

        return unique_points

    def sample_curve_at_intervals(
        self, calibration: CalibrationData, interval: float = None
    ) -> list[tuple]:
        """Sample the curve at regular X intervals.

        Args:
            calibration: Calibration data for coordinate transformation
            interval: Sampling interval in coordinate units (default: from config)

        Returns:
            List of (time, survival) data points
        """
        if interval is None:
            # Use pixel interval and convert
            pixel_interval = config.SAMPLING_INTERVAL
        else:
            # Convert coordinate interval to pixels
            pixel_interval = int(
                interval * (calibration.x_max_pixel[0] - calibration.origin_pixel[0])
                / calibration.x_max_value
            )

        # Sample points along X axis
        data_points = []
        x_start = calibration.origin_pixel[0]
        x_end = calibration.x_max_pixel[0]

        for pixel_x in range(x_start, x_end + 1, max(1, pixel_interval)):
            # Find Y value at this X position
            pixel_y = self._get_y_at_x(pixel_x)

            if pixel_y is not None:
                # Convert to coordinates
                x_val, y_val = calibration.pixel_to_coord(pixel_x, pixel_y)
                data_points.append((x_val, y_val))

        return data_points

    def _get_y_at_x(self, target_x: int) -> Optional[int]:
        """Get Y value of curve at given X position.

        Uses nearest detected curve point.

        Args:
            target_x: X pixel coordinate

        Returns:
            Y pixel coordinate or None if not found
        """
        if not self.step_curve_points:
            return None

        # Find closest point
        min_dist = float('inf')
        closest_y = None

        for x, y in self.step_curve_points:
            dist = abs(x - target_x)
            if dist < min_dist:
                min_dist = dist
                closest_y = y

        return closest_y

    def trace_curve_by_color(
        self,
        color_range: tuple = None,
        calibration: CalibrationData = None
    ) -> list[tuple]:
        """Trace curve by detecting specific color.

        Args:
            color_range: ((lower_bgr), (upper_bgr)) color bounds
            calibration: Optional calibration data

        Returns:
            List of (x, y) points
        """
        if color_range is None:
            # Default to detecting dark colors (black curves)
            lower = np.array([0, 0, 0])
            upper = np.array([100, 100, 100])
        else:
            lower, upper = color_range
            lower = np.array(lower)
            upper = np.array(upper)

        # Create mask
        mask = cv2.inRange(self.image, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract points from largest contour
        if contours:
            largest = max(contours, key=cv2.contourArea)
            points = [(pt[0][0], pt[0][1]) for pt in largest]
            points.sort(key=lambda p: p[0])
            return points

        return []

    def get_visualization(self) -> np.ndarray:
        """Get visualization of detected lines and curve.

        Returns:
            Image with detections drawn
        """
        vis = self.image.copy()

        # Draw all detected lines
        for line in self.horizontal_lines:
            cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 1)

        for line in self.vertical_lines:
            cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 1)

        # Draw step curve points
        for i, point in enumerate(self.step_curve_points):
            cv2.circle(vis, point, 3, (0, 0, 255), -1)
            if i > 0:
                cv2.line(vis, self.step_curve_points[i-1], point, (0, 0, 255), 2)

        return vis


def trace_km_curve(
    image: Image.Image | np.ndarray,
    calibration: CalibrationData,
    roi: tuple = None
) -> list[tuple]:
    """Convenience function to trace KM curve from image.

    Args:
        image: Input image
        calibration: Calibration data for coordinate transformation
        roi: Optional region of interest (x, y, width, height)

    Returns:
        List of (time, survival) data points
    """
    tracer = CurveTracer(image)
    tracer.detect_lines()
    tracer.find_step_curve(roi)

    return tracer.sample_curve_at_intervals(calibration)
