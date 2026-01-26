"""Axis calibration for KM curve extraction - Version 2.

This module provides a clean, two-step calibration process:
1. Axis Detection & Calibration: Find axis lines and labels, calculate pixel-to-data mapping
2. Plot Rectangle: Define the extraction area using calibrated coordinates

No AI in the core calibration path - AI is optional for post-extraction validation only.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2
import numpy as np
import re


@dataclass
class AxisCalibration:
    """Complete calibration result for coordinate mapping.

    All coordinates are in pixels. The calibration defines:
    - Where (0, 0) is in pixel space (origin)
    - Where (time_max, survival_max) is in pixel space
    - The conversion factors between pixels and data units
    """
    # Pixel coordinates of key points
    x_0_pixel: int          # X pixel where time = 0
    x_max_pixel: int        # X pixel where time = time_max
    y_0_pixel: int          # Y pixel where survival = 0 (bottom)
    y_100_pixel: int        # Y pixel where survival = 1.0/100% (top)

    # Data ranges
    time_min: float         # Usually 0
    time_max: float         # e.g., 24 months
    survival_min: float     # Usually 0
    survival_max: float     # Usually 1.0 (or 100 if percentage)

    # Conversion factors (data units per pixel)
    time_per_pixel: float
    survival_per_pixel: float

    # Plot rectangle (extraction area) - derived from axis coordinates
    plot_x: int             # = x_0_pixel
    plot_y: int             # = y_100_pixel
    plot_width: int         # = x_max_pixel - x_0_pixel
    plot_height: int        # = y_0_pixel - y_100_pixel

    # Detected labels for debugging
    detected_x_labels: Optional[List[Tuple[float, int]]] = None
    detected_y_labels: Optional[List[Tuple[float, int]]] = None

    @property
    def origin(self) -> Tuple[int, int]:
        """Pixel coordinates of (time=0, survival=0)."""
        return (self.x_0_pixel, self.y_0_pixel)

    @property
    def plot_rectangle(self) -> Tuple[int, int, int, int]:
        """Plot area as (x, y, width, height)."""
        return (self.plot_x, self.plot_y, self.plot_width, self.plot_height)

    # Compatibility properties for existing code
    @property
    def x_data_range(self) -> Tuple[float, float]:
        """Data range for X-axis (time)."""
        return (self.time_min, self.time_max)

    @x_data_range.setter
    def x_data_range(self, value: Tuple[float, float]):
        """Set X-axis data range and recalculate conversion factor."""
        self.time_min, self.time_max = value
        x_pixel_range = self.x_max_pixel - self.x_0_pixel
        if x_pixel_range > 0:
            self.time_per_pixel = (self.time_max - self.time_min) / x_pixel_range

    @property
    def y_data_range(self) -> Tuple[float, float]:
        """Data range for Y-axis (survival)."""
        return (self.survival_min, self.survival_max)

    @y_data_range.setter
    def y_data_range(self, value: Tuple[float, float]):
        """Set Y-axis data range and recalculate conversion factor."""
        self.survival_min, self.survival_max = value
        y_pixel_range = self.y_0_pixel - self.y_100_pixel
        if y_pixel_range > 0:
            self.survival_per_pixel = (self.survival_max - self.survival_min) / y_pixel_range

    @property
    def x_axis_end(self) -> Tuple[int, int]:
        """Pixel coordinates of X-axis end point (for backward compatibility)."""
        return (self.x_max_pixel, self.y_0_pixel)

    @x_axis_end.setter
    def x_axis_end(self, value: Tuple[int, int]):
        """Set X-axis end pixel position and recalculate conversion factor."""
        self.x_max_pixel = value[0]
        x_pixel_range = self.x_max_pixel - self.x_0_pixel
        if x_pixel_range > 0:
            self.time_per_pixel = (self.time_max - self.time_min) / x_pixel_range
        # Update plot width
        self.plot_width = self.x_max_pixel - self.x_0_pixel

    @property
    def y_axis_end(self) -> Tuple[int, int]:
        """Pixel coordinates of Y-axis end point (top, where survival=100%) (for backward compatibility)."""
        return (self.x_0_pixel, self.y_100_pixel)

    def pixel_to_time(self, pixel_x: int) -> float:
        """Convert X pixel to time value."""
        return self.time_min + (pixel_x - self.x_0_pixel) * self.time_per_pixel

    def pixel_to_survival(self, pixel_y: int) -> float:
        """Convert Y pixel to survival value (normalized to 0-1)."""
        # Y pixels increase downward, survival increases upward
        raw_survival = self.survival_min + (self.y_0_pixel - pixel_y) * self.survival_per_pixel
        # Normalize to 0-1 if using percentage scale
        if self.survival_max > 1.5:
            return raw_survival / self.survival_max
        return raw_survival

    def pixel_to_coord(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert pixel coordinates to (time, survival) data coordinates."""
        return (self.pixel_to_time(pixel_x), self.pixel_to_survival(pixel_y))

    def time_to_pixel(self, time: float) -> int:
        """Convert time value to X pixel."""
        return int(self.x_0_pixel + (time - self.time_min) / self.time_per_pixel)

    def survival_to_pixel(self, survival: float) -> int:
        """Convert survival value (0-1) to Y pixel."""
        # Denormalize if using percentage scale
        if self.survival_max > 1.5:
            survival = survival * self.survival_max
        return int(self.y_0_pixel - (survival - self.survival_min) / self.survival_per_pixel)

    def coord_to_pixel(self, time: float, survival: float) -> Tuple[int, int]:
        """Convert (time, survival) data coordinates to pixel coordinates."""
        return (self.time_to_pixel(time), self.survival_to_pixel(survival))


class AxisCalibrator:
    """Two-step axis calibration for KM plots.

    Step 1: Detect axis lines and labels
    Step 2: Calculate calibration and define plot rectangle

    Usage:
        calibrator = AxisCalibrator(image)
        calibration = calibrator.calibrate(time_max=24)

        # Convert coordinates
        time, survival = calibration.pixel_to_coord(pixel_x, pixel_y)

        # Get plot area for extraction
        x, y, w, h = calibration.plot_rectangle
    """

    def __init__(self, image: np.ndarray):
        """Initialize with a BGR image."""
        self.image = image
        self.height, self.width = image.shape[:2]
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detected axis lines (set during calibration)
        self._x_axis_y: Optional[int] = None  # Y pixel of X-axis line
        self._y_axis_x: Optional[int] = None  # X pixel of Y-axis line
        self._x_axis_start: Optional[int] = None
        self._x_axis_end: Optional[int] = None
        self._y_axis_top: Optional[int] = None
        self._y_axis_bottom: Optional[int] = None

    # =========================================================================
    # STEP 1: Axis Line Detection
    # =========================================================================

    def detect_axis_lines(self) -> bool:
        """Detect X and Y axis lines using Hough transform.

        Returns:
            True if both axes detected, False otherwise
        """
        # Edge detection
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=50, minLineLength=30, maxLineGap=20
        )

        if lines is None:
            return False

        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Classify lines
            if abs(angle) < 10 and length > self.width * 0.3:
                # Horizontal line (potential X-axis)
                y_mid = (y1 + y2) // 2
                if y_mid > self.height * 0.4:  # Bottom half
                    horizontal_lines.append((y_mid, min(x1, x2), max(x1, x2), length))
            elif abs(abs(angle) - 90) < 10 and length > self.height * 0.3:
                # Vertical line (potential Y-axis)
                x_mid = (x1 + x2) // 2
                if x_mid < self.width * 0.4:  # Left side
                    vertical_lines.append((x_mid, min(y1, y2), max(y1, y2), length))

        if not horizontal_lines or not vertical_lines:
            return False

        # Select best X-axis (longest in bottom region)
        x_axis = max(horizontal_lines, key=lambda l: l[3])
        self._x_axis_y = x_axis[0]
        self._x_axis_start = x_axis[1]
        self._x_axis_end = x_axis[2]

        # Select best Y-axis (longest in left region)
        y_axis = max(vertical_lines, key=lambda l: l[3])
        self._y_axis_x = y_axis[0]
        self._y_axis_top = y_axis[1]
        self._y_axis_bottom = y_axis[2]

        return True

    # =========================================================================
    # STEP 1: OCR Label Detection
    # =========================================================================

    def _get_ocr_reader(self):
        """Get OCR reader (lazy loading)."""
        if not hasattr(self, '_ocr_reader'):
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

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a numeric value from OCR text."""
        text = text.strip().replace('%', '').replace(',', '').replace(' ', '')

        # Handle common OCR errors for short text
        if len(text) <= 3:
            text = text.replace('O', '0').replace('o', '0')
            text = text.replace('l', '1').replace('I', '1')

        # Handle ".X" patterns
        if text.startswith('.') and len(text) >= 2:
            text = '0' + text

        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    def _ocr_region(self, region: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """Run OCR on a region, return list of (text, bbox, confidence)."""
        reader, ocr_type = self._get_ocr_reader()

        if reader is None:
            return []

        results = []

        if ocr_type == 'easyocr':
            ocr_results = reader.readtext(region)
            for bbox_points, text, conf in ocr_results:
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - x), int(max(ys) - y)
                results.append((text, (x, y, w, h), conf))

        elif ocr_type == 'pytesseract':
            import pytesseract
            from PIL import Image

            if len(region.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(region)

            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            for i, text in enumerate(data['text']):
                if text.strip():
                    conf = data['conf'][i] / 100.0 if data['conf'][i] != -1 else 0.5
                    results.append((text.strip(),
                                   (data['left'][i], data['top'][i], data['width'][i], data['height'][i]),
                                   conf))

        return results

    def detect_x_labels(self) -> List[Tuple[float, int]]:
        """Detect X-axis labels (time values).

        Returns:
            List of (value, x_pixel) tuples sorted by x position
        """
        if self._x_axis_y is None:
            return []

        # Region below X-axis (skip tick marks)
        y_start = self._x_axis_y + 8
        y_end = min(self.height, self._x_axis_y + 40)
        x_start = max(0, self._y_axis_x - 30) if self._y_axis_x else 0
        x_end = min(self.width, self._x_axis_end + 50) if self._x_axis_end else self.width

        if y_end <= y_start or x_end <= x_start:
            return []

        region = self.image[y_start:y_end, x_start:x_end]
        scaled = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Try OCR
        ocr_results = self._ocr_region(scaled)

        # Try pytesseract with sparse text mode as fallback
        if not ocr_results or all(self._parse_number(t) is None for t, _, _ in ocr_results):
            try:
                import pytesseract
                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if len(scaled.shape) == 3 else scaled
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT,
                                                 config='--psm 11 -c tessedit_char_whitelist=0123456789')
                ocr_results = []
                for i, text in enumerate(data['text']):
                    if text.strip() and data['conf'][i] > 20:
                        value = self._parse_number(text.strip())
                        if value is not None and 0 <= value <= 200:
                            x_pos = data['left'][i] + data['width'][i] // 2
                            ocr_results.append((text.strip(), (x_pos, 0, data['width'][i], data['height'][i]),
                                               data['conf'][i] / 100.0))
            except ImportError:
                pass

        # Parse results
        labels = []
        for text, (x, y, w, h), conf in ocr_results:
            value = self._parse_number(text)
            if value is not None and conf > 0.2 and 0 <= value <= 200:
                x_center = x_start + (x + w // 2) // 3  # Account for 3x scaling
                labels.append((value, x_center))

        labels.sort(key=lambda item: item[1])
        return labels

    def detect_y_labels(self) -> List[Tuple[float, int]]:
        """Detect Y-axis labels (survival values).

        Returns:
            List of (value, y_pixel) tuples sorted by y position (top to bottom)
        """
        if self._y_axis_x is None:
            return []

        # Region left of Y-axis
        x_start = max(0, self._y_axis_x - 60)
        x_end = self._y_axis_x + 5
        y_start = max(0, self._y_axis_top - 10) if self._y_axis_top else 0
        y_end = min(self.height, self._y_axis_bottom + 10) if self._y_axis_bottom else self.height

        if x_end <= x_start or y_end <= y_start:
            return []

        region = self.image[y_start:y_end, x_start:x_end]
        scaled = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Try OCR
        ocr_results = self._ocr_region(scaled)

        # Try pytesseract as fallback
        if not ocr_results or all(self._parse_number(t) is None for t, _, _ in ocr_results):
            try:
                import pytesseract
                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if len(scaled.shape) == 3 else scaled
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT,
                                                 config='--psm 6 -c tessedit_char_whitelist=0123456789.')
                ocr_results = []
                for i, text in enumerate(data['text']):
                    if text.strip() and data['conf'][i] > 0:
                        value = self._parse_number(text.strip())
                        if value is not None and (0 <= value <= 1.5 or 0 <= value <= 150):
                            y_pos = data['top'][i] + data['height'][i] // 2
                            ocr_results.append((text.strip(), (0, y_pos, data['width'][i], data['height'][i]),
                                               data['conf'][i] / 100.0))
            except ImportError:
                pass

        # Parse results
        labels = []
        for text, (x, y, w, h), conf in ocr_results:
            value = self._parse_number(text)
            if value is not None and conf > 0.2:
                if 0 <= value <= 1.5 or 0 <= value <= 150:
                    y_center = y_start + (y + h // 2) // 3  # Account for 3x scaling
                    labels.append((value, y_center))

        labels.sort(key=lambda item: item[1])
        return labels

    # =========================================================================
    # STEP 1: OCR Error Correction
    # =========================================================================

    def correct_y_labels(self, labels: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """Correct common OCR errors in Y-axis labels using spacing-based inference.

        Common errors: "75" read as "5", "25" read as "5", "100" read as "0"

        This method uses the spacing between detected labels to infer correct values.
        The key insight is that the BOTTOMMOST 0.0 label is the actual zero (survival=0),
        while 0.0 labels near the top are OCR misreads of "100".
        """
        if len(labels) < 2:
            return labels

        # Sort by y position (top to bottom)
        sorted_labels = sorted(labels, key=lambda x: x[1])

        # Find the BOTTOMMOST 0.0 label - this is the actual zero (survival=0)
        # Labels near the top with 0.0 are likely OCR misreads of "100"
        zero_y = None
        zero_candidates = [(v, y) for v, y in sorted_labels if abs(v) < 0.1]
        if zero_candidates:
            # Use the bottommost (highest y value) as actual zero
            zero_y = max(y for _, y in zero_candidates)

        if zero_y is None:
            return labels

        # Calculate spacing between all labels to estimate the grid
        y_positions = sorted(set(y for _, y in sorted_labels))
        if len(y_positions) < 2:
            return labels

        spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions) - 1)]

        # Filter out very small spacings (noise) and very large ones (gaps)
        median_spacing = sorted(spacings)[len(spacings) // 2] if spacings else 100
        valid_spacings = [s for s in spacings if 0.5 * median_spacing < s < 2 * median_spacing]

        if not valid_spacings:
            valid_spacings = spacings

        avg_spacing = sum(valid_spacings) / len(valid_spacings) if valid_spacings else 100

        # Check if we need correction:
        # 1. Multiple 0.0 values (100 misread as 0)
        # 2. Duplicate small values like 5 (75/25 misread as 5)
        values = [v for v, _ in sorted_labels]
        zero_count = sum(1 for v in values if abs(v) < 0.1)
        small_value_counts = {}
        for v in values:
            if 2 <= v <= 9:
                small_value_counts[v] = small_value_counts.get(v, 0) + 1

        has_duplicate_zeros = zero_count > 1
        has_duplicate_smalls = any(c > 1 for c in small_value_counts.values())

        if not has_duplicate_zeros and not has_duplicate_smalls:
            # No obvious OCR errors detected
            return labels

        # Correct labels based on distance from the actual zero (bottom)
        corrected = []
        for orig_v, orig_y in sorted_labels:
            # Distance from zero: positive = above zero line (higher survival)
            distance_from_zero = zero_y - orig_y

            # Calculate number of grid steps above zero
            steps_from_zero = round(distance_from_zero / avg_spacing)

            # Infer value (assuming 25% increments which is common: 0, 25, 50, 75, 100)
            inferred_value = steps_from_zero * 25.0
            inferred_value = max(0, min(100, inferred_value))

            corrected.append((inferred_value, orig_y))

        corrected.sort(key=lambda x: x[1])
        return corrected

    # =========================================================================
    # STEP 1: Find Key Pixel Positions
    # =========================================================================

    def find_x_0_pixel(self, x_labels: List[Tuple[float, int]]) -> int:
        """Find X pixel where time = 0."""
        # Look for label with value 0
        for value, x_pixel in x_labels:
            if abs(value) < 0.1:
                return x_pixel

        # Extrapolate from other labels
        if len(x_labels) >= 2:
            sorted_labels = sorted(x_labels, key=lambda x: x[0])
            val1, x1 = sorted_labels[0]
            val2, x2 = sorted_labels[1]
            if val1 > 0 and val2 != val1:
                pixels_per_unit = (x2 - x1) / (val2 - val1)
                x0 = x1 - val1 * pixels_per_unit
                if x0 >= 0:
                    return int(x0)

        # Fallback to Y-axis position
        return self._y_axis_x if self._y_axis_x else int(self.width * 0.1)

    def find_x_max_pixel(self, x_labels: List[Tuple[float, int]], time_max: float, x_0_pixel: int) -> Tuple[int, float]:
        """Find X pixel where time = time_max and the detected max value.

        Args:
            x_labels: Detected X-axis labels
            time_max: Expected maximum time value
            x_0_pixel: X pixel where time = 0 (for extrapolation)

        Returns:
            (x_pixel, detected_time_max)
        """
        # If we have multiple labels with good coverage, use them
        if len(x_labels) >= 2:
            sorted_labels = sorted(x_labels, key=lambda x: x[0])
            rightmost = max(x_labels, key=lambda x: x[1])

            # Check if we have the exact time_max label
            for value, x_pixel in x_labels:
                if abs(value - time_max) < 0.5:
                    return (x_pixel, time_max)

            # Extrapolate to time_max if rightmost label is less than time_max
            if len(x_labels) >= 2 and rightmost[0] < time_max:
                # Calculate pixels per time unit from two labels
                val1, x1 = sorted_labels[0]
                val2, x2 = sorted_labels[-1]
                if val2 != val1:
                    pixels_per_unit = (x2 - x1) / (val2 - val1)
                    x_max = x1 + (time_max - val1) * pixels_per_unit
                    return (int(x_max), time_max)

            return (rightmost[1], rightmost[0])

        # With only one label or no labels, use the X-axis line extent
        # This is more reliable than using a single possibly-wrong label
        x_axis_end = self._x_axis_end if self._x_axis_end else int(self.width * 0.9)

        # If we have one label, try to extrapolate
        if len(x_labels) == 1:
            val, x = x_labels[0]
            if val > 0 and x > x_0_pixel:
                # Calculate pixels per unit from this label and x=0
                pixels_per_unit = (x - x_0_pixel) / val
                x_max = x_0_pixel + time_max * pixels_per_unit
                # Only use extrapolation if it covers most of the axis extent (>70%)
                # If much smaller, the single label is probably wrong/misread
                if x_max >= x_axis_end * 0.7:
                    return (int(x_max), time_max)

        # Fallback to X-axis end - use the detected line extent
        return (x_axis_end, time_max)

    def find_y_0_pixel(self, y_labels: List[Tuple[float, int]]) -> int:
        """Find Y pixel where survival = 0 (bottom of plot)."""
        for value, y_pixel in y_labels:
            if abs(value) < 0.1:
                return y_pixel

        # Fallback to X-axis position
        return self._x_axis_y if self._x_axis_y else int(self.height * 0.8)

    def find_y_100_pixel(self, y_labels: List[Tuple[float, int]]) -> int:
        """Find Y pixel where survival = 100% (top of plot)."""
        # Look for 100 or 1.0 label
        for value, y_pixel in y_labels:
            if abs(value - 100) < 1 or abs(value - 1.0) < 0.05:
                return y_pixel

        # Extrapolate from corrected labels
        if len(y_labels) >= 2:
            # Find 0 position
            zero_y = None
            for v, y in y_labels:
                if abs(v) < 0.1:
                    zero_y = y
                    break

            if zero_y is not None:
                # Get highest detected value
                sorted_by_value = sorted(y_labels, key=lambda x: x[0], reverse=True)
                highest_val, highest_y = sorted_by_value[0]

                if highest_val > 0:
                    pixels_per_percent = (zero_y - highest_y) / highest_val
                    y_100 = zero_y - (100 * pixels_per_percent)
                    if y_100 >= 0:
                        return int(y_100)

        # Fallback to Y-axis top
        return self._y_axis_top if self._y_axis_top else int(self.height * 0.1)

    # =========================================================================
    # STEP 2: Main Calibration
    # =========================================================================

    def calibrate(self, time_max: float, verbose: bool = False) -> Optional[AxisCalibration]:
        """Perform complete axis calibration.

        Args:
            time_max: Maximum time value on X-axis (required)
            verbose: Print calibration details

        Returns:
            AxisCalibration object, or None if calibration fails
        """
        if verbose:
            print("=" * 60)
            print("AXIS CALIBRATION")
            print("=" * 60)

        # Step 1a: Detect axis lines
        if verbose:
            print("\n[Step 1a] Detecting axis lines...")

        if not self.detect_axis_lines():
            if verbose:
                print("  WARNING: Axis line detection failed, using estimates")
            self._y_axis_x = int(self.width * 0.1)
            self._x_axis_y = int(self.height * 0.8)
            self._y_axis_top = int(self.height * 0.1)
            self._y_axis_bottom = self._x_axis_y
            self._x_axis_start = self._y_axis_x
            self._x_axis_end = int(self.width * 0.9)

        if verbose:
            print(f"  X-axis line at y={self._x_axis_y}")
            print(f"  Y-axis line at x={self._y_axis_x}")

        # Step 1b: Detect axis labels via OCR
        if verbose:
            print("\n[Step 1b] Detecting axis labels (OCR)...")

        x_labels_raw = self.detect_x_labels()
        y_labels_raw = self.detect_y_labels()

        if verbose:
            print(f"  X-axis labels (raw): {x_labels_raw}")
            print(f"  Y-axis labels (raw): {y_labels_raw}")

        # Step 1c: Correct OCR errors
        if verbose:
            print("\n[Step 1c] Correcting OCR errors...")

        y_labels = self.correct_y_labels(y_labels_raw)
        x_labels = x_labels_raw  # X labels usually don't need correction

        if y_labels != y_labels_raw and verbose:
            print(f"  Y-axis labels (corrected): {y_labels}")

        # Step 1d: Find key pixel positions
        if verbose:
            print("\n[Step 1d] Finding key pixel positions...")

        x_0_pixel = self.find_x_0_pixel(x_labels)
        x_max_pixel, detected_time_max = self.find_x_max_pixel(x_labels, time_max, x_0_pixel)
        y_0_pixel = self.find_y_0_pixel(y_labels)
        y_100_pixel = self.find_y_100_pixel(y_labels)

        if verbose:
            print(f"  X=0 at pixel {x_0_pixel}")
            print(f"  X={time_max} at pixel {x_max_pixel} (detected max: {detected_time_max})")
            print(f"  Y=0% at pixel {y_0_pixel}")
            print(f"  Y=100% at pixel {y_100_pixel}")

        # Determine Y scale (0-1 or 0-100)
        survival_max = 1.0
        if y_labels:
            max_y_value = max(v for v, _ in y_labels)
            if max_y_value > 1.5:
                survival_max = 100.0

        # Step 2: Calculate calibration
        if verbose:
            print("\n[Step 2] Calculating calibration...")

        # Calculate conversion factors
        x_pixel_range = x_max_pixel - x_0_pixel
        y_pixel_range = y_0_pixel - y_100_pixel  # Note: y increases downward

        if x_pixel_range <= 0 or y_pixel_range <= 0:
            if verbose:
                print("  ERROR: Invalid pixel ranges")
            return None

        time_per_pixel = time_max / x_pixel_range
        survival_per_pixel = survival_max / y_pixel_range

        # Plot rectangle is directly from calibrated coordinates
        plot_x = x_0_pixel
        plot_y = y_100_pixel
        plot_width = x_max_pixel - x_0_pixel
        plot_height = y_0_pixel - y_100_pixel

        if verbose:
            print(f"  Time range: 0 - {time_max}")
            print(f"  Survival range: 0 - {survival_max}")
            print(f"  Time per pixel: {time_per_pixel:.6f}")
            print(f"  Survival per pixel: {survival_per_pixel:.6f}")
            print(f"  Plot rectangle: x={plot_x}, y={plot_y}, w={plot_width}, h={plot_height}")

        # Create calibration result
        calibration = AxisCalibration(
            x_0_pixel=x_0_pixel,
            x_max_pixel=x_max_pixel,
            y_0_pixel=y_0_pixel,
            y_100_pixel=y_100_pixel,
            time_min=0.0,
            time_max=time_max,
            survival_min=0.0,
            survival_max=survival_max,
            time_per_pixel=time_per_pixel,
            survival_per_pixel=survival_per_pixel,
            plot_x=plot_x,
            plot_y=plot_y,
            plot_width=plot_width,
            plot_height=plot_height,
            detected_x_labels=x_labels_raw,
            detected_y_labels=y_labels
        )

        # Verification
        if verbose:
            print("\n[Verification]")
            t, s = calibration.pixel_to_coord(x_0_pixel, y_0_pixel)
            print(f"  Origin ({x_0_pixel}, {y_0_pixel}) -> ({t:.2f}, {s:.2f}) [expected: (0, 0)]")
            t, s = calibration.pixel_to_coord(x_0_pixel, y_100_pixel)
            print(f"  Top-left ({x_0_pixel}, {y_100_pixel}) -> ({t:.2f}, {s:.2f}) [expected: (0, 1.0)]")
            t, s = calibration.pixel_to_coord(x_max_pixel, y_0_pixel)
            print(f"  Bottom-right ({x_max_pixel}, {y_0_pixel}) -> ({t:.2f}, {s:.2f}) [expected: ({time_max}, 0)]")

        return calibration


# =============================================================================
# Convenience function
# =============================================================================

def calibrate_image(image: np.ndarray, time_max: float, verbose: bool = False) -> Optional[AxisCalibration]:
    """Convenience function to calibrate an image.

    Args:
        image: BGR image
        time_max: Maximum time value on X-axis
        verbose: Print calibration details

    Returns:
        AxisCalibration object, or None if calibration fails
    """
    calibrator = AxisCalibrator(image)
    return calibrator.calibrate(time_max, verbose=verbose)
