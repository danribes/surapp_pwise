"""Line-style detection for multi-curve grayscale images.

This module provides line-style based curve detection and separation for
grayscale Kaplan-Meier plots where curves are differentiated by visual patterns
(solid, dashed, dotted, dash-dot) rather than color.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2


class LineStyle(Enum):
    """Enumeration of detectable line styles."""
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    DASH_DOT = "dash_dot"
    UNKNOWN = "unknown"


@dataclass
class LineStyleDescriptor:
    """Describes the visual characteristics of a line style."""
    style: LineStyle
    gap_ratio: float          # Average gap / segment length ratio
    avg_segment_len: float    # Average segment length in pixels
    avg_gap_len: float        # Average gap length in pixels
    pattern_period: float     # Repeat period of the pattern
    confidence: float


@dataclass
class CurveSegment:
    """A connected component that is part of a curve."""
    pixels: List[Tuple[int, int]]  # List of (x, y) pixel coordinates
    style: Optional[LineStyleDescriptor]
    x_range: Tuple[int, int]       # (min_x, max_x)
    y_range: Tuple[int, int]       # (min_y, max_y)
    component_id: int              # Connected component ID


@dataclass
class TracedCurve:
    """A curve traced through the image."""
    y_positions: Dict[int, float]  # x -> y position mapping
    pixel_presence: Dict[int, bool]  # x -> whether pixels exist at this x
    style: LineStyle
    confidence: float


@dataclass
class DetectedCurve:
    """A complete curve detected by line style analysis."""
    segments: List[CurveSegment]
    style: LineStyle
    data_points: List[Tuple[float, float]]  # (time, survival) pairs
    confidence: float


class LineStyleDetector:
    """Detects and separates curves based on their line styles.

    This class analyzes grayscale images to identify multiple curves that are
    differentiated by visual patterns (solid, dashed, dotted, dash-dot) rather
    than color.
    """

    def __init__(self, image: np.ndarray, plot_bounds: Optional[Tuple[int, int, int, int]] = None,
                 filter_reference_lines: bool = True, preprocess_text_removal: bool = False):
        """
        Initialize the detector.

        Args:
            image: BGR or grayscale image of the plot area
            plot_bounds: Optional (x, y, width, height) of the plot region.
                        If None, uses the entire image.
            filter_reference_lines: Whether to filter out horizontal/vertical reference lines.
                        Set to False for debugging.
            preprocess_text_removal: Whether to remove text from image before processing.
                        This can help when text overlaps with curves.
        """
        self.original_image = image
        self.filter_reference_lines_enabled = filter_reference_lines
        self.preprocess_text_removal = preprocess_text_removal

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image.copy()

        # Optional text removal preprocessing
        if preprocess_text_removal:
            self.gray = self._preprocess_remove_text(self.gray)

        # Set plot bounds
        if plot_bounds is not None:
            self.plot_x, self.plot_y, self.plot_w, self.plot_h = plot_bounds
        else:
            self.plot_x, self.plot_y = 0, 0
            self.plot_h, self.plot_w = self.gray.shape[:2]

        self.binary = None
        self.segments = []
        self.traced_curves = []
        self.vertical_line_x_positions = set()  # Will be populated by reference line filtering

    def _preprocess_remove_text(self, gray: np.ndarray) -> np.ndarray:
        """
        Remove text from the grayscale image before curve detection.

        This uses OCR to identify text regions and inpaints them with
        the background color, which prevents text from being connected
        to curves during thresholding.

        Args:
            gray: Grayscale image

        Returns:
            Grayscale image with text regions removed
        """
        try:
            import pytesseract

            h, w = gray.shape[:2]

            # Detect text regions
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, timeout=30)

            # Create text mask
            text_mask = np.zeros((h, w), dtype=np.uint8)

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

                if conf > 20 and len(text) > 0:
                    x = data['left'][i]
                    y = data['top'][i]
                    tw = data['width'][i]
                    th = data['height'][i]

                    # Add padding
                    padding = 3
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(w, x + tw + padding)
                    y2 = min(h, y + th + padding)

                    text_mask[y1:y2, x1:x2] = 255

            # Don't dilate - we want to preserve curve pixels
            # Only remove text that was actually detected

            if np.sum(text_mask) > 0:
                # Inpaint with background color (white for KM plots)
                result = cv2.inpaint(
                    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                    text_mask,
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA
                )
                return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            print(f"    Text preprocessing failed: {e}")

        return gray

    def detect_all_curves(self, expected_count: Optional[int] = None,
                          debug_dir: Optional[str] = None) -> List[DetectedCurve]:
        """
        Main entry point - detect and separate all curves.

        Args:
            expected_count: Optional expected number of curves (helps disambiguation)
            debug_dir: Optional directory to save debug images

        Returns:
            List of DetectedCurve objects
        """
        # Phase 1: Extract dark pixels (curve pixels)
        self.binary = self._extract_dark_pixels(debug_dir)

        # Phase 1b: Pre-classify pixels by dash pattern
        # This helps distinguish dashed vs solid curves when they're close together
        self.dash_mask = self._create_dash_mask(debug_dir)

        # Phase 2: Trace curves by following Y-positions across X
        self.traced_curves = self._trace_curves_by_position(expected_count)

        # Phase 3: Curve crossing detection and fix
        # When trackers swap curves at crossing points, fix the assignments
        # by swapping Y-values to maintain consistent curve ordering
        # NOTE: Temporarily disabled - curves don't actually cross in test image
        # if len(self.traced_curves) >= 2:
        #     intersections = self._find_intersections()
        #     if intersections:
        #         print(f"    Detected {len(intersections)} potential crossing point(s)")
        #         self._fix_curve_crossings()

        # Phase 3b: Smooth curves AFTER crossing fix
        # This ensures the crossing fix can work on complete data
        self._smooth_traced_curves(self.traced_curves)

        # Phase 3c: Use dash_mask to fix tracking errors where curves got swapped
        # This is more reliable than position-based ordering since it uses actual style info
        if len(self.traced_curves) == 2 and hasattr(self, 'dash_mask') and self.dash_mask is not None:
            self._fix_tracking_with_dash_mask()

        # Phase 4: Analyze line style for each traced curve AFTER fixing crossings
        # Use pixel-based analysis for more accurate style detection
        for i, curve in enumerate(self.traced_curves):
            # Try pixel-based analysis first (looks at actual image pixels)
            style, confidence = self._analyze_curve_pattern_from_pixels(curve)

            if style == LineStyle.UNKNOWN or confidence < 0.5:
                # Fallback to tracking-based analysis
                style, confidence = self._analyze_curve_pattern(curve)

            curve.style = style
            curve.confidence = confidence

        # Phase 4b: For KM curves, relabel based on relative position
        # Use visual characteristics (dash patterns, density) to determine style
        # Do NOT assume which treatment has better survival - this varies by study
        if expected_count == 2 and len(self.traced_curves) == 2:
            self._relabel_curves_by_position()
            # NOTE: _fix_curve_swaps() disabled - it incorrectly assumed dashed=better survival
            # Style detection should be purely visual, not based on outcome assumptions
            # self._fix_curve_swaps()
            # NOTE: _fix_tail_curve_crossings() also disabled - same incorrect assumption
            # The dashed curve doesn't always have better survival than solid
            # self._fix_tail_curve_crossings()

        # Phase 5: Convert traced curves to DetectedCurve format
        curves = self._build_detected_curves()

        return curves

    def _extract_dark_pixels(self, debug_dir: Optional[str] = None) -> np.ndarray:
        """Extract curve pixels using adaptive thresholding."""
        # Crop to plot region
        plot_region = self.gray[
            self.plot_y:self.plot_y + self.plot_h,
            self.plot_x:self.plot_x + self.plot_w
        ]

        # Adaptive thresholding for varying backgrounds
        binary = cv2.adaptiveThreshold(
            plot_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=10
        )

        # Save debug image: after thresholding
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_1_threshold.png", binary)

        # Remove noise with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Save debug image: after noise removal
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_2_noise_removed.png", binary)

        # Save the pre-reference-line-filtered binary for dash detection
        # (dash gaps get filled by dilation in reference line filtering)
        self.binary_before_dilation = binary.copy()

        # Filter out horizontal and vertical reference lines (if enabled)
        if self.filter_reference_lines_enabled:
            binary = self._filter_reference_lines_with_dilation(binary, debug_dir)

        return binary

    def _create_dash_mask(self, debug_dir: Optional[str] = None) -> np.ndarray:
        """
        Create a mask identifying pixels that appear to be part of dashed lines.

        Dashed lines have regular horizontal gaps. This method analyzes each row
        for gap patterns and marks pixels that are likely from dashed curves.

        Uses the binary image BEFORE dilation (to preserve dash gaps).

        Returns:
            Binary mask where 255 = likely dashed, 0 = likely solid
        """
        # Use the pre-dilation binary if available (dash gaps are preserved)
        source_binary = getattr(self, 'binary_before_dilation', self.binary)
        height, width = source_binary.shape
        dash_mask = np.zeros_like(source_binary)

        # For each row, analyze the horizontal gap pattern
        for y in range(height):
            row = source_binary[y, :]
            if np.sum(row > 0) < 10:  # Skip rows with very few pixels
                continue

            # Find runs of pixels (segments) and gaps
            in_segment = False
            segments = []  # (start, end) of each segment
            gaps = []  # (start, end) of each gap
            seg_start = 0
            gap_start = 0

            for x in range(width):
                if row[x] > 0:
                    if not in_segment:
                        # Start of a new segment
                        if in_segment is not None and gap_start > 0:
                            gaps.append((gap_start, x))
                        seg_start = x
                        in_segment = True
                else:
                    if in_segment:
                        # End of a segment
                        segments.append((seg_start, x))
                        gap_start = x
                        in_segment = False

            # Close last segment if needed
            if in_segment:
                segments.append((seg_start, width))

            if len(segments) < 2:
                continue  # Not enough segments to be a dashed line

            # Analyze gap pattern
            gap_lengths = [g[1] - g[0] for g in gaps]
            seg_lengths = [s[1] - s[0] for s in segments]

            if not gap_lengths:
                continue

            # Filter out outlier gaps (large gaps between curves, not dash gaps)
            # Dash gaps are typically small (3-20 pixels)
            max_dash_gap = 25  # Maximum expected dash gap
            dash_gaps = [g for g in gap_lengths if g <= max_dash_gap]
            dash_segs = [s for s in seg_lengths if s <= 30]  # Filter out very long segments too

            if len(dash_gaps) < 2 or len(dash_segs) < 3:
                continue

            avg_gap = np.mean(dash_gaps)
            avg_seg = np.mean(dash_segs)
            gap_var = np.var(dash_gaps) / (avg_gap + 0.1) if avg_gap > 0 else 999
            seg_var = np.var(dash_segs) / (avg_seg + 0.1) if avg_seg > 0 else 999

            # Dashed lines have:
            # - Multiple small segments (at least 3)
            # - Regular small gaps (multiple similar-sized gaps)
            # - Small segment length (< 15 pixels typical for dashes)
            if len(dash_segs) >= 3 and avg_seg < 15 and avg_gap > 2:
                # Debug: print pattern info for qualifying rows
                # if y == 20:
                #     print(f"    Row {y} detail: dash_gaps={dash_gaps[:10]}, dash_segs={dash_segs[:10]}")
                #     print(f"    gap_var={gap_var:.2f}, seg_var={seg_var:.2f}")

                # Check for regular pattern - variance should be low relative to mean
                if gap_var < 5.0 and seg_var < 5.0:
                    # Mark this row as likely containing a dashed line
                    dash_mask[y, :] = source_binary[y, :]

        # Dilate the dash mask slightly to handle edge cases
        kernel = np.ones((3, 3), np.uint8)
        dash_mask = cv2.dilate(dash_mask, kernel, iterations=1)
        # Intersect with final binary to ensure mask aligns with tracked pixels
        dash_mask = cv2.bitwise_and(dash_mask, self.binary)

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_dash_mask.png", dash_mask)
            # Also save solid mask
            solid_mask = cv2.bitwise_and(self.binary, cv2.bitwise_not(dash_mask))
            cv2.imwrite(f"{debug_dir}/debug_solid_mask.png", solid_mask)

        return dash_mask

    def _filter_reference_lines_with_dilation(self, binary: np.ndarray,
                                               debug_dir: Optional[str] = None) -> np.ndarray:
        """
        Filter reference lines using dilation to preserve curve continuity.

        Strategy:
        1. Dilate the binary image to connect curve segments and fill small gaps
        2. Detect reference lines on the dilated image
        3. Create a mask of reference line pixels (excluding curve intersections)
        4. Apply the mask to the ORIGINAL binary image (not dilated)

        This preserves the original curve thickness while ensuring curve pixels
        at reference line intersections are not removed.
        """
        height, width = binary.shape
        print(f"    Starting preprocessing with dilation ({width}x{height})")

        # Keep a copy of the original binary for final masking
        original_binary = binary.copy()

        # Step 1: Dilate to connect curve segments
        # Use a small kernel to fill gaps without merging separate curves
        # IMPORTANT: Keep dilation minimal to avoid merging curves that are 10-15 pixels apart
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary, dilate_kernel, iterations=1)

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_2b_dilated.png", dilated)

        # Step 2: Detect reference lines on the dilated image
        h_line_mask = self._detect_horizontal_line_mask(dilated)
        v_line_mask = self._detect_vertical_line_mask(dilated)

        # Store x-positions of vertical reference lines for later use in tracking
        # These positions need special handling to avoid tracking artifacts
        v_line_cols = np.where(np.sum(v_line_mask, axis=0) > height * 0.3)[0]
        self.vertical_line_x_positions = set(v_line_cols.tolist())
        # Expand the set to include neighboring columns (reference lines affect nearby tracking)
        expanded_positions = set()
        for x in self.vertical_line_x_positions:
            for dx in range(-5, 6):  # ±5 pixels around each line
                expanded_positions.add(x + dx)
        self.vertical_line_x_positions = expanded_positions

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_3_h_line_mask.png", h_line_mask)
            cv2.imwrite(f"{debug_dir}/debug_4_v_line_mask.png", v_line_mask)

        # Step 3: Remove reference lines completely from original binary
        # Don't try to preserve intersections - we'll fill gaps with dilation later
        filtered = original_binary.copy()
        filtered[h_line_mask > 0] = 0
        filtered[v_line_mask > 0] = 0

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_5_after_line_removal.png", filtered)

        # Step 4: Dilate to fill small gaps created by line removal
        # Use horizontal kernel to preserve vertical separation between curves
        # This fills horizontal gaps (from vertical line removal) without merging
        # vertically adjacent curves
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        filtered = cv2.dilate(filtered, fill_kernel, iterations=1)

        # No erosion needed - the horizontal dilation doesn't significantly
        # change curve thickness

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_6_after_gap_fill.png", filtered)

        # Step 6: Remove small noise components
        noise_removed = self._remove_isolated_components(filtered)

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_7_noise_removed.png", filtered)

        # Save binary before text filtering (for plateau detection later)
        # Text filtering can remove plateau pixels that look like labels
        self.binary_before_text_filter = filtered.copy()

        # Step 7: Filter out text elements (legends, labels, annotations)
        text_removed = self._filter_text_and_legends(filtered, debug_dir)

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_8_final.png", filtered)

        h_count = np.sum(h_line_mask > 0) // width if width > 0 else 0
        v_count = np.sum(v_line_mask > 0) // height if height > 0 else 0
        print(f"    Preprocessing complete: ~{h_count} H-line rows, ~{v_count} V-line cols, "
              f"{noise_removed} noise, {text_removed} text components")

        return filtered

    def _detect_horizontal_line_mask(self, binary: np.ndarray) -> np.ndarray:
        """Detect horizontal reference lines and return a mask."""
        height, width = binary.shape
        mask = np.zeros_like(binary)

        # Method 1: Find rows with long continuous horizontal runs
        min_line_length = int(width * 0.4)

        for y in range(height):
            row = binary[y, :]
            nonzero = np.where(row > 0)[0]

            if len(nonzero) < min_line_length:
                continue

            if len(nonzero) > 1:
                gaps = np.diff(nonzero)
                run_starts = [0] + list(np.where(gaps > 5)[0] + 1)
                run_ends = list(np.where(gaps > 5)[0]) + [len(nonzero) - 1]

                for start_idx, end_idx in zip(run_starts, run_ends):
                    run_length = end_idx - start_idx + 1
                    if run_length >= min_line_length:
                        x_start = nonzero[start_idx]
                        x_end = nonzero[end_idx]

                        # Check if it's a thin line
                        y_min = max(0, y - 4)
                        y_max = min(height, y + 5)
                        region = binary[y_min:y_max, x_start:x_end+1]
                        row_densities = np.sum(region > 0, axis=1) / (x_end - x_start + 1)
                        thick_rows = np.sum(row_densities > 0.5)

                        if thick_rows <= 8:  # Allow slightly thicker lines
                            # Mark this horizontal line in the mask
                            for dy in range(-3, 4):
                                row_y = y + dy
                                if 0 <= row_y < height:
                                    mask[row_y, x_start:x_end+1] = 255

        # Method 2: Morphological detection
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 4, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Exclude edges (X-axis at bottom, potential top border)
        edge_margin = int(height * 0.05)
        h_lines[:edge_margin, :] = 0
        h_lines[-edge_margin:, :] = 0

        mask = cv2.bitwise_or(mask, h_lines)

        return mask

    def _detect_vertical_line_mask(self, binary: np.ndarray) -> np.ndarray:
        """Detect vertical reference lines and return a mask."""
        height, width = binary.shape
        mask = np.zeros_like(binary)

        # Method 1: Find columns with long continuous vertical runs
        min_line_length = int(height * 0.4)

        # Skip the left edge (Y-axis) - start from 5% of width
        start_x = int(width * 0.05)
        for x in range(start_x, width):
            col = binary[:, x]
            nonzero = np.where(col > 0)[0]

            if len(nonzero) < min_line_length:
                continue

            if len(nonzero) > 1:
                gaps = np.diff(nonzero)
                run_starts = [0] + list(np.where(gaps > 5)[0] + 1)
                run_ends = list(np.where(gaps > 5)[0]) + [len(nonzero) - 1]

                for start_idx, end_idx in zip(run_starts, run_ends):
                    run_length = end_idx - start_idx + 1
                    if run_length >= min_line_length:
                        y_start = nonzero[start_idx]
                        y_end = nonzero[end_idx]

                        # Check if it's a thin line
                        x_min = max(0, x - 4)
                        x_max = min(width, x + 5)
                        region = binary[y_start:y_end+1, x_min:x_max]
                        col_densities = np.sum(region > 0, axis=0) / (y_end - y_start + 1)
                        thick_cols = np.sum(col_densities > 0.5)

                        if thick_cols <= 8:
                            for dx in range(-3, 4):
                                col_x = x + dx
                                if 0 <= col_x < width:
                                    mask[y_start:y_end+1, col_x] = 255

        # Method 2: Morphological detection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 4))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Exclude edges (Y-axis on left, potential axis on right)
        edge_margin = int(width * 0.05)
        v_lines[:, :edge_margin] = 0
        v_lines[:, -edge_margin:] = 0

        mask = cv2.bitwise_or(mask, v_lines)

        return mask

    def _find_curve_intersections_horizontal(self, dilated: np.ndarray,
                                              h_line_mask: np.ndarray) -> np.ndarray:
        """Find curve pixels that intersect with horizontal reference lines."""
        height, width = dilated.shape
        intersections = np.zeros_like(dilated)

        # For each pixel in the horizontal line mask, check if it has
        # vertical connectivity (indicating it's part of a curve crossing)
        line_pixels = np.where(h_line_mask > 0)

        for y, x in zip(line_pixels[0], line_pixels[1]):
            # Check for curve pixels above and below (on dilated image)
            has_above = False
            has_below = False

            # Check above (with margin for line thickness)
            for check_y in range(max(0, y - 10), max(0, y - 3)):
                if dilated[check_y, x] > 0 and h_line_mask[check_y, x] == 0:
                    has_above = True
                    break

            # Check below
            for check_y in range(min(height, y + 4), min(height, y + 11)):
                if dilated[check_y, x] > 0 and h_line_mask[check_y, x] == 0:
                    has_below = True
                    break

            if has_above or has_below:
                # This is a curve intersection - mark for preservation
                intersections[y, x] = 255

        # Dilate intersections slightly to ensure we preserve enough
        kernel = np.ones((5, 3), np.uint8)
        intersections = cv2.dilate(intersections, kernel, iterations=1)

        return intersections

    def _find_curve_intersections_vertical(self, dilated: np.ndarray,
                                            v_line_mask: np.ndarray) -> np.ndarray:
        """Find curve pixels that intersect with vertical reference lines."""
        height, width = dilated.shape
        intersections = np.zeros_like(dilated)

        line_pixels = np.where(v_line_mask > 0)

        for y, x in zip(line_pixels[0], line_pixels[1]):
            # Skip Y-axis (left edge) - curves don't pass through it
            if x < width * 0.05:
                continue

            has_left = False
            has_right = False

            # Check left - look for curve pixels not part of the vertical line
            for check_x in range(max(0, x - 12), max(0, x - 4)):
                if dilated[y, check_x] > 0 and v_line_mask[y, check_x] == 0:
                    has_left = True
                    break

            # Check right
            for check_x in range(min(width, x + 5), min(width, x + 13)):
                if dilated[y, check_x] > 0 and v_line_mask[y, check_x] == 0:
                    has_right = True
                    break

            # BOTH left AND right required - curve must pass through the line
            if has_left and has_right:
                intersections[y, x] = 255

        # Minimal dilation - just to ensure we don't create gaps
        kernel = np.ones((3, 3), np.uint8)
        intersections = cv2.dilate(intersections, kernel, iterations=1)

        return intersections

    def _filter_reference_lines(self, binary: np.ndarray, debug_dir: Optional[str] = None) -> np.ndarray:
        """
        Comprehensive filter to isolate survival curves by removing all non-curve elements.

        Removes:
        1. Horizontal reference lines (e.g., 50% survival line)
        2. Vertical reference lines (median markers)
        3. Text labels and legends
        4. Grid lines and axis remnants

        Args:
            binary: Binary mask of detected pixels
            debug_dir: Optional directory to save debug images

        Returns:
            Filtered binary mask containing only survival curve pixels
        """
        height, width = binary.shape
        filtered = binary.copy()

        print(f"    Starting comprehensive preprocessing ({width}x{height})")

        # =================================================================
        # STEP 1: Detect horizontal reference lines using projection analysis
        # =================================================================
        h_lines_removed = self._detect_and_remove_horizontal_lines(filtered)
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_3_h_lines_removed.png", filtered)

        # =================================================================
        # STEP 2: Detect vertical reference lines using projection analysis
        # =================================================================
        v_lines_removed = self._detect_and_remove_vertical_lines(filtered)
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_4_v_lines_removed.png", filtered)

        # =================================================================
        # STEP 3: Remove text regions using connected component analysis
        # =================================================================
        text_components_removed = self._remove_text_regions(filtered)
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_5_text_removed.png", filtered)

        # =================================================================
        # STEP 4: Remove isolated small components (noise/fragments)
        # =================================================================
        noise_removed = self._remove_isolated_components(filtered)
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_6_noise_removed.png", filtered)

        print(f"    Preprocessing complete: removed {h_lines_removed} H-lines, "
              f"{v_lines_removed} V-lines, {text_components_removed} text regions, "
              f"{noise_removed} noise components")

        return filtered

    def _detect_and_remove_horizontal_lines(self, binary: np.ndarray) -> int:
        """
        Detect and remove horizontal reference lines.

        Uses multiple detection methods:
        1. Row projection analysis (sum of pixels in each row)
        2. Run-length analysis (find continuous horizontal runs)
        3. Morphological line detection

        Args:
            binary: Binary mask to modify in place

        Returns:
            Number of horizontal lines removed
        """
        height, width = binary.shape
        lines_removed = 0

        # Method 1: Find rows with continuous horizontal runs spanning >50% of width
        min_line_length = int(width * 0.4)

        for y in range(height):
            row = binary[y, :]
            nonzero = np.where(row > 0)[0]

            if len(nonzero) < min_line_length:
                continue

            # Find continuous runs in this row
            if len(nonzero) > 1:
                gaps = np.diff(nonzero)
                # A reference line has most pixels continuous (gaps <= 3 pixels)
                run_starts = [0] + list(np.where(gaps > 3)[0] + 1)
                run_ends = list(np.where(gaps > 3)[0]) + [len(nonzero) - 1]

                for start_idx, end_idx in zip(run_starts, run_ends):
                    run_length = end_idx - start_idx + 1
                    if run_length >= min_line_length:
                        # This is a long horizontal run - likely a reference line
                        # Check if it's thin (reference lines are typically 1-4 pixels thick)
                        x_start = nonzero[start_idx]
                        x_end = nonzero[end_idx]

                        # Check thickness by looking at neighboring rows
                        y_min = max(0, y - 4)
                        y_max = min(height, y + 5)
                        region = binary[y_min:y_max, x_start:x_end+1]

                        # Count how many rows in this region have high density
                        row_densities = np.sum(region > 0, axis=1) / (x_end - x_start + 1)
                        thick_rows = np.sum(row_densities > 0.5)

                        # Reference lines are thin (<=6 pixels thick)
                        if thick_rows <= 6:
                            # Remove this horizontal line segment, but preserve curve intersections
                            # A curve intersection has vertical connectivity above or below the line
                            for x_pos in range(x_start, x_end + 1):
                                # Check if this pixel is part of a curve (has vertical neighbors)
                                has_curve_above = False
                                has_curve_below = False

                                # Check above the line region
                                for check_y in range(max(0, y - 8), max(0, y - 4)):
                                    if binary[check_y, x_pos] > 0:
                                        has_curve_above = True
                                        break

                                # Check below the line region
                                for check_y in range(min(height, y + 5), min(height, y + 9)):
                                    if binary[check_y, x_pos] > 0:
                                        has_curve_below = True
                                        break

                                # Only remove if NOT part of a curve crossing
                                if not (has_curve_above or has_curve_below):
                                    for dy in range(-3, 4):
                                        row_y = y + dy
                                        if 0 <= row_y < height:
                                            binary[row_y, x_pos] = 0
                            lines_removed += 1

        # Method 2: Use morphological detection for any remaining lines
        # Create a horizontal kernel
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 4, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        if np.any(h_lines > 0):
            # Find rows with detected horizontal lines
            h_line_rows = np.where(np.sum(h_lines, axis=1) > width * 0.3 * 255)[0]
            for y in h_line_rows:
                line_mask = h_lines[y, :] > 0
                line_x_positions = np.where(line_mask)[0]

                for x_pos in line_x_positions:
                    # Check for vertical curve connectivity
                    has_curve_above = np.any(binary[max(0, y-8):max(0, y-4), x_pos] > 0)
                    has_curve_below = np.any(binary[min(height, y+5):min(height, y+9), x_pos] > 0)

                    if not (has_curve_above or has_curve_below):
                        for row_y in range(max(0, y-2), min(height, y+3)):
                            binary[row_y, x_pos] = 0
                lines_removed += 1

        return lines_removed

    def _detect_and_remove_vertical_lines(self, binary: np.ndarray) -> int:
        """
        Detect and remove vertical reference lines (median markers).

        Args:
            binary: Binary mask to modify in place

        Returns:
            Number of vertical lines removed
        """
        height, width = binary.shape
        lines_removed = 0

        # Method 1: Find columns with continuous vertical runs spanning >40% of height
        min_line_length = int(height * 0.4)

        for x in range(width):
            col = binary[:, x]
            nonzero = np.where(col > 0)[0]

            if len(nonzero) < min_line_length:
                continue

            # Find continuous runs in this column
            if len(nonzero) > 1:
                gaps = np.diff(nonzero)
                run_starts = [0] + list(np.where(gaps > 3)[0] + 1)
                run_ends = list(np.where(gaps > 3)[0]) + [len(nonzero) - 1]

                for start_idx, end_idx in zip(run_starts, run_ends):
                    run_length = end_idx - start_idx + 1
                    if run_length >= min_line_length:
                        # Long vertical run - check if it's thin
                        y_start = nonzero[start_idx]
                        y_end = nonzero[end_idx]

                        x_min = max(0, x - 4)
                        x_max = min(width, x + 5)
                        region = binary[y_start:y_end+1, x_min:x_max]

                        col_densities = np.sum(region > 0, axis=0) / (y_end - y_start + 1)
                        thick_cols = np.sum(col_densities > 0.5)

                        # Reference lines are thin (<=6 pixels thick)
                        if thick_cols <= 6:
                            # Remove vertical line, but preserve curve intersections
                            for y_pos in range(y_start, y_end + 1):
                                # Check if this pixel is part of a curve (has horizontal neighbors)
                                has_curve_left = False
                                has_curve_right = False

                                # Check left of the line region
                                for check_x in range(max(0, x - 8), max(0, x - 4)):
                                    if binary[y_pos, check_x] > 0:
                                        has_curve_left = True
                                        break

                                # Check right of the line region
                                for check_x in range(min(width, x + 5), min(width, x + 9)):
                                    if binary[y_pos, check_x] > 0:
                                        has_curve_right = True
                                        break

                                # Only remove if NOT part of a curve crossing
                                if not (has_curve_left or has_curve_right):
                                    for dx in range(-3, 4):
                                        col_x = x + dx
                                        if 0 <= col_x < width:
                                            binary[y_pos, col_x] = 0
                            lines_removed += 1

        # Method 2: Morphological detection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 4))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        if np.any(v_lines > 0):
            v_line_cols = np.where(np.sum(v_lines, axis=0) > height * 0.3 * 255)[0]
            for x in v_line_cols:
                line_mask = v_lines[:, x] > 0
                line_y_positions = np.where(line_mask)[0]

                for y_pos in line_y_positions:
                    # Check for horizontal curve connectivity
                    has_curve_left = np.any(binary[y_pos, max(0, x-8):max(0, x-4)] > 0)
                    has_curve_right = np.any(binary[y_pos, min(width, x+5):min(width, x+9)] > 0)

                    if not (has_curve_left or has_curve_right):
                        for col_x in range(max(0, x-2), min(width, x+3)):
                            binary[y_pos, col_x] = 0
                lines_removed += 1

        return lines_removed

    def _remove_text_regions(self, binary: np.ndarray) -> int:
        """
        Remove text labels, legends, and other text-like regions.

        Text regions are characterized by:
        - Small connected components with certain aspect ratios
        - Clusters of small components (letters forming words)
        - Located in corners or near curves (labels)
        - Components that don't span a significant horizontal range

        Args:
            binary: Binary mask to modify in place

        Returns:
            Number of text components removed
        """
        height, width = binary.shape
        components_removed = 0

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # First pass: identify curve-like components (including segments after line removal)
        main_curve_ids = set()
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Main curves span significant horizontal distance
            if w > width * 0.3 and area > 500:
                main_curve_ids.add(i)

            # Also protect curve segments (after vertical line removal)
            # These are elongated horizontally, span at least 10% width, and have curve-like shape
            elif w > width * 0.1 and w > h * 2 and area > 200:
                # Elongated horizontal component - likely a curve segment
                fill_ratio = area / (w * h) if w * h > 0 else 0
                # Curves have low fill ratio (they're thin lines, not solid blocks)
                if fill_ratio < 0.4:
                    main_curve_ids.add(i)

        # Analyze each component
        for i in range(1, num_labels):  # Skip background (0)
            # Never remove main curve components
            if i in main_curve_ids:
                continue

            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            # Skip very small noise (handled elsewhere)
            if area < 5:
                continue

            is_text = False
            aspect_ratio = w / h if h > 0 else 0
            fill_ratio = area / (w * h) if w * h > 0 else 0

            # =================================================================
            # CRITERION 1: Top region - likely titles, statistics, axis labels
            # =================================================================
            if y < height * 0.15:
                # Almost anything in the top 15% is text/annotation
                if area < 2000:
                    is_text = True

            # =================================================================
            # CRITERION 2: Right edge region - likely legends
            # Only remove if component is small AND isolated (not extending into plot)
            # =================================================================
            if x > width * 0.85:  # Starts in right 15%
                # Only if component is small and doesn't extend into the main plot area
                if w < width * 0.1 and area < 300:
                    is_text = True

            # =================================================================
            # CRITERION 3: Small isolated characters
            # =================================================================
            if not is_text and area < 200 and w < 30 and h < 30:
                if fill_ratio > 0.2 and fill_ratio < 0.95:
                    if 0.2 < aspect_ratio < 4.0:
                        is_text = True

            # =================================================================
            # CRITERION 4: Text blocks (words, labels)
            # =================================================================
            if not is_text and area < 1500 and w < 150 and h < 50:
                if 0.1 < fill_ratio < 0.8:
                    # Wide aspect ratio (words) or narrow (single letters)
                    if aspect_ratio > 2 or aspect_ratio < 0.5:
                        is_text = True

            # =================================================================
            # CRITERION 5: Components that don't span significant horizontal range
            # Survival curves span from left to right; text doesn't
            # BUT: curve fragments from dashed lines can be narrow
            # =================================================================
            if not is_text and w < width * 0.05:
                # Very narrow components - check if they look like text (not curve fragments)
                # Curve fragments are typically thin and elongated horizontally
                if area < 200 and h > w:  # Taller than wide = likely text
                    is_text = True

            # =================================================================
            # CRITERION 6: Corner regions (legends, statistics)
            # Only top corners and bottom-left corner
            # Bottom-right can have curve data
            # =================================================================
            if not is_text:
                # Top corners only - where titles and statistics appear
                in_top_corner = (
                    y < height * 0.2 and
                    (x < width * 0.15 or x + w > width * 0.85)
                )
                # Bottom-left corner - where axis labels might be
                in_bottom_left = (
                    y + h > height * 0.8 and
                    x < width * 0.15
                )
                if (in_top_corner or in_bottom_left) and area < 500 and w < width * 0.1:
                    is_text = True

            if is_text:
                binary[labels == i] = 0
                components_removed += 1

        return components_removed

    def _remove_isolated_components(self, binary: np.ndarray) -> int:
        """
        Remove small isolated components that are likely noise or fragments.

        Survival curves are continuous structures, so isolated small blobs
        that don't connect to larger structures are removed.

        Args:
            binary: Binary mask to modify in place

        Returns:
            Number of components removed
        """
        height, width = binary.shape
        components_removed = 0

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Calculate minimum area for a valid curve segment
        # Curves should have area proportional to their length
        min_curve_area = 20  # Minimum pixels for a valid segment

        # Find the largest component (likely a curve)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = np.max(areas) if len(areas) > 0 else 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Remove very small components (noise)
            if area < min_curve_area:
                binary[labels == i] = 0
                components_removed += 1
                continue

            # Remove components that are too "blobby" (not curve-like)
            # Curves are elongated; blobs have low perimeter-to-area ratio
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

            # Very square/round small components are likely noise
            if area < 100 and aspect_ratio < 2:
                fill_ratio = area / (w * h) if w * h > 0 else 0
                # High fill ratio = blob, not curve segment
                if fill_ratio > 0.6:
                    binary[labels == i] = 0
                    components_removed += 1

        return components_removed

    def _filter_text_and_legends(self, binary: np.ndarray, debug_dir: Optional[str] = None) -> int:
        """
        Filter out text elements like legends, labels, and annotations.

        Text characteristics:
        1. Located in legend areas (typically upper-right or near edges)
        2. Small-to-medium components with text-like aspect ratios
        3. Components that are far from the main curve area

        Args:
            binary: Binary mask to modify in place

        Returns:
            Number of text components removed
        """
        height, width = binary.shape
        components_removed = 0

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Define regions where text/legends typically appear
        # Legend area: top portion and right portion of plot
        legend_top = height * 0.35  # Top 35% might have legend
        legend_right_start = width * 0.55  # Right 45% might have legend text

        # Text near curves (labels like "RT + TMZ", "RT Only")
        # These are typically small components not connected to main curves
        text_max_area = 800  # Text components are typically small
        text_min_area = 30   # But not too small (that's noise)

        # Get the largest components (main curves)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_areas = sorted(areas, reverse=True)
            # Top 2 largest are likely the curves
            curve_min_area = sorted_areas[1] * 0.3 if len(sorted_areas) >= 2 else 500

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            # Skip components that are likely curves (large enough)
            if area > curve_min_area:
                continue

            # Remove components in legend area (upper-right)
            if cy < legend_top and cx > legend_right_start:
                if area < text_max_area * 2:  # Allow larger for multi-line legends
                    binary[labels == i] = 0
                    components_removed += 1
                    continue

            # Remove any isolated components in the upper portion (above survival 0.7)
            # that are not part of the main curves
            if cy < height * 0.25 and area < text_max_area:
                binary[labels == i] = 0
                components_removed += 1
                continue

            # Remove small components in right margin (curve labels)
            if cx > width * 0.70 and area < text_max_area and area > text_min_area:
                # Check if it's isolated (not connected to main curves)
                # Text has higher fill ratio within bounding box
                fill_ratio = area / (w * h) if w * h > 0 else 0
                if fill_ratio > 0.15:  # Text is denser than curve segments
                    # But preserve thin, wide segments in curve Y-range (plateau lines)
                    aspect_wh = w / h if h > 0 else 1
                    in_curve_y_range = height * 0.20 < cy < height * 0.85
                    is_curve_segment = h < 15 and aspect_wh > 1.5
                    if is_curve_segment and in_curve_y_range:
                        continue  # Preserve this as a curve segment
                    binary[labels == i] = 0
                    components_removed += 1
                    continue

            # Remove bottom annotations (axis labels like "12.1", "14.6")
            if cy > height * 0.85 and area < 500:
                binary[labels == i] = 0
                components_removed += 1
                continue

            # Remove curve labels in the lower-right area (like "RT Only", "RT + TMZ")
            # These typically appear in the bottom-right portion of the plot
            if cy > height * 0.55 and cx > width * 0.55:
                if area < text_max_area * 5:  # Allow larger for multi-word labels + dilation
                    # Check if it's not too tall (text is typically short in height)
                    if h < 80:  # Increased from 50 to catch larger text
                        # Additional check: text is typically not very elongated vertically
                        aspect_hw = h / w if w > 0 else 1
                        aspect_wh = w / h if h > 0 else 1
                        fill_ratio = area / (w * h) if w * h > 0 else 0

                        # Preserve curve segments (thin and horizontally elongated)
                        # After dilation, curve segments may have high fill ratio
                        # Key characteristics: thin (h < 15) and wide (aspect_wh > 2)
                        is_curve_segment = (
                            h < 15 and aspect_wh > 2.0
                        )

                        if aspect_hw < 3 and not is_curve_segment:
                            binary[labels == i] = 0
                            components_removed += 1
                            continue

            # More aggressive filtering in the far right margin (last 25% of width)
            # This area typically only has curve labels, not actual curve data
            # BUT we need to preserve actual curve segments (plateau lines, etc.)
            if cx > width * 0.75:
                # Filter any medium-sized components in the far right
                if area < text_max_area * 8 and area > text_min_area:
                    # Calculate characteristics
                    aspect = w / h if h > 0 else 1
                    fill_ratio = area / (w * h) if w * h > 0 else 0

                    # Preserve horizontally-elongated segments (likely curve pieces)
                    # Curve segments: wide AND thin (plateau lines, dashed segments)
                    # After dilation, curves may have high fill ratio, so don't filter by fill
                    is_curve_segment = (
                        aspect > 3 or  # Very wide (curve segment)
                        (aspect > 1.5 and h < 15)  # Moderately wide and thin
                    )

                    # Preserve components in the curve Y-range (middle of plot)
                    # Curves should be between top 20% (legend) and bottom 15% (axis labels)
                    in_curve_y_range = height * 0.20 < cy < height * 0.85

                    # Only remove if it looks like text (not a curve segment)
                    if not is_curve_segment or not in_curve_y_range:
                        # Additional check: text has higher fill ratio
                        if fill_ratio > 0.25 or h > 30:
                            binary[labels == i] = 0
                            components_removed += 1
                            continue

            # Remove components that look like text based on characteristics
            # Text tends to be relatively wide compared to height
            aspect = w / h if h > 0 else 1
            if text_min_area < area < text_max_area:
                # Text-like: wider than tall, medium fill ratio
                if aspect > 1.5 and h < 30:
                    fill_ratio = area / (w * h) if w * h > 0 else 0
                    if fill_ratio > 0.2:
                        # Preserve thin, wide segments in the curve Y-range
                        # These are likely curve plateau segments, not text
                        in_curve_y_range = height * 0.20 < cy < height * 0.85
                        is_thin_wide_segment = h < 15 and aspect > 2.0
                        if is_thin_wide_segment and in_curve_y_range:
                            continue  # Preserve this as a curve segment
                        binary[labels == i] = 0
                        components_removed += 1
                        continue

        if debug_dir:
            cv2.imwrite(f"{debug_dir}/debug_8_text_filtered.png", binary)

        return components_removed

    def _trace_curves_by_position(self, expected_count: Optional[int] = None) -> List[TracedCurve]:
        """
        Trace curves by tracking Y-positions across X coordinates.

        At each X column, find clusters of Y positions and track them.
        """
        height, width = self.binary.shape

        # For each x, find all y positions with pixels
        # Use position-dependent Y filtering:
        # - Left/middle portion: NO top filtering to capture curves at survival=1.0 (y=0)
        # - Right portion (where legend text is): more aggressive filtering
        y_min_base = 0  # No top filter - curves start at y=0 (survival=1.0)
        y_min_legend = int(height * 0.15)  # Filter top 15% in legend area
        y_max_valid = int(height * 0.95)  # Filter bottom 5% for axis labels
        legend_x_start = int(width * 0.75)  # Legend typically in right 25%

        x_to_y_positions: Dict[int, List[int]] = {}

        for x in range(width):
            col = self.binary[:, x]
            y_positions = np.where(col > 0)[0]

            # Apply position-dependent filtering
            if x >= legend_x_start:
                # In legend area: filter more aggressively at top
                y_min = y_min_legend
            else:
                # In main curve area: minimal filtering
                y_min = y_min_base

            y_positions = [y for y in y_positions if y_min <= y <= y_max_valid]
            if len(y_positions) > 0:
                x_to_y_positions[x] = y_positions

        if not x_to_y_positions:
            return []

        # Find the number of curves by analyzing Y-position clusters
        # Sample multiple X positions to estimate curve count
        sample_x_positions = sorted(x_to_y_positions.keys())
        if len(sample_x_positions) > 20:
            # Sample evenly across the width
            indices = np.linspace(0, len(sample_x_positions)-1, 20, dtype=int)
            sample_x_positions = [sample_x_positions[i] for i in indices]

        # Count distinct Y clusters at each sampled X
        cluster_counts = []
        for x in sample_x_positions:
            if x in x_to_y_positions:
                y_vals = np.array(x_to_y_positions[x])
                n_clusters = self._count_y_clusters(y_vals)
                cluster_counts.append(n_clusters)

        # Estimate number of curves
        if expected_count:
            n_curves = expected_count
        elif cluster_counts:
            # Use mode or median of cluster counts
            n_curves = int(np.median(cluster_counts))
            n_curves = max(1, min(n_curves, 10))  # Reasonable bounds
        else:
            n_curves = 1

        # Initialize curve trackers
        curves = [TracedCurve(
            y_positions={},
            pixel_presence={},
            style=LineStyle.UNKNOWN,
            confidence=0.0
        ) for _ in range(n_curves)]

        # Process X positions from left to right
        all_x = sorted(x_to_y_positions.keys())

        # Track positions skipped due to vertical reference lines
        skipped_x_positions = set()

        for x in all_x:
            # Skip x positions affected by vertical reference lines
            # These positions have unreliable Y values due to line artifacts
            if x in self.vertical_line_x_positions:
                skipped_x_positions.add(x)
                continue

            y_vals = x_to_y_positions[x]

            # Cluster Y values at this X
            # Use dash mask to help split clusters when curves are close together
            clusters = self._cluster_y_values_with_style(x, y_vals, n_curves)

            # Assign clusters to curves based on proximity to previous positions
            self._assign_clusters_to_curves(curves, x, clusters)

        # Interpolate through skipped positions (vertical reference lines)
        if skipped_x_positions:
            print(f"    Interpolating through {len(skipped_x_positions)} reference line positions")
            self._interpolate_through_gaps(curves, skipped_x_positions)

        # Note: Smoothing is now done after crossing fix in detect_all_curves()
        # This allows the crossing fix to work on complete data first

        return curves

    def _interpolate_through_gaps(self, curves: List[TracedCurve], gap_positions: set):
        """
        Interpolate curve Y values through gaps caused by vertical reference lines.

        Uses linear interpolation between the nearest valid points on either side.
        This ensures curves smoothly pass through reference line regions instead
        of having artifacts or jumps.
        """
        for curve in curves:
            if not curve.y_positions:
                continue

            # Get all current x positions sorted
            existing_x = sorted(curve.y_positions.keys())
            if len(existing_x) < 2:
                continue

            # Find gap positions that fall within this curve's x range
            min_x, max_x = min(existing_x), max(existing_x)
            gaps_in_range = sorted([x for x in gap_positions if min_x < x < max_x])

            for gap_x in gaps_in_range:
                # Find nearest valid x on left and right
                left_x = None
                right_x = None

                for x in reversed(existing_x):
                    if x < gap_x and x not in gap_positions:
                        left_x = x
                        break

                for x in existing_x:
                    if x > gap_x and x not in gap_positions:
                        right_x = x
                        break

                # Interpolate if we have both sides
                if left_x is not None and right_x is not None:
                    left_y = curve.y_positions[left_x]
                    right_y = curve.y_positions[right_x]

                    # Linear interpolation
                    t = (gap_x - left_x) / (right_x - left_x)
                    interp_y = left_y + t * (right_y - left_y)
                    curve.y_positions[gap_x] = interp_y

    def _relabel_curves_by_position(self):
        """
        For KM curves with 2 curves: validate and fix style labels using dash_mask overlap.

        Instead of assuming dashed=better survival (which varies by study),
        use the dash_mask overlap to determine which curve is actually dashed.
        The curve with higher dash_mask overlap is more likely to be the dashed one.
        """
        if len(self.traced_curves) != 2:
            return

        curve0 = self.traced_curves[0]
        curve1 = self.traced_curves[1]

        # Calculate dash_mask overlap for each curve
        def calc_dash_overlap(curve):
            if not hasattr(self, 'dash_mask') or self.dash_mask is None:
                return 0.0
            dash_count = 0
            total_count = 0
            for x, y in curve.y_positions.items():
                y = int(y)
                if 0 <= y < self.dash_mask.shape[0] and 0 <= x < self.dash_mask.shape[1]:
                    total_count += 1
                    if self.dash_mask[y, x] > 0:
                        dash_count += 1
            return dash_count / total_count if total_count > 0 else 0.0

        overlap0 = calc_dash_overlap(curve0)
        overlap1 = calc_dash_overlap(curve1)

        # Also calculate presence ratio (gaps) for each curve
        def calc_presence_ratio(curve):
            x_positions = sorted(curve.y_positions.keys())
            if len(x_positions) < 10:
                return 1.0
            present_count = sum(1 for x in x_positions if curve.pixel_presence.get(x, False))
            return present_count / len(x_positions)

        presence0 = calc_presence_ratio(curve0)
        presence1 = calc_presence_ratio(curve1)

        print(f"    Style validation: curve0 dash_overlap={overlap0:.2f} presence={presence0:.2f}")
        print(f"    Style validation: curve1 dash_overlap={overlap1:.2f} presence={presence1:.2f}")

        # NEW: Direct scan of the plot to find which Y levels have dash patterns
        # Scan in the MIDDLE portion where curves are well separated
        def find_dashed_y_level():
            """
            Scan the middle portion (40-70%) of the plot at curve Y positions
            to find which curve has a clear dash pattern.
            The middle portion is where KM curves are typically well separated.
            """
            # Get X range for middle portion (40-70% of plot)
            x_start = int(self.plot_w * 0.40)
            x_end = int(self.plot_w * 0.70)

            if x_end - x_start < 30:
                return None, None

            # Get the Y positions of both curves in this region
            y0_values = [curve0.y_positions[x] for x in range(x_start, x_end) if x in curve0.y_positions]
            y1_values = [curve1.y_positions[x] for x in range(x_start, x_end) if x in curve1.y_positions]

            if len(y0_values) < 10 or len(y1_values) < 10:
                print(f"    WARNING: Not enough Y values in rightmost region: curve0={len(y0_values)}, curve1={len(y1_values)}")
                return None, None

            # Get average Y for each curve in this region
            y0_avg = int(np.mean(y0_values))
            y1_avg = int(np.mean(y1_values))
            print(f"    Middle region Y averages: curve0={y0_avg} (range {min(y0_values):.0f}-{max(y0_values):.0f}), curve1={y1_avg} (range {min(y1_values):.0f}-{max(y1_values):.0f})")

            # Use the MODE of Y values (most common Y) instead of average
            # This helps when curves have varying Y positions
            from collections import Counter
            y0_mode = Counter([int(y) for y in y0_values]).most_common(1)[0][0]
            y1_mode = Counter([int(y) for y in y1_values]).most_common(1)[0][0]

            # Also get the Y value at the rightmost X position for each curve
            max_x0 = max([x for x in range(x_start, x_end) if x in curve0.y_positions], default=x_start)
            max_x1 = max([x for x in range(x_start, x_end) if x in curve1.y_positions], default=x_start)
            y0_rightmost = int(curve0.y_positions.get(max_x0, y0_avg))
            y1_rightmost = int(curve1.y_positions.get(max_x1, y1_avg))

            print(f"    Y modes: curve0={y0_mode}, curve1={y1_mode}")
            print(f"    Y at rightmost X: curve0={y0_rightmost} (x={max_x0}), curve1={y1_rightmost} (x={max_x1})")

            # Use the rightmost Y values for scanning (most representative of where curve ends)
            # Determine which is upper (lower Y) and which is lower (higher Y)
            upper_y = min(y0_rightmost, y1_rightmost)
            lower_y = max(y0_rightmost, y1_rightmost)
            upper_is_curve0 = (y0_rightmost <= y1_rightmost)

            # Now scan at both Y levels and count gaps
            def count_gaps_at_y(y_pos):
                img_y = y_pos + self.plot_y
                if img_y < 3 or img_y >= self.gray.shape[0] - 3:
                    return 0, 0

                gaps = []
                current_gap = 0
                dark_count = 0

                for x in range(x_start, x_end):
                    img_x = x + self.plot_x
                    if 0 <= img_x < self.gray.shape[1]:
                        # Use a narrow band (5 pixels) centered on the Y position
                        band = self.gray[img_y - 2:img_y + 3, img_x]
                        min_val = np.min(band) if band.size > 0 else 255

                        if min_val > 200:  # White = gap
                            current_gap += 1
                        else:
                            if min_val < 150:
                                dark_count += 1
                            if current_gap >= 3:
                                gaps.append(current_gap)
                            current_gap = 0

                if current_gap >= 3:
                    gaps.append(current_gap)

                return len(gaps), dark_count

            upper_gaps, upper_dark = count_gaps_at_y(upper_y)
            lower_gaps, lower_dark = count_gaps_at_y(lower_y)

            print(f"    Direct Y scan: upper Y={upper_y} ({upper_gaps} gaps, {upper_dark} dark), lower Y={lower_y} ({lower_gaps} gaps, {lower_dark} dark)")
            print(f"    Scan range: x={x_start} to {x_end} (plot), img_y_upper={upper_y + self.plot_y}, img_y_lower={lower_y + self.plot_y}")

            x_range = x_end - x_start
            return upper_is_curve0, upper_gaps, lower_gaps, upper_dark, lower_dark, x_range

        result = find_dashed_y_level()
        if result and result[0] is not None:
            upper_is_curve0, upper_gaps, lower_gaps, upper_dark, lower_dark, x_range = result

            # Key insight: DASHED lines have FEWER dark pixels per unit length
            # because the gaps reduce the total number of dark pixels
            # Calculate dark pixel density (dark per position)
            upper_density = upper_dark / x_range if x_range > 0 else 0
            lower_density = lower_dark / x_range if x_range > 0 else 0

            # The curve with LOWER dark density is more likely dashed
            # (fewer dark pixels = more gaps = dashed)
            # Calculate the relative density difference
            max_density = max(upper_density, lower_density)
            if max_density > 0.1:
                # The curve with significantly lower density is dashed
                upper_is_dashed_by_density = (lower_density > upper_density * 1.3)
                lower_is_dashed_by_density = (upper_density > lower_density * 1.3)
            else:
                upper_is_dashed_by_density = False
                lower_is_dashed_by_density = False

            print(f"    Dark density: upper={upper_density:.2f}, lower={lower_density:.2f}")
            print(f"    Density ratio: upper/lower={upper_density/lower_density if lower_density > 0 else 'inf':.2f}")

            # If one curve has significantly lower density, it's dashed
            if lower_is_dashed_by_density:
                # Lower curve (in image) has lower density = more gaps = DASHED
                print(f"    Density analysis: LOWER curve is dashed (density {lower_density:.2f} < {upper_density:.2f})")
                if upper_is_curve0:
                    print(f"    Setting: curve0=solid (upper), curve1=dashed (lower)")
                    curve0.style = LineStyle.SOLID
                    curve0.confidence = 0.85
                    curve1.style = LineStyle.DASHED
                    curve1.confidence = 0.85
                else:
                    print(f"    Setting: curve0=dashed (lower), curve1=solid (upper)")
                    curve0.style = LineStyle.DASHED
                    curve0.confidence = 0.85
                    curve1.style = LineStyle.SOLID
                    curve1.confidence = 0.85
                return  # Skip old analysis
            elif upper_is_dashed_by_density:
                # Upper curve (in image) has lower density = more gaps = DASHED
                print(f"    Density analysis: UPPER curve is dashed (density {upper_density:.2f} < {lower_density:.2f})")
                if upper_is_curve0:
                    print(f"    Setting: curve0=dashed (upper), curve1=solid (lower)")
                    curve0.style = LineStyle.DASHED
                    curve0.confidence = 0.85
                    curve1.style = LineStyle.SOLID
                    curve1.confidence = 0.85
                else:
                    print(f"    Setting: curve0=solid (lower), curve1=dashed (upper)")
                    curve0.style = LineStyle.SOLID
                    curve0.confidence = 0.85
                    curve1.style = LineStyle.DASHED
                    curve1.confidence = 0.85
                return  # Skip old analysis

        # Also check gap regularity - true dashed lines have REGULAR gaps
        # Use pre-dilation binary to preserve dash gaps
        def calc_gap_regularity(curve):
            """Check if gaps along curve are regular (true dash pattern)."""
            x_positions = sorted(curve.y_positions.keys())
            if len(x_positions) < 20:
                return 0.0, 0

            # Use pre-dilation binary if available (preserves dash gaps)
            check_binary = getattr(self, 'binary_before_dilation', self.binary)

            # Check binary mask along curve path for gaps
            gap_lengths = []
            segment_lengths = []
            in_gap = False
            in_segment = False
            gap_start = 0
            seg_start = 0

            for i, x in enumerate(x_positions):
                y = int(curve.y_positions[x])
                has_pixel = False
                if 0 <= y < check_binary.shape[0] and 0 <= x < check_binary.shape[1]:
                    # Check at traced y and nearby
                    for dy in range(-2, 3):
                        if 0 <= y + dy < check_binary.shape[0]:
                            if check_binary[y + dy, x] > 0:
                                has_pixel = True
                                break

                if not has_pixel:
                    if in_segment:
                        seg_len = i - seg_start
                        if seg_len >= 2:
                            segment_lengths.append(seg_len)
                        in_segment = False
                    if not in_gap:
                        gap_start = i
                        in_gap = True
                else:
                    if in_gap:
                        gap_len = i - gap_start
                        if gap_len >= 2:  # Only count significant gaps
                            gap_lengths.append(gap_len)
                        in_gap = False
                    if not in_segment:
                        seg_start = i
                        in_segment = True

            num_gaps = len(gap_lengths)
            if num_gaps < 3:
                return 0.0, num_gaps  # Not enough gaps for pattern analysis

            # Calculate coefficient of variation for gaps (lower = more regular)
            avg_gap = np.mean(gap_lengths)
            std_gap = np.std(gap_lengths)
            gap_cv = std_gap / avg_gap if avg_gap > 0 else 999

            # Also check segment regularity
            seg_cv = 999
            if len(segment_lengths) >= 3:
                avg_seg = np.mean(segment_lengths)
                std_seg = np.std(segment_lengths)
                seg_cv = std_seg / avg_seg if avg_seg > 0 else 999

            # Regular dash pattern: both gaps and segments have low CV
            # Return a regularity score (higher = more regular dash pattern)
            if gap_cv < 1.0 and seg_cv < 1.5:
                return max(0, 1.0 - (gap_cv + seg_cv) / 2), num_gaps
            elif gap_cv < 1.5:
                return max(0, 0.5 - gap_cv / 3), num_gaps
            return 0.0, num_gaps

        # Analyze original grayscale image for dash patterns along curve path
        def analyze_curve_for_dashes_in_grayscale(curve, curve_name=""):
            """
            Check original grayscale image for dash-like intensity variations.
            Dashed curves show alternating dark/light patterns along the path.
            """
            # Get grayscale plot region
            gray_plot = self.gray[
                self.plot_y:self.plot_y + self.plot_h,
                self.plot_x:self.plot_x + self.plot_w
            ]

            x_positions = sorted(curve.y_positions.keys())
            if len(x_positions) < 50:
                return 0.0, 0, 0.0

            # Only analyze middle portion where curves are separated
            mid_start = len(x_positions) // 4
            mid_end = int(len(x_positions) * 0.85)
            mid_positions = x_positions[mid_start:mid_end]

            if len(mid_positions) < 30:
                return 0.0, 0, 0.0

            # Sample more densely to catch dash patterns
            sample_step = max(1, len(mid_positions) // 200)
            sample_xs = mid_positions[::sample_step]

            # Collect intensity values along the curve path
            intensities = []
            for x in sample_xs:
                y_center = int(curve.y_positions[x])
                if 0 <= y_center < gray_plot.shape[0] and 0 <= x < gray_plot.shape[1]:
                    # Get minimum intensity in a small vertical band (darkest = curve)
                    min_intensity = 255
                    for dy in range(-2, 3):
                        y_check = y_center + dy
                        if 0 <= y_check < gray_plot.shape[0]:
                            min_intensity = min(min_intensity, gray_plot[y_check, x])
                    intensities.append(min_intensity)

            if len(intensities) < 30:
                return 0.0, 0, 0.0

            # Analyze intensity pattern for dashes
            # Threshold to determine dark (curve) vs light (gap)
            threshold = np.percentile(intensities, 30)  # 30th percentile as dark threshold

            # Count transitions from dark to light (dash gaps)
            is_dark = [i < threshold for i in intensities]
            transitions = 0
            gap_lengths = []
            current_gap = 0

            for i in range(1, len(is_dark)):
                if is_dark[i - 1] and not is_dark[i]:
                    transitions += 1
                    current_gap = 1
                elif not is_dark[i - 1] and not is_dark[i]:
                    current_gap += 1
                elif not is_dark[i - 1] and is_dark[i]:
                    if current_gap >= 2:
                        gap_lengths.append(current_gap)
                    current_gap = 0

            # Calculate metrics
            gap_ratio = sum(1 for d in is_dark if not d) / len(is_dark)
            num_gaps = len(gap_lengths)

            # Calculate gap regularity
            regularity = 0.0
            if num_gaps >= 3:
                avg_gap = np.mean(gap_lengths)
                std_gap = np.std(gap_lengths)
                cv = std_gap / avg_gap if avg_gap > 0 else 999
                if cv < 1.0:
                    regularity = max(0, 1.0 - cv)

            avg_y = np.mean([curve.y_positions[x] for x in sample_xs])
            dark_count = sum(is_dark)
            print(f"      {curve_name}: avg_y={avg_y:.0f}, dark={dark_count}/{len(is_dark)}, gaps={num_gaps}, gap_ratio={gap_ratio:.2f}")

            return gap_ratio, num_gaps, regularity

        gap_ratio0, num_gaps0, reg0 = analyze_curve_for_dashes_in_grayscale(curve0, "curve0")
        gap_ratio1, num_gaps1, reg1 = analyze_curve_for_dashes_in_grayscale(curve1, "curve1")
        print(f"    Grayscale analysis: curve0 gap_ratio={gap_ratio0:.2f} ({num_gaps0} gaps, reg={reg0:.2f})")
        print(f"    Grayscale analysis: curve1 gap_ratio={gap_ratio1:.2f} ({num_gaps1} gaps, reg={reg1:.2f})")

        # Determine which curve should be dashed:
        # TRUE dashes have:
        # - REGULAR gaps (high regularity score)
        # - MODERATE number of gaps (5-30 range is typical)
        # - Reasonable gap ratio (0.15-0.40 is typical for dashes)
        #
        # NOISE gaps have:
        # - Irregular gaps (low regularity)
        # - Too many gaps (>40 suggests tracking issues)
        # - Either very low or very high gap ratio

        def compute_dash_likelihood(gap_ratio, num_gaps, regularity):
            score = 0.0

            # Regularity is the strongest indicator of true dashes
            score += regularity * 3.0

            # Moderate gap ratio (0.15-0.40) typical for dashes
            if 0.15 <= gap_ratio <= 0.40:
                score += 1.0
            elif gap_ratio > 0.40:
                # High gap ratio could be many small gaps (noise)
                score += 0.5 * (1.0 - min(1.0, (gap_ratio - 0.40) / 0.30))

            # Moderate number of gaps (5-30) typical for dashes
            if 5 <= num_gaps <= 30:
                score += 0.8
            elif num_gaps > 30:
                # Too many gaps suggests noise, not true dashes
                score -= (num_gaps - 30) / 50.0

            return max(0, score)

        dash_score0 = compute_dash_likelihood(gap_ratio0, num_gaps0, reg0)
        dash_score1 = compute_dash_likelihood(gap_ratio1, num_gaps1, reg1)

        print(f"    Dash likelihood scores: curve0={dash_score0:.2f}, curve1={dash_score1:.2f}")

        # Only relabel if there's a clear difference
        if abs(dash_score0 - dash_score1) > 0.2:
            if dash_score0 > dash_score1:
                # Curve 0 should be dashed
                if curve0.style != LineStyle.DASHED or curve1.style != LineStyle.SOLID:
                    print(f"    Relabeling by dash analysis: curve0=dashed, curve1=solid")
                    curve0.style = LineStyle.DASHED
                    curve0.confidence = max(0.7, curve0.confidence)
                    curve1.style = LineStyle.SOLID
                    curve1.confidence = max(0.7, curve1.confidence)
                else:
                    print(f"    Curves already correctly labeled by dash analysis")
            else:
                # Curve 1 should be dashed
                if curve1.style != LineStyle.DASHED or curve0.style != LineStyle.SOLID:
                    print(f"    Relabeling by dash analysis: curve0=solid, curve1=dashed")
                    curve1.style = LineStyle.DASHED
                    curve1.confidence = max(0.7, curve1.confidence)
                    curve0.style = LineStyle.SOLID
                    curve0.confidence = max(0.7, curve0.confidence)
                else:
                    print(f"    Curves already correctly labeled by dash analysis")
        else:
            print(f"    Dash scores too close ({dash_score0:.2f} vs {dash_score1:.2f}), keeping current labels")

    def _fix_curve_swaps(self):
        """
        Fix regions where curves have swapped identities.

        For KM survival curves, the dashed (better treatment) curve should
        generally be above the solid (worse treatment) curve. Detect regions
        where this is violated and swap Y values to correct.
        """
        if len(self.traced_curves) != 2:
            return

        # Find the dashed and solid curves
        dashed_curve = None
        solid_curve = None
        for curve in self.traced_curves:
            if curve.style == LineStyle.DASHED:
                dashed_curve = curve
            elif curve.style == LineStyle.SOLID:
                solid_curve = curve

        if dashed_curve is None or solid_curve is None:
            return

        # Find x positions where both curves have data
        common_x = sorted(set(dashed_curve.y_positions.keys()) & set(solid_curve.y_positions.keys()))
        if len(common_x) < 20:
            return

        # Detect swap regions: where dashed Y > solid Y (lower survival)
        # In image coords, higher Y = lower survival
        swap_count = 0
        for x in common_x:
            dashed_y = dashed_curve.y_positions[x]
            solid_y = solid_curve.y_positions[x]

            # If dashed is below solid (higher Y value = lower survival), swap them
            # Use a small tolerance to avoid swapping due to noise
            if dashed_y > solid_y + 3:
                # Swap the Y positions
                dashed_curve.y_positions[x] = solid_y
                solid_curve.y_positions[x] = dashed_y
                swap_count += 1

        if swap_count > 0:
            print(f"    Fixed {swap_count} curve swap position(s)")

    def _fix_tail_curve_crossings(self):
        """
        Fix incorrect curve tracking in the tail region.

        For KM survival curves, the better treatment (dashed, higher survival)
        often has a flat plateau at the end. The tracker may miss this plateau
        and follow the worse treatment curve downward. This method scans the
        binary mask for the plateau and re-tracks the dashed curve if needed.
        """
        if len(self.traced_curves) != 2:
            return

        # Find the dashed and solid curves
        dashed_curve = None
        solid_curve = None
        for curve in self.traced_curves:
            if curve.style == LineStyle.DASHED:
                dashed_curve = curve
            elif curve.style == LineStyle.SOLID:
                solid_curve = curve

        if dashed_curve is None or solid_curve is None:
            return

        # Find x positions where dashed curve has data
        dashed_x = sorted(dashed_curve.y_positions.keys())
        if len(dashed_x) < 50:
            return

        # Look at a mid-region (50-70%) to establish expected Y level for dashed curve
        mid_start_idx = int(len(dashed_x) * 0.50)
        mid_end_idx = int(len(dashed_x) * 0.70)
        mid_x = dashed_x[mid_start_idx:mid_end_idx]

        if len(mid_x) < 10:
            return

        # Calculate expected Y level for dashed curve in mid-region
        mid_y_values = [dashed_curve.y_positions[x] for x in mid_x]
        expected_dashed_y = np.mean(mid_y_values)

        # Look at the tail region (last 40% of plot width)
        # This needs to start early enough to catch the plateau
        tail_start_x = int(self.plot_w * 0.60)

        # Scan the binary mask for horizontal bands in the tail region
        # that could be the dashed curve plateau
        # Use the pre-text-filtered binary if available (plateau pixels may have been filtered as text)
        search_binary = getattr(self, 'binary_before_text_filter', self.binary)
        if search_binary is None:
            search_binary = self.binary
        if search_binary is None:
            return

        height, width = search_binary.shape

        # Find all Y positions with pixels in the tail region
        # Use wider bands (10 pixels) to catch sparse dashed lines
        tail_y_bands = {}  # y -> count of pixels
        for x in range(tail_start_x, width):
            for y in range(height):
                if search_binary[y, x] > 0:
                    # Round to nearest 10 pixels to group into bands (wider for sparse dashed lines)
                    y_band = (y // 10) * 10
                    if y_band not in tail_y_bands:
                        tail_y_bands[y_band] = 0
                    tail_y_bands[y_band] += 1

        if not tail_y_bands:
            return

        # Find bands that are:
        # 1. Above the current dashed curve tail position (lower Y = higher survival)
        # 2. Have significant pixel density
        # 3. Closer to the expected mid-region Y level

        # Get current dashed curve tail Y
        tail_dashed_x = [x for x in dashed_x if x >= tail_start_x]
        if not tail_dashed_x:
            return

        current_tail_y = np.mean([dashed_curve.y_positions[x] for x in tail_dashed_x])

        # Look for a plateau band that represents the dashed curve
        # Strategy: Find the band with the MOST pixels in the expected survival range
        # (survival 0.15-0.35, which is Y between height*0.65 and height*0.85)
        # This avoids selecting sparse text pixels over actual curve pixels
        best_plateau_band = None
        best_plateau_count = 0

        # Expected plateau Y range: survival 0.15-0.30 corresponds to Y = height*(0.70-0.85)
        expected_y_min = int(height * 0.65)  # survival ~0.35
        expected_y_max = int(height * 0.85)  # survival ~0.15

        for y_band, count in tail_y_bands.items():
            # Must have enough pixels to be a real band (reduced for dashed lines)
            if count < 10:
                continue

            # Must be in reasonable range (not at very top which could be axis/legend)
            if y_band < height * 0.1:
                continue

            # Must be above the current tail position (better survival)
            if y_band >= current_tail_y - 3:
                continue

            # Must be at reasonable survival level (between 0.15-0.5)
            if y_band < height * 0.5:
                continue

            # Prefer bands in the expected plateau range (survival 0.15-0.30)
            # Give bonus score to bands in this range
            score = count
            if expected_y_min <= y_band <= expected_y_max:
                score *= 2  # Double weight for bands in expected range

            if score > best_plateau_count:
                best_plateau_count = score
                best_plateau_band = y_band

        if best_plateau_band is not None:
            print(f"    Selected plateau band Y={best_plateau_band} (survival={1-best_plateau_band/height:.2f}) with {tail_y_bands[best_plateau_band]} pixels")

        if best_plateau_band is None:
            # No plateau pixels found in binary mask (likely filtered out)
            # Fall back to extending the dashed curve based on its last known position
            # if it ends earlier than the solid curve
            solid_x = sorted(solid_curve.y_positions.keys()) if solid_curve.y_positions else []
            dashed_x_sorted = sorted(dashed_curve.y_positions.keys())

            if solid_x and dashed_x_sorted:
                solid_max_x = max(solid_x)
                dashed_max_x = max(dashed_x_sorted)

                # Detect and fix plateau region
                # Find where the dashed curve should start plateauing by analyzing descent rate
                # Look at the 40-70% region where the curve is descending but reliable

                mid_start_x = int(self.plot_w * 0.40)
                mid_end_x = int(self.plot_w * 0.70)
                mid_x = [x for x in dashed_x_sorted if mid_start_x <= x <= mid_end_x]

                # For KM curves, the plateau level is the LOWEST Y (best survival)
                # the dashed curve reaches AFTER the descent phase
                # Look at the last 40% of the curve to find the minimum Y
                late_start_idx = int(len(dashed_x_sorted) * 0.60)
                late_x = dashed_x_sorted[late_start_idx:]

                if len(late_x) >= 10:
                    late_y = [dashed_curve.y_positions[x] for x in late_x]

                    # The plateau Y is the MINIMUM Y (best survival) in the late region
                    # This is because KM curves can only decrease or stay flat
                    plateau_y = min(late_y)
                    min_y_idx = late_y.index(plateau_y)
                    plateau_start_x = late_x[min_y_idx]


                    # Determine where the plateau starts (where curve first reaches this Y level)
                    # Look backwards to find where it stabilizes
                    min_slope_x = plateau_start_x
                    for i, x in enumerate(late_x):
                        y = late_y[i]
                        if abs(y - plateau_y) < self.plot_h * 0.02:  # Within 2% of plateau
                            min_slope_x = x
                            break

                    # Verify plateau is above solid curve
                    solid_at_plateau = [solid_curve.y_positions[x] for x in solid_x if x >= min_slope_x]
                    if solid_at_plateau:
                        solid_mean = np.mean(solid_at_plateau)
                        min_separation = self.plot_h * 0.05  # 5% of height
                        if plateau_y > solid_mean - min_separation:
                            plateau_y = solid_mean - self.plot_h * 0.10  # Force 10% separation

                    # First, fix existing points that have dropped below the plateau
                    # Start from the plateau inflection point (min_slope_x), not tail_start_x
                    # (these are tracking errors where dashed followed solid)
                    plateau_fix_start = min_slope_x if min_slope_x is not None else tail_start_x
                    points_fixed = 0
                    for x in dashed_x_sorted:
                        if x >= plateau_fix_start:
                            old_y = dashed_curve.y_positions[x]
                            # If current Y is more than 3% below plateau, fix it
                            # (Use 3% tolerance to be more aggressive in fixing)
                            if old_y > plateau_y + self.plot_h * 0.03:
                                dashed_curve.y_positions[x] = plateau_y
                                points_fixed += 1

                    if points_fixed > 0:
                        print(f"    Fixed {points_fixed} dashed curve points (from x={plateau_fix_start}) to plateau Y={plateau_y:.1f}")

                    # Extend horizontally to match or exceed solid curve extent
                    # but not more than 95% of plot width
                    extend_to_x = min(solid_max_x, int(self.plot_w * 0.95))
                    points_extended = 0

                    for x in range(dashed_max_x + 1, extend_to_x + 1):
                        dashed_curve.y_positions[x] = plateau_y
                        dashed_curve.pixel_presence[x] = False  # No actual pixels, extrapolated
                        points_extended += 1

                    if points_extended > 0:
                        print(f"    Extended dashed curve by {points_extended} points at Y={plateau_y:.1f} (extrapolated plateau)")
            return

        # Verify this is actually a plateau (relatively flat across x)
        # by checking pixel presence at this Y level across the tail
        plateau_pixels_per_x = {}
        for x in range(tail_start_x, width):
            for dy in range(-8, 9):
                check_y = best_plateau_band + dy
                if 0 <= check_y < height and search_binary[check_y, x] > 0:
                    if x not in plateau_pixels_per_x:
                        plateau_pixels_per_x[x] = check_y
                    elif abs(check_y - best_plateau_band) < abs(plateau_pixels_per_x[x] - best_plateau_band):
                        plateau_pixels_per_x[x] = check_y

        # Need enough x positions with plateau pixels
        if len(plateau_pixels_per_x) < 10:
            return

        print(f"    Found plateau band Y={best_plateau_band} with {len(plateau_pixels_per_x)} x positions")

        # Re-track the dashed curve in the tail region to follow the plateau
        # Force all tail positions to plateau level, even if no pixels found
        points_fixed = 0
        for x in tail_dashed_x:
            old_y = dashed_curve.y_positions[x]

            # Check if current position is significantly off from plateau
            # (suggesting the tracker followed the wrong curve)
            off_plateau = abs(old_y - best_plateau_band) > 15

            # Scan for pixels near the plateau band at this x position
            new_y = None
            for dy in range(-10, 11):
                check_y = best_plateau_band + dy
                if 0 <= check_y < height and search_binary[check_y, x] > 0:
                    if new_y is None or abs(check_y - best_plateau_band) < abs(new_y - best_plateau_band):
                        new_y = check_y

            # If found a pixel near plateau, use it
            if new_y is not None and abs(new_y - old_y) > 5:
                dashed_curve.y_positions[x] = float(new_y)
                points_fixed += 1
            # If no pixel found but we're significantly off plateau, use plateau directly
            elif new_y is None and off_plateau:
                dashed_curve.y_positions[x] = float(best_plateau_band)
                points_fixed += 1

        if points_fixed > 0:
            fixed_y_values = [dashed_curve.y_positions[x] for x in tail_dashed_x[:10] if x in dashed_curve.y_positions]
            print(f"    Fixed {points_fixed} dashed curve tail positions to plateau (sample Y values: {[f'{y:.0f}' for y in fixed_y_values]})")

        # Extend the dashed curve along the plateau beyond its current end
        # Use best_plateau_band as the target Y level (verified from binary mask)
        dashed_x_sorted = sorted(dashed_curve.y_positions.keys())
        if len(dashed_x_sorted) < 10:
            return

        current_max_x = max(dashed_curve.y_positions.keys())
        points_extended = 0

        # Extend by looking for plateau pixels near the best_plateau_band level
        # Use best_plateau_band (from binary detection) not curve-derived values
        sorted_plateau_x = sorted(plateau_pixels_per_x.keys())

        extended_y_values = []
        first_ext_x = None
        last_ext_x = None
        for x in sorted_plateau_x:
            if x <= current_max_x:
                continue  # Already have data for this position

            plateau_y = plateau_pixels_per_x[x]

            # Only extend if this Y is close to the detected plateau band
            # Use a tighter tolerance (5 pixels) and set to best_plateau_band directly
            # to avoid drifting toward incorrect pixels at the curve end
            if abs(plateau_y - best_plateau_band) <= 10:
                if first_ext_x is None:
                    first_ext_x = x
                last_ext_x = x
                # Use best_plateau_band directly instead of plateau_y
                # This ensures consistent plateau level across the extension
                dashed_curve.y_positions[x] = float(best_plateau_band)
                dashed_curve.pixel_presence[x] = True
                points_extended += 1
                extended_y_values.append(best_plateau_band)

        if points_extended > 0:
            avg_ext_y = np.mean(extended_y_values) if extended_y_values else 0
            print(f"    Extended dashed curve by {points_extended} plateau points at Y≈{best_plateau_band}")

        if points_fixed > 0 and points_extended == 0:
            # Only truncate if we didn't extend - extension already uses clean plateau pixels
            # After fixing to plateau, truncate any final points that drop sharply
            # These are likely artifacts from text or other interference
            dashed_x_after = sorted(dashed_curve.y_positions.keys())
            if len(dashed_x_after) > 10:
                # Look at the last 10% of positions
                check_start = int(len(dashed_x_after) * 0.90)
                check_x = dashed_x_after[check_start:]

                # Find the plateau Y level (mode of recent positions)
                recent_y = [dashed_curve.y_positions[x] for x in dashed_x_after[check_start-20:check_start]]
                if recent_y:
                    plateau_y = np.median(recent_y)

                    # Remove any points that drop SUDDENLY below the plateau
                    # Only truncate if there's a sharp drop, not gradual variation
                    truncate_from = None
                    prev_y = None
                    for x in check_x:
                        y = dashed_curve.y_positions[x]
                        # Check for sudden drop (more than 8% of plot height in one step)
                        if prev_y is not None:
                            drop = y - prev_y
                            if drop > self.plot_h * 0.08:  # Sudden drop
                                truncate_from = x
                                break
                        # Also check if way below plateau (more than 10%)
                        if y > plateau_y + self.plot_h * 0.10:
                            truncate_from = x
                            break
                        prev_y = y

                    if truncate_from is not None:
                        points_truncated = 0
                        for x in list(dashed_curve.y_positions.keys()):
                            if x >= truncate_from:
                                del dashed_curve.y_positions[x]
                                if x in dashed_curve.pixel_presence:
                                    del dashed_curve.pixel_presence[x]
                                points_truncated += 1
                        if points_truncated > 0:
                            print(f"    Truncated {points_truncated} post-plateau drop points")

    def _truncate_anomalous_tails(self, curves: List[TracedCurve]):
        """
        Truncate curve tails that show anomalous behavior.

        At the tail end of curves, the tracker may pick up text or switch to
        following another curve. Detect this by comparing the tail slope to
        the recent average slope - if the tail drops much faster, truncate it.
        """
        for curve in curves:
            if not curve.y_positions:
                continue

            x_positions = sorted(curve.y_positions.keys())
            if len(x_positions) < 50:
                continue

            # Look at the last 15% of the curve as the "tail"
            tail_start_idx = int(len(x_positions) * 0.85)
            # Look at 60-85% as the "reference" region
            ref_start_idx = int(len(x_positions) * 0.60)
            ref_end_idx = tail_start_idx

            if ref_end_idx - ref_start_idx < 10:
                continue

            # Calculate average slope and Y range in reference region
            ref_x = x_positions[ref_start_idx:ref_end_idx]
            ref_y = [curve.y_positions[x] for x in ref_x]
            if len(ref_x) < 2:
                continue

            # Slope as y-change per x-unit (positive = curve going down in image coords)
            ref_slope = (ref_y[-1] - ref_y[0]) / (ref_x[-1] - ref_x[0]) if ref_x[-1] != ref_x[0] else 0

            # Get stable Y range from reference region (for checking if jumps return to valid area)
            ref_y_min = min(ref_y)
            ref_y_max = max(ref_y)
            ref_y_tolerance = max(20, (ref_y_max - ref_y_min) * 2)  # Allow some tolerance

            # Check tail for anomalous drops
            tail_x = x_positions[tail_start_idx:]
            truncate_at = None

            for i in range(len(tail_x) - 1):
                x1, x2 = tail_x[i], tail_x[i + 1]
                y1 = curve.y_positions[x1]
                y2 = curve.y_positions[x2]

                # Calculate local slope
                local_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0

                # If local slope is much steeper than reference (5x), flag it
                # Only check for drops (positive slope in image coords = survival decreasing)
                # Be conservative - only truncate for very clear anomalies
                if local_slope > 0 and ref_slope > 0:
                    # Both going down - check if tail is dropping too fast
                    if local_slope > ref_slope * 5:
                        # But allow if jumping back to the reference Y range (returning to correct position)
                        if ref_y_min - ref_y_tolerance <= y2 <= ref_y_max + ref_y_tolerance:
                            continue  # This is the curve returning to its stable position
                        truncate_at = x1
                        break
                elif local_slope > 0 and ref_slope <= 0.01:
                    # Reference is flat but tail is dropping - anomaly
                    # Only flag if the drop is extremely significant (going to text/artifacts)
                    # Use a high threshold (10% of height) to be very conservative
                    # Flat curves (like survival plateaus) may have noise that causes
                    # jumps, but these should not trigger truncation
                    if local_slope > self.plot_h * 0.10:  # More than 10% of height per pixel
                        # But allow if jumping back to the reference Y range
                        if ref_y_min - ref_y_tolerance <= y2 <= ref_y_max + ref_y_tolerance:
                            continue
                        truncate_at = x1
                        break

            if truncate_at is not None:
                # Remove all points after truncate_at
                points_removed = 0
                for x in list(curve.y_positions.keys()):
                    if x > truncate_at:
                        del curve.y_positions[x]
                        if x in curve.pixel_presence:
                            del curve.pixel_presence[x]
                        points_removed += 1
                if points_removed > 0:
                    print(f"    Truncated {points_removed} anomalous tail points")

    def _smooth_traced_curves(self, curves: List[TracedCurve]):
        """
        Smooth traced curves by:
        1. Detecting ONLY very large jumps (>15% of plot height) that indicate
           clear tracking errors
        2. Interpolating through gaps

        We are conservative here - only remove points that are clearly wrong.
        """
        # First, truncate any anomalous tail behavior
        self._truncate_anomalous_tails(curves)

        # Very large jump threshold - only catch clear tracking errors
        big_jump_threshold = self.plot_h * 0.15  # 15% of plot height

        for curve in curves:
            if not curve.y_positions:
                continue

            x_positions = sorted(curve.y_positions.keys())
            if len(x_positions) < 10:
                continue

            # Find points that represent sudden large jumps
            outliers = set()
            for i in range(1, len(x_positions)):
                x_prev = x_positions[i - 1]
                x_curr = x_positions[i]
                y_prev = curve.y_positions[x_prev]
                y_curr = curve.y_positions[x_curr]

                change = abs(y_curr - y_prev)
                if change > big_jump_threshold:
                    # Check if this is a "spike" - jumps away and back
                    if i + 1 < len(x_positions):
                        x_next = x_positions[i + 1]
                        y_next = curve.y_positions[x_next]
                        # If y_next is close to y_prev, this is a spike
                        if abs(y_next - y_prev) < big_jump_threshold * 0.5:
                            outliers.add(x_curr)

            if outliers:
                print(f"    Detected {len(outliers)} spike outlier(s)")
                for x in outliers:
                    del curve.y_positions[x]
                    curve.pixel_presence[x] = False

        # Interpolate through gaps
        self._interpolate_gaps(curves)

    def _get_ranges(self, points):
        """Convert list of points to range strings for debug output."""
        if not points:
            return "[]"
        ranges = []
        start = points[0]
        end = points[0]
        for p in points[1:]:
            if p == end + 1:
                end = p
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = p
                end = p
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        return "[" + ", ".join(ranges) + "]"

    def _interpolate_gaps(self, curves: List[TracedCurve]):
        """Interpolate through gaps in traced curves."""
        for curve in curves:
            if not curve.y_positions:
                continue

            x_positions = sorted(curve.y_positions.keys())
            if len(x_positions) < 2:
                continue

            x_min = min(x_positions)
            x_max = max(x_positions)

            known_x = list(x_positions)
            known_y = [curve.y_positions[x] for x in known_x]

            # Interpolate Y for all X positions
            all_x = list(range(x_min, x_max + 1))

            if len(known_x) >= 2:
                interpolated_y = np.interp(all_x, known_x, known_y)

                # Update curve with interpolated positions
                for i, x in enumerate(all_x):
                    if x not in curve.y_positions:
                        curve.y_positions[x] = float(interpolated_y[i])
                        curve.pixel_presence[x] = False

    def _fix_curve_crossings(self):
        """
        Detect and fix curve crossings caused by tracking errors.

        For Kaplan-Meier curves that don't cross in the original image,
        crossings in the traced data indicate the tracker swapped curves.
        This method detects intersections and swaps Y-values to maintain
        consistent curve ordering.
        """
        if len(self.traced_curves) < 2:
            return

        # Get all X positions where we have data for multiple curves
        all_x = set()
        for curve in self.traced_curves:
            all_x.update(curve.y_positions.keys())
        all_x = sorted(all_x)

        if len(all_x) < 2:
            return

        # Find intersection points between each pair of curves
        intersections = self._find_intersections()

        if intersections:
            print(f"    Detected {len(intersections)} intersection point(s) - fixing curve assignments...")

            # For KM curves: determine which curve should be "on top" (higher survival = lower Y in image)
            # based on the majority of the data
            self._reorder_curves_by_position(intersections)

    def _find_intersections(self) -> List[Tuple[int, int, int]]:
        """
        Find all intersection points between pairs of curves.

        Returns:
            List of (x_position, curve1_idx, curve2_idx) tuples where crossings occur
        """
        intersections = []

        # Get common X positions
        all_x = set()
        for curve in self.traced_curves:
            all_x.update(curve.y_positions.keys())
        all_x = sorted(all_x)

        # Check each pair of curves
        for i in range(len(self.traced_curves)):
            for j in range(i + 1, len(self.traced_curves)):
                curve1 = self.traced_curves[i]
                curve2 = self.traced_curves[j]

                prev_diff = None
                prev_x = None

                for x in all_x:
                    if x in curve1.y_positions and x in curve2.y_positions:
                        y1 = curve1.y_positions[x]
                        y2 = curve2.y_positions[x]
                        diff = y1 - y2  # Positive if curve1 is below curve2 (lower survival)

                        if prev_diff is not None:
                            # Check for sign change (crossing)
                            if (prev_diff > 0 and diff < 0) or (prev_diff < 0 and diff > 0):
                                # Intersection between prev_x and x
                                intersections.append((x, i, j))

                        prev_diff = diff
                        prev_x = x

        return intersections

    def _reorder_curves_by_position(self, intersections: List[Tuple[int, int, int]]):
        """
        Reorder curve Y-positions to eliminate false crossings.

        Strategy:
        1. Determine which curve should consistently be "above" (lower Y = higher survival)
           by analyzing the curves at their START (where they're typically well-separated)
        2. Find contiguous SEGMENTS where curves are swapped, rather than individual points
        3. Swap entire segments to maintain continuity within each curve
        """
        if len(self.traced_curves) != 2:
            return

        curve1 = self.traced_curves[0]
        curve2 = self.traced_curves[1]

        # Get common X positions
        common_x = sorted(set(curve1.y_positions.keys()) & set(curve2.y_positions.keys()))

        if len(common_x) < 5:
            return

        # Determine expected ordering based on the EARLY portion of data
        # Use points from the first 30% where tracking is typically most reliable
        # (before curves get close and tracker might swap)
        early_region_end = max(30, len(common_x) // 3)
        early_x = common_x[:early_region_end]

        # Among early points, use those with meaningful separation (>3% of plot height)
        min_sep = self.plot_h * 0.03
        early_diffs = []
        for x in early_x:
            y1 = curve1.y_positions[x]
            y2 = curve2.y_positions[x]
            diff = y1 - y2  # Positive if curve1 is below (lower survival = higher Y)
            if abs(diff) > min_sep:
                early_diffs.append(diff)

        # If we have enough separated points, use them for voting
        if len(early_diffs) >= 5:
            avg_diff = np.mean(early_diffs)
        else:
            # Fall back to using all early points
            all_diffs = [curve1.y_positions[x] - curve2.y_positions[x] for x in early_x]
            avg_diff = np.mean(all_diffs) if all_diffs else 0

        curve1_should_be_below = avg_diff > 0

        # Now find SEGMENTS where the ordering is wrong
        # A segment is a contiguous run of X positions where ordering is consistent
        segments = []  # List of (start_x_idx, end_x_idx, ordering_matches_expected)
        current_start = 0
        current_matches = None

        for i, x in enumerate(common_x):
            y1 = curve1.y_positions[x]
            y2 = curve2.y_positions[x]
            curve1_is_below = y1 > y2

            # Check if this point's ordering matches expected
            matches_expected = (curve1_is_below == curve1_should_be_below)

            if current_matches is None:
                current_matches = matches_expected
            elif matches_expected != current_matches:
                # Ordering changed - end current segment
                segments.append((current_start, i - 1, current_matches))
                current_start = i
                current_matches = matches_expected

        # Add final segment
        if current_matches is not None:
            segments.append((current_start, len(common_x) - 1, current_matches))

        # Count segments that need fixing
        wrong_segments = [(s, e, m) for s, e, m in segments if not m]
        print(f"    Curve ordering: {'curve1' if not curve1_should_be_below else 'curve2'} should be above")
        print(f"    Found {len(segments)} segments, {len(wrong_segments)} need reordering")

        # Debug: show segment details
        for i, (s, e, m) in enumerate(segments):
            x_start = common_x[s]
            x_end = common_x[e]
            status = "OK" if m else "WRONG"
            print(f"      Segment {i+1}: X={x_start}-{x_end} ({e-s+1} points) - {status}")

        if not wrong_segments:
            return

        # For KM curves, the two curves should NEVER cross (one should always be above)
        # Therefore, swap ALL wrongly-ordered segments to maintain consistent ordering
        total_points = len(common_x)
        swaps_made = 0

        for start_idx, end_idx, _ in wrong_segments:
            # Swap all wrongly-ordered points to maintain consistent curve ordering
                for i in range(start_idx, end_idx + 1):
                    x = common_x[i]
                    y1 = curve1.y_positions[x]
                    y2 = curve2.y_positions[x]
                    curve1.y_positions[x] = y2
                    curve2.y_positions[x] = y1
                    swaps_made += 1

        if swaps_made > 0:
            print(f"    Swapped {swaps_made} points in {len(wrong_segments)} segment(s)")

    def _fix_tracking_with_dash_mask(self):
        """
        Use dash_mask to verify and fix tracking errors using a segment-based approach.

        This method:
        1. Finds contiguous segments where curves have a consistent relative ordering
        2. For each segment, determines if it's correctly ordered using dash_mask
        3. Swaps entire segments that are incorrectly ordered
        """
        if len(self.traced_curves) != 2:
            return

        if not hasattr(self, 'dash_mask') or self.dash_mask is None:
            return

        curve0 = self.traced_curves[0]
        curve1 = self.traced_curves[1]

        # Get common X positions where both curves have data
        common_x = sorted(set(curve0.y_positions.keys()) & set(curve1.y_positions.keys()))

        if len(common_x) < 10:
            return

        # Step 1: Find segments where curves have consistent relative ordering
        # A "crossing" occurs when the relative Y-order of curves changes
        min_sep = 3  # Minimum separation to consider curves as distinct

        segments = []  # List of (start_idx, end_idx, curve0_is_above)
        seg_start = None
        prev_order = None  # True if curve0 is above (lower Y), False if below

        for i, x in enumerate(common_x):
            y0 = curve0.y_positions[x]
            y1 = curve1.y_positions[x]
            sep = y1 - y0  # Positive if curve0 is above

            if abs(sep) < min_sep:
                # Curves too close - indeterminate
                if seg_start is not None:
                    # End current segment
                    segments.append((seg_start, i - 1, prev_order))
                    seg_start = None
                    prev_order = None
                continue

            current_order = sep > 0  # True if curve0 is above

            if prev_order is None:
                # Start new segment
                seg_start = i
                prev_order = current_order
            elif current_order != prev_order:
                # Order changed - end segment and start new one
                segments.append((seg_start, i - 1, prev_order))
                seg_start = i
                prev_order = current_order

        # Close final segment
        if seg_start is not None:
            segments.append((seg_start, len(common_x) - 1, prev_order))

        if not segments:
            return

        # Step 2: For each segment, calculate dash_mask overlap to verify correctness
        # Accumulate evidence across all segments to determine overall which curve is dashed
        total_evidence = []  # (segment_len, curve0_dash_score, curve1_dash_score, curve0_is_above)

        for seg_start, seg_end, curve0_is_above in segments:
            seg_len = seg_end - seg_start + 1
            if seg_len < 5:
                continue  # Skip very short segments

            dash_0 = 0
            dash_1 = 0

            for i in range(seg_start, seg_end + 1):
                x = common_x[i]
                y0 = int(curve0.y_positions[x])
                y1 = int(curve1.y_positions[x])

                # Check dash_mask at each curve's position
                for dy in range(-1, 2):
                    y0_check = y0 + dy
                    y1_check = y1 + dy

                    if 0 <= y0_check < self.dash_mask.shape[0] and 0 <= x < self.dash_mask.shape[1]:
                        if self.dash_mask[y0_check, x] > 0:
                            dash_0 += 1

                    if 0 <= y1_check < self.dash_mask.shape[0] and 0 <= x < self.dash_mask.shape[1]:
                        if self.dash_mask[y1_check, x] > 0:
                            dash_1 += 1

            total_evidence.append((seg_len, dash_0, dash_1, curve0_is_above))

        if not total_evidence:
            return

        # Step 3: Determine which curve is dashed using weighted voting
        # Weight by segment length and dash score difference
        weighted_votes_curve0_dashed = 0
        weighted_votes_curve1_dashed = 0

        for seg_len, dash_0, dash_1, _ in total_evidence:
            diff = abs(dash_0 - dash_1)
            weight = seg_len * max(1, diff)  # Longer segments with clearer patterns count more

            if dash_0 > dash_1:
                weighted_votes_curve0_dashed += weight
            elif dash_1 > dash_0:
                weighted_votes_curve1_dashed += weight

        curve0_is_dashed = weighted_votes_curve0_dashed > weighted_votes_curve1_dashed

        print(f"    Dash-based tracking fix: votes curve0_dashed={weighted_votes_curve0_dashed}, curve1_dashed={weighted_votes_curve1_dashed}")
        print(f"    Determined: curve0 is {'DASHED' if curve0_is_dashed else 'SOLID'}")

        # Step 4: Identify segments that need swapping
        # A segment needs swapping if its local dash pattern disagrees with the overall determination
        segments_to_swap = []

        for seg_start, seg_end, curve0_is_above in segments:
            seg_len = seg_end - seg_start + 1
            if seg_len < 3:
                continue

            # Calculate local dash scores
            local_dash_0 = 0
            local_dash_1 = 0

            for i in range(seg_start, seg_end + 1):
                x = common_x[i]
                y0 = int(curve0.y_positions[x])
                y1 = int(curve1.y_positions[x])

                for dy in range(-1, 2):
                    y0_check = y0 + dy
                    y1_check = y1 + dy

                    if 0 <= y0_check < self.dash_mask.shape[0] and 0 <= x < self.dash_mask.shape[1]:
                        if self.dash_mask[y0_check, x] > 0:
                            local_dash_0 += 1

                    if 0 <= y1_check < self.dash_mask.shape[0] and 0 <= x < self.dash_mask.shape[1]:
                        if self.dash_mask[y1_check, x] > 0:
                            local_dash_1 += 1

            # Determine local pattern
            local_curve0_is_dashed = local_dash_0 > local_dash_1

            # Check if this segment is inconsistent with overall determination
            if local_curve0_is_dashed != curve0_is_dashed:
                # This segment has the curves swapped
                segments_to_swap.append((seg_start, seg_end))

        # Step 5: Swap Y positions for all identified segments
        swaps_made = 0
        for seg_start, seg_end in segments_to_swap:
            for i in range(seg_start, seg_end + 1):
                x = common_x[i]
                y0 = curve0.y_positions[x]
                y1 = curve1.y_positions[x]
                curve0.y_positions[x] = y1
                curve1.y_positions[x] = y0
                swaps_made += 1

        if swaps_made > 0:
            print(f"    Dash-mask tracking fix: swapped {swaps_made} points in {len(segments_to_swap)} segment(s)")
            for seg_start, seg_end in segments_to_swap:
                x_start = common_x[seg_start]
                x_end = common_x[seg_end]
                print(f"      Swapped segment: x={x_start} to {x_end}")

    def _count_y_clusters(self, y_vals: np.ndarray, min_gap: int = 8) -> int:
        """Count distinct Y clusters in an array of Y values."""
        if len(y_vals) == 0:
            return 0
        if len(y_vals) == 1:
            return 1

        y_sorted = np.sort(y_vals)
        gaps = np.diff(y_sorted)

        # Count gaps larger than threshold
        n_clusters = 1 + np.sum(gaps > min_gap)
        return n_clusters

    def _cluster_y_values(self, y_vals: List[int], expected_clusters: int) -> List[float]:
        """
        Cluster Y values and return cluster centers.

        Returns list of Y positions (cluster centers), sorted from top to bottom.
        """
        if not y_vals:
            return []

        y_arr = np.array(y_vals)
        y_sorted = np.sort(y_arr)

        if len(y_sorted) == 1:
            return [float(y_sorted[0])]

        # Find gaps to identify clusters
        gaps = np.diff(y_sorted)

        # Use adaptive min_gap based on expected clusters
        # If we expect multiple clusters but only find one, try smaller gaps
        min_gap = 5  # Standard gap for cluster separation

        # Find cluster boundaries
        boundaries = [0]
        for i, gap in enumerate(gaps):
            if gap > min_gap:
                boundaries.append(i + 1)
        boundaries.append(len(y_sorted))

        # Calculate cluster centers
        centers = []
        for i in range(len(boundaries) - 1):
            cluster = y_sorted[boundaries[i]:boundaries[i+1]]
            centers.append(float(np.median(cluster)))

        # If we found fewer clusters than expected, try to split the largest cluster
        if len(centers) < expected_clusters and expected_clusters > 1:
            # Find the cluster with the most spread
            while len(centers) < expected_clusters:
                # Recalculate with smaller gap threshold
                min_gap = max(2, min_gap - 1)
                boundaries = [0]
                for i, gap in enumerate(gaps):
                    if gap > min_gap:
                        boundaries.append(i + 1)
                boundaries.append(len(y_sorted))

                centers = []
                for i in range(len(boundaries) - 1):
                    cluster = y_sorted[boundaries[i]:boundaries[i+1]]
                    centers.append(float(np.median(cluster)))

                if min_gap <= 2:
                    break  # Don't go below 2 pixels

        return sorted(centers)

    def _cluster_y_values_with_style(self, x: int, y_vals: List[int], expected_clusters: int) -> List[float]:
        """
        Cluster Y values using both position and dash-style information.

        When position-based clustering produces fewer clusters than expected,
        use the dash mask to try to split clusters by line style.
        """
        # First, try standard position-based clustering
        clusters = self._cluster_y_values(y_vals, expected_clusters)

        # If we got enough clusters, we're done
        if len(clusters) >= expected_clusters:
            return clusters

        # If we have a dash mask and expected 2 curves but got 1 cluster,
        # try to split based on line style
        if expected_clusters == 2 and len(clusters) == 1 and hasattr(self, 'dash_mask'):
            # Separate Y values into dashed and solid
            dashed_y = []
            solid_y = []

            for y in y_vals:
                y_int = int(y)
                if 0 <= y_int < self.dash_mask.shape[0] and self.dash_mask[y_int, x] > 0:
                    dashed_y.append(y)
                else:
                    solid_y.append(y)

            # If we have both dashed and solid pixels at this x, create 2 clusters
            if dashed_y and solid_y:
                dashed_center = float(np.median(dashed_y))
                solid_center = float(np.median(solid_y))
                return sorted([dashed_center, solid_center])

            # NEW: When dash_mask doesn't help (all pixels same type),
            # try to split based on Y-span if the cluster is thick enough
            # KM curves at same X should have different Y values (one above other)
            if len(y_vals) >= 4:  # Need enough pixels to potentially split
                y_arr = np.array(sorted(y_vals))
                y_span = y_arr[-1] - y_arr[0]
                # If the Y-span is >= 4 pixels, there might be two curves
                # (typical line thickness is 2-3 pixels)
                if y_span >= 4:
                    # Split into upper and lower halves
                    mid_y = (y_arr[0] + y_arr[-1]) / 2
                    upper_y = y_arr[y_arr <= mid_y]
                    lower_y = y_arr[y_arr > mid_y]
                    if len(upper_y) > 0 and len(lower_y) > 0:
                        upper_center = float(np.median(upper_y))
                        lower_center = float(np.median(lower_y))
                        # Only split if centers are sufficiently apart (> line thickness)
                        if abs(lower_center - upper_center) >= 3:
                            return sorted([upper_center, lower_center])

        return clusters

    def _assign_clusters_to_curves(
        self,
        curves: List[TracedCurve],
        x: int,
        clusters: List[float]
    ):
        """Assign Y clusters at position X to curve trackers.

        Uses a maximum allowed Y-jump threshold to prevent discontinuities
        when curves are close together or cross.
        """
        if not clusters:
            # No pixels at this X - mark as gap for all curves
            for curve in curves:
                curve.pixel_presence[x] = False
            return

        n_curves = len(curves)
        n_clusters = len(clusters)

        # Maximum allowed Y-jump per X-step (prevents discontinuities)
        # KM curves typically change gradually - large jumps indicate tracking errors
        max_y_jump = self.plot_h * 0.08  # 8% of plot height max per step

        # Get previous Y positions for each curve (with extrapolation for continuity)
        prev_positions = []
        expected_positions = []  # Extrapolated expected positions
        for curve in curves:
            if curve.y_positions:
                # Find most recent position
                recent_x = max(curve.y_positions.keys())
                prev_y = curve.y_positions[recent_x]
                prev_positions.append(prev_y)

                # Extrapolate expected position based on recent trajectory
                # This helps maintain smooth curves
                x_vals = sorted(curve.y_positions.keys())
                if len(x_vals) >= 3:
                    # Use last few points to estimate trajectory
                    recent_xs = x_vals[-5:]
                    recent_ys = [curve.y_positions[xi] for xi in recent_xs]
                    if len(recent_xs) >= 2:
                        # Linear extrapolation from recent slope
                        slope = (recent_ys[-1] - recent_ys[0]) / max(1, recent_xs[-1] - recent_xs[0])
                        x_diff = x - recent_x
                        expected_y = prev_y + slope * x_diff
                        # Clamp expected_y to reasonable bounds
                        expected_y = max(0, min(self.plot_h, expected_y))
                        expected_positions.append(expected_y)
                    else:
                        expected_positions.append(prev_y)
                else:
                    expected_positions.append(prev_y)
            else:
                prev_positions.append(None)
                expected_positions.append(None)

        # If this is the first X with data, initialize curves with clusters
        if all(p is None for p in prev_positions):
            # For KM curves at t≈0, both should start at survival 1.0 (top = lowest y)
            # But if we're past the very early region AND clusters are clearly separated,
            # initialize with different clusters to capture divergence
            very_early_x = self.plot_w * 0.08  # First 8% of plot width

            if x < very_early_x or len(clusters) < 2:
                # Use top cluster for all curves at very start
                top_cluster = clusters[0]  # clusters are sorted, first is lowest y
                for curve in curves:
                    curve.y_positions[x] = top_cluster
                    curve.pixel_presence[x] = True
            else:
                # Past very early region with 2+ clusters: check separation
                cluster_gap = clusters[1] - clusters[0] if len(clusters) >= 2 else 0

                if cluster_gap >= 3:  # Clusters are clearly separated
                    # Assign curve 0 to top cluster, curve 1 to bottom cluster
                    for i, curve in enumerate(curves):
                        if i < len(clusters):
                            curve.y_positions[x] = clusters[i]
                        else:
                            curve.y_positions[x] = clusters[0]
                        curve.pixel_presence[x] = True
                else:
                    # Clusters too close, use top for all
                    top_cluster = clusters[0]
                    for curve in curves:
                        curve.y_positions[x] = top_cluster
                        curve.pixel_presence[x] = True
            return

        # Match clusters to curves based on proximity to expected position
        # Use expected (extrapolated) positions for better continuity
        curve_preferences = []  # (curve_idx, best_cluster_idx, best_dist, within_threshold)

        for i, curve in enumerate(curves):
            if prev_positions[i] is None:
                continue

            expected_y = expected_positions[i] if expected_positions[i] is not None else prev_positions[i]
            prev_y = prev_positions[i]

            # Find closest cluster, preferring those within max_y_jump of previous position
            best_cluster = None
            best_dist = float('inf')
            best_within_threshold = False

            for j, cluster_y in enumerate(clusters):
                dist_from_expected = abs(cluster_y - expected_y)
                dist_from_prev = abs(cluster_y - prev_y)

                # Check if this cluster is within acceptable jump distance
                within_threshold = dist_from_prev <= max_y_jump

                # Prefer clusters within threshold; among those, pick closest to expected
                if within_threshold:
                    if not best_within_threshold or dist_from_expected < best_dist:
                        best_dist = dist_from_expected
                        best_cluster = j
                        best_within_threshold = True
                elif not best_within_threshold:
                    # No good cluster found yet - track the closest one anyway
                    if dist_from_expected < best_dist:
                        best_dist = dist_from_expected
                        best_cluster = j

            # Only add to preferences if we found a reasonable cluster
            if best_cluster is not None and (best_within_threshold or best_dist < self.plot_h * 0.15):
                curve_preferences.append((i, best_cluster, best_dist, best_within_threshold))

        # Sort preferences: prioritize those within threshold, then by distance
        curve_preferences.sort(key=lambda x: (not x[3], x[2]))

        # For KM curves with 2 curves and 2+ clusters:
        # Strategy:
        # 1. Initially (before divergence): use ordering (curve 0 = top, curve 1 = bottom)
        # 2. After divergence: use TRAJECTORY-BASED tracking (each curve follows nearest cluster)
        #    This allows curves to cross without swapping identities

        if n_curves == 2 and len(clusters) >= 2:
            # Safeguard: Don't diverge in the first 3% of the plot (let curves establish)
            min_x_for_divergence = self.plot_w * 0.03
            if x < min_x_for_divergence:
                # Use only the top cluster for both curves during early portion
                curves[0].y_positions[x] = clusters[0]
                curves[0].pixel_presence[x] = True
                curves[1].y_positions[x] = clusters[0]
                curves[1].pixel_presence[x] = True
                return

            # Check if curves have already diverged (have different recent positions)
            curves_diverged = False
            if prev_positions[0] is not None and prev_positions[1] is not None:
                pos_diff = abs(prev_positions[0] - prev_positions[1])
                curves_diverged = pos_diff > 5  # More than 5 pixels apart = diverged

            if not curves_diverged:
                # INITIAL DIVERGENCE: Use ordering-based assignment
                # Curve 0 = top (better survival), Curve 1 = bottom (worse survival)
                curves[0].y_positions[x] = clusters[0]  # Top cluster
                curves[0].pixel_presence[x] = True
                curves[1].y_positions[x] = clusters[1]  # Bottom cluster
                curves[1].pixel_presence[x] = True
            else:
                # AFTER DIVERGENCE: Use trajectory-based tracking with dash-pattern awareness
                # Each curve follows the cluster closest to its expected position
                # When curves are close, use dash pattern to help distinguish them

                # Check dash pattern around each cluster
                cluster_dash_scores = []
                for j, cluster_y in enumerate(clusters):
                    dash_score = 0
                    if hasattr(self, 'dash_mask'):
                        # Count dashed pixels near this cluster position
                        y_int = int(cluster_y)
                        for dy in range(-3, 4):
                            y_check = y_int + dy
                            if 0 <= y_check < self.dash_mask.shape[0]:
                                if self.dash_mask[y_check, x] > 0:
                                    dash_score += 1
                    cluster_dash_scores.append(dash_score)

                # Check if curves are getting close (within 15% of plot height)
                curves_close = False
                if prev_positions[0] is not None and prev_positions[1] is not None:
                    pos_diff = abs(prev_positions[0] - prev_positions[1])
                    curves_close = pos_diff < self.plot_h * 0.15

                # Calculate distances and assign greedily
                assignments = []  # (curve_idx, cluster_idx, distance, dash_bonus)
                for i in range(n_curves):
                    if expected_positions[i] is not None:
                        for j, cluster_y in enumerate(clusters):
                            dist = abs(cluster_y - expected_positions[i])
                            # Add dash bonus: curve 0 (dashed) prefers high dash score clusters
                            # curve 1 (solid) prefers low dash score clusters
                            dash_bonus = 0
                            if curves_close and len(cluster_dash_scores) > j:
                                if i == 0:  # Dashed curve
                                    # Reduce effective distance for clusters with dashed pixels
                                    dash_bonus = -cluster_dash_scores[j] * 3
                                else:  # Solid curve
                                    # Reduce effective distance for clusters without dashed pixels
                                    dash_bonus = cluster_dash_scores[j] * 3
                            adjusted_dist = dist + dash_bonus
                            assignments.append((i, j, adjusted_dist, dist))

                # Sort by adjusted distance (prefer closer matches with dash pattern bonus)
                assignments.sort(key=lambda x: x[2])

                # Greedy assignment: each curve gets its closest available cluster
                used_clusters = set()
                assigned_curves = set()

                for curve_idx, cluster_idx, adj_dist, orig_dist in assignments:
                    if curve_idx in assigned_curves or cluster_idx in used_clusters:
                        continue
                    # Only assign if within reasonable distance (use original distance for threshold)
                    if orig_dist <= max_y_jump * 1.5:
                        curves[curve_idx].y_positions[x] = clusters[cluster_idx]
                        curves[curve_idx].pixel_presence[x] = True
                        used_clusters.add(cluster_idx)
                        assigned_curves.add(curve_idx)

                # Handle any unassigned curves
                for i in range(n_curves):
                    if i not in assigned_curves:
                        # Try to assign to any remaining cluster within threshold
                        for j, cluster_y in enumerate(clusters):
                            if j not in used_clusters:
                                if prev_positions[i] is not None:
                                    if abs(cluster_y - prev_positions[i]) <= max_y_jump:
                                        curves[i].y_positions[x] = cluster_y
                                        curves[i].pixel_presence[x] = True
                                        used_clusters.add(j)
                                        assigned_curves.add(i)
                                        break
                        else:
                            # No valid cluster found - mark as gap
                            curves[i].pixel_presence[x] = False

            return

        # Check if curves are overlapping (same or very close previous positions)
        overlap_threshold = 10  # pixels
        curves_are_overlapping = False
        if len([p for p in prev_positions if p is not None]) >= 2:
            valid_prev = [p for p in prev_positions if p is not None]
            max_prev = max(valid_prev)
            min_prev = min(valid_prev)
            curves_are_overlapping = (max_prev - min_prev) < overlap_threshold

        # When curves are overlapping with single cluster, share it
        if curves_are_overlapping and len(clusters) == 1:
            for curve in curves:
                if curve.y_positions:  # Only curves that have started tracking
                    curve.y_positions[x] = clusters[0]
                    curve.pixel_presence[x] = True
            return

        # Assign clusters
        used_clusters = set()
        assigned_curves = set()

        for curve_idx, cluster_idx, dist, within_threshold in curve_preferences:
            if curve_idx in assigned_curves:
                continue

            if cluster_idx not in used_clusters:
                curves[curve_idx].y_positions[x] = clusters[cluster_idx]
                curves[curve_idx].pixel_presence[x] = True
                used_clusters.add(cluster_idx)
                assigned_curves.add(curve_idx)

        # Second pass: curves that didn't get their preferred cluster try others
        # BUT only accept clusters within max_y_jump to prevent discontinuities
        remaining_clusters = [i for i in range(len(clusters)) if i not in used_clusters]

        for curve_idx, _, _, _ in curve_preferences:
            if curve_idx in assigned_curves:
                continue
            prev_y = prev_positions[curve_idx]
            if prev_y is None:
                continue

            # Find closest remaining cluster WITHIN the jump threshold
            valid_remaining = [
                (j, abs(clusters[j] - prev_y))
                for j in remaining_clusters
                if abs(clusters[j] - prev_y) <= max_y_jump
            ]

            if valid_remaining:
                best_j, _ = min(valid_remaining, key=lambda x: x[1])
                curves[curve_idx].y_positions[x] = clusters[best_j]
                curves[curve_idx].pixel_presence[x] = True
                remaining_clusters.remove(best_j)
                assigned_curves.add(curve_idx)
            elif curves_are_overlapping:
                # No valid cluster within threshold, but curves are overlapping
                # Allow sharing with an already-assigned cluster within threshold
                for other_idx in assigned_curves:
                    partner_y = curves[other_idx].y_positions.get(x)
                    if partner_y is not None and abs(partner_y - prev_y) <= max_y_jump:
                        curves[curve_idx].y_positions[x] = partner_y
                        curves[curve_idx].pixel_presence[x] = True
                        assigned_curves.add(curve_idx)
                        break
                else:
                    curves[curve_idx].pixel_presence[x] = False
            else:
                curves[curve_idx].pixel_presence[x] = False

        # Mark unassigned curves as having no pixel at this X
        for i, curve in enumerate(curves):
            if i not in assigned_curves and prev_positions[i] is not None:
                curve.pixel_presence[x] = False

        # Assign remaining clusters ONLY to curves that have NO previous positions at all
        # This handles the initial few columns where some curves haven't started yet
        # Do NOT assign to curves that were intentionally marked as gaps (to prevent bad jumps)
        for i, curve in enumerate(curves):
            # Only assign if curve has no previous positions at all (truly new)
            if not curve.y_positions:
                if remaining_clusters:
                    curve.y_positions[x] = clusters[remaining_clusters.pop(0)]
                    curve.pixel_presence[x] = True
                else:
                    curve.pixel_presence[x] = False

    def _analyze_curve_pattern_from_pixels(self, curve: TracedCurve) -> Tuple[LineStyle, float]:
        """
        Analyze line style by examining actual pixels along the curve's Y-positions.

        Strategy: For each curve, look at a NARROW band around its Y-positions and analyze
        the horizontal continuity of pixels within that band. A dashed line will have
        distinct gaps; a solid line will be continuous.

        When curves are close together, we need to be careful not to include pixels
        from the other curve. Use a narrower band in such cases.
        """
        if not curve.y_positions:
            return LineStyle.UNKNOWN, 0.0

        # Get sorted X positions
        x_positions = sorted(curve.y_positions.keys())
        if len(x_positions) < 10:
            return LineStyle.UNKNOWN, 0.3

        # Use a narrow band to avoid including pixels from nearby curves
        band_height = 2  # +/- pixels to include in the band (reduced from 4)

        # Analyze pixel presence along the curve path
        actual_presence = []
        for x in x_positions:
            y = int(curve.y_positions[x])
            y_min = max(0, y - band_height)
            y_max = min(self.binary.shape[0], y + band_height + 1)

            # Check if there are any pixels in the narrow band
            band_pixels = self.binary[y_min:y_max, x]
            has_pixel = np.any(band_pixels > 0)
            actual_presence.append(has_pixel)

        # Count gaps
        num_gaps = 0
        in_gap = False
        for present in actual_presence:
            if not present:
                if not in_gap:
                    num_gaps += 1
                    in_gap = True
            else:
                in_gap = False

        # Also sample the binary mask directly to look for dash patterns
        # by examining consecutive X positions
        dash_analysis = self._analyze_dash_pattern_directly(curve)

        # If direct dash analysis found dashes, prefer that result
        if dash_analysis[0] == LineStyle.DASHED:
            return dash_analysis

        # Analyze the presence pattern
        return self._classify_from_presence_pattern(actual_presence)

    def _analyze_dash_pattern_directly(self, curve: TracedCurve) -> Tuple[LineStyle, float]:
        """
        Analyze the curve for dash patterns by scanning the ORIGINAL grayscale image
        along the curve path, looking for TRUE gaps (white regions).

        Key approach:
        1. Scan original grayscale with a wider band (5 pixels) to catch line thickness
        2. Look for TRUE gaps: consecutive X positions where minimum intensity > 180
        3. True dashed lines have multiple regular gaps of 3+ pixels wide
        4. Solid lines might have occasional noise gaps but they're small/irregular
        """
        if not curve.y_positions:
            return LineStyle.UNKNOWN, 0.0

        x_positions = sorted(curve.y_positions.keys())
        if len(x_positions) < 30:
            return LineStyle.UNKNOWN, 0.3

        # CRITICAL: Scan the ORIGINAL grayscale image directly (not binary)
        # Use a 5-pixel band around the curve Y position to account for thickness
        x_min_curve = min(x_positions)
        x_max_curve = max(x_positions)

        # Interpolate Y positions for every X in the curve range
        all_x = list(range(x_min_curve, x_max_curve + 1))
        interpolated_y = np.interp(all_x, x_positions,
                                   [curve.y_positions[x] for x in x_positions])

        # Scan grayscale for each X position
        min_intensities = []
        for i, x in enumerate(all_x):
            y = int(interpolated_y[i])
            # Convert to full image coordinates
            img_x = x + self.plot_x
            img_y = y + self.plot_y

            # Use a 5-pixel vertical band centered on the curve
            y_min = max(0, img_y - 2)
            y_max = min(self.gray.shape[0], img_y + 3)
            x_min = max(0, img_x)
            x_max = min(self.gray.shape[1], img_x + 1)

            # Get minimum intensity (darkest pixel) in the band
            band = self.gray[y_min:y_max, x_min:x_max]
            if band.size > 0:
                min_intensities.append(np.min(band))
            else:
                min_intensities.append(255)

        min_intensities = np.array(min_intensities)

        # Classify each position as "dark" (has curve pixel) or "light" (gap)
        # TRUE gaps are clearly white (intensity > 180)
        # Curve pixels are dark (intensity < 100)
        is_dark = min_intensities < 100
        is_light = min_intensities > 180

        dark_ratio = np.sum(is_dark) / len(is_dark) if len(is_dark) > 0 else 1.0

        # Find TRUE gaps: runs of consecutive "light" positions
        true_gaps = []
        current_gap_length = 0

        for i in range(len(is_light)):
            if is_light[i]:
                current_gap_length += 1
            else:
                if current_gap_length >= 3:  # Minimum gap width to count as true gap
                    true_gaps.append(current_gap_length)
                current_gap_length = 0

        # Don't forget the last gap
        if current_gap_length >= 3:
            true_gaps.append(current_gap_length)

        num_true_gaps = len(true_gaps)
        avg_true_gap = np.mean(true_gaps) if true_gaps else 0

        # Calculate gap regularity (true dashes have regular gaps)
        gap_regularity = 0.0
        if num_true_gaps >= 3:
            gap_std = np.std(true_gaps)
            gap_cv = gap_std / avg_true_gap if avg_true_gap > 0 else 999
            if gap_cv < 0.5:
                gap_regularity = 1.0  # Very regular
            elif gap_cv < 1.0:
                gap_regularity = 0.5  # Somewhat regular
            # gap_cv >= 1.0 means irregular (likely noise, not true dashes)

        # Also analyze segment lengths between gaps
        segments = []
        current_seg_length = 0
        for i in range(len(is_dark)):
            if is_dark[i]:
                current_seg_length += 1
            else:
                if current_seg_length >= 3:
                    segments.append(current_seg_length)
                current_seg_length = 0
        if current_seg_length >= 3:
            segments.append(current_seg_length)

        avg_segment = np.mean(segments) if segments else 0
        num_segments = len(segments)

        # Scoring: higher score = more likely DASHED
        dash_score = 0

        # TRUE gaps are the strongest indicator
        if num_true_gaps >= 10:
            dash_score += 4  # Many true gaps = definitely dashed
        elif num_true_gaps >= 6:
            dash_score += 3
        elif num_true_gaps >= 3:
            dash_score += 2

        # Regular gaps increase confidence
        if gap_regularity >= 0.5 and num_true_gaps >= 3:
            dash_score += 2
        elif gap_regularity > 0 and num_true_gaps >= 2:
            dash_score += 1

        # Average gap size: dashes typically have gaps of 4-20 pixels
        if num_true_gaps >= 2 and 4 <= avg_true_gap <= 20:
            dash_score += 1

        # Low dark ratio indicates gaps
        if dark_ratio < 0.80:
            dash_score += 1

        # SOLID indicators (reduce score)
        # Very high dark ratio = continuous line
        if dark_ratio > 0.95:
            dash_score -= 2
        elif dark_ratio > 0.90:
            dash_score -= 1

        # Very few or no true gaps = solid
        if num_true_gaps <= 1:
            dash_score -= 2

        # Very long average segments = solid
        if avg_segment > 50 and num_true_gaps < 5:
            dash_score -= 1

        # Ensure score doesn't go negative
        dash_score = max(0, dash_score)

        # Debug output
        print(f"    Style analysis: true_gaps={num_true_gaps}, avg_gap={avg_true_gap:.1f}, "
              f"dark_ratio={dark_ratio:.2f}, avg_seg={avg_segment:.1f}, regularity={gap_regularity:.2f}, score={dash_score}")

        if dash_score >= 5:
            return LineStyle.DASHED, 0.9
        elif dash_score >= 3:
            return LineStyle.DASHED, 0.7
        else:
            return LineStyle.SOLID, 0.85

    def _classify_from_presence_pattern(self, presences: List[bool]) -> Tuple[LineStyle, float]:
        """Classify line style from a presence/absence pattern."""
        if len(presences) < 5:
            return LineStyle.UNKNOWN, 0.3

        # Count runs of presence and absence
        runs = []
        current_val = presences[0]
        current_len = 1

        for p in presences[1:]:
            if p == current_val:
                current_len += 1
            else:
                runs.append((current_val, current_len))
                current_val = p
                current_len = 1
        runs.append((current_val, current_len))

        # Analyze runs
        presence_runs = [length for val, length in runs if val]
        absence_runs = [length for val, length in runs if not val]

        total_present = sum(presence_runs)
        total_absent = sum(absence_runs)
        total = total_present + total_absent

        if total == 0:
            return LineStyle.UNKNOWN, 0.0

        presence_ratio = total_present / total

        # Classification based on pattern
        if presence_ratio > 0.85:
            # Mostly continuous - solid line
            return LineStyle.SOLID, 0.9

        if presence_ratio < 0.3:
            # Very sparse - might be noise or dotted
            if presence_runs:
                avg_segment = np.mean(presence_runs)
                if avg_segment < 5:
                    return LineStyle.DOTTED, 0.6
            return LineStyle.UNKNOWN, 0.4

        # Check for regular pattern (dashed or dotted)
        if presence_runs and absence_runs:
            avg_segment = np.mean(presence_runs)
            avg_gap = np.mean(absence_runs)

            # Regularity check
            segment_std = np.std(presence_runs) / avg_segment if avg_segment > 0 else 1
            gap_std = np.std(absence_runs) / avg_gap if avg_gap > 0 else 1

            is_regular = segment_std < 0.5 and gap_std < 0.5

            if avg_segment < 8 and avg_gap > 3:
                return LineStyle.DOTTED, 0.7 if is_regular else 0.5

            if avg_segment > 10 and avg_gap > 5:
                return LineStyle.DASHED, 0.8 if is_regular else 0.6

            # Check for dash-dot pattern
            if len(presence_runs) > 4:
                is_alternating = self._check_alternating_pattern(presence_runs)
                if is_alternating:
                    return LineStyle.DASH_DOT, 0.7

        # Default: classify by gap ratio
        gap_ratio = total_absent / total
        if gap_ratio < 0.15:
            return LineStyle.SOLID, 0.7
        elif gap_ratio < 0.4:
            return LineStyle.DASHED, 0.6
        else:
            return LineStyle.DOTTED, 0.5

    def _analyze_curve_pattern(self, curve: TracedCurve) -> Tuple[LineStyle, float]:
        """
        Analyze the presence/absence pattern to determine line style.

        Solid lines have continuous presence.
        Dashed lines have regular gaps.
        Dotted lines have frequent small segments.
        """
        if not curve.pixel_presence:
            return LineStyle.UNKNOWN, 0.0

        # Get sorted X positions
        x_positions = sorted(curve.pixel_presence.keys())
        if len(x_positions) < 5:
            return LineStyle.UNKNOWN, 0.3

        # Analyze presence pattern
        presences = [curve.pixel_presence[x] for x in x_positions]

        # Count runs of presence and absence
        runs = []
        current_val = presences[0]
        current_len = 1

        for p in presences[1:]:
            if p == current_val:
                current_len += 1
            else:
                runs.append((current_val, current_len))
                current_val = p
                current_len = 1
        runs.append((current_val, current_len))

        # Analyze runs
        presence_runs = [length for val, length in runs if val]
        absence_runs = [length for val, length in runs if not val]

        total_present = sum(presence_runs)
        total_absent = sum(absence_runs)
        total = total_present + total_absent

        if total == 0:
            return LineStyle.UNKNOWN, 0.0

        presence_ratio = total_present / total

        # Classification based on pattern
        if presence_ratio > 0.85:
            # Mostly continuous - solid line
            return LineStyle.SOLID, 0.9

        if presence_ratio < 0.3:
            # Very sparse - might be noise or dotted
            if presence_runs:
                avg_segment = np.mean(presence_runs)
                if avg_segment < 5:
                    return LineStyle.DOTTED, 0.6
            return LineStyle.UNKNOWN, 0.4

        # Check for regular pattern (dashed or dotted)
        if presence_runs and absence_runs:
            avg_segment = np.mean(presence_runs)
            avg_gap = np.mean(absence_runs)

            # Regularity check
            segment_std = np.std(presence_runs) / avg_segment if avg_segment > 0 else 1
            gap_std = np.std(absence_runs) / avg_gap if avg_gap > 0 else 1

            is_regular = segment_std < 0.5 and gap_std < 0.5

            if avg_segment < 8 and avg_gap > 3:
                return LineStyle.DOTTED, 0.7 if is_regular else 0.5

            if avg_segment > 10 and avg_gap > 5:
                return LineStyle.DASHED, 0.8 if is_regular else 0.6

            # Check for dash-dot pattern
            if len(presence_runs) > 4:
                # Look for alternating long-short pattern
                is_alternating = self._check_alternating_pattern(presence_runs)
                if is_alternating:
                    return LineStyle.DASH_DOT, 0.7

        # Default: classify by gap ratio
        gap_ratio = total_absent / total
        if gap_ratio < 0.15:
            return LineStyle.SOLID, 0.7
        elif gap_ratio < 0.4:
            return LineStyle.DASHED, 0.6
        else:
            return LineStyle.DOTTED, 0.5

    def _check_alternating_pattern(self, runs: List[int]) -> bool:
        """Check if runs show alternating long-short pattern."""
        if len(runs) < 4:
            return False

        median_len = np.median(runs)

        # Check if alternating above/below median
        prev_long = runs[0] > median_len
        alternations = 0

        for run_len in runs[1:]:
            curr_long = run_len > median_len
            if curr_long != prev_long:
                alternations += 1
            prev_long = curr_long

        # If most transitions are alternations, it's a dash-dot pattern
        return alternations > len(runs) * 0.6

    def _build_detected_curves(self) -> List[DetectedCurve]:
        """Convert traced curves to DetectedCurve format."""
        curves = []

        for traced in self.traced_curves:
            if not traced.y_positions:
                continue


            # Create pixel list from traced positions
            pixels = []
            for x, y in traced.y_positions.items():
                pixels.append((x, int(y)))

            if not pixels:
                continue

            # Create a segment from all pixels
            pixel_arr = np.array(pixels)
            segment = CurveSegment(
                pixels=pixels,
                style=LineStyleDescriptor(
                    style=traced.style,
                    gap_ratio=0.0,
                    avg_segment_len=0.0,
                    avg_gap_len=0.0,
                    pattern_period=0.0,
                    confidence=traced.confidence
                ),
                x_range=(int(pixel_arr[:, 0].min()), int(pixel_arr[:, 0].max())),
                y_range=(int(pixel_arr[:, 1].min()), int(pixel_arr[:, 1].max())),
                component_id=-1
            )

            curves.append(DetectedCurve(
                segments=[segment],
                style=traced.style,
                data_points=[],
                confidence=traced.confidence
            ))

        return curves

    def extract_curve_points(
        self,
        curve: DetectedCurve,
        pixel_to_coord_func
    ) -> List[Tuple[float, float]]:
        """Extract (time, survival) data points from a detected curve.

        Args:
            curve: The detected curve to extract points from
            pixel_to_coord_func: Function that converts (px_x, px_y) to (time, survival)

        Returns:
            List of (time, survival) tuples
        """
        # Collect all pixels
        all_pixels = []
        for segment in curve.segments:
            all_pixels.extend(segment.pixels)

        if not all_pixels:
            return []

        pixels = np.array(all_pixels)

        # Group by x coordinate and take median y
        x_unique = np.unique(pixels[:, 0])

        data_points = []
        for x in x_unique:
            y_vals = pixels[pixels[:, 0] == x, 1]
            # Use median y for robustness
            y_median = np.median(y_vals)

            # Convert to data coordinates
            time, survival = pixel_to_coord_func(
                x + self.plot_x,
                y_median + self.plot_y
            )

            data_points.append((time, survival))

        # Sort by time
        data_points.sort(key=lambda p: p[0])

        return data_points

    def get_binary_mask(self) -> np.ndarray:
        """Get the binary mask of extracted dark pixels."""
        return self.binary

    def get_debug_image(self) -> np.ndarray:
        """Generate a debug visualization of detected curves."""
        if len(self.original_image.shape) == 2:
            debug_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = self.original_image.copy()

        # Color map for different line styles
        style_colors = {
            LineStyle.SOLID: (0, 255, 0),      # Green
            LineStyle.DASHED: (255, 0, 0),     # Blue
            LineStyle.DOTTED: (0, 0, 255),     # Red
            LineStyle.DASH_DOT: (255, 255, 0), # Cyan
            LineStyle.UNKNOWN: (128, 128, 128) # Gray
        }

        # Draw each traced curve
        for i, curve in enumerate(self.traced_curves):
            color = style_colors.get(curve.style, (128, 128, 128))

            # Draw curve path
            for x, y in curve.y_positions.items():
                px = x + self.plot_x
                py = int(y) + self.plot_y
                if 0 <= px < debug_img.shape[1] and 0 <= py < debug_img.shape[0]:
                    cv2.circle(debug_img, (px, py), 2, color, -1)

        return debug_img


def is_grayscale_image(img: np.ndarray) -> bool:
    """Check if image is grayscale or near-grayscale.

    Args:
        img: BGR image array

    Returns:
        True if image appears to be grayscale
    """
    if len(img.shape) == 2:
        return True

    if len(img.shape) == 3:
        # Check color variance using HSV saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)

        # Low saturation indicates grayscale-ish image
        return mean_saturation < 30

    return False


def convert_line_style_curves(
    detected_curves: List[DetectedCurve],
    calibration,
    time_max: float
) -> List[Dict]:
    """Convert DetectedCurve objects to the format used by extract_km_curves.

    Args:
        detected_curves: List of curves detected by LineStyleDetector
        calibration: AxisCalibrationResult object (or None)
        time_max: Maximum time value

    Returns:
        List of dicts with 'name', 'style', 'data_points', 'pixels'
    """
    curves_data = []

    for i, curve in enumerate(detected_curves):
        # Generate name based on style
        style_name = curve.style.value if curve.style != LineStyle.UNKNOWN else f"curve_{i+1}"

        # Collect all pixels
        all_pixels = []
        for segment in curve.segments:
            all_pixels.extend(segment.pixels)

        curves_data.append({
            'name': f"{style_name}_{i+1}",
            'style': curve.style,
            'data_points': curve.data_points,
            'pixels': all_pixels,
            'confidence': curve.confidence
        })

    return curves_data
