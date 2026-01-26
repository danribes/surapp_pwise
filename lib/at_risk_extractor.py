#!/usr/bin/env python3
"""
At-Risk Table Extractor for Kaplan-Meier Curves

This module extracts "Number at Risk" tables that appear below KM curves.
The extracted data is essential for the Guyot algorithm to reconstruct
individual patient data (IPD) from published survival curves.

Features:
- Automatic detection of at-risk table presence
- Graceful handling when no table is present
- OCR-based text extraction with layout preservation
- Validation against extracted curve data
- AI-assisted fallback when OCR has low confidence

Usage:
    extractor = AtRiskExtractor(image, plot_bounds, calibration)
    result = extractor.extract()

    if result is not None:
        # Table found and extracted
        print(result.to_dataframe())
    else:
        # No table detected - continue without at-risk data
        pass
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import re
import json

# Optional AI support
try:
    from .ai_table_reader import AITableReader, get_ai_table_reader
    AI_TABLE_AVAILABLE = True
except ImportError:
    AI_TABLE_AVAILABLE = False


@dataclass
class AtRiskData:
    """Container for extracted at-risk table data."""
    groups: Dict[str, Dict[float, int]] = field(default_factory=dict)
    time_points: List[float] = field(default_factory=list)
    table_bounds: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    confidence: float = 0.0
    raw_ocr_results: List[Any] = field(default_factory=list)

    def to_dataframe(self):
        """Convert to pandas DataFrame in Guyot-compatible format."""
        import pandas as pd

        rows = []
        for group_name, time_data in self.groups.items():
            for time, at_risk in sorted(time_data.items()):
                rows.append({
                    'Group': group_name,
                    'Time': time,
                    'AtRisk': at_risk
                })

        return pd.DataFrame(rows)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'groups': self.groups,
            'time_points': self.time_points,
            'confidence': self.confidence
        }

    def add_events_column(self):
        """
        Calculate events between intervals (for Guyot algorithm).
        Events = n_at_risk[t] - n_at_risk[t+1] (approximately)
        """
        import pandas as pd

        df = self.to_dataframe()
        df['Events'] = 0

        for group in df['Group'].unique():
            mask = df['Group'] == group
            group_df = df[mask].sort_values('Time')

            # Calculate events as difference in at-risk counts
            at_risk = group_df['AtRisk'].values
            events = np.zeros(len(at_risk), dtype=int)

            for i in range(len(at_risk) - 1):
                # Events + censored = drop in at-risk count
                # This is an approximation; true events need survival curve
                events[i] = max(0, at_risk[i] - at_risk[i + 1])

            df.loc[mask, 'Events'] = events

        return df


class AtRiskExtractor:
    """
    Extracts "Number at Risk" tables from Kaplan-Meier plot images.

    Handles both cases:
    - Table present: Extracts and returns structured data
    - Table absent: Returns None gracefully
    """

    # Common header patterns for at-risk tables
    HEADER_PATTERNS = [
        r'number\s*at\s*risk',
        r'n\s*at\s*risk',
        r'patients\s*at\s*risk',
        r'no\.?\s*at\s*risk',
        r'at\s*risk',
    ]

    def __init__(
        self,
        image: np.ndarray,
        plot_bounds: Tuple[int, int, int, int],
        calibration: Any = None,
        debug: bool = False
    ):
        """
        Initialize the at-risk extractor.

        Args:
            image: Original image (BGR format from cv2)
            plot_bounds: (x, y, width, height) of the plot area
            calibration: AxisCalibrationResult with axis ranges
            debug: Enable debug output
        """
        self.image = image
        self.plot_bounds = plot_bounds
        self.calibration = calibration
        self.debug = debug
        self.height, self.width = image.shape[:2]

        # OCR reader (lazy initialization)
        self._ocr_reader = None
        self._ocr_available = None

    def _check_ocr_available(self) -> bool:
        """Check if OCR library is available."""
        if self._ocr_available is not None:
            return self._ocr_available

        try:
            import easyocr
            self._ocr_available = True
        except ImportError:
            try:
                import pytesseract
                self._ocr_available = True
            except ImportError:
                self._ocr_available = False

        return self._ocr_available

    def _preprocess_for_ocr(self, region: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        Preprocess image region for better OCR accuracy.

        Applies:
        1. Upscaling (2x default) for better character recognition
        2. Grayscale conversion
        3. Contrast enhancement (CLAHE)
        4. Denoising
        5. Adaptive thresholding for clean binarization

        Args:
            region: Image region to preprocess
            scale_factor: Upscaling factor (default 2.0)

        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # Upscale for better OCR on small text
        if scale_factor > 1.0:
            new_width = int(gray.shape[1] * scale_factor)
            new_height = int(gray.shape[0] * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Adaptive thresholding for clean text
        # Use a larger block size for upscaled images
        block_size = max(11, int(15 * scale_factor) | 1)  # Must be odd
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 2
        )

        # Morphological operations to clean up
        kernel_clean = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)

        return cleaned

    def _preprocess_for_ocr_color(self, region: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        Preprocess color image for OCR while preserving some color information.
        Useful for tables where group names might be color-coded.

        Args:
            region: Image region to preprocess
            scale_factor: Upscaling factor

        Returns:
            Preprocessed color image
        """
        # Upscale
        if scale_factor > 1.0:
            new_width = int(region.shape[1] * scale_factor)
            new_height = int(region.shape[0] * scale_factor)
            region = cv2.resize(region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Convert to LAB color space for better contrast enhancement
        if len(region.shape) == 3:
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(region)

        # Light sharpening
        kernel = np.array([[0, -1, 0],
                          [-1,  5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def _get_ocr_reader(self):
        """Get or create OCR reader (lazy initialization)."""
        if self._ocr_reader is not None:
            return self._ocr_reader

        try:
            import easyocr
            # Suppress easyocr download messages
            self._ocr_reader = easyocr.Reader(['en'], verbose=False)
            self._ocr_type = 'easyocr'
        except ImportError:
            try:
                import pytesseract
                self._ocr_reader = pytesseract
                self._ocr_type = 'pytesseract'
            except ImportError:
                raise ImportError(
                    "OCR library required. Install with: pip install easyocr\n"
                    "Or alternatively: pip install pytesseract (requires Tesseract binary)"
                )

        return self._ocr_reader

    def extract(self, curve_names: List[str] = None) -> Optional[AtRiskData]:
        """
        Extract at-risk table data from the image.

        Args:
            curve_names: Optional list of expected group names (from curve detection)

        Returns:
            AtRiskData if table found and extracted, None if no table detected
        """
        # Step 1: Detect table region
        table_region, table_bounds = self._detect_table_region()

        if table_region is None:
            if self.debug:
                print("  No at-risk table region detected")
            return None

        if self.debug:
            print(f"  Table region: x={table_bounds[0]}, y={table_bounds[1]}, "
                  f"w={table_bounds[2]}, h={table_bounds[3]}")

        # Step 2: Check if OCR is available
        if not self._check_ocr_available():
            if self.debug:
                print("  OCR not available - skipping at-risk extraction")
            return None

        # Step 3: Extract text with positions
        ocr_results = self._extract_text(table_region)

        if not ocr_results:
            if self.debug:
                print("  No text detected in table region")
            return None

        # Step 4: Parse table structure
        result = self._parse_table_structure(ocr_results, table_bounds, curve_names)

        if result is None or not result.groups:
            if self.debug:
                print("  Could not parse table structure")
            return None

        result.table_bounds = table_bounds
        result.raw_ocr_results = ocr_results

        return result

    def _detect_table_region(self) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detect the at-risk table region below the plot.

        Returns:
            Tuple of (table_image_region, bounds) or (None, None) if not found
        """
        plot_x, plot_y, plot_w, plot_h = self.plot_bounds

        # Table should be below the plot area
        # Start search from 75% of plot height to catch tables that might
        # overlap slightly with the bottom of the plot area
        # This helps with cases where plot bounds are over-estimated
        search_start_fraction = 0.75
        table_search_y_start = int(plot_y + plot_h * search_start_fraction)
        table_search_y_end = self.height

        # Ensure we have enough space for a table (at least 50 pixels)
        min_table_height = 50

        if self.debug:
            print(f"    Table search: y={table_search_y_start} to {table_search_y_end} "
                  f"(height={table_search_y_end - table_search_y_start})")

        if table_search_y_end - table_search_y_start < min_table_height:
            if self.debug:
                print(f"    Region too small for table")
            return None, None

        # Extract region below plot - use FULL width for multi-panel figures
        # Tables often span across multiple panels
        below_plot = self.image[table_search_y_start:table_search_y_end, :]

        # Convert to grayscale for analysis
        if len(below_plot.shape) == 3:
            gray = cv2.cvtColor(below_plot, cv2.COLOR_BGR2GRAY)
        else:
            gray = below_plot.copy()

        # Look for text regions using morphological operations
        # Threshold to find dark pixels (text)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Dilate to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours of text regions
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # Find bounding box of all text regions
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Validate: table should have reasonable dimensions
        if w < 100 or h < min_table_height:
            return None, None

        # Add margins
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(self.width - x, w + 2 * margin)
        h = min(below_plot.shape[0] - y, h + 2 * margin)

        # Extract table region
        table_region = below_plot[y:y+h, x:x+w]

        # Convert bounds to full image coordinates
        full_y = table_search_y_start + y
        table_bounds = (x, full_y, w, h)

        # Validation: check for "at risk" header OR structured numeric grid
        has_header = self._validate_table_region(table_region)
        has_numeric_grid = self._has_numeric_grid(table_region)

        if self.debug:
            print(f"    Table validation: header={has_header}, numeric_grid={has_numeric_grid}")

        # Accept if either condition is met
        if not has_header and not has_numeric_grid:
            return None, None

        return table_region, table_bounds

    def _validate_table_region(self, region: np.ndarray) -> bool:
        """
        Validate that the region contains an at-risk table.
        Looks for header text patterns.
        """
        # Quick OCR of just the header region (top portion)
        header_height = min(40, region.shape[0] // 3)
        header_region = region[:header_height, :]

        try:
            reader = self._get_ocr_reader()

            if self._ocr_type == 'easyocr':
                results = reader.readtext(header_region)
                text = ' '.join([r[1].lower() for r in results])
            else:
                from PIL import Image
                pil_img = Image.fromarray(cv2.cvtColor(header_region, cv2.COLOR_BGR2RGB))
                text = reader.image_to_string(pil_img).lower()

            # Check for header patterns
            for pattern in self.HEADER_PATTERNS:
                if re.search(pattern, text):
                    return True

        except Exception:
            pass

        return False

    def _has_numeric_grid(self, region: np.ndarray) -> bool:
        """
        Check if region contains a grid of numbers (backup validation).
        Must have multiple rows of numbers to be considered a table.
        """
        try:
            reader = self._get_ocr_reader()

            if self._ocr_type == 'easyocr':
                results = reader.readtext(region)
                if not results:
                    return False

                # Group by Y coordinate to check for multiple rows
                y_positions = {}
                for bbox, text, conf in results:
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    y_bucket = int(y_center / 20)  # Group within 20 pixels
                    if y_bucket not in y_positions:
                        y_positions[y_bucket] = []
                    y_positions[y_bucket].append(text)

                # Need at least 2 distinct rows
                if len(y_positions) < 2:
                    return False

                # Count rows that have multiple numbers
                rows_with_numbers = 0
                for y, texts in y_positions.items():
                    numeric_count = sum(1 for t in texts if re.match(r'^\d+$', t.strip()))
                    if numeric_count >= 3:
                        rows_with_numbers += 1

                # Need at least 2 rows with numbers (data rows or data + time)
                return rows_with_numbers >= 2

            else:
                from PIL import Image
                pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                text = reader.image_to_string(pil_img)
                lines = text.strip().split('\n')

                # Need at least 2 lines with multiple numbers
                lines_with_numbers = 0
                for line in lines:
                    nums = re.findall(r'\b\d+\b', line)
                    if len(nums) >= 3:
                        lines_with_numbers += 1

                return lines_with_numbers >= 2

        except Exception:
            return False

    def _extract_text(self, region: np.ndarray, use_preprocessing: bool = True) -> List[Dict]:
        """
        Extract text with bounding boxes from the table region.

        Args:
            region: Image region to extract text from
            use_preprocessing: Whether to apply preprocessing for better OCR

        Returns:
            List of dicts with 'text', 'bbox', 'confidence' keys
        """
        reader = self._get_ocr_reader()
        results = []

        # Apply preprocessing for better OCR
        if use_preprocessing:
            # Try color-preserved preprocessing first (for colored group names)
            processed_color = self._preprocess_for_ocr_color(region, scale_factor=2.0)
            # Also create binary version for number detection
            processed_binary = self._preprocess_for_ocr(region, scale_factor=2.0)

            if self.debug:
                print(f"    Preprocessing: {region.shape} -> {processed_color.shape}")
        else:
            processed_color = region
            processed_binary = region

        scale_factor = 2.0 if use_preprocessing else 1.0

        if self._ocr_type == 'easyocr':
            # Run OCR on color-enhanced image
            ocr_results = reader.readtext(processed_color)

            # If few results from color, also try binary
            if len(ocr_results) < 5:
                binary_results = reader.readtext(processed_binary)
                # Merge results, avoiding duplicates
                existing_texts = {r[1].lower().strip() for r in ocr_results}
                for br in binary_results:
                    if br[1].lower().strip() not in existing_texts:
                        ocr_results.append(br)

            for bbox, text, conf in ocr_results:
                # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [p[0] / scale_factor for p in bbox]  # Scale back
                y_coords = [p[1] / scale_factor for p in bbox]

                results.append({
                    'text': text,
                    'bbox': (min(x_coords), min(y_coords),
                            max(x_coords) - min(x_coords),
                            max(y_coords) - min(y_coords)),
                    'center_x': sum(x_coords) / 4,
                    'center_y': sum(y_coords) / 4,
                    'confidence': conf
                })

        else:  # pytesseract
            from PIL import Image
            if len(processed_color.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(processed_color, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(processed_color)

            # Get detailed data with bounding boxes
            data = reader.image_to_data(pil_img, output_type=reader.Output.DICT)

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:
                    # Scale coordinates back to original size
                    x = data['left'][i] / scale_factor
                    y = data['top'][i] / scale_factor
                    w = data['width'][i] / scale_factor
                    h = data['height'][i] / scale_factor
                    conf = float(data['conf'][i]) / 100 if data['conf'][i] != -1 else 0.5

                    results.append({
                        'text': text,
                        'bbox': (x, y, w, h),
                        'center_x': x + w / 2,
                        'center_y': y + h / 2,
                        'confidence': conf
                    })

        return results

    def _parse_table_structure(
        self,
        ocr_results: List[Dict],
        table_bounds: Tuple[int, int, int, int],
        curve_names: List[str] = None
    ) -> Optional[AtRiskData]:
        """
        Parse OCR results into structured table data.

        The table typically has:
        - Header row: "Number at risk"
        - Group rows: "GroupName  N1  N2  N3  ..."
        - Time row: "0  3  6  9  12  ..." (at bottom)
        """
        if not ocr_results:
            return None

        # Group results by Y coordinate (rows)
        rows = self._group_by_rows(ocr_results)

        if len(rows) < 2:
            return None

        # Sort rows by Y position
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])

        if self.debug:
            print(f"    Found {len(sorted_rows)} text rows")
            for row_y, row_items in sorted_rows:
                texts = [item['text'] for item in row_items]
                print(f"      y={row_y}: {texts[:10]}{'...' if len(texts) > 10 else ''}")

        # Identify row types
        header_row = None
        time_row = None
        data_rows = []

        for row_y, row_items in sorted_rows:
            row_text = ' '.join(item['text'].lower() for item in row_items)
            row_texts = [item['text'] for item in row_items]

            # Check if this is the header
            is_header = any(re.search(p, row_text) for p in self.HEADER_PATTERNS)

            # Count numeric vs non-numeric items
            numeric_items = [t for t in row_texts if re.match(r'^\d+\.?\d*$', t.strip())]
            non_numeric_items = [t for t in row_texts if not re.match(r'^\d+\.?\d*$', t.strip()) and len(t.strip()) > 0]

            numeric_ratio = len(numeric_items) / max(len(row_texts), 1)

            # Skip header rows
            if is_header:
                header_row = row_items
                continue

            # Skip rows that are just axis labels like "Time (Months)"
            if re.search(r'time\s*\(?months?\)?|months?|years?', row_text):
                continue

            # Data row: has a text label AND multiple numbers
            # The label is typically at the left (first non-numeric item)
            has_label = len(non_numeric_items) >= 1
            has_numbers = len(numeric_items) >= 3  # At least 3 time points

            if self.debug:
                print(f"      Row y={row_y}: {len(numeric_items)} nums, {len(non_numeric_items)} text, "
                      f"header={is_header}")

            if has_label and has_numbers:
                data_rows.append((row_y, row_items))
                if self.debug:
                    print(f"        -> DATA ROW (label + numbers)")
            elif numeric_ratio > 0.8 and len(numeric_items) >= 3:
                # Mostly numeric row - could be time row or unlabeled data
                # Time row is usually at the bottom and has evenly spaced small numbers
                is_likely_time = all(float(n) < 100 for n in numeric_items if re.match(r'^\d+\.?\d*$', n))
                if is_likely_time and (time_row is None or row_y > sorted_rows[-2][0]):
                    time_row = row_items
                    if self.debug:
                        print(f"        -> TIME ROW")
                else:
                    # Unlabeled data row
                    data_rows.append((row_y, row_items))
                    if self.debug:
                        print(f"        -> UNLABELED DATA ROW")

        # If no explicit time row found, try the last row if it's mostly numeric
        if time_row is None and sorted_rows:
            last_row_y, last_row_items = sorted_rows[-1]
            last_texts = [item['text'] for item in last_row_items]
            numeric_count = sum(1 for t in last_texts if re.match(r'^\d+\.?\d*$', t.strip()))
            if numeric_count >= 3 and numeric_count / len(last_texts) > 0.7:
                time_row = last_row_items
                # Remove from data_rows if it was added there
                data_rows = [(y, items) for y, items in data_rows if y != last_row_y]

        # Extract time points
        time_points = self._extract_time_points(time_row, table_bounds)

        if not time_points and data_rows:
            # Estimate from calibration or default
            # Count numbers in first data row to estimate column count
            first_row_texts = [item['text'] for item in data_rows[0][1]]
            n_numbers = sum(1 for t in first_row_texts if re.match(r'^\d+$', t.strip()))
            time_points = self._estimate_time_points(max(n_numbers, 7))

        if not time_points:
            return None

        if self.debug:
            print(f"    Found {len(data_rows)} data rows, time_points={time_points}")

        # Parse data rows
        result = AtRiskData(time_points=time_points)

        for row_y, row_items in data_rows:
            group_name, values = self._parse_data_row(row_items, len(time_points))

            if self.debug:
                print(f"    Parsed row y={row_y}: group='{group_name}', values={values[:5]}{'...' if len(values) > 5 else ''}")

            if group_name and values and len(values) >= 3:
                # Clean group name
                group_name = self._clean_group_name(group_name)
                result.groups[group_name] = {}
                for i, val in enumerate(values):
                    if i < len(time_points):
                        result.groups[group_name][time_points[i]] = val

        # Validate: need at least one group with reasonable data
        if not result.groups:
            return None

        # Calculate confidence based on extraction quality
        total_cells = sum(len(v) for v in result.groups.values())
        expected_cells = len(result.groups) * len(time_points)
        result.confidence = total_cells / max(expected_cells, 1)

        return result

    def _clean_group_name(self, name: str) -> str:
        """Clean up group name from OCR artifacts."""
        # Remove common OCR artifacts
        name = re.sub(r'^[|:\-_\s]+', '', name)
        name = re.sub(r'[|:\-_\s]+$', '', name)
        # Remove leading/trailing punctuation
        name = name.strip('.,;:|-_ ')
        return name if name else "Unknown"

    def _group_by_rows(self, items: List[Dict], tolerance: int = 15) -> Dict[int, List[Dict]]:
        """Group OCR items by their Y coordinate (row)."""
        rows = {}

        for item in items:
            y = int(item['center_y'])

            # Find existing row within tolerance
            matched_row = None
            for row_y in rows.keys():
                if abs(row_y - y) < tolerance:
                    matched_row = row_y
                    break

            if matched_row is not None:
                rows[matched_row].append(item)
            else:
                rows[y] = [item]

        # Sort items within each row by X coordinate
        for row_y in rows:
            rows[row_y] = sorted(rows[row_y], key=lambda x: x['center_x'])

        return rows

    def _extract_time_points(
        self,
        time_row: List[Dict],
        table_bounds: Tuple[int, int, int, int]
    ) -> List[float]:
        """Extract time points from the time row."""
        if not time_row:
            return []

        time_points = []
        for item in time_row:
            text = item['text'].strip()
            # Match numbers (integers or decimals)
            match = re.match(r'^(\d+\.?\d*)$', text)
            if match:
                time_points.append(float(match.group(1)))

        return sorted(time_points)

    def _estimate_time_points(self, n_points: int) -> List[float]:
        """Estimate time points from calibration or defaults."""
        if self.calibration is not None:
            x_min, x_max = self.calibration.x_data_range
            # Generate evenly spaced points
            return [x_min + i * (x_max - x_min) / (n_points - 1)
                    for i in range(n_points)]

        # Default: 0, 3, 6, 9, 12, 15, 18 (common pattern)
        return [0, 3, 6, 9, 12, 15, 18][:n_points]

    def _parse_data_row(
        self,
        row_items: List[Dict],
        expected_values: int
    ) -> Tuple[Optional[str], List[int]]:
        """
        Parse a data row into group name and numeric values.

        Returns:
            Tuple of (group_name, list_of_at_risk_counts)
        """
        if not row_items:
            return None, []

        # Sort items by X position (left to right)
        sorted_items = sorted(row_items, key=lambda x: x['center_x'])

        group_name_parts = []
        values = []
        found_first_number = False

        for item in sorted_items:
            text = item['text'].strip()

            # Try to parse as number
            if re.match(r'^\d+$', text):
                values.append(int(text))
                found_first_number = True
            elif re.match(r'^\d+\.\d+$', text):
                values.append(int(float(text)))
                found_first_number = True
            elif not found_first_number and len(text) > 0:
                # Text before first number is part of group name
                # Skip common separators
                if text not in ['|', '-', ':', '_', '.', ',']:
                    group_name_parts.append(text)

        # Combine group name parts
        if group_name_parts:
            group_name = ' '.join(group_name_parts)
            # Clean up
            group_name = re.sub(r'\s+', ' ', group_name).strip()
            group_name = re.sub(r'^[|:\-_\s]+', '', group_name)
            group_name = re.sub(r'[|:\-_\s]+$', '', group_name)
        else:
            group_name = None

        # If no group name found but we have values, generate one
        if not group_name and values:
            group_name = f"Group_{len(values)}"

        return group_name, values

    def save_debug_image(self, output_path: str, result: AtRiskData = None):
        """Save debug image showing detected table region."""
        debug_img = self.image.copy()

        if result and result.table_bounds:
            x, y, w, h = result.table_bounds
            # Draw rectangle around table
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, "At-Risk Table", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw plot bounds
        px, py, pw, ph = self.plot_bounds
        cv2.rectangle(debug_img, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
        cv2.putText(debug_img, "Plot Area", (px, py - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imwrite(output_path, debug_img)

    def save_preprocessed_image(self, output_path: str, table_region: np.ndarray):
        """Save preprocessed table region for debugging OCR issues."""
        processed = self._preprocess_for_ocr(table_region, scale_factor=2.0)
        cv2.imwrite(output_path, processed)

    def extract_with_ai_fallback(
        self,
        curve_names: List[str] = None,
        use_ai: bool = True,
        ai_config=None
    ) -> Optional[AtRiskData]:
        """
        Extract at-risk table with AI fallback for low-confidence cases.

        This method:
        1. First attempts OCR-based table extraction
        2. If OCR fails or has low confidence, uses AI vision model
        3. Validates OCR results against AI when both are available

        Args:
            curve_names: Optional list of expected group names
            use_ai: Whether to use AI fallback (default True)
            ai_config: Optional AIConfig for AI detection

        Returns:
            AtRiskData if table found and extracted, None otherwise
        """
        # First try standard extraction
        result = self.extract(curve_names)

        # Evaluate extraction confidence
        ocr_confidence = result.confidence if result else 0.0

        if self.debug:
            print(f"  OCR extraction confidence: {ocr_confidence:.1%}")

        # Use AI fallback if available and needed
        if use_ai and AI_TABLE_AVAILABLE and (ocr_confidence < 0.7 or result is None):
            ai_reader = get_ai_table_reader(ai_config)

            if ai_reader is not None and ai_reader.is_available:
                if self.debug:
                    print("  Using AI fallback for table reading...")

                # Get table region
                table_region, table_bounds = self._detect_table_region()

                if table_region is not None:
                    ai_result = ai_reader.read_table(table_region, quiet=not self.debug)

                    if ai_result is not None and ai_result.is_valid:
                        # Use AI result to create/improve extraction
                        if result is None:
                            # No OCR result - create from AI
                            return self._create_from_ai_result(ai_result, table_bounds)
                        else:
                            # Validate/correct OCR with AI
                            return self._merge_ocr_and_ai_result(result, ai_result)

        return result

    def _create_from_ai_result(
        self,
        ai_result,
        table_bounds: Tuple[int, int, int, int]
    ) -> AtRiskData:
        """
        Create AtRiskData from AI reading result.

        Args:
            ai_result: AITableResult from AI reading
            table_bounds: Detected table bounds

        Returns:
            AtRiskData
        """
        result = AtRiskData(
            groups=ai_result.groups,
            time_points=ai_result.time_points,
            table_bounds=table_bounds,
            confidence=ai_result.confidence * 0.9,  # Slightly discount AI-only
            raw_ocr_results=[]
        )

        if self.debug:
            print(f"  Created table from AI: {len(result.groups)} groups, "
                  f"{len(result.time_points)} time points")

        return result

    def _merge_ocr_and_ai_result(
        self,
        ocr_result: AtRiskData,
        ai_result
    ) -> AtRiskData:
        """
        Merge OCR and AI results, preferring higher confidence source.

        Args:
            ocr_result: Result from OCR extraction
            ai_result: AITableResult from AI reading

        Returns:
            Merged AtRiskData
        """
        # Start with OCR result
        time_points = ocr_result.time_points
        groups = ocr_result.groups.copy()
        confidence = ocr_result.confidence

        # Validate time points
        if ai_result.time_points:
            if not time_points:
                time_points = ai_result.time_points
            elif len(ai_result.time_points) > len(time_points) and ai_result.confidence > 0.7:
                # AI found more time points
                if self.debug:
                    print(f"  AI override time points: {time_points} -> {ai_result.time_points}")
                time_points = ai_result.time_points

        # Validate and merge groups
        if ai_result.groups:
            for ai_name, ai_counts in ai_result.groups.items():
                # Find matching OCR group
                ocr_match = None
                for ocr_name in groups:
                    if (ocr_name.lower() in ai_name.lower() or
                        ai_name.lower() in ocr_name.lower() or
                        self._names_similar(ocr_name, ai_name)):
                        ocr_match = ocr_name
                        break

                if ocr_match:
                    # Validate counts
                    ocr_counts = groups[ocr_match]
                    for t, ai_count in ai_counts.items():
                        ocr_count = ocr_counts.get(t)
                        if ocr_count is None:
                            # AI has value OCR missed
                            if ai_result.confidence > 0.6:
                                ocr_counts[t] = ai_count
                        elif abs(ai_count - ocr_count) / max(ai_count, 1) > 0.3:
                            # Significant disagreement
                            if ai_result.confidence > 0.8:
                                if self.debug:
                                    print(f"  AI override {ocr_match}[{t}]: {ocr_count} -> {ai_count}")
                                ocr_counts[t] = ai_count
                else:
                    # AI found group OCR missed
                    if ai_result.confidence > 0.7:
                        if self.debug:
                            print(f"  AI added group: {ai_name}")
                        groups[ai_name] = ai_counts

        # Update confidence
        if ai_result.is_valid:
            confidence = max(confidence, (confidence + ai_result.confidence) / 2)

        return AtRiskData(
            groups=groups,
            time_points=time_points,
            table_bounds=ocr_result.table_bounds,
            confidence=confidence,
            raw_ocr_results=ocr_result.raw_ocr_results
        )

    def _names_similar(self, name1: str, name2: str) -> bool:
        """Check if two group names are similar (fuzzy match)."""
        # Normalize
        n1 = re.sub(r'[^a-z0-9]', '', name1.lower())
        n2 = re.sub(r'[^a-z0-9]', '', name2.lower())

        if not n1 or not n2:
            return False

        # Check substring
        if n1 in n2 or n2 in n1:
            return True

        # Check first few chars
        min_len = min(len(n1), len(n2), 4)
        if n1[:min_len] == n2[:min_len]:
            return True

        return False


def extract_at_risk_table(
    image_path: str,
    plot_bounds: Tuple[int, int, int, int],
    calibration: Any = None,
    output_dir: str = None,
    debug: bool = False
) -> Optional[AtRiskData]:
    """
    Convenience function to extract at-risk table from an image file.

    Args:
        image_path: Path to the KM plot image
        plot_bounds: (x, y, width, height) of the plot area
        calibration: Optional AxisCalibrationResult
        output_dir: Optional directory for debug output
        debug: Enable debug output

    Returns:
        AtRiskData if table found, None otherwise
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    extractor = AtRiskExtractor(img, plot_bounds, calibration, debug=debug)
    result = extractor.extract()

    if result and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save debug image
        extractor.save_debug_image(str(output_path / "debug_at_risk_detection.png"), result)

        # Save CSV
        df = result.add_events_column()
        df.to_csv(output_path / "at_risk_table.csv", index=False)

    return result
