"""
Hybrid AI + Pixel-based Curve Extraction.

This module combines:
1. AI for curve IDENTIFICATION (colors, styles, count, legend)
2. Color-based ISOLATION (separate image per curve)
3. Pixel-based EXTRACTION (accurate coordinate extraction)
4. AI for VALIDATION (quality check)

This approach leverages AI's strength (understanding image content) while
avoiding its weakness (precise numerical reading from graphs).
"""

import cv2
import numpy as np
import pandas as pd
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any

# Import color transformation utilities
from .color_transform import ColorTransformer

# Import AI service
try:
    from .ai_service import AIService, get_ai_service
    from .ai_curve_extractor import AICurveExtractor
    HAS_AI = True
except ImportError:
    HAS_AI = False
    AIService = None

# Import extraction components
try:
    from .color_detector import ColorCurveDetector, detect_curve_colors
    HAS_COLOR_DETECTOR = True
except ImportError:
    HAS_COLOR_DETECTOR = False

try:
    from .calibrator import Calibrator
    HAS_CALIBRATOR = True
except ImportError:
    HAS_CALIBRATOR = False


@dataclass
class CurveIdentification:
    """Result of AI curve identification."""
    color: str  # e.g., "green", "red", "black"
    style: str  # e.g., "solid", "dashed"
    description: str  # Full AI description
    legend_label: Optional[str] = None  # If readable from legend
    position: str = ""  # "top", "middle", "bottom" based on survival


@dataclass
class IsolatedCurve:
    """A curve isolated into its own image."""
    identification: CurveIdentification
    isolated_image: np.ndarray
    isolated_image_path: str
    mask: np.ndarray
    original_color_hsv: Tuple[int, int, int] = (0, 0, 0)
    transformed: bool = False  # True if color transformation applied
    extraction_color: str = ""  # Color to use for extraction (may differ after transform)


@dataclass
class ExtractedCurveResult:
    """Result of extracting a single curve."""
    identification: CurveIdentification
    dataframe: pd.DataFrame  # Time, Survival columns
    points_count: int
    time_range: Tuple[float, float]
    survival_range: Tuple[float, float]
    confidence: float = 0.8
    extraction_method: str = "pixel"  # "pixel", "waypoint", "interpolated"


@dataclass
class HybridExtractionResult:
    """Complete result of hybrid extraction."""
    curves: List[ExtractedCurveResult]
    combined_df: pd.DataFrame
    ai_identification: Dict[str, Any]
    validation_result: Optional[Dict[str, Any]] = None
    output_dir: str = ""


class HybridExtractor:
    """
    Hybrid AI + Pixel-based curve extractor.

    Workflow:
    1. AI identifies curves in the image (or use color detection fallback)
    2. Each curve is isolated using color masking
    3. Pixel-based extraction runs on each isolated image
    4. Results are combined and validated with AI
    """

    # Standard color definitions in HSV
    COLOR_RANGES = {
        'red': {'h_ranges': [(0, 10), (170, 180)], 's_min': 50, 'v_min': 50},
        'orange': {'h_ranges': [(10, 25)], 's_min': 50, 'v_min': 50},
        'yellow': {'h_ranges': [(25, 35)], 's_min': 50, 'v_min': 50},
        'green': {'h_ranges': [(35, 85)], 's_min': 50, 'v_min': 50},
        'cyan': {'h_ranges': [(85, 100)], 's_min': 50, 'v_min': 50},
        'blue': {'h_ranges': [(100, 130)], 's_min': 50, 'v_min': 50},
        'purple': {'h_ranges': [(130, 150)], 's_min': 50, 'v_min': 50},
        'magenta': {'h_ranges': [(150, 170)], 's_min': 50, 'v_min': 50},
        'black': {'h_ranges': [(0, 180)], 's_max': 50, 'v_min': 10, 'v_max': 80},
        'gray': {'h_ranges': [(0, 180)], 's_max': 50, 'v_min': 60, 'v_max': 180},
        'white': {'h_ranges': [(0, 180)], 's_max': 30, 'v_min': 200},
    }

    def __init__(self, quiet: bool = False, use_ai: bool = True):
        """
        Initialize the hybrid extractor.

        Args:
            quiet: Suppress progress messages
            use_ai: Whether to use AI for curve identification (if False, uses color detection)
        """
        self.quiet = quiet
        self.use_ai = use_ai
        self.ai_service = None
        self.temp_files = []

        if HAS_AI and use_ai:
            self.ai_service = get_ai_service()

    def cleanup(self):
        """Remove temporary files."""
        for f in self.temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        self.temp_files = []

    def _log(self, message: str):
        """Print message if not quiet."""
        if not self.quiet:
            print(message)

    def _parse_color_from_description(self, description: str) -> str:
        """
        Parse color name from AI description.

        Args:
            description: AI curve description like "cyan solid" or "dashed gray"

        Returns:
            Color name (lowercase)
        """
        description = description.lower()

        # Check for known colors
        for color in self.COLOR_RANGES.keys():
            if color in description:
                return color

        # Check common color variations
        color_aliases = {
            'grey': 'gray',
            'dark': 'black',
            'light blue': 'cyan',
            'teal': 'cyan',
            'aqua': 'cyan',
            'pink': 'magenta',
            'lime': 'green',
            'navy': 'blue',
            'maroon': 'red',
            'olive': 'green',
        }

        for alias, color in color_aliases.items():
            if alias in description:
                return color

        return "unknown"

    def _parse_style_from_description(self, description: str) -> str:
        """Parse line style from AI description."""
        description = description.lower()

        if 'dash' in description:
            return 'dashed'
        elif 'dot' in description:
            return 'dotted'
        else:
            return 'solid'

    def identify_curves_with_ai(self, image_path: str) -> List[CurveIdentification]:
        """
        Use AI to identify curves in the image.

        Args:
            image_path: Path to the KM plot image

        Returns:
            List of identified curves
        """
        if not self.use_ai:
            self._log("  AI disabled, using color detection")
            return self._identify_curves_fallback(image_path)

        if not HAS_AI or self.ai_service is None or not self.ai_service.is_available:
            self._log("  AI not available, using fallback color detection")
            return self._identify_curves_fallback(image_path)

        self._log("  Step 1: AI curve identification...")

        # Use the curve extractor's identification capability
        extractor = self.ai_service.curve_extractor
        if extractor is None:
            return self._identify_curves_fallback(image_path)

        # Call the identification prompt
        response = extractor._call_vision_model(
            image_path,
            extractor.IDENTIFY_CURVES_PROMPT,
            quiet=self.quiet
        )

        if not response:
            self._log("    AI identification failed, using fallback")
            return self._identify_curves_fallback(image_path)

        # Parse the response
        curves, x_range, y_range, confidence = extractor._parse_curve_identification(response)

        identifications = []
        for i, desc in enumerate(curves):
            color = self._parse_color_from_description(desc)
            style = self._parse_style_from_description(desc)

            # Determine position based on order (AI returns top to bottom)
            if i == 0:
                position = "top"
            elif i == len(curves) - 1:
                position = "bottom"
            else:
                position = "middle"

            identifications.append(CurveIdentification(
                color=color,
                style=style,
                description=desc,
                position=position
            ))

        self._log(f"    Found {len(identifications)} curves: {[c.description for c in identifications]}")
        return identifications

    def _identify_curves_fallback(self, image_path: str) -> List[CurveIdentification]:
        """
        Fallback curve identification using color histogram analysis.
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        colors = detect_curve_colors(img, min_pixels=100, max_colors=6)

        identifications = []
        for color_info in colors:
            name = color_info.get('name', 'unknown')
            identifications.append(CurveIdentification(
                color=name,
                style='solid',
                description=f"{name} curve",
                position=""
            ))

        return identifications

    def create_color_mask(self, image: np.ndarray, color: str) -> np.ndarray:
        """
        Create a binary mask for pixels of the specified color.

        Args:
            image: BGR image
            color: Color name

        Returns:
            Binary mask (255 for matching pixels, 0 otherwise)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if color not in self.COLOR_RANGES:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        color_def = self.COLOR_RANGES[color]

        # Start with all False
        mask = np.zeros(image.shape[:2], dtype=bool)

        # Add hue ranges
        for h_min, h_max in color_def['h_ranges']:
            mask |= (h >= h_min) & (h <= h_max)

        # Apply saturation constraints
        s_min = color_def.get('s_min', 0)
        s_max = color_def.get('s_max', 255)
        mask &= (s >= s_min) & (s <= s_max)

        # Apply value constraints
        v_min = color_def.get('v_min', 0)
        v_max = color_def.get('v_max', 255)
        mask &= (v >= v_min) & (v <= v_max)

        return (mask * 255).astype(np.uint8)

    def isolate_curve(self,
                      image: np.ndarray,
                      identification: CurveIdentification,
                      other_colors: List[str] = None) -> IsolatedCurve:
        """
        Isolate a single curve from the image.

        For colored curves: Keep only pixels of that color
        For black/gray curves: Apply inversion after removing other colors

        Args:
            image: Original BGR image
            identification: Curve identification info
            other_colors: List of other curve colors to remove

        Returns:
            IsolatedCurve with the isolated image
        """
        color = identification.color
        other_colors = other_colors or []

        # Create color transformer
        transformer = ColorTransformer(image)

        if color in ['black', 'gray']:
            # For dark curves, need special handling
            # Remove other colored curves first, then invert
            colored_to_remove = [c for c in other_colors if c not in ['black', 'gray', 'white']]
            if colored_to_remove:
                # Remove other colors
                cleaned = transformer.remove_colors(colored_to_remove)
            else:
                cleaned = image.copy()

            # Invert to make black curves white (easier to extract)
            isolated = transformer.invert(cleaned)
            extraction_color = 'white'
            transformed = True
        else:
            # For colored curves, isolate just that color
            mask = self.create_color_mask(image, color)

            # Create isolated image with white background
            isolated = np.full_like(image, 255)
            isolated[mask > 0] = image[mask > 0]
            extraction_color = color
            transformed = False

        # Save to temp file
        temp_path = tempfile.mktemp(suffix=f'_isolated_{color}.png')
        cv2.imwrite(temp_path, isolated)
        self.temp_files.append(temp_path)

        # Create mask
        mask = self.create_color_mask(image, color)

        return IsolatedCurve(
            identification=identification,
            isolated_image=isolated,
            isolated_image_path=temp_path,
            mask=mask,
            transformed=transformed,
            extraction_color=extraction_color
        )

    def _deduplicate_identifications(self, identifications: List[CurveIdentification]) -> List[CurveIdentification]:
        """
        Remove duplicate curve identifications (e.g., black and gray being the same).
        """
        # Group similar colors
        color_groups = {
            'dark': ['black', 'gray', 'dark'],
            'red': ['red', 'maroon'],
            'green': ['green', 'lime'],
            'blue': ['blue', 'navy'],
            'cyan': ['cyan', 'teal', 'aqua'],
        }

        seen_groups = set()
        unique = []

        for ident in identifications:
            color = ident.color

            # Find which group this color belongs to
            group = color
            for group_name, members in color_groups.items():
                if color in members:
                    group = group_name
                    break

            if group not in seen_groups:
                seen_groups.add(group)
                unique.append(ident)
                self._log(f"    Keeping {ident.color} (group: {group})")
            else:
                self._log(f"    Skipping duplicate {ident.color} (group: {group})")

        return unique

    def extract_black_curve_smart(self,
                                   image: np.ndarray,
                                   calibration: Dict[str, float],
                                   other_curves: List[ExtractedCurveResult],
                                   time_max: float = None) -> Optional[ExtractedCurveResult]:
        """
        Smart extraction of black/gray curves.

        Strategy:
        1. Find all black pixels in the plot area
        2. Use other curves as upper bounds where available
        3. Exclude pixels near the axes (straight lines)
        4. Look for step-pattern pixels typical of KM curves
        """
        x_0 = int(calibration['x_0_pixel'])
        x_max = int(calibration['x_max_pixel'])
        y_0 = int(calibration['y_0_pixel'])
        y_100 = int(calibration['y_100_pixel'])
        time_max_val = time_max or calibration.get('time_max', 24)

        # Convert to HSV for black detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Black pixels: low saturation, low-medium value (not too dark, not too light)
        black_mask = (s < 50) & (v > 20) & (v < 100)

        # Exclude pixels very close to axes (within 5 pixels)
        axis_margin = 5
        black_mask[:, :x_0 + axis_margin] = False  # Left of Y-axis
        black_mask[y_0 - axis_margin:, :] = False  # Below X-axis

        # Build upper bound from other curves
        upper_bound = {}
        lower_bound = {}
        for curve in other_curves:
            for _, row in curve.dataframe.iterrows():
                t, surv = row['Time'], row['Survival']
                x = int(x_0 + (t / time_max_val) * (x_max - x_0))
                y = int(y_0 - surv * (y_0 - y_100))

                # The black curve should be BELOW other curves (higher y value)
                if x not in upper_bound or y < upper_bound[x]:
                    upper_bound[x] = y

        # Scan for curve pixels
        # For each x, find black pixels that could be part of the curve
        points = []
        for x in range(x_0 + axis_margin, x_max - axis_margin):
            col_mask = black_mask[:, x]
            y_indices = np.where(col_mask)[0]

            # Filter to plot area (above X-axis, below top)
            y_indices = y_indices[(y_indices >= y_100 + 10) & (y_indices <= y_0 - 10)]

            if len(y_indices) == 0:
                continue

            # If we have upper bound from other curves, filter
            if x in upper_bound:
                # Black curve should be below (higher y) than other curves
                y_indices = y_indices[y_indices >= upper_bound[x] - 3]

            if len(y_indices) == 0:
                continue

            # Find the topmost black pixel in this column (highest survival)
            # This helps avoid picking up text/labels lower in the plot
            y = int(np.min(y_indices))

            # Additional check: the y should be reasonable (not at the very bottom)
            # The black curve in this image ends around 21%, so y should be above y_0 * 0.8
            if y > y_0 - 0.15 * (y_0 - y_100):  # Below 15% survival
                continue

            points.append((x, y))

        if len(points) < 10:
            self._log(f"      Only {len(points)} points found, trying fallback")
            return self._extract_black_curve_fallback(image, calibration, time_max_val)

        # Convert to time/survival
        data = []
        for x, y in points:
            t = (x - x_0) / (x_max - x_0) * time_max_val
            surv = (y_0 - y) / (y_0 - y_100)
            surv = max(0, min(1, surv))
            data.append({'Time': t, 'Survival': surv})

        df = pd.DataFrame(data)
        df = self._enforce_monotonicity(df)
        df = self._resample_curve(df, time_max_val)

        # Validate: survival should not go below ~15% for this type of curve
        if df['Survival'].min() < 0.10:
            self._log(f"      Result suspicious (min survival={df['Survival'].min():.1%}), trying fallback")
            return self._extract_black_curve_fallback(image, calibration, time_max_val)

        return ExtractedCurveResult(
            identification=CurveIdentification(
                color='black',
                style='solid',
                description='black curve (smart extraction)',
                position='middle'
            ),
            dataframe=df,
            points_count=len(df),
            time_range=(df['Time'].min(), df['Time'].max()),
            survival_range=(df['Survival'].min(), df['Survival'].max()),
            confidence=0.7,
            extraction_method='smart_black'
        )

    def _extract_black_curve_fallback(self,
                                       image: np.ndarray,
                                       calibration: Dict[str, float],
                                       time_max: float) -> Optional[ExtractedCurveResult]:
        """
        Fallback extraction using censoring mark detection.

        Looks for + marks on the curve as anchor points.
        """
        from .color_transform import extract_curve_from_marks

        try:
            df = extract_curve_from_marks(image, calibration, exclude_colors=['green', 'red'])

            if df is not None and len(df) > 10:
                return ExtractedCurveResult(
                    identification=CurveIdentification(
                        color='black',
                        style='solid',
                        description='black curve (mark detection)',
                        position='middle'
                    ),
                    dataframe=df,
                    points_count=len(df),
                    time_range=(df['Time'].min(), df['Time'].max()),
                    survival_range=(df['Survival'].min(), df['Survival'].max()),
                    confidence=0.6,
                    extraction_method='mark_detection'
                )
        except Exception as e:
            self._log(f"      Mark detection failed: {e}")

        return None

    def extract_from_isolated(self,
                               isolated: IsolatedCurve,
                               calibration: Dict[str, float],
                               time_max: float = None) -> Optional[ExtractedCurveResult]:
        """
        Extract curve data from an isolated image.

        Args:
            isolated: Isolated curve data
            calibration: Calibration dict with x_0_pixel, x_max_pixel, etc.
            time_max: Maximum time value

        Returns:
            Extracted curve result or None if extraction fails
        """
        img = isolated.isolated_image

        # Determine extraction color
        if isolated.transformed:
            # After inversion, look for white/bright pixels
            extract_color = 'white'
        else:
            extract_color = isolated.extraction_color

        # Get pixel bounds
        x_0 = int(calibration['x_0_pixel'])
        x_max = int(calibration['x_max_pixel'])
        y_0 = int(calibration['y_0_pixel'])  # y for survival=0
        y_100 = int(calibration['y_100_pixel'])  # y for survival=1
        time_max_val = time_max or calibration.get('time_max', 24)

        # Scan for curve pixels
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if extract_color == 'white':
            # After inversion, the curve is white/bright
            mask = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 50)
        else:
            mask = self.create_color_mask(img, extract_color) > 0

        # Find curve points
        points = []
        for x in range(x_0, x_max + 1):
            col_mask = mask[:, x]
            y_indices = np.where(col_mask)[0]

            # Filter to plot area
            y_indices = y_indices[(y_indices >= y_100) & (y_indices <= y_0)]

            if len(y_indices) > 0:
                # Use median y for this x
                y = int(np.median(y_indices))
                points.append((x, y))

        if len(points) < 10:
            self._log(f"    Warning: Only {len(points)} points found for {isolated.identification.color}")
            return None

        # Convert to time/survival coordinates
        data = []
        for x, y in points:
            t = (x - x_0) / (x_max - x_0) * time_max_val
            s = (y_0 - y) / (y_0 - y_100)
            s = max(0, min(1, s))  # Clamp to [0, 1]
            data.append({'Time': t, 'Survival': s})

        df = pd.DataFrame(data)

        # Apply monotonicity constraint
        df = self._enforce_monotonicity(df)

        # Resample to regular intervals
        df = self._resample_curve(df, time_max_val)

        return ExtractedCurveResult(
            identification=isolated.identification,
            dataframe=df,
            points_count=len(df),
            time_range=(df['Time'].min(), df['Time'].max()),
            survival_range=(df['Survival'].min(), df['Survival'].max()),
            confidence=0.8,
            extraction_method='pixel'
        )

    def _enforce_monotonicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure survival is monotonically decreasing."""
        df = df.sort_values('Time').reset_index(drop=True)

        max_survival = 1.0
        survivals = []
        for s in df['Survival']:
            s = min(s, max_survival)
            survivals.append(s)
            max_survival = s

        df['Survival'] = survivals
        return df

    def _resample_curve(self, df: pd.DataFrame, time_max: float, step: float = 0.5) -> pd.DataFrame:
        """Resample curve to regular time intervals."""
        if len(df) < 2:
            return df

        # Create regular time grid
        times = np.arange(0, min(df['Time'].max(), time_max) + step, step)

        # Interpolate
        survivals = np.interp(times, df['Time'].values, df['Survival'].values)

        return pd.DataFrame({'Time': times, 'Survival': survivals})

    def extract(self,
                image_path: str,
                calibration: Dict[str, float],
                time_max: float = None,
                output_dir: str = None) -> HybridExtractionResult:
        """
        Run the full hybrid extraction pipeline.

        Args:
            image_path: Path to KM plot image
            calibration: Calibration dictionary
            time_max: Maximum time value
            output_dir: Output directory for results

        Returns:
            HybridExtractionResult with all extracted curves
        """
        self._log("Starting hybrid extraction...")

        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: AI identification
        identifications = self.identify_curves_with_ai(image_path)

        if not identifications:
            raise ValueError("No curves identified in image")

        # Deduplicate similar colors (e.g., black and gray)
        identifications = self._deduplicate_identifications(identifications)

        # Get all colors for isolation
        all_colors = [c.color for c in identifications if c.color != 'unknown']

        # Step 2: Isolate each curve
        self._log("  Step 2: Isolating curves...")
        isolated_curves = []
        for ident in identifications:
            if ident.color == 'unknown':
                self._log(f"    Skipping unknown color: {ident.description}")
                continue

            other_colors = [c for c in all_colors if c != ident.color]
            isolated = self.isolate_curve(image, ident, other_colors)
            isolated_curves.append(isolated)

            # Save isolated image if output dir specified
            if output_dir:
                out_path = os.path.join(output_dir, f"isolated_{ident.color}.png")
                cv2.imwrite(out_path, isolated.isolated_image)
                self._log(f"    Saved: {out_path}")

        # Step 3: Extract each curve
        self._log("  Step 3: Extracting curves...")
        extracted_curves = []
        black_curve_pending = None

        for isolated in isolated_curves:
            self._log(f"    Extracting {isolated.identification.color}...")

            # For black/gray curves, try regular extraction first
            # but we may need smart extraction later
            if isolated.identification.color in ['black', 'gray']:
                result = self.extract_from_isolated(isolated, calibration, time_max)

                # Check if result seems reasonable (not just axes/text)
                if result and result.survival_range[0] < 0.05:
                    # Survival goes too close to 0, probably picked up axes
                    self._log(f"      Result suspicious (survival min={result.survival_range[0]:.1%}), will try smart extraction")
                    black_curve_pending = isolated
                    continue
                elif result:
                    extracted_curves.append(result)
                    self._log(f"      Found {result.points_count} points, "
                             f"time: {result.time_range[0]:.1f}-{result.time_range[1]:.1f}, "
                             f"survival: {result.survival_range[1]:.1%}-{result.survival_range[0]:.1%}")
                else:
                    black_curve_pending = isolated
            else:
                result = self.extract_from_isolated(isolated, calibration, time_max)

                if result:
                    extracted_curves.append(result)
                    self._log(f"      Found {result.points_count} points, "
                             f"time: {result.time_range[0]:.1f}-{result.time_range[1]:.1f}, "
                             f"survival: {result.survival_range[1]:.1%}-{result.survival_range[0]:.1%}")
                else:
                    self._log(f"      Extraction failed for {isolated.identification.color}")

        # Try smart extraction for black curve if needed
        if black_curve_pending:
            self._log(f"    Trying specialized black curve extraction...")

            # First try mark-based extraction (most accurate for curves with censoring marks)
            result = self._extract_black_curve_fallback(image, calibration, time_max or calibration.get('time_max', 24))

            if result and result.survival_range[0] > 0.10:  # Reasonable result
                extracted_curves.append(result)
                self._log(f"      Mark-based extraction found {result.points_count} points, "
                         f"time: {result.time_range[0]:.1f}-{result.time_range[1]:.1f}, "
                         f"survival: {result.survival_range[1]:.1%}-{result.survival_range[0]:.1%}")
            elif extracted_curves:
                # Fall back to smart extraction using other curves as reference
                self._log(f"      Mark detection failed or suspicious, trying constraint-based extraction...")
                result = self.extract_black_curve_smart(image, calibration, extracted_curves, time_max)
                if result:
                    extracted_curves.append(result)
                    self._log(f"      Constraint-based extraction found {result.points_count} points, "
                             f"time: {result.time_range[0]:.1f}-{result.time_range[1]:.1f}, "
                             f"survival: {result.survival_range[1]:.1%}-{result.survival_range[0]:.1%}")
                else:
                    self._log(f"      All black curve extraction methods failed")

        if not extracted_curves:
            raise ValueError("No curves could be extracted")

        # Step 4: Combine results
        self._log("  Step 4: Combining results...")
        combined_df = self._combine_curves(extracted_curves)

        # Save individual curves
        if output_dir:
            for curve in extracted_curves:
                filename = f"curve_{curve.identification.color}.csv"
                path = os.path.join(output_dir, filename)
                curve.dataframe.to_csv(path, index=False)
                self._log(f"    Saved: {path}")

            # Save combined
            combined_path = os.path.join(output_dir, "all_curves.csv")
            combined_df.to_csv(combined_path, index=False)
            self._log(f"    Saved: {combined_path}")

        # Step 5: AI validation (if available)
        validation_result = None
        if output_dir and self.ai_service and self.ai_service.is_available:
            self._log("  Step 5: AI validation...")
            validation_result = self._validate_with_ai(
                image_path,
                extracted_curves,
                output_dir
            )

        result = HybridExtractionResult(
            curves=extracted_curves,
            combined_df=combined_df,
            ai_identification={
                'curves': [
                    {
                        'color': c.color,
                        'style': c.style,
                        'description': c.description,
                        'position': c.position
                    }
                    for c in identifications
                ]
            },
            validation_result=validation_result,
            output_dir=output_dir or ""
        )

        self.cleanup()
        self._log("Hybrid extraction complete!")

        return result

    def _combine_curves(self, curves: List[ExtractedCurveResult]) -> pd.DataFrame:
        """Combine all curves into a single DataFrame."""
        dfs = []
        for curve in curves:
            df = curve.dataframe.copy()
            df['Curve'] = curve.identification.color
            if curve.identification.legend_label:
                df['Label'] = curve.identification.legend_label
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _validate_with_ai(self,
                          original_path: str,
                          curves: List[ExtractedCurveResult],
                          output_dir: str) -> Optional[Dict[str, Any]]:
        """
        Validate extracted curves using AI.
        """
        try:
            # Create overlay image
            original = cv2.imread(original_path)
            overlay = original.copy()

            # Draw extracted curves on overlay
            colors_bgr = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'cyan': (255, 255, 0),
                'magenta': (255, 0, 255),
                'yellow': (0, 255, 255),
                'orange': (0, 165, 255),
                'black': (0, 0, 0),
                'gray': (128, 128, 128),
            }

            # We'd need calibration to draw... skip for now
            # Just save original and use AI to validate curve count

            overlay_path = os.path.join(output_dir, "overlay_validation.png")
            cv2.imwrite(overlay_path, original)

            if self.ai_service.capabilities.validation:
                result = self.ai_service.validate_extraction(
                    original_path,
                    overlay_path,
                    quiet=self.quiet
                )
                if result:
                    return {
                        'match': result.match,
                        'confidence': result.confidence,
                        'is_valid': result.is_valid,
                        'issues': result.issues,
                        'suggestions': result.suggestions
                    }
        except Exception as e:
            self._log(f"    Validation error: {e}")

        return None


def hybrid_extract(image_path: str,
                   calibration: Dict[str, float],
                   time_max: float = None,
                   output_dir: str = None,
                   quiet: bool = False,
                   use_ai: bool = True) -> HybridExtractionResult:
    """
    Convenience function for hybrid extraction.

    Args:
        image_path: Path to KM plot image
        calibration: Calibration dictionary with pixel bounds
        time_max: Maximum time value
        output_dir: Output directory
        quiet: Suppress progress messages
        use_ai: Whether to use AI for curve identification

    Returns:
        HybridExtractionResult
    """
    extractor = HybridExtractor(quiet=quiet, use_ai=use_ai)
    return extractor.extract(image_path, calibration, time_max, output_dir)
