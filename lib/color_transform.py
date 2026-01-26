"""
Color transformation module for improving curve extraction.

When curves are difficult to extract due to color similarity with background
elements (e.g., black curves on white background with black axes/text),
this module applies automatic color transformations to improve visibility.
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import tempfile
import os


@dataclass
class ColorTransformResult:
    """Result of a color transformation attempt."""
    transform_name: str
    transformed_image: np.ndarray
    temp_path: str
    curve_color_in_transformed: str  # What color the target curve appears as after transform


class ColorTransformer:
    """
    Automatically applies color transformations to improve curve extraction.

    Strategies:
    1. Inversion - for dark curves (black, dark gray)
    2. Channel isolation - for colored curves mixed with similar colors
    3. HSV manipulation - for curves with specific hue ranges
    """

    def __init__(self, image: np.ndarray):
        """
        Initialize with the original image.

        Args:
            image: BGR image array
        """
        self.original = image.copy()
        self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.temp_files = []

    def cleanup(self):
        """Remove temporary files."""
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)
        self.temp_files = []

    def detect_difficult_colors(self) -> List[str]:
        """
        Detect which curve colors might be difficult to extract.

        Returns:
            List of color names that may need transformation
        """
        difficult = []

        # Check for black/dark gray curves
        # These are difficult because they blend with axes, text, gridlines
        dark_pixels = (self.gray < 80) & (self.hsv[:,:,1] < 50)  # Low saturation = grayscale
        if np.sum(dark_pixels) > 500:  # Significant dark pixels present
            difficult.append('black')

        return difficult

    def isolate_color(self, target_color: str) -> Optional[np.ndarray]:
        """
        Create an image with only the target color visible.

        Args:
            target_color: Color name ('black', 'red', 'green', 'blue', etc.)

        Returns:
            Image with only target color pixels, others set to white
        """
        h, s, v = cv2.split(self.hsv)

        if target_color == 'black':
            # Black: low saturation, low-medium value
            mask = (s < 50) & (v > 10) & (v < 120)
        elif target_color == 'gray':
            # Gray: low saturation, medium value
            mask = (s < 50) & (v >= 80) & (v < 180)
        elif target_color == 'red':
            # Red wraps around in HSV
            mask = ((h < 10) | (h > 170)) & (s > 50) & (v > 50)
        elif target_color == 'green':
            mask = (h >= 35) & (h <= 85) & (s > 50) & (v > 50)
        elif target_color == 'blue':
            mask = (h >= 100) & (h <= 130) & (s > 50) & (v > 50)
        elif target_color == 'cyan':
            mask = (h >= 85) & (h <= 100) & (s > 50) & (v > 50)
        else:
            return None

        # Create isolated image
        result = np.full_like(self.original, 255)  # White background
        result[mask] = self.original[mask]

        return result

    def remove_colors(self, colors_to_remove: List[str]) -> np.ndarray:
        """
        Remove specified colors from image, setting them to white.

        Args:
            colors_to_remove: List of color names to remove

        Returns:
            Image with specified colors removed
        """
        result = self.original.copy()
        h, s, v = cv2.split(self.hsv)

        combined_mask = np.zeros(self.gray.shape, dtype=bool)

        for color in colors_to_remove:
            if color == 'red':
                mask = ((h < 10) | (h > 170)) & (s > 50) & (v > 50)
            elif color == 'green':
                mask = (h >= 35) & (h <= 85) & (s > 50) & (v > 50)
            elif color == 'blue':
                mask = (h >= 100) & (h <= 130) & (s > 50) & (v > 50)
            elif color == 'cyan':
                mask = (h >= 85) & (h <= 100) & (s > 50) & (v > 50)
            else:
                continue
            combined_mask = combined_mask | mask

        result[combined_mask] = [255, 255, 255]
        return result

    def invert(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Invert image colors.

        Args:
            image: Image to invert (uses original if None)

        Returns:
            Inverted image
        """
        img = image if image is not None else self.original
        return cv2.bitwise_not(img)

    def enhance_contrast(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enhance contrast using CLAHE.

        Args:
            image: Image to enhance (uses original if None)

        Returns:
            Contrast-enhanced image
        """
        img = image if image is not None else self.original

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def get_transforms_for_color(self, target_color: str,
                                  other_colors: List[str] = None) -> List[ColorTransformResult]:
        """
        Get appropriate transformations to extract a specific color curve.

        Args:
            target_color: The color of the curve to extract
            other_colors: Other curve colors present in the image

        Returns:
            List of transformation results to try
        """
        transforms = []
        other_colors = other_colors or []

        if target_color in ['black', 'gray']:
            # Strategy 1: Remove other colored curves, then invert
            if other_colors:
                img_no_colors = self.remove_colors(other_colors)
                img_inverted = self.invert(img_no_colors)

                # Save to temp file
                temp_path = tempfile.mktemp(suffix='_inverted.png')
                cv2.imwrite(temp_path, img_inverted)
                self.temp_files.append(temp_path)

                transforms.append(ColorTransformResult(
                    transform_name=f"remove_{'+'.join(other_colors)}_then_invert",
                    transformed_image=img_inverted,
                    temp_path=temp_path,
                    curve_color_in_transformed='white'  # Black becomes white after inversion
                ))

            # Strategy 2: Just invert the original
            img_inverted = self.invert()
            temp_path = tempfile.mktemp(suffix='_inverted_full.png')
            cv2.imwrite(temp_path, img_inverted)
            self.temp_files.append(temp_path)

            transforms.append(ColorTransformResult(
                transform_name="invert_full",
                transformed_image=img_inverted,
                temp_path=temp_path,
                curve_color_in_transformed='white'
            ))

            # Strategy 3: Isolate black pixels and invert
            img_isolated = self.isolate_color(target_color)
            if img_isolated is not None:
                img_isolated_inv = self.invert(img_isolated)
                temp_path = tempfile.mktemp(suffix='_isolated_inverted.png')
                cv2.imwrite(temp_path, img_isolated_inv)
                self.temp_files.append(temp_path)

                transforms.append(ColorTransformResult(
                    transform_name=f"isolate_{target_color}_then_invert",
                    transformed_image=img_isolated_inv,
                    temp_path=temp_path,
                    curve_color_in_transformed='white'
                ))

        elif target_color in ['red', 'green', 'blue', 'cyan']:
            # For colored curves, try removing other colors first
            if other_colors:
                img_no_others = self.remove_colors(other_colors)
                temp_path = tempfile.mktemp(suffix='_cleaned.png')
                cv2.imwrite(temp_path, img_no_others)
                self.temp_files.append(temp_path)

                transforms.append(ColorTransformResult(
                    transform_name=f"remove_{'+'.join(other_colors)}",
                    transformed_image=img_no_others,
                    temp_path=temp_path,
                    curve_color_in_transformed=target_color
                ))

            # Try contrast enhancement
            img_enhanced = self.enhance_contrast()
            temp_path = tempfile.mktemp(suffix='_enhanced.png')
            cv2.imwrite(temp_path, img_enhanced)
            self.temp_files.append(temp_path)

            transforms.append(ColorTransformResult(
                transform_name="contrast_enhanced",
                transformed_image=img_enhanced,
                temp_path=temp_path,
                curve_color_in_transformed=target_color
            ))

        return transforms


def extract_curve_from_marks(image: np.ndarray,
                             calibration: Dict,
                             exclude_colors: List[str] = None) -> Optional[pd.DataFrame]:
    """
    Extract a curve by detecting censoring marks (+) as anchor points.

    This is a fallback method for black/dark curves that are difficult to
    separate from axes and text. It detects the small + marks that indicate
    censoring events and interpolates the curve between them.

    Args:
        image: BGR image array
        calibration: Dict with keys 'x_0_pixel', 'x_max_pixel', 'y_0_pixel',
                     'y_100_pixel', 'time_max'
        exclude_colors: List of colors to exclude (e.g., ['green', 'red'])

    Returns:
        DataFrame with 'Time' and 'Survival' columns, or None if extraction fails
    """
    import pandas as pd
    from scipy import ndimage

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Create masks for colors to exclude
    exclude_mask = np.zeros(image.shape[:2], dtype=bool)
    exclude_colors = exclude_colors or []

    for color in exclude_colors:
        if color == 'green':
            exclude_mask |= (h >= 35) & (h <= 85) & (s > 50) & (v > 50)
        elif color == 'red':
            exclude_mask |= ((h < 10) | (h > 170)) & (s > 50) & (v > 50)
        elif color == 'blue':
            exclude_mask |= (h >= 100) & (h <= 130) & (s > 50) & (v > 50)
        elif color == 'cyan':
            exclude_mask |= (h >= 85) & (h <= 100) & (s > 50) & (v > 50)

    # Detect black pixels (low saturation, low-medium value)
    black_only = (s < 30) & (v > 20) & (v < 100) & ~exclude_mask

    # The censoring marks (+) are small cross-shaped features
    cross_kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)

    black_uint8 = black_only.astype(np.uint8) * 255
    eroded = cv2.erode(black_uint8, cross_kernel, iterations=1)

    # Find connected components
    labeled, num_features = ndimage.label(eroded)

    # Extract calibration values
    x_0 = calibration['x_0_pixel']
    x_max = calibration['x_max_pixel']
    y_0 = calibration['y_0_pixel']  # y for survival=0
    y_100 = calibration['y_100_pixel']  # y for survival=1
    time_max = calibration['time_max']

    def pixel_to_coords(px, py):
        t = (px - x_0) / (x_max - x_0) * time_max
        s = (y_0 - py) / (y_0 - y_100)
        return t, s

    # Get centroids of mark components
    mark_points = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        y_coords, x_coords = np.where(component)
        if len(x_coords) > 0:
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))

            # Check if in plot area and not too big
            if (x_0 <= cx <= x_max and y_100 <= cy <= y_0 and
                len(x_coords) < 50 and not exclude_mask[cy, cx]):
                t, s = pixel_to_coords(cx, cy)
                if 0 <= t <= time_max and 0 <= s <= 1:
                    mark_points.append((t, s))

    if len(mark_points) < 3:
        return None

    # Sort by time and apply monotonicity
    mark_points.sort(key=lambda p: p[0])

    mono_points = [(0.0, 1.0)]
    last_s = 1.0
    for t, s in mark_points:
        if s < last_s + 0.03:
            mono_points.append((t, min(s, last_s)))
            last_s = min(last_s, s)

    if len(mono_points) < 3:
        return None

    # Interpolate to fine grid
    df = pd.DataFrame(mono_points, columns=['Time', 'Survival'])
    times = np.arange(0, df['Time'].max() + 0.5, 0.5)
    survivals = np.interp(times, df['Time'], df['Survival'])

    return pd.DataFrame({'Time': times, 'Survival': survivals})


def validate_curve_extraction(extracted_df: pd.DataFrame,
                               reference_df: pd.DataFrame = None,
                               expected_end_survival: Tuple[float, float] = None) -> bool:
    """
    Validate if an extracted curve is reasonable.

    Args:
        extracted_df: Extracted curve DataFrame
        reference_df: Optional reference curve for comparison
        expected_end_survival: Optional (min, max) tuple for expected end survival

    Returns:
        True if extraction appears valid, False otherwise
    """
    if extracted_df is None or len(extracted_df) < 10:
        return False

    # Check monotonicity
    diffs = extracted_df['Survival'].diff().dropna()
    if (diffs > 0.01).sum() > len(diffs) * 0.1:  # More than 10% increasing
        return False

    # Check time coverage
    if extracted_df['Time'].max() < 50:  # Should cover at least 50 units
        return False

    # Check end survival if expected range provided
    if expected_end_survival:
        end_s = extracted_df['Survival'].iloc[-1]
        if not (expected_end_survival[0] <= end_s <= expected_end_survival[1]):
            return False

    return True


def auto_transform_and_extract(image_path: str,
                                difficult_colors: List[str],
                                other_colors: List[str],
                                output_dir: str) -> Dict[str, str]:
    """
    Automatically transform image and save variants for difficult colors.

    Args:
        image_path: Path to original image
        difficult_colors: Colors that are difficult to extract
        other_colors: Other colors present in the image
        output_dir: Directory to save transformed images

    Returns:
        Dict mapping color name to best transformed image path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    transformer = ColorTransformer(img)

    results = {}

    for color in difficult_colors:
        transforms = transformer.get_transforms_for_color(color, other_colors)

        if transforms:
            # Use the first transform (usually the best strategy)
            best = transforms[0]

            # Save with descriptive name
            output_path = os.path.join(output_dir, f"transformed_for_{color}.png")
            cv2.imwrite(output_path, best.transformed_image)

            results[color] = {
                'path': output_path,
                'transform': best.transform_name,
                'target_color_after': best.curve_color_in_transformed
            }

            print(f"  {color}: applied '{best.transform_name}' -> {output_path}")

    transformer.cleanup()
    return results
