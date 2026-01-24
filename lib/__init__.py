# Kaplan-Meier Curve Extraction Library
# Main modules for curve detection and axis calibration

from .detector import LineStyleDetector, LineStyle, DetectedCurve, is_grayscale_image
from .calibrator import AxisCalibrator, AxisCalibrationResult
from .color_detector import (
    ColorCurveDetector, is_color_image, detect_curve_colors,
    extract_curves_with_overlap_handling
)

__all__ = [
    # Grayscale/line-style detection
    'LineStyleDetector',
    'LineStyle',
    'DetectedCurve',
    'is_grayscale_image',
    # Color-based detection
    'ColorCurveDetector',
    'is_color_image',
    'detect_curve_colors',
    # Axis calibration
    'AxisCalibrator',
    'AxisCalibrationResult',
]
