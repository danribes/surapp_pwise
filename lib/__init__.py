# Kaplan-Meier Curve Extraction Library
# Main modules for curve detection and axis calibration

from .detector import LineStyleDetector, LineStyle, DetectedCurve, is_grayscale_image
from .calibrator import AxisCalibrator, AxisCalibrationResult
from .color_detector import (
    ColorCurveDetector, is_color_image, detect_curve_colors,
    extract_curves_with_overlap_handling
)

# AI modules (optional - require ollama package)
try:
    from .ai_validator import AIValidator, check_ollama_status
    from .ai_config import AIConfig, ValidationResult, ExtractionParameters
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False

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

# Add AI exports if available
if _AI_AVAILABLE:
    __all__.extend([
        'AIValidator',
        'AIConfig',
        'ValidationResult',
        'ExtractionParameters',
        'check_ollama_status',
    ])
