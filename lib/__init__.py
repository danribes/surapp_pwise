# Kaplan-Meier Curve Extraction Library
# Main modules for curve detection and axis calibration

from .detector import LineStyleDetector, LineStyle, DetectedCurve, is_grayscale_image
from .calibrator_v2 import AxisCalibrator, AxisCalibration, calibrate_image
# Keep old import for backward compatibility
from .calibrator import AxisCalibrationResult
from .color_detector import (
    ColorCurveDetector, is_color_image, detect_curve_colors,
    extract_curves_with_overlap_handling
)

# At-risk table extraction (for Guyot algorithm compatibility)
from .at_risk_extractor import AtRiskExtractor, AtRiskData, extract_at_risk_table

# Curve quality control
from .curve_quality_control import (
    CurveQualityChecker, QCReport, CurveQCResult, QCIssue,
    QCIssueType, QCSeverity, run_quality_control, format_qc_report
)

# Extraction validation
from .validator import (
    KMCurveValidator, ValidationReport, ValidationResult, ValidationStatus,
    validate_extraction
)

# AI modules (optional - require ollama package)
try:
    from .ai_validator import AIValidator, check_ollama_status
    from .ai_config import AIConfig, ValidationResult as AIValidationResult, ExtractionParameters
    _AI_VALIDATOR_AVAILABLE = True
except ImportError:
    _AI_VALIDATOR_AVAILABLE = False

# Extended AI modules (axis detection, table reading, curve reconstruction)
try:
    from .ai_axis_detector import AIAxisDetector, AIAxisResult, get_ai_axis_detector
    _AI_AXIS_AVAILABLE = True
except ImportError:
    _AI_AXIS_AVAILABLE = False

try:
    from .ai_table_reader import AITableReader, AITableResult, get_ai_table_reader
    _AI_TABLE_AVAILABLE = True
except ImportError:
    _AI_TABLE_AVAILABLE = False

try:
    from .ai_service import AIService, AICapabilities, get_ai_service, is_ai_available
    _AI_SERVICE_AVAILABLE = True
except ImportError:
    _AI_SERVICE_AVAILABLE = False

try:
    from .ai_curve_extractor import AICurveExtractor, AIExtractionResult, ExtractedCurve, get_ai_curve_extractor
    _AI_CURVE_EXTRACTOR_AVAILABLE = True
except ImportError:
    _AI_CURVE_EXTRACTOR_AVAILABLE = False

# Combined AI availability
_AI_AVAILABLE = _AI_VALIDATOR_AVAILABLE

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
    'AxisCalibration',
    'AxisCalibrationResult',
    'calibrate_image',
    # At-risk table extraction
    'AtRiskExtractor',
    'AtRiskData',
    'extract_at_risk_table',
    # Curve quality control
    'CurveQualityChecker',
    'QCReport',
    'CurveQCResult',
    'QCIssue',
    'QCIssueType',
    'QCSeverity',
    'run_quality_control',
    'format_qc_report',
    # Extraction validation
    'KMCurveValidator',
    'ValidationReport',
    'ValidationResult',
    'ValidationStatus',
    'validate_extraction',
]

# Add AI exports if available
if _AI_VALIDATOR_AVAILABLE:
    __all__.extend([
        'AIValidator',
        'AIConfig',
        'AIValidationResult',
        'ExtractionParameters',
        'check_ollama_status',
    ])

if _AI_AXIS_AVAILABLE:
    __all__.extend([
        'AIAxisDetector',
        'AIAxisResult',
        'get_ai_axis_detector',
    ])

if _AI_TABLE_AVAILABLE:
    __all__.extend([
        'AITableReader',
        'AITableResult',
        'get_ai_table_reader',
    ])

if _AI_SERVICE_AVAILABLE:
    __all__.extend([
        'AIService',
        'AICapabilities',
        'get_ai_service',
        'is_ai_available',
    ])

if _AI_CURVE_EXTRACTOR_AVAILABLE:
    __all__.extend([
        'AICurveExtractor',
        'AIExtractionResult',
        'ExtractedCurve',
        'get_ai_curve_extractor',
    ])

# Hybrid extractor (AI + pixel-based)
try:
    from .hybrid_extractor import (
        HybridExtractor, HybridExtractionResult,
        CurveIdentification, IsolatedCurve, ExtractedCurveResult,
        hybrid_extract
    )
    _HYBRID_AVAILABLE = True
    __all__.extend([
        'HybridExtractor',
        'HybridExtractionResult',
        'CurveIdentification',
        'IsolatedCurve',
        'ExtractedCurveResult',
        'hybrid_extract',
    ])
except ImportError:
    _HYBRID_AVAILABLE = False

# Color transformation utilities
try:
    from .color_transform import (
        ColorTransformer, ColorTransformResult,
        extract_curve_from_marks, validate_curve_extraction,
        auto_transform_and_extract
    )
    _COLOR_TRANSFORM_AVAILABLE = True
    __all__.extend([
        'ColorTransformer',
        'ColorTransformResult',
        'extract_curve_from_marks',
        'validate_curve_extraction',
        'auto_transform_and_extract',
    ])
except ImportError:
    _COLOR_TRANSFORM_AVAILABLE = False
