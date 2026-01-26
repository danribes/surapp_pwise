"""
Unified AI Service for KM Curve Extraction.

Consolidates all AI functionality (validation, axis detection, curve reconstruction,
table reading) into a single service interface.

Usage:
    from lib.ai_service import AIService

    service = AIService()

    if service.is_available:
        # AI-assisted extraction
        axes = service.detect_axes(image_path)
        table = service.read_at_risk_table(table_image)
        validation = service.validate_extraction(original, overlay)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any

import cv2
import numpy as np

# Import configuration
from .ai_config import AIConfig

# Import AI components
try:
    from .ai_validator import AIValidator, ValidationResult, check_ollama_status
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False
    AIValidator = None
    ValidationResult = None

try:
    from .ai_axis_detector import AIAxisDetector, AIAxisResult, get_ai_axis_detector
    HAS_AXIS_DETECTOR = True
except ImportError:
    HAS_AXIS_DETECTOR = False
    AIAxisDetector = None
    AIAxisResult = None

try:
    from .ai_table_reader import AITableReader, AITableResult, get_ai_table_reader
    HAS_TABLE_READER = True
except ImportError:
    HAS_TABLE_READER = False
    AITableReader = None
    AITableResult = None

try:
    from .ai_curve_extractor import AICurveExtractor, AIExtractionResult, ExtractedCurve, get_ai_curve_extractor
    HAS_CURVE_EXTRACTOR = True
except ImportError:
    HAS_CURVE_EXTRACTOR = False
    AICurveExtractor = None
    AIExtractionResult = None
    ExtractedCurve = None


@dataclass
class AICapabilities:
    """Available AI capabilities."""
    validation: bool = False
    axis_detection: bool = False
    table_reading: bool = False
    curve_reconstruction: bool = False
    curve_extraction: bool = False

    @property
    def any_available(self) -> bool:
        """Check if any AI capability is available."""
        return any([
            self.validation,
            self.axis_detection,
            self.table_reading,
            self.curve_reconstruction,
            self.curve_extraction
        ])

    def __str__(self) -> str:
        caps = []
        if self.validation:
            caps.append("validation")
        if self.axis_detection:
            caps.append("axis_detection")
        if self.table_reading:
            caps.append("table_reading")
        if self.curve_reconstruction:
            caps.append("curve_reconstruction")
        if self.curve_extraction:
            caps.append("curve_extraction")
        return f"AICapabilities({', '.join(caps) if caps else 'none'})"


class AIService:
    """
    Unified AI service for KM curve extraction assistance.

    Provides a single interface to all AI capabilities:
    - Post-extraction validation
    - Axis detection and calibration
    - Curve reconstruction for compressed images
    - At-risk table reading

    All operations gracefully degrade when AI is unavailable.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize the AI service.

        Args:
            config: AI configuration. If None, loads from environment.
        """
        self.config = config or AIConfig.from_environment()
        self._status_checked = False
        self._available = False
        self._capabilities = None

        # Lazy-loaded components
        self._validator = None
        self._axis_detector = None
        self._table_reader = None
        self._curve_extractor = None

    def _check_status(self) -> bool:
        """Check if AI service is reachable."""
        if self._status_checked:
            return self._available

        if not self.config.enabled:
            self._status_checked = True
            self._available = False
            return False

        if HAS_VALIDATOR:
            status = check_ollama_status(self.config.host)
            self._available = status.get('available', False)
        else:
            self._available = False

        self._status_checked = True
        return self._available

    @property
    def is_available(self) -> bool:
        """Check if AI service is available and reachable."""
        return self._check_status()

    @property
    def capabilities(self) -> AICapabilities:
        """Get available AI capabilities."""
        if self._capabilities is not None:
            return self._capabilities

        caps = AICapabilities()

        if not self.is_available:
            self._capabilities = caps
            return caps

        # Check each component
        caps.validation = HAS_VALIDATOR
        caps.axis_detection = HAS_AXIS_DETECTOR
        caps.table_reading = HAS_TABLE_READER
        caps.curve_reconstruction = HAS_TABLE_READER  # Uses same AI connection
        caps.curve_extraction = HAS_CURVE_EXTRACTOR

        self._capabilities = caps
        return caps

    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status of the AI service.

        Returns:
            Dictionary with status information
        """
        status = {
            'enabled': self.config.enabled,
            'host': self.config.host,
            'model': self.config.model,
            'available': self.is_available,
            'capabilities': {
                'validation': self.capabilities.validation,
                'axis_detection': self.capabilities.axis_detection,
                'table_reading': self.capabilities.table_reading,
                'curve_reconstruction': self.capabilities.curve_reconstruction,
                'curve_extraction': self.capabilities.curve_extraction
            }
        }

        if HAS_VALIDATOR and self.is_available:
            ollama_status = check_ollama_status(self.config.host)
            status['models'] = ollama_status.get('models', [])
            status['error'] = ollama_status.get('error')

        return status

    # =========================================================================
    # Validation
    # =========================================================================

    @property
    def validator(self) -> Optional['AIValidator']:
        """Get the validator component (lazy initialization)."""
        if self._validator is not None:
            return self._validator

        if not HAS_VALIDATOR or not self.is_available:
            return None

        self._validator = AIValidator(self.config)
        return self._validator

    def validate_extraction(
        self,
        original_image: Union[str, Path],
        overlay_image: Union[str, Path],
        quiet: bool = True
    ) -> Optional['ValidationResult']:
        """
        Validate extracted curves against original image.

        Args:
            original_image: Path to original KM plot image
            overlay_image: Path to image with extracted curves overlaid
            quiet: Suppress progress messages

        Returns:
            ValidationResult if validation succeeded, None otherwise
        """
        if self.validator is None:
            return None

        return self.validator.validate(
            str(original_image),
            str(overlay_image),
            quiet=quiet
        )

    # =========================================================================
    # Axis Detection
    # =========================================================================

    @property
    def axis_detector(self) -> Optional['AIAxisDetector']:
        """Get the axis detector component (lazy initialization)."""
        if self._axis_detector is not None:
            return self._axis_detector

        if not HAS_AXIS_DETECTOR or not self.is_available:
            return None

        self._axis_detector = get_ai_axis_detector(self.config)
        return self._axis_detector

    def detect_axes(
        self,
        image: Union[str, Path, np.ndarray],
        quiet: bool = True
    ) -> Optional['AIAxisResult']:
        """
        Detect axis labels and ranges using AI.

        Args:
            image: Path to image or numpy array (BGR)
            quiet: Suppress progress messages

        Returns:
            AIAxisResult if detection succeeded, None otherwise
        """
        if self.axis_detector is None:
            return None

        return self.axis_detector.detect_axes(image, quiet=quiet)

    # =========================================================================
    # Table Reading
    # =========================================================================

    @property
    def table_reader(self) -> Optional['AITableReader']:
        """Get the table reader component (lazy initialization)."""
        if self._table_reader is not None:
            return self._table_reader

        if not HAS_TABLE_READER or not self.is_available:
            return None

        self._table_reader = get_ai_table_reader(self.config)
        return self._table_reader

    @property
    def curve_extractor(self) -> Optional['AICurveExtractor']:
        """Get the curve extractor component (lazy initialization)."""
        if self._curve_extractor is not None:
            return self._curve_extractor

        if not HAS_CURVE_EXTRACTOR or not self.is_available:
            return None

        self._curve_extractor = get_ai_curve_extractor(self.config)
        return self._curve_extractor

    def extract_curves(
        self,
        image: Union[str, Path, np.ndarray],
        time_max: float = 24.0,
        time_step: float = 1.0,
        quiet: bool = True
    ) -> Optional['AIExtractionResult']:
        """
        Extract curve points directly using AI vision.

        This method uses AI to read survival values directly from the image
        at specified time points, rather than relying on pixel detection.
        Can be more accurate for compressed or low-quality images.

        Args:
            image: Path to image or numpy array (BGR format)
            time_max: Maximum time value to extract
            time_step: Step size for time points (smaller = more detail)
            quiet: Suppress progress messages

        Returns:
            AIExtractionResult if successful, None otherwise
        """
        if self.curve_extractor is None:
            return None

        return self.curve_extractor.extract_curves(
            image, time_max=time_max, time_step=time_step, quiet=quiet
        )

    def extract_single_curve(
        self,
        image: Union[str, Path, np.ndarray],
        curve_description: str,
        time_points: List[float],
        quiet: bool = True
    ) -> Optional['ExtractedCurve']:
        """
        Extract points for a single specified curve.

        Args:
            image: Path to image or numpy array
            curve_description: Description of the curve (e.g., "cyan", "upper curve")
            time_points: List of time values to read
            quiet: Suppress progress messages

        Returns:
            ExtractedCurve if successful, None otherwise
        """
        if self.curve_extractor is None:
            return None

        return self.curve_extractor.extract_single_curve(
            image, curve_description, time_points, quiet=quiet
        )

    def read_at_risk_table(
        self,
        table_image: Union[str, Path, np.ndarray],
        quiet: bool = True
    ) -> Optional['AITableResult']:
        """
        Read at-risk table using AI.

        Args:
            table_image: Path to image or numpy array (BGR) of table region
            quiet: Suppress progress messages

        Returns:
            AITableResult if reading succeeded, None otherwise
        """
        if self.table_reader is None:
            return None

        return self.table_reader.read_table(table_image, quiet=quiet)

    # =========================================================================
    # Curve Reconstruction
    # =========================================================================

    def reconstruct_curve(
        self,
        image: Union[str, Path, np.ndarray],
        curve_points: List[Tuple[float, float]],
        curve_color: str,
        time_max: float = 24.0,
        quiet: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Reconstruct missing portions of a curve using AI.

        When detected curve coverage is low, uses AI to trace the likely
        curve path in missing regions.

        Args:
            image: Path to image or numpy array (BGR)
            curve_points: Detected (time, survival) points
            curve_color: Color name of the curve
            time_max: Maximum time value
            quiet: Suppress progress messages

        Returns:
            Enhanced list of (time, survival) points
        """
        if not self.is_available:
            return curve_points

        # Import here to avoid circular import
        try:
            from .color_detector import ColorCurveDetector
        except ImportError:
            return curve_points

        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                return curve_points
        else:
            img = image

        # Create detector with minimal bounds (just for AI call)
        h, w = img.shape[:2]
        detector = ColorCurveDetector(img, (0, 0, w, h))

        return detector.ai_reconstruct_curve(
            curve_points, curve_color, time_max,
            ai_config=self.config, quiet=quiet
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def assist_extraction(
        self,
        image_path: Union[str, Path],
        use_axis_detection: bool = True,
        use_validation: bool = True,
        time_max: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
        quiet: bool = True
    ) -> Dict[str, Any]:
        """
        Provide AI assistance for the full extraction pipeline.

        This convenience method orchestrates AI assistance for:
        1. Axis detection (pre-extraction)
        2. Extraction validation (post-extraction)

        Args:
            image_path: Path to the KM plot image
            use_axis_detection: Whether to use AI for axis detection
            use_validation: Whether to validate extraction results
            time_max: Optional maximum time value hint
            output_dir: Output directory for validation images
            quiet: Suppress progress messages

        Returns:
            Dictionary with AI assistance results
        """
        results = {
            'axis_detection': None,
            'validation': None,
            'recommendations': []
        }

        image_path = Path(image_path)

        # Pre-extraction: Axis detection
        if use_axis_detection and self.capabilities.axis_detection:
            if not quiet:
                print("Running AI axis detection...")

            axis_result = self.detect_axes(str(image_path), quiet=quiet)
            if axis_result and axis_result.is_valid:
                results['axis_detection'] = {
                    'x_range': axis_result.x_range,
                    'y_range': axis_result.y_range,
                    'x_label': axis_result.x_label,
                    'y_label': axis_result.y_label,
                    'curve_count': axis_result.curve_count,
                    'confidence': axis_result.confidence
                }

                if time_max is None and axis_result.x_range:
                    results['recommendations'].append(
                        f"Use --time-max {axis_result.x_range[1]}"
                    )

        # Post-extraction: Validation (if output_dir provided)
        if use_validation and output_dir and self.capabilities.validation:
            output_dir = Path(output_dir)
            overlay_path = output_dir / "comparison_overlay.png"

            if overlay_path.exists():
                if not quiet:
                    print("Running AI validation...")

                validation = self.validate_extraction(
                    str(image_path), str(overlay_path), quiet=quiet
                )

                if validation:
                    results['validation'] = {
                        'match': validation.match,
                        'confidence': validation.confidence,
                        'is_valid': validation.is_valid,
                        'issues': validation.issues,
                        'suggestions': validation.suggestions
                    }

                    if not validation.is_valid:
                        results['recommendations'].extend(validation.suggestions)

        return results


def get_ai_service(config: Optional[AIConfig] = None) -> AIService:
    """
    Get an AI service instance.

    Args:
        config: AI configuration. If None, loads from environment.

    Returns:
        AIService instance (may not be available - check is_available)
    """
    return AIService(config)


def is_ai_available(config: Optional[AIConfig] = None) -> bool:
    """
    Quick check if AI service is available.

    Args:
        config: AI configuration. If None, loads from environment.

    Returns:
        True if AI service is available and ready
    """
    return get_ai_service(config).is_available
