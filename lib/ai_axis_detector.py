"""
AI-Assisted Axis Detection for KM Curve Extraction.

Uses vision models (Ollama + llama3.2-vision) to detect axis labels and ranges
when traditional OCR/Hough methods have low confidence.
"""

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Union

import cv2
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .ai_config import AIConfig


@dataclass
class AIAxisResult:
    """Result from AI axis detection."""
    x_range: Optional[Tuple[float, float]] = None  # (min, max)
    y_range: Optional[Tuple[float, float]] = None  # (min, max)
    x_label: Optional[str] = None  # e.g., "Time (months)"
    y_label: Optional[str] = None  # e.g., "Survival probability"
    curve_count: Optional[int] = None
    confidence: float = 0.0
    raw_response: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if result contains usable axis information."""
        return (self.x_range is not None and self.y_range is not None
                and self.confidence >= 0.5)


class AIAxisDetector:
    """Use vision model to detect axis labels and ranges."""

    # Prompt template for axis detection
    AXIS_DETECTION_PROMPT = """Analyze this Kaplan-Meier survival plot image.
Identify:
1. X-axis label and numeric range (e.g., "Time (months): 0 to 24")
2. Y-axis label and numeric range (e.g., "Survival probability: 0 to 1.0")
3. Number of distinct curves visible

Look at the axis tick mark labels to determine the exact numeric ranges.

Respond in EXACTLY this format (use the exact labels shown):
X_LABEL: <axis label text>
X_RANGE: <min>, <max>
Y_LABEL: <axis label text>
Y_RANGE: <min>, <max>
CURVES: <count>
CONFIDENCE: <0.0-1.0>

Example response:
X_LABEL: Time (months)
X_RANGE: 0, 24
Y_LABEL: Overall survival
Y_RANGE: 0.0, 1.0
CURVES: 2
CONFIDENCE: 0.85
"""

    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize the detector.

        Args:
            config: AI configuration. If None, loads from environment.
        """
        self.config = config or AIConfig.from_environment()
        self._model_ready = False

    @property
    def is_available(self) -> bool:
        """Check if AI axis detection is available."""
        if not OLLAMA_AVAILABLE:
            return False
        if not self.config.enabled:
            return False
        return self._check_connection()

    def _check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            client = ollama.Client(host=self.config.host)
            client.list()
            return True
        except Exception:
            return False

    def _ensure_model(self, quiet: bool = True) -> bool:
        """Ensure the vision model is available."""
        if self._model_ready:
            return True

        try:
            client = ollama.Client(host=self.config.host)
            models = client.list()
            model_names = [m.get('name', '').split(':')[0] for m in models.get('models', [])]

            if self.config.model.split(':')[0] not in model_names:
                if not quiet:
                    print(f"Pulling {self.config.model} model...")
                client.pull(self.config.model)

            self._model_ready = True
            return True
        except Exception as e:
            if not quiet:
                print(f"Error ensuring model: {e}")
            return False

    def _encode_image(self, image: Union[str, Path, np.ndarray]) -> str:
        """Encode image to base64 for API.

        Args:
            image: Path to image file or numpy array (BGR format)

        Returns:
            Base64 encoded string
        """
        if isinstance(image, (str, Path)):
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            # NumPy array - encode as PNG
            success, buffer = cv2.imencode('.png', image)
            if not success:
                raise ValueError("Failed to encode image")
            return base64.b64encode(buffer).decode('utf-8')

    def _parse_response(self, response_text: str) -> AIAxisResult:
        """Parse the AI response into structured result.

        Handles both plain text and markdown formatted responses.
        """
        result = AIAxisResult(raw_response=response_text)

        # Remove markdown formatting (**, *, #, etc.)
        clean_text = re.sub(r'\*+', '', response_text)
        clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)

        lines = clean_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Normalize line - remove leading bullets/dashes
            line = re.sub(r'^[-*•]\s*', '', line)

            # Parse X_LABEL (handles "X_LABEL: value" or "X-LABEL: value")
            if re.match(r'^X[-_]?LABEL\s*:', line, re.IGNORECASE):
                result.x_label = re.split(r':\s*', line, 1)[1].strip()

            # Parse X_RANGE
            elif re.match(r'^X[-_]?RANGE\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    # Parse "min, max" format
                    parts = re.split(r'[,\s]+', value)
                    nums = [float(re.sub(r'[^\d.\-]', '', p)) for p in parts if re.search(r'\d', p)]
                    if len(nums) >= 2:
                        result.x_range = (nums[0], nums[1])
                except (ValueError, IndexError):
                    pass

            # Parse Y_LABEL
            elif re.match(r'^Y[-_]?LABEL\s*:', line, re.IGNORECASE):
                result.y_label = re.split(r':\s*', line, 1)[1].strip()

            # Parse Y_RANGE
            elif re.match(r'^Y[-_]?RANGE\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    parts = re.split(r'[,\s]+', value)
                    nums = [float(re.sub(r'[^\d.\-]', '', p)) for p in parts if re.search(r'\d', p)]
                    if len(nums) >= 2:
                        result.y_range = (nums[0], nums[1])
                except (ValueError, IndexError):
                    pass

            # Parse CURVES
            elif re.match(r'^CURVES?\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    nums = re.findall(r'\d+', value)
                    if nums:
                        result.curve_count = int(nums[0])
                except (ValueError, IndexError):
                    pass

            # Parse CONFIDENCE
            elif re.match(r'^CONFIDENCE\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    value = re.sub(r'[%/].*', '', value)
                    result.confidence = float(value)
                    if result.confidence > 1:
                        result.confidence /= 100
                except (ValueError, IndexError):
                    result.confidence = 0.5

        return result

    def detect_axes(
        self,
        image: Union[str, Path, np.ndarray],
        quiet: bool = True
    ) -> Optional[AIAxisResult]:
        """Use AI to detect axis labels and ranges.

        Args:
            image: Path to image file or numpy array (BGR format)
            quiet: Suppress progress messages

        Returns:
            AIAxisResult if successful, None if detection failed
        """
        if not self.is_available:
            if not quiet:
                print("AI axis detection not available")
            return None

        if not self._ensure_model(quiet=quiet):
            return None

        try:
            client = ollama.Client(host=self.config.host)

            if not quiet:
                print("  Running AI axis detection...")

            # Prepare image path for Ollama
            if isinstance(image, np.ndarray):
                # Save temp file for Ollama
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    cv2.imwrite(f.name, image)
                    image_path = f.name
            else:
                image_path = str(image)

            # Send to vision model
            response = client.chat(
                model=self.config.model,
                messages=[
                    {
                        'role': 'user',
                        'content': self.AXIS_DETECTION_PROMPT,
                        'images': [image_path]
                    }
                ],
                options={
                    'temperature': 0.1,  # Low temperature for consistent output
                }
            )

            response_text = response['message']['content']
            result = self._parse_response(response_text)

            if not quiet:
                print(f"    AI detected X range: {result.x_range}")
                print(f"    AI detected Y range: {result.y_range}")
                print(f"    Confidence: {result.confidence:.1%}")

            return result

        except Exception as e:
            if not quiet:
                print(f"AI axis detection error: {e}")
            return None

    def validate_axis_detection(
        self,
        ocr_x_range: Optional[Tuple[float, float]],
        ocr_y_range: Optional[Tuple[float, float]],
        ai_result: AIAxisResult,
        tolerance: float = 0.1
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Validate OCR-detected axis ranges against AI detection.

        Args:
            ocr_x_range: X range from OCR (min, max)
            ocr_y_range: Y range from OCR (min, max)
            ai_result: Result from AI axis detection
            tolerance: Relative tolerance for matching (default 10%)

        Returns:
            Tuple of (best_x_range, best_y_range, confidence)
        """
        x_range = ocr_x_range
        y_range = ocr_y_range
        confidence = 0.5

        # If AI result is valid, use it to validate/correct OCR
        if ai_result and ai_result.is_valid:
            # Check X range
            if ai_result.x_range:
                if ocr_x_range is None:
                    # No OCR result - use AI
                    x_range = ai_result.x_range
                    confidence = ai_result.confidence * 0.9
                else:
                    # Compare OCR and AI
                    ocr_span = ocr_x_range[1] - ocr_x_range[0]
                    ai_span = ai_result.x_range[1] - ai_result.x_range[0]

                    if abs(ocr_span - ai_span) / max(ai_span, 0.1) > tolerance:
                        # Significant disagreement - prefer AI if high confidence
                        if ai_result.confidence > 0.7:
                            x_range = ai_result.x_range
                            confidence = ai_result.confidence
                    else:
                        # Agreement - use OCR (usually more precise)
                        confidence = min(0.95, ai_result.confidence + 0.1)

            # Check Y range
            if ai_result.y_range:
                if ocr_y_range is None:
                    # No OCR result - use AI
                    y_range = ai_result.y_range
                    confidence = ai_result.confidence * 0.9
                else:
                    # For Y-axis, check if scale matches (0-1 vs 0-100)
                    ai_is_percent = ai_result.y_range[1] > 1.5
                    ocr_is_percent = ocr_y_range[1] > 1.5

                    if ai_is_percent != ocr_is_percent:
                        # Scale disagreement - prefer AI
                        if ai_result.confidence > 0.6:
                            y_range = ai_result.y_range
                    # Otherwise, agreement is good
                    confidence = min(0.95, confidence + 0.1)

        return x_range, y_range, confidence


def get_ai_axis_detector(config: Optional[AIConfig] = None) -> Optional[AIAxisDetector]:
    """Get an AI axis detector if available.

    Args:
        config: AI configuration. If None, loads from environment.

    Returns:
        AIAxisDetector instance if available, None otherwise
    """
    detector = AIAxisDetector(config)
    if detector.is_available:
        return detector
    return None
