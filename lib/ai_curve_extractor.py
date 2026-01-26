"""
AI-Assisted Curve Point Extraction for KM Plots.

Uses vision models to directly read survival values from Kaplan-Meier curves
at specified time points.

IMPORTANT LIMITATIONS:
- Current vision models (llama3.2-vision) are NOT reliable for precise numerical
  reading from graphs. They tend to generate plausible-looking values rather than
  accurately reading the actual curve positions.
- AI is useful for CURVE IDENTIFICATION (color, style) but NOT for extracting
  precise survival values.
- For accurate data extraction, use pixel-based detection (extract_km.py).
- AI extraction may be useful for:
  1. Quick rough estimates when precision isn't critical
  2. Validating that curves were correctly identified
  3. Cross-checking axis ranges and labels
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import cv2
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from .ai_config import AIConfig
    HAS_AI_CONFIG = True
except ImportError:
    HAS_AI_CONFIG = False


@dataclass
class ExtractedCurve:
    """Extracted curve data from AI."""
    name: str
    color: str = ""
    points: List[Tuple[float, float]] = field(default_factory=list)  # (time, survival)
    confidence: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if curve has enough points."""
        return len(self.points) >= 3 and self.confidence >= 0.5


@dataclass
class AIExtractionResult:
    """Result from AI curve extraction."""
    curves: List[ExtractedCurve] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    confidence: float = 0.0
    raw_response: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if extraction produced usable data."""
        return (len(self.curves) >= 1
                and all(c.is_valid for c in self.curves)
                and self.confidence >= 0.5)


class AICurveExtractor:
    """Use vision model to directly extract curve points."""

    # Prompt for initial curve identification
    IDENTIFY_CURVES_PROMPT = """Analyze this Kaplan-Meier survival plot image carefully.

Identify:
1. How many distinct survival curves are shown?
2. What color or style is each curve? (e.g., "cyan solid", "gray dashed", "blue", "red")
   - List curves from TOP to BOTTOM (the curve with HIGHER survival values first)
3. What is the X-axis range (time units shown)?
4. What is the Y-axis range? (Usually 0.0 to 1.0 for probability, or 0% to 100%)

IMPORTANT: Look at the Y-axis labels carefully:
- If Y-axis shows 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 -> survival is in decimal form
- If Y-axis shows 0, 20, 40, 60, 80, 100 -> survival is in percentage form

Respond in this exact format:
CURVE_COUNT: <number>
CURVE_1: <color/style description> (top/better survival curve)
CURVE_2: <color/style description> (bottom/worse survival curve)
X_RANGE: <min>, <max>
Y_RANGE: <min>, <max>
Y_SCALE: <decimal or percentage>
CONFIDENCE: <0.0-1.0>
"""

    # Prompt template for reading points
    READ_POINTS_PROMPT_TEMPLATE = """Look at this Kaplan-Meier survival plot image very carefully.

Focus ONLY on the {curve_description} curve.

STEP 1 - CALIBRATE: First, look at the Y-axis scale on the left side:
- Find the gridlines and their labels (0.0, 0.2, 0.4, 0.6, 0.8, 1.0 or similar)
- Note where 0.5 (50%) falls on the axis

STEP 2 - READ: For each time point below, find where the {curve_description} curve crosses that X value, then read the Y value:
{time_points_list}

STEP 3 - VERIFY:
- At time 0, survival should be 1.0 (or close to it)
- The curve can only go DOWN or stay flat over time (never up)
- If the {curve_description} curve drops quickly, later values should be much lower than 1.0
- Many KM curves drop to 0.2-0.5 range by the end

Report values as decimals between 0 and 1.

Respond in this EXACT format (one line per time point):
CURVE: {curve_name}
TIME: 0 | SURVIVAL: <value as decimal 0.0-1.0>
TIME: 3 | SURVIVAL: <value as decimal 0.0-1.0>
TIME: 6 | SURVIVAL: <value as decimal 0.0-1.0>
... (continue for all time points)
CONFIDENCE: <0.0-1.0>
"""

    def __init__(self, config: Optional['AIConfig'] = None):
        """Initialize the extractor.

        Args:
            config: AI configuration. If None, loads from environment.
        """
        if HAS_AI_CONFIG:
            self.config = config or AIConfig.from_environment()
        else:
            self.config = None
        self._model_ready = False

    @property
    def is_available(self) -> bool:
        """Check if AI extraction is available."""
        if not OLLAMA_AVAILABLE:
            return False
        if self.config is None or not self.config.enabled:
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

    def _call_vision_model(self, image_path: str, prompt: str, quiet: bool = True) -> Optional[str]:
        """Send image and prompt to vision model."""
        try:
            client = ollama.Client(host=self.config.host)

            response = client.chat(
                model=self.config.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                ],
                options={
                    'temperature': 0.1,  # Low temperature for consistent output
                }
            )

            return response['message']['content']
        except Exception as e:
            if not quiet:
                print(f"Vision model error: {e}")
            return None

    def _parse_curve_identification(self, response_text: str) -> Tuple[List[str], Tuple[float, float], Tuple[float, float], float]:
        """Parse curve identification response.

        Returns:
            Tuple of (curve_descriptions, x_range, y_range, confidence)
        """
        curves = []
        x_range = (0.0, 24.0)
        y_range = (0.0, 1.0)
        confidence = 0.5
        y_scale = "decimal"

        # Clean markdown
        clean_text = re.sub(r'\*+', '', response_text)
        clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)

        lines = clean_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove bullets
            line = re.sub(r'^[-*•]\s*', '', line)

            # Parse CURVE_N lines
            curve_match = re.match(r'^CURVE[-_]?\d+\s*:\s*(.+)', line, re.IGNORECASE)
            if curve_match:
                # Remove parenthetical comments like "(top/better survival curve)"
                curve_desc = curve_match.group(1).strip()
                curve_desc = re.sub(r'\s*\([^)]*\)\s*$', '', curve_desc)
                curves.append(curve_desc)

            # Parse X_RANGE
            elif re.match(r'^X[-_]?RANGE\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    parts = re.split(r'[,\s]+', value)
                    nums = [float(re.sub(r'[^\d.\-]', '', p)) for p in parts if re.search(r'\d', p)]
                    if len(nums) >= 2:
                        x_range = (nums[0], nums[1])
                except (ValueError, IndexError):
                    pass

            # Parse Y_RANGE
            elif re.match(r'^Y[-_]?RANGE\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    parts = re.split(r'[,\s]+', value)
                    nums = [float(re.sub(r'[^\d.\-]', '', p)) for p in parts if re.search(r'\d', p)]
                    if len(nums) >= 2:
                        y_range = (nums[0], nums[1])
                except (ValueError, IndexError):
                    pass

            # Parse Y_SCALE
            elif re.match(r'^Y[-_]?SCALE\s*:', line, re.IGNORECASE):
                value = re.split(r':\s*', line, 1)[1].strip().lower()
                if 'percent' in value:
                    y_scale = "percentage"

            # Parse CONFIDENCE
            elif re.match(r'^CONFIDENCE\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    value = re.sub(r'[%/].*', '', value)
                    confidence = float(value)
                    if confidence > 1:
                        confidence /= 100
                except (ValueError, IndexError):
                    pass

        return curves, x_range, y_range, confidence

    def _parse_points_response(self, response_text: str) -> Tuple[str, List[Tuple[float, float]], float]:
        """Parse curve points response.

        Returns:
            Tuple of (curve_name, points, confidence)
        """
        curve_name = ""
        points = []
        confidence = 0.5

        # Clean markdown
        clean_text = re.sub(r'\*+', '', response_text)
        clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)

        lines = clean_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove bullets
            line = re.sub(r'^[-*•]\s*', '', line)

            # Parse CURVE name
            if re.match(r'^CURVE\s*:', line, re.IGNORECASE):
                curve_name = re.split(r':\s*', line, 1)[1].strip()

            # Parse TIME | SURVIVAL lines
            elif re.match(r'^TIME\s*:', line, re.IGNORECASE):
                try:
                    # Format: "TIME: 3 | SURVIVAL: 0.85" or "TIME: 3, SURVIVAL: 0.85"
                    parts = re.split(r'\|', line)
                    if len(parts) >= 2:
                        time_part = parts[0]
                        surv_part = parts[1]
                    else:
                        # Try comma separation
                        parts = re.split(r',', line)
                        if len(parts) >= 2:
                            time_part = parts[0]
                            surv_part = parts[1]
                        else:
                            continue

                    # Extract time value
                    time_match = re.search(r'TIME\s*:\s*([\d.]+)', time_part, re.IGNORECASE)
                    surv_match = re.search(r'SURVIVAL\s*:\s*([\d.]+)', surv_part, re.IGNORECASE)

                    if time_match and surv_match:
                        t = float(time_match.group(1))
                        s = float(surv_match.group(1))
                        # Normalize survival to 0-1 if given as percentage
                        if s > 1.5:
                            s /= 100
                        points.append((t, s))
                except (ValueError, IndexError):
                    pass

            # Parse CONFIDENCE
            elif re.match(r'^CONFIDENCE\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    value = re.sub(r'[%/].*', '', value)
                    confidence = float(value)
                    if confidence > 1:
                        confidence /= 100
                except (ValueError, IndexError):
                    pass

        return curve_name, points, confidence

    def extract_curves(
        self,
        image: Union[str, Path, np.ndarray],
        time_max: float = 24.0,
        time_step: float = 1.0,
        quiet: bool = True
    ) -> Optional[AIExtractionResult]:
        """Extract curve points directly using AI vision.

        Args:
            image: Path to image file or numpy array (BGR format)
            time_max: Maximum time value to extract
            time_step: Step size for time points (smaller = more detail)
            quiet: Suppress progress messages

        Returns:
            AIExtractionResult if successful, None if extraction failed
        """
        if not self.is_available:
            if not quiet:
                print("AI curve extraction not available")
            return None

        if not self._ensure_model(quiet=quiet):
            return None

        # Prepare image path
        if isinstance(image, np.ndarray):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                cv2.imwrite(f.name, image)
                image_path = f.name
        else:
            image_path = str(image)

        result = AIExtractionResult(raw_response="")

        try:
            # Step 1: Identify curves in the image
            if not quiet:
                print("  AI: Identifying curves...")

            id_response = self._call_vision_model(image_path, self.IDENTIFY_CURVES_PROMPT, quiet)
            if not id_response:
                return None

            result.raw_response += f"=== IDENTIFICATION ===\n{id_response}\n\n"

            curve_descriptions, x_range, y_range, id_confidence = self._parse_curve_identification(id_response)

            if not curve_descriptions:
                if not quiet:
                    print("  AI: Could not identify curves")
                return None

            if not quiet:
                print(f"  AI: Found {len(curve_descriptions)} curves: {curve_descriptions}")
                print(f"  AI: X range: {x_range}, Y range: {y_range}")

            # Generate time points
            time_points = []
            t = 0.0
            while t <= time_max:
                time_points.append(t)
                t += time_step

            result.time_points = time_points
            time_points_str = ", ".join([str(int(t)) if t == int(t) else str(t) for t in time_points])

            # Step 2: Read points for each curve
            for i, curve_desc in enumerate(curve_descriptions):
                if not quiet:
                    print(f"  AI: Reading points for curve '{curve_desc}'...")

                # Create prompt for this curve
                prompt = self.READ_POINTS_PROMPT_TEMPLATE.format(
                    curve_description=curve_desc,
                    curve_name=curve_desc,
                    time_points_list=time_points_str
                )

                points_response = self._call_vision_model(image_path, prompt, quiet)
                if not points_response:
                    continue

                result.raw_response += f"=== CURVE {i+1}: {curve_desc} ===\n{points_response}\n\n"

                curve_name, points, confidence = self._parse_points_response(points_response)

                if points:
                    # Sort by time and ensure monotonicity
                    points = sorted(points, key=lambda p: p[0])

                    # Ensure starts at (0, 1.0) if not present
                    if not points or points[0][0] > 0.01:
                        points.insert(0, (0.0, 1.0))

                    # Enforce monotonic decrease (KM constraint)
                    cleaned_points = []
                    max_survival = 1.0
                    for t, s in points:
                        s = min(s, max_survival)
                        cleaned_points.append((t, s))
                        max_survival = s

                    curve = ExtractedCurve(
                        name=curve_name or f"curve_{i+1}",
                        color=curve_desc,
                        points=cleaned_points,
                        confidence=confidence
                    )
                    result.curves.append(curve)

                    if not quiet:
                        print(f"    Extracted {len(cleaned_points)} points, confidence: {confidence:.0%}")

            # Calculate overall confidence
            if result.curves:
                result.confidence = sum(c.confidence for c in result.curves) / len(result.curves)

            return result

        except Exception as e:
            if not quiet:
                print(f"AI curve extraction error: {e}")
            return None

    def extract_single_curve(
        self,
        image: Union[str, Path, np.ndarray],
        curve_description: str,
        time_points: List[float],
        quiet: bool = True
    ) -> Optional[ExtractedCurve]:
        """Extract points for a single specified curve.

        Args:
            image: Path to image file or numpy array
            curve_description: Description of the curve (e.g., "cyan", "dashed", "upper curve")
            time_points: List of time values to read
            quiet: Suppress progress messages

        Returns:
            ExtractedCurve if successful, None otherwise
        """
        if not self.is_available:
            return None

        if not self._ensure_model(quiet=quiet):
            return None

        # Prepare image path
        if isinstance(image, np.ndarray):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                cv2.imwrite(f.name, image)
                image_path = f.name
        else:
            image_path = str(image)

        try:
            time_points_str = ", ".join([str(int(t)) if t == int(t) else str(t) for t in time_points])

            prompt = self.READ_POINTS_PROMPT_TEMPLATE.format(
                curve_description=curve_description,
                curve_name=curve_description,
                time_points_list=time_points_str
            )

            response = self._call_vision_model(image_path, prompt, quiet)
            if not response:
                return None

            curve_name, points, confidence = self._parse_points_response(response)

            if not points:
                return None

            # Sort and clean points
            points = sorted(points, key=lambda p: p[0])

            if not points or points[0][0] > 0.01:
                points.insert(0, (0.0, 1.0))

            # Enforce monotonic decrease
            cleaned_points = []
            max_survival = 1.0
            for t, s in points:
                s = min(s, max_survival)
                cleaned_points.append((t, s))
                max_survival = s

            return ExtractedCurve(
                name=curve_name or curve_description,
                color=curve_description,
                points=cleaned_points,
                confidence=confidence
            )

        except Exception as e:
            if not quiet:
                print(f"AI single curve extraction error: {e}")
            return None


def get_ai_curve_extractor(config=None) -> Optional[AICurveExtractor]:
    """Get an AI curve extractor if available.

    Args:
        config: AI configuration. If None, loads from environment.

    Returns:
        AICurveExtractor instance if available, None otherwise
    """
    extractor = AICurveExtractor(config)
    if extractor.is_available:
        return extractor
    return None
