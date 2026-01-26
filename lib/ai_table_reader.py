"""
AI-Assisted At-Risk Table Reader for KM Curve Extraction.

Uses vision models (Ollama + llama3.2-vision) to read at-risk tables
when traditional OCR has low confidence or misreads values.
"""

import base64
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
class AITableResult:
    """Result from AI table reading."""
    time_points: List[float] = field(default_factory=list)
    groups: Dict[str, Dict[float, int]] = field(default_factory=dict)  # group_name -> {time: count}
    confidence: float = 0.0
    raw_response: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if result contains usable table data."""
        return (len(self.time_points) >= 2
                and len(self.groups) >= 1
                and self.confidence >= 0.5)


class AITableReader:
    """Use vision model to read at-risk tables."""

    # Prompt template for table reading
    TABLE_READING_PROMPT = """Analyze this "Number at Risk" table from a Kaplan-Meier survival plot.

Read and extract:
1. The time points from the header or bottom row (e.g., 0, 3, 6, 9, 12, ...)
2. The group/treatment names in the leftmost column
3. The at-risk counts for each group at each time point

Important:
- Each row corresponds to a treatment group
- Each column corresponds to a time point
- Numbers are patient counts (integers)

Respond in this exact format:

TIME_POINTS: <comma-separated list>
GROUP: <name> | <counts at each time point, comma-separated>
GROUP: <name> | <counts at each time point, comma-separated>
CONFIDENCE: <0.0-1.0>

Example:
TIME_POINTS: 0, 3, 6, 9, 12, 15, 18
GROUP: Pre-ICI | 40, 35, 28, 20, 15, 10, 5
GROUP: Post-ICI | 131, 120, 100, 85, 70, 55, 40
CONFIDENCE: 0.85
"""

    def __init__(self, config: Optional['AIConfig'] = None):
        """Initialize the reader.

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
        """Check if AI table reading is available."""
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

    def _parse_response(self, response_text: str) -> AITableResult:
        """Parse the AI response into structured result.

        Handles both plain text and markdown formatted responses.
        """
        result = AITableResult(raw_response=response_text)

        # Remove markdown formatting (**, *, #, etc.)
        clean_text = re.sub(r'\*+', '', response_text)
        clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)

        lines = clean_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Normalize line - remove leading bullets/dashes
            line = re.sub(r'^[-*•]\s*', '', line)

            # Parse TIME_POINTS (handles various formats)
            if re.match(r'^TIME[-_\s]*POINTS?\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    # Parse comma-separated numbers
                    parts = re.split(r'[,\s]+', value)
                    result.time_points = [float(p) for p in parts if re.match(r'^[\d.]+$', p.strip())]
                except (ValueError, IndexError):
                    pass

            # Parse GROUP lines
            elif re.match(r'^GROUP\s*:', line, re.IGNORECASE):
                try:
                    value = re.split(r':\s*', line, 1)[1].strip()
                    # Format: "group_name | count1, count2, count3, ..."
                    if '|' in value:
                        name_part, counts_part = value.split('|', 1)
                        group_name = name_part.strip()
                        count_str = counts_part.strip()
                    else:
                        # Try to find where name ends and numbers begin
                        parts = value.split()
                        name_parts = []
                        count_parts = []
                        found_number = False

                        for part in parts:
                            part = part.strip().rstrip(',')
                            if re.match(r'^[\d.]+$', part):
                                found_number = True
                                count_parts.append(part)
                            elif not found_number:
                                name_parts.append(part)

                        group_name = ' '.join(name_parts)
                        count_str = ', '.join(count_parts)

                    # Parse counts
                    count_parts = re.split(r'[,\s]+', count_str)
                    counts = []
                    for p in count_parts:
                        p = p.strip().rstrip(',')
                        if re.match(r'^[\d.]+$', p):
                            counts.append(int(float(p)))

                    if group_name and counts:
                        result.groups[group_name] = {}
                        for i, count in enumerate(counts):
                            if i < len(result.time_points):
                                result.groups[group_name][result.time_points[i]] = count

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

        # If we got groups but no time points, generate default
        if result.groups and not result.time_points:
            max_counts = max(len(v) for v in result.groups.values())
            result.time_points = [float(i * 3) for i in range(max_counts)]

            # Re-map groups to use generated time points
            for group_name in result.groups:
                old_data = result.groups[group_name]
                new_data = {}
                for i, (_, count) in enumerate(sorted(old_data.items())):
                    if i < len(result.time_points):
                        new_data[result.time_points[i]] = count
                result.groups[group_name] = new_data

        return result

    def read_table(
        self,
        table_image: Union[str, Path, np.ndarray],
        quiet: bool = True
    ) -> Optional[AITableResult]:
        """Use AI to read at-risk table.

        Args:
            table_image: Path to image file or numpy array (BGR format) of table region
            quiet: Suppress progress messages

        Returns:
            AITableResult if successful, None if reading failed
        """
        if not self.is_available:
            if not quiet:
                print("AI table reading not available")
            return None

        if not self._ensure_model(quiet=quiet):
            return None

        try:
            client = ollama.Client(host=self.config.host)

            if not quiet:
                print("  Running AI table reading...")

            # Prepare image path for Ollama
            if isinstance(table_image, np.ndarray):
                # Save temp file for Ollama
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    cv2.imwrite(f.name, table_image)
                    image_path = f.name
            else:
                image_path = str(table_image)

            # Send to vision model
            response = client.chat(
                model=self.config.model,
                messages=[
                    {
                        'role': 'user',
                        'content': self.TABLE_READING_PROMPT,
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
                print(f"    AI detected {len(result.groups)} groups")
                print(f"    Time points: {result.time_points}")
                print(f"    Confidence: {result.confidence:.1%}")

            return result

        except Exception as e:
            if not quiet:
                print(f"AI table reading error: {e}")
            return None

    def validate_ocr_result(
        self,
        ocr_time_points: List[float],
        ocr_groups: Dict[str, Dict[float, int]],
        ai_result: AITableResult
    ) -> Tuple[List[float], Dict[str, Dict[float, int]], float]:
        """Validate OCR-extracted table against AI reading.

        Args:
            ocr_time_points: Time points from OCR
            ocr_groups: Groups data from OCR
            ai_result: Result from AI table reading

        Returns:
            Tuple of (best_time_points, best_groups, confidence)
        """
        time_points = ocr_time_points
        groups = ocr_groups
        confidence = 0.5

        if not ai_result or not ai_result.is_valid:
            return time_points, groups, confidence

        # Compare time points
        if ai_result.time_points:
            if not time_points:
                # No OCR result - use AI
                time_points = ai_result.time_points
                confidence = ai_result.confidence
            else:
                # Compare structures
                ocr_len = len(time_points)
                ai_len = len(ai_result.time_points)

                # If AI detected more time points, it might be more accurate
                if ai_len > ocr_len and ai_result.confidence > 0.7:
                    time_points = ai_result.time_points
                    confidence = ai_result.confidence

        # Compare groups
        if ai_result.groups:
            if not groups:
                # No OCR groups - use AI
                groups = ai_result.groups
                confidence = ai_result.confidence
            else:
                # Validate group counts
                for group_name, ai_counts in ai_result.groups.items():
                    # Find matching OCR group
                    ocr_match = None
                    for ocr_name in groups:
                        if ocr_name.lower() in group_name.lower() or group_name.lower() in ocr_name.lower():
                            ocr_match = ocr_name
                            break

                    if ocr_match:
                        ocr_counts = groups[ocr_match]

                        # Check for significant disagreements
                        for t, ai_count in ai_counts.items():
                            ocr_count = ocr_counts.get(t)
                            if ocr_count is not None:
                                # Significant disagreement (>20% difference)
                                if abs(ai_count - ocr_count) / max(ai_count, 1) > 0.2:
                                    # Prefer AI if high confidence
                                    if ai_result.confidence > 0.8:
                                        ocr_counts[t] = ai_count

                # Add any groups AI found that OCR missed
                for ai_name, ai_counts in ai_result.groups.items():
                    found = any(
                        ai_name.lower() in g.lower() or g.lower() in ai_name.lower()
                        for g in groups
                    )
                    if not found and ai_result.confidence > 0.7:
                        groups[ai_name] = ai_counts

                confidence = max(confidence, ai_result.confidence * 0.9)

        return time_points, groups, confidence


def get_ai_table_reader(config=None) -> Optional[AITableReader]:
    """Get an AI table reader if available.

    Args:
        config: AI configuration. If None, loads from environment.

    Returns:
        AITableReader instance if available, None otherwise
    """
    reader = AITableReader(config)
    if reader.is_available:
        return reader
    return None
