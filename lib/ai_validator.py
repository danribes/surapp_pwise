"""
AI Validator for SURAPP

Uses Ollama with llama3.2-vision to validate curve extraction accuracy
by comparing original images with extracted overlays.
"""

import base64
import re
import time
import tempfile
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .ai_config import AIConfig, ValidationResult, ExtractionParameters


class AIValidator:
    """Validates curve extraction using vision AI."""

    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize the validator.

        Args:
            config: AI configuration. If None, loads from environment.
        """
        self.config = config or AIConfig.from_environment()
        self._client = None
        self._model_ready = False

    @property
    def is_available(self) -> bool:
        """Check if AI validation is available."""
        if not OLLAMA_AVAILABLE:
            return False
        if not self.config.enabled:
            return False
        return self._check_ollama_connection()

    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            if HTTPX_AVAILABLE:
                response = httpx.get(
                    f"{self.config.host}/api/tags",
                    timeout=5.0
                )
                return response.status_code == 200
            else:
                # Fallback: try to list models
                client = ollama.Client(host=self.config.host)
                client.list()
                return True
        except Exception:
            return False

    def ensure_model(self, quiet: bool = False) -> bool:
        """Ensure the vision model is available, pulling if necessary.

        Args:
            quiet: Suppress progress messages.

        Returns:
            True if model is ready, False otherwise.
        """
        if self._model_ready:
            return True

        try:
            client = ollama.Client(host=self.config.host)

            # Check if model exists
            models = client.list()
            model_names = [m.get('name', '').split(':')[0] for m in models.get('models', [])]

            if self.config.model.split(':')[0] not in model_names:
                if not quiet:
                    print(f"Pulling {self.config.model} model (this may take a while)...")

                # Pull the model
                for progress in client.pull(self.config.model, stream=True):
                    if not quiet and 'completed' in progress and 'total' in progress:
                        pct = progress['completed'] / progress['total'] * 100
                        print(f"\r  Progress: {pct:.1f}%", end='', flush=True)

                if not quiet:
                    print("\n  Model ready!")

            self._model_ready = True
            return True

        except Exception as e:
            if not quiet:
                print(f"Error ensuring model: {e}")
            return False

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for API."""
        image_path = Path(image_path)
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _combine_images_side_by_side(
        self,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path]
    ) -> str:
        """
        Combine two images side-by-side into a single image.

        This is needed because llama3.2-vision only supports one image per request.

        Args:
            image1_path: Path to the first image (original)
            image2_path: Path to the second image (overlay)

        Returns:
            Path to the combined temporary image
        """
        img1 = cv2.imread(str(image1_path))
        img2 = cv2.imread(str(image2_path))

        if img1 is None or img2 is None:
            raise ValueError("Failed to load one or both images")

        # Resize images to same height if needed
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        target_h = max(h1, h2)

        if h1 != target_h:
            scale = target_h / h1
            img1 = cv2.resize(img1, (int(w1 * scale), target_h))
            w1 = img1.shape[1]

        if h2 != target_h:
            scale = target_h / h2
            img2 = cv2.resize(img2, (int(w2 * scale), target_h))
            w2 = img2.shape[1]

        # Create side-by-side image with divider
        divider_width = 10
        combined = np.zeros((target_h, w1 + divider_width + w2, 3), dtype=np.uint8)
        combined[:, :w1] = img1
        combined[:, w1:w1+divider_width] = [255, 255, 255]  # White divider
        combined[:, w1+divider_width:] = img2

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "ORIGINAL", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(combined, "EXTRACTED", (w1 + divider_width + 10, 30), font, 1, (0, 255, 0), 2)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(temp_file.name, combined)

        return temp_file.name

    def _parse_response(self, response_text: str) -> ValidationResult:
        """Parse the AI response into a structured result."""
        # Default values
        match = "PARTIAL"
        confidence = 0.5
        issues = []
        suggestions = []

        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Parse MATCH line
            if line.upper().startswith('MATCH:'):
                value = line.split(':', 1)[1].strip().upper()
                if 'YES' in value:
                    match = "YES"
                elif 'NO' in value:
                    match = "NO"
                else:
                    match = "PARTIAL"

            # Parse CONFIDENCE line
            elif line.upper().startswith('CONFIDENCE:'):
                try:
                    value = line.split(':', 1)[1].strip()
                    # Handle formats like "0.8", "80%", "0.8/1.0"
                    value = re.sub(r'[%/].*', '', value)
                    confidence = float(value)
                    if confidence > 1:
                        confidence /= 100
                except (ValueError, IndexError):
                    confidence = 0.5

            # Parse ISSUES line
            elif line.upper().startswith('ISSUES:'):
                value = line.split(':', 1)[1].strip()
                if value.lower() != 'none':
                    issues = [i.strip() for i in value.split(',') if i.strip()]

            # Parse SUGGESTIONS line
            elif line.upper().startswith('SUGGESTIONS:'):
                value = line.split(':', 1)[1].strip()
                if value.lower() != 'none':
                    suggestions = [s.strip() for s in value.split(',') if s.strip()]

        return ValidationResult(
            match=match,
            confidence=confidence,
            issues=issues if issues else ["None"],
            suggestions=suggestions if suggestions else ["None"],
            raw_response=response_text
        )

    def validate(
        self,
        original_image: Union[str, Path],
        overlay_image: Union[str, Path],
        quiet: bool = False
    ) -> Optional[ValidationResult]:
        """Validate extraction by comparing original and overlay images.

        Args:
            original_image: Path to the original KM plot image.
            overlay_image: Path to the extraction overlay image.
            quiet: Suppress progress messages.

        Returns:
            ValidationResult or None if validation failed.
        """
        if not self.is_available:
            if not quiet:
                print("AI validation not available (Ollama not running or not installed)")
            return None

        if not self.ensure_model(quiet=quiet):
            return None

        combined_path = None
        try:
            client = ollama.Client(host=self.config.host)

            if not quiet:
                print("Validating extraction with AI...")

            # Combine images side-by-side (llama3.2-vision only supports one image)
            combined_path = self._combine_images_side_by_side(original_image, overlay_image)

            # Updated prompt for side-by-side comparison
            combined_prompt = """Analyze this side-by-side comparison of a Kaplan-Meier survival curve plot.

LEFT SIDE (labeled "ORIGINAL"): The original plot image
RIGHT SIDE (labeled "EXTRACTED"): The extracted curves overlaid on the original

Your task is to validate if the extraction is accurate. Check:
1. Do the extracted colored lines on the RIGHT follow the original curves precisely?
2. Are there any sections where the extraction deviates from the original?
3. Are all curves from the original captured in the extraction?
4. Do the curves start at the correct position (usually survival=1.0 at time=0)?
5. Are the curve endpoints correctly captured?

Respond in this exact format:
MATCH: [YES/NO/PARTIAL]
CONFIDENCE: [0.0-1.0]
ISSUES: [List any specific issues found, or "None" if perfect match]
SUGGESTIONS: [Parameter adjustments if needed, or "None"]
"""

            # Send combined image to the vision model
            response = client.chat(
                model=self.config.model,
                messages=[
                    {
                        'role': 'user',
                        'content': combined_prompt,
                        'images': [combined_path]
                    }
                ],
                options={
                    'temperature': 0.1,  # Low temperature for consistent analysis
                }
            )

            response_text = response['message']['content']
            result = self._parse_response(response_text)

            if not quiet:
                print(result)

            return result

        except Exception as e:
            if not quiet:
                print(f"Validation error: {e}")
            return None

        finally:
            # Clean up temp file
            if combined_path:
                try:
                    import os
                    os.unlink(combined_path)
                except Exception:
                    pass

    def validate_with_retry(
        self,
        original_image: Union[str, Path],
        extraction_func: callable,
        output_dir: Union[str, Path],
        initial_params: Optional[ExtractionParameters] = None,
        quiet: bool = False
    ) -> tuple[bool, ExtractionParameters, Optional[ValidationResult]]:
        """Run extraction with AI validation and automatic retry.

        Args:
            original_image: Path to the original KM plot image.
            extraction_func: Function that performs extraction, takes (image, params, output_dir).
            output_dir: Directory for output files.
            initial_params: Starting extraction parameters.
            quiet: Suppress progress messages.

        Returns:
            Tuple of (success, final_params, final_result).
        """
        params = initial_params or ExtractionParameters()
        output_dir = Path(output_dir)

        for attempt in range(self.config.max_retries):
            if not quiet:
                print(f"\n{'='*50}")
                print(f"Extraction attempt {attempt + 1}/{self.config.max_retries}")
                print(f"Parameters: {params.to_dict()}")
                print('='*50)

            # Run extraction
            overlay_image = extraction_func(original_image, params, output_dir)

            if overlay_image is None:
                if not quiet:
                    print("Extraction failed to produce overlay image")
                continue

            # Validate result
            result = self.validate(original_image, overlay_image, quiet=quiet)

            if result is None:
                if not quiet:
                    print("Validation unavailable, accepting extraction")
                return True, params, None

            if result.is_valid:
                if not quiet:
                    print(f"\nExtraction validated successfully!")
                return True, params, result

            if not result.needs_retry or attempt == self.config.max_retries - 1:
                if not quiet:
                    print(f"\nExtraction completed with issues (no more retries)")
                return False, params, result

            # Adjust parameters based on suggestions
            if not quiet:
                print(f"\nAdjusting parameters based on AI feedback...")
            params = params.adjust_from_suggestions(result.suggestions)

            # Small delay before retry
            time.sleep(1)

        return False, params, result


def check_ollama_status(host: str = "http://localhost:11434") -> dict:
    """Check Ollama server status and available models.

    Args:
        host: Ollama server URL.

    Returns:
        Status dictionary with server info.
    """
    status = {
        "available": False,
        "host": host,
        "models": [],
        "error": None
    }

    try:
        if not OLLAMA_AVAILABLE:
            status["error"] = "ollama package not installed"
            return status

        client = ollama.Client(host=host)
        models = client.list()

        status["available"] = True
        status["models"] = [m.get('name', 'unknown') for m in models.get('models', [])]

    except Exception as e:
        status["error"] = str(e)

    return status
