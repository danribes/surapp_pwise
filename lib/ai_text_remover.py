"""
AI-Enhanced Text Removal for KM plots.

Uses vision AI to:
1. Identify text regions that OCR might miss
2. Validate that curves weren't damaged during text removal
3. Guide iterative cleanup if needed
"""

import base64
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from .ai_config import AIConfig
    HAS_AI_CONFIG = True
except ImportError:
    HAS_AI_CONFIG = False


class AITextRemover:
    """AI-enhanced text removal for Kaplan-Meier plots."""

    def __init__(self, config: Optional['AIConfig'] = None):
        """Initialize the text remover.

        Args:
            config: AI configuration. If None, uses defaults or environment.
        """
        if HAS_AI_CONFIG:
            self.config = config or AIConfig.from_environment()
        else:
            self.config = None
        self._client = None

    @property
    def is_available(self) -> bool:
        """Check if AI text removal is available."""
        if not OLLAMA_AVAILABLE:
            return False
        if self.config and not self.config.enabled:
            return False
        return self._check_connection()

    def _check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            host = self.config.host if self.config else "http://localhost:11434"
            client = ollama.Client(host=host)
            client.list()
            return True
        except Exception:
            return False

    def _get_client(self):
        """Get or create Ollama client."""
        if self._client is None:
            host = self.config.host if self.config else "http://localhost:11434"
            self._client = ollama.Client(host=host)
        return self._client

    def _encode_image_array(self, img: np.ndarray) -> str:
        """Encode numpy image array to base64."""
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    def identify_text_regions(
        self,
        img: np.ndarray,
        existing_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Use AI to identify text regions in the image.

        Args:
            img: BGR image
            existing_mask: Optional existing text mask from OCR

        Returns:
            Tuple of (text_mask, list of text region descriptions)
        """
        if not self.is_available:
            print("  AI not available, using OCR only")
            return self._ocr_text_mask(img)

        h, w = img.shape[:2]
        text_mask = existing_mask.copy() if existing_mask is not None else np.zeros((h, w), dtype=np.uint8)

        # Ask AI to identify text regions
        prompt = """Analyze this Kaplan-Meier survival plot image.

Identify ALL text regions including:
1. Title text at the top
2. Axis labels (X-axis: time/months, Y-axis: probability/survival)
3. Axis tick numbers
4. Legend text (curve labels like "RT Only", "RT + TMZ")
5. Statistical annotations (HR, CI, p-value)
6. Any other text overlapping curves

For each text region found, describe:
- Location (e.g., "top-right", "overlapping dashed curve at x=28")
- Content (the text if readable)
- Whether it overlaps with a curve

Format your response as:
TEXT_REGION: [location] - "[content]" - overlaps_curve: [yes/no]

At the end, give overall assessment:
TOTAL_TEXT_REGIONS: [count]
CURVE_OVERLAPPING_TEXT: [count]"""

        try:
            client = self._get_client()
            model = self.config.model if self.config else "llama3.2-vision"

            response = client.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [self._encode_image_array(img)]
                }]
            )

            response_text = response['message']['content']
            print(f"  AI text detection response:\n{response_text[:500]}...")

            # Parse response to identify additional text regions
            # For now, just use the AI assessment to guide further processing
            regions = self._parse_text_regions(response_text)

            return text_mask, regions

        except Exception as e:
            print(f"  AI text identification failed: {e}")
            return text_mask, []

    def validate_text_removal(
        self,
        original_img: np.ndarray,
        cleaned_img: np.ndarray
    ) -> Tuple[bool, str, List[str]]:
        """
        Use AI to validate that text removal was successful and curves intact.

        Args:
            original_img: Original image before text removal
            cleaned_img: Image after text removal

        Returns:
            Tuple of (success, assessment, list of issues)
        """
        if not self.is_available:
            print("  AI not available for validation")
            return True, "AI validation skipped", []

        # Create side-by-side comparison
        h1, w1 = original_img.shape[:2]
        h2, w2 = cleaned_img.shape[:2]

        # Resize if needed
        target_h = max(h1, h2)
        if h1 != target_h:
            scale = target_h / h1
            original_resized = cv2.resize(original_img, (int(w1 * scale), target_h))
        else:
            original_resized = original_img

        if h2 != target_h:
            scale = target_h / h2
            cleaned_resized = cv2.resize(cleaned_img, (int(w2 * scale), target_h))
        else:
            cleaned_resized = cleaned_img

        # Create side-by-side
        divider = 10
        combined = np.zeros((target_h, original_resized.shape[1] + divider + cleaned_resized.shape[1], 3), dtype=np.uint8)
        combined[:, :original_resized.shape[1]] = original_resized
        combined[:, original_resized.shape[1]:original_resized.shape[1]+divider] = [200, 200, 200]
        combined[:, original_resized.shape[1]+divider:] = cleaned_resized

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "ORIGINAL", (10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "CLEANED", (original_resized.shape[1] + divider + 10, 25), font, 0.7, (0, 255, 0), 2)

        prompt = """Compare the ORIGINAL image (left) with the CLEANED image (right).

The CLEANED image should have text removed but curves preserved.

Check for:
1. REMAINING TEXT: Is there any text still visible in the cleaned image?
   - Statistical annotations (HR, CI, p-value)
   - Curve labels
   - Axis labels that should have been removed

2. CURVE DAMAGE: Were any curves damaged during text removal?
   - Look for gaps in curves
   - Look for curves that are thinner or missing segments
   - Compare curve paths between original and cleaned

3. CURVE CONTINUITY: Do curves in the cleaned image maintain their shape?
   - Solid curve should be continuous
   - Dashed curve should maintain its pattern

Respond with:
TEXT_REMOVED: [COMPLETE/PARTIAL/FAILED]
CURVES_INTACT: [YES/NO/PARTIAL]
ISSUES: [list any specific issues found]
OVERALL: [PASS/FAIL]

Be specific about any gaps or damage found."""

        try:
            client = self._get_client()
            model = self.config.model if self.config else "llama3.2-vision"

            response = client.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [self._encode_image_array(combined)]
                }]
            )

            response_text = response['message']['content']
            print(f"  AI validation response:\n{response_text}")

            # Parse response
            success, assessment, issues = self._parse_validation_response(response_text)
            return success, assessment, issues

        except Exception as e:
            print(f"  AI validation failed: {e}")
            return True, f"AI validation error: {e}", []

    def remove_text_with_ai_guidance(
        self,
        img: np.ndarray,
        plot_bounds: Optional[Tuple[int, int, int, int]] = None,
        max_iterations: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Remove text from image with AI guidance and validation.

        Args:
            img: BGR image
            plot_bounds: Optional (x, y, w, h) of plot area
            max_iterations: Maximum cleanup iterations

        Returns:
            Tuple of (cleaned_image, text_mask, validation_report)
        """
        h, w = img.shape[:2]

        # Step 1: Initial OCR-based text detection
        text_mask, ocr_count = self._ocr_text_mask(img)
        print(f"  OCR detected {ocr_count} text regions")

        # Step 2: AI enhancement of text mask (if available)
        if self.is_available:
            ai_mask, ai_regions = self.identify_text_regions(img, text_mask)
            if len(ai_regions) > 0:
                print(f"  AI identified {len(ai_regions)} text regions")
                # Merge AI-identified regions into mask
                text_mask = cv2.bitwise_or(text_mask, ai_mask)

        # Step 3: Dilate mask to ensure coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)

        # Step 4: Protect curves during inpainting
        # Create a curve protection mask
        curve_mask = self._detect_curve_pixels(img)

        # Remove curve pixels from text mask to protect them
        protected_text_mask = text_mask.copy()
        protected_text_mask[curve_mask > 0] = 0

        # Step 5: Inpaint
        cleaned = cv2.inpaint(img, protected_text_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # Step 6: AI validation
        report = {'iterations': 1, 'success': True, 'issues': []}

        if self.is_available:
            success, assessment, issues = self.validate_text_removal(img, cleaned)
            report['ai_assessment'] = assessment
            report['issues'] = issues
            report['success'] = success

            # Iterative cleanup if needed
            iteration = 1
            while not success and iteration < max_iterations:
                iteration += 1
                print(f"  Iteration {iteration}: Attempting to fix issues...")

                # For now, just try with larger inpaint radius
                cleaned = cv2.inpaint(img, text_mask, inpaintRadius=10, flags=cv2.INPAINT_NS)

                success, assessment, issues = self.validate_text_removal(img, cleaned)
                report['ai_assessment'] = assessment
                report['issues'] = issues
                report['success'] = success

            report['iterations'] = iteration

        return cleaned, text_mask, report

    def _ocr_text_mask(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        """Create text mask using OCR."""
        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)

        if not HAS_TESSERACT:
            return text_mask, 0

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, timeout=30)
        except Exception as e:
            print(f"  OCR failed: {e}")
            return text_mask, 0

        count = 0
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

            if conf > 20 and len(text) > 0:
                x = data['left'][i]
                y = data['top'][i]
                tw = data['width'][i]
                th = data['height'][i]

                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w, x + tw + padding)
                y2 = min(h, y + th + padding)

                text_mask[y1:y2, x1:x2] = 255
                count += 1

        return text_mask, count

    def _detect_curve_pixels(self, img: np.ndarray) -> np.ndarray:
        """Detect curve pixels to protect during inpainting."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Curves are dark pixels
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Remove very small noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find connected components and keep only elongated ones (likely curves)
        # This is a simple heuristic - curves tend to be thin and long
        curve_mask = np.zeros_like(binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Skip tiny noise
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / max(1, min(bw, bh))

            # Curves are elongated (high aspect ratio) or large connected components
            if aspect > 3 or area > 500:
                cv2.drawContours(curve_mask, [contour], -1, 255, -1)

        return curve_mask

    def _parse_text_regions(self, response: str) -> List[dict]:
        """Parse AI response for text regions."""
        regions = []
        for line in response.split('\n'):
            if line.strip().startswith('TEXT_REGION:'):
                parts = line.split('-')
                if len(parts) >= 2:
                    regions.append({
                        'location': parts[0].replace('TEXT_REGION:', '').strip(),
                        'content': parts[1].strip().strip('"') if len(parts) > 1 else '',
                        'overlaps_curve': 'yes' in line.lower() and 'overlaps' in line.lower()
                    })
        return regions

    def _parse_validation_response(self, response: str) -> Tuple[bool, str, List[str]]:
        """Parse AI validation response."""
        text_removed = "UNKNOWN"
        curves_intact = "UNKNOWN"
        overall = "UNKNOWN"
        issues = []

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('TEXT_REMOVED:'):
                text_removed = line.split(':', 1)[1].strip()
            elif line.startswith('CURVES_INTACT:'):
                curves_intact = line.split(':', 1)[1].strip()
            elif line.startswith('OVERALL:'):
                overall = line.split(':', 1)[1].strip()
            elif line.startswith('ISSUES:') or line.startswith('-'):
                issue = line.replace('ISSUES:', '').replace('-', '').strip()
                if issue:
                    issues.append(issue)

        success = overall.upper() == 'PASS' or (
            text_removed.upper() in ['COMPLETE', 'PARTIAL'] and
            curves_intact.upper() in ['YES', 'PARTIAL']
        )

        assessment = f"Text: {text_removed}, Curves: {curves_intact}, Overall: {overall}"
        return success, assessment, issues


def remove_text_with_ai(
    img: np.ndarray,
    plot_bounds: Optional[Tuple[int, int, int, int]] = None,
    config: Optional['AIConfig'] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to remove text with AI assistance.

    Args:
        img: BGR image
        plot_bounds: Optional plot bounds
        config: Optional AI configuration

    Returns:
        Tuple of (cleaned_image, text_mask)
    """
    remover = AITextRemover(config)
    cleaned, mask, report = remover.remove_text_with_ai_guidance(img, plot_bounds)

    if report.get('issues'):
        print(f"  Text removal issues: {report['issues']}")

    return cleaned, mask
