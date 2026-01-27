#!/usr/bin/env python3
"""
AI-Assisted Extraction Documentation with Iterative Refinement.

This script uses AI vision to evaluate extraction results and automatically
adjusts parameters until the objective is achieved:
1. All curves are complete (no gaps)
2. All text/annotations are removed
3. No curve data is accidentally removed

Uses Ollama + llama3.2-vision for assessment.
"""

import cv2
import numpy as np
import json
import shutil
import base64
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import time


@dataclass
class ExtractionParameters:
    """Parameters that can be tuned for curve extraction."""
    # Text box detection
    light_threshold: int = 230          # Gray value to detect white backgrounds (was 215, too sensitive)
    box_min_area: int = 500             # Minimum area for text boxes (was 100, too small)
    box_max_area: int = 25000           # Maximum area for text boxes
    box_aspect_min: float = 1.2         # Minimum aspect ratio for boxes (was 0.8, too permissive)
    box_margin: int = 10                # Margin around detected boxes (was 15, too large)

    # Black curve extraction
    black_threshold: int = 100          # Threshold for black pixel detection
    min_component_area: int = 3         # Minimum area for curve components

    # Edge detection for rectangles
    canny_low: int = 50
    canny_high: int = 150
    rect_min_area: int = 300
    rect_max_area: int = 15000
    rect_aspect_min: float = 1.5
    rect_aspect_max: float = 12.0
    rect_max_height: int = 40

    # Top region (P-value) detection
    top_region_ratio: float = 0.12      # Top 12% of image

    # Right region text detection
    right_region_ratio: float = 0.55    # Right 45% of image


@dataclass
class AIAssessment:
    """Result of AI assessment of an extraction step."""
    is_acceptable: bool
    issues: List[str] = field(default_factory=list)
    suggestions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_response: str = ""


class AIExtractionAssessor:
    """Uses AI vision to assess extraction quality and suggest improvements."""

    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3.2-vision"):
        self.ollama_host = ollama_host
        self.model = model
        self.timeout = 60  # seconds - balanced timeout for vision model

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _encode_image(self, image_path: str, max_size: int = 512) -> str:
        """Encode image to base64, resizing for faster AI processing."""
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            # Resize if larger than max_size while maintaining aspect ratio
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Encode resized image to JPEG for smaller size
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer.tobytes()).decode("utf-8")
        else:
            # Fallback to original file
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    def _call_vision_model(self, prompt: str, image_path: str) -> str:
        """Call the vision model with an image."""
        image_b64 = self._encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent assessments
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    def assess_plot_area(self, image_path: str) -> AIAssessment:
        """Assess the plot area image to check if axis labels are incorrectly included."""

        prompt = """Look at the BOTTOM EDGE: Are there numbers (0, 1, 3, 6, 12, 18, 24)?
Look at the LEFT EDGE: Are there numbers (0.0, 0.2, 0.4, etc)?

Answer:
X_AXIS_LABELS: [YES/NO]
Y_AXIS_LABELS: [YES/NO]
CORRECT: [YES if no axis labels, NO if any labels visible]"""

        response = self._call_vision_model(prompt, image_path)
        return self._parse_plot_area_assessment(response)

    def _parse_plot_area_assessment(self, response: str) -> AIAssessment:
        """Parse AI response for plot area assessment."""
        # Check for errors first
        if response.startswith("Error:"):
            return AIAssessment(
                is_acceptable=False,
                issues=["AI call failed - will retry with adjusted boundaries"],
                confidence=0.0,
                raw_response=response
            )

        assessment = AIAssessment(
            is_acceptable=True,  # Start optimistic
            raw_response=response
        )

        response_upper = response.upper()
        response_lower = response.lower()

        # Parse simplified format
        for line in response.strip().split('\n'):
            line_upper = line.upper().strip()
            if 'X_AXIS' in line_upper or 'X AXIS' in line_upper:
                if 'YES' in line_upper:
                    assessment.is_acceptable = False
                    assessment.issues.append("X-axis labels present")
                    assessment.suggestions["adjust_y_bottom"] = True
            elif 'Y_AXIS' in line_upper or 'Y AXIS' in line_upper:
                if 'YES' in line_upper:
                    assessment.is_acceptable = False
                    assessment.issues.append("Y-axis labels present")
                    assessment.suggestions["adjust_x_left"] = True
            elif 'CORRECT' in line_upper:
                if 'NO' in line_upper:
                    assessment.is_acceptable = False

        # Fallback: look for YES/NO patterns anywhere
        if 'X_AXIS_LABELS: YES' in response_upper or 'X AXIS LABELS: YES' in response_upper:
            assessment.is_acceptable = False
            assessment.suggestions["adjust_y_bottom"] = True
        if 'Y_AXIS_LABELS: YES' in response_upper or 'Y AXIS LABELS: YES' in response_upper:
            assessment.is_acceptable = False
            assessment.suggestions["adjust_x_left"] = True

        # Set confidence based on whether we found clear structured response
        if any(marker in response_upper for marker in ['X_AXIS', 'Y_AXIS', 'CORRECT']):
            assessment.confidence = 0.8
        else:
            assessment.confidence = 0.5
            # Look for mentions of numbers at edges
            if any(phrase in response_lower for phrase in ["numbers at bottom", "numbers on left", "see numbers", "axis label"]):
                assessment.is_acceptable = False
                assessment.suggestions["adjust_boundaries"] = True

        return assessment

    def assess_curves_only(self, image_path: str, original_path: str) -> AIAssessment:
        """Assess the cleaned curves image for completeness and cleanliness."""

        # First, do a quick pixel-based sanity check
        img = cv2.imread(image_path)
        if img is not None:
            # Check if image is mostly white (no curves extracted)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            non_white = np.sum(gray < 250)
            total = gray.size
            non_white_ratio = non_white / total

            if non_white_ratio < 0.01:  # Less than 1% non-white pixels
                return AIAssessment(
                    is_acceptable=False,
                    issues=["No curves detected - image is mostly blank"],
                    suggestions={"add_color_detection": True},
                    confidence=0.0,
                    raw_response="AUTOMATIC REJECTION: Image contains no visible curves"
                )

        prompt = """Analyze this Kaplan-Meier survival curve extraction result.

The image should show ONLY the survival curves (step functions) with censoring marks (+).
There should be NO text, labels, annotation boxes, axes, or grid lines.

Evaluate these criteria:
1. CURVE COMPLETENESS: Are all curves continuous from left to right without gaps?
   Look for: broken lines, missing segments, curves that stop abruptly

2. TEXT REMOVAL: Is ALL text removed?
   Look for: any letters, numbers, words, annotation boxes with labels

3. ARTIFACT REMOVAL: Are non-curve elements removed?
   Look for: rectangular boxes (even empty ones), axis lines, tick marks

Respond in this EXACT format:
CURVES_COMPLETE: [YES/NO]
CURVES_ISSUE: [description if NO, or "none" if YES]
TEXT_REMOVED: [YES/NO]
TEXT_ISSUE: [description if NO, or "none" if YES]
ARTIFACTS_REMOVED: [YES/NO]
ARTIFACTS_ISSUE: [description if NO, or "none" if YES]
OVERALL_ACCEPTABLE: [YES/NO]
CONFIDENCE: [0-100]
SUGGESTIONS: [specific parameter adjustments if needed, or "none"]"""

        response = self._call_vision_model(prompt, image_path)
        return self._parse_assessment(response)

    def assess_with_original(self, cleaned_path: str, original_path: str) -> AIAssessment:
        """Compare cleaned image with original to verify no curve data was lost."""

        # First assess the cleaned image alone
        assessment = self.assess_curves_only(cleaned_path, original_path)

        # If basic assessment passes, do a comparison check
        if assessment.is_acceptable:
            comparison_prompt = """Compare these two images:
1. Original Kaplan-Meier plot (with all labels and annotations)
2. Cleaned version (should have only curves)

Check if ANY curve data was accidentally removed in the cleaning process.
Look for: curves that are shorter in the cleaned version, missing curve segments

Respond with:
DATA_PRESERVED: [YES/NO]
ISSUE: [description if NO, or "none" if YES]"""

            # Note: For a full comparison, we'd need to send both images
            # For now, we rely on the single-image assessment

        return assessment

    def _parse_assessment(self, response: str) -> AIAssessment:
        """Parse AI response into structured assessment."""
        assessment = AIAssessment(
            is_acceptable=False,
            raw_response=response
        )

        response_lower = response.lower()

        # Try to extract structured fields
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue

            parts = line.split(':', 1)
            if len(parts) != 2:
                continue

            key = parts[0].strip().upper().replace('_', ' ').replace('-', ' ')
            value = parts[1].strip()

            # Normalize key variations
            if "CURVES" in key and "COMPLETE" in key:
                if "NO" in value.upper() or "INCOMPLETE" in value.upper():
                    assessment.issues.append("Curves incomplete")
            elif "CURVES" in key and "ISSUE" in key:
                if value.lower() not in ["none", "n/a", "no issues", ""]:
                    assessment.issues.append(f"Curve issue: {value}")
            elif "TEXT" in key and "REMOVED" in key:
                if "NO" in value.upper():
                    assessment.issues.append("Text not fully removed")
                    assessment.suggestions["increase_box_margin"] = True
            elif "TEXT" in key and "ISSUE" in key:
                if value.lower() not in ["none", "n/a", "no issues", ""]:
                    assessment.issues.append(f"Text issue: {value}")
                    assessment.suggestions["increase_box_margin"] = True
            elif "ARTIFACT" in key and "REMOVED" in key:
                if "NO" in value.upper():
                    assessment.issues.append("Artifacts remain")
                    assessment.suggestions["increase_box_detection"] = True
            elif "ARTIFACT" in key and "ISSUE" in key:
                if value.lower() not in ["none", "n/a", "no issues", ""]:
                    assessment.issues.append(f"Artifact issue: {value}")
                    assessment.suggestions["increase_box_detection"] = True
            elif "OVERALL" in key or "ACCEPTABLE" in key:
                assessment.is_acceptable = "YES" in value.upper()
            elif "CONFIDENCE" in key:
                try:
                    # Extract number from value
                    import re
                    nums = re.findall(r'[\d.]+', value)
                    if nums:
                        conf = float(nums[0])
                        if conf > 1:  # Assume percentage
                            conf = conf / 100
                        assessment.confidence = min(conf, 1.0)
                except:
                    pass
            elif "SUGGESTION" in key:
                if value.lower() not in ["none", "n/a", ""]:
                    assessment.suggestions["ai_suggestion"] = value

        # Fallback: analyze response text if no structured format found
        if assessment.confidence == 0 and len(assessment.issues) == 0:
            # Look for positive indicators
            positive_phrases = ["complete", "all curves", "fully visible", "clean",
                              "no text", "removed", "acceptable", "good"]
            negative_phrases = ["gap", "missing", "incomplete", "text visible",
                              "annotation", "box", "artifact", "not removed"]

            positive_count = sum(1 for p in positive_phrases if p in response_lower)
            negative_count = sum(1 for p in negative_phrases if p in response_lower)

            if positive_count > negative_count:
                assessment.is_acceptable = True
                assessment.confidence = 0.6 + (0.1 * positive_count)
            else:
                assessment.is_acceptable = False
                assessment.confidence = 0.3
                if "gap" in response_lower or "missing" in response_lower:
                    assessment.issues.append("Possible curve gaps detected")
                if "text" in response_lower or "annotation" in response_lower or "box" in response_lower:
                    assessment.issues.append("Possible text/annotation remaining")
                    assessment.suggestions["increase_box_margin"] = True

        return assessment

    def count_curves_in_original(self, image_path: str) -> AIAssessment:
        """Count the number of distinct curves in the original image."""

        prompt = """How many step-function curves are in this Kaplan-Meier plot?
Answer format:
CURVES: 2
COLORS: purple, cyan"""

        response = self._call_vision_model(prompt, image_path)
        return self._parse_curve_count(response)

    def _parse_curve_count(self, response: str) -> AIAssessment:
        """Parse curve count response."""
        assessment = AIAssessment(
            is_acceptable=True,
            raw_response=response
        )

        import re

        for line in response.strip().split('\n'):
            line = line.strip()
            if ':' not in line:
                continue

            parts = line.split(':', 1)
            key = parts[0].strip().upper()
            value = parts[1].strip()

            if "CURVES" in key or "CURVE_COUNT" in key or "COUNT" in key:
                nums = re.findall(r'\d+', value)
                if nums:
                    assessment.suggestions["curve_count"] = int(nums[0])
            elif "COLOR" in key:
                assessment.suggestions["curve_colors"] = value.lower()

        # Fallback: look for numbers anywhere in response
        if "curve_count" not in assessment.suggestions:
            # Look for patterns like "2 curves", "two curves", etc.
            nums = re.findall(r'(\d+)\s*(?:curve|line|survival)', response.lower())
            if nums:
                count = int(nums[0])
                if 1 <= count <= 10:  # Reasonable curve count
                    assessment.suggestions["curve_count"] = count
            else:
                # Try to find small numbers (1-10) which are likely curve counts
                all_nums = re.findall(r'\b(\d)\b', response)  # Single digits only
                for num in all_nums:
                    count = int(num)
                    if 1 <= count <= 10:
                        assessment.suggestions["curve_count"] = count
                        break

        assessment.confidence = 0.8 if "curve_count" in assessment.suggestions else 0.3
        return assessment

    def assess_extraction_completeness(
        self,
        original_path: str,
        extracted_path: str,
        expected_curves: int = None
    ) -> AIAssessment:
        """Assess if all curves from original are present in extraction."""

        prompt = f"""Count the distinct colored curves in this image.
Expected: {expected_curves if expected_curves else 2} curves.
If you see fewer curves than expected, identify missing colors.

Reply ONLY:
FOUND: [number of curves visible]
COMPLETE: [YES if found >= expected, NO otherwise]
MISSING: [color of missing curve, or "none"]"""

        response = self._call_vision_model(prompt, extracted_path)
        return self._parse_completeness(response)

    def _parse_completeness(self, response: str) -> AIAssessment:
        """Parse completeness assessment response."""
        import re

        assessment = AIAssessment(
            is_acceptable=True,
            raw_response=response
        )

        for line in response.strip().split('\n'):
            line = line.strip()
            if ':' not in line:
                continue

            parts = line.split(':', 1)
            key = parts[0].strip().upper()
            value = parts[1].strip()

            if "FOUND" in key:
                nums = re.findall(r'\d+', value)
                if nums:
                    assessment.suggestions["found_curves"] = int(nums[0])
            elif "COMPLETE" in key:
                if "NO" in value.upper():
                    assessment.is_acceptable = False
                    assessment.issues.append("Extraction incomplete")
                    assessment.suggestions["needs_enhancement"] = True
            elif "MISSING" in key:
                if value.lower() not in ["none", "n/a", "no", ""]:
                    assessment.is_acceptable = False
                    assessment.issues.append(f"Missing: {value}")
                    assessment.suggestions["missing_curves"] = value
                    assessment.suggestions["needs_enhancement"] = True

        assessment.confidence = 0.8
        return assessment


class AdaptiveExtractionDocumentor:
    """Creates extraction documentation with AI-assisted iterative refinement."""

    def __init__(self, max_iterations: int = 2, use_ai: bool = True, skip_plot_refinement: bool = False):
        self.max_iterations = max_iterations
        self.use_ai = use_ai
        self.skip_plot_refinement = skip_plot_refinement
        self.assessor = AIExtractionAssessor() if use_ai else None
        self.params = ExtractionParameters()

    def _enhance_image_for_extraction(
        self,
        img: np.ndarray,
        enhancement_type: str = "auto"
    ) -> List[Tuple[str, np.ndarray]]:
        """Apply various color enhancements to improve curve extraction.

        Returns a list of (name, enhanced_image) tuples to try.
        """
        enhancements = []

        # 1. Color inversion (good for light curves on white background)
        inverted = cv2.bitwise_not(img)
        enhancements.append(("inverted", inverted))

        # 2. Increase saturation (makes colors more distinct)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int32) * 1.5, 0, 255).astype(np.uint8)
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        enhancements.append(("saturated", saturated))

        # 3. Isolate each color channel
        b, g, r = cv2.split(img)

        # Blue channel (good for cyan/blue curves)
        blue_enhanced = cv2.merge([b, b, b])
        enhancements.append(("blue_channel", blue_enhanced))

        # Red channel (good for red/magenta curves)
        red_enhanced = cv2.merge([r, r, r])
        enhancements.append(("red_channel", red_enhanced))

        # Green channel
        green_enhanced = cv2.merge([g, g, g])
        enhancements.append(("green_channel", green_enhanced))

        # 4. Contrast enhancement using CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhancements.append(("contrast_enhanced", contrast_enhanced))

        # 5. Color space conversion to emphasize differences
        # Convert to LAB and enhance A channel (red-green) and B channel (blue-yellow)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)

        # Enhance A channel (helps distinguish red/magenta from cyan)
        a_enhanced = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        enhancements.append(("lab_a_channel", cv2.merge([a_enhanced, a_enhanced, a_enhanced])))

        # Enhance B channel (helps distinguish blue from yellow)
        b_enhanced = cv2.normalize(b_ch, None, 0, 255, cv2.NORM_MINMAX)
        enhancements.append(("lab_b_channel", cv2.merge([b_enhanced, b_enhanced, b_enhanced])))

        return enhancements

    def _count_curve_colors(self, img: np.ndarray) -> Dict[str, int]:
        """Count pixels of different colors that could be curves."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Only consider saturated, non-white pixels
        mask = (s > 30) & (v > 50) & (v < 250)

        color_counts = {
            'cyan': np.sum(mask & (h >= 80) & (h <= 100)),
            'blue': np.sum(mask & (h >= 100) & (h <= 130)),
            'purple': np.sum(mask & (h >= 130) & (h <= 160)),
            'magenta': np.sum(mask & (h >= 160) & (h <= 175)),
            'red': np.sum(mask & ((h <= 10) | (h >= 175))),
            'orange': np.sum(mask & (h >= 10) & (h <= 25)),
            'green': np.sum(mask & (h >= 35) & (h <= 80)),
        }

        # Also count grayscale/black pixels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        color_counts['black'] = np.sum((gray < 80) & (s < 50))
        color_counts['gray'] = np.sum((gray >= 80) & (gray < 180) & (s < 50))

        return color_counts

    def _count_extracted_curves(self, img: np.ndarray, min_pixels: int = 500) -> int:
        """Count the number of distinct curve colors in an extracted image.

        Uses a higher threshold (500 pixels) to filter out noise and only
        count colors that represent actual curves.
        """
        color_counts = self._count_curve_colors(img)
        # Only count saturated colors (not gray/black which are often noise)
        real_curves = 0
        for color, count in color_counts.items():
            if color not in ['gray', 'black'] and count >= min_pixels:
                real_curves += 1
            elif color in ['gray', 'black'] and count >= min_pixels * 5:
                # Gray/black need much more pixels to be considered real curves
                real_curves += 1
        return real_curves

    def _merge_extractions(
        self,
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> np.ndarray:
        """Merge curves from enhanced extraction with original extraction.

        Takes the union of non-white pixels from both images.
        """
        # Create masks of non-white pixels
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        orig_mask = orig_gray < 250
        enh_mask = enh_gray < 250

        # Start with the original image
        result = original.copy()

        # Add pixels from enhanced that aren't in original
        new_pixels = enh_mask & ~orig_mask

        # For new pixels, we need to assign a color
        # Use the color from the enhanced image
        result[new_pixels] = enhanced[new_pixels]

        return result

    def create_documentation(
        self,
        source_image: str,
        results_dir: str,
        calibration: dict,
        verbose: bool = True
    ) -> dict:
        """Create step-by-step documentation with AI refinement."""

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Load original image
        img = cv2.imread(source_image)
        if img is None:
            raise ValueError(f"Could not load image: {source_image}")

        h, w = img.shape[:2]

        # Extract calibration values
        x_0 = calibration['x_0_pixel']
        x_max = calibration['x_max_pixel']
        y_0 = calibration['y_0_pixel']
        y_100 = calibration['y_100_pixel']

        if verbose:
            print(f"Original image size: {w}x{h}")
            print(f"Calibration: x=[{x_0}, {x_max}], y=[{y_100}, {y_0}]")

        # Step 1: Copy original image
        step1_path = results_path / "step1_original.png"
        shutil.copy(source_image, step1_path)
        if verbose:
            print(f"Step 1: Original image saved")

        # Step 2: Extract rectangle with axes and labels
        # Large margins to include axis labels (Y-axis label on left, X-axis label "Months" below)
        margin_left, margin_bottom, margin_top, margin_right = 130, 180, 70, 40
        x1 = max(0, x_0 - margin_left)
        y1 = max(0, y_100 - margin_top)
        x2 = min(w, x_max + margin_right)
        y2 = min(h, y_0 + margin_bottom)

        step2_img = img[y1:y2, x1:x2].copy()
        step2_path = results_path / "step2_with_axes_labels.png"
        cv2.imwrite(str(step2_path), step2_img)
        if verbose:
            print(f"Step 2: Rectangle with axes/labels saved")

        # Step 3: Plot area only (simple cropping, no AI)
        step3_path = results_path / "step3_plot_area.png"
        top_margin = 15
        left_margin = 5
        bottom_margin = 10
        right_margin = 5
        step3_img = img[max(0, y_100 - top_margin):min(h, y_0 + bottom_margin),
                       max(0, x_0 - left_margin):min(w, x_max + right_margin)].copy()
        cv2.imwrite(str(step3_path), step3_img)

        if verbose:
            print(f"Step 3: Plot area saved")

        # Step 4: Extract curves (no AI, fast extraction)
        step4_path = results_path / "step4_curves_only.png"
        step4_img = self._extract_curves(step3_img, self.params)
        cv2.imwrite(str(step4_path), step4_img)

        if verbose:
            print(f"Step 4: Cleaned curves saved")

        # AI Assessment: Check if all curves were captured
        enhancement_applied = None
        if self.use_ai and self.assessor and self.assessor.is_available():
            if verbose:
                print("  AI: Counting curves in original image...")

            # Count expected curves
            curve_count_assessment = self.assessor.count_curves_in_original(str(step1_path))
            expected_curves = curve_count_assessment.suggestions.get("curve_count", 0)
            expected_colors = curve_count_assessment.suggestions.get("curve_colors", "")

            # Fallback: count colors in original image if AI failed
            if expected_curves == 0:
                original_colors = self._count_curve_colors(step3_img)
                expected_curves = sum(1 for c in original_colors.values() if c > 200)
                if verbose:
                    print(f"  AI timed out, using color detection: {expected_curves} colors found")
            else:
                if verbose:
                    print(f"  AI: Found {expected_curves} curves ({expected_colors})")

            # Count extracted curves and compare with expected
            extracted_curve_count = self._count_extracted_curves(step4_img)
            if verbose:
                print(f"  Extracted {extracted_curve_count} curve colors, expected {expected_curves}")

            # Also try AI assessment (but don't rely on it alone)
            if verbose:
                print("  AI: Assessing extraction completeness...")

            completeness = self.assessor.assess_extraction_completeness(
                str(step1_path), str(step4_path), expected_curves
            )

            # Trigger enhancement if extracted < expected OR AI says incomplete
            needs_enhancement = (extracted_curve_count < expected_curves) or \
                               (not completeness.is_acceptable) or \
                               completeness.suggestions.get("needs_enhancement", False)

            if needs_enhancement:
                if verbose:
                    print(f"  AI: Issues detected: {completeness.issues}")
                    print(f"  AI: Suggested enhancement: {completeness.suggestions.get('enhancement_type', 'auto')}")

                # Try color enhancements
                best_enhancement = None
                best_curve_count = self._count_extracted_curves(step4_img)

                enhancements = self._enhance_image_for_extraction(step3_img)
                for enhance_name, enhanced_img in enhancements:
                    # Extract curves from enhanced image
                    enhanced_curves = self._extract_curves(enhanced_img, self.params)

                    # Count curves in enhanced extraction
                    curve_count = self._count_extracted_curves(enhanced_curves)

                    if verbose:
                        print(f"    Enhancement '{enhance_name}': {curve_count} color regions detected")

                    if curve_count > best_curve_count:
                        best_curve_count = curve_count
                        best_enhancement = (enhance_name, enhanced_curves)

                # If enhancement improved results, use it
                if best_enhancement:
                    enhance_name, enhanced_curves = best_enhancement
                    if verbose:
                        print(f"  AI: Using enhancement '{enhance_name}' (improved from {self._count_extracted_curves(step4_img)} to {best_curve_count} curves)")

                    # Merge enhanced curves with original extraction
                    step4_img = self._merge_extractions(step4_img, enhanced_curves)
                    cv2.imwrite(str(step4_path), step4_img)
                    enhancement_applied = enhance_name

                    # Save the enhanced source for reference
                    cv2.imwrite(str(results_path / f"step3_enhanced_{enhance_name}.png"),
                               dict(enhancements)[enhance_name])
            else:
                if verbose:
                    print("  AI: Extraction looks complete")

        # Step 5: Skeleton of the curves (thinned to 1-pixel width)
        step5_path = results_path / "step5_skeleton.png"
        step5_img = self._create_skeleton(step4_img)
        cv2.imwrite(str(step5_path), step5_img)
        if verbose:
            print(f"Step 5: Curve skeleton saved")

        # Step 6: Original image with extracted curves overlay
        step6_path = results_path / "step6_overlay.png"
        step6_img = self._create_overlay(img, step4_img, x_0, x_max, y_0, y_100)
        cv2.imwrite(str(step6_path), step6_img)
        if verbose:
            print(f"Step 6: Overlay visualization saved")

        # Step 7: Extract curve coordinates to CSV
        csv_paths = self._extract_curves_to_csv(
            step4_img, calibration, results_path, verbose
        )
        if verbose:
            print(f"Step 7: CSV files saved")
            if enhancement_applied:
                print(f"  Note: Enhancement '{enhancement_applied}' was applied to improve extraction")

        # Create summary visualization
        self._create_summary(img, calibration, results_path)

        result = {
            'step1': str(step1_path),
            'step2': str(step2_path),
            'step3': str(step3_path),
            'step4': str(step4_path),
            'step5': str(step5_path),
            'step6': str(step6_path)
        }
        result.update(csv_paths)
        return result

    def _refine_plot_area(
        self,
        img: np.ndarray,
        x_0: int,
        x_max: int,
        y_0: int,
        y_100: int,
        output_path: Path,
        verbose: bool
    ) -> Tuple[np.ndarray, int, int]:
        """Refine plot area boundaries using AI to exclude axis labels."""

        h, w = img.shape[:2]
        current_y_0 = y_0
        current_x_0 = x_0
        prev_y_0 = y_0
        prev_x_0 = x_0

        max_boundary_iterations = 3  # Reduced from 8 for faster processing
        y_adjustment_step = 50  # pixels to move boundary up (increased for faster convergence)
        x_adjustment_step = 40  # pixels to move boundary right

        for iteration in range(max_boundary_iterations):
            if verbose:
                print(f"\n  Plot area refinement - Iteration {iteration + 1}/{max_boundary_iterations}")
                print(f"  Current boundaries: x=[{current_x_0}, {x_max}], y=[{y_100}, {current_y_0}]")

            # Extract with current boundaries
            step3_img = img[y_100:current_y_0, current_x_0:x_max].copy()
            cv2.imwrite(str(output_path), step3_img)

            # Get AI assessment
            if verbose:
                print("  Waiting for AI assessment of plot area...")

            assessment = self.assessor.assess_plot_area(str(output_path))

            if verbose:
                print(f"  AI Assessment: {'PASS' if assessment.is_acceptable else 'FAIL'}")
                print(f"  Confidence: {assessment.confidence:.0%}")
                if assessment.issues:
                    print(f"  Issues: {', '.join(assessment.issues)}")
                # Print raw response for debugging when confidence is low
                if assessment.confidence < 0.7 and assessment.raw_response:
                    print(f"  AI Response (first 300 chars): {assessment.raw_response[:300]}")

            # Check if acceptable - require higher confidence
            if assessment.is_acceptable and assessment.confidence >= 0.75:
                if verbose:
                    print("  Plot area boundaries accepted!")
                return step3_img, current_y_0, current_x_0

            # Track if we made any adjustments this iteration
            made_adjustment = False

            # Adjust boundaries based on assessment
            if assessment.suggestions.get("adjust_y_bottom") or \
               "X-axis" in ' '.join(assessment.issues) or \
               "x-axis" in ' '.join(assessment.issues).lower() or \
               "bottom" in assessment.raw_response.lower():
                # Move the bottom boundary up to exclude X-axis labels
                prev_y_0 = current_y_0
                current_y_0 = max(y_100 + 100, current_y_0 - y_adjustment_step)
                if current_y_0 != prev_y_0:
                    made_adjustment = True
                    if verbose:
                        print(f"  Adjusting: moving bottom boundary up from {prev_y_0} to {current_y_0}")

            if assessment.suggestions.get("adjust_x_left") or \
               "Y-axis" in ' '.join(assessment.issues) or \
               "y-axis" in ' '.join(assessment.issues).lower() or \
               "left" in assessment.raw_response.lower():
                # Move the left boundary right to exclude Y-axis labels
                prev_x_0 = current_x_0
                current_x_0 = min(x_max - 100, current_x_0 + x_adjustment_step)
                if current_x_0 != prev_x_0:
                    made_adjustment = True
                    if verbose:
                        print(f"  Adjusting: moving left boundary right from {prev_x_0} to {current_x_0}")

            # If assessment failed but no specific adjustment suggested, try both
            if not assessment.is_acceptable and not made_adjustment:
                # Try moving bottom boundary up as default action
                prev_y_0 = current_y_0
                current_y_0 = max(y_100 + 100, current_y_0 - y_adjustment_step)
                if current_y_0 != prev_y_0:
                    made_adjustment = True
                    if verbose:
                        print(f"  Auto-adjusting: moving bottom boundary up from {prev_y_0} to {current_y_0}")

            # If we've made no progress, we're stuck
            if not made_adjustment:
                if verbose:
                    print("  No further adjustments possible")
                break

        # After all iterations, use the refined boundaries
        step3_img = img[y_100:current_y_0, current_x_0:x_max].copy()
        cv2.imwrite(str(output_path), step3_img)

        if verbose:
            print(f"\n  Max iterations reached for plot area refinement")
            print(f"  Final boundaries: x=[{current_x_0}, {x_max}], y=[{y_100}, {current_y_0}]")

        return step3_img, current_y_0, current_x_0

    def _extract_with_refinement(
        self,
        plot_img: np.ndarray,
        output_path: Path,
        original_path: str,
        verbose: bool
    ) -> np.ndarray:
        """Extract curves with AI-guided iterative refinement."""

        params = ExtractionParameters()  # Start with defaults
        best_result = None
        best_assessment = None

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n  Iteration {iteration + 1}/{self.max_iterations}")

            # Extract with current parameters
            result = self._extract_curves(plot_img, params)
            cv2.imwrite(str(output_path), result)

            # Get AI assessment
            if verbose:
                print("  Waiting for AI assessment...")

            assessment = self.assessor.assess_curves_only(str(output_path), original_path)

            if verbose:
                print(f"  AI Assessment: {'PASS' if assessment.is_acceptable else 'FAIL'}")
                print(f"  Confidence: {assessment.confidence:.0%}")
                if assessment.issues:
                    print(f"  Issues: {', '.join(assessment.issues)}")
                # Show raw response for debugging
                if assessment.confidence == 0:
                    print(f"  Raw AI response (first 500 chars):")
                    print(f"  {assessment.raw_response[:500]}")

            # Track best result
            if best_assessment is None or assessment.confidence > best_assessment.confidence:
                best_result = result.copy()
                best_assessment = assessment

            # Check if acceptable
            if assessment.is_acceptable and assessment.confidence >= 0.7:
                if verbose:
                    print("  Result accepted!")
                return result

            # Adjust parameters based on assessment
            params = self._adjust_parameters(params, assessment, verbose)

        # Return best result after all iterations
        if verbose:
            print(f"\nMax iterations reached. Using best result (confidence: {best_assessment.confidence:.0%})")

        cv2.imwrite(str(output_path), best_result)
        return best_result

    def _adjust_parameters(
        self,
        params: ExtractionParameters,
        assessment: AIAssessment,
        verbose: bool
    ) -> ExtractionParameters:
        """Adjust extraction parameters based on AI assessment."""

        new_params = ExtractionParameters(
            light_threshold=params.light_threshold,
            box_min_area=params.box_min_area,
            box_max_area=params.box_max_area,
            box_aspect_min=params.box_aspect_min,
            box_margin=params.box_margin,
            black_threshold=params.black_threshold,
            min_component_area=params.min_component_area,
            canny_low=params.canny_low,
            canny_high=params.canny_high,
            rect_min_area=params.rect_min_area,
            rect_max_area=params.rect_max_area,
            rect_aspect_min=params.rect_aspect_min,
            rect_aspect_max=params.rect_aspect_max,
            rect_max_height=params.rect_max_height,
            top_region_ratio=params.top_region_ratio,
            right_region_ratio=params.right_region_ratio,
        )

        adjustments = []

        # Check for text/artifact issues - need more aggressive removal
        if assessment.suggestions.get("increase_box_margin") or \
           "text" in ' '.join(assessment.issues).lower() or \
           "annotation" in ' '.join(assessment.issues).lower() or \
           "box" in ' '.join(assessment.issues).lower():
            new_params.box_margin = min(params.box_margin + 5, 30)
            new_params.box_min_area = max(params.box_min_area - 30, 50)
            new_params.light_threshold = min(params.light_threshold + 10, 240)
            adjustments.append(f"box_margin: {params.box_margin} -> {new_params.box_margin}")

        # Check for incomplete curves - need less aggressive removal
        if "incomplete" in ' '.join(assessment.issues).lower() or \
           "gap" in ' '.join(assessment.issues).lower() or \
           "missing" in ' '.join(assessment.issues).lower():
            new_params.black_threshold = min(params.black_threshold + 20, 150)
            new_params.min_component_area = max(params.min_component_area - 1, 1)
            adjustments.append(f"black_threshold: {params.black_threshold} -> {new_params.black_threshold}")

        # Check for artifact issues
        if assessment.suggestions.get("increase_box_detection") or \
           "artifact" in ' '.join(assessment.issues).lower() or \
           "rectangle" in ' '.join(assessment.issues).lower():
            new_params.rect_min_area = max(params.rect_min_area - 100, 100)
            new_params.rect_max_height = min(params.rect_max_height + 10, 60)
            adjustments.append(f"rect_detection adjusted")

        if verbose and adjustments:
            print(f"  Adjustments: {', '.join(adjustments)}")

        return new_params

    def _extract_curves(self, img: np.ndarray, params: ExtractionParameters) -> np.ndarray:
        """Extract curves with given parameters."""
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        output = np.ones_like(img) * 255

        # === Detect text/annotation regions ===
        text_box_regions = np.zeros((h, w), dtype=np.uint8)
        text_box_list = []

        # Method 1: White background detection
        light_mask = (gray > params.light_threshold).astype(np.uint8) * 255
        contours_light, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_light:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            if params.box_min_area < area < params.box_max_area:
                aspect = cw / max(ch, 1)
                if aspect > params.box_aspect_min:
                    margin = params.box_margin
                    x1, y1 = max(0, x - margin), max(0, y - margin)
                    x2, y2 = min(w, x + cw + margin), min(h, y + ch + margin)
                    cv2.rectangle(text_box_regions, (x1, y1), (x2, y2), 255, -1)
                    text_box_list.append((x1, y1, x2-x1, y2-y1))

        # Method 2: Rectangular contour detection
        edges = cv2.Canny(gray, params.canny_low, params.canny_high)
        contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_edges:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                area = cw * ch
                aspect = cw / max(ch, 1)
                if params.rect_min_area < area < params.rect_max_area and \
                   params.rect_aspect_min < aspect < params.rect_aspect_max and \
                   ch < params.rect_max_height:
                    margin = 5
                    x1, y1 = max(0, x - margin), max(0, y - margin)
                    x2, y2 = min(w, x + cw + margin), min(h, y + ch + margin)
                    cv2.rectangle(text_box_regions, (x1, y1), (x2, y2), 255, -1)
                    text_box_list.append((x1, y1, x2-x1, y2-y1))

        # Method 3: Top region (P-value) - only apply to right half where text typically is
        # Don't mask the left side where curves start at 100%
        top_region_h = int(h * params.top_region_ratio)
        left_curve_margin = int(w * 0.3)  # Protect left 30% where curves start
        top_dark = (gray[:top_region_h, left_curve_margin:] < 200).astype(np.uint8) * 255
        if np.sum(top_dark) > 500:
            cols_with_dark = np.any(top_dark > 0, axis=0)
            if np.any(cols_with_dark):
                left = np.argmax(cols_with_dark) + left_curve_margin
                right = w - np.argmax(cols_with_dark[::-1])
                cv2.rectangle(text_box_regions, (max(left_curve_margin, left-20), 0),
                             (min(w, right+20), top_region_h + 15), 255, -1)

        # Method 4: Right region text
        right_region_x = int(w * params.right_region_ratio)
        right_region = gray[:, right_region_x:]
        right_dark = (right_region < 150).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(right_dark, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            rx = stats[i, cv2.CC_STAT_LEFT] + right_region_x
            ry = stats[i, cv2.CC_STAT_TOP]
            rcw = stats[i, cv2.CC_STAT_WIDTH]
            rch = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = rcw / max(rch, 1)
            if 200 < area < 8000 and 1.5 < aspect < 10 and rch < 35:
                margin = 10
                x1, y1 = max(0, rx - margin), max(0, ry - margin)
                x2, y2 = min(w, rx + rcw + margin), min(h, ry + rch + margin)
                cv2.rectangle(text_box_regions, (x1, y1), (x2, y2), 255, -1)

        # === Extract curves ===

        # Create saturation-protected text mask - don't mask high-saturation pixels (colored curves)
        # Text is typically low saturation (grayscale), so we only apply text masking to low-sat pixels
        saturation = hsv[:, :, 1]
        high_saturation_mask = saturation > 40  # Pixels with saturation > 40 are protected
        text_mask_for_colors = text_box_regions.copy()
        text_mask_for_colors[high_saturation_mask] = 0  # Don't mask colored pixels

        # Green curve (H: 35-85)
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        green_mask[text_mask_for_colors > 0] = 0

        # Cyan/Teal curve (H: 85-105) - common in medical publications
        cyan_mask = cv2.inRange(hsv, np.array([85, 50, 50]), np.array([105, 255, 255]))
        cyan_mask[text_mask_for_colors > 0] = 0

        # Blue curve (H: 105-130)
        blue_mask = cv2.inRange(hsv, np.array([105, 50, 50]), np.array([130, 255, 255]))
        blue_mask[text_mask_for_colors > 0] = 0

        # Purple/Magenta curve (H: 130-165)
        purple_mask = cv2.inRange(hsv, np.array([130, 50, 50]), np.array([165, 255, 255]))
        purple_mask[text_mask_for_colors > 0] = 0

        # Red curve (H: 0-15 or 165-180)
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
        red_mask |= cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
        red_mask[text_mask_for_colors > 0] = 0

        # Orange curve (H: 15-35)
        orange_mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))
        orange_mask[text_mask_for_colors > 0] = 0

        # Gray curve - low saturation, medium value (not white, not black)
        # This catches gray/silver curves that are common in KM plots
        gray_curve_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 200]))
        gray_curve_mask[text_box_regions > 0] = 0
        # Remove background (high value, low saturation areas that are too bright)
        background = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 30, 255]))
        gray_curve_mask[background > 0] = 0

        # Note: Gray antialiased edge detection was removed as it caused overlap issues
        # The main gray detection (S<50, V=80-200) captures the core gray pixels

        # Black curve
        b, g, r = cv2.split(img)
        black_mask = ((b < params.black_threshold) &
                      (g < params.black_threshold) &
                      (r < params.black_threshold)).astype(np.uint8) * 255

        text_box_expanded = cv2.dilate(text_box_regions, np.ones((5, 5), np.uint8), iterations=1)
        black_mask[text_box_expanded > 0] = 0

        # Filter black components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
        curve_black = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            if area < params.min_component_area:
                continue
            if y > h - 10:
                continue

            # Check if in text box
            in_text_box = False
            for tx, ty, tcw, tch in text_box_list:
                if (tx - 5 <= x <= tx + tcw + 5) and (ty - 5 <= y <= ty + tch + 5):
                    in_text_box = True
                    break
            if in_text_box:
                continue

            aspect = cw / max(ch, 1)
            is_h_line = (cw > 10 and ch < 8)
            is_v_line = (ch > 10 and cw < 8)
            is_censor = (area < 60 and 0.3 < aspect < 3 and cw < 18 and ch < 18)
            is_curve_segment = (aspect > 2.5 or aspect < 0.4)
            is_large_segment = (area > 50)

            if is_h_line or is_v_line or is_censor or is_curve_segment or is_large_segment:
                curve_black[labels == i] = 255

        # Apply to output
        kernel = np.ones((2, 2), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        # Clean gray mask - only apply morphological opening to right portion
        # to preserve thin curve segments on the left where curves start
        curve_protection_x = int(w * 0.20)  # Protect left 20% where curves start
        gray_right = gray_curve_mask[:, curve_protection_x:].copy()
        gray_right = cv2.morphologyEx(gray_right, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        gray_curve_mask[:, curve_protection_x:] = gray_right

        # === Remove text-like components from low-saturation masks ===
        # Text characters are typically small, roughly square blobs
        # Include curve_black to catch black text like "83" from "P = 0.0183"
        # Note: Don't filter high-saturation colors (cyan, purple, etc.) - they can't be text
        # Define protected zone for curve starts (top-left corner)
        text_filter_start_zone_x = int(w * 0.15)  # Left 15% of image
        text_filter_start_zone_y = int(h * 0.15)  # Top 15% of image

        for color_mask in [gray_curve_mask, curve_black]:
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                cw = stats[i, cv2.CC_STAT_WIDTH]
                ch = stats[i, cv2.CC_STAT_HEIGHT]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]

                # Protect components in the curve start zone (top-left)
                in_curve_start_zone = (x < text_filter_start_zone_x and y < text_filter_start_zone_y)
                if in_curve_start_zone:
                    continue  # Don't filter components in the curve start zone

                aspect = cw / max(ch, 1)
                fill_ratio = area / max(cw * ch, 1)

                # Text characteristics: small-medium size, roughly square, moderate fill
                is_text_like = (
                    (20 < area < 2000) and
                    (0.3 < aspect < 3) and
                    (cw < 50) and (ch < 50) and
                    (fill_ratio > 0.2)
                )

                # Curve characteristics: elongated (horizontal steps or vertical drops)
                is_curve_like = (
                    (aspect > 4 or aspect < 0.25) or  # Very elongated
                    (cw > 50 or ch > 50) or  # Large
                    (area > 2000)  # Very large
                )

                # Remove if text-like and not curve-like
                if is_text_like and not is_curve_like:
                    color_mask[labels == i] = 0

        # === Remove dashed vertical lines ===
        # Dashed lines are series of short vertical segments aligned vertically
        # Check each column for vertical line patterns
        for x_check in range(0, w):
            col_slice = curve_black[:, max(0, x_check):min(w, x_check+4)]
            if col_slice.shape[1] > 0:
                # Count rows with any black pixels in this column slice
                rows_with_black = np.sum(np.any(col_slice > 0, axis=1))
                # If more than 30% of rows have black pixels, it's likely a vertical line
                if rows_with_black > h * 0.3:
                    curve_black[:, max(0, x_check):min(w, x_check+4)] = 0

        # === Remove edge artifacts ===
        # Remove bottom edge (X-axis tick marks and line)
        bottom_margin = 20
        curve_black[h-bottom_margin:, :] = 0
        gray_curve_mask[h-bottom_margin:, :] = 0
        cyan_mask[h-bottom_margin:, :] = 0

        # Remove left margin to catch Y-axis remnants
        # Use larger margin for black (axis line) and smaller for curves
        left_margin_black = 15  # Y-axis line is thick
        left_margin_curve = 5   # Small margin for curve edges
        curve_black[:, :left_margin_black] = 0
        gray_curve_mask[:, :left_margin_curve] = 0

        # Remove rightmost columns only (potential axis or annotation remnants)
        right_margin = 5
        curve_black[:, w-right_margin:] = 0
        gray_curve_mask[:, w-right_margin:] = 0
        cyan_mask[:, w-right_margin:] = 0

        # === Completely remove colors that are likely just annotation text ===
        # Red and orange are typically used for text annotations, not curves
        # Only keep them if they form large connected regions (actual curves)
        for annotation_mask in [red_mask, orange_mask]:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(annotation_mask, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                cw = stats[i, cv2.CC_STAT_WIDTH]
                ch = stats[i, cv2.CC_STAT_HEIGHT]
                # Only keep if it's large enough to be a curve segment
                if area < 800 or (cw < 80 and ch < 80):
                    annotation_mask[labels == i] = 0

        # === Remove small isolated components (likely text remnants like parentheses) ===
        # Define protected zone for curve starts (top-left corner)
        curve_start_zone_x = int(w * 0.15)  # Left 15% of image
        curve_start_zone_y = int(h * 0.15)  # Top 15% of image

        # Note: Don't filter cyan/blue/purple/green/red - they require high saturation (S>50)
        # so they can't be text. Only filter low-saturation colors (gray, black, orange).
        for color_mask in [gray_curve_mask, curve_black, orange_mask]:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                cw = stats[i, cv2.CC_STAT_WIDTH]
                ch = stats[i, cv2.CC_STAT_HEIGHT]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]

                # Protect components in the curve start zone (top-left)
                # These are likely the beginning of curves at 100% survival
                in_curve_start_zone = (x < curve_start_zone_x and y < curve_start_zone_y)
                if in_curve_start_zone:
                    continue  # Don't filter components in the curve start zone

                aspect = cw / max(ch, 1)
                fill_ratio = area / max(cw * ch, 1)

                # Small isolated components are likely text/punctuation
                is_small_isolated = (area < 300 and cw < 35 and ch < 35)

                # Parentheses are tall and thin with low fill ratio
                is_parenthesis = (
                    (5 < area < 800) and
                    (0.1 < aspect < 1.2) and  # Taller than wide or roughly square
                    (ch > 8) and (cw < 35) and
                    (fill_ratio < 0.65)
                )

                # Dots and small punctuation
                is_punctuation = (area < 50 and cw < 15 and ch < 15)

                # Text-like: not very elongated (curves are elongated)
                is_text_like = (
                    (area < 400) and
                    (0.3 < aspect < 3.0) and  # Not elongated like curve segments
                    (cw < 40) and (ch < 40)
                )

                # Curve segments are typically elongated (horizontal steps or vertical drops)
                is_curve_segment = (
                    (aspect > 5 or aspect < 0.2) or  # Very elongated
                    (cw > 70) or (ch > 70) or  # Large
                    (area > 600)  # Large area
                )

                # Additional check: if component is isolated (not near curve), likely text
                # Check if there are other pixels of the same mask nearby
                margin_check = 15
                x1_check = max(0, x - margin_check)
                x2_check = min(w, x + cw + margin_check)
                y1_check = max(0, y - margin_check)
                y2_check = min(h, y + ch + margin_check)
                nearby_region = color_mask[y1_check:y2_check, x1_check:x2_check].copy()
                nearby_region[y-y1_check:y-y1_check+ch, x-x1_check:x-x1_check+cw] = 0  # Exclude self
                nearby_pixels = np.sum(nearby_region > 0)
                is_isolated = nearby_pixels < 50

                # Remove if it looks like text/punctuation and not like a curve
                # Also remove isolated small components
                if ((is_small_isolated or is_parenthesis or is_punctuation or is_text_like) and not is_curve_segment) or \
                   (is_isolated and area < 400):
                    color_mask[labels == i] = 0

        # Apply colors (BGR format)
        output[green_mask > 0] = [0, 180, 0]        # Green
        output[cyan_mask > 0] = [180, 180, 0]      # Cyan
        output[blue_mask > 0] = [180, 0, 0]        # Blue
        output[purple_mask > 0] = [180, 0, 180]    # Purple
        output[red_mask > 0] = [0, 0, 180]         # Red
        output[orange_mask > 0] = [0, 128, 255]    # Orange
        output[gray_curve_mask > 0] = [128, 128, 128]  # Gray
        output[curve_black > 0] = [0, 0, 0]        # Black

        return output

    def _create_skeleton(self, curves_img: np.ndarray) -> np.ndarray:
        """Create a skeleton (thinned) version of the extracted curves.

        Uses morphological skeletonization to reduce curves to 1-pixel width,
        which is useful for curve tracing and data extraction.

        Also filters out censoring tick marks to show only the main curve line.
        """
        h, w = curves_img.shape[:2]

        # Convert to grayscale for processing
        gray = cv2.cvtColor(curves_img, cv2.COLOR_BGR2GRAY)

        # Create binary mask of all curve pixels (non-white pixels)
        binary = (gray < 250).astype(np.uint8) * 255

        # Step 1: Apply morphological opening to remove small protrusions (tick marks)
        # Use a horizontal kernel to preserve horizontal lines (curve steps)
        # and a vertical kernel to preserve vertical lines (curve drops)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

        # Extract horizontal and vertical line segments
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

        # Combine: keep pixels that are part of horizontal OR vertical segments
        cleaned = cv2.bitwise_or(horizontal, vertical)

        # Step 2: Dilate slightly to reconnect any gaps
        kernel_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)

        # Step 3: Apply skeletonization
        skeleton = np.zeros_like(cleaned)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        temp = cleaned.copy()
        while True:
            eroded = cv2.erode(temp, element)
            dilated = cv2.dilate(eroded, element)
            diff = cv2.subtract(temp, dilated)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()

            if cv2.countNonZero(temp) == 0:
                break

        # Step 4: Remove any remaining small isolated segments (tick mark remnants)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Remove very small isolated segments
            if area < 10:
                skeleton[labels == i] = 0

        # Create output image with skeleton on white background
        output = np.ones_like(curves_img) * 255

        # Color the skeleton based on original curve colors
        skeleton_points = np.where(skeleton > 0)
        for y, x in zip(skeleton_points[0], skeleton_points[1]):
            # Check a small neighborhood in the original curves image for color
            y1, y2 = max(0, y-2), min(curves_img.shape[0], y+3)
            x1, x2 = max(0, x-2), min(curves_img.shape[1], x+3)
            neighborhood = curves_img[y1:y2, x1:x2]

            # Find non-white pixels in neighborhood
            non_white_mask = np.any(neighborhood < 250, axis=2)
            if np.any(non_white_mask):
                non_white_pixels = neighborhood[non_white_mask]
                if len(non_white_pixels) > 0:
                    output[y, x] = non_white_pixels[0]
                else:
                    output[y, x] = [0, 0, 0]
            else:
                output[y, x] = [0, 0, 0]

        return output

    def _create_overlay(
        self,
        original_img: np.ndarray,
        curves_img: np.ndarray,
        x_0: int,
        x_max: int,
        y_0: int,
        y_100: int
    ) -> np.ndarray:
        """Create an overlay of extracted curves on the original image.

        This helps visualize how well the extraction aligns with the original.
        Uses bright contrasting colors (magenta/red) to make the overlay visible.
        """
        output = original_img.copy()
        h, w = original_img.shape[:2]

        # Calculate the region where the curves were extracted from
        # (matching step 3 margins)
        top_margin = 15
        left_margin = 5
        bottom_margin = 10
        right_margin = 5

        region_y1 = max(0, y_100 - top_margin)
        region_y2 = min(h, y_0 + bottom_margin)
        region_x1 = max(0, x_0 - left_margin)
        region_x2 = min(w, x_max + right_margin)

        # Resize curves image to match the region if needed
        region_h = region_y2 - region_y1
        region_w = region_x2 - region_x1
        curves_h, curves_w = curves_img.shape[:2]

        if curves_h != region_h or curves_w != region_w:
            curves_resized = cv2.resize(curves_img, (region_w, region_h), interpolation=cv2.INTER_NEAREST)
        else:
            curves_resized = curves_img

        # Create a mask of curve pixels (non-white pixels in the curves image)
        gray_curves = cv2.cvtColor(curves_resized, cv2.COLOR_BGR2GRAY)
        curve_mask = (gray_curves < 250).astype(np.uint8) * 255

        # Skeletonize to get 1-pixel thin lines for better visibility
        # This allows seeing both the original curves and the extracted overlay
        skeleton = np.zeros_like(curve_mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = curve_mask.copy()
        while True:
            eroded = cv2.erode(temp, element)
            dilated = cv2.dilate(eroded, element)
            diff = cv2.subtract(temp, dilated)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

        # Use the thin skeleton for the overlay
        curve_mask_thin = skeleton

        # Create the overlay with a bright magenta/red color for visibility
        overlay_color = np.array([255, 0, 255], dtype=np.uint8)  # Bright magenta (BGR)

        # Get the region to overlay on
        overlay_region = output[region_y1:region_y2, region_x1:region_x2].copy()

        # Apply the bright overlay color where curves exist
        for c in range(3):
            overlay_region[:, :, c] = np.where(
                curve_mask_thin > 0,
                overlay_color[c],
                overlay_region[:, :, c]
            )

        output[region_y1:region_y2, region_x1:region_x2] = overlay_region

        # Add a border around the plot area to show the extraction region
        cv2.rectangle(output, (region_x1, region_y1), (region_x2, region_y2), (0, 255, 0), 2)

        return output

    def _extract_curves_to_csv(
        self,
        curves_img: np.ndarray,
        calibration: dict,
        results_path: Path,
        verbose: bool
    ) -> dict:
        """Extract curve coordinates from the cleaned curves image and save to CSV.

        Returns:
            Dictionary with paths to generated CSV files
        """
        import pandas as pd

        h, w = curves_img.shape[:2]
        time_max = calibration.get('time_max', 24.0)

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(curves_img, cv2.COLOR_BGR2HSV)

        # Define color ranges for different curves (HSV color space)
        color_ranges = {
            'cyan': {'h_min': 80, 'h_max': 100, 's_min': 40, 'v_min': 50},
            'teal': {'h_min': 100, 'h_max': 110, 's_min': 40, 'v_min': 50},
            'green': {'h_min': 35, 'h_max': 80, 's_min': 50, 'v_min': 50},
            'blue': {'h_min': 110, 'h_max': 130, 's_min': 50, 'v_min': 50},
            'purple': {'h_min': 130, 'h_max': 155, 's_min': 30, 'v_min': 50},
            'magenta': {'h_min': 155, 'h_max': 175, 's_min': 30, 'v_min': 50},
            'red': {'h_min': 0, 'h_max': 10, 's_min': 50, 'v_min': 50},
            'red2': {'h_min': 175, 'h_max': 180, 's_min': 50, 'v_min': 50},  # Red wraps around
            'orange': {'h_min': 10, 'h_max': 25, 's_min': 50, 'v_min': 50},
            'yellow': {'h_min': 25, 'h_max': 35, 's_min': 50, 'v_min': 50},
            'gray': {'h_min': 0, 'h_max': 180, 's_min': 0, 's_max': 50, 'v_min': 80, 'v_max': 180},
            'black': {'h_min': 0, 'h_max': 180, 's_min': 0, 's_max': 50, 'v_min': 0, 'v_max': 80},
        }

        csv_paths = {}
        all_curves_data = []

        for color_name, ranges in color_ranges.items():
            # Create mask for this color
            h_channel, s_channel, v_channel = cv2.split(hsv)

            mask = (h_channel >= ranges['h_min']) & (h_channel <= ranges['h_max'])
            mask &= (s_channel >= ranges.get('s_min', 0))
            mask &= (s_channel <= ranges.get('s_max', 255))
            mask &= (v_channel >= ranges.get('v_min', 0))
            mask &= (v_channel <= ranges.get('v_max', 255))

            # Check if there are enough pixels of this color
            if np.sum(mask) < 100:
                continue

            # Extract curve points
            points = []
            prev_y = None

            for x in range(w):
                col_mask = mask[:, x]
                y_indices = np.where(col_mask)[0]

                if len(y_indices) > 0:
                    # Use minimum y (highest survival) to filter out tick marks
                    # Tick marks extend below the curve line
                    if len(y_indices) <= 3:
                        y = int(np.median(y_indices))
                    else:
                        # For multiple pixels, prefer continuity with previous point
                        y_min = int(np.min(y_indices))
                        if prev_y is not None and abs(y_min - prev_y) > 20:
                            # Large jump - use median instead
                            y = int(np.median(y_indices))
                        else:
                            y = y_min

                    # Convert to time/survival coordinates
                    t = (x / w) * time_max
                    s = 1.0 - (y / h)  # y=0 is top (survival=1), y=h is bottom (survival=0)
                    s = max(0, min(1, s))

                    points.append((t, s))
                    prev_y = y

            if len(points) < 10:
                continue

            # Enforce monotonicity by removing violating points
            monotonic_points = []
            max_s = 1.0
            for t, s in points:
                if s <= max_s + 0.005:  # Small tolerance
                    s = min(s, max_s)
                    monotonic_points.append((t, s))
                    max_s = s

            # Remove consecutive duplicates
            cleaned_points = [monotonic_points[0]] if monotonic_points else []
            for i in range(1, len(monotonic_points)):
                t, s = monotonic_points[i]
                prev_t, prev_s = cleaned_points[-1]
                if t - prev_t > 0.1 or prev_s - s > 0.005:
                    cleaned_points.append((t, s))

            if len(cleaned_points) < 5:
                continue

            # Create DataFrame
            df = pd.DataFrame(cleaned_points, columns=['Time', 'Survival'])

            # Save individual curve CSV
            csv_path = results_path / f"curve_{color_name}.csv"
            df.to_csv(csv_path, index=False)
            csv_paths[f'csv_{color_name}'] = str(csv_path)

            if verbose:
                print(f"  - {color_name}: {len(df)} points, "
                      f"survival {df['Survival'].max():.1%} -> {df['Survival'].min():.1%}")

            # Add to combined data
            for _, row in df.iterrows():
                all_curves_data.append({
                    'Curve': color_name,
                    'Time': row['Time'],
                    'Survival': row['Survival']
                })

        # Save combined CSV
        if all_curves_data:
            combined_df = pd.DataFrame(all_curves_data)
            combined_path = results_path / "all_curves.csv"
            combined_df.to_csv(combined_path, index=False)
            csv_paths['csv_all'] = str(combined_path)

        return csv_paths

    def _create_summary(self, img: np.ndarray, calibration: dict, results_path: Path):
        """Create summary visualization."""
        output = img.copy()
        h, w = img.shape[:2]

        x_0 = calibration['x_0_pixel']
        x_max = calibration['x_max_pixel']
        y_0 = calibration['y_0_pixel']
        y_100 = calibration['y_100_pixel']

        cv2.rectangle(output, (x_0, y_100), (x_max, y_0), (255, 0, 0), 2)
        cv2.line(output, (x_0, y_0), (x_max, y_0), (0, 255, 0), 2)
        cv2.line(output, (x_0, y_100), (x_0, y_0), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, "Plot Area", (x_0 + 10, y_100 + 25), font, 0.6, (255, 0, 0), 2)

        cv2.imwrite(str(results_path / "extraction_regions.png"), output)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AI-assisted extraction documentation')
    parser.add_argument('--source', '-s', required=True, help='Source image path')
    parser.add_argument('--results', '-r', required=True, help='Results directory')
    parser.add_argument('--calibration', '-c', help='Calibration JSON file')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI refinement completely')
    parser.add_argument('--fast', '-f', action='store_true', help='Fast mode: skip AI, use optimized defaults')
    parser.add_argument('--skip-plot-refinement', action='store_true',
                        help='Skip AI plot area refinement (use when calibration is already accurate)')
    parser.add_argument('--max-iterations', '-m', type=int, default=2, help='Max refinement iterations (default: 2)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Fast mode implies no AI
    if args.fast:
        args.no_ai = True
        args.skip_plot_refinement = True

    # Load calibration
    if args.calibration:
        with open(args.calibration, 'r') as f:
            calibration = json.load(f)
    else:
        cal_path = Path(args.results) / 'calibration.json'
        if cal_path.exists():
            with open(cal_path, 'r') as f:
                calibration = json.load(f)
        else:
            raise ValueError("No calibration file found")

    # Create documentor
    documentor = AdaptiveExtractionDocumentor(
        max_iterations=args.max_iterations,
        use_ai=not args.no_ai,
        skip_plot_refinement=getattr(args, 'skip_plot_refinement', False)
    )

    # Check AI availability and mode
    if args.fast:
        print("Fast mode: AI disabled, using optimized defaults")
    elif args.no_ai:
        print("AI disabled, using default parameters")
    else:
        if documentor.assessor and documentor.assessor.is_available():
            print("AI assessment enabled (Ollama available)")
            if getattr(args, 'skip_plot_refinement', False):
                print("  - Plot area refinement: SKIPPED (using calibration as-is)")
        else:
            print("Warning: AI not available, using default parameters")

    # Create documentation
    paths = documentor.create_documentation(
        args.source,
        args.results,
        calibration,
        verbose=args.verbose or True
    )

    print("\nDocumentation images created:")
    for step, path in paths.items():
        print(f"  {step}: {path}")


if __name__ == '__main__':
    main()
