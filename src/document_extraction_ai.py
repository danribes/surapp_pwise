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
    light_threshold: int = 215          # Gray value to detect white backgrounds
    box_min_area: int = 100             # Minimum area for text boxes
    box_max_area: int = 25000           # Maximum area for text boxes
    box_aspect_min: float = 0.8         # Minimum aspect ratio for boxes
    box_margin: int = 15                # Margin around detected boxes

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
        self.timeout = 120  # seconds

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
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

    def assess_curves_only(self, image_path: str, original_path: str) -> AIAssessment:
        """Assess the cleaned curves image for completeness and cleanliness."""

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


class AdaptiveExtractionDocumentor:
    """Creates extraction documentation with AI-assisted iterative refinement."""

    def __init__(self, max_iterations: int = 5, use_ai: bool = True):
        self.max_iterations = max_iterations
        self.use_ai = use_ai
        self.assessor = AIExtractionAssessor() if use_ai else None
        self.params = ExtractionParameters()

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
        margin_left, margin_bottom, margin_top, margin_right = 70, 50, 20, 20
        x1 = max(0, x_0 - margin_left)
        y1 = max(0, y_100 - margin_top)
        x2 = min(w, x_max + margin_right)
        y2 = min(h, y_0 + margin_bottom)

        step2_img = img[y1:y2, x1:x2].copy()
        step2_path = results_path / "step2_with_axes_labels.png"
        cv2.imwrite(str(step2_path), step2_img)
        if verbose:
            print(f"Step 2: Rectangle with axes/labels saved")

        # Step 3: Plot area only
        step3_img = img[y_100:y_0, x_0:x_max].copy()
        step3_path = results_path / "step3_plot_area.png"
        cv2.imwrite(str(step3_path), step3_img)
        if verbose:
            print(f"Step 3: Plot area saved")

        # Step 4: Cleaned curves with AI refinement
        step4_path = results_path / "step4_curves_only.png"

        if self.use_ai and self.assessor and self.assessor.is_available():
            step4_img = self._extract_with_refinement(
                step3_img, step4_path, str(step3_path), verbose
            )
        else:
            if verbose:
                print("AI not available, using default parameters")
            step4_img = self._extract_curves(step3_img, self.params)
            cv2.imwrite(str(step4_path), step4_img)

        if verbose:
            print(f"Step 4: Cleaned curves saved")

        # Create summary visualization
        self._create_summary(img, calibration, results_path)

        return {
            'step1': str(step1_path),
            'step2': str(step2_path),
            'step3': str(step3_path),
            'step4': str(step4_path)
        }

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

        # Method 3: Top region (P-value)
        top_region_h = int(h * params.top_region_ratio)
        top_dark = (gray[:top_region_h, :] < 200).astype(np.uint8) * 255
        if np.sum(top_dark) > 500:
            cols_with_dark = np.any(top_dark > 0, axis=0)
            if np.any(cols_with_dark):
                left = np.argmax(cols_with_dark)
                right = w - np.argmax(cols_with_dark[::-1])
                cv2.rectangle(text_box_regions, (max(0, left-20), 0),
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

        # Green curve
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        green_mask[text_box_regions > 0] = 0

        # Red curve
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
        red_mask |= cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
        red_mask[text_box_regions > 0] = 0

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
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        output[green_mask > 0] = [0, 180, 0]
        output[red_mask > 0] = [0, 0, 180]
        output[curve_black > 0] = [0, 0, 0]

        return output

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
    parser.add_argument('--no-ai', action='store_true', help='Disable AI refinement')
    parser.add_argument('--max-iterations', '-m', type=int, default=5, help='Max refinement iterations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

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
        use_ai=not args.no_ai
    )

    # Check AI availability
    if not args.no_ai:
        if documentor.assessor and documentor.assessor.is_available():
            print("AI assessment enabled (Ollama available)")
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
