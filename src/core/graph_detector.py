"""Graph detection module for identifying Kaplan-Meier curves."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..utils.config import config
from ..utils.image_utils import pil_to_cv2, preprocess_for_curve_detection


@dataclass
class DetectedGraph:
    """Information about a detected graph region."""
    bbox: tuple  # (x, y, width, height)
    confidence: float
    graph_type: str  # 'km_curve', 'generic', 'unknown'
    image: Optional[np.ndarray] = None

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]


class GraphDetector:
    """Detects graphs and potential Kaplan-Meier curves in images."""

    def __init__(self, image: Image.Image | np.ndarray):
        """Initialize graph detector.

        Args:
            image: Input image (PIL Image or OpenCV array)
        """
        if isinstance(image, Image.Image):
            self.image = pil_to_cv2(image)
        else:
            self.image = image.copy()

        self.height, self.width = self.image.shape[:2]
        self._yolo_model = None

    def detect_graphs_heuristic(self) -> list[DetectedGraph]:
        """Detect graphs using heuristic methods (no ML).

        Uses axis detection and step pattern matching.

        Returns:
            List of DetectedGraph objects
        """
        detected = []

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Find rectangular regions with lines (potential plot areas)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and score contours
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size (graph should be reasonable size)
            if w < 100 or h < 100:
                continue

            if w > self.width * 0.95 or h > self.height * 0.95:
                continue

            # Check aspect ratio (typical graphs are wider than tall or square)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 3:
                continue

            # Extract region
            roi = self.image[y:y+h, x:x+w]

            # Score as potential KM curve
            confidence, graph_type = self._score_km_likelihood(roi)

            if confidence > 0.3:
                detected.append(DetectedGraph(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    graph_type=graph_type,
                    image=roi
                ))

        # Sort by confidence
        detected.sort(key=lambda d: d.confidence, reverse=True)

        # Non-maximum suppression to remove overlapping detections
        detected = self._non_max_suppression(detected)

        return detected

    def detect_graphs_template(self, min_confidence: float = 0.5) -> list[DetectedGraph]:
        """Detect graphs using template matching for step patterns.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of DetectedGraph objects
        """
        detected = []

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Create step function templates of various sizes
        templates = self._create_step_templates()

        for template in templates:
            # Match template
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

            # Find locations above threshold
            locations = np.where(result >= min_confidence)

            for pt in zip(*locations[::-1]):  # x, y
                th, tw = template.shape[:2]

                # Expand bounding box to likely graph area
                margin = 50
                x = max(0, pt[0] - margin)
                y = max(0, pt[1] - margin)
                w = min(self.width - x, tw + 2 * margin)
                h = min(self.height - y, th + 2 * margin)

                detected.append(DetectedGraph(
                    bbox=(x, y, w, h),
                    confidence=float(result[pt[1], pt[0]]),
                    graph_type='km_curve',
                    image=self.image[y:y+h, x:x+w]
                ))

        # Non-maximum suppression
        detected = self._non_max_suppression(detected)

        return detected

    def detect_graphs_yolo(self) -> list[DetectedGraph]:
        """Detect graphs using YOLO model (if available).

        Returns:
            List of DetectedGraph objects
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return []

        # Try to load a document/chart detection model
        # For now, return empty - would need a trained model
        return []

    def detect_all(self) -> list[DetectedGraph]:
        """Run all detection methods and combine results.

        Returns:
            Combined list of DetectedGraph objects
        """
        all_detected = []

        # Run heuristic detection
        heuristic_results = self.detect_graphs_heuristic()
        all_detected.extend(heuristic_results)

        # Run template detection
        template_results = self.detect_graphs_template()

        # Add template results that don't overlap significantly
        for detection in template_results:
            if not self._overlaps_existing(detection, all_detected):
                all_detected.append(detection)

        # Sort by confidence
        all_detected.sort(key=lambda d: d.confidence, reverse=True)

        return all_detected

    def _score_km_likelihood(self, roi: np.ndarray) -> tuple:
        """Score likelihood that region contains a KM curve.

        Args:
            roi: Region of interest image

        Returns:
            (confidence, graph_type) tuple
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Feature 1: Check for L-shaped axes
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)

        if lines is None:
            return (0.0, 'unknown')

        # Count horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 15:
                h_lines.append(line[0])
            elif angle > 75:
                v_lines.append(line[0])

        # Score based on axis presence
        axis_score = 0.0
        if h_lines and v_lines:
            axis_score = 0.3

            # Check for L-shape (vertical line near left, horizontal near bottom)
            left_v = any(min(l[0], l[2]) < w * 0.3 for l in v_lines)
            bottom_h = any(max(l[1], l[3]) > h * 0.7 for l in h_lines)

            if left_v and bottom_h:
                axis_score = 0.5

        # Feature 2: Check for step pattern
        step_score = self._detect_step_pattern(gray)

        # Feature 3: Check for monotonic decrease pattern
        monotonic_score = self._check_monotonic_pattern(gray)

        # Combine scores
        total_score = axis_score * 0.3 + step_score * 0.4 + monotonic_score * 0.3

        # Determine type
        if total_score > 0.5 and step_score > 0.3:
            graph_type = 'km_curve'
        elif axis_score > 0.3:
            graph_type = 'generic'
        else:
            graph_type = 'unknown'

        return (total_score, graph_type)

    def _detect_step_pattern(self, gray: np.ndarray) -> float:
        """Detect step function pattern in grayscale image.

        Args:
            gray: Grayscale image

        Returns:
            Score from 0 to 1
        """
        h, w = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=10, maxLineGap=5)

        if lines is None:
            return 0.0

        # Count alternating horizontal-vertical patterns
        h_segments = []
        v_segments = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if length < 5:
                continue

            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            if angle < 10:
                h_segments.append((min(x1, x2), y1, max(x1, x2), y2))
            elif angle > 80:
                v_segments.append((x1, min(y1, y2), x2, max(y1, y2)))

        if len(h_segments) < 3 or len(v_segments) < 2:
            return 0.1

        # Check if segments connect (step pattern)
        connections = 0
        tolerance = 15

        for h_seg in h_segments:
            h_end_x = h_seg[2]  # Right end of horizontal
            h_y = h_seg[1]

            for v_seg in v_segments:
                v_x = v_seg[0]
                v_top = v_seg[1]
                v_bottom = v_seg[3]

                # Check if horizontal end connects to vertical
                if abs(h_end_x - v_x) < tolerance and v_top <= h_y <= v_bottom:
                    connections += 1
                    break

        # Score based on connection ratio
        max_possible = min(len(h_segments), len(v_segments))
        if max_possible > 0:
            return min(1.0, connections / max_possible)

        return 0.0

    def _check_monotonic_pattern(self, gray: np.ndarray) -> float:
        """Check for monotonically decreasing pattern.

        Args:
            gray: Grayscale image

        Returns:
            Score from 0 to 1
        """
        h, w = gray.shape

        # Sample columns and find the "curve" position (darkest pixels)
        samples = []
        for x in range(w // 10, w, w // 10):
            col = gray[:, x]

            # Find dark pixels (potential curve)
            dark_pixels = np.where(col < 128)[0]

            if len(dark_pixels) > 0:
                # Use the topmost dark pixel in the lower portion
                # (KM curves start high and go down)
                lower_dark = dark_pixels[dark_pixels > h // 4]
                if len(lower_dark) > 0:
                    samples.append(lower_dark[0])

        if len(samples) < 3:
            return 0.0

        # Check if generally increasing Y (decreasing survival in image coords)
        increases = sum(1 for i in range(1, len(samples)) if samples[i] >= samples[i-1])
        return increases / (len(samples) - 1)

    def _create_step_templates(self) -> list[np.ndarray]:
        """Create step function templates for matching.

        Returns:
            List of template images
        """
        templates = []

        # Create simple step templates of various sizes
        for size in [50, 100, 150]:
            template = np.ones((size, size), dtype=np.uint8) * 255

            # Draw a simple step pattern
            step_height = size // 3
            cv2.line(template, (0, step_height), (size//3, step_height), 0, 2)
            cv2.line(template, (size//3, step_height), (size//3, 2*step_height), 0, 2)
            cv2.line(template, (size//3, 2*step_height), (2*size//3, 2*step_height), 0, 2)
            cv2.line(template, (2*size//3, 2*step_height), (2*size//3, size-10), 0, 2)
            cv2.line(template, (2*size//3, size-10), (size-10, size-10), 0, 2)

            templates.append(template)

        return templates

    def _non_max_suppression(
        self,
        detections: list[DetectedGraph],
        overlap_threshold: float = 0.5
    ) -> list[DetectedGraph]:
        """Apply non-maximum suppression to remove overlapping detections.

        Args:
            detections: List of detections
            overlap_threshold: IoU threshold for suppression

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        for detection in sorted_detections:
            # Check overlap with kept detections
            should_keep = True
            for kept in keep:
                iou = self._calculate_iou(detection.bbox, kept.bbox)
                if iou > overlap_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep.append(detection)

        return keep

    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate intersection over union of two bounding boxes.

        Args:
            bbox1: (x, y, width, height)
            bbox2: (x, y, width, height)

        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi

        if wi <= 0 or hi <= 0:
            return 0.0

        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def _overlaps_existing(
        self,
        detection: DetectedGraph,
        existing: list[DetectedGraph],
        threshold: float = 0.3
    ) -> bool:
        """Check if detection overlaps significantly with existing detections.

        Args:
            detection: Detection to check
            existing: List of existing detections
            threshold: IoU threshold

        Returns:
            True if overlaps
        """
        for existing_det in existing:
            iou = self._calculate_iou(detection.bbox, existing_det.bbox)
            if iou > threshold:
                return True
        return False


def detect_km_curves(image: Image.Image | np.ndarray) -> list[DetectedGraph]:
    """Convenience function to detect KM curves in an image.

    Args:
        image: Input image

    Returns:
        List of detected graphs
    """
    detector = GraphDetector(image)
    return detector.detect_all()


def get_graph_regions(
    image: Image.Image | np.ndarray,
    min_confidence: float = 0.3
) -> list[tuple]:
    """Get bounding boxes of detected graph regions.

    Args:
        image: Input image
        min_confidence: Minimum confidence threshold

    Returns:
        List of (x, y, width, height) tuples
    """
    detections = detect_km_curves(image)
    return [d.bbox for d in detections if d.confidence >= min_confidence]
