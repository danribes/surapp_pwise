"""
KM Curve Extraction Validator

Comprehensive validation module to ensure extracted curves match original images.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class ValidationStatus(Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationReport:
    """Complete validation report for an extraction."""
    results: List[ValidationResult] = field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.PASS
    sample_point_errors: List[Dict] = field(default_factory=list)
    suggested_adjustments: Dict = field(default_factory=dict)

    def add_result(self, result: ValidationResult):
        self.results.append(result)
        # Update overall status (FAIL > WARNING > PASS)
        if result.status == ValidationStatus.FAIL:
            self.overall_status = ValidationStatus.FAIL
        elif result.status == ValidationStatus.WARNING and self.overall_status != ValidationStatus.FAIL:
            self.overall_status = ValidationStatus.WARNING

    def summary(self) -> str:
        """Generate a summary string of the validation report."""
        lines = [f"Overall Status: {self.overall_status.value}"]
        lines.append("-" * 40)

        for result in self.results:
            icon = {"PASS": "[OK]", "WARNING": "[!!]", "FAIL": "[XX]"}[result.status.value]
            lines.append(f"{icon} {result.name}: {result.message}")

        if self.suggested_adjustments:
            lines.append("-" * 40)
            lines.append("Suggested Adjustments:")
            for key, value in self.suggested_adjustments.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class KMCurveValidator:
    """Validator for Kaplan-Meier curve extractions."""

    def __init__(self, original_img: np.ndarray, calibration, plot_bounds: Tuple[int, int, int, int]):
        """
        Initialize validator.

        Args:
            original_img: Original image (grayscale or BGR)
            calibration: Calibration object with axis information
            plot_bounds: (x, y, width, height) of plot area
        """
        self.original_img = original_img
        self.calibration = calibration
        self.plot_bounds = plot_bounds

        # Convert to grayscale if needed
        if len(original_img.shape) == 3:
            self.gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = original_img

        # Extract calibration info
        if calibration:
            self.time_range = calibration.x_data_range
            self.survival_range = calibration.y_data_range
            self.origin = calibration.origin
            self.x_axis_end = calibration.x_axis_end
            self.y_axis_end = calibration.y_axis_end
        else:
            self.time_range = (0, 10)
            self.survival_range = (0, 1)
            self.origin = (plot_bounds[0], plot_bounds[1] + plot_bounds[3])
            self.x_axis_end = (plot_bounds[0] + plot_bounds[2], plot_bounds[1] + plot_bounds[3])
            self.y_axis_end = (plot_bounds[0], plot_bounds[1])

    def validate_all(self, curves_data: List[Dict], verbose: bool = False) -> ValidationReport:
        """
        Run all validation checks on extracted curves.

        Args:
            curves_data: List of curve data dictionaries
            verbose: Print progress messages

        Returns:
            ValidationReport with all results
        """
        report = ValidationReport()

        if verbose:
            print("\n[Validation] Running extraction validation...")

        # 1. Axis Calibration Validation
        report.add_result(self._validate_calibration())

        # 2. Curve Count Validation
        report.add_result(self._validate_curve_count(curves_data))

        # 3. KM Properties Validation (per curve)
        for curve_data in curves_data:
            report.add_result(self._validate_starting_point(curve_data))
            report.add_result(self._validate_monotonicity(curve_data))
            report.add_result(self._validate_range_bounds(curve_data))
            report.add_result(self._validate_early_region(curve_data))
            report.add_result(self._validate_tail_coverage(curve_data))

        # 4. Sample Point Validation
        sample_results = self._validate_sample_points(curves_data)
        report.add_result(sample_results['result'])
        report.sample_point_errors = sample_results['errors']

        # 5. Overlay Comparison
        overlay_result = self._validate_overlay_match(curves_data)
        report.add_result(overlay_result['result'])
        if overlay_result.get('adjustments'):
            report.suggested_adjustments = overlay_result['adjustments']

        # 6. Curve Separation Validation
        if len(curves_data) >= 2:
            report.add_result(self._validate_curve_separation(curves_data))

        if verbose:
            print(report.summary())

        return report

    def _validate_calibration(self) -> ValidationResult:
        """Validate axis calibration."""
        issues = []

        if self.calibration is None:
            return ValidationResult(
                name="Axis Calibration",
                status=ValidationStatus.FAIL,
                message="No calibration data available"
            )

        # Check time range
        t_min, t_max = self.time_range
        if t_min != 0:
            issues.append(f"Time should start at 0, got {t_min}")
        if t_max <= 0:
            issues.append(f"Invalid max time: {t_max}")

        # Check survival range
        s_min, s_max = self.survival_range
        if s_min != 0:
            issues.append(f"Survival min should be 0, got {s_min}")
        if s_max not in [1.0, 100.0]:
            issues.append(f"Survival max should be 1.0 or 100, got {s_max}")

        # Check origin and endpoints
        if self.origin[0] >= self.x_axis_end[0]:
            issues.append("Origin X should be less than X-axis end")
        if self.origin[1] <= self.y_axis_end[1]:
            issues.append("Origin Y should be greater than Y-axis end (Y increases downward)")

        if issues:
            return ValidationResult(
                name="Axis Calibration",
                status=ValidationStatus.WARNING,
                message=f"{len(issues)} issue(s) found",
                details={"issues": issues}
            )

        return ValidationResult(
            name="Axis Calibration",
            status=ValidationStatus.PASS,
            message=f"X: 0-{t_max}, Y: 0-{s_max}"
        )

    def _validate_curve_count(self, curves_data: List[Dict]) -> ValidationResult:
        """Validate number of detected curves."""
        count = len(curves_data)

        if count == 0:
            return ValidationResult(
                name="Curve Count",
                status=ValidationStatus.FAIL,
                message="No curves detected"
            )
        elif count == 1:
            return ValidationResult(
                name="Curve Count",
                status=ValidationStatus.WARNING,
                message="Only 1 curve detected (expected 2 for comparison)"
            )
        elif count == 2:
            return ValidationResult(
                name="Curve Count",
                status=ValidationStatus.PASS,
                message="2 curves detected"
            )
        else:
            return ValidationResult(
                name="Curve Count",
                status=ValidationStatus.WARNING,
                message=f"{count} curves detected (expected 2)"
            )

    def _validate_starting_point(self, curve_data: Dict) -> ValidationResult:
        """Validate that curve starts at (0, 1.0)."""
        name = curve_data.get('name', 'Unknown')
        points = curve_data.get('clean_points', [])

        if not points:
            return ValidationResult(
                name=f"Starting Point ({name})",
                status=ValidationStatus.FAIL,
                message="No points in curve"
            )

        first_t, first_s = points[0]

        issues = []
        if abs(first_t) > 0.01:
            issues.append(f"First time={first_t:.3f}, should be 0")
        if abs(first_s - 1.0) > 0.01:
            issues.append(f"First survival={first_s:.3f}, should be 1.0")

        if issues:
            return ValidationResult(
                name=f"Starting Point ({name})",
                status=ValidationStatus.WARNING if abs(first_s - 1.0) < 0.05 else ValidationStatus.FAIL,
                message="; ".join(issues)
            )

        return ValidationResult(
            name=f"Starting Point ({name})",
            status=ValidationStatus.PASS,
            message=f"Starts at ({first_t:.2f}, {first_s:.3f})"
        )

    def _validate_monotonicity(self, curve_data: Dict) -> ValidationResult:
        """Validate that survival is monotonically non-increasing."""
        name = curve_data.get('name', 'Unknown')
        points = curve_data.get('clean_points', [])

        if len(points) < 2:
            return ValidationResult(
                name=f"Monotonicity ({name})",
                status=ValidationStatus.WARNING,
                message="Not enough points to check"
            )

        violations = []
        for i in range(1, len(points)):
            prev_t, prev_s = points[i-1]
            curr_t, curr_s = points[i]

            # Time should increase
            if curr_t < prev_t:
                violations.append(f"Time decreased at t={curr_t:.2f}")

            # Survival should not increase
            if curr_s > prev_s + 0.001:  # Small tolerance for floating point
                violations.append(f"Survival increased at t={curr_t:.2f}: {prev_s:.3f} -> {curr_s:.3f}")

        if violations:
            return ValidationResult(
                name=f"Monotonicity ({name})",
                status=ValidationStatus.FAIL,
                message=f"{len(violations)} violation(s)",
                details={"violations": violations[:5]}  # First 5
            )

        return ValidationResult(
            name=f"Monotonicity ({name})",
            status=ValidationStatus.PASS,
            message="Monotonically non-increasing"
        )

    def _validate_range_bounds(self, curve_data: Dict) -> ValidationResult:
        """Validate that all points are within valid ranges."""
        name = curve_data.get('name', 'Unknown')
        points = curve_data.get('clean_points', [])

        if not points:
            return ValidationResult(
                name=f"Range Bounds ({name})",
                status=ValidationStatus.FAIL,
                message="No points"
            )

        t_min = min(p[0] for p in points)
        t_max = max(p[0] for p in points)
        s_min = min(p[1] for p in points)
        s_max = max(p[1] for p in points)

        issues = []
        if t_min < -0.01:
            issues.append(f"Negative time: {t_min:.3f}")
        if s_min < -0.01:
            issues.append(f"Negative survival: {s_min:.3f}")
        if s_max > 1.01:
            issues.append(f"Survival > 1: {s_max:.3f}")

        if issues:
            return ValidationResult(
                name=f"Range Bounds ({name})",
                status=ValidationStatus.WARNING,
                message="; ".join(issues)
            )

        return ValidationResult(
            name=f"Range Bounds ({name})",
            status=ValidationStatus.PASS,
            message=f"t:[{t_min:.1f},{t_max:.1f}] s:[{s_min:.2f},{s_max:.2f}]"
        )

    def _validate_early_region(self, curve_data: Dict) -> ValidationResult:
        """Validate curve behavior in the early region (t=0 to t=20% of range).

        Checks:
        1. Curve starts exactly at (0, 1.0)
        2. Early points are properly detected and aligned with original
        3. No artificial steps or gaps in the first portion
        4. Pixel-level alignment check for detecting curve mixing
        """
        name = curve_data.get('name', 'Unknown')
        points = curve_data.get('clean_points', [])

        if not points:
            return ValidationResult(
                name=f"Early Region ({name})",
                status=ValidationStatus.FAIL,
                message="No points"
            )

        t_min, t_max = self.time_range
        early_threshold = t_max * 0.20  # First 20% of time range (expanded from 5%)

        # Get points in early region
        early_points = [(t, s) for t, s in points if t <= early_threshold]

        issues = []
        details = {"early_point_count": len(early_points), "pixel_errors": []}

        # Check 1: First point should be at (0, 1.0)
        first_t, first_s = points[0]
        if abs(first_t) > 0.001:
            issues.append(f"First point time={first_t:.3f} (should be 0)")
        if abs(first_s - 1.0) > 0.005:
            issues.append(f"First point survival={first_s:.4f} (should be 1.0)")

        # Check 2: Should have points in early region
        if len(early_points) < 3:
            issues.append(f"Only {len(early_points)} points in early region (t<{early_threshold:.1f})")

        # Check 3: Early points should be at or very close to s=1.0
        # (KM curves typically stay at 1.0 briefly before first event)
        if early_points:
            early_survivals = [s for _, s in early_points]
            max_early_survival = max(early_survivals)
            if max_early_survival < 0.99:
                issues.append(f"Max survival in early region={max_early_survival:.3f} (should be ~1.0)")

        # Check 4: Validate early points against original image at multiple sample points
        # Sample at regular intervals through the early region
        sample_times = [t_max * p for p in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]]
        early_errors = []

        for sample_t in sample_times:
            # Find the closest extracted point to this sample time
            closest_point = None
            min_time_diff = float('inf')
            for t, s in early_points:
                time_diff = abs(t - sample_t)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_point = (t, s)

            if closest_point and min_time_diff < t_max * 0.02:  # Within 2% of range
                pixel = self._data_to_pixel(closest_point[0], closest_point[1])
                distance = self._measure_pixel_distance(pixel)
                if distance != float('inf'):
                    early_errors.append({
                        'time': closest_point[0],
                        'survival': closest_point[1],
                        'pixel_error': distance
                    })
                    details["pixel_errors"].append({
                        't': round(closest_point[0], 2),
                        's': round(closest_point[1], 3),
                        'err': round(distance, 1)
                    })

        if early_errors:
            errors_only = [e['pixel_error'] for e in early_errors]
            avg_early_error = sum(errors_only) / len(errors_only)
            max_early_error = max(errors_only)
            details["avg_error"] = round(avg_early_error, 1)
            details["max_error"] = round(max_early_error, 1)

            # More strict threshold for early region (curves should align well here)
            if max_early_error > 4:
                issues.append(f"Early region pixel error: max={max_early_error:.1f}px, avg={avg_early_error:.1f}px")
            # Check for systematic offset (all errors in same direction suggests curve mixing)
            if len(errors_only) >= 3 and avg_early_error > 2:
                issues.append(f"Possible curve misalignment in early region (avg error={avg_early_error:.1f}px)")

        if issues:
            status = ValidationStatus.FAIL if any('First point' in i for i in issues) else ValidationStatus.WARNING
            return ValidationResult(
                name=f"Early Region ({name})",
                status=status,
                message="; ".join(issues[:3]),  # Limit to first 3 issues
                details=details
            )

        return ValidationResult(
            name=f"Early Region ({name})",
            status=ValidationStatus.PASS,
            message=f"Starts at (0, 1.0), {len(early_points)} early points OK"
        )

    def _validate_tail_coverage(self, curve_data: Dict) -> ValidationResult:
        """Validate curve tail coverage (last 10% of time range).

        Checks:
        1. Curve extends close to the expected time_max
        2. Tail points are properly detected and aligned with original
        3. No premature truncation of the curve
        """
        name = curve_data.get('name', 'Unknown')
        points = curve_data.get('clean_points', [])

        if not points:
            return ValidationResult(
                name=f"Tail Coverage ({name})",
                status=ValidationStatus.FAIL,
                message="No points"
            )

        t_min, t_max = self.time_range
        tail_start = t_max * 0.90  # Last 10% of time range

        # Get actual time range of extracted curve
        actual_t_max = max(t for t, _ in points)
        actual_t_min = min(t for t, _ in points)

        # Get points in tail region
        tail_points = [(t, s) for t, s in points if t >= tail_start]

        issues = []

        # Check 1: Curve should extend close to expected time_max
        coverage = actual_t_max / t_max if t_max > 0 else 0
        if coverage < 0.95:
            issues.append(f"Curve ends at t={actual_t_max:.1f}, expected ~{t_max:.1f} ({coverage*100:.0f}% coverage)")

        # Check 2: Should have points in tail region
        expected_tail_points = max(3, int(len(points) * 0.05))  # At least 3 or 5% of total
        if len(tail_points) < expected_tail_points:
            issues.append(f"Only {len(tail_points)} points in tail region (t>{tail_start:.1f})")

        # Check 3: Validate tail points against original image
        tail_errors = []
        for t, s in tail_points[-5:]:  # Check last 5 tail points
            pixel = self._data_to_pixel(t, s)
            distance = self._measure_pixel_distance(pixel)
            if distance != float('inf'):
                tail_errors.append(distance)

        if tail_errors:
            avg_tail_error = sum(tail_errors) / len(tail_errors)
            max_tail_error = max(tail_errors)
            if max_tail_error > 5:
                issues.append(f"Tail region pixel error: max={max_tail_error:.1f}px, avg={avg_tail_error:.1f}px")

        # Check 4: Verify tail survival values are reasonable (should be lower than early region)
        if tail_points:
            tail_survivals = [s for _, s in tail_points]
            max_tail_survival = max(tail_survivals)
            min_tail_survival = min(tail_survivals)

            # Tail should have lower survival than the overall max (unless curve is flat)
            overall_max_s = max(s for _, s in points)
            if max_tail_survival >= overall_max_s - 0.01 and len(points) > 10:
                issues.append(f"Tail survival={max_tail_survival:.2f} not lower than max={overall_max_s:.2f}")

        if issues:
            # Coverage issue is more serious
            status = ValidationStatus.FAIL if coverage < 0.90 else ValidationStatus.WARNING
            return ValidationResult(
                name=f"Tail Coverage ({name})",
                status=status,
                message="; ".join(issues[:3]),
                details={"issues": issues, "coverage": coverage, "tail_point_count": len(tail_points)}
            )

        return ValidationResult(
            name=f"Tail Coverage ({name})",
            status=ValidationStatus.PASS,
            message=f"Extends to t={actual_t_max:.1f} ({coverage*100:.0f}% coverage), {len(tail_points)} tail points"
        )

    def _validate_sample_points(self, curves_data: List[Dict]) -> Dict:
        """Validate curves at regular sample points."""
        t_min, t_max = self.time_range

        # Generate sample times based on axis scale
        sample_times = [
            0,
            t_max * 0.1,
            t_max * 0.2,
            t_max * 0.25,
            t_max * 0.5,
            t_max * 0.75,
            t_max * 0.9,
            t_max
        ]

        errors = []
        max_error = 0
        total_error = 0
        count = 0

        for curve_data in curves_data:
            name = curve_data.get('name', 'Unknown')
            points = curve_data.get('clean_points', [])

            if not points:
                continue

            for sample_t in sample_times:
                # Find closest point in extracted data
                closest_idx = min(range(len(points)), key=lambda i: abs(points[i][0] - sample_t))
                extracted_t, extracted_s = points[closest_idx]

                # Get expected pixel position
                expected_pixel = self._data_to_pixel(sample_t, extracted_s)

                # Check if there's a dark pixel nearby in original image
                pixel_error = self._measure_pixel_distance(expected_pixel)

                error_entry = {
                    'curve': name,
                    'sample_time': sample_t,
                    'extracted_survival': extracted_s,
                    'pixel_error': pixel_error
                }
                errors.append(error_entry)

                # Only count finite errors in statistics
                if pixel_error != float('inf'):
                    max_error = max(max_error, pixel_error)
                    total_error += pixel_error
                    count += 1

        avg_error = total_error / count if count > 0 else 0

        # Count out-of-bounds points
        oob_count = sum(1 for e in errors if e['pixel_error'] == float('inf'))

        # Determine status based on errors
        if count == 0:
            status = ValidationStatus.WARNING
            message = "No valid sample points to measure"
        elif max_error > 10:  # More than 10 pixels off
            status = ValidationStatus.FAIL
            message = f"Max pixel error: {max_error:.1f}px (threshold: 10px)"
        elif max_error > 5:
            status = ValidationStatus.WARNING
            message = f"Max pixel error: {max_error:.1f}px, avg: {avg_error:.1f}px"
        else:
            status = ValidationStatus.PASS
            message = f"Max pixel error: {max_error:.1f}px, avg: {avg_error:.1f}px"

        # Add OOB info if any
        if oob_count > 0:
            message += f" ({oob_count} sample(s) out of bounds)"

        return {
            'result': ValidationResult(
                name="Sample Point Accuracy",
                status=status,
                message=message,
                details={'max_error': max_error, 'avg_error': avg_error}
            ),
            'errors': errors
        }

    def _validate_overlay_match(self, curves_data: List[Dict]) -> Dict:
        """Validate overall overlay match with original image."""
        total_points = 0
        matched_points = 0
        y_offset_sum = 0
        y_offset_count = 0

        for curve_data in curves_data:
            points = curve_data.get('clean_points', [])

            for t, s in points:
                pixel = self._data_to_pixel(t, s)
                px, py = int(pixel[0]), int(pixel[1])

                # Check bounds
                if not (0 <= px < self.gray_img.shape[1] and 0 <= py < self.gray_img.shape[0]):
                    continue

                total_points += 1

                # Check if there's a dark pixel nearby (curve line)
                match_distance, best_y = self._find_nearest_dark_pixel(px, py, search_radius=8)

                if match_distance <= 5:
                    matched_points += 1

                if best_y is not None:
                    y_offset_sum += (best_y - py)
                    y_offset_count += 1

        match_ratio = matched_points / total_points if total_points > 0 else 0
        avg_y_offset = y_offset_sum / y_offset_count if y_offset_count > 0 else 0

        # Determine status
        if match_ratio >= 0.95:
            status = ValidationStatus.PASS
            message = f"{match_ratio*100:.1f}% points match original"
        elif match_ratio >= 0.85:
            status = ValidationStatus.WARNING
            message = f"{match_ratio*100:.1f}% points match (target: 95%)"
        else:
            status = ValidationStatus.FAIL
            message = f"Only {match_ratio*100:.1f}% points match original"

        # Suggest adjustments if needed
        adjustments = {}
        if abs(avg_y_offset) > 2:
            adjustments['y_offset_adjustment'] = f"{avg_y_offset:.1f} pixels"

        return {
            'result': ValidationResult(
                name="Overlay Match",
                status=status,
                message=message,
                details={'match_ratio': match_ratio, 'avg_y_offset': avg_y_offset}
            ),
            'adjustments': adjustments
        }

    def _validate_curve_separation(self, curves_data: List[Dict]) -> ValidationResult:
        """Validate that curves are properly separated (not merged)."""
        if len(curves_data) < 2:
            return ValidationResult(
                name="Curve Separation",
                status=ValidationStatus.PASS,
                message="Single curve, no separation check needed"
            )

        # Compare first two curves
        curve1_points = curves_data[0].get('clean_points', [])
        curve2_points = curves_data[1].get('clean_points', [])

        if not curve1_points or not curve2_points:
            return ValidationResult(
                name="Curve Separation",
                status=ValidationStatus.WARNING,
                message="Missing curve points"
            )

        # Check survival difference at various time points
        t_max = self.time_range[1]
        sample_times = [t_max * 0.25, t_max * 0.5, t_max * 0.75]

        differences = []
        for sample_t in sample_times:
            # Get survival for each curve at this time
            s1 = self._get_survival_at_time(curve1_points, sample_t)
            s2 = self._get_survival_at_time(curve2_points, sample_t)

            if s1 is not None and s2 is not None:
                differences.append(abs(s1 - s2))

        if not differences:
            return ValidationResult(
                name="Curve Separation",
                status=ValidationStatus.WARNING,
                message="Could not measure separation"
            )

        avg_diff = sum(differences) / len(differences)
        max_diff = max(differences)

        if max_diff < 0.02:  # Curves are essentially the same
            return ValidationResult(
                name="Curve Separation",
                status=ValidationStatus.WARNING,
                message=f"Curves may be merged (max diff: {max_diff:.3f})"
            )

        return ValidationResult(
            name="Curve Separation",
            status=ValidationStatus.PASS,
            message=f"Curves separated (avg diff: {avg_diff:.3f})"
        )

    def _data_to_pixel(self, time: float, survival: float) -> Tuple[float, float]:
        """Convert data coordinates to pixel coordinates."""
        t_min, t_max = self.time_range
        s_min, s_max = self.survival_range

        # Normalize survival if using percentage scale
        if s_max > 1.5:
            survival = survival * s_max

        # X coordinate
        origin_x = self.origin[0]
        x_end = self.x_axis_end[0]
        px = origin_x + (time - t_min) / (t_max - t_min) * (x_end - origin_x)

        # Y coordinate (inverted - Y increases downward in pixels)
        origin_y = self.origin[1]  # Bottom (s=0)
        y_end = self.y_axis_end[1]  # Top (s=1)
        py = y_end + (1.0 - survival) * (origin_y - y_end)

        return (px, py)

    def _measure_pixel_distance(self, pixel: Tuple[float, float]) -> float:
        """Measure distance to nearest dark pixel in original image."""
        px, py = int(pixel[0]), int(pixel[1])

        # Check bounds
        if not (0 <= px < self.gray_img.shape[1] and 0 <= py < self.gray_img.shape[0]):
            return float('inf')

        distance, _ = self._find_nearest_dark_pixel(px, py, search_radius=15)
        return distance

    def _find_nearest_dark_pixel(self, px: int, py: int, search_radius: int = 10) -> Tuple[float, Optional[int]]:
        """Find nearest dark pixel to given position."""
        min_distance = float('inf')
        best_y = None

        threshold = 128  # Dark pixel threshold

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = px + dx, py + dy

                if not (0 <= nx < self.gray_img.shape[1] and 0 <= ny < self.gray_img.shape[0]):
                    continue

                if self.gray_img[ny, nx] < threshold:
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_y = ny

        return (min_distance, best_y)

    def _get_survival_at_time(self, points: List[Tuple[float, float]], target_time: float) -> Optional[float]:
        """Get survival value at a specific time using linear interpolation."""
        if not points:
            return None

        # Find surrounding points
        prev_point = None
        next_point = None

        for t, s in points:
            if t <= target_time:
                prev_point = (t, s)
            if t >= target_time and next_point is None:
                next_point = (t, s)
                break

        if prev_point is None:
            return points[0][1] if points else None
        if next_point is None:
            return prev_point[1]

        # Interpolate (though KM curves are step functions, use the previous value)
        return prev_point[1]


class DenseValidationReport:
    """Detailed dense validation report."""

    def __init__(self):
        self.curve_reports = {}  # {curve_name: CurveDenseReport}
        self.overall_accuracy = 0.0
        self.missing_steps = []
        self.offset_analysis = {}

    def summary(self) -> str:
        lines = ["=" * 60, "DENSE VALIDATION REPORT", "=" * 60]

        lines.append(f"\nOverall Accuracy: {self.overall_accuracy*100:.1f}%")

        for curve_name, report in self.curve_reports.items():
            lines.append(f"\n--- {curve_name} ---")
            lines.append(f"  Points matched: {report['matched_points']}/{report['total_points']} ({report['match_rate']*100:.1f}%)")
            lines.append(f"  Average Y-offset: {report['avg_offset']:.2f} pixels")
            lines.append(f"  Max Y-offset: {report['max_offset']:.2f} pixels")
            lines.append(f"  Steps detected: {report['steps_detected']}")
            lines.append(f"  Steps in original: {report['steps_in_original']}")

            if report.get('missing_step_regions'):
                lines.append(f"  Missing step regions: {len(report['missing_step_regions'])}")
                for region in report['missing_step_regions'][:5]:  # Show first 5
                    lines.append(f"    - t={region['time_start']:.1f} to t={region['time_end']:.1f} (gap of {region['survival_gap']:.3f})")

            if report.get('offset_regions'):
                lines.append(f"  Significant offset regions:")
                for region in report['offset_regions'][:5]:
                    lines.append(f"    - t={region['time']:.1f}: offset={region['offset']:.1f}px ({region['direction']})")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class DenseValidator:
    """Dense validation that checks every pixel position."""

    def __init__(self, original_img: np.ndarray, calibration, plot_bounds: Tuple[int, int, int, int]):
        self.original_img = original_img
        self.calibration = calibration
        self.plot_bounds = plot_bounds

        # Convert to grayscale if needed
        if len(original_img.shape) == 3:
            self.gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = original_img

        # Extract calibration info
        if calibration:
            self.time_range = calibration.x_data_range
            self.survival_range = calibration.y_data_range
            self.origin = calibration.origin
            self.x_axis_end = calibration.x_axis_end
            self.y_axis_end = calibration.y_axis_end
        else:
            self.time_range = (0, 10)
            self.survival_range = (0, 1)
            px, py, pw, ph = plot_bounds
            self.origin = (px, py + ph)
            self.x_axis_end = (px + pw, py + ph)
            self.y_axis_end = (px, py)

    def validate_dense(self, curves_data: List[Dict], color_masks: Dict[str, np.ndarray] = None) -> DenseValidationReport:
        """
        Perform dense validation checking every pixel position.

        Args:
            curves_data: List of curve data dictionaries
            color_masks: Optional dict of {curve_name: mask} for color-based checking

        Returns:
            DenseValidationReport with detailed analysis
        """
        report = DenseValidationReport()

        px, py, pw, ph = self.plot_bounds
        t_min, t_max = self.time_range

        total_matched = 0
        total_points = 0

        for curve_data in curves_data:
            curve_name = curve_data.get('name', 'Unknown')
            points = curve_data.get('clean_points', [])

            if not points:
                continue

            curve_report = self._validate_curve_dense(
                curve_name, points, color_masks.get(curve_name) if color_masks else None
            )
            report.curve_reports[curve_name] = curve_report

            total_matched += curve_report['matched_points']
            total_points += curve_report['total_points']

        report.overall_accuracy = total_matched / total_points if total_points > 0 else 0

        return report

    def _validate_curve_dense(self, curve_name: str, points: List[Tuple[float, float]],
                              color_mask: np.ndarray = None) -> Dict:
        """Validate a single curve with dense checking."""
        px, py, pw, ph = self.plot_bounds
        t_min, t_max = self.time_range

        # Build a lookup dict: time -> survival
        time_to_survival = {}
        for t, s in points:
            t_rounded = round(t, 3)
            if t_rounded not in time_to_survival:
                time_to_survival[t_rounded] = s

        # Statistics
        matched_points = 0
        total_points = 0
        y_offsets = []
        missing_step_regions = []
        offset_regions = []

        # Track steps in extracted curve
        steps_extracted = self._find_steps(points)

        # Dense check: iterate through every X pixel position
        prev_extracted_s = None
        gap_start = None

        for x_pixel in range(px, px + pw, 2):  # Check every 2 pixels for speed
            # Convert pixel X to time
            t = t_min + (x_pixel - px) / pw * (t_max - t_min)
            t_rounded = round(t, 3)

            # Get extracted survival at this time
            extracted_s = self._get_survival_at_time_dense(points, t)

            if extracted_s is None:
                continue

            total_points += 1

            # Convert to pixel Y
            extracted_y = int(self._survival_to_pixel_y(extracted_s))

            # Check if there's a dark pixel near the extracted position
            # This validates if the extracted curve matches something in the original
            offset, found_y = self._find_nearest_curve_pixel(x_pixel, extracted_y, color_mask)

            if offset is not None:
                y_offsets.append(offset)

                # Check if matched (within tolerance)
                if abs(offset) <= 5:
                    matched_points += 1
                elif abs(offset) > 8:
                    # Record significant offset
                    direction = "extracted below" if offset > 0 else "extracted above"
                    offset_regions.append({
                        'time': t,
                        'x_pixel': x_pixel,
                        'offset': offset,
                        'direction': direction,
                        'extracted_s': extracted_s,
                        'extracted_y': extracted_y,
                        'found_y': found_y
                    })

            # Track potential missing steps (large gaps in extracted survival)
            if prev_extracted_s is not None and extracted_s is not None:
                survival_gap = abs(prev_extracted_s - extracted_s)
                if survival_gap > 0.05:
                    # Check if original has pixels in this gap region
                    gap_y_start = int(self._survival_to_pixel_y(prev_extracted_s))
                    gap_y_end = int(self._survival_to_pixel_y(extracted_s))

                    # Look for curve pixels in the gap (potential missing steps)
                    has_pixels_in_gap = self._check_pixels_in_gap(
                        x_pixel, min(gap_y_start, gap_y_end), max(gap_y_start, gap_y_end), color_mask
                    )

                    if has_pixels_in_gap and gap_start is None:
                        gap_start = t
                    elif not has_pixels_in_gap and gap_start is not None:
                        missing_step_regions.append({
                            'time_start': gap_start,
                            'time_end': t,
                            'survival_gap': survival_gap
                        })
                        gap_start = None

            prev_extracted_s = extracted_s

        # Find steps in original image
        steps_original = self._count_steps_in_original(color_mask) if color_mask is not None else len(steps_extracted)

        # Calculate statistics
        avg_offset = np.mean(y_offsets) if y_offsets else 0
        max_offset = max(abs(o) for o in y_offsets) if y_offsets else 0

        # Consolidate offset regions (merge nearby ones)
        consolidated_offsets = self._consolidate_regions(offset_regions, 'time', threshold=1.0)

        return {
            'matched_points': matched_points,
            'total_points': total_points,
            'match_rate': matched_points / total_points if total_points > 0 else 0,
            'avg_offset': avg_offset,
            'max_offset': max_offset,
            'steps_detected': len(steps_extracted),
            'steps_in_original': steps_original,
            'missing_step_regions': missing_step_regions,
            'offset_regions': consolidated_offsets[:10],  # Top 10
            'all_offsets': y_offsets
        }

    def _survival_to_pixel_y(self, survival: float) -> float:
        """Convert survival value to pixel Y coordinate."""
        origin_y = self.origin[1]  # Bottom (s=0)
        y_end = self.y_axis_end[1]  # Top (s=1)
        return y_end + (1.0 - survival) * (origin_y - y_end)

    def _get_survival_at_time_dense(self, points: List[Tuple[float, float]], target_time: float) -> Optional[float]:
        """Get survival at specific time using step function interpolation."""
        if not points:
            return None

        # Find the last point with t <= target_time (step function behavior)
        result = None
        for t, s in points:
            if t <= target_time:
                result = s
            else:
                break

        return result

    def _find_nearest_curve_pixel(self, x_pixel: int, expected_y: int,
                                    color_mask: np.ndarray = None,
                                    search_radius: int = 15) -> Tuple[Optional[int], Optional[int]]:
        """Find the nearest dark/curve pixel to the expected position.

        Args:
            x_pixel: X coordinate to check
            expected_y: Expected Y coordinate from extracted curve
            color_mask: Optional color-specific mask
            search_radius: How far to search vertically

        Returns:
            (offset, found_y) where offset = found_y - expected_y, or (None, None) if not found
        """
        px, py, pw, ph = self.plot_bounds

        # Ensure we're within plot bounds
        if x_pixel < px or x_pixel >= px + pw:
            return None, None

        best_offset = None
        found_y = None

        for dy in range(-search_radius, search_radius + 1):
            check_y = expected_y + dy

            if check_y < py or check_y >= py + ph:
                continue

            if color_mask is not None:
                # Use color mask
                if 0 <= check_y < color_mask.shape[0] and 0 <= x_pixel < color_mask.shape[1]:
                    if color_mask[check_y, x_pixel] > 0:
                        if best_offset is None or abs(dy) < abs(best_offset):
                            best_offset = dy
                            found_y = check_y
            else:
                # Use grayscale pixel detection - look for non-white pixels
                # Gray curves have values around 100-170, so we use a higher threshold
                if 0 <= check_y < self.gray_img.shape[0] and 0 <= x_pixel < self.gray_img.shape[1]:
                    pixel_val = self.gray_img[check_y, x_pixel]
                    # Check for any non-white pixel (could be dark or gray curve)
                    if pixel_val < 200:  # More permissive threshold
                        if best_offset is None or abs(dy) < abs(best_offset):
                            best_offset = dy
                            found_y = check_y

        return best_offset, found_y

    def _check_pixels_in_gap(self, x_pixel: int, y_start: int, y_end: int,
                             color_mask: np.ndarray = None) -> bool:
        """Check if there are curve pixels in a vertical gap region.

        Returns True if there might be missing steps (pixels found in gap).
        """
        if y_end <= y_start:
            return False

        px, py, pw, ph = self.plot_bounds

        for y in range(y_start + 2, y_end - 2):  # Exclude boundaries
            if y < py or y >= py + ph:
                continue

            if color_mask is not None:
                if 0 <= y < color_mask.shape[0] and 0 <= x_pixel < color_mask.shape[1]:
                    if color_mask[y, x_pixel] > 0:
                        return True
            else:
                if 0 <= y < self.gray_img.shape[0] and 0 <= x_pixel < self.gray_img.shape[1]:
                    if self.gray_img[y, x_pixel] < 200:  # More permissive for gray curves
                        return True

        return False

    def _find_curve_at_x(self, x_pixel: int, color_mask: np.ndarray = None) -> Optional[int]:
        """Find the curve's Y position at a given X pixel."""
        px, py, pw, ph = self.plot_bounds

        if color_mask is not None:
            # Use color mask if available
            col = color_mask[py:py+ph, x_pixel]
            y_positions = np.where(col > 0)[0]
            if len(y_positions) > 0:
                # Return the topmost position (highest survival)
                return py + int(np.min(y_positions))
            return None

        # Otherwise, use grayscale dark pixel detection
        threshold = 128
        for y in range(py, py + ph):
            if 0 <= y < self.gray_img.shape[0] and 0 <= x_pixel < self.gray_img.shape[1]:
                if self.gray_img[y, x_pixel] < threshold:
                    return y

        return None

    def _find_steps(self, points: List[Tuple[float, float]], threshold: float = 0.01) -> List[Dict]:
        """Find all steps (survival changes) in the extracted curve."""
        steps = []
        prev_s = None
        prev_t = None

        for t, s in points:
            if prev_s is not None:
                if prev_s - s > threshold:  # Survival dropped
                    steps.append({
                        'time': t,
                        'from_survival': prev_s,
                        'to_survival': s,
                        'drop': prev_s - s
                    })
            prev_s = s
            prev_t = t

        return steps

    def _count_steps_in_original(self, color_mask: np.ndarray) -> int:
        """Count approximate number of steps in the original curve from the mask."""
        px, py, pw, ph = self.plot_bounds

        prev_y = None
        step_count = 0

        for x in range(px, px + pw, 5):  # Sample every 5 pixels
            col = color_mask[py:py+ph, x]
            y_positions = np.where(col > 0)[0]

            if len(y_positions) > 0:
                curr_y = int(np.min(y_positions))  # Topmost

                if prev_y is not None:
                    if abs(curr_y - prev_y) > 3:  # Step detected
                        step_count += 1

                prev_y = curr_y

        return step_count

    def _consolidate_regions(self, regions: List[Dict], time_key: str, threshold: float = 1.0) -> List[Dict]:
        """Consolidate nearby regions into single entries."""
        if not regions:
            return []

        # Sort by time
        sorted_regions = sorted(regions, key=lambda r: r[time_key])

        consolidated = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            if region[time_key] - consolidated[-1][time_key] < threshold:
                # Merge: keep the one with larger offset
                if abs(region.get('offset', 0)) > abs(consolidated[-1].get('offset', 0)):
                    consolidated[-1] = region
            else:
                consolidated.append(region)

        # Sort by offset magnitude
        consolidated.sort(key=lambda r: abs(r.get('offset', 0)), reverse=True)

        return consolidated


def run_dense_validation(original_img: np.ndarray,
                        calibration,
                        plot_bounds: Tuple[int, int, int, int],
                        curves_data: List[Dict],
                        color_masks: Dict[str, np.ndarray] = None,
                        verbose: bool = False) -> DenseValidationReport:
    """
    Run dense validation on extracted curves.

    Args:
        original_img: Original image
        calibration: Calibration object
        plot_bounds: Plot bounds
        curves_data: Extracted curves
        color_masks: Optional color masks for each curve
        verbose: Print progress

    Returns:
        DenseValidationReport
    """
    validator = DenseValidator(original_img, calibration, plot_bounds)
    report = validator.validate_dense(curves_data, color_masks)

    if verbose:
        print(report.summary())

    return report


def validate_extraction(original_img: np.ndarray,
                       calibration,
                       plot_bounds: Tuple[int, int, int, int],
                       curves_data: List[Dict],
                       verbose: bool = False) -> ValidationReport:
    """
    Convenience function to validate an extraction.

    Args:
        original_img: Original image
        calibration: Calibration object
        plot_bounds: Plot area bounds
        curves_data: Extracted curve data
        verbose: Print progress

    Returns:
        ValidationReport
    """
    validator = KMCurveValidator(original_img, calibration, plot_bounds)
    return validator.validate_all(curves_data, verbose=verbose)
