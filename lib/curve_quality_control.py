"""
Curve Quality Control Module

Validates extracted Kaplan-Meier curves for common extraction errors:
- Sudden unrealistic drops in survival
- Curves that go to zero too early
- Curves picking up reference lines or axes
- Missing data regions
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import numpy as np


class QCIssueType(Enum):
    """Types of quality control issues."""
    SUDDEN_DROP = "sudden_drop"
    EARLY_ZERO = "early_zero"
    FLAT_REGION = "flat_at_zero"
    NEGATIVE_TIME = "negative_time"
    SURVIVAL_OUT_OF_RANGE = "survival_out_of_range"
    TOO_FEW_POINTS = "too_few_points"
    NON_MONOTONIC = "non_monotonic"
    REFERENCE_LINE_CONTAMINATION = "reference_line"


class QCSeverity(Enum):
    """Severity levels for QC issues."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QCIssue:
    """A single quality control issue."""
    issue_type: QCIssueType
    severity: QCSeverity
    message: str
    time_range: Optional[Tuple[float, float]] = None
    survival_range: Optional[Tuple[float, float]] = None
    suggestion: str = ""


@dataclass
class CurveQCResult:
    """Quality control result for a single curve."""
    curve_name: str
    passed: bool
    issues: List[QCIssue] = field(default_factory=list)
    score: float = 1.0  # 0.0 = completely invalid, 1.0 = perfect

    def add_issue(self, issue: QCIssue):
        self.issues.append(issue)
        # Adjust score based on severity
        if issue.severity == QCSeverity.CRITICAL:
            self.score *= 0.1
            self.passed = False
        elif issue.severity == QCSeverity.ERROR:
            self.score *= 0.5
            self.passed = False
        elif issue.severity == QCSeverity.WARNING:
            self.score *= 0.9


@dataclass
class QCReport:
    """Complete QC report for all curves."""
    curve_results: Dict[str, CurveQCResult] = field(default_factory=dict)
    overall_passed: bool = True

    def add_curve_result(self, result: CurveQCResult):
        self.curve_results[result.curve_name] = result
        if not result.passed:
            self.overall_passed = False


class CurveQualityChecker:
    """
    Quality control checker for extracted KM curves.
    """

    def __init__(
        self,
        max_sudden_drop: float = 0.25,  # Max allowed drop between consecutive points
        min_points: int = 10,
        early_zero_threshold: float = 0.3,  # Fraction of time range where zero is suspicious
        flat_zero_threshold: float = 0.5,  # If >50% of curve is at zero, flag it
        reference_line_values: List[float] = None  # Common reference lines (0.5, etc.)
    ):
        self.max_sudden_drop = max_sudden_drop
        self.min_points = min_points
        self.early_zero_threshold = early_zero_threshold
        self.flat_zero_threshold = flat_zero_threshold
        self.reference_line_values = reference_line_values or [0.5, 0.25, 0.75]
        self.reference_line_tolerance = 0.03  # Tolerance for detecting reference lines

    def check_curve(
        self,
        points: List[Tuple[float, float]],
        curve_name: str,
        expected_time_max: float = None
    ) -> CurveQCResult:
        """
        Run all quality checks on a curve.

        Args:
            points: List of (time, survival) tuples
            curve_name: Name of the curve for reporting
            expected_time_max: Expected maximum time (for early zero detection)

        Returns:
            CurveQCResult with all detected issues
        """
        result = CurveQCResult(curve_name=curve_name, passed=True)

        if not points:
            result.add_issue(QCIssue(
                issue_type=QCIssueType.TOO_FEW_POINTS,
                severity=QCSeverity.CRITICAL,
                message="No data points in curve",
                suggestion="Check curve detection settings"
            ))
            return result

        times = np.array([p[0] for p in points])
        survivals = np.array([p[1] for p in points])

        if expected_time_max is None:
            expected_time_max = times.max()

        # Check 1: Minimum number of points
        self._check_min_points(result, len(points))

        # Check 2: Negative times
        self._check_negative_times(result, times)

        # Check 3: Survival values in range
        self._check_survival_range(result, survivals)

        # Check 4: Sudden drops
        self._check_sudden_drops(result, times, survivals)

        # Check 5: Early zero
        self._check_early_zero(result, times, survivals, expected_time_max)

        # Check 6: Flat at zero (stuck at zero for most of curve)
        self._check_flat_zero(result, survivals)

        # Check 7: Reference line contamination
        self._check_reference_line_contamination(result, times, survivals)

        # Check 8: Non-monotonic (survival increasing)
        self._check_monotonicity(result, times, survivals)

        return result

    def _check_min_points(self, result: CurveQCResult, num_points: int):
        if num_points < self.min_points:
            result.add_issue(QCIssue(
                issue_type=QCIssueType.TOO_FEW_POINTS,
                severity=QCSeverity.ERROR,
                message=f"Only {num_points} points (minimum: {self.min_points})",
                suggestion="Increase image resolution or check detection"
            ))

    def _check_negative_times(self, result: CurveQCResult, times: np.ndarray):
        neg_times = times[times < 0]
        if len(neg_times) > 0:
            result.add_issue(QCIssue(
                issue_type=QCIssueType.NEGATIVE_TIME,
                severity=QCSeverity.ERROR,
                message=f"Found {len(neg_times)} points with negative time",
                time_range=(float(neg_times.min()), float(neg_times.max())),
                suggestion="Check axis calibration"
            ))

    def _check_survival_range(self, result: CurveQCResult, survivals: np.ndarray):
        out_of_range = survivals[(survivals < 0) | (survivals > 1.0)]
        if len(out_of_range) > 0:
            result.add_issue(QCIssue(
                issue_type=QCIssueType.SURVIVAL_OUT_OF_RANGE,
                severity=QCSeverity.ERROR,
                message=f"Found {len(out_of_range)} points with survival outside [0,1]",
                survival_range=(float(out_of_range.min()), float(out_of_range.max())),
                suggestion="Check Y-axis calibration"
            ))

    def _check_sudden_drops(self, result: CurveQCResult, times: np.ndarray, survivals: np.ndarray):
        """Check for unrealistically sudden drops in survival."""
        if len(survivals) < 2:
            return

        drops = np.diff(survivals)
        # Find large negative drops (survival decreasing rapidly)
        large_drop_indices = np.where(drops < -self.max_sudden_drop)[0]

        for idx in large_drop_indices:
            drop_size = -drops[idx]
            time_before = times[idx]
            time_after = times[idx + 1]
            surv_before = survivals[idx]
            surv_after = survivals[idx + 1]

            result.add_issue(QCIssue(
                issue_type=QCIssueType.SUDDEN_DROP,
                severity=QCSeverity.ERROR,
                message=f"Sudden drop of {drop_size:.1%} at time {time_before:.1f}-{time_after:.1f} "
                        f"(from {surv_before:.1%} to {surv_after:.1%})",
                time_range=(float(time_before), float(time_after)),
                survival_range=(float(surv_after), float(surv_before)),
                suggestion="May be picking up reference line or wrong curve segment"
            ))

    def _check_early_zero(self, result: CurveQCResult, times: np.ndarray,
                          survivals: np.ndarray, expected_time_max: float):
        """Check if curve goes to zero suspiciously early."""
        if len(survivals) == 0:
            return

        zero_threshold = 0.01  # Consider values < 1% as "zero"
        zero_indices = np.where(survivals < zero_threshold)[0]

        if len(zero_indices) == 0:
            return

        first_zero_time = times[zero_indices[0]]
        early_cutoff = expected_time_max * self.early_zero_threshold

        if first_zero_time < early_cutoff:
            result.add_issue(QCIssue(
                issue_type=QCIssueType.EARLY_ZERO,
                severity=QCSeverity.CRITICAL,
                message=f"Curve reaches zero at time {first_zero_time:.1f}, "
                        f"which is only {first_zero_time/expected_time_max:.0%} of the time range",
                time_range=(float(first_zero_time), float(expected_time_max)),
                suggestion="Curve may be following the X-axis instead of actual data"
            ))

    def _check_flat_zero(self, result: CurveQCResult, survivals: np.ndarray):
        """Check if curve is flat at zero for most of its length."""
        if len(survivals) == 0:
            return

        zero_count = np.sum(survivals < 0.01)
        zero_fraction = zero_count / len(survivals)

        if zero_fraction > self.flat_zero_threshold:
            result.add_issue(QCIssue(
                issue_type=QCIssueType.FLAT_REGION,
                severity=QCSeverity.CRITICAL,
                message=f"{zero_fraction:.0%} of curve points are at zero",
                suggestion="Curve may be following the X-axis or bottom of plot"
            ))

    def _check_reference_line_contamination(self, result: CurveQCResult,
                                            times: np.ndarray, survivals: np.ndarray):
        """Check if curve suddenly jumps to a common reference line value."""
        if len(survivals) < 10:
            return

        for ref_value in self.reference_line_values:
            # Check if a significant portion of the curve is near this reference value
            near_ref = np.abs(survivals - ref_value) < self.reference_line_tolerance

            if np.sum(near_ref) > len(survivals) * 0.3:  # More than 30% near reference
                # Check if there's a sudden jump to this value
                diffs = np.diff(survivals)
                jumps_to_ref = np.where(
                    (np.abs(survivals[1:] - ref_value) < self.reference_line_tolerance) &
                    (np.abs(diffs) > 0.1)
                )[0]

                if len(jumps_to_ref) > 0:
                    jump_idx = jumps_to_ref[0]
                    result.add_issue(QCIssue(
                        issue_type=QCIssueType.REFERENCE_LINE_CONTAMINATION,
                        severity=QCSeverity.ERROR,
                        message=f"Curve appears to follow {ref_value:.0%} reference line "
                                f"after time {times[jump_idx]:.1f}",
                        time_range=(float(times[jump_idx]), float(times[-1])),
                        suggestion=f"May be picking up the dashed {ref_value:.0%} reference line"
                    ))
                    break

    def _check_monotonicity(self, result: CurveQCResult, times: np.ndarray, survivals: np.ndarray):
        """Check that survival is monotonically non-increasing."""
        if len(survivals) < 2:
            return

        increases = np.diff(survivals)
        significant_increases = increases > 0.02  # Allow tiny numerical errors

        if np.any(significant_increases):
            increase_count = np.sum(significant_increases)
            max_increase = np.max(increases)
            result.add_issue(QCIssue(
                issue_type=QCIssueType.NON_MONOTONIC,
                severity=QCSeverity.WARNING,
                message=f"Survival increases {increase_count} times (max increase: {max_increase:.1%})",
                suggestion="May indicate noise or detection errors"
            ))


def run_quality_control(
    curves_data: List[Dict],
    expected_time_max: float = None,
    strict: bool = False
) -> QCReport:
    """
    Run quality control on extracted curves.

    Args:
        curves_data: List of curve dictionaries with 'name' and 'clean_points'
        expected_time_max: Expected maximum time value
        strict: If True, use stricter thresholds

    Returns:
        QCReport with results for all curves
    """
    checker = CurveQualityChecker(
        max_sudden_drop=0.15 if strict else 0.25,
        early_zero_threshold=0.2 if strict else 0.3
    )

    report = QCReport()

    for curve_data in curves_data:
        name = curve_data.get('name', 'unknown')
        points = curve_data.get('clean_points', [])

        result = checker.check_curve(points, name, expected_time_max)
        report.add_curve_result(result)

    return report


def format_qc_report(report: QCReport, use_colors: bool = True) -> str:
    """Format QC report for display."""
    if use_colors:
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        BOLD = '\033[1m'
        NC = '\033[0m'
    else:
        RED = GREEN = YELLOW = BOLD = NC = ''

    lines = []
    lines.append(f"\n{BOLD}QUALITY CONTROL REPORT{NC}")
    lines.append("=" * 50)

    for curve_name, result in report.curve_results.items():
        status = f"{GREEN}PASSED{NC}" if result.passed else f"{RED}FAILED{NC}"
        score_color = GREEN if result.score > 0.8 else (YELLOW if result.score > 0.5 else RED)

        lines.append(f"\n{BOLD}{curve_name}{NC}: {status} (score: {score_color}{result.score:.0%}{NC})")

        if result.issues:
            for issue in result.issues:
                severity_color = {
                    QCSeverity.WARNING: YELLOW,
                    QCSeverity.ERROR: RED,
                    QCSeverity.CRITICAL: RED
                }.get(issue.severity, NC)

                severity_icon = {
                    QCSeverity.WARNING: "⚠",
                    QCSeverity.ERROR: "✗",
                    QCSeverity.CRITICAL: "✗✗"
                }.get(issue.severity, "?")

                lines.append(f"  {severity_color}{severity_icon} [{issue.severity.value.upper()}]{NC} {issue.message}")
                if issue.suggestion:
                    lines.append(f"      → {issue.suggestion}")

    lines.append("")
    overall = f"{GREEN}ALL PASSED{NC}" if report.overall_passed else f"{RED}ISSUES DETECTED{NC}"
    lines.append(f"{BOLD}Overall:{NC} {overall}")

    return "\n".join(lines)
