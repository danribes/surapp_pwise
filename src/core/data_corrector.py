"""Data correction module with isotonic regression."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Try to import sklearn's IsotonicRegression, fall back to our PAVA implementation
try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class CorrectionResult:
    """Result of data correction."""
    original_points: list[tuple]
    corrected_points: list[tuple]
    changes_made: int


def apply_isotonic_regression(
    data_points: list[tuple],
    decreasing: bool = True,
    force_start: Optional[tuple] = None
) -> list[tuple]:
    """Apply isotonic regression to ensure monotonicity.

    For Kaplan-Meier curves, survival should be monotonically decreasing.

    Args:
        data_points: List of (x, y) data points
        decreasing: If True, ensure monotonically decreasing (default for KM curves)
        force_start: Optional starting point to force (e.g., (0, 100))

    Returns:
        List of corrected (x, y) data points
    """
    if not data_points:
        return []

    # Sort by X values
    sorted_points = sorted(data_points, key=lambda p: p[0])

    # Extract X and Y arrays
    x_values = np.array([p[0] for p in sorted_points])
    y_values = np.array([p[1] for p in sorted_points])

    # Apply isotonic regression
    if HAS_SKLEARN:
        if decreasing:
            # For decreasing, we negate, fit increasing, then negate back
            iso_reg = IsotonicRegression(increasing=True)
            y_corrected = -iso_reg.fit_transform(x_values, -y_values)
        else:
            iso_reg = IsotonicRegression(increasing=True)
            y_corrected = iso_reg.fit_transform(x_values, y_values)
    else:
        # Use our PAVA implementation
        y_list = y_values.tolist()
        if decreasing:
            y_corrected = np.array(pava_algorithm(y_list))
        else:
            # For increasing, negate, apply PAVA, negate back
            negated = [-y for y in y_list]
            corrected = pava_algorithm(negated)
            y_corrected = np.array([-y for y in corrected])

    # Force starting point if specified
    if force_start is not None:
        x_start, y_start = force_start

        # Find if we have a point at x_start
        if len(x_values) > 0 and x_values[0] <= x_start:
            # Adjust first point
            y_corrected[0] = y_start
        else:
            # Prepend the starting point
            x_values = np.insert(x_values, 0, x_start)
            y_corrected = np.insert(y_corrected, 0, y_start)

    # Reconstruct points
    corrected_points = [(float(x), float(y)) for x, y in zip(x_values, y_corrected)]

    return corrected_points


def pava_algorithm(values: list[float], weights: list[float] = None) -> list[float]:
    """Pool Adjacent Violators Algorithm for isotonic regression.

    This implements the classic PAVA algorithm for decreasing monotonicity.

    Args:
        values: List of Y values
        weights: Optional weights for each value

    Returns:
        Isotonically corrected Y values (monotonically decreasing)
    """
    n = len(values)
    if n == 0:
        return []

    if weights is None:
        weights = [1.0] * n

    # Work with copies
    y = list(values)
    w = list(weights)

    # Pool Adjacent Violators
    # For decreasing, we look for violations where y[i] < y[i+1]
    i = 0
    while i < n - 1:
        if y[i] < y[i + 1]:
            # Violation found - pool adjacent values
            # Find the extent of the violation
            j = i + 1
            while j < n - 1 and y[j] < y[j + 1]:
                j += 1

            # Pool values from i to j
            total_weight = sum(w[i:j+1])
            weighted_avg = sum(y[k] * w[k] for k in range(i, j+1)) / total_weight

            # Replace with weighted average
            for k in range(i, j+1):
                y[k] = weighted_avg

            # Go back to check for new violations
            if i > 0:
                i -= 1
        else:
            i += 1

    return y


def correct_km_data(
    data_points: list[tuple],
    force_start_at_100: bool = True,
    y_scale: float = 100.0
) -> CorrectionResult:
    """Correct Kaplan-Meier survival data.

    Ensures:
    1. Starts at (0, max_survival) if force_start_at_100
    2. Monotonically decreasing
    3. Values bounded between 0 and max_survival

    Args:
        data_points: List of (time, survival) data points
        force_start_at_100: If True, force starting point to (0, y_scale)
        y_scale: Scale for Y axis (100 for percentage, 1.0 for proportion)

    Returns:
        CorrectionResult with original and corrected data
    """
    if not data_points:
        return CorrectionResult([], [], 0)

    original = list(data_points)

    # Sort by time
    sorted_points = sorted(data_points, key=lambda p: p[0])

    # Extract values
    times = [p[0] for p in sorted_points]
    survivals = [p[1] for p in sorted_points]

    # Clip values to valid range
    survivals = [max(0, min(y_scale, s)) for s in survivals]

    # Apply PAVA for monotonic decreasing
    corrected_survivals = pava_algorithm(survivals)

    # Force start at max if requested
    if force_start_at_100:
        # Ensure first point is at time 0 with max survival
        if len(times) == 0 or times[0] > 0:
            times.insert(0, 0.0)
            corrected_survivals.insert(0, y_scale)
        else:
            corrected_survivals[0] = y_scale

    # Count changes
    changes = sum(
        1 for orig, corr in zip(survivals, corrected_survivals)
        if abs(orig - corr) > 0.001
    )

    # Reconstruct points
    corrected_points = list(zip(times, corrected_survivals))

    return CorrectionResult(original, corrected_points, changes)


def interpolate_missing_points(
    data_points: list[tuple],
    interval: float = 1.0,
    max_time: float = None
) -> list[tuple]:
    """Interpolate missing time points using step function.

    For KM curves, we use backward fill (hold previous value until next step).

    Args:
        data_points: List of (time, survival) data points
        interval: Time interval for interpolation
        max_time: Maximum time value (default: max from data)

    Returns:
        List of interpolated data points
    """
    if not data_points:
        return []

    sorted_points = sorted(data_points, key=lambda p: p[0])

    if max_time is None:
        max_time = sorted_points[-1][0]

    # Generate time points
    interpolated = []
    current_survival = sorted_points[0][1]
    point_idx = 0

    t = 0.0
    while t <= max_time:
        # Update survival if we've passed a data point
        while point_idx < len(sorted_points) and sorted_points[point_idx][0] <= t:
            current_survival = sorted_points[point_idx][1]
            point_idx += 1

        interpolated.append((t, current_survival))
        t += interval

    return interpolated


def calculate_statistics(data_points: list[tuple]) -> dict:
    """Calculate summary statistics for the survival data.

    Args:
        data_points: List of (time, survival) data points

    Returns:
        Dictionary with statistics
    """
    if not data_points:
        return {}

    sorted_points = sorted(data_points, key=lambda p: p[0])

    times = [p[0] for p in sorted_points]
    survivals = [p[1] for p in sorted_points]

    # Find median survival time (time when survival drops to 50%)
    median_time = None
    for t, s in sorted_points:
        if s <= 50:
            median_time = t
            break

    # Calculate area under curve (rough estimate using trapezoidal rule)
    auc = 0
    for i in range(1, len(sorted_points)):
        dt = times[i] - times[i-1]
        avg_survival = (survivals[i] + survivals[i-1]) / 2
        auc += dt * avg_survival

    return {
        "n_points": len(data_points),
        "time_range": (times[0], times[-1]),
        "survival_range": (min(survivals), max(survivals)),
        "median_survival_time": median_time,
        "area_under_curve": auc,
        "final_survival": survivals[-1]
    }
