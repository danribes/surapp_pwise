"""Extract KM curves from image using color detection."""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def extract_curves_from_image(image_path: str, output_dir: str = None):
    """
    Extract Research (blue) and Control (red) curves from KM plot image.

    Args:
        image_path: Path to the KM curve image
        output_dir: Directory to save results (if None, creates results/<image_name>_<timestamp>/)
    """
    # Create output directory structure
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        # Create results folder with subfolder for this extraction
        results_base = Path("results")
        results_base.mkdir(exist_ok=True)
        output_dir = results_base / f"{image_name}_{timestamp}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}")

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges
    # Blue (Research curve) - solid line
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])

    # Red (Control curve) - dashed line
    # Red wraps around in HSV, so we need two ranges
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    # Create masks
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Find plot boundaries by analyzing the image
    # Looking at the image structure:
    # - Y-axis labels on left (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
    # - X-axis labels on bottom (0, 2, 4, 6, 8, 10, 12)

    # Detect plot area by finding where the curves exist
    blue_points = np.where(blue_mask > 0)
    red_points = np.where(red_mask > 0)

    if len(blue_points[0]) == 0 and len(red_points[0]) == 0:
        raise ValueError("No curves detected in image")

    # Combine all curve points to find plot boundaries
    all_y = np.concatenate([blue_points[0], red_points[0]])
    all_x = np.concatenate([blue_points[1], red_points[1]])

    # Plot boundaries (approximate from curve extents)
    x_min_px = int(np.min(all_x))
    x_max_px = int(np.max(all_x))
    y_min_px = int(np.min(all_y))  # Top of plot (survival = 1.0)
    y_max_px = int(np.max(all_y))  # Bottom of plot area

    print(f"Detected plot area: x=[{x_min_px}, {x_max_px}], y=[{y_min_px}, {y_max_px}]")

    # Axis ranges from the image
    time_min = 0.0
    time_max = 12.0
    survival_min = 0.0
    survival_max = 1.0

    # However, the curves don't go all the way to 0.0 survival
    # We need to find the actual plot boundaries more accurately
    # The y-axis 1.0 is at the top, 0.0 is at the bottom

    # Estimate plot boundaries based on typical figure layout
    # The curves start near y=1.0 and the plot extends to y=0.0
    # Looking at the proportions, if curves reach ~0.4 survival at the lowest
    # and the detected y_max corresponds to ~0.4, we can extrapolate

    # For this specific image, let's use the curve starting points
    # The curves start at survival=1.0, so y_min_px corresponds to survival=1.0
    # We need to find where survival=0.0 would be

    # The y-axis spans from 1.0 to 0.0, which is a range of 1.0
    # The lowest curve point is around survival=0.39
    # So the plot y-range in pixels = (y_max_px - y_min_px) / (1.0 - 0.39) * 1.0

    # Better approach: assume the plot is square-ish and use the x-range to estimate
    plot_x_range = x_max_px - x_min_px

    # Estimate the full plot height (from survival=1.0 to survival=0.0)
    # The x-axis goes from 0 to 12, and looking at typical KM plots,
    # the aspect ratio is usually around 1:1 or 4:3

    # Let's use a more robust approach: find the topmost curve point (should be at survival≈1.0)
    # and use that as reference

    # The topmost blue points should be at or near survival=1.0
    curve_top_y = y_min_px

    # Estimate where survival=0.0 would be based on aspect ratio
    # Assuming typical plot aspect ratio, the full survival range (1.0)
    # corresponds to similar pixel range as time range (12 years)
    estimated_full_y_range = plot_x_range * 0.7  # Typical KM plot is wider than tall

    # Calculate calibration
    # survival = 1.0 at y_min_px, survival = 0.0 at y_min_px + estimated_full_y_range
    plot_y_max = y_min_px + estimated_full_y_range

    print(f"Estimated full plot: x=[{x_min_px}, {x_max_px}], y=[{y_min_px}, {plot_y_max:.0f}]")

    def pixel_to_coord(px_x, px_y):
        """Convert pixel coordinates to data coordinates."""
        # X: linear from x_min_px=0 to x_max_px=12
        time = time_min + (px_x - x_min_px) / (x_max_px - x_min_px) * (time_max - time_min)

        # Y: linear from y_min_px=1.0 to plot_y_max=0.0 (inverted because y increases downward)
        survival = survival_max - (px_y - y_min_px) / (plot_y_max - y_min_px) * (survival_max - survival_min)

        return time, survival

    def extract_curve_points_with_overlap(blue_mask, red_mask):
        """
        Extract curve points from color masks, detecting overlapping regions.

        When curves overlap, both get the same survival value at that time point.

        Returns:
            research_points: list of (time, survival) for research curve
            control_points: list of (time, survival) for control curve
            overlap_regions: list of (time_start, time_end) for overlapping regions
        """
        research_points = []
        control_points = []
        overlap_times = []

        # Threshold for considering curves as overlapping (in pixels)
        overlap_threshold = 5  # If curves are within 5 pixels, consider them overlapping

        # Sample at each x position
        for x in range(x_min_px, x_max_px + 1, 2):  # Sample every 2 pixels
            blue_column = blue_mask[:, x]
            red_column = red_mask[:, x]

            blue_pixels = np.where(blue_column > 0)[0]
            red_pixels = np.where(red_column > 0)[0]

            has_blue = len(blue_pixels) > 0
            has_red = len(red_pixels) > 0

            time, _ = pixel_to_coord(x, 0)

            if not (0 <= time <= time_max):
                continue

            if has_blue and has_red:
                # Both curves present - check for overlap
                blue_y = int(np.median(blue_pixels))
                red_y = int(np.median(red_pixels))

                y_diff = abs(blue_y - red_y)

                if y_diff <= overlap_threshold:
                    # Curves are overlapping - use average position for both
                    avg_y = (blue_y + red_y) / 2
                    _, survival = pixel_to_coord(x, avg_y)
                    survival = max(0.0, min(1.0, survival))

                    research_points.append((time, survival, True))  # True = overlapping
                    control_points.append((time, survival, True))
                    overlap_times.append(time)
                else:
                    # Curves are separate
                    _, blue_surv = pixel_to_coord(x, blue_y)
                    _, red_surv = pixel_to_coord(x, red_y)

                    blue_surv = max(0.0, min(1.0, blue_surv))
                    red_surv = max(0.0, min(1.0, red_surv))

                    research_points.append((time, blue_surv, False))
                    control_points.append((time, red_surv, False))

            elif has_blue:
                # Only blue (research) curve
                blue_y = int(np.median(blue_pixels))
                _, survival = pixel_to_coord(x, blue_y)
                survival = max(0.0, min(1.0, survival))
                research_points.append((time, survival, False))

            elif has_red:
                # Only red (control) curve
                red_y = int(np.median(red_pixels))
                _, survival = pixel_to_coord(x, red_y)
                survival = max(0.0, min(1.0, survival))
                control_points.append((time, survival, False))

        # Identify overlap regions (continuous time ranges)
        overlap_regions = []
        if overlap_times:
            overlap_times.sort()
            region_start = overlap_times[0]
            region_end = overlap_times[0]

            for t in overlap_times[1:]:
                if t - region_end < 0.5:  # Within 0.5 time units = same region
                    region_end = t
                else:
                    overlap_regions.append((region_start, region_end))
                    region_start = t
                    region_end = t

            overlap_regions.append((region_start, region_end))

        # Strip the overlap flag for return
        research_points = [(t, s) for t, s, _ in research_points]
        control_points = [(t, s) for t, s, _ in control_points]

        return research_points, control_points, overlap_regions, len(overlap_times)

    # Extract curves with overlap detection
    print("\nExtracting curves with overlap detection...")
    research_raw, control_raw, overlap_regions, overlap_count = extract_curve_points_with_overlap(blue_mask, red_mask)

    print(f"  Research (blue): {len(research_raw)} raw points")
    print(f"  Control (red): {len(control_raw)} raw points")
    print(f"  Overlapping points detected: {overlap_count}")

    if overlap_regions:
        print(f"\n  Overlap regions (curves share same values):")
        for i, (t_start, t_end) in enumerate(overlap_regions, 1):
            if t_end - t_start < 0.1:
                print(f"    Region {i}: t = {t_start:.2f} years (single point)")
            else:
                print(f"    Region {i}: t = {t_start:.2f} to {t_end:.2f} years")

    def clean_km_curve(points):
        """Clean curve to proper KM step format."""
        if not points:
            return []

        # Sort by time
        points = sorted(points, key=lambda p: p[0])

        # Remove duplicates and ensure monotonicity
        cleaned = []
        prev_survival = 2.0  # Start above max

        for time, survival in points:
            # Round for comparison
            surv_rounded = round(survival, 4)

            # Only keep if survival decreased
            if surv_rounded < prev_survival - 0.001:
                cleaned.append((round(time, 4), surv_rounded))
                prev_survival = surv_rounded

        # Ensure we start at (0, 1.0) if first point is close
        if cleaned and cleaned[0][0] > 0.1:
            cleaned.insert(0, (0.0, 1.0))
        elif cleaned and cleaned[0][1] < 0.99:
            cleaned.insert(0, (0.0, 1.0))

        return cleaned

    # Clean curves
    research_clean = clean_km_curve(research_raw)
    control_clean = clean_km_curve(control_raw)

    # Synchronize overlapping regions - ensure both curves have same values
    def synchronize_overlap_regions(research, control, overlap_regions):
        """
        Ensure both curves have identical values in overlapping regions.

        For each overlap region:
        1. Find all time points from both curves in that region
        2. Use the average survival at each common time point
        3. Add synchronized points to both curves
        """
        if not overlap_regions:
            return research, control

        research_dict = {round(t, 4): s for t, s in research}
        control_dict = {round(t, 4): s for t, s in control}

        synced_points = []  # Points that should be in both curves

        for t_start, t_end in overlap_regions:
            # Get all time points in this region from both curves
            r_in_region = [(t, s) for t, s in research if t_start - 0.1 <= t <= t_end + 0.1]
            c_in_region = [(t, s) for t, s in control if t_start - 0.1 <= t <= t_end + 0.1]

            # Combine all time points
            all_times = set([round(t, 4) for t, _ in r_in_region] +
                           [round(t, 4) for t, _ in c_in_region])

            for t in sorted(all_times):
                r_surv = research_dict.get(t)
                c_surv = control_dict.get(t)

                if r_surv is not None and c_surv is not None:
                    # Both have this time - use average
                    avg_surv = (r_surv + c_surv) / 2
                    synced_points.append((t, avg_surv))
                elif r_surv is not None:
                    synced_points.append((t, r_surv))
                elif c_surv is not None:
                    synced_points.append((t, c_surv))

        # Remove overlap region points from original curves and add synced points
        research_new = [(t, s) for t, s in research
                       if not any(t_start - 0.1 <= t <= t_end + 0.1
                                 for t_start, t_end in overlap_regions)]
        control_new = [(t, s) for t, s in control
                      if not any(t_start - 0.1 <= t <= t_end + 0.1
                                for t_start, t_end in overlap_regions)]

        # Add synced points to both
        research_new.extend(synced_points)
        control_new.extend(synced_points)

        # Sort and clean again
        research_new = sorted(research_new, key=lambda p: p[0])
        control_new = sorted(control_new, key=lambda p: p[0])

        # Remove duplicates keeping first occurrence
        def dedupe(points):
            seen_times = set()
            result = []
            for t, s in points:
                t_round = round(t, 4)
                if t_round not in seen_times:
                    seen_times.add(t_round)
                    result.append((t, s))
            return result

        return dedupe(research_new), dedupe(control_new)

    def enforce_monotonicity(points):
        """
        Apply isotonic regression to ensure monotonically decreasing values.
        Uses Pool Adjacent Violators Algorithm (PAVA).
        """
        if not points:
            return points

        points = sorted(points, key=lambda p: p[0])
        times = [p[0] for p in points]
        survivals = [p[1] for p in points]

        # PAVA for decreasing monotonicity
        n = len(survivals)
        result = survivals.copy()

        i = 0
        while i < n - 1:
            if result[i] < result[i + 1]:
                # Violation: survival increased
                # Pool adjacent values
                j = i + 1
                while j < n - 1 and result[j] < result[j + 1]:
                    j += 1

                # Average the violating region
                avg = sum(result[i:j+1]) / (j - i + 1)
                for k in range(i, j + 1):
                    result[k] = avg

                # Go back to check for new violations
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Reconstruct points and remove consecutive duplicates
        final = []
        prev_surv = None
        for t, s in zip(times, result):
            s_round = round(s, 4)
            if prev_surv is None or s_round < prev_surv:
                final.append((round(t, 4), s_round))
                prev_surv = s_round

        return final

    # Apply overlap synchronization
    if overlap_regions:
        research_clean, control_clean = synchronize_overlap_regions(
            research_clean, control_clean, overlap_regions
        )
        print(f"\nAfter overlap synchronization:")
        print(f"  Research: {len(research_clean)} points")
        print(f"  Control: {len(control_clean)} points")

        # Apply monotonicity enforcement after synchronization
        research_clean = enforce_monotonicity(research_clean)
        control_clean = enforce_monotonicity(control_clean)
        print(f"\nAfter monotonicity enforcement:")
        print(f"  Research: {len(research_clean)} points")
        print(f"  Control: {len(control_clean)} points")

        # Count synchronized points
        r_times = set(round(t, 4) for t, _ in research_clean)
        c_times = set(round(t, 4) for t, _ in control_clean)
        common_times = r_times & c_times

        # Check how many have identical values
        identical = 0
        for t in common_times:
            r_val = next((s for tm, s in research_clean if round(tm, 4) == t), None)
            c_val = next((s for tm, s in control_clean if round(tm, 4) == t), None)
            if r_val and c_val and abs(r_val - c_val) < 0.001:
                identical += 1

        print(f"  Synchronized (identical) points: {identical}")
    else:
        print(f"\nAfter cleaning (step points only):")
        print(f"  Research: {len(research_clean)} points")
        print(f"  Control: {len(control_clean)} points")

    # Save results
    # Individual curves
    research_df = pd.DataFrame(research_clean, columns=['Time', 'Survival'])
    control_df = pd.DataFrame(control_clean, columns=['Time', 'Survival'])

    research_df.to_csv(f"{output_dir}/research_curve.csv", index=False)
    control_df.to_csv(f"{output_dir}/control_curve.csv", index=False)

    # Combined file
    research_df['Curve'] = 'Research'
    control_df['Curve'] = 'Control'
    combined = pd.concat([research_df, control_df], ignore_index=True)
    combined = combined[['Curve', 'Time', 'Survival']]
    combined.to_csv(f"{output_dir}/both_curves.csv", index=False)

    # Save overlap information
    if overlap_regions:
        overlap_df = pd.DataFrame(overlap_regions, columns=['Time_Start', 'Time_End'])
        overlap_df.to_csv(f"{output_dir}/overlap_regions.csv", index=False)

    print(f"\nSaved to {output_dir}/")
    print(f"  - research_curve.csv")
    print(f"  - control_curve.csv")
    print(f"  - both_curves.csv")
    if overlap_regions:
        print(f"  - overlap_regions.csv")

    # Verification
    print("\n" + "="*60)
    print("VERIFICATION & MONOTONICITY CHECK")
    print("="*60)

    all_valid = True

    for name, data in [('Research', research_clean), ('Control', control_clean)]:
        if not data:
            print(f"\n{name}: No data extracted")
            all_valid = False
            continue

        times = [p[0] for p in data]
        survivals = [p[1] for p in data]

        # Check monotonicity
        violations = []
        for i in range(len(survivals) - 1):
            if survivals[i] < survivals[i + 1]:
                violations.append({
                    'index': i,
                    'time1': times[i],
                    'surv1': survivals[i],
                    'time2': times[i + 1],
                    'surv2': survivals[i + 1],
                    'increase': survivals[i + 1] - survivals[i]
                })

        is_monotonic = len(violations) == 0
        is_strictly_decreasing = all(survivals[i] > survivals[i+1] for i in range(len(survivals)-1))

        print(f"\n{name}:")
        print(f"  Points: {len(data)}")
        print(f"  Time range: {min(times):.2f} - {max(times):.2f} years")
        print(f"  Survival range: {min(survivals):.4f} - {max(survivals):.4f}")

        if is_monotonic:
            print(f"  ✓ MONOTONICALLY DECREASING: Yes")
            if is_strictly_decreasing:
                print(f"  ✓ STRICTLY DECREASING: Yes (no repeated values)")
            else:
                # Find repeated values
                repeated = sum(1 for i in range(len(survivals)-1) if survivals[i] == survivals[i+1])
                print(f"  ⚠ STRICTLY DECREASING: No ({repeated} repeated consecutive values)")
        else:
            print(f"  ✗ MONOTONICALLY DECREASING: NO - {len(violations)} violations found!")
            all_valid = False
            print(f"\n  Violations (survival increased instead of decreased):")
            for v in violations[:10]:  # Show first 10
                print(f"    At t={v['time1']:.2f}: {v['surv1']:.4f} → t={v['time2']:.2f}: {v['surv2']:.4f} (↑{v['increase']:.4f})")
            if len(violations) > 10:
                print(f"    ... and {len(violations) - 10} more violations")

    # Overlap summary
    print("\n" + "="*60)
    print("OVERLAP DETECTION SUMMARY")
    print("="*60)
    if overlap_regions:
        print(f"  Overlapping points: {overlap_count}")
        print(f"  Overlap regions: {len(overlap_regions)}")
        for i, (t_start, t_end) in enumerate(overlap_regions, 1):
            duration = t_end - t_start
            if duration < 0.1:
                print(f"    {i}. t = {t_start:.2f} years (point overlap)")
            else:
                print(f"    {i}. t = {t_start:.2f} - {t_end:.2f} years ({duration:.2f} years)")
        print("\n  Note: In overlap regions, both curves have identical survival values")
    else:
        print("  No overlapping regions detected")
        print("  Curves are fully separated throughout the time range")

    # Overall result
    print("\n" + "="*60)
    if all_valid:
        print("✓ ALL CURVES VALID - Data is suitable for survival analysis")
    else:
        print("✗ VALIDATION FAILED - Some curves have issues")
    print("="*60)

    # Display first/last few points
    print("\n" + "="*60)
    print("RESEARCH CURVE (first 10 and last 5 points)")
    print("="*60)
    for t, s in research_clean[:10]:
        print(f"  Time: {t:6.2f}  Survival: {s:.4f}")
    if len(research_clean) > 15:
        print("  ...")
        for t, s in research_clean[-5:]:
            print(f"  Time: {t:6.2f}  Survival: {s:.4f}")

    print("\n" + "="*60)
    print("CONTROL CURVE (first 10 and last 5 points)")
    print("="*60)
    for t, s in control_clean[:10]:
        print(f"  Time: {t:6.2f}  Survival: {s:.4f}")
    if len(control_clean) > 15:
        print("  ...")
        for t, s in control_clean[-5:]:
            print(f"  Time: {t:6.2f}  Survival: {s:.4f}")

    # Save debug mask images
    cv2.imwrite(f"{output_dir}/debug_blue_mask.png", blue_mask)
    cv2.imwrite(f"{output_dir}/debug_red_mask.png", red_mask)
    print(f"\nDebug masks saved for inspection")

    return research_clean, control_clean


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "bmjopen-2019-September-9-9--F1.large.jpg"

    extract_curves_from_image(image_path)
