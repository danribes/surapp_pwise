"""Core extraction module for KM curves."""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractionResult:
    """Result of KM curve extraction."""
    research_curve: list
    control_curve: list
    overlap_regions: list
    output_dir: Path
    is_valid: bool
    research_points: int
    control_points: int
    overlap_count: int


def extract_curves_from_image(
    image_path: str,
    output_dir: Optional[str] = None,
    time_max: float = 12.0,
    overlap_threshold: int = 5,
    quiet: bool = False
) -> ExtractionResult:
    """
    Extract Research (blue) and Control (red) curves from KM plot image.

    Args:
        image_path: Path to the KM curve image
        output_dir: Directory to save results (if None, creates results/<image_name>_<timestamp>/)
        time_max: Maximum time value on X-axis (default: 12.0)
        overlap_threshold: Pixel threshold for considering curves as overlapping (default: 5)
        quiet: If True, suppress output messages

    Returns:
        ExtractionResult with extracted curves and metadata
    """
    def log(msg):
        if not quiet:
            print(msg)

    # Create output directory structure
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        results_base = Path("results")
        results_base.mkdir(exist_ok=True)
        output_dir = results_base / f"{image_name}_{timestamp}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Output directory: {output_dir}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]
    log(f"Image size: {width}x{height}")

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])

    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    # Create masks
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Detect plot area
    blue_points = np.where(blue_mask > 0)
    red_points = np.where(red_mask > 0)

    if len(blue_points[0]) == 0 and len(red_points[0]) == 0:
        raise ValueError("No curves detected in image")

    all_y = np.concatenate([blue_points[0], red_points[0]])
    all_x = np.concatenate([blue_points[1], red_points[1]])

    x_min_px = int(np.min(all_x))
    x_max_px = int(np.max(all_x))
    y_min_px = int(np.min(all_y))

    log(f"Detected plot area: x=[{x_min_px}, {x_max_px}], y_min={y_min_px}")

    # Axis ranges
    time_min = 0.0
    survival_min = 0.0
    survival_max = 1.0

    plot_x_range = x_max_px - x_min_px
    estimated_full_y_range = plot_x_range * 0.7
    plot_y_max = y_min_px + estimated_full_y_range

    def pixel_to_coord(px_x, px_y):
        time = time_min + (px_x - x_min_px) / (x_max_px - x_min_px) * (time_max - time_min)
        survival = survival_max - (px_y - y_min_px) / (plot_y_max - y_min_px) * (survival_max - survival_min)
        return time, survival

    def extract_curve_points_with_overlap():
        research_points = []
        control_points = []
        overlap_times = []

        for x in range(x_min_px, x_max_px + 1, 2):
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
                blue_y = int(np.median(blue_pixels))
                red_y = int(np.median(red_pixels))
                y_diff = abs(blue_y - red_y)

                if y_diff <= overlap_threshold:
                    avg_y = (blue_y + red_y) / 2
                    _, survival = pixel_to_coord(x, avg_y)
                    survival = max(0.0, min(1.0, survival))
                    research_points.append((time, survival, True))
                    control_points.append((time, survival, True))
                    overlap_times.append(time)
                else:
                    _, blue_surv = pixel_to_coord(x, blue_y)
                    _, red_surv = pixel_to_coord(x, red_y)
                    blue_surv = max(0.0, min(1.0, blue_surv))
                    red_surv = max(0.0, min(1.0, red_surv))
                    research_points.append((time, blue_surv, False))
                    control_points.append((time, red_surv, False))
            elif has_blue:
                blue_y = int(np.median(blue_pixels))
                _, survival = pixel_to_coord(x, blue_y)
                survival = max(0.0, min(1.0, survival))
                research_points.append((time, survival, False))
            elif has_red:
                red_y = int(np.median(red_pixels))
                _, survival = pixel_to_coord(x, red_y)
                survival = max(0.0, min(1.0, survival))
                control_points.append((time, survival, False))

        # Identify overlap regions
        overlap_regions = []
        if overlap_times:
            overlap_times.sort()
            region_start = overlap_times[0]
            region_end = overlap_times[0]

            for t in overlap_times[1:]:
                if t - region_end < 0.5:
                    region_end = t
                else:
                    overlap_regions.append((region_start, region_end))
                    region_start = t
                    region_end = t
            overlap_regions.append((region_start, region_end))

        research_points = [(t, s) for t, s, _ in research_points]
        control_points = [(t, s) for t, s, _ in control_points]

        return research_points, control_points, overlap_regions, len(overlap_times)

    log("\nExtracting curves with overlap detection...")
    research_raw, control_raw, overlap_regions, overlap_count = extract_curve_points_with_overlap()

    log(f"  Research (blue): {len(research_raw)} raw points")
    log(f"  Control (red): {len(control_raw)} raw points")
    log(f"  Overlapping points detected: {overlap_count}")

    if overlap_regions:
        log(f"\n  Overlap regions (curves share same values):")
        for i, (t_start, t_end) in enumerate(overlap_regions, 1):
            if t_end - t_start < 0.1:
                log(f"    Region {i}: t = {t_start:.2f} years (single point)")
            else:
                log(f"    Region {i}: t = {t_start:.2f} to {t_end:.2f} years")

    def clean_km_curve(points):
        if not points:
            return []
        points = sorted(points, key=lambda p: p[0])
        cleaned = []
        prev_survival = 2.0

        for time, survival in points:
            surv_rounded = round(survival, 4)
            if surv_rounded < prev_survival - 0.001:
                cleaned.append((round(time, 4), surv_rounded))
                prev_survival = surv_rounded

        if cleaned and cleaned[0][0] > 0.1:
            cleaned.insert(0, (0.0, 1.0))
        elif cleaned and cleaned[0][1] < 0.99:
            cleaned.insert(0, (0.0, 1.0))

        return cleaned

    research_clean = clean_km_curve(research_raw)
    control_clean = clean_km_curve(control_raw)

    def synchronize_overlap_regions(research, control, regions):
        if not regions:
            return research, control

        research_dict = {round(t, 4): s for t, s in research}
        control_dict = {round(t, 4): s for t, s in control}
        synced_points = []

        for t_start, t_end in regions:
            r_in_region = [(t, s) for t, s in research if t_start - 0.1 <= t <= t_end + 0.1]
            c_in_region = [(t, s) for t, s in control if t_start - 0.1 <= t <= t_end + 0.1]
            all_times = set([round(t, 4) for t, _ in r_in_region] +
                           [round(t, 4) for t, _ in c_in_region])

            for t in sorted(all_times):
                r_surv = research_dict.get(t)
                c_surv = control_dict.get(t)
                if r_surv is not None and c_surv is not None:
                    synced_points.append((t, (r_surv + c_surv) / 2))
                elif r_surv is not None:
                    synced_points.append((t, r_surv))
                elif c_surv is not None:
                    synced_points.append((t, c_surv))

        research_new = [(t, s) for t, s in research
                       if not any(t_start - 0.1 <= t <= t_end + 0.1 for t_start, t_end in regions)]
        control_new = [(t, s) for t, s in control
                      if not any(t_start - 0.1 <= t <= t_end + 0.1 for t_start, t_end in regions)]

        research_new.extend(synced_points)
        control_new.extend(synced_points)
        research_new = sorted(research_new, key=lambda p: p[0])
        control_new = sorted(control_new, key=lambda p: p[0])

        def dedupe(points):
            seen = set()
            result = []
            for t, s in points:
                t_round = round(t, 4)
                if t_round not in seen:
                    seen.add(t_round)
                    result.append((t, s))
            return result

        return dedupe(research_new), dedupe(control_new)

    def enforce_monotonicity(points):
        if not points:
            return points

        points = sorted(points, key=lambda p: p[0])
        times = [p[0] for p in points]
        survivals = [p[1] for p in points]

        n = len(survivals)
        result = survivals.copy()

        i = 0
        while i < n - 1:
            if result[i] < result[i + 1]:
                j = i + 1
                while j < n - 1 and result[j] < result[j + 1]:
                    j += 1
                avg = sum(result[i:j+1]) / (j - i + 1)
                for k in range(i, j + 1):
                    result[k] = avg
                if i > 0:
                    i -= 1
            else:
                i += 1

        final = []
        prev_surv = None
        for t, s in zip(times, result):
            s_round = round(s, 4)
            if prev_surv is None or s_round < prev_surv:
                final.append((round(t, 4), s_round))
                prev_surv = s_round

        return final

    if overlap_regions:
        research_clean, control_clean = synchronize_overlap_regions(
            research_clean, control_clean, overlap_regions
        )
        log(f"\nAfter overlap synchronization:")
        log(f"  Research: {len(research_clean)} points")
        log(f"  Control: {len(control_clean)} points")

        research_clean = enforce_monotonicity(research_clean)
        control_clean = enforce_monotonicity(control_clean)
        log(f"\nAfter monotonicity enforcement:")
        log(f"  Research: {len(research_clean)} points")
        log(f"  Control: {len(control_clean)} points")
    else:
        log(f"\nAfter cleaning (step points only):")
        log(f"  Research: {len(research_clean)} points")
        log(f"  Control: {len(control_clean)} points")

    # Save results
    research_df = pd.DataFrame(research_clean, columns=['Time', 'Survival'])
    control_df = pd.DataFrame(control_clean, columns=['Time', 'Survival'])

    research_df.to_csv(output_dir / "research_curve.csv", index=False)
    control_df.to_csv(output_dir / "control_curve.csv", index=False)

    research_df['Curve'] = 'Research'
    control_df['Curve'] = 'Control'
    combined = pd.concat([research_df, control_df], ignore_index=True)
    combined = combined[['Curve', 'Time', 'Survival']]
    combined.to_csv(output_dir / "both_curves.csv", index=False)

    if overlap_regions:
        overlap_df = pd.DataFrame(overlap_regions, columns=['Time_Start', 'Time_End'])
        overlap_df.to_csv(output_dir / "overlap_regions.csv", index=False)

    cv2.imwrite(str(output_dir / "debug_blue_mask.png"), blue_mask)
    cv2.imwrite(str(output_dir / "debug_red_mask.png"), red_mask)

    log(f"\nSaved to {output_dir}/")

    # Validation
    all_valid = True
    for name, data in [('Research', research_clean), ('Control', control_clean)]:
        if data:
            survivals = [p[1] for p in data]
            is_monotonic = all(survivals[i] >= survivals[i+1] for i in range(len(survivals)-1))
            if not is_monotonic:
                all_valid = False

    return ExtractionResult(
        research_curve=research_clean,
        control_curve=control_clean,
        overlap_regions=overlap_regions,
        output_dir=output_dir,
        is_valid=all_valid,
        research_points=len(research_clean),
        control_points=len(control_clean),
        overlap_count=overlap_count
    )
