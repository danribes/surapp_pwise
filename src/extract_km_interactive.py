#!/usr/bin/env python3
"""
Kaplan-Meier Curve Extractor - Interactive Mode

This script provides an interactive workflow for extracting KM curves with
optional at-risk table data for the Guyot algorithm.

Workflow:
1. Extract survival curves from image
2. Attempt to detect at-risk table
3. Show detected values and ask user to verify/correct
4. Option to skip at-risk data and use direct parameterization

Usage:
    python extract_km_interactive.py <image_path> [options]
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies
def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    for pkg_name, import_name in [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import local modules
from lib import (
    LineStyleDetector, AxisCalibrator, is_grayscale_image,
    ColorCurveDetector, is_color_image
)

try:
    from lib import AtRiskExtractor, AtRiskData
    AT_RISK_AVAILABLE = True
except ImportError:
    AT_RISK_AVAILABLE = False

try:
    from lib import run_quality_control, format_qc_report, QCSeverity
    QC_AVAILABLE = True
except ImportError:
    QC_AVAILABLE = False


# Terminal colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    NC = '\033[0m'


def print_header(title: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"{Colors.NC}")


def print_step(step: int, total: int, msg: str):
    print(f"\n{Colors.BOLD}[Step {step}/{total}]{Colors.NC} {msg}")


def print_success(msg: str):
    print(f"{Colors.GREEN}  ✓ {msg}{Colors.NC}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}  ⚠ {msg}{Colors.NC}")


def print_error(msg: str):
    print(f"{Colors.RED}  ✗ {msg}{Colors.NC}")


def print_info(msg: str):
    print(f"{Colors.DIM}  {msg}{Colors.NC}")


def display_at_risk_table(groups: Dict[str, Dict[float, int]], time_points: List[float]):
    """Display at-risk table in a formatted way."""
    if not groups or not time_points:
        print("  No data to display.")
        return

    # Header
    header = f"  {'Group':<20}"
    for t in time_points:
        header += f" {t:>6.0f}"
    print(f"\n{Colors.BOLD}{header}{Colors.NC}")
    print("  " + "-" * (20 + 7 * len(time_points)))

    # Data rows
    for group, values in groups.items():
        row = f"  {group:<20}"
        for t in time_points:
            val = values.get(t, "-")
            if val == "-":
                row += f" {Colors.DIM}{val:>6}{Colors.NC}"
            else:
                row += f" {val:>6}"
        print(row)
    print()


def get_user_choice(prompt: str, options: List[str], default: int = 1) -> int:
    """Get user choice from numbered options."""
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        marker = f"{Colors.GREEN}→{Colors.NC}" if i == default else " "
        print(f"  {marker} [{i}] {opt}")

    while True:
        try:
            choice = input(f"\n  Select [1-{len(options)}] (default={default}): ").strip()
            if choice == "":
                return default
            idx = int(choice)
            if 1 <= idx <= len(options):
                return idx
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}")


def manual_at_risk_entry(curve_names: List[str], suggested_times: List[float] = None) -> Tuple[Dict, List[float]]:
    """Interactive manual entry of at-risk data."""
    print_header("MANUAL AT-RISK DATA ENTRY")

    # Get time points
    if suggested_times:
        print(f"  Suggested time points: {suggested_times}")
        use_suggested = input("  Use these time points? [Y/n]: ").strip().lower()
        if use_suggested != 'n':
            time_points = suggested_times
        else:
            time_points = None
    else:
        time_points = None

    if time_points is None:
        print("\n  Enter time points (comma-separated, e.g., 0,3,6,9,12,15,18):")
        try:
            time_str = input("  > ").strip()
            time_points = sorted([float(t.strip()) for t in time_str.split(',')])
        except ValueError:
            print_error("Invalid time points")
            return {}, []

    print(f"\n  Time points: {time_points}")

    # Get values for each group
    groups = {}
    for curve_name in curve_names:
        print(f"\n  Enter at-risk values for '{Colors.BOLD}{curve_name}{Colors.NC}':")
        print(f"  {Colors.DIM}(comma-separated values for times: {time_points}){Colors.NC}")
        print(f"  {Colors.DIM}(or press Enter to skip this group){Colors.NC}")

        values_str = input("  > ").strip()
        if not values_str:
            print_info(f"Skipped {curve_name}")
            continue

        try:
            values = [int(v.strip()) for v in values_str.split(',')]
            if len(values) != len(time_points):
                print_warning(f"Expected {len(time_points)} values, got {len(values)}")
                # Pad or truncate
                while len(values) < len(time_points):
                    values.append(0)
                values = values[:len(time_points)]

            groups[curve_name] = dict(zip(time_points, values))
            print_success(f"Added {curve_name}: {values}")
        except ValueError as e:
            print_error(f"Invalid values: {e}")

    return groups, time_points


def verify_and_correct_at_risk(groups: Dict, time_points: List[float],
                                curve_names: List[str]) -> Tuple[Dict, List[float], bool]:
    """
    Verify detected at-risk data and allow corrections.
    Returns (groups, time_points, use_at_risk_data)
    """
    print_header("AT-RISK DATA VERIFICATION")

    if groups and time_points:
        print(f"  {Colors.GREEN}Detected at-risk table:{Colors.NC}")
        display_at_risk_table(groups, time_points)

        # Show confidence assessment
        print(f"  {Colors.BOLD}Please verify these values against the original image.{Colors.NC}")
        print(f"  {Colors.DIM}OCR may have errors, especially with small text.{Colors.NC}")
    else:
        print_warning("No at-risk table was detected in the image.")

    # Get user choice
    options = [
        "Accept detected values (proceed with Guyot algorithm)",
        "Enter/correct values manually",
        "Skip at-risk data (use direct curve parameterization)"
    ]

    if not groups:
        options[0] = f"{Colors.DIM}Accept detected values (none detected){Colors.NC}"

    choice = get_user_choice("How would you like to proceed?", options,
                             default=2 if not groups else 1)

    if choice == 1 and groups:
        # Accept detected values
        print_success("Using detected at-risk values")
        return groups, time_points, True

    elif choice == 2:
        # Manual entry/correction
        if groups:
            # Show current values and allow correction
            correct_choice = get_user_choice(
                "Correction mode:",
                ["Re-enter all values from scratch", "Correct specific values"],
                default=1
            )

            if correct_choice == 1:
                groups, time_points = manual_at_risk_entry(curve_names, time_points)
            else:
                groups, time_points = correct_specific_values(groups, time_points, curve_names)
        else:
            groups, time_points = manual_at_risk_entry(curve_names)

        if groups:
            print(f"\n  {Colors.GREEN}Updated at-risk table:{Colors.NC}")
            display_at_risk_table(groups, time_points)
            return groups, time_points, True
        else:
            print_warning("No at-risk data entered")
            return {}, [], False

    else:
        # Skip at-risk data
        print_info("Skipping at-risk data - will use direct curve parameterization")
        return {}, [], False


def rename_curves_interactive(curve_names: List[str], curve_colors: List[str]) -> List[str]:
    """
    Allow user to rename curves interactively.
    Returns updated curve names.
    """
    print_header("CURVE NAMING")

    print(f"  {Colors.BOLD}Detected curves:{Colors.NC}")
    for i, (name, color) in enumerate(zip(curve_names, curve_colors), 1):
        print(f"    [{i}] {name} ({color})")

    print(f"\n  {Colors.DIM}You can rename curves to meaningful names (e.g., 'KEYNOTE 024', 'real-world'){Colors.NC}")

    options = [
        "Keep auto-detected names",
        "Rename curves"
    ]

    choice = get_user_choice("Would you like to rename the curves?", options, default=2)

    if choice == 1:
        print_info("Keeping auto-detected names")
        return curve_names

    # Rename curves
    new_names = []
    for i, (old_name, color) in enumerate(zip(curve_names, curve_colors), 1):
        print(f"\n  Curve {i} ({Colors.BOLD}{color}{Colors.NC}), current name: '{old_name}'")
        new_name = input(f"  Enter new name (or press Enter to keep '{old_name}'): ").strip()

        if new_name:
            # Sanitize name for filenames
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in new_name)
            new_names.append(safe_name)
            print_success(f"Renamed to '{safe_name}'")
        else:
            new_names.append(old_name)
            print_info(f"Kept '{old_name}'")

    return new_names


def display_qc_results(qc_report, curves_data: List[Dict]) -> Tuple[List[Dict], bool]:
    """
    Display QC results and allow user to handle issues.
    Returns (updated_curves_data, should_continue)
    """
    print_header("CURVE QUALITY CONTROL")

    if not QC_AVAILABLE:
        print_warning("Quality control module not available")
        return curves_data, True

    # Display summary
    all_passed = qc_report.overall_passed
    passed_count = sum(1 for r in qc_report.curve_results.values() if r.passed)
    total_count = len(qc_report.curve_results)

    if all_passed:
        print_success(f"All {total_count} curves passed quality control")
        return curves_data, True

    print_warning(f"{passed_count}/{total_count} curves passed quality control")
    print()

    # Display detailed issues for each curve
    for curve_name, result in qc_report.curve_results.items():
        if result.passed:
            print(f"  {Colors.GREEN}✓{Colors.NC} {Colors.BOLD}{curve_name}{Colors.NC}: PASSED (score: {result.score:.0%})")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} {Colors.BOLD}{curve_name}{Colors.NC}: FAILED (score: {result.score:.0%})")
            for issue in result.issues:
                severity_color = Colors.RED if issue.severity in [QCSeverity.ERROR, QCSeverity.CRITICAL] else Colors.YELLOW
                print(f"      {severity_color}• {issue.message}{Colors.NC}")
                if issue.suggestion:
                    print(f"        {Colors.DIM}→ {issue.suggestion}{Colors.NC}")

    print()

    # User options
    options = [
        "Continue anyway (use curves as-is)",
        "Remove failed curves and continue",
        "Abort extraction"
    ]

    choice = get_user_choice("How would you like to proceed?", options, default=2)

    if choice == 1:
        print_warning("Continuing with potentially problematic curves")
        return curves_data, True
    elif choice == 2:
        # Filter out failed curves
        valid_curves = []
        for curve_data in curves_data:
            curve_name = curve_data.get('name', '')
            if curve_name in qc_report.curve_results:
                if qc_report.curve_results[curve_name].passed:
                    valid_curves.append(curve_data)
                    print_success(f"Keeping '{curve_name}'")
                else:
                    print_info(f"Removing '{curve_name}'")
            else:
                valid_curves.append(curve_data)

        if not valid_curves:
            print_error("No valid curves remaining!")
            return [], False

        print_success(f"Continuing with {len(valid_curves)} valid curve(s)")
        return valid_curves, True
    else:
        print_info("Extraction aborted by user")
        return [], False


def correct_specific_values(groups: Dict, time_points: List[float],
                           curve_names: List[str]) -> Tuple[Dict, List[float]]:
    """Allow correction of specific values."""
    print("\n  Correction mode - enter 'done' when finished")

    group_names = list(groups.keys())

    while True:
        print(f"\n  Current values:")
        display_at_risk_table(groups, time_points)

        print("  Options:")
        print("    - Enter 'GROUP TIME VALUE' to correct (e.g., 'KEYNOTE 024 12 95')")
        print("    - Enter 'time NEW_TIMES' to change time points (e.g., 'time 0,3,6,9,12')")
        print("    - Enter 'done' to finish")

        cmd = input("\n  > ").strip()

        if cmd.lower() == 'done':
            break

        if cmd.lower().startswith('time '):
            # Change time points
            try:
                new_times = [float(t.strip()) for t in cmd[5:].split(',')]
                new_times = sorted(new_times)

                # Remap values
                for group in groups:
                    old_values = groups[group]
                    new_values = {}
                    for t in new_times:
                        if t in old_values:
                            new_values[t] = old_values[t]
                    groups[group] = new_values

                time_points = new_times
                print_success(f"Updated time points: {time_points}")
            except ValueError:
                print_error("Invalid time points format")
            continue

        # Parse GROUP TIME VALUE
        parts = cmd.rsplit(' ', 2)
        if len(parts) == 3:
            group_name, time_str, value_str = parts
            try:
                time = float(time_str)
                value = int(value_str)

                # Find matching group (case-insensitive partial match)
                matched_group = None
                for g in groups:
                    if group_name.lower() in g.lower():
                        matched_group = g
                        break

                if matched_group:
                    if time not in time_points:
                        print_warning(f"Time {time} not in time points, adding it")
                        time_points = sorted(time_points + [time])

                    groups[matched_group][time] = value
                    print_success(f"Set {matched_group} at t={time} to {value}")
                else:
                    print_error(f"Group '{group_name}' not found")
            except ValueError:
                print_error("Invalid format. Use: GROUP TIME VALUE")
        else:
            print_error("Invalid format. Use: GROUP TIME VALUE")

    return groups, time_points


def save_at_risk_data(groups: Dict, time_points: List[float], output_dir: Path,
                      source: str = "manual") -> pd.DataFrame:
    """Save at-risk data to CSV and JSON with events calculation."""
    import json

    rows = []
    for group_name, time_data in groups.items():
        for time in sorted(time_data.keys()):
            rows.append({
                'Group': group_name,
                'Time': time,
                'AtRisk': time_data[time]
            })

    df = pd.DataFrame(rows)

    # Calculate events
    df['Events'] = 0
    for group in df['Group'].unique():
        mask = df['Group'] == group
        group_df = df[mask].sort_values('Time')
        at_risk = group_df['AtRisk'].values
        events = np.zeros(len(at_risk), dtype=int)
        for i in range(len(at_risk) - 1):
            events[i] = max(0, at_risk[i] - at_risk[i + 1])
        df.loc[mask, 'Events'] = events

    # Save CSV
    df.to_csv(output_dir / "at_risk_table.csv", index=False)

    # Save JSON format
    json_data = {
        "groups": {str(k): {str(t): v for t, v in vals.items()} for k, vals in groups.items()},
        "time_points": time_points,
        "source": source
    }
    with open(output_dir / "at_risk_table.json", 'w') as f:
        json.dump(json_data, f, indent=2)

    return df


def extract_curves_interactive(
    image_path: str,
    output_dir: str = None,
    time_max: float = None,
    expected_curves: int = 2,
    interactive: bool = True
):
    """
    Extract KM curves with interactive at-risk data handling.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print_error(f"Image not found: {image_path}")
        return None

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{image_path.stem}_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("KAPLAN-MEIER CURVE EXTRACTOR")
    print(f"  Image: {image_path}")
    print(f"  Output: {output_dir}")

    total_steps = 7 if interactive else 5

    # Step 1: Load image
    print_step(1, total_steps, "Loading image...")

    img = cv2.imread(str(image_path))
    if img is None:
        print_error(f"Could not load image: {image_path}")
        return None

    height, width = img.shape[:2]
    print_success(f"Loaded {width} x {height} pixels")

    # Step 2: Calibrate axes
    print_step(2, total_steps, "Calibrating axes...")

    calibrator = AxisCalibrator(img)
    calibration = calibrator.calibrate()

    if calibration is None:
        print_warning("Auto-calibration failed, using estimates")
        plot_bounds = (int(width * 0.1), int(height * 0.1),
                      int(width * 0.8), int(height * 0.7))
    else:
        plot_bounds = calibration.plot_rectangle
        refined = calibrator.refine_plot_bounds_from_curves()
        if refined:
            plot_bounds = refined
        print_success(f"Plot area: {plot_bounds}")
        print_info(f"X-axis: {calibration.x_data_range[0]} - {calibration.x_data_range[1]}")

    # Override time_max if specified
    if time_max is not None and calibration is not None:
        calibration.x_data_range = (calibration.x_data_range[0], time_max)
        print_info(f"X-axis (override): 0 - {time_max}")

    # Step 3: Detect curves
    print_step(3, total_steps, "Detecting curves...")

    use_color = is_color_image(img)

    if use_color:
        print_info("Using color-based detection")
        detector = ColorCurveDetector(img, plot_bounds)
        detected_curves = detector.detect_all_curves(
            expected_count=expected_curves,
            debug_dir=str(output_dir)
        )
    else:
        print_info("Using line-style detection")
        detector = LineStyleDetector(img, plot_bounds, filter_reference_lines=True)
        detected_curves = detector.detect_all_curves(
            expected_count=expected_curves,
            debug_dir=str(output_dir)
        )

    if not detected_curves:
        print_error("No curves detected!")
        return None

    # Get auto-generated curve names and colors/styles
    curve_names = []
    curve_colors = []
    for i, curve in enumerate(detected_curves):
        if use_color:
            name = curve.name
            color = name.split('_')[0]  # Extract color from name like "cyan_1"
        else:
            name = f"{curve.style.value}_{i+1}"
            color = curve.style.value
        curve_names.append(name)
        curve_colors.append(color)
        print_success(f"Curve {i+1}: {name} (confidence: {curve.confidence:.2f})")

    # Step 4: Curve naming (interactive only)
    if interactive:
        print_step(4, total_steps, "Naming curves...")
        curve_names = rename_curves_interactive(curve_names, curve_colors)

    # Step 5: At-risk table handling (interactive) / Step 4 (non-interactive)
    step_at_risk = 5 if interactive else 4
    print_step(step_at_risk, total_steps, "Processing at-risk table...")

    at_risk_groups = {}
    at_risk_times = []
    use_at_risk = False

    if AT_RISK_AVAILABLE:
        # Try automatic extraction
        extractor = AtRiskExtractor(img, plot_bounds, calibration, debug=False)
        result = extractor.extract()

        if result and result.groups:
            at_risk_groups = result.groups
            at_risk_times = result.time_points
            print_success(f"Detected {len(result.groups)} group(s) in at-risk table")
        else:
            print_warning("No at-risk table detected automatically")

        # Save debug image
        extractor.save_debug_image(str(output_dir / "debug_at_risk_detection.png"), result)

    if interactive:
        # Interactive verification
        at_risk_groups, at_risk_times, use_at_risk = verify_and_correct_at_risk(
            at_risk_groups, at_risk_times, curve_names
        )
    elif at_risk_groups:
        use_at_risk = True

    # Step 6: Extract coordinates (interactive) / Step 5 (non-interactive)
    step_extract = 6 if interactive else 5
    print_step(step_extract, total_steps, "Extracting curve coordinates...")

    # Set up coordinate conversion
    plot_x, plot_y, plot_w, plot_h = plot_bounds

    if calibration is not None:
        time_min, time_max_cal = calibration.x_data_range
        survival_min, survival_max = calibration.y_data_range
        if time_max is not None:
            time_max_cal = time_max
    else:
        time_min = 0.0
        time_max_cal = time_max if time_max else 10.0
        survival_min, survival_max = 0.0, 1.0

    def pixel_to_coord(px_x, px_y):
        t = time_min + (px_x - plot_x) / plot_w * (time_max_cal - time_min)
        s = survival_max - (px_y - plot_y) / plot_h * (survival_max - survival_min)
        return t, s

    # Extract and clean curve data
    all_curves_data = []

    for i, curve in enumerate(detected_curves):
        raw_points = detector.extract_curve_points(curve, pixel_to_coord)

        if not raw_points:
            print_warning(f"No points for curve {i+1}")
            continue

        clean_points = _clean_curve_data(raw_points)

        curve_data = {
            'name': curve_names[i],
            'style': curve_names[i].split('_')[0] if use_color else curve.style.value,
            'raw_points': raw_points,
            'clean_points': clean_points,
            'confidence': curve.confidence
        }
        all_curves_data.append(curve_data)

        if clean_points:
            print_success(f"{curve_names[i]}: {len(clean_points)} points, "
                         f"time: {clean_points[0][0]:.1f}-{clean_points[-1][0]:.1f}")

    # Step 7 (interactive only): Quality Control
    if interactive and QC_AVAILABLE:
        print_step(7, total_steps, "Quality control check...")

        qc_report = run_quality_control(all_curves_data, expected_time_max=time_max_cal)
        all_curves_data, should_continue = display_qc_results(qc_report, all_curves_data)

        if not should_continue or not all_curves_data:
            print_error("Extraction aborted due to quality issues")
            return None

        # Update curve_names to match remaining curves
        curve_names = [c['name'] for c in all_curves_data]
    elif QC_AVAILABLE:
        # Non-interactive: just report issues but continue
        qc_report = run_quality_control(all_curves_data, expected_time_max=time_max_cal)
        if not qc_report.overall_passed:
            print_warning("Quality control issues detected (run with interactive mode to review)")

    # Save curve CSVs
    for curve_data in all_curves_data:
        df = pd.DataFrame(curve_data['clean_points'], columns=['Time', 'Survival'])
        df.to_csv(output_dir / f"curve_{curve_data['name']}.csv", index=False)

    # Save combined CSV
    combined_rows = []
    for curve_data in all_curves_data:
        for time, survival in curve_data['clean_points']:
            combined_rows.append({
                'Curve': curve_data['name'],
                'Style': curve_data['style'],
                'Time': time,
                'Survival': survival
            })

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(output_dir / "all_curves.csv", index=False)

    # Save at-risk data if available
    if use_at_risk and at_risk_groups:
        at_risk_df = save_at_risk_data(at_risk_groups, at_risk_times, output_dir)
        print_success(f"Saved at-risk table ({len(at_risk_groups)} groups, {len(at_risk_times)} time points)")

    # Generate plots
    _plot_curves(all_curves_data, output_dir / "extracted_curves.png")
    _plot_comparison(img, all_curves_data, plot_bounds, time_max_cal,
                    output_dir / "comparison_overlay.png")

    # Summary
    print_header("EXTRACTION COMPLETE")

    print(f"  {Colors.BOLD}Output directory:{Colors.NC} {output_dir}/")
    print()
    print(f"  {Colors.BOLD}Curve data:{Colors.NC}")
    print(f"    - all_curves.csv")
    for curve_data in all_curves_data:
        print(f"    - curve_{curve_data['name']}.csv")

    if use_at_risk:
        print()
        print(f"  {Colors.BOLD}At-risk data:{Colors.NC}")
        print(f"    - at_risk_table.csv")
        print(f"    {Colors.GREEN}→ Ready for Guyot algorithm{Colors.NC}")
    else:
        print()
        print(f"  {Colors.BOLD}Analysis mode:{Colors.NC}")
        print(f"    {Colors.YELLOW}→ Direct curve parameterization (no at-risk data){Colors.NC}")
        print(f"    {Colors.DIM}  Curve coordinates can be used directly for:{Colors.NC}")
        print(f"    {Colors.DIM}  - Parametric fitting (Weibull, exponential, etc.){Colors.NC}")
        print(f"    {Colors.DIM}  - Non-parametric interpolation{Colors.NC}")
        print(f"    {Colors.DIM}  - Digitized curve analysis{Colors.NC}")

    print()
    print(f"  {Colors.BOLD}Visualizations:{Colors.NC}")
    print(f"    - extracted_curves.png")
    print(f"    - comparison_overlay.png")

    return {
        'curves': all_curves_data,
        'output_dir': str(output_dir),
        'calibration': calibration,
        'at_risk_data': at_risk_groups if use_at_risk else None,
        'at_risk_times': at_risk_times if use_at_risk else None,
        'use_guyot': use_at_risk
    }


def _clean_curve_data(points, tolerance=0.005):
    """Clean curve data - same as in extract_km.py"""
    if not points:
        return []

    sorted_points = sorted(points, key=lambda p: p[0])
    sorted_points = [(max(0.0, t), s) for t, s in sorted_points]

    time_groups = {}
    for t, s in sorted_points:
        t_rounded = round(t, 3)
        if t_rounded not in time_groups:
            time_groups[t_rounded] = []
        time_groups[t_rounded].append(s)

    deduplicated = []
    for t in sorted(time_groups.keys()):
        survivals = time_groups[t]
        median_s = sorted(survivals)[len(survivals) // 2]
        deduplicated.append((t, median_s))

    if not deduplicated:
        return []

    max_detected = max(s for _, s in deduplicated)
    if max_detected < 0.95 and max_detected > 0.1:
        scale_factor = 1.0 / max_detected
        deduplicated = [(t, min(1.0, s * scale_factor)) for t, s in deduplicated]

    first_t, first_s = deduplicated[0]
    if first_t > 0.01:
        deduplicated.insert(0, (0.0, 1.0))
    elif first_s < 0.999:
        deduplicated[0] = (0.0, 1.0)

    monotonic = []
    max_survival = 1.0
    for t, s in deduplicated:
        s = min(s, max_survival)
        monotonic.append((t, s))
        max_survival = s

    cleaned = [monotonic[0]] if monotonic else []
    for i in range(1, len(monotonic)):
        t, s = monotonic[i]
        prev_t, prev_s = cleaned[-1]
        if t - prev_t > tolerance or prev_s - s > tolerance:
            cleaned.append((t, s))

    return cleaned


def _plot_curves(curves_data, output_path):
    """Generate curve visualization."""
    plt.figure(figsize=(10, 6))

    color_map = {
        'red': 'red', 'orange': 'orange', 'yellow': 'gold',
        'green': 'green', 'cyan': 'cyan', 'blue': 'blue',
        'purple': 'purple', 'magenta': 'magenta',
        'solid': 'blue', 'dashed': 'red', 'dotted': 'green'
    }
    linestyles = {'solid': '-', 'dashed': '--', 'dotted': ':'}

    for curve_data in curves_data:
        points = curve_data['clean_points']
        if not points:
            continue

        times = [p[0] for p in points]
        survivals = [p[1] for p in points]
        style = curve_data['style']

        color = color_map.get(style, 'black')
        ls = linestyles.get(style, '-')

        plt.step(times, survivals, where='post', color=color,
                linestyle=ls, linewidth=2, label=curve_data['name'])

    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Extracted Kaplan-Meier Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_comparison(original_img, curves_data, plot_bounds, time_max, output_path):
    """Generate comparison overlay."""
    if len(original_img.shape) == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    plot_x, plot_y, plot_w, plot_h = plot_bounds

    def data_to_pixel(time, survival):
        pixel_x = plot_x + (time / time_max) * plot_w
        pixel_y = plot_y + (1.0 - survival) * plot_h
        return pixel_x, pixel_y

    color_map = {
        'red': '#FF0000', 'orange': '#FF4500', 'cyan': '#00CED1',
        'blue': '#0000FF', 'magenta': '#FF00FF', 'green': '#00FF00',
        'solid': '#0000FF', 'dashed': '#FF0000', 'dotted': '#00AA00'
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb, extent=[0, img_rgb.shape[1], img_rgb.shape[0], 0])

    for curve_data in curves_data:
        points = curve_data['clean_points']
        if not points:
            continue

        style = curve_data['style']
        color = color_map.get(style, '#FF4500')

        pixel_coords = [data_to_pixel(t, s) for t, s in points]
        px_x = [p[0] for p in pixel_coords]
        px_y = [p[1] for p in pixel_coords]

        ax.plot(px_x, px_y, '-', color=color, linewidth=2.5,
                alpha=0.85, label=f"Extracted: {curve_data['name']}")

    ax.set_title('Comparison: Extracted Curves Overlaid on Original')
    ax.legend(loc='upper right')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Kaplan-Meier curves with interactive at-risk data handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_km_interactive.py my_plot.png
  python extract_km_interactive.py my_plot.png --time-max 36
  python extract_km_interactive.py my_plot.png --no-interactive
        """
    )

    parser.add_argument("image", help="Path to KM plot image")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--time-max", type=float, help="Maximum time value")
    parser.add_argument("--curves", type=int, default=2, help="Expected curves")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Skip interactive prompts")

    args = parser.parse_args()

    try:
        result = extract_curves_interactive(
            args.image,
            output_dir=args.output,
            time_max=args.time_max,
            expected_curves=args.curves,
            interactive=not args.no_interactive
        )

        if result is None:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
