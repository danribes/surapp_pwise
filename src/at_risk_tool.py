#!/usr/bin/env python3
"""
At-Risk Table Tool - Standalone utility for at-risk data management

This tool provides:
1. OCR extraction with verification/correction
2. Manual entry of at-risk data
3. Import from CSV or PDF files
4. Export in Guyot-compatible format

Usage:
    python at_risk_tool.py extract <image> [--output DIR]
    python at_risk_tool.py manual [--output DIR]
    python at_risk_tool.py import <file.csv|file.pdf> [--output DIR]
    python at_risk_tool.py verify <at_risk_table.csv> [--image IMAGE]
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies
def check_dependencies():
    """Check that required packages are installed."""
    missing = []

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import cv2
import numpy as np
import pandas as pd


# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


def print_header(title: str):
    """Print a styled header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"{Colors.NC}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.NC}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.NC}")


class AtRiskDataManager:
    """Manages at-risk table data with verification and manual entry support."""

    def __init__(self):
        self.groups: Dict[str, Dict[float, int]] = {}
        self.time_points: List[float] = []
        self.source: str = "manual"
        self.confidence: float = 1.0

    def add_group(self, name: str, time_values: Dict[float, int]):
        """Add a group with its at-risk values."""
        self.groups[name] = time_values
        # Update time points
        for t in time_values.keys():
            if t not in self.time_points:
                self.time_points.append(t)
        self.time_points = sorted(self.time_points)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        rows = []
        for group_name, time_data in self.groups.items():
            for time, at_risk in sorted(time_data.items()):
                rows.append({
                    'Group': group_name,
                    'Time': time,
                    'AtRisk': at_risk
                })
        return pd.DataFrame(rows)

    def to_guyot_format(self) -> pd.DataFrame:
        """Convert to Guyot algorithm format with events column."""
        df = self.to_dataframe()
        df['Events'] = 0

        for group in df['Group'].unique():
            mask = df['Group'] == group
            group_df = df[mask].sort_values('Time')
            at_risk = group_df['AtRisk'].values
            events = np.zeros(len(at_risk), dtype=int)

            for i in range(len(at_risk) - 1):
                events[i] = max(0, at_risk[i] - at_risk[i + 1])

            df.loc[mask, 'Events'] = events

        return df

    def save(self, output_path: str):
        """Save to CSV file."""
        df = self.to_guyot_format()
        df.to_csv(output_path, index=False)
        return df

    def display(self):
        """Display current data in a formatted table."""
        if not self.groups:
            print("  No data loaded.")
            return

        # Create display table
        print(f"\n  {Colors.BOLD}Time Points:{Colors.NC} {self.time_points}")
        print()

        # Header
        header = f"  {'Group':<20}"
        for t in self.time_points:
            header += f" {t:>6}"
        print(f"{Colors.BOLD}{header}{Colors.NC}")
        print("  " + "-" * (20 + 7 * len(self.time_points)))

        # Data rows
        for group, values in self.groups.items():
            row = f"  {group:<20}"
            for t in self.time_points:
                val = values.get(t, "-")
                row += f" {val:>6}"
            print(row)

        print()


def extract_from_image(image_path: str, output_dir: str = None, interactive: bool = True) -> Optional[AtRiskDataManager]:
    """Extract at-risk data from an image using OCR."""
    from lib import AxisCalibrator
    from lib.at_risk_extractor import AtRiskExtractor

    print_header("AT-RISK TABLE EXTRACTION")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print_error(f"Could not load image: {image_path}")
        return None

    print(f"  Image: {image_path}")
    print(f"  Size: {img.shape[1]} x {img.shape[0]} pixels")

    # Calibrate to get plot bounds
    print("\n  Detecting plot area...")
    calibrator = AxisCalibrator(img)
    calibration = calibrator.calibrate()

    if calibration is None:
        print_warning("Could not detect plot bounds automatically.")
        # Use default bounds
        h, w = img.shape[:2]
        plot_bounds = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.7))
    else:
        plot_bounds = calibration.plot_rectangle
        refined = calibrator.refine_plot_bounds_from_curves()
        if refined:
            plot_bounds = refined

    print(f"  Plot bounds: x={plot_bounds[0]}, y={plot_bounds[1]}, "
          f"w={plot_bounds[2]}, h={plot_bounds[3]}")

    # Extract at-risk table
    print("\n  Extracting at-risk table (with preprocessing)...")
    extractor = AtRiskExtractor(img, plot_bounds, calibration, debug=True)
    result = extractor.extract()

    if result is None or not result.groups:
        print_warning("No at-risk table detected or could not parse data.")

        if interactive:
            print("\n  Options:")
            print("    [1] Enter data manually")
            print("    [2] Import from CSV file")
            print("    [3] Cancel")

            choice = input("\n  Select option [1-3]: ").strip()

            if choice == '1':
                return manual_entry(output_dir)
            elif choice == '2':
                csv_path = input("  Enter CSV file path: ").strip()
                return import_from_csv(csv_path)
            else:
                return None
        return None

    # Convert to manager
    manager = AtRiskDataManager()
    manager.time_points = result.time_points
    manager.groups = result.groups
    manager.source = "ocr"
    manager.confidence = result.confidence

    print_success(f"Extracted {len(result.groups)} group(s)")
    manager.display()

    # Save debug images
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        extractor.save_debug_image(str(output_path / "debug_at_risk_detection.png"), result)

        # Save preprocessed image for debugging
        table_region, _ = extractor._detect_table_region()
        if table_region is not None:
            extractor.save_preprocessed_image(str(output_path / "debug_preprocessed_table.png"), table_region)

    # Interactive verification
    if interactive:
        manager = verify_and_correct(manager)

    return manager


def verify_and_correct(manager: AtRiskDataManager) -> AtRiskDataManager:
    """Interactive verification and correction of extracted data."""
    print_header("VERIFY & CORRECT DATA")

    manager.display()

    while True:
        print("  Options:")
        print("    [1] Correct a value")
        print("    [2] Add a group")
        print("    [3] Remove a group")
        print("    [4] Change time points")
        print("    [5] Accept and continue")
        print("    [q] Cancel")

        choice = input("\n  Select option: ").strip().lower()

        if choice == '1':
            manager = correct_value(manager)
        elif choice == '2':
            manager = add_group_interactive(manager)
        elif choice == '3':
            manager = remove_group_interactive(manager)
        elif choice == '4':
            manager = change_time_points(manager)
        elif choice == '5' or choice == '':
            break
        elif choice == 'q':
            return manager

        manager.display()

    return manager


def correct_value(manager: AtRiskDataManager) -> AtRiskDataManager:
    """Correct a specific value in the data."""
    groups = list(manager.groups.keys())

    print("\n  Available groups:")
    for i, g in enumerate(groups, 1):
        print(f"    [{i}] {g}")

    try:
        group_idx = int(input("  Select group number: ").strip()) - 1
        if 0 <= group_idx < len(groups):
            group_name = groups[group_idx]

            print(f"\n  Time points for {group_name}:")
            for t in manager.time_points:
                val = manager.groups[group_name].get(t, "N/A")
                print(f"    t={t}: {val}")

            time_str = input("  Enter time point to correct: ").strip()
            time = float(time_str)

            new_val = input(f"  Enter new value for t={time}: ").strip()
            manager.groups[group_name][time] = int(new_val)
            print_success(f"Updated {group_name} at t={time} to {new_val}")
    except (ValueError, IndexError) as e:
        print_error(f"Invalid input: {e}")

    return manager


def add_group_interactive(manager: AtRiskDataManager) -> AtRiskDataManager:
    """Add a new group interactively."""
    print("\n  Adding new group")

    name = input("  Group name: ").strip()
    if not name:
        print_error("Group name cannot be empty")
        return manager

    print(f"  Enter at-risk values for each time point ({manager.time_points}):")
    print("  (Press Enter to skip a time point)")

    values = {}
    for t in manager.time_points:
        val_str = input(f"    t={t}: ").strip()
        if val_str:
            try:
                values[t] = int(val_str)
            except ValueError:
                print_warning(f"Invalid value '{val_str}', skipping")

    if values:
        manager.groups[name] = values
        print_success(f"Added group '{name}' with {len(values)} values")
    else:
        print_warning("No values entered, group not added")

    return manager


def remove_group_interactive(manager: AtRiskDataManager) -> AtRiskDataManager:
    """Remove a group interactively."""
    groups = list(manager.groups.keys())

    print("\n  Available groups:")
    for i, g in enumerate(groups, 1):
        print(f"    [{i}] {g}")

    try:
        group_idx = int(input("  Select group number to remove: ").strip()) - 1
        if 0 <= group_idx < len(groups):
            group_name = groups[group_idx]
            confirm = input(f"  Remove '{group_name}'? [y/N]: ").strip().lower()
            if confirm == 'y':
                del manager.groups[group_name]
                print_success(f"Removed group '{group_name}'")
    except (ValueError, IndexError) as e:
        print_error(f"Invalid input: {e}")

    return manager


def change_time_points(manager: AtRiskDataManager) -> AtRiskDataManager:
    """Change the time points."""
    print(f"\n  Current time points: {manager.time_points}")
    print("  Enter new time points (comma-separated):")

    try:
        new_points = input("  > ").strip()
        time_points = [float(t.strip()) for t in new_points.split(',')]
        time_points = sorted(time_points)

        # Update manager
        old_points = manager.time_points
        manager.time_points = time_points

        # Remap existing data where possible
        for group_name in manager.groups:
            new_values = {}
            for t in time_points:
                if t in manager.groups[group_name]:
                    new_values[t] = manager.groups[group_name][t]
            manager.groups[group_name] = new_values

        print_success(f"Updated time points: {time_points}")
    except ValueError as e:
        print_error(f"Invalid input: {e}")

    return manager


def manual_entry(output_dir: str = None) -> AtRiskDataManager:
    """Manual entry of at-risk data."""
    print_header("MANUAL AT-RISK DATA ENTRY")

    manager = AtRiskDataManager()
    manager.source = "manual"

    # Get time points
    print("  Enter time points (comma-separated, e.g., 0,3,6,9,12,15,18):")
    try:
        time_str = input("  > ").strip()
        manager.time_points = sorted([float(t.strip()) for t in time_str.split(',')])
    except ValueError:
        print_error("Invalid time points format")
        return None

    print(f"\n  Time points: {manager.time_points}")

    # Get groups
    print("\n  Enter group data (empty group name to finish):")

    while True:
        name = input("\n  Group name (or Enter to finish): ").strip()
        if not name:
            break

        print(f"  Enter at-risk values for '{name}':")
        print(f"  (Format: value1,value2,value3,... for times {manager.time_points})")

        try:
            values_str = input("  > ").strip()
            values = [int(v.strip()) for v in values_str.split(',')]

            if len(values) != len(manager.time_points):
                print_warning(f"Expected {len(manager.time_points)} values, got {len(values)}")
                # Pad or truncate
                while len(values) < len(manager.time_points):
                    values.append(0)
                values = values[:len(manager.time_points)]

            manager.groups[name] = dict(zip(manager.time_points, values))
            print_success(f"Added group '{name}'")
        except ValueError as e:
            print_error(f"Invalid values: {e}")

    if not manager.groups:
        print_error("No groups entered")
        return None

    manager.display()
    return manager


def import_from_csv(csv_path: str) -> Optional[AtRiskDataManager]:
    """Import at-risk data from a CSV file."""
    print_header("IMPORT FROM CSV")

    path = Path(csv_path)
    if not path.exists():
        print_error(f"File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"  Loaded: {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")

        manager = AtRiskDataManager()
        manager.source = f"csv:{csv_path}"

        # Try to detect format
        # Expected format: Group, Time, AtRisk (and optionally Events)
        if 'Group' in df.columns and 'Time' in df.columns and 'AtRisk' in df.columns:
            for group in df['Group'].unique():
                group_df = df[df['Group'] == group]
                values = dict(zip(group_df['Time'], group_df['AtRisk'].astype(int)))
                manager.add_group(group, values)

        # Alternative: wide format (Group as rows, Time as columns)
        elif 'Group' in df.columns:
            time_cols = [c for c in df.columns if c != 'Group']
            for _, row in df.iterrows():
                group_name = row['Group']
                values = {}
                for col in time_cols:
                    try:
                        t = float(col)
                        values[t] = int(row[col])
                    except ValueError:
                        continue
                if values:
                    manager.add_group(group_name, values)

        else:
            print_error("Could not detect CSV format. Expected columns: Group, Time, AtRisk")
            return None

        print_success(f"Imported {len(manager.groups)} group(s)")
        manager.display()
        return manager

    except Exception as e:
        print_error(f"Error reading CSV: {e}")
        return None


def import_from_pdf(pdf_path: str) -> Optional[AtRiskDataManager]:
    """Import at-risk data from a PDF file."""
    print_header("IMPORT FROM PDF")

    path = Path(pdf_path)
    if not path.exists():
        print_error(f"File not found: {pdf_path}")
        return None

    # Check for PDF library
    try:
        import pdfplumber
    except ImportError:
        print_error("pdfplumber not installed. Install with: pip install pdfplumber")
        return None

    try:
        tables_found = []

        with pdfplumber.open(pdf_path) as pdf:
            print(f"  PDF has {len(pdf.pages)} page(s)")

            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    print(f"  Page {i+1}: Found {len(tables)} table(s)")
                    tables_found.extend(tables)

        if not tables_found:
            print_warning("No tables found in PDF")
            print("  Would you like to enter data manually? [y/N]: ")
            if input().strip().lower() == 'y':
                return manual_entry()
            return None

        # Display found tables
        print(f"\n  Found {len(tables_found)} table(s):")
        for i, table in enumerate(tables_found):
            print(f"\n  Table {i+1}:")
            for row in table[:5]:  # Show first 5 rows
                print(f"    {row}")
            if len(table) > 5:
                print(f"    ... ({len(table) - 5} more rows)")

        # Ask which table to use
        table_idx = int(input("\n  Select table number to import: ").strip()) - 1

        if 0 <= table_idx < len(tables_found):
            table = tables_found[table_idx]
            return parse_table_to_manager(table)

    except Exception as e:
        print_error(f"Error reading PDF: {e}")
        return None

    return None


def parse_table_to_manager(table: List[List[str]]) -> Optional[AtRiskDataManager]:
    """Parse a table (list of lists) into AtRiskDataManager."""
    if not table or len(table) < 2:
        print_error("Table too small")
        return None

    manager = AtRiskDataManager()
    manager.source = "pdf"

    # Try to detect structure
    # First row might be headers (time points)
    # First column might be group names

    header = table[0]
    print(f"\n  Header row: {header}")

    # Try to extract time points from header
    time_points = []
    time_col_start = 1  # Skip first column (group names)

    for i, cell in enumerate(header[time_col_start:], time_col_start):
        try:
            t = float(cell.strip() if cell else "")
            time_points.append((i, t))
        except (ValueError, AttributeError):
            continue

    if time_points:
        print(f"  Detected time points: {[t for _, t in time_points]}")
        manager.time_points = [t for _, t in time_points]

        # Parse data rows
        for row in table[1:]:
            if not row or not row[0]:
                continue

            group_name = str(row[0]).strip()
            if not group_name or group_name.lower() in ['time', 'months', 'number at risk']:
                continue

            values = {}
            for col_idx, t in time_points:
                if col_idx < len(row) and row[col_idx]:
                    try:
                        val = int(float(str(row[col_idx]).strip()))
                        values[t] = val
                    except (ValueError, AttributeError):
                        continue

            if values:
                manager.add_group(group_name, values)

    if manager.groups:
        print_success(f"Parsed {len(manager.groups)} group(s)")
        manager.display()
        return manager
    else:
        print_error("Could not parse table structure")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="At-Risk Table Tool - Extract, verify, and manage at-risk data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  extract   Extract at-risk table from a KM plot image
  manual    Enter at-risk data manually
  import    Import from CSV or PDF file
  verify    Verify and correct existing at-risk data

Examples:
  python at_risk_tool.py extract my_plot.png
  python at_risk_tool.py manual -o results/
  python at_risk_tool.py import data.csv
  python at_risk_tool.py import paper.pdf
  python at_risk_tool.py verify at_risk_table.csv --image my_plot.png
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract from image')
    extract_parser.add_argument('image', help='Path to KM plot image')
    extract_parser.add_argument('-o', '--output', help='Output directory')
    extract_parser.add_argument('--no-verify', action='store_true',
                               help='Skip interactive verification')

    # Manual entry command
    manual_parser = subparsers.add_parser('manual', help='Manual data entry')
    manual_parser.add_argument('-o', '--output', help='Output directory')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import from file')
    import_parser.add_argument('file', help='CSV or PDF file to import')
    import_parser.add_argument('-o', '--output', help='Output directory')

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify existing data')
    verify_parser.add_argument('csv', help='CSV file with at-risk data')
    verify_parser.add_argument('--image', help='Optional image for reference')
    verify_parser.add_argument('-o', '--output', help='Output directory')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Set up output directory
    if hasattr(args, 'output') and args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/at_risk_{timestamp}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    manager = None

    # Execute command
    if args.command == 'extract':
        manager = extract_from_image(
            args.image,
            output_dir,
            interactive=not args.no_verify
        )

    elif args.command == 'manual':
        manager = manual_entry(output_dir)

    elif args.command == 'import':
        file_path = args.file
        if file_path.lower().endswith('.pdf'):
            manager = import_from_pdf(file_path)
        else:
            manager = import_from_csv(file_path)

        if manager:
            # Offer verification
            verify = input("\n  Verify and correct data? [Y/n]: ").strip().lower()
            if verify != 'n':
                manager = verify_and_correct(manager)

    elif args.command == 'verify':
        manager = import_from_csv(args.csv)
        if manager:
            manager = verify_and_correct(manager)

    # Save results
    if manager and manager.groups:
        output_path = Path(output_dir) / "at_risk_table.csv"
        df = manager.save(str(output_path))
        print_header("EXPORT COMPLETE")
        print(f"  Output: {output_path}")
        print(f"  Groups: {len(manager.groups)}")
        print(f"  Time points: {len(manager.time_points)}")
        print(f"  Source: {manager.source}")

        # Also save JSON for programmatic use
        json_path = Path(output_dir) / "at_risk_table.json"
        with open(json_path, 'w') as f:
            json.dump({
                'groups': {k: {str(t): v for t, v in vals.items()}
                          for k, vals in manager.groups.items()},
                'time_points': manager.time_points,
                'source': manager.source
            }, f, indent=2)

        print(f"  JSON: {json_path}")
    else:
        print_warning("No data to save")
        sys.exit(1)


if __name__ == "__main__":
    main()
