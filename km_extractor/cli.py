#!/usr/bin/env python3
"""Command-line interface for KM curve extraction."""

import argparse
import sys
from pathlib import Path

# Handle both installed package and PyInstaller bundled modes
try:
    from km_extractor import __version__
    from km_extractor.extractor import extract_curves_from_image
except ImportError:
    from . import __version__
    from .extractor import extract_curves_from_image


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          KM-EXTRACTOR v{version}                          ║
║     Extract survival data from Kaplan-Meier curves        ║
╚═══════════════════════════════════════════════════════════╝
""".format(version=__version__))


def print_validation_report(result):
    """Print detailed validation report."""
    print("\n" + "="*60)
    print("VERIFICATION & MONOTONICITY CHECK")
    print("="*60)

    for name, data in [('Research', result.research_curve), ('Control', result.control_curve)]:
        if not data:
            print(f"\n{name}: No data extracted")
            continue

        times = [p[0] for p in data]
        survivals = [p[1] for p in data]

        violations = []
        for i in range(len(survivals) - 1):
            if survivals[i] < survivals[i + 1]:
                violations.append((times[i], survivals[i], times[i+1], survivals[i+1]))

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
                repeated = sum(1 for i in range(len(survivals)-1) if survivals[i] == survivals[i+1])
                print(f"  ⚠ STRICTLY DECREASING: No ({repeated} repeated consecutive values)")
        else:
            print(f"  ✗ MONOTONICALLY DECREASING: NO - {len(violations)} violations found!")

    # Overlap summary
    print("\n" + "="*60)
    print("OVERLAP DETECTION SUMMARY")
    print("="*60)
    if result.overlap_regions:
        print(f"  Overlapping points: {result.overlap_count}")
        print(f"  Overlap regions: {len(result.overlap_regions)}")
        for i, (t_start, t_end) in enumerate(result.overlap_regions, 1):
            duration = t_end - t_start
            if duration < 0.1:
                print(f"    {i}. t = {t_start:.2f} years (point overlap)")
            else:
                print(f"    {i}. t = {t_start:.2f} - {t_end:.2f} years ({duration:.2f} years)")
        print("\n  Note: In overlap regions, both curves have identical survival values")
    else:
        print("  No overlapping regions detected")

    # Overall result
    print("\n" + "="*60)
    if result.is_valid:
        print("✓ ALL CURVES VALID - Data is suitable for survival analysis")
    else:
        print("✗ VALIDATION FAILED - Some curves have issues")
    print("="*60)


def print_curve_sample(result):
    """Print sample of extracted curves."""
    print("\n" + "="*60)
    print("RESEARCH CURVE (first 10 and last 5 points)")
    print("="*60)
    for t, s in result.research_curve[:10]:
        print(f"  Time: {t:6.2f}  Survival: {s:.4f}")
    if len(result.research_curve) > 15:
        print("  ...")
        for t, s in result.research_curve[-5:]:
            print(f"  Time: {t:6.2f}  Survival: {s:.4f}")

    print("\n" + "="*60)
    print("CONTROL CURVE (first 10 and last 5 points)")
    print("="*60)
    for t, s in result.control_curve[:10]:
        print(f"  Time: {t:6.2f}  Survival: {s:.4f}")
    if len(result.control_curve) > 15:
        print("  ...")
        for t, s in result.control_curve[-5:]:
            print(f"  Time: {t:6.2f}  Survival: {s:.4f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='km-extract',
        description='Extract survival data from Kaplan-Meier curve images',
        epilog='Example: km-extract figure1.png --time-max 10'
    )

    parser.add_argument(
        'image',
        help='Path to the KM curve image (PNG, JPG, etc.)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output directory (default: results/<image_name>_<timestamp>/)',
        default=None
    )

    parser.add_argument(
        '-t', '--time-max',
        type=float,
        default=12.0,
        help='Maximum time value on X-axis (default: 12.0)'
    )

    parser.add_argument(
        '--overlap-threshold',
        type=int,
        default=5,
        help='Pixel threshold for curve overlap detection (default: 5)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip validation report'
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print_banner()

    try:
        result = extract_curves_from_image(
            image_path=str(image_path),
            output_dir=args.output,
            time_max=args.time_max,
            overlap_threshold=args.overlap_threshold,
            quiet=args.quiet
        )

        if not args.quiet and not args.no_report:
            print_validation_report(result)
            print_curve_sample(result)

        if not args.quiet:
            print(f"\n✓ Extraction complete!")
            print(f"  Output: {result.output_dir}/")
            print(f"  Research: {result.research_points} points")
            print(f"  Control: {result.control_points} points")

        sys.exit(0 if result.is_valid else 1)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
