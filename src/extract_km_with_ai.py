#!/usr/bin/env python3
"""
Kaplan-Meier Curve Extractor with AI Options.

This script offers multiple extraction modes:
1. Standard pixel-based detection (fast, good for high-quality images)
2. AI-assisted detection (uses AI to validate/improve results)
3. Pure AI extraction (uses AI to read values directly from image)

Usage:
    python extract_km_with_ai.py <image_path> [--mode MODE] [--time-max TIME]

Modes:
    standard  - Pixel-based detection only (default, fastest, most accurate)
    assisted  - Pixel detection with AI validation/correction (recommended)
    ai-only   - Pure AI extraction (EXPERIMENTAL - low accuracy, slow)
                Note: AI vision models are NOT reliable for precise numerical
                reading. Use this only for rough estimates or debugging.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import pandas as pd
import matplotlib.pyplot as plt


def extract_with_ai_only(image_path, output_dir, time_max=24.0, time_step=3.0, quiet=False):
    """Extract curves using pure AI vision (no pixel detection).

    This method asks the AI to directly read survival values from the image
    at specified time points.
    """
    from lib.ai_service import get_ai_service

    service = get_ai_service()
    if not service:
        print("ERROR: AI service not available")
        return None

    if not quiet:
        print("=" * 60)
        print("AI-ONLY KAPLAN-MEIER CURVE EXTRACTION")
        print("=" * 60)
        print(f"\nInput image: {image_path}")
        print(f"Output directory: {output_dir}")
        print(f"Time range: 0 - {time_max}")
        print(f"Time step: {time_step}")
        print(f"Model: {service.config.model}")
        print("\nNote: AI extraction may take several minutes on CPU...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run AI extraction
    if not quiet:
        print("\n[Step 1/3] Running AI extraction...")

    result = service.extract_curves(
        image_path,
        time_max=time_max,
        time_step=time_step,
        quiet=quiet
    )

    if result is None or not result.curves:
        print("ERROR: AI extraction failed")
        return None

    if not quiet:
        print(f"\n[Step 2/3] Processing {len(result.curves)} curves...")

    all_curves_data = []
    for curve in result.curves:
        curve_data = {
            'name': curve.name.replace(' ', '_'),
            'style': curve.color,
            'clean_points': curve.points,
            'confidence': curve.confidence
        }
        all_curves_data.append(curve_data)

        if not quiet:
            print(f"  {curve.name}: {len(curve.points)} points, confidence: {curve.confidence:.0%}")

    # Save CSV files
    if not quiet:
        print("\n[Step 3/3] Saving results...")

    for curve_data in all_curves_data:
        df = pd.DataFrame(curve_data['clean_points'], columns=['Time', 'Survival'])
        csv_path = output_dir / f"curve_{curve_data['name']}.csv"
        df.to_csv(csv_path, index=False)

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

    # Save raw AI response
    with open(output_dir / "ai_raw_response.txt", 'w') as f:
        f.write(result.raw_response)

    # Generate visualization
    _plot_ai_curves(all_curves_data, output_dir / "extracted_curves.png", time_max)

    # Generate overlay
    img = cv2.imread(str(image_path))
    _plot_ai_overlay(img, all_curves_data, output_dir / "comparison_overlay.png", time_max)

    if not quiet:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nOutput files saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  - all_curves.csv        : Combined data from all curves")
        for curve_data in all_curves_data:
            print(f"  - curve_{curve_data['name']}.csv : {curve_data['style']} curve data")
        print("  - extracted_curves.png  : Visualization plot")
        print("  - comparison_overlay.png: Original with extracted curves overlay")
        print("  - ai_raw_response.txt   : Raw AI responses")
        print(f"\nOverall confidence: {result.confidence:.0%}")

    return {
        'curves': all_curves_data,
        'output_dir': str(output_dir),
        'confidence': result.confidence
    }


def _plot_ai_curves(curves_data, output_path, time_max):
    """Generate a plot of AI-extracted curves."""
    plt.figure(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    for i, curve_data in enumerate(curves_data):
        points = curve_data['clean_points']
        if not points:
            continue

        times = [p[0] for p in points]
        survivals = [p[1] for p in points]
        color = colors[i % len(colors)]

        plt.step(times, survivals,
                where='post',
                color=color,
                linewidth=2,
                label=f"{curve_data['name']} (AI)")

    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('AI-Extracted Kaplan-Meier Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, time_max)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_ai_overlay(img, curves_data, output_path, time_max):
    """Generate comparison overlay for AI extraction."""
    # Convert BGR to RGB
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    height, width = img.shape[:2]

    # Estimate plot bounds (assume standard layout)
    plot_x = int(width * 0.15)
    plot_w = int(width * 0.75)
    plot_y = int(height * 0.1)
    plot_h = int(height * 0.6)

    colors = ['#0000FF', '#FF0000', '#00AA00', '#FF8800', '#8800FF', '#00CCCC']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb, extent=[0, width, height, 0])

    for i, curve_data in enumerate(curves_data):
        points = curve_data['clean_points']
        if not points:
            continue

        color = colors[i % len(colors)]

        # Convert to pixel coordinates
        pixel_coords = []
        for t, s in points:
            px_x = plot_x + (t / time_max) * plot_w
            px_y = plot_y + (1.0 - s) * plot_h
            pixel_coords.append((px_x, px_y))

        # Create step function
        step_x = []
        step_y = []
        for j, (px, py) in enumerate(pixel_coords):
            if j > 0:
                step_x.append(px)
                step_y.append(step_y[-1])
            step_x.append(px)
            step_y.append(py)

        ax.plot(step_x, step_y, '-', color=color, linewidth=2.5,
                alpha=0.85, label=f"AI: {curve_data['name']}")

    ax.set_title('AI-Extracted Curves Overlaid on Original', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract KM curves with multiple AI options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  standard  - Pixel-based detection only (fast)
  assisted  - Pixel detection with AI validation
  ai-only   - Pure AI extraction (slowest, best for low-quality images)

Examples:
  python extract_km_with_ai.py image.png                    # Standard mode
  python extract_km_with_ai.py image.png --mode assisted    # AI-assisted
  python extract_km_with_ai.py image.png --mode ai-only     # Pure AI
        """
    )

    parser.add_argument("image", help="Path to the KM plot image")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--time-max", type=float, default=24.0, help="Maximum time value")
    parser.add_argument("--time-step", type=float, default=3.0,
                       help="Time step for AI extraction (smaller = more detail, slower)")
    parser.add_argument("--mode", choices=['standard', 'assisted', 'ai-only'],
                       default='standard', help="Extraction mode")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    # Generate output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(args.image).stem
        args.output = f"results/{image_name}_{args.mode}_{timestamp}"

    if args.mode == 'ai-only':
        print("\n" + "!" * 60)
        print("WARNING: ai-only mode is EXPERIMENTAL")
        print("AI vision models are NOT reliable for precise numerical reading.")
        print("Values may be inaccurate. Use 'standard' or 'assisted' mode")
        print("for accurate data extraction.")
        print("!" * 60 + "\n")
        result = extract_with_ai_only(
            args.image,
            args.output,
            time_max=args.time_max,
            time_step=args.time_step,
            quiet=args.quiet
        )
    elif args.mode == 'assisted':
        # Use standard extraction with AI assistance
        from extract_km import extract_km_curves
        result = extract_km_curves(
            args.image,
            output_dir=args.output,
            time_max=args.time_max,
            show_progress=not args.quiet,
            use_ai=True
        )
    else:
        # Standard extraction (no AI)
        from extract_km import extract_km_curves
        result = extract_km_curves(
            args.image,
            output_dir=args.output,
            time_max=args.time_max,
            show_progress=not args.quiet,
            use_ai=False
        )

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
