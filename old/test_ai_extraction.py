#!/usr/bin/env python3
"""
Test AI-based curve point extraction.

This script tests using AI vision to directly read survival values
from Kaplan-Meier curves at specified time points.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ai_service import get_ai_service


def main():
    print("=" * 60)
    print("AI CURVE EXTRACTION TEST")
    print("=" * 60)

    # Get AI service
    service = get_ai_service()
    if not service:
        print("AI service not available")
        return

    print(f"Model: {service.config.model}")
    print(f"Capabilities: {service.capabilities}")

    # Test image
    image_path = "input/Kaplan-Meier NSQ OS.webp"
    print(f"\nImage: {image_path}")

    # Extract curves with coarse time steps first (faster)
    print("\n--- Extracting curves (time step = 3) ---")
    print("(This may take a few minutes on CPU...)")

    result = service.extract_curves(
        image_path,
        time_max=24.0,
        time_step=3.0,  # Coarse steps for faster testing
        quiet=False
    )

    if result is None:
        print("Extraction failed")
        return

    print(f"\n--- Results ---")
    print(f"Curves found: {len(result.curves)}")
    print(f"Overall confidence: {result.confidence:.0%}")

    for curve in result.curves:
        print(f"\n  {curve.name} ({curve.color}):")
        print(f"    Points: {len(curve.points)}")
        print(f"    Confidence: {curve.confidence:.0%}")
        print(f"    Data:")
        for t, s in curve.points:
            print(f"      t={t:5.1f}: survival={s:.3f}")

    # Save to CSV
    import pandas as pd
    output_dir = Path("results/ai_extraction_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    for curve in result.curves:
        df = pd.DataFrame(curve.points, columns=['Time', 'Survival'])
        csv_path = output_dir / f"ai_curve_{curve.name.replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    # Save raw response
    with open(output_dir / "ai_raw_response.txt", 'w') as f:
        f.write(result.raw_response)
    print(f"Saved: {output_dir / 'ai_raw_response.txt'}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
