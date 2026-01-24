#!/usr/bin/env python3
"""
AI-Enhanced Kaplan-Meier Curve Extractor

This script extends the standard extractor with AI-powered validation
using Ollama with llama3.2-vision. The AI compares the original image
with the extracted curves overlay to validate accuracy and suggest
parameter adjustments if needed.

Usage:
    python extract_km_ai.py <image_path> [--time-max TIME] [--curves N] [--validate]

Example:
    python extract_km_ai.py my_km_plot.png --time-max 40 --validate

Requirements:
    - Ollama running with llama3.2-vision model
    - Or use with docker-compose.ai.yml for containerized setup
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

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

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

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
import matplotlib.pyplot as plt

# Import local modules
try:
    from lib import (
        LineStyleDetector, AxisCalibrator, is_grayscale_image,
        ColorCurveDetector, is_color_image
    )
    from lib.ai_validator import AIValidator, check_ollama_status
    from lib.ai_config import AIConfig, ExtractionParameters, ValidationResult
except ImportError as e:
    print(f"ERROR: Could not import local modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def extract_km_curves_with_params(
    image_path: str,
    params: ExtractionParameters,
    output_dir: str,
    time_max: float = None,
    expected_curves: int = 2,
    show_progress: bool = True
) -> str:
    """
    Extract KM curves with specific parameters.

    This is a wrapper around the standard extraction that accepts
    ExtractionParameters for AI-guided tuning.

    Returns:
        Path to the comparison overlay image, or None if extraction failed.
    """
    # Import the standard extraction function
    from extract_km import extract_km_curves

    # Run standard extraction
    # Note: In a full implementation, params would modify the extraction behavior
    # For now, we use the standard extraction and validate the results
    result = extract_km_curves(
        image_path,
        output_dir=output_dir,
        time_max=time_max,
        expected_curves=expected_curves,
        show_progress=show_progress
    )

    if result is None:
        return None

    # Return path to comparison overlay for validation
    overlay_path = Path(output_dir) / "comparison_overlay.png"
    if overlay_path.exists():
        return str(overlay_path)

    return None


def extract_with_ai_validation(
    image_path: str,
    output_dir: str = None,
    time_max: float = None,
    expected_curves: int = 2,
    validate: bool = True,
    auto_retry: bool = True,
    show_progress: bool = True
) -> dict:
    """
    Extract KM curves with optional AI validation.

    Args:
        image_path: Path to the KM plot image
        output_dir: Output directory
        time_max: Maximum time value on X-axis
        expected_curves: Expected number of curves
        validate: Whether to run AI validation
        auto_retry: Whether to automatically retry with adjusted parameters
        show_progress: Print progress messages

    Returns:
        Dictionary with extraction results and validation info
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{image_path.stem}_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize AI validator
    ai_config = AIConfig.from_environment()
    validator = AIValidator(ai_config)

    # Check AI availability
    ai_available = validator.is_available if validate else False

    if show_progress:
        print("=" * 60)
        print("AI-ENHANCED KAPLAN-MEIER CURVE EXTRACTOR")
        print("=" * 60)
        print(f"\nInput image: {image_path}")
        print(f"Output directory: {output_dir}")
        print(f"AI Validation: {'Enabled' if ai_available else 'Disabled'}")

        if validate and not ai_available:
            status = check_ollama_status(ai_config.host)
            if status['error']:
                print(f"  (Reason: {status['error']})")
            print("  Continuing without AI validation...")

    # Run extraction
    from extract_km import extract_km_curves

    result = extract_km_curves(
        str(image_path),
        output_dir=str(output_dir),
        time_max=time_max,
        expected_curves=expected_curves,
        show_progress=show_progress
    )

    if result is None:
        return {'success': False, 'error': 'Extraction failed', 'validation': None}

    # Run AI validation if available
    validation_result = None

    if ai_available:
        overlay_path = output_dir / "comparison_overlay.png"

        if overlay_path.exists():
            if show_progress:
                print("\n" + "-" * 60)
                print("AI VALIDATION")
                print("-" * 60)

            validation_result = validator.validate(
                str(image_path),
                str(overlay_path),
                quiet=not show_progress
            )

            if validation_result:
                # Save validation report
                report_path = output_dir / "ai_validation_report.txt"
                with open(report_path, 'w') as f:
                    f.write("AI Validation Report\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Model: {ai_config.model}\n")
                    f.write(f"Image: {image_path.name}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                    f.write(str(validation_result) + "\n\n")
                    f.write("Raw Response:\n")
                    f.write("-" * 40 + "\n")
                    f.write(validation_result.raw_response)

                if show_progress:
                    print(f"\n  Validation report saved to: {report_path}")

    # Build result
    final_result = {
        'success': True,
        'curves': result['curves'],
        'output_dir': str(output_dir),
        'calibration': result.get('calibration'),
        'validation': None
    }

    if validation_result:
        final_result['validation'] = {
            'match': validation_result.match,
            'confidence': validation_result.confidence,
            'is_valid': validation_result.is_valid,
            'issues': validation_result.issues,
            'suggestions': validation_result.suggestions
        }

    if show_progress:
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)

        if validation_result:
            status = "VALIDATED" if validation_result.is_valid else "NEEDS REVIEW"
            print(f"\nValidation Status: {status}")
            print(f"Confidence: {validation_result.confidence:.1%}")

    return final_result


def check_ai_status(host: str = None):
    """Check and display AI service status."""
    if host is None:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    print("=" * 60)
    print("AI SERVICE STATUS")
    print("=" * 60)

    status = check_ollama_status(host)

    print(f"\nOllama Host: {status['host']}")
    print(f"Available: {'Yes' if status['available'] else 'No'}")

    if status['error']:
        print(f"Error: {status['error']}")

    if status['models']:
        print(f"\nInstalled Models:")
        for model in status['models']:
            print(f"  - {model}")
    else:
        print("\nNo models installed.")
        print("\nTo install llama3.2-vision:")
        print("  ollama pull llama3.2-vision")

    # Check for required model
    config = AIConfig.from_environment()
    required_model = config.model.split(':')[0]

    model_names = [m.split(':')[0] for m in status.get('models', [])]

    if required_model in model_names:
        print(f"\nRequired model '{required_model}': Installed")
    else:
        print(f"\nRequired model '{required_model}': NOT INSTALLED")
        print(f"  Run: ollama pull {config.model}")


def find_images(directory="."):
    """Find image files in the specified directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    images = []

    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in image_extensions:
            images.append(path)

    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return images


def select_image_interactive():
    """Let user select an image from the current directory."""
    print("=" * 60)
    print("AI-ENHANCED KAPLAN-MEIER CURVE EXTRACTOR")
    print("=" * 60)
    print("\nSearching for image files...")

    images = find_images()

    if not images:
        print("\nNo image files found in current directory.")
        print("Supported formats: PNG, JPG, JPEG, BMP, TIFF, GIF")
        print("\nUsage: python extract_km_ai.py <image_path>")
        return None

    print(f"\nFound {len(images)} image(s):\n")

    for i, img in enumerate(images, 1):
        size = img.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  [{i}] {img.name}  ({size_str})")

    print(f"\n  [0] Cancel")

    while True:
        try:
            choice = input("\nSelect image number: ").strip()

            if choice == '0' or choice.lower() == 'q':
                print("Cancelled.")
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return images[idx]
            else:
                print(f"Please enter a number between 1 and {len(images)}")

        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Enhanced Kaplan-Meier survival curve extractor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_km_ai.py                           # Interactive mode
  python extract_km_ai.py my_plot.png               # Extract without validation
  python extract_km_ai.py my_plot.png --validate    # Extract with AI validation
  python extract_km_ai.py --status                  # Check AI service status

AI Validation:
  Requires Ollama running with llama3.2-vision model.
  Install: ollama pull llama3.2-vision

  Or use Docker:
  docker-compose -f docker-compose.yml -f docker-compose.ai.yml up -d
        """
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to the Kaplan-Meier plot image"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: results/<image>_<timestamp>)"
    )

    parser.add_argument(
        "--time-max",
        type=float,
        help="Maximum time value on X-axis"
    )

    parser.add_argument(
        "--curves",
        type=int,
        default=2,
        help="Expected number of curves (default: 2)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable AI validation of extraction results"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Check AI service status and exit"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Check status mode
    if args.status:
        check_ai_status()
        sys.exit(0)

    # If no image provided, use interactive selection
    if args.image is None:
        selected = select_image_interactive()
        if selected is None:
            sys.exit(0)
        image_path = str(selected)
    else:
        image_path = args.image

    try:
        result = extract_with_ai_validation(
            image_path,
            output_dir=args.output,
            time_max=args.time_max,
            expected_curves=args.curves,
            validate=args.validate,
            show_progress=not args.quiet
        )

        if not result['success']:
            print(f"ERROR: {result.get('error', 'Unknown error')}")
            sys.exit(1)

        # Exit with appropriate code based on validation
        if result.get('validation'):
            if result['validation']['is_valid']:
                sys.exit(0)
            else:
                sys.exit(2)  # Extraction succeeded but validation flagged issues

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
