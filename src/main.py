#!/usr/bin/env python3
"""
Kaplan-Meier Curve Extractor

A desktop application for extracting survival data from Kaplan-Meier curves in PDFs.

Usage:
    python -m src.main
    or
    python src/main.py
"""

import sys
from pathlib import Path

# Add src directory to path for imports when running directly
if __name__ == "__main__":
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir.parent))


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []

    try:
        import customtkinter
    except ImportError:
        missing.append("customtkinter")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        missing.append("PyMuPDF")

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
        import scipy
    except ImportError:
        missing.append("scipy")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def check_optional_dependencies():
    """Check optional dependencies and print warnings."""
    warnings = []

    try:
        import pytesseract
    except ImportError:
        warnings.append(
            "pytesseract not installed - OCR features will be disabled"
        )

    try:
        import matplotlib
    except ImportError:
        warnings.append(
            "matplotlib not installed - plot preview will be disabled"
        )

    try:
        import openpyxl
    except ImportError:
        warnings.append(
            "openpyxl not installed - Excel export will be disabled"
        )

    if warnings:
        print("Optional dependencies missing:")
        for warning in warnings:
            print(f"  - {warning}")
        print()


def main():
    """Main entry point."""
    print("KM Curve Extractor - Starting...")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    check_optional_dependencies()

    # Import and run app
    try:
        from src.gui.app import run_app
        run_app()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
