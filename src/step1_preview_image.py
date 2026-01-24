#!/usr/bin/env python3
"""
Step 1: Preview Image and Detect Basic Properties

This script loads a Kaplan-Meier plot image and displays basic information
about it, including size, color type, and a preview.

Usage:
    python step1_preview_image.py <image_path>

Output:
    - Displays image information
    - Saves a preview to results/step1_preview.png
"""

import sys
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check dependencies
try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)

try:
    from lib import is_grayscale_image
except ImportError:
    # Fallback implementation
    def is_grayscale_image(img):
        if len(img.shape) == 2:
            return True
        b, g, r = cv2.split(img)
        return np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)


def preview_image(image_path):
    """Preview an image and display its properties."""
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None

    print("=" * 60)
    print("STEP 1: IMAGE PREVIEW")
    print("=" * 60)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None

    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1

    print(f"\nImage: {image_path.name}")
    print(f"  Full path: {image_path.absolute()}")
    print(f"\nProperties:")
    print(f"  Width:    {width} pixels")
    print(f"  Height:   {height} pixels")
    print(f"  Channels: {channels}")

    # Check if grayscale
    grayscale = is_grayscale_image(img)
    print(f"  Type:     {'Grayscale' if grayscale else 'Color'}")

    # Analyze color distribution
    if not grayscale:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        high_sat_pixels = np.sum(saturation > 50)
        total_pixels = width * height
        color_ratio = high_sat_pixels / total_pixels * 100
        print(f"  Colored pixels: {color_ratio:.1f}%")

    # Estimate aspect ratio interpretation
    aspect = width / height
    print(f"\nAspect ratio: {aspect:.2f}")
    if aspect > 1.5:
        print("  (Wide image - may contain multiple panels)")
    elif aspect < 0.8:
        print("  (Tall image - vertical orientation)")
    else:
        print("  (Standard aspect ratio)")

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Save preview with annotations
    preview = img.copy()

    # Add border to show image bounds
    cv2.rectangle(preview, (0, 0), (width-1, height-1), (0, 255, 0), 2)

    # Add text overlay
    text = f"{width}x{height}"
    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)

    preview_path = output_dir / "step1_preview.png"
    cv2.imwrite(str(preview_path), preview)

    print(f"\nPreview saved to: {preview_path}")
    print("\n" + "=" * 60)
    print("NEXT STEP: Run step2_calibrate_axes.py to detect plot axes")
    print("=" * 60)

    return {
        'image': img,
        'width': width,
        'height': height,
        'grayscale': grayscale,
        'path': image_path
    }


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
    print("Searching for image files...\n")
    images = find_images()

    if not images:
        print("No image files found in current directory.")
        return None

    print(f"Found {len(images)} image(s):\n")
    for i, img in enumerate(images, 1):
        size = img.stat().st_size
        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"
        print(f"  [{i}] {img.name}  ({size_str})")
    print(f"\n  [0] Cancel")

    while True:
        try:
            choice = input("\nSelect image number: ").strip()
            if choice == '0':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return images[idx]
            print(f"Please enter 1-{len(images)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None


def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        selected = select_image_interactive()
        if selected is None:
            sys.exit(0)
        image_path = str(selected)

    preview_image(image_path)


if __name__ == "__main__":
    main()
