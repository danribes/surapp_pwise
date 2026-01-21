#!/usr/bin/env python3
"""
Cross-platform build script for km-extractor.

Usage:
    python build.py          # Build for current platform
    python build.py --all    # Build for all platforms (requires each OS)

Requirements:
    pip install pyinstaller
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def get_platform():
    """Detect current platform."""
    system = platform.system().lower()
    if system == 'linux':
        return 'linux'
    elif system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def build(spec_file: str, output_name: str):
    """Run PyInstaller build."""
    print(f"\n{'='*60}")
    print(f"Building {output_name}...")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean',
        '--noconfirm',
        spec_file
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Build successful: dist/{output_name}")
    else:
        print(f"\n✗ Build failed for {output_name}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Build km-extractor executable')
    parser.add_argument('--platform', choices=['linux', 'macos', 'windows'],
                       help='Target platform (default: current)')
    args = parser.parse_args()

    # Determine platform
    if args.platform:
        target = args.platform
    else:
        target = get_platform()

    print(f"Building for: {target}")

    # Select spec file
    spec_files = {
        'linux': 'build_linux.spec',
        'macos': 'build_macos.spec',
        'windows': 'build_windows.spec'
    }

    spec_file = spec_files.get(target)
    if not spec_file:
        print(f"Error: No spec file for platform: {target}")
        sys.exit(1)

    if not Path(spec_file).exists():
        print(f"Error: Spec file not found: {spec_file}")
        sys.exit(1)

    # Build
    output_name = 'km-extract' + ('.exe' if target == 'windows' else '')
    build(spec_file, output_name)

    print(f"""
{'='*60}
BUILD COMPLETE
{'='*60}

Executable: dist/km-extract{'.exe' if target == 'windows' else ''}

Usage:
  ./dist/km-extract image.png
  ./dist/km-extract image.jpg --time-max 10 --output ./my_results

The executable is standalone and can be distributed without Python.
""")


if __name__ == '__main__':
    main()
