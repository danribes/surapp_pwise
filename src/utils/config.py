"""Application configuration constants."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """Application configuration settings."""

    # Application info
    APP_NAME: str = "KM Curve Extractor"
    APP_VERSION: str = "1.0.0"

    # PDF processing
    PDF_DPI: int = 300

    # Image processing
    CANNY_THRESHOLD1: int = 50
    CANNY_THRESHOLD2: int = 150
    HOUGH_THRESHOLD: int = 50
    HOUGH_MIN_LINE_LENGTH: int = 20
    HOUGH_MAX_LINE_GAP: int = 10
    LINE_ANGLE_TOLERANCE: float = 5.0  # degrees
    ENDPOINT_TOLERANCE: int = 10  # pixels

    # Curve detection
    MIN_STEP_HEIGHT: int = 5  # minimum Y change for a valid step
    SAMPLING_INTERVAL: int = 5  # pixels between sampled points

    # OCR settings
    OCR_CONFIG: str = "--psm 6"  # Assume uniform block of text

    # GUI settings
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800
    THUMBNAIL_SIZE: tuple = (150, 200)

    # Export settings
    DEFAULT_DECIMAL_PLACES: int = 4

    # Colors for visualization
    CURVE_COLOR: tuple = (255, 0, 0)  # Red for detected curve
    CALIBRATION_COLOR: tuple = (0, 255, 0)  # Green for calibration points
    AXIS_COLOR: tuple = (0, 0, 255)  # Blue for detected axes


# Global configuration instance
config = AppConfig()


def get_assets_path() -> Path:
    """Get the path to the assets directory."""
    return Path(__file__).parent.parent.parent / "assets"


def get_tesseract_path() -> str | None:
    """Get the path to Tesseract OCR executable if bundled."""
    import platform
    import shutil

    # Try to find tesseract in system path
    tesseract = shutil.which("tesseract")
    if tesseract:
        return tesseract

    # Platform-specific default locations
    if platform.system() == "Windows":
        default_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in default_paths:
            if Path(path).exists():
                return path

    return None
