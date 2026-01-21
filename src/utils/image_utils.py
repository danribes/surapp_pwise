"""Image preprocessing utilities."""

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    # Convert to RGB if needed
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to numpy array and swap RGB to BGR
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    return cv_image


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to PIL Image."""
    if len(cv_image.shape) == 2:
        # Grayscale
        return Image.fromarray(cv_image)
    else:
        # BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)


def preprocess_for_curve_detection(image: np.ndarray) -> np.ndarray:
    """Preprocess image for curve detection.

    Args:
        image: Input image in BGR format

    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    return filtered


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess image for OCR.

    Args:
        image: Input image in BGR format

    Returns:
        Preprocessed binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return binary


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE.

    Args:
        image: Input image in BGR format

    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge channels
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def detect_edges(gray_image: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """Detect edges using Canny edge detector.

    Args:
        gray_image: Grayscale input image
        threshold1: First threshold for hysteresis
        threshold2: Second threshold for hysteresis

    Returns:
        Binary edge image
    """
    return cv2.Canny(gray_image, threshold1, threshold2)


def find_contours(binary_image: np.ndarray) -> list:
    """Find contours in a binary image.

    Args:
        binary_image: Binary input image

    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def crop_region(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop a region from an image.

    Args:
        image: Input image
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of region
        height: Height of region

    Returns:
        Cropped image region
    """
    return image[y:y+height, x:x+width].copy()


def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """Resize image maintaining aspect ratio.

    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / h
        new_width = int(w * ratio)
        new_height = height
    elif height is None:
        ratio = width / w
        new_width = width
        new_height = int(h * ratio)
    else:
        new_width = width
        new_height = height

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def draw_point(image: np.ndarray, point: tuple, color: tuple = (0, 255, 0), radius: int = 5) -> np.ndarray:
    """Draw a point on an image.

    Args:
        image: Input image
        point: (x, y) coordinates
        color: BGR color tuple
        radius: Circle radius

    Returns:
        Image with point drawn
    """
    output = image.copy()
    cv2.circle(output, point, radius, color, -1)
    return output


def draw_line(image: np.ndarray, pt1: tuple, pt2: tuple, color: tuple = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    """Draw a line on an image.

    Args:
        image: Input image
        pt1: Start point (x, y)
        pt2: End point (x, y)
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with line drawn
    """
    output = image.copy()
    cv2.line(output, pt1, pt2, color, thickness)
    return output


def draw_rectangle(image: np.ndarray, pt1: tuple, pt2: tuple, color: tuple = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draw a rectangle on an image.

    Args:
        image: Input image
        pt1: Top-left corner (x, y)
        pt2: Bottom-right corner (x, y)
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with rectangle drawn
    """
    output = image.copy()
    cv2.rectangle(output, pt1, pt2, color, thickness)
    return output
