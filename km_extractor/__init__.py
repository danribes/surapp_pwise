"""KM Extractor - Extract survival data from Kaplan-Meier curve images."""

__version__ = "1.0.0"
__author__ = "Dan Ribes"

from .extractor import extract_curves_from_image

__all__ = ["extract_curves_from_image", "__version__"]
