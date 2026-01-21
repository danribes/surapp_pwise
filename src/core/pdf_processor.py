"""PDF to image conversion module."""

from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from PIL import Image

from ..utils.config import config


class PDFProcessor:
    """Handles PDF loading and conversion to images."""

    def __init__(self, pdf_path: str | Path):
        """Initialize PDF processor.

        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        self._document = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self):
        """Open the PDF document."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self._document = fitz.open(str(self.pdf_path))

    def close(self):
        """Close the PDF document."""
        if self._document:
            self._document.close()
            self._document = None

    @property
    def page_count(self) -> int:
        """Get the number of pages in the PDF."""
        if self._document is None:
            raise RuntimeError("PDF document is not open")
        return len(self._document)

    def get_page_image(self, page_num: int, dpi: int = None) -> Image.Image:
        """Convert a PDF page to a PIL Image.

        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution in DPI (default: from config)

        Returns:
            PIL Image of the page
        """
        if self._document is None:
            raise RuntimeError("PDF document is not open")

        if page_num < 0 or page_num >= len(self._document):
            raise ValueError(f"Invalid page number: {page_num}")

        if dpi is None:
            dpi = config.PDF_DPI

        # Calculate zoom factor from DPI
        # PDF default is 72 DPI
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        page = self._document[page_num]
        pixmap = page.get_pixmap(matrix=matrix)

        # Convert to PIL Image
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        return image

    def get_all_pages(self, dpi: int = None) -> Generator[Image.Image, None, None]:
        """Generator that yields all pages as PIL Images.

        Args:
            dpi: Resolution in DPI (default: from config)

        Yields:
            PIL Image for each page
        """
        for page_num in range(self.page_count):
            yield self.get_page_image(page_num, dpi)

    def get_page_thumbnail(self, page_num: int, size: tuple = None) -> Image.Image:
        """Get a thumbnail of a PDF page.

        Args:
            page_num: Page number (0-indexed)
            size: Thumbnail size (width, height)

        Returns:
            PIL Image thumbnail
        """
        if size is None:
            size = config.THUMBNAIL_SIZE

        # Use lower DPI for thumbnails
        image = self.get_page_image(page_num, dpi=72)
        image.thumbnail(size, Image.Resampling.LANCZOS)

        return image

    def get_all_thumbnails(self, size: tuple = None) -> Generator[Image.Image, None, None]:
        """Generator that yields thumbnails for all pages.

        Args:
            size: Thumbnail size (width, height)

        Yields:
            PIL Image thumbnail for each page
        """
        for page_num in range(self.page_count):
            yield self.get_page_thumbnail(page_num, size)


def load_pdf(pdf_path: str | Path) -> PDFProcessor:
    """Convenience function to create a PDFProcessor.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        PDFProcessor instance (remember to close or use as context manager)
    """
    processor = PDFProcessor(pdf_path)
    processor.open()
    return processor


def extract_pages_as_images(pdf_path: str | Path, dpi: int = None) -> list[Image.Image]:
    """Extract all pages from a PDF as images.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution in DPI

    Returns:
        List of PIL Images
    """
    with PDFProcessor(pdf_path) as processor:
        return list(processor.get_all_pages(dpi))
