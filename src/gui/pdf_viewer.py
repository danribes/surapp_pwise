"""PDF page viewer and thumbnail gallery."""

import tkinter as tk
from pathlib import Path
from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from ..core.pdf_processor import PDFProcessor
from ..utils.config import config


class PDFThumbnailGallery(ctk.CTkFrame):
    """Scrollable gallery of PDF page thumbnails."""

    def __init__(
        self,
        parent,
        on_page_select: Callable[[int, Image.Image], None] = None,
        thumbnail_size: tuple = None
    ):
        """Initialize thumbnail gallery.

        Args:
            parent: Parent widget
            on_page_select: Callback when page is selected (page_num, full_image)
            thumbnail_size: Size of thumbnails (width, height)
        """
        super().__init__(parent)

        self.on_page_select = on_page_select
        self.thumbnail_size = thumbnail_size or config.THUMBNAIL_SIZE

        self._pdf_processor: Optional[PDFProcessor] = None
        self._thumbnails: list[ImageTk.PhotoImage] = []
        self._full_images: list[Image.Image] = []
        self._selected_page = -1

        self._create_widgets()

    def _create_widgets(self):
        """Create gallery widgets."""
        # Header
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=5, pady=5)

        self.title_label = ctk.CTkLabel(
            header,
            text="PDF Pages",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.pack(side="left")

        self.page_count_label = ctk.CTkLabel(
            header,
            text="0 pages",
            text_color="gray"
        )
        self.page_count_label.pack(side="right")

        # Scrollable thumbnail area
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            orientation="vertical"
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Placeholder message
        self.placeholder = ctk.CTkLabel(
            self.scroll_frame,
            text="Load a PDF file to view pages",
            text_color="gray"
        )
        self.placeholder.pack(pady=50)

    def load_pdf(self, pdf_path: str | Path):
        """Load PDF and create thumbnails.

        Args:
            pdf_path: Path to PDF file
        """
        # Clear existing
        self._clear()

        try:
            self._pdf_processor = PDFProcessor(pdf_path)
            self._pdf_processor.open()

            page_count = self._pdf_processor.page_count
            self.page_count_label.configure(text=f"{page_count} pages")

            # Hide placeholder
            self.placeholder.pack_forget()

            # Load thumbnails and full images
            for i in range(page_count):
                self._load_page(i)

            # Select first page by default
            if page_count > 0:
                self._select_page(0)

        except Exception as e:
            self.placeholder.configure(text=f"Error: {str(e)}")
            self.placeholder.pack(pady=50)

    def _load_page(self, page_num: int):
        """Load a single page thumbnail.

        Args:
            page_num: Page number (0-indexed)
        """
        # Get thumbnail
        thumbnail = self._pdf_processor.get_page_thumbnail(
            page_num, self.thumbnail_size
        )

        # Store full image for later use
        full_image = self._pdf_processor.get_page_image(page_num)
        self._full_images.append(full_image)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(thumbnail)
        self._thumbnails.append(photo)

        # Create thumbnail frame
        thumb_frame = ctk.CTkFrame(
            self.scroll_frame,
            fg_color="transparent"
        )
        thumb_frame.pack(pady=5)

        # Thumbnail button
        thumb_btn = ctk.CTkButton(
            thumb_frame,
            image=photo,
            text="",
            width=self.thumbnail_size[0] + 10,
            height=self.thumbnail_size[1] + 10,
            fg_color="transparent",
            hover_color=("gray80", "gray30"),
            command=lambda p=page_num: self._select_page(p)
        )
        thumb_btn.pack()

        # Page label
        page_label = ctk.CTkLabel(
            thumb_frame,
            text=f"Page {page_num + 1}",
            font=ctk.CTkFont(size=11)
        )
        page_label.pack()

        # Store reference to frame for selection highlighting
        thumb_frame.page_num = page_num
        thumb_frame.button = thumb_btn

    def _select_page(self, page_num: int):
        """Select a page.

        Args:
            page_num: Page number to select
        """
        if page_num == self._selected_page:
            return

        # Update visual selection
        for widget in self.scroll_frame.winfo_children():
            if hasattr(widget, 'page_num'):
                if widget.page_num == page_num:
                    widget.button.configure(fg_color=("gray70", "gray40"))
                elif widget.page_num == self._selected_page:
                    widget.button.configure(fg_color="transparent")

        self._selected_page = page_num

        # Callback with full image
        if self.on_page_select and page_num < len(self._full_images):
            self.on_page_select(page_num, self._full_images[page_num])

    def _clear(self):
        """Clear all loaded content."""
        # Close PDF
        if self._pdf_processor:
            self._pdf_processor.close()
            self._pdf_processor = None

        # Clear images
        self._thumbnails.clear()
        self._full_images.clear()
        self._selected_page = -1

        # Clear widgets
        for widget in self.scroll_frame.winfo_children():
            if widget != self.placeholder:
                widget.destroy()

        # Show placeholder
        self.placeholder.pack(pady=50)
        self.page_count_label.configure(text="0 pages")

    def get_selected_page(self) -> Optional[Image.Image]:
        """Get the currently selected page image.

        Returns:
            PIL Image of selected page or None
        """
        if 0 <= self._selected_page < len(self._full_images):
            return self._full_images[self._selected_page]
        return None

    def get_selected_page_number(self) -> int:
        """Get the selected page number.

        Returns:
            Page number (0-indexed) or -1 if none selected
        """
        return self._selected_page


class PDFPageViewer(ctk.CTkFrame):
    """Large view of a single PDF page with zoom and pan."""

    def __init__(
        self,
        parent,
        on_click: Callable[[int, int], None] = None
    ):
        """Initialize page viewer.

        Args:
            parent: Parent widget
            on_click: Callback when image is clicked (x, y in image coords)
        """
        super().__init__(parent)

        self.on_click = on_click

        self._current_image: Optional[Image.Image] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._scale = 1.0
        self._offset = (0, 0)

        self._create_widgets()

    def _create_widgets(self):
        """Create viewer widgets."""
        # Toolbar
        toolbar = ctk.CTkFrame(self)
        toolbar.pack(fill="x", padx=5, pady=5)

        # Zoom controls
        ctk.CTkLabel(toolbar, text="Zoom:").pack(side="left", padx=5)

        self.zoom_out_btn = ctk.CTkButton(
            toolbar, text="-", width=30,
            command=self._zoom_out
        )
        self.zoom_out_btn.pack(side="left", padx=2)

        self.zoom_label = ctk.CTkLabel(toolbar, text="100%", width=50)
        self.zoom_label.pack(side="left", padx=5)

        self.zoom_in_btn = ctk.CTkButton(
            toolbar, text="+", width=30,
            command=self._zoom_in
        )
        self.zoom_in_btn.pack(side="left", padx=2)

        self.fit_btn = ctk.CTkButton(
            toolbar, text="Fit", width=50,
            command=self._fit_to_view
        )
        self.fit_btn.pack(side="left", padx=10)

        # Page info
        self.page_info = ctk.CTkLabel(toolbar, text="No page loaded")
        self.page_info.pack(side="right", padx=5)

        # Canvas with scrollbars
        canvas_frame = ctk.CTkFrame(self)
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="gray20",
            highlightthickness=0
        )

        # Scrollbars
        self.v_scroll = ctk.CTkScrollbar(
            canvas_frame,
            orientation="vertical",
            command=self.canvas.yview
        )
        self.h_scroll = ctk.CTkScrollbar(
            canvas_frame,
            orientation="horizontal",
            command=self.canvas.xview
        )

        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set
        )

        # Grid layout for scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Bind events
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def set_image(self, image: Image.Image, page_num: int = None):
        """Set the image to display.

        Args:
            image: PIL Image to display
            page_num: Optional page number for info display
        """
        self._current_image = image

        if page_num is not None:
            self.page_info.configure(
                text=f"Page {page_num + 1} ({image.width}x{image.height})"
            )

        self._fit_to_view()

    def _display_image(self):
        """Display the current image at current scale."""
        if self._current_image is None:
            return

        # Calculate scaled size
        scaled_width = int(self._current_image.width * self._scale)
        scaled_height = int(self._current_image.height * self._scale)

        # Resize image
        scaled = self._current_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.LANCZOS
        )

        self._photo_image = ImageTk.PhotoImage(scaled)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0,
            anchor="nw",
            image=self._photo_image,
            tags="image"
        )

        # Update scroll region
        self.canvas.configure(scrollregion=(0, 0, scaled_width, scaled_height))

        # Update zoom label
        self.zoom_label.configure(text=f"{int(self._scale * 100)}%")

    def _zoom_in(self):
        """Zoom in."""
        self._scale = min(self._scale * 1.25, 5.0)
        self._display_image()

    def _zoom_out(self):
        """Zoom out."""
        self._scale = max(self._scale / 1.25, 0.1)
        self._display_image()

    def _fit_to_view(self):
        """Fit image to view."""
        if self._current_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.after(100, self._fit_to_view)
            return

        scale_x = canvas_width / self._current_image.width
        scale_y = canvas_height / self._current_image.height
        self._scale = min(scale_x, scale_y, 1.0)

        self._display_image()

    def _on_resize(self, event):
        """Handle canvas resize."""
        # Only auto-fit if at fit scale
        pass

    def _on_click(self, event):
        """Handle canvas click."""
        if self._current_image is None or self.on_click is None:
            return

        # Convert canvas coords to image coords
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        img_x = int(canvas_x / self._scale)
        img_y = int(canvas_y / self._scale)

        # Clamp to image bounds
        img_x = max(0, min(img_x, self._current_image.width - 1))
        img_y = max(0, min(img_y, self._current_image.height - 1))

        self.on_click(img_x, img_y)

    def _on_mousewheel(self, event):
        """Handle mouse wheel for zooming."""
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def clear(self):
        """Clear the viewer."""
        self._current_image = None
        self._photo_image = None
        self.canvas.delete("all")
        self.page_info.configure(text="No page loaded")
        self.zoom_label.configure(text="100%")
