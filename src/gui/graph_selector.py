"""Graph selection interface for detected graphs."""

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw

from ..core.graph_detector import DetectedGraph, GraphDetector
from ..utils.image_utils import pil_to_cv2, cv2_to_pil


class GraphSelector(ctk.CTkFrame):
    """Interface for selecting detected graphs."""

    def __init__(
        self,
        parent,
        on_selection_change: Callable[[list[DetectedGraph]], None] = None
    ):
        """Initialize graph selector.

        Args:
            parent: Parent widget
            on_selection_change: Callback when selection changes
        """
        super().__init__(parent)

        self.on_selection_change = on_selection_change

        self._current_image: Optional[Image.Image] = None
        self._detected_graphs: list[DetectedGraph] = []
        self._selected_indices: set[int] = set()

        self._photo_image = None
        self._scale_factor = 1.0
        self._image_offset = (0, 0)

        self._create_widgets()

    def _create_widgets(self):
        """Create selector widgets."""
        # Header
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=5, pady=5)

        self.title_label = ctk.CTkLabel(
            header,
            text="Detected Graphs",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.pack(side="left")

        self.count_label = ctk.CTkLabel(
            header,
            text="0 found",
            text_color="gray"
        )
        self.count_label.pack(side="right")

        # Main content - split view
        content = ctk.CTkFrame(self)
        content.pack(fill="both", expand=True, padx=5, pady=5)

        # Left - Image with bounding boxes
        image_frame = ctk.CTkFrame(content)
        image_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(image_frame, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Right - List of detected graphs
        list_frame = ctk.CTkFrame(content, width=200)
        list_frame.pack(side="right", fill="y", padx=(5, 0))
        list_frame.pack_propagate(False)

        list_header = ctk.CTkLabel(
            list_frame,
            text="Select Graphs",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        list_header.pack(pady=5)

        self.graph_list = ctk.CTkScrollableFrame(list_frame)
        self.graph_list.pack(fill="both", expand=True, pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(list_frame)
        btn_frame.pack(fill="x", pady=5)

        self.select_all_btn = ctk.CTkButton(
            btn_frame,
            text="Select All",
            command=self._select_all,
            width=80
        )
        self.select_all_btn.pack(side="left", padx=2)

        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="Clear",
            command=self._clear_selection,
            width=80,
            fg_color="gray"
        )
        self.clear_btn.pack(side="right", padx=2)

        # Manual selection hint
        hint = ctk.CTkLabel(
            list_frame,
            text="Click on image or\ncheckboxes to select",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        hint.pack(pady=5)

    def set_image(self, image: Image.Image):
        """Set the image to display and detect graphs.

        Args:
            image: PIL Image
        """
        self._current_image = image
        self._detected_graphs = []
        self._selected_indices = set()

        # Detect graphs
        self._detect_graphs()

        # Update display
        self._display_image()
        self._update_list()

    def _detect_graphs(self):
        """Run graph detection on current image."""
        if self._current_image is None:
            return

        # Convert to OpenCV format
        cv_image = pil_to_cv2(self._current_image)

        # Run detection
        detector = GraphDetector(cv_image)
        self._detected_graphs = detector.detect_all()

        # Update count
        self.count_label.configure(text=f"{len(self._detected_graphs)} found")

        # Auto-select high-confidence KM curves
        for i, graph in enumerate(self._detected_graphs):
            if graph.graph_type == 'km_curve' and graph.confidence > 0.5:
                self._selected_indices.add(i)

    def _display_image(self):
        """Display image with detection boxes."""
        if self._current_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.after(100, self._display_image)
            return

        # Calculate scale
        img_width, img_height = self._current_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self._scale_factor = min(scale_x, scale_y, 1.0)

        # Create display image with boxes
        display = self._current_image.copy()
        draw = ImageDraw.Draw(display)

        # Draw bounding boxes
        for i, graph in enumerate(self._detected_graphs):
            x, y, w, h = graph.bbox

            # Color based on selection and type
            if i in self._selected_indices:
                color = (0, 200, 0)  # Green for selected
                width = 3
            elif graph.graph_type == 'km_curve':
                color = (0, 100, 255)  # Blue for KM curves
                width = 2
            else:
                color = (200, 200, 0)  # Yellow for generic
                width = 1

            draw.rectangle(
                [x, y, x + w, y + h],
                outline=color,
                width=width
            )

            # Draw label
            label = f"{i + 1}: {graph.graph_type} ({graph.confidence:.0%})"
            draw.text((x + 5, y + 5), label, fill=color)

        # Resize for display
        new_width = int(img_width * self._scale_factor)
        new_height = int(img_height * self._scale_factor)
        display = display.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self._photo_image = ImageTk.PhotoImage(display)

        # Center on canvas
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self._image_offset = (x_offset, y_offset)

        # Draw on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            x_offset, y_offset,
            anchor="nw",
            image=self._photo_image
        )

    def _update_list(self):
        """Update the graph list panel."""
        # Clear existing items
        for widget in self.graph_list.winfo_children():
            widget.destroy()

        if not self._detected_graphs:
            label = ctk.CTkLabel(
                self.graph_list,
                text="No graphs detected",
                text_color="gray"
            )
            label.pack(pady=20)
            return

        # Create checkbox for each graph
        self._checkboxes = []
        for i, graph in enumerate(self._detected_graphs):
            frame = ctk.CTkFrame(self.graph_list, fg_color="transparent")
            frame.pack(fill="x", pady=2)

            var = tk.BooleanVar(value=i in self._selected_indices)

            cb = ctk.CTkCheckBox(
                frame,
                text=f"Graph {i + 1}",
                variable=var,
                command=lambda idx=i, v=var: self._toggle_selection(idx, v)
            )
            cb.pack(side="left")

            # Type label
            type_label = ctk.CTkLabel(
                frame,
                text=graph.graph_type[:8],
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            type_label.pack(side="right", padx=5)

            self._checkboxes.append((var, cb))

    def _toggle_selection(self, index: int, var: tk.BooleanVar):
        """Toggle selection of a graph.

        Args:
            index: Graph index
            var: Associated checkbox variable
        """
        if var.get():
            self._selected_indices.add(index)
        else:
            self._selected_indices.discard(index)

        self._display_image()
        self._notify_selection_change()

    def _on_canvas_click(self, event):
        """Handle click on canvas to select graph."""
        if not self._detected_graphs:
            return

        # Convert to image coordinates
        x_offset, y_offset = self._image_offset
        img_x = (event.x - x_offset) / self._scale_factor
        img_y = (event.y - y_offset) / self._scale_factor

        # Find clicked graph
        for i, graph in enumerate(self._detected_graphs):
            gx, gy, gw, gh = graph.bbox
            if gx <= img_x <= gx + gw and gy <= img_y <= gy + gh:
                # Toggle selection
                if i in self._selected_indices:
                    self._selected_indices.discard(i)
                else:
                    self._selected_indices.add(i)

                # Update checkbox
                if i < len(self._checkboxes):
                    self._checkboxes[i][0].set(i in self._selected_indices)

                self._display_image()
                self._notify_selection_change()
                break

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        self._display_image()

    def _select_all(self):
        """Select all detected graphs."""
        self._selected_indices = set(range(len(self._detected_graphs)))

        # Update checkboxes
        for var, _ in self._checkboxes:
            var.set(True)

        self._display_image()
        self._notify_selection_change()

    def _clear_selection(self):
        """Clear selection."""
        self._selected_indices.clear()

        # Update checkboxes
        for var, _ in self._checkboxes:
            var.set(False)

        self._display_image()
        self._notify_selection_change()

    def _notify_selection_change(self):
        """Notify parent of selection change."""
        if self.on_selection_change:
            selected = [
                self._detected_graphs[i]
                for i in sorted(self._selected_indices)
            ]
            self.on_selection_change(selected)

    def get_selected_graphs(self) -> list[DetectedGraph]:
        """Get list of selected graphs.

        Returns:
            List of selected DetectedGraph objects
        """
        return [
            self._detected_graphs[i]
            for i in sorted(self._selected_indices)
        ]

    def get_selected_regions(self) -> list[Image.Image]:
        """Get cropped images of selected graphs.

        Returns:
            List of PIL Images
        """
        if self._current_image is None:
            return []

        regions = []
        for i in sorted(self._selected_indices):
            graph = self._detected_graphs[i]
            x, y, w, h = graph.bbox
            region = self._current_image.crop((x, y, x + w, y + h))
            regions.append(region)

        return regions

    def add_manual_region(self, bbox: tuple):
        """Add a manually selected region.

        Args:
            bbox: (x, y, width, height) bounding box
        """
        graph = DetectedGraph(
            bbox=bbox,
            confidence=1.0,
            graph_type='manual'
        )

        # Crop image region
        if self._current_image:
            x, y, w, h = bbox
            graph.image = pil_to_cv2(
                self._current_image.crop((x, y, x + w, y + h))
            )

        self._detected_graphs.append(graph)
        self._selected_indices.add(len(self._detected_graphs) - 1)

        self._display_image()
        self._update_list()
        self._notify_selection_change()
