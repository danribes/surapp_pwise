"""Main application window."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
from PIL import Image

from ..core.curve_tracer import CalibrationData, CurveTracer
from ..core.graph_detector import DetectedGraph
from ..utils.config import config
from ..utils.image_utils import pil_to_cv2

from .calibration import CalibrationDialog, QuickCalibrationWidget
from .graph_selector import GraphSelector
from .pdf_viewer import PDFThumbnailGallery, PDFPageViewer
from .preview import ResultsPreview, MultiCurvePreview


class KMExtractorApp(ctk.CTk):
    """Main application window for KM Curve Extractor."""

    def __init__(self):
        """Initialize the application."""
        super().__init__()

        self.title(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(900, 600)

        # Set appearance
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # State
        self._current_pdf_path: Optional[Path] = None
        self._current_image: Optional[Image.Image] = None
        self._calibration: Optional[CalibrationData] = None
        self._selected_graphs: list[DetectedGraph] = []
        self._extraction_mode = tk.StringVar(value="auto")

        self._create_menu()
        self._create_widgets()
        self._create_bindings()

    def _create_menu(self):
        """Create application menu."""
        # For cross-platform compatibility, we use a frame-based toolbar
        # instead of native menus
        pass

    def _create_widgets(self):
        """Create application widgets."""
        # Toolbar
        self._create_toolbar()

        # Main content area with panes
        self.main_paned = ctk.CTkFrame(self)
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left panel - PDF thumbnails
        self.left_panel = ctk.CTkFrame(self.main_paned, width=200)
        self.left_panel.pack(side="left", fill="y", padx=(0, 5))
        self.left_panel.pack_propagate(False)

        self.pdf_gallery = PDFThumbnailGallery(
            self.left_panel,
            on_page_select=self._on_page_selected
        )
        self.pdf_gallery.pack(fill="both", expand=True)

        # Center panel - Page viewer / Graph selector
        self.center_panel = ctk.CTkFrame(self.main_paned)
        self.center_panel.pack(side="left", fill="both", expand=True, padx=5)

        # Mode switcher
        mode_frame = ctk.CTkFrame(self.center_panel)
        mode_frame.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(mode_frame, text="Mode:").pack(side="left", padx=5)

        self.view_mode = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Page View", "Graph Selection"],
            command=self._on_mode_change
        )
        self.view_mode.set("Page View")
        self.view_mode.pack(side="left", padx=5)

        # Stacked frames for different views
        self.view_container = ctk.CTkFrame(self.center_panel)
        self.view_container.pack(fill="both", expand=True)

        self.page_viewer = PDFPageViewer(
            self.view_container,
            on_click=self._on_image_click
        )

        self.graph_selector = GraphSelector(
            self.view_container,
            on_selection_change=self._on_graph_selection_change
        )

        # Show page viewer by default
        self.page_viewer.pack(fill="both", expand=True)

        # Right panel - Controls
        self.right_panel = ctk.CTkFrame(self.main_paned, width=280)
        self.right_panel.pack(side="right", fill="y", padx=(5, 0))
        self.right_panel.pack_propagate(False)

        self._create_control_panel()

        # Status bar
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.pack(fill="x", side="bottom")

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready. Load a PDF file to begin.",
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10)

    def _create_toolbar(self):
        """Create toolbar."""
        toolbar = ctk.CTkFrame(self, height=40)
        toolbar.pack(fill="x", padx=5, pady=5)

        # File operations
        self.load_btn = ctk.CTkButton(
            toolbar,
            text="Load PDF",
            command=self._load_pdf,
            width=100
        )
        self.load_btn.pack(side="left", padx=5)

        self.load_image_btn = ctk.CTkButton(
            toolbar,
            text="Load Image",
            command=self._load_image,
            width=100
        )
        self.load_image_btn.pack(side="left", padx=5)

        # Separator
        sep = ctk.CTkFrame(toolbar, width=2)
        sep.pack(side="left", fill="y", padx=10, pady=5)

        # Extraction controls
        self.extract_btn = ctk.CTkButton(
            toolbar,
            text="Extract Curve",
            command=self._extract_curve,
            width=120,
            state="disabled"
        )
        self.extract_btn.pack(side="left", padx=5)

        self.batch_extract_btn = ctk.CTkButton(
            toolbar,
            text="Batch Extract",
            command=self._batch_extract,
            width=120,
            state="disabled"
        )
        self.batch_extract_btn.pack(side="left", padx=5)

        # Help
        self.help_btn = ctk.CTkButton(
            toolbar,
            text="Help",
            command=self._show_help,
            width=60,
            fg_color="transparent",
            border_width=1
        )
        self.help_btn.pack(side="right", padx=5)

    def _create_control_panel(self):
        """Create right control panel."""
        # Extraction mode
        mode_frame = ctk.CTkFrame(self.right_panel)
        mode_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            mode_frame,
            text="Extraction Mode",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        ctk.CTkRadioButton(
            mode_frame,
            text="Auto-detect graphs",
            variable=self._extraction_mode,
            value="auto"
        ).pack(padx=10, anchor="w")

        ctk.CTkRadioButton(
            mode_frame,
            text="Manual calibration",
            variable=self._extraction_mode,
            value="manual"
        ).pack(padx=10, anchor="w")

        # Calibration widget
        self.calibration_widget = QuickCalibrationWidget(
            self.right_panel,
            on_calibration_change=self._on_calibration_change
        )
        self.calibration_widget.pack(fill="x", padx=5, pady=10)

        # Calibrate button
        self.calibrate_btn = ctk.CTkButton(
            self.right_panel,
            text="Open Calibration",
            command=self._open_calibration,
            state="disabled"
        )
        self.calibrate_btn.pack(fill="x", padx=5, pady=5)

        # Separator
        sep = ctk.CTkFrame(self.right_panel, height=2)
        sep.pack(fill="x", padx=5, pady=10)

        # Selected graphs info
        self.selection_frame = ctk.CTkFrame(self.right_panel)
        self.selection_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            self.selection_frame,
            text="Selection",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        self.selection_label = ctk.CTkLabel(
            self.selection_frame,
            text="No graphs selected",
            text_color="gray"
        )
        self.selection_label.pack()

        # Options
        options_frame = ctk.CTkFrame(self.right_panel)
        options_frame.pack(fill="x", padx=5, pady=10)

        ctk.CTkLabel(
            options_frame,
            text="Options",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        self.apply_correction_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_frame,
            text="Apply isotonic correction",
            variable=self.apply_correction_var
        ).pack(padx=10, anchor="w")

        self.force_start_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_frame,
            text="Force start at 100%",
            variable=self.force_start_var
        ).pack(padx=10, anchor="w")

        # Y-axis scale
        scale_frame = ctk.CTkFrame(options_frame)
        scale_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(scale_frame, text="Y scale:").pack(side="left")

        self.y_scale_var = tk.StringVar(value="100")
        self.y_scale_combo = ctk.CTkOptionMenu(
            scale_frame,
            values=["100 (percentage)", "1.0 (proportion)"],
            variable=self.y_scale_var,
            width=150
        )
        self.y_scale_combo.pack(side="right")

    def _create_bindings(self):
        """Create keyboard bindings."""
        self.bind("<Control-o>", lambda e: self._load_pdf())
        self.bind("<Control-e>", lambda e: self._extract_curve())
        self.bind("<Control-c>", lambda e: self._open_calibration())
        self.bind("<F1>", lambda e: self._show_help())

    def _on_mode_change(self, mode: str):
        """Handle view mode change."""
        # Hide all views
        self.page_viewer.pack_forget()
        self.graph_selector.pack_forget()

        # Show selected view
        if mode == "Page View":
            self.page_viewer.pack(fill="both", expand=True)
        else:
            self.graph_selector.pack(fill="both", expand=True)

            # Update graph selector with current image
            if self._current_image:
                self.graph_selector.set_image(self._current_image)

    def _load_pdf(self):
        """Load a PDF file."""
        filepath = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )

        if filepath:
            self._current_pdf_path = Path(filepath)
            self.pdf_gallery.load_pdf(filepath)
            self._update_status(f"Loaded: {self._current_pdf_path.name}")
            self._update_button_states()

    def _load_image(self):
        """Load an image file directly."""
        filepath = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("All files", "*.*")
            ]
        )

        if filepath:
            try:
                image = Image.open(filepath)
                self._current_image = image.convert("RGB")
                self.page_viewer.set_image(self._current_image)

                if self.view_mode.get() == "Graph Selection":
                    self.graph_selector.set_image(self._current_image)

                self._update_status(f"Loaded image: {Path(filepath).name}")
                self._update_button_states()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def _on_page_selected(self, page_num: int, image: Image.Image):
        """Handle page selection from gallery."""
        self._current_image = image
        self.page_viewer.set_image(image, page_num)

        if self.view_mode.get() == "Graph Selection":
            self.graph_selector.set_image(image)

        self._calibration = None
        self.calibration_widget.clear_calibration()
        self._update_button_states()
        self._update_status(f"Selected page {page_num + 1}")

    def _on_image_click(self, x: int, y: int):
        """Handle click on image."""
        self._update_status(f"Clicked: ({x}, {y})")

    def _on_graph_selection_change(self, selected: list[DetectedGraph]):
        """Handle graph selection change."""
        self._selected_graphs = selected

        if selected:
            self.selection_label.configure(
                text=f"{len(selected)} graph(s) selected",
                text_color=("gray10", "gray90")
            )
        else:
            self.selection_label.configure(
                text="No graphs selected",
                text_color="gray"
            )

        self._update_button_states()

    def _on_calibration_change(self, calibration: CalibrationData):
        """Handle calibration change."""
        self._calibration = calibration
        self._update_button_states()

    def _open_calibration(self):
        """Open calibration dialog."""
        if self._current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        CalibrationDialog(
            self,
            self._current_image,
            self._set_calibration,
            self._calibration
        )

    def _set_calibration(self, calibration: CalibrationData):
        """Set calibration from dialog."""
        self._calibration = calibration
        self.calibration_widget.set_calibration(calibration)
        self._update_button_states()
        self._update_status("Calibration set successfully")

    def _extract_curve(self):
        """Extract curve from current selection."""
        if self._current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        mode = self._extraction_mode.get()

        if mode == "manual":
            if self._calibration is None:
                messagebox.showwarning(
                    "Warning",
                    "Please calibrate the axes first using the Calibration button"
                )
                return

            self._extract_with_calibration()
        else:
            self._extract_auto()

    def _extract_with_calibration(self):
        """Extract curve using manual calibration."""
        try:
            # Create tracer
            tracer = CurveTracer(self._current_image)
            tracer.detect_lines()
            tracer.find_step_curve()

            # Sample curve
            data_points = tracer.sample_curve_at_intervals(self._calibration)

            if not data_points:
                messagebox.showwarning(
                    "Warning",
                    "No curve data could be extracted. Try adjusting the calibration."
                )
                return

            # Show preview
            ResultsPreview(
                self,
                data_points,
                self._current_image,
                "Extracted Curve"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed: {e}")

    def _extract_auto(self):
        """Extract curve using auto-detection."""
        if not self._selected_graphs:
            # Try to detect graphs first
            self.view_mode.set("Graph Selection")
            self._on_mode_change("Graph Selection")

            if not self.graph_selector.get_selected_graphs():
                messagebox.showinfo(
                    "Info",
                    "Please select the graph region(s) you want to extract."
                )
                return

        # Get selected regions
        regions = self.graph_selector.get_selected_regions()

        if len(regions) == 1:
            # Single curve - open calibration for it
            CalibrationDialog(
                self,
                regions[0],
                lambda cal: self._extract_region_with_calibration(regions[0], cal)
            )
        else:
            # Multiple curves
            messagebox.showinfo(
                "Info",
                f"Selected {len(regions)} regions. "
                "Please calibrate each one individually."
            )

    def _extract_region_with_calibration(
        self,
        region: Image.Image,
        calibration: CalibrationData
    ):
        """Extract curve from a region with calibration."""
        try:
            tracer = CurveTracer(region)
            tracer.detect_lines()
            tracer.find_step_curve()

            data_points = tracer.sample_curve_at_intervals(calibration)

            if not data_points:
                messagebox.showwarning(
                    "Warning",
                    "No curve data could be extracted from this region."
                )
                return

            ResultsPreview(self, data_points, region, "Extracted Curve")

        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed: {e}")

    def _batch_extract(self):
        """Batch extract multiple curves."""
        if not self._selected_graphs:
            messagebox.showinfo(
                "Info",
                "Please select graphs to extract in Graph Selection mode."
            )
            return

        # For batch mode, we would need calibration for each
        # This is a simplified implementation
        messagebox.showinfo(
            "Batch Extract",
            f"Batch extraction of {len(self._selected_graphs)} graphs "
            "would require calibrating each one. "
            "This feature will be enhanced in a future update."
        )

    def _update_button_states(self):
        """Update button enabled states based on current state."""
        has_image = self._current_image is not None
        has_calibration = self._calibration is not None
        has_selection = len(self._selected_graphs) > 0
        mode = self._extraction_mode.get()

        self.calibrate_btn.configure(state="normal" if has_image else "disabled")

        if mode == "manual":
            can_extract = has_image and has_calibration
        else:
            can_extract = has_image

        self.extract_btn.configure(state="normal" if can_extract else "disabled")
        self.batch_extract_btn.configure(
            state="normal" if has_selection else "disabled"
        )

    def _update_status(self, message: str):
        """Update status bar message."""
        self.status_label.configure(text=message)

    def _show_help(self):
        """Show help dialog."""
        help_text = """
KM Curve Extractor - Help

QUICK START:
1. Load a PDF or image containing a Kaplan-Meier curve
2. Select the page/region containing the curve
3. Use Manual Calibration mode:
   - Click "Open Calibration"
   - Click on the origin point (0,0)
   - Click on the max X point
   - Click on the max Y point (100%)
   - Enter the axis values
4. Click "Extract Curve"
5. Preview and export results

KEYBOARD SHORTCUTS:
- Ctrl+O: Open PDF
- Ctrl+E: Extract curve
- Ctrl+C: Open calibration
- F1: Show help

TIPS:
- For best results, use high-quality PDF or image files
- The isotonic correction ensures the curve is monotonically decreasing
- Export to Excel includes metadata and statistics
        """

        dialog = ctk.CTkToplevel(self)
        dialog.title("Help")
        dialog.geometry("500x500")
        dialog.transient(self)

        text = ctk.CTkTextbox(dialog, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", help_text)
        text.configure(state="disabled")

        ctk.CTkButton(
            dialog, text="Close", command=dialog.destroy
        ).pack(pady=10)


def run_app():
    """Run the application."""
    app = KMExtractorApp()
    app.mainloop()
