"""Manual calibration dialog for axis point selection."""

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from ..core.curve_tracer import CalibrationData


class CalibrationDialog(ctk.CTkToplevel):
    """Dialog for manual calibration of graph axes."""

    def __init__(
        self,
        parent,
        image: Image.Image,
        callback: Callable[[CalibrationData], None],
        initial_calibration: Optional[CalibrationData] = None
    ):
        """Initialize calibration dialog.

        Args:
            parent: Parent window
            image: Graph image to calibrate
            callback: Function to call with calibration data when done
            initial_calibration: Optional initial calibration to edit
        """
        super().__init__(parent)

        self.title("Calibrate Axes")
        self.geometry("900x700")
        self.minsize(800, 600)

        self.image = image
        self.callback = callback
        self.initial_calibration = initial_calibration

        # Calibration points
        self.origin_point = None
        self.x_max_point = None
        self.y_max_point = None

        # Scale values
        self.x_max_value = tk.DoubleVar(value=100.0)
        self.y_max_value = tk.DoubleVar(value=100.0)

        # Current selection mode
        self.selection_mode = tk.StringVar(value="origin")

        # Canvas image reference
        self._photo_image = None
        self._scale_factor = 1.0

        self._create_widgets()
        self._load_initial_calibration()

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

    def _create_widgets(self):
        """Create dialog widgets."""
        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left side - Image canvas
        canvas_frame = ctk.CTkFrame(main_frame)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Instructions label
        self.instructions_label = ctk.CTkLabel(
            canvas_frame,
            text="Click on the graph to set calibration points",
            font=ctk.CTkFont(size=14)
        )
        self.instructions_label.pack(pady=5)

        # Canvas for image
        self.canvas = tk.Canvas(canvas_frame, bg="white", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Right side - Controls
        control_frame = ctk.CTkFrame(main_frame, width=250)
        control_frame.pack(side="right", fill="y")
        control_frame.pack_propagate(False)

        # Selection mode
        mode_label = ctk.CTkLabel(
            control_frame,
            text="Select Point:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        mode_label.pack(pady=(10, 5))

        modes = [
            ("origin", "Origin (0, 0)", "Green"),
            ("x_max", "Max X Point", "Blue"),
            ("y_max", "Max Y Point (100%)", "Red")
        ]

        for mode, label, color in modes:
            rb = ctk.CTkRadioButton(
                control_frame,
                text=label,
                variable=self.selection_mode,
                value=mode,
                command=self._update_instructions
            )
            rb.pack(pady=5, padx=10, anchor="w")

        # Separator
        sep = ctk.CTkFrame(control_frame, height=2)
        sep.pack(fill="x", pady=15, padx=10)

        # Scale values
        scale_label = ctk.CTkLabel(
            control_frame,
            text="Axis Values:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        scale_label.pack(pady=(5, 10))

        # X max value
        x_frame = ctk.CTkFrame(control_frame)
        x_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(x_frame, text="Max X (Time):").pack(side="left")
        self.x_max_entry = ctk.CTkEntry(
            x_frame,
            textvariable=self.x_max_value,
            width=80
        )
        self.x_max_entry.pack(side="right")

        # Y max value
        y_frame = ctk.CTkFrame(control_frame)
        y_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(y_frame, text="Max Y:").pack(side="left")
        self.y_max_entry = ctk.CTkEntry(
            y_frame,
            textvariable=self.y_max_value,
            width=80
        )
        self.y_max_entry.pack(side="right")

        # Y scale hint
        y_hint = ctk.CTkLabel(
            control_frame,
            text="(100 for %, 1.0 for proportion)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        y_hint.pack(pady=(0, 10))

        # Current points display
        points_label = ctk.CTkLabel(
            control_frame,
            text="Selected Points:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        points_label.pack(pady=(15, 5))

        self.points_display = ctk.CTkTextbox(control_frame, height=120, width=220)
        self.points_display.pack(padx=10, pady=5)
        self.points_display.configure(state="disabled")

        # Buttons
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=self._clear_points,
            fg_color="gray"
        )
        self.clear_btn.pack(fill="x", pady=5)

        self.confirm_btn = ctk.CTkButton(
            button_frame,
            text="Confirm Calibration",
            command=self._confirm,
            state="disabled"
        )
        self.confirm_btn.pack(fill="x", pady=5)

        self.cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy,
            fg_color="transparent",
            border_width=1
        )
        self.cancel_btn.pack(fill="x", pady=5)

    def _load_initial_calibration(self):
        """Load initial calibration if provided."""
        if self.initial_calibration:
            self.origin_point = self.initial_calibration.origin_pixel
            self.x_max_point = self.initial_calibration.x_max_pixel
            self.y_max_point = self.initial_calibration.y_max_pixel
            self.x_max_value.set(self.initial_calibration.x_max_value)
            self.y_max_value.set(self.initial_calibration.y_max_value)
            self._update_display()

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        self._display_image()

    def _display_image(self):
        """Display image on canvas with scaling."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet
            self.after(100, self._display_image)
            return

        # Calculate scale factor
        img_width, img_height = self.image.size

        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self._scale_factor = min(scale_x, scale_y, 1.0)

        # Resize image
        new_width = int(img_width * self._scale_factor)
        new_height = int(img_height * self._scale_factor)

        resized = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(resized)

        # Clear and redraw
        self.canvas.delete("all")

        # Center image
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2

        self.canvas.create_image(
            x_offset, y_offset,
            anchor="nw",
            image=self._photo_image,
            tags="image"
        )

        # Store offset for coordinate conversion
        self._image_offset = (x_offset, y_offset)

        # Redraw calibration points
        self._draw_points()

    def _canvas_to_image(self, canvas_x: int, canvas_y: int) -> tuple:
        """Convert canvas coordinates to image coordinates."""
        if not hasattr(self, '_image_offset'):
            return (0, 0)

        x_offset, y_offset = self._image_offset

        img_x = int((canvas_x - x_offset) / self._scale_factor)
        img_y = int((canvas_y - y_offset) / self._scale_factor)

        # Clamp to image bounds
        img_x = max(0, min(img_x, self.image.width - 1))
        img_y = max(0, min(img_y, self.image.height - 1))

        return (img_x, img_y)

    def _image_to_canvas(self, img_x: int, img_y: int) -> tuple:
        """Convert image coordinates to canvas coordinates."""
        if not hasattr(self, '_image_offset'):
            return (0, 0)

        x_offset, y_offset = self._image_offset

        canvas_x = int(img_x * self._scale_factor + x_offset)
        canvas_y = int(img_y * self._scale_factor + y_offset)

        return (canvas_x, canvas_y)

    def _on_canvas_click(self, event):
        """Handle canvas click to set calibration point."""
        # Convert to image coordinates
        img_coords = self._canvas_to_image(event.x, event.y)

        # Set appropriate point
        mode = self.selection_mode.get()

        if mode == "origin":
            self.origin_point = img_coords
            # Auto-advance to next mode
            self.selection_mode.set("x_max")
        elif mode == "x_max":
            self.x_max_point = img_coords
            self.selection_mode.set("y_max")
        elif mode == "y_max":
            self.y_max_point = img_coords

        self._update_display()
        self._draw_points()

    def _draw_points(self):
        """Draw calibration points on canvas."""
        # Clear existing point markers
        self.canvas.delete("point")

        radius = 6

        # Draw origin (green)
        if self.origin_point:
            cx, cy = self._image_to_canvas(*self.origin_point)
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="green", outline="white", width=2, tags="point"
            )
            self.canvas.create_text(
                cx + 10, cy - 10, text="Origin", fill="green",
                anchor="sw", font=("Arial", 10, "bold"), tags="point"
            )

        # Draw X max (blue)
        if self.x_max_point:
            cx, cy = self._image_to_canvas(*self.x_max_point)
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="blue", outline="white", width=2, tags="point"
            )
            self.canvas.create_text(
                cx + 10, cy - 10, text="X Max", fill="blue",
                anchor="sw", font=("Arial", 10, "bold"), tags="point"
            )

        # Draw Y max (red)
        if self.y_max_point:
            cx, cy = self._image_to_canvas(*self.y_max_point)
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="red", outline="white", width=2, tags="point"
            )
            self.canvas.create_text(
                cx + 10, cy - 10, text="Y Max", fill="red",
                anchor="sw", font=("Arial", 10, "bold"), tags="point"
            )

        # Draw lines connecting points
        if self.origin_point and self.x_max_point:
            o_canvas = self._image_to_canvas(*self.origin_point)
            x_canvas = self._image_to_canvas(*self.x_max_point)
            self.canvas.create_line(
                o_canvas[0], o_canvas[1], x_canvas[0], x_canvas[1],
                fill="blue", dash=(4, 4), tags="point"
            )

        if self.origin_point and self.y_max_point:
            o_canvas = self._image_to_canvas(*self.origin_point)
            y_canvas = self._image_to_canvas(*self.y_max_point)
            self.canvas.create_line(
                o_canvas[0], o_canvas[1], y_canvas[0], y_canvas[1],
                fill="red", dash=(4, 4), tags="point"
            )

    def _update_display(self):
        """Update points display and button states."""
        # Update text display
        self.points_display.configure(state="normal")
        self.points_display.delete("1.0", "end")

        text = ""
        if self.origin_point:
            text += f"Origin: {self.origin_point}\n"
        else:
            text += "Origin: Not set\n"

        if self.x_max_point:
            text += f"X Max: {self.x_max_point}\n"
        else:
            text += "X Max: Not set\n"

        if self.y_max_point:
            text += f"Y Max: {self.y_max_point}\n"
        else:
            text += "Y Max: Not set\n"

        self.points_display.insert("1.0", text)
        self.points_display.configure(state="disabled")

        # Update button state
        all_set = all([self.origin_point, self.x_max_point, self.y_max_point])
        self.confirm_btn.configure(state="normal" if all_set else "disabled")

        self._update_instructions()

    def _update_instructions(self):
        """Update instruction label based on current mode."""
        mode = self.selection_mode.get()

        instructions = {
            "origin": "Click on the graph origin (where axes meet, usually 0,0)",
            "x_max": "Click on the rightmost point of the X-axis",
            "y_max": "Click on the topmost point of the Y-axis (100% survival)"
        }

        self.instructions_label.configure(text=instructions.get(mode, ""))

    def _clear_points(self):
        """Clear all calibration points."""
        self.origin_point = None
        self.x_max_point = None
        self.y_max_point = None
        self.selection_mode.set("origin")
        self._update_display()
        self._draw_points()

    def _confirm(self):
        """Confirm calibration and return data."""
        try:
            x_max = float(self.x_max_value.get())
            y_max = float(self.y_max_value.get())
        except ValueError:
            # Show error
            return

        calibration = CalibrationData(
            origin_pixel=self.origin_point,
            x_max_pixel=self.x_max_point,
            y_max_pixel=self.y_max_point,
            x_max_value=x_max,
            y_max_value=y_max
        )

        self.callback(calibration)
        self.destroy()


class QuickCalibrationWidget(ctk.CTkFrame):
    """Compact calibration widget for embedding in main window."""

    def __init__(
        self,
        parent,
        on_calibration_change: Callable[[CalibrationData], None] = None
    ):
        """Initialize quick calibration widget.

        Args:
            parent: Parent widget
            on_calibration_change: Callback when calibration changes
        """
        super().__init__(parent)

        self.on_calibration_change = on_calibration_change
        self.calibration: Optional[CalibrationData] = None

        self._create_widgets()

    def _create_widgets(self):
        """Create widget components."""
        # Title
        title = ctk.CTkLabel(
            self,
            text="Calibration",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=(5, 10))

        # Status
        self.status_label = ctk.CTkLabel(
            self,
            text="Not calibrated",
            text_color="orange"
        )
        self.status_label.pack()

        # Values display
        self.values_frame = ctk.CTkFrame(self)
        self.values_frame.pack(fill="x", padx=5, pady=5)

        self.x_label = ctk.CTkLabel(self.values_frame, text="X: 0 - ?")
        self.x_label.pack()

        self.y_label = ctk.CTkLabel(self.values_frame, text="Y: 0 - ?")
        self.y_label.pack()

        # Calibrate button
        self.calibrate_btn = ctk.CTkButton(
            self,
            text="Set Calibration",
            command=self._request_calibration
        )
        self.calibrate_btn.pack(pady=5)

    def _request_calibration(self):
        """Request calibration dialog from parent."""
        # This should be connected by the parent
        pass

    def set_calibration(self, calibration: CalibrationData):
        """Set calibration data.

        Args:
            calibration: Calibration data
        """
        self.calibration = calibration

        self.status_label.configure(text="Calibrated", text_color="green")
        self.x_label.configure(text=f"X: 0 - {calibration.x_max_value}")
        self.y_label.configure(text=f"Y: 0 - {calibration.y_max_value}")

        if self.on_calibration_change:
            self.on_calibration_change(calibration)

    def clear_calibration(self):
        """Clear calibration data."""
        self.calibration = None
        self.status_label.configure(text="Not calibrated", text_color="orange")
        self.x_label.configure(text="X: 0 - ?")
        self.y_label.configure(text="Y: 0 - ?")
