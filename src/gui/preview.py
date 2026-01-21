"""Results preview window with data visualization."""

import tkinter as tk
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from PIL import Image

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..core.data_corrector import correct_km_data, calculate_statistics
from ..core.exporter import DataExporter, MultiCurveExporter


class ResultsPreview(ctk.CTkToplevel):
    """Dialog for previewing and exporting extracted data."""

    def __init__(
        self,
        parent,
        data_points: list[tuple],
        original_image: Optional[Image.Image] = None,
        curve_name: str = "Curve 1"
    ):
        """Initialize results preview.

        Args:
            parent: Parent window
            data_points: List of (time, survival) tuples
            original_image: Optional original graph image
            curve_name: Name for the curve
        """
        super().__init__(parent)

        self.title("Extraction Results")
        self.geometry("1000x700")
        self.minsize(800, 600)

        self.data_points = data_points
        self.original_image = original_image
        self.curve_name = curve_name

        # Corrected data
        self.correction_result = correct_km_data(data_points)
        self.use_corrected = tk.BooleanVar(value=True)

        self._create_widgets()

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

    def _create_widgets(self):
        """Create preview widgets."""
        # Main container with tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs
        self.tabview.add("Plot")
        self.tabview.add("Data")
        self.tabview.add("Statistics")

        # Plot tab
        self._create_plot_tab()

        # Data tab
        self._create_data_tab()

        # Statistics tab
        self._create_stats_tab()

        # Bottom buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)

        self.export_csv_btn = ctk.CTkButton(
            btn_frame,
            text="Export CSV",
            command=self._export_csv
        )
        self.export_csv_btn.pack(side="left", padx=5)

        self.export_excel_btn = ctk.CTkButton(
            btn_frame,
            text="Export Excel",
            command=self._export_excel
        )
        self.export_excel_btn.pack(side="left", padx=5)

        self.close_btn = ctk.CTkButton(
            btn_frame,
            text="Close",
            command=self.destroy,
            fg_color="transparent",
            border_width=1
        )
        self.close_btn.pack(side="right", padx=5)

    def _create_plot_tab(self):
        """Create the plot tab."""
        plot_frame = self.tabview.tab("Plot")

        if not HAS_MATPLOTLIB:
            label = ctk.CTkLabel(
                plot_frame,
                text="Matplotlib not available for plotting",
                text_color="orange"
            )
            label.pack(pady=50)
            return

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Controls
        control_frame = ctk.CTkFrame(plot_frame)
        control_frame.pack(fill="x", pady=5)

        # Correction toggle
        self.correction_cb = ctk.CTkCheckBox(
            control_frame,
            text="Apply isotonic correction",
            variable=self.use_corrected,
            command=self._update_plot
        )
        self.correction_cb.pack(side="left", padx=10)

        # Show original toggle
        self.show_original_var = tk.BooleanVar(value=True)
        if self.correction_result.changes_made > 0:
            self.show_original_cb = ctk.CTkCheckBox(
                control_frame,
                text="Show original data",
                variable=self.show_original_var,
                command=self._update_plot
            )
            self.show_original_cb.pack(side="left", padx=10)

        # Update plot
        self._update_plot()

    def _update_plot(self):
        """Update the matplotlib plot."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()

        # Get data based on correction setting
        if self.use_corrected.get():
            points = self.correction_result.corrected_points
        else:
            points = self.correction_result.original_points

        # Plot data
        times = [p[0] for p in points]
        survivals = [p[1] for p in points]

        self.ax.step(times, survivals, where='post', linewidth=2,
                     label='Extracted', color='blue')

        # Show original if different and toggle enabled
        if (self.use_corrected.get() and
            self.show_original_var.get() and
            self.correction_result.changes_made > 0):

            orig_times = [p[0] for p in self.correction_result.original_points]
            orig_survivals = [p[1] for p in self.correction_result.original_points]
            self.ax.step(orig_times, orig_survivals, where='post',
                        linewidth=1, linestyle='--', alpha=0.5,
                        label='Original', color='gray')

        # Configure plot
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Survival (%)')
        self.ax.set_title(f'{self.curve_name} - Kaplan-Meier Curve')
        self.ax.set_ylim(-5, 105)
        self.ax.set_xlim(left=0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        self.canvas.draw()

    def _create_data_tab(self):
        """Create the data tab with table view."""
        data_frame = self.tabview.tab("Data")

        # Header
        header = ctk.CTkFrame(data_frame)
        header.pack(fill="x", pady=5)

        ctk.CTkLabel(
            header,
            text=f"Extracted Data Points ({len(self.data_points)} points)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")

        if self.correction_result.changes_made > 0:
            ctk.CTkLabel(
                header,
                text=f"({self.correction_result.changes_made} points corrected)",
                text_color="orange"
            ).pack(side="right")

        # Data table
        table_frame = ctk.CTkScrollableFrame(data_frame)
        table_frame.pack(fill="both", expand=True, pady=5)

        # Table header
        header_frame = ctk.CTkFrame(table_frame)
        header_frame.pack(fill="x")

        ctk.CTkLabel(
            header_frame, text="#", width=50,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=5)
        ctk.CTkLabel(
            header_frame, text="Time", width=100,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=5)
        ctk.CTkLabel(
            header_frame, text="Survival", width=100,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=5)
        ctk.CTkLabel(
            header_frame, text="Corrected", width=100,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=5)

        # Data rows
        original = {p[0]: p[1] for p in self.correction_result.original_points}
        corrected = {p[0]: p[1] for p in self.correction_result.corrected_points}

        all_times = sorted(set(original.keys()) | set(corrected.keys()))

        for i, t in enumerate(all_times[:100]):  # Limit display to 100 rows
            row = ctk.CTkFrame(table_frame, fg_color="transparent")
            row.pack(fill="x")

            orig_val = original.get(t, "-")
            corr_val = corrected.get(t, "-")

            # Highlight corrections
            if orig_val != corr_val and orig_val != "-" and corr_val != "-":
                row.configure(fg_color=("gray90", "gray20"))

            ctk.CTkLabel(row, text=str(i+1), width=50).pack(side="left", padx=5)
            ctk.CTkLabel(
                row,
                text=f"{t:.4f}" if isinstance(t, float) else str(t),
                width=100
            ).pack(side="left", padx=5)
            ctk.CTkLabel(
                row,
                text=f"{orig_val:.4f}" if isinstance(orig_val, float) else str(orig_val),
                width=100
            ).pack(side="left", padx=5)
            ctk.CTkLabel(
                row,
                text=f"{corr_val:.4f}" if isinstance(corr_val, float) else str(corr_val),
                width=100
            ).pack(side="left", padx=5)

        if len(all_times) > 100:
            ctk.CTkLabel(
                table_frame,
                text=f"... and {len(all_times) - 100} more rows",
                text_color="gray"
            ).pack(pady=10)

    def _create_stats_tab(self):
        """Create the statistics tab."""
        stats_frame = self.tabview.tab("Statistics")

        # Calculate statistics
        points = (self.correction_result.corrected_points
                  if self.use_corrected.get()
                  else self.correction_result.original_points)
        stats = calculate_statistics(points)

        # Display statistics
        ctk.CTkLabel(
            stats_frame,
            text="Survival Curve Statistics",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        stats_display = ctk.CTkFrame(stats_frame)
        stats_display.pack(fill="both", expand=True, padx=20, pady=10)

        stat_items = [
            ("Number of data points", stats.get('n_points', 'N/A')),
            ("Time range", f"{stats.get('time_range', ('N/A', 'N/A'))[0]:.2f} - {stats.get('time_range', ('N/A', 'N/A'))[1]:.2f}"),
            ("Survival range", f"{stats.get('survival_range', ('N/A', 'N/A'))[0]:.2f}% - {stats.get('survival_range', ('N/A', 'N/A'))[1]:.2f}%"),
            ("Median survival time", f"{stats.get('median_survival_time', 'N/A')}" if stats.get('median_survival_time') else "Not reached"),
            ("Final survival", f"{stats.get('final_survival', 'N/A'):.2f}%"),
            ("Area under curve", f"{stats.get('area_under_curve', 'N/A'):.2f}"),
            ("Corrections applied", f"{self.correction_result.changes_made} points"),
        ]

        for label, value in stat_items:
            row = ctk.CTkFrame(stats_display, fg_color="transparent")
            row.pack(fill="x", pady=5)

            ctk.CTkLabel(
                row, text=label + ":",
                font=ctk.CTkFont(weight="bold"),
                width=200
            ).pack(side="left", padx=10)

            ctk.CTkLabel(row, text=str(value)).pack(side="left")

    def _export_csv(self):
        """Export data to CSV."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.curve_name.replace(' ', '_')}_survival_data.csv"
        )

        if filepath:
            points = (self.correction_result.corrected_points
                      if self.use_corrected.get()
                      else self.correction_result.original_points)

            exporter = DataExporter(points)
            exporter.to_csv(filepath)

            # Show confirmation
            self._show_message(f"Data exported to:\n{filepath}")

    def _export_excel(self):
        """Export data to Excel."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile=f"{self.curve_name.replace(' ', '_')}_survival_data.xlsx"
        )

        if filepath:
            points = (self.correction_result.corrected_points
                      if self.use_corrected.get()
                      else self.correction_result.original_points)

            stats = calculate_statistics(points)

            exporter = DataExporter(
                points,
                metadata={
                    "Curve Name": self.curve_name,
                    "Points Count": len(points),
                    "Corrected": "Yes" if self.use_corrected.get() else "No",
                    "Corrections Made": self.correction_result.changes_made,
                    **{k: str(v) for k, v in stats.items()}
                }
            )
            exporter.to_excel(filepath)

            self._show_message(f"Data exported to:\n{filepath}")

    def _show_message(self, message: str):
        """Show a message dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Export Complete")
        dialog.geometry("400x150")
        dialog.transient(self)

        ctk.CTkLabel(dialog, text=message, wraplength=350).pack(pady=20)
        ctk.CTkButton(dialog, text="OK", command=dialog.destroy).pack(pady=10)

        dialog.grab_set()


class MultiCurvePreview(ctk.CTkToplevel):
    """Preview for multiple extracted curves."""

    def __init__(self, parent, curves: dict[str, list[tuple]]):
        """Initialize multi-curve preview.

        Args:
            parent: Parent window
            curves: Dictionary of {curve_name: data_points}
        """
        super().__init__(parent)

        self.title("Multi-Curve Results")
        self.geometry("1100x750")
        self.minsize(900, 650)

        self.curves = curves
        self.corrected_curves = {}

        # Apply corrections to all curves
        for name, points in curves.items():
            result = correct_km_data(points)
            self.corrected_curves[name] = result.corrected_points

        self._create_widgets()

        self.transient(parent)
        self.grab_set()

    def _create_widgets(self):
        """Create widgets."""
        # Plot
        plot_frame = ctk.CTkFrame(self)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        if HAS_MATPLOTLIB:
            self.fig = Figure(figsize=(9, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)

            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

            self._update_plot()
        else:
            ctk.CTkLabel(
                plot_frame,
                text="Matplotlib not available",
                text_color="orange"
            ).pack(pady=50)

        # Buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="Export All (CSV)",
            command=self._export_all_csv
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Export All (Excel)",
            command=self._export_all_excel
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Close",
            command=self.destroy,
            fg_color="transparent",
            border_width=1
        ).pack(side="right", padx=5)

    def _update_plot(self):
        """Update the plot with all curves."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for i, (name, points) in enumerate(self.corrected_curves.items()):
            times = [p[0] for p in points]
            survivals = [p[1] for p in points]

            color = colors[i % len(colors)]
            self.ax.step(times, survivals, where='post', linewidth=2,
                        label=name, color=color)

        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Survival (%)')
        self.ax.set_title('Kaplan-Meier Survival Curves')
        self.ax.set_ylim(-5, 105)
        self.ax.set_xlim(left=0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        self.canvas.draw()

    def _export_all_csv(self):
        """Export all curves to CSV."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="km_curves_data.csv"
        )

        if filepath:
            exporter = MultiCurveExporter()
            for name, points in self.corrected_curves.items():
                exporter.add_curve(name, points)

            exporter.to_csv_long(filepath)

    def _export_all_excel(self):
        """Export all curves to Excel."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="km_curves_data.xlsx"
        )

        if filepath:
            exporter = MultiCurveExporter()
            for name, points in self.corrected_curves.items():
                exporter.add_curve(name, points)

            exporter.to_excel(filepath)
