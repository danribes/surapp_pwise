"""Data export module for CSV and Excel formats."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.config import config


class DataExporter:
    """Handles exporting survival data to various formats."""

    def __init__(
        self,
        data_points: list[tuple],
        time_label: str = "Time",
        survival_label: str = "Survival (%)",
        metadata: Optional[dict] = None
    ):
        """Initialize exporter.

        Args:
            data_points: List of (time, survival) tuples
            time_label: Label for time column
            survival_label: Label for survival column
            metadata: Optional metadata to include in export
        """
        self.data_points = data_points
        self.time_label = time_label
        self.survival_label = survival_label
        self.metadata = metadata or {}
        self._df = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get data as pandas DataFrame."""
        if self._df is None:
            self._df = pd.DataFrame(
                self.data_points,
                columns=[self.time_label, self.survival_label]
            )
            # Round to configured decimal places
            self._df = self._df.round(config.DEFAULT_DECIMAL_PLACES)
        return self._df

    def to_csv(
        self,
        filepath: str | Path,
        include_header: bool = True,
        decimal_places: int = None
    ) -> Path:
        """Export data to CSV file.

        Args:
            filepath: Output file path
            include_header: Include column headers
            decimal_places: Number of decimal places (default: from config)

        Returns:
            Path to created file
        """
        filepath = Path(filepath)

        if decimal_places is None:
            decimal_places = config.DEFAULT_DECIMAL_PLACES

        df = self.dataframe.round(decimal_places)
        df.to_csv(filepath, index=False, header=include_header)

        return filepath

    def to_excel(
        self,
        filepath: str | Path,
        sheet_name: str = "Survival Data",
        include_metadata: bool = True,
        decimal_places: int = None
    ) -> Path:
        """Export data to Excel file.

        Args:
            filepath: Output file path
            sheet_name: Name of the worksheet
            include_metadata: Include metadata in a separate sheet
            decimal_places: Number of decimal places (default: from config)

        Returns:
            Path to created file
        """
        filepath = Path(filepath)

        if decimal_places is None:
            decimal_places = config.DEFAULT_DECIMAL_PLACES

        df = self.dataframe.round(decimal_places)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Write metadata if requested
            if include_metadata and self.metadata:
                meta_df = pd.DataFrame([
                    {"Property": k, "Value": str(v)}
                    for k, v in self.metadata.items()
                ])
                meta_df.to_excel(writer, sheet_name="Metadata", index=False)

        return filepath

    def to_dict(self) -> dict:
        """Export data as dictionary.

        Returns:
            Dictionary with time and survival arrays
        """
        return {
            self.time_label: [p[0] for p in self.data_points],
            self.survival_label: [p[1] for p in self.data_points]
        }

    def to_json(self, filepath: str | Path, indent: int = 2) -> Path:
        """Export data to JSON file.

        Args:
            filepath: Output file path
            indent: JSON indentation

        Returns:
            Path to created file
        """
        filepath = Path(filepath)

        self.dataframe.to_json(filepath, orient='records', indent=indent)

        return filepath


class MultiCurveExporter:
    """Handles exporting multiple survival curves."""

    def __init__(self):
        """Initialize multi-curve exporter."""
        self.curves: dict[str, list[tuple]] = {}
        self.metadata: dict = {}

    def add_curve(self, name: str, data_points: list[tuple]):
        """Add a survival curve.

        Args:
            name: Name/identifier for the curve
            data_points: List of (time, survival) tuples
        """
        self.curves[name] = data_points

    def set_metadata(self, metadata: dict):
        """Set metadata for the export.

        Args:
            metadata: Dictionary of metadata
        """
        self.metadata = metadata

    def to_csv_wide(self, filepath: str | Path) -> Path:
        """Export all curves to CSV in wide format.

        Each curve gets its own survival column.

        Args:
            filepath: Output file path

        Returns:
            Path to created file
        """
        filepath = Path(filepath)

        if not self.curves:
            raise ValueError("No curves to export")

        # Collect all time points
        all_times = set()
        for points in self.curves.values():
            all_times.update(p[0] for p in points)

        times = sorted(all_times)

        # Build dataframe
        data = {"Time": times}

        for name, points in self.curves.items():
            # Create lookup dictionary
            point_dict = {p[0]: p[1] for p in points}

            # Fill in survival values (forward fill for missing)
            survivals = []
            last_value = None
            for t in times:
                if t in point_dict:
                    last_value = point_dict[t]
                survivals.append(last_value)

            data[f"Survival_{name}"] = survivals

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        return filepath

    def to_csv_long(self, filepath: str | Path) -> Path:
        """Export all curves to CSV in long format.

        Includes a curve identifier column.

        Args:
            filepath: Output file path

        Returns:
            Path to created file
        """
        filepath = Path(filepath)

        if not self.curves:
            raise ValueError("No curves to export")

        rows = []
        for name, points in self.curves.items():
            for time, survival in points:
                rows.append({
                    "Curve": name,
                    "Time": time,
                    "Survival": survival
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        return filepath

    def to_excel(self, filepath: str | Path) -> Path:
        """Export all curves to Excel with separate sheets.

        Args:
            filepath: Output file path

        Returns:
            Path to created file
        """
        filepath = Path(filepath)

        if not self.curves:
            raise ValueError("No curves to export")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet with all curves in wide format
            all_times = set()
            for points in self.curves.values():
                all_times.update(p[0] for p in points)

            times = sorted(all_times)
            summary_data = {"Time": times}

            for name, points in self.curves.items():
                point_dict = {p[0]: p[1] for p in points}
                survivals = []
                last_value = None
                for t in times:
                    if t in point_dict:
                        last_value = point_dict[t]
                    survivals.append(last_value)
                summary_data[name] = survivals

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Individual sheets for each curve
            for name, points in self.curves.items():
                # Sanitize sheet name (max 31 chars, no special chars)
                sheet_name = name[:31].replace("/", "-").replace("\\", "-")
                df = pd.DataFrame(points, columns=["Time", "Survival"])
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Metadata sheet
            if self.metadata:
                meta_df = pd.DataFrame([
                    {"Property": k, "Value": str(v)}
                    for k, v in self.metadata.items()
                ])
                meta_df.to_excel(writer, sheet_name="Metadata", index=False)

        return filepath


def export_to_csv(
    data_points: list[tuple],
    filepath: str | Path,
    time_label: str = "Time",
    survival_label: str = "Survival (%)"
) -> Path:
    """Convenience function to export data to CSV.

    Args:
        data_points: List of (time, survival) tuples
        filepath: Output file path
        time_label: Label for time column
        survival_label: Label for survival column

    Returns:
        Path to created file
    """
    exporter = DataExporter(data_points, time_label, survival_label)
    return exporter.to_csv(filepath)


def export_to_excel(
    data_points: list[tuple],
    filepath: str | Path,
    time_label: str = "Time",
    survival_label: str = "Survival (%)",
    metadata: dict = None
) -> Path:
    """Convenience function to export data to Excel.

    Args:
        data_points: List of (time, survival) tuples
        filepath: Output file path
        time_label: Label for time column
        survival_label: Label for survival column
        metadata: Optional metadata dictionary

    Returns:
        Path to created file
    """
    exporter = DataExporter(data_points, time_label, survival_label, metadata)
    return exporter.to_excel(filepath)
