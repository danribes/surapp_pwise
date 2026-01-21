"""Tests for curve extraction functionality."""

import unittest
import numpy as np
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class TestDataCorrector(unittest.TestCase):
    """Tests for data correction module."""

    def test_pava_algorithm_decreasing(self):
        """Test PAVA algorithm for decreasing monotonicity."""
        from src.core.data_corrector import pava_algorithm

        # Values that violate monotonicity
        values = [100, 90, 95, 80, 85, 70]

        result = pava_algorithm(values)

        # Should be monotonically decreasing
        for i in range(1, len(result)):
            self.assertLessEqual(result[i], result[i-1])

    def test_pava_already_monotonic(self):
        """Test PAVA with already monotonic data."""
        from src.core.data_corrector import pava_algorithm

        values = [100, 90, 80, 70, 60, 50]

        result = pava_algorithm(values)

        # Should be unchanged
        self.assertEqual(values, result)

    def test_isotonic_regression(self):
        """Test isotonic regression function."""
        from src.core.data_corrector import apply_isotonic_regression

        data = [(0, 100), (10, 95), (20, 98), (30, 80), (40, 70)]

        result = apply_isotonic_regression(data, decreasing=True)

        # Check monotonicity
        for i in range(1, len(result)):
            self.assertLessEqual(result[i][1], result[i-1][1])

    def test_correct_km_data(self):
        """Test KM data correction."""
        from src.core.data_corrector import correct_km_data

        data = [(5, 95), (10, 90), (15, 92), (20, 80)]

        result = correct_km_data(data, force_start_at_100=True)

        # Should start at 100
        self.assertEqual(result.corrected_points[0], (0.0, 100.0))

        # Should be monotonically decreasing
        for i in range(1, len(result.corrected_points)):
            self.assertLessEqual(
                result.corrected_points[i][1],
                result.corrected_points[i-1][1]
            )


@unittest.skipUnless(HAS_CV2, "OpenCV (cv2) not installed")
class TestCurveTracer(unittest.TestCase):
    """Tests for curve tracing module."""

    def test_calibration_data_conversion(self):
        """Test pixel to coordinate conversion."""
        from src.core.curve_tracer import CalibrationData

        calibration = CalibrationData(
            origin_pixel=(100, 400),
            x_max_pixel=(500, 400),
            y_max_pixel=(100, 100),
            x_max_value=50.0,
            y_max_value=100.0
        )

        # Test origin conversion
        x, y = calibration.pixel_to_coord(100, 400)
        self.assertAlmostEqual(x, 0.0, places=2)
        self.assertAlmostEqual(y, 0.0, places=2)

        # Test max X point
        x, y = calibration.pixel_to_coord(500, 400)
        self.assertAlmostEqual(x, 50.0, places=2)

        # Test max Y point
        x, y = calibration.pixel_to_coord(100, 100)
        self.assertAlmostEqual(y, 100.0, places=2)

        # Test middle point
        x, y = calibration.pixel_to_coord(300, 250)
        self.assertAlmostEqual(x, 25.0, places=2)
        self.assertAlmostEqual(y, 50.0, places=2)

    def test_calibration_inverse_conversion(self):
        """Test coordinate to pixel conversion."""
        from src.core.curve_tracer import CalibrationData

        calibration = CalibrationData(
            origin_pixel=(100, 400),
            x_max_pixel=(500, 400),
            y_max_pixel=(100, 100),
            x_max_value=50.0,
            y_max_value=100.0
        )

        # Convert to pixels and back
        original_x, original_y = 25.0, 50.0
        px, py = calibration.coord_to_pixel(original_x, original_y)
        back_x, back_y = calibration.pixel_to_coord(px, py)

        self.assertAlmostEqual(original_x, back_x, places=1)
        self.assertAlmostEqual(original_y, back_y, places=1)


class TestExporter(unittest.TestCase):
    """Tests for data export module."""

    def test_exporter_dataframe(self):
        """Test dataframe creation."""
        from src.core.exporter import DataExporter

        data = [(0, 100), (10, 90), (20, 80)]
        exporter = DataExporter(data)

        df = exporter.dataframe

        self.assertEqual(len(df), 3)
        self.assertIn("Time", df.columns)
        self.assertIn("Survival (%)", df.columns)

    def test_exporter_to_dict(self):
        """Test dictionary export."""
        from src.core.exporter import DataExporter

        data = [(0, 100), (10, 90), (20, 80)]
        exporter = DataExporter(data)

        result = exporter.to_dict()

        self.assertEqual(result["Time"], [0, 10, 20])
        self.assertEqual(result["Survival (%)"], [100, 90, 80])


class TestStatistics(unittest.TestCase):
    """Tests for statistics calculation."""

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        from src.core.data_corrector import calculate_statistics

        data = [(0, 100), (10, 80), (20, 60), (30, 40), (40, 20)]

        stats = calculate_statistics(data)

        self.assertEqual(stats["n_points"], 5)
        self.assertEqual(stats["time_range"], (0, 40))
        self.assertEqual(stats["survival_range"], (20, 100))
        self.assertEqual(stats["final_survival"], 20)

        # Median should be reached at t=20 (60%) or t=30 (40%)
        self.assertIsNotNone(stats["median_survival_time"])


@unittest.skipUnless(HAS_CV2, "OpenCV (cv2) not installed")
class TestImageUtils(unittest.TestCase):
    """Tests for image utilities."""

    def test_pil_cv2_conversion(self):
        """Test PIL to CV2 and back conversion."""
        from src.utils.image_utils import pil_to_cv2, cv2_to_pil
        from PIL import Image

        # Create a simple test image
        pil_img = Image.new("RGB", (100, 100), color=(255, 0, 0))

        # Convert to CV2
        cv2_img = pil_to_cv2(pil_img)

        self.assertEqual(cv2_img.shape, (100, 100, 3))

        # Convert back
        back_pil = cv2_to_pil(cv2_img)

        self.assertEqual(back_pil.size, (100, 100))


if __name__ == "__main__":
    unittest.main()
