import csv
import math
import tempfile
import unittest
from pathlib import Path

from apply_wfi_photodiode_correction import apply_correction, apply_scale_correction


class TestPhotodiodeCorrection(unittest.TestCase):
    def test_applies_count_and_log_moment_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            summary.write_text(
                "exposure_start_time,led,pd1_v_mean,pd2_v_mean\n"
                "2026-09-12T01:25:50.714,1,10,20\n"
                "2026-09-12T01:26:00.714,1,8,16\n"
            )
            statistics = root / "statistics.csv"
            fields = [
                "led",
                "exposure_start_time",
                "n_pair_valid",
                "sum_reference_counts",
                "sum_squared_reference_counts",
                "sum_current_counts",
                "sum_squared_current_counts",
                "sum_reference_current_counts",
                "min_reference_count",
                "max_reference_count",
                "min_current_count",
                "max_current_count",
                "sum_log_change",
                "sum_squared_log_change",
                "min_log_change",
                "max_log_change",
            ]
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "led": 1,
                        "exposure_start_time": "2026-09-12T01:25:50.7140000",
                        "n_pair_valid": 2,
                        "sum_reference_counts": 20,
                        "sum_squared_reference_counts": 202,
                        "sum_current_counts": 40,
                        "sum_squared_current_counts": 802,
                        "sum_reference_current_counts": 402,
                        "min_reference_count": 9,
                        "max_reference_count": 11,
                        "min_current_count": 19,
                        "max_current_count": 21,
                        "sum_log_change": 0.3,
                        "sum_squared_log_change": 0.05,
                        "min_log_change": 0.1,
                        "max_log_change": 0.2,
                    }
                )
                writer.writerow(
                    {
                        "led": 1,
                        "exposure_start_time": "2026-09-12T01:26:00.7140000",
                        "n_pair_valid": 2,
                        "sum_reference_counts": 20,
                        "sum_squared_reference_counts": 202,
                        "sum_current_counts": 40,
                        "sum_squared_current_counts": 802,
                        "sum_reference_current_counts": 402,
                        "min_reference_count": 9,
                        "max_reference_count": 11,
                        "min_current_count": 19,
                        "max_current_count": 21,
                        "sum_log_change": 0.3,
                        "sum_squared_log_change": 0.05,
                        "min_log_change": 0.1,
                        "max_log_change": 0.2,
                    }
                )
            output = root / "corrected.csv"
            result = apply_correction(statistics, summary, output, "pd1")
            self.assertEqual(result["n_rows"], 2)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            first = rows[0]
            self.assertAlmostEqual(float(first["sum_current_counts"]), 40.0)
            second = rows[1]
            factor = 1.25
            log_factor = math.log(factor)
            self.assertAlmostEqual(float(second["sum_current_counts"]), 50.0)
            self.assertAlmostEqual(float(second["sum_squared_current_counts"]), 802.0 * factor**2)
            self.assertAlmostEqual(float(second["sum_reference_current_counts"]), 402.0 * factor)
            self.assertAlmostEqual(float(second["min_current_count"]), 19.0 * factor)
            self.assertAlmostEqual(float(second["max_current_count"]), 21.0 * factor)
            self.assertAlmostEqual(float(second["sum_log_change"]), 0.3 + 2.0 * log_factor)
            self.assertAlmostEqual(
                float(second["sum_squared_log_change"]),
                0.05 + 2.0 * log_factor * 0.3 + 2.0 * log_factor**2,
            )
            self.assertAlmostEqual(float(second["min_log_change"]), 0.1 + log_factor)
            self.assertAlmostEqual(float(second["max_log_change"]), 0.2 + log_factor)

    def test_applies_log_only_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            summary.write_text(
                "exposure_start_time,led,pd1_v_mean,pd2_v_mean\n"
                "2026-09-12T01:25:50.714,1,10,20\n"
                "2026-09-12T01:26:00.714,1,8,16\n"
            )
            statistics = root / "statistics.csv"
            statistics.write_text(
                "led,exposure_start_time,n_pair_valid,sum_log_change,"
                "sum_squared_log_change,min_log_change,max_log_change\n"
                "1,2026-09-12T01:25:50.7140000,2,0.3,0.05,0.1,0.2\n"
                "1,2026-09-12T01:26:00.7140000,2,0.3,0.05,0.1,0.2\n"
            )
            output = root / "corrected.csv"
            result = apply_correction(statistics, summary, output, "pd2")
            self.assertFalse(result["count_statistics_available"])
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            log_factor = math.log(1.25)
            self.assertAlmostEqual(float(rows[1]["sum_log_change"]), 0.3 + 2.0 * log_factor)

    def test_applies_direct_scale_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "scales.csv"
            summary.write_text(
                "exposure_start_time,led,replacement_scale_ratio\n"
                "2026-09-12T01:25:50.714,1,1.0\n"
                "2026-09-12T01:26:00.714,1,1.25\n"
            )
            statistics = root / "statistics.csv"
            statistics.write_text(
                "led,exposure_start_time,n_pair_valid,sum_log_change,"
                "sum_squared_log_change,min_log_change,max_log_change\n"
                "1,2026-09-12T01:25:50.7140000,2,0.3,0.05,0.1,0.2\n"
                "1,2026-09-12T01:26:00.7140000,2,0.3,0.05,0.1,0.2\n"
            )
            output = root / "corrected.csv"
            result = apply_scale_correction(
                statistics, summary, output, "replacement_scale_ratio"
            )
            self.assertEqual(result["n_exposures"], 2)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(
                float(rows[1]["sum_log_change"]), 0.3 + 2.0 * math.log(1.25)
            )


if __name__ == "__main__":
    unittest.main()
