import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np

from fit_wfi_side_swap import fit_side_swap_series
from wfi_side_swap import apply_side_swap_correction, fit_log_step_spline


class SideSwapFitTests(unittest.TestCase):
    def test_recovers_multiplicative_step_from_continuous_trend(self):
        times = np.asarray([-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        trend = 2.0 + 0.03 * times
        observed = np.exp(trend + (times >= 0.0) * np.log(1.25))

        fit = fit_log_step_spline(times, observed, 0.0)

        self.assertAlmostEqual(fit.step_log, np.log(1.25), places=12)
        self.assertAlmostEqual(fit.correction_factor, 0.8, places=12)
        self.assertAlmostEqual(fit.rmse_log, 0.0, places=12)
        self.assertEqual(fit.selected_indices.tolist(), list(range(10)))

    def test_correction_only_changes_right_hand_samples(self):
        times = np.asarray([-2.0, -1.0, 1.0, 2.0])
        values = np.asarray([2.0, 3.0, 12.0, 16.0])

        corrected = apply_side_swap_correction(times, values, 0.0, 0.5)

        np.testing.assert_array_equal(corrected, [2.0, 3.0, 6.0, 8.0])

    def test_uses_nearest_requested_points(self):
        times = np.asarray([-10.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        values = np.exp(1.0 + 0.02 * times)

        fit = fit_log_step_spline(
            times,
            values,
            0.0,
            pre_points=3,
            post_points=3,
        )

        self.assertEqual(fit.selected_indices.tolist(), [3, 4, 5, 6, 7, 8])

    def test_rejects_insufficient_side(self):
        with self.assertRaisesRegex(ValueError, "Need 5 pre-event and 5 post-event"):
            fit_log_step_spline(
                np.asarray([-2.0, -1.0, 1.0, 2.0]),
                np.ones(4),
                0.0,
            )

    def test_csv_calibration_fits_image_and_photodiode_separately(self):
        image_fields = (
            "repeat",
            "led",
            "file_set_name",
            "days",
            "n_pair_valid",
            "sum_current_counts",
        )
        inventory_fields = (
            "repeat",
            "led",
            "file_set_name",
            "days",
            "photodiode_mean",
            "photodiode_dark_subtracted",
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "raw.csv"
            inventory_path = Path(directory) / "inventory.csv"
            with image_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=image_fields)
                writer.writeheader()
                for repeat in range(10):
                    days = float(repeat)
                    value = np.exp(2.0 + 0.02 * days)
                    if repeat >= 5:
                        value *= 1.2
                    writer.writerow(
                        {
                            "repeat": repeat,
                            "led": 1,
                            "file_set_name": f"led1_{repeat}",
                            "days": days,
                            "n_pair_valid": 1,
                            "sum_current_counts": value,
                        }
                    )
            with inventory_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=inventory_fields)
                writer.writeheader()
                for repeat in range(10):
                    days = float(repeat)
                    value = np.exp(1.0 + 0.01 * days)
                    if repeat >= 5:
                        value *= 0.8
                    writer.writerow(
                        {
                            "repeat": repeat,
                            "led": 1,
                            "file_set_name": f"led1_{repeat}",
                            "days": days,
                            "photodiode_mean": value,
                            "photodiode_dark_subtracted": "True",
                        }
                    )

            rows = fit_side_swap_series(
                image_path,
                inventory_path,
                event_repeat=5,
            )

        fits = {row["channel"]: row for row in rows}
        self.assertAlmostEqual(fits["image"]["correction_factor"], 1.0 / 1.2, places=12)
        self.assertAlmostEqual(fits["photodiode"]["correction_factor"], 1.0 / 0.8, places=12)


if __name__ == "__main__":
    unittest.main()
