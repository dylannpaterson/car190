import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_wfi_full_focal_plane_rates import (
    load_rms_map,
    load_rate_map,
    load_reset_rate_values,
    latest_statistics_observation,
    mean_rate_over_interval,
    sca_position,
)


class TestWfiFocalPlaneOrientation(unittest.TestCase):
    def test_mast_science_frame_tiles_are_not_flipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tile_rate_posterior.csv"
            fields = (
                "detector",
                "tile_row",
                "tile_col",
                "initial_rate_nm_day_median",
                "final_rate_nm_day_median",
                "mean_rate_nm_day_median",
            )
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in range(8):
                    for column in range(8):
                        value = 10 * row + column
                        writer.writerow(
                            {
                                "detector": "WFI01",
                                "tile_row": row,
                                "tile_col": column,
                                "initial_rate_nm_day_median": value,
                                "final_rate_nm_day_median": value,
                                "mean_rate_nm_day_median": value,
                            }
                        )

            maps = load_rate_map(path, "WFI01")
            expected = np.asarray(
                [[10 * row + column for column in range(8)] for row in range(8)]
            )
            np.testing.assert_array_equal(
                maps["mean_rate_nm_day_median"], expected
            )

    def test_detector_order_matches_science_focal_plane_layout(self):
        self.assertEqual(sca_position("WFI16"), (0.0, 18.0))
        self.assertEqual(sca_position("WFI01"), (27.0, 12.0))
        self.assertEqual(sca_position("WFI03"), (27.0, -6.0))
        self.assertEqual(sca_position("WFI07"), (45.0, 18.0))

    def test_256_tile_rate_map_uses_16_by_16_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tile_rate_posterior.csv"
            fields = (
                "detector",
                "tile_row",
                "tile_col",
                "initial_rate_nm_day_median",
                "final_rate_nm_day_median",
                "mean_rate_nm_day_median",
            )
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in range(16):
                    for column in range(16):
                        value = 10 * row + column
                        writer.writerow(
                            {
                                "detector": "WFI01",
                                "tile_row": row,
                                "tile_col": column,
                                "initial_rate_nm_day_median": value,
                                "final_rate_nm_day_median": value,
                                "mean_rate_nm_day_median": value,
                            }
                        )
            maps = load_rate_map(path, "WFI01")
            self.assertEqual(maps["mean_rate_nm_day_median"].shape, (16, 16))
            self.assertEqual(maps["mean_rate_nm_day_median"][15, 15], 165)

    def test_reset_rates_use_endpoint_thickness_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "posterior_samples.npz"
            np.savez(
                path,
                observation_days=np.asarray([0.0, 1.0, 2.0, 4.0, 5.0, 6.0]),
                ice_reset_day=np.asarray([3.0]),
                tile_ids=np.asarray([0, 1]),
                tile_ice_thickness=np.asarray(
                    [
                        [
                            [1.0, 3.0, 5.0, 2.0, 4.0, 8.0],
                            [2.0, 4.0, 8.0, 4.0, 7.0, 13.0],
                        ]
                    ]
                ),
            )

            tile_ids, pre_values, post_values, reset_day = load_reset_rate_values(path)

            np.testing.assert_array_equal(tile_ids, [0, 1])
            np.testing.assert_allclose(pre_values, [2.0, 3.0])
            np.testing.assert_allclose(post_values, [3.0, 4.5])
            self.assertEqual(reset_day, 3.0)

    def test_last_48h_rate_uses_interpolated_thickness_endpoints(self):
        days = np.asarray([0.0, 0.25, 1.25, 2.0, 3.5, 4.0])
        thickness = np.asarray([2.0, 2.75, 5.75, 8.0, 12.5, 14.0])

        self.assertAlmostEqual(
            mean_rate_over_interval(days, thickness, duration_days=2.0), 3.0
        )

    def test_latest_statistics_observation_is_normalized_to_utc(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wfi01_tile_log_statistics_512.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("exposure_start_time",))
                writer.writeheader()
                writer.writerow({"exposure_start_time": "2026-10-11T06:00:00Z"})
                writer.writerow(
                    {"exposure_start_time": "2026-10-11T06:17:38.939000+00:00"}
                )

            self.assertEqual(
                latest_statistics_observation(path),
                "2026-10-11 06:17:38.939 UTC",
            )

    def test_rms_map_excludes_reference_repeat_and_uses_millilog(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            fit_dir = output_dir / "fits" / "tile_000"
            fit_dir.mkdir(parents=True)
            with (output_dir / "tile_rate_posterior.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("detector", "tile_id", "tile_row", "tile_col"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "detector": "WFI01",
                        "tile_id": 0,
                        "tile_row": 0,
                        "tile_col": 0,
                    }
                )
            with (fit_dir / "tile_group_predictions.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("repeat", "residual_log"),
                )
                writer.writeheader()
                writer.writerow({"repeat": 0, "residual_log": 100.0})
                writer.writerow({"repeat": 1, "residual_log": 0.003})
                writer.writerow({"repeat": 1, "residual_log": -0.004})

            maps = load_rms_map(output_dir, "WFI01")

            self.assertEqual(maps["rms_residual_millilog"].shape, (1, 1))
            self.assertAlmostEqual(
                maps["rms_residual_millilog"][0, 0],
                3.5355339059327378,
            )


if __name__ == "__main__":
    unittest.main()
