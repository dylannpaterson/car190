import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_wfi_full_focal_plane_rates import load_rate_map, sca_position


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


if __name__ == "__main__":
    unittest.main()
