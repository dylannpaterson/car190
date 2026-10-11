import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_wfi_reset_grey_throughput_mosaic import (
    DEFAULT_PRODUCTION_PREFIX,
    SCAS,
    load_production_maps,
)


class ResetGreyThroughputMosaicTests(unittest.TestCase):
    def write_production_posteriors(self, root, rows_per_detector=64):
        fields = (
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
            "post_reset_log_offset_p16",
            "post_reset_log_offset_median",
            "post_reset_log_offset_p84",
        )
        for detector in SCAS:
            output = (
                root
                / f"{DEFAULT_PRODUCTION_PREFIX}{detector}_map10000_emcee1000"
                / "post_reset_tile_log_offset_posterior.csv"
            )
            output.parent.mkdir(parents=True)
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for tile_id in range(rows_per_detector):
                    tile_row, tile_col = divmod(tile_id, 8)
                    writer.writerow(
                        {
                            "detector": detector,
                            "tile_id": tile_id,
                            "tile_row": tile_row,
                            "tile_col": tile_col,
                            "post_reset_log_offset_p16": 0.09,
                            "post_reset_log_offset_median": 0.1,
                            "post_reset_log_offset_p84": 0.11,
                        }
                    )

    def test_loads_and_converts_per_tile_log_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_production_posteriors(root)
            maps = load_production_maps(root)
        np.testing.assert_allclose(
            maps["WFI01"]["change"][0, 0], 100.0 * np.expm1(0.1)
        )
        np.testing.assert_allclose(
            maps["WFI01"]["interval_width"][0, 0],
            100.0 * (np.expm1(0.11) - np.expm1(0.09)),
        )

    def test_rejects_incomplete_detector_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_production_posteriors(root, rows_per_detector=63)
            with self.assertRaises(ValueError):
                load_production_maps(root)


if __name__ == "__main__":
    unittest.main()
