import unittest
import shutil
import tempfile
from pathlib import Path

import numpy as np

from fit_wfi_tmm_numpyro import (
    DEFAULT_SINGLE_SCA_TILE_STATISTICS,
    load_tile_statistics,
)


class TestTmmTileData(unittest.TestCase):
    def test_repeat_nodes_are_shared_and_time_ordered(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=2)

        np.testing.assert_array_equal(data["repeat_values"], np.arange(19))
        np.testing.assert_array_equal(
            data["repeat_index"][:19], np.arange(19)
        )
        self.assertEqual(len(data["node_days"]), 19)
        self.assertTrue(np.all(np.diff(data["node_days"]) > 0.0))
        self.assertEqual(data["counts"].shape, (2, 114))

    def test_combines_sca_statistics_with_shared_observation_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copy(
                DEFAULT_SINGLE_SCA_TILE_STATISTICS,
                directory / "wfi01_tile_log_statistics_512.csv",
            )
            shutil.copy(
                DEFAULT_SINGLE_SCA_TILE_STATISTICS,
                directory / "wfi02_tile_log_statistics_512.csv",
            )
            data = load_tile_statistics(directory, max_tiles=2)

        self.assertEqual(data["sca_names"].tolist(), ["WFI01", "WFI02"])
        self.assertEqual(data["n_scas"], 2)
        self.assertEqual(data["n_tiles"], 2)
        self.assertEqual(data["counts"].shape, (4, 114))


if __name__ == "__main__":
    unittest.main()
