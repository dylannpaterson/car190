import tempfile
import unittest
from pathlib import Path

import numpy as np

import plot_wfi_fitted_incidence_angle_mosaic as plotter


class FittedIncidenceAngleMosaicTests(unittest.TestCase):
    def write_archive(self, path, tile_id, angle, fitted=True):
        path.mkdir(parents=True)
        np.savez(
            path / "posterior_samples.npz",
            tile_ids=np.asarray([tile_id]),
            incidence_angle_deg=np.asarray([angle]),
            fit_incidence_angle=np.asarray(fitted),
        )

    def test_loads_new_production_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_archive(
                root / "production" / "wfi01" / "fits" / "tile_009",
                9,
                2.5,
            )
            values = plotter.load_incidence_angle_map(root, "WFI01")
            self.assertAlmostEqual(values[1, 1], 2.5)
            self.assertTrue(np.isnan(values[0, 0]))

    def test_loads_legacy_layout_and_ignores_fixed_angles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = (
                root
                / "wfi02_absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares"
                / "fits"
            )
            self.write_archive(fit_root / "tile_000_least_squares", 0, 1.25)
            self.write_archive(
                fit_root / "tile_001_least_squares", 1, 7.5, fitted=False
            )
            values = plotter.load_incidence_angle_map(root, "WFI02")
            self.assertAlmostEqual(values[0, 0], 1.25)
            self.assertTrue(np.isnan(values[0, 1]))

    def test_loads_256_tile_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = root / "production" / "tile_size_256" / "wfi01" / "fits"
            for tile_id in range(256):
                self.write_archive(
                    fit_root / f"tile_{tile_id:03d}",
                    tile_id,
                    3.5 if tile_id == 255 else 1.0,
                )
            values = plotter.load_incidence_angle_map(root, "WFI01", tile_size=256)
            self.assertEqual(values.shape, (16, 16))
            self.assertAlmostEqual(values[15, 15], 3.5)


if __name__ == "__main__":
    unittest.main()
