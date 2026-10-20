import tempfile
import unittest
from pathlib import Path

import numpy as np

import plot_wfi_post_reset_initial_thickness_mosaic as plotter


class PostResetInitialThicknessMosaicTests(unittest.TestCase):
    def write_archive(self, path, thickness=None, log_thickness=None):
        path.mkdir(parents=True)
        values = {}
        if thickness is not None:
            values["reset_initial_thickness_nm"] = np.asarray([thickness])
        if log_thickness is not None:
            values["reset_initial_thickness_log_nm"] = np.asarray([log_thickness])
        np.savez(path / "posterior_samples.npz", **values)

    def write_complete_legacy_archives(self, root):
        fit_root = (
            root
            / "wfi01_absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares"
            / "fits"
        )
        for tile_id in range(64):
            self.write_archive(
                fit_root / f"tile_{tile_id:03d}_least_squares",
                thickness=float(tile_id + 1),
            )

    def test_loads_current_log_parameter_and_converts_to_nm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = root / "production" / "wfi01" / "fits"
            for tile_id in range(64):
                self.write_archive(
                    fit_root / f"tile_{tile_id:03d}",
                    log_thickness=np.log(12.5 if tile_id == 9 else 5.0),
                )
            values = plotter.load_post_reset_initial_thickness_map(root, "WFI01")
            self.assertAlmostEqual(values[1, 1], 12.5)

    def test_prefers_complete_legacy_layout_over_partial_current_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_archive(
                root / "production" / "wfi01" / "fits" / "tile_009",
                thickness=999.0,
            )
            self.write_complete_legacy_archives(root)
            values = plotter.load_post_reset_initial_thickness_map(root, "WFI01")
            self.assertAlmostEqual(values[0, 0], 1.0)
            self.assertAlmostEqual(values[1, 1], 10.0)
            self.assertAlmostEqual(values[7, 7], 64.0)

    def test_draws_mosaic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mosaic.png"
            maps = {detector: None for detector in plotter.SCAS}
            maps["WFI01"] = np.arange(64, dtype=float).reshape(8, 8) + 1.0
            plotter.draw_post_reset_initial_thickness_mosaic(path, maps)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_loads_256_tile_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = root / "production" / "tile_size_256" / "wfi01" / "fits"
            for tile_id in range(256):
                self.write_archive(
                    fit_root / f"tile_{tile_id:03d}",
                    thickness=17.5 if tile_id == 255 else 5.0,
                )
            values = plotter.load_post_reset_initial_thickness_map(
                root,
                "WFI01",
                tile_size=256,
            )
            self.assertEqual(values.shape, (16, 16))
            self.assertAlmostEqual(values[15, 15], 17.5)


if __name__ == "__main__":
    unittest.main()
