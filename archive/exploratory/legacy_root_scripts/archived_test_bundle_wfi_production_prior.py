import tempfile
import unittest
from pathlib import Path

import numpy as np

import bundle_wfi_production_prior as bundler


class WfiProductionBundleTests(unittest.TestCase):
    def write_archives(
        self,
        root,
        layout,
        tile_size=bundler.DEFAULT_TILE_SIZE,
        tile_count=64,
    ):
        fit_root = root / "production"
        if tile_size != bundler.DEFAULT_TILE_SIZE:
            fit_root /= f"tile_size_{tile_size}"
        fit_root /= "wfi01"
        fit_root /= "fits"
        if layout == "legacy":
            fit_root = (
                root
                / "wfi01_absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares"
                / "fits"
            )
        for tile_id in range(tile_count):
            suffix = "_least_squares" if layout == "legacy" else ""
            path = fit_root / f"tile_{tile_id:03d}{suffix}"
            path.mkdir(parents=True)
            np.savez(path / "posterior_samples.npz")

    def test_timestamp_is_explicit_utc_mjd(self):
        value = bundler.canonical_utc("2026-09-12T01:25:50.7140000")
        self.assertEqual(value, "2026-09-12T01:25:50.7140000Z")
        self.assertAlmostEqual(
            bundler.timestamp_mjd_utc(value), 61295.05961474537, places=12
        )

    def test_complete_new_layout_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_archives(root, "new")
            archives, layout = bundler.find_fit_archives(root, "WFI01")
            self.assertEqual(layout, "production")
            self.assertEqual(sorted(archives), list(range(64)))

    def test_complete_legacy_layout_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_archives(root, "legacy")
            archives, layout = bundler.find_fit_archives(root, "WFI01")
            self.assertEqual(layout, "legacy_independent_production")
            self.assertEqual(sorted(archives), list(range(64)))

    def test_complete_alternate_tile_layout_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_archives(root, "new", tile_size=512, tile_count=64)
            archives, layout = bundler.find_fit_archives(
                root,
                "WFI01",
                tile_size=512,
                expected_tile_ids=range(64),
            )
            self.assertEqual(layout, "production")
            self.assertEqual(sorted(archives), list(range(64)))

    def test_ar_schema_exposes_raw_and_physical_thickness(self):
        names = {name for name, _fmt, _unit, _description in bundler.bundle_schemas()["PARAMETERS"]}
        for layer in range(1, 4):
            self.assertIn(f"ar_layer{layer}_thickness_scale_100nm", names)
            self.assertIn(f"ar_layer{layer}_thickness_nm", names)

    def test_hgcdte_schema_exposes_reference_and_physical_values(self):
        schemas = bundler.bundle_schemas(3)
        reference_names = {
            name for name, _fmt, _unit, _description in schemas["HGCDTE_REFERENCE"]
        }
        parameter_specs = {
            name: (fmt, unit)
            for name, fmt, unit, _description in schemas["PARAMETERS"]
        }
        self.assertEqual(reference_names, {"wavelength_nm", "reference_n", "reference_k"})
        self.assertEqual(parameter_specs["absolute_hgcdte_n"], ("3D", ""))
        self.assertEqual(parameter_specs["absolute_hgcdte_k"], ("3D", ""))

    def test_hgcdte_physical_values_reconstruct_from_log_scales(self):
        reference_n = np.asarray([2.0, 2.1, 2.2])
        reference_k = np.asarray([0.1, 0.2, 0.3])
        physical_n, physical_k = bundler.reconstruct_hgcdte(
            reference_n, reference_k, np.log(1.5), np.log(2.0)
        )
        np.testing.assert_allclose(physical_n, reference_n * 1.5)
        np.testing.assert_allclose(physical_k, reference_k * 2.0)


if __name__ == "__main__":
    unittest.main()
