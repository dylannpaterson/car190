import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import export_eric_rcs_trend_comparison as exporter


class EricRcsTrendExportTests(unittest.TestCase):
    def test_cli_defaults_to_eric_s3_prefix(self):
        args = exporter.parse_args([])

        self.assertIsNone(args.source_root)
        self.assertEqual(args.s3_uri, exporter.DEFAULT_S3_URI)

    def test_factor_values_anchor_each_step_to_first_product(self):
        baseline = {
            "photodiode": 1.0,
            "acm": 1.0,
            "acm_bridge": 1.0,
            "side_bridge": 0.9,
            "tempgain": 1.1,
        }
        factors = {
            "photodiode": 1.02,
            "acm": 0.98,
            "acm_bridge": 1.01,
            "side_bridge": 0.95,
            "tempgain": 1.2,
        }

        row = exporter._factor_values(factors, baseline)

        self.assertAlmostEqual(row["total_multiplier_no_tempgain"], 1.02 * 0.98 * 1.01 * 0.95)
        self.assertAlmostEqual(
            row["log_side_bridge_ratio_to_first"],
            np.log(0.95 / 0.9),
        )
        self.assertAlmostEqual(
            row["log_total_no_tempgain_ratio_to_first"],
            np.log((1.02 * 0.98 * 1.01 * 0.95) / 0.9),
        )
        self.assertAlmostEqual(
            row["log_total_calibrated_ratio_to_first"],
            np.log((1.02 * 0.98 * 1.01 * 0.95 * 1.2) / (0.9 * 1.1)),
        )

    def test_tile_136_maps_to_local_256_pixel_coordinates(self):
        tiles_per_axis, block_size = exporter._tile_geometry(256, 16, 256)

        self.assertEqual(tiles_per_axis, 16)
        self.assertEqual(block_size, 16)
        self.assertEqual(
            exporter._tile_slices(136, tiles_per_axis, block_size),
            (8, 8, 128, 144, 128, 144),
        )

    def test_rate_values_report_raw_and_separately_calibrated_trends(self):
        baseline = np.full((2, 2), 10.0)
        current = np.full((2, 2), 12.0)
        baseline_factors = {
            "photodiode": 1.0,
            "acm": 1.0,
            "acm_bridge": 1.0,
            "side_bridge": 1.0,
            "tempgain": 1.0,
        }
        factors = {
            "photodiode": 1.1,
            "acm": 1.0,
            "acm_bridge": 1.0,
            "side_bridge": 1.0,
            "tempgain": 1.2,
        }

        row = exporter._rate_values(
            current,
            baseline,
            factors,
            baseline_factors,
        )

        self.assertAlmostEqual(row["raw_mean_dn_s"], 12.0)
        self.assertAlmostEqual(row["raw_log_change_mean"], np.log(1.2))
        self.assertAlmostEqual(row["no_tempgain_mean_dn_s"], 13.2)
        self.assertAlmostEqual(
            row["no_tempgain_log_change_mean"],
            np.log(1.2 * 1.1),
        )
        self.assertAlmostEqual(row["calibrated_mean_dn_s"], 15.84)
        self.assertAlmostEqual(
            row["calibrated_log_change_mean"],
            np.log(1.2 * 1.1 * 1.2),
        )

    @unittest.skipIf(exporter.h5py is None, "h5py is not installed")
    def test_hdf5_export_writes_compact_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "rcs_trend_v4"
            cal_dir = source_root / "cal"
            raw_dir = source_root / "h5"
            cal_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            for index, scale in enumerate((1.0, 1.1)):
                cal_path = cal_dir / f"epoch_{index:03d}_cal.h5"
                raw_path = raw_dir / f"epoch_{index:03d}_raw.h5"
                raw = np.full((18, 4, 4), 10.0 * scale)
                with exporter.h5py.File(cal_path, "w") as handle:
                    handle.attrs["timestamp"] = (
                        f"2026-09-{12 + index:02d}T00:00:00Z"
                    )
                    handle.attrs["run_label"] = f"run{index}"
                    handle.attrs["bin_factor"] = 16
                    for led in range(1, 7):
                        calibrated_scale = 0.9 if index == 1 else 1.0
                        handle.create_dataset(
                            f"cal/LED{led}",
                            data=raw * calibrated_scale,
                        )
                        for step in exporter.CALIBRATION_STEPS:
                            value = 1.0
                            if step == "side_bridge" and index == 1:
                                value = 0.9
                            handle.attrs[f"cal_{step}_LED{led}"] = value
                with exporter.h5py.File(raw_path, "w") as handle:
                    handle.attrs["bin_factor"] = 16
                    for led in range(1, 7):
                        handle.create_dataset(f"raw/LED{led}", data=raw)
            output = Path(directory) / "export"

            manifest = exporter.export_bundle(
                source_root,
                output,
                detectors=("WFI01",),
                tile_ids=(0,),
                tile_size=64,
                include_sca_summary=True,
            )

            self.assertEqual(manifest["products_selected"], 2)
            self.assertTrue((output / "manifest.json").is_file())
            with (output / "eric_v4_corrections.csv").open(newline="") as handle:
                correction_rows = list(csv.DictReader(handle))
            self.assertEqual(len(correction_rows), 12)
            with (output / "eric_v4_tile_trends.csv").open(newline="") as handle:
                tile_rows = list(csv.DictReader(handle))
            self.assertEqual(len(tile_rows), 12)
            with (output / "eric_v4_sca_trends.csv").open(newline="") as handle:
                sca_rows = list(csv.DictReader(handle))
            self.assertEqual(len(sca_rows), 12)
            self.assertAlmostEqual(
                float(tile_rows[-1]["raw_log_change_mean"]),
                np.log(1.1),
            )

    @unittest.skipIf(exporter.h5py is None, "h5py is not installed")
    def test_s3_export_uses_a_temporary_source_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote"
            cal_path = remote / "cal" / "epoch_000_cal.h5"
            raw_path = remote / "h5" / "epoch_000_raw.h5"
            cal_path.parent.mkdir(parents=True)
            raw_path.parent.mkdir(parents=True)
            with exporter.h5py.File(cal_path, "w") as handle:
                handle.attrs["timestamp"] = "2026-09-12T00:00:00Z"
                handle.attrs["bin_factor"] = 16
                raw = np.full((18, 4, 4), 10.0)
                for step in exporter.CALIBRATION_STEPS:
                    values = np.ones((6, 18))
                    if step == "side_bridge":
                        values[:, :] = 1.1
                    handle.create_dataset(
                        f"calibration/{step}/multiplier",
                        data=values,
                    )
                for led in range(1, 7):
                    handle.create_dataset(
                        f"sca_arrays/LED{led}",
                        data=raw * 1.1,
                    )
            with exporter.h5py.File(raw_path, "w") as handle:
                handle.attrs["bin_factor"] = 16
                raw = np.full((18, 4, 4), 10.0)
                for led in range(1, 7):
                    handle.create_dataset(f"sca_arrays/LED{led}", data=raw)

            objects = [
                {"Key": "prefix/", "Size": 0},
                {
                    "Key": "prefix/digests/digests_v4_10_12_26",
                    "Size": 0,
                },
                {
                    "Key": "prefix/cal/epoch_000_cal.h5",
                    "Size": cal_path.stat().st_size,
                },
                {
                    "Key": "prefix/h5/epoch_000_raw.h5",
                    "Size": raw_path.stat().st_size,
                },
            ]

            def fake_download(arguments, aws_cli="aws", aws_profile=None):
                self.assertEqual(arguments[0:2], ("s3", "cp"))
                key = str(arguments[2]).split("prefix/", 1)[1]
                shutil.copy2(remote / key, Path(arguments[3]))
                return ""

            output = root / "export"
            with patch.object(
                exporter,
                "_list_s3_objects",
                return_value=("bucket", "prefix/", objects),
            ), patch.object(exporter, "_run_aws", side_effect=fake_download):
                manifest = exporter.export_s3_bundle(
                    "s3://bucket/prefix/",
                    output,
                    detectors=("WFI01",),
                    tile_ids=(0,),
                    tile_size=64,
                    copy_metadata=False,
                )

            self.assertEqual(manifest["source_mode"], "s3")
            self.assertEqual(manifest["source_uri"], "s3://bucket/prefix/")
            self.assertTrue((output / "s3_inventory.json").is_file())
            self.assertFalse(
                any(path.name.startswith(".eric_v4_s3_source_") for path in output.iterdir())
            )


if __name__ == "__main__":
    unittest.main()
