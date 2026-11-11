import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import export_wfi_production_ice_csv as exporter


class ExportWfiProductionIceCsvTests(unittest.TestCase):
    def test_infer_date_label(self):
        self.assertEqual(
            exporter.infer_date_label("/workspace/run/20261101/corrected_statistics"),
            "20261101",
        )
        with self.assertRaises(ValueError):
            exporter.infer_date_label("/workspace/run/corrected_statistics")

    def write_inputs(self, root):
        statistics_dir = root / "statistics"
        statistics_dir.mkdir()
        statistics_path = (
            statistics_dir / "wfi01_tile_log_statistics_256_count_corrected.csv"
        )
        fields = (
            "tile_id",
            "tile_row",
            "tile_col",
            "row_start",
            "row_stop_exclusive",
            "col_start",
            "col_stop_exclusive",
            "tile_n_rows",
            "tile_n_cols",
            "group",
            "led",
            "repeat",
            "file_set_name",
            "baseline_file_set_name",
            "exposure_start_time",
            "days",
        )
        with statistics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(3):
                writer.writerow(
                    {
                        "tile_id": 0,
                        "tile_row": 0,
                        "tile_col": 0,
                        "row_start": 0,
                        "row_stop_exclusive": 256,
                        "col_start": 0,
                        "col_stop_exclusive": 256,
                        "tile_n_rows": 256,
                        "tile_n_cols": 256,
                        "group": index,
                        "led": 1,
                        "repeat": index,
                        "file_set_name": f"current_{index}",
                        "baseline_file_set_name": f"baseline_{index}",
                        "exposure_start_time": f"2026-10-{10 + index:02d}T00:00:00",
                        "days": float(index),
                    }
                )

        fit_dir = (
            root / "production" / "wfi01" / "relative_response" / "fits" / "tile_000"
        )
        fit_dir.mkdir(parents=True)
        np.savez_compressed(
            fit_dir / "posterior_samples.npz",
            observation_days=np.asarray([0.0, 1.0, 2.0]),
            tile_ice_thickness=np.asarray([1.0, 2.0, 3.0]),
            tile_instantaneous_rate_nm_day=np.asarray([4.0, 5.0, 6.0]),
            np_node_thickness=np.asarray([1.0, 2.0, 3.0]),
            node_instantaneous_rate_nm_day=np.asarray([4.0, 5.0, 6.0]),
            initial_thickness_log_nm=np.asarray([np.log(1.0)]),
            ice_reset_day=np.asarray([0.5]),
            third_decon_day=np.asarray([1.5]),
            fit_mode=np.asarray("relative_response_warmup_centered_thickness_prior"),
            constant_rate_map_warmup=np.asarray(1),
        )
        return statistics_dir, root

    def test_current_and_trajectory_rows_include_mpa_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            statistics_dir, fit_root = self.write_inputs(Path(temporary))
            current, trajectory = exporter.collect_rows(
                statistics_dir, fit_root, detectors=("WFI01",)
            )

            self.assertEqual(len(current), 1)
            self.assertEqual(len(trajectory), 3)
            self.assertAlmostEqual(current[0]["mpa_x_mm"], -2.96)
            self.assertAlmostEqual(current[0]["mpa_y_mm"], -7.03)
            self.assertEqual(current[0]["ice_thickness_nm"], 3.0)
            self.assertEqual(current[0]["accumulation_rate_nm_day"], 6.0)
            self.assertTrue(current[0]["post_third_decon"])
            self.assertEqual(trajectory[-1]["days_since_first_exposure"], 2.0)
            self.assertEqual(trajectory[-1]["node_index"], 2)
            self.assertNotIn("led", trajectory[0])

    def test_write_rows_uses_stable_spreadsheet_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            statistics_dir, fit_root = self.write_inputs(Path(temporary))
            current, _trajectory = exporter.collect_rows(
                statistics_dir, fit_root, detectors=("WFI01",)
            )
            output = Path(temporary) / "current.csv"
            exporter.write_rows(output, current, exporter.CURRENT_FIELDS)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(tuple(rows[0]), exporter.CURRENT_FIELDS)
            self.assertEqual(rows[0]["mpa_x_mm"], "-2.96")
            self.assertEqual(rows[0]["ice_thickness_nm"], "3.0")

    def test_streaming_export_writes_all_epochs_and_optional_current_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            statistics_dir, fit_root = self.write_inputs(Path(temporary))
            trajectory_path = Path(temporary) / "trajectory.csv"
            current_path = Path(temporary) / "current.csv"
            counts = exporter.write_export(
                statistics_dir,
                fit_root,
                trajectory_path,
                current_path=current_path,
                detectors=("WFI01",),
            )

            self.assertEqual(counts, (1, 3))
            with trajectory_path.open(newline="") as handle:
                trajectory = list(csv.DictReader(handle))
            with current_path.open(newline="") as handle:
                current = list(csv.DictReader(handle))
            self.assertEqual([row["node_index"] for row in trajectory], ["0", "1", "2"])
            self.assertEqual(trajectory[-1]["ice_thickness_nm"], "3.0")
            self.assertEqual(current[0]["latest_node_index"], "2")


if __name__ == "__main__":
    unittest.main()
