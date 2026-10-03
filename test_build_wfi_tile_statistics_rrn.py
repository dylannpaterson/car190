import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from build_wfi_tile_statistics_rrn import (
    FIELDS,
    attach_photodiode_corrections,
    build_observations,
    fetch_photodiode_telemetry,
    load_ice_monitor_photodiode_summary,
    parse_iso_time,
    parse_product_name,
    resolve_record_times,
    scan_products,
    tile_sufficient_statistics,
    update_detector_statistics_incremental,
    write_detector_statistics,
    write_inventory,
)


class RrnFilenameTests(unittest.TestCase):
    def test_tile_statistics_include_raw_counts_and_zero_self_ratio(self):
        image = np.asarray([[2.0, 4.0], [8.0, 16.0]])
        dq = np.zeros_like(image, dtype=np.uint32)
        tile = next(iter(tile_sufficient_statistics(image, dq, image, dq, 2)))

        self.assertTrue(set(FIELDS).issuperset(tile))
        self.assertEqual(tile["n_pair_valid"], 4)
        self.assertEqual(tile["sum_reference_counts"], 30.0)
        self.assertEqual(tile["sum_current_counts"], 30.0)
        self.assertEqual(tile["sum_reference_current_counts"], 340.0)
        self.assertEqual(tile["sum_log_change"], 0.0)
        self.assertEqual(tile["sum_squared_log_change"], 0.0)

    def test_tile_statistics_apply_positive_current_scale(self):
        image = np.asarray([[2.0, 4.0], [8.0, 16.0]])
        dq = np.zeros_like(image, dtype=np.uint32)
        tile = next(iter(tile_sufficient_statistics(image, dq, image, dq, 2, 2.0)))

        self.assertEqual(tile["sum_reference_counts"], 30.0)
        self.assertEqual(tile["sum_current_counts"], 60.0)
        self.assertEqual(tile["sum_squared_current_counts"], 1360.0)
        self.assertEqual(tile["sum_reference_current_counts"], 680.0)
        self.assertAlmostEqual(tile["sum_log_change"], 4.0 * np.log(2.0))

    def test_photodiode_corrections_are_normalized_per_led(self):
        observations = [
            {
                "led": 1,
                "group": 0,
                "file_set_name": "led1_first",
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "exposure_end_time": "2026-01-01T00:01:00Z",
            },
            {
                "led": 1,
                "group": 1,
                "file_set_name": "led1_second",
                "exposure_start_time": "2026-01-01T01:00:00Z",
                "exposure_end_time": "2026-01-01T01:01:00Z",
            },
            {
                "led": 2,
                "group": 2,
                "file_set_name": "led2_first",
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "exposure_end_time": "2026-01-01T00:01:00Z",
            },
            {
                "led": 2,
                "group": 3,
                "file_set_name": "led2_second",
                "exposure_start_time": "2026-01-01T01:00:00Z",
                "exposure_end_time": "2026-01-01T01:01:00Z",
            },
        ]
        telemetry = {
            "channel": "pd1",
            "values": [
                (datetime(2026, 1, 1, 0, 0), 10.0),
                (datetime(2026, 1, 1, 0, 1), 10.0),
                (datetime(2026, 1, 1, 1, 0), 8.0),
                (datetime(2026, 1, 1, 1, 1), 8.0),
            ],
        }
        attach_photodiode_corrections(observations, telemetry)

        self.assertEqual(observations[0]["photodiode_scale"], 1.0)
        self.assertEqual(observations[1]["photodiode_scale"], 1.25)

    def test_ice_monitor_summary_is_dark_subtracted_and_normalized(self):
        fields = (
            "mnemonic",
            "step",
            "led",
            "start",
            "end",
            "signal",
            "dark_median",
            "dark_subtracted",
            "n_samples",
            "ref_pass",
            "ref_signal",
            "change_pct",
        )
        rows = [
            {
                "mnemonic": "WFI_MCE_SRCS_PD1_V",
                "step": "band1",
                "led": "1",
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-01T00:01:00Z",
                "signal": "10.0",
                "dark_median": "0.2",
                "dark_subtracted": "True",
                "n_samples": "100",
                "ref_pass": "0103902002",
                "ref_signal": "10.0",
                "change_pct": "0.0",
            },
            {
                "mnemonic": "WFI_MCE_SRCS_PD1_V",
                "step": "band1",
                "led": "1",
                "start": "2026-01-01T01:00:00Z",
                "end": "2026-01-01T01:01:00Z",
                "signal": "8.0",
                "dark_median": "0.3",
                "dark_subtracted": "True",
                "n_samples": "101",
                "ref_pass": "0103902002",
                "ref_signal": "10.0",
                "change_pct": "-20.0",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ice_monitor.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            telemetry = load_ice_monitor_photodiode_summary(path, "pd1")

        observations = [
            {
                "led": 1,
                "group": 0,
                "file_set_name": "led1_first",
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "exposure_end_time": "2026-01-01T00:01:00Z",
            },
            {
                "led": 1,
                "group": 1,
                "file_set_name": "led1_second",
                "exposure_start_time": "2026-01-01T01:00:00Z",
                "exposure_end_time": "2026-01-01T01:01:00Z",
            },
        ]
        attach_photodiode_corrections(observations, telemetry)

        self.assertEqual(observations[0]["photodiode_scale"], 1.0)
        self.assertEqual(observations[1]["photodiode_scale"], 1.25)
        self.assertEqual(observations[1]["photodiode_dark_median"], 0.3)
        self.assertEqual(observations[1]["photodiode_source"], "roman-telem-ice-monitor")
        self.assertEqual(observations[1]["photodiode_reference_pass"], "0103902002")

    def test_parse_iso_time_normalizes_timezone_offsets(self):
        self.assertEqual(
            parse_iso_time("2026-01-01T01:00:00+01:00"),
            datetime(2026, 1, 1, 0, 0),
        )

    def test_detector_writer_uses_photodiode_scale_for_current_image(self):
        observations = []
        for led in range(1, 7):
            for repeat in range(3):
                file_set_name = f"led{led}_{repeat}"
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": len(observations),
                        "file_set_name": file_set_name,
                        "baseline_file_set_name": f"led{led}_0",
                        "exposure_start_time": f"2026-01-01T{repeat:02d}:00:00Z",
                        "days": float(repeat),
                        "cal_uri": file_set_name,
                        "source": "mast",
                        "path": None,
                        "photodiode_scale": 1.0 if repeat == 0 else 2.0,
                    }
                )

        image = np.asarray([[2.0, 4.0], [8.0, 16.0]])
        dq = np.zeros_like(image, dtype=np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "statistics.csv"
            write_detector_statistics(
                observations,
                output,
                tile_size=2,
                resume=False,
                fetch_image=lambda uri: (image.copy(), dq.copy()),
            )
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        current_sum = {
            int(row["repeat"]): float(row["sum_current_counts"])
            for row in rows
            if int(row["led"]) == 1
        }
        self.assertEqual(current_sum, {0: 30.0, 1: 60.0, 2: 60.0})

    def test_fetch_photodiode_telemetry_uses_index_window_and_token(self):
        rows = [
            {
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "exposure_end_time": "2026-01-01T00:01:00Z",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token.txt"
            token_path.write_text("MAST_API_TOKEN=secret\n")
            with patch("build_wfi_tile_statistics_rrn._query_photodiode_mnemonic") as query:
                query.return_value = [(datetime(2026, 1, 1, 0, 0), 0.4)]
                telemetry = fetch_photodiode_telemetry(rows, token_path, "pd1")

        query.assert_called_once()
        token, mnemonic, start, end = query.call_args.args
        self.assertEqual(token, "secret")
        self.assertEqual(mnemonic, "WFI_MCE_SRCS_PD1_V")
        self.assertEqual(start, datetime(2025, 12, 31, 23, 59))
        self.assertEqual(end, datetime(2026, 1, 1, 0, 2))
        self.assertEqual(telemetry["channel"], "pd1")

    def test_parse_product_name_derives_detector_led_and_epoch(self):
        path = Path("r0103902017001007001_0001_wfi01_dark_cal.asdf")
        record = parse_product_name(path)
        self.assertEqual(record["detector"], "WFI01")
        self.assertEqual(record["led"], 6)
        self.assertEqual(record["epoch_key"], "010390201700100")
        self.assertEqual(record["file_set_name"], path.name[:-len("_cal.asdf")])

    def test_scan_filters_detector_and_rejects_duplicate_file_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "r0103902001001002001_0001_wfi01_dark_cal.asdf").touch()
            (root / "r0103902001001003001_0001_wfi02_dark_cal.asdf").touch()
            records = scan_products(root, "WFI01")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["detector"], "WFI01")

    def test_build_observations_orders_epochs_and_assigns_baselines(self):
        records = []
        for epoch, hour in enumerate((1, 5, 9)):
            for led in range(1, 7):
                visit_id = f"01039020{epoch + 1:02d}00100{led + 1}001"
                path = Path(
                    f"r{visit_id}_0001_wfi01_dark_cal.asdf"
                )
                records.append(
                    {
                        "path": path,
                        "file_name": path.name,
                        "file_set_name": path.name[:-len("_cal.asdf")],
                        "visit_id": visit_id,
                        "detector": "WFI01",
                        "led": led,
                        "epoch_key": visit_id[:-4],
                        "exposure_start_time": f"2026-01-01T{hour:02d}:00:00Z",
                    }
                )
        observations = build_observations(records)
        self.assertEqual(len(observations), 18)
        self.assertEqual(sorted({row["repeat"] for row in observations}), [0, 1, 2])
        self.assertEqual(observations[0]["led"], 1)
        self.assertEqual(observations[0]["days"], 0.0)
        self.assertEqual(
            observations[0]["baseline_file_set_name"], observations[0]["file_set_name"]
        )
        self.assertEqual(observations[6]["baseline_file_set_name"], observations[0]["file_set_name"])

    def test_inventory_records_optional_index_timestamp_source(self):
        path = Path("r0103902001001002001_0001_wfi01_dark_cal.asdf")
        record = parse_product_name(path)
        index = {
            record["file_set_name"]: {
                "fileSetName": record["file_set_name"],
                "detector": "WFI01",
                "led": "1",
                "exposure_start_time": "2026-01-01T00:00:00Z",
            }
        }
        resolve_record_times([record], index)
        self.assertEqual(record["timestamp_source"], "optional_index")
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / "inventory.csv"
            write_inventory(
                inventory,
                [record],
                [
                    {
                        "file_set_name": record["file_set_name"],
                        "led": 1,
                        "repeat": 0,
                        "exposure_start_time": record["exposure_start_time"],
                        "days": 0.0,
                    }
                ],
            )
            self.assertIn("optional_index", inventory.read_text())

    def test_incremental_writer_reuses_existing_file_sets(self):
        def observations(repeats):
            rows = []
            for led in range(1, 7):
                for repeat in repeats:
                    name = f"led{led}_repeat{repeat}"
                    rows.append(
                        {
                            "led": led,
                            "repeat": repeat,
                            "group": len(rows),
                            "file_set_name": name,
                            "baseline_file_set_name": f"led{led}_repeat0",
                            "exposure_start_time": (
                                f"2026-01-01T{repeat:02d}:00:00Z"
                            ),
                            "days": float(repeat),
                            "path": Path(name),
                        }
                    )
            return rows

        image = np.ones((4, 4), dtype=float)
        dq = np.zeros((4, 4), dtype=np.uint32)

        def load_image(observation, fetch_image=None):
            repeat = int(observation["repeat"])
            return image + 0.1 * repeat, dq

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "statistics.csv"
            initial = observations((0, 1))
            all_rows = observations((0, 1, 2))
            with patch(
                "build_wfi_tile_statistics_rrn.load_observation_image",
                side_effect=load_image,
            ):
                write_detector_statistics(initial, output, tile_size=2, resume=False)
                update_detector_statistics_incremental(
                    all_rows,
                    output,
                    tile_size=2,
                )
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 6 * 3 * 4)
        added = [
            row
            for row in rows
            if row["file_set_name"] == "led1_repeat2"
        ]
        self.assertEqual(len(added), 4)
        self.assertAlmostEqual(
            float(added[-1]["sum_log_change"]), 4.0 * np.log(1.2), places=12
        )

if __name__ == "__main__":
    unittest.main()
