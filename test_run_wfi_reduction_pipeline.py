import csv
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import run_wfi_reduction_pipeline as pipeline
from run_wfi_reduction_pipeline import (
    find_previous_statistics,
    attach_photodiode_corrections,
    load_ice_monitor_photodiode_summary,
    list_s3_products,
    run_detector,
    source_changes_require_rebuild,
    stream_s3_image,
    utc_run_date,
)


class ReductionPipelineTests(unittest.TestCase):
    def test_run_date_uses_compact_utc_date(self):
        self.assertEqual(utc_run_date("20261003"), "20261003")
        with self.assertRaises(ValueError):
            utc_run_date("2026-10-03")

    def test_s3_listing_does_not_create_a_local_image_cache(self):
        product = "r0103902017001007001_0001_wfi01_dark_cal.asdf"
        payload = json.dumps(
            {
                "Contents": [
                    {
                        "Key": f"activities/CAR190/{product}",
                        "Size": 123,
                        "ETag": '"etag"',
                        "LastModified": "2026-10-03T00:00:00+00:00",
                    }
                ]
            }
        )
        with patch("run_wfi_reduction_pipeline.subprocess.run") as run:
            run.return_value.stdout = payload
            records = list_s3_products(
                "s3://roman-s3-romcom/activities/CAR190/",
                aws_binary="aws",
                aws_profile="nexus",
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "s3")
        self.assertIsNone(records[0]["path"])
        self.assertEqual(
            records[0]["s3_uri"],
            f"s3://roman-s3-romcom/activities/CAR190/{product}",
        )
        command = run.call_args.args[0]
        self.assertEqual(command[:5], ["aws", "--profile", "nexus", "s3api", "list-objects-v2"])
        self.assertIn("--prefix", command)
        self.assertIn("activities/CAR190/", command)
        run.assert_called_once_with(
            command,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

    def test_find_previous_statistics_ignores_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "20261002" / "statistics" / "wfi01_tile_log_statistics_512.csv"
            current = root / "20261003"
            old.parent.mkdir(parents=True)
            old.write_text("old")
            self.assertEqual(
                find_previous_statistics(root, current, "WFI01", 512),
                old,
            )

    def test_s3_image_stream_uses_stdout_without_writing_a_product(self):
        with patch("run_wfi_reduction_pipeline.subprocess.run") as run, \
             patch("run_wfi_reduction_pipeline.load_asdf_buffer", return_value=("data", "dq")) as load:
            run.return_value.returncode = 0
            run.return_value.stdout = b"asdf-bytes"
            run.return_value.stderr = b""
            result = stream_s3_image("s3://bucket/path/image_cal.asdf")

        self.assertEqual(result, ("data", "dq"))
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            ["aws", "s3", "cp", "s3://bucket/path/image_cal.asdf", "-", "--only-show-errors"],
        )
        load.assert_called_once()
        self.assertEqual(load.call_args.args[1], "s3://bucket/path/image_cal.asdf")

    def test_photodiode_correction_requires_dark_subtracted_reference_signal(self):
        fields = (
            "mnemonic",
            "step",
            "led_band",
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
                "led_band": "1",
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
                "led_band": "1",
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
            summary_path = Path(directory) / "ice_monitor.csv"
            with summary_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            telemetry = load_ice_monitor_photodiode_summary(summary_path, "pd1")

        observations = [
            {
                "led": 1,
                "group": 0,
                "file_set_name": "first",
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "exposure_end_time": "2026-01-01T00:01:00Z",
            },
            {
                "led": 1,
                "group": 1,
                "file_set_name": "second",
                "exposure_start_time": "2026-01-01T01:00:00Z",
                "exposure_end_time": "2026-01-01T01:01:00Z",
            },
        ]
        attach_photodiode_corrections(observations, telemetry)
        self.assertEqual(observations[0]["photodiode_scale"], 1.0)
        self.assertEqual(observations[1]["photodiode_scale"], 1.25)
        self.assertEqual(observations[1]["photodiode_dark_median"], 0.3)
        self.assertEqual(observations[1]["photodiode_reference_pass"], "0103902002")
        self.assertEqual(observations[1]["photodiode_source"], "roman-telem-ice-monitor")

    def test_observations_preserve_s3_uri_for_streaming(self):
        records = []
        for led in range(1, 7):
            records.append(
                {
                    "path": None,
                    "file_name": f"led{led}.asdf",
                    "file_set_name": f"led{led}",
                    "visit_id": f"visit{led}",
                    "detector": "WFI01",
                    "led": led,
                    "epoch_key": "epoch0",
                    "source": "s3",
                    "s3_uri": f"s3://bucket/led{led}.asdf",
                    "cal_uri": f"s3://bucket/led{led}.asdf",
                    "exposure_start_time": "2026-01-01T00:00:00Z",
                    "exposure_end_time": "2026-01-01T00:00:01Z",
                }
            )
        records.extend(
            {
                **record,
                "file_name": f"{record['file_name']}_repeat",
                "file_set_name": f"{record['file_set_name']}_repeat",
                "epoch_key": "epoch1",
                "visit_id": f"{record['visit_id']}_repeat",
                "s3_uri": f"{record['s3_uri']}_repeat",
                "cal_uri": f"{record['cal_uri']}_repeat",
                "exposure_start_time": "2026-01-02T00:00:00Z",
                "exposure_end_time": "2026-01-02T00:00:01Z",
            }
            for record in records[:6]
        )
        records.extend(
            {
                **record,
                "file_name": f"{record['file_name']}_repeat2",
                "file_set_name": f"{record['file_set_name']}_repeat2",
                "epoch_key": "epoch2",
                "visit_id": f"{record['visit_id']}_repeat2",
                "s3_uri": f"{record['s3_uri']}_repeat2",
                "cal_uri": f"{record['cal_uri']}_repeat2",
                "exposure_start_time": "2026-01-03T00:00:00Z",
                "exposure_end_time": "2026-01-03T00:00:01Z",
            }
            for record in records[:6]
        )
        observations = pipeline.build_observations(records)
        self.assertTrue(all(row["s3_uri"] for row in observations))

    def test_source_signature_detects_replacement_but_allows_new_products(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / "inventory.csv"
            inventory.write_text(
                "file_set_name,source_signature,photodiode_scale\n"
                "old,10:20,\n"
            )
            records = [{"file_set_name": "old", "source_signature": "10:20"}]
            observations = [{"file_set_name": "old"}]
            self.assertFalse(
                source_changes_require_rebuild(inventory, records, observations)
            )

            records.append({"file_set_name": "new", "source_signature": "1:2"})
            observations.append({"file_set_name": "new"})
            self.assertFalse(
                source_changes_require_rebuild(inventory, records, observations)
            )

            records[0]["source_signature"] = "11:21"
            self.assertTrue(
                source_changes_require_rebuild(inventory, records, observations)
            )

    def test_detector_wires_standalone_statistics_and_correction_outputs(self):
        records = []
        index_metadata = {}
        for repeat in range(3):
            timestamp = (
                datetime(2026, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=repeat)
            ).isoformat().replace("+00:00", "Z")
            for led in range(1, 7):
                file_set_name = f"epoch{repeat}_led{led}"
                record = {
                    "path": Path(file_set_name),
                    "file_name": f"{file_set_name}_cal.asdf",
                    "file_set_name": file_set_name,
                    "visit_id": f"visit{repeat}{led}",
                    "detector": "WFI01",
                    "led": led,
                    "epoch_key": f"epoch{repeat}",
                    "source": "rrn",
                    "source_signature": "1:1",
                }
                records.append(record)
                index_metadata[file_set_name] = {
                    "fileSetName": file_set_name,
                    "detector": "WFI01",
                    "led": str(led),
                    "exposure_start_time": timestamp,
                    "exposure_end_time": timestamp,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "20261003"
            manifest = {"detectors": {}}
            with patch.object(pipeline, "DEFAULT_TILE_SIZES", (512,)), \
                 patch.object(pipeline, "mast_records_from_index", return_value=[]), \
                 patch.object(pipeline, "attach_photodiode_corrections"), \
                 patch.object(pipeline, "reduce_statistics", return_value="rebuilt"), \
                 patch.object(pipeline, "build_corrections", return_value=([], [])), \
                 patch.object(pipeline, "write_csv"), \
                 patch.object(pipeline, "write_corrected_statistics") as write_corrected:
                run_detector(
                    records,
                    run_dir,
                    root,
                    "WFI01",
                    index_metadata,
                    [],
                    None,
                    None,
                    manifest,
                )

        self.assertIn("WFI01", manifest["detectors"])
        self.assertEqual(write_corrected.call_count, 1)
        self.assertEqual(len(write_corrected.call_args.args), 3)


if __name__ == "__main__":
    unittest.main()
