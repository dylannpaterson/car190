import csv
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import run_wfi_reduction_pipeline as pipeline
from run_wfi_reduction_pipeline import (
    attach_photodiode_corrections,
    build_corrections,
        build_observations,
        load_ice_monitor_photodiode_summary,
        list_s3_products,
        merge_telemetry_summaries,
        merge_product_records,
    read_index_rows,
    resolve_record_times,
    run_detector,
    source_changes_require_rebuild,
    stream_s3_image,
    utc_run_date,
    write_side_swap_scaled_statistics,
)


class ReductionPipelineTests(unittest.TestCase):
    def test_unified_index_requires_both_programs(self):
        fields = (
            "fileSetName",
            "detector",
            "led",
            "visit_id",
            "exposure_start_time",
            "program",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for program in ("1039", "1063"):
                    writer.writerow(
                        {
                            "fileSetName": f"{program}_product",
                            "detector": "WFI01",
                            "led": "1",
                            "visit_id": f"{program}visit",
                            "exposure_start_time": "2026-01-01T00:00:00Z",
                            "program": program,
                        }
                    )
            rows = read_index_rows(path, require_program=True)

        self.assertEqual({row["program"] for row in rows}, {"1039", "1063"})

    def test_detector_cli_selects_one_detector(self):
        self.assertEqual(
            pipeline.parse_args(["--detector", "WFI01"]).detector,
            "WFI01",
        )
        self.assertEqual(pipeline.parse_args(["--workers", "4"]).workers, 4)
        self.assertEqual(
            pipeline.parse_args(["--run-root", "/tmp/car190"]).run_root,
            Path("/tmp/car190"),
        )
        self.assertIsNone(pipeline.parse_args([]).detector)
        with self.assertRaises(SystemExit):
            pipeline.parse_args(["--detector", "WFI99"])
        with self.assertRaises(SystemExit):
            pipeline.parse_args(["--workers", "0"])

    def test_main_dispatches_all_detectors_to_worker_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.csv"
            index.write_text("index")
            index_rows = [
                {"detector": detector, "program": program}
                for detector in pipeline.DEFAULT_DETECTORS
                for program in pipeline.UNIFIED_PROGRAMS
            ]
            telemetry = {
                "values": {},
                "source": "roman-telem-ice-monitor",
                "channel": "pd2",
                "mnemonic": "WFI_MCE_SRCS_PD2_V",
                "summary_path": "telemetry",
                "reference_mode": "first_non_warmup_per_led",
                "reference_values": {},
            }

            def worker(*arguments):
                return {"detector": arguments[3]}

            with patch.object(
                pipeline,
                "parse_args",
                return_value=SimpleNamespace(
                    detector=None,
                    workers=2,
                    run_root=root / "runs",
                ),
            ), patch.object(pipeline, "utc_run_date", return_value="20261007"), patch.object(
                pipeline, "DEFAULT_RUN_ROOT", root / "runs"
            ), patch.object(
                pipeline, "DEFAULT_MAST_INDEX", index
            ), patch.object(pipeline, "list_s3_products", return_value=[]), patch.object(
                pipeline, "refresh_index"
            ), patch.object(pipeline, "read_index_rows", return_value=index_rows), patch.object(
                pipeline, "read_optional_index", return_value={}
            ), patch.object(
                pipeline, "run_ice_monitor_telemetry", return_value=telemetry
            ), patch.object(
                pipeline, "merge_telemetry_summaries", return_value=telemetry
            ), patch.object(
                pipeline, "run_detector_worker", side_effect=worker
            ) as run_worker:
                pipeline.main([])
                manifest_path = root / "runs" / "20261007" / "pipeline_manifest.json"
                with manifest_path.open() as handle:
                    manifest = json.load(handle)

            self.assertEqual(run_worker.call_count, len(pipeline.DEFAULT_DETECTORS))
            self.assertEqual(manifest["detector_workers"]["used"], 2)
            self.assertEqual(
                set(manifest["detectors"]), set(pipeline.DEFAULT_DETECTORS)
            )
            self.assertEqual(manifest["photodiode_channel"], "pd2")

    def test_statistics_writer_preserves_raw_and_scaled_counts(self):
        observations = []
        for led in range(1, 7):
            for repeat in range(2):
                observations.append(
                    {
                        "led": led,
                        "program": "1039",
                        "repeat": repeat,
                        "group": (led - 1) * 2 + repeat,
                        "file_set_name": f"led{led}_repeat{repeat}",
                        "baseline_file_set_name": f"led{led}_repeat0",
                        "exposure_start_time": f"2026-01-01T00:{led:02d}:{repeat:02d}Z",
                        "days": float(repeat),
                        "source": "s3",
                        "photodiode_scale": 1.5 if repeat else 1.0,
                        "s3_uri": f"s3://example/led{led}_repeat{repeat}",
                    }
                )
        baseline = np.ones((4, 4), dtype=np.float32)
        dq = np.zeros((4, 4), dtype=np.uint32)

        def fetch_image(observation):
            repeat = int(observation["repeat"])
            return baseline * (1.0 if repeat == 0 else 2.0), dq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scaled = root / "statistics" / "wfi01.csv"
            raw = root / "raw_statistics" / "wfi01.csv"
            pipeline.write_detector_statistics(
                observations,
                scaled,
                tile_size=2,
                fetch_image=fetch_image,
                raw_output_path=raw,
            )

            with scaled.open(newline="") as handle:
                scaled_rows = list(csv.DictReader(handle))
            with raw.open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))

            scaled_repeat = next(row for row in scaled_rows if row["group"] == "1")
            raw_repeat = next(row for row in raw_rows if row["group"] == "1")
            self.assertAlmostEqual(float(raw_repeat["sum_current_counts"]), 8.0)
            self.assertAlmostEqual(float(scaled_repeat["sum_current_counts"]), 12.0)
            self.assertAlmostEqual(
                float(raw_repeat["sum_log_change"]), 4.0 * np.log(2.0)
            )
            self.assertAlmostEqual(
                float(scaled_repeat["sum_log_change"]), 4.0 * np.log(3.0)
            )

            def should_not_fetch(_observation):
                raise AssertionError("completed paired statistics were refetched")

            pipeline.write_detector_statistics(
                observations,
                scaled,
                tile_size=2,
                fetch_image=should_not_fetch,
                raw_output_path=raw,
            )

    def test_side_swap_scaling_rebuilds_scaled_output_without_changing_raw(self):
        fields = (
            "led",
            "repeat",
            "group",
            "file_set_name",
            "baseline_file_set_name",
            "days",
            "n_pair_valid",
            "sum_reference_counts",
            "sum_squared_reference_counts",
            "sum_current_counts",
            "sum_squared_current_counts",
            "sum_reference_current_counts",
            "min_reference_count",
            "max_reference_count",
            "min_current_count",
            "max_current_count",
            "sum_log_change",
            "sum_squared_log_change",
            "min_log_change",
            "max_log_change",
        )
        raw_row = {
            "led": 1,
            "repeat": 1,
            "group": 1,
            "file_set_name": "current",
            "baseline_file_set_name": "baseline",
            "days": 1.0,
            "n_pair_valid": 2,
            "sum_reference_counts": 4.0,
            "sum_squared_reference_counts": 8.0,
            "sum_current_counts": 8.0,
            "sum_squared_current_counts": 32.0,
            "sum_reference_current_counts": 16.0,
            "min_reference_count": 2.0,
            "max_reference_count": 2.0,
            "min_current_count": 4.0,
            "max_current_count": 4.0,
            "sum_log_change": 0.0,
            "sum_squared_log_change": 0.0,
            "min_log_change": np.log(2.0),
            "max_log_change": np.log(2.0),
        }
        observations = [
            {
                "led": 1,
                "repeat": 0,
                "file_set_name": "baseline",
                "photodiode_scale": 1.0,
            },
            {
                "led": 1,
                "repeat": 1,
                "file_set_name": "current",
                "photodiode_scale": 1.5,
            },
        ]
        side_swap_rows = [
            {
                "led": led,
                "channel": channel,
                "correction_factor": 2.0 if channel == "image" else 4.0,
            }
            for led in range(1, 7)
            for channel in ("image", "photodiode")
        ]
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.csv"
            scaled_path = Path(directory) / "scaled.csv"
            with raw_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(raw_row)

            write_side_swap_scaled_statistics(
                raw_path,
                scaled_path,
                observations,
                side_swap_rows,
                event_repeat=1,
            )
            with raw_path.open(newline="") as handle:
                raw_result = next(csv.DictReader(handle))
            with scaled_path.open(newline="") as handle:
                scaled_result = next(csv.DictReader(handle))

        self.assertAlmostEqual(float(raw_result["sum_current_counts"]), 8.0)
        self.assertAlmostEqual(float(scaled_result["sum_current_counts"]), 6.0)
        self.assertAlmostEqual(
            float(scaled_result["sum_log_change"]), 2.0 * np.log(0.75)
        )

    def test_incremental_statistics_updates_raw_and_scaled_companions(self):
        def make_observations(repeats):
            return [
                {
                    "led": led,
                    "program": "1039",
                    "repeat": repeat,
                    "group": (led - 1) * repeats + repeat,
                    "file_set_name": f"led{led}_repeat{repeat}",
                    "baseline_file_set_name": f"led{led}_repeat0",
                    "exposure_start_time": f"2026-01-01T{led:02d}:{repeat:02d}:00Z",
                    "days": float(repeat),
                    "source": "s3",
                    "photodiode_scale": 1.0 + 0.25 * repeat,
                    "s3_uri": f"s3://example/led{led}_repeat{repeat}",
                }
                for led in range(1, 7)
                for repeat in range(repeats)
            ]

        old_observations = make_observations(2)
        all_observations = make_observations(3)
        baseline = np.ones((2, 2), dtype=np.float32)
        dq = np.zeros((2, 2), dtype=np.uint32)

        def fetch_image(observation):
            return baseline * (1.0 + float(observation["repeat"])), dq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scaled = root / "statistics.csv"
            raw = root / "raw_statistics.csv"
            pipeline.write_detector_statistics(
                old_observations,
                scaled,
                tile_size=2,
                fetch_image=fetch_image,
                raw_output_path=raw,
            )
            fetched = []

            def fetch_incremental(observation):
                fetched.append(observation["file_set_name"])
                return fetch_image(observation)

            pipeline.update_detector_statistics_incremental(
                all_observations,
                scaled,
                tile_size=2,
                fetch_image=fetch_incremental,
                raw_output_path=raw,
            )
            with scaled.open(newline="") as handle:
                scaled_rows = list(csv.DictReader(handle))
            with raw.open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))

        self.assertEqual(len(fetched), 12)
        scaled_new = next(row for row in scaled_rows if row["group"] == "2")
        raw_new = next(row for row in raw_rows if row["group"] == "2")
        self.assertAlmostEqual(float(raw_new["sum_current_counts"]), 12.0)
        self.assertAlmostEqual(float(scaled_new["sum_current_counts"]), 18.0)

    def test_missing_raw_companion_rebuilds_existing_scaled_output(self):
        observations = [
            {
                "led": led,
                "program": "1039",
                "repeat": repeat,
                "group": (led - 1) * 3 + repeat,
                "file_set_name": f"led{led}_repeat{repeat}",
                "baseline_file_set_name": f"led{led}_repeat0",
                "exposure_start_time": f"2026-01-01T{led:02d}:{repeat:02d}:00Z",
                "days": float(repeat),
                "source": "s3",
                "photodiode_scale": 1.0 + 0.1 * repeat,
                "s3_uri": f"s3://example/led{led}_repeat{repeat}",
            }
            for led in range(1, 7)
            for repeat in range(3)
        ]
        records = [{"detector": "WFI01"}]
        baseline = np.ones((2, 2), dtype=np.float32)
        dq = np.zeros((2, 2), dtype=np.uint32)

        def fetch_image(observation):
            return baseline * (1.0 + float(observation["repeat"])), dq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "20261007"
            scaled = run_dir / "statistics" / "wfi01.csv"
            raw = run_dir / "raw_statistics" / "wfi01.csv"
            pipeline.write_detector_statistics(
                observations,
                scaled,
                tile_size=2,
                fetch_image=fetch_image,
            )
            pipeline.reduce_statistics(
                observations,
                records,
                scaled,
                tile_size=2,
                output_root=root,
                run_dir=run_dir,
                no_resume=False,
                fetch_image=fetch_image,
                raw_output_path=raw,
            )
            with raw.open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))

        self.assertFalse(scaled.exists())
        self.assertEqual(len(raw_rows), 18)

    def test_reduce_statistics_reuses_raw_only_and_ignores_stale_scaled_file(self):
        observations = [
            {
                "led": led,
                "program": "1039",
                "repeat": repeat,
                "group": (led - 1) * 3 + repeat,
                "file_set_name": f"led{led}_repeat{repeat}",
                "baseline_file_set_name": f"led{led}_repeat0",
                "exposure_start_time": f"2026-01-01T{led:02d}:{repeat:02d}:00Z",
                "days": float(repeat),
                "source": "s3",
                "photodiode_scale": 1.0 + 0.1 * repeat,
                "s3_uri": f"s3://example/led{led}_repeat{repeat}",
            }
            for led in range(1, 7)
            for repeat in range(3)
        ]
        old_observations = [row for row in observations if row["repeat"] < 2]
        records = [{"detector": "WFI01"}]
        baseline = np.ones((2, 2), dtype=np.float32)
        dq = np.zeros((2, 2), dtype=np.uint32)

        def fetch_image(observation):
            return baseline * (1.0 + float(observation["repeat"])), dq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_raw = (
                root
                / "20261006"
                / "raw_statistics"
                / "wfi01_tile_log_statistics_2.csv"
            )
            current_run = root / "20261007"
            current_scaled = (
                current_run / "statistics" / "wfi01_tile_log_statistics_2.csv"
            )
            current_raw = (
                current_run / "raw_statistics" / "wfi01_tile_log_statistics_2.csv"
            )
            pipeline.write_raw_detector_statistics(
                old_observations,
                previous_raw,
                tile_size=2,
                fetch_image=fetch_image,
            )
            stale_scaled = (
                root
                / "20261006"
                / "statistics"
                / "wfi01_tile_log_statistics_2.csv"
            )
            stale_scaled.parent.mkdir(parents=True, exist_ok=True)
            stale_scaled.write_text("this is deliberately not a statistics CSV")

            mode = pipeline.reduce_statistics(
                observations,
                records,
                current_scaled,
                tile_size=2,
                output_root=root,
                run_dir=current_run,
                no_resume=False,
                fetch_image=fetch_image,
                raw_output_path=current_raw,
            )
            with current_raw.open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))

        self.assertEqual(mode, "incremental")
        self.assertFalse(current_scaled.exists())
        raw_new = next(row for row in raw_rows if row["group"] == "2")
        self.assertAlmostEqual(float(raw_new["sum_current_counts"]), 12.0)

    def test_product_merge_prefers_s3_for_matching_physical_identity(self):
        s3_name = "r0106302001001002001_0001_wfi01_dark"
        mast_name = "r0106302001001002001_0001_wfi01_dark_mast"
        local = {
            "file_set_name": s3_name,
            "file_name": f"{s3_name}_cal.asdf",
            "detector": "WFI01",
            "visit_id": "0106302001001002001",
            "led": 1,
            "source": "s3",
        }
        mast = {
            "file_set_name": mast_name,
            "file_name": f"{mast_name}_cal.asdf",
            "detector": "WFI01",
            "visit_id": "0106302001001002001",
            "program": "1063",
            "led": 1,
            "source": "mast",
        }
        merged = merge_product_records([local], [mast])

        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0], local)
        self.assertEqual(merged[0]["mast_file_set_names"], [mast_name])
        resolved = resolve_record_times(
            merged,
            {
                mast_name: {
                    "fileSetName": mast_name,
                    "detector": "WFI01",
                    "visit_id": "0106302001001002001",
                    "led": "1",
                    "program": "1063",
                    "exposure_start_time": "2026-09-27T00:00:00Z",
                    "exposure_end_time": "2026-09-27T00:01:00Z",
                }
            },
        )
        self.assertEqual(resolved[0]["program"], "1063")

        second_exposure = {
            **mast,
            "file_set_name": "r0106302001001002001_0002_wfi01_dark",
            "file_name": "r0106302001001002001_0002_wfi01_dark_cal.asdf",
        }
        merged_exposures = merge_product_records(
            [local], [mast, second_exposure]
        )
        self.assertEqual(len(merged_exposures), 2)
        self.assertEqual(
            {row["file_set_name"] for row in merged_exposures},
            {s3_name, second_exposure["file_set_name"]},
        )

    def test_observations_split_reused_visit_prefix_when_led_repeats(self):
        records = []
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sequences = [
            ("shared", 0, (1, 2, 3, 4, 5, 6, 3)),
            ("next0", 10, (1, 2, 3, 4, 5, 6)),
            ("next1", 20, (1, 2, 3, 4, 5, 6)),
        ]
        for prefix, day_offset, leds in sequences:
            for index, led in enumerate(leds):
                visit_id = f"1063{prefix}{index:04d}"
                timestamp = start + timedelta(days=day_offset + index / 24.0)
                records.append(
                    {
                        "file_set_name": f"{prefix}_{index}",
                        "file_name": f"{prefix}_{index}.asdf",
                        "detector": "WFI01",
                        "visit_id": visit_id,
                        "program": "1063",
                        "led": led,
                        "epoch_key": prefix,
                        "source": "s3",
                        "exposure_start_time": timestamp.isoformat().replace(
                            "+00:00", "Z"
                        ),
                    }
                )

        observations = build_observations(records)

        shared = [row for row in observations if row["file_set_name"].startswith("shared_")]
        self.assertEqual(len(shared), 7)
        self.assertEqual(
            len({row["repeat"] for row in shared if row["led"] == 3}),
            2,
        )

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

    def test_find_previous_raw_statistics_ignores_current_run_and_scaled_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_raw = root / "20261002" / "raw_statistics" / "wfi01_tile_log_statistics_512.csv"
            stale_scaled = root / "20261003" / "statistics" / "wfi01_tile_log_statistics_512.csv"
            current = root / "20261003"
            old_raw.parent.mkdir(parents=True)
            old_raw.write_text("raw")
            stale_scaled.parent.mkdir(parents=True)
            stale_scaled.write_text("stale scaled data")
            self.assertEqual(
                pipeline.find_previous_raw_statistics(root, current, "WFI01", 512),
                old_raw,
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

    def test_ice_monitor_fetch_passes_explicit_program_to_roman_telem(self):
        index_rows = [
            {
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "exposure_end_time": "2026-01-01T00:01:00Z",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "telemetry" / "program_1063.csv"
            with patch.object(pipeline, "_read_mast_token", return_value="token"), \
                 patch.object(pipeline, "load_ice_monitor_photodiode_summary", return_value={"values": {}}), \
                 patch.object(pipeline.subprocess, "run") as run:
                pipeline.run_ice_monitor_telemetry(
                    index_rows,
                    Path(directory) / "mast_api_token.txt",
                    "pd1",
                    "1063",
                    output,
                )

        command = run.call_args.args[0]
        program_index = command.index("--ice-program")
        self.assertEqual(command[program_index + 1], "1063")

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
            "warmup",
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
                "warmup": "False",
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
                "warmup": "False",
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
            pipeline.apply_first_non_warmup_references(telemetry)

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

    def test_merged_telemetry_uses_first_non_warmup_reference_across_programs(self):
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
            "warmup",
            "ref_pass",
            "ref_signal",
            "change_pct",
            "pass_id",
        )
        rows_by_program = {
            "1039": [
                {
                    "mnemonic": "WFI_MCE_SRCS_PD1_V",
                    "step": "band1",
                    "led_band": "1",
                    "start": "2026-01-01T00:00:00Z",
                    "end": "2026-01-01T00:01:00Z",
                    "signal": "9.0",
                    "dark_median": "0.2",
                    "dark_subtracted": "True",
                    "n_samples": "100",
                    "warmup": "True",
                    "ref_pass": "tool-warmup",
                    "ref_signal": "10.0",
                    "change_pct": "-10.0",
                    "pass_id": "0103902001",
                },
                {
                    "mnemonic": "WFI_MCE_SRCS_PD1_V",
                    "step": "band1",
                    "led_band": "1",
                    "start": "2026-01-01T01:00:00Z",
                    "end": "2026-01-01T01:01:00Z",
                    "signal": "10.0",
                    "dark_median": "0.2",
                    "dark_subtracted": "True",
                    "n_samples": "100",
                    "warmup": "False",
                    "ref_pass": "tool-reference",
                    "ref_signal": "10.0",
                    "change_pct": "0.0",
                    "pass_id": "0103902002",
                },
            ],
            "1063": [
                {
                    "mnemonic": "WFI_MCE_SRCS_PD1_V",
                    "step": "band1",
                    "led_band": "1",
                    "start": "2026-01-01T02:00:00Z",
                    "end": "2026-01-01T02:01:00Z",
                    "signal": "8.0",
                    "dark_median": "0.2",
                    "dark_subtracted": "True",
                    "n_samples": "100",
                    "warmup": "False",
                    "ref_pass": "1063-self-reference",
                    "ref_signal": "8.0",
                    "change_pct": "0.0",
                    "pass_id": "0106302001",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            summaries = []
            for program, rows in rows_by_program.items():
                path = Path(directory) / f"{program}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                summaries.append(load_ice_monitor_photodiode_summary(path, "pd1"))
            telemetry = merge_telemetry_summaries(summaries)

        self.assertEqual(telemetry["reference_mode"], "first_non_warmup_per_led")
        self.assertEqual(telemetry["reference_values"]["1"]["pass_id"], "0103902002")
        self.assertEqual(telemetry["reference_values"]["1"]["signal"], 10.0)
        value = telemetry["values"][(
            datetime(2026, 1, 1, 2, 0),
            datetime(2026, 1, 1, 2, 1),
            1,
        )]
        self.assertEqual(value["reference_pass"], "0103902002")
        self.assertEqual(value["reference_signal"], 10.0)
        self.assertAlmostEqual(value["change_pct"], -20.0)

        observations = [
            {
                "led": 1,
                "group": 0,
                "file_set_name": "program-1063",
                "exposure_start_time": "2026-01-01T02:00:00Z",
                "exposure_end_time": "2026-01-01T02:01:00Z",
            }
        ]
        attach_photodiode_corrections(observations, telemetry)
        self.assertEqual(observations[0]["photodiode_scale"], 1.25)
        self.assertEqual(observations[0]["photodiode_reference_pass"], "0103902002")

    def test_photodiode_correction_falls_back_to_visit_identity(self):
        telemetry = {
            "source": "roman-telem-ice-monitor",
            "channel": "pd1",
            "reference_mode": "first_non_warmup_per_led",
            "values": {},
            "identity_values": {
                ("0103902001", "001.002.001", 1): [
                    (
                        datetime(2026, 1, 1, 0, 0, 1),
                        datetime(2026, 1, 1, 0, 1, 1),
                        {
                            "signal": 8.0,
                            "dark_median": 0.3,
                            "n_samples": 100,
                            "reference_pass": "0103902002",
                            "reference_signal": 10.0,
                            "change_pct": -20.0,
                        },
                    )
                ]
            },
        }
        observations = [
            {
                "led": 1,
                "visit_id": "0103902001001002001",
                "file_set_name": "exposure",
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "exposure_end_time": "2026-01-01T00:01:00Z",
            }
        ]

        attach_photodiode_corrections(observations, telemetry)

        self.assertEqual(observations[0]["photodiode_scale"], 1.25)

    def test_telemetry_identity_splits_grouped_observations(self):
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
            "warmup",
            "ref_pass",
            "ref_signal",
            "change_pct",
            "pass_id",
            "observation",
        )
        row = {
            "mnemonic": "WFI_MCE_SRCS_PD1_V",
            "step": "band3",
            "led_band": "3",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-01-01T00:03:00Z",
            "signal": "8.0",
            "dark_median": "0.3",
            "dark_subtracted": "True",
            "n_samples": "300",
            "warmup": "False",
            "ref_pass": "0106302001",
            "ref_signal": "10.0",
            "change_pct": "-20.0",
            "pass_id": "0106302001",
            "observation": "001.004.001,001.009.001",
        }
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "ice_monitor.csv"
            with summary_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            telemetry = load_ice_monitor_photodiode_summary(summary_path, "pd1")
            pipeline.apply_first_non_warmup_references(telemetry)

        observations = [
            {
                "led": 3,
                "visit_id": "0106302001001004001",
                "file_set_name": "first-band3",
                "exposure_start_time": "2026-01-01T01:00:00Z",
                "exposure_end_time": "2026-01-01T01:01:00Z",
            },
            {
                "led": 3,
                "visit_id": "0106302001001009001",
                "file_set_name": "second-band3",
                "exposure_start_time": "2026-01-01T02:00:00Z",
                "exposure_end_time": "2026-01-01T02:01:00Z",
            },
        ]

        attach_photodiode_corrections(observations, telemetry)

        self.assertEqual(observations[0]["photodiode_scale"], 1.0)
        self.assertEqual(observations[1]["photodiode_scale"], 1.0)

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

            observations[0]["photodiode_scale"] = 1.75
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
                    "program": "1063" if repeat == 2 else "1039",
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
                    "program": "1063" if repeat == 2 else "1039",
                    "exposure_start_time": timestamp,
                    "exposure_end_time": timestamp,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "20261003"
            manifest = {"detectors": {}}
            side_swap_rows = [
                {
                    "led": led,
                    "channel": channel,
                    "correction_factor": 1.0,
                }
                for led in range(1, 7)
                for channel in ("image", "photodiode")
            ]
            with patch.object(pipeline, "DEFAULT_TILE_SIZES", (512,)), \
                 patch.object(pipeline, "mast_records_from_index", return_value=[]), \
                 patch.object(pipeline, "attach_photodiode_corrections"), \
                  patch.object(pipeline, "reduce_statistics", return_value="rebuilt") as reduce, \
                  patch.object(pipeline, "available_side_swap_points", return_value=(5, 3)), \
                  patch.object(pipeline, "fit_side_swap_series", return_value=side_swap_rows), \
                  patch.object(pipeline, "write_side_swap_scaled_statistics"), \
                  patch.object(pipeline, "post_temperature_repeats_by_tile", return_value={0: ()}), \
                  patch.object(pipeline, "build_corrections", return_value=([], [])) as build_corrections, \
                 patch.object(pipeline, "write_csv"), \
                 patch.object(pipeline, "write_corrected_statistics") as write_corrected, \
                  patch.object(pipeline, "plot_count_diagnostic") as plot_diagnostic, \
                  patch.object(pipeline, "plot_reset_aligned_counts") as reset_plot, \
                  patch.object(pipeline, "plot_led3_post_temperature_fit") as temp_fit_plot, \
                  patch.object(pipeline, "plot_led3_pre_temperature_fit") as pre_temp_fit_plot:
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
        self.assertEqual(
            reduce.call_args.kwargs["raw_output_path"],
            run_dir / "raw_statistics" / "wfi01_tile_log_statistics_512.csv",
        )
        self.assertEqual(plot_diagnostic.call_count, 1)
        self.assertEqual(reset_plot.call_count, 1)
        self.assertEqual(temp_fit_plot.call_count, 1)
        self.assertEqual(pre_temp_fit_plot.call_count, 1)
        self.assertEqual(temp_fit_plot.call_args.args[5], 52)
        self.assertEqual(temp_fit_plot.call_args.args[6], 39)
        self.assertEqual(
            build_corrections.call_args.kwargs["post_acm_fit_start_repeat"],
            36,
        )
        self.assertEqual(
            build_corrections.call_args.kwargs[
                "post_temperature_fit_start_repeat"
            ],
            39,
        )
        self.assertEqual(
            build_corrections.call_args.kwargs["post_temperature_fit_end_repeat"],
            52,
        )
        self.assertEqual(
            build_corrections.call_args.kwargs["day22_acm_step_repeat"],
            52,
        )
        self.assertEqual(
            build_corrections.call_args.kwargs["day26_acm_step_repeat"],
            61,
        )
        self.assertEqual(build_corrections.call_args.args[6], 56)
        self.assertEqual(
            build_corrections.call_args.kwargs["pre_temperature_fit_start_repeat"],
            14,
        )
        self.assertEqual(
            build_corrections.call_args.kwargs["pre_temperature_fit_end_repeat"],
            35,
        )

    def test_legacy_post_operational_repeats_map_to_unified_epochs(self):
        observations = [
            {
                "program": "1039",
                "repeat": repeat,
                "days": 15.0 + repeat - 666,
                "exposure_start_time": "2026-09-27T03:00:00Z",
            }
            for repeat in range(666, 695)
        ]
        with patch.object(pipeline, "day_for_timestamp", return_value=15.0):
            self.assertEqual(
                pipeline.effective_post_operational_repeat(observations, 39),
                669,
            )
            self.assertEqual(
                pipeline.effective_post_operational_repeat(observations, 56),
                686,
            )
            self.assertEqual(
                pipeline.effective_post_operational_repeat(observations, 61),
                691,
            )

    def test_current_program_inventory_does_not_reuse_unified_repeat_39(self):
        fields = ("repeat", "program", "exposure_start_time")
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / "inventory.csv"
            with inventory.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "repeat": "39",
                            "program": "1063",
                            "exposure_start_time": "2026-09-24T23:30:55Z",
                        },
                        {
                            "repeat": "56",
                            "program": "1039",
                            "exposure_start_time": "2026-10-06T11:57:50Z",
                        },
                    )
                )

            self.assertIsNone(
                pipeline.previous_repeat_time(inventory, 39, pipeline.REFERENCE_PROGRAM)
            )
            self.assertEqual(
                pipeline.previous_repeat_time(inventory, 56, pipeline.REFERENCE_PROGRAM),
                "2026-10-06T11:57:50Z",
            )

    def test_day26_legacy_repeat_maps_through_all_programs(self):
        observations = [
            {
                "program": "1039",
                "repeat": repeat,
                "days": 15.0 + repeat - 666,
                "exposure_start_time": "2026-09-27T03:00:00Z",
            }
            for repeat in range(666, 691)
        ]
        observations.append(
            {
                "program": "1063",
                "repeat": 691,
                "days": 27.14,
                "exposure_start_time": "2026-10-09T04:49:15Z",
            }
        )
        with patch.object(pipeline, "day_for_timestamp", return_value=15.0):
            self.assertEqual(
                pipeline.effective_post_operational_repeat(
                    observations,
                    pipeline.DEFAULT_DAY26_ACM_STEP_REPEAT,
                    include_all_programs=True,
                ),
                691,
            )

    def test_explicit_late_step_remains_active_after_event(self):
        fields = (
            "tile_id",
            "group",
            "led",
            "repeat",
            "days",
            "exposure_start_time",
            "sum_current_counts",
            "n_pair_valid",
        )
        with tempfile.TemporaryDirectory() as directory:
            statistics = Path(directory) / "wfi01_tile_log_statistics_512.csv"
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                group = 0
                for repeat in range(3):
                    for led in range(1, 7):
                        writer.writerow(
                            {
                                "tile_id": 0,
                                "group": group,
                                "led": led,
                                "repeat": repeat,
                                "days": float(repeat),
                                "exposure_start_time": "2026-09-01T00:00:00Z",
                                "sum_current_counts": 100.0,
                                "n_pair_valid": 1.0,
                            }
                        )
                        group += 1

            tiles = pipeline.load_count_statistics(
                statistics,
                additional_acm_step_repeat=None,
                pre_reset_acm_step_repeat=None,
                late_acm_step_repeat=1,
                day22_acm_step_repeat=2,
            )
            tiles_without_late_step = pipeline.load_count_statistics(
                statistics,
                additional_acm_step_repeat=None,
                pre_reset_acm_step_repeat=None,
                late_acm_step_repeat=1,
                disable_late_acm_step=True,
            )

        np.testing.assert_array_equal(
            tiles[0]["led"][1]["late_step"],
            np.asarray([0.0, 1.0, 1.0]),
        )
        np.testing.assert_array_equal(
            tiles[0]["led"][1]["day22_step"],
            np.asarray([0.0, 0.0, 1.0]),
        )
        self.assertIsNone(tiles_without_late_step[0]["led"][1]["late_step"])

    def test_count_diagnostic_plot_compares_raw_and_corrected_counts(self):
        statistics_fields = (
            "led",
            "program",
            "group",
            "tile_id",
            "days",
            "sum_current_counts",
            "n_pair_valid",
        )
        correction_fields = ("tile_id", "group", "count_log_correction")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statistics = root / "statistics.csv"
            corrections = root / "corrections.csv"
            output = root / "diagnostics" / "counts.png"
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=statistics_fields)
                writer.writeheader()
                for tile_id in (0, 1):
                    for group, (program, day) in enumerate(
                        (("1039", 1.0), ("1063", 2.0))
                    ):
                        for led in range(1, 7):
                            writer.writerow(
                                {
                                    "led": led,
                                    "program": program,
                                    "group": group,
                                    "tile_id": tile_id,
                                    "days": day,
                                    "sum_current_counts": 100.0 + led,
                                    "n_pair_valid": 10.0,
                                }
                            )
            with corrections.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=correction_fields)
                writer.writeheader()
                for tile_id in (0, 1):
                    for group in (0, 1):
                        writer.writerow(
                            {
                                "tile_id": tile_id,
                                "group": group,
                                "count_log_correction": 0.1 * group,
                            }
                        )

            result = pipeline.plot_count_diagnostic(
                output,
                statistics,
                corrections,
                "WFI01",
                operational_temperature_day=1.5,
                physical_reset_day=2.0,
            )

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_reset_aligned_count_plot_compares_reset_sides(self):
        fields = (
            "led",
            "program",
            "group",
            "tile_id",
            "days",
            "sum_current_counts",
            "n_pair_valid",
        )
        with tempfile.TemporaryDirectory() as directory:
            statistics = Path(directory) / "corrected_statistics.csv"
            output = Path(directory) / "diagnostics" / "reset_aligned.png"
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for tile_id in (0, 1):
                    for group, (program, day) in enumerate(
                        (("1039", 1.0), ("1063", 2.0))
                    ):
                        for led in range(1, 7):
                            writer.writerow(
                                {
                                    "led": led,
                                    "program": program,
                                    "group": group,
                                    "tile_id": tile_id,
                                    "days": day,
                                    "sum_current_counts": 100.0
                                    * (10.0 if day >= 2.0 else 1.0),
                                    "n_pair_valid": 10.0,
                                }
                            )

            result = pipeline.plot_reset_aligned_counts(
                output,
                statistics,
                "WFI01",
                physical_reset_day=1.5,
            )

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_led3_temperature_fit_plot_marks_next_step_cutoff(self):
        statistics_fields = (
            "led",
            "program",
            "repeat",
            "group",
            "tile_id",
            "days",
            "sum_current_counts",
            "n_pair_valid",
        )
        correction_fields = (
            "tile_id",
            "group",
            "count_log_correction",
            "temperature_gain_log_correction",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statistics = root / "statistics.csv"
            corrections = root / "corrections.csv"
            output = root / "diagnostics" / "led3_temperature_fit.png"
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=statistics_fields)
                writer.writeheader()
                for repeat, (program, day) in enumerate(
                    (("1039", 9.0), ("1039", 10.0), ("1039", 11.0), ("1039", 12.0), ("1063", 13.0))
                ):
                    writer.writerow(
                        {
                            "led": 3,
                            "program": program,
                            "repeat": repeat,
                            "group": repeat,
                            "tile_id": 0,
                            "days": day,
                            "sum_current_counts": 1000.0 + repeat,
                            "n_pair_valid": 10.0,
                        }
                    )
            with corrections.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=correction_fields)
                writer.writeheader()
                for group in range(5):
                    writer.writerow(
                        {
                            "tile_id": 0,
                            "group": group,
                            "count_log_correction": 0.0,
                            "temperature_gain_log_correction": 0.0,
                        }
                    )

            result = pipeline.plot_led3_post_temperature_fit(
                output,
                statistics,
                corrections,
                "WFI01",
                operational_temperature_day=9.0,
                post_temperature_fit_end_repeat=3,
            )

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_led3_pre_temperature_fit_plot_marks_acm_step_interval(self):
        statistics_fields = (
            "led",
            "program",
            "repeat",
            "group",
            "tile_id",
            "days",
            "sum_current_counts",
            "n_pair_valid",
        )
        correction_fields = (
            "tile_id",
            "group",
            "count_log_correction",
            "log_gain_correction",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statistics = root / "statistics.csv"
            corrections = root / "corrections.csv"
            output = root / "diagnostics" / "led3_pre_temperature_fit.png"
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=statistics_fields)
                writer.writeheader()
                for repeat in range(7):
                    writer.writerow(
                        {
                            "led": 3,
                            "program": "1063" if repeat == 6 else "1039",
                            "repeat": repeat,
                            "group": repeat,
                            "tile_id": 0,
                            "days": 2.0 + repeat,
                            "sum_current_counts": 1000.0 + repeat,
                            "n_pair_valid": 10.0,
                        }
                    )
            with corrections.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=correction_fields)
                writer.writeheader()
                for group in range(7):
                    writer.writerow(
                        {
                            "tile_id": 0,
                            "group": group,
                            "count_log_correction": 0.0,
                            "log_gain_correction": 0.0,
                        }
                    )

            result = pipeline.plot_led3_pre_temperature_fit(
                output,
                statistics,
                corrections,
                "WFI01",
                first_acm_step_repeat=2,
                second_acm_step_repeat=5,
            )

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_post_temperature_fit_uses_rows_through_late_step(self):
        rows = [
            {
                "program": "1039",
                "days": 15.0 + 0.1 * repeat,
                "repeat": repeat,
            }
            for repeat in range(666, 686)
        ]

        selected = pipeline.select_post_temperature_fit_rows(rows, 15.0, 686)

        self.assertEqual(len(selected), 20)
        self.assertEqual(selected[0]["repeat"], 666)
        self.assertEqual(selected[-1]["repeat"], 685)

    def test_temperature_fit_boundary_is_after_additional_step(self):
        self.assertEqual(pipeline.temperature_fit_end_repeat(669, 686), 686)
        self.assertEqual(pipeline.temperature_fit_end_repeat(669, 686, 682), 682)
        with self.assertRaises(ValueError):
            pipeline.temperature_fit_end_repeat(669, 669)
        with self.assertRaises(ValueError):
            pipeline.temperature_fit_end_repeat(669, 686, 686)

    def test_unified_temperature_correction_only_applies_to_first_1063_batch(self):
        fields = (
            "led",
            "program",
            "repeat",
            "group",
            "tile_id",
            "days",
            "exposure_start_time",
            "sum_current_counts",
            "n_pair_valid",
        )
        epochs = [
            ("1039", 0.0),
            ("1039", 2.0),
            ("1039", 4.0),
            ("1039", 6.0),
            ("1063", 10.0),
            ("1063", 12.0),
            ("1039", 16.0),
            ("1039", 18.0),
            ("1039", 20.0),
            ("1039", 22.0),
            ("1063", 24.0),
        ]
        rows = []
        for repeat, (program, day) in enumerate(epochs):
            for led in range(1, 7):
                count = 1000.0 + led * 10.0 + 20.0 * np.sin(
                    2.0 * np.pi * (day - 16.0) / 12.0
                )
                if program == "1063":
                    count *= 1.25
                rows.append(
                    {
                        "led": led,
                        "program": program,
                        "repeat": repeat,
                        "group": len(rows),
                        "tile_id": 0,
                        "days": day,
                        "exposure_start_time": (
                            datetime(2026, 1, 1) + timedelta(days=day)
                        ).isoformat()
                        + "Z",
                        "sum_current_counts": count,
                        "n_pair_valid": 100.0,
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            statistics = Path(directory) / "wfi01_tile_log_statistics_512.csv"
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            corrections, summaries = build_corrections(
                statistics,
                reference_led=3,
                minimum_day=1.0,
                additional_acm_step_repeat=9,
                operational_temperature_day=15.0,
                pre_reset_acm_step_repeat=2,
                late_acm_step_repeat=10,
                post_temperature_fit_start_repeat=6,
                pre_temperature_fit_start_repeat=1,
                pre_temperature_fit_end_repeat=3,
            )

        self.assertEqual(len(summaries), 1)
        self.assertGreater(
            summaries[0]["post_operational_temperature_period_days"], 0.0
        )
        self.assertEqual(
            summaries[0]["post_operational_temperature_fit_points"], 4
        )
        self.assertEqual(
            summaries[0]["post_operational_temperature_fit_end_repeat"], 10
        )
        self.assertEqual(
            summaries[0]["pre_operational_temperature_fit_points"], 2
        )
        self.assertEqual(
            summaries[0]["pre_operational_temperature_fit_start_repeat"], 1
        )
        self.assertEqual(
            summaries[0]["pre_operational_temperature_fit_end_repeat"], 3
        )
        first_temperature_batch_rows = [
            row
            for row in corrections
            if row["program"] == "1063" and float(row["days"]) < 15.0
        ]
        later_temperature_batch_rows = [
            row
            for row in corrections
            if row["program"] == "1063" and float(row["days"]) >= 15.0
        ]
        reference_rows = [
            row for row in corrections if row["program"] == "1039"
        ]
        self.assertEqual(len(first_temperature_batch_rows), 12)
        self.assertEqual(len(later_temperature_batch_rows), 6)
        self.assertTrue(reference_rows)
        self.assertTrue(
            any(
                float(row["temperature_gain_log_correction"]) != 0.0
                for row in first_temperature_batch_rows
            )
        )
        self.assertTrue(
            all(
                float(row["temperature_gain_log_correction"]) == 0.0
                for row in later_temperature_batch_rows
            )
        )
        self.assertTrue(
            all(float(row["temperature_gain_log_correction"]) == 0.0 for row in reference_rows)
        )

    def test_shared_acm_fit_pools_all_leds_and_temperature_uses_reference_led(self):
        rows = []
        led_data = {}
        programs = ("1039", "1039", "1039", "1039", "1063", "1039", "1063", "1039")
        for led in range(1, 7):
            led_rows = []
            for repeat, program in enumerate(programs):
                row = {
                    "tile_id": 0,
                    "group": repeat * 6 + led - 1,
                    "days": float(repeat),
                    "repeat": repeat,
                    "led": led,
                    "program": program,
                }
                rows.append(row)
                led_rows.append(row)
            days = np.arange(8, dtype=float) + 0.01 * led
            values = np.full(8, 100.0)
            values[3:] *= np.exp(-0.001 * led)
            led_data[led] = {
                "rows": led_rows,
                "days": days,
                "values": values,
                "program": np.asarray(programs, dtype=object),
                "first_step": np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                "additional_step": np.asarray([float(repeat >= 4) for repeat in range(8)]),
                "pre_reset_step": np.asarray([float(repeat >= 2) for repeat in range(8)]),
                "day22_step": np.asarray([float(repeat >= 3) for repeat in range(8)]),
                "late_step": np.asarray([float(repeat >= 6) for repeat in range(8)]),
                "repeat": np.arange(8),
            }
        tile = {"rows": rows, "led": led_data}
        calls = []
        mask_counts = []

        def fake_shared_fit(entries, masks, step_names, per_led_step_names=()):
            calls.append((tuple(entry["led"] for entry in entries), tuple(step_names)))
            mask_counts.append(tuple(np.count_nonzero(mask) for mask in masks))
            led_order = tuple(entry["led"] for entry in entries)
            step_coefficients = {
                name: value
                for name, value in (
                    ("first_step", np.log(1.1)),
                    ("pre_reset_step", np.log(1.2)),
                    ("additional_step", np.log(1.3)),
                    ("late_step", np.log(0.9)),
                )
                if name in step_names
            }
            return {
                "period_days": 10.0,
                "origin_day": 0.0,
                "led_order": led_order,
                "led_offsets": {led: 0.0 for led in led_order},
                "led_sine_coefficients": {led: 0.0 for led in led_order},
                "led_cosine_coefficients": {led: 0.0 for led in led_order},
                    "step_coefficients": step_coefficients,
                    "led_step_coefficients": {
                        name: {
                            led: np.log(1.3) + 0.01 * led
                            for led in led_order
                        }
                        for name in per_led_step_names
                    },
                    "rmse_log_change": 0.0,
                "r2": 1.0,
            }

        temperature_fit = {
            "period_days": 10.0,
            "origin_day": 0.0,
            "offset": 100.0,
            "sine": 0.0,
            "cosine": 0.0,
            "rmse": 0.0,
            "r2": 1.0,
        }
        temperature_calls = []

        def record_temperature_fit(days, values):
            temperature_calls.append((np.asarray(days), np.asarray(values)))
            return temperature_fit

        with patch.object(pipeline, "_fit_shared_log_curve", side_effect=fake_shared_fit), \
             patch.object(pipeline, "_fit_count_sinusoid", side_effect=record_temperature_fit):
            corrections, _ = pipeline.fit_count_corrections(
                tile,
                reference_led=3,
                minimum_day=1.5,
                additional_acm_step_repeat=4,
                operational_temperature_day=4.0,
                pre_reset_acm_step_repeat=2,
                late_acm_step_repeat=8,
                post_temperature_fit_start_repeat=6,
                post_acm_fit_start_repeat=4,
                day22_acm_step_repeat=3,
                post_acm_fit_end_repeat=8,
                per_led_acm_step_names=("day22_step",),
            )

        self.assertEqual(calls[0], ((1, 2, 3, 4, 5, 6), ("first_step", "pre_reset_step")))
        self.assertEqual(calls[1], ((1, 2, 3, 4, 5, 6), ("additional_step", "late_step")))
        self.assertEqual(mask_counts[0], (2, 2, 2, 2, 2, 2))
        self.assertEqual(mask_counts[1], (2, 2, 2, 2, 2, 2))
        self.assertEqual(len(temperature_calls), 2)
        np.testing.assert_allclose(temperature_calls[1][0], np.asarray([7.03]))
        by_led_repeat = {
            (int(row["group"]) % 6 + 1, int(row["group"]) // 6): row
            for row in corrections
        }
        for repeat in range(8):
            reference_row = by_led_repeat[(1, repeat)]
            for led in range(2, 7):
                row = by_led_repeat[(led, repeat)]
                for field in (
                    "acm_step_log_offset",
                    "pre_reset_acm_step_log_offset",
                    "additional_acm_step_log_offset",
                    "late_acm_step_log_offset",
                    "temperature_gain_log_correction",
                ):
                    self.assertAlmostEqual(float(row[field]), float(reference_row[field]))
        self.assertNotAlmostEqual(
            float(by_led_repeat[(1, 5)]["day22_acm_step_log_offset"]),
            float(by_led_repeat[(2, 5)]["day22_acm_step_log_offset"]),
        )
        self.assertTrue(
            all(
                float(by_led_repeat[(led, 4)]["temperature_gain_log_correction"])
                == 0.0
                for led in range(1, 7)
            )
        )

        temperature_calls.clear()
        with patch.object(pipeline, "_fit_shared_log_curve", side_effect=fake_shared_fit), \
             patch.object(pipeline, "_fit_count_sinusoid", side_effect=record_temperature_fit):
            per_led_corrections, _ = pipeline.fit_count_corrections(
                tile,
                reference_led=3,
                minimum_day=1.5,
                additional_acm_step_repeat=4,
                operational_temperature_day=4.0,
                pre_reset_acm_step_repeat=2,
                late_acm_step_repeat=8,
                post_temperature_fit_start_repeat=6,
                post_acm_fit_start_repeat=4,
                per_led_acm_step_names=("additional_step",),
            )
        per_led_by_repeat = {
            (int(row["group"]) % 6 + 1, int(row["group"]) // 6): row
            for row in per_led_corrections
        }
        self.assertNotAlmostEqual(
            float(per_led_by_repeat[(1, 5)]["additional_acm_step_log_offset"]),
            float(per_led_by_repeat[(2, 5)]["additional_acm_step_log_offset"]),
        )

        tile_without_late_step = {
            "rows": tile["rows"],
            "led": {
                led: {**entry, "late_step": None}
                for led, entry in tile["led"].items()
            },
        }
        temperature_calls.clear()
        with patch.object(pipeline, "_fit_shared_log_curve", side_effect=fake_shared_fit), \
             patch.object(pipeline, "_fit_count_sinusoid", side_effect=record_temperature_fit):
            pipeline.fit_count_corrections(
                tile_without_late_step,
                reference_led=3,
                minimum_day=1.5,
                additional_acm_step_repeat=5,
                operational_temperature_day=4.0,
                pre_reset_acm_step_repeat=2,
                late_acm_step_repeat=7,
                disable_late_acm_step=True,
            )
        np.testing.assert_allclose(temperature_calls[1][0], np.asarray([5.03]))


    def test_day26_acm_step_is_shared_grey(self):
        rows = []
        led_data = {}
        step = 0.002
        for led in range(1, 7):
            led_rows = []
            days = np.arange(9, dtype=float) + 0.01 * led
            values = 100.0 * np.exp(0.0005 * days)
            values[4:] *= np.exp(step)
            for repeat, day in enumerate(days):
                row = {
                    "tile_id": 0,
                    "group": repeat * 6 + led - 1,
                    "days": float(day),
                    "repeat": repeat,
                    "led": led,
                    "program": "1039",
                }
                rows.append(row)
                led_rows.append(row)
            led_data[led] = {
                "rows": led_rows,
                "days": days,
                "values": values,
                "program": np.full(9, "1039", dtype=object),
                "first_step": None,
                "additional_step": None,
                "pre_reset_step": None,
                "day22_step": None,
                "day26_step": np.asarray(
                    [float(repeat >= 4) for repeat in range(9)]
                ),
                "late_step": None,
                "repeat": np.arange(9),
            }

        corrections, summary = pipeline.fit_count_corrections(
            {"rows": rows, "led": led_data},
            reference_led=3,
            minimum_day=1.0,
            additional_acm_step_repeat=None,
            operational_temperature_day=None,
            pre_reset_acm_step_repeat=None,
            late_acm_step_repeat=None,
            day26_acm_step_repeat=4,
            post_temperature_fit_end_repeat=9,
        )

        self.assertAlmostEqual(
            summary["reference_day26_acm_step_counts"], step, places=7
        )
        self.assertEqual(summary["day26_acm_step_fit_scope"], "shared_grey")
        self.assertTrue(
            all(
                abs(float(row["day26_acm_step_log_offset"]) - step) < 1.0e-7
                for row in corrections
            )
        )
        self.assertTrue(
            all(
                abs(float(row["count_log_correction"]) + step) < 1.0e-7
                for row in corrections
                if int(row["group"]) // 6 >= 4
            )
        )


if __name__ == "__main__":
    unittest.main()
