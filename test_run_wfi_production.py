import csv
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_wfi_production as runner
import run_wfi_relative_response_production as relative_runner


def _timeout_test_process(job, finished):
    if job[0] == 3:
        time.sleep(6.0)
    finished.set()


class ProductionRunnerTests(unittest.TestCase):
    def write_statistics(self, path, scale=1.0):
        fieldnames = [
            "tile_id",
            "group",
            "n_pair_valid",
            "sum_log_change",
            "sum_squared_log_change",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for tile_id in (0, 1):
                for group in (1, 0):
                    writer.writerow(
                        {
                            "tile_id": tile_id,
                            "group": group,
                            "n_pair_valid": 10,
                            "sum_log_change": scale * (group + 1),
                            "sum_squared_log_change": scale * (group + 2),
                        }
                    )

    def test_custom_paths_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "wfi01_corrected.csv"
            self.write_statistics(source)
            split_root = root / "split"
            paths = runner.tile_statistics_paths("WFI01", source, split_root)
            self.assertEqual(sorted(paths), [0, 1])

            fit_dir = root / "fit"
            fit_dir.mkdir()
            np.savez(
                fit_dir / "posterior_samples.npz",
                statistics_row_fingerprints=runner.statistics_row_fingerprints(paths[0]),
                statistics_fingerprint=runner.statistics_row_fingerprints(paths[0]),
            )
            self.assertTrue(runner.archive_matches_statistics(fit_dir, paths[0]))

            self.write_statistics(source, scale=2.0)
            runner.tile_statistics_paths("WFI01", source, split_root)
            self.assertFalse(runner.archive_matches_statistics(fit_dir, paths[0]))

    def test_detector_paths_use_fixed_production_layout(self):
        statistics, output, split, fits, emcee = runner.detector_paths(
            "WFI03", Path("corrected")
        )
        self.assertEqual(
            statistics,
            Path("corrected/wfi03_tile_log_statistics_256_count_corrected.csv"),
        )
        self.assertEqual(
            output,
            Path("corrected") / "production" / "wfi03",
        )
        self.assertEqual(split.parent, output)
        self.assertEqual(fits.parent, output)
        self.assertEqual(emcee, output / "emcee")

    def test_detector_paths_support_alternate_tile_size_layout(self):
        statistics, output, split, fits, emcee = runner.detector_paths(
            "WFI03",
            Path("corrected"),
            tile_size=512,
        )
        self.assertEqual(
            statistics,
            Path("corrected/wfi03_tile_log_statistics_512_count_corrected.csv"),
        )
        self.assertEqual(
            output,
            Path("corrected") / "production" / "tile_size_512" / "wfi03",
        )
        self.assertEqual(split, output / "tile_statistics")
        self.assertEqual(fits, output / "fits")
        self.assertEqual(emcee, output / "emcee")

    def test_tile_size_argument_defaults_to_256(self):
        arguments = runner.parse_args(["--statistics-dir", "."])
        self.assertEqual(arguments.tile_size, 256)
        self.assertEqual(arguments.workers, 2)
        self.assertEqual(arguments.tile_timeout_seconds, 1800.0)
        self.assertEqual(
            arguments.absolute_hgcdte_n_scale_prior_sigma_log,
            0.25,
        )
        self.assertEqual(
            runner.parse_args(["--statistics-dir", ".", "--tile-size", "256"]).tile_size,
            256,
        )

    def test_hgcdte_n_prior_sigma_argument_parses(self):
        arguments = runner.parse_args(
            [
                "--statistics-dir",
                ".",
                "--absolute-hgcdte-n-scale-prior-sigma-log",
                "0.1",
            ]
        )
        self.assertEqual(arguments.absolute_hgcdte_n_scale_prior_sigma_log, 0.1)

    def test_fixed_hgcdte_n_argument_parses(self):
        arguments = runner.parse_args(
            [
                "--statistics-dir",
                ".",
                "--fixed-absolute-hgcdte-n-log-scale",
                "0.0",
            ]
        )
        self.assertEqual(arguments.fixed_absolute_hgcdte_n_log_scale, 0.0)

    def test_workers_argument_requires_a_positive_integer(self):
        self.assertEqual(
            runner.parse_args(["--statistics-dir", ".", "--workers", "2"]).workers,
            2,
        )
        with self.assertRaises(SystemExit):
            runner.parse_args(["--statistics-dir", ".", "--workers", "0"])

    def test_response_cache_argument_selects_explicit_cached_fit_mode(self):
        arguments = runner.parse_args(
            [
                "--statistics-dir",
                ".",
                "--detector",
                "WFI01",
                "--tile-size",
                "256",
                "--response-cache-root",
                "cache",
            ]
        )
        with patch.object(relative_runner, "run_detector") as run_detector:
            runner.run_cached_response(arguments, ["WFI01"])
        run_detector.assert_called_once_with(
            "WFI01",
            Path("."),
            Path("cache"),
            selected_tile_ids=None,
            tile_size=256,
            output_root=None,
            force=False,
            workers=2,
            batch_size=8,
            joint_batch=False,
        )

    def test_response_cache_argument_requires_one_detector(self):
        arguments = runner.parse_args(
            ["--statistics-dir", ".", "--response-cache-root", "cache"]
        )
        with self.assertRaisesRegex(ValueError, "exactly one --detector"):
            runner.run_cached_response(arguments, list(runner.DETECTORS))

    def test_response_cache_can_reuse_external_source_provenance(self):
        arguments = runner.parse_args(
            [
                "--statistics-dir",
                ".",
                "--detector",
                "WFI01",
                "--response-cache-root",
                "cache",
                "--allow-external-cache-source",
            ]
        )
        with patch.object(relative_runner, "run_detector") as run_detector:
            runner.run_cached_response(arguments, ["WFI01"])
        self.assertTrue(run_detector.call_args.kwargs["allow_external_cache_source"])

    def test_run_detector_uses_data_driven_alternate_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            statistics = root / "wfi01_tile_log_statistics_256_count_corrected.csv"
            fields = ("tile_id", "group")
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for tile_id in range(4):
                    writer.writerow({"tile_id": tile_id, "group": 0})
            reference_dir = root / "reference"
            reference_dir.mkdir()
            for name in (
                "WFI_RCS_LEDP_spectral_energy_distribution.ecsv",
                "WFI_Quantum_efficiency.ecsv",
            ):
                (reference_dir / name).write_text("reference\n")

            with patch.object(runner, "REFERENCE_DATA_DIR", reference_dir), patch.object(
                runner.emcee_warmup, "run_emcee"
            ) as run_emcee, patch.object(
                runner, "_run_tile_jobs", return_value=[]
            ) as run_tile_jobs, patch.object(
                runner, "write_detector_rate_posterior", return_value=None
            ):
                runner.run_detector(
                    "WFI01",
                    None,
                    root,
                    False,
                    0.1,
                    tile_size=256,
                    workers=1,
                )

            output_root = root / "production" / "wfi01"
            self.assertEqual(run_emcee.call_args.args[0].name, "wfi01_tile_log_statistics_256_tile_000.csv")
            self.assertEqual(run_emcee.call_args.kwargs["trajectory"], "constant-rate")
            self.assertFalse(run_emcee.call_args.kwargs["fit_incidence_angle"])
            self.assertEqual(
                run_emcee.call_args.kwargs["trajectory_rate_prior_mean_nm_day"],
                runner.PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
            )
            self.assertEqual(
                run_emcee.call_args.kwargs["absolute_hgcdte_n_scale_prior_sigma_log"],
                runner.PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
            )
            self.assertTrue(run_emcee.call_args.kwargs["fit_ice_n_constant"])
            self.assertFalse(run_emcee.call_args.kwargs["fit_ar_k_constant"])
            self.assertEqual(
                run_emcee.call_args.kwargs["fixed_ice_n_log_constant"],
                runner.PRODUCTION_FIXED_ICE_N_LOG_CONSTANT,
            )
            self.assertEqual(len(run_tile_jobs.call_args.args[0]), 4)
            self.assertEqual(run_tile_jobs.call_args.args[1:], (1, 1800.0))
            self.assertTrue(output_root.is_dir())

    def test_run_detector_uses_supervised_tile_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            statistics = root / "wfi01_tile_log_statistics_256_count_corrected.csv"
            fields = ("tile_id", "group")
            with statistics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for tile_id in range(4):
                    writer.writerow({"tile_id": tile_id, "group": 0})
            reference_dir = root / "reference"
            reference_dir.mkdir()
            for name in (
                "WFI_RCS_LEDP_spectral_energy_distribution.ecsv",
                "WFI_Quantum_efficiency.ecsv",
            ):
                (reference_dir / name).write_text("reference\n")
            with patch.object(runner, "REFERENCE_DATA_DIR", reference_dir), patch.object(
                runner.emcee_warmup, "run_emcee"
            ), patch.object(
                runner, "_run_tile_jobs", return_value=[]
            ) as run_tile_jobs, patch.object(
                runner, "fit_tile"
            ), patch.object(runner, "write_detector_rate_posterior", return_value=None):
                runner.run_detector(
                    "WFI01",
                    None,
                    root,
                    False,
                    0.1,
                    tile_size=256,
                    workers=4,
                )

            self.assertEqual(len(run_tile_jobs.call_args.args[0]), 4)
            self.assertEqual(run_tile_jobs.call_args.args[1:], (2, 1800.0))

    def test_tile_timeout_is_retried_serially_without_aborting_batch(self):
        jobs = [(3, "tile-3"), (7, "tile-7")]
        with patch.object(
            runner,
            "_run_tile_batch",
            side_effect=[[(jobs[0], "exceeded timeout")], []],
        ) as run_tile_batch:
            failures = runner._run_tile_jobs(
                jobs,
                2,
                10.0,
                retry_timeout_seconds=20.0,
            )

        self.assertEqual(failures, [])
        self.assertEqual(run_tile_batch.call_count, 2)
        self.assertEqual(run_tile_batch.call_args_list[0].args[0], jobs)
        self.assertEqual(run_tile_batch.call_args_list[0].args[1:3], (2, 10.0))
        self.assertEqual(run_tile_batch.call_args_list[1].args[0], [jobs[0]])
        self.assertEqual(run_tile_batch.call_args_list[1].args[1:3], (1, 20.0))

    def test_tile_timeout_kills_only_that_worker_and_continues(self):
        failures = runner._run_tile_jobs(
            [(3,), (7,)],
            2,
            5.0,
            process_target=_timeout_test_process,
            retry_timeout_seconds=1.0,
        )

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], (3,))
        self.assertIn("timeout", failures[0][1])

    def test_statistics_reset_day_uses_first_program_1063_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wfi01_corrected.csv"
            fieldnames = [
                "tile_id",
                "group",
                "program",
                "days",
                "n_pair_valid",
                "sum_log_change",
                "sum_squared_log_change",
            ]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for group, program, day in (
                    (0, "1039", 0.0),
                    (1, "1063", 12.907397928240743),
                    (2, "1063", 12.91),
                ):
                    writer.writerow(
                        {
                            "tile_id": 0,
                            "group": group,
                            "program": program,
                            "days": day,
                            "n_pair_valid": 10,
                            "sum_log_change": 1.0,
                            "sum_squared_log_change": 2.0,
                        }
                    )
            self.assertAlmostEqual(
                runner.statistics_reset_day(path),
                12.907397928240743,
            )

    def test_archive_matches_production_model_requires_maxent_warmup_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            fit_dir = Path(temporary)
            np.savez(
                fit_dir / "posterior_samples.npz",
                production_model_version=np.asarray(runner.PRODUCTION_MODEL_VERSION),
                trajectory_regularization=np.asarray("none"),
                trajectory_entropy_weight=np.asarray(0.0),
                thickness_penalty_enabled=np.asarray(0),
                constant_rate_map_warmup=np.asarray(1),
            )
            self.assertTrue(runner.archive_matches_production_model(fit_dir))
            self.assertFalse(runner.archive_matches_production_model(fit_dir, 0.03))
            legacy_dir = fit_dir / "legacy"
            legacy_dir.mkdir()
            np.savez(
                legacy_dir / "posterior_samples.npz",
                trajectory_regularization=np.asarray("maxent"),
                trajectory_entropy_weight=np.asarray(0.1),
                constant_rate_map_warmup=np.asarray(1),
            )
            self.assertFalse(runner.archive_matches_production_model(legacy_dir))


if __name__ == "__main__":
    unittest.main()
