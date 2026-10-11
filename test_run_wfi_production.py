import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_wfi_production as runner


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
            Path("corrected/wfi03_tile_log_statistics_512_count_corrected.csv"),
        )
        self.assertEqual(
            output,
            Path("corrected") / "production" / "wfi03",
        )
        self.assertEqual(split.parent, output)
        self.assertEqual(fits.parent, output)
        self.assertEqual(emcee, output / "emcee")

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
                trajectory_regularization=np.asarray("maxent"),
                trajectory_entropy_weight=np.asarray(0.1),
                constant_rate_map_warmup=np.asarray(1),
            )
            self.assertTrue(runner.archive_matches_production_model(fit_dir))
            self.assertFalse(runner.archive_matches_production_model(fit_dir, 0.03))


if __name__ == "__main__":
    unittest.main()
