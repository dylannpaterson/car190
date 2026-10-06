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


if __name__ == "__main__":
    unittest.main()
