import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_wfi_tile_statistics_stream import (
    detector_observations,
    read_index,
    write_detector_statistics,
)


ROOT = Path(__file__).resolve().parent


class TestStreamingStatistics(unittest.TestCase):
    def test_all_sca_index_has_the_same_wfi01_sequence(self):
        rows = read_index(ROOT / "all_wfi_rcs_flat_index.csv")
        observations = detector_observations(rows, "WFI01")

        self.assertEqual(len(observations), 114)
        self.assertEqual({row["led"] for row in observations}, set(range(1, 7)))
        self.assertEqual({row["repeat"] for row in observations}, set(range(19)))
        self.assertEqual(observations[0]["group"], 0)
        self.assertEqual(observations[-1]["group"], 113)

    def test_statistics_writer_uses_only_fetched_arrays(self):
        observations = []
        for led in range(1, 7):
            for repeat in range(2):
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * 2 + repeat,
                        "file_set_name": f"led{led}_repeat{repeat}",
                        "baseline_file_set_name": f"led{led}_repeat0",
                        "exposure_start_time": f"2026-01-01T00:0{led}:00Z",
                        "days": float(repeat),
                        "cal_uri": f"led{led}_repeat{repeat}",
                    }
                )

        baseline = np.ones((4, 4), dtype=np.float32)
        dq = np.zeros((4, 4), dtype=np.uint32)
        fetched = []

        def fetch_image(uri):
            fetched.append(uri)
            repeat = int(uri[-1])
            return baseline + repeat * 0.1, dq

        with tempfile.TemporaryDirectory() as directory:
            output = write_detector_statistics(
                observations,
                fetch_image,
                Path(directory) / "tile_statistics.csv",
                tile_size=2,
            )
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            output.unlink()

            def should_not_fetch(_uri):
                raise AssertionError("resume unexpectedly redownloaded a completed exposure")

            resumed = write_detector_statistics(
                observations,
                should_not_fetch,
                output,
                tile_size=2,
            )
            self.assertEqual(resumed, output)
            self.assertTrue(output.exists())
            self.assertEqual(len(list((output.with_name(output.name + ".parts")).glob("group_*.csv"))), 12)

        self.assertEqual(len(rows), 6 * 2 * 4)
        self.assertEqual(len(fetched), 6 * 2)
        expected = 4 * np.log(np.float64(np.float32(1.1)))
        self.assertAlmostEqual(float(rows[-1]["sum_log_change"]), expected, places=12)


if __name__ == "__main__":
    unittest.main()
