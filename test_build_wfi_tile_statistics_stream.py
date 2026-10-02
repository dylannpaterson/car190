import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_wfi_tile_statistics_stream import (
    LEGACY_FIELDS,
    detector_observations,
    read_index,
    update_detector_statistics_incremental,
    write_detector_statistics,
)


ROOT = Path(__file__).resolve().parent


class TestStreamingStatistics(unittest.TestCase):
    def test_index_rejects_non_rcs_programs(self):
        rows = read_index(ROOT / "all_wfi_rcs_flat_index.csv")
        rows[0] = dict(rows[0], program="1025")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed_index.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "only RCS program 1039"):
                read_index(path)

    def test_all_sca_index_has_the_same_wfi01_sequence(self):
        rows = read_index(ROOT / "all_wfi_rcs_flat_index.csv")
        observations = detector_observations(rows, "WFI01")
        sequence_count = len(
            {
                row["visit_id"][:-4]
                for row in rows
                if row["detector"] == "WFI01"
            }
        )

        self.assertEqual(len(observations), 257)
        self.assertEqual({row["led"] for row in observations}, set(range(1, 7)))
        self.assertEqual(
            {row["repeat"] for row in observations},
            set(range(sequence_count)),
        )
        self.assertEqual(observations[0]["group"], 0)
        self.assertEqual(observations[-1]["group"], len(observations) - 1)

    def test_observations_align_inserted_and_missing_led_visits(self):
        def row(led, sequence, start):
            return {
                "detector": "WFI01",
                "led": str(led),
                "fileSetName": f"led{led}_{sequence}",
                "exposure_start_time": f"2026-01-01T{start}:00Z",
                "visit_id": f"{sequence}{led + 1}001",
            }

        rows = []
        for led in range(1, 7):
            rows.append(row(led, "010390200100100", "00:00"))
            rows.append(row(led, "010390200200100", "01:00"))
            rows.append(row(led, "010390200300100", "02:00"))
        rows.extend(
            (
                row(1, "010250101000100", "00:10"),
                row(2, "010250101300100", "01:10"),
            )
        )

        observations = detector_observations(rows, "WFI01")
        by_file_set = {row["file_set_name"]: row for row in observations}

        self.assertEqual(len(observations), 20)
        self.assertEqual(
            {
                by_file_set[f"led{led}_010390200200100"]["repeat"]
                for led in range(1, 7)
            },
            {2},
        )
        self.assertEqual(
            by_file_set["led1_010250101000100"]["repeat"],
            1,
        )
        self.assertEqual(
            by_file_set["led2_010250101300100"]["repeat"],
            3,
        )
        self.assertEqual(
            sorted({row["group"] for row in observations}),
            list(range(len(observations))),
        )
        node_days = [
            np.mean(
                [row["days"] for row in observations if row["repeat"] == repeat]
            )
            for repeat in sorted({row["repeat"] for row in observations})
        ]
        self.assertTrue(np.all(np.diff(node_days) > 0.0))
        for led in range(1, 7):
            self.assertEqual(
                by_file_set[f"led{led}_010390200100100"][
                    "baseline_file_set_name"
                ],
                f"led{led}_010390200100100",
            )

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
        current_value = np.float64(np.float32(1.1))
        self.assertAlmostEqual(float(rows[-1]["sum_reference_counts"]), 4.0)
        self.assertAlmostEqual(
            float(rows[-1]["sum_squared_reference_counts"]), 4.0
        )
        self.assertAlmostEqual(
            float(rows[-1]["sum_current_counts"]), 4.0 * current_value
        )
        self.assertAlmostEqual(
            float(rows[-1]["sum_squared_current_counts"]), 4.0 * current_value**2
        )
        self.assertAlmostEqual(
            float(rows[-1]["sum_reference_current_counts"]), 4.0 * current_value
        )
        self.assertAlmostEqual(float(rows[-1]["min_reference_count"]), 1.0)
        self.assertAlmostEqual(float(rows[-1]["max_reference_count"]), 1.0)
        self.assertAlmostEqual(float(rows[-1]["min_current_count"]), current_value)
        self.assertAlmostEqual(float(rows[-1]["max_current_count"]), current_value)

    def test_incremental_update_does_not_refetch_existing_exposures(self):
        def observation(led, repeat, name, days):
            return {
                "led": led,
                "repeat": repeat,
                "group": (led - 1) * 4 + repeat,
                "file_set_name": name,
                "baseline_file_set_name": f"led{led}_old0",
                "exposure_start_time": f"2026-01-01T00:{led:02d}:{repeat:02d}Z",
                "days": days,
                "cal_uri": name,
            }

        old_observations = [
            observation(led, repeat, f"led{led}_old{repeat}", float(repeat))
            for led in range(1, 7)
            for repeat in range(2)
        ]
        new_observations = []
        for led in range(1, 7):
            new_observations.extend(
                (
                    observation(led, 0, f"led{led}_old0", 0.0),
                    observation(led, 1, f"led{led}_newmid", 0.5),
                    observation(led, 2, f"led{led}_old1", 1.0),
                    observation(led, 3, f"led{led}_newlast", 1.5),
                )
            )

        baseline = np.ones((4, 4), dtype=np.float32)
        dq = np.zeros((4, 4), dtype=np.uint32)

        def fetch_initial(uri):
            return baseline + (0.1 if "old1" in uri else 0.0), dq

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tile_statistics.csv"
            write_detector_statistics(
                old_observations,
                fetch_initial,
                output,
                tile_size=2,
            )
            fetched = []

            def fetch_incremental(uri):
                fetched.append(uri)
                return baseline + (0.2 if "newmid" in uri else 0.3), dq

            update_detector_statistics_incremental(
                new_observations,
                fetch_incremental,
                output,
                tile_size=2,
            )
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            fingerprints = {
                row["file_set_name"]: row["sum_log_change"] for row in rows
            }
            initial_fetches = list(fetched)

            relabeled_observations = []
            for index, item in enumerate(new_observations):
                relabeled = dict(item)
                relabeled["repeat"] = 10 + index
                relabeled["group"] = 100 + index
                relabeled["days"] = 10.0 + index
                relabeled_observations.append(relabeled)
            fetched.clear()
            update_detector_statistics_incremental(
                relabeled_observations,
                fetch_incremental,
                output,
                tile_size=2,
            )
            with output.open(newline="") as handle:
                relabeled_rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 6 * 4 * 4)
        self.assertEqual(len(initial_fetches), 6 * 3)
        self.assertFalse(any("old1" in uri for uri in initial_fetches))
        old1_rows = [row for row in rows if row["file_set_name"].endswith("old1")]
        self.assertEqual({int(row["group"]) for row in old1_rows}, {2, 6, 10, 14, 18, 22})
        self.assertEqual({int(row["repeat"]) for row in old1_rows}, {2})
        self.assertEqual(fetched, [])
        self.assertEqual(
            {
                row["file_set_name"]: row["sum_log_change"]
                for row in relabeled_rows
            },
            fingerprints,
        )
        relabeled_old1 = [
            row for row in relabeled_rows if row["file_set_name"].endswith("old1")
        ]
        self.assertEqual({int(row["group"]) for row in relabeled_old1}, {102, 106, 110, 114, 118, 122})
        self.assertEqual(
            {int(row["repeat"]) for row in relabeled_old1},
            {12, 16, 20, 24, 28, 32},
        )

    def test_incremental_update_preserves_legacy_log_schema(self):
        def observation(led, repeat, name):
            return {
                "led": led,
                "repeat": repeat,
                "group": (led - 1) * 2 + repeat,
                "file_set_name": name,
                "baseline_file_set_name": f"led{led}_old",
                "exposure_start_time": f"2026-01-01T00:{led:02d}:{repeat:02d}Z",
                "days": float(repeat),
                "cal_uri": name,
            }

        old_observations = [
            observation(led, 0, f"led{led}_old") for led in range(1, 7)
        ]
        all_observations = old_observations + [
            observation(led, 1, f"led{led}_new") for led in range(1, 7)
        ]
        baseline = np.ones((4, 4), dtype=np.float32)
        dq = np.zeros((4, 4), dtype=np.uint32)

        def fetch_image(uri):
            return baseline + (0.1 if uri.endswith("new") else 0.0), dq

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "legacy_statistics.csv"
            write_detector_statistics(
                old_observations,
                fetch_image,
                output,
                tile_size=2,
            )
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEGACY_FIELDS)
                writer.writeheader()
                writer.writerows(
                    {field: row[field] for field in LEGACY_FIELDS} for row in rows
                )

            update_detector_statistics_incremental(
                all_observations,
                fetch_image,
                output,
                tile_size=2,
            )
            with output.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), LEGACY_FIELDS)
                self.assertEqual(len(list(reader)), 6 * 2 * 4)


if __name__ == "__main__":
    unittest.main()
