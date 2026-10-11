import csv
from datetime import datetime, timedelta, timezone
import tempfile
from pathlib import Path
import unittest

from thin_wfi_statistics import (
    complete_program_cycles,
    select_program_1063_repeats,
    thin_statistics_directory,
)


def row(program, repeat, led, time):
    return {
        "led": str(led),
        "program": str(program),
        "repeat": str(repeat),
        "group": str(repeat * 10 + led),
        "exposure_start_time": time.isoformat().replace("+00:00", "Z"),
        "days": "0",
        "n_pair_valid": "1",
        "sum_log_change": "0",
        "sum_squared_log_change": "0",
    }


class ThinWfiStatisticsTests(unittest.TestCase):
    def test_selects_complete_cycles_at_least_one_hour_apart(self):
        start = datetime(2026, 9, 24, tzinfo=timezone.utc)
        rows = []
        for repeat in range(10, 18):
            for led in range(1, 7):
                rows.append(row("1063", repeat, led, start + timedelta(minutes=30 * (repeat - 10))))
        rows.append(row("1063", 99, 3, start + timedelta(hours=20)))

        selected = select_program_1063_repeats(rows, 1.0)

        self.assertEqual([cycle[2] for cycle in selected], [10, 12, 14, 16])
        self.assertEqual(len(complete_program_cycles(rows)), 8)
        for previous, current in zip(selected, selected[1:]):
            self.assertGreaterEqual(
                (current[0] - previous[1]).total_seconds(), 3600.0
            )

    def test_directory_preserves_program_1039_and_writes_manifest(self):
        start = datetime(2026, 9, 24, tzinfo=timezone.utc)
        fields = [
            "led",
            "program",
            "repeat",
            "group",
            "exposure_start_time",
            "days",
            "n_pair_valid",
            "sum_log_change",
            "sum_squared_log_change",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            source_path = source / "wfi01_tile_log_statistics_512_count_corrected.csv"
            rows = [row("1039", 0, 1, start)]
            for repeat in range(10, 14):
                for led in range(1, 7):
                    rows.append(
                        row(
                            "1063",
                            repeat,
                            led,
                            start + timedelta(minutes=30 * (repeat - 10)),
                        )
                    )
            with source_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            manifest = thin_statistics_directory(source, output, 1.0)

            with (output / source_path.name).open(newline="") as handle:
                retained = list(csv.DictReader(handle))
            self.assertEqual(len(retained), 1 + 2 * 6)
            self.assertEqual(retained[0]["program"], "1039")
            self.assertEqual(manifest["files"]["WFI01"]["selected_repeats"], [10, 12])
            self.assertTrue(
                (output / "program1063_cadence_thinning_manifest.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
