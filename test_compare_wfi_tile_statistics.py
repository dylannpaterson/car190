import csv
import tempfile
import unittest
from pathlib import Path

from compare_wfi_tile_statistics import COUNT_FIELDS, compare


class CompareTileStatisticsTests(unittest.TestCase):
    def write_rows(self, path, rows):
        fields = [
            "file_set_name",
            "tile_id",
            "n_pair_valid",
            "sum_log_change",
            "sum_squared_log_change",
        ]
        if rows and set(COUNT_FIELDS).issubset(rows[0]):
            fields.extend(COUNT_FIELDS)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_reports_common_and_unmatched_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.csv"
            new = root / "new.csv"
            output = root / "comparison.csv"
            common = root / "common.csv"
            self.write_rows(
                old,
                [
                    {
                        "file_set_name": "a",
                        "tile_id": 0,
                        "n_pair_valid": 10,
                        "sum_log_change": 2,
                        "sum_squared_log_change": 0.5,
                    },
                    {
                        "file_set_name": "old-only",
                        "tile_id": 0,
                        "n_pair_valid": 10,
                        "sum_log_change": 0,
                        "sum_squared_log_change": 0,
                    },
                ],
            )
            self.write_rows(
                new,
                [
                    {
                        "file_set_name": "a",
                        "tile_id": 0,
                        "n_pair_valid": 9,
                        "sum_log_change": 1,
                        "sum_squared_log_change": 0.25,
                    },
                    {
                        "file_set_name": "new-only",
                        "tile_id": 0,
                        "n_pair_valid": 10,
                        "sum_log_change": 0,
                        "sum_squared_log_change": 0,
                    },
                ],
            )
            summary = compare(old, new, output, common)
            self.assertEqual(summary["common_rows"], 1)
            self.assertEqual(summary["old_only_rows"], 1)
            self.assertEqual(summary["new_only_rows"], 1)
            self.assertEqual(summary["valid_count_delta_min"], -1)
            self.assertTrue(output.exists())
            with common.open() as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, 1)

    def test_reports_reference_and_current_count_mean_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.csv"
            new = root / "new.csv"
            count_values = {
                "sum_reference_counts": 100,
                "sum_squared_reference_counts": 1000,
                "sum_current_counts": 120,
                "sum_squared_current_counts": 1440,
                "sum_reference_current_counts": 1100,
                "min_reference_count": 1,
                "max_reference_count": 20,
                "min_current_count": 2,
                "max_current_count": 24,
            }
            old_row = {
                "file_set_name": "a",
                "tile_id": 0,
                "n_pair_valid": 10,
                "sum_log_change": 0,
                "sum_squared_log_change": 0,
                **count_values,
            }
            new_row = {
                **old_row,
                "sum_reference_counts": 110,
                "sum_current_counts": 130,
            }
            self.write_rows(old, [old_row])
            self.write_rows(new, [new_row])

            summary = compare(old, new)

            self.assertTrue(summary["count_statistics_compared"])
            self.assertEqual(summary["max_abs_reference_mean_count_delta"], 1.0)
            self.assertEqual(summary["max_abs_current_mean_count_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
