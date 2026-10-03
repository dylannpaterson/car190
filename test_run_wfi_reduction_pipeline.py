import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_wfi_reduction_pipeline import (
    find_previous_statistics,
    source_changes_require_rebuild,
    sync_s3_source,
    utc_run_date,
)


class ReductionPipelineTests(unittest.TestCase):
    def test_run_date_uses_compact_utc_date(self):
        self.assertEqual(utc_run_date("20261003"), "20261003")
        with self.assertRaises(ValueError):
            utc_run_date("2026-10-03")

    def test_s3_sync_restricts_downloads_to_calibrated_products(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            with patch("run_wfi_reduction_pipeline.subprocess.run") as run:
                result = sync_s3_source(
                    "s3://roman-s3-romcom/activities/CAR190/",
                    cache,
                    aws_binary="aws",
                    aws_profile="nexus",
                )

        self.assertEqual(result, cache)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["aws", "--profile", "nexus", "s3"])
        self.assertIn("--exclude", command)
        self.assertIn("*_cal.asdf", command)
        self.assertIn("--delete", command)
        run.assert_called_once_with(command, check=True)

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


if __name__ == "__main__":
    unittest.main()
