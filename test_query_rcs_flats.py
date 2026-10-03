import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from query_rcs_flats import INDEX_FIELDS, refresh_index


class QueryRcsFlatsTests(unittest.TestCase):
    def test_refresh_index_uses_token_and_writes_all_rows(self):
        row = {field: f"{field}-value" for field in INDEX_FIELDS}
        row.update(
            {
                "fileSetName": "r0103902017001007001_0001",
                "detector": "WFI01",
                "exposure_start_time": "2026-01-01T00:00:00Z",
                "led": "6",
                "visit_id": "0103902017001007001",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "mast_api_token.txt"
            output_path = Path(directory) / "all_wfi_rcs_flat_index.csv"
            token_path.write_text("MAST_API_TOKEN=secret-token\n")
            with (
                patch("query_rcs_flats.MastMissions") as missions_class,
                patch("query_rcs_flats.query_detector", return_value=[object()]),
                patch("query_rcs_flats.table_to_index_rows", return_value=[row]),
            ):
                rows = refresh_index(
                    output_path,
                    token_path=token_path,
                    detectors=("WFI01",),
                )
            output_text = output_path.read_text()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fileSetName"], row["fileSetName"])
        self.assertEqual(
            missions_class.return_value.login.call_args.kwargs["token"],
            "secret-token",
        )
        self.assertIn("fileSetName", output_text)


if __name__ == "__main__":
    unittest.main()
