import csv
import tempfile
import unittest
from pathlib import Path

from plot_wfi_photodiode_correction import (
    load_inventory,
    load_side_swap_events,
    plot_correction,
    write_rows,
)


class TestPhotodiodeCorrectionPlot(unittest.TestCase):
    def make_inventory(self, path):
        fields = (
            "detector",
            "led",
            "program",
            "repeat",
            "exposure_start_time",
            "days",
            "file_set_name",
            "photodiode_mean",
            "photodiode_reference",
            "photodiode_scale",
            "photodiode_change_pct",
            "photodiode_channel",
            "photodiode_source",
        )
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for led in range(1, 7):
                writer.writerow(
                    {
                        "detector": "WFI01",
                        "led": led,
                        "program": 1039,
                        "repeat": led,
                        "exposure_start_time": f"2026-09-12T00:0{led}:00Z",
                        "days": led / 10.0,
                        "file_set_name": f"set_{led}",
                        "photodiode_mean": 0.25,
                        "photodiode_reference": 0.30,
                        "photodiode_scale": 1.2,
                        "photodiode_change_pct": -16.6667,
                        "photodiode_channel": "pd2",
                        "photodiode_source": "roman-telem-ice-monitor",
                    }
                )

    def test_load_inventory_exposes_applied_scale_and_percent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wfi01_rrn_inventory.csv"
            self.make_inventory(path)

            rows, detector = load_inventory(path)

            self.assertEqual(detector, "WFI01")
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["led"], 1)
            self.assertAlmostEqual(rows[0]["photodiode_scale"], 1.2)
            self.assertAlmostEqual(rows[0]["correction_percent"], 20.0)

    def test_plot_and_tidy_csv_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "wfi01_rrn_inventory.csv"
            self.make_inventory(inventory)
            rows, detector = load_inventory(inventory)
            plot = root / "diagnostics" / "wfi01_pd.png"

            plot_correction(plot, rows, detector, quantity="percent")
            tidy = root / "pd.csv"
            write_rows(tidy, rows)

            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 0)
            self.assertTrue(tidy.is_file())
            with tidy.open(newline="") as handle:
                self.assertEqual(sum(1 for _row in csv.DictReader(handle)), 6)

    def test_side_swap_event_loads_photodiode_rows_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "side_swap.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "detector",
                        "led",
                        "channel",
                        "event_repeat",
                        "event_day",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "detector": "WFI01",
                        "led": 1,
                        "channel": "image",
                        "event_repeat": 686,
                        "event_day": 24.2,
                    }
                )
                writer.writerow(
                    {
                        "detector": "WFI01",
                        "led": 1,
                        "channel": "photodiode",
                        "event_repeat": 686,
                        "event_day": 24.3,
                    }
                )

            events = load_side_swap_events(path, detector="WFI01")

            self.assertEqual(events[1]["event_repeat"], 686)
            self.assertAlmostEqual(events[1]["event_day"], 24.3)


if __name__ == "__main__":
    unittest.main()
