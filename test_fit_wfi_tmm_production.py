import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fit_wfi_tmm_production as production


class FixedProductionSettingsTests(unittest.TestCase):
    def test_settings_use_pre_corrected_statistics_and_disable_fit_corrections(self):
        statistics_path = Path(
            "rrn_reduction_20261002/wfi02_tile_log_statistics_512_count_corrected.csv"
        )
        with patch.object(production, "_PRODUCTION_STATISTICS_PATH", statistics_path):
            args = production.parse_args()

        self.assertIsNone(args.fixed_corrections)
        self.assertTrue(args.corrections_already_applied)
        self.assertFalse(args.fit_late_acm_step)
        self.assertFalse(args.fit_additional_acm_step)
        self.assertTrue(args.fit_post_reset_tile_log_offset)
        self.assertEqual(args.post_reset_tile_log_offset_prior_sigma, 0.1)
        self.assertTrue(args.fit_pre_post_ice_index)
        self.assertEqual(args.ice_reset_day, production.PRODUCTION_RESET_DAY)
        self.assertTrue(args.tile_initial_thickness_lognormal)
        self.assertEqual(args.tile_initial_thickness_log_median_nm, 10.0)
        self.assertEqual(args.tile_initial_thickness_log_sigma, 1.5)
        self.assertTrue(args.tile_rate_variation)
        self.assertTrue(args.tile_rate_trajectory_variation)
        self.assertFalse(args.tile_rate_neighbor_smoothness)
        self.assertTrue(args.map_only)
        self.assertEqual(args.map_steps, 10000)

    def test_main_runs_every_corrected_statistics_file_without_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wfi01_tile_log_statistics_512_count_corrected.csv").touch()
            (root / "wfi02_tile_log_statistics_512_count_corrected.csv").touch()
            output_root = root / "outputs"
            with (
                patch.object(production, "PRODUCTION_INPUT_ROOT", root),
                patch.object(production, "PRODUCTION_OUTPUT_ROOT", output_root),
                patch.object(production, "_fit_one") as fit_one,
                patch.object(sys, "argv", ["fit_wfi_tmm_production.py"]),
            ):
                production.main()

        self.assertEqual(fit_one.call_count, 2)
        self.assertEqual(
            production._PRODUCTION_STATISTICS_PATH.name,
            "wfi02_tile_log_statistics_512_count_corrected.csv",
        )

    def test_main_rejects_command_line_arguments(self):
        with patch.object(
            sys, "argv", ["fit_wfi_tmm_production.py", "--map-steps", "1"]
        ):
            with self.assertRaises(SystemExit):
                production.main()


if __name__ == "__main__":
    unittest.main()
