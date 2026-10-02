import tempfile
import unittest
from pathlib import Path

import numpy as np

from fit_wfi_count_shift_scale_mcmc import (
    _correction_path,
    load_model_data,
    _segment_prediction,
    _statistics_paths,
    log_probability,
    segment_log_probability,
)


def make_segment(detector, led, parameter_index, offset):
    return {
        "detector": detector,
        "detector_index": 0,
        "led": led,
        "parameter_index": parameter_index,
        "reset_day": 10.0,
        "pre_days": np.asarray([0.0, 1.0, 2.0]),
        "post_days": np.asarray([2.0, 3.0]),
        "pre_values": np.asarray([[offset, offset + 10.0, offset + 20.0]]),
        "post_values": np.asarray([[offset + 10.0, offset + 20.0]]),
        "post_weights": np.ones((1, 2)),
        "shift_prior_lower": -3.0,
        "shift_prior_upper": -1.0,
    }


class ShiftScaleModelTests(unittest.TestCase):
    def test_multi_detector_segments_keep_each_detector_tiles(self):
        model = load_model_data(
            "rrn_reduction_20260930",
            "rrn_reduction_20260930",
            15.064405023148149,
            39,
            35,
            ["WFI01", "WFI03"],
        )
        self.assertEqual(len(model["segments"]), 12)
        for segment in model["segments"]:
            self.assertEqual(segment["pre_values"].shape[0], 64)
            self.assertEqual(segment["post_values"].shape[0], 64)

    def test_segment_prediction_uses_segment_reset_day(self):
        segment = make_segment("WFI01", 1, 0, 0.0)
        prediction = _segment_prediction(segment, 9.0)
        np.testing.assert_allclose(prediction, [[10.0, 20.0]])

    def test_independent_segment_posterior_uses_its_own_scale_and_noise(self):
        segment = make_segment("WFI01", 1, 0, 0.0)
        exact = segment_log_probability(segment, np.asarray([9.0, 0.0, 0.0]))
        wrong_scale = segment_log_probability(segment, np.asarray([9.0, np.log(2.0), 0.0]))
        self.assertGreater(exact, wrong_scale)

    def test_detector_led_scales_are_independent(self):
        model = {
            "reset_day": 10.0,
            "n_segments": 2,
            "n_data": 4.0,
            "segments": [
                make_segment("WFI01", 1, 0, 0.0),
                make_segment("WFI03", 1, 1, 100.0),
            ],
        }
        model["segments"][1]["post_values"] *= 2.0
        parameters = np.asarray([9.0, 9.0, 0.0, np.log(2.0), np.log(1.0)])
        independent = log_probability(model, parameters)
        shared = log_probability(model, np.asarray([9.0, 9.0, 0.0, 0.0, np.log(1.0)]))
        self.assertTrue(np.isfinite(independent))
        self.assertGreater(independent, shared)

    def test_directory_inputs_select_raw_512_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "wfi01_tile_log_statistics_512.csv",
                "wfi01_tile_log_statistics_512_count_corrected.csv",
                "wfi01_tile_log_statistics_64.csv",
                "wfi03_tile_log_statistics_512.csv",
            ):
                (root / name).touch()
            selected = _statistics_paths(root, ["WFI03", "WFI01"])
        self.assertEqual([name for name, _ in selected], ["WFI03", "WFI01"])
        self.assertEqual([path.name for _, path in selected], [
            "wfi03_tile_log_statistics_512.csv",
            "wfi01_tile_log_statistics_512.csv",
        ])

    def test_detector_correction_table_is_selected_from_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "wfi03_count_space_fixed_corrections_three_events.csv"
            selected.touch()
            self.assertEqual(_correction_path(root, "WFI03", 2), selected)


if __name__ == "__main__":
    unittest.main()
