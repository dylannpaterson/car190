import unittest

import numpy as np

from summarize_wfi01_optical_prior_sensitivity import one_standard_error_selection


class TestOpticalPriorSelection(unittest.TestCase):
    def test_selects_simplest_model_inside_best_models_one_se_limit(self):
        rows = [
            {"held_out_rmse_log": 1.0, "admissible_physical_thickness": True},
            {"held_out_rmse_log": 0.9, "admissible_physical_thickness": True},
            {"held_out_rmse_log": 0.7, "admissible_physical_thickness": False},
        ]
        scores = [
            np.asarray([0.99, 1.01, 1.00, 1.00]),
            np.asarray([0.7, 1.1, 0.7, 1.1]),
            np.asarray([0.7, 0.7, 0.7, 0.7]),
        ]

        primary, best, standard_error, limit = one_standard_error_selection(
            rows, scores
        )

        self.assertEqual(best, 1)
        self.assertEqual(primary, 0)
        self.assertAlmostEqual(standard_error, np.std(scores[1], ddof=1) / 2.0)
        self.assertAlmostEqual(limit, 0.9 + standard_error)

    def test_uses_fold_scores_aligned_with_actual_best_row(self):
        rows = [
            {"held_out_rmse_log": 0.8, "admissible_physical_thickness": True},
            {"held_out_rmse_log": 0.9, "admissible_physical_thickness": True},
        ]
        scores = [np.asarray([0.8, 0.8]), np.asarray([0.0, 1.8])]

        primary, best, standard_error, limit = one_standard_error_selection(
            rows, scores
        )

        self.assertEqual(best, 0)
        self.assertEqual(primary, 0)
        self.assertEqual(standard_error, 0.0)
        self.assertEqual(limit, 0.8)


if __name__ == "__main__":
    unittest.main()
