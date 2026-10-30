import unittest

import numpy as np

from plot_wfi_tile136_model_residuals import (
    _experiment_comparison_label,
    _experiment_index_curve,
)


class ExperimentIndexPlotTests(unittest.TestCase):
    def test_comparison_labels_include_porosity_prior_width(self):
        self.assertEqual(
            _experiment_comparison_label(
                {
                    "porosity_model": True,
                    "metrics": {"porosity_solid_n_prior_sigma_log": 0.012},
                }
            ),
            "thickness-based porosity (n prior sigma_log = 0.012)",
        )

    def test_monotonic_plot_curve_does_not_overshoot(self):
        grid = np.linspace(0.0, 300.0, 1001)
        curve = _experiment_index_curve(
            np.asarray([0.0, 100.0, 200.0, 300.0]),
            np.asarray([1.30, 1.30, 1.26, 1.26]),
            grid,
            monotonic_decreasing=True,
        )

        self.assertLessEqual(float(np.max(np.diff(curve))), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
