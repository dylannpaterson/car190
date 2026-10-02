import unittest

import numpy as np

from fit_wfi_count_common_shift_free_scale_mcmc import (
    common_delta_bounds,
    initial_center,
    log_probability,
)


def make_model():
    segments = []
    for index in range(2):
        segments.append(
            {
                "detector": f"WFI0{index + 1}",
                "led": 1,
                "reset_day": 10.0,
                "shift_prior_lower": -3.0,
                "shift_prior_upper": -1.0,
                "pre_days": np.asarray([0.0, 1.0, 2.0]),
                "post_days": np.asarray([2.0, 3.0]),
                "pre_values": np.asarray([[10.0, 20.0, 30.0]]),
                "post_values": np.asarray([[20.0, 30.0]]) * (1.0 + 0.1 * index),
                "post_weights": np.ones((1, 2)),
            }
        )
    return {"segments": segments, "n_segments": 2, "reset_day": 10.0}


class CommonShiftTests(unittest.TestCase):
    def test_common_prior_is_intersection_of_segment_priors(self):
        model = make_model()
        self.assertEqual(common_delta_bounds(model), (7.0, 9.0))

    def test_common_posterior_requires_the_shared_shift(self):
        model = make_model()
        exact = log_probability(model, np.asarray([9.0, 0.0, np.log(1.1), 0.0, 0.0]))
        wrong = log_probability(model, np.asarray([7.5, 0.0, np.log(1.1), 0.0, 0.0]))
        self.assertGreater(exact, wrong)

    def test_initial_center_has_one_shift_and_one_scale_per_segment(self):
        center, _ = initial_center(make_model(), grid_size=21)
        self.assertEqual(center.shape, (5,))
        self.assertAlmostEqual(center[0], 9.0, places=2)


if __name__ == "__main__":
    unittest.main()
