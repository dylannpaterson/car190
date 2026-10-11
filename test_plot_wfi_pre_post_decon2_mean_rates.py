import unittest

import numpy as np

from plot_wfi_pre_post_decon2_mean_rates import endpoint_mean_rates


class TestEndpointMeanRates(unittest.TestCase):
    def test_uses_final_minus_initial_over_each_phase_duration(self):
        pre, post, pre_duration, post_duration = endpoint_mean_rates(
            np.asarray([0.0, 1.0, 2.0, 4.0, 5.0, 6.0]),
            np.asarray([1.0, 3.0, 5.0, 2.0, 4.0, 8.0]),
            decon2_day=3.0,
        )

        self.assertEqual(pre_duration, 2.0)
        self.assertEqual(post_duration, 2.0)
        self.assertEqual(pre, 2.0)
        self.assertEqual(post, 3.0)


if __name__ == "__main__":
    unittest.main()
