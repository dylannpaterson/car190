import unittest

import numpy as np

from plot_wfi_pre_post_ice_ratio import matched_epoch_ratio


class TestMatchedEpochRatio(unittest.TestCase):
    def test_aligns_decon2_to_decon2_day_and_computes_fractional_difference(self):
        elapsed, pre, post, ratio = matched_epoch_ratio(
            np.asarray([0.0, 1.0, 2.0, 4.0, 5.0, 6.0]),
            np.asarray([1.0, 3.0, 5.0, 2.0, 4.0, 8.0]),
            decon2_day=3.0,
            sample_count=3,
        )

        np.testing.assert_allclose(elapsed, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(pre, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(post, [2.0, 4.0, 8.0])
        np.testing.assert_allclose(ratio, [1.0, 1.0 / 3.0, 3.0 / 5.0])


if __name__ == "__main__":
    unittest.main()
