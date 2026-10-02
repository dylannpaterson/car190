import unittest

import numpy as np

from optimize_wfi_tmm_power import (
    optimization_bounds,
    pack_parameters,
    unpack_parameters,
)


class TestWfiTmmPowerOptimization(unittest.TestCase):
    def test_pack_and_unpack_round_trip(self):
        optical = np.asarray([0.5, 1.2, 2.0, 1.8, 1.5, 2.4])
        packed = pack_parameters(optical, 12.5, 37.5, 1.1, 0.8)
        (
            unpacked_optical,
            unpacked_rate,
            unpacked_initial_thickness,
            unpacked_n_scale,
            unpacked_k_scale,
        ) = unpack_parameters(packed)
        np.testing.assert_allclose(unpacked_optical, optical)
        self.assertAlmostEqual(unpacked_rate, 12.5)
        self.assertAlmostEqual(unpacked_initial_thickness, 37.5)
        self.assertAlmostEqual(unpacked_n_scale, 1.1)
        self.assertAlmostEqual(unpacked_k_scale, 0.8)

    def test_bounds_cover_optics_rate_thickness_and_substrate_scales(self):
        bounds = optimization_bounds((1.0, 50.0))
        self.assertEqual(len(bounds), 10)
        self.assertEqual(bounds[6], (np.log(1.0), np.log(50.0)))
        self.assertEqual(bounds[7], (0.0, 100.0))
        self.assertEqual(bounds[8], (np.log(0.5), np.log(3.0)))
        self.assertEqual(bounds[9], (np.log(0.1), np.log(5.0)))
