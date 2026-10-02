import unittest

import numpy as np

from grid_search_wfi_tmm_power import (
    build_source_integrations,
    integrated_absolute_response,
    optical_parameter_grid,
    predict_constant_rate_log_ratios,
    tmm_power_curves,
)


class TestWfiTmmPowerGrid(unittest.TestCase):
    def test_zero_ice_has_unit_absolute_ratio(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        parameters = np.asarray([0.5, 1.5, 0.8, 2.0, 1.2, 2.5])
        clean, coated = tmm_power_curves(
            wavelengths,
            parameters,
            0.0,
            use_hgcdte_optical_constants=False,
        )
        np.testing.assert_allclose(coated, clean, rtol=0.0, atol=1.0e-12)

    def test_vectorized_thickness_matches_scalar_tmm(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        parameters = np.asarray([0.5, 1.5, 0.8, 2.0, 1.2, 2.5])
        thicknesses = np.asarray([0.0, 10.0, 25.0])
        _, vectorized = tmm_power_curves(
            wavelengths,
            parameters,
            thicknesses,
            use_hgcdte_optical_constants=False,
        )
        scalar = np.column_stack(
            [
                tmm_power_curves(
                    wavelengths,
                    parameters,
                    thickness,
                    use_hgcdte_optical_constants=False,
                )[1]
                for thickness in thicknesses
            ]
        )
        np.testing.assert_allclose(vectorized, scalar, rtol=0.0, atol=1.0e-12)

    def test_hgcdte_scales_change_absolute_substrate_power(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        parameters = np.asarray([0.5, 1.5, 0.8, 2.0, 1.2, 2.5])
        clean_default, coated_default = tmm_power_curves(
            wavelengths,
            parameters,
            20.0,
            use_hgcdte_optical_constants=False,
        )
        clean_scaled, coated_scaled = tmm_power_curves(
            wavelengths,
            parameters,
            20.0,
            use_hgcdte_optical_constants=False,
            hgcdte_n_scale=1.2,
            hgcdte_k_scale=0.7,
        )
        self.assertFalse(np.allclose(clean_default, clean_scaled))
        self.assertFalse(np.allclose(coated_default, coated_scaled))

    def test_absolute_response_uses_no_qe_and_normalizes_at_zero(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        spectra = {
            led: {
                "wavelength": wavelengths,
                "raw_output": np.asarray([1.0, 2.0, 1.0]),
            }
            for led in range(1, 7)
        }
        source = build_source_integrations(spectra, sed_mode="core", sed_weighting="energy")
        response = integrated_absolute_response(
            source,
            np.asarray([0.5, 1.5, 0.8, 2.0, 1.2, 2.5]),
            [0.0, 10.0],
            use_hgcdte_optical_constants=False,
        )
        np.testing.assert_allclose(response[:, 0], 1.0, rtol=0.0, atol=1.0e-12)

    def test_grid_has_six_optical_dimensions(self):
        candidates = list(optical_parameter_grid(3))
        self.assertEqual(len(candidates), 3**6)
        self.assertTrue(all(np.asarray(candidate).shape == (6,) for candidate in candidates))

    def test_constant_rate_prediction_uses_each_led_baseline(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        spectra = {
            led: {
                "wavelength": wavelengths,
                "raw_output": np.asarray([1.0, 2.0, 1.0]),
            }
            for led in range(1, 7)
        }
        source = build_source_integrations(spectra, sed_mode="core", sed_weighting="energy")
        data = {
            "days": np.asarray([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2),
            "counts": np.ones((1, 12)),
            "sums": np.zeros((1, 12)),
        }
        data["sums"][0, ::2] = 0.01
        prediction = predict_constant_rate_log_ratios(
            data,
            source,
            np.asarray([0.5, 1.5, 0.8, 2.0, 1.2, 2.5]),
            10.0,
            use_hgcdte_optical_constants=False,
        )
        np.testing.assert_allclose(prediction[::2], 0.01, rtol=0.0, atol=1.0e-12)
        self.assertTrue(np.all(np.isfinite(prediction)))
