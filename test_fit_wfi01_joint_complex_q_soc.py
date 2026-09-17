import unittest

import numpy as np

from fit_wfi01_joint_complex_q_soc import (
    build_wfi01_response,
    q_factor_grid,
    trajectory_plot_series,
)


class TestComplexQResponse(unittest.TestCase):
    def test_trajectory_plot_exposes_reference_baseline(self):
        fit = {
            "node_days": np.asarray([0.00348543016975, 0.5]),
            "node_incremental_thickness_nm": np.asarray([0.05, 4.0]),
            "node_thickness_nm": np.asarray([10.05, 14.0]),
            "initial_thickness_nm": 10.0,
        }

        days, incremental, absolute = trajectory_plot_series(fit)

        np.testing.assert_array_equal(days, [0.0, 0.00348543016975, 0.5])
        np.testing.assert_array_equal(incremental, [0.0, 0.05, 4.0])
        np.testing.assert_array_equal(absolute, [10.0, 10.05, 14.0])

    def test_precomputed_broadband_integration_matches_direct_trapezoid(self):
        model_wavelengths = np.asarray([500.0, 600.0, 700.0])
        q_values = np.asarray([0.04 + 0.01j, 0.03 - 0.01j, 0.02 + 0.02j])
        alpha_values = np.zeros(3)
        d_grid = np.asarray([0.0, 10.0, 25.0])
        wavelengths = np.asarray([450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0])
        spectra = {
            band: {
                "wavelength": wavelengths,
                "raw_output": np.ones_like(wavelengths),
            }
            for band in range(1, 7)
        }
        qe_values = np.ones((len(wavelengths), 1))
        parameters = np.asarray([0.1, -0.2, 0.03, 0.1, 1.0, 0.0])

        response = build_wfi01_response(
            model_wavelengths,
            q_values,
            alpha_values,
            spectra,
            wavelengths,
            qe_values,
            d_grid,
            "signed",
            parameters,
        )
        model_factors = q_factor_grid(
            model_wavelengths,
            q_values,
            alpha_values,
            d_grid,
            parameters,
        )
        factors = np.ones((len(d_grid), len(wavelengths)))
        supported = (wavelengths >= model_wavelengths[0]) & (
            wavelengths <= model_wavelengths[-1]
        )
        for index, row in enumerate(model_factors):
            factors[index, supported] = np.interp(
                wavelengths[supported], model_wavelengths, row
            )
        expected = np.trapezoid(factors * wavelengths[None, :], wavelengths, axis=1)
        expected /= np.trapezoid(wavelengths, wavelengths)

        np.testing.assert_allclose(response[0, 0], expected, rtol=1.0e-13)
        for led in range(1, 6):
            np.testing.assert_allclose(response[0, led], response[0, 0])

    def test_q_model_rejects_nonphysical_attenuation(self):
        factors = q_factor_grid(
            np.asarray([500.0, 600.0]),
            np.asarray([0.1 + 0.0j, 0.1 + 0.0j]),
            np.asarray([0.01, 0.01]),
            np.asarray([0.0, 10.0]),
            np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
        )
        self.assertIsNone(factors)

    def test_optical_path_phase_uses_wavenumber(self):
        wavelengths = np.asarray([1000.0, 1400.0, 2000.0])
        q_base = np.asarray([0.1 + 0.02j] * 3)
        alpha = np.zeros(3)
        d_grid = np.asarray([20.0])
        parameters = np.asarray([0.0, 0.0, 0.0, 100.0, 1.0, 0.0])

        factors = q_factor_grid(
            wavelengths,
            q_base,
            alpha,
            d_grid,
            parameters,
            phase_model="optical_path",
        )
        phase_shift = 2.0 * np.pi * 100.0 * (
            1.0 / wavelengths - 1.0 / 1400.0
        )
        q = q_base * np.exp(1j * phase_shift)
        thickness_phase = 4.0 * np.pi * 1.25 * 20.0 / wavelengths
        expected = np.abs(1.0 - q) ** 2 / np.abs(
            1.0 - q * np.exp(1j * thickness_phase)
        ) ** 2

        np.testing.assert_allclose(factors[0], expected)


if __name__ == "__main__":
    unittest.main()
