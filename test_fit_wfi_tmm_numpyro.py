import unittest
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fit_wfi_tmm_numpyro import (
    DEFAULT_SINGLE_SCA_TILE_STATISTICS,
    apply_fixed_corrections,
    ar_coordinate_rotation,
    REFERENCE_THICKNESS_NM,
    REFERENCE_TIME_DAYS,
    RESET_DAY,
    build_model,
    build_prediction_function,
    CONDUCTIVE_EPSILON_INF,
    CONDUCTIVE_REFERENCE_WAVELENGTH_NM,
    default_likelihood_weight,
    dielectric_ar_reference,
    early_time_scale_configuration,
    format_command_line,
    integrate_rate_trajectory,
    initial_values_for_data,
    load_map_warm_start,
    load_response_context,
    load_tile_statistics,
    map_parameter_scale_values,
    prepare_response_integration,
    ar_prior_arrays,
    aggregate_tile_sufficient_statistics,
    load_asw_optical_constants,
    hgcdte_reference_optical_constants,
    hgcdte_modulation_basis,
    ice_modulation_basis,
    incidence_cone_quadrature,
    incidence_pupil_quadrature,
    incidence_pupil_ray_sines,
    jitter_unconstrained_initial_params,
    load_incidence_chief_ray_tilts,
    qe_columns_for_scas,
    require_numpyro,
    qe_rbf_configuration,
    run_map_optimization,
    run_trust_exact_map_optimization,
    statistics_fingerprint,
    tile_neighbor_edges,
    rate_curvature_matrix,
    trajectory_rate_whitener,
    trajectory_rate_posterior_rotation,
    trajectory_rate_per_scale,
    transfer_map_initial_values,
)
from plot_wfi_tmm_numpyro import (
    apply_early_time_data_correction,
    acm_step_corner_values,
    hierarchy_sigma_labels,
    save_global_corner_plots,
)


def scalar_coherent_transmission(wavelength, layers, substrate_index):
    matrix = np.eye(2, dtype=complex)
    for index, thickness in layers:
        phase = 2.0 * np.pi * index * thickness / wavelength
        layer = np.asarray(
            [
                [np.cos(phase), -1j * np.sin(phase) / index],
                [-1j * index * np.sin(phase), np.cos(phase)],
            ]
        )
        matrix = matrix @ layer
    denominator = (
        matrix[0, 0]
        + substrate_index * matrix[0, 1]
        + matrix[1, 0]
        + substrate_index * matrix[1, 1]
    )
    return np.real(substrate_index) * abs(2.0 / denominator) ** 2


def scalar_single_film_scattering(wavelength, index, thickness):
    """Independent Airy calculation for a film between matched air media."""
    phase = 2.0 * np.pi * index * thickness / wavelength
    reflection_01 = (1.0 - index) / (1.0 + index)
    reflection_12 = (index - 1.0) / (index + 1.0)
    transmission_01 = 2.0 / (1.0 + index)
    transmission_12 = 2.0 * index / (index + 1.0)
    denominator = 1.0 + reflection_01 * reflection_12 * np.exp(2j * phase)
    reflection = (
        reflection_01 + reflection_12 * np.exp(2j * phase)
    ) / denominator
    transmission = (
        transmission_01 * transmission_12 * np.exp(1j * phase)
    ) / denominator
    return abs(reflection) ** 2, abs(transmission) ** 2


def scalar_detector_absorption(wavelength, layers, detector_index, detector_thickness, backside_index):
    """Return finite-detector absorption from independent boundary fluxes."""
    def matrix(index, thickness):
        phase = 2.0 * np.pi * index * thickness / wavelength
        return np.asarray(
            [
                [np.cos(phase), -1j * np.sin(phase) / index],
                [-1j * index * np.sin(phase), np.cos(phase)],
            ]
        )

    detector_boundary = matrix(detector_index, detector_thickness) @ np.asarray(
        [1.0, backside_index], dtype=complex
    )
    full_boundary = detector_boundary
    for index, thickness in reversed(layers):
        full_boundary = matrix(index, thickness) @ full_boundary
    normalization = 2.0 / (full_boundary[0] + full_boundary[1])
    return np.real(
        detector_boundary[0] * np.conj(detector_boundary[1])
        - backside_index
    ) * abs(normalization) ** 2


class TestTmmTileData(unittest.TestCase):
    def test_fixed_corrections_transform_log_ratio_sufficient_statistics(self):
        data = {
            "days": np.asarray([0.0, 1.0]),
            "counts": np.asarray([[2.0, 3.0]]),
            "sums": np.asarray([[1.0, 2.0]]),
            "sum_squared": np.asarray([[0.5, 2.0]]),
            "step_design": np.asarray([0.0, 1.0]),
            "late_step_design": np.asarray([0.0, 0.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.csv"
            path.write_text(
                "group,days,log_gain_correction,acm_step_log_offset,"
                "late_acm_step_log_offset\n"
                "0,0,0.1,0.3,0.4\n"
                "1,1,0.2,0.3,0.4\n"
            )
            corrected = apply_fixed_corrections(data, path)
        delta = np.asarray([[0.1, -0.1]])
        np.testing.assert_allclose(corrected["sums"], data["sums"] + data["counts"] * delta)
        np.testing.assert_allclose(
            corrected["sum_squared"],
            data["sum_squared"] + 2.0 * delta * data["sums"] + data["counts"] * delta**2,
        )
        self.assertTrue(corrected["fixed_corrections_applied"])
        self.assertEqual(
            corrected["fixed_acm_step_log_offset"],
            0.3,
        )
        self.assertEqual(
            corrected["fixed_late_acm_step_log_offset"],
            0.4,
        )

    def test_fixed_corrections_follow_original_group_ids(self):
        data = {
            "groups": np.asarray([4, 7]),
            "days": np.asarray([0.0, 1.0]),
            "counts": np.asarray([[2.0, 3.0]]),
            "sums": np.asarray([[1.0, 2.0]]),
            "sum_squared": np.asarray([[0.5, 2.0]]),
            "step_design": np.asarray([0.0, 1.0]),
            "late_step_design": np.asarray([0.0, 0.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.csv"
            path.write_text(
                "group,days,log_gain_correction,acm_step_log_offset,"
                "late_acm_step_log_offset\n"
                "4,0,0.1,0.3,0.4\n"
                "7,1,0.2,0.3,0.4\n"
            )
            corrected = apply_fixed_corrections(data, path)
        np.testing.assert_allclose(
            corrected["fixed_log_gain_correction"],
            [0.1, 0.2],
        )
        self.assertTrue(Path(corrected["fixed_corrections_path"]).is_absolute())

    def test_fixed_corrections_apply_additional_acm_step(self):
        data = {
            "groups": np.asarray([0, 1]),
            "days": np.asarray([0.0, 1.0]),
            "counts": np.asarray([[2.0, 3.0]]),
            "sums": np.asarray([[1.0, 2.0]]),
            "sum_squared": np.asarray([[0.5, 2.0]]),
            "step_design": np.asarray([0.0, 1.0]),
            "late_step_design": np.asarray([0.0, 0.0]),
            "additional_step_design": np.asarray([0.0, 1.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.csv"
            path.write_text(
                "group,days,log_gain_correction,acm_step_log_offset,"
                "additional_acm_step_log_offset,late_acm_step_log_offset\n"
                "0,0,0.1,0.3,0.7,0.4\n"
                "1,1,0.2,0.3,0.7,0.4\n"
            )
            corrected = apply_fixed_corrections(data, path)
        delta = np.asarray([[0.1, -0.8]])
        np.testing.assert_allclose(
            corrected["sums"], data["sums"] + data["counts"] * delta
        )
        self.assertEqual(
            corrected["fixed_additional_acm_step_log_offset"],
            0.7,
        )

    def test_fixed_corrections_apply_pre_reset_acm_step(self):
        data = {
            "groups": np.asarray([0, 1]),
            "days": np.asarray([0.0, 1.0]),
            "counts": np.asarray([[2.0, 3.0]]),
            "sums": np.asarray([[1.0, 2.0]]),
            "sum_squared": np.asarray([[0.5, 2.0]]),
            "step_design": np.asarray([0.0, 1.0]),
            "pre_reset_step_design": np.asarray([0.0, 1.0]),
            "late_step_design": np.asarray([0.0, 0.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.csv"
            path.write_text(
                "group,days,log_gain_correction,acm_step_log_offset,"
                "pre_reset_acm_step_log_offset,late_acm_step_log_offset\n"
                "0,0,0.1,0.3,0.7,0.4\n"
                "1,1,0.2,0.3,0.7,0.4\n"
            )
            corrected = apply_fixed_corrections(data, path)
        delta = np.asarray([[0.1, -0.8]])
        np.testing.assert_allclose(
            corrected["sums"], data["sums"] + data["counts"] * delta
        )
        self.assertEqual(
            corrected["fixed_pre_reset_acm_step_log_offset"],
            0.7,
        )

    def test_tile_fixed_corrections_apply_before_tile_likelihood(self):
        data = {
            "groups": np.asarray([0, 1]),
            "days": np.asarray([0.0, 1.0]),
            "tile_ids": np.asarray([0, 1]),
            "n_tiles": 2,
            "n_scas": 1,
            "sca_names": np.asarray(["WFI03"]),
            "counts": np.asarray([[2.0, 3.0], [4.0, 5.0]]),
            "sums": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "sum_squared": np.asarray(
                [[0.5, 2.0], [4.5, 8.0]]
            ),
            "step_design": np.asarray([0.0, 1.0]),
            "late_step_design": np.asarray([0.0, 0.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed_tile.csv"
            path.write_text(
                "detector,tile_id,group,days,log_gain_correction,"
                "acm_step_log_offset,late_acm_step_log_offset\n"
                "WFI03,0,0,0,0.1,0.3,0.4\n"
                "WFI03,0,1,1,0.2,0.3,0.4\n"
                "WFI03,1,0,0,-0.1,0.5,0.6\n"
                "WFI03,1,1,1,-0.2,0.5,0.6\n"
            )
            corrected = apply_fixed_corrections(data, path)
        delta = np.asarray([[0.1, -0.1], [-0.1, -0.7]])
        np.testing.assert_allclose(
            corrected["sums"], data["sums"] + data["counts"] * delta
        )
        np.testing.assert_allclose(
            corrected["sum_squared"],
            data["sum_squared"]
            + 2.0 * delta * data["sums"]
            + data["counts"] * delta**2,
        )
        self.assertTrue(corrected["fixed_corrections_tile_aware"])
        np.testing.assert_allclose(
            corrected["fixed_acm_step_log_offset_by_row"], [0.3, 0.5]
        )

    def test_format_command_line_quotes_arguments(self):
        command = format_command_line(
            ["fit_wfi_tmm_numpyro.py", "--output-dir", "path with spaces"],
            "/venv/bin/python",
        )
        self.assertEqual(
            command,
            "/venv/bin/python fit_wfi_tmm_numpyro.py --output-dir 'path with spaces'",
        )

    def test_loads_asw_optical_constants(self):
        with patch(
            "fit_wfi_tmm_numpyro.read_source",
            return_value="1.1 1.29 0.0 0.0\n1.2 1.30 0.001 0.0\n",
        ):
            wavelengths, real_index, imaginary_index = load_asw_optical_constants(
                "https://example.test/asw.tab"
            )
        np.testing.assert_array_equal(wavelengths, [1100.0, 1200.0])
        np.testing.assert_array_equal(real_index, [1.29, 1.30])
        np.testing.assert_array_equal(imaginary_index, [0.0, 0.001])

    def test_hgcdte_reference_optical_constants_are_finite_and_passive(self):
        wavelengths = np.asarray([480.0, 1600.0, 2000.0, 2400.0, 2600.0, 2700.0])

        real_index, imaginary_index = hgcdte_reference_optical_constants(wavelengths)

        self.assertTrue(np.all(np.isfinite(real_index)))
        self.assertTrue(np.all(np.isfinite(imaginary_index)))
        self.assertTrue(np.all(real_index > 0.0))
        self.assertTrue(np.all(imaginary_index >= 0.0))
        self.assertGreater(imaginary_index[4], imaginary_index[-1])
        self.assertLess(imaginary_index[-1], 1.0e-3)
        np.testing.assert_allclose(real_index[1], 2.906516129032258, rtol=0.0, atol=1.0e-12)

    def test_hgcdte_modulation_basis_is_normalized(self):
        basis = hgcdte_modulation_basis([480.0, 1600.0, 2700.0])

        self.assertEqual(basis.shape[0], 3)
        self.assertTrue(np.all(np.isfinite(basis)))
        self.assertTrue(np.all(basis >= 0.0))
        np.testing.assert_allclose(np.sum(basis, axis=1), 1.0)

    def test_qe_rbf_configuration_validates_custom_basis(self):
        nodes, width, prior = qe_rbf_configuration(
            [600.0, 800.0, 1000.0], 120.0, 0.5
        )
        np.testing.assert_array_equal(nodes, [600.0, 800.0, 1000.0])
        self.assertEqual(width, 120.0)
        self.assertEqual(prior, 0.5)
        with self.assertRaises(ValueError):
            qe_rbf_configuration([800.0, 600.0], 120.0, 0.5)

    def test_incidence_cone_quadrature_is_solid_angle_normalized(self):
        angles_deg, weights = incidence_cone_quadrature(3.0)

        self.assertEqual(angles_deg.shape, (8,))
        self.assertEqual(weights.shape, (8,))
        self.assertTrue(np.all(angles_deg > 0.0))
        self.assertTrue(np.all(angles_deg < 3.0))
        np.testing.assert_allclose(np.sum(weights), 1.0)
        zero_angles, zero_weights = incidence_cone_quadrature(0.0)
        np.testing.assert_array_equal(zero_angles, [0.0])
        np.testing.assert_array_equal(zero_weights, [1.0])
        with self.assertRaises(ValueError):
            incidence_cone_quadrature(-1.0)
        with self.assertRaises(ValueError):
            incidence_cone_quadrature(90.0)

    def test_incidence_pupil_quadrature_is_uniform_area_and_normalized(self):
        directions, weights = incidence_pupil_quadrature(3.576334)

        self.assertEqual(directions.shape, (32, 3))
        self.assertEqual(weights.shape, (32,))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        np.testing.assert_allclose(np.sum(weights), 1.0)
        np.testing.assert_allclose(
            np.sum(directions[:, :2] * weights[:, None], axis=0), 0.0, atol=1e-15
        )
        self.assertTrue(np.all(directions[:, 2] > 0.0))
        zero_directions, zero_weights = incidence_pupil_quadrature(0.0)
        np.testing.assert_array_equal(zero_directions, [[0.0, 0.0, 1.0]])
        np.testing.assert_array_equal(zero_weights, [1.0])

    def test_incidence_pupil_applies_row_specific_chief_rays(self):
        sine, weights = incidence_pupil_ray_sines(
            3.576334, [[0.0, 0.0], [8.0, 0.0]]
        )

        self.assertEqual(sine.shape, (2, 32))
        self.assertEqual(weights.shape, (32,))
        self.assertLess(np.average(sine[0], weights=weights), 0.1)
        self.assertGreater(np.average(sine[1], weights=weights), 0.1)

    def test_load_incidence_chief_ray_tilts_requires_selected_rows(self):
        data = {
            "sca_names": np.asarray(["WFI01"]),
            "tile_ids": np.asarray([0, 3]),
            "n_scas": 1,
            "n_tiles": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.csv"
            path.write_text(
                "detector,tile_id,chief_ray_x_deg,chief_ray_y_deg\n"
                "WFI01,0,1.0,-2.0\n"
                "WFI01,3,3.0,4.0\n"
            )
            tilts = load_incidence_chief_ray_tilts(path, data)
        np.testing.assert_array_equal(tilts, [[1.0, -2.0], [3.0, 4.0]])

    def test_absorbing_layer_is_passive_and_matches_airy_solution(self):
        wavelength = 1000.0
        index = 2.0 + 0.2j
        thickness = 100.0

        reflection, expected_transmission = scalar_single_film_scattering(
            wavelength, index, thickness
        )
        transmission = scalar_coherent_transmission(
            wavelength, [(index, thickness)], 1.0 + 0.0j
        )

        self.assertAlmostEqual(transmission, expected_transmission, places=12)
        self.assertGreaterEqual(1.0 - reflection - transmission, 0.0)

    def test_pre_post_complex_ice_index_sites_and_response(self):
        jax, jnp, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((1, 2)),
            "sums": np.zeros((1, 2)),
            "sum_squared": np.ones((1, 2)),
            "days": np.asarray([0.0, 2.0]),
            "repeat_index": np.asarray([0, 1]),
            "node_days": np.asarray([0.0, 2.0]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
            "n_scas": 1,
            "n_tiles": 1,
            "sca_index": np.zeros(1, dtype=int),
            "repeat_values": np.asarray([0, 1]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            fully_np_trajectory=True,
            ice_reset_day=1.0,
            fit_pre_post_ice_index=True,
        )
        sites = trace(seed(model, jax.random.PRNGKey(53))).get_trace()
        for name in (
            "pre_reset_ice_n",
            "pre_reset_ice_k",
            "post_reset_ice_n",
            "post_reset_ice_k",
        ):
            self.assertIn(name, sites)
            self.assertTrue(np.isfinite(np.asarray(sites[name]["value"])))

        predictor = build_prediction_function(
            data,
            context,
            row_indices=np.asarray([0]),
            post_reset_ice_reset_day=1.0,
        )
        prediction = np.asarray(
            predictor(
                jnp.asarray([0.0, 100.0]),
                jnp.asarray([[0.0, 1.0, 0.0]]),
                jnp.asarray(0.0),
                post_reset_ice_index=jnp.asarray(1.30 + 0.01j),
                pre_reset_ice_index=jnp.asarray(1.20 + 0.02j),
            )
        )
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertNotAlmostEqual(prediction[0, 1], 0.0)

    def test_finite_detector_matches_independent_boundary_flux_calculation(self):
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "n_scas": 1,
            "sca_index": np.asarray([0]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
            "late_step_design": np.zeros(2),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        predictor = build_prediction_function(
            data,
            context,
            finite_hgcdte_detector=True,
            detector_backside_index=1.4,
            row_indices=np.asarray([0]),
        )
        prediction = np.asarray(
            predictor(
                jnp.asarray([0.0, 100.0]),
                jnp.asarray([[0.0, 1.0, 0.0]]),
                jnp.asarray(0.0),
                None,
                None,
                None,
                None,
                None,
                jnp.asarray(2500.0),
            )
        )
        detector_index = 3.6 + 0.7j
        clean_absorption = scalar_detector_absorption(
            1000.0, [], detector_index, 2500.0, 1.4
        )
        coated_absorption = scalar_detector_absorption(
            1000.0, [(1.25 + 0.0j, 100.0)], detector_index, 2500.0, 1.4
        )
        np.testing.assert_allclose(
            np.exp(prediction[0, 1]),
            coated_absorption / clean_absorption,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_absolute_substrate_power_normalizes_clean_power_and_keeps_baseline(self):
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "n_scas": 1,
            "sca_index": np.asarray([0]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
            "late_step_design": np.zeros(2),
            "counts": np.ones((1, 2)),
            "sums": np.asarray([[0.02, 0.03]]),
        }
        context = {
            "wavelengths": np.asarray([500.0, 700.0]),
            "model_weights": np.ones((1, 6, 2)),
            "outside_output": np.zeros((1, 6)),
            "clean_output": np.ones((1, 6)),
            "hgcdte_n": np.asarray([2.9, 2.9]),
            "hgcdte_k": np.asarray([0.05, 0.08]),
            "absolute_substrate_power": True,
        }
        predictor = build_prediction_function(
            data,
            context,
            row_indices=np.asarray([0]),
            use_hgcdte_optical_constants=True,
        )
        prediction = np.asarray(
            predictor(
                jnp.asarray([0.0, 100.0]),
                jnp.asarray([[0.0, 1.0, 0.0]]),
                jnp.asarray(0.0),
                jnp.asarray([0.0, 0.0]),
            )
        )
        self.assertAlmostEqual(prediction[0, 0], 0.02, places=12)
        self.assertTrue(np.isfinite(prediction[0, 1]))
        shifted_prediction = np.asarray(
            predictor(
                jnp.asarray([0.0, 100.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                jnp.asarray([0.0, 0.0]),
                sca_ar_opd_nm=jnp.asarray([15.0]),
            )
        )
        self.assertTrue(np.all(np.isfinite(shifted_prediction)))

        row_data = dict(data)
        row_data["sca_index"] = np.asarray([0, 0])
        row_data["counts"] = np.ones((2, 2))
        row_data["sums"] = np.asarray([[0.02, 0.03], [0.02, 0.03]])
        row_predictor = build_prediction_function(
            row_data,
            context,
            row_indices=np.asarray([0, 1]),
            use_hgcdte_optical_constants=True,
            fit_incidence_angle=True,
        )
        row_prediction = np.asarray(
            row_predictor(
                jnp.asarray([0.0, 100.0]),
                jnp.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
                jnp.asarray(0.0),
                jnp.asarray([[0.0, 0.0], [np.log(1.2), np.log(0.8)]]),
                incidence_angle_deg=jnp.asarray(10.0),
            )
        )
        self.assertEqual(row_prediction.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(row_prediction)))
        self.assertNotAlmostEqual(row_prediction[0, 1], row_prediction[1, 1])

    def test_absolute_response_context_omits_qe_weighting(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        spectra = {
            band: {
                "wavelength": wavelengths,
                "raw_output": np.ones_like(wavelengths),
            }
            for band in range(1, 7)
        }
        qe_values = np.column_stack(
            (np.ones(len(wavelengths)), np.asarray([1.0, 2.0, 4.0]))
        )
        with patch("fit_wfi_tmm_numpyro.load_sed", return_value=spectra), patch(
            "fit_wfi_tmm_numpyro.load_qe",
            return_value=(wavelengths, qe_values, ["SCU 1", "SCU 2"]),
        ):
            context = load_response_context(
                Path("model.ecsv"),
                "signed",
                1,
                ["WFI01", "WFI02"],
                absolute_substrate_power=True,
            )
        self.assertTrue(context["absolute_substrate_power"])
        np.testing.assert_allclose(context["outside_output"], 0.0)
        np.testing.assert_allclose(
            context["clean_output"][0], context["clean_output"][1]
        )
        np.testing.assert_allclose(
            context["model_weights"][0], context["model_weights"][1]
        )

    def test_absolute_response_context_prunes_zero_weight_wavelengths(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        spectra = {
            band: {
                "wavelength": wavelengths,
                "raw_output": np.ones_like(wavelengths),
            }
            for band in range(1, 7)
        }
        qe_values = np.ones((len(wavelengths), 1))
        full_grid = np.asarray([500.0, 550.0, 600.0, 650.0, 700.0])
        expected = prepare_response_integration(
            full_grid,
            spectra,
            wavelengths,
            qe_values,
            "signed",
        )
        with patch("fit_wfi_tmm_numpyro.load_sed", return_value=spectra), patch(
            "fit_wfi_tmm_numpyro.load_qe",
            return_value=(wavelengths, qe_values, ["SCU 1"]),
        ):
            context = load_response_context(
                Path("model.ecsv"),
                "signed",
                1,
                ["WFI01"],
                wavelength_step_nm=50.0,
                absolute_substrate_power=True,
            )

        np.testing.assert_array_equal(context["wavelengths"], wavelengths)
        np.testing.assert_allclose(
            context["model_weights"], expected[0][:, [0, 2, 4]][None, :, :]
        )
        np.testing.assert_allclose(context["outside_output"], expected[1][None, :])
        np.testing.assert_allclose(context["clean_output"], expected[2][None, :])

        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2, dtype=int),
            "step_design": np.zeros(12),
            "counts": np.ones((1, 12)),
            "sums": np.zeros((1, 12)),
            "sum_squared": np.ones((1, 12)),
        }
        full_context = {
            "wavelengths": full_grid,
            "model_weights": expected[0],
            "outside_output": expected[1],
            "clean_output": expected[2],
            "hgcdte_n": np.full(full_grid.shape, 3.6),
            "hgcdte_k": np.full(full_grid.shape, 0.7),
        }
        compressed_context = {
            **full_context,
            "wavelengths": wavelengths,
            "model_weights": expected[0][:, [0, 2, 4]],
            "hgcdte_n": np.full(wavelengths.shape, 3.6),
            "hgcdte_k": np.full(wavelengths.shape, 0.7),
        }
        full_predictor = build_prediction_function(
            data,
            full_context,
            use_hgcdte_optical_constants=True,
            absolute_substrate_power=True,
        )
        compressed_predictor = build_prediction_function(
            data,
            compressed_context,
            use_hgcdte_optical_constants=True,
            absolute_substrate_power=True,
        )
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        arguments = (
            jnp.asarray(np.linspace(0.0, 50.0, 12)),
            jnp.asarray([[1.25, 2.0, 0.0]]),
            jnp.asarray(0.0),
            jnp.asarray([[0.0, 0.0]]),
        )
        np.testing.assert_allclose(
            full_predictor(*arguments),
            compressed_predictor(*arguments),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_finite_detector_thickness_samples_are_per_tile_and_positive(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((2, 3)),
            "sums": np.zeros((2, 3)),
            "sum_squared": np.ones((2, 3)),
            "days": np.asarray([0.0, 1.0, 2.0]),
            "repeat_index": np.asarray([0, 1, 2]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.asarray([1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(3),
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "metadata": {
                0: {"tile_row": 0, "tile_col": 0},
                1: {"tile_row": 0, "tile_col": 1},
            },
            "sca_index": np.zeros(2, dtype=int),
            "repeat_values": np.asarray([0, 1, 2]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            finite_hgcdte_detector=True,
            fit_detector_thickness=True,
        )
        sites = trace(seed(model, jax.random.PRNGKey(43))).get_trace()

        self.assertEqual(
            sites["detector_log_thickness_nm"]["value"].shape, (1, 2)
        )
        thickness = np.asarray(sites["detector_thickness_nm"]["value"])
        self.assertEqual(thickness.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(thickness)))
        self.assertTrue(np.all(thickness > 0.0))

    def test_absolute_hgcdte_scales_are_sampled_per_tile(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((2, 3)),
            "sums": np.zeros((2, 3)),
            "sum_squared": np.ones((2, 3)),
            "days": np.asarray([0.0, 1.0, 2.0]),
            "repeat_index": np.asarray([0, 1, 2]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.asarray([1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(3),
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "metadata": {
                0: {"tile_row": 0, "tile_col": 0},
                1: {"tile_row": 0, "tile_col": 1},
            },
            "sca_index": np.zeros(2, dtype=int),
            "repeat_values": np.asarray([0, 1, 2]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "absolute_substrate_power": True,
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            absolute_substrate_power=True,
            use_hgcdte_optical_constants=True,
        )
        sites = trace(seed(model, jax.random.PRNGKey(47))).get_trace()

        self.assertEqual(
            sites["absolute_hgcdte_n_log_scale"]["value"].shape, (1, 2)
        )
        self.assertEqual(
            sites["absolute_hgcdte_k_log_scale"]["value"].shape, (1, 2)
        )

    def test_absolute_hgcdte_scales_can_be_shared_without_spatial_factors(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((2, 3)),
            "sums": np.zeros((2, 3)),
            "sum_squared": np.ones((2, 3)),
            "days": np.asarray([0.0, 1.0, 2.0]),
            "repeat_index": np.asarray([0, 1, 2]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.asarray([1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(3),
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "metadata": {
                0: {"tile_row": 0, "tile_col": 0},
                1: {"tile_row": 0, "tile_col": 1},
            },
            "sca_index": np.zeros(2, dtype=int),
            "repeat_values": np.asarray([0, 1, 2]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "absolute_substrate_power": True,
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            absolute_substrate_power=True,
            use_hgcdte_optical_constants=True,
            shared_absolute_hgcdte_scales=True,
            tile_rate_variation=True,
            tile_rate_neighbor_smoothness=False,
        )
        sites = trace(seed(model, jax.random.PRNGKey(48))).get_trace()

        self.assertEqual(
            sites["absolute_hgcdte_n_log_scale"]["value"].shape, (1,)
        )
        self.assertEqual(
            sites["absolute_hgcdte_k_log_scale"]["value"].shape, (1,)
        )
        self.assertNotIn("trajectory_rate_neighbor_smoothness", sites)
        self.assertNotIn("trajectory_rate_tile_node_neighbor_smoothness", sites)

    def test_independent_tile_trajectories_have_no_global_or_smoothing_terms(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((2, 4)),
            "sums": np.zeros((2, 4)),
            "sum_squared": np.ones((2, 4)),
            "days": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "repeat_index": np.asarray([0, 1, 2, 3]),
            "node_days": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "led": np.asarray([1, 1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(4),
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "metadata": {
                0: {"tile_row": 0, "tile_col": 0},
                1: {"tile_row": 0, "tile_col": 1},
            },
            "sca_index": np.zeros(2, dtype=int),
            "repeat_values": np.asarray([0, 1, 2, 3]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            fully_np_trajectory=True,
            tile_rate_trajectory_variation=True,
            independent_tile_trajectories=True,
            tile_rate_neighbor_smoothness=False,
            tile_rate_trajectory_smoothness_weight=0.0,
        )
        sites = trace(seed(model, jax.random.PRNGKey(49))).get_trace()

        self.assertEqual(
            sites["trajectory_rate_tile_node_log_rates"]["value"].shape,
            (1, 2, 4),
        )
        self.assertNotIn("np_log_rate_nodes", sites)
        self.assertNotIn("trajectory_rate_tile_offsets", sites)
        self.assertNotIn("trajectory_rate_tile_node_offsets", sites)
        self.assertNotIn("np_smoothness_regularization", sites)
        self.assertNotIn("trajectory_rate_neighbor_smoothness", sites)
        self.assertNotIn("trajectory_rate_tile_node_neighbor_smoothness", sites)
        self.assertNotIn("trajectory_rate_tile_node_smoothness", sites)

        smooth_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            fully_np_trajectory=True,
            tile_rate_trajectory_variation=True,
            independent_tile_trajectories=True,
            tile_rate_neighbor_smoothness=False,
            tile_rate_trajectory_smoothness_weight=0.01,
        )
        smooth_sites = trace(seed(smooth_model, jax.random.PRNGKey(50))).get_trace()

        self.assertIn("trajectory_rate_tile_node_smoothness", smooth_sites)
        self.assertNotIn("trajectory_rate_tile_node_neighbor_smoothness", smooth_sites)

    def test_per_tile_initial_thickness_uses_positive_lognormal_sites(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((2, 4)),
            "sums": np.zeros((2, 4)),
            "sum_squared": np.ones((2, 4)),
            "days": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "repeat_index": np.asarray([0, 1, 2, 3]),
            "node_days": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "led": np.asarray([1, 1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(4),
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "metadata": {
                0: {"tile_row": 0, "tile_col": 0},
                1: {"tile_row": 0, "tile_col": 1},
            },
            "sca_index": np.zeros(2, dtype=int),
            "repeat_values": np.asarray([0, 1, 2, 3]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            fully_np_trajectory=True,
            tile_rate_trajectory_variation=True,
            independent_tile_trajectories=True,
            tile_rate_neighbor_smoothness=False,
            tile_initial_thickness_lognormal=True,
            tile_initial_thickness_log_median_nm=10.0,
            tile_initial_thickness_log_sigma=1.5,
        )
        sites = trace(seed(model, jax.random.PRNGKey(51))).get_trace()

        log_thickness = np.asarray(sites["initial_thickness_log_nm"]["value"])
        thickness = np.asarray(sites["initial_thickness_nm"]["value"])
        self.assertEqual(log_thickness.shape, (1, 2))
        self.assertEqual(thickness.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(log_thickness)))
        self.assertTrue(np.all(np.isfinite(thickness)))
        self.assertTrue(np.all(thickness > 0.0))
        np.testing.assert_allclose(thickness, np.exp(log_thickness))
        self.assertNotIn("initial_thickness_prior", sites)
        self.assertNotIn("initial_thickness_pivot_z", sites)
        self.assertEqual(
            sites["tile_ice_thickness"]["value"].shape,
            (2, 4),
        )

        reset_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_parameters_across_tiles=True,
            fully_np_trajectory=True,
            tile_rate_trajectory_variation=True,
            independent_tile_trajectories=True,
            tile_rate_neighbor_smoothness=False,
            tile_initial_thickness_lognormal=True,
            tile_initial_thickness_log_median_nm=10.0,
            tile_initial_thickness_log_sigma=1.5,
            ice_reset_day=2.0,
        )
        reset_sites = trace(seed(reset_model, jax.random.PRNGKey(52))).get_trace()

        self.assertEqual(
            reset_sites["reset_initial_thickness_log_nm"]["value"].shape,
            (1, 2),
        )
        self.assertEqual(
            reset_sites["reset_initial_thickness_nm"]["value"].shape,
            (1, 2),
        )
        reset_thickness = np.asarray(
            reset_sites["reset_initial_thickness_nm"]["value"]
        ).reshape(-1)
        reset_trajectory = np.asarray(
            reset_sites["tile_ice_thickness"]["value"]
        )
        np.testing.assert_allclose(
            reset_trajectory[:, 2], reset_thickness, rtol=0.0, atol=1.0e-12
        )
        self.assertFalse(
            np.allclose(reset_trajectory[:, 0], reset_trajectory[:, 2])
        )

    def test_repeat_nodes_are_shared_and_time_ordered(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=2)

        repeat_count = len(data["repeat_values"])
        np.testing.assert_array_equal(
            data["repeat_values"], np.arange(repeat_count)
        )
        self.assertEqual(data["repeat_index"].min(), 0)
        self.assertEqual(data["repeat_index"].max(), repeat_count - 1)
        self.assertTrue(np.all(np.diff(data["repeat_index"]) >= 0))
        for repeat_index, repeat_value in enumerate(data["repeat_values"]):
            groups = data["repeat"] == repeat_value
            np.testing.assert_array_equal(
                np.unique(data["repeat_index"][groups]), [repeat_index]
            )
        self.assertEqual(len(data["node_days"]), repeat_count)
        self.assertTrue(np.all(np.diff(data["node_days"]) > 0.0))
        self.assertEqual(data["counts"].shape, (2, len(data["led"])))
        np.testing.assert_array_equal(data["sca_index"], np.zeros(2, dtype=int))

    def test_additional_acm_step_design_activates_at_requested_repeat(self):
        data = load_tile_statistics(
            DEFAULT_SINGLE_SCA_TILE_STATISTICS,
            max_tiles=1,
            additional_acm_step_repeat=39,
        )
        self.assertEqual(data["additional_acm_step_repeat"], 39)
        np.testing.assert_array_equal(
            np.unique(data["additional_step_design"][data["repeat"] < 39]),
            [0.0],
        )
        np.testing.assert_array_equal(
            np.unique(data["additional_step_design"][data["repeat"] >= 39]),
            [1.0],
        )

    def test_pre_reset_acm_step_design_activates_at_requested_repeat(self):
        data = load_tile_statistics(
            DEFAULT_SINGLE_SCA_TILE_STATISTICS,
            max_tiles=1,
            pre_reset_acm_step_repeat=35,
        )
        self.assertEqual(data["pre_reset_acm_step_repeat"], 35)
        np.testing.assert_array_equal(
            np.unique(data["pre_reset_step_design"][data["repeat"] < 35]),
            [0.0],
        )
        np.testing.assert_array_equal(
            np.unique(data["pre_reset_step_design"][data["repeat"] >= 35]),
            [1.0],
        )

    def test_early_time_scales_are_shared_by_repeat_epoch(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=1)

        configuration = early_time_scale_configuration(data, 1.5)
        expected_nodes = np.flatnonzero(data["node_days"] <= 1.5)
        np.testing.assert_array_equal(
            configuration["repeat"], data["repeat_values"][expected_nodes]
        )
        np.testing.assert_allclose(
            configuration["days"], data["node_days"][expected_nodes]
        )
        for repeat_index in expected_nodes:
            groups = data["repeat_index"] == repeat_index
            np.testing.assert_array_equal(
                np.unique(configuration["observation_scale_index"][groups]),
                [np.flatnonzero(expected_nodes == repeat_index)[0]],
            )
        self.assertTrue(
            np.all(configuration["observation_scale_index"][data["node_days"][data["repeat_index"]] > 1.5] == -1)
        )
        last_repeat = data["repeat_values"][-1]
        self.assertTrue(
            np.all(data["late_step_design"][data["repeat"] < last_repeat] == 0.0)
        )
        self.assertTrue(
            np.all(data["late_step_design"][data["repeat"] == last_repeat] == 1.0)
        )

    def test_plot_early_time_correction_is_applied_to_data(self):
        data = {
            "node_days": np.asarray([0.0, 0.5, 1.0, 2.0]),
            "repeat_index": np.asarray([0, 1, 2, 3]),
            "repeat_values": np.asarray([0, 1, 2, 3]),
            "counts": np.asarray([[2.0, 2.0, 2.0, 2.0]]),
            "sums": np.asarray([[2.0, 4.0, 6.0, 8.0]]),
            "sum_squared": np.asarray([[2.0, 8.0, 18.0, 32.0]]),
        }
        predictions = np.zeros((3, 1, 4))
        corrected_data, corrected_predictions = apply_early_time_data_correction(
            data,
            predictions,
            np.asarray([[1.0, 2.0, 4.0]]),
            1.0,
        )
        correction = np.asarray([0.0, np.log(2.0), np.log(4.0), 0.0])
        np.testing.assert_allclose(
            corrected_data["sums"], data["sums"] + data["counts"] * correction
        )
        np.testing.assert_allclose(
            corrected_data["sum_squared"],
            data["sum_squared"]
            + 2.0 * data["sums"] * correction
            + data["counts"] * correction * correction,
        )
        np.testing.assert_allclose(
            corrected_predictions, predictions
        )
        np.testing.assert_allclose(
            corrected_data["early_time_scale_log_correction"], correction
        )

    def test_tile_neighbor_edges_follow_detector_grid(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS)

        edges = tile_neighbor_edges(data)

        self.assertGreater(len(edges), 0)
        self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
        self.assertTrue(np.all(edges < data["n_tiles"]))

    def test_combines_sca_statistics_with_shared_observation_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copy(
                DEFAULT_SINGLE_SCA_TILE_STATISTICS,
                directory / "wfi01_tile_log_statistics_512.csv",
            )
            shutil.copy(
                DEFAULT_SINGLE_SCA_TILE_STATISTICS,
                directory / "wfi02_tile_log_statistics_512.csv",
            )
            data = load_tile_statistics(directory, max_tiles=2)

        self.assertEqual(data["sca_names"].tolist(), ["WFI01", "WFI02"])
        self.assertEqual(data["n_scas"], 2)
        self.assertEqual(data["n_tiles"], 2)
        self.assertEqual(data["counts"].shape, (4, len(data["led"])))
        np.testing.assert_array_equal(data["sca_index"], [0, 0, 1, 1])

    def test_maps_wfi_labels_to_public_scu_qe_columns(self):
        fieldnames = [f"SCU {index}" for index in range(1, 19)]
        np.testing.assert_array_equal(
            qe_columns_for_scas(["WFI01", "WFI02", "WFI18"], fieldnames),
            [0, 1, 17],
        )
        with self.assertRaises(ValueError):
            qe_columns_for_scas(["WFI19"], fieldnames)

    def test_ar_prior_arrays_support_one_through_three_layers(self):
        reference, lower, upper = ar_prior_arrays(3)
        self.assertEqual(reference.shape, (9,))
        self.assertEqual(lower.shape, (9,))
        self.assertEqual(upper.shape, (9,))
        layers = reference.reshape(3, 3)
        self.assertTrue(np.all(np.diff(layers[:, 1]) > 0.0))
        np.testing.assert_allclose(layers[:, 2], 1.0e-4)
        np.testing.assert_allclose(
            4.0 * 100.0 * layers[:, 0] * layers[:, 1], 1050.0
        )
        np.testing.assert_allclose(reference, dielectric_ar_reference(3))
        with self.assertRaises(ValueError):
            ar_prior_arrays(4)

    def test_ar_coordinate_rotation_is_full_rank_and_orthogonal(self):
        active = np.asarray([0, 1, 3, 4, 6, 7])
        rotation = ar_coordinate_rotation(active, 3)

        self.assertEqual(rotation.shape, (6, 6))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(6), atol=1e-14)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(6), atol=1e-14)
        np.testing.assert_allclose(
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            @ rotation.T
            @ rotation,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

    def test_ar_coordinate_rotation_leaves_absorption_coordinates_unrotated(self):
        active = np.arange(9)
        rotation = ar_coordinate_rotation(active, 3)

        np.testing.assert_allclose(rotation.T @ rotation, np.eye(9), atol=1e-14)
        np.testing.assert_allclose(rotation[2], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(rotation[5], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_default_likelihood_weight_is_one_effective_pixel_per_group(self):
        counts = np.asarray([[100.0, 200.0, 300.0]])

        self.assertEqual(default_likelihood_weight(counts), 1.0 / 200.0)
        with self.assertRaises(ValueError):
            default_likelihood_weight([0.0, 1.0])

    def test_statistics_fingerprint_changes_when_sufficient_statistics_change(self):
        data = {
            "counts": np.asarray([[1.0, 2.0]]),
            "sums": np.asarray([[3.0, 4.0]]),
            "sum_squared": np.asarray([[5.0, 6.0]]),
        }
        changed = {key: value.copy() for key, value in data.items()}
        changed["sums"][0, 1] += 1.0

        self.assertNotEqual(statistics_fingerprint(data), statistics_fingerprint(changed))

    def test_trajectory_rate_scale_conversion_matches_fixture(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=1)

        self.assertAlmostEqual(
            trajectory_rate_per_scale(data["days"]), 6.705819170890919, places=12
        )

    def test_rate_integral_reproduces_square_root_baseline_without_np(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=1)
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        integration_times = np.unique(
            np.concatenate(
                (
                    [np.min(data["days"])],
                    data["node_days"],
                    [np.max(data["days"])],
                )
            )
        )
        integrated, instantaneous_rate, _, _ = integrate_rate_trajectory(
            jax,
            jnp,
            jnp.asarray(data["days"]),
            jnp.asarray(data["node_days"]),
            integration_times,
            0.0,
            jnp.zeros(len(data["node_days"])),
        )
        expected_thickness = REFERENCE_THICKNESS_NM * (
            np.sqrt((RESET_DAY + data["days"]) / REFERENCE_TIME_DAYS)
            - np.sqrt(RESET_DAY / REFERENCE_TIME_DAYS)
        )

        np.testing.assert_allclose(
            np.asarray(integrated), expected_thickness, rtol=0.0, atol=1.0e-10
        )
        self.assertAlmostEqual(
            float(np.asarray(instantaneous_rate)[0]),
            trajectory_rate_per_scale(data["days"]),
            places=12,
        )

    def test_rate_integral_supports_fully_nonparametric_positive_rates(self):
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        integrated, instantaneous_rate, node_integrated, node_rate = (
            integrate_rate_trajectory(
                jax,
                jnp,
                jnp.asarray([0.0, 1.0, 2.0]),
                jnp.asarray([0.0, 2.0]),
                jnp.asarray([0.0, 2.0]),
                0.0,
                jnp.full(2, np.log(2.0)),
                use_square_root=False,
            )
        )

        np.testing.assert_allclose(
            np.asarray(integrated), [0.0, 2.0, 4.0], rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            np.asarray(instantaneous_rate), 2.0, rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            np.asarray(node_integrated), [0.0, 4.0], rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            np.asarray(node_rate), 2.0, rtol=0.0, atol=1.0e-12
        )

    def test_rate_integral_returns_reset_anchor_integral(self):
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        values = integrate_rate_trajectory(
            jax,
            jnp,
            jnp.asarray([0.0, 1.0, 3.0]),
            jnp.asarray([0.0, 3.0]),
            jnp.asarray([0.0, 3.0]),
            0.0,
            jnp.full(2, np.log(2.0)),
            use_square_root=False,
            anchor_day=1.0,
        )

        integrated, _, _, _, anchor_integral = values
        np.testing.assert_allclose(
            np.asarray(integrated), [0.0, 2.0, 6.0], rtol=0.0, atol=1.0e-12
        )
        self.assertAlmostEqual(float(np.asarray(anchor_integral)), 2.0, places=12)

    def test_lbfgs_map_optimization_reports_gradient_convergence(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        import numpyro
        import numpyro.distributions as dist

        def model():
            value = numpyro.sample("value", dist.Normal(0.0, 1.0))
            numpyro.factor("data", -0.5 * ((value - 3.0) / 0.1) ** 2)

        map_values, diagnostics = run_map_optimization(
            model,
            {"value": np.asarray(0.0)},
            jax,
            maxiter=20,
            tolerance=1.0e-10,
        )

        self.assertEqual(diagnostics["optimizer"], "L-BFGS-B")
        self.assertTrue(diagnostics["converged"])
        self.assertLess(diagnostics["gradient_norm"], 1.0e-6)
        self.assertLess(diagnostics["projected_gradient_norm"], 1.0e-10)
        self.assertEqual(diagnostics["function_tolerance"], 0.0)
        self.assertLess(diagnostics["scaled_projected_gradient_norm"], 1.0e-10)
        self.assertAlmostEqual(float(map_values["value"]), 2.97029703, places=6)

    def test_lbfgs_prior_scaled_coordinates_preserve_map_solution(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        import numpyro
        import numpyro.distributions as dist

        def model():
            value = numpyro.sample("value", dist.Normal(0.0, 1.0))
            numpyro.factor("data", -0.5 * ((value - 3.0) / 0.1) ** 2)

        map_values, diagnostics = run_map_optimization(
            model,
            {"value": np.asarray(0.0)},
            jax,
            maxiter=20,
            tolerance=1.0e-10,
            parameter_scales={"value": np.asarray(0.1)},
        )

        self.assertTrue(diagnostics["converged"])
        self.assertAlmostEqual(float(map_values["value"]), 2.97029703, places=6)
        self.assertAlmostEqual(diagnostics["parameter_scale_min"], 0.1)
        self.assertAlmostEqual(diagnostics["parameter_scale_max"], 0.1)

    def test_map_parameter_scales_use_site_prior_widths(self):
        initial_values = {
            "ar_mu_raw": np.zeros(2),
            "ar_sigma_raw": np.zeros(2),
            "hgcdte_n_log_modulation": np.zeros(3),
            "early_time_log_scale": np.zeros(4),
            "acm_step_log_offset": np.zeros(6),
            "np_node_raw": np.zeros(2),
        }
        args = SimpleNamespace(
            ar_mean_prior_sigma=10.0,
            ar_tile_sigma_prior=2.5,
            hgcdte_modulation_prior_sigma_log=5.0,
            early_time_scale_prior_sigma=1.0,
            acm_step_prior_sigma=0.5,
        )

        scales = map_parameter_scale_values(initial_values, args)

        np.testing.assert_array_equal(scales["ar_mu_raw"], [10.0, 10.0])
        np.testing.assert_array_equal(scales["ar_sigma_raw"], [2.5, 2.5])
        np.testing.assert_array_equal(
            scales["hgcdte_n_log_modulation"], [5.0, 5.0, 5.0]
        )
        np.testing.assert_array_equal(
            scales["early_time_log_scale"], [1.0, 1.0, 1.0, 1.0]
        )
        np.testing.assert_array_equal(scales["acm_step_log_offset"], np.full(6, 0.5))
        np.testing.assert_array_equal(scales["np_node_raw"], [1.0, 1.0])

    def test_nuts_initial_jitter_is_reproducible_and_batched(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        initial_params = {
            "scalar": jax.numpy.asarray(2.0),
            "vector": jax.numpy.asarray([1.0, -1.0]),
        }

        first = jitter_unconstrained_initial_params(
            initial_params, jax, seed=17, chains=3, scale=0.05
        )
        second = jitter_unconstrained_initial_params(
            initial_params, jax, seed=17, chains=3, scale=0.05
        )

        for name in initial_params:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(first[name].shape, (3,) + initial_params[name].shape)
        self.assertFalse(np.array_equal(np.asarray(first["scalar"]), [2.0, 2.0, 2.0]))

        exact = jitter_unconstrained_initial_params(
            initial_params, jax, seed=17, chains=2, scale=0.0
        )
        np.testing.assert_array_equal(exact["scalar"], [2.0, 2.0])
        np.testing.assert_array_equal(exact["vector"], [[1.0, -1.0], [1.0, -1.0]])

    def test_rate_whitener_preserves_independent_plus_curvature_prior(self):
        node_days = np.asarray([0.0, 1.0, 2.5, 4.0, 7.0])
        prior_sigma = 1.5
        likelihood_scale = 3.2
        smoothness_weight = 4.6
        whitener = trajectory_rate_whitener(
            node_days, prior_sigma, likelihood_scale, smoothness_weight
        )
        curvature = rate_curvature_matrix(node_days)
        standard = np.asarray([0.3, -1.1, 0.7, 1.4, -0.2])
        mean = np.full(node_days.size, np.log(5.0))
        log_rates = mean + whitener @ standard

        original_quadratic = (
            -0.5 * np.sum(((log_rates - mean) / prior_sigma) ** 2)
            - likelihood_scale
            * smoothness_weight
            * np.mean((curvature @ log_rates) ** 2)
        )
        whitened_quadratic = -0.5 * np.sum(standard**2)
        self.assertAlmostEqual(original_quadratic, whitened_quadratic, places=10)

    def test_trajectory_posterior_rotation_uses_conditional_hessian(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        import numpyro
        import numpyro.distributions as dist

        precision = jax.numpy.asarray(
            [[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 3.0]]
        )

        def model():
            rate = numpyro.sample(
                "np_log_rate_z",
                dist.Normal(0.0, 1.0).expand((3,)).to_event(1),
            )
            numpyro.factor("conditional_rate_curvature", -0.5 * rate @ precision @ rate)

        rotation = trajectory_rate_posterior_rotation(
            model, {"np_log_rate_z": np.zeros(3)}, jax
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
        expected_eigenvalues = np.linalg.eigvalsh(np.eye(3) + np.asarray(precision))
        rotated_precision = rotation.T @ (np.eye(3) + np.asarray(precision)) @ rotation
        np.testing.assert_allclose(
            np.sort(np.diag(rotated_precision)), expected_eigenvalues, atol=1.0e-10
        )

    def test_whitened_model_does_not_double_count_rate_smoothness(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((1, 3)),
            "sums": np.zeros((1, 3)),
            "sum_squared": np.ones((1, 3)),
            "days": np.asarray([0.0, 1.0, 2.0]),
            "repeat_index": np.asarray([0, 1, 2]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.asarray([1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(3),
            "n_scas": 1,
            "n_tiles": 1,
            "sca_index": np.zeros(1, dtype=int),
            "repeat_values": np.asarray([0, 1, 2]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        common = dict(
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            fully_np_trajectory=True,
            np_smoothness_weight=1.0,
        )
        direct_sites = trace(
            seed(build_model(data, context, whiten_np_trajectory=False, **common), jax.random.PRNGKey(61))
        ).get_trace()
        whitened_sites = trace(
            seed(build_model(data, context, whiten_np_trajectory=True, **common), jax.random.PRNGKey(61))
        ).get_trace()

        self.assertIn("np_smoothness_regularization", direct_sites)
        self.assertNotIn("np_smoothness_regularization", whitened_sites)

        rotation = np.eye(3)
        rotated_sites = trace(
            seed(
                build_model(
                    data,
                    context,
                    whiten_np_trajectory=True,
                    np_rate_likelihood_rotation=rotation,
                    **common,
                ),
                jax.random.PRNGKey(62),
            )
        ).get_trace()
        self.assertIn("np_log_rate_likelihood_rotated", rotated_sites)
        self.assertEqual(
            rotated_sites["np_log_rate_likelihood_rotated"]["value"].shape,
            (3,),
        )

    def test_map_safe_pivot_extension_is_only_used_for_cold_map(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace
        from numpyro.infer.util import log_density

        data = {
            "counts": np.ones((1, 3)),
            "sums": np.zeros((1, 3)),
            "sum_squared": np.ones((1, 3)),
            "days": np.asarray([0.0, 1.0, 2.0]),
            "repeat_index": np.asarray([0, 1, 2]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.asarray([1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(3),
            "n_scas": 1,
            "n_tiles": 1,
            "sca_index": np.zeros(1, dtype=int),
            "repeat_values": np.asarray([0, 1, 2]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        common = dict(
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            fully_np_trajectory=True,
            whiten_np_trajectory=True,
            pivot_initial_thickness=True,
            initial_thickness_prior_mean_nm=0.0,
            initial_thickness_prior_sigma_nm=50.0,
        )
        safe_model = build_model(
            data, context, map_safe_initial_thickness=True, **common
        )
        sample_trace = trace(seed(safe_model, jax.random.PRNGKey(71))).get_trace()
        values = {
            name: site["value"]
            for name, site in sample_trace.items()
            if site["type"] == "sample"
        }
        values["initial_thickness_pivot_z"] = -100.0
        safe_log_joint, _ = log_density(safe_model, (), {}, values)
        self.assertTrue(np.isfinite(float(safe_log_joint)))

        exact_model = build_model(data, context, **common)
        exact_log_joint, _ = log_density(exact_model, (), {}, values)
        self.assertFalse(np.isfinite(float(exact_log_joint)))

    def test_map_warm_start_checks_data_fingerprint_and_site_shapes(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=1)
        with tempfile.TemporaryDirectory() as directory:
            warm_start_path = Path(directory) / "posterior_samples.npz"
            np.savez(
                warm_start_path,
                map_only=np.asarray(1),
                statistics_fingerprint=np.asarray(statistics_fingerprint(data)),
                value=np.asarray([[1.5, -0.25]]),
                acm_step_log_offset=np.asarray(
                    [[-0.01, -0.02, -0.03, -0.04, -0.05, -0.06]]
                ),
            )

            values = load_map_warm_start(
                warm_start_path,
                {
                    "value": np.zeros(2),
                    "new_site": np.asarray([0.25]),
                    "acm_step_log_offset": np.asarray(0.0),
                },
                data,
            )

        np.testing.assert_array_equal(values["value"], [1.5, -0.25])
        np.testing.assert_array_equal(values["new_site"], [0.25])
        self.assertAlmostEqual(float(values["acm_step_log_offset"]), -0.035)

        changed_data = dict(data)
        changed_data["counts"] = np.asarray(data["counts"]).copy()
        changed_data["counts"][0, 0] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            warm_start_path = Path(directory) / "posterior_samples.npz"
            np.savez(
                warm_start_path,
                map_only=np.asarray(1),
                statistics_fingerprint=np.asarray(statistics_fingerprint(data)),
                value=np.asarray([[1.5, -0.25]]),
            )
            with self.assertRaisesRegex(ValueError, "statistics do not match"):
                load_map_warm_start(
                    warm_start_path, {"value": np.zeros(2)}, changed_data
                )

    def test_map_warm_start_prefers_reparameterized_source_site(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=1)
        with tempfile.TemporaryDirectory() as directory:
            warm_start_path = Path(directory) / "posterior_samples.npz"
            np.savez(
                warm_start_path,
                map_only=np.asarray(1),
                statistics_fingerprint=np.asarray(statistics_fingerprint(data)),
                latent=np.asarray([99.0, 98.0]),
                constrained=np.asarray([1.5, -0.25]),
            )

            values = load_map_warm_start(
                warm_start_path,
                {"latent": np.zeros(2)},
                data,
                value_converters={
                    "latent": (
                        "constrained",
                        lambda value, _: np.asarray(value) + 2.0,
                    )
                },
            )

        np.testing.assert_array_equal(values["latent"], [3.5, 1.75])

    def test_lbfgs_map_optimization_bounds_hgcdte_sites(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        import numpyro
        import numpyro.distributions as dist

        def model():
            hgcdte_n = numpyro.sample(
                "hgcdte_n", dist.TruncatedNormal(3.6, 1.0, low=1.0, high=6.0)
            )
            hgcdte_k = numpyro.sample(
                "hgcdte_k", dist.TruncatedNormal(0.7, 1.0, low=0.0, high=3.0)
            )
            numpyro.factor(
                "data",
                -0.5 * ((hgcdte_n - 10.0) / 0.1) ** 2
                - 0.5 * ((hgcdte_k + 2.0) / 0.1) ** 2,
            )

        map_values, _ = run_map_optimization(
            model,
            {"hgcdte_n": np.asarray(3.6), "hgcdte_k": np.asarray(0.7)},
            jax,
            maxiter=20,
        )

        self.assertGreaterEqual(float(map_values["hgcdte_n"]), 1.0)
        self.assertLessEqual(float(map_values["hgcdte_n"]), 6.0)
        self.assertGreaterEqual(float(map_values["hgcdte_k"]), 0.0)
        self.assertLessEqual(float(map_values["hgcdte_k"]), 3.0)

    def test_trust_exact_map_optimization_reaches_interior_solution(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        import numpyro
        import numpyro.distributions as dist

        def model():
            x = numpyro.sample("x", dist.Normal(0.0, 1.0))
            numpyro.factor("target", -0.5 * ((x - 2.0) / 0.1) ** 2)

        map_values, diagnostics = run_trust_exact_map_optimization(
            model,
            {"x": np.asarray(0.0)},
            jax,
            maxiter=20,
            tolerance=1.0e-6,
        )

        self.assertTrue(diagnostics["converged"])
        self.assertAlmostEqual(float(map_values["x"]), 1.980198, places=5)
        with self.assertRaisesRegex(ValueError, "raw sample-site coordinates"):
            run_trust_exact_map_optimization(
                model,
                {"x": np.asarray(0.0)},
                jax,
                maxiter=20,
                parameter_scales={"x": np.asarray(1.0)},
            )

    def test_hierarchy_labels_follow_active_ar_parameter_width(self):
        self.assertEqual(
            hierarchy_sigma_labels(3, 6, "tile"),
            (
                "layer 1 tile sigma scale",
                "layer 1 tile sigma n",
                "layer 2 tile sigma scale",
                "layer 2 tile sigma n",
                "layer 3 tile sigma scale",
                "layer 3 tile sigma n",
            ),
        )
        self.assertEqual(
            hierarchy_sigma_labels(3, 3, "SCA"),
            (
                "layer 1 SCA sigma scale",
                "layer 2 SCA sigma scale",
                "layer 3 SCA sigma scale",
            ),
        )
        with self.assertRaises(ValueError):
            hierarchy_sigma_labels(3, 5, "tile")

    def test_global_corner_accepts_lossless_ar_hierarchy(self):
        sample_count = 20
        samples = {
            "ar_layers": np.asarray(3),
            "ar_mean": np.ones((sample_count, 9)),
            "ar_sigma_raw": np.ones((sample_count, 6)),
            "initial_thickness_nm": np.ones(sample_count),
            "mean_rate_nm_day": np.ones(sample_count),
            "acm_step_log_offset": np.zeros(sample_count),
        }

        with patch("plot_wfi_tmm_numpyro.save_corner_plot") as save_corner:
            save_global_corner_plots(Path("plots"), samples)

        self.assertEqual(save_corner.call_count, 2)
        hyper_values = save_corner.call_args_list[1].args[1]
        hyper_labels = save_corner.call_args_list[1].args[2]
        self.assertEqual(hyper_values.shape, (sample_count, 15))
        self.assertEqual(len(hyper_labels), hyper_values.shape[1])
        self.assertNotIn("layer 1 tile sigma k", hyper_labels)

    def test_global_corner_accepts_led_dependent_acm_steps(self):
        sample_count = 20
        samples = {
            "ar_layers": np.asarray(3),
            "ar_mean": np.ones((sample_count, 9)),
            "ar_sigma_raw": np.ones((sample_count, 6)),
            "initial_thickness_nm": np.ones(sample_count),
            "mean_rate_nm_day": np.ones(sample_count),
            "acm_step_log_offset": np.zeros((sample_count, 6)),
        }

        values, labels = acm_step_corner_values(samples)
        self.assertEqual(values.shape, (sample_count, 6))
        self.assertEqual(labels, tuple(f"ACM LED {led} step (%)" for led in range(1, 7)))
        with patch("plot_wfi_tmm_numpyro.save_corner_plot") as save_corner:
            save_global_corner_plots(Path("plots"), samples)

        global_values = save_corner.call_args_list[0].args[1]
        global_labels = save_corner.call_args_list[0].args[2]
        self.assertEqual(global_values.shape, (sample_count, 17))
        self.assertEqual(len(global_labels), global_values.shape[1])

    def test_single_tile_initialization_has_no_unidentifiable_tile_offsets(self):
        reference, lower, upper = ar_prior_arrays(3)

        class Args:
            ar_layers = 3
            shared_ar_optical_properties = True
            map_design_wavelength_nm = None
            map_middle_layer_eighth_wave = False
            np_modulation = False

        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "repeat_values": np.arange(2),
        }
        initial_values = initial_values_for_data(
            data, Args(), reference, lower, upper
        )

        self.assertNotIn("ar_sigma_raw", initial_values)
        self.assertNotIn("ar_z", initial_values)

    def test_initial_thickness_prior_and_map_start_are_configurable(self):
        reference, lower, upper = ar_prior_arrays(3)

        class Args:
            ar_layers = 3
            shared_ar_optical_properties = True
            map_design_wavelength_nm = None
            map_middle_layer_eighth_wave = False
            map_middle_layer_half_wave = False
            fit_ar_absorption = False
            np_modulation = False
            initial_thickness_init_nm = 1.0e-6

        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "repeat_values": np.arange(2),
        }
        initial_values = initial_values_for_data(
            data, Args(), reference, lower, upper
        )

        self.assertAlmostEqual(
            float(initial_values["initial_thickness_nm"]), 1.0e-6
        )

    def test_single_tile_model_samples_direct_ar_parameters(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((1, 2)),
            "sums": np.zeros((1, 2)),
            "sum_squared": np.ones((1, 2)),
            "days": np.asarray([0.0, 1.0]),
            "repeat_index": np.asarray([0, 1]),
            "node_days": np.asarray([0.0, 1.0]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
            "n_scas": 1,
            "n_tiles": 1,
            "sca_index": np.zeros(1, dtype=int),
            "repeat_values": np.asarray([0, 1]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
        )
        sites = trace(seed(model, jax.random.PRNGKey(4))).get_trace()

        self.assertIn("ar_mu_raw", sites)
        self.assertEqual(sites["ar_mu_raw"]["value"].shape, (6,))
        ar_parameters = np.asarray(sites["ar_parameters"]["value"]).reshape(3, 3)
        np.testing.assert_array_equal(ar_parameters[:, 2], 0.0)
        self.assertEqual(sites["instantaneous_rate_nm_day"]["value"].shape, (2,))
        self.assertEqual(
            sites["node_instantaneous_rate_nm_day"]["value"].shape, (2,)
        )
        self.assertEqual(sites["initial_rate_nm_day"]["value"].shape, ())
        self.assertNotIn("ar_sigma_raw", sites)
        self.assertNotIn("ar_z", sites)

        angle_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            fit_incidence_angle=True,
            incidence_angle_prior_sigma_deg=3.0,
        )
        angle_sites = trace(seed(angle_model, jax.random.PRNGKey(45))).get_trace()
        self.assertEqual(angle_sites["incidence_angle_deg"]["value"].shape, ())
        self.assertGreaterEqual(
            float(angle_sites["incidence_angle_deg"]["value"]), 0.0
        )
        self.assertLessEqual(
            float(angle_sites["incidence_angle_deg"]["value"]), 15.0
        )

        rotated_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            standardize_ar_mean=True,
            rotate_ar_coordinates=True,
        )
        rotated_sites = trace(seed(rotated_model, jax.random.PRNGKey(44))).get_trace()
        self.assertIn("ar_mu_z", rotated_sites)
        self.assertEqual(rotated_sites["ar_mu_z"]["value"].shape, (6,))
        self.assertTrue(np.all(np.isfinite(rotated_sites["ar_parameters"]["value"])))

        led_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            led_dependent_acm_step=True,
        )
        led_sites = trace(seed(led_model, jax.random.PRNGKey(5))).get_trace()
        self.assertEqual(led_sites["acm_step_log_offset"]["value"].shape, (6,))

        common_steps_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            led_dependent_acm_step=False,
            fit_late_acm_step=True,
        )
        common_steps = trace(
            seed(common_steps_model, jax.random.PRNGKey(51))
        ).get_trace()
        self.assertEqual(common_steps["acm_step_log_offset"]["value"].shape, ())
        self.assertEqual(common_steps["late_acm_step_log_offset"]["value"].shape, ())

        early_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            early_time_cutoff_day=1.5,
        )
        early_sites = trace(seed(early_model, jax.random.PRNGKey(9))).get_trace()
        self.assertIn("early_time_log_scale", early_sites)
        self.assertEqual(early_sites["early_time_log_scale"]["value"].shape, (1,))
        self.assertTrue(
            np.all(np.asarray(early_sites["early_time_scale"]["value"]) > 0.0)
        )
        self.assertEqual(
            float(np.asarray(early_sites["early_time_scale"]["value"])[0]),
            1.0,
        )

        free_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            fully_np_trajectory=True,
        )
        free_sites = trace(seed(free_model, jax.random.PRNGKey(6))).get_trace()
        self.assertIn("np_log_rate_nodes", free_sites)
        self.assertNotIn("log_trajectory_scale", free_sites)
        self.assertEqual(free_sites["np_log_rate_nodes"]["value"].shape, (2,))
        self.assertEqual(
            free_sites["node_instantaneous_rate_nm_day"]["value"].shape,
            (2,),
        )
        free_multipliers = np.asarray(
            free_sites["np_node_multipliers"]["value"]
        )
        self.assertEqual(free_multipliers.shape, (2,))
        self.assertTrue(np.all(np.isfinite(free_multipliers)))
        self.assertTrue(np.all(free_multipliers > 0.0))

        constant_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            constant_rate_modulation=True,
            fixed_initial_thickness_nm=0.0,
        )
        constant_sites = trace(
            seed(constant_model, jax.random.PRNGKey(8))
        ).get_trace()
        self.assertIn("log_trajectory_scale", constant_sites)
        self.assertNotIn("np_log_rate_nodes", constant_sites)
        self.assertIn("np_entropy_regularization", constant_sites)
        self.assertNotIn("np_smoothness_regularization", constant_sites)
        self.assertEqual(constant_sites["np_node_raw"]["value"].shape, (1,))

        smoothed_constant_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            constant_rate_modulation=True,
            constant_rate_smoothness_weight=0.01,
            fixed_initial_thickness_nm=0.0,
        )
        smoothed_constant_sites = trace(
            seed(smoothed_constant_model, jax.random.PRNGKey(10))
        ).get_trace()
        self.assertIn("np_smoothness_regularization", smoothed_constant_sites)

        fixed_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            fully_np_trajectory=True,
            fixed_initial_thickness_nm=0.0,
        )
        fixed_sites = trace(seed(fixed_model, jax.random.PRNGKey(7))).get_trace()
        self.assertEqual(
            float(np.asarray(fixed_sites["initial_thickness_nm"]["value"])), 0.0
        )

        post_reset_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_optical_properties=True,
            fixed_initial_thickness_nm=0.0,
            ice_reset_day=0.5,
            fit_post_reset_tile_log_offset=True,
        )
        post_reset_sites = trace(
            seed(post_reset_model, jax.random.PRNGKey(11))
        ).get_trace()
        self.assertEqual(
            post_reset_sites["post_reset_tile_log_offset"]["value"].shape,
            (1, 1),
        )

    def test_shared_ar_parameters_remove_all_tile_level_ar_sites(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((2, 3)),
            "sums": np.zeros((2, 3)),
            "sum_squared": np.ones((2, 3)),
            "days": np.asarray([0.0, 1.0, 2.0]),
            "repeat_index": np.asarray([0, 1, 2]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.asarray([1, 1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(3),
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "metadata": {
                0: {"tile_row": 0, "tile_col": 0},
                1: {"tile_row": 0, "tile_col": 1},
            },
            "sca_index": np.zeros(2, dtype=int),
            "repeat_values": np.asarray([0, 1, 2]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_parameters_across_tiles=True,
        )
        sites = trace(seed(model, jax.random.PRNGKey(41))).get_trace()

        self.assertIn("ar_mu_raw", sites)
        self.assertNotIn("ar_sigma_raw", sites)
        self.assertNotIn("ar_z", sites)
        ar_parameters = np.asarray(sites["ar_parameters"]["value"])
        self.assertEqual(ar_parameters.shape, (2, 9))
        np.testing.assert_array_equal(ar_parameters[0], ar_parameters[1])

        opd_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_parameters_across_tiles=True,
            fit_sca_ar_opd=True,
        )
        opd_sites = trace(seed(opd_model, jax.random.PRNGKey(42))).get_trace()
        self.assertEqual(opd_sites["sca_ar_opd_nm"]["value"].shape, (1,))

        tile_rate_model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=3,
            shared_ar_parameters_across_tiles=True,
            tile_rate_trajectory_variation=True,
            constant_rate_modulation=True,
            fixed_initial_thickness_nm=0.0,
        )
        tile_rate_sites = trace(
            seed(tile_rate_model, jax.random.PRNGKey(42))
        ).get_trace()
        self.assertIn("trajectory_rate_tile_offsets", tile_rate_sites)
        centered_offsets = np.asarray(
            tile_rate_sites["centered_trajectory_rate_tile_offsets"]["value"]
        )
        np.testing.assert_allclose(
            np.mean(centered_offsets, axis=1), 0.0, rtol=0.0, atol=1.0e-12
        )
        tile_node_offsets = np.asarray(
            tile_rate_sites["trajectory_rate_tile_node_offsets"]["value"]
        )
        self.assertEqual(tile_node_offsets.shape, (1, 2, 2))
        tile_ice = np.asarray(tile_rate_sites["ice_thickness"]["value"])
        self.assertEqual(tile_ice.shape, (2, 3))
        self.assertFalse(np.allclose(tile_ice[0], tile_ice[1]))
        tile_rates = np.asarray(
            tile_rate_sites["tile_instantaneous_rate_nm_day"]["value"]
        )
        self.assertFalse(
            np.allclose(tile_rates[0] / tile_rates[1], (tile_rates[0] / tile_rates[1])[0])
        )

    def test_aggregated_tile_moments_preserve_shared_ar_likelihood_gradient(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "counts": np.asarray([[2.0, 3.0], [4.0, 5.0]]),
            "sums": np.asarray([[0.0, 0.03], [0.0, 0.06]]),
            "sum_squared": np.asarray([[0.01, 0.02], [0.02, 0.04]]),
            "sca_index": np.zeros(2, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        aggregated = aggregate_tile_sufficient_statistics(data)
        predictor = build_prediction_function(data, context)
        aggregated_predictor = build_prediction_function(aggregated, context)
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        ice = jnp.asarray([0.0, 50.0])

        full_value, full_gradient = jax.value_and_grad(
            lambda ar: predictor.residual_sum_squares(
                ice, jnp.broadcast_to(ar[None, :], (2, 3)), jnp.asarray(0.0)
            )
        )(jnp.asarray([1.25, 2.0, 0.0]))
        aggregate_value, aggregate_gradient = jax.value_and_grad(
            lambda ar: aggregated_predictor.residual_sum_squares(
                ice, ar[None, :], jnp.asarray(0.0)
            )
        )(jnp.asarray([1.25, 2.0, 0.0]))

        self.assertEqual(aggregated["counts"].shape, (1, 2))
        np.testing.assert_array_equal(aggregated["counts"], [[6.0, 8.0]])
        np.testing.assert_allclose(
            aggregate_value, full_value, rtol=2.0e-7, atol=1.0e-10
        )
        np.testing.assert_allclose(
            aggregate_gradient, full_gradient, rtol=2.0e-7, atol=1.0e-10
        )

        shared_ar_predictor = build_prediction_function(
            data, context, share_ar_across_rows=True
        )
        reference_prediction = build_prediction_function(data, context)
        shared_ar_value, shared_ar_gradient = jax.value_and_grad(
            lambda ar: shared_ar_predictor(
                ice,
                jnp.broadcast_to(ar[None, :], (2, 3)),
                jnp.asarray(0.0),
            ).sum()
        )(jnp.asarray([1.25, 2.0, 0.0]))
        reference_value, reference_gradient = jax.value_and_grad(
            lambda ar: reference_prediction(
                ice,
                jnp.broadcast_to(ar[None, :], (2, 3)),
                jnp.asarray(0.0),
            ).sum()
        )(jnp.asarray([1.25, 2.0, 0.0]))
        np.testing.assert_allclose(
            shared_ar_value, reference_value, rtol=2.0e-7, atol=1.0e-10
        )
        np.testing.assert_allclose(
            shared_ar_gradient, reference_gradient, rtol=2.0e-7, atol=1.0e-10
        )

    def test_model_wires_qe_rbf_flag_to_prediction(self):
        data = {
            "counts": np.ones((1, 2)),
            "sums": np.zeros((1, 2)),
            "sum_squared": np.ones((1, 2)),
            "days": np.asarray([0.0, 1.0]),
            "repeat_index": np.asarray([0, 1]),
            "node_days": np.asarray([0.0, 1.0]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
            "n_scas": 1,
            "n_tiles": 1,
            "sca_index": np.zeros(1, dtype=int),
            "repeat_values": np.asarray([0, 1]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        with patch("fit_wfi_tmm_numpyro.build_prediction_function") as builder:
            build_model(
                data,
                context,
                log_noise_sigma=0.005,
                likelihood_weight=1.0e-8,
                ar_layers=3,
                fit_qe_rbf=True,
            )

        self.assertTrue(builder.call_args.kwargs["fit_qe_rbf"])

    def test_reduced_map_does_not_copy_tile_offsets_to_full_fit(self):
        full_initial = {
            "ar_mu_raw": np.zeros(3),
            "ar_sca_z": np.zeros((2, 3)),
            "ar_tile_z": np.zeros((2, 4, 3)),
        }
        reduced_map = {
            "ar_mu_raw": np.ones(3),
            "ar_sca_z": np.ones((2, 3)),
            "ar_tile_z": np.ones((2, 1, 3)),
        }
        transferred = transfer_map_initial_values(reduced_map, full_initial)

        np.testing.assert_array_equal(transferred["ar_mu_raw"], np.ones(3))
        np.testing.assert_array_equal(transferred["ar_sca_z"], np.ones((2, 3)))
        np.testing.assert_array_equal(
            transferred["ar_tile_z"], full_initial["ar_tile_z"]
        )

    def test_response_context_keeps_one_qe_integration_per_sca(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        spectra = {
            band: {
                "wavelength": wavelengths,
                "raw_output": np.ones_like(wavelengths),
            }
            for band in range(1, 7)
        }
        qe_values = np.column_stack(
            (np.ones(len(wavelengths)), np.asarray([1.0, 2.0, 4.0]))
        )
        with patch("fit_wfi_tmm_numpyro.load_sed", return_value=spectra), patch(
            "fit_wfi_tmm_numpyro.load_qe",
            return_value=(wavelengths, qe_values, ["SCU 1", "SCU 2"]),
        ):
            context = load_response_context(
                Path("model.ecsv"), "signed", 1, ["WFI01", "WFI02"]
            )
            fine_context = load_response_context(
                Path("model.ecsv"), "signed", 1, ["WFI01", "WFI02"], 0.5
            )

        self.assertEqual(context["model_weights"].shape, (2, 6, 3))
        self.assertEqual(fine_context["model_weights"].shape, (2, 6, 401))
        self.assertEqual(context["outside_output"].shape, (2, 6))
        self.assertEqual(context["clean_output"].shape, (2, 6))
        np.testing.assert_array_equal(context["qe_columns"], [0, 1])
        self.assertFalse(
            np.allclose(context["model_weights"][0], context["model_weights"][1])
        )

    def test_response_integration_supports_energy_sensitivity_weighting(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        spectra = {
            band: {
                "wavelength": wavelengths,
                "raw_output": np.ones_like(wavelengths),
            }
            for band in range(1, 7)
        }
        qe = np.ones((len(wavelengths), 1))
        photon = prepare_response_integration(
            wavelengths, spectra, wavelengths, qe, "signed", sed_weighting="photon"
        )
        energy = prepare_response_integration(
            wavelengths, spectra, wavelengths, qe, "signed", sed_weighting="energy"
        )

        self.assertEqual(photon[2].shape, (6,))
        self.assertGreater(photon[2][0], energy[2][0])
        self.assertNotEqual(photon[0][0, 0], energy[0][0, 0])
        with self.assertRaises(ValueError):
            prepare_response_integration(
                wavelengths,
                spectra,
                wavelengths,
                qe,
                "signed",
                sed_weighting="invalid",
            )

    def test_prediction_uses_the_matching_sca_qe_context(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        spectra = {
            band: {
                "wavelength": wavelengths,
                "raw_output": np.ones_like(wavelengths),
            }
            for band in range(1, 7)
        }
        qe_values = np.column_stack(
            (np.ones(len(wavelengths)), np.asarray([1.0, 2.0, 4.0]))
        )
        with patch("fit_wfi_tmm_numpyro.load_sed", return_value=spectra), patch(
            "fit_wfi_tmm_numpyro.load_qe",
            return_value=(wavelengths, qe_values, ["SCU 1", "SCU 2"]),
        ):
            context = load_response_context(
                Path("model.ecsv"), "signed", 1, ["WFI01", "WFI02"]
            )

        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "n_scas": 2,
            "sca_index": np.asarray([0, 1]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        predictor = build_prediction_function(data, context)
        predictions = np.asarray(
            predictor(
                jnp.asarray([10.0, 20.0]),
                jnp.asarray([[2.5, 2.0, 0.2], [2.5, 2.0, 0.2]]),
                jnp.asarray(0.0),
            )
        )

        self.assertEqual(predictions.shape, (2, 2))
        self.assertFalse(np.allclose(predictions[0], predictions[1]))
        tile_predictions = np.asarray(
            predictor(
                jnp.asarray([[10.0, 20.0], [11.0, 21.0]]),
                jnp.asarray([[2.5, 2.0, 0.2], [2.5, 2.0, 0.2]]),
                jnp.asarray(0.0),
            )
        )
        self.assertEqual(tile_predictions.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(tile_predictions)))
        subset_predictor = build_prediction_function(
            data, context, row_indices=np.asarray([1])
        )
        subset_prediction = np.asarray(
            subset_predictor(
                jnp.asarray([10.0, 20.0]),
                jnp.asarray([[2.5, 2.0, 0.2]]),
                jnp.asarray(0.0),
            )
        )
        np.testing.assert_allclose(subset_prediction, predictions[1:2])
        for layer_count in (2, 3):
            layered_parameters = np.tile(
                np.asarray([2.5, 2.0, 0.2]), (2, layer_count)
            )
            layered_predictions = np.asarray(
                predictor(
                    jnp.asarray([10.0, 20.0]),
                    jnp.asarray(layered_parameters),
                    jnp.asarray(0.0),
                )
            )
            self.assertEqual(layered_predictions.shape, (2, 2))
            self.assertTrue(np.all(np.isfinite(layered_predictions)))

    def test_prediction_uses_each_leds_baseline_group(self):
        wavelengths = np.asarray([500.0, 600.0, 700.0])
        context = {
            "wavelengths": wavelengths,
            "model_weights": np.full((6, len(wavelengths)), 1.0 / len(wavelengths)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2, dtype=int),
            "step_design": np.zeros(12),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context)
        predictions = np.asarray(
            predictor(
                jnp.arange(12.0) + 10.0,
                jnp.asarray([[2.5, 2.0, 0.2]]),
                jnp.asarray(0.0),
            )
        )

        np.testing.assert_allclose(
            predictions[0, data["first_by_led"]], 0.0, atol=1.0e-12
        )

    def test_prediction_applies_led_dependent_acm_steps(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2, dtype=int),
            "step_design": np.tile([0.0, 1.0], 6),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context)
        acm_steps = np.log1p(np.arange(1, 7, dtype=float) * 0.01)
        predictions = np.asarray(
            predictor(
                jnp.zeros(12),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(acm_steps),
            )
        )

        expected = np.column_stack((np.zeros(6), acm_steps)).reshape(-1)
        np.testing.assert_allclose(predictions[0], expected)

    def test_prediction_applies_two_common_acm_steps(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2, dtype=int),
            "step_design": np.tile([0.0, 1.0], 6),
            "late_step_design": np.tile([0.0, 1.0], 6),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(
            data, context, fit_late_acm_step=True
        )
        base_step = jnp.asarray(np.log(0.96))
        late_step = jnp.asarray(np.log(0.992))
        predictions = np.asarray(
            predictor(
                jnp.zeros(12),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                base_step,
                late_acm_step=late_step,
            )
        )

        expected = np.tile([0.0, np.log(0.96) + np.log(0.992)], 6)
        np.testing.assert_allclose(predictions[0], expected)

    def test_prediction_applies_relative_count_scale_in_log_space(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2, dtype=int),
            "step_design": np.zeros(12),
            "repeat_index": np.tile([0, 1], 6),
            "repeat_values": np.asarray([0, 1]),
            "node_days": np.asarray([0.0, 1.0]),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(
            data, context, early_time_cutoff_day=1.5
        )
        predictions = np.asarray(
            predictor(
                jnp.zeros(12),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                early_time_scale=jnp.asarray([1.0, 2.0]),
            )
        )

        np.testing.assert_allclose(
            predictions[0], np.tile([0.0, -np.log(2.0)], 6)
        )

    def test_single_chunk_likelihood_matches_scanned_value_and_gradient(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        observed = np.tile([0.0, 0.01], 6)
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(4, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2),
            "step_design": np.zeros(12),
            "counts": np.full((4, 12), 3.0),
            "sums": np.broadcast_to(3.0 * observed, (4, 12)),
            "sum_squared": np.broadcast_to(
                3.0 * (observed**2 + 0.0001), (4, 12)
            ),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        ice_thickness = jnp.asarray(np.tile([0.0, 50.0], 6))
        ar_parameters = jnp.asarray(
            [[1.25 + index * 0.01, 2.0, 0.0] for index in range(4)]
        )

        def evaluate(chunk_size):
            prediction_data = dict(data, prediction_row_chunk_size=chunk_size)
            predictor = build_prediction_function(prediction_data, context)
            return jax.jit(
                jax.value_and_grad(
                    lambda ar: predictor.residual_sum_squares(
                        ice_thickness, ar, jnp.asarray(0.0)
                    )
                )
            )(ar_parameters)

        direct_value, direct_gradient = evaluate(4)
        scanned_value, scanned_gradient = evaluate(2)
        np.testing.assert_allclose(direct_value, scanned_value, rtol=1.0e-12)
        np.testing.assert_allclose(
            direct_gradient, scanned_gradient, rtol=1.0e-10, atol=1.0e-12
        )

    def test_single_sca_shared_integration_inputs_match_rowwise_inputs(self):
        wavelengths = np.asarray([900.0, 1000.0, 1100.0])
        model_weights = np.asarray(
            [[0.8, 1.0, 0.7], [0.9, 1.1, 0.6]] * 3,
            dtype=float,
        )
        context_single = {
            "wavelengths": wavelengths,
            "model_weights": model_weights,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        context_two_sca = {
            "wavelengths": wavelengths,
            "model_weights": np.stack((model_weights, 1.1 * model_weights)),
            "outside_output": np.zeros((2, 6)),
            "clean_output": np.ones((2, 6)),
        }
        common_data = {
            "sca_index": np.zeros(4, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2, dtype=int),
            "step_design": np.zeros(12),
            "counts": np.full((4, 12), 3.0),
            "sums": np.zeros((4, 12)),
            "sum_squared": np.full((4, 12), 0.0001),
        }
        data_single = dict(common_data, n_scas=1)
        data_two_sca = dict(common_data, n_scas=2)
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        thickness = jnp.asarray(np.linspace(0.0, 50.0, 12))
        ar_parameters = jnp.asarray(
            [[1.25 + index * 0.01, 2.0, 0.0] for index in range(4)]
        )

        single = build_prediction_function(data_single, context_single)
        rowwise = build_prediction_function(data_two_sca, context_two_sca)

        def evaluate(predictor):
            return jax.jit(
                jax.value_and_grad(
                    lambda ar: predictor.residual_sum_squares(
                        thickness, ar, jnp.asarray(0.0)
                    )
                )
            )(ar_parameters)

        single_value, single_gradient = evaluate(single)
        rowwise_value, rowwise_gradient = evaluate(rowwise)
        np.testing.assert_allclose(single_value, rowwise_value, rtol=1.0e-12)
        np.testing.assert_allclose(
            single_gradient, rowwise_gradient, rtol=1.0e-10, atol=1.0e-12
        )

    def test_prediction_matches_scalar_coherent_stack(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context)
        predictions = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
            )
        )

        substrate_index = 3.6 + 0.7j
        clean = scalar_coherent_transmission(
            1000.0,
            [(2.0 + 0.0j, 125.0)],
            substrate_index,
        )
        coated = scalar_coherent_transmission(
            1000.0,
            [(1.25 + 0.0j, 50.0), (2.0 + 0.0j, 125.0)],
            substrate_index,
        )
        expected = np.asarray([[0.0, np.log(coated / clean)]])
        np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)

    def test_incidence_cone_preserves_normal_limit_and_returns_finite_response(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        parameters = jnp.asarray([[1.25, 2.0, 0.05]])
        thickness = jnp.asarray([0.0, 50.0])
        normal_predictions = np.asarray(
            build_prediction_function(data, context)(
                thickness, parameters, jnp.asarray(0.0)
            )
        )
        zero_cone_predictions = np.asarray(
            build_prediction_function(
                data, context, incidence_cone_half_angle_deg=0.0
            )(thickness, parameters, jnp.asarray(0.0))
        )
        tiny_cone_predictions = np.asarray(
            build_prediction_function(
                data, context, incidence_cone_half_angle_deg=1.0e-6
            )(thickness, parameters, jnp.asarray(0.0))
        )
        cone_predictions = np.asarray(
            build_prediction_function(
                data, context, incidence_cone_half_angle_deg=20.0
            )(thickness, parameters, jnp.asarray(0.0))
        )

        np.testing.assert_allclose(zero_cone_predictions, normal_predictions)
        np.testing.assert_allclose(
            tiny_cone_predictions, normal_predictions, rtol=1.0e-10, atol=1.0e-12
        )
        self.assertTrue(np.all(np.isfinite(cone_predictions)))
        self.assertFalse(np.allclose(cone_predictions, normal_predictions))

        fitted_predictor = build_prediction_function(
            data, context, fit_incidence_angle=True
        )
        fitted_normal = np.asarray(
            fitted_predictor(
                thickness,
                parameters,
                jnp.asarray(0.0),
                incidence_angle_deg=jnp.asarray(0.0),
            )
        )
        fitted_tilted = np.asarray(
            fitted_predictor(
                thickness,
                parameters,
                jnp.asarray(0.0),
                incidence_angle_deg=jnp.asarray(10.0),
            )
        )
        self.assertTrue(np.all(np.isfinite(fitted_tilted)))
        np.testing.assert_allclose(fitted_normal, normal_predictions)
        self.assertFalse(np.allclose(fitted_tilted, fitted_normal))

        def fitted_sum(angle):
            return jnp.sum(
                fitted_predictor(
                    thickness,
                    parameters,
                    jnp.asarray(0.0),
                    incidence_angle_deg=angle,
                )
            )

        zero_gradient = jax.grad(fitted_sum)(jnp.asarray(0.0))
        self.assertTrue(np.isfinite(float(zero_gradient)))
        self.assertAlmostEqual(float(zero_gradient), 0.0, places=12)

    def test_prediction_uses_row_specific_chief_ray_geometry(self):
        context = {
            "wavelengths": np.asarray([900.0, 1300.0]),
            "model_weights": np.ones((6, 2)) / 2.0,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(2, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        _, jnp, _, _, _, _, _ = require_numpyro()
        predictor = build_prediction_function(
            data,
            context,
            incidence_pupil_half_angle_deg=3.576334,
            incidence_chief_ray_tilts_deg=np.asarray([[0.0, 0.0], [10.0, 0.0]]),
        )
        prediction = np.asarray(
            predictor(
                jnp.asarray([0.0, 100.0]),
                jnp.asarray([[1.0, 1.5, 0.0], [1.0, 1.5, 0.0]]),
                jnp.asarray(0.0),
            )
        )

        self.assertEqual(prediction.shape, (2, 2))
        self.assertFalse(np.allclose(prediction[0], prediction[1]))

    def test_prediction_matches_scalar_multilayer_coherent_stack(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context)
        ar_parameters = np.asarray(
            [[
                1.25,
                2.0,
                0.05,
                1.70,
                1.8,
                0.02,
                1.95,
                2.3,
                0.01,
            ]]
        )
        predictions = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray(ar_parameters),
                jnp.asarray(0.0),
            )
        )

        substrate_index = 3.6 + 0.7j
        ar_layers = [
            (2.0 + 0.05j, 125.0),
            (1.8 + 0.02j, 170.0),
            (2.3 + 0.01j, 195.0),
        ]
        clean = scalar_coherent_transmission(1000.0, ar_layers, substrate_index)
        coated = scalar_coherent_transmission(
            1000.0,
            [(1.25 + 0.0j, 50.0)] + ar_layers,
            substrate_index,
        )
        expected = np.asarray([[0.0, np.log(coated / clean)]])
        np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)

    def test_prediction_applies_sca_ar_opd_to_first_layer_phase(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context)
        ar_parameters = jnp.asarray([[1.25, 2.0, 0.0]])
        thickness = jnp.asarray([0.0, 50.0])
        baseline = np.asarray(predictor(thickness, ar_parameters, jnp.asarray(0.0)))
        zero_offset = np.asarray(
            predictor(
                thickness,
                ar_parameters,
                jnp.asarray(0.0),
                sca_ar_opd_nm=jnp.asarray([0.0]),
            )
        )
        shifted = np.asarray(
            predictor(
                thickness,
                ar_parameters,
                jnp.asarray(0.0),
                sca_ar_opd_nm=jnp.asarray([20.0]),
            )
        )

        np.testing.assert_allclose(zero_offset, baseline, rtol=0.0, atol=0.0)
        substrate_index = 3.6 + 0.7j
        clean = scalar_coherent_transmission(
            1000.0,
            [(2.0 + 0.0j, 125.0 + 10.0)],
            substrate_index,
        )
        coated = scalar_coherent_transmission(
            1000.0,
            [(1.25 + 0.0j, 50.0), (2.0 + 0.0j, 125.0 + 10.0)],
            substrate_index,
        )
        expected = np.asarray([[0.0, np.log(coated / clean)]])
        np.testing.assert_allclose(shifted, expected, rtol=1e-10, atol=1e-10)

    def test_prediction_broadcasts_sca_ar_opd_by_row_sca(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((2, 6, 1)),
            "outside_output": np.zeros((2, 6)),
            "clean_output": np.ones((2, 6)),
        }
        data = {
            "n_scas": 2,
            "sca_index": np.asarray([0, 1]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context)
        ar_parameters = jnp.asarray([[1.25, 2.0, 0.0], [1.25, 2.0, 0.0]])
        thickness = jnp.asarray([0.0, 50.0])
        baseline = np.asarray(predictor(thickness, ar_parameters, jnp.asarray(0.0)))
        shifted = np.asarray(
            predictor(
                thickness,
                ar_parameters,
                jnp.asarray(0.0),
                sca_ar_opd_nm=jnp.asarray([0.0, 20.0]),
            )
        )

        np.testing.assert_allclose(shifted[0], baseline[0], rtol=0.0, atol=0.0)
        self.assertFalse(np.allclose(shifted[1], baseline[1]))

        cone_predictor = build_prediction_function(
            data,
            context,
            incidence_cone_half_angle_deg=20.0,
            share_polarization_work=True,
        )
        cone_shifted = np.asarray(
            cone_predictor(
                thickness,
                ar_parameters,
                jnp.asarray(0.0),
                sca_ar_opd_nm=jnp.asarray([0.0, 20.0]),
            )
        )
        self.assertTrue(np.all(np.isfinite(cone_shifted)))
        self.assertFalse(np.allclose(cone_shifted[1], shifted[1]))

    def test_model_samples_optional_sca_ar_opd_per_sca(self):
        jax, _, _, _, _, _, _ = require_numpyro()
        from numpyro.handlers import seed, trace

        data = {
            "counts": np.ones((2, 2)),
            "sums": np.zeros((2, 2)),
            "sum_squared": np.ones((2, 2)),
            "days": np.asarray([0.0, 1.0]),
            "repeat_index": np.asarray([0, 1]),
            "node_days": np.asarray([0.0, 1.0]),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
            "n_scas": 2,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0, 1]),
            "repeat_values": np.asarray([0, 1]),
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((2, 6, 1)),
            "outside_output": np.zeros((2, 6)),
            "clean_output": np.ones((2, 6)),
        }
        model = build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            shared_ar_optical_properties=True,
            fit_sca_ar_opd=True,
            sca_ar_opd_prior_sigma_nm=2.0,
        )
        sites = trace(seed(model, jax.random.PRNGKey(91))).get_trace()
        self.assertEqual(sites["sca_ar_opd_nm"]["value"].shape, (2,))

    def test_prediction_accepts_fitted_hgcdte_parameters(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context, fit_hgcdte=True)
        ar_parameters = jnp.asarray([[1.25, 2.0, 0.0]])
        predictions = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                ar_parameters,
                jnp.asarray(0.0),
                jnp.asarray([4.0, 0.2]),
            )
        )

        substrate_index = 4.0 + 0.2j
        clean = scalar_coherent_transmission(
            1000.0,
            [(2.0 + 0.0j, 125.0)],
            substrate_index,
        )
        coated = scalar_coherent_transmission(
            1000.0,
            [(1.25 + 0.0j, 50.0), (2.0 + 0.0j, 125.0)],
            substrate_index,
        )
        expected = np.asarray([[0.0, np.log(coated / clean)]])
        np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)

    def test_prediction_accepts_wavelength_dependent_conductive_layer(self):
        context = {
            "wavelengths": np.asarray([1000.0, 1600.0, 2200.0]),
            "model_weights": np.ones((6, 3)) / 3.0,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(
            data, context, fit_conductive_layer=True
        )
        predictions = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                None,
                jnp.asarray([np.log(5.0), np.log(1.0), np.log(1.0)]),
            )
        )

        wavelengths = context["wavelengths"]
        omega_ratio = CONDUCTIVE_REFERENCE_WAVELENGTH_NM / wavelengths
        conductive_index = np.sqrt(
            CONDUCTIVE_EPSILON_INF
            - 1.0 / (omega_ratio**2 + 1j * omega_ratio)
        )
        substrate_index = 3.6 + 0.7j
        clean = np.asarray(
            [
                scalar_coherent_transmission(
                    wavelength,
                    [(2.0 + 0.0j, 125.0), (index, 5.0)],
                    substrate_index,
                )
                for wavelength, index in zip(wavelengths, conductive_index)
            ]
        )
        coated = np.asarray(
            [
                scalar_coherent_transmission(
                    wavelength,
                    [
                        (1.25 + 0.0j, 50.0),
                        (2.0 + 0.0j, 125.0),
                        (index, 5.0),
                    ],
                    substrate_index,
                )
                for wavelength, index in zip(wavelengths, conductive_index)
            ]
        )
        expected = np.asarray([[0.0, np.log(np.mean(coated / clean))]])
        np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)

    def test_prediction_accepts_wavelength_dependent_asw_constants(self):
        context = {
            "wavelengths": np.asarray([1200.0, 1600.0, 2200.0]),
            "ice_n": np.asarray([1.29, 1.29, 1.30]),
            "ice_k": np.asarray([0.0, 0.0005, 0.001]),
            "model_weights": np.ones((6, 3)) / 3.0,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(
            data, context, use_asw_optical_constants=True
        )
        predictions = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
            )
        )

        substrate_index = 3.6 + 0.7j
        clean = np.asarray(
            [
                scalar_coherent_transmission(
                    wavelength,
                    [(2.0 + 0.0j, 125.0)],
                    substrate_index,
                )
                for wavelength in context["wavelengths"]
            ]
        )
        coated = np.asarray(
            [
                scalar_coherent_transmission(
                    wavelength,
                    [
                        (index, 50.0),
                        (2.0 + 0.0j, 125.0),
                    ],
                    substrate_index,
                )
                for wavelength, index in zip(
                    context["wavelengths"], context["ice_n"] + 1j * context["ice_k"]
                )
            ]
        )
        expected = np.asarray([[0.0, np.log(np.mean(coated / clean))]])
        np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)

        cone_predictions = np.asarray(
            build_prediction_function(
                data,
                context,
                use_asw_optical_constants=True,
                incidence_cone_half_angle_deg=3.0,
            )(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
            )
        )
        self.assertTrue(np.all(np.isfinite(cone_predictions)))

    def test_prediction_accepts_wavelength_dependent_hgcdte_constants(self):
        wavelengths = np.asarray([1000.0, 1600.0, 2200.0])
        hgcdte_n, hgcdte_k = hgcdte_reference_optical_constants(wavelengths)
        context = {
            "wavelengths": wavelengths,
            "hgcdte_n": hgcdte_n,
            "hgcdte_k": hgcdte_k,
            "model_weights": np.ones((6, 3)) / 3.0,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(
            data, context, use_hgcdte_optical_constants=True
        )
        predictions = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
            )
        )

        substrate_indices = hgcdte_n + 1j * hgcdte_k
        clean = np.asarray(
            [
                scalar_coherent_transmission(
                    wavelength,
                    [(2.0 + 0.0j, 125.0)],
                    substrate_index,
                )
                for wavelength, substrate_index in zip(wavelengths, substrate_indices)
            ]
        )
        coated = np.asarray(
            [
                scalar_coherent_transmission(
                    wavelength,
                    [(1.25 + 0.0j, 50.0), (2.0 + 0.0j, 125.0)],
                    substrate_index,
                )
                for wavelength, substrate_index in zip(wavelengths, substrate_indices)
            ]
        )
        expected = np.asarray([[0.0, np.log(np.mean(coated / clean))]])
        np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)

    def test_zero_hgcdte_modulation_matches_reference_constants(self):
        wavelengths = np.asarray([1000.0, 1600.0, 2200.0])
        hgcdte_n, hgcdte_k = hgcdte_reference_optical_constants(wavelengths)
        context = {
            "wavelengths": wavelengths,
            "hgcdte_n": hgcdte_n,
            "hgcdte_k": hgcdte_k,
            "hgcdte_modulation_basis": hgcdte_modulation_basis(wavelengths),
            "model_weights": np.ones((6, 3)) / 3.0,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(
            data,
            context,
            fit_hgcdte_modulation=True,
            use_hgcdte_optical_constants=True,
        )
        zero_modulation = jnp.zeros(2 * context["hgcdte_modulation_basis"].shape[1])
        modulated = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                zero_modulation,
            )
        )
        reference_predictor = build_prediction_function(
            data, context, use_hgcdte_optical_constants=True
        )
        reference = np.asarray(
            reference_predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
            )
        )

        np.testing.assert_allclose(modulated, reference, rtol=1e-10, atol=1e-10)

        changed = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                jnp.concatenate(
                    (jnp.full(zero_modulation.shape[0] // 2, 0.5),
                     jnp.full(zero_modulation.shape[0] // 2, -0.5))
                ),
            )
        )
        self.assertFalse(np.allclose(changed, reference))

    def test_zero_ice_modulation_matches_asw_reference_constants(self):
        wavelengths = np.asarray([1000.0, 1600.0, 2200.0])
        ice_n = np.asarray([1.30, 1.28, 1.26])
        ice_k = np.asarray([0.01, 0.02, 0.03])
        context = {
            "wavelengths": wavelengths,
            "ice_n": ice_n,
            "ice_k": ice_k,
            "ice_modulation_basis": ice_modulation_basis(wavelengths),
            "model_weights": np.ones((6, 3)) / 3.0,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(
            data,
            context,
            fit_ice_optical_modulation=True,
            use_asw_optical_constants=True,
        )
        zero_modulation = jnp.zeros(2 * context["ice_modulation_basis"].shape[1])
        modulated = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                ice_modulation_parameters=zero_modulation,
            )
        )
        reference_predictor = build_prediction_function(
            data, context, use_asw_optical_constants=True
        )
        reference = np.asarray(
            reference_predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
            )
        )
        np.testing.assert_allclose(modulated, reference, rtol=1e-10, atol=1e-10)

        changed = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                ice_modulation_parameters=jnp.concatenate(
                    (
                        jnp.full(zero_modulation.shape[0] // 2, 0.1),
                        jnp.full(zero_modulation.shape[0] // 2, -0.1),
                    )
                ),
            )
        )
        self.assertFalse(np.allclose(changed, reference))

    def test_constant_and_wavelength_dependent_hgcdte_modes_are_exclusive(self):
        context = {
            "wavelengths": np.asarray([1000.0]),
            "hgcdte_n": np.asarray([3.0]),
            "hgcdte_k": np.asarray([0.1]),
            "model_weights": np.ones((6, 1)),
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        with self.assertRaises(ValueError):
            build_prediction_function(
                data,
                context,
                fit_hgcdte=True,
                use_hgcdte_optical_constants=True,
            )

    def test_prediction_accepts_rbf_qe_modulation(self):
        context = {
            "wavelengths": np.asarray([650.0, 950.0, 1250.0, 1600.0]),
            "model_weights": np.ones((6, 4)) / 4.0,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.asarray([1, 1]),
            "first_by_led": np.zeros(6, dtype=int),
            "step_design": np.zeros(2),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context, fit_qe_rbf=True)
        baseline = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                None,
                None,
                jnp.zeros(4),
            )
        )
        predictions = np.asarray(
            predictor(
                jnp.asarray([0.0, 50.0]),
                jnp.asarray([[1.25, 2.0, 0.0]]),
                jnp.asarray(0.0),
                None,
                None,
                jnp.asarray([0.1, -0.1, 0.05, 0.0]),
            )
        )

        self.assertEqual(predictions.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertNotEqual(float(predictions[0, 1]), 0.0)
        self.assertFalse(np.allclose(predictions, baseline))

    def test_prediction_recovers_injected_rbf_qe_shape(self):
        from scipy.optimize import minimize

        wavelengths = np.asarray([500.0, 650.0, 900.0, 1200.0, 1500.0, 2000.0, 2600.0])
        led_centers = np.asarray([550.0, 700.0, 850.0, 1100.0, 1500.0, 2200.0])
        model_weights = np.exp(
            -0.5 * ((wavelengths[None, :] - led_centers[:, None]) / 350.0) ** 2
        )
        model_weights /= np.sum(model_weights, axis=1, keepdims=True)
        context = {
            "wavelengths": wavelengths,
            "model_weights": model_weights,
            "outside_output": np.zeros(6),
            "clean_output": np.ones(6),
        }
        data = {
            "n_scas": 1,
            "sca_index": np.zeros(1, dtype=int),
            "led": np.repeat(np.arange(1, 7), 2),
            "first_by_led": np.arange(0, 12, 2),
            "step_design": np.zeros(12),
        }
        jax, jnp, _, _, _, _, _ = require_numpyro()
        jax.config.update("jax_enable_x64", True)
        predictor = build_prediction_function(data, context, fit_qe_rbf=True)
        ice_thickness = jnp.asarray(np.tile([20.0, 26.0], 6))
        ar_parameters = jnp.asarray([[1.25, 2.0, 0.0]])
        injected = jnp.asarray([0.0, -0.08, 0.06, 0.12])
        target = predictor(
            ice_thickness,
            ar_parameters,
            jnp.asarray(0.0),
            None,
            None,
            injected,
        )

        def objective(free_values):
            qe_values = jnp.concatenate((jnp.zeros(1), free_values))
            prediction = predictor(
                ice_thickness,
                ar_parameters,
                jnp.asarray(0.0),
                None,
                None,
                qe_values,
            )
            return jnp.sum((prediction - target) ** 2)

        value_and_gradient = jax.jit(jax.value_and_grad(objective))

        def scipy_objective(values):
            value, gradient = value_and_gradient(jnp.asarray(values))
            return float(value), np.asarray(gradient, dtype=float)

        result = minimize(
            scipy_objective,
            np.zeros(3),
            jac=True,
            method="BFGS",
            options={"gtol": 1.0e-12, "maxiter": 100},
        )

        self.assertLess(result.fun, 1.0e-18)


if __name__ == "__main__":
    unittest.main()
