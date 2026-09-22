import unittest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fit_wfi_tmm_numpyro import (
    DEFAULT_SINGLE_SCA_TILE_STATISTICS,
    REFERENCE_THICKNESS_NM,
    REFERENCE_TIME_DAYS,
    RESET_DAY,
    build_model,
    build_prediction_function,
    default_likelihood_weight,
    dielectric_ar_reference,
    integrate_rate_trajectory,
    initial_values_for_data,
    load_response_context,
    load_tile_statistics,
    ar_prior_arrays,
    load_asw_optical_constants,
    qe_columns_for_scas,
    require_numpyro,
    qe_rbf_configuration,
    run_map_optimization,
    statistics_fingerprint,
    tile_neighbor_edges,
    trajectory_rate_per_scale,
    transfer_map_initial_values,
)
from plot_wfi_tmm_numpyro import (
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


class TestTmmTileData(unittest.TestCase):
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

    def test_qe_rbf_configuration_validates_custom_basis(self):
        nodes, width, prior = qe_rbf_configuration(
            [600.0, 800.0, 1000.0], 120.0, 0.5
        )
        np.testing.assert_array_equal(nodes, [600.0, 800.0, 1000.0])
        self.assertEqual(width, 120.0)
        self.assertEqual(prior, 0.5)
        with self.assertRaises(ValueError):
            qe_rbf_configuration([800.0, 600.0], 120.0, 0.5)

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

    def test_repeat_nodes_are_shared_and_time_ordered(self):
        data = load_tile_statistics(DEFAULT_SINGLE_SCA_TILE_STATISTICS, max_tiles=2)

        np.testing.assert_array_equal(data["repeat_values"], np.arange(19))
        np.testing.assert_array_equal(
            data["repeat_index"][:19], np.arange(19)
        )
        self.assertEqual(len(data["node_days"]), 19)
        self.assertTrue(np.all(np.diff(data["node_days"]) > 0.0))
        self.assertEqual(data["counts"].shape, (2, 114))
        np.testing.assert_array_equal(data["sca_index"], np.zeros(2, dtype=int))

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
        self.assertEqual(data["counts"].shape, (4, 114))
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
        self.assertAlmostEqual(float(map_values["value"]), 2.97029703, places=6)

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

        np.testing.assert_allclose(predictions[0, data["first_by_led"]], 0.0)

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

        self.assertEqual(predictions.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertNotEqual(float(predictions[0, 1]), 0.0)

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

        self.assertEqual(predictions.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertNotEqual(float(predictions[0, 1]), 0.0)

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
