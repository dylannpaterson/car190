import tempfile
import unittest
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


def _prediction_fixture():
    wavelengths = np.asarray([900.0, 1100.0, 1300.0])
    data = {
        "counts": np.ones((1, 12)),
        "sums": np.zeros((1, 12)),
        "sum_squared": np.ones((1, 12)),
        "days": np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 5.0, 5.1]),
        "node_days": np.asarray([0.0, 1.0, 2.0]),
        "led": np.repeat(np.arange(1, 7), 2),
        "first_by_led": np.arange(0, 12, 2),
        "sca_index": np.asarray([0]),
    }
    led_weights = np.asarray(
        [
            [1.0, 0.8, 1.2],
            [0.9, 1.1, 0.7],
            [1.2, 0.9, 1.0],
            [0.8, 1.2, 0.9],
            [1.1, 0.7, 1.3],
            [0.95, 1.05, 0.85],
        ]
    )
    context = {
        "wavelengths": wavelengths,
        "model_weights": np.repeat(led_weights, 2, axis=0)[None, :, :],
        "hgcdte_n": np.asarray([3.6, 3.5, 3.4]),
        "hgcdte_k": np.asarray([0.7, 0.65, 0.6]),
        "absolute_substrate_power": True,
    }
    reference, _, _ = production.ar_prior_arrays()
    ar_parameters = reference[None, :]
    ice_thickness = np.asarray(
        [[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7]]
    )
    hgcdte_parameters = np.zeros((1, 2))
    return data, context, ar_parameters, ice_thickness, hgcdte_parameters


def _two_reset_trajectory_data():
    return {
        "days": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "node_days": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "reset_days": np.asarray([2.0, 5.0]),
        "reset_day": 2.0,
    }


class FixedProductionSettingsTests(unittest.TestCase):
    def test_settings_are_fixed_to_the_production_contract(self):
        config = production.ProductionConfig(Path("stats.csv"), Path("output"))

        self.assertEqual(config.tile_statistics, Path("stats.csv"))
        self.assertEqual(config.output_dir, Path("output"))
        self.assertIsNone(config.map_init_from)
        self.assertEqual(config.trajectory_rate_prior_sigma_log, 1.5)
        self.assertEqual(config.initial_thickness_prior, "lognormal")
        self.assertEqual(config.initial_thickness_prior_sigma_nm, 5.0)
        self.assertEqual(config.absolute_hgcdte_n_scale_prior_sigma_log, 0.25)
        self.assertIsNone(config.fixed_absolute_hgcdte_n_log_scale)
        self.assertTrue(config.fit_incidence_angle)
        self.assertEqual(config.incidence_pupil_half_angle_deg, 0.0)
        self.assertEqual(production.AR_LAYERS, 3)
        self.assertEqual(production.ICE_INDEX_DEFAULT, 1.25)
        self.assertEqual(production.INITIAL_THICKNESS_LOG_MEDIAN_NM, 10.0)
        self.assertEqual(production.INITIAL_THICKNESS_LOG_SIGMA, 1.5)
        self.assertEqual(production.TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY, 5.0)
        self.assertEqual(production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG, 1.5)
        self.assertEqual(production.ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG, 0.25)
        self.assertEqual(production.PRODUCTION_MAP_STEPS, 10000)
        self.assertEqual(production.PRODUCTION_MAP_FTOL, 1.0e-9)
        self.assertEqual(production.PRODUCTION_MAP_XTOL, 1.0e-9)

    def test_rate_prior_sigma_controls_map_coordinate_scale(self):
        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "node_days": np.asarray([0.0, 1.0, 2.0]),
        }
        values = production.initial_values_for_data(data)
        scales = production.map_parameter_scale_values(
            values,
            trajectory_rate_prior_sigma_log=0.1,
        )

        np.testing.assert_allclose(scales["np_log_rate_nodes"], 0.1)
        np.testing.assert_allclose(scales["incidence_angle_deg"], 15.0)

    def test_optional_third_decon_adds_independent_sites(self):
        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "node_days": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            "reset_days": np.asarray([1.5, 3.5]),
        }
        values = production.initial_values_for_data(data, fit_incidence_angle=False)

        self.assertIn("third_decon_initial_thickness_log_nm", values)
        self.assertIn("post_third_decon_tile_log_offset", values)
        self.assertEqual(values["third_decon_initial_thickness_log_nm"].shape, (1, 1))

    def test_optional_third_decon_resets_trajectory_baseline(self):
        data = _two_reset_trajectory_data()
        jax, jnp, _, _ = production.require_numpyro()
        ice, _, node_ice, _ = production._trajectory_state(
            data,
            np.zeros((1, 7)),
            np.asarray([[np.log(1.0)]]),
            np.asarray(
                [[[np.log(2.0)]], [[np.log(3.0)]]]
            ),
            jax,
            jnp,
        )

        np.testing.assert_allclose(
            np.asarray(ice),
            [[1.0, 2.0, 2.0, 3.0, 4.0, 3.0, 4.0]],
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(np.asarray(node_ice), np.asarray(ice), rtol=1.0e-6)

    def test_hgcdte_n_prior_sigma_controls_only_n_map_coordinate_scale(self):
        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "node_days": np.asarray([0.0, 1.0, 2.0]),
        }
        values = production.initial_values_for_data(data)
        scales = production.map_parameter_scale_values(
            values,
            absolute_hgcdte_n_scale_prior_sigma_log=0.1,
        )

        np.testing.assert_allclose(scales["absolute_hgcdte_n_log_scale"], 0.1)
        np.testing.assert_allclose(
            scales["absolute_hgcdte_k_log_scale"],
            production.ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        )

    def test_fixed_ice_n_bounds_remove_the_coordinate_from_map_search(self):
        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "node_days": np.asarray([0.0, 1.0, 2.0]),
        }
        values = production.initial_values_for_data(
            data,
            fit_incidence_angle=False,
            fit_ice_n_constant=True,
        )
        jax, _, _, _ = production.require_numpyro()
        flat_initial, unravel, flat_lower, flat_upper = production.map_parameter_bounds(
            values,
            jax,
            fixed_parameter_values={"ice_n_log_constant": np.zeros((1, 1))},
        )
        lower = np.asarray(flat_lower, dtype=float)
        upper = np.asarray(flat_upper, dtype=float)
        initial = np.asarray(flat_initial, dtype=float)
        fixed = np.flatnonzero(np.isclose(lower, -1.0e-12))
        self.assertEqual(fixed.size, 1)
        np.testing.assert_allclose(initial[fixed], 0.0)
        np.testing.assert_allclose(upper[fixed], 1.0e-12)
        self.assertEqual(unravel(jax.numpy.asarray(initial))["ice_n_log_constant"].shape, (1, 1))

    def test_initial_values_use_direct_independent_tile_sites(self):
        data = {
            "n_scas": 1,
            "n_tiles": 2,
            "node_days": np.asarray([0.0, 1.0, 2.0]),
        }
        values = production.initial_values_for_data(data)

        self.assertEqual(
            set(values),
            {
                "ar_tile_raw",
                "initial_thickness_log_nm",
                "reset_initial_thickness_log_nm",
                "incidence_angle_deg",
                "np_log_rate_nodes",
                "post_reset_tile_log_offset",
                "absolute_hgcdte_n_log_scale",
                "absolute_hgcdte_k_log_scale",
            },
        )
        self.assertEqual(values["ar_tile_raw"].shape, (1, 2, 6))
        self.assertEqual(values["np_log_rate_nodes"].shape, (1, 2, 3))
        self.assertEqual(values["initial_thickness_log_nm"].shape, (1, 2))
        self.assertEqual(values["incidence_angle_deg"], 7.5)

    def test_led_log_offset_initial_values_have_one_site_per_led(self):
        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "first_by_led": np.arange(6),
        }
        values = production.initial_values_for_data(
            data,
            fit_incidence_angle=False,
            fit_led_log_offsets=True,
        )

        self.assertEqual(values["led_log_offsets"].shape, (1, 6))
        np.testing.assert_array_equal(values["led_log_offsets"], 0.0)
        scales = production.map_parameter_scale_values(values)
        np.testing.assert_allclose(
            scales["led_log_offsets"], production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG
        )

    def test_half_normal_initial_values_use_physical_sites_and_bounds(self):
        data = {
            "n_scas": 1,
            "n_tiles": 2,
            "node_days": np.asarray([0.0, 1.0, 2.0]),
        }
        values = production.initial_values_for_data(
            data,
            initial_thickness_prior="half_normal",
            initial_thickness_prior_sigma_nm=5.0,
        )

        self.assertIn("initial_thickness_nm", values)
        self.assertIn("reset_initial_thickness_nm", values)
        self.assertNotIn("initial_thickness_log_nm", values)
        np.testing.assert_allclose(
            values["initial_thickness_nm"],
            5.0 * np.sqrt(2.0 / np.pi),
        )
        scales = production.map_parameter_scale_values(values)
        np.testing.assert_allclose(scales["initial_thickness_nm"], 5.0)

        jax, _, _, _ = production.require_numpyro()
        _, unravel, flat_lower, _ = production.map_parameter_bounds(values, jax)
        lower = unravel(jax.numpy.asarray(flat_lower))
        np.testing.assert_allclose(lower["initial_thickness_nm"], 0.0)
        np.testing.assert_allclose(lower["reset_initial_thickness_nm"], 0.0)

    def test_ordered_ar_reference_round_trips_to_increasing_indices(self):
        reference, lower, upper = production.ar_prior_arrays()
        active = production.ar_active_parameter_indices()
        raw = production.ar_reference_raw(
            reference,
            lower,
            upper,
            active,
        )

        jax, jnp, _, _ = production.require_numpyro()
        physical = np.asarray(
            production.ordered_ar_indices_from_raw(
                jnp.asarray(raw), jax, jnp
            )
        )

        np.testing.assert_allclose(physical[active], reference[active], rtol=1e-5)
        self.assertTrue(np.all(np.diff(physical[[1, 4, 7]]) > 0.0))

    def test_warm_start_accepts_scalar_sites(self):
        data = {
            "counts": np.ones((1, 1)),
            "sums": np.zeros((1, 1)),
            "sum_squared": np.ones((1, 1)),
        }
        initial = {"incidence_angle_deg": 7.5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posterior_samples.npz"
            np.savez(
                path,
                map_only=np.asarray(1),
                incidence_angle_deg=np.asarray([6.25]),
            )
            values = production.load_map_warm_start(path, initial, data)

        self.assertEqual(values["incidence_angle_deg"], 6.25)

    def test_warm_start_reduces_node_rate_archive_to_scalar_rate_site(self):
        data = {"counts": np.ones((1, 1)), "sums": np.zeros((1, 1)), "sum_squared": np.ones((1, 1))}
        initial = {"log_rate_scale": np.zeros((1, 1))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posterior_samples.npz"
            np.savez(
                path,
                map_only=np.asarray(1),
                np_log_rate_nodes=np.asarray([[[1.25, 1.25, 1.25]]]),
            )
            values = production.load_map_warm_start(path, initial, data)

        np.testing.assert_allclose(values["log_rate_scale"], [[1.25]])

    def test_legacy_emcee_warm_start_leaves_new_spectral_sites_at_zero(self):
        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "node_days": np.asarray([0.0, 1.0, 2.0]),
        }
        initial = production.initial_values_for_data(
            data,
            constant_rate=True,
            fit_incidence_angle=False,
            fit_ice_n_constant=True,
            fit_ar_hgcdte_modulation=True,
        )
        new_sites = (
            "ice_n_log_constant",
            "ar_n_log_modulation",
            "hgcdte_n_log_modulation",
            "hgcdte_k_log_modulation",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posterior_samples.npz"
            np.savez(path, map_only=np.asarray(1))
            values = production.load_map_warm_start(path, initial, data)

        for name in new_sites:
            np.testing.assert_array_equal(values[name], np.zeros_like(initial[name]))

    def test_warm_start_converts_legacy_log_thickness_sites(self):
        data = {
            "counts": np.ones((1, 1)),
            "sums": np.zeros((1, 1)),
            "sum_squared": np.ones((1, 1)),
        }
        initial = {
            "initial_thickness_nm": np.full((1, 1), 5.0),
            "reset_initial_thickness_nm": np.full((1, 1), 5.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posterior_samples.npz"
            np.savez(
                path,
                map_only=np.asarray(1),
                initial_thickness_log_nm=np.asarray([[np.log(2.5)]]),
                reset_initial_thickness_log_nm=np.asarray([[np.log(1.25)]]),
            )
            values = production.load_map_warm_start(path, initial, data)

        np.testing.assert_allclose(values["initial_thickness_nm"], [[2.5]])
        np.testing.assert_allclose(values["reset_initial_thickness_nm"], [[1.25]])

    def test_model_exposes_only_fixed_production_latents(self):
        jax, _, _, _ = production.require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "counts": np.ones((1, 6)),
            "sums": np.zeros((1, 6)),
            "sum_squared": np.ones((1, 6)),
            "days": np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.arange(1, 7),
            "first_by_led": np.arange(6),
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0]),
            "sca_names": np.asarray(["WFI01"]),
            "groups": np.arange(6),
            "repeat": np.asarray([0, 0, 1, 1, 2, 2]),
            "metadata": {0: {"tile_row": 0, "tile_col": 0}},
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((1, 6, 1)),
            "outside_output": np.zeros((1, 6)),
            "clean_output": np.ones((1, 6)),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "absolute_substrate_power": True,
        }
        model = production.build_model(data, context)
        from numpyro.handlers import seed, trace

        sites = trace(seed(model, jax.random.PRNGKey(102))).get_trace()

        self.assertEqual(sites["ar_tile_raw"]["value"].shape, (1, 1, 6))
        self.assertEqual(
            sites["np_log_rate_nodes"]["value"].shape,
            (1, 1, 3),
        )
        self.assertEqual(
            sites["absolute_hgcdte_n_log_scale"]["value"].shape,
            (1, 1),
        )
        for site in ("ar_mu_raw", "log_trajectory_scale", "complex_q_parameters"):
            self.assertNotIn(site, sites)

        half_model = production.build_model(
            data,
            context,
            initial_thickness_prior="half_normal",
            initial_thickness_prior_sigma_nm=5.0,
        )
        half_sites = trace(seed(half_model, jax.random.PRNGKey(107))).get_trace()
        self.assertEqual(
            half_sites["initial_thickness_nm"]["value"].shape,
            (1, 1),
        )
        self.assertEqual(
            half_sites["reset_initial_thickness_nm"]["value"].shape,
            (1, 1),
        )
        self.assertNotIn("initial_thickness_log_nm", half_sites)
        self.assertNotIn("reset_initial_thickness_log_nm", half_sites)
        self.assertTrue(
            np.all(np.asarray(half_sites["initial_thickness_nm"]["value"]) >= 0.0)
        )

    def test_fixed_hgcdte_n_is_not_an_optimized_sample_site(self):
        jax, _, _, _ = production.require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "counts": np.ones((1, 6)),
            "sums": np.zeros((1, 6)),
            "sum_squared": np.ones((1, 6)),
            "days": np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.arange(1, 7),
            "first_by_led": np.arange(6),
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0]),
            "sca_names": np.asarray(["WFI01"]),
            "groups": np.arange(6),
            "repeat": np.asarray([0, 0, 1, 1, 2, 2]),
            "metadata": {0: {"tile_row": 0, "tile_col": 0}},
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((1, 6, 1)),
            "outside_output": np.zeros((1, 6)),
            "clean_output": np.ones((1, 6)),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "absolute_substrate_power": True,
        }
        model = production.build_model(
            data,
            context,
            fixed_absolute_hgcdte_n_log_scale=0.0,
        )
        initial = production.initial_values_for_data(
            data,
            fixed_absolute_hgcdte_n_log_scale=0.0,
        )
        self.assertNotIn("absolute_hgcdte_n_log_scale", initial)
        from numpyro.handlers import seed, trace

        sites = trace(seed(model, jax.random.PRNGKey(104))).get_trace()
        self.assertEqual(
            sites["absolute_hgcdte_n_log_scale"]["type"],
            "deterministic",
        )
        np.testing.assert_allclose(
            np.asarray(sites["absolute_hgcdte_n_log_scale"]["value"]),
            0.0,
        )
        residuals = model.production_residuals(
            {name: jax.numpy.asarray(value) for name, value in initial.items()}
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(residuals))))

    def test_reset_aware_rate_metrics_exclude_the_reset_jump(self):
        data, context, *_ = _prediction_fixture()
        data.update(
            {
                "n_scas": 1,
                "n_tiles": 1,
                "tile_ids": np.asarray([0]),
                "sca_names": np.asarray(["WFI01"]),
                "groups": np.arange(12),
                "repeat": np.tile(np.asarray([0, 1]), 6),
                "metadata": {0: {"tile_row": 0, "tile_col": 0}},
                "node_days": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
                "reset_day": 2.0,
            }
        )
        model = production.build_model(
            data,
            context,
            fit_incidence_angle=False,
            smoothness_weight=0.0,
        )
        from numpyro.handlers import seed, substitute, trace

        sites = trace(
            seed(
                substitute(
                    model,
                    data={
                        "np_log_rate_nodes": np.log(
                            np.asarray([[[2.0, 2.0, 4.0, 4.0, 4.0]]])
                        )
                    },
                ),
                107,
            )
        ).get_trace()

        pre_accumulation = 2.0 + (4.0 - 2.0) / np.log(2.0)
        post_accumulation = 4.0 * (5.1 - 2.0)
        expected_pre = pre_accumulation / 2.0
        expected_post = post_accumulation / (5.1 - 2.0)
        expected_reset_aware = (pre_accumulation + post_accumulation) / 5.1

        np.testing.assert_allclose(
            sites["tile_pre_reset_mean_rate_nm_day"]["value"],
            [[expected_pre]],
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            sites["tile_post_reset_mean_rate_nm_day"]["value"],
            [[expected_post]],
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            sites["tile_mean_rate_nm_day"]["value"],
            [[expected_reset_aware]],
            rtol=1.0e-6,
        )
        self.assertGreater(
            float(sites["tile_cross_reset_mean_rate_nm_day"]["value"][0, 0]),
            expected_reset_aware,
        )

    def test_experimental_entropy_regularization_replaces_curvature_factor(self):
        jax, _, _, _ = production.require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "counts": np.ones((1, 6)),
            "sums": np.zeros((1, 6)),
            "sum_squared": np.ones((1, 6)),
            "days": np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.arange(1, 7),
            "first_by_led": np.arange(6),
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0]),
            "sca_names": np.asarray(["WFI01"]),
            "groups": np.arange(6),
            "repeat": np.asarray([0, 0, 1, 1, 2, 2]),
            "metadata": {0: {"tile_row": 0, "tile_col": 0}},
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((1, 6, 1)),
            "outside_output": np.zeros((1, 6)),
            "clean_output": np.ones((1, 6)),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "absolute_substrate_power": True,
        }
        model = production.build_model(
            data,
            context,
            smoothness_weight=0.0,
            entropy_reference_log_rates=np.zeros((1, 1, 3)),
            entropy_weight=0.1,
            rate_prior_log_mean=np.zeros((1, 1, 3)),
        )
        from numpyro.handlers import seed, trace

        sites = trace(seed(model, jax.random.PRNGKey(103))).get_trace()

        self.assertIn("trajectory_entropy", sites)
        self.assertNotIn("trajectory_curvature", sites)

    def test_constant_rate_model_replaces_node_rate_site(self):
        jax, _, _, _ = production.require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "counts": np.ones((1, 6)),
            "sums": np.zeros((1, 6)),
            "sum_squared": np.ones((1, 6)),
            "days": np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.arange(1, 7),
            "first_by_led": np.arange(6),
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0]),
            "sca_names": np.asarray(["WFI01"]),
            "groups": np.arange(6),
            "repeat": np.asarray([0, 0, 1, 1, 2, 2]),
            "metadata": {0: {"tile_row": 0, "tile_col": 0}},
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((1, 6, 1)),
            "outside_output": np.zeros((1, 6)),
            "clean_output": np.ones((1, 6)),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "absolute_substrate_power": True,
        }
        model = production.build_model(data, context, constant_rate=True)
        initial = production.initial_values_for_data(data, constant_rate=True)
        self.assertIn("log_rate_scale", initial)
        self.assertNotIn("np_log_rate_nodes", initial)
        from numpyro.handlers import seed, trace

        sites = trace(seed(model, jax.random.PRNGKey(104))).get_trace()

        self.assertEqual(sites["log_rate_scale"]["value"].shape, (1, 1))
        self.assertNotIn("np_log_rate_nodes", sites)

    def test_ice_n_modulation_basis_is_normalized(self):
        basis = production.ice_n_modulation_basis(
            np.asarray([600.0, 1000.0, 1600.0])
        )
        self.assertEqual(
            basis.shape,
            (3, len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM)),
        )
        np.testing.assert_allclose(np.sum(basis, axis=1), 1.0)

    def test_spectral_modulation_basis_is_anchored(self):
        basis = production.spectral_modulation_basis(
            np.asarray([600.0, production.AR_HGCDTE_MODULATION_ANCHOR_NM, 1600.0])
        )
        np.testing.assert_allclose(basis[1], 0.0)

    def test_pupil_quadrature_is_normalized_and_has_normal_limit(self):
        directions, weights = production.incidence_pupil_quadrature(0.0)
        np.testing.assert_allclose(directions, [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(weights, [1.0])

        directions, weights = production.incidence_pupil_quadrature(3.576334)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        self.assertTrue(np.all(directions[:, 2] > 0.0))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_pupil_predictor_is_finite_and_changes_oblique_power_average(self):
        data, context, ar_parameters, ice_thickness, hgcdte_parameters = _prediction_fixture()
        single_ray = production.build_prediction_function(data, context)
        pupil = production.build_prediction_function(
            data,
            context,
            incidence_pupil_half_angle_deg=3.576334,
        )

        single_prediction = np.asarray(
            single_ray(ice_thickness, ar_parameters, hgcdte_parameters, 0.0)
        )
        pupil_prediction = np.asarray(
            pupil(ice_thickness, ar_parameters, hgcdte_parameters, 0.0)
        )

        self.assertEqual(single_prediction.shape, (1, 12))
        self.assertEqual(pupil_prediction.shape, (1, 12))
        self.assertTrue(np.all(np.isfinite(pupil_prediction)))
        self.assertFalse(np.allclose(single_prediction, pupil_prediction))

    def test_fixed_normal_incidence_predictor_matches_two_polarization_limit(self):
        data, context, ar_parameters, ice_thickness, hgcdte_parameters = _prediction_fixture()
        general = production.build_prediction_function(data, context)
        normal = production.build_prediction_function(
            data,
            context,
            fixed_normal_incidence=True,
        )

        general_prediction = np.asarray(
            general(ice_thickness, ar_parameters, hgcdte_parameters, 0.0)
        )
        normal_prediction = np.asarray(
            normal(ice_thickness, ar_parameters, hgcdte_parameters, 0.0)
        )
        np.testing.assert_allclose(normal_prediction, general_prediction, rtol=1.0e-12, atol=1.0e-12)

    def test_clean_relative_response_is_unity_without_ice(self):
        data, context, ar_parameters, _, hgcdte_parameters = _prediction_fixture()
        predictor = production.build_prediction_function(
            data,
            context,
            normalize_clean_response=True,
        )
        prediction = predictor(
            np.zeros((1, 12)),
            ar_parameters,
            hgcdte_parameters,
            0.0,
        )
        np.testing.assert_allclose(prediction, 0.0, atol=1.0e-12)

    def test_porosity_ice_index_matches_direct_effective_index_values(self):
        data, context, ar_parameters, ice_thickness, hgcdte_parameters = _prediction_fixture()
        parameters = (0.0, 0.4, 100.0, 1.32)
        porosity = parameters[1] * (
            1.0 - np.exp(-ice_thickness / parameters[2])
        )
        solid_lorentz_lorenz = (parameters[3] ** 2 - 1.0) / (
            parameters[3] ** 2 + 2.0
        )
        effective_lorentz_lorenz = (1.0 - porosity) * solid_lorentz_lorenz
        effective_n = np.sqrt(
            (1.0 + 2.0 * effective_lorentz_lorenz)
            / (1.0 - effective_lorentz_lorenz)
        )
        porosity_predictor = production.build_prediction_function(
            data,
            context,
            ice_n_porosity_parameters=parameters,
        )
        direct_predictor = production.build_prediction_function(data, context)
        np.testing.assert_allclose(
            porosity_predictor(
                ice_thickness,
                ar_parameters,
                hgcdte_parameters,
                0.0,
            ),
            direct_predictor(
                ice_thickness,
                ar_parameters,
                hgcdte_parameters,
                0.0,
                ice_n_log_values=np.log(effective_n / production.ICE_INDEX_DEFAULT),
            ),
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    def test_incidence_angle_can_be_fixed_at_zero(self):
        data, context, *_ = _prediction_fixture()
        data.update(
            {
                "n_scas": 1,
                "n_tiles": 1,
                "tile_ids": np.asarray([0]),
                "sca_names": np.asarray(["WFI01"]),
                "groups": np.arange(12),
                "repeat": np.tile(np.asarray([0, 1]), 6),
                "metadata": {0: {"tile_row": 0, "tile_col": 0}},
            }
        )
        model = production.build_model(
            data,
            context,
            fit_incidence_angle=False,
            incidence_pupil_half_angle_deg=3.576334,
        )
        initial = production.initial_values_for_data(data, fit_incidence_angle=False)
        self.assertNotIn("incidence_angle_deg", initial)
        from numpyro.handlers import seed, trace

        jax, _, _, _ = production.require_numpyro()
        sites = trace(seed(model, jax.random.PRNGKey(106))).get_trace()
        self.assertNotIn("incidence_angle_deg", sites)

    def test_constant_rate_model_can_add_ice_n_modulation(self):
        jax, _, _, _ = production.require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "counts": np.ones((1, 6)),
            "sums": np.zeros((1, 6)),
            "sum_squared": np.ones((1, 6)),
            "days": np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.arange(1, 7),
            "first_by_led": np.arange(6),
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0]),
            "sca_names": np.asarray(["WFI01"]),
            "groups": np.arange(6),
            "repeat": np.asarray([0, 0, 1, 1, 2, 2]),
            "metadata": {0: {"tile_row": 0, "tile_col": 0}},
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((1, 6, 1)),
            "outside_output": np.zeros((1, 6)),
            "clean_output": np.ones((1, 6)),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "ice_n_modulation_basis": np.ones(
                (1, len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
            )
            / len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM),
            "absolute_substrate_power": True,
        }
        model = production.build_model(
            data,
            context,
            constant_rate=True,
            fit_ice_n_modulation=True,
        )
        from numpyro.handlers import seed, trace

        sites = trace(seed(model, jax.random.PRNGKey(105))).get_trace()
        self.assertEqual(
            sites["ice_n_log_modulation"]["value"].shape,
            (1, 1, len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM)),
        )

    def test_constant_rate_model_can_fit_constant_ice_n(self):
        jax, _, _, _ = production.require_numpyro()
        jax.config.update("jax_enable_x64", True)
        data = {
            "counts": np.ones((1, 6)),
            "sums": np.zeros((1, 6)),
            "sum_squared": np.ones((1, 6)),
            "days": np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1]),
            "node_days": np.asarray([0.0, 1.0, 2.0]),
            "led": np.arange(1, 7),
            "first_by_led": np.arange(6),
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0]),
            "sca_names": np.asarray(["WFI01"]),
            "groups": np.arange(6),
            "repeat": np.asarray([0, 0, 1, 1, 2, 2]),
            "metadata": {0: {"tile_row": 0, "tile_col": 0}},
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((1, 6, 1)),
            "outside_output": np.zeros((1, 6)),
            "clean_output": np.ones((1, 6)),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "absolute_substrate_power": True,
        }
        model = production.build_model(
            data,
            context,
            constant_rate=True,
            fit_ice_n_constant=True,
        )
        initial = production.initial_values_for_data(
            data,
            constant_rate=True,
            fit_ice_n_constant=True,
        )
        self.assertEqual(initial["ice_n_log_constant"].shape, (1, 1))
        from numpyro.handlers import seed, trace

        sites = trace(seed(model, jax.random.PRNGKey(107))).get_trace()
        self.assertEqual(sites["ice_n_log_constant"]["value"].shape, (1, 1))
        self.assertEqual(sites["ice_n_constant"]["value"].shape, (1, 1))

    def test_ice_n_modulation_and_constant_are_mutually_exclusive(self):
        data = {
            "counts": np.ones((1, 1)),
            "sums": np.zeros((1, 1)),
            "sum_squared": np.ones((1, 1)),
            "days": np.asarray([0.0]),
            "node_days": np.asarray([0.0]),
            "led": np.asarray([1]),
            "first_by_led": np.asarray([0]),
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([0]),
            "sca_index": np.asarray([0]),
            "sca_names": np.asarray(["WFI01"]),
            "groups": np.asarray([0]),
            "repeat": np.asarray([0]),
            "metadata": {0: {"tile_row": 0, "tile_col": 0}},
        }
        context = {
            "wavelengths": np.asarray([1000.0]),
            "model_weights": np.ones((1, 1, 1)),
            "outside_output": np.zeros((1, 1)),
            "clean_output": np.ones((1, 1)),
            "hgcdte_n": np.asarray([3.6]),
            "hgcdte_k": np.asarray([0.7]),
            "ice_n_modulation_basis": np.ones(
                (1, len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
            )
            / len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM),
            "absolute_substrate_power": True,
        }
        with self.assertRaises(ValueError):
            production.build_model(
                data,
                context,
                fit_ice_n_modulation=True,
                fit_ice_n_constant=True,
            )

if __name__ == "__main__":
    unittest.main()
