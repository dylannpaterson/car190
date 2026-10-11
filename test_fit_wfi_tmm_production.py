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


class FixedProductionSettingsTests(unittest.TestCase):
    def test_settings_are_fixed_to_the_production_contract(self):
        config = production.ProductionConfig(Path("stats.csv"), Path("output"))

        self.assertEqual(config.tile_statistics, Path("stats.csv"))
        self.assertEqual(config.output_dir, Path("output"))
        self.assertIsNone(config.map_init_from)
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

if __name__ == "__main__":
    unittest.main()
