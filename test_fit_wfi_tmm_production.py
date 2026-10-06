import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fit_wfi_tmm_production as production


class FixedProductionSettingsTests(unittest.TestCase):
    def test_settings_use_pre_corrected_statistics_and_disable_fit_corrections(self):
        statistics_path = Path(
            "rrn_reduction_20261002/wfi01_tile_log_statistics_512_count_corrected.csv"
        )
        with patch.object(production, "_PRODUCTION_STATISTICS_PATH", statistics_path):
            args = production.parse_args()

        self.assertIsNone(args.fixed_corrections)
        self.assertTrue(args.corrections_already_applied)
        self.assertFalse(args.fit_late_acm_step)
        self.assertFalse(args.fit_additional_acm_step)
        self.assertTrue(args.fit_post_reset_tile_log_offset)
        self.assertEqual(args.post_reset_tile_log_offset_prior_sigma, 0.1)
        self.assertFalse(args.fit_pre_post_ice_index)
        self.assertFalse(args.fit_post_reset_ice_index)
        self.assertIsNone(args.fixed_trajectory_rate_nm_day)
        self.assertTrue(args.shared_absolute_hgcdte_scales)
        self.assertTrue(args.ordered_ar_indices)
        self.assertIsNone(args.fixed_ar_indices)
        self.assertEqual(args.ice_reset_day, production.PRODUCTION_RESET_DAY)
        self.assertTrue(args.tile_initial_thickness_lognormal)
        self.assertEqual(args.tile_initial_thickness_log_median_nm, 10.0)
        self.assertEqual(args.tile_initial_thickness_log_sigma, 1.5)
        self.assertTrue(args.tile_rate_variation)
        self.assertTrue(args.tile_rate_trajectory_variation)
        self.assertFalse(args.tile_rate_neighbor_smoothness)
        self.assertTrue(args.fully_np_trajectory)
        self.assertTrue(args.whiten_np_trajectory)
        self.assertEqual(args.np_smoothness_weight, 0.1)
        self.assertEqual(args.tile_rate_prior_sigma_log, 10.0)
        self.assertEqual(args.tile_rate_trajectory_smoothness_weight, 0.001)
        self.assertFalse(args.constant_rate_modulation)
        self.assertTrue(args.map_only)
        self.assertEqual(args.map_steps, 10000)

    def test_fixed_rate_initial_values_have_no_rate_latents(self):
        statistics_path = Path(
            "rrn_reduction_20261002/wfi01_tile_log_statistics_512_count_corrected.csv"
        )
        with patch.object(production, "_PRODUCTION_STATISTICS_PATH", statistics_path):
            args = production.parse_args()
        ar_reference, ar_lower, ar_upper = production.ar_prior_arrays(args.ar_layers)
        data = {
            "n_scas": 1,
            "n_tiles": 1,
            "tile_ids": np.asarray([1]),
            "days": np.asarray([1.0, 2.0]),
            "node_days": np.asarray([1.0, 2.0]),
            "repeat_values": np.asarray([1, 2]),
        }

        warmup_args = production.fixed_rate_warmup_args(args)
        initial_values = production.initial_values_for_data(
            data, warmup_args, ar_reference, ar_lower, ar_upper
        )

        self.assertEqual(warmup_args.fixed_trajectory_rate_nm_day, 10.0)
        self.assertFalse(warmup_args.fixed_absolute_hgcdte_scales)
        self.assertTrue(warmup_args.shared_absolute_hgcdte_scales)
        self.assertFalse(warmup_args.ordered_ar_indices)
        np.testing.assert_allclose(
            warmup_args.fixed_ar_indices, [1.31, 1.73, 2.27]
        )
        self.assertFalse(warmup_args.fully_np_trajectory)
        self.assertNotIn("log_trajectory_scale", initial_values)
        self.assertNotIn("np_node_raw", initial_values)

    def test_shared_hgcdte_warmup_values_broadcast_to_tile_sites(self):
        transferred = production.transfer_map_initial_values(
            {
                "absolute_hgcdte_n_log_scale": np.asarray([0.25]),
                "absolute_hgcdte_k_log_scale": np.asarray([-0.1]),
            },
            {
                "absolute_hgcdte_n_log_scale": np.zeros((1, 3)),
                "absolute_hgcdte_k_log_scale": np.zeros((1, 3)),
            },
        )

        np.testing.assert_allclose(
            transferred["absolute_hgcdte_n_log_scale"], [[0.25, 0.25, 0.25]]
        )
        np.testing.assert_allclose(
            transferred["absolute_hgcdte_k_log_scale"], [[-0.1, -0.1, -0.1]]
        )

    def test_shared_hgcdte_values_keep_shared_target_shape(self):
        transferred = production.transfer_map_initial_values(
            {
                "absolute_hgcdte_n_log_scale": np.asarray([0.25]),
                "absolute_hgcdte_k_log_scale": np.asarray([-0.1]),
            },
            {
                "absolute_hgcdte_n_log_scale": np.zeros(1),
                "absolute_hgcdte_k_log_scale": np.zeros(1),
            },
            source_tile_ids=np.asarray([0]),
            target_tile_ids=np.arange(64),
            broadcast_tile_values=True,
        )

        np.testing.assert_allclose(
            transferred["absolute_hgcdte_n_log_scale"], [0.25]
        )
        np.testing.assert_allclose(
            transferred["absolute_hgcdte_k_log_scale"], [-0.1]
        )

    def test_tile_subset_warm_start_values_follow_tile_ids(self):
        transferred = production.transfer_map_initial_values(
            {
                "initial_thickness_log_nm": np.asarray([[1.2]]),
                "post_reset_tile_log_offset": np.asarray([[0.3]]),
                "absolute_hgcdte_n_log_scale": np.asarray([[0.4]]),
            },
            {
                "initial_thickness_log_nm": np.zeros((1, 3)),
                "post_reset_tile_log_offset": np.zeros((1, 3)),
                "absolute_hgcdte_n_log_scale": np.zeros((1, 3)),
            },
            source_tile_ids=np.asarray([2]),
            target_tile_ids=np.asarray([0, 2, 4]),
        )

        np.testing.assert_allclose(
            transferred["initial_thickness_log_nm"], [[0.0, 1.2, 0.0]]
        )
        np.testing.assert_allclose(
            transferred["post_reset_tile_log_offset"], [[0.0, 0.3, 0.0]]
        )
        np.testing.assert_allclose(
            transferred["absolute_hgcdte_n_log_scale"], [[0.0, 0.4, 0.0]]
        )

    def test_single_tile_warm_start_can_broadcast_tile_local_sites(self):
        transferred = production.transfer_map_initial_values(
            {"initial_thickness_log_nm": np.asarray([[1.2]])},
            {"initial_thickness_log_nm": np.zeros((1, 3))},
            source_tile_ids=np.asarray([2]),
            target_tile_ids=np.asarray([0, 2, 4]),
            broadcast_tile_values=True,
        )

        np.testing.assert_allclose(
            transferred["initial_thickness_log_nm"], [[1.2, 1.2, 1.2]]
        )

    def test_joint_independent_mode_has_direct_tile_ar_and_rate_sites(self):
        from numpyro.handlers import seed, trace

        jax, _, _, _, _, _, _ = production.require_numpyro()
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
            "late_step_design": np.zeros(4),
            "additional_step_design": np.zeros(4),
            "pre_reset_step_design": np.zeros(4),
            "n_scas": 1,
            "n_tiles": 2,
            "tile_ids": np.asarray([0, 1]),
            "metadata": {
                0: {"tile_row": 0, "tile_col": 0},
                1: {"tile_row": 0, "tile_col": 1},
            },
            "sca_index": np.zeros(2, dtype=int),
            "repeat_values": np.asarray([0, 1, 2, 3]),
            "groups": np.arange(4),
            "observation_days": np.asarray([0.0, 1.0, 2.0, 3.0]),
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
        model = production.build_model(
            data,
            context,
            log_noise_sigma=0.005,
            likelihood_weight=1.0e-8,
            ar_layers=1,
            independent_tile_ar_parameters=True,
            fully_np_trajectory=True,
            tile_rate_trajectory_variation=True,
            independent_tile_trajectories=True,
            tile_rate_neighbor_smoothness=False,
            tile_rate_trajectory_smoothness_weight=0.0,
            tile_initial_thickness_lognormal=True,
            absolute_substrate_power=True,
            use_hgcdte_optical_constants=True,
            shared_absolute_hgcdte_scales=False,
        )
        sites = trace(seed(model, jax.random.PRNGKey(102))).get_trace()

        self.assertEqual(sites["ar_tile_raw"]["value"].shape, (1, 2, 2))
        self.assertEqual(
            sites["trajectory_rate_tile_node_log_rates"]["value"].shape,
            (1, 2, 4),
        )
        self.assertEqual(
            sites["absolute_hgcdte_n_log_scale"]["value"].shape, (1, 2)
        )
        self.assertNotIn("ar_mu_raw", sites)
        for site in (
            "ar_sigma_raw",
            "ar_tile_sigma",
            "ar_z",
            "ar_tile_z",
            "np_log_rate_nodes",
            "trajectory_rate_tile_offsets",
            "trajectory_rate_tile_node_offsets",
        ):
            self.assertNotIn(site, sites)

    def test_constant_rate_mode_is_distinct_from_node_trajectory(self):
        with patch.object(
            production, "_PRODUCTION_STATISTICS_PATH", Path(
                "rrn_reduction_20261002/wfi01_tile_log_statistics_512_count_corrected.csv"
            )
        ):
            args = production.parse_args()

        self.assertTrue(hasattr(args, "constant_rate_trajectory"))
        self.assertFalse(args.constant_rate_trajectory)
        self.assertIsNone(args.fixed_trajectory_rate_nm_day)
        self.assertFalse(args.map_init_allow_tile_subset)

    def test_ordered_ar_reference_round_trips_to_increasing_indices(self):
        reference, lower, upper = production.ar_prior_arrays(3)
        active = production.ar_active_parameter_indices(3, False)
        raw = production.ar_reference_raw(
            reference,
            lower,
            upper,
            active,
            ordered_ar_indices=True,
        )

        jax, jnp, *_ = production.require_numpyro()
        physical = np.asarray(
            production.ordered_ar_indices_from_raw(
                jnp.asarray(raw), active, lower, upper, len(reference), jax, jnp
            )
        )

        np.testing.assert_allclose(physical[active], reference[active], rtol=1e-5)
        self.assertTrue(np.all(np.diff(physical[[1, 4, 7]]) > 0.0))

    def test_main_runs_the_wfi01_corrected_statistics_without_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wfi01_tile_log_statistics_512_count_corrected.csv").touch()
            (root / "wfi02_tile_log_statistics_512_count_corrected.csv").touch()
            output_root = root / "outputs"
            with (
                patch.object(production, "PRODUCTION_INPUT_ROOT", root),
                patch.object(production, "PRODUCTION_OUTPUT_ROOT", output_root),
                patch.object(production, "_fit_one") as fit_one,
                patch.object(sys, "argv", ["fit_wfi_tmm_production.py"]),
            ):
                production.main()

        self.assertEqual(fit_one.call_count, 1)
        self.assertEqual(
            production._PRODUCTION_STATISTICS_PATH.name,
            "wfi01_tile_log_statistics_512_count_corrected.csv",
        )

    def test_main_rejects_command_line_arguments(self):
        with patch.object(
            sys, "argv", ["fit_wfi_tmm_production.py", "--map-steps", "1"]
        ):
            with self.assertRaises(SystemExit):
                production.main()


if __name__ == "__main__":
    unittest.main()
