import unittest

import numpy as np

from analyze_rcs_soc_ice_model import (
    fit_entropy_sqrt_trajectory,
    fit_trajectory,
    generalized_relative_entropy,
    mean_squared_log_node_curvature,
    model_prediction,
    prepare_fit_observations,
    project_out,
)


def synthetic_ice_problem():
    d_grid = np.linspace(0.0, 100.0, 201)
    response_curves = np.empty((18, 6, len(d_grid)), dtype=float)
    for detector in range(18):
        for led in range(6):
            response_curves[detector, led] = np.exp(
                -(led + 1) * 0.0005 * d_grid
            )

    observations = []
    for led in range(1, 7):
        for repeat, days in enumerate((0.0, 0.5, 1.0, 1.5)):
            for detector in range(1, 19):
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * 4 + repeat,
                        "detector": detector,
                        "days": days,
                    }
                )

    prediction, _ = model_prediction(
        observations,
        response_curves,
        d_grid,
        "linear",
        15.0,
        50.0,
        4.65,
        18.0,
    )
    for row, value in zip(observations, prediction):
        row["log_change"] = value
    return observations, response_curves, d_grid, prediction


class TestFitSetup(unittest.TestCase):
    def test_generalized_relative_entropy_is_minimized_at_unity(self):
        self.assertEqual(generalized_relative_entropy(np.ones(4)), 0.0)
        self.assertGreater(
            generalized_relative_entropy(np.asarray([0.8, 1.0, 1.2])),
            0.0,
        )

    def test_time_aware_curvature_vanishes_for_linear_log_nodes(self):
        days = np.asarray([0.0, 0.2, 0.7, 1.6])
        log_nodes = 0.03 * days - 0.1
        self.assertAlmostEqual(
            mean_squared_log_node_curvature(log_nodes, days), 0.0, places=28
        )

    def test_direct_fit_recovers_known_rate(self):
        observations, response_curves, d_grid, _ = synthetic_ice_problem()
        selected, projection = prepare_fit_observations(
            observations, "direct", detector=1
        )
        observed = np.asarray([row["log_change"] for row in selected])

        result = fit_trajectory(
            selected,
            observed,
            projection,
            response_curves,
            d_grid,
            "linear",
            np.linspace(0.0, 30.0, 61),
            50.0,
            4.65,
            18.0,
        )

        self.assertEqual(result["parameter"], 15.0)
        self.assertLess(result["model_rms"], 1.0e-12)
        self.assertAlmostEqual(result["signal_retained_fraction"], 1.0)

    def test_model_prediction_accepts_single_led_subset(self):
        observations, response_curves, d_grid, all_prediction = synthetic_ice_problem()
        selected = [row for row in observations if row["led"] == 1]
        prediction, _ = model_prediction(
            selected,
            response_curves,
            d_grid,
            "linear",
            15.0,
            50.0,
            4.65,
            18.0,
        )

        self.assertEqual(len(prediction), len(selected))
        np.testing.assert_allclose(
            prediction,
            [value for row, value in zip(observations, all_prediction) if row["led"] == 1],
            atol=1.0e-12,
        )

    def test_entropy_nodes_recover_square_root_prior(self):
        observations, response_curves, d_grid, _ = synthetic_ice_problem()
        selected, projection = prepare_fit_observations(
            observations, "direct", detector=1
        )
        observed, _ = model_prediction(
            selected,
            response_curves,
            d_grid,
            "soc_sqrt",
            2.0,
            50.0,
            4.65,
            18.0,
        )

        result = fit_entropy_sqrt_trajectory(
            selected,
            observed,
            projection,
            response_curves,
            d_grid,
            initial_scale=1.8,
            max_scale=5.0,
            reference_thickness_nm=50.0,
            reference_time_days=4.65,
            reset_day=18.0,
            entropy_weight=0.1,
            smoothness_weight=0.1,
            max_node_multiplier=2.0,
        )

        self.assertTrue(result["optimizer_converged"])
        self.assertAlmostEqual(result["parameter"], 2.0, places=2)
        np.testing.assert_allclose(result["node_multipliers"], 1.0, atol=0.01)
        self.assertLess(result["model_rms"], 1.0e-5)

    def test_entropy_fit_can_select_zero_scale(self):
        observations, response_curves, d_grid, _ = synthetic_ice_problem()
        selected, projection = prepare_fit_observations(
            observations, "direct", detector=1
        )
        result = fit_entropy_sqrt_trajectory(
            selected,
            np.zeros(len(selected)),
            projection,
            response_curves,
            d_grid,
            initial_scale=0.1,
            max_scale=5.0,
            reference_thickness_nm=50.0,
            reference_time_days=4.65,
            reset_day=18.0,
            entropy_weight=0.1,
            smoothness_weight=0.1,
            max_node_multiplier=2.0,
        )

        self.assertEqual(result["parameter"], 0.0)
        self.assertEqual(result["model_rms"], 0.0)

    def test_projected_fit_removes_band_common_signal(self):
        observations, _, _, prediction = synthetic_ice_problem()
        _, projection = prepare_fit_observations(
            observations, "projected", detector=1
        )

        projected = project_out(prediction, projection)

        self.assertLess(
            np.sqrt(np.mean(projected**2)),
            np.sqrt(np.mean(prediction**2)) * 1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
