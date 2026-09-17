import unittest
from unittest.mock import patch

import numpy as np

from analyze_wfi01_all_pixel_soc_model import (
    DEFAULT_STEP_TIME,
    all_pixel_loss_components,
    apply_step_offset,
    event_step_design,
    fit_global_entropy_sqrt,
    fit_global_soc_sqrt,
    pixel_log_statistics,
    parse_args,
    profile_step_offset,
)


try:
    import scipy.optimize  # noqa: F401

    SCIPY_AVAILABLE = True
except ModuleNotFoundError:
    SCIPY_AVAILABLE = False


class TestAllPixelGlobalFit(unittest.TestCase):
    def test_cli_defaults_to_known_acm_step(self):
        with patch("sys.argv", ["analyze_wfi01_all_pixel_soc_model.py"]):
            args = parse_args()

        self.assertEqual(args.step_time, DEFAULT_STEP_TIME)

    def test_pixel_log_statistics_match_direct_calculation(self):
        baseline = np.asarray([[2.0, 4.0], [8.0, 16.0]])
        current = baseline * np.exp(np.asarray([[0.1, -0.2], [0.3, 0.4]]))
        baseline_dq = np.zeros((2, 2), dtype=np.uint32)
        current_dq = np.zeros((2, 2), dtype=np.uint32)
        current_dq[0, 1] = 1

        result = pixel_log_statistics(
            baseline, baseline_dq, current, current_dq, chunk_rows=1
        )
        expected = np.asarray([0.1, 0.3, 0.4])

        self.assertEqual(result["pixel_count"], 3)
        self.assertAlmostEqual(result["sum_log_change"], np.sum(expected))
        self.assertAlmostEqual(
            result["sum_squared_log_change"], np.sum(expected**2)
        )
        self.assertAlmostEqual(result["mean_log_change"], np.mean(expected))

    def test_sufficient_statistic_loss_matches_explicit_pixel_loss(self):
        values = [np.asarray([0.0, 0.1, 0.2]), np.asarray([-0.2, -0.1])]
        observations = []
        for index, group_values in enumerate(values):
            observations.append(
                {
                    "pixel_count": len(group_values),
                    "log_change": float(np.mean(group_values)),
                    "within_sum_squares": float(
                        np.sum((group_values - np.mean(group_values)) ** 2)
                    ),
                }
            )
        prediction = np.asarray([0.05, -0.15])

        result = all_pixel_loss_components(observations, prediction)
        expected = sum(
            np.sum((group_values - fitted) ** 2)
            for group_values, fitted in zip(values, prediction)
        )

        self.assertAlmostEqual(result["total_sse"], expected)
        self.assertAlmostEqual(
            result["total_sse"], result["within_sse"] + result["between_sse"]
        )

    def test_global_grid_fit_recovers_known_scale(self):
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
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * 4 + repeat,
                        "detector": 1,
                        "days": days,
                        "pixel_count": 3,
                    }
                )
        from analyze_rcs_soc_ice_model import model_prediction

        prediction, _ = model_prediction(
            observations,
            response_curves,
            d_grid,
            "soc_sqrt",
            2.0,
            50.0,
            4.65,
            18.0,
        )
        offsets = np.asarray([-0.01, 0.0, 0.01])
        for row, value in zip(observations, prediction):
            pixels = value + offsets
            row["log_change"] = float(np.mean(pixels))
            row["within_sum_squares"] = float(
                np.sum((pixels - row["log_change"]) ** 2)
            )

        result = fit_global_soc_sqrt(
            observations,
            response_curves,
            d_grid,
            max_scale=5.0,
            scale_steps=101,
            reference_thickness_nm=50.0,
            reference_time_days=4.65,
            reset_day=18.0,
        )

        self.assertTrue(result["optimizer_converged"])
        self.assertAlmostEqual(result["parameter"], 2.0, places=6)
        self.assertLess(result["between_sse"], 1.0e-20)
        self.assertEqual(result["pixel_count"], 72)
        self.assertEqual(result["nonreference_pixel_count"], 54)

    def test_global_grid_fit_profiles_known_event_step(self):
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
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * 4 + repeat,
                        "detector": 1,
                        "days": days,
                        "pixel_count": 3,
                    }
                )
        from analyze_rcs_soc_ice_model import model_prediction

        prediction, _ = model_prediction(
            observations,
            response_curves,
            d_grid,
            "soc_sqrt",
            2.0,
            50.0,
            4.65,
            18.0,
        )
        step_mask = np.asarray([row["repeat"] >= 2 for row in observations])
        true_step = -0.04
        for row, value, is_after_step in zip(observations, prediction, step_mask):
            pixels = value + (true_step if is_after_step else 0.0) + np.asarray(
                [-0.01, 0.0, 0.01]
            )
            row["log_change"] = float(np.mean(pixels))
            row["within_sum_squares"] = float(
                np.sum((pixels - row["log_change"]) ** 2)
            )

        result = fit_global_soc_sqrt(
            observations,
            response_curves,
            d_grid,
            max_scale=5.0,
            scale_steps=101,
            reference_thickness_nm=50.0,
            reference_time_days=4.65,
            reset_day=18.0,
            step_mask=step_mask,
        )

        self.assertAlmostEqual(result["parameter"], 2.0, places=6)
        self.assertAlmostEqual(result["step_log_offset"], true_step, places=6)
        self.assertLess(result["between_sse"], 1.0e-20)

    def test_event_step_is_centered_on_each_led_reference(self):
        observations = [
            {"led": 1, "days": 0.0, "pixel_count": 5, "log_change": 0.0},
            {"led": 1, "days": 2.0, "pixel_count": 5, "log_change": -0.04},
            {"led": 2, "days": 1.0, "pixel_count": 5, "log_change": 0.0},
            {"led": 2, "days": 2.5, "pixel_count": 5, "log_change": 0.0},
        ]
        event_mask = np.asarray([False, True, True, True])

        design = event_step_design(observations, event_mask)
        step = profile_step_offset(observations, np.zeros(4), event_mask)
        prediction = apply_step_offset(
            np.zeros(4), event_mask, step, observations=observations
        )

        np.testing.assert_array_equal(design, [0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(step, -0.04)
        np.testing.assert_allclose(prediction, [0.0, -0.04, 0.0, 0.0])

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for the NP optimizer")
    def test_global_np_fit_recovers_known_nodes(self):
        d_grid = np.linspace(0.0, 100.0, 201)
        response_curves = np.empty((18, 6, len(d_grid)), dtype=float)
        for detector in range(18):
            for led in range(6):
                response_curves[detector, led] = np.exp(
                    -(led + 1) * 0.0005 * d_grid
                )
        observations = []
        days_by_repeat = (0.0, 0.5, 1.0, 1.5)
        node_multipliers = np.asarray((1.0, 1.02, 1.04, 1.05))
        for led in range(1, 7):
            for repeat, days in enumerate(days_by_repeat):
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * 4 + repeat,
                        "detector": 1,
                        "days": days,
                        "pixel_count": 3,
                    }
                )
        from analyze_rcs_soc_ice_model import soc_sqrt_thickness

        base = soc_sqrt_thickness(
            np.asarray([row["days"] for row in observations]),
            50.0,
            4.65,
            18.0,
        )
        thickness = base * node_multipliers[
            np.asarray([row["repeat"] for row in observations])
        ] * 2.0
        from analyze_rcs_soc_ice_model import response_prediction

        prediction = response_prediction(
            observations, response_curves, d_grid, thickness
        )
        offsets = np.asarray([-0.01, 0.0, 0.01])
        for row, value in zip(observations, prediction):
            pixels = value + offsets
            row["log_change"] = float(np.mean(pixels))
            row["within_sum_squares"] = float(
                np.sum((pixels - row["log_change"]) ** 2)
            )

        result = fit_global_entropy_sqrt(
            observations,
            response_curves,
            d_grid,
            initial_scale=2.0,
            initial_scales=(2.0,),
            max_scale=5.0,
            reference_thickness_nm=50.0,
            reference_time_days=4.65,
            reset_day=18.0,
            entropy_weight=0.0,
            smoothness_weight=0.0,
            max_node_multiplier=2.0,
        )

        self.assertTrue(result["optimizer_converged"])
        self.assertLess(result["between_sse"], 1.0e-16)
        expected_node_thickness = (
            2.0 * soc_sqrt_thickness(np.asarray(days_by_repeat), 50.0, 4.65, 18.0)
            * node_multipliers
        )
        np.testing.assert_allclose(
            result["node_thickness_nm"], expected_node_thickness, atol=1.0e-5
        )

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for the NP optimizer")
    def test_global_np_fit_recovers_nonzero_initial_thickness(self):
        d_grid = np.linspace(0.0, 100.0, 401)
        response_curves = np.empty((18, 6, len(d_grid)), dtype=float)
        for detector in range(18):
            for led in range(6):
                linear = -0.004 + 0.0003 * led
                quadratic = (led + 1) * 1.0e-5
                response_curves[detector, led] = np.exp(
                    linear * d_grid + quadratic * d_grid**2
                )
        observations = []
        days_by_repeat = (0.0, 0.5, 1.0, 1.5, 2.0)
        for led in range(1, 7):
            for repeat, days in enumerate(days_by_repeat):
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * len(days_by_repeat) + repeat,
                        "detector": 1,
                        "days": days,
                        "pixel_count": 3,
                    }
                )

        from analyze_rcs_soc_ice_model import response_prediction, soc_sqrt_thickness

        true_initial_thickness = 25.0
        incremental = 2.0 * soc_sqrt_thickness(
            np.asarray([row["days"] for row in observations]),
            50.0,
            4.65,
            18.0,
        )
        prediction = response_prediction(
            observations,
            response_curves,
            d_grid,
            true_initial_thickness + incremental,
        )
        offsets = np.asarray([-0.01, 0.0, 0.01])
        for row, value in zip(observations, prediction):
            pixels = value + offsets
            row["log_change"] = float(np.mean(pixels))
            row["within_sum_squares"] = float(
                np.sum((pixels - row["log_change"]) ** 2)
            )

        sqrt_result = fit_global_soc_sqrt(
            observations,
            response_curves,
            d_grid,
            max_scale=5.0,
            scale_steps=101,
            reference_thickness_nm=50.0,
            reference_time_days=4.65,
            reset_day=18.0,
            fit_initial_thickness=True,
            initial_thickness_starts=(0.0, 20.0, 40.0),
            max_initial_thickness_nm=50.0,
        )
        result = fit_global_entropy_sqrt(
            observations,
            response_curves,
            d_grid,
            initial_scale=2.0,
            initial_scales=(1.0, 2.0),
            max_scale=5.0,
            reference_thickness_nm=50.0,
            reference_time_days=4.65,
            reset_day=18.0,
            entropy_weight=0.01,
            smoothness_weight=0.1,
            max_node_multiplier=1.5,
            fit_initial_thickness=True,
            initial_thickness_starts=(0.0, 20.0, 40.0),
            max_initial_thickness_nm=50.0,
        )

        self.assertAlmostEqual(
            sqrt_result["initial_thickness_nm"], true_initial_thickness, places=3
        )
        self.assertAlmostEqual(sqrt_result["parameter"], 2.0, places=4)
        self.assertLess(sqrt_result["between_sse"], 1.0e-12)
        self.assertTrue(result["optimizer_converged"])
        self.assertEqual(result["optimizer_starts"], 18)
        self.assertAlmostEqual(
            result["initial_thickness_nm"], true_initial_thickness, places=3
        )
        np.testing.assert_allclose(
            result["thickness_nm"] - result["incremental_thickness_nm"],
            true_initial_thickness,
            atol=1.0e-3,
        )
        self.assertLess(result["between_sse"], 1.0e-12)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for the NP optimizer")
    def test_joint_response_factory_excludes_held_out_observations(self):
        d_grid = np.linspace(0.0, 100.0, 401)

        def response_factory(parameters):
            optical_slope = float(parameters[0])
            curves = np.empty((18, 6, len(d_grid)), dtype=float)
            for detector in range(18):
                for led in range(6):
                    linear = -0.004 + 0.0003 * led
                    quadratic = (led + 1) * 1.0e-5 * (
                        1.0 + optical_slope * (led - 2.5) / 3.0
                    )
                    curves[detector, led] = np.exp(
                        linear * d_grid + quadratic * d_grid**2
                    )
            return curves

        observations = []
        days_by_repeat = (0.0, 0.5, 1.0, 1.5, 2.0)
        for led in range(1, 7):
            for repeat, days in enumerate(days_by_repeat):
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * len(days_by_repeat) + repeat,
                        "detector": 1,
                        "days": days,
                        "pixel_count": 3,
                    }
                )

        from analyze_rcs_soc_ice_model import response_prediction, soc_sqrt_thickness

        incremental = 2.0 * soc_sqrt_thickness(
            np.asarray([row["days"] for row in observations]),
            50.0,
            4.65,
            18.0,
        )
        prediction = response_prediction(
            observations,
            response_factory(np.asarray([0.2])),
            d_grid,
            25.0 + incremental,
        )
        for row, value in zip(observations, prediction):
            row["log_change"] = float(value)
            row["within_sum_squares"] = 0.0
        fit_mask = np.asarray([row["repeat"] != 3 for row in observations])

        def fit(rows):
            return fit_global_entropy_sqrt(
                rows,
                None,
                d_grid,
                initial_scale=2.0,
                initial_scales=(2.0,),
                max_scale=5.0,
                reference_thickness_nm=50.0,
                reference_time_days=4.65,
                reset_day=18.0,
                entropy_weight=0.01,
                smoothness_weight=0.1,
                max_node_multiplier=1.5,
                initial_thickness_nm=25.0,
                response_factory=response_factory,
                optical_initial_parameters=np.asarray([0.0]),
                optical_parameter_starts=(np.asarray([0.0]),),
                optical_parameter_bounds=((-0.5, 0.5),),
                fit_mask=fit_mask,
                node_start_perturbation=0.0,
            )

        result = fit(observations)
        modified = [dict(row) for row in observations]
        for row in modified:
            if row["repeat"] == 3:
                row["log_change"] += 1.0
        modified_result = fit(modified)

        self.assertEqual(result["held_out_observations"], 6)
        self.assertAlmostEqual(result["optical_parameters"][0], 0.2, places=3)
        np.testing.assert_allclose(
            result["optical_parameters"],
            modified_result["optical_parameters"],
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            result["node_thickness_nm"],
            modified_result["node_thickness_nm"],
            atol=1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
