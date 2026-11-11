import numpy as np

from experiment_joint_wfi01_inverse_wavelength_curvature import InverseWavelengthCurvature
from experiment_joint_wfi01_fixed_ice import full_basis


def test_reciprocal_wavelength_grid_and_curvature_gradient():
    bounds = (474.0, 2333.0)
    mode_count = full_basis(np.array(bounds), bounds)[0].shape[0]
    curvature = InverseWavelengthCurvature(bounds, mode_count, grid_points=65)
    np.testing.assert_allclose(np.diff(curvature.u), np.diff(curvature.u)[0], rtol=1e-13)
    reciprocal = 1.0 / bounds[1] + curvature.u * (1.0 / bounds[0] - 1.0 / bounds[1])
    np.testing.assert_allclose(curvature.wavelengths_nm, 1.0 / reciprocal, rtol=1e-14)

    rng = np.random.default_rng(20261111)
    parameters = rng.normal(0.0, 0.03, size=2 * mode_count)
    weight = 2.7
    value, gradient, amplitude_cost, phase_cost = curvature.evaluate(parameters, weight)
    assert value == amplitude_cost + phase_cost
    assert amplitude_cost > 0.0
    assert phase_cost > 0.0

    for index in (0, mode_count // 2, mode_count, mode_count + mode_count // 2):
        step = 1e-6
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        numerical = (curvature.evaluate(upper, weight)[0]
                     - curvature.evaluate(lower, weight)[0]) / (2.0 * step)
        np.testing.assert_allclose(gradient[index], numerical, rtol=2e-7, atol=2e-8)


def test_zero_curvature_weight_is_a_null_penalty():
    bounds = (474.0, 2333.0)
    mode_count = full_basis(np.array(bounds), bounds)[0].shape[0]
    curvature = InverseWavelengthCurvature(bounds, mode_count, grid_points=33)
    value, gradient, amplitude_cost, phase_cost = curvature.evaluate(
        np.ones(2 * mode_count), 0.0
    )
    assert value == amplitude_cost == phase_cost == 0.0
    np.testing.assert_array_equal(gradient, np.zeros(2 * mode_count))
