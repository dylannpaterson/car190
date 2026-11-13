import numpy as np

from experiment_joint_wfi01_fixed_ice import full_basis
from experiment_joint_wfi01_inverse_wavelength_curvature import InverseWavelengthCurvature
from experiment_joint_wfi01_inverse_wavelength_maxent import ReciprocalWavelengthMaxEnt


def test_reciprocal_wavelength_functional_maxent_gradient():
    bounds = (474.0, 2333.0)
    mode_count = full_basis(np.array(bounds), bounds)[0].shape[0]
    grid = InverseWavelengthCurvature(bounds, mode_count, grid_points=65)
    maxent = ReciprocalWavelengthMaxEnt(grid)
    rng = np.random.default_rng(20261111)
    parameters = rng.normal(0.0, 0.025, size=2 * mode_count)
    weight = 850.0
    value, gradient = maxent.evaluate(parameters, weight)
    assert value > 0.0
    np.testing.assert_array_equal(gradient[mode_count:], np.zeros(mode_count))

    for index in (0, mode_count // 2, mode_count - 1):
        step = 1e-6
        upper, lower = parameters.copy(), parameters.copy()
        upper[index] += step
        lower[index] -= step
        numerical = (maxent.evaluate(upper, weight)[0]
                     - maxent.evaluate(lower, weight)[0]) / (2.0 * step)
        np.testing.assert_allclose(gradient[index], numerical, rtol=2e-7, atol=2e-8)
