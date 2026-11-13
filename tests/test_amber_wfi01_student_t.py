import numpy as np
from scipy.optimize import minimize_scalar

from fit_amber_wfi01_student_t import profile_student_locations


def test_robust_locations_match_independent_scalar_search():
    values = np.array([
        [-1., 0., 1., 0.5, 120.],
        [0., 0.1, 0.2, 80., 81.],
        [-50., -49., -48., 40., 42.],
        [-1., 0., 1., np.nan, np.nan],
    ])
    scales = np.array([
        [2., 2., 3., 2., 2.],
        [1., 2., 1., 3., 3.],
        [5., 2., 4., 3., 7.],
        [1., 1., 1., 1., 1.],
    ])
    valid = np.isfinite(values)
    locations, residuals, scores = profile_student_locations(values, scales, valid)
    for i, location in enumerate(locations):
        y, s = values[i, valid[i]], scales[i, valid[i]]
        def objective(center):
            return 2.5 * np.log1p((y-center)**2/(4*s*s)).sum()
        grid = np.linspace(y.min()-1., y.max()+1., 20001)
        costs = np.array([objective(center) for center in grid])
        best = int(costs.argmin())
        reference = minimize_scalar(objective, bounds=(grid[max(0,best-2)],grid[min(len(grid)-1,best+2)]),
                                    method="bounded", options={"xatol":1e-10})
        assert abs(objective(location)-reference.fun) < 1e-8
        assert abs(scores[i].sum()) < 1e-6
    assert abs(locations[0]) < 1.
    assert np.all(residuals[~valid] == 0.)


def test_symmetric_student_location_is_zero():
    values = np.array([[-3., -1., 0., 1., 3.]])
    location, _, _ = profile_student_locations(values, np.ones_like(values), np.ones_like(values,dtype=bool))
    np.testing.assert_allclose(location, 0., atol=1e-10)
