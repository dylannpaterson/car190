import numpy as np

from experiment_wfi_spline_trajectory import (
    build_spline_knot_days,
    natural_spline_values,
)


def test_build_spline_knot_days_adds_reset_and_endpoint():
    data = {
        "days": np.asarray([0.0, 1.25, 2.1]),
        "reset_day": 1.1,
    }
    knots = build_spline_knot_days(data, spacing_hours=12.0)
    np.testing.assert_allclose(knots, [0.0, 0.5, 1.0, 1.1, 1.5, 2.0, 2.1])


def test_natural_spline_matches_values_at_knots():
    import jax
    import jax.numpy as jnp

    knots = np.asarray([0.0, 0.5, 1.25, 2.0])
    values = np.asarray([[0.0, 1.0, -0.5, 2.0]])
    evaluated = natural_spline_values(knots, values, knots, jax, jnp)
    np.testing.assert_allclose(np.asarray(evaluated), values, rtol=0.0, atol=1.0e-12)
