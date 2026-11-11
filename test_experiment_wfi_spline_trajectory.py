import numpy as np
import pytest

from experiment_wfi_spline_trajectory import (
    build_incremental_knot_layout,
    build_spline_knot_layout,
    build_spline_knot_days,
    fit_spline_reusable,
    incremental_rate_initialization,
    incremental_new_knot_mask,
    natural_spline_values,
    spline_knot_layout_from_metadata,
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


def test_piecewise_layout_keeps_independent_reset_endpoints():
    data = {
        "days": np.arange(0.0, 7.5, 0.5),
        "node_days": np.arange(0.0, 7.5, 0.5),
        "reset_days": np.asarray([2.5, 5.0]),
        "reset_day": 2.5,
    }
    layout = build_spline_knot_layout(data, spacing_hours=48.0)

    np.testing.assert_allclose(layout["boundaries"], [0.0, 2.5, 5.0, 7.0])
    np.testing.assert_array_equal(layout["segments"][0], [0.0, 2.0, 2.5])
    np.testing.assert_array_equal(layout["segments"][1], [2.5, 4.0, 5.0])
    np.testing.assert_array_equal(layout["segments"][2], [5.0, 6.0, 7.0])


def _incremental_statistics(days):
    days = np.asarray(days, dtype=float)
    count = days.size
    return {
        "days": days,
        "node_days": days[~np.isin(days, [2.5, 5.0])],
        "groups": np.arange(count, dtype=int),
        "led": np.tile(np.arange(1, 7, dtype=int), (count + 5) // 6)[:count],
        "repeat": np.arange(count, dtype=int),
        "program": np.full(count, -1, dtype=int),
        "counts": np.ones((1, count), dtype=float),
        "sums": np.zeros((1, count), dtype=float),
        "sum_squared": np.zeros((1, count), dtype=float),
        "reset_days": np.asarray([2.5, 5.0]),
        "reset_day": 2.5,
    }


def test_incremental_layout_preserves_old_segments_and_appends_suffix():
    old_data = _incremental_statistics(
        np.arange(0.0, 7.5, 0.5)
    )
    new_data = _incremental_statistics(
        np.arange(0.0, 9.5, 0.5)
    )
    old_layout = build_spline_knot_layout(old_data, spacing_hours=48.0)
    layout = build_incremental_knot_layout(
        old_data,
        new_data,
        old_layout,
        spacing_hours=48.0,
    )

    assert len(layout["segments"]) == len(old_layout["segments"])
    np.testing.assert_allclose(layout["segments"][:-1], old_layout["segments"][:-1])
    np.testing.assert_allclose(layout["segments"][-1][:3], old_layout["segments"][-1])
    np.testing.assert_allclose(layout["segments"][-1][-1], 9.0)
    np.testing.assert_allclose(layout["boundaries"], [0.0, 2.5, 5.0, 9.0])
    np.testing.assert_array_equal(
        incremental_new_knot_mask(layout, 7.0),
        np.asarray(layout["flat_knots"] > 7.0),
    )
    initialized = incremental_rate_initialization(
        old_layout,
        np.arange(old_layout["flat_knots"].size, dtype=float),
        layout,
    )
    assert initialized.size == layout["flat_knots"].size


def test_incremental_layout_rejects_changed_historical_statistics():
    old_data = _incremental_statistics(
        np.arange(0.0, 7.5, 0.5)
    )
    new_data = _incremental_statistics(
        np.arange(0.0, 8.5, 0.5)
    )
    new_data["sums"][0, 2] = 1.0
    old_layout = build_spline_knot_layout(old_data, spacing_hours=48.0)

    with pytest.raises(ValueError, match="historical sums"):
        build_incremental_knot_layout(old_data, new_data, old_layout, spacing_hours=48.0)


def test_incremental_layout_can_refit_two_old_edge_knots():
    old_data = _incremental_statistics(
        np.arange(0.0, 7.5, 0.5)
    )
    new_data = _incremental_statistics(
        np.arange(0.0, 9.5, 0.5)
    )
    old_layout = build_spline_knot_layout(
        old_data,
        spacing_hours=48.0,
        extra_days=(5.1, 5.3),
    )
    layout = build_incremental_knot_layout(
        old_data,
        new_data,
        old_layout,
        spacing_hours=48.0,
        overlap_knots=2,
    )
    mask = incremental_new_knot_mask(layout, 3.0)

    assert layout["incremental_overlap_knots"] == 2
    assert np.count_nonzero(mask) == 4
    assert layout["incremental_anchor_day"] is None
    assert np.isclose(layout["segments"][-1][0], 5.0)
    assert np.isclose(layout["segments"][-1][-1], 9.0)
    restored = spline_knot_layout_from_metadata(
        layout["flat_knots"],
        layout["boundaries"],
        [len(segment) for segment in layout["segments"]],
    )
    np.testing.assert_array_equal(restored["flat_knots"], layout["flat_knots"])
    np.testing.assert_array_equal(restored["boundaries"], layout["boundaries"])
    old_rates = np.arange(old_layout["flat_knots"].size, dtype=float)
    initialized = incremental_rate_initialization(old_layout, old_rates, layout)
    assert initialized.size == layout["flat_knots"].size


def test_reusable_fit_changes_only_selected_rate_nodes():
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    current_values = {
        "initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "reset_initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "post_reset_tile_log_offset": np.asarray([[0.0]]),
        "np_log_rate_nodes": np.asarray([[[np.log(12.0)] * 5]]),
    }
    _, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in current_values.items()}
    )
    parameter_count = sum(np.asarray(value).size for value in current_values.values())
    target = jnp.ones(parameter_count)

    def residual(scaled_values, *unused):
        return scaled_values - target

    def jacobian(scaled_values, *unused):
        del scaled_values
        return jnp.eye(parameter_count)

    evaluator = {
        "residual": residual,
        "jacobian": jacobian,
        "unravel": unravel,
    }
    data = {
        "counts": np.ones((1, 5)),
        "sums": np.zeros((1, 5)),
        "sum_squared": np.zeros((1, 5)),
    }
    rate_reference = np.full_like(current_values["np_log_rate_nodes"], np.log(12.0))
    values, diagnostics = fit_spline_reusable(
        data,
        {"relative_log_response": np.zeros((6, 1))},
        current_values,
        rate_reference,
        evaluator,
        jax,
        jnp,
        max_nfev=10,
        free_rate_mask=np.asarray([[[False, False, False, True, True]]]),
    )

    np.testing.assert_allclose(
        np.asarray(values["np_log_rate_nodes"])[..., :3],
        np.asarray(current_values["np_log_rate_nodes"])[..., :3],
    )
    assert np.all(
        np.asarray(values["np_log_rate_nodes"])[..., 3:]
        > np.asarray(current_values["np_log_rate_nodes"])[..., 3:]
    )
    np.testing.assert_allclose(
        np.asarray(values["initial_thickness_log_nm"]),
        np.asarray(current_values["initial_thickness_log_nm"]),
    )
    assert diagnostics["parameter_count"] == 2


def test_incremental_layout_can_append_after_an_overlap_fit():
    old_data = _incremental_statistics(
        np.arange(0.0, 7.5, 0.5)
    )
    new_data = _incremental_statistics(
        np.arange(0.0, 9.5, 0.5)
    )
    newer_data = _incremental_statistics(
        np.arange(0.0, 11.5, 0.5)
    )
    old_layout = build_spline_knot_layout(
        old_data,
        spacing_hours=48.0,
        extra_days=(5.1, 5.3),
    )
    first_layout = build_incremental_knot_layout(
        old_data,
        new_data,
        old_layout,
        spacing_hours=48.0,
        overlap_knots=2,
    )
    second_layout = build_incremental_knot_layout(
        new_data,
        newer_data,
        first_layout,
        spacing_hours=48.0,
        overlap_knots=0,
    )

    assert second_layout["boundaries"][-2:] == pytest.approx([5.0, 11.0])
