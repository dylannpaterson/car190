"""Compare a reduced-knot spline ice trajectory with a cached NP fit.

This is an experiment-only driver. It deliberately keeps the production
non-parametric trajectory implementation unchanged.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import fit_wfi_tmm_production as production
import wfi_relative_response as relative
import run_wfi_relative_response_production as relative_runner


def build_spline_knot_days(data, spacing_hours=12.0, extra_days=()):
    """Return regular knots plus exact physical-reset and requested boundaries."""
    spacing_days = float(spacing_hours) / 24.0
    if not np.isfinite(spacing_days) or spacing_days <= 0.0:
        raise ValueError("Spline spacing must be positive and finite")
    start = float(np.min(data["days"]))
    end = float(np.max(data["days"]))
    regular = np.arange(start, end + 0.5 * spacing_days, spacing_days)
    candidates = np.concatenate(
        (
            regular,
            np.asarray([start, end, float(data["reset_day"])], dtype=float),
            np.asarray(tuple(extra_days), dtype=float),
        )
    )
    knots = np.unique(candidates)
    if np.any(knots < start) or np.any(knots > end):
        raise ValueError("Spline knots must lie within the observation span")
    if knots.size < 3 or np.any(np.diff(knots) <= 0.0):
        raise ValueError("Spline requires at least three increasing knots")
    return knots


def build_spline_knot_layout(data, spacing_hours=48.0, extra_days=()):
    """Build independent spline knot sets separated at every reset boundary.

    The reset day is included as the endpoint of both neighboring segments so
    the fitted log-rate can jump there.  Apart from those physical boundaries,
    knots are placed at the requested regular spacing and at the final data
    epoch.
    """
    spacing_days = float(spacing_hours) / 24.0
    if not np.isfinite(spacing_days) or spacing_days <= 0.0:
        raise ValueError("Spline spacing must be positive and finite")
    observation_days = np.asarray(data["days"], dtype=float).reshape(-1)
    node_days = np.asarray(data["node_days"], dtype=float).reshape(-1)
    if (
        observation_days.size < 2
        or np.any(~np.isfinite(observation_days))
        or np.any(np.diff(np.sort(observation_days)) < 0.0)
    ):
        raise ValueError("Spline observations must contain finite days")
    if node_days.size < 2 or np.any(~np.isfinite(node_days)):
        raise ValueError("Spline repeat nodes must contain finite days")
    start = float(np.min(observation_days))
    end = float(np.max(observation_days))
    if end <= start:
        raise ValueError("Spline observations must span a positive time interval")
    reset_days = np.asarray(production.reset_days_for_data(data), dtype=float)
    active_reset_days = reset_days[(reset_days > start) & (reset_days < end)]
    boundaries = np.concatenate(([start], active_reset_days, [end]))
    extra_days = np.asarray(tuple(extra_days), dtype=float).reshape(-1)
    if extra_days.size and (
        np.any(~np.isfinite(extra_days))
        or np.any(extra_days < start)
        or np.any(extra_days > end)
    ):
        raise ValueError("Spline extra knots must lie within the observation span")

    regular_days = np.arange(start, end + 0.5 * spacing_days, spacing_days)
    segment_knots = []
    for segment_index, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        candidates = [left, right]
        candidates.extend(
            regular_days[(regular_days >= left) & (regular_days <= right)]
        )
        candidates.extend(extra_days[(extra_days >= left) & (extra_days <= right)])
        knots = np.unique(np.asarray(candidates, dtype=float))
        if knots.size < 3 or np.any(np.diff(knots) <= 0.0):
            raise ValueError(
                "Each reset-separated spline segment requires at least three knots"
            )
        segment_knots.append(knots)

    flat_knots = np.concatenate(segment_knots)
    slices = []
    offset = 0
    for knots in segment_knots:
        slices.append((offset, offset + knots.size))
        offset += knots.size
    return {
        "boundaries": boundaries,
        "segments": tuple(segment_knots),
        "slices": tuple(slices),
        "flat_knots": flat_knots,
        "spacing_hours": float(spacing_hours),
    }


def validate_incremental_data_prefix(old_data, new_data):
    """Require the new statistics to append observations to the old fit."""
    old_days = np.asarray(old_data["days"], dtype=float)
    new_days = np.asarray(new_data["days"], dtype=float)
    if new_days.size <= old_days.size:
        raise ValueError("Incremental statistics must contain new observations")
    for name in ("groups", "days", "led", "repeat", "program"):
        old_values = np.asarray(old_data[name])
        new_values = np.asarray(new_data[name])
        if new_values.size < old_values.size or not np.array_equal(
            new_values[: old_values.size], old_values
        ):
            raise ValueError(
                f"Incremental statistics do not preserve the old {name} prefix"
            )
    old_node_days = np.asarray(old_data["node_days"], dtype=float)
    new_node_days = np.asarray(new_data["node_days"], dtype=float)
    if new_node_days.size < old_node_days.size or not np.array_equal(
        new_node_days[: old_node_days.size], old_node_days
    ):
        raise ValueError("Incremental statistics do not preserve the old node_days prefix")
    for name in ("counts", "sums", "sum_squared"):
        old_values = np.asarray(old_data[name], dtype=float)
        new_values = np.asarray(new_data[name], dtype=float)
        old_prefix = new_values[..., : old_values.shape[-1]]
        if old_prefix.shape != old_values.shape or not np.array_equal(
            old_prefix, old_values
        ):
            raise ValueError(
                f"Incremental statistics changed historical {name} values"
            )
    old_resets = np.asarray(production.reset_days_for_data(old_data), dtype=float)
    new_resets = np.asarray(production.reset_days_for_data(new_data), dtype=float)
    if not np.array_equal(old_resets, new_resets):
        raise ValueError("Incremental fitting does not support new or moved reset days")
    if not np.isclose(np.min(old_days), np.min(new_days), rtol=0.0, atol=1.0e-12):
        raise ValueError("Incremental statistics must have the same start day")
    if not np.max(new_days) > np.max(old_days):
        raise ValueError("Incremental statistics must extend the old time span")


def spline_knot_layout_from_metadata(
    knot_days,
    boundaries,
    segment_sizes,
    spacing_hours=None,
):
    """Reconstruct a possibly incremental layout stored in a fit archive."""
    knot_days = np.asarray(knot_days, dtype=float).reshape(-1)
    boundaries = np.asarray(boundaries, dtype=float).reshape(-1)
    segment_sizes = np.asarray(segment_sizes, dtype=int).reshape(-1)
    if (
        knot_days.size == 0
        or boundaries.size != segment_sizes.size + 1
        or np.any(segment_sizes < 2)
        or int(np.sum(segment_sizes)) != knot_days.size
        or np.any(~np.isfinite(knot_days))
        or np.any(~np.isfinite(boundaries))
        or np.any(np.diff(boundaries) <= 0.0)
    ):
        raise ValueError("Invalid spline knot layout metadata")
    segments = []
    slices = []
    offset = 0
    for index, size in enumerate(segment_sizes):
        stop = offset + int(size)
        segment = knot_days[offset:stop]
        if (
            np.any(np.diff(segment) <= 0.0)
            or not np.isclose(segment[0], boundaries[index], rtol=0.0, atol=1.0e-12)
            or not np.isclose(segment[-1], boundaries[index + 1], rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError("Spline knot layout metadata is inconsistent")
        segments.append(segment)
        slices.append((offset, stop))
        offset = stop
    return {
        "boundaries": boundaries,
        "segments": tuple(segments),
        "slices": tuple(slices),
        "flat_knots": knot_days,
        "spacing_hours": None if spacing_hours is None else float(spacing_hours),
    }


def build_incremental_knot_layout(
    old_data,
    new_data,
    old_layout,
    spacing_hours=48.0,
    extra_days=(),
    overlap_knots=0,
):
    """Build the full new spline while selecting only the edge nodes to refit."""
    overlap_knots = int(overlap_knots)
    if overlap_knots < 0:
        raise ValueError("Incremental overlap knot count must be non-negative")
    validate_incremental_data_prefix(old_data, new_data)
    old_end = float(np.max(np.asarray(old_data["days"], dtype=float)))
    new_end = float(np.max(np.asarray(new_data["days"], dtype=float)))
    old_flat_knots = np.asarray(old_layout["flat_knots"], dtype=float)
    full_layout = build_spline_knot_layout(
        new_data,
        spacing_hours=spacing_hours,
        extra_days=tuple(extra_days) + tuple(old_flat_knots.tolist()),
    )
    old_boundaries = np.asarray(old_layout["boundaries"], dtype=float)
    full_boundaries = np.asarray(full_layout["boundaries"], dtype=float)
    if old_boundaries.size < 2 or full_boundaries.size < 2:
        raise ValueError("Incremental fitting requires valid spline boundaries")
    old_start = float(np.min(np.asarray(old_data["days"], dtype=float)))
    active_old_resets = np.asarray(
        production.reset_days_for_data(old_data), dtype=float
    )
    active_old_resets = active_old_resets[
        (active_old_resets > old_start) & (active_old_resets < old_end)
    ]
    expected_old_boundaries = np.concatenate(
        ([old_start], active_old_resets, [old_end])
    )
    if not np.array_equal(old_boundaries, expected_old_boundaries):
        raise ValueError(
            "Incremental source must use the ordinary physical-reset spline layout"
        )
    if not np.isclose(
        old_boundaries[0], full_boundaries[0], rtol=0.0, atol=1.0e-12
    ) or not np.array_equal(full_boundaries[1:-1], active_old_resets):
        raise ValueError("Incremental fitting requires unchanged reset boundaries")
    if not np.isclose(old_boundaries[-1], old_end, rtol=0.0, atol=1.0e-12):
        raise ValueError("The old spline endpoint does not match its statistics")
    if not np.isclose(full_boundaries[-1], new_end, rtol=0.0, atol=1.0e-12):
        raise ValueError("The new spline endpoint does not match its statistics")

    old_segments = tuple(np.asarray(segment, dtype=float) for segment in old_layout["segments"])
    full_segments = tuple(
        np.asarray(segment, dtype=float) for segment in full_layout["segments"]
    )
    if len(old_segments) != len(full_segments):
        raise ValueError("Incremental fitting requires the same reset segment count")
    for old_segment, full_segment in zip(old_segments, full_segments):
        full_old_prefix = full_segment[full_segment <= old_end + 1.0e-12]
        if not np.array_equal(old_segment, full_old_prefix):
            raise ValueError("New knots were inserted before the incremental boundary")

    free_rate_mask = np.asarray(full_layout["flat_knots"], dtype=float) > (
        old_end + 1.0e-12
    )
    if overlap_knots:
        old_final = old_segments[-1]
        if overlap_knots > old_final.size:
            raise ValueError("Incremental overlap exceeds the historical knot count")
        final_start, _ = full_layout["slices"][-1]
        full_final = full_segments[-1]
        for historical_day in old_final[-overlap_knots:]:
            matching = np.flatnonzero(
                np.isclose(
                    full_final,
                    historical_day,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            if matching.size != 1:
                raise ValueError(
                    "Historical overlap knot is missing from the new spline layout"
                )
            free_rate_mask[final_start + int(matching[0])] = True
    if not np.any(free_rate_mask):
        raise ValueError("Incremental fitting found no new spline knots")

    result = dict(full_layout)
    result.update(
        {
            "incremental_boundary_day": old_end,
            "incremental_overlap_knots": overlap_knots,
            "incremental_anchor_day": None,
            "incremental_free_rate_mask": free_rate_mask,
        }
    )
    return result


def incremental_new_knot_mask(knot_layout, old_end):
    """Return the flattened rate-node mask for newly fitted edge knots."""
    if "incremental_free_rate_mask" in knot_layout:
        mask = np.asarray(knot_layout["incremental_free_rate_mask"], dtype=bool).copy()
        if not np.any(mask):
            raise ValueError("Incremental fitting found no free spline knots")
        return mask
    knots = np.asarray(knot_layout["flat_knots"], dtype=float)
    mask = knots > float(old_end) + 1.0e-12
    if not np.any(mask):
        raise ValueError("Incremental fitting found no new spline knots")
    return mask


def incremental_rate_initialization(old_layout, old_log_rates, knot_layout):
    """Map old rate nodes into the full new layout and initialize new nodes at the edge rate."""
    old_log_rates = np.asarray(old_log_rates, dtype=float).reshape(-1)
    old_flat_knots = np.asarray(old_layout["flat_knots"], dtype=float)
    if old_log_rates.size != old_flat_knots.size:
        raise ValueError("Old rate-node values do not match the old knot layout")
    old_segments = tuple(np.asarray(segment, dtype=float) for segment in old_layout["segments"])
    old_values = []
    offset = 0
    for segment in old_segments:
        old_values.append(old_log_rates[offset : offset + segment.size])
        offset += segment.size
    old_end = float(knot_layout["incremental_boundary_day"])
    new_values = []
    for old_segment, old_segment_values, new_segment in zip(
        old_segments,
        old_values,
        knot_layout["segments"],
    ):
        historical = new_segment <= old_end + 1.0e-12
        mapped = np.interp(
            new_segment[historical],
            old_segment,
            old_segment_values,
        )
        extension = np.full(
            np.count_nonzero(~historical),
            old_segment_values[-1],
            dtype=float,
        )
        new_values.append(np.concatenate((mapped, extension)))
    values = np.concatenate(new_values)
    if values.size != np.asarray(knot_layout["flat_knots"]).size:
        raise ValueError("Incremental rate initialization does not match knot layout")
    return values


def _natural_spline_coefficients(knots, values, jnp):
    """Return natural-cubic coefficients for values with a final knot axis."""
    knots = jnp.asarray(knots)
    values = jnp.asarray(values)
    knot_count = int(knots.size)
    flat_values = values.reshape((-1, knot_count))
    h = jnp.diff(knots)
    matrix = jnp.zeros((knot_count, knot_count), dtype=values.dtype)
    matrix = matrix.at[0, 0].set(1.0)
    matrix = matrix.at[-1, -1].set(1.0)
    for index in range(1, knot_count - 1):
        matrix = matrix.at[index, index - 1].set(h[index - 1])
        matrix = matrix.at[index, index].set(2.0 * (h[index - 1] + h[index]))
        matrix = matrix.at[index, index + 1].set(h[index])
    slopes = jnp.diff(flat_values, axis=1) / h[None, :]
    rhs = jnp.zeros_like(flat_values)
    rhs = rhs.at[:, 1:-1].set(6.0 * jnp.diff(slopes, axis=1))
    moments = jnp.linalg.solve(jnp.asarray(matrix), rhs.T).T
    coefficient_a = flat_values[:, :-1]
    coefficient_b = slopes - h[None, :] * (2.0 * moments[:, :-1] + moments[:, 1:]) / 6.0
    coefficient_c = moments[:, :-1] / 2.0
    coefficient_d = jnp.diff(moments, axis=1) / (6.0 * h[None, :])
    return coefficient_a, coefficient_b, coefficient_c, coefficient_d


def _natural_spline_eval(knots, coefficients, target, jnp):
    target = jnp.asarray(target)
    index = jnp.searchsorted(knots, target, side="right") - 1
    index = jnp.clip(index, 0, knots.size - 2)
    delta = target - knots[index]
    coefficient_a, coefficient_b, coefficient_c, coefficient_d = coefficients
    return (
        coefficient_a[:, index]
        + coefficient_b[:, index] * delta
        + coefficient_c[:, index] * delta**2
        + coefficient_d[:, index] * delta**3
    )


def natural_spline_values(knots, values, targets, jax, jnp):
    """Evaluate a natural cubic spline, preserving a leading batch axis."""
    del jax
    coefficients = _natural_spline_coefficients(knots, values, jnp)
    return _natural_spline_eval(jnp.asarray(knots), coefficients, targets, jnp)


def _integrate_piecewise_spline_profile(jax, jnp, targets, layout, log_rates):
    """Evaluate the integral and rate of reset-separated spline segments."""
    gauss_nodes = jnp.asarray(production.GAUSS_LEGENDRE_NODES)
    gauss_weights = jnp.asarray(production.GAUSS_LEGENDRE_WEIGHTS)
    boundaries = jnp.asarray(layout["boundaries"])
    target_values = jnp.asarray(targets)
    segment_data = []
    running = jnp.asarray(0.0, dtype=jnp.asarray(log_rates).dtype)

    for (left, right), (start, stop), knots_np in zip(
        zip(layout["boundaries"][:-1], layout["boundaries"][1:]),
        layout["slices"],
        layout["segments"],
    ):
        knots = jnp.asarray(knots_np)
        segment_rates = jnp.asarray(log_rates)[start:stop]
        coefficients = _natural_spline_coefficients(
            knots, segment_rates[None, :], jnp
        )
        left = jnp.asarray(left)
        right = jnp.asarray(right)

        def rate_at(time_value, segment_knots=knots, segment_coefficients=coefficients):
            log_rate = _natural_spline_eval(
                segment_knots, segment_coefficients, time_value, jnp
            )[0]
            return jnp.exp(log_rate)

        def integrate_interval(
            interval_left,
            interval_right,
            segment_rate=rate_at,
        ):
            midpoint = 0.5 * (interval_left + interval_right)
            half_width = 0.5 * (interval_right - interval_left)
            return half_width * jnp.sum(
                gauss_weights
                * segment_rate(midpoint + half_width * gauss_nodes)
            )

        total = integrate_interval(left, right)
        segment_data.append((left, right, running, total, rate_at, integrate_interval))
        running = running + total

    segment_index = jnp.searchsorted(boundaries, target_values, side="right") - 1
    segment_index = jnp.clip(segment_index, 0, len(segment_data) - 1)
    integrated = jnp.zeros_like(target_values)
    rates = jnp.zeros_like(target_values)
    for index, (left, right, prefix, _total, rate_at, integrate_interval) in enumerate(
        segment_data
    ):
        clipped = jnp.clip(target_values, left, right)
        candidate_integrated = prefix + jax.vmap(
            lambda target: integrate_interval(left, target)
        )(clipped)
        candidate_rates = jax.vmap(rate_at)(clipped)
        selected = segment_index == index
        integrated = jnp.where(selected, candidate_integrated, integrated)
        rates = jnp.where(selected, candidate_rates, rates)
    return integrated, rates


def _layout_for_data(data, knot_layout):
    if isinstance(knot_layout, dict):
        return knot_layout
    knots = np.asarray(knot_layout, dtype=float).reshape(-1)
    if knots.size < 3 or np.any(np.diff(knots) <= 0.0):
        raise ValueError("Spline knots must be strictly increasing")
    slices = ((0, knots.size),)
    return {
        "boundaries": np.asarray(
            [float(np.min(data["days"])), float(np.max(data["days"]))],
            dtype=float,
        ),
        "segments": (knots,),
        "slices": slices,
        "flat_knots": knots,
        "spacing_hours": None,
    }


def _integrate_spline_profile(jax, jnp, days, node_days, layout, log_rates):
    day_values = _integrate_piecewise_spline_profile(
        jax, jnp, days, layout, log_rates
    )
    node_values = _integrate_piecewise_spline_profile(
        jax, jnp, node_days, layout, log_rates
    )
    knot_values = _integrate_piecewise_spline_profile(
        jax, jnp, jnp.asarray(layout["flat_knots"]), layout, log_rates
    )
    return (
        day_values[0],
        day_values[1],
        node_values[0],
        node_values[1],
        knot_values[0],
        knot_values[1],
    )


def spline_trajectory_state(
    data,
    log_rate_nodes,
    knot_layout,
    initial_thickness,
    reset_initial_thickness,
    jax,
    jnp,
):
    """Evaluate the reset-aware thickness state for spline log-rate knots."""
    days = jnp.asarray(data["days"])
    node_days = jnp.asarray(data["node_days"])
    layout = _layout_for_data(data, knot_layout)
    knot_days = jnp.asarray(layout["flat_knots"])
    reset_days = np.asarray(production.reset_days_for_data(data), dtype=float)
    profiles = jnp.asarray(log_rate_nodes).reshape((-1, knot_days.size))
    integrated, rates, node_integrated, node_rates, knot_integrated, knot_rates = jax.vmap(
        lambda profile: _integrate_spline_profile(
            jax, jnp, days, node_days, layout, profile
        )
    )(profiles)
    reset_integrated = jax.vmap(
        lambda profile: _integrate_piecewise_spline_profile(
            jax, jnp, jnp.asarray(reset_days), layout, profile
        )[0]
    )(profiles)
    initial = jnp.exp(jnp.asarray(initial_thickness)).reshape(-1)
    reset_initial = jnp.exp(jnp.asarray(reset_initial_thickness)).reshape(-1)
    reset_initial = reset_initial.reshape((len(reset_days), -1))
    ice = initial[:, None] + integrated
    node_ice = initial[:, None] + node_integrated
    knot_ice = initial[:, None] + knot_integrated
    for reset_index, reset_day in enumerate(reset_days):
        post = days >= reset_day
        ice = jnp.where(
            post[None, :],
            reset_initial[reset_index, :, None]
            + integrated
            - reset_integrated[:, reset_index, None],
            ice,
        )
        node_post = node_days[None, :] >= reset_day
        node_ice = jnp.where(
            node_post,
            reset_initial[reset_index, :, None]
            + node_integrated
            - reset_integrated[:, reset_index, None],
            node_ice,
        )
        knot_post = knot_days[None, :] >= reset_day
        knot_ice = jnp.where(
            knot_post,
            reset_initial[reset_index, :, None]
            + knot_integrated
            - reset_integrated[:, reset_index, None],
            knot_ice,
        )
    return ice, rates, node_ice, node_rates, knot_ice, knot_rates


def spline_relative_state(data, values, cache, knot_layout, jax, jnp):
    """Evaluate the cached relative-response prediction for spline values."""
    ice, rates, node_ice, node_rates, knot_ice, knot_rates = spline_trajectory_state(
        data,
        values["np_log_rate_nodes"],
        knot_layout,
        values["initial_thickness_log_nm"],
        jnp.stack(
            [
                values[name]
                for name in production.reset_site_names(data)[0][1:]
            ],
            axis=0,
        ),
        jax,
        jnp,
    )
    grid = jnp.asarray(cache["grid_nm"])
    curves = jnp.asarray(cache["relative_log_response"])
    group_index = jnp.arange(len(data["groups"]), dtype=jnp.int32)
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1, dtype=jnp.int32)
    current_by_led = jnp.stack(
        [jnp.interp(ice[0], grid, curves[led]) for led in range(6)]
    )
    current = current_by_led[led_index, group_index]
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    baseline_thickness = ice[0, first_by_led][led_index]
    baseline_by_led = jnp.stack(
        [jnp.interp(baseline_thickness, grid, curves[led]) for led in range(6)]
    )
    baseline = baseline_by_led[led_index, group_index]
    prediction = (current - baseline)[None, :] + relative._led_log_offset_values(
        data,
        values,
        jnp,
    )
    return {
        "ice": ice,
        "rates": rates,
        "node_ice": node_ice,
        "node_rates": node_rates,
        "knot_ice": knot_ice,
        "knot_rates": knot_rates,
        "prediction": prediction,
    }


def _fit_shape_value(value, target_shape):
    value = np.asarray(value, dtype=float)
    while value.ndim > len(target_shape) and value.shape[0] == 1:
        value = value[0]
    if value.shape != tuple(target_shape):
        raise ValueError(f"Fit value shape {value.shape} does not match {target_shape}")
    return value.copy()


def _load_initial_and_current_values(
    data,
    current_fit,
    initial_fit,
    initial_statistics,
    knot_layout,
    fit_led_log_offsets=False,
):
    reference_values = relative.load_warm_values(
        data,
        initial_fit,
        initial_statistics,
        fit_led_log_offsets=fit_led_log_offsets,
    )
    with np.load(Path(current_fit), allow_pickle=False) as source:
        current_log_rates = np.asarray(source["np_log_rate_nodes"], dtype=float).reshape(-1)
        source_knot_days = (
            np.asarray(source["spline_knot_days"], dtype=float).reshape(-1)
            if "spline_knot_days" in source
            else None
        )
        source_knot_boundaries = (
            np.asarray(source["spline_knot_boundaries"], dtype=float).reshape(-1)
            if "spline_knot_boundaries" in source
            else None
        )
        source_knot_segment_sizes = (
            np.asarray(source["spline_knot_segment_sizes"], dtype=int).reshape(-1)
            if "spline_knot_segment_sizes" in source
            else None
        )
        current_values = {}
        for name, reference in reference_values.items():
            if name == "np_log_rate_nodes":
                continue
            if name in source:
                current_values[name] = _fit_shape_value(
                    source[name], np.asarray(reference).shape
                )
            else:
                current_values[name] = np.asarray(reference, dtype=float).copy()
    knot_layout = _layout_for_data(data, knot_layout)
    knot_days = np.asarray(knot_layout["flat_knots"], dtype=float)
    if source_knot_days is not None:
        if source_knot_days.size != current_log_rates.size:
            raise ValueError(
                "Current fit knot metadata does not match its rate-node count"
            )
        if source_knot_days.size == knot_days.size and np.allclose(
            source_knot_days, knot_days, rtol=0.0, atol=1.0e-12
        ):
            initialized_log_rates = current_log_rates
        else:
            if source_knot_boundaries is None or source_knot_segment_sizes is None:
                raise ValueError(
                    "Current fit knot layout does not match the requested spline layout"
                )
            target_boundaries = np.asarray(knot_layout["boundaries"], dtype=float)
            target_segments = [
                np.asarray(segment, dtype=float)
                for segment in knot_layout["segments"]
            ]
            if (
                source_knot_boundaries.size != target_boundaries.size
                or not np.allclose(
                    source_knot_boundaries[:-1],
                    target_boundaries[:-1],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or target_boundaries[-1]
                < source_knot_boundaries[-1] - 1.0e-12
                or source_knot_segment_sizes.size != len(target_segments)
                or int(np.sum(source_knot_segment_sizes)) != source_knot_days.size
            ):
                raise ValueError(
                    "Current fit reset-segment layout does not match the requested "
                    "spline layout"
                )
            initialized_segments = []
            source_start = 0
            for source_size, target_segment in zip(
                source_knot_segment_sizes, target_segments
            ):
                source_stop = source_start + int(source_size)
                initialized_segments.append(
                    np.interp(
                        target_segment,
                        source_knot_days[source_start:source_stop],
                        current_log_rates[source_start:source_stop],
                    )
                )
                source_start = source_stop
            initialized_log_rates = np.concatenate(initialized_segments)
    else:
        source_node_days = np.asarray(data["node_days"], dtype=float)
        if source_node_days.size != current_log_rates.size:
            raise ValueError(
                "Current fit rate-node count does not match the data node layout"
            )
        initialized_log_rates = np.interp(
            knot_days,
            source_node_days,
            current_log_rates,
        )
    current_values["np_log_rate_nodes"] = initialized_log_rates.reshape(
        np.asarray(reference_values["np_log_rate_nodes"]).shape[:-1]
        + (knot_days.size,)
    )
    rate_reference = relative.rate_prior_reference(current_values)
    return current_values, rate_reference


def _load_incremental_initial_and_current_values(
    data,
    current_fit,
    initial_fit,
    initial_statistics,
    old_data,
    old_layout,
    knot_layout,
    fit_led_log_offsets=False,
):
    """Load an old spline state and append fixed-rate initial values for new knots."""
    reference_values = relative.load_warm_values(
        data,
        initial_fit,
        initial_statistics,
        fit_led_log_offsets=fit_led_log_offsets,
    )
    with np.load(Path(current_fit), allow_pickle=False) as source:
        current_values = {}
        for name, reference in reference_values.items():
            if name == "np_log_rate_nodes":
                continue
            if name in source:
                current_values[name] = _fit_shape_value(
                    source[name], np.asarray(reference).shape
                )
            else:
                current_values[name] = np.asarray(reference, dtype=float).copy()
        if "np_log_rate_nodes" not in source:
            raise ValueError(f"Incremental source is missing np_log_rate_nodes: {current_fit}")
        old_log_rates = np.asarray(source["np_log_rate_nodes"], dtype=float).reshape(-1)

    old_flat_knots = np.asarray(old_layout["flat_knots"], dtype=float)
    if old_log_rates.size != old_flat_knots.size:
        raise ValueError(
            "Incremental source knot count does not match its historical layout"
        )
    old_end = float(np.max(np.asarray(old_data["days"], dtype=float)))
    new_knots = np.asarray(knot_layout["flat_knots"], dtype=float)
    if not np.any(np.isclose(new_knots, old_end, rtol=0.0, atol=1.0e-12)) or not np.any(
        new_knots > old_end + 1.0e-12
    ):
        raise ValueError("Incremental knot layout has no valid extension")
    combined_log_rates = incremental_rate_initialization(
        old_layout,
        old_log_rates,
        knot_layout,
    )
    target_shape = np.asarray(reference_values["np_log_rate_nodes"]).shape[:-1]
    current_values["np_log_rate_nodes"] = combined_log_rates.reshape(
        target_shape + (combined_log_rates.size,)
    )
    rate_reference = relative.rate_prior_reference(current_values)
    return current_values, rate_reference


def _data_residuals(data, values, prediction, jnp):
    observed = relative._observed_with_offset(data, values, jnp)
    scale = (
        jnp.sqrt(production.default_likelihood_weight(data["counts"]) / int(data["n_tiles"]))
        / production.LOG_NOISE_SIGMA
        * jnp.sqrt(jnp.asarray(data["counts"]))
    )
    return scale * (prediction - observed)


def build_spline_evaluator(
    data,
    initial_values,
    cache,
    knot_layout,
    jax,
    jnp,
    *,
    thickness_prior_log_sigma=production.INITIAL_THICKNESS_LOG_SIGMA,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Compile a spline residual/Jacobian reusable across same-shaped tiles."""
    from jax.flatten_util import ravel_pytree

    variable_site_names = relative.variable_sites(initial_values)
    template = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_site_names
    }
    _, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in template.items()}
    )
    layout_data = {
        "days": np.asarray(data["days"], dtype=float),
        "node_days": np.asarray(data["node_days"], dtype=float),
        "groups": np.asarray(data["groups"], dtype=int),
        "led": np.asarray(data["led"], dtype=int),
        "repeat": np.asarray(data["repeat"], dtype=int),
        "first_by_led": np.asarray(data["first_by_led"], dtype=int),
        "reset_day": float(data["reset_day"]),
        "reset_days": np.asarray(production.reset_days_for_data(data), dtype=float),
    }
    grid = jnp.asarray(cache["grid_nm"])

    def residual_function(
        scaled_values,
        flat_initial,
        flat_scales,
        rate_reference,
        counts,
        sums,
        sum_squared,
        curves,
        thickness_prior_centers,
    ):
        values = unravel(flat_initial + flat_scales * scaled_values)
        state = spline_relative_state(
            layout_data,
            values,
            {"grid_nm": grid, "relative_log_response": curves},
            knot_layout,
            jax,
            jnp,
        )
        _, offset_names = production.reset_site_names(layout_data)
        reset_mask = jnp.asarray(
            layout_data["days"]
            >= production.reset_days_for_data(layout_data)[:, None]
        )
        reset_offsets = jnp.stack(
            [values[name].reshape(-1) for name in offset_names], axis=0
        )
        observed = sums / counts + jnp.sum(
            reset_offsets[:, :, None] * reset_mask[:, None, :], axis=0
        )
        data_residual = (
            jnp.sqrt(1.0 / jnp.median(counts))
            / production.LOG_NOISE_SIGMA
            * jnp.sqrt(counts)
            * (state["prediction"] - observed)
        ).reshape(-1)
        prior_residual = relative._prior_residuals(
            layout_data,
            values,
            rate_reference,
            jnp,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
            thickness_prior_log_centers=thickness_prior_centers,
            thickness_prior_log_sigma=thickness_prior_log_sigma,
        )
        return jnp.concatenate((data_residual, prior_residual))

    return {
        "residual": jax.jit(residual_function),
        "jacobian": jax.jit(jax.jacfwd(residual_function)),
        "unravel": unravel,
    }


def fit_spline_reusable(
    data,
    cache,
    current_values,
    rate_reference,
    evaluator,
    jax,
    jnp,
    max_nfev,
    thickness_prior_centers=None,
    thickness_prior_log_sigma=production.INITIAL_THICKNESS_LOG_SIGMA,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    free_rate_mask=None,
):
    """Fit one tile using a spline evaluator compiled for reuse."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    variable_site_names = relative.variable_sites(current_values)
    variable_initial = {
        name: np.asarray(current_values[name], dtype=float).copy()
        for name in variable_site_names
    }
    fixed_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in current_values.items()
        if name not in variable_site_names
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(
        variable_initial,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    if isinstance(thickness_prior_log_sigma, dict):
        for name, sigma in thickness_prior_log_sigma.items():
            if name in parameter_scales:
                parameter_scales[name] = np.full_like(
                    np.asarray(parameter_scales[name], dtype=float),
                    float(sigma),
                )
    flat_scales, _ = ravel_pytree(
        {
            name: jnp.asarray(parameter_scales[name])
            for name in variable_site_names
        }
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)
    if free_rate_mask is None:
        free_mask_values = {
            name: np.ones_like(np.asarray(variable_initial[name]), dtype=bool)
            for name in variable_site_names
        }
    else:
        rate_mask = np.asarray(free_rate_mask, dtype=bool)
        if rate_mask.shape != np.asarray(variable_initial["np_log_rate_nodes"]).shape:
            raise ValueError("Incremental rate mask does not match spline rate nodes")
        free_mask_values = {
            name: np.zeros_like(np.asarray(variable_initial[name]), dtype=bool)
            for name in variable_site_names
        }
        free_mask_values["np_log_rate_nodes"] = rate_mask
    flat_free_mask, _ = ravel_pytree(
        {name: jnp.asarray(free_mask_values[name]) for name in variable_site_names}
    )
    free_indices = np.flatnonzero(np.asarray(flat_free_mask, dtype=bool))
    if free_indices.size == 0:
        raise ValueError("Spline fit has no free parameters")
    free_indices_jnp = jnp.asarray(free_indices, dtype=jnp.int32)
    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    curves = jnp.asarray(cache["relative_log_response"])
    rate_reference = jnp.asarray(rate_reference)
    if thickness_prior_centers is None:
        centers = None
    else:
        centers = {
            name: jnp.asarray(value)
            for name, value in thickness_prior_centers.items()
        }

    def expanded_scaled_values(scaled_values):
        full_scaled = jnp.zeros_like(jnp.asarray(flat_initial))
        return full_scaled.at[free_indices_jnp].set(scaled_values)

    def full_values(scaled_values):
        values = dict(fixed_values)
        values.update(
            unravel(
                jnp.asarray(flat_initial)
                + jnp.asarray(flat_scales) * expanded_scaled_values(scaled_values)
            )
        )
        return values

    def scipy_residual(scaled_values):
        return np.asarray(
            evaluator["residual"](
                expanded_scaled_values(jnp.asarray(scaled_values)),
                jnp.asarray(flat_initial),
                jnp.asarray(flat_scales),
                rate_reference,
                counts,
                sums,
                sum_squared,
                curves,
                centers,
            ),
            dtype=float,
        )

    def scipy_jacobian(scaled_values):
        full_jacobian = np.asarray(
            evaluator["jacobian"](
                expanded_scaled_values(jnp.asarray(scaled_values)),
                jnp.asarray(flat_initial),
                jnp.asarray(flat_scales),
                rate_reference,
                counts,
                sums,
                sum_squared,
                curves,
                centers,
            ),
            dtype=float,
        )
        return full_jacobian[:, free_indices]

    initial_scaled = np.zeros(free_indices.size, dtype=float)
    started = time.perf_counter()
    initial_residual = scipy_residual(initial_scaled)
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=int(max_nfev),
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    values = full_values(np.asarray(result.x, dtype=float))
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(free_indices.size),
        "fixed_parameter_count": int(flat_initial.size - free_indices.size),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
        "trajectory_rate_initialization_nm_day": float(
            np.exp(np.median(variable_initial["np_log_rate_nodes"]))
        ),
    }


def _stack_spline_value_arrays(values_list, name):
    """Stack production-shaped per-tile arrays on a leading tile axis."""
    if not values_list:
        raise ValueError("At least one value tree is required")
    if isinstance(values_list[0], dict):
        arrays = [np.asarray(values[name], dtype=float) for values in values_list]
    else:
        arrays = [np.asarray(values, dtype=float) for values in values_list]
    if name == "led_log_offsets":
        return np.concatenate(
            [array.reshape((-1, array.shape[-1])) for array in arrays], axis=0
        )
    return np.concatenate(
        [array.reshape((-1,) + array.shape[2:]) for array in arrays], axis=0
    )


def fit_spline_batch(
    data_list,
    cache_list,
    initial_values_list,
    rate_reference_list,
    knot_layout,
    jax,
    jnp,
    max_nfev,
    thickness_prior_centers_list=None,
    thickness_prior_log_sigma=production.INITIAL_THICKNESS_LOG_SIGMA,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Fit independent spline tiles with one vectorized JAX objective."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    if not data_list:
        raise ValueError("At least one tile is required for a batch fit")
    if len(data_list) != len(cache_list) or len(data_list) != len(initial_values_list):
        raise ValueError("Spline batch inputs must have matching tile counts")
    if len(data_list) != len(rate_reference_list):
        raise ValueError("Spline rate references must match the tile count")

    variable_site_names = relative.variable_sites(initial_values_list[0])
    if any(
        relative.variable_sites(values) != variable_site_names
        for values in initial_values_list[1:]
    ):
        raise ValueError("Spline batch tiles must expose the same variable sites")
    # Reuse the production validation for shared observation layouts.  The
    # spline evaluator keeps the tile axis separate so its Jacobian remains
    # block diagonal rather than becoming a dense global matrix.
    relative._stack_tile_data(data_list)
    grids = [np.asarray(cache["grid_nm"], dtype=float) for cache in cache_list]
    if not all(np.array_equal(grids[0], grid) for grid in grids[1:]):
        raise ValueError("Cannot batch spline response caches with different grids")
    tile_count = len(data_list)

    if thickness_prior_centers_list is None:
        thickness_prior_centers = None
    else:
        if len(thickness_prior_centers_list) != tile_count:
            raise ValueError("Thickness prior centers must match the tile count")
        if all(center is None for center in thickness_prior_centers_list):
            thickness_prior_centers = None
        elif any(center is None for center in thickness_prior_centers_list):
            raise ValueError("All spline batch tiles require thickness prior centers")
        else:
            center_names = tuple(thickness_prior_centers_list[0])
            if any(
                tuple(center) != center_names
                for center in thickness_prior_centers_list[1:]
            ):
                raise ValueError("Spline thickness-prior center sites differ")
            thickness_prior_centers = {
                name: np.stack(
                    [
                        np.asarray(center[name], dtype=float)
                        for center in thickness_prior_centers_list
                    ],
                    axis=0,
                )
                for name in center_names
            }

    evaluator = build_spline_evaluator(
        data_list[0],
        initial_values_list[0],
        cache_list[0],
        knot_layout,
        jax,
        jnp,
        thickness_prior_log_sigma=thickness_prior_log_sigma,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )

    flat_initial_list = []
    flat_scales_list = []
    for initial_values in initial_values_list:
        variable_initial = {
            name: np.asarray(initial_values[name], dtype=float).copy()
            for name in variable_site_names
        }
        parameter_scales = production.map_parameter_scale_values(
            variable_initial,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        )
        if isinstance(thickness_prior_log_sigma, dict):
            for name, sigma in thickness_prior_log_sigma.items():
                if name in parameter_scales:
                    parameter_scales[name] = np.full_like(
                        np.asarray(parameter_scales[name], dtype=float),
                        float(sigma),
                    )
        flat_initial, _ = ravel_pytree(
            {name: jnp.asarray(value) for name, value in variable_initial.items()}
        )
        flat_scales, _ = ravel_pytree(
            {name: jnp.asarray(parameter_scales[name]) for name in variable_site_names}
        )
        flat_initial_list.append(np.asarray(flat_initial, dtype=float))
        flat_scales_list.append(np.asarray(flat_scales, dtype=float))
    flat_initial = np.stack(flat_initial_list, axis=0)
    flat_scales = np.stack(flat_scales_list, axis=0)
    parameter_count = flat_initial.shape[1]
    rate_reference = np.stack(
        [np.asarray(reference, dtype=float) for reference in rate_reference_list],
        axis=0,
    )
    counts = np.stack(
        [np.asarray(item["counts"], dtype=float) for item in data_list], axis=0
    )
    sums = np.stack(
        [np.asarray(item["sums"], dtype=float) for item in data_list], axis=0
    )
    sum_squared = np.stack(
        [np.asarray(item["sum_squared"], dtype=float) for item in data_list], axis=0
    )
    curves = np.stack(
        [np.asarray(cache["relative_log_response"], dtype=float) for cache in cache_list],
        axis=0,
    )

    if thickness_prior_centers is None:
        def residual_for_tile(
            scaled_values,
            tile_initial,
            tile_scales,
            tile_rate_reference,
            tile_counts,
            tile_sums,
            tile_sum_squared,
            tile_curves,
        ):
            return evaluator["residual"](
                scaled_values,
                tile_initial,
                tile_scales,
                tile_rate_reference,
                tile_counts,
                tile_sums,
                tile_sum_squared,
                tile_curves,
                None,
            )

        residual_in_axes = (0,) * 8
        residual_arguments = (
            jnp.asarray(flat_initial),
            jnp.asarray(flat_scales),
            jnp.asarray(rate_reference),
            jnp.asarray(counts),
            jnp.asarray(sums),
            jnp.asarray(sum_squared),
            jnp.asarray(curves),
        )
    else:
        center_names = tuple(thickness_prior_centers)

        def residual_for_tile(
            scaled_values,
            tile_initial,
            tile_scales,
            tile_rate_reference,
            tile_counts,
            tile_sums,
            tile_sum_squared,
            tile_curves,
            tile_centers,
        ):
            centers = {
                name: tile_centers[index] for index, name in enumerate(center_names)
            }
            return evaluator["residual"](
                scaled_values,
                tile_initial,
                tile_scales,
                tile_rate_reference,
                tile_counts,
                tile_sums,
                tile_sum_squared,
                tile_curves,
                centers,
            )

        residual_in_axes = (0,) * 9
        residual_arguments = (
            jnp.asarray(flat_initial),
            jnp.asarray(flat_scales),
            jnp.asarray(rate_reference),
            jnp.asarray(counts),
            jnp.asarray(sums),
            jnp.asarray(sum_squared),
            jnp.asarray(curves),
            tuple(
                jnp.asarray(thickness_prior_centers[name])
                for name in center_names
            ),
        )

    residual_vmap = jax.vmap(residual_for_tile, in_axes=residual_in_axes)
    jacobian_vmap = jax.vmap(
        jax.jacfwd(residual_for_tile),
        in_axes=residual_in_axes,
    )
    residual_jit = jax.jit(residual_vmap)
    jacobian_jit = jax.jit(jacobian_vmap)

    def evaluate_residual_matrix(scaled_values):
        scaled_matrix = np.asarray(scaled_values, dtype=float).reshape(
            tile_count, parameter_count
        )
        return np.asarray(
            residual_jit(jnp.asarray(scaled_matrix), *residual_arguments),
            dtype=float,
        )

    def scipy_residual(scaled_values):
        return evaluate_residual_matrix(scaled_values).reshape(-1)

    def scipy_jacobian(scaled_values):
        from scipy.sparse import block_diag

        scaled_matrix = np.asarray(scaled_values, dtype=float).reshape(
            tile_count, parameter_count
        )
        jacobians = np.asarray(
            jacobian_jit(jnp.asarray(scaled_matrix), *residual_arguments),
            dtype=float,
        )
        return block_diag(
            [jacobians[index] for index in range(tile_count)], format="csr"
        )

    initial_scaled = np.zeros(tile_count * parameter_count, dtype=float)
    started = time.perf_counter()
    initial_matrix = evaluate_residual_matrix(initial_scaled)
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=int(max_nfev),
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    final_scaled = np.asarray(result.x, dtype=float).reshape(tile_count, parameter_count)
    final_matrix = evaluate_residual_matrix(result.x)
    tile_values = [
        evaluator["unravel"](
            jnp.asarray(flat_initial[index] + flat_scales[index] * final_scaled[index])
        )
        for index in range(tile_count)
    ]
    values = {
        name: jnp.asarray(_stack_spline_value_arrays(tile_values, name))
        for name in variable_site_names
    }
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(parameter_count),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_matrix**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_matrix**2)),
        "initial_residual_costs": 0.5 * np.sum(initial_matrix**2, axis=1),
        "residual_costs": 0.5 * np.sum(final_matrix**2, axis=1),
        "trajectory_rate_initialization_nm_day": np.exp(
            np.asarray(
                [
                    np.median(np.asarray(values_tree["np_log_rate_nodes"]))
                    for values_tree in initial_values_list
                ],
                dtype=float,
            )
        ),
        "tile_count": tile_count,
    }


def _fit_spline(data, cache, current_values, rate_reference, knot_days, jax, jnp, max_nfev):
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    variable_initial = {
        name: np.asarray(current_values[name], dtype=float).copy()
        for name in relative.VARIABLE_SITES
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(variable_initial)
    flat_scales, _ = ravel_pytree(
        {name: jnp.asarray(parameter_scales[name]) for name in relative.VARIABLE_SITES}
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)

    def full_values(scaled_values):
        return unravel(
            jnp.asarray(flat_initial) + jnp.asarray(flat_scales) * scaled_values
        )

    def residuals(scaled_values):
        values = full_values(scaled_values)
        state = spline_relative_state(data, values, cache, knot_days, jax, jnp)
        data_residual = _data_residuals(data, values, state["prediction"], jnp).reshape(-1)
        return jnp.concatenate(
            (data_residual, relative._prior_residuals(data, values, rate_reference, jnp))
        )

    residual_jit = jax.jit(residuals)
    jacobian_jit = jax.jit(jax.jacfwd(residuals))

    def scipy_residual(scaled_values):
        return np.asarray(residual_jit(jnp.asarray(scaled_values)), dtype=float)

    def scipy_jacobian(scaled_values):
        return np.asarray(jacobian_jit(jnp.asarray(scaled_values)), dtype=float)

    initial_scaled = np.zeros_like(flat_initial)
    started = time.perf_counter()
    initial_residual = scipy_residual(initial_scaled)
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=int(max_nfev),
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    values = full_values(np.asarray(result.x, dtype=float))
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(flat_initial.size),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
    }


def _state_metrics(data, values, state, jnp):
    observed = np.asarray(relative._observed_with_offset(data, values, jnp), dtype=float)
    prediction = np.asarray(state["prediction"], dtype=float)
    residual = prediction - observed
    valid = np.ones(residual.shape[1], dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    valid_residual = residual[0, valid]
    ice = np.asarray(state["ice"], dtype=float).reshape(-1)
    rates = np.asarray(state["rates"], dtype=float).reshape(-1)
    data_residual = np.asarray(_data_residuals(data, values, state["prediction"], jnp), dtype=float)
    return {
        "residual_rms_log": float(np.sqrt(np.mean(valid_residual**2))),
        "residual_max_abs_log": float(np.max(np.abs(valid_residual))),
        "weighted_data_cost": float(0.5 * np.sum(data_residual**2)),
        "initial_rate_nm_day": float(rates[0]),
        "final_rate_nm_day": float(rates[-1]),
        "mean_rate_nm_day": float(
            (ice[-1] - ice[0]) / (np.max(data["days"]) - np.min(data["days"]))
        ),
        "initial_thickness_nm": float(ice[0]),
        "final_thickness_nm": float(ice[-1]),
        "ice_min_nm": float(np.min(ice)),
        "ice_max_nm": float(np.max(ice)),
    }


def run_comparison(statistics_path, response_cache_path, current_fit, output_dir, spacing_hours=12.0, extra_days=(), max_nfev=10000):
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    statistics_path = Path(statistics_path)
    response_cache_path = Path(response_cache_path)
    current_fit = Path(current_fit)
    output_dir = Path(output_dir)
    data = production.load_tile_statistics(statistics_path)
    data["statistics_path"] = str(statistics_path)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    cache = relative.load_response_cache(response_cache_path, data, context)
    initial_fit, initial_statistics = relative_runner.cache_source_paths(response_cache_path)
    knot_days = build_spline_knot_days(data, spacing_hours, extra_days)
    current_values, rate_reference = _load_initial_and_current_values(
        data,
        current_fit,
        initial_fit,
        initial_statistics,
        knot_days,
    )

    with np.load(current_fit, allow_pickle=False) as source:
        current_fit_values = {
            name: _fit_shape_value(
                source[name],
                np.asarray(current_values[name]).shape,
            )
            for name in (
                "initial_thickness_log_nm",
                "reset_initial_thickness_log_nm",
                "post_reset_tile_log_offset",
            )
        }
        current_fit_values["np_log_rate_nodes"] = np.asarray(
            source["np_log_rate_nodes"], dtype=float
        )
        current_objective_history = np.asarray(
            source["map_objective_history"], dtype=float
        )
    current_state = relative.relative_state(data, current_fit_values, cache, jax, jnp)
    current_metrics = _state_metrics(data, current_fit_values, current_state, jnp)
    current_metrics["full_residual_cost"] = float(current_objective_history[-1])
    spline_values, diagnostics = _fit_spline(
        data,
        cache,
        current_values,
        rate_reference,
        knot_days,
        jax,
        jnp,
        max_nfev,
    )
    spline_state = spline_relative_state(data, spline_values, cache, knot_days, jax, jnp)
    spline_metrics = _state_metrics(data, spline_values, spline_state, jnp)
    spline_metrics["full_residual_cost"] = diagnostics["residual_cost"]

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "statistics": str(statistics_path.resolve()),
        "response_cache": str(response_cache_path.resolve()),
        "current_fit": str(current_fit.resolve()),
        "spacing_hours": float(spacing_hours),
        "extra_days": [float(value) for value in extra_days],
        "reset_day": float(data["reset_day"]),
        "observation_start_day": float(np.min(data["days"])),
        "observation_end_day": float(np.max(data["days"])),
        "current_node_count": int(np.asarray(current_fit_values["np_log_rate_nodes"]).size),
        "current_parameter_count": int(
            np.asarray(current_fit_values["np_log_rate_nodes"]).size + 3
        ),
        "spline_knot_count": int(knot_days.size),
        "spline_knot_days": knot_days.tolist(),
        "current": current_metrics,
        "spline": {**spline_metrics, **diagnostics},
        "delta_spline_minus_current": {
            name: float(spline_metrics[name] - current_metrics[name])
            for name in spline_metrics
        },
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "spline_fit.npz",
        knot_days=knot_days,
        np_log_rate_nodes=np.asarray(spline_values["np_log_rate_nodes"]),
        initial_thickness_log_nm=np.asarray(spline_values["initial_thickness_log_nm"]),
        reset_initial_thickness_log_nm=np.asarray(
            spline_values["reset_initial_thickness_log_nm"]
        ),
        post_reset_tile_log_offset=np.asarray(spline_values["post_reset_tile_log_offset"]),
        observation_days=np.asarray(data["days"]),
        ice_thickness=np.asarray(spline_state["ice"]),
        instantaneous_rate_nm_day=np.asarray(spline_state["rates"]),
    )
    with (output_dir / "trajectory_comparison.csv").open("w") as handle:
        handle.write(
            "days,current_ice_nm,spline_ice_nm,delta_ice_nm,current_rate_nm_day,"
            "spline_rate_nm_day,delta_rate_nm_day\n"
        )
        current_ice = np.asarray(current_state["ice"], dtype=float).reshape(-1)
        spline_ice = np.asarray(spline_state["ice"], dtype=float).reshape(-1)
        current_rates = np.asarray(current_state["rates"], dtype=float).reshape(-1)
        spline_rates = np.asarray(spline_state["rates"], dtype=float).reshape(-1)
        for index, day in enumerate(data["days"]):
            handle.write(
                f"{float(day):.17g},{current_ice[index]:.17g},{spline_ice[index]:.17g},"
                f"{spline_ice[index] - current_ice[index]:.17g},"
                f"{current_rates[index]:.17g},{spline_rates[index]:.17g},"
                f"{spline_rates[index] - current_rates[index]:.17g}\n"
            )
    summary = "\n".join(
        (
            "# Spline trajectory comparison",
            "",
            f"- Current trajectory nodes: `{comparison['current_node_count']}` (`{comparison['current_parameter_count']}` free parameters).",
            f"- Spline knots: `{comparison['spline_knot_count']}` at 12-hour spacing plus exact boundaries.",
            f"- Spline fit converged: `{diagnostics['converged']}` after `{diagnostics['nfev']}` evaluations.",
            f"- Current weighted data cost: `{current_metrics['weighted_data_cost']:.12g}`.",
            f"- Spline weighted data cost: `{spline_metrics['weighted_data_cost']:.12g}`.",
            f"- Current full residual cost: `{current_metrics['full_residual_cost']:.12g}`.",
            f"- Spline full residual cost: `{diagnostics['residual_cost']:.12g}`.",
            f"- Current residual RMS: `{current_metrics['residual_rms_log']:.12g}` log.",
            f"- Spline residual RMS: `{spline_metrics['residual_rms_log']:.12g}` log.",
            f"- Mean-rate change: `{spline_metrics['mean_rate_nm_day'] - current_metrics['mean_rate_nm_day']:.12g}` nm/day.",
            f"- Final-thickness change: `{spline_metrics['final_thickness_nm'] - current_metrics['final_thickness_nm']:.12g}` nm.",
            "",
        )
    )
    (output_dir / "comparison.md").write_text(summary)
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--response-cache", type=Path, required=True)
    parser.add_argument("--current-fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spacing-hours", type=float, default=12.0)
    parser.add_argument("--extra-knot", type=float, action="append", default=[])
    parser.add_argument("--max-nfev", type=int, default=production.PRODUCTION_MAP_STEPS)
    return parser.parse_args()


def main():
    args = parse_args()
    comparison = run_comparison(
        args.statistics,
        args.response_cache,
        args.current_fit,
        args.output_dir,
        spacing_hours=args.spacing_hours,
        extra_days=args.extra_knot,
        max_nfev=args.max_nfev,
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
