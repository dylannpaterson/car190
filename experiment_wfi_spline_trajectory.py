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


def _integrate_spline_profile(jax, jnp, days, knot_days, integration_times, log_rates):
    gauss_nodes = jnp.asarray(production.GAUSS_LEGENDRE_NODES)
    gauss_weights = jnp.asarray(production.GAUSS_LEGENDRE_WEIGHTS)
    coefficients = _natural_spline_coefficients(knot_days, log_rates[None, :], jnp)

    def instantaneous_rate(time_value):
        log_rate = _natural_spline_eval(
            jnp.asarray(knot_days), coefficients, time_value, jnp
        )[0]
        return jnp.exp(log_rate)

    def integrate_interval(left, right):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        return half_width * jnp.sum(
            gauss_weights
            * instantaneous_rate(midpoint + half_width * gauss_nodes)
        )

    interval_integrals = jax.vmap(integrate_interval)(
        integration_times[:-1], integration_times[1:]
    )
    cumulative = jnp.concatenate((jnp.zeros(1), jnp.cumsum(interval_integrals)))

    def integrated_at(target):
        index = jnp.searchsorted(integration_times, target, side="right") - 1
        index = jnp.clip(index, 0, integration_times.size - 2)
        return cumulative[index] + integrate_interval(integration_times[index], target)

    return (
        jax.vmap(integrated_at)(jnp.asarray(days)),
        jax.vmap(instantaneous_rate)(jnp.asarray(days)),
        jax.vmap(integrated_at)(jnp.asarray(knot_days)),
        jax.vmap(instantaneous_rate)(jnp.asarray(knot_days)),
    )


def spline_trajectory_state(data, log_rate_nodes, knot_days, initial_thickness, reset_initial_thickness, jax, jnp):
    """Evaluate the reset-aware thickness state for spline log-rate knots."""
    days = jnp.asarray(data["days"])
    knot_days_np = np.asarray(knot_days, dtype=float)
    knot_days = jnp.asarray(knot_days_np)
    reset_day = float(data["reset_day"])
    integration_times = np.unique(
        np.concatenate(
            (
                [float(np.min(data["days"]))],
                knot_days_np,
                [float(np.max(data["days"]))],
                [reset_day],
            )
        )
    )
    profiles = jnp.asarray(log_rate_nodes).reshape((-1, knot_days.size))
    integrated, rates, node_integrated, node_rates = jax.vmap(
        lambda profile: _integrate_spline_profile(
            jax,
            jnp,
            days,
            knot_days,
            jnp.asarray(integration_times),
            profile,
        )
    )(profiles)
    reset_integrated = jax.vmap(
        lambda profile: _integrate_spline_profile(
            jax,
            jnp,
            jnp.asarray([reset_day]),
            knot_days,
            jnp.asarray(integration_times),
            profile,
        )[0][0]
    )(profiles)
    initial = jnp.exp(jnp.asarray(initial_thickness)).reshape(-1)
    reset_initial = jnp.exp(jnp.asarray(reset_initial_thickness)).reshape(-1)
    post = days[None, :] >= reset_day
    integrated = jnp.where(post, integrated - reset_integrated[:, None], integrated)
    node_post = knot_days[None, :] >= reset_day
    node_integrated = jnp.where(
        node_post,
        node_integrated - reset_integrated[:, None],
        node_integrated,
    )
    ice = jnp.where(post, reset_initial[:, None], initial[:, None]) + integrated
    node_ice = jnp.where(
        node_post,
        reset_initial[:, None],
        initial[:, None],
    ) + node_integrated
    return ice, rates, node_ice, node_rates


def spline_relative_state(data, values, cache, knot_days, jax, jnp):
    """Evaluate the cached relative-response prediction for spline values."""
    ice, rates, node_ice, node_rates = spline_trajectory_state(
        data,
        values["np_log_rate_nodes"],
        knot_days,
        values["initial_thickness_log_nm"],
        values["reset_initial_thickness_log_nm"],
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
    return {
        "ice": ice,
        "rates": rates,
        "node_ice": node_ice,
        "node_rates": node_rates,
        "prediction": (current - baseline)[None, :],
    }


def _fit_shape_value(value, target_shape):
    value = np.asarray(value, dtype=float)
    while value.ndim > len(target_shape) and value.shape[0] == 1:
        value = value[0]
    if value.shape != tuple(target_shape):
        raise ValueError(f"Fit value shape {value.shape} does not match {target_shape}")
    return value.copy()


def _load_initial_and_current_values(data, current_fit, initial_fit, initial_statistics, knot_days):
    reference_values = relative.load_warm_values(
        data,
        initial_fit,
        initial_statistics,
    )
    with np.load(Path(current_fit), allow_pickle=False) as source:
        current_log_rates = np.asarray(source["np_log_rate_nodes"], dtype=float).reshape(-1)
        current_values = {
            name: _fit_shape_value(source[name], np.asarray(reference_values[name]).shape)
            for name in (
                "initial_thickness_log_nm",
                "reset_initial_thickness_log_nm",
                "post_reset_tile_log_offset",
            )
        }
    current_values["np_log_rate_nodes"] = np.interp(
        knot_days,
        np.asarray(data["node_days"], dtype=float),
        current_log_rates,
    ).reshape(np.asarray(reference_values["np_log_rate_nodes"]).shape[:-1] + (knot_days.size,))
    reference_log_rates = np.asarray(reference_values["np_log_rate_nodes"], dtype=float).reshape(-1)
    rate_reference = np.interp(
        knot_days,
        np.asarray(data["node_days"], dtype=float),
        reference_log_rates,
    ).reshape(np.asarray(current_values["np_log_rate_nodes"]).shape)
    return current_values, rate_reference


def _data_residuals(data, values, prediction, jnp):
    observed = relative._observed_with_offset(data, values, jnp)
    scale = (
        jnp.sqrt(production.default_likelihood_weight(data["counts"]) / int(data["n_tiles"]))
        / production.LOG_NOISE_SIGMA
        * jnp.sqrt(jnp.asarray(data["counts"]))
    )
    return scale * (prediction - observed)


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
