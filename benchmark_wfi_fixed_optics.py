"""Benchmark fixed-optics and relative-response WFI tile fits.

This is a diagnostic benchmark.  It does not modify the production model or
production archives.  The reference archive supplies the optical MAP values
and a warm trajectory; the target statistics are then fit with either the
full production parameter set, fixed optical parameters with the production
TMM evaluator, or a precomputed relative-response curve.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import fit_wfi_tmm_production as production


OPTICAL_SITES = (
    "ar_tile_raw",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
    "incidence_angle_deg",
)
VARIABLE_SITES = (
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "post_reset_tile_log_offset",
    "np_log_rate_nodes",
)


def _strip_leading_axes(value, target_shape):
    value = np.asarray(value, dtype=float)
    while value.ndim > len(target_shape) and value.shape[0] == 1:
        value = value[0]
    if value.shape != tuple(target_shape):
        raise ValueError(f"Reference shape {value.shape} does not match {target_shape}")
    return value.copy()


def _load_reference_values(data, reference_fit, reference_statistics):
    """Load a previous tile state and interpolate its rate nodes to the target."""
    values = production.initial_values_for_data(data, constant_rate=False)
    reference_fit = Path(reference_fit)
    with np.load(reference_fit, allow_pickle=False) as source:
        for name in (
            "ar_tile_raw",
            "initial_thickness_log_nm",
            "reset_initial_thickness_log_nm",
            "post_reset_tile_log_offset",
            "absolute_hgcdte_n_log_scale",
            "absolute_hgcdte_k_log_scale",
        ):
            if name in source:
                values[name] = _strip_leading_axes(source[name], np.asarray(values[name]).shape)
        if "incidence_angle_deg" in source and "incidence_angle_deg" in values:
            values["incidence_angle_deg"] = float(
                np.asarray(source["incidence_angle_deg"]).reshape(-1)[0]
            )
        old_rates = np.asarray(source["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)

    old_data = production.load_tile_statistics(reference_statistics)
    if old_rates.size != old_data["node_days"].size:
        raise ValueError("Reference rate nodes do not match reference statistics")
    target_rate_nodes = np.interp(
        data["node_days"], old_data["node_days"], old_rates
    )
    values["np_log_rate_nodes"] = np.log(target_rate_nodes).reshape(
        np.asarray(values["np_log_rate_nodes"]).shape
    )
    return values


def _load_median_optics(reference_fit_root, target_values):
    arrays = {name: [] for name in OPTICAL_SITES}
    for fit_path in sorted(Path(reference_fit_root).glob("tile_*/posterior_samples.npz")):
        with np.load(fit_path, allow_pickle=False) as source:
            for name in OPTICAL_SITES:
                if name not in source:
                    raise ValueError(f"{name} is missing from {fit_path}")
                arrays[name].append(np.asarray(source[name], dtype=float).reshape(-1))
    if not arrays["ar_tile_raw"]:
        raise FileNotFoundError(f"No posterior archives found under {reference_fit_root}")
    median = {}
    for name, values in arrays.items():
        value = np.median(np.asarray(values), axis=0)
        target_shape = np.asarray(target_values[name]).shape
        median[name] = value.reshape(target_shape)
    return median


def _build_production_model(data, context, values):
    rate_reference = np.asarray(values["np_log_rate_nodes"], dtype=float)
    return production.build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_reference,
        entropy_weight=0.1,
        rate_prior_log_mean=rate_reference,
        constant_rate=False,
        fit_incidence_angle=True,
    )


def _observed_with_offset(data, values, jnp):
    observed = jnp.asarray(data["sums"]) / jnp.asarray(data["counts"])
    reset_mask = jnp.asarray(
        np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    )
    return observed + (
        jnp.asarray(values["post_reset_tile_log_offset"]).reshape(-1, 1)
        * reset_mask[None, :]
    )


def _ice_prior_residuals(values, data, rate_reference, jnp):
    residuals = [
        (
            jnp.asarray(values["initial_thickness_log_nm"])
            - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
        ).reshape(-1)
        / production.INITIAL_THICKNESS_LOG_SIGMA,
        (
            jnp.asarray(values["reset_initial_thickness_log_nm"])
            - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
        ).reshape(-1)
        / production.INITIAL_THICKNESS_LOG_SIGMA,
        (
            jnp.asarray(values["np_log_rate_nodes"])
            - jnp.asarray(rate_reference)
        ).reshape(-1)
        / production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        jnp.asarray(values["post_reset_tile_log_offset"]).reshape(-1)
        / production.POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
    ]
    likelihood_scale = (
        0.5
        * production.default_likelihood_weight(data["counts"])
        / int(data["n_tiles"])
        * float(np.sum(data["sum_squared"]))
        / production.LOG_NOISE_SIGMA**2
    )
    entropy = production._signed_entropy_residual(
        jnp.asarray(values["np_log_rate_nodes"]),
        jnp.asarray(rate_reference),
        jnp,
    ).reshape(-1)
    residuals.append(
        entropy
        * jnp.sqrt(
            likelihood_scale * 0.1 / entropy.size
        )
    )
    return jnp.concatenate(residuals)


def _scaled_data_residual(data, prediction, values, jnp):
    observed = _observed_with_offset(data, values, jnp)
    weight = production.default_likelihood_weight(data["counts"])
    return (
        jnp.sqrt(weight / int(data["n_tiles"]))
        / production.LOG_NOISE_SIGMA
        * jnp.sqrt(jnp.asarray(data["counts"]))
        * (prediction - observed)
    ).reshape(-1)


def _fixed_tmm_residual(model, data, values, rate_reference, jnp):
    state = model.production_state(values)
    return jnp.concatenate(
        (
            _scaled_data_residual(data, state["prediction"], values, jnp),
            _ice_prior_residuals(values, data, rate_reference, jnp),
        )
    )


def _reduced_least_squares(model_residual, initial_values, variable_names, jax, jnp):
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    variable_initial = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_names
    }
    fixed_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in initial_values.items()
        if name not in variable_names
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(initial_values)
    scale_pytree = {
        name: jnp.asarray(parameter_scales[name])
        for name in variable_names
    }
    flat_scales, _ = ravel_pytree(scale_pytree)
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)

    def full_values(scaled_values):
        values = dict(fixed_values)
        values.update(
            unravel(
                jnp.asarray(flat_initial) + jnp.asarray(flat_scales) * scaled_values
            )
        )
        return values

    def residuals(scaled_values):
        return model_residual(full_values(scaled_values))

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
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    optimized = full_values(np.asarray(result.x, dtype=float))
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    return optimized, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(flat_initial.size),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
    }


def _build_relative_lookup(data, context, values, grid):
    jax, jnp, _, _ = production.require_numpyro()
    selected = []
    for led in range(1, 7):
        indices = np.flatnonzero(np.asarray(data["led"]) == led)
        if indices.size < 2:
            raise ValueError(f"LED {led} has no non-reference group")
        selected.extend((int(indices[0]), int(indices[1])))
    selected = np.asarray(selected, dtype=int)
    mini = dict(data)
    for name in ("counts", "sums", "sum_squared"):
        mini[name] = np.asarray(data[name])[:, selected]
    for name in ("groups", "led", "repeat", "program", "days", "exposure_times"):
        mini[name] = np.asarray(data[name])[selected]
    mini["first_by_led"] = np.arange(0, 12, 2, dtype=int)
    mini["repeat_values"] = np.asarray(sorted(set(mini["repeat"].tolist())), dtype=int)
    mini["repeat_index"] = np.zeros(12, dtype=int)

    predictor = production.build_prediction_function(mini, context)
    ar_raw = np.asarray(values["ar_tile_raw"], dtype=float).reshape(-1)
    ar_physical = np.asarray(
        production._physical_ar_parameters(
            jnp.asarray(ar_raw[None, :]), jax, jnp
        )
    )
    hgcdte = np.asarray(
        [
            float(np.asarray(values["absolute_hgcdte_n_log_scale"]).reshape(-1)[0]),
            float(np.asarray(values["absolute_hgcdte_k_log_scale"]).reshape(-1)[0]),
        ]
    )
    row_count = 6 * grid.size
    ice = np.zeros((row_count, 12), dtype=float)
    for led_index in range(6):
        start = led_index * grid.size
        stop = (led_index + 1) * grid.size
        ice[start:stop, 2 * led_index + 1] = grid
    ar_rows = np.broadcast_to(ar_physical, (row_count, ar_physical.shape[-1]))
    hg_rows = np.broadcast_to(hgcdte, (row_count, 2))
    started = time.perf_counter()
    prediction = np.asarray(
        jax.jit(predictor)(
            jnp.asarray(ice),
            jnp.asarray(ar_rows),
            jnp.asarray(hg_rows),
            float(np.asarray(values["incidence_angle_deg"]).reshape(-1)[0]),
        )
    )
    elapsed = time.perf_counter() - started
    baseline_offsets = np.asarray(
        [
            np.average(
                np.asarray(data["sums"])[:, group]
                / np.asarray(data["counts"])[:, group],
                weights=np.asarray(data["counts"])[:, group],
            )
            for group in np.asarray(data["first_by_led"], dtype=int)
        ]
    )
    curves = np.empty((6, grid.size), dtype=float)
    for led_index in range(6):
        start = led_index * grid.size
        stop = (led_index + 1) * grid.size
        curves[led_index] = prediction[start:stop, 2 * led_index + 1] - baseline_offsets[led_index]
    return grid, curves, baseline_offsets, elapsed


def _relative_state(data, values, grid, curves, baseline_offsets, jax, jnp):
    ice, rates, node_ice, node_rates = production._trajectory_state(
        data,
        values["np_log_rate_nodes"],
        values["initial_thickness_log_nm"],
        values["reset_initial_thickness_log_nm"],
        jax,
        jnp,
    )
    group_index = jnp.arange(len(data["groups"]), dtype=jnp.int32)
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1, dtype=jnp.int32)
    grid_jnp = jnp.asarray(grid)
    curve_jnp = jnp.asarray(curves)
    current_by_led = jnp.stack(
        [jnp.interp(ice[0], grid_jnp, curve_jnp[led]) for led in range(6)]
    )
    current = current_by_led[led_index, group_index]
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    baseline_thickness = ice[0, first_by_led][led_index]
    baseline_by_led = jnp.stack(
        [jnp.interp(baseline_thickness, grid_jnp, curve_jnp[led]) for led in range(6)]
    )
    baseline = baseline_by_led[led_index, group_index]
    prediction = (
        current
        - baseline
        + jnp.asarray(baseline_offsets)[led_index]
    )[None, :]
    return {
        "ice": ice,
        "rates": rates,
        "node_ice": node_ice,
        "node_rates": node_rates,
        "prediction": prediction,
    }


def _metrics(data, values, state, diagnostics, mode, tile_id, lookup_seconds=0.0):
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(data["counts"], dtype=float)
    observed += np.asarray(values["post_reset_tile_log_offset"]).reshape(-1, 1) * (
        np.asarray(data["days"])[None, :] >= float(data["reset_day"])
    )
    prediction = np.asarray(state["prediction"], dtype=float)
    residual = observed - prediction
    valid = np.ones(residual.shape[1], dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    values_residual = residual[0, valid]
    rates = np.asarray(state["rates"], dtype=float).reshape(-1)
    ice = np.asarray(state["ice"], dtype=float).reshape(-1)
    days = np.asarray(data["days"], dtype=float)
    result = {
        "mode": mode,
        "tile_id": int(tile_id),
        "residual_rms_log": float(np.sqrt(np.mean(values_residual**2))),
        "residual_max_abs_log": float(np.max(np.abs(values_residual))),
        "initial_rate_nm_day": float(rates[0]),
        "final_rate_nm_day": float(rates[-1]),
        "mean_rate_nm_day": float((ice[-1] - ice[0]) / (np.max(days) - np.min(days))),
        "lookup_seconds": float(lookup_seconds),
        **diagnostics,
    }
    return result


def run_benchmark(args):
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data = production.load_tile_statistics(args.statistics)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    reference_values = _load_reference_values(
        data, args.reference_fit, args.reference_statistics
    )
    shared_optics = _load_median_optics(args.reference_fit_dir, reference_values)
    rate_reference = np.asarray(reference_values["np_log_rate_nodes"], dtype=float)
    model = _build_production_model(data, context, reference_values)
    tile_id = int(data["tile_ids"][0])
    results = []

    started = time.perf_counter()
    full_values, full_diagnostics = production.run_least_squares_map_optimization(
        model, reference_values, jax
    )
    full_elapsed = time.perf_counter() - started
    full_state = model.production_state(full_values)
    full_diagnostics = {
        "converged": bool(full_diagnostics["converged"]),
        "message": full_diagnostics["message"],
        "nfev": int(full_diagnostics["function_evaluations"]),
        "parameter_count": int(
            sum(np.asarray(value).size for value in reference_values.values())
        ),
        "compile_seconds": None,
        "total_seconds": float(full_elapsed),
        "residual_cost": float(full_diagnostics["objective"]),
        "initial_residual_cost": float(full_diagnostics["initial_objective"]),
    }
    results.append(_metrics(data, full_values, full_state, full_diagnostics, "full_tmm", tile_id))

    fixed_values = dict(reference_values)
    fixed_values.update(shared_optics if args.shared_optics else {})
    fixed_model = _build_production_model(data, context, fixed_values)
    fixed_rate_reference = np.asarray(fixed_values["np_log_rate_nodes"], dtype=float)
    fixed_values, fixed_diagnostics = _reduced_least_squares(
        lambda values: _fixed_tmm_residual(
            fixed_model, data, values, fixed_rate_reference, jnp
        ),
        fixed_values,
        VARIABLE_SITES,
        jax,
        jnp,
    )
    fixed_state = fixed_model.production_state(fixed_values)
    results.append(
        _metrics(
            data,
            fixed_values,
            fixed_state,
            fixed_diagnostics,
            "fixed_optics_tmm" if not args.shared_optics else "shared_optics_tmm",
            tile_id,
        )
    )

    grid = np.linspace(0.0, float(args.max_thickness), int(args.max_thickness / args.grid_step) + 1)
    lookup_started = time.perf_counter()
    grid, curves, baseline_offsets, lookup_seconds = _build_relative_lookup(
        data, context, fixed_values, grid
    )
    lookup_values = dict(fixed_values)

    def relative_residual(values):
        state = _relative_state(
            data, values, grid, curves, baseline_offsets, jax, jnp
        )
        return jnp.concatenate(
            (
                _scaled_data_residual(data, state["prediction"], values, jnp),
                _ice_prior_residuals(values, data, fixed_rate_reference, jnp),
            )
        )

    relative_values, relative_diagnostics = _reduced_least_squares(
        relative_residual,
        lookup_values,
        VARIABLE_SITES,
        jax,
        jnp,
    )
    relative_state = _relative_state(
        data, relative_values, grid, curves, baseline_offsets, jax, jnp
    )
    relative_diagnostics["lookup_seconds"] = float(lookup_seconds)
    results.append(
        _metrics(
            data,
            relative_values,
            relative_state,
            relative_diagnostics,
            "relative_response",
            tile_id,
            lookup_seconds=lookup_seconds,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"tile_{tile_id:03d}_fixed_optics_benchmark.json"
    payload = {
        "statistics": str(Path(args.statistics).resolve()),
        "reference_fit": str(Path(args.reference_fit).resolve()),
        "reference_statistics": str(Path(args.reference_statistics).resolve()),
        "shared_optics": bool(args.shared_optics),
        "grid_step_nm": float(args.grid_step),
        "max_thickness_nm": float(args.max_thickness),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--reference-fit", type=Path, required=True)
    parser.add_argument("--reference-statistics", type=Path, required=True)
    parser.add_argument("--reference-fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-optics", action="store_true")
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--max-thickness", type=float, default=260.0)
    args = parser.parse_args()
    if args.grid_step <= 0.0 or args.max_thickness <= 0.0:
        raise ValueError("grid-step and max-thickness must be positive")
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
