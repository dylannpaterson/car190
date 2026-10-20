"""Fit one WFI tile with another detector's rate trajectory fixed.

This is a diagnostic only. The target tile's non-parametric ice-rate nodes
are fixed to the source detector's median trajectory while optical and phase
nuisance parameters are reoptimized with the production TMM residual.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import fit_wfi_tmm_production as production
from plot_wfi_tile_rate_counterfactual import load_source_median_node_rates
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    detector_output_root,
    positive_integer,
    tile_statistics_path,
)


DEFAULT_TARGET_DETECTOR = "WFI01"
DEFAULT_SOURCE_DETECTOR = "WFI10"
DEFAULT_TILE_ID = 0
PRODUCTION_MAXENT_WEIGHT = 0.1
VARIABLE_SITES = (
    "ar_tile_raw",
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "post_reset_tile_log_offset",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
    "incidence_angle_deg",
)


def _scalar(values, name):
    return float(np.asarray(values[name], dtype=float).reshape(-1)[0])


def _fit_summary_objective(path):
    if not path.is_file():
        return None
    text = path.read_text()
    match = re.search(r"^- Objective: `([^`]+)`\.$", text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def _source_mean_rate(source_output):
    path = Path(source_output) / "tile_rate_posterior.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No source tile-rate rows found in {path}")
    return float(np.median([float(row["mean_rate_nm_day_median"]) for row in rows])), len(rows)


def _load_target_values(target_fit_dir, data, source_rates):
    values = production.initial_values_for_data(data, constant_rate=False)
    values = production.load_map_warm_start(
        target_fit_dir / "posterior_samples.npz",
        values,
        data,
    )
    values["np_log_rate_nodes"] = np.log(source_rates).reshape(
        np.asarray(values["np_log_rate_nodes"]).shape
    )
    return values


def _bounded_reduced_least_squares(model_residual, initial_values, variable_names, jax, jnp):
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
    flat_scales, _ = ravel_pytree(
        {
            name: jnp.asarray(parameter_scales[name])
            for name in variable_names
        }
    )
    lower = {}
    upper = {}
    for name, value in variable_initial.items():
        lower[name] = np.full(np.asarray(value).shape, -np.inf)
        upper[name] = np.full(np.asarray(value).shape, np.inf)
        if name == "incidence_angle_deg":
            lower[name] = np.zeros(np.asarray(value).shape)
            upper[name] = np.full(
                np.asarray(value).shape,
                production.INCIDENCE_ANGLE_MAX_DEG,
            )
    flat_lower, _ = ravel_pytree(
        {name: jnp.asarray(lower[name]) for name in variable_names}
    )
    flat_upper, _ = ravel_pytree(
        {name: jnp.asarray(upper[name]) for name in variable_names}
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)
    scaled_lower = (np.asarray(flat_lower, dtype=float) - flat_initial) / flat_scales
    scaled_upper = (np.asarray(flat_upper, dtype=float) - flat_initial) / flat_scales

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
    initial_residual = scipy_residual(initial_scaled)
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        bounds=(scaled_lower, scaled_upper),
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=0,
    )
    optimized = full_values(np.asarray(result.x, dtype=float))
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    return optimized, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
    }


def _base_observed(data):
    return (
        np.asarray(data["sums"], dtype=float)
        / np.asarray(data["counts"], dtype=float)
    ).reshape(-1)


def _raw_prediction(data, prediction, values):
    post_mask = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    offset = _scalar(values, "post_reset_tile_log_offset")
    return np.asarray(prediction, dtype=float).reshape(-1) - offset * post_mask


def _metrics(data, raw_observed, raw_prediction):
    residual = np.asarray(raw_observed, dtype=float) - np.asarray(raw_prediction, dtype=float)
    valid = np.ones(residual.size, dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    counts = np.asarray(data["counts"], dtype=float).reshape(-1)
    weights = counts[valid]
    log_residual = residual[valid]
    percent_residual = 100.0 * np.expm1(log_residual)
    likelihood_weight = production.default_likelihood_weight(data["counts"])
    return {
        "residual_rms_log": float(np.sqrt(np.mean(log_residual**2))),
        "weighted_residual_rms_log": float(
            np.sqrt(np.sum(weights * log_residual**2) / np.sum(weights))
        ),
        "weighted_residual_rms_percent": float(
            np.sqrt(np.sum(weights * percent_residual**2) / np.sum(weights))
        ),
        "max_abs_residual_percent": float(np.max(np.abs(percent_residual))),
        "data_objective": float(
            0.5
            * likelihood_weight
            * np.sum(weights * log_residual**2)
            / production.LOG_NOISE_SIGMA**2
        ),
    }, residual


def _write_csv(path, data, raw_observed, free_prediction, forced_prediction):
    fields = (
        "group",
        "days",
        "led",
        "repeat",
        "observed_percent",
        "free_rate_model_percent",
        "forced_rate_model_percent",
        "free_rate_residual_percent",
        "forced_rate_residual_percent",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, group in enumerate(data["groups"]):
            writer.writerow(
                {
                    "group": int(group),
                    "days": float(data["days"][index]),
                    "led": int(data["led"][index]),
                    "repeat": int(data["repeat"][index]),
                    "observed_percent": 100.0 * np.expm1(raw_observed[index]),
                    "free_rate_model_percent": 100.0 * np.expm1(free_prediction[index]),
                    "forced_rate_model_percent": 100.0 * np.expm1(forced_prediction[index]),
                    "free_rate_residual_percent": 100.0 * np.expm1(
                        raw_observed[index] - free_prediction[index]
                    ),
                    "forced_rate_residual_percent": 100.0 * np.expm1(
                        raw_observed[index] - forced_prediction[index]
                    ),
                }
            )


def _plot(path, data, raw_observed, free_prediction, forced_prediction, source_rate):
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    observed_percent = 100.0 * np.expm1(raw_observed)
    free_percent = 100.0 * np.expm1(free_prediction)
    forced_percent = 100.0 * np.expm1(forced_prediction)
    free_residual = 100.0 * np.expm1(raw_observed - free_prediction)
    forced_residual = 100.0 * np.expm1(raw_observed - forced_prediction)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.get_cmap("tab10", 6)
    for led in range(1, 7):
        mask = leds == led
        color = colors(led - 1)
        axes[0].plot(days[mask], observed_percent[mask], "o", color=color, ms=3.0)
        axes[0].plot(days[mask], free_percent[mask], color=color, linewidth=1.1)
        axes[0].plot(
            days[mask], forced_percent[mask], color=color, linestyle="--", linewidth=1.1
        )
        axes[1].plot(days[mask], free_residual[mask], color=color, linewidth=1.0)
        axes[1].plot(
            days[mask], forced_residual[mask], color=color, linestyle="--", linewidth=1.0
        )

    for axis in axes:
        axis.axvline(float(data["reset_day"]), color="0.35", linestyle=":", linewidth=1.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Log-change equivalent (%)")
    axes[0].set_title("Observed WFI01 data with free-rate and fixed-rate optical refits")
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("Observed - model (%)")
    axes[1].set_xlabel("Days since first WFI01 exposure")
    axes[1].set_title("Solid = free WFI01 rate; dashed = WFI10 median rate")
    style_handles = [
        Line2D([], [], marker="o", linestyle="", color="black", label="Observed WFI01"),
        Line2D([], [], color="black", label="Free-rate WFI01 model"),
        Line2D([], [], color="black", linestyle="--", label="Fixed WFI10 median-rate model"),
    ]
    led_handles = [
        Line2D([], [], color=colors(led - 1), label=f"LED {led}")
        for led in range(1, 7)
    ]
    figure.legend(
        handles=style_handles + led_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "WFI01 tile 000: WFI10 median rate fixed, optical and phase parameters refit\n"
        f"WFI10 median node rate {np.median(source_rate):.4f} nm/day",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.91))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-id", type=int, default=DEFAULT_TILE_ID)
    parser.add_argument("--target-detector", default=DEFAULT_TARGET_DETECTOR)
    parser.add_argument("--source-detector", default=DEFAULT_SOURCE_DETECTOR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_id < 0:
        raise ValueError("--tile-id must be non-negative")
    target_detector = args.target_detector.upper()
    source_detector = args.source_detector.upper()
    target_output = detector_output_root(args.root, target_detector, args.tile_size)
    source_output = detector_output_root(args.root, source_detector, args.tile_size)
    target_fit_dir = target_output / "fits" / f"tile_{args.tile_id:03d}"
    target_statistics = tile_statistics_path(
        target_output / "tile_statistics",
        target_detector,
        args.tile_size,
        args.tile_id,
    )
    if not target_statistics.is_file():
        raise FileNotFoundError(f"Missing target tile statistics: {target_statistics}")

    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data = production.load_tile_statistics(target_statistics)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)

    source_rates = load_source_median_node_rates(
        source_output,
        len(data["node_days"]),
    )
    source_mean_rate, source_tile_count = _source_mean_rate(source_output)
    with np.load(target_fit_dir / "posterior_samples.npz", allow_pickle=False) as archive:
        target_warmup_rate = _scalar(archive, "constant_rate_map_rate_nm_day")
        free_prediction = np.asarray(
            archive["_map_least_squares_prediction"], dtype=float
        ).reshape(-1)
        archive_values = {name: np.asarray(archive[name]) for name in archive.files}

    initial_values = _load_target_values(target_fit_dir, data, source_rates)
    rate_reference = np.full_like(
        np.asarray(initial_values["np_log_rate_nodes"], dtype=float),
        np.log(target_warmup_rate),
    )
    model = production.build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_reference,
        entropy_weight=PRODUCTION_MAXENT_WEIGHT,
        rate_prior_log_mean=rate_reference,
        constant_rate=False,
        fit_incidence_angle=True,
    )
    forced_values, diagnostics = _bounded_reduced_least_squares(
        model.production_residuals,
        initial_values,
        VARIABLE_SITES,
        jax,
        jnp,
    )
    forced_state = model.production_state(forced_values)
    forced_prediction = np.asarray(
        forced_state["prediction"], dtype=float
    ).reshape(-1)
    raw_observed = _base_observed(data)
    free_prediction = _raw_prediction(data, free_prediction, archive_values)
    forced_prediction = _raw_prediction(data, forced_prediction, forced_values)
    free_metrics, _free_residual = _metrics(data, raw_observed, free_prediction)
    forced_metrics, _forced_residual = _metrics(
        data,
        raw_observed,
        forced_prediction,
    )

    output = args.output or (
        target_fit_dir
        / f"{target_detector.lower()}_tile_{args.tile_id:03d}_"
        f"{source_detector.lower()}_median_rate_optical_refit.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    json_path = output.with_suffix(".json")
    _write_csv(csv_path, data, raw_observed, free_prediction, forced_prediction)
    _plot(output, data, raw_observed, free_prediction, forced_prediction, source_rates)

    ar_parameters = np.asarray(forced_state["ar_parameters"], dtype=float).reshape(-1)
    node_days = np.asarray(data["node_days"], dtype=float)
    source_time_mean = float(
        np.trapezoid(source_rates, node_days)
        / (node_days[-1] - node_days[0])
    )
    payload = {
        "diagnostic": "fixed_source_rate_with_target_optical_refit",
        "target_detector": target_detector,
        "source_detector": source_detector,
        "tile_id": int(args.tile_id),
        "statistics": str(target_statistics.resolve()),
        "target_posterior": str((target_fit_dir / "posterior_samples.npz").resolve()),
        "source_rate_definition": (
            "median over source-detector tiles of node_instantaneous_rate_nm_day; "
            "the full node trajectory is fixed"
        ),
        "source_tile_count": int(source_tile_count),
        "source_detector_median_mean_rate_nm_day": source_mean_rate,
        "source_node_rate_summary_nm_day": {
            "first": float(source_rates[0]),
            "last": float(source_rates[-1]),
            "median": float(np.median(source_rates)),
            "time_average": source_time_mean,
            "minimum": float(np.min(source_rates)),
            "maximum": float(np.max(source_rates)),
        },
        "fixed_rate_node_days": node_days.tolist(),
        "fixed_rate_nodes_nm_day": source_rates.tolist(),
        "variable_sites": list(VARIABLE_SITES),
        "free_fit": {
            "map_objective": _fit_summary_objective(target_fit_dir / "map_summary.md"),
            "metrics": free_metrics,
        },
        "forced_fit": {
            "least_squares_residual_cost": float(diagnostics["residual_cost"]),
            "initial_least_squares_residual_cost": float(
                diagnostics["initial_residual_cost"]
            ),
            "converged": bool(diagnostics["converged"]),
            "function_evaluations": int(diagnostics["nfev"]),
            "message": diagnostics["message"],
            "metrics": forced_metrics,
        },
        "optimized_parameters": {
            "initial_thickness_nm": float(
                np.exp(_scalar(forced_values, "initial_thickness_log_nm"))
            ),
            "reset_initial_thickness_nm": float(
                np.exp(_scalar(forced_values, "reset_initial_thickness_log_nm"))
            ),
            "post_reset_tile_log_offset": _scalar(
                forced_values,
                "post_reset_tile_log_offset",
            ),
            "incidence_angle_deg": _scalar(forced_values, "incidence_angle_deg"),
            "absolute_hgcdte_n_log_scale": _scalar(
                forced_values,
                "absolute_hgcdte_n_log_scale",
            ),
            "absolute_hgcdte_k_log_scale": _scalar(
                forced_values,
                "absolute_hgcdte_k_log_scale",
            ),
            "ar_tile_raw": np.asarray(forced_values["ar_tile_raw"], dtype=float)
            .reshape(-1)
            .tolist(),
            "ar_physical": ar_parameters.tolist(),
        },
        "outputs": {
            "plot": str(output.resolve()),
            "csv": str(csv_path.resolve()),
            "json": str(json_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Source median node rate: {np.median(source_rates):.9g} nm/day")
    print(
        "Free weighted residual RMS: "
        f"{free_metrics['weighted_residual_rms_percent']:.9g}%"
    )
    print(
        "Forced weighted residual RMS: "
        f"{forced_metrics['weighted_residual_rms_percent']:.9g}%"
    )
    print(f"Forced objective residual cost: {diagnostics['residual_cost']:.9g}")
    print(
        "Forced optimized HgCdTe log scales: "
        f"{_scalar(forced_values, 'absolute_hgcdte_n_log_scale'):.9g}, "
        f"{_scalar(forced_values, 'absolute_hgcdte_k_log_scale'):.9g}"
    )
    print(
        "Forced optimized incidence angle: "
        f"{_scalar(forced_values, 'incidence_angle_deg'):.9g} deg"
    )


if __name__ == "__main__":
    main()
