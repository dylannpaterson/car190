"""Fit a shared initial ice thickness and accumulation trajectory to all LEDs.

The observations are normalized to each LED's first exposure, but that first
exposure is not known to be ice-free. This fit therefore evaluates the optical
response as R(d0 + delta_d) / R(d0), with one global d0 and one global smooth
accumulation trajectory shared by all six LEDs.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_rcs_soc_ice_model import (
    DEFAULT_MODEL_PATH,
    broadband_response_curves,
    fit_periodic_model,
    generalized_relative_entropy,
    mean_squared_log_node_curvature,
    read_model_table,
    response_prediction,
    soc_sqrt_thickness,
)
from analyze_wfi01_all_pixel_soc_model import (
    ROOT,
    STATISTICS_PATH,
    all_pixel_loss_components,
    apply_step_offset,
    load_all_pixel_statistics,
    profile_step_offset,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_wfi01_joint_optical_soc import DEFAULT_STEP_TIME, led_summary, parse_step_mask


DEFAULT_OUTPUT = ROOT / "wfi01_initial_thickness_soc"


def fit_initial_thickness_trajectory(
    observations,
    response_curves,
    d_grid,
    reference_thickness_nm,
    reference_time_days,
    reset_day,
    entropy_weight,
    smoothness_weight,
    max_scale,
    max_node_multiplier,
    step_mask,
    initial_thickness_starts,
    scale_starts,
):
    from scipy.optimize import minimize

    repeats = sorted({row["repeat"] for row in observations})
    repeat_to_index = {repeat: index for index, repeat in enumerate(repeats)}
    observation_nodes = np.asarray(
        [repeat_to_index[row["repeat"]] for row in observations], dtype=int
    )
    days = np.asarray([row["days"] for row in observations], dtype=float)
    node_days = np.asarray(
        [
            np.mean([row["days"] for row in observations if row["repeat"] == repeat])
            for repeat in repeats
        ]
    )
    base_thickness = soc_sqrt_thickness(
        days,
        reference_thickness_nm,
        reference_time_days,
        reset_day,
    )
    base_node_thickness = soc_sqrt_thickness(
        node_days,
        reference_thickness_nm,
        reference_time_days,
        reset_day,
    )
    null_loss = all_pixel_loss_components(observations, np.zeros(len(observations)))
    null_between_sse = null_loss["between_sse"]

    def unpack(parameters):
        initial_thickness = float(parameters[0])
        scale = float(np.exp(parameters[1]))
        log_nodes = np.zeros(len(repeats), dtype=float)
        log_nodes[1:] = parameters[2:]
        node_multipliers = np.exp(log_nodes)
        incremental = scale * base_thickness * node_multipliers[observation_nodes]
        node_incremental = scale * base_node_thickness * node_multipliers
        absolute = initial_thickness + incremental
        node_absolute = initial_thickness + node_incremental
        return (
            initial_thickness,
            scale,
            log_nodes,
            node_multipliers,
            incremental,
            node_incremental,
            absolute,
            node_absolute,
        )

    def evaluate(parameters, details=False):
        state = unpack(parameters)
        (
            initial_thickness,
            scale,
            log_nodes,
            node_multipliers,
            incremental,
            node_incremental,
            absolute,
            node_absolute,
        ) = state
        if (
            np.any(np.diff(node_absolute) < -1.0e-9)
            or np.min(absolute) < d_grid[0]
            or np.max(absolute) > d_grid[-1]
        ):
            return np.inf
        prediction = response_prediction(
            observations,
            response_curves,
            d_grid,
            absolute,
        )
        step_log_offset = profile_step_offset(observations, prediction, step_mask)
        prediction = apply_step_offset(
            prediction, step_mask, step_log_offset, observations=observations
        )
        loss = all_pixel_loss_components(observations, prediction)
        data_term = loss["between_sse"] / null_between_sse
        entropy = generalized_relative_entropy(node_multipliers)
        smoothness = mean_squared_log_node_curvature(log_nodes, node_days)
        objective = data_term + entropy_weight * entropy + smoothness_weight * smoothness
        if details:
            return {
                "initial_thickness_nm": initial_thickness,
                "scale": scale,
                "log_nodes": log_nodes,
                "node_multipliers": node_multipliers,
                "incremental_thickness_nm": incremental,
                "node_incremental_thickness_nm": node_incremental,
                "absolute_thickness_nm": absolute,
                "node_absolute_thickness_nm": node_absolute,
                "node_days": node_days,
                "prediction": prediction,
                "step_log_offset": step_log_offset,
                "loss": loss,
                "data_term": data_term,
                "entropy": entropy,
                "smoothness": smoothness,
                "objective": objective,
            }
        return objective

    def constraints(parameters):
        state = unpack(parameters)
        absolute = state[6]
        node_absolute = state[7]
        return np.concatenate(
            (
                np.diff(node_absolute),
                [np.min(absolute) - d_grid[0], d_grid[-1] - np.max(absolute)],
            )
        )

    node_bound = float(np.log(max_node_multiplier))
    bounds = [
        (float(d_grid[0]), float(d_grid[-1])),
        (np.log(1.0e-6), np.log(max_scale)),
        *[(-node_bound, node_bound)] * (len(repeats) - 1),
    ]
    phase = np.linspace(0.0, 2.0 * np.pi, len(repeats) - 1)
    node_starts = (
        np.zeros(len(repeats) - 1),
        0.02 * np.sin(phase),
        -0.02 * np.sin(phase),
    )
    starts = [
        np.concatenate(([d0, np.log(scale)], node_start))
        for d0 in initial_thickness_starts
        for scale in scale_starts
        for node_start in node_starts
    ]
    results = []
    for start_index, start in enumerate(starts):
        if np.min(constraints(start)) < 0.0:
            continue
        result = minimize(
            evaluate,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 2500, "ftol": 1.0e-12, "disp": False},
        )
        constraint_values = constraints(result.x)
        print(
            f"start {start_index + 1}/{len(starts)}: objective={result.fun:.9g}; "
            f"d0={result.x[0]:.6g}; success={result.success}"
        )
        if result.success and np.isfinite(result.fun) and np.min(constraint_values) >= -1.0e-7:
            results.append(result)
    if not results:
        raise RuntimeError("No initial-thickness trajectory fit converged")
    results.sort(key=lambda result: result.fun)
    details = evaluate(results[0].x, details=True)
    details["optimizer_starts"] = len(starts)
    details["optimizer_converged_starts"] = len(results)
    details["converged_initial_thicknesses_nm"] = [
        float(result.x[0]) for result in results
    ]
    details["converged_objectives"] = [float(result.fun) for result in results]
    details["null_loss"] = null_loss
    return details


def write_outputs(output_dir, observations, fit):
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "initial_thickness_predictions.csv"
    with prediction_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "led",
                "repeat",
                "days",
                "observed_log_change",
                "predicted_log_change",
                "residual_log_change",
                "incremental_thickness_nm",
                "absolute_thickness_nm",
            ]
        )
        for row, prediction, incremental, absolute in zip(
            observations,
            fit["prediction"],
            fit["incremental_thickness_nm"],
            fit["absolute_thickness_nm"],
        ):
            writer.writerow(
                [
                    row["led"],
                    row["repeat"],
                    f'{row["days"]:.12g}',
                    f'{row["log_change"]:.12g}',
                    f"{prediction:.12g}",
                    f'{row["log_change"] - prediction:.12g}',
                    f"{incremental:.12g}",
                    f"{absolute:.12g}",
                ]
            )

    node_path = output_dir / "initial_thickness_nodes.csv"
    with node_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "repeat",
                "days",
                "node_multiplier",
                "incremental_thickness_nm",
                "absolute_thickness_nm",
            ]
        )
        for repeat, days, multiplier, incremental, absolute in zip(
            range(len(fit["node_days"])),
            fit["node_days"],
            fit["node_multipliers"],
            fit["node_incremental_thickness_nm"],
            fit["node_absolute_thickness_nm"],
        ):
            writer.writerow(
                [
                    repeat,
                    f"{days:.12g}",
                    f"{multiplier:.12g}",
                    f"{incremental:.12g}",
                    f"{absolute:.12g}",
                ]
            )

    elapsed = float(np.max(fit["node_days"]) - np.min(fit["node_days"]))
    growth = float(
        fit["node_incremental_thickness_nm"][-1]
        - fit["node_incremental_thickness_nm"][0]
    )
    rate = growth / elapsed
    loss = fit["loss"]
    pixel_rms = np.sqrt(loss["total_sse"] / loss["pixel_count"])
    between_rms = np.sqrt(loss["between_sse"] / loss["pixel_count"])
    summary_path = output_dir / "initial_thickness_fit.md"
    lines = [
        "# Shared initial-thickness ice fit",
        "",
        "The first LED exposures are working references, not assumed ice-free measurements. The fitted observable is R(d0 + delta_d) / R(d0).",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Initial physical thickness | {fit['initial_thickness_nm']:.12g} nm |",
        f"| Accumulated thickness | {growth:.12g} nm |",
        f"| Mean accumulation rate | {rate:.12g} nm/day |",
        f"| Final physical thickness | {fit['node_absolute_thickness_nm'][-1]:.12g} nm |",
        f"| ACM step | {100.0 * np.expm1(fit['step_log_offset']):.12g}% |",
        f"| All-pixel RMS | {pixel_rms:.12g} |",
        f"| Between-exposure RMS | {between_rms:.12g} |",
        f"| Between-exposure SSE | {loss['between_sse']:.12g} |",
        f"| Penalized objective | {fit['objective']:.12g} |",
        "",
        "## LED residuals",
        "",
        "| LED | RMS mean residual (log) | Final residual (%) |",
        "|---:|---:|---:|",
    ]
    for row in led_summary(observations, fit["prediction"]):
        lines.append(
            f"| {row['led']} | {row['rms_mean_residual_log']:.12g} | {row['final_residual_percent']:.12g} |"
        )
    lines.extend(
        [
            "",
            f"Converged starts: {fit['optimizer_converged_starts']} / {fit['optimizer_starts']}.",
            f"Prediction table: `{prediction_path}`",
            f"Thickness nodes: `{node_path}`",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True, constrained_layout=True)
    for led, axis in enumerate(axes.flat, start=1):
        indices = [index for index, row in enumerate(observations) if row["led"] == led]
        days = np.asarray([observations[index]["days"] for index in indices])
        observed = 100.0 * np.expm1([observations[index]["log_change"] for index in indices])
        predicted = 100.0 * np.expm1(fit["prediction"][indices])
        axis.plot(days, observed, "ko", ms=3.5, label="Observed")
        axis.plot(days, predicted, color="#D55E00", lw=1.6, label="Initial-thickness fit")
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Change (%)")
    axes[1, 0].set_ylabel("Change (%)")
    axes[1, 1].set_xlabel("Days since first exposure")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("All LEDs: shared initial thickness and accumulation trajectory")
    fit_plot_path = output_dir / "initial_thickness_all_led_fit.png"
    figure.savefig(fit_plot_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(
        fit["node_days"],
        fit["node_incremental_thickness_nm"],
        "o-",
        color="#D55E00",
    )
    axes[0].set_ylabel("Accumulated thickness (nm)")
    axes[0].set_title(
        f"Shared accumulation: {rate:.3f} nm/day mean; d0={fit['initial_thickness_nm']:.3f} nm"
    )
    axes[0].grid(alpha=0.2)
    interval_rates = np.diff(fit["node_incremental_thickness_nm"]) / np.diff(fit["node_days"])
    axes[1].step(
        fit["node_days"][1:],
        interval_rates,
        where="post",
        color="#0072B2",
    )
    axes[1].set_xlabel("Days since first exposure")
    axes[1].set_ylabel("Interval rate (nm/day)")
    axes[1].grid(alpha=0.2)
    trajectory_plot_path = output_dir / "initial_thickness_trajectory.png"
    figure.savefig(trajectory_plot_path, dpi=180)
    plt.close(figure)
    return summary_path, prediction_path, node_path, fit_plot_path, trajectory_plot_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-path", type=Path, default=STATISTICS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--step-time", default=DEFAULT_STEP_TIME)
    parser.add_argument("--sqrt-ref-thickness-nm", type=float, default=122.7)
    parser.add_argument("--sqrt-ref-time-days", type=float, default=4.65)
    parser.add_argument("--reset-day", type=float, default=18.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=4.64158883361)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--max-node-multiplier", type=float, default=2.0)
    parser.add_argument("--thickness-grid-steps", type=int, default=1201)
    parser.add_argument(
        "--initial-thickness-starts",
        default="0,15,40,80,120,160,200,240",
    )
    parser.add_argument("--scale-starts", default="1.5,2.0,2.5")
    return parser.parse_args()


def parse_float_list(value):
    values = [float(item) for item in value.split(",")]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Parameter starts must be finite comma-separated values")
    return values


def main():
    args = parse_args()
    observations = load_all_pixel_statistics(args.statistics_path)
    step_mask = parse_step_mask(observations, args.step_time)
    wavelengths, thicknesses, ratios = read_model_table(args.model_path)
    coefficients, omegas = fit_periodic_model(wavelengths, thicknesses, ratios)
    model = (wavelengths, thicknesses, coefficients, omegas)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    d_grid = np.linspace(0.0, 300.0, args.thickness_grid_steps)
    response = broadband_response_curves(
        model,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
    )[args.sed_mode]
    fit = fit_initial_thickness_trajectory(
        observations,
        response,
        d_grid,
        args.sqrt_ref_thickness_nm,
        args.sqrt_ref_time_days,
        args.reset_day,
        args.entropy_weight,
        args.smoothness_weight,
        args.max_scale,
        args.max_node_multiplier,
        step_mask,
        parse_float_list(args.initial_thickness_starts),
        parse_float_list(args.scale_starts),
    )
    paths = write_outputs(args.output_dir, observations, fit)
    elapsed = fit["node_days"][-1] - fit["node_days"][0]
    growth = (
        fit["node_incremental_thickness_nm"][-1]
        - fit["node_incremental_thickness_nm"][0]
    )
    print(f"Initial physical thickness: {fit['initial_thickness_nm']:.8g} nm")
    print(f"Accumulated thickness: {growth:.8g} nm")
    print(f"Mean accumulation rate: {growth / elapsed:.8g} nm/day")
    print(
        "All-pixel RMS: "
        f"{np.sqrt(fit['loss']['total_sse'] / fit['loss']['pixel_count']):.8g}"
    )
    print(f"Between-exposure SSE: {fit['loss']['between_sse']:.8g}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
