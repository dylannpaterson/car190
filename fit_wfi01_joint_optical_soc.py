"""Jointly fit one WFI01 ice trajectory and a small optical calibration.

This diagnostic keeps the all-pixel loss and one common ice trajectory, while
allowing the empirical SCA1 response to change through a few global optical
parameters:

* ``n_ice`` changes the fringe phase with thickness;
* ``fringe_contrast`` changes the fitted interference amplitude;
* ``absorption_per_50nm`` adds a shared exponential attenuation term.
* ``phase_offset`` shifts the common effective interference phase.

The fit is intentionally low-dimensional. It does not modify the production
response table or the existing all-pixel fit outputs.
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analyze_rcs_soc_ice_model import (
    DEFAULT_MODEL_PATH,
    fit_periodic_model,
    mean_endpoint_rate,
    read_model_table,
    response_prediction,
    soc_sqrt_thickness,
    sed_output,
)
from analyze_wfi01_all_pixel_soc_model import (
    DATA_DIR,
    INDEX_PATH,
    ROOT,
    STATISTICS_PATH,
    all_pixel_loss_components,
    apply_step_offset,
    fit_global_entropy_sqrt,
    fit_global_soc_sqrt,
    load_all_pixel_statistics,
    parse_mast_time,
    profile_step_offset,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed


DEFAULT_OUTPUT = ROOT / "wfi01_joint_optical_soc"
DEFAULT_STEP_TIME = "2026-09-14T23:38:00Z"
MODES = ("signed", "clipped", "core")


def parse_step_mask(observations, step_time):
    if step_time is None:
        return None
    event_time = parse_mast_time(step_time)
    mask = np.asarray(
        [parse_mast_time(row["exposure_start_time"]) >= event_time for row in observations],
        dtype=bool,
    )
    if not np.any(mask) or np.all(mask):
        raise ValueError("The event time must split the supplied observations")
    return mask


def read_optical_model(path):
    wavelengths, thicknesses, ratios = read_model_table(path)
    coefficients, omegas = fit_periodic_model(wavelengths, thicknesses, ratios)
    return wavelengths, coefficients, omegas


def optical_factor_grid(wavelengths, coefficients, omegas, d_grid, parameters):
    """Evaluate the globally perturbed periodic response on model wavelengths."""
    n_ice, fringe_contrast, absorption_per_50nm, phase_offset = parameters
    phase = (
        omegas[None, :]
        * (n_ice / 1.25)
        * d_grid[:, None]
        / 50.0
        + phase_offset
    )
    numerator = coefficients[:, 0][None, :] + fringe_contrast * (
        coefficients[:, 1][None, :] * np.cos(phase)
        + coefficients[:, 2][None, :] * np.sin(phase)
    )
    clean = coefficients[:, 0] + fringe_contrast * (
        coefficients[:, 1] * np.cos(phase_offset)
        + coefficients[:, 2] * np.sin(phase_offset)
    )
    factors = (
        np.exp(-absorption_per_50nm * d_grid[:, None] / 50.0)
        * numerator
        / clean[None, :]
    )
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        return None
    return factors


def build_wfi01_response(
    model_wavelengths,
    coefficients,
    omegas,
    spectra,
    qe_wavelengths,
    qe_values,
    d_grid,
    sed_mode,
    optical_parameters,
):
    """Build only the WFI01/SCA1 response curves needed by this fit."""
    model_factors = optical_factor_grid(
        model_wavelengths,
        coefficients,
        omegas,
        d_grid,
        optical_parameters,
    )
    if model_factors is None:
        return None

    response_curves = np.zeros((1, 6, len(d_grid)), dtype=float)
    for band in range(1, 7):
        spectrum = spectra[band]
        wavelengths = spectrum["wavelength"]
        supported = (wavelengths >= model_wavelengths[0]) & (
            wavelengths <= model_wavelengths[-1]
        )
        factors = np.ones((len(d_grid), len(wavelengths)), dtype=float)
        if np.any(supported):
            factors[:, supported] = np.asarray(
                [
                    np.interp(
                        wavelengths[supported],
                        model_wavelengths,
                        row,
                    )
                    for row in model_factors
                ]
            )

        qe = np.interp(
            wavelengths,
            qe_wavelengths,
            qe_values[:, 0],
            left=0.0,
            right=0.0,
        )
        weighted_output = sed_output(spectrum, sed_mode) * wavelengths * qe
        clean_response = float(np.trapezoid(weighted_output, wavelengths))
        if clean_response <= 0.0:
            raise ValueError(f"Nonpositive clean WFI01 response in LED {band}")
        response_curves[0, band - 1] = np.trapezoid(
            factors * weighted_output[None, :],
            wavelengths,
            axis=1,
        ) / clean_response
    return response_curves


def joint_prediction(observations, response_curves, d_grid, scale, references, step_mask):
    reference_thickness_nm, reference_time_days, reset_day = references
    days = np.asarray([row["days"] for row in observations], dtype=float)
    thickness = scale * soc_sqrt_thickness(
        days,
        reference_thickness_nm,
        reference_time_days,
        reset_day,
    )
    if np.min(thickness) < d_grid[0] or np.max(thickness) > d_grid[-1]:
        return None
    prediction = response_prediction(observations, response_curves, d_grid, thickness)
    step_log_offset = profile_step_offset(observations, prediction, step_mask)
    prediction = apply_step_offset(
        prediction, step_mask, step_log_offset, observations=observations
    )
    return prediction, thickness, step_log_offset


def make_objective(
    observations,
    model_wavelengths,
    coefficients,
    omegas,
    spectra,
    qe_wavelengths,
    qe_values,
    d_grid,
    sed_mode,
    references,
    step_mask,
):
    null_between_sse = all_pixel_loss_components(
        observations,
        np.zeros(len(observations)),
    )["between_sse"]

    def objective(parameters, details=False):
        scale = float(parameters[0])
        if scale <= 0.0:
            return np.inf
        response_curves = build_wfi01_response(
            model_wavelengths,
            coefficients,
            omegas,
            spectra,
            qe_wavelengths,
            qe_values,
            d_grid,
            sed_mode,
            parameters[1:],
        )
        if response_curves is None:
            return np.inf
        prediction_result = joint_prediction(
            observations,
            response_curves,
            d_grid,
            scale,
            references,
            step_mask,
        )
        if prediction_result is None:
            return np.inf
        prediction, thickness, step_log_offset = prediction_result
        loss = all_pixel_loss_components(observations, prediction)
        value = loss["between_sse"] / null_between_sse
        if details:
            return {
                "objective": value,
                "prediction": prediction,
                "thickness_nm": thickness,
                "step_log_offset": step_log_offset,
                "loss": loss,
                "response_curves": response_curves,
            }
        return value

    return objective


def led_summary(observations, prediction):
    rows = []
    for led in sorted({row["led"] for row in observations}):
        indices = [index for index, row in enumerate(observations) if row["led"] == led]
        residuals = np.asarray(
            [observations[index]["log_change"] - prediction[index] for index in indices]
        )
        rows.append(
            {
                "led": led,
                "rms_mean_residual_log": float(np.sqrt(np.mean(residuals**2))),
                "final_residual_percent": float(
                    100.0 * np.expm1(residuals[-1])
                ),
            }
        )
    return rows


def write_prediction(path, observations, prediction, label):
    fields = [
        "model",
        "led",
        "repeat",
        "days",
        "observed_log_change",
        "predicted_log_change",
        "residual_log_change",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, value in zip(observations, prediction):
            writer.writerow(
                {
                    "model": label,
                    "led": row["led"],
                    "repeat": row["repeat"],
                    "days": f'{row["days"]:.12g}',
                    "observed_log_change": f'{row["log_change"]:.12g}',
                    "predicted_log_change": f"{value:.12g}",
                    "residual_log_change": f'{row["log_change"] - value:.12g}',
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-path", type=Path, default=STATISTICS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sed-mode", choices=MODES, default="core")
    parser.add_argument("--step-time", default=DEFAULT_STEP_TIME)
    parser.add_argument("--max-scale", type=float, default=20.0)
    parser.add_argument("--scale-steps", type=int, default=1001)
    parser.add_argument("--thickness-grid-steps", type=int, default=1201)
    parser.add_argument("--optimization-thickness-grid-steps", type=int, default=401)
    parser.add_argument("--sqrt-ref-thickness-nm", type=float, default=122.7)
    parser.add_argument("--sqrt-ref-time-days", type=float, default=4.65)
    parser.add_argument("--reset-day", type=float, default=18.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=4.64158883361)
    parser.add_argument("--max-node-multiplier", type=float, default=2.0)
    parser.add_argument("--optimizer-starts", type=int, default=5)
    parser.add_argument("--allow-negative-absorption", action="store_true")
    parser.add_argument("--n-min", type=float, default=1.15)
    parser.add_argument("--n-max", type=float, default=1.35)
    parser.add_argument("--contrast-min", type=float, default=0.5)
    parser.add_argument("--contrast-max", type=float, default=1.5)
    parser.add_argument("--phase-max", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_scale <= 0.0 or args.thickness_grid_steps < 2:
        raise ValueError("Invalid scale or thickness-grid settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = load_all_pixel_statistics(args.statistics_path)
    step_mask = parse_step_mask(observations, args.step_time)
    model_wavelengths, coefficients, omegas = read_optical_model(args.model_path)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    references = (
        args.sqrt_ref_thickness_nm,
        args.sqrt_ref_time_days,
        args.reset_day,
    )
    elapsed_max = max(row["days"] for row in observations)
    d_max = max(
        300.0,
        args.max_scale
        * args.max_node_multiplier
        * soc_sqrt_thickness(
            elapsed_max,
            args.sqrt_ref_thickness_nm,
            args.sqrt_ref_time_days,
            args.reset_day,
        ),
    )
    d_grid = np.linspace(0.0, d_max, args.thickness_grid_steps)
    optimization_d_grid = np.linspace(
        0.0,
        d_max,
        args.optimization_thickness_grid_steps,
    )

    baseline_response = build_wfi01_response(
        model_wavelengths,
        coefficients,
        omegas,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
        args.sed_mode,
        (1.25, 1.0, 0.0, 0.0),
    )
    baseline_fit = fit_global_entropy_sqrt(
        observations,
        baseline_response,
        d_grid,
        2.0,
        [1.0, 2.0, 3.0],
        args.max_scale,
        args.sqrt_ref_thickness_nm,
        args.sqrt_ref_time_days,
        args.reset_day,
        args.entropy_weight,
        args.smoothness_weight,
        args.max_node_multiplier,
        step_mask=step_mask,
    )

    lower_absorption = -0.05 if args.allow_negative_absorption else 0.0
    bounds = [
        (1.0e-6, args.max_scale),
        (args.n_min, args.n_max),
        (args.contrast_min, args.contrast_max),
        (lower_absorption, 0.05),
        (-args.phase_max, args.phase_max),
    ]
    objective = make_objective(
        observations,
        model_wavelengths,
        coefficients,
        omegas,
        spectra,
        qe_wavelengths,
        qe_values,
        optimization_d_grid,
        args.sed_mode,
        references,
        step_mask,
    )
    starts = [
        np.asarray([baseline_fit["parameter"], 1.25, 1.0, 0.0, 0.0]),
        np.asarray([2.0, 1.20, 0.9, 0.0, -0.2]),
        np.asarray([2.0, 1.30, 1.1, 0.0, 0.2]),
        np.asarray([3.0, 1.25, 0.8, 0.01, -0.4]),
        np.asarray([1.0, 1.25, 1.2, 0.01, 0.4]),
    ][: max(args.optimizer_starts, 1)]

    from scipy.optimize import minimize

    results = []
    for index, start in enumerate(starts):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250, "ftol": 1.0e-11, "maxls": 40},
        )
        results.append(result)
        print(
            f"start {index + 1}/{len(starts)}: objective={result.fun:.8g}; "
            f"parameters={result.x}; success={result.success}"
        )
    best_result = min(results, key=lambda result: result.fun)
    best_parameters = best_result.x

    final_response = build_wfi01_response(
        model_wavelengths,
        coefficients,
        omegas,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
        args.sed_mode,
        best_parameters[1:],
    )
    joint_sqrt_result = joint_prediction(
        observations,
        final_response,
        d_grid,
        best_parameters[0],
        references,
        step_mask,
    )
    if joint_sqrt_result is None:
        raise RuntimeError("The final joint square-root prediction is outside the response grid")
    joint_sqrt_prediction, joint_sqrt_thickness, joint_sqrt_step = joint_sqrt_result
    joint_sqrt_details = {
        "prediction": joint_sqrt_prediction,
        "thickness_nm": joint_sqrt_thickness,
        "step_log_offset": joint_sqrt_step,
        "loss": all_pixel_loss_components(observations, joint_sqrt_prediction),
    }
    joint_np = fit_global_entropy_sqrt(
        observations,
        final_response,
        d_grid,
        best_parameters[0],
        [best_parameters[0]],
        args.max_scale,
        args.sqrt_ref_thickness_nm,
        args.sqrt_ref_time_days,
        args.reset_day,
        args.entropy_weight,
        args.smoothness_weight,
        args.max_node_multiplier,
        step_mask=step_mask,
    )

    write_prediction(
        args.output_dir / "baseline_predictions.csv",
        observations,
        baseline_fit["prediction"],
        "fixed_empirical_response",
    )
    write_prediction(
        args.output_dir / "joint_sqrt_predictions.csv",
        observations,
        joint_sqrt_details["prediction"],
        "joint_optical_sqrt",
    )
    write_prediction(
        args.output_dir / "joint_np_predictions.csv",
        observations,
        joint_np["prediction"],
        "joint_optical_entropy_sqrt",
    )

    summary_path = args.output_dir / "joint_optical_fit.md"
    baseline_loss = all_pixel_loss_components(observations, baseline_fit["prediction"])
    joint_sqrt_loss = joint_sqrt_details["loss"]
    joint_np_loss = {
        "between_sse": joint_np["between_sse"],
        "total_sse": joint_np["total_sse"],
        "pixel_rms_log": joint_np["pixel_rms_log"],
    }
    summary_lines = [
        "# WFI01 joint optical and ice fit",
        "",
        f"SED mode: `{args.sed_mode}`",
        f"ACM step time: `{args.step_time}`",
        "",
        "## Joint parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| SOC scale | {best_parameters[0]:.12g} |",
        f"| n_ice | {best_parameters[1]:.12g} |",
        f"| fringe contrast | {best_parameters[2]:.12g} |",
        f"| absorption per 50 nm | {best_parameters[3]:.12g} |",
        f"| phase offset (rad) | {best_parameters[4]:.12g} |",
        f"| optimizer objective | {best_result.fun:.12g} |",
        "",
        "## Fit comparison",
        "",
        "| Fit | Between-exposure SSE | All-pixel RMS | Step |",
        "|---|---:|---:|---:|",
        f"| Fixed empirical + entropy trajectory | {baseline_fit['between_sse']:.12g} | {baseline_fit['pixel_rms_log']:.12g} | {baseline_fit['step_percent']:.12g}% |",
        f"| Joint optical + square-root trajectory | {joint_sqrt_loss['between_sse']:.12g} | {np.sqrt(joint_sqrt_loss['total_sse'] / joint_sqrt_loss['pixel_count']):.12g} | {100.0 * np.expm1(joint_sqrt_details['step_log_offset']):.12g}% |",
        f"| Joint optical + entropy trajectory | {joint_np_loss['between_sse']:.12g} | {joint_np_loss['pixel_rms_log']:.12g} | {joint_np['step_percent']:.12g}% |",
        "",
        "## LED residuals",
        "",
        "| Model | LED | RMS mean residual (log) | Final residual (%) |",
        "|---|---:|---:|---:|",
    ]
    for label, prediction in (
        ("fixed empirical", baseline_fit["prediction"]),
        ("joint square-root", joint_sqrt_details["prediction"]),
        ("joint entropy", joint_np["prediction"]),
    ):
        for row in led_summary(observations, prediction):
            summary_lines.append(
                f"| {label} | {row['led']} | {row['rms_mean_residual_log']:.12g} | {row['final_residual_percent']:.12g} |"
            )
    summary_lines.extend(
        [
            "",
            "The optical parameters are global across all LEDs; no LED-specific optical scale is fitted.",
            "",
            "Inputs and outputs:",
            f"- Statistics: `{args.statistics_path}`",
            f"- Empirical model: `{args.model_path}`",
            f"- Baseline predictions: `{args.output_dir / 'baseline_predictions.csv'}`",
            f"- Joint predictions: `{args.output_dir / 'joint_np_predictions.csv'}`",
        ]
    )
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Baseline entropy RMS: {baseline_fit['pixel_rms_log']:.8g}")
    print(f"Joint square-root RMS: {np.sqrt(joint_sqrt_loss['total_sse'] / joint_sqrt_loss['pixel_count']):.8g}")
    print(f"Joint entropy RMS: {joint_np['pixel_rms_log']:.8g}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
