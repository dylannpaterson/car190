"""Simultaneously fit one ice trajectory and globally shared complex-q nuisance."""

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
    read_model_table,
    sed_output,
)
from analyze_wfi01_all_pixel_soc_model import (
    ROOT,
    STATISTICS_PATH,
    fit_global_entropy_sqrt,
    load_all_pixel_statistics,
    parse_mast_time,
    write_prediction_csv,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_wfi01_joint_optical_soc import (
    DEFAULT_STEP_TIME,
    led_summary,
    parse_step_mask,
)


DEFAULT_EFFECTIVE_TABLE = (
    ROOT / "effective_sca1_reflectance_physical" / "effective_sca1_reflectance_table.csv"
)
DEFAULT_OUTPUT = ROOT / "wfi01_joint_complex_q_soc"


def load_effective_q(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No effective q rows found in {path}")
    wavelengths = np.asarray([float(row["wavelength_nm"]) for row in rows])
    rho = np.asarray([float(row["rho"]) for row in rows])
    phase = np.asarray([float(row["phase_offset_rad"]) for row in rows])
    alpha = np.asarray([float(row["alpha_per_nm"]) for row in rows])
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("Effective q wavelengths must be strictly increasing")
    return wavelengths, rho * np.exp(1j * phase), alpha


def q_factor_grid(
    wavelengths,
    q_base,
    alpha_base,
    d_grid,
    parameters,
    phase_model="linear_wavelength",
):
    """Evaluate a smooth global perturbation of the effective q spectrum."""
    log_rho_scale, rho_slope, phase_offset, phase_slope, alpha_scale, alpha_add = parameters
    x = (wavelengths - 1400.0) / 1000.0
    if phase_model == "linear_wavelength":
        phase_shape = phase_offset + phase_slope * x
    elif phase_model == "optical_path":
        # A thin-path phase is linear in wavenumber, not wavelength.
        phase_shape = phase_offset + 2.0 * np.pi * phase_slope * (
            1.0 / wavelengths - 1.0 / 1400.0
        )
    else:
        raise ValueError(f"Unknown phase model: {phase_model}")
    q = q_base * np.exp(
        log_rho_scale + rho_slope * x + 1j * phase_shape
    )
    alpha = alpha_scale * alpha_base + alpha_add / 50.0
    if np.any(alpha < 0.0) or np.any(np.abs(q) >= 0.95):
        return None
    phase = 4.0 * np.pi * 1.25 * d_grid[:, None] / wavelengths[None, :]
    denominator = np.abs(1.0 - q[None, :] * np.exp(1j * phase)) ** 2
    numerator = np.abs(1.0 - q[None, :]) ** 2
    factors = np.exp(-alpha[None, :] * d_grid[:, None]) * numerator / denominator
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        return None
    return factors


def trajectory_plot_series(fit):
    """Return reference-anchored absolute and accumulated thickness series."""
    node_days = np.asarray(fit["node_days"], dtype=float)
    node_incremental = np.asarray(
        fit["node_incremental_thickness_nm"], dtype=float
    )
    node_absolute = np.asarray(fit["node_thickness_nm"], dtype=float)
    initial = float(fit["initial_thickness_nm"])
    if not (
        node_days.shape == node_incremental.shape == node_absolute.shape
        and node_days.ndim == 1
        and node_days.size
    ):
        raise ValueError("Trajectory node arrays must be non-empty one-dimensional arrays")
    if np.any(node_days < 0.0) or not np.all(
        np.isclose(node_absolute, initial + node_incremental, rtol=1.0e-10, atol=1.0e-10)
    ):
        raise ValueError("Trajectory nodes must satisfy absolute = initial + incremental")
    if node_days[0] < 0.0:
        raise ValueError("Trajectory node days must be non-negative")

    if node_days[0] > 0.0:
        return (
            np.concatenate(([0.0], node_days)),
            np.concatenate(([0.0], node_incremental)),
            np.concatenate(([initial], node_absolute)),
        )
    if not np.isclose(node_incremental[0], 0.0, atol=1.0e-10):
        raise ValueError("Accumulation must be zero at the reference epoch")
    return node_days, node_incremental, node_absolute


def trapezoid_sample_weights(wavelengths):
    """Return coefficients whose dot product reproduces np.trapezoid."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    weights = np.empty_like(wavelengths)
    weights[0] = 0.5 * (wavelengths[1] - wavelengths[0])
    weights[-1] = 0.5 * (wavelengths[-1] - wavelengths[-2])
    weights[1:-1] = 0.5 * (wavelengths[2:] - wavelengths[:-2])
    return weights


def prepare_response_integration(
    model_wavelengths,
    spectra,
    qe_wavelengths,
    qe_values,
    sed_mode,
    qe_column=0,
):
    """Precompute one QE-column's LED integrals on the model wavelength grid."""
    qe_values = np.asarray(qe_values)
    if qe_values.ndim != 2 or qe_values.shape[0] != len(qe_wavelengths):
        raise ValueError("QE values must be a wavelength-aligned two-dimensional array")
    qe_column = int(qe_column)
    if qe_column < 0 or qe_column >= qe_values.shape[1]:
        raise ValueError(f"QE column {qe_column} is outside the supplied QE table")
    model_weights = np.zeros((6, len(model_wavelengths)), dtype=float)
    outside_output = np.zeros(6, dtype=float)
    clean_output = np.zeros(6, dtype=float)
    for band in range(1, 7):
        spectrum = spectra[band]
        wavelengths = spectrum["wavelength"]
        qe = np.interp(
            wavelengths,
            qe_wavelengths,
            qe_values[:, qe_column],
            left=0.0,
            right=0.0,
        )
        weighted_output = sed_output(spectrum, sed_mode) * wavelengths * qe
        sample_weights = weighted_output * trapezoid_sample_weights(wavelengths)
        clean_output[band - 1] = np.sum(sample_weights)
        supported = (wavelengths >= model_wavelengths[0]) & (
            wavelengths <= model_wavelengths[-1]
        )
        outside_output[band - 1] = np.sum(sample_weights[~supported])
        supported_wavelengths = wavelengths[supported]
        supported_weights = sample_weights[supported]
        lower = np.searchsorted(
            model_wavelengths,
            supported_wavelengths,
            side="right",
        ) - 1
        lower = np.clip(lower, 0, len(model_wavelengths) - 2)
        upper = lower + 1
        fraction = (
            (supported_wavelengths - model_wavelengths[lower])
            / (model_wavelengths[upper] - model_wavelengths[lower])
        )
        np.add.at(
            model_weights[band - 1],
            lower,
            supported_weights * (1.0 - fraction),
        )
        np.add.at(
            model_weights[band - 1],
            upper,
            supported_weights * fraction,
        )
    if np.any(clean_output <= 0.0):
        raise ValueError("Each LED requires positive clean broadband output")
    return model_weights, outside_output, clean_output


def build_wfi01_response(
    model_wavelengths,
    q_values,
    alpha_values,
    spectra,
    qe_wavelengths,
    qe_values,
    d_grid,
    sed_mode,
    parameters,
    response_integration=None,
    phase_model="linear_wavelength",
):
    model_factors = q_factor_grid(
        model_wavelengths,
        q_values,
        alpha_values,
        d_grid,
        parameters,
        phase_model=phase_model,
    )
    if model_factors is None:
        return None
    if response_integration is None:
        response_integration = prepare_response_integration(
            model_wavelengths,
            spectra,
            qe_wavelengths,
            qe_values,
            sed_mode,
        )
    model_weights, outside_output, clean_output = response_integration
    integrated = model_factors @ model_weights.T + outside_output[None, :]
    response_curves = (integrated / clean_output[None, :]).T[None, :, :]
    if not np.all(np.isfinite(response_curves)) or np.any(response_curves <= 0.0):
        return None
    return response_curves


OPTICAL_PARAMETER_NAMES = (
    "log_q_magnitude_scale",
    "q_magnitude_slope",
    "q_phase_offset_rad",
    "q_phase_slope",
)


def parse_float_list(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty comma-separated list of finite numbers")
    return values


def make_response_factory(
    model_wavelengths,
    q_values,
    alpha_values,
    spectra,
    qe_wavelengths,
    qe_values,
    d_grid,
    sed_mode,
    optical_parameter_count,
    phase_model="linear_wavelength",
):
    integration = prepare_response_integration(
        model_wavelengths,
        spectra,
        qe_wavelengths,
        qe_values,
        sed_mode,
    )
    cache = {}

    def response_factory(parameters):
        parameters = np.asarray(parameters, dtype=float)
        key = tuple(parameters)
        if key not in cache:
            full_parameters = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            full_parameters[:optical_parameter_count] = parameters
            cache[key] = build_wfi01_response(
                model_wavelengths,
                q_values,
                alpha_values,
                spectra,
                qe_wavelengths,
                qe_values,
                d_grid,
                sed_mode,
                full_parameters,
                response_integration=integration,
                phase_model=phase_model,
            )
            if len(cache) > 512:
                latest = cache[key]
                cache.clear()
                cache[key] = latest
        return cache[key]

    return response_factory, integration


def held_out_score(observations, prediction, mask):
    indices = np.flatnonzero(mask)
    residuals = np.asarray(
        [observations[index]["log_change"] - prediction[index] for index in indices]
    )
    counts = np.asarray(
        [observations[index]["pixel_count"] for index in indices], dtype=float
    )
    return {
        "equal_exposure_rmse_log": float(np.sqrt(np.mean(residuals**2))),
        "count_weighted_rmse_log": float(
            np.sqrt(np.average(residuals**2, weights=counts))
        ),
        "mean_residual_log": float(np.mean(residuals)),
    }


def run_validation(
    observations,
    repeats,
    empirical_response,
    baseline_response,
    response_factory,
    d_grid,
    optical_initial,
    optical_bounds,
    optical_penalty,
    args,
    step_mask,
):
    rows = []
    validation_d0_starts = parse_float_list(args.validation_initial_thickness_starts)
    for repeat in repeats:
        held_out = np.asarray(
            [row["repeat"] == repeat for row in observations], dtype=bool
        )
        fit_mask = ~held_out
        empirical = fit_global_entropy_sqrt(
            observations,
            empirical_response,
            d_grid,
            initial_scale=2.0,
            initial_scales=(2.0,),
            max_scale=args.max_scale,
            reference_thickness_nm=args.sqrt_ref_thickness_nm,
            reference_time_days=args.sqrt_ref_time_days,
            reset_day=args.reset_day,
            entropy_weight=args.entropy_weight,
            smoothness_weight=args.smoothness_weight,
            max_node_multiplier=args.max_node_multiplier,
            step_mask=step_mask,
            fit_initial_thickness=True,
            initial_thickness_starts=validation_d0_starts,
            max_initial_thickness_nm=args.max_initial_thickness_nm,
            fit_mask=fit_mask,
            node_start_perturbation=0.0,
            optimizer_maxiter=args.validation_maxiter,
            optimizer_ftol=args.validation_ftol,
        )
        baseline = fit_global_entropy_sqrt(
            observations,
            baseline_response,
            d_grid,
            initial_scale=2.0,
            initial_scales=(2.0,),
            max_scale=args.max_scale,
            reference_thickness_nm=args.sqrt_ref_thickness_nm,
            reference_time_days=args.sqrt_ref_time_days,
            reset_day=args.reset_day,
            entropy_weight=args.entropy_weight,
            smoothness_weight=args.smoothness_weight,
            max_node_multiplier=args.max_node_multiplier,
            step_mask=step_mask,
            fit_initial_thickness=True,
            initial_thickness_starts=validation_d0_starts,
            max_initial_thickness_nm=args.max_initial_thickness_nm,
            fit_mask=fit_mask,
            node_start_perturbation=0.0,
            optimizer_maxiter=args.validation_maxiter,
            optimizer_ftol=args.validation_ftol,
        )
        joint = fit_global_entropy_sqrt(
            observations,
            None,
            d_grid,
            initial_scale=baseline["parameter"],
            initial_scales=(baseline["parameter"],),
            max_scale=args.max_scale,
            reference_thickness_nm=args.sqrt_ref_thickness_nm,
            reference_time_days=args.sqrt_ref_time_days,
            reset_day=args.reset_day,
            entropy_weight=args.entropy_weight,
            smoothness_weight=args.smoothness_weight,
            max_node_multiplier=args.max_node_multiplier,
            step_mask=step_mask,
            initial_thickness_nm=baseline["initial_thickness_nm"],
            fit_initial_thickness=True,
            # The fixed-q fold has already searched the fringe aliases.  Its
            # fitted trajectory is the physically matched warm start for the
            # substantially more expensive joint optical refinement.
            initial_thickness_starts=(baseline["initial_thickness_nm"],),
            max_initial_thickness_nm=args.max_initial_thickness_nm,
            response_factory=response_factory,
            optical_initial_parameters=optical_initial,
            optical_parameter_starts=(optical_initial,),
            optical_parameter_bounds=optical_bounds,
            optical_penalty=optical_penalty,
            fit_mask=fit_mask,
            initial_node_multipliers=baseline["node_multipliers"],
            node_start_perturbation=0.0,
            optimizer_maxiter=args.validation_maxiter,
            optimizer_ftol=args.validation_ftol,
        )
        for label, fit in (
            ("empirical_table", empirical),
            ("fixed_q", baseline),
            ("joint_q", joint),
        ):
            score = held_out_score(observations, fit["prediction"], held_out)
            rows.append(
                {
                    "model": label,
                    "held_out_repeat": repeat,
                    "held_out_observations": int(np.count_nonzero(held_out)),
                    **score,
                    "initial_thickness_nm": fit["initial_thickness_nm"],
                    "mean_rate_nm_day": fit["mean_rate_nm_day"],
                    "step_percent": fit["step_percent"],
                    "optimizer_iterations": fit["optimizer_iterations"],
                    "optimizer_evaluations": fit["optimizer_evaluations"],
                }
            )
        print(f"Validated held-out repeat {repeat}")
    return rows


def write_validation(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_nodes(path, fits):
    fields = [
        "model",
        "repeat",
        "days",
        "node_multiplier",
        "initial_thickness_nm",
        "incremental_thickness_nm",
        "absolute_thickness_nm",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, fit in fits:
            for repeat, days, multiplier, incremental, absolute in zip(
                fit["node_repeats"],
                fit["node_days"],
                fit["node_multipliers"],
                fit["node_incremental_thickness_nm"],
                fit["node_thickness_nm"],
            ):
                writer.writerow(
                    {
                        "model": label,
                        "repeat": int(repeat),
                        "days": f"{days:.12g}",
                        "node_multiplier": f"{multiplier:.12g}",
                        "initial_thickness_nm": f'{fit["initial_thickness_nm"]:.12g}',
                        "incremental_thickness_nm": f"{incremental:.12g}",
                        "absolute_thickness_nm": f"{absolute:.12g}",
                    }
                )


def event_day(observations, step_time):
    if step_time is None:
        return None
    event = parse_mast_time(step_time)
    start = min(parse_mast_time(row["exposure_start_time"]) for row in observations)
    return (event - start).total_seconds() / 86400.0


def write_plots(
    output_dir,
    observations,
    empirical,
    joint,
    step_time,
    validation_rows,
):
    step_day = event_day(observations, step_time)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(17, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for led, axis in enumerate(axes.flat, start=1):
        indices = [
            index for index, row in enumerate(observations) if row["led"] == led
        ]
        days = np.asarray([observations[index]["days"] for index in indices])
        observed = 100.0 * np.expm1(
            [observations[index]["log_change"] for index in indices]
        )
        axis.plot(days, observed, "ko", ms=3.5, label="Observed")
        axis.plot(
            days,
            100.0 * np.expm1(empirical["prediction"][indices]),
            "--",
            color="#777777",
            lw=1.5,
            label="Fixed empirical response",
        )
        axis.plot(
            days,
            100.0 * np.expm1(joint["prediction"][indices]),
            color="#D55E00",
            lw=1.8,
            label="Simultaneous shared-optics fit",
        )
        if step_day is not None:
            axis.axvline(step_day, color="#c1121f", ls=":", lw=1.0)
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Change (%)")
    axes[1, 0].set_ylabel("Change (%)")
    axes[1, 1].set_xlabel("Days since first exposure")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("All LEDs: one trajectory with globally shared optical nuisance")
    fit_plot = output_dir / "simultaneous_complex_q_all_led_fit.png"
    figure.savefig(fit_plot, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    for label, fit, color, linestyle in (
        ("Fixed empirical response", empirical, "#777777", "--"),
        ("Simultaneous shared-optics fit", joint, "#D55E00", "-"),
    ):
        plot_days, plot_incremental, plot_absolute = trajectory_plot_series(fit)
        axes[0].plot(
            plot_days,
            plot_absolute,
            marker="o",
            color=color,
            linestyle=linestyle,
            label=f"{label}; d0={fit['initial_thickness_nm']:.2f} nm",
        )
        axes[1].plot(
            plot_days,
            plot_incremental,
            marker="o",
            color=color,
            linestyle=linestyle,
            label=f"{label}; accumulation",
        )
        rates = np.diff(fit["node_incremental_thickness_nm"]) / np.diff(
            fit["node_days"]
        )
        axes[2].step(
            fit["node_days"][1:],
            rates,
            where="post",
            color=color,
            linestyle=linestyle,
            label=f"{label}; rate",
        )
    for axis in axes:
        if step_day is not None:
            axis.axvline(step_day, color="#c1121f", ls=":", lw=1.0)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Absolute thickness d (nm)")
    axes[0].set_title("Absolute physical thickness: d(t) = d0 + Delta d(t)")
    axes[1].set_ylabel("Accumulated thickness Delta d (nm)")
    axes[1].set_title(
        f"Accumulation relative to the reference epoch; "
        f"{joint['mean_rate_nm_day']:.3f} nm/day mean"
    )
    axes[2].set_xlabel("Days since first exposure")
    axes[2].set_ylabel("Interval rate d(Delta d)/dt (nm/day)")
    trajectory_plot = output_dir / "simultaneous_complex_q_trajectory.png"
    figure.savefig(trajectory_plot, dpi=180)
    plt.close(figure)

    validation_plot = None
    if validation_rows:
        validation_plot = output_dir / "repeat_blocked_validation.png"
        figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
        for label, color, marker in (
            ("empirical_table", "#0072B2", "^"),
            ("fixed_q", "#777777", "o"),
            ("joint_q", "#D55E00", "s"),
        ):
            rows = [row for row in validation_rows if row["model"] == label]
            axis.plot(
                [row["held_out_repeat"] for row in rows],
                [row["equal_exposure_rmse_log"] for row in rows],
                marker=marker,
                color=color,
                label=label.replace("_", " "),
            )
        axis.set_xlabel("Held-out repeat")
        axis.set_ylabel("Held-out equal-exposure RMSE (log)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        figure.savefig(validation_plot, dpi=180)
        plt.close(figure)
    return fit_plot, trajectory_plot, validation_plot


def write_summary(
    path,
    observations,
    empirical,
    baseline,
    joint,
    parameter_names,
    parameter_sigmas,
    args,
    node_path,
    validation_path,
    validation_rows,
):
    lines = [
        "# WFI01 simultaneous shared-optics ice fit",
        "",
        "The initial physical thickness, shared accumulation nodes, global optical nuisance parameters, and common ACM step are optimized in one objective. No parameter is indexed by LED.",
        "",
        f"Optical perturbation model: `{args.optical_model}`.",
        "",
        "The observable remains `R(d_g) / R(d_first, LED)`, so the working reference is not assumed to be ice-free. The effective ice index is fixed at 1.25 and absolute thickness is constrained to the calibrated 0-300 nm interval.",
        "",
        "## Ice Result",
        "",
        "| Model | Initial thickness (nm) | Accumulated thickness (nm) | Mean rate (nm/day) | Final physical thickness (nm) | ACM step (%) | All-pixel RMS | Non-reference RMS | Between-exposure SSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, fit in (
        ("Fixed empirical table", empirical),
        ("Fixed effective q", baseline),
        ("Simultaneous q nuisance", joint),
    ):
        lines.append(
            f"| {label} | {fit['initial_thickness_nm']:.6f} | "
            f"{np.max(fit['incremental_thickness_nm']):.6f} | "
            f"{fit['mean_rate_nm_day']:.6f} | {np.max(fit['thickness_nm']):.6f} | "
            f"{fit['step_percent']:.6f} | {fit['pixel_rms_log']:.8f} | "
            f"{fit['nonreference_pixel_rms_log']:.8f} | "
            f"{fit['between_sse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The all-pixel RMS includes each LED's identity-normalized first exposure. The non-reference RMS excludes those six reference-vs-itself rows and is the less optimistic descriptive residual metric.",
            "",
            "## LED Residuals",
            "",
            "| Model | LED | RMS mean residual (log) | Final residual (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, fit in (
        ("empirical table", empirical),
        ("fixed q", baseline),
        ("simultaneous q", joint),
    ):
        for row in led_summary(observations, fit["prediction"]):
            lines.append(
                f"| {label} | {row['led']} | {row['rms_mean_residual_log']:.12g} | "
                f"{row['final_residual_percent']:.12g} |"
            )
    lines.extend(
        [
            "",
            "## Nuisance Diagnostics",
            "",
            f"Optical prior weight: `{args.optical_prior_weight}`. These values calibrate the response and are not interpreted as uniquely measured coating constants.",
            "",
            "| Shared optical parameter | Fitted value | Prior sigma |",
            "|---|---:|---:|",
        ]
    )
    for name, value, sigma in zip(
        parameter_names,
        joint["optical_parameters"],
        parameter_sigmas,
    ):
        lines.append(f"| {name} | {value:.12g} | {sigma:.12g} |")
    lines.extend(
        [
            "",
            f"The joint optimizer used `{joint['optimizer_starts']}` starts; `{joint['optimizer_converged_starts']}` converged. Its optical-prior contribution is `{joint['optical_penalty']:.12g}`.",
        ]
    )
    if validation_rows:
        lines.extend(
            [
                "",
                "## Repeat-Blocked Validation",
                "",
                "Each listed repeat was excluded for all six LEDs while preserving the full observation layout and original per-LED reference exposures.",
                "",
                "| Model | Mean held-out RMSE (log) | Median held-out RMSE (log) |",
                "|---|---:|---:|",
            ]
        )
        for label in ("empirical_table", "fixed_q", "joint_q"):
            scores = np.asarray(
                [
                    row["equal_exposure_rmse_log"]
                    for row in validation_rows
                    if row["model"] == label
                ]
            )
            lines.append(
                f"| {label.replace('_', ' ')} | {np.mean(scores):.12g} | "
                f"{np.median(scores):.12g} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Effective q input: `{args.effective_q_table}`",
            f"- Thickness nodes: `{node_path}`",
            f"- Repeat-blocked scores: `{validation_path}`" if validation_rows else "- Repeat-blocked validation was not requested.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-path", type=Path, default=STATISTICS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--effective-q-table", type=Path, default=DEFAULT_EFFECTIVE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument(
        "--optical-model",
        choices=("magnitude", "magnitude_phase", "magnitude_optical_path"),
        default="magnitude",
    )
    parser.add_argument("--step-time", default=DEFAULT_STEP_TIME)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--thickness-grid-steps", type=int, default=601)
    parser.add_argument("--sqrt-ref-thickness-nm", type=float, default=122.7)
    parser.add_argument("--sqrt-ref-time-days", type=float, default=4.65)
    parser.add_argument("--reset-day", type=float, default=18.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=4.64158883361)
    parser.add_argument("--max-node-multiplier", type=float, default=2.0)
    parser.add_argument("--max-initial-thickness-nm", type=float, default=250.0)
    parser.add_argument("--initial-thickness-starts", default="0,15,40,80,120,160,200,240")
    parser.add_argument("--optimizer-maxiter", type=int, default=2000)
    parser.add_argument("--optimizer-ftol", type=float, default=1.0e-12)
    parser.add_argument("--optical-prior-weight", type=float, default=0.1)
    parser.add_argument("--log-rho-prior-sigma", type=float, default=0.25)
    parser.add_argument("--rho-slope-prior-sigma", type=float, default=0.5)
    parser.add_argument("--phase-offset-prior-sigma", type=float, default=0.05)
    parser.add_argument("--phase-slope-prior-sigma", type=float, default=0.2)
    parser.add_argument("--phase-opd-prior-sigma", type=float, default=100.0)
    parser.add_argument(
        "--validation-repeats",
        default="",
        help="Comma-separated repeats from 1 through 15 to hold out one at a time",
    )
    parser.add_argument(
        "--validation-initial-thickness-starts",
        default="0,20,80,160,240",
        help="Comma-separated fringe-alias starts for every held-out fit",
    )
    parser.add_argument("--validation-maxiter", type=int, default=500)
    parser.add_argument("--validation-ftol", type=float, default=1.0e-9)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.thickness_grid_steps < 2:
        raise ValueError("--thickness-grid-steps must be at least two")
    if args.optical_prior_weight < 0.0:
        raise ValueError("--optical-prior-weight must be non-negative")
    d0_starts = parse_float_list(args.initial_thickness_starts)
    validation_repeats = (
        [int(value) for value in parse_float_list(args.validation_repeats)]
        if args.validation_repeats.strip()
        else []
    )
    if any(repeat < 1 or repeat > 15 for repeat in validation_repeats):
        raise ValueError("Validation repeats must lie between 1 and 15")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = load_all_pixel_statistics(args.statistics_path)
    step_mask = parse_step_mask(observations, args.step_time)
    model_wavelengths, calibrated_thicknesses, empirical_ratios = read_model_table(
        args.model_path
    )
    q_wavelengths, q_values, alpha_values = load_effective_q(args.effective_q_table)
    if not np.allclose(model_wavelengths, q_wavelengths):
        raise ValueError(
            "The effective q table must use the empirical model wavelength grid"
        )
    d_max = float(np.max(calibrated_thicknesses))
    if args.max_initial_thickness_nm > d_max:
        raise ValueError("Initial-thickness bound exceeds the calibrated range")
    d_grid = np.linspace(0.0, d_max, args.thickness_grid_steps)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    empirical_coefficients, empirical_omegas = fit_periodic_model(
        model_wavelengths,
        calibrated_thicknesses,
        empirical_ratios,
    )
    empirical_response = broadband_response_curves(
        (
            model_wavelengths,
            calibrated_thicknesses,
            empirical_coefficients,
            empirical_omegas,
        ),
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
    )[args.sed_mode]

    parameter_count = 2 if args.optical_model == "magnitude" else 4
    if args.optical_model == "magnitude_optical_path":
        parameter_names = OPTICAL_PARAMETER_NAMES[:3] + ("q_phase_optical_path_nm",)
        phase_model = "optical_path"
        optical_prior_sigmas = [
            args.log_rho_prior_sigma,
            args.rho_slope_prior_sigma,
            args.phase_offset_prior_sigma,
            args.phase_opd_prior_sigma,
        ]
    else:
        parameter_names = OPTICAL_PARAMETER_NAMES[:parameter_count]
        phase_model = "linear_wavelength"
        optical_prior_sigmas = [
            args.log_rho_prior_sigma,
            args.rho_slope_prior_sigma,
            args.phase_offset_prior_sigma,
            args.phase_slope_prior_sigma,
        ][:parameter_count]
    parameter_sigmas = np.asarray(optical_prior_sigmas)
    if np.any(parameter_sigmas <= 0.0):
        raise ValueError("Optical prior sigmas must be positive")
    if args.optical_model == "magnitude_optical_path":
        optical_bounds = (
            (-0.8, 0.8),
            (-1.5, 1.5),
            (-0.3, 0.3),
            (-500.0, 500.0),
        )
    else:
        optical_bounds = (
            (-0.8, 0.8),
            (-1.5, 1.5),
            (-0.3, 0.3),
            (-1.0, 1.0),
        )[:parameter_count]
    optical_initial = np.zeros(parameter_count, dtype=float)

    def optical_penalty(parameters):
        return args.optical_prior_weight * float(
            np.mean((np.asarray(parameters) / parameter_sigmas) ** 2)
        )

    response_factory, integration = make_response_factory(
        model_wavelengths,
        q_values,
        alpha_values,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
        args.sed_mode,
        parameter_count,
        phase_model=phase_model,
    )
    baseline_response = build_wfi01_response(
        model_wavelengths,
        q_values,
        alpha_values,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
        args.sed_mode,
        np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        response_integration=integration,
        phase_model=phase_model,
    )
    empirical = fit_global_entropy_sqrt(
        observations,
        empirical_response,
        d_grid,
        initial_scale=2.0,
        initial_scales=(2.0,),
        max_scale=args.max_scale,
        reference_thickness_nm=args.sqrt_ref_thickness_nm,
        reference_time_days=args.sqrt_ref_time_days,
        reset_day=args.reset_day,
        entropy_weight=args.entropy_weight,
        smoothness_weight=args.smoothness_weight,
        max_node_multiplier=args.max_node_multiplier,
        step_mask=step_mask,
        fit_initial_thickness=True,
        initial_thickness_starts=d0_starts,
        max_initial_thickness_nm=args.max_initial_thickness_nm,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_ftol=args.optimizer_ftol,
    )
    baseline = fit_global_entropy_sqrt(
        observations,
        baseline_response,
        d_grid,
        initial_scale=2.0,
        initial_scales=(2.0,),
        max_scale=args.max_scale,
        reference_thickness_nm=args.sqrt_ref_thickness_nm,
        reference_time_days=args.sqrt_ref_time_days,
        reset_day=args.reset_day,
        entropy_weight=args.entropy_weight,
        smoothness_weight=args.smoothness_weight,
        max_node_multiplier=args.max_node_multiplier,
        step_mask=step_mask,
        fit_initial_thickness=True,
        initial_thickness_starts=d0_starts,
        max_initial_thickness_nm=args.max_initial_thickness_nm,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_ftol=args.optimizer_ftol,
    )
    joint_d0_starts = sorted(set(d0_starts + [baseline["initial_thickness_nm"]]))
    joint = fit_global_entropy_sqrt(
        observations,
        None,
        d_grid,
        initial_scale=baseline["parameter"],
        initial_scales=(baseline["parameter"],),
        max_scale=args.max_scale,
        reference_thickness_nm=args.sqrt_ref_thickness_nm,
        reference_time_days=args.sqrt_ref_time_days,
        reset_day=args.reset_day,
        entropy_weight=args.entropy_weight,
        smoothness_weight=args.smoothness_weight,
        max_node_multiplier=args.max_node_multiplier,
        step_mask=step_mask,
        initial_thickness_nm=baseline["initial_thickness_nm"],
        fit_initial_thickness=True,
        initial_thickness_starts=joint_d0_starts,
        max_initial_thickness_nm=args.max_initial_thickness_nm,
        response_factory=response_factory,
        optical_initial_parameters=optical_initial,
        optical_parameter_starts=(optical_initial,),
        optical_parameter_bounds=optical_bounds,
        optical_penalty=optical_penalty,
        initial_node_multipliers=baseline["node_multipliers"],
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_ftol=args.optimizer_ftol,
    )

    validation_rows = run_validation(
        observations,
        validation_repeats,
        empirical_response,
        baseline_response,
        response_factory,
        d_grid,
        optical_initial,
        optical_bounds,
        optical_penalty,
        args,
        step_mask,
    )
    baseline_prediction_path = args.output_dir / "fixed_q_predictions.csv"
    empirical_prediction_path = args.output_dir / "empirical_table_predictions.csv"
    joint_prediction_path = args.output_dir / "simultaneous_q_predictions.csv"
    write_prediction_csv(
        empirical_prediction_path,
        observations,
        empirical,
        args.sed_mode,
        step_mask=step_mask,
    )
    write_prediction_csv(
        baseline_prediction_path,
        observations,
        baseline,
        args.sed_mode,
        step_mask=step_mask,
    )
    write_prediction_csv(
        joint_prediction_path,
        observations,
        joint,
        args.sed_mode,
        step_mask=step_mask,
    )
    node_path = args.output_dir / "simultaneous_q_nodes.csv"
    write_nodes(
        node_path,
        (
            ("empirical_table", empirical),
            ("fixed_q", baseline),
            ("joint_q", joint),
        ),
    )
    validation_path = args.output_dir / "repeat_blocked_validation.csv"
    write_validation(validation_path, validation_rows)
    plot_paths = write_plots(
        args.output_dir,
        observations,
        empirical,
        joint,
        args.step_time,
        validation_rows,
    )
    summary_path = args.output_dir / "simultaneous_complex_q_fit.md"
    write_summary(
        summary_path,
        observations,
        empirical,
        baseline,
        joint,
        parameter_names,
        parameter_sigmas,
        args,
        node_path,
        validation_path,
        validation_rows,
    )

    print(
        f"Empirical table: d0={empirical['initial_thickness_nm']:.8g} nm; "
        f"rate={empirical['mean_rate_nm_day']:.8g} nm/day; "
        f"RMS={empirical['pixel_rms_log']:.8g}"
    )
    print(
        f"Fixed q: d0={baseline['initial_thickness_nm']:.8g} nm; "
        f"rate={baseline['mean_rate_nm_day']:.8g} nm/day; "
        f"RMS={baseline['pixel_rms_log']:.8g}"
    )
    print(
        f"Simultaneous q: d0={joint['initial_thickness_nm']:.8g} nm; "
        f"rate={joint['mean_rate_nm_day']:.8g} nm/day; "
        f"RMS={joint['pixel_rms_log']:.8g}; step={joint['step_percent']:.8g}%"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {empirical_prediction_path}")
    print(f"Wrote {baseline_prediction_path}")
    print(f"Wrote {joint_prediction_path}")
    print(f"Wrote {node_path}")
    if validation_rows:
        print(f"Wrote {validation_path}")
    for plot_path in plot_paths:
        if plot_path is not None:
            print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
