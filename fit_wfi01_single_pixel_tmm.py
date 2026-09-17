"""Fit one WFI01 pixel with a static air/ice/AR/HgCdTe TMM response."""

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
    response_prediction,
)
from analyze_wfi01_all_pixel_soc_model import (
    ROOT,
    fit_global_entropy_sqrt,
    load_all_pixel_statistics,
    parse_mast_time,
    write_prediction_csv,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_effective_sca1_tmm import load_empirical_table, transmission_ratios
from fit_wfi01_joint_complex_q_soc import (
    led_summary,
    prepare_response_integration,
    trapezoid_sample_weights,
    write_nodes,
)
from fit_wfi01_joint_optical_soc import DEFAULT_STEP_TIME, parse_step_mask


DEFAULT_STATISTICS = ROOT / "wfi01_single_pixel_1000_1000/single_pixel_log_statistics.csv"
DEFAULT_OUTPUT = ROOT / "wfi01_single_pixel_1000_1000_tmm"
TMM_PARAMETER_NAMES = (
    "ar_thickness_nm",
    "ar_refractive_index",
    "ar_absorption_index",
    "hgcdte_refractive_index",
    "hgcdte_absorption_index",
)
TMM_CENTER = np.asarray([2.0, 2.0, 0.05, 3.5, 0.5], dtype=float)
TMM_SIGMAS = np.asarray([3.0, 1.0, 0.2, 1.0, 0.5], dtype=float)
TMM_BOUNDS = (
    (0.01, 10.0),
    (1.0, 4.0),
    (0.0, 0.5),
    (2.0, 6.0),
    (0.0, 2.0),
)
AR_THICKNESS_SCALE_NM = 100.0
ICE_REAL_INDEX = 1.25
ICE_IMAGINARY_INDEX = 0.0
HGCDTE_MODEL_CHOICES = ("free", "nominal-2p5um")
# Constant diagnostic baseline for a nominal 2.5 um-cutoff MCT absorber. The
# published optical constants are composition- and wavelength-dependent; this
# approximation is kept explicit until a detector-matched table is available.
NOMINAL_HGCDTE = np.asarray([3.6, 0.7], dtype=float)
HGCDTE_FIXED_HALF_WIDTH = 1.0e-6
NOMINAL_HGCDTE_REFERENCE = (
    "Johs et al., Thin Solid Films 313-314, 137-142 (1998), "
    "doi:10.1016/S0040-6090(97)00800-6; the values here are a provisional "
    "constant approximation, not a direct evaluation of that library."
)


def tmm_configuration(hgcdte_model="free"):
    if hgcdte_model not in HGCDTE_MODEL_CHOICES:
        raise ValueError(f"Unknown HgCdTe model: {hgcdte_model}")
    center = TMM_CENTER.copy()
    bounds = list(TMM_BOUNDS)
    fixed_hgcdte = None
    if hgcdte_model == "nominal-2p5um":
        center[3:5] = NOMINAL_HGCDTE
        bounds[3] = (
            NOMINAL_HGCDTE[0] - HGCDTE_FIXED_HALF_WIDTH,
            NOMINAL_HGCDTE[0] + HGCDTE_FIXED_HALF_WIDTH,
        )
        bounds[4] = (
            NOMINAL_HGCDTE[1] - HGCDTE_FIXED_HALF_WIDTH,
            NOMINAL_HGCDTE[1] + HGCDTE_FIXED_HALF_WIDTH,
        )
        fixed_hgcdte = NOMINAL_HGCDTE.copy()
    return center, tuple(bounds), fixed_hgcdte


def parse_float_list(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty comma-separated list of finite numbers")
    return values


def tmm_parameters(parameters, fixed_hgcdte=None):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (5,) or not np.all(np.isfinite(parameters)):
        raise ValueError("TMM parameters must be five finite values")
    if fixed_hgcdte is not None:
        parameters = parameters.copy()
        parameters[3:5] = fixed_hgcdte
    return np.asarray(
        [
            ICE_REAL_INDEX,
            ICE_IMAGINARY_INDEX,
            AR_THICKNESS_SCALE_NM * parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
        ],
        dtype=float,
    )


def build_tmm_response(
    model_wavelengths,
    spectra,
    qe_wavelengths,
    qe_values,
    d_grid,
    parameters,
    integration,
    fixed_hgcdte=None,
):
    factors = transmission_ratios(
        model_wavelengths,
        d_grid,
        tmm_parameters(parameters, fixed_hgcdte=fixed_hgcdte),
    ).T
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        return None
    model_weights, outside_output, clean_output = integration
    integrated = factors @ model_weights.T + outside_output[None, :]
    response_curves = (integrated / clean_output[None, :]).T[None, :, :]
    if not np.all(np.isfinite(response_curves)) or np.any(response_curves <= 0.0):
        return None
    return response_curves


def build_empirical_response(model_path, spectra, qe_wavelengths, qe_values, d_grid, sed_mode):
    wavelengths, calibrated_thicknesses, ratios = read_model_table(model_path)
    coefficients, omegas = fit_periodic_model(
        wavelengths,
        calibrated_thicknesses,
        ratios,
    )
    return broadband_response_curves(
        (wavelengths, calibrated_thicknesses, coefficients, omegas),
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
    )[sed_mode]


def write_summary(
    path,
    observations,
    empirical,
    tmm,
    parameters,
    validation_note,
    optical_center,
    hgcdte_model,
):
    lines = [
        "# WFI01 single-pixel transfer-matrix fit",
        "",
        "This fit uses only WFI01 pixel (1000, 1000): six LEDs across 19 epochs. "
        "The optical stack is vacuum / ice / one effective static AR layer / absorbing HgCdTe.",
        "",
        f"HgCdTe model: {hgcdte_model}. It is represented by a constant effective "
        "complex index. The AR layer is not a claim about the proprietary coating recipe.",
        (
            f"Nominal baseline n={NOMINAL_HGCDTE[0]:g}, k={NOMINAL_HGCDTE[1]:g}. "
            f"{NOMINAL_HGCDTE_REFERENCE}"
            if hgcdte_model == "nominal-2p5um"
            else ""
        ),
        "",
        "## Fit comparison",
        "",
        "| Model | Initial thickness (nm) | Accumulated thickness (nm) | Mean rate (nm/day) | Final thickness (nm) | ACM step (%) | RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, fit in (("Empirical response", empirical), ("TMM response", tmm)):
        lines.append(
            f"| {label} | {fit['initial_thickness_nm']:.8f} | "
            f"{np.max(fit['incremental_thickness_nm']):.8f} | "
            f"{fit['mean_rate_nm_day']:.8f} | {np.max(fit['thickness_nm']):.8f} | "
            f"{fit['step_percent']:.8f} | {fit['pixel_rms_log']:.10f} |"
        )
    lines.extend(
        [
            "",
            "## Fitted static optical parameters",
            "",
            "| Parameter | Value | Prior center |",
            "|---|---:|---:|",
            f"| AR thickness (nm) | {AR_THICKNESS_SCALE_NM * parameters[0]:.8f} | {AR_THICKNESS_SCALE_NM * optical_center[0]:.8f} |",
            f"| AR refractive index | {parameters[1]:.8f} | {optical_center[1]:.8f} |",
            f"| AR absorption index | {parameters[2]:.8f} | {optical_center[2]:.8f} |",
            f"| HgCdTe refractive index | {parameters[3]:.8f} | {optical_center[3]:.8f} |",
            f"| HgCdTe absorption index | {parameters[4]:.8f} | {optical_center[4]:.8f} |",
            "",
            "## LED residuals",
            "",
            "| Model | LED | RMS residual (log) | Final residual (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, fit in (("empirical", empirical), ("TMM", tmm)):
        for row in led_summary(observations, fit["prediction"]):
            lines.append(
                f"| {label} | {row['led']} | {row['rms_mean_residual_log']:.10g} | "
                f"{row['final_residual_percent']:.10g} |"
            )
    lines.extend(
        [
            "",
            validation_note,
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_plot(path, observations, empirical, tmm):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, constrained_layout=True)
    for led, axis in enumerate(axes.flat, start=1):
        indices = [index for index, row in enumerate(observations) if row["led"] == led]
        days = np.asarray([observations[index]["days"] for index in indices])
        observed = 100.0 * np.expm1([observations[index]["log_change"] for index in indices])
        axis.plot(days, observed, "ko", ms=3, label="Observed")
        axis.plot(days, 100.0 * np.expm1(empirical["prediction"][indices]), "--", color="#777777", label="Empirical")
        axis.plot(days, 100.0 * np.expm1(tmm["prediction"][indices]), color="#D55E00", label="TMM")
        axis.axhline(0.0, color="black", lw=0.6)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Change (%)")
    axes[1, 0].set_ylabel("Change (%)")
    axes[1, 1].set_xlabel("Days since first exposure")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Pixel (1000, 1000): empirical response versus static TMM")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def held_out_score(observations, prediction, mask):
    indices = np.flatnonzero(mask)
    residuals = np.asarray(
        [observations[index]["log_change"] - prediction[index] for index in indices]
    )
    return float(np.sqrt(np.mean(residuals**2)))


def write_validation(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(
    observations,
    repeats,
    empirical_response,
    d_grid,
    response_factory,
    step_mask,
    args,
):
    rows = []
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
            max_scale=10.0,
            reference_thickness_nm=122.7,
            reference_time_days=4.65,
            reset_day=18.0,
            entropy_weight=0.01,
            smoothness_weight=4.64158883361,
            max_node_multiplier=2.0,
            step_mask=step_mask,
            fit_initial_thickness=True,
            initial_thickness_starts=parse_float_list(
                args.validation_initial_thickness_starts
            ),
            max_initial_thickness_nm=args.max_initial_thickness_nm,
            fit_mask=fit_mask,
            optimizer_maxiter=args.validation_maxiter,
            optimizer_ftol=args.validation_ftol,
        )
        tmm = fit_global_entropy_sqrt(
            observations,
            None,
            d_grid,
            initial_scale=empirical["parameter"],
            initial_scales=(empirical["parameter"],),
            max_scale=10.0,
            reference_thickness_nm=122.7,
            reference_time_days=4.65,
            reset_day=18.0,
            entropy_weight=0.01,
            smoothness_weight=4.64158883361,
            max_node_multiplier=2.0,
            step_mask=step_mask,
            initial_thickness_nm=empirical["initial_thickness_nm"],
            fit_initial_thickness=True,
            initial_thickness_starts=(empirical["initial_thickness_nm"],),
            max_initial_thickness_nm=args.max_initial_thickness_nm,
            response_factory=response_factory,
            optical_initial_parameters=args.tmm_center,
            optical_parameter_starts=(args.tmm_center,),
            optical_parameter_bounds=args.tmm_bounds,
            optical_penalty=lambda parameters: args.optical_prior_weight
            * float(np.mean(((np.asarray(parameters) - args.tmm_center) / TMM_SIGMAS) ** 2)),
            fit_mask=fit_mask,
            initial_node_multipliers=empirical["node_multipliers"],
            optimizer_maxiter=args.validation_maxiter,
            optimizer_ftol=args.validation_ftol,
        )
        for label, fit in (("empirical", empirical), ("tmm", tmm)):
            rows.append(
                {
                    "model": label,
                    "held_out_epoch": repeat,
                    "held_out_observations": int(np.count_nonzero(held_out)),
                    "held_out_rmse_log": held_out_score(
                        observations, fit["prediction"], held_out
                    ),
                    "initial_thickness_nm": fit["initial_thickness_nm"],
                    "mean_rate_nm_day": fit["mean_rate_nm_day"],
                }
            )
        print(f"Validated held-out epoch {repeat}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-path", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--hgcdte-model", choices=HGCDTE_MODEL_CHOICES, default="free")
    parser.add_argument("--step-time", default=DEFAULT_STEP_TIME)
    parser.add_argument("--thickness-grid-steps", type=int, default=601)
    parser.add_argument("--max-initial-thickness-nm", type=float, default=50.0)
    parser.add_argument("--initial-thickness-starts", default="0,5,10,20,35,50")
    parser.add_argument("--optimizer-maxiter", type=int, default=1000)
    parser.add_argument("--optimizer-ftol", type=float, default=1.0e-10)
    parser.add_argument("--optical-prior-weight", type=float, default=0.001)
    parser.add_argument(
        "--validation-epochs",
        default="",
        help="Comma-separated epochs to hold out during validation",
    )
    parser.add_argument("--validation-initial-thickness-starts", default="0,5,10,20,35,50")
    parser.add_argument("--validation-maxiter", type=int, default=300)
    parser.add_argument("--validation-ftol", type=float, default=1.0e-8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.thickness_grid_steps < 2:
        raise ValueError("--thickness-grid-steps must be at least two")
    observations = load_all_pixel_statistics(args.statistics_path)
    tmm_center, tmm_bounds, fixed_hgcdte = tmm_configuration(args.hgcdte_model)
    args.tmm_center = tmm_center
    args.tmm_bounds = tmm_bounds
    step_mask = parse_step_mask(observations, args.step_time)
    model_wavelengths, calibrated_thicknesses, _ = read_model_table(args.model_path)
    tmm_wavelengths, _ = load_empirical_table(args.model_path)
    if not np.allclose(model_wavelengths, tmm_wavelengths):
        raise ValueError("The TMM and empirical model wavelength grids differ")
    d_grid = np.linspace(0.0, float(np.max(calibrated_thicknesses)), args.thickness_grid_steps)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    integration = prepare_response_integration(
        model_wavelengths,
        spectra,
        qe_wavelengths,
        qe_values,
        args.sed_mode,
    )
    empirical_response = build_empirical_response(
        args.model_path,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
        args.sed_mode,
    )

    cache = {}

    def response_factory(parameters):
        key = tuple(np.asarray(parameters, dtype=float))
        if key not in cache:
            cache[key] = build_tmm_response(
                model_wavelengths,
                spectra,
                qe_wavelengths,
                qe_values,
                d_grid,
                key,
                integration,
                fixed_hgcdte=fixed_hgcdte,
            )
        return cache[key]

    def optical_penalty(parameters):
        return args.optical_prior_weight * float(
            np.mean(((np.asarray(parameters) - tmm_center) / TMM_SIGMAS) ** 2)
        )

    empirical = fit_global_entropy_sqrt(
        observations,
        empirical_response,
        d_grid,
        initial_scale=2.0,
        initial_scales=(2.0,),
        max_scale=10.0,
        reference_thickness_nm=122.7,
        reference_time_days=4.65,
        reset_day=18.0,
        entropy_weight=0.01,
        smoothness_weight=4.64158883361,
        max_node_multiplier=2.0,
        step_mask=step_mask,
        fit_initial_thickness=True,
        initial_thickness_starts=parse_float_list(args.initial_thickness_starts),
        max_initial_thickness_nm=args.max_initial_thickness_nm,
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_ftol=args.optimizer_ftol,
    )
    tmm = fit_global_entropy_sqrt(
        observations,
        None,
        d_grid,
            initial_scale=empirical["parameter"],
        initial_scales=(empirical["parameter"],),
        max_scale=10.0,
        reference_thickness_nm=122.7,
        reference_time_days=4.65,
        reset_day=18.0,
        entropy_weight=0.01,
        smoothness_weight=4.64158883361,
        max_node_multiplier=2.0,
        step_mask=step_mask,
        initial_thickness_nm=empirical["initial_thickness_nm"],
        fit_initial_thickness=True,
        initial_thickness_starts=(empirical["initial_thickness_nm"],),
        max_initial_thickness_nm=args.max_initial_thickness_nm,
        response_factory=response_factory,
        optical_initial_parameters=tmm_center,
        optical_parameter_starts=(tmm_center,),
        optical_parameter_bounds=tmm_bounds,
        optical_penalty=optical_penalty,
        initial_node_multipliers=empirical["node_multipliers"],
        optimizer_maxiter=args.optimizer_maxiter,
        optimizer_ftol=args.optimizer_ftol,
    )

    validation_epochs = (
        [int(value) for value in parse_float_list(args.validation_epochs)]
        if args.validation_epochs.strip()
        else []
    )
    if any(epoch < 0 or epoch >= len({row["repeat"] for row in observations}) for epoch in validation_epochs):
        raise ValueError("Validation epochs must be valid zero-based epoch numbers")
    validation_rows = run_validation(
        observations,
        validation_epochs,
        empirical_response,
        d_grid,
        response_factory,
        step_mask,
        args,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(
        args.output_dir / "empirical_predictions.csv",
        observations,
        empirical,
        args.sed_mode,
        step_mask=step_mask,
    )
    write_prediction_csv(
        args.output_dir / "tmm_predictions.csv",
        observations,
        tmm,
        args.sed_mode,
        step_mask=step_mask,
    )
    write_nodes(
        args.output_dir / "tmm_nodes.csv",
        (("empirical", empirical), ("tmm", tmm)),
    )
    validation_path = args.output_dir / "repeat_blocked_validation.csv"
    write_validation(validation_path, validation_rows)
    if validation_rows:
        validation_note = (
            "Validation mean held-out RMSE: "
            + ", ".join(
                f"{label}={np.mean([row['held_out_rmse_log'] for row in validation_rows if row['model'] == label]):.10g}"
                for label in ("empirical", "tmm")
            )
            + "."
        )
    else:
        validation_note = "No withheld-epoch validation was requested in this first one-pixel TMM run."
    write_summary(
        args.output_dir / "single_pixel_tmm_fit.md",
        observations,
        empirical,
        tmm,
        tmm["optical_parameters"],
        validation_note,
        tmm_center,
        args.hgcdte_model,
    )
    write_plot(args.output_dir / "single_pixel_tmm_fit.png", observations, empirical, tmm)
    print(
        f"Empirical: d0={empirical['initial_thickness_nm']:.8g} nm; "
        f"rate={empirical['mean_rate_nm_day']:.8g} nm/day; "
        f"RMS={empirical['pixel_rms_log']:.8g}"
    )
    print(
        f"TMM: d0={tmm['initial_thickness_nm']:.8g} nm; "
        f"rate={tmm['mean_rate_nm_day']:.8g} nm/day; "
        f"RMS={tmm['pixel_rms_log']:.8g}"
    )
    print(f"TMM parameters: {tmm['optical_parameters']}")
    print(f"Wrote {args.output_dir / 'single_pixel_tmm_fit.md'}")


if __name__ == "__main__":
    main()
