#!/usr/bin/env python3
"""Sweep maximum-entropy strengths for a fixed-CALSPEC response model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import fit_wfi_blackbody_sed as base
import fit_wfi_calspec_response_modulation as response_fit


ROOT = Path(__file__).resolve().parent
DEFAULT_RESPONSE_RATIO = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_098/plots/ice_response_ratio_20261015_vs_20260928.csv"
)
DEFAULT_DATA = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_136/plots/blackbody_sed_digitized.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_098/plots/calspec_response_map_strength_sweep_spacing100"
)
DEFAULT_STRENGTHS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--response-ratio", type=Path, default=DEFAULT_RESPONSE_RATIO)
    parser.add_argument("--bandpass-file", type=Path, default=base.ROMAN_EFFECTIVE_AREA_PATH)
    parser.add_argument("--calspec-spectrum", type=Path, default=base.DEFAULT_CALSPEC_SPECTRUM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extrapolation-temperature", type=float, default=46750.0)
    parser.add_argument("--response-modulation-node-spacing", type=float, default=100.0)
    parser.add_argument("--response-maxent-log-bound", type=float, default=5.0)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_STRENGTHS),
        help="Positive maximum-entropy strengths to sweep.",
    )
    return parser.parse_args(argv)


def map_fit(
    strength,
    starts,
    fixed_sed,
    observations,
    area_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    response_ratio,
    response_node_wavelengths,
    entropy_wavelengths,
    entropy_weights,
    log_bound,
    maxiter,
):
    objective_args = (
        fixed_sed,
        observations,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        response_node_wavelengths,
        None,
        strength,
        log_bound,
        entropy_wavelengths,
        entropy_weights,
    )

    def objective(theta):
        value = response_fit.log_probability(theta, *objective_args)
        return 1.0e100 if not np.isfinite(value) else -value

    bounds = [(-log_bound + 1.0e-8, log_bound - 1.0e-8)] * len(
        response_node_wavelengths
    )
    candidates = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "maxls": 50,
            },
        )
        candidates.append(result)
    result = min(candidates, key=lambda candidate: float(candidate.fun))
    theta = np.asarray(result.x, dtype=float)
    filters = [row["filter"] for row in observations]
    model = response_fit.bandpass_ratios(
        fixed_sed,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        filters,
        theta,
        response_node_wavelengths,
    )
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    chi_squared = float(np.sum(((observed - model) / sigma) ** 2))
    entropy_score = base.relative_maxent_score(
        entropy_wavelengths,
        theta,
        response_node_wavelengths,
        np.asarray([], dtype=float),
        entropy_weights,
    )
    modulated_response = response_ratio * response_fit.response_modulation_factor(
        response_wavelengths_nm,
        theta,
        response_node_wavelengths,
    )
    return {
        "strength": float(strength),
        "theta": theta,
        "model": model,
        "modulated_response": modulated_response,
        "chi_squared": chi_squared,
        "entropy_score": float(entropy_score),
        "log_posterior": float(-result.fun),
        "max_abs_log_amplitude": float(np.max(np.abs(theta))),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
    }


def write_csvs(
    output_dir,
    results,
    observations,
    response_wavelengths_nm,
    response_ratio,
):
    summary_path = output_dir / "map_strength_sweep_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "strength",
                "chi_squared",
                "entropy_score",
                "log_posterior",
                "max_abs_log_amplitude",
                "success",
                "iterations",
                "function_evaluations",
                "message",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: result[key]
                    for key in writer.fieldnames
                }
            )

    model_path = output_dir / "map_strength_sweep_filter_models.csv"
    with model_path.open("w", newline="") as handle:
        fields = ["filter", "pivot_wavelength_nm", "observed_cr_ratio", "ratio_sigma"]
        fields.extend(f"map_model_strength_{result['strength']:.9g}" for result in results)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(observations):
            output = {
                "filter": row["filter"],
                "pivot_wavelength_nm": f"{row['pivot_wavelength_nm']:.9g}",
                "observed_cr_ratio": f"{row['cr_ratio']:.9g}",
                "ratio_sigma": f"{row['ratio_sigma']:.9g}",
            }
            output.update(
                {
                    f"map_model_strength_{result['strength']:.9g}": f"{result['model'][index]:.9g}"
                    for result in results
                }
            )
            writer.writerow(output)

    response_path = output_dir / "map_strength_sweep_responses.csv"
    fields = ["wavelength_nm", "unmodulated_response_ratio"]
    fields.extend(
        f"map_response_strength_{result['strength']:.9g}" for result in results
    )
    with response_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, wavelength in enumerate(response_wavelengths_nm):
            output = {
                "wavelength_nm": f"{wavelength:.9g}",
                "unmodulated_response_ratio": f"{response_ratio[index]:.9g}",
            }
            output.update(
                {
                    f"map_response_strength_{result['strength']:.9g}": f"{result['modulated_response'][index]:.9g}"
                    for result in results
                }
            )
            writer.writerow(output)
    return summary_path, model_path, response_path


def write_plots(
    output_dir,
    results,
    observations,
    response_wavelengths_nm,
    response_ratio,
    area_wavelengths_nm,
    bandpasses,
):
    wavelength_upper = max(
        2200.0,
        max(
            float(np.max(area_wavelengths_nm[np.asarray(area) >= 0.1 * np.max(area)]))
            for area in bandpasses.values()
        ),
    )
    response_mask = (response_wavelengths_nm >= 450.0) & (
        response_wavelengths_nm <= wavelength_upper
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(results)))

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(
        response_wavelengths_nm[response_mask],
        response_ratio[response_mask],
        color="#6F6F6F",
        linestyle="--",
        linewidth=1.5,
        label="Unmodulated response",
    )
    for color, result in zip(colors, results):
        axis.plot(
            response_wavelengths_nm[response_mask],
            result["modulated_response"][response_mask],
            color=color,
            linewidth=1.7,
            label=f"MAP max-ent strength={result['strength']:.0g}",
        )
    axis.set_xlim(450.0, wavelength_upper)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Ice response ratio")
    axis.set_title("MAP maximum-entropy response-strength sweep")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    response_plot = output_dir / "map_strength_sweep_responses.png"
    figure.savefig(response_plot, dpi=180)
    plt.close(figure)

    pivot = np.asarray([row["pivot_wavelength_nm"] for row in observations])
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    order = np.argsort(pivot)
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.errorbar(
        pivot[order],
        observed[order],
        yerr=sigma[order],
        fmt="o",
        color="black",
        capsize=3,
        label="Digitized CR ratios",
    )
    for color, result in zip(colors, results):
        axis.plot(
            pivot[order],
            result["model"][order],
            color=color,
            linewidth=1.7,
            label=(
                f"strength={result['strength']:.0g}; "
                f"MAP $\\chi^2$={result['chi_squared']:.2f}"
            ),
        )
    axis.set_xlim(450.0, wavelength_upper)
    axis.set_xlabel("Roman effective wavelength (nm)")
    axis.set_ylabel("Oct 15 / Sep 28 CR ratio")
    axis.set_title("MAP maximum-entropy response-strength sweep")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    model_plot = output_dir / "map_strength_sweep_filter_models.png"
    figure.savefig(model_plot, dpi=180)
    plt.close(figure)
    return response_plot, model_plot


def main(argv=None):
    args = parse_args(argv)
    if args.extrapolation_temperature <= 0.0:
        raise ValueError("Extrapolation temperature must be positive")
    if args.response_modulation_node_spacing <= 0.0:
        raise ValueError("Response node spacing must be positive")
    if args.response_maxent_log_bound <= 0.0:
        raise ValueError("Response log bound must be positive")
    if args.maxiter < 1:
        raise ValueError("MAP maxiter must be positive")
    if not args.strengths or any(strength <= 0.0 for strength in args.strengths):
        raise ValueError("All maximum-entropy strengths must be positive")
    strengths = sorted(set(float(strength) for strength in args.strengths))

    area_wavelengths_nm, bandpasses = base.load_bandpasses(args.bandpass_file)
    response_wavelengths_nm, response_ratio = base.load_response_ratio(args.response_ratio)
    observations = base.load_observations(args.data)
    calspec = base.load_calspec_spectrum(args.calspec_spectrum)
    if calspec is None:
        raise FileNotFoundError(f"CALSPEC spectrum not found: {args.calspec_spectrum}")
    calspec_wavelengths_nm, calspec_shape = calspec
    fixed_sed = lambda wavelengths: response_fit.calspec_blackbody_extrapolated_shape(
        wavelengths,
        calspec_wavelengths_nm,
        calspec_shape,
        args.extrapolation_temperature,
    )
    response_node_wavelengths = base.build_full_area_modulation_nodes(
        area_wavelengths_nm,
        bandpasses,
        args.response_modulation_node_spacing,
    )
    entropy_wavelengths = area_wavelengths_nm
    entropy_weights = np.zeros_like(area_wavelengths_nm, dtype=float)
    for area in bandpasses.values():
        entropy_weights += np.asarray(area, dtype=float) / np.max(area)
    entropy_weights *= fixed_sed(entropy_wavelengths)

    zero = np.zeros(len(response_node_wavelengths), dtype=float)
    starts = [zero]
    results = []
    for strength in strengths:
        result = map_fit(
            strength,
            starts,
            fixed_sed,
            observations,
            area_wavelengths_nm,
            bandpasses,
            response_wavelengths_nm,
            response_ratio,
            response_node_wavelengths,
            entropy_wavelengths,
            entropy_weights,
            args.response_maxent_log_bound,
            args.maxiter,
        )
        results.append(result)
        starts = [zero, result["theta"]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path, model_path, response_path = write_csvs(
        output_dir,
        results,
        observations,
        response_wavelengths_nm,
        response_ratio,
    )
    response_plot, model_plot = write_plots(
        output_dir,
        results,
        observations,
        response_wavelengths_nm,
        response_ratio,
        area_wavelengths_nm,
        bandpasses,
    )
    summary = {
        "data": str(Path(args.data).resolve()),
        "response_ratio": str(Path(args.response_ratio).resolve()),
        "calspec_spectrum": str(Path(args.calspec_spectrum).resolve()),
        "extrapolation_temperature_k": float(args.extrapolation_temperature),
        "response_node_spacing_nm": float(args.response_modulation_node_spacing),
        "response_node_count": len(response_node_wavelengths),
        "response_maxent_log_bound": float(args.response_maxent_log_bound),
        "strengths": strengths,
        "summary_csv": str(summary_path.resolve()),
        "filter_model_csv": str(model_path.resolve()),
        "response_csv": str(response_path.resolve()),
        "response_plot": str(response_plot.resolve()),
        "filter_model_plot": str(model_plot.resolve()),
        "results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"theta", "model", "modulated_response"}
            }
            for result in results
        ],
    }
    (output_dir / "map_strength_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
