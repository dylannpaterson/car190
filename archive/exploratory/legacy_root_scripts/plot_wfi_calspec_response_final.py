#!/usr/bin/env python3
"""Write clean final plots for a selected CALSPEC response-modulation MAP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_blackbody_sed as base
import fit_wfi_calspec_response_modulation as response_fit


ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_DIR = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_098/plots/calspec_response_map_strength_sweep_1e4_1e5_spacing100"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_098/plots/calspec_response_map_final_strength5e4"
)
DEFAULT_RESPONSE_RATIO = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_098/plots/ice_response_ratio_20261015_vs_20260928.csv"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strength", type=float, default=50000.0)
    parser.add_argument("--data", type=Path, default=base.DEFAULT_DATA)
    parser.add_argument("--response-ratio", type=Path, default=DEFAULT_RESPONSE_RATIO)
    parser.add_argument("--bandpass-file", type=Path, default=base.ROMAN_EFFECTIVE_AREA_PATH)
    parser.add_argument("--calspec-spectrum", type=Path, default=base.DEFAULT_CALSPEC_SPECTRUM)
    parser.add_argument("--extrapolation-temperature", type=float, default=46750.0)
    parser.add_argument("--response-node-spacing", type=float, default=100.0)
    return parser.parse_args(argv)


def effective_area_upper(area_wavelengths_nm, bandpasses):
    return max(
        2200.0,
        max(
            float(np.max(area_wavelengths_nm[np.asarray(area) >= 0.1 * np.max(area)]))
            for area in bandpasses.values()
        ),
    )


def add_filter_spans(axis, area_wavelengths_nm, bandpasses):
    for name in base.ROMAN_FILTERS:
        area = np.asarray(bandpasses[name], dtype=float)
        support = area >= 0.1 * np.max(area)
        if not np.any(support):
            continue
        lower = float(np.min(area_wavelengths_nm[support]))
        upper = float(np.max(area_wavelengths_nm[support]))
        axis.axvspan(lower, upper, color="0.5", alpha=0.045, linewidth=0.0)
        axis.text(
            0.5 * (lower + upper),
            0.98,
            name,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.35",
        )


def read_selected_sweep(sweep_dir, strength):
    with (Path(sweep_dir) / "map_strength_sweep_summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary = next(
        (row for row in summary_rows if np.isclose(float(row["strength"]), strength)),
        None,
    )
    if summary is None:
        raise ValueError(f"Strength {strength:g} is not in the sweep: {sweep_dir}")
    with (Path(sweep_dir) / "map_strength_sweep_responses.csv").open(newline="") as handle:
        response_rows = list(csv.DictReader(handle))
    with (Path(sweep_dir) / "map_strength_sweep_filter_models.csv").open(newline="") as handle:
        filter_rows = list(csv.DictReader(handle))
    response_field = f"map_response_strength_{strength:.9g}"
    model_field = f"map_model_strength_{strength:.9g}"
    if not response_rows or response_field not in response_rows[0]:
        raise ValueError(f"Missing response column {response_field}")
    if not filter_rows or model_field not in filter_rows[0]:
        raise ValueError(f"Missing filter-model column {model_field}")
    return summary, response_rows, filter_rows, response_field, model_field


def main(argv=None):
    args = parse_args(argv)
    if args.strength <= 0.0 or args.extrapolation_temperature <= 0.0:
        raise ValueError("Strength and extrapolation temperature must be positive")
    summary_row, response_rows, filter_rows, response_field, model_field = read_selected_sweep(
        args.sweep_dir,
        args.strength,
    )
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
        args.response_node_spacing,
    )
    filters = [row["filter"] for row in observations]
    unmodulated_model = response_fit.bandpass_ratios(
        fixed_sed,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        filters,
        np.zeros(len(response_node_wavelengths), dtype=float),
        response_node_wavelengths,
    )
    selected_model_by_filter = {
        row["filter"]: float(row[model_field]) for row in filter_rows
    }
    selected_model = np.asarray([selected_model_by_filter[name] for name in filters])
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    pivot = np.asarray([row["pivot_wavelength_nm"] for row in observations])
    order = np.argsort(pivot)
    residual_sigma = (observed - selected_model) / sigma
    wavelength_grid = np.asarray([float(row["wavelength_nm"]) for row in response_rows])
    unmodulated_grid = np.asarray(
        [float(row["unmodulated_response_ratio"]) for row in response_rows]
    )
    selected_grid = np.asarray([float(row[response_field]) for row in response_rows])
    wavelength_upper = effective_area_upper(area_wavelengths_nm, bandpasses)
    plot_mask = (wavelength_grid >= 450.0) & (wavelength_grid <= wavelength_upper)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response_csv = output_dir / "final_selected_map_spectral_response.csv"
    with response_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("wavelength_nm", "unmodulated_response_ratio", "selected_map_response_ratio"))
        writer.writerows(zip(wavelength_grid, unmodulated_grid, selected_grid))

    fit_csv = output_dir / "final_selected_map_filter_fit.csv"
    with fit_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "filter",
                "pivot_wavelength_nm",
                "observed_cr_ratio",
                "ratio_sigma",
                "unmodulated_model",
                "selected_map_model",
                "residual_sigma",
            )
        )
        for index, row in enumerate(observations):
            writer.writerow(
                (
                    row["filter"],
                    row["pivot_wavelength_nm"],
                    row["cr_ratio"],
                    row["ratio_sigma"],
                    f"{unmodulated_model[index]:.9g}",
                    f"{selected_model[index]:.9g}",
                    f"{residual_sigma[index]:.9g}",
                )
            )

    figure, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    add_filter_spans(axis, area_wavelengths_nm, bandpasses)
    axis.plot(
        wavelength_grid[plot_mask],
        unmodulated_grid[plot_mask],
        color="#6F6F6F",
        linestyle="--",
        linewidth=1.7,
        label="Unmodulated ice response",
    )
    axis.plot(
        wavelength_grid[plot_mask],
        selected_grid[plot_mask],
        color="#0072B2",
        linewidth=2.5,
        label=f"Final MAP response (max-ent strength={args.strength:.0e})",
    )
    axis.set_xlim(450.0, wavelength_upper)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Oct 15 / Sep 28 ice response ratio")
    axis.set_title("Final CALSPEC-weighted modulated spectral response")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    response_plot = output_dir / "final_modulated_spectral_response.png"
    figure.savefig(response_plot, dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.0)},
        constrained_layout=True,
    )
    axes[0].errorbar(
        pivot[order],
        observed[order],
        yerr=sigma[order],
        fmt="o",
        color="black",
        capsize=3,
        markersize=6,
        label="Digitized CR ratios",
    )
    axes[0].plot(
        pivot[order],
        unmodulated_model[order],
        color="#6F6F6F",
        linestyle="--",
        marker="o",
        linewidth=1.5,
        label="Unmodulated model",
    )
    axes[0].plot(
        pivot[order],
        selected_model[order],
        color="#0072B2",
        marker="o",
        linewidth=2.2,
        label=f"Final MAP model (strength={args.strength:.0e})",
    )
    axes[0].set_xlim(450.0, wavelength_upper)
    axes[0].set_ylabel("CR ratio")
    axes[0].set_title(
        f"Final CALSPEC-weighted response fit; MAP $\\chi^2$={float(summary_row['chi_squared']):.2f}"
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].axhspan(-1.0, 1.0, color="#56B4E9", alpha=0.18, label="$\\pm1\\sigma$")
    axes[1].axhline(0.0, color="0.25", linewidth=1.0)
    axes[1].plot(
        pivot[order],
        residual_sigma[order],
        color="#D55E00",
        marker="o",
        linewidth=1.8,
    )
    axes[1].set_xlim(450.0, wavelength_upper)
    axes[1].set_xlabel("Roman effective wavelength (nm)")
    axes[1].set_ylabel("Residual / $\\sigma$")
    axes[1].set_ylim(-2.0, 2.0)
    axes[1].grid(alpha=0.2)
    fit_plot = output_dir / "final_modulated_response_fit.png"
    figure.savefig(fit_plot, dpi=220)
    plt.close(figure)

    final_summary = {
        "selected_strength": float(args.strength),
        "map_chi_squared": float(summary_row["chi_squared"]),
        "max_abs_log_amplitude": float(summary_row["max_abs_log_amplitude"]),
        "max_multiplicative_deviation_percent": float(
            100.0 * np.expm1(float(summary_row["max_abs_log_amplitude"]))
        ),
        "node_spacing_nm": float(args.response_node_spacing),
        "node_count": len(response_node_wavelengths),
        "calspec_spectrum": str(Path(args.calspec_spectrum).resolve()),
        "extrapolation_temperature_k": float(args.extrapolation_temperature),
        "response_csv": str(response_csv.resolve()),
        "fit_csv": str(fit_csv.resolve()),
        "response_plot": str(response_plot.resolve()),
        "fit_plot": str(fit_plot.resolve()),
        "source_sweep": str(Path(args.sweep_dir).resolve()),
    }
    (output_dir / "final_selected_map_summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
