"""Fit a central WFI01 tile with a fixed WFI10 rate trajectory.

This diagnostic retains the production count-corrected sufficient statistics
only where both pre- and post-operational temperature-gain terms are zero, then
holds the WFI10 median rate trajectory fixed while refitting the target optics.
The six original reference rows are retained because their log changes are
zero and they anchor the sufficient statistics for each LED.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import fit_wfi_tmm_production as production
from benchmark_wfi_fixed_optics import _load_reference_values
from fit_wfi_fixed_rate_optical_refit import (
    VARIABLE_SITES,
    _bounded_reduced_least_squares,
    _write_csv,
)
from plot_wfi_tile_rate_counterfactual import load_source_median_node_rates
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    detector_output_root,
    positive_integer,
    tile_statistics_path,
)


DEFAULT_DETECTOR = "WFI01"
DEFAULT_SOURCE_DETECTOR = "WFI10"
DEFAULT_TILE_ID = 136
TEMPERATURE_TOLERANCE = 1.0e-12
MAXENT_WEIGHT = 0.1


def _scalar(values, name):
    return float(np.asarray(values[name], dtype=float).reshape(-1)[0])


def _read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _temperature_free(value):
    return abs(float(value or 0.0)) <= TEMPERATURE_TOLERANCE


def write_filtered_statistics(source_path, corrections_path, output_path, tile_id):
    source_rows = [
        row for row in _read_rows(source_path) if int(row["tile_id"]) == int(tile_id)
    ]
    if not source_rows:
        raise ValueError(f"No rows found for tile {tile_id} in {source_path}")
    corrections = {
        (int(row["tile_id"]), int(row["group"])): row
        for row in _read_rows(corrections_path)
    }
    reference_groups = {}
    for row in sorted(source_rows, key=lambda item: int(item["group"])):
        reference_groups.setdefault(int(row["led"]), int(row["group"]))

    selected = []
    excluded = []
    for row in source_rows:
        key = (int(row["tile_id"]), int(row["group"]))
        correction = corrections.get(key)
        if correction is None:
            raise ValueError(f"Missing correction row for {key}")
        is_reference = int(row["group"]) in reference_groups.values()
        has_no_temperature_gain = (
            _temperature_free(correction.get("log_gain_correction"))
            and _temperature_free(correction.get("temperature_gain_log_correction"))
        )
        if is_reference or has_no_temperature_gain:
            selected.append(row)
        else:
            excluded.append(row)

    selected.sort(key=lambda item: int(item["group"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    return {
        "source_rows": len(source_rows),
        "selected_rows": len(selected),
        "excluded_rows": len(excluded),
        "reference_groups": sorted(reference_groups.values()),
        "selected_groups": sorted({int(row["group"]) for row in selected}),
        "excluded_groups": sorted({int(row["group"]) for row in excluded}),
        "selected_rows_by_led": {
            str(led): sum(int(row["led"]) == led for row in selected)
            for led in range(1, 7)
        },
        "excluded_rows_by_led": {
            str(led): sum(int(row["led"]) == led for row in excluded)
            for led in range(1, 7)
        },
        "temperature_fields": [
            "log_gain_correction",
            "temperature_gain_log_correction",
        ],
    }


def _metrics(data, raw_observed, raw_prediction):
    residual = np.asarray(raw_observed) - np.asarray(raw_prediction)
    valid = np.ones(residual.size, dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    weights = np.asarray(data["counts"], dtype=float).reshape(-1)[valid]
    log_residual = residual[valid]
    percent_residual = 100.0 * np.expm1(log_residual)
    return {
        "residual_rms_log": float(np.sqrt(np.mean(log_residual**2))),
        "weighted_residual_rms_percent": float(
            np.sqrt(np.sum(weights * percent_residual**2) / np.sum(weights))
        ),
        "max_abs_residual_percent": float(np.max(np.abs(percent_residual))),
        "data_objective": float(
            0.5
            * production.default_likelihood_weight(data["counts"])
            * np.sum(weights * log_residual**2)
            / production.LOG_NOISE_SIGMA**2
        ),
    }


def _raw_observed(data):
    return (
        np.asarray(data["sums"], dtype=float)
        / np.asarray(data["counts"], dtype=float)
    ).reshape(-1)


def _raw_prediction(data, prediction, values):
    post = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    return np.asarray(prediction, dtype=float).reshape(-1) - (
        _scalar(values, "post_reset_tile_log_offset") * post
    )


def _plot(path, data, raw_observed, full_prediction, fixed_prediction, source_rate):
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    observed = 100.0 * np.expm1(raw_observed)
    full = 100.0 * np.expm1(full_prediction)
    fixed = 100.0 * np.expm1(fixed_prediction)
    full_residual = 100.0 * np.expm1(raw_observed - full_prediction)
    fixed_residual = 100.0 * np.expm1(raw_observed - fixed_prediction)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.get_cmap("tab10", 6)
    for led in range(1, 7):
        mask = leds == led
        color = colors(led - 1)
        axes[0].plot(days[mask], observed[mask], "o", color=color, ms=3.0)
        axes[0].plot(days[mask], full[mask], color=color, linewidth=1.0)
        axes[0].plot(
            days[mask],
            fixed[mask],
            color=color,
            linestyle="--",
            linewidth=1.1,
        )
        axes[1].plot(days[mask], full_residual[mask], color=color, linewidth=1.0)
        axes[1].plot(
            days[mask],
            fixed_residual[mask],
            color=color,
            linestyle="--",
            linewidth=1.1,
        )
    for axis in axes:
        axis.axvline(float(data["reset_day"]), color="0.35", linestyle=":")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Log-change equivalent (%)")
    axes[0].set_title("Central WFI01 tile using only no-temperature-gain groups")
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("Observed - model (%)")
    axes[1].set_xlabel("Days since first WFI01 exposure")
    axes[1].set_title("Solid = full production fit; dashed = fixed WFI10 median-rate fit")
    style_handles = [
        Line2D([], [], marker="o", linestyle="", color="black", label="Observed"),
        Line2D([], [], color="black", label="Full production fit"),
        Line2D([], [], color="black", linestyle="--", label="Fixed WFI10 median-rate fit"),
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
        "WFI01 tile 136 (grid 8,8): temperature-dependent gain groups excluded",
        fontsize=12,
    )
    axes[0].text(
        0.01,
        0.98,
        f"Fixed WFI10 median node rate: {np.median(source_rate):.4f} nm/day",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corrections-dir", type=Path, default=None)
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--source-detector", default=DEFAULT_SOURCE_DETECTOR)
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-id", type=int, default=DEFAULT_TILE_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = args.detector.upper()
    source_detector = args.source_detector.upper()
    if args.tile_id < 0:
        raise ValueError("--tile-id must be non-negative")

    detector_output = detector_output_root(args.root, detector, args.tile_size)
    source_statistics = tile_statistics_path(
        detector_output / "tile_statistics",
        detector,
        args.tile_size,
        args.tile_id,
    )
    corrections_dir = args.corrections_dir or args.root.parent / "corrections"
    corrections_path = corrections_dir / (
        f"{detector.lower()}_count_space_unified_temperature_corrections.csv"
    )
    full_fit_dir = detector_output / "fits" / f"tile_{args.tile_id:03d}"
    full_fit_path = full_fit_dir / "posterior_samples.npz"
    output_dir = args.output_dir or full_fit_dir / "fixed_rate_no_temperature_gain"
    # The production loader derives the detector name from this filename.
    filtered_statistics = output_dir / (
        f"{detector.lower()}_tile_log_statistics_{args.tile_size}_tile_"
        f"{args.tile_id:03d}.csv"
    )
    selection = write_filtered_statistics(
        source_statistics,
        corrections_path,
        filtered_statistics,
        args.tile_id,
    )

    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data = production.load_tile_statistics(filtered_statistics)
    full_data = production.load_tile_statistics(source_statistics)
    data["reset_day"] = float(full_data["reset_day"])
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)

    with np.load(full_fit_path, allow_pickle=False) as full_archive:
        full_archive_values = {
            name: np.asarray(full_archive[name]) for name in full_archive.files
        }
        full_prediction = np.asarray(
            full_archive["_map_least_squares_prediction"], dtype=float
        ).reshape(-1)

    source_output = detector_output_root(args.root, source_detector, args.tile_size)
    source_rates_full = load_source_median_node_rates(
        source_output,
        len(full_data["node_days"]),
    )
    source_rates = np.interp(
        np.asarray(data["node_days"], dtype=float),
        np.asarray(full_data["node_days"], dtype=float),
        source_rates_full,
    )

    reference_values = _load_reference_values(
        data,
        full_fit_path,
        source_statistics,
    )
    reference_values["np_log_rate_nodes"] = np.log(source_rates).reshape(
        np.asarray(reference_values["np_log_rate_nodes"]).shape
    )
    rate_reference = np.asarray(reference_values["np_log_rate_nodes"], dtype=float)
    fixed_model = production.build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_reference,
        entropy_weight=MAXENT_WEIGHT,
        rate_prior_log_mean=rate_reference,
        constant_rate=False,
        fit_incidence_angle=True,
    )
    fixed_values, diagnostics = _bounded_reduced_least_squares(
        fixed_model.production_residuals,
        reference_values,
        VARIABLE_SITES,
        jax,
        jnp,
    )
    fixed_state = fixed_model.production_state(fixed_values)
    fixed_prediction = np.asarray(fixed_state["prediction"], dtype=float).reshape(-1)

    filtered_base = _raw_observed(data)
    fixed_prediction = _raw_prediction(data, fixed_prediction, fixed_values)
    full_group_to_index = {
        int(group): index for index, group in enumerate(full_data["groups"])
    }
    full_offset = _scalar(full_archive_values, "post_reset_tile_log_offset")
    full_post = np.asarray(full_data["days"], dtype=float) >= float(full_data["reset_day"])
    full_raw_prediction = np.asarray(full_prediction, dtype=float) - full_offset * full_post
    selected_indices = [full_group_to_index[int(group)] for group in data["groups"]]
    full_filtered_prediction = full_raw_prediction[selected_indices]
    full_filtered_metrics = _metrics(data, filtered_base, full_filtered_prediction)
    fixed_metrics = _metrics(data, filtered_base, fixed_prediction)
    csv_path = output_dir / "fixed_rate_vs_full_fit.csv"
    plot_path = output_dir / "fixed_rate_vs_full_fit.png"
    _write_csv(
        csv_path,
        data,
        filtered_base,
        full_filtered_prediction,
        fixed_prediction,
    )
    _plot(
        plot_path,
        data,
        filtered_base,
        full_filtered_prediction,
        fixed_prediction,
        source_rates,
    )

    source_time_average = float(
        np.trapezoid(source_rates, np.asarray(data["node_days"], dtype=float))
        / (
            float(data["node_days"][-1])
            - float(data["node_days"][0])
        )
    )

    payload = {
        "diagnostic": "wfi01_no_temperature_gain_fixed_rate",
        "detector": detector,
        "source_detector": source_detector,
        "tile_id": int(args.tile_id),
        "tile_row": 8,
        "tile_col": 8,
        "source_statistics": str(source_statistics.resolve()),
        "filtered_statistics": str(filtered_statistics.resolve()),
        "corrections": str(corrections_path.resolve()),
        "full_fit": str(full_fit_path.resolve()),
        "source_fit_root": str((source_output / "fits").resolve()),
        "reset_day_used": float(data["reset_day"]),
        "selection": selection,
        "fixed_rate_definition": (
            "median over source-detector tiles of node_instantaneous_rate_nm_day; "
            "interpolated onto the retained target repeat nodes and held fixed"
        ),
        "fixed_rate_summary_nm_day": {
            "first": float(source_rates[0]),
            "last": float(source_rates[-1]),
            "median": float(np.median(source_rates)),
            "time_average": source_time_average,
            "minimum": float(np.min(source_rates)),
            "maximum": float(np.max(source_rates)),
        },
        "fixed_rate_nodes_nm_day": source_rates.tolist(),
        "variable_sites": list(VARIABLE_SITES),
        "full_fit_on_selected_groups": full_filtered_metrics,
        "fixed_rate_fit": {
            **fixed_metrics,
            "least_squares_residual_cost": float(diagnostics["residual_cost"]),
            "initial_least_squares_residual_cost": float(
                diagnostics["initial_residual_cost"]
            ),
            "function_evaluations": int(diagnostics["nfev"]),
            "converged": bool(diagnostics["converged"]),
            "message": diagnostics["message"],
        },
        "optimized_parameters": {
            "initial_thickness_nm": float(
                np.exp(_scalar(fixed_values, "initial_thickness_log_nm"))
            ),
            "reset_initial_thickness_nm": float(
                np.exp(_scalar(fixed_values, "reset_initial_thickness_log_nm"))
            ),
            "post_reset_tile_log_offset": _scalar(fixed_values, "post_reset_tile_log_offset"),
            "incidence_angle_deg": _scalar(fixed_values, "incidence_angle_deg"),
            "absolute_hgcdte_n_log_scale": _scalar(
                fixed_values,
                "absolute_hgcdte_n_log_scale",
            ),
            "absolute_hgcdte_k_log_scale": _scalar(
                fixed_values,
                "absolute_hgcdte_k_log_scale",
            ),
        },
        "outputs": {
            "directory": str(output_dir.resolve()),
            "csv": str(csv_path.resolve()),
            "plot": str(plot_path.resolve()),
        },
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote filtered statistics: {filtered_statistics}")
    print(f"Wrote diagnostic outputs: {output_dir}")
    print(f"Wrote fixed-rate comparison: {csv_path}")
    print(f"Wrote comparison plot: {plot_path}")
    print(f"Selected rows: {selection['selected_rows']} / {selection['source_rows']}")
    print(
        "Full-fit selected-group RMS: "
        f"{full_filtered_metrics['weighted_residual_rms_percent']:.9g}%"
    )
    print(
        "Fixed-rate-fit RMS: "
        f"{fixed_metrics['weighted_residual_rms_percent']:.9g}%"
    )
    print(
        "Fixed WFI10 rate median/time-average: "
        f"{payload['fixed_rate_summary_nm_day']['median']:.9g}/"
        f"{payload['fixed_rate_summary_nm_day']['time_average']:.9g} nm/day"
    )


if __name__ == "__main__":
    main()
