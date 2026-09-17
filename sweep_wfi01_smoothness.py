"""Sweep the smoothness weight for the WFI01 all-pixel NP trajectory.

The sweep is intended as a diagnostic. It fits one selected LED and SED mode
at each smoothness weight, writes the exact exposure-level data term and node
curvature, and marks an L-curve knee candidate without changing the primary
fit outputs.
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
    MODES,
    broadband_response_curves,
    fit_periodic_model,
    read_model_table,
    soc_sqrt_thickness,
)
from analyze_wfi01_all_pixel_soc_model import (
    ROOT,
    STATISTICS_PATH,
    fit_global_entropy_sqrt,
    fit_global_soc_sqrt,
    load_all_pixel_statistics,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed


DEFAULT_OUTPUT_DIR = ROOT / "wfi01_led1_only_latest"
DEFAULT_WEIGHTS = np.logspace(-4.0, 3.0, 25)


def parse_weights(value):
    weights = np.asarray([float(item) for item in value.split(",")], dtype=float)
    if len(weights) < 3 or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Provide at least three positive finite smoothness weights")
    return np.unique(weights)


def l_curve_knee_index(data_terms, roughness):
    """Return the interior point farthest from the log-log endpoint chord."""
    x = np.log10(np.maximum(np.asarray(data_terms, dtype=float), 1.0e-15))
    y = np.log10(np.maximum(np.asarray(roughness, dtype=float), 1.0e-15))
    points = np.column_stack((x, y))
    spans = np.ptp(points, axis=0)
    spans[spans == 0.0] = 1.0
    points = (points - points.min(axis=0)) / spans
    chord = points[-1] - points[0]
    chord_length = np.linalg.norm(chord)
    if chord_length == 0.0:
        return len(points) // 2, np.zeros(len(points), dtype=float)
    distances = np.abs(
        chord[0] * (points[:, 1] - points[0, 1])
        - chord[1] * (points[:, 0] - points[0, 0])
    ) / chord_length
    distances[[0, -1]] = -np.inf
    return int(np.argmax(distances)), distances


def write_sweep_csv(path, rows, knee_index):
    fields = [
        "smoothness_weight",
        "knee_candidate",
        "normalized_data_term",
        "between_rms_log",
        "pixel_rms_log",
        "smoothness_penalty",
        "entropy_penalty",
        "penalized_objective",
        "mean_rate_nm_day",
        "max_thickness_nm",
        "min_node_multiplier",
        "max_node_multiplier",
        "max_interval_rate_nm_day",
        "optimizer_converged_starts",
        "optimizer_evaluations",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "smoothness_weight": f'{row["smoothness_weight"]:.12g}',
                    "knee_candidate": index == knee_index,
                    "normalized_data_term": f'{row["fit"]["normalized_data_term"]:.12g}',
                    "between_rms_log": f'{row["fit"]["between_rms_log"]:.12g}',
                    "pixel_rms_log": f'{row["fit"]["pixel_rms_log"]:.12g}',
                    "smoothness_penalty": f'{row["fit"]["smoothness_penalty"]:.12g}',
                    "entropy_penalty": f'{row["fit"]["entropy_penalty"]:.12g}',
                    "penalized_objective": f'{row["fit"]["penalized_objective"]:.12g}',
                    "mean_rate_nm_day": f'{row["fit"]["mean_rate_nm_day"]:.12g}',
                    "max_thickness_nm": f'{np.max(row["fit"]["thickness_nm"]):.12g}',
                    "min_node_multiplier": f'{np.min(row["fit"]["node_multipliers"]):.12g}',
                    "max_node_multiplier": f'{np.max(row["fit"]["node_multipliers"]):.12g}',
                    "max_interval_rate_nm_day": f'{np.max(row["interval_rates"]):.12g}',
                    "optimizer_converged_starts": row["fit"]["optimizer_converged_starts"],
                    "optimizer_evaluations": row["fit"]["optimizer_evaluations"],
                }
            )


def write_knee_nodes(path, fit):
    node_days = np.asarray(fit["node_days"])
    node_thickness = np.asarray(fit["node_thickness_nm"])
    rates = np.divide(np.diff(node_thickness), np.diff(node_days))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "repeat",
                "days",
                "node_multiplier",
                "fitted_thickness_nm",
                "interval_rate_nm_day",
            ),
        )
        writer.writeheader()
        for index, (repeat, days, multiplier, thickness) in enumerate(
            zip(
                fit["node_repeats"],
                node_days,
                fit["node_multipliers"],
                node_thickness,
            )
        ):
            writer.writerow(
                {
                    "repeat": int(repeat),
                    "days": f"{days:.12g}",
                    "node_multiplier": f"{multiplier:.12g}",
                    "fitted_thickness_nm": f"{thickness:.12g}",
                    "interval_rate_nm_day": (
                        "" if index == 0 else f"{rates[index - 1]:.12g}"
                    ),
                }
            )


def plot_sweep(path, rows, knee_index, sed_mode, led):
    weights = np.asarray([row["smoothness_weight"] for row in rows])
    data_terms = np.asarray(
        [row["fit"]["normalized_data_term"] for row in rows]
    )
    roughness = np.asarray(
        [row["fit"]["smoothness_penalty"] for row in rows]
    )
    between_rms = np.asarray([row["fit"]["between_rms_log"] for row in rows])
    max_rates = np.asarray([np.max(row["interval_rates"]) for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].loglog(data_terms, roughness, "o-", color="#111111")
    axes[0].scatter(
        data_terms[knee_index],
        roughness[knee_index],
        s=100,
        color="#c1121f",
        zorder=3,
        label=f"knee candidate, weight={weights[knee_index]:.3g}",
    )
    for index, weight in enumerate(weights):
        axes[0].annotate(
            f"{weight:.2g}",
            (data_terms[index], roughness[index]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=7,
        )
    axes[0].set_xlabel("Normalized between-exposure SSE")
    axes[0].set_ylabel("Mean squared log-node curvature")
    axes[0].set_title("Smoothness L-curve")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2, which="both")

    axes[1].semilogx(weights, between_rms, "o-", color="#111111", label="Between RMS")
    axes[1].set_xlabel("Smoothness weight")
    axes[1].set_ylabel("Between-exposure RMS (log)")
    rate_axis = axes[1].twinx()
    rate_axis.semilogx(
        weights,
        max_rates,
        "s--",
        color="#c1121f",
        label="Maximum interval rate",
    )
    rate_axis.set_ylabel("Maximum interval rate (nm/day)", color="#c1121f")
    rate_axis.tick_params(axis="y", labelcolor="#c1121f")
    axes[1].axvline(weights[knee_index], color="#c1121f", alpha=0.5)
    axes[1].set_title(f"LED {led}, {sed_mode} SED")
    axes[1].grid(alpha=0.2, which="both")
    handles, labels = axes[1].get_legend_handles_labels()
    rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
    axes[1].legend(handles + rate_handles, labels + rate_labels, frameon=False, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--led", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--sed-mode", choices=MODES, default="clipped")
    parser.add_argument("--statistics-path", type=Path, default=STATISTICS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--smoothness-values",
        default=",".join(f"{value:.12g}" for value in DEFAULT_WEIGHTS),
    )
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--max-scale", type=float, default=20.0)
    parser.add_argument("--scale-steps", type=int, default=4001)
    parser.add_argument("--thickness-grid-steps", type=int, default=2401)
    parser.add_argument("--sqrt-ref-thickness-nm", type=float, default=122.7)
    parser.add_argument("--sqrt-ref-time-days", type=float, default=4.65)
    parser.add_argument("--reset-day", type=float, default=18.0)
    parser.add_argument("--max-node-multiplier", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = parse_weights(args.smoothness_values)
    observations = [
        row
        for row in load_all_pixel_statistics(args.statistics_path)
        if row["led"] == args.led
    ]
    if not observations:
        raise ValueError(f"No observations found for LED {args.led}")

    model_wavelengths, thicknesses, ratios = read_model_table(args.model_path)
    coefficients, omegas = fit_periodic_model(model_wavelengths, thicknesses, ratios)
    model = (model_wavelengths, thicknesses, coefficients, omegas)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
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
    responses = broadband_response_curves(
        model, spectra, qe_wavelengths, qe_values, d_grid
    )
    sqrt_fit = fit_global_soc_sqrt(
        observations,
        responses[args.sed_mode],
        d_grid,
        args.max_scale,
        args.scale_steps,
        args.sqrt_ref_thickness_nm,
        args.sqrt_ref_time_days,
        args.reset_day,
    )

    rows = []
    for smoothness_weight in weights:
        fit = fit_global_entropy_sqrt(
            observations,
            responses[args.sed_mode],
            d_grid,
            sqrt_fit["parameter"],
            sqrt_fit["coarse_local_minimum_scales"],
            args.max_scale,
            args.sqrt_ref_thickness_nm,
            args.sqrt_ref_time_days,
            args.reset_day,
            args.entropy_weight,
            float(smoothness_weight),
            args.max_node_multiplier,
        )
        interval_rates = np.diff(fit["node_thickness_nm"]) / np.diff(fit["node_days"])
        rows.append(
            {
                "smoothness_weight": float(smoothness_weight),
                "fit": fit,
                "interval_rates": interval_rates,
            }
        )
        print(
            f"smoothness={smoothness_weight:.8g}; "
            f"between RMS={fit['between_rms_log']:.8g}; "
            f"curvature={fit['smoothness_penalty']:.8g}; "
            f"max rate={np.max(interval_rates):.8g} nm/day"
        )

    data_terms = [row["fit"]["normalized_data_term"] for row in rows]
    roughness = [row["fit"]["smoothness_penalty"] for row in rows]
    knee_index, _ = l_curve_knee_index(data_terms, roughness)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "wfi01_smoothness_sweep.csv"
    plot_path = args.output_dir / "wfi01_smoothness_sweep.png"
    node_path = args.output_dir / "wfi01_smoothness_knee_nodes.csv"
    write_sweep_csv(csv_path, rows, knee_index)
    write_knee_nodes(node_path, rows[knee_index]["fit"])
    plot_sweep(plot_path, rows, knee_index, args.sed_mode, args.led)
    print(
        f"Knee candidate: smoothness={rows[knee_index]['smoothness_weight']:.8g}; "
        f"between RMS={rows[knee_index]['fit']['between_rms_log']:.8g}; "
        f"curvature={rows[knee_index]['fit']['smoothness_penalty']:.8g}"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {node_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
