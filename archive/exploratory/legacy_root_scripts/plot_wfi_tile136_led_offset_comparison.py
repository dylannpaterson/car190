"""Compare a LED-offset fit with the main production MAP fits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import fit_wfi_tmm_production as production


DETECTORS = ("wfi01", "wfi10")
LED_COLORS = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#f0e442",
    "#56b4e9",
)
RESET_DAY = 12.907397928240743


def load_fit(root, detector, fit_dir):
    detector_root = Path(root) / detector
    statistics_path = (
        detector_root
        / "n_prior_sigma_0p05"
        / "tile_statistics"
        / f"{detector}_tile_log_statistics_256_tile_136.csv"
    )
    fit_path = detector_root / fit_dir
    with (fit_path / "tile_group_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    data = {
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "observed": np.asarray(
            [float(row["observed_log_change"]) for row in rows]
        ),
        "model": np.asarray([float(row["model_log_change"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }
    with np.load(fit_path / "posterior_samples.npz", allow_pickle=False) as archive:
        data["node_rates"] = np.asarray(
            archive["node_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
        data["node_thickness"] = np.asarray(
            archive["node_ice_thickness"], dtype=float
        ).reshape(-1)
        data["mean_rate"] = float(
            np.asarray(archive["tile_mean_rate_nm_day"]).reshape(-1)[0]
        )
        data["initial_rate"] = float(
            np.asarray(archive["tile_initial_rate_nm_day"]).reshape(-1)[0]
        )
        data["final_rate"] = float(
            np.asarray(archive["tile_final_rate_nm_day"]).reshape(-1)[0]
        )
        data["rms"] = float(
            np.asarray(archive["map_group_rms_log"]).reshape(-1)[0]
        )
        data["led_offsets"] = (
            np.asarray(archive["led_log_offsets"], dtype=float).reshape(-1)
            if "led_log_offsets" in archive
            else np.full(6, np.nan)
        )
    data["node_days"] = np.asarray(
        production.load_tile_statistics(statistics_path)["node_days"], dtype=float
    )
    return data


def format_axes(axis):
    axis.axvline(RESET_DAY, color="0.4", linestyle="--", linewidth=0.8)
    axis.grid(alpha=0.2)


def plot_data_model_residuals(
    series,
    output,
    offset_fit_name,
    offset_label,
    baseline_fit_name,
    baseline_label,
):
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20, 8),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    fit_order = (offset_fit_name, baseline_fit_name)
    fit_titles = {
        offset_fit_name: offset_label,
        baseline_fit_name: baseline_label,
    }
    for row, detector in enumerate(DETECTORS):
        for column, fit_name in enumerate(fit_order):
            values = series[detector][fit_name]
            data_axis = axes[row, column]
            residual_axis = axes[row, column + 2]
            for led, color in enumerate(LED_COLORS, start=1):
                mask = values["led"] == led
                order = np.argsort(values["days"][mask])
                data_axis.scatter(
                    values["days"][mask],
                    1000.0 * values["observed"][mask],
                    color=color,
                    s=7,
                    alpha=0.4,
                    edgecolors="none",
                )
                data_axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["model"][mask][order],
                    color=color,
                    linewidth=0.9,
                )
                residual_axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["residual"][mask][order],
                    color=color,
                    marker=".",
                    markersize=2.5,
                    linewidth=0.7,
                )
            format_axes(data_axis)
            format_axes(residual_axis)
            residual_axis.axhline(0.0, color="black", linewidth=0.7)
            data_axis.set_title(
                f"{detector.upper()} | {fit_titles[fit_name]}\n"
                f"RMS={values['rms']:.6f}"
            )
            residual_axis.set_title(
                f"{detector.upper()} | {fit_titles[fit_name]}\n"
                f"RMS={values['rms']:.6f}"
            )
            if row == 1:
                data_axis.set_xlabel("Days since first exposure")
                residual_axis.set_xlabel("Days since first exposure")
            if column == 0:
                data_axis.set_ylabel(f"{detector.upper()}\nlog change [10^-3]")
                residual_axis.set_ylabel(
                    f"{detector.upper()}\nobserved - model [10^-3]"
                )
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="data"),
        Line2D([], [], color="black", linewidth=1.2, label="model"),
        Line2D([], [], color="0.4", linestyle="--", linewidth=0.8, label="ice reset"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, loc="best")
    residual_handles = [
        Line2D([], [], color=color, linewidth=1.2, label=f"LED {led}")
        for led, color in enumerate(LED_COLORS, start=1)
    ]
    axes[0, 2].legend(
        handles=residual_handles,
        ncol=3,
        frameon=False,
        fontsize=8,
        loc="best",
    )
    figure.suptitle(f"WFI tile 136: {offset_label} versus {baseline_label}")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rate_trajectory_difference(
    series,
    output,
    offset_fit_name,
    offset_label,
    baseline_fit_name,
    baseline_label,
):
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 8),
        sharex="col",
        constrained_layout=True,
    )
    for row, detector in enumerate(DETECTORS):
        offset = series[detector][offset_fit_name]
        production_fit = series[detector][baseline_fit_name]
        rate_axis, thickness_axis, difference_axis = axes[row]
        rate_axis.plot(
            offset["node_days"],
            offset["node_rates"],
            color="#d55e00",
            linewidth=1.8,
            label=f"{offset_label} ({offset['initial_rate']:.2f} nm/day)",
        )
        rate_axis.plot(
            production_fit["node_days"],
            production_fit["node_rates"],
            color="#0072b2",
            linewidth=1.8,
            label=baseline_label,
        )
        thickness_axis.plot(
            offset["node_days"],
            offset["node_thickness"],
            color="#d55e00",
            linewidth=1.8,
            label=offset_label,
        )
        thickness_axis.plot(
            production_fit["node_days"],
            production_fit["node_thickness"],
            color="#0072b2",
            linewidth=1.8,
            label=baseline_label,
        )
        difference = offset["node_thickness"] - production_fit["node_thickness"]
        difference_axis.plot(
            offset["node_days"],
            difference,
            color="#542788",
            linewidth=1.8,
        )
        for axis in (rate_axis, thickness_axis, difference_axis):
            format_axes(axis)
        difference_axis.axhline(0.0, color="black", linewidth=0.7)
        rate_axis.set_title(f"{detector.upper()} rate")
        thickness_axis.set_title(f"{detector.upper()} ice trajectory")
        difference_axis.set_title(f"{detector.upper()} trajectory difference")
        rate_axis.set_ylabel("rate [nm/day]")
        thickness_axis.set_ylabel("ice thickness [nm]")
        difference_axis.set_ylabel("offset model - baseline [nm]")
        rate_axis.legend(frameon=False, fontsize=8, loc="best")
        thickness_axis.legend(frameon=False, fontsize=8, loc="best")
        if row == 1:
            for axis in (rate_axis, thickness_axis, difference_axis):
                axis.set_xlabel("Days since first exposure")
    figure.suptitle("WFI tile 136: rate and trajectory comparison", fontsize=14)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(series, output, offset_fit_name, baseline_fit_name):
    fields = [
        "detector",
        "fit",
        "group_rms_log",
        "mean_rate_nm_day",
        "initial_rate_nm_day",
        "final_rate_nm_day",
        *[f"led_{led}_log_offset" for led in range(1, 7)],
    ]
    with Path(output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector in DETECTORS:
            for fit_name in (offset_fit_name, baseline_fit_name):
                values = series[detector][fit_name]
                row = {
                    "detector": detector.upper(),
                    "fit": fit_name,
                    "group_rms_log": values["rms"],
                    "mean_rate_nm_day": values["mean_rate"],
                    "initial_rate_nm_day": values["initial_rate"],
                    "final_rate_nm_day": values["final_rate"],
                }
                row.update(
                    {
                        f"led_{led}_log_offset": values["led_offsets"][led - 1]
                        for led in range(1, 7)
                    }
                )
                writer.writerow(row)


def main(
    root,
    output_dir,
    offset_fit_dir="diagnostics/tile_136_led_offset_constant_rate/constant_rate",
    offset_fit_name="led_offset",
    offset_label="Constant rate + LED offsets",
    output_prefix="tile136_led_offset",
    baseline_fit_dir="diagnostics/production_arhgcdte_v2/fits/tile_136",
    baseline_fit_name="production",
    baseline_label="Main production MAP",
):
    series = {}
    for detector in DETECTORS:
        series[detector] = {
            offset_fit_name: load_fit(
                root,
                detector,
                offset_fit_dir,
            ),
            baseline_fit_name: load_fit(
                root,
                detector,
                baseline_fit_dir,
            ),
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data_model_residuals(
        series,
        output_dir / f"{output_prefix}_data_model_residuals.png",
        offset_fit_name,
        offset_label,
        baseline_fit_name,
        baseline_label,
    )
    plot_rate_trajectory_difference(
        series,
        output_dir / f"{output_prefix}_rate_trajectory_difference.png",
        offset_fit_name,
        offset_label,
        baseline_fit_name,
        baseline_label,
    )
    write_summary(
        series,
        output_dir / f"{output_prefix}_comparison.csv",
        offset_fit_name,
        baseline_fit_name,
    )
    print(f"Wrote {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--offset-fit-dir",
        default="diagnostics/tile_136_led_offset_constant_rate/constant_rate",
        help="Detector-relative fit directory for the LED-offset comparison.",
    )
    parser.add_argument(
        "--offset-fit-name",
        default="led_offset",
        help="CSV fit name for the LED-offset comparison.",
    )
    parser.add_argument(
        "--offset-label",
        default="Constant rate + LED offsets",
        help="Plot label for the LED-offset comparison.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tile136_led_offset",
        help="Prefix for generated plot and CSV filenames.",
    )
    parser.add_argument(
        "--baseline-fit-dir",
        default="diagnostics/production_arhgcdte_v2/fits/tile_136",
        help="Detector-relative fit directory for the baseline comparison.",
    )
    parser.add_argument(
        "--baseline-fit-name",
        default="production",
        help="CSV fit name for the baseline comparison.",
    )
    parser.add_argument(
        "--baseline-label",
        default="Main production MAP",
        help="Plot label for the baseline comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.root,
        args.output_dir,
        offset_fit_dir=args.offset_fit_dir,
        offset_fit_name=args.offset_fit_name,
        offset_label=args.offset_label,
        output_prefix=args.output_prefix,
        baseline_fit_dir=args.baseline_fit_dir,
        baseline_fit_name=args.baseline_fit_name,
        baseline_label=args.baseline_label,
    )
