"""Align operating-temperature rate trajectories by time since decon reset."""

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

import plot_wfi_operating_temperature_rate_only as temperature_filter


DECON_DAYS = (0.0, 12.907397928240743, 38.79784510416666)
CYCLE_NAMES = ("cycle1", "cycle2", "cycle3")
CYCLE_LABELS = ("Cycle 1", "Cycle 2", "Cycle 3")
CYCLE_COLORS = ("#0072b2", "#d55e00", "#009e73")
DEFAULT_AGES = (3.0, 5.0, 7.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--telemetry-path", type=Path, required=True)
    parser.add_argument("--temperature-coefficients", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-subdir", default="relative_response")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--temperature-tolerance-k", type=float, default=0.1)
    parser.add_argument(
        "--assume-missing-operating-temperature",
        action="store_true",
        help="Retain repeats with missing telemetry when observed values are in-window.",
    )
    parser.add_argument(
        "--age-days",
        dest="ages",
        type=float,
        nargs="+",
        default=list(DEFAULT_AGES),
        help="Ages at which to report cross-cycle offsets; default: 3 5 7.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def cycle_mask(days, cycle_index):
    lower = DECON_DAYS[cycle_index]
    upper = DECON_DAYS[cycle_index + 1] if cycle_index < 2 else np.inf
    return (days >= lower) & (days < upper)


def cycle_age_values(tile, cycle_index):
    mask = cycle_mask(tile["filtered_days"], cycle_index)
    return (
        tile["filtered_days"][mask] - DECON_DAYS[cycle_index],
        tile["filtered_rates"][mask],
    )


def common_age_grid(tiles):
    minima = []
    maxima = []
    for tile in tiles:
        for cycle_index in range(3):
            ages, _rates = cycle_age_values(tile, cycle_index)
            if ages.size:
                minima.append(float(np.min(ages)))
                maxima.append(float(np.max(ages)))
    if not minima or max(minima) >= min(maxima):
        raise ValueError(f"No common tau interval for {CYCLE_NAMES[cycle_index]}")
    lower = max(max(minima), 2.5)
    upper = min(maxima)
    if lower >= upper:
        raise ValueError(f"No common tau interval above 2.5 days for {CYCLE_NAMES[cycle_index]}")
    return np.arange(lower, upper + 0.001, 0.25)


def interpolate_cycle_rate(tile, cycle_index, ages):
    cycle_ages, cycle_rates = cycle_age_values(tile, cycle_index)
    if cycle_ages.size == 0:
        return np.full(ages.shape, np.nan)
    order = np.argsort(cycle_ages, kind="stable")
    return np.interp(ages, cycle_ages[order], cycle_rates[order])


def percentile_curve(tiles, cycle_index, ages):
    values = np.asarray(
        [interpolate_cycle_rate(tile, cycle_index, ages) for tile in tiles],
        dtype=float,
    )
    return values, np.percentile(values, [16, 50, 84], axis=0)


def tile_slopes(tiles, cycle_index, ages):
    slopes = []
    for tile in tiles:
        rates = interpolate_cycle_rate(tile, cycle_index, ages)
        if np.all(np.isfinite(rates)):
            slopes.append(np.polyfit(ages, rates, 1)[0])
    return np.asarray(slopes, dtype=float)


def write_slope_summary(path, detector_data):
    fields = (
        "detector",
        "cycle",
        "tiles",
        "tau_min_days",
        "tau_max_days",
        "rate_slope_p16_nm_day_per_day",
        "rate_slope_median_nm_day_per_day",
        "rate_slope_p84_nm_day_per_day",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector, data in detector_data.items():
            for cycle_index, cycle in enumerate(CYCLE_NAMES):
                ages = data["grids"][cycle_index]
                slopes = tile_slopes(data["tiles"], cycle_index, ages)
                writer.writerow(
                    {
                        "detector": detector,
                        "cycle": cycle,
                        "tiles": slopes.size,
                        "tau_min_days": np.min(ages),
                        "tau_max_days": np.max(ages),
                        "rate_slope_p16_nm_day_per_day": np.percentile(slopes, 16),
                        "rate_slope_median_nm_day_per_day": np.percentile(slopes, 50),
                        "rate_slope_p84_nm_day_per_day": np.percentile(slopes, 84),
                    }
                )


def write_offset_summary(path, detector_data, ages):
    fields = (
        "detector",
        "tau_days",
        "cycle2_minus_cycle1_p16_nm_day",
        "cycle2_minus_cycle1_median_nm_day",
        "cycle2_minus_cycle1_p84_nm_day",
        "cycle3_minus_cycle2_p16_nm_day",
        "cycle3_minus_cycle2_median_nm_day",
        "cycle3_minus_cycle2_p84_nm_day",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector, data in detector_data.items():
            for index, age in enumerate(ages):
                rates = []
                for tile in data["tiles"]:
                    rates.append(
                        [
                            interpolate_cycle_rate(tile, cycle_index, np.asarray([age]))[0]
                            for cycle_index in range(3)
                        ]
                    )
                rates = np.asarray(rates, dtype=float)
                cycle2_minus_cycle1 = rates[:, 1] - rates[:, 0]
                cycle3_minus_cycle2 = rates[:, 2] - rates[:, 1]
                writer.writerow(
                    {
                        "detector": detector,
                        "tau_days": age,
                        "cycle2_minus_cycle1_p16_nm_day": np.percentile(cycle2_minus_cycle1, 16),
                        "cycle2_minus_cycle1_median_nm_day": np.percentile(cycle2_minus_cycle1, 50),
                        "cycle2_minus_cycle1_p84_nm_day": np.percentile(cycle2_minus_cycle1, 84),
                        "cycle3_minus_cycle2_p16_nm_day": np.percentile(cycle3_minus_cycle2, 16),
                        "cycle3_minus_cycle2_median_nm_day": np.percentile(cycle3_minus_cycle2, 50),
                        "cycle3_minus_cycle2_p84_nm_day": np.percentile(cycle3_minus_cycle2, 84),
                    }
                )


def plot_tau_figure(path, detector_data, dpi):
    detectors = tuple(detector_data)
    figure, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=True, sharey=True)
    plotted_percentiles = []
    for axis, detector in zip(axes.flat, detectors):
        data = detector_data[detector]
        for cycle_index, (label, color) in enumerate(zip(CYCLE_LABELS, CYCLE_COLORS)):
            ages = data["grids"][cycle_index]
            _values, percentiles = percentile_curve(data["tiles"], cycle_index, ages)
            plotted_percentiles.append(percentiles[[0, 2]])
            axis.fill_between(
                ages,
                percentiles[0],
                percentiles[2],
                color=color,
                alpha=0.18,
            )
            axis.plot(ages, percentiles[1], color=color, linewidth=1.5)
        slopes = []
        for cycle_index in range(3):
            cycle_ages = data["grids"][cycle_index]
            slopes.append(np.median(tile_slopes(data["tiles"], cycle_index, cycle_ages)))
        axis.set_title(
            f"{detector}\n"
            f"slopes: {slopes[0]:+.2f}, {slopes[1]:+.2f}, {slopes[2]:+.2f}",
            fontsize=8,
        )
        axis.axhline(0.0, color="0.45", linewidth=0.5)
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=7)
    plotted_percentiles = np.concatenate(plotted_percentiles, axis=1)
    y_min = float(np.min(plotted_percentiles))
    y_max = float(np.max(plotted_percentiles))
    y_padding = max((y_max - y_min) * 0.06, 0.1)
    for axis in axes.flat:
        axis.set_ylim(y_min - y_padding, y_max + y_padding)
    for axis in axes[-1]:
        axis.set_xlabel("tau since decon [days]", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Rate [nm/day]", fontsize=8)
    figure.legend(
        handles=[
            Line2D([], [], color=color, linewidth=2.0, label=label)
            for label, color in zip(CYCLE_LABELS, CYCLE_COLORS)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Operating-temperature rate trajectories aligned by time since decon\n"
        "Lines are tile medians; bands are tile 16/84th percentiles; titles show median linear slopes",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.93), pad=0.8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main():
    args = parse_args()
    if args.tile_size <= 0 or args.temperature_tolerance_k <= 0 or args.dpi <= 0:
        raise ValueError("tile-size, temperature-tolerance-k, and dpi must be positive")
    if any(age <= 0.0 for age in args.ages):
        raise ValueError("Ages must be positive")

    telemetry_temperatures, telemetry_info = temperature_filter.load_telemetry(
        args.telemetry_path
    )
    operating_temperatures = temperature_filter.load_operating_temperatures(
        args.temperature_coefficients,
        args.tile_size,
    )
    detector_data = {}
    detectors = temperature_filter.DETECTORS
    for detector in detectors:
        tiles, _statistics_path = temperature_filter.load_detector_data(
            detector,
            Path(args.production_root) / detector.lower(),
            args.tile_size,
            telemetry_temperatures,
            operating_temperatures,
            args.temperature_tolerance_k,
            args.assume_missing_operating_temperature,
            args.fit_subdir,
        )
        detector_data[detector] = {
            "tiles": tiles,
            "grids": [common_age_grid(tiles) for _cycle_index in range(3)],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slope_path = args.output_dir / "rate_vs_tau_slope_summary.csv"
    offset_path = args.output_dir / "rate_vs_tau_cycle_offset_summary.csv"
    plot_path = args.output_dir / "rate_vs_tau_percentiles_all_detectors.png"
    write_slope_summary(slope_path, detector_data)
    write_offset_summary(offset_path, detector_data, args.ages)
    plot_tau_figure(plot_path, detector_data, args.dpi)
    metadata = {
        "detectors": detectors,
        "ages_days": args.ages,
        "decon_days": DECON_DAYS,
        "temperature_tolerance_k": args.temperature_tolerance_k,
        "assume_missing_operating_temperature": args.assume_missing_operating_temperature,
        "fit_subdir": args.fit_subdir,
        "telemetry": telemetry_info,
        "plot": str(plot_path.resolve()),
        "slope_summary": str(slope_path.resolve()),
        "offset_summary": str(offset_path.resolve()),
    }
    metadata_path = args.output_dir / "rate_vs_tau_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {plot_path}")
    print(f"Wrote {slope_path}")
    print(f"Wrote {offset_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
