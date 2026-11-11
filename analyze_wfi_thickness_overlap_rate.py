"""Plot rate versus ice thickness only where reset cycles overlap."""

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
    parser.add_argument("--grid-points", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def cycle_mask(days, cycle_index):
    lower = DECON_DAYS[cycle_index]
    upper = DECON_DAYS[cycle_index + 1] if cycle_index < 2 else np.inf
    return (days >= lower) & (days < upper)


def cycle_xy(tile, cycle_index):
    mask = cycle_mask(tile["filtered_days"], cycle_index)
    x_values = tile["filtered_thickness"][mask]
    y_values = tile["filtered_rates"][mask]
    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]
    y_values = y_values[order]
    unique_x, inverse = np.unique(x_values, return_inverse=True)
    if unique_x.size == x_values.size:
        return unique_x, y_values
    unique_y = np.asarray(
        [np.mean(y_values[inverse == index]) for index in range(unique_x.size)],
        dtype=float,
    )
    return unique_x, unique_y


def overlap_range(tiles):
    cycle_ranges = []
    for cycle_index in range(3):
        ranges = [
            (float(np.min(cycle_xy(tile, cycle_index)[0])), float(np.max(cycle_xy(tile, cycle_index)[0])))
            for tile in tiles
        ]
        cycle_ranges.append(ranges)
    all_tile_lower = max(max(values[0] for values in ranges) for ranges in cycle_ranges)
    all_tile_upper = min(min(values[1] for values in ranges) for ranges in cycle_ranges)
    if all_tile_lower < all_tile_upper:
        return all_tile_lower, all_tile_upper, "all_tiles"

    aggregate_lower = max(
        min(value[0] for value in ranges) for ranges in cycle_ranges
    )
    aggregate_upper = min(
        max(value[1] for value in ranges) for ranges in cycle_ranges
    )
    if aggregate_lower >= aggregate_upper:
        raise ValueError("No overlapping thickness range across cycles")
    return aggregate_lower, aggregate_upper, "aggregate_tiles"


def interpolate_rate(x_values, y_values, thickness_grid):
    mask = (thickness_grid >= x_values[0]) & (thickness_grid <= x_values[-1])
    result = np.full(thickness_grid.shape, np.nan, dtype=float)
    result[mask] = np.interp(thickness_grid[mask], x_values, y_values)
    return result


def percentile_curves(tiles, thickness_grid, cycle_index):
    values = np.asarray(
        [
            interpolate_rate(*cycle_xy(tile, cycle_index), thickness_grid)
            for tile in tiles
        ],
        dtype=float,
    )
    percentiles = np.full((3, thickness_grid.size), np.nan, dtype=float)
    counts = np.count_nonzero(np.isfinite(values), axis=0)
    for index in range(thickness_grid.size):
        observed = values[:, index]
        observed = observed[np.isfinite(observed)]
        if observed.size:
            percentiles[:, index] = np.percentile(observed, [16, 50, 84])
    return percentiles, counts


def write_summary(path, detector_data):
    fields = (
        "detector",
        "cycle",
        "overlap_mode",
        "overlap_min_nm",
        "overlap_max_nm",
        "thickness_nm",
        "rate_p16_nm_day",
        "rate_median_nm_day",
        "rate_p84_nm_day",
        "tiles_available",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector, data in detector_data.items():
            for cycle_index, cycle in enumerate(CYCLE_NAMES):
                percentiles, counts = data["curves"][cycle_index]
                for index, thickness in enumerate(data["grid"]):
                    writer.writerow(
                        {
                            "detector": detector,
                            "cycle": cycle,
                            "overlap_mode": data["overlap_mode"],
                            "overlap_min_nm": data["overlap_min_nm"],
                            "overlap_max_nm": data["overlap_max_nm"],
                            "thickness_nm": thickness,
                            "rate_p16_nm_day": percentiles[0, index],
                            "rate_median_nm_day": percentiles[1, index],
                            "rate_p84_nm_day": percentiles[2, index],
                            "tiles_available": counts[index],
                        }
                    )


def plot_figure(path, detector_data, dpi):
    detectors = tuple(detector_data)
    figure, axes = plt.subplots(3, 6, figsize=(18, 9), sharey=True)
    plotted = []
    for axis, detector in zip(axes.flat, detectors):
        data = detector_data[detector]
        for cycle_index, (label, color) in enumerate(zip(CYCLE_LABELS, CYCLE_COLORS)):
            percentiles, counts = data["curves"][cycle_index]
            valid = np.isfinite(percentiles[1])
            if not np.any(valid):
                continue
            plotted.append(percentiles[:, valid][[0, 2]])
            axis.fill_between(
                data["grid"][valid],
                percentiles[0, valid],
                percentiles[2, valid],
                color=color,
                alpha=0.18,
            )
            axis.plot(
                data["grid"][valid],
                percentiles[1, valid],
                color=color,
                linewidth=1.5,
            )
        axis.set_title(
            f"{detector}\n"
            f"overlap {data['overlap_min_nm']:.0f}-{data['overlap_max_nm']:.0f} nm",
            fontsize=8,
        )
        axis.set_xlabel("ice thickness [nm]", fontsize=8)
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=7)
    if plotted:
        plotted = np.concatenate(plotted, axis=1)
        y_min = float(np.nanmin(plotted))
        y_max = float(np.nanmax(plotted))
        padding = max((y_max - y_min) * 0.06, 0.1)
        for axis in axes.flat:
            axis.set_ylim(y_min - padding, y_max + padding)
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
        "Operating-temperature rate versus ice thickness in the reset-cycle overlap\n"
        "Lines are tile medians; bands are tile 16/84th percentiles; no extrapolation",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.93), pad=0.8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main():
    args = parse_args()
    if args.tile_size <= 0 or args.temperature_tolerance_k <= 0:
        raise ValueError("tile-size and temperature-tolerance-k must be positive")
    if args.grid_points < 3:
        raise ValueError("grid-points must be at least 3")

    telemetry_temperatures, telemetry_info = temperature_filter.load_telemetry(
        args.telemetry_path
    )
    operating_temperatures = temperature_filter.load_operating_temperatures(
        args.temperature_coefficients,
        args.tile_size,
    )
    detector_data = {}
    for detector in temperature_filter.DETECTORS:
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
        lower, upper, mode = overlap_range(tiles)
        grid = np.linspace(lower, upper, args.grid_points)
        detector_data[detector] = {
            "tiles": tiles,
            "grid": grid,
            "overlap_min_nm": lower,
            "overlap_max_nm": upper,
            "overlap_mode": mode,
            "curves": [percentile_curves(tiles, grid, cycle_index) for cycle_index in range(3)],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "rate_vs_thickness_overlap_percentiles_all_detectors.png"
    summary_path = args.output_dir / "rate_vs_thickness_overlap_percentiles.csv"
    plot_figure(plot_path, detector_data, args.dpi)
    write_summary(summary_path, detector_data)
    metadata = {
        "detectors": temperature_filter.DETECTORS,
        "temperature_tolerance_k": args.temperature_tolerance_k,
        "assume_missing_operating_temperature": args.assume_missing_operating_temperature,
        "fit_subdir": args.fit_subdir,
        "grid_points": args.grid_points,
        "telemetry": telemetry_info,
        "plot": str(plot_path.resolve()),
        "summary": str(summary_path.resolve()),
        "overlap_ranges_nm": {
            detector: [data["overlap_min_nm"], data["overlap_max_nm"]]
            for detector, data in detector_data.items()
        },
        "overlap_modes": {
            detector: data["overlap_mode"] for detector, data in detector_data.items()
        },
    }
    metadata_path = args.output_dir / "rate_vs_thickness_overlap_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
