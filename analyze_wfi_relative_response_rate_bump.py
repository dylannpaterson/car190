"""Measure the full-WFI accumulation-rate excess across a time gap.

For every relative-response tile fit, a shape-preserving cubic spline is interpolated
through the fitted accumulation-rate samples outside the requested day range.
The observed fitted rate and this background are then compared on one common
time grid inside the excluded range.  The focal-plane maps are evaluated at
the full-focal-plane peak time, rather than at an independent peak time for
each tile.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from plot_wfi_full_focal_plane_rates import (
    SCAS,
    draw_rate_mosaic,
    percentile_color_limits,
)
from wfi_production_plot_annotations import load_time_annotations
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    tile_directory_entries,
    tile_grid_size_for_tile_size,
)


RATE_METRICS = (
    ("Rate difference at full-plane peak", "rate_difference_nm_day"),
    ("Percentage difference at full-plane peak", "rate_difference_percent"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detector",
        action="append",
        choices=SCAS,
        default=None,
        help="Detector to include; repeat for multiple detectors. Defaults to all WFI detectors.",
    )
    parser.add_argument("--window-start-day", type=float, default=48.0)
    parser.add_argument("--window-end-day", type=float, default=55.0)
    parser.add_argument(
        "--time-points",
        type=int,
        default=None,
        help="Optional dense evaluation grid; default uses the union of excluded observation times.",
    )
    parser.add_argument(
        "--context-days",
        type=float,
        default=5.0,
        help="Days shown on each side of the excluded interval in the diagnostic plot.",
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    return parser.parse_args()


def _collapse_duplicate_days(days, rates):
    unique_days, inverse = np.unique(days, return_inverse=True)
    if unique_days.size == days.size:
        return unique_days, rates
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=rates)
    return unique_days, sums / counts


def load_tile_rate(path):
    with np.load(path, allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
        rates = np.asarray(
            samples["tile_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
        detector = str(np.asarray(samples["sca_names"]).reshape(-1)[0])
        tile_id = int(np.asarray(samples["tile_ids"]).reshape(-1)[0])
        statistics_path = Path(str(np.asarray(samples["statistics_path"]).item()))
    if days.size != rates.size or days.size < 4:
        raise ValueError(f"Unexpected time/rate shape in {path}")
    finite = np.isfinite(days) & np.isfinite(rates) & (rates > 0.0)
    days, rates = _collapse_duplicate_days(days[finite], rates[finite])
    order = np.argsort(days)
    days = days[order]
    rates = rates[order]
    if days.size < 4 or np.any(np.diff(days) <= 0.0):
        raise ValueError(f"Tile rate samples are not strictly increasing in {path}")
    return detector, tile_id, days, rates, statistics_path


def tile_rate_difference(days, rates, time_grid, window_start, window_end):
    excluded = (days >= window_start) & (days <= window_end)
    training = ~excluded
    if np.count_nonzero(training) < 4 or np.count_nonzero(excluded) < 1:
        raise ValueError("The requested window leaves too few tile rate samples")
    background_spline = PchipInterpolator(days[training], rates[training])
    observed = np.interp(time_grid, days, rates)
    background = np.asarray(background_spline(time_grid), dtype=float)
    if not np.all(np.isfinite(background)) or np.any(background <= 0.0):
        raise ValueError("Spline background is non-finite or non-positive")
    difference = observed - background
    percent = 100.0 * difference / background
    return (
        observed,
        background,
        difference,
        percent,
        int(np.count_nonzero(excluded)),
        background_spline,
    )


def _fit_root(statistics_root, detector):
    return (
        Path(statistics_root)
        / "production"
        / detector.lower()
        / "relative_response"
        / "fits"
    )


def load_detector(statistics_root, detector, tile_grid_size):
    fit_root = _fit_root(statistics_root, detector)
    entries = [
        (tile_id, path / "posterior_samples.npz")
        for tile_id, path in tile_directory_entries(fit_root)
        if (path / "posterior_samples.npz").is_file()
    ]
    tile_ids = complete_tile_ids((tile_id for tile_id, _path in entries))
    expected = tuple(range(tile_grid_size * tile_grid_size))
    if tile_ids != expected:
        raise ValueError(
            f"{detector} contains {len(tile_ids)} tiles, expected "
            f"{len(expected)} for tile size {tile_grid_size}"
        )
    tiles = []
    for tile_id, path in entries:
        archive_detector, archive_tile_id, days, rates, statistics_path = load_tile_rate(path)
        if archive_detector != detector or archive_tile_id != tile_id:
            raise ValueError(f"Archive identity mismatch in {path}")
        tiles.append(
            {
                "detector": detector,
                "tile_id": tile_id,
                "tile_row": tile_id // tile_grid_size,
                "tile_col": tile_id % tile_grid_size,
                "days": days,
                "rates": rates,
                "statistics_path": statistics_path,
            }
        )
    return tiles


def load_temperature_events(detector_tiles):
    reference = None
    for tiles in detector_tiles.values():
        annotations = load_time_annotations(tiles[0]["statistics_path"])
        events = tuple(event for event in annotations.events if event.kind == "temperature")
        if reference is None:
            reference = events
            continue
        if len(events) != len(reference) or any(
            first.label != second.label
            or not np.isclose(first.day, second.day, rtol=0.0, atol=1.0e-10)
            for first, second in zip(reference, events)
        ):
            raise ValueError("Temperature-event timing differs between detectors")
    return reference or ()


def _map_for_time(tiles, field, time_index, tile_grid_size):
    values = np.full((tile_grid_size, tile_grid_size), np.nan, dtype=float)
    for tile in tiles:
        values[tile["tile_row"], tile["tile_col"]] = tile[field][time_index]
    return values


def write_tile_summary(path, tiles, detectors, time_grid, rate_peak_index, percent_peak_index):
    fields = [
        "detector",
        "tile_id",
        "tile_row",
        "tile_col",
        "rate_peak_day",
        "observed_rate_at_rate_peak_nm_day",
        "background_rate_at_rate_peak_nm_day",
        "rate_difference_at_rate_peak_nm_day",
        "rate_difference_at_rate_peak_percent",
        "percent_peak_day",
        "observed_rate_at_percent_peak_nm_day",
        "background_rate_at_percent_peak_nm_day",
        "rate_difference_at_percent_peak_nm_day",
        "rate_difference_at_percent_peak_percent",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector in detectors:
            for tile in tiles[detector]:
                row = {
                    "detector": detector,
                    "tile_id": tile["tile_id"],
                    "tile_row": tile["tile_row"],
                    "tile_col": tile["tile_col"],
                    "rate_peak_day": time_grid[rate_peak_index],
                    "observed_rate_at_rate_peak_nm_day": tile["observed"][rate_peak_index],
                    "background_rate_at_rate_peak_nm_day": tile["background"][rate_peak_index],
                    "rate_difference_at_rate_peak_nm_day": tile["difference"][rate_peak_index],
                    "rate_difference_at_rate_peak_percent": tile["percent"][rate_peak_index],
                    "percent_peak_day": time_grid[percent_peak_index],
                    "observed_rate_at_percent_peak_nm_day": tile["observed"][percent_peak_index],
                    "background_rate_at_percent_peak_nm_day": tile["background"][percent_peak_index],
                    "rate_difference_at_percent_peak_nm_day": tile["difference"][percent_peak_index],
                    "rate_difference_at_percent_peak_percent": tile["percent"][percent_peak_index],
                }
                writer.writerow(row)


def write_detector_summary(
    path, detector_series, time_grid, rate_peak_index, percent_peak_index
):
    fields = [
        "detector",
        "tile_count",
        "rate_peak_day",
        "observed_mean_rate_at_rate_peak_nm_day",
        "background_mean_rate_at_rate_peak_nm_day",
        "rate_difference_at_rate_peak_nm_day",
        "rate_difference_at_rate_peak_percent",
        "percent_peak_day",
        "observed_mean_rate_at_percent_peak_nm_day",
        "background_mean_rate_at_percent_peak_nm_day",
        "rate_difference_at_percent_peak_nm_day",
        "rate_difference_at_percent_peak_percent",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector, series in detector_series.items():
            observed = series["observed"]
            background = series["background"]
            difference = observed - background
            percent = 100.0 * difference / background
            writer.writerow(
                {
                    "detector": detector,
                    "tile_count": len(series["tiles"]),
                    "rate_peak_day": time_grid[rate_peak_index],
                    "observed_mean_rate_at_rate_peak_nm_day": observed[rate_peak_index],
                    "background_mean_rate_at_rate_peak_nm_day": background[rate_peak_index],
                    "rate_difference_at_rate_peak_nm_day": difference[rate_peak_index],
                    "rate_difference_at_rate_peak_percent": percent[rate_peak_index],
                    "percent_peak_day": time_grid[percent_peak_index],
                    "observed_mean_rate_at_percent_peak_nm_day": observed[percent_peak_index],
                    "background_mean_rate_at_percent_peak_nm_day": background[percent_peak_index],
                    "rate_difference_at_percent_peak_nm_day": difference[percent_peak_index],
                    "rate_difference_at_percent_peak_percent": percent[percent_peak_index],
                }
            )


def write_time_series(path, time_grid, observed, background, difference, percent):
    fields = [
        "day",
        "observed_mean_rate_nm_day",
        "background_mean_rate_nm_day",
        "rate_difference_nm_day",
        "rate_difference_percent",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, day in enumerate(time_grid):
            writer.writerow(
                {
                    "day": day,
                    "observed_mean_rate_nm_day": observed[index],
                    "background_mean_rate_nm_day": background[index],
                    "rate_difference_nm_day": difference[index],
                    "rate_difference_percent": percent[index],
                }
            )


def plot_time_series(
    path,
    time_grid,
    detector_series,
    observed_mean,
    background_mean,
    difference_mean,
    percent_mean,
    peak_day,
    window_start,
    window_end,
    events,
):
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(detector_series)))
    for color, (detector, series) in zip(colors, detector_series.items()):
        axes[0].plot(
            time_grid,
            series["observed"],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            label=detector,
        )
        axes[0].plot(
            time_grid,
            series["background"],
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.35,
        )
        axes[1].plot(time_grid, series["difference"], color=color, linewidth=0.9, alpha=0.75)
        axes[2].plot(time_grid, series["percent"], color=color, linewidth=0.9, alpha=0.75)
    axes[0].plot(
        time_grid,
        observed_mean,
        color="black",
        linewidth=2.4,
        label="18-detector mean rate",
    )
    axes[0].plot(
        time_grid,
        background_mean,
        color="#D55E00",
        linewidth=2.0,
        linestyle="--",
        label="18-detector mean spline background",
    )
    for axis in axes:
        axis.axvspan(window_start, window_end, color="#E69F00", alpha=0.12)
    axes[0].set_ylabel("Detector mean rate (nm/day)")
    axes[0].legend(frameon=False, fontsize=7, ncol=4)
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        time_grid,
        difference_mean,
        color="black",
        linewidth=2.4,
        label="18-detector mean difference",
    )
    axes[1].axvline(peak_day, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Rate difference (nm/day)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[2].plot(
        time_grid,
        percent_mean,
        color="black",
        linewidth=2.4,
        label="18-detector mean percentage difference",
    )
    axes[2].axvline(peak_day, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Day")
    axes[2].set_ylabel("Percentage difference (%)")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)
    for event in events:
        if not time_grid[0] <= event.day <= time_grid[-1]:
            continue
        color = "#009E73" if event.kind == "temperature" else "#D55E00"
        linestyle = "-." if event.kind == "temperature" else "--"
        for axis in axes:
            axis.axvline(
                event.day,
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
                alpha=0.9,
            )
        axes[0].text(
            event.day,
            0.99,
            event.label,
            transform=axes[0].get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=color,
            backgroundcolor="white",
        )
    figure.suptitle(
        f"WFI detector accumulation-rate trajectories and excess; peak day {peak_day:.6f}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_mosaic(path, maps, peak_day, title_suffix):
    limits = {
        field: percentile_color_limits(maps, field, 99.0)
        for _title, field in RATE_METRICS
    }
    draw_rate_mosaic(
        path,
        maps,
        "relative-response fitted accumulation rates",
        "",
        metrics=RATE_METRICS,
        title_prefix=(
            f"Full-WFI accumulation-rate difference at day {peak_day:.6f}"
            f" ({title_suffix})"
        ),
        observation_label="",
        color_limits_by_field=limits,
        colorbar_labels={
            "rate_difference_nm_day": "Rate difference (nm/day)",
            "rate_difference_percent": "Rate difference (%)",
        },
        metric_unit_labels={
            "rate_difference_nm_day": "nm/day",
            "rate_difference_percent": "%",
        },
        suptitle_fontsize=11,
        axis_title_fontsize=10,
    )


def main():
    args = parse_args()
    if not args.window_start_day < args.window_end_day:
        raise ValueError("window start must be before window end")
    if args.time_points is not None and args.time_points < 2:
        raise ValueError("--time-points must be at least 2")
    if args.context_days <= 0.0:
        raise ValueError("--context-days must be positive")
    detectors = tuple(args.detector or SCAS)
    tile_grid_size = tile_grid_size_for_tile_size(args.tile_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector_tiles = {}
    all_tiles = []
    for detector in detectors:
        tiles = load_detector(args.statistics_root, detector, tile_grid_size)
        detector_tiles[detector] = tiles
        all_tiles.extend(tiles)

    if args.time_points is None:
        time_grid = np.unique(
            np.concatenate(
                [
                    tile["days"][
                        (tile["days"] >= args.window_start_day)
                        & (tile["days"] <= args.window_end_day)
                    ]
                    for tile in all_tiles
                ]
            )
        )
    else:
        time_grid = np.linspace(
            args.window_start_day, args.window_end_day, args.time_points
        )
    if time_grid.size < 2:
        raise ValueError("The requested window has too few observation times")

    for tile in all_tiles:
        (
            tile["observed"],
            tile["background"],
            tile["difference"],
            tile["percent"],
            tile["excluded_samples"],
            tile["background_spline"],
        ) = tile_rate_difference(
            tile["days"],
            tile["rates"],
            time_grid,
            args.window_start_day,
            args.window_end_day,
        )

    detector_series = {}
    for detector, tiles in detector_tiles.items():
        detector_series[detector] = {
            "tiles": tiles,
            "observed": np.mean([tile["observed"] for tile in tiles], axis=0),
            "background": np.mean(
                [tile["background"] for tile in tiles], axis=0
            ),
        }
    observed_mean = np.mean(
        [series["observed"] for series in detector_series.values()], axis=0
    )
    background_mean = np.mean(
        [series["background"] for series in detector_series.values()], axis=0
    )
    difference_mean = observed_mean - background_mean
    percent_mean = 100.0 * difference_mean / background_mean
    rate_peak_index = int(np.argmax(difference_mean))
    percent_peak_index = int(np.argmax(percent_mean))
    rate_peak_day = float(time_grid[rate_peak_index])

    write_tile_summary(
        args.output_dir / "tile_difference_at_full_plane_peaks.csv",
        detector_tiles,
        detectors,
        time_grid,
        rate_peak_index,
        percent_peak_index,
    )
    write_detector_summary(
        args.output_dir / "detector_difference_at_full_plane_peaks.csv",
        detector_series,
        time_grid,
        rate_peak_index,
        percent_peak_index,
    )
    write_time_series(
        args.output_dir / "full_focal_plane_rate_difference.csv",
        time_grid,
        observed_mean,
        background_mean,
        difference_mean,
        percent_mean,
    )
    context_start = max(
        args.window_start_day - args.context_days,
        max(tile["days"][0] for tile in all_tiles),
    )
    context_end = min(
        args.window_end_day + args.context_days,
        min(tile["days"][-1] for tile in all_tiles),
    )
    if not context_start < context_end:
        raise ValueError("The requested plot context is outside the fitted data")
    context_grid = np.linspace(context_start, context_end, 401)
    context_detector_series = {}
    for detector, tiles in detector_tiles.items():
        observed = np.mean(
            [np.interp(context_grid, tile["days"], tile["rates"]) for tile in tiles],
            axis=0,
        )
        background = np.mean(
            [tile["background_spline"](context_grid) for tile in tiles],
            axis=0,
        )
        difference = observed - background
        context_detector_series[detector] = {
            "observed": observed,
            "background": background,
            "difference": difference,
            "percent": 100.0 * difference / background,
        }
    context_observed_mean = np.mean(
        [series["observed"] for series in context_detector_series.values()], axis=0
    )
    context_background_mean = np.mean(
        [series["background"] for series in context_detector_series.values()], axis=0
    )
    context_difference_mean = context_observed_mean - context_background_mean
    context_percent_mean = 100.0 * context_difference_mean / context_background_mean
    temperature_events = load_temperature_events(detector_tiles)
    write_time_series(
        args.output_dir / "full_focal_plane_rate_difference_context.csv",
        context_grid,
        context_observed_mean,
        context_background_mean,
        context_difference_mean,
        context_percent_mean,
    )
    plot_time_series(
        args.output_dir / "full_focal_plane_rate_difference.png",
        context_grid,
        context_detector_series,
        context_observed_mean,
        context_background_mean,
        context_difference_mean,
        context_percent_mean,
        rate_peak_day,
        args.window_start_day,
        args.window_end_day,
        temperature_events,
    )

    rate_peak_maps = {
        detector: {
            "rate_difference_nm_day": _map_for_time(
                tiles, "difference", rate_peak_index, tile_grid_size
            ),
            "rate_difference_percent": _map_for_time(
                tiles, "percent", rate_peak_index, tile_grid_size
            ),
        }
        for detector, tiles in detector_tiles.items()
    }
    write_mosaic(
        args.output_dir / "full_focal_plane_difference_at_rate_peak.png",
        rate_peak_maps,
        time_grid[rate_peak_index],
        "maximum mean rate difference",
    )

    percent_peak_maps = {
        detector: {
            "rate_difference_nm_day": _map_for_time(
                tiles, "difference", percent_peak_index, tile_grid_size
            ),
            "rate_difference_percent": _map_for_time(
                tiles, "percent", percent_peak_index, tile_grid_size
            ),
        }
        for detector, tiles in detector_tiles.items()
    }
    write_mosaic(
        args.output_dir / "full_focal_plane_difference_at_percent_peak.png",
        percent_peak_maps,
        time_grid[percent_peak_index],
        "maximum mean percentage difference",
    )

    summary = {
        "statistics_root": str(Path(args.statistics_root).resolve()),
        "detectors": list(detectors),
        "tiles_per_detector": tile_grid_size * tile_grid_size,
        "total_tiles": len(all_tiles),
        "window_start_day": args.window_start_day,
        "window_end_day": args.window_end_day,
        "temperature_events": [
            {"label": event.label, "day": float(event.day)}
            for event in temperature_events
        ],
        "background": "shape-preserving cubic spline interpolation on fitted tile accumulation rate over time, excluding the inclusive window",
        "rate_source": "tile_instantaneous_rate_nm_day evaluated at observation_days",
        "rate_peak_day": float(time_grid[rate_peak_index]),
        "aggregation": "mean of the 256 tile rates within each detector, then mean of the 18 detector means",
        "rate_peak_mean_difference_nm_day": float(difference_mean[rate_peak_index]),
        "rate_peak_mean_difference_percent": float(percent_mean[rate_peak_index]),
        "percent_peak_day": float(time_grid[percent_peak_index]),
        "percent_peak_mean_difference_nm_day": float(difference_mean[percent_peak_index]),
        "percent_peak_mean_difference_percent": float(percent_mean[percent_peak_index]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
