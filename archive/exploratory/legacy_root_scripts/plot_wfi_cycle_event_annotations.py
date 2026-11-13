"""Add estimated temperature-event markers to the Cycle 2/3 WFI plots."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


UTC = timezone.utc
DECON_DAYS = (0.0, 12.907397928240743, 38.79784510416666)
CYCLE_INDICES = (1, 2)
CYCLE_LABELS = ("Cycle 2", "Cycle 3")
CYCLE_COLORS = ("#d55e00", "#009e73")
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
EVENTS = (
    {
        "name": "AOM",
        "timestamp": datetime(2026, 10, 30, 1, 43, tzinfo=UTC),
        "label": "AOM temperature bump",
        "color": "#0072b2",
        "linestyle": "--",
    },
    {
        "name": "AMS",
        "timestamp": datetime(2026, 11, 2, 0, 0, tzinfo=UTC),
        "label": "AMS temperature bump (estimated)",
        "color": "#cc79a7",
        "linestyle": ":",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--fit-subdir", default="spline_relative_response_normal")
    parser.add_argument("--statistics-path", type=Path, required=True)
    parser.add_argument("--trajectory-quantiles", type=Path, required=True)
    parser.add_argument("--thickness-summary", type=Path, required=True)
    parser.add_argument("--thickness-output", type=Path, required=True)
    parser.add_argument("--tau-output", type=Path, required=True)
    parser.add_argument("--detector", dest="detectors", action="append", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    timestamp = datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def first_statistics_timestamp(path):
    with Path(path).open(newline="") as handle:
        timestamps = [
            parse_timestamp(row["exposure_start_time"])
            for row in csv.DictReader(handle)
            if row.get("exposure_start_time")
        ]
    if not timestamps:
        raise ValueError(f"No exposure timestamps found in {path}")
    return min(timestamps)


def load_trajectory_quantiles(path, detectors):
    values = {detector: [] for detector in detectors}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            detector = row["detector"].upper()
            if detector not in values:
                continue
            values[detector].append(
                (
                    float(row["days"]),
                    float(row["ice_thickness_median_nm"]),
                )
            )
    trajectories = {}
    for detector, rows in values.items():
        if not rows:
            raise FileNotFoundError(f"No trajectory quantiles found for {detector}")
        rows.sort()
        trajectories[detector] = (
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
        )
    return trajectories


def event_markers(trajectories, start_utc):
    markers = []
    for event in EVENTS:
        event_day = (event["timestamp"] - start_utc).total_seconds() / 86400.0
        thicknesses = []
        for days, thickness in trajectories.values():
            if event_day < days[0] or event_day > days[-1]:
                continue
            thicknesses.append(float(np.interp(event_day, days, thickness)))
        if not thicknesses:
            raise ValueError(f"Event is outside the trajectory range: {event['name']}")
        marker = dict(event)
        marker["event_day"] = event_day
        marker["median_thickness_nm"] = float(np.median(thicknesses))
        marker["cycle3_tau_days"] = event_day - DECON_DAYS[2]
        markers.append(marker)
    return markers


def write_event_markers(path, markers):
    fields = (
        "event",
        "timestamp_utc",
        "event_day",
        "cycle3_tau_days",
        "median_thickness_nm",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for marker in markers:
            writer.writerow(
                {
                    "event": marker["name"],
                    "timestamp_utc": marker["timestamp"].isoformat(),
                    "event_day": marker["event_day"],
                    "cycle3_tau_days": marker["cycle3_tau_days"],
                    "median_thickness_nm": marker["median_thickness_nm"],
                }
            )


def load_thickness_summary(path, detectors):
    curves = {detector: {"cycle2": [], "cycle3": []} for detector in detectors}
    ranges = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            detector = row["detector"].upper()
            if detector not in curves or row["cycle"] not in curves[detector]:
                continue
            cycle = row["cycle"]
            curves[detector][cycle].append(
                (
                    float(row["thickness_nm"]),
                    float(row["rate_p16_nm_day"]),
                    float(row["rate_median_nm_day"]),
                    float(row["rate_p84_nm_day"]),
                )
            )
            ranges[detector] = (
                float(row["range_min_nm"]),
                float(row["range_max_nm"]),
            )
    for detector in detectors:
        for cycle in curves[detector]:
            if not curves[detector][cycle]:
                raise FileNotFoundError(f"No {cycle} thickness curve found for {detector}")
            curves[detector][cycle] = np.asarray(curves[detector][cycle], dtype=float)
    return curves, ranges


def event_legend_handles(markers, mode):
    handles = []
    for marker in markers:
        if mode == "thickness":
            label = (
                f"{marker['name']} at {marker['median_thickness_nm']:.0f} nm "
                f"({marker['timestamp'].strftime('%Y-%m-%d %H:%M UTC')})"
            )
        else:
            label = (
                f"{marker['name']} tau={marker['cycle3_tau_days']:.1f} d, "
                f"{marker['median_thickness_nm']:.0f} nm"
            )
        handles.append(
            Line2D(
                [],
                [],
                color=marker["color"],
                linestyle=marker["linestyle"],
                linewidth=1.5,
                label=label,
            )
        )
    return handles


def plot_thickness(path, curves, ranges, markers, dpi):
    figure, axes = plt.subplots(3, 6, figsize=(18, 9), sharey=True)
    for axis, detector in zip(axes.flat, curves):
        for cycle, label, color in zip(("cycle2", "cycle3"), CYCLE_LABELS, CYCLE_COLORS):
            values = curves[detector][cycle]
            valid = np.isfinite(values[:, 1])
            values = values[valid]
            axis.fill_between(values[:, 0], values[:, 1], values[:, 3], color=color, alpha=0.18)
            axis.plot(values[:, 0], values[:, 2], color=color, linewidth=1.5)
        for marker in markers:
            axis.axvline(
                marker["median_thickness_nm"],
                color=marker["color"],
                linestyle=marker["linestyle"],
                linewidth=1.0,
                alpha=0.9,
            )
        axis.set_title(f"{detector}\n{ranges[detector][0]:.0f}-{ranges[detector][1]:.0f} nm", fontsize=8)
        axis.set_xlim(*ranges[detector])
        axis.set_ylim(5.0, 15.0)
        axis.set_xlabel("ice thickness [nm]", fontsize=8)
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=7)
    for axis in axes[:, 0]:
        axis.set_ylabel("Rate [nm/day]", fontsize=8)
    cycle_handles = [
        Line2D([], [], color=color, linewidth=2.0, label=label)
        for label, color in zip(CYCLE_LABELS, CYCLE_COLORS)
    ]
    figure.legend(
        handles=cycle_handles + event_legend_handles(markers, "thickness"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Fixed-knot spline rate versus ice thickness: Cycle 2 versus Cycle 3\n"
        "Lines are tile medians; bands are tile 16/84th percentiles; vertical lines use the 18-detector median thickness",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.92), pad=0.8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def tile_fit_paths(production_root, detector, fit_subdir):
    fit_root = Path(production_root) / "production" / detector.lower() / fit_subdir / "fits"
    paths = sorted(fit_root.glob("tile_*/posterior_samples.npz"))
    if not paths:
        raise FileNotFoundError(f"No fit archives found in {fit_root}")
    return paths


def load_tile_trajectory(path):
    with np.load(path, allow_pickle=False) as archive:
        repeats = np.asarray(archive["observation_repeat"], dtype=int).reshape(-1)
        observation_days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
        thickness = np.asarray(archive["np_node_thickness"], dtype=float).reshape(-1)
        rates = np.asarray(archive["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
    repeat_values = np.unique(repeats)
    days = np.asarray(
        [np.mean(observation_days[repeats == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    if thickness.size != days.size or rates.size != days.size:
        raise ValueError(f"Node trajectory size mismatch in {path}")
    return days, thickness, rates


def load_detector_tiles(production_root, detector, fit_subdir):
    return [load_tile_trajectory(path) for path in tile_fit_paths(production_root, detector, fit_subdir)]


def cycle_age_values(tile, cycle_index):
    days, _thickness, rates = tile
    lower = DECON_DAYS[cycle_index]
    upper = DECON_DAYS[cycle_index + 1] if cycle_index < 2 else np.inf
    mask = (days >= lower) & (days < upper)
    order = np.argsort(days[mask], kind="stable")
    return days[mask][order] - lower, rates[mask][order]


def common_age_grid(detector_tiles):
    minima = []
    maxima = []
    for tiles in detector_tiles.values():
        for cycle_index in CYCLE_INDICES:
            for tile in tiles:
                ages, _rates = cycle_age_values(tile, cycle_index)
                minima.append(float(np.min(ages)))
                maxima.append(float(np.max(ages)))
    lower = max(max(minima), 2.5)
    upper = min(maxima)
    if lower >= upper:
        raise ValueError("No common Cycle 2/3 tau interval")
    return np.arange(lower, upper + 0.001, 0.25)


def tau_percentiles(tiles, cycle_index, ages):
    values = []
    for tile in tiles:
        cycle_ages, rates = cycle_age_values(tile, cycle_index)
        values.append(np.interp(ages, cycle_ages, rates))
    return np.percentile(np.asarray(values, dtype=float), [16, 50, 84], axis=0)


def median_slope(tiles, cycle_index, ages):
    slopes = []
    for tile in tiles:
        cycle_ages, rates = cycle_age_values(tile, cycle_index)
        slopes.append(np.polyfit(ages, np.interp(ages, cycle_ages, rates), 1)[0])
    return float(np.median(slopes))


def plot_tau(path, detector_tiles, ages, markers, dpi):
    figure, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=True, sharey=True)
    for axis, detector in zip(axes.flat, detector_tiles):
        slopes = []
        for cycle_index, label, color in zip(CYCLE_INDICES, CYCLE_LABELS, CYCLE_COLORS):
            percentiles = tau_percentiles(detector_tiles[detector], cycle_index, ages)
            slopes.append(median_slope(detector_tiles[detector], cycle_index, ages))
            axis.fill_between(ages, percentiles[0], percentiles[2], color=color, alpha=0.18)
            axis.plot(ages, percentiles[1], color=color, linewidth=1.5)
        for marker in markers:
            if ages[0] <= marker["cycle3_tau_days"] <= ages[-1]:
                axis.axvline(
                    marker["cycle3_tau_days"],
                    color=marker["color"],
                    linestyle=marker["linestyle"],
                    linewidth=1.0,
                    alpha=0.9,
                )
        axis.set_title(f"{detector}\nslopes: {slopes[0]:+.2f}, {slopes[1]:+.2f}", fontsize=8)
        axis.set_ylim(5.0, 15.0)
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("tau since decon [days]", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Rate [nm/day]", fontsize=8)
    cycle_handles = [
        Line2D([], [], color=color, linewidth=2.0, label=label)
        for label, color in zip(CYCLE_LABELS, CYCLE_COLORS)
    ]
    figure.legend(
        handles=cycle_handles + event_legend_handles(markers, "tau"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Fixed-knot spline rate trajectories: Cycle 2 versus Cycle 3 aligned by tau since decon\n"
        "Lines are tile medians; bands are tile 16/84th percentiles; vertical lines mark Cycle 3 event tau",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.92), pad=0.8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def update_metadata(path, markers):
    path = Path(path)
    metadata = json.loads(path.read_text()) if path.is_file() else {}
    metadata["y_axis_min_nm_day"] = 5.0
    metadata["y_axis_max_nm_day"] = 15.0
    metadata["event_markers"] = [
        {
            "event": marker["name"],
            "timestamp_utc": marker["timestamp"].isoformat(),
            "event_day": marker["event_day"],
            "cycle3_tau_days": marker["cycle3_tau_days"],
            "median_thickness_nm": marker["median_thickness_nm"],
        }
        for marker in markers
    ]
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    args = parse_args()
    detectors = tuple(detector.upper() for detector in args.detectors)
    unknown = set(detectors).difference(DETECTORS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")

    start_utc = first_statistics_timestamp(args.statistics_path)
    trajectories = load_trajectory_quantiles(args.trajectory_quantiles, detectors)
    markers = event_markers(trajectories, start_utc)
    curves, ranges = load_thickness_summary(args.thickness_summary, detectors)
    detector_tiles = {
        detector: load_detector_tiles(args.production_root, detector, args.fit_subdir)
        for detector in detectors
    }
    ages = common_age_grid(detector_tiles)

    plot_thickness(args.thickness_output, curves, ranges, markers, args.dpi)
    plot_tau(args.tau_output, detector_tiles, ages, markers, args.dpi)

    event_csv = Path(args.thickness_output).with_name("cycle_event_markers.csv")
    write_event_markers(event_csv, markers)
    update_metadata(
        Path(args.thickness_output).with_suffix(".json"), markers
    )
    update_metadata(Path(args.tau_output).with_name("rate_vs_tau_metadata.json"), markers)

    for marker in markers:
        print(
            f"{marker['name']}: day={marker['event_day']:.4f}, "
            f"cycle3_tau={marker['cycle3_tau_days']:.4f}, "
            f"median_thickness={marker['median_thickness_nm']:.2f} nm"
        )
    print(f"Wrote {args.thickness_output}")
    print(f"Wrote {args.tau_output}")
    print(f"Wrote {event_csv}")


if __name__ == "__main__":
    main()
