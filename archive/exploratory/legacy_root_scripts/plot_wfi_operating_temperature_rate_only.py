"""Plot operating-temperature ice-rate trajectories for every WFI tile."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
RESET_DAYS = (12.907397928240743, 38.79784510416666)
SEGMENTS = (
    ("pre-reset", "Pre-reset", "#0072b2"),
    ("post-reset", "Post-reset", "#d55e00"),
    ("post-third-decon", "Post-third decon", "#009e73"),
)
TILE_PATTERN = re.compile(r"tile_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--telemetry-path", type=Path, required=True)
    parser.add_argument(
        "--temperature-coefficients",
        type=Path,
        required=True,
        help="CSV containing per-detector, per-tile operating temperatures.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--temperature-tolerance-k", type=float, default=0.1)
    parser.add_argument(
        "--detector",
        dest="detectors",
        action="append",
        default=None,
        help="Restrict output to selected detector(s); may be repeated.",
    )
    parser.add_argument(
        "--assume-missing-operating-temperature",
        action="store_true",
        help="Retain repeats with missing telemetry when observed values are in-window.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Write only the all-detector percentile plot and summaries.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_timestamp(value):
    timestamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return timestamp.replace(tzinfo=None)


def five_minute_bucket(timestamp):
    return timestamp.replace(
        minute=(timestamp.minute // 5) * 5,
        second=0,
        microsecond=0,
    )


def load_telemetry(path):
    sums = defaultdict(float)
    counts = defaultdict(int)
    row_count = 0
    first_timestamp = None
    last_timestamp = None
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ObsTime", "EUValue", "mnemonic"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Telemetry is missing columns: {sorted(missing)}")
        for row in reader:
            if row["mnemonic"] != "WFI_MCE_ADC_WTSM00":
                continue
            timestamp = parse_timestamp(row["ObsTime"])
            bucket = five_minute_bucket(timestamp)
            sums[bucket] += float(row["EUValue"])
            counts[bucket] += 1
            row_count += 1
            first_timestamp = (
                timestamp
                if first_timestamp is None
                else min(first_timestamp, timestamp)
            )
            last_timestamp = (
                timestamp if last_timestamp is None else max(last_timestamp, timestamp)
            )
    if not counts:
        raise ValueError(f"No WFI_MCE_ADC_WTSM00 rows found in {path}")
    temperatures = {bucket: sums[bucket] / counts[bucket] for bucket in counts}
    return temperatures, {
        "telemetry_path": str(Path(path).resolve()),
        "telemetry_rows": row_count,
        "telemetry_buckets": len(temperatures),
        "telemetry_first_utc": first_timestamp.isoformat(),
        "telemetry_last_utc": last_timestamp.isoformat(),
    }


def load_operating_temperatures(path, tile_size):
    references = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"detector", "tile_size", "tile_id", "reference_temperature_k"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Temperature coefficients are missing: {sorted(missing)}")
        for row in reader:
            if int(row["tile_size"]) != tile_size:
                continue
            key = (row["detector"].upper(), int(row["tile_id"]))
            if key in references:
                raise ValueError(f"Duplicate operating temperature for {key}")
            references[key] = float(row["reference_temperature_k"])
    return references


def tile_id_from_path(path):
    match = TILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected tile directory name: {path}")
    return int(match.group(1))


def tile_fit_paths(detector_root, fit_subdir="relative_response"):
    fits_root = Path(detector_root) / fit_subdir / "fits"
    paths = sorted(
        (path.parent for path in fits_root.glob("tile_*/posterior_samples.npz")),
        key=tile_id_from_path,
    )
    if not paths:
        raise FileNotFoundError(f"No relative-response tile fits found in {fits_root}")
    return paths


def load_node_trajectory(fit_dir):
    with np.load(Path(fit_dir) / "posterior_samples.npz", allow_pickle=False) as archive:
        repeats = np.asarray(archive["observation_repeat"], dtype=int).reshape(-1)
        observation_days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
        thickness = np.asarray(archive["np_node_thickness"], dtype=float).reshape(-1)
        rates = np.asarray(archive["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
    repeat_values = np.unique(repeats)
    if thickness.size != rates.size or rates.size != repeat_values.size:
        raise ValueError(
            f"Node/repeat count mismatch in {fit_dir}: "
            f"{thickness.size} node thicknesses, {rates.size} node rates, "
            f"and {repeat_values.size} repeats"
        )
    days = np.asarray(
        [np.mean(observation_days[repeats == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    return repeat_values, days, thickness, rates


def load_repeat_temperature_groups(stats_path, repeat_values, telemetry_temperatures):
    groups = defaultdict(list)
    with Path(stats_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"repeat", "exposure_start_time"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Statistics are missing columns: {sorted(missing)}")
        for row in reader:
            repeat = int(row["repeat"])
            bucket = five_minute_bucket(parse_timestamp(row["exposure_start_time"]))
            groups[repeat].append(telemetry_temperatures.get(bucket, np.nan))

    if set(groups) != set(int(value) for value in repeat_values):
        missing = sorted(set(int(value) for value in repeat_values).difference(groups))
        extra = sorted(set(groups).difference(int(value) for value in repeat_values))
        raise ValueError(
            f"Repeat mismatch for {stats_path}: missing={missing}, extra={extra}"
        )
    return [np.asarray(groups[int(repeat)], dtype=float) for repeat in repeat_values]


def operating_temperature_mask(
    temperature_groups,
    reference_temperature,
    tolerance,
    assume_missing,
):
    keep = []
    node_temperatures = []
    complete_nodes = 0
    missing_nodes = 0
    outside_nodes = 0
    assumed_nodes = 0
    for values in temperature_groups:
        observed = values[np.isfinite(values)]
        complete = observed.size == values.size
        if complete:
            complete_nodes += 1
        else:
            missing_nodes += 1
        within_window = observed.size == 0 or np.all(
            np.abs(observed - reference_temperature) <= tolerance
        )
        if not within_window:
            outside_nodes += 1
        retain = within_window and (complete or (assume_missing and not complete))
        if retain and not complete:
            assumed_nodes += 1
        keep.append(retain)
        node_temperatures.append(float(np.mean(observed)) if observed.size else np.nan)
    return np.asarray(keep, dtype=bool), np.asarray(node_temperatures), {
        "complete_nodes": complete_nodes,
        "missing_nodes": missing_nodes,
        "outside_temperature_nodes": outside_nodes,
        "assumed_operating_nodes": assumed_nodes,
        "operating_nodes": int(np.count_nonzero(keep)),
    }


def segment_mask(days, name):
    if name == "pre-reset":
        return days < RESET_DAYS[0]
    if name == "post-reset":
        return (days >= RESET_DAYS[0]) & (days < RESET_DAYS[1])
    return days >= RESET_DAYS[1]


def load_detector_data(
    detector,
    detector_root,
    tile_size,
    telemetry_temperatures,
    operating_temperatures,
    tolerance,
    assume_missing,
    fit_subdir="relative_response",
):
    fit_paths = tile_fit_paths(detector_root, fit_subdir)
    reference_path = next(
        path for path in fit_paths if tile_id_from_path(path) == 0
    )
    repeat_values, days, _thickness, _rates = load_node_trajectory(reference_path)
    statistics_path = (
        Path(detector_root)
        / fit_subdir
        / "tile_statistics"
        / f"{detector.lower()}_tile_log_statistics_{tile_size}_tile_000.csv"
    )
    temperature_groups = load_repeat_temperature_groups(
        statistics_path,
        repeat_values,
        telemetry_temperatures,
    )
    tiles = []
    for fit_path in fit_paths:
        tile_id = tile_id_from_path(fit_path)
        current_repeats, current_days, thickness, rates = load_node_trajectory(fit_path)
        if not np.array_equal(current_repeats, repeat_values) or not np.array_equal(
            current_days, days
        ):
            raise ValueError(f"Observation nodes differ in {fit_path}")
        reference_temperature = operating_temperatures[(detector, tile_id)]
        keep, node_temperatures, filter_info = operating_temperature_mask(
            temperature_groups,
            reference_temperature,
            tolerance,
            assume_missing,
        )
        tiles.append(
            {
                "detector": detector,
                "tile_id": tile_id,
                "days": days,
                "thickness": thickness,
                "rates": rates,
                "filtered_days": days[keep],
                "filtered_thickness": thickness[keep],
                "filtered_rates": rates[keep],
                "node_temperatures": node_temperatures,
                "reference_temperature_k": reference_temperature,
                "repeat_values": repeat_values,
                "keep": keep,
                "filter_info": filter_info,
            }
        )
    return tiles, statistics_path


def plot_detector_rate_panels(detector, tiles, output, tolerance, assume_missing, dpi):
    figure, axes = plt.subplots(2, 1, figsize=(13, 10))
    rate_axis, raw_axis = axes
    finite_days = np.concatenate(
        [tile["filtered_days"] for tile in tiles if tile["filtered_days"].size]
    )
    x_max = float(np.max(finite_days)) if finite_days.size else 1.0

    raw_x = {name: [] for name, _label, _color in SEGMENTS}
    raw_y = {name: [] for name, _label, _color in SEGMENTS}
    for tile in tiles:
        days = tile["filtered_days"]
        rates = tile["filtered_rates"]
        thickness = tile["filtered_thickness"]
        for name, _label, color in SEGMENTS:
            mask = segment_mask(days, name)
            if not np.any(mask):
                continue
            rate_axis.plot(
                days[mask],
                rates[mask],
                color=color,
                linewidth=0.45,
                alpha=0.18,
            )
            raw_x[name].append(thickness[mask])
            raw_y[name].append(rates[mask])

    rate_axis.axvline(RESET_DAYS[0], color="0.35", linestyle="--", linewidth=0.9)
    rate_axis.axvline(RESET_DAYS[1], color="0.35", linestyle=":", linewidth=1.0)
    rate_axis.set_xlim(0.0, x_max)
    rate_axis.set_xlabel("Days since first exposure")
    rate_axis.set_ylabel("Rate [nm/day]")
    rate_axis.set_title("Relative-response MAP rate trajectory: all retained tiles")
    rate_axis.grid(alpha=0.2)

    for name, label, color in SEGMENTS:
        if not raw_x[name]:
            continue
        x_values = np.concatenate(raw_x[name])
        y_values = np.concatenate(raw_y[name])
        raw_axis.scatter(
            x_values,
            y_values,
            color=color,
            s=7,
            alpha=0.12,
            label=label,
        )
        if x_values.size >= 2 and np.ptp(x_values) > 0.0:
            coefficients = np.polyfit(x_values, y_values, 1)
            x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
            raw_axis.plot(
                x_line,
                np.polyval(coefficients, x_line),
                color=color,
                linewidth=1.6,
            )
    raw_axis.set_title("Raw rate versus thickness: all retained tiles")
    raw_axis.set_xlabel("Ice thickness [nm]")
    raw_axis.set_ylabel("Rate [nm/day]")
    raw_axis.grid(alpha=0.2)
    raw_axis.legend(frameon=False, loc="best")

    references = np.asarray([tile["reference_temperature_k"] for tile in tiles])
    complete = sum(tile["filter_info"]["complete_nodes"] for tile in tiles)
    missing = sum(tile["filter_info"]["missing_nodes"] for tile in tiles)
    retained = sum(tile["filter_info"]["operating_nodes"] for tile in tiles)
    assumption = "missing retained" if assume_missing else "complete telemetry only"
    figure.suptitle(
        f"{detector} operating-temperature ice-rate diagnostics\n"
        f"{len(tiles)} tiles; WTSM00 reference {references.min():.4f}-"
        f"{references.max():.4f} K +/- {tolerance:.3f} K; {assumption}\n"
        f"{retained} retained tile-nodes; {complete} complete and {missing} with missing telemetry",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=1.0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def binned_percentiles(x_values, y_values, bin_count=40):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size == 0:
        return tuple(np.asarray([], dtype=float) for _ in range(4))
    lower = float(np.min(x_values))
    upper = float(np.max(x_values))
    if lower == upper:
        return (
            np.asarray([lower]),
            np.asarray([np.percentile(y_values, 16)]),
            np.asarray([np.percentile(y_values, 50)]),
            np.asarray([np.percentile(y_values, 84)]),
            np.asarray([y_values.size], dtype=int),
        )
    edges = np.linspace(lower, upper, bin_count + 1)
    centers = []
    low = []
    median = []
    high = []
    counts = []
    for index in range(bin_count):
        if index == bin_count - 1:
            mask = (x_values >= edges[index]) & (x_values <= edges[index + 1])
        else:
            mask = (x_values >= edges[index]) & (x_values < edges[index + 1])
        if not np.any(mask):
            continue
        centers.append((edges[index] + edges[index + 1]) / 2.0)
        low.append(np.percentile(y_values[mask], 16))
        median.append(np.percentile(y_values[mask], 50))
        high.append(np.percentile(y_values[mask], 84))
        counts.append(int(np.count_nonzero(mask)))
    return (
        np.asarray(centers),
        np.asarray(low),
        np.asarray(median),
        np.asarray(high),
        np.asarray(counts, dtype=int),
    )


def plot_detector_percentiles(
    detector,
    tiles,
    output,
    tolerance,
    assume_missing,
    dpi,
):
    if not tiles:
        raise ValueError("No tile trajectories available for percentile plot")
    reference_days = tiles[0]["filtered_days"]
    if not reference_days.size:
        raise ValueError("No operating-temperature nodes available for percentile plot")
    if any(
        not np.array_equal(tile["filtered_days"], reference_days)
        for tile in tiles[1:]
    ):
        raise ValueError("Filtered node days differ across tiles; cannot form shared percentiles")

    rate_matrix = np.asarray([tile["filtered_rates"] for tile in tiles], dtype=float)
    rate_percentiles = np.percentile(rate_matrix, [16, 50, 84], axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(13, 10))
    rate_axis, raw_axis = axes
    for index, (_name, label, color) in enumerate(SEGMENTS):
        mask = segment_mask(reference_days, _name)
        if not np.any(mask):
            continue
        rate_axis.fill_between(
            reference_days[mask],
            rate_percentiles[0, mask],
            rate_percentiles[2, mask],
            color=color,
            alpha=0.20,
            label=f"{label} 16/84%",
        )
        rate_axis.plot(
            reference_days[mask],
            rate_percentiles[1, mask],
            color=color,
            linewidth=2.0,
            label=f"{label} median",
        )
    rate_axis.axvline(RESET_DAYS[0], color="0.35", linestyle="--", linewidth=0.9)
    rate_axis.axvline(RESET_DAYS[1], color="0.35", linestyle=":", linewidth=1.0)
    rate_axis.set_xlim(0.0, float(np.max(reference_days)))
    rate_axis.set_xlabel("Days since first exposure")
    rate_axis.set_ylabel("Rate [nm/day]")
    rate_axis.set_title("Operating-temperature ice-rate trajectory")
    rate_axis.grid(alpha=0.2)

    for name, label, color in SEGMENTS:
        x_values = []
        y_values = []
        for tile in tiles:
            mask = segment_mask(tile["filtered_days"], name)
            x_values.append(tile["filtered_thickness"][mask])
            y_values.append(tile["filtered_rates"][mask])
        x_values = np.concatenate(x_values)
        y_values = np.concatenate(y_values)
        centers, low, median, high, _counts = binned_percentiles(x_values, y_values)
        if not centers.size:
            continue
        raw_axis.fill_between(
            centers,
            low,
            high,
            color=color,
            alpha=0.20,
        )
        raw_axis.plot(
            centers,
            median,
            color=color,
            linewidth=2.0,
            label=f"{label} median, binned 16/84%",
        )
    raw_axis.set_title("Raw rate versus thickness: percentile summary")
    raw_axis.set_xlabel("Ice thickness [nm]")
    raw_axis.set_ylabel("Rate [nm/day]")
    raw_axis.grid(alpha=0.2)

    references = np.asarray([tile["reference_temperature_k"] for tile in tiles])
    retained = sum(tile["filter_info"]["operating_nodes"] for tile in tiles)
    complete = sum(tile["filter_info"]["complete_nodes"] for tile in tiles)
    missing = sum(tile["filter_info"]["missing_nodes"] for tile in tiles)
    assumption = "missing retained" if assume_missing else "complete telemetry only"
    figure.suptitle(
        f"{detector} operating-temperature ice-rate percentile summary\n"
        f"{len(tiles)} tiles; percentiles computed across tiles; WTSM00 reference "
        f"{references.min():.4f}-{references.max():.4f} K +/- {tolerance:.3f} K; {assumption}\n"
        f"{retained} retained tile-nodes; {complete} complete and {missing} with missing telemetry",
        fontsize=12,
    )
    rate_axis.legend(frameon=False, fontsize=8, ncol=3, loc="best")
    raw_axis.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=1.0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return reference_days, rate_percentiles


def write_percentile_summaries(output_dir, detector, tiles, days, rate_percentiles):
    prefix = detector.lower()
    trajectory_path = Path(output_dir) / f"{prefix}_rate_trajectory_percentiles.csv"
    with trajectory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("days", "rate_p16_nm_day", "rate_median_nm_day", "rate_p84_nm_day"),
        )
        writer.writeheader()
        for index, day in enumerate(days):
            writer.writerow(
                {
                    "days": day,
                    "rate_p16_nm_day": rate_percentiles[0, index],
                    "rate_median_nm_day": rate_percentiles[1, index],
                    "rate_p84_nm_day": rate_percentiles[2, index],
                }
            )

    fields = (
        "segment",
        "thickness_center_nm",
        "rate_p16_nm_day",
        "rate_median_nm_day",
        "rate_p84_nm_day",
        "points",
    )
    relation_path = Path(output_dir) / f"{prefix}_rate_thickness_percentiles.csv"
    with relation_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, _label, _color in SEGMENTS:
            x_values = []
            y_values = []
            for tile in tiles:
                mask = segment_mask(tile["filtered_days"], name)
                x_values.append(tile["filtered_thickness"][mask])
                y_values.append(tile["filtered_rates"][mask])
            x_values = np.concatenate(x_values)
            y_values = np.concatenate(y_values)
            centers, low, median, high, counts = binned_percentiles(x_values, y_values)
            for index in range(centers.size):
                writer.writerow(
                    {
                        "segment": name,
                        "thickness_center_nm": centers[index],
                        "rate_p16_nm_day": low[index],
                        "rate_median_nm_day": median[index],
                        "rate_p84_nm_day": high[index],
                        "points": int(counts[index]),
                    }
                )
    return trajectory_path, relation_path


def write_summary(path, rows):
    fields = (
        "detector",
        "tile_id",
        "reference_temperature_k",
        "temperature_tolerance_k",
        "total_nodes",
        "complete_nodes",
        "missing_nodes",
        "outside_temperature_nodes",
        "assumed_operating_nodes",
        "operating_nodes",
        "filtered_day_min",
        "filtered_day_max",
        "filtered_rate_min_nm_day",
        "filtered_rate_max_nm_day",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.tile_size <= 0 or args.temperature_tolerance_k <= 0 or args.dpi <= 0:
        raise ValueError("tile-size, temperature-tolerance-k, and dpi must be positive")
    detectors = tuple(detector.upper() for detector in (args.detectors or DETECTORS))
    unknown = set(detectors).difference(DETECTORS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")

    telemetry_temperatures, telemetry_info = load_telemetry(args.telemetry_path)
    operating_temperatures = load_operating_temperatures(
        args.temperature_coefficients,
        args.tile_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detector_counts = {}
    percentile_outputs = []
    for detector in detectors:
        detector_root = Path(args.production_root) / detector.lower()
        tiles, statistics_path = load_detector_data(
            detector,
            detector_root,
            args.tile_size,
            telemetry_temperatures,
            operating_temperatures,
            args.temperature_tolerance_k,
            args.assume_missing_operating_temperature,
        )
        detector_counts[detector] = len(tiles)
        if args.aggregate_only:
            output = (
                Path(args.output_dir)
                / f"{detector.lower()}_operating_temperature_ice_rate_percentiles.png"
            )
            days, rate_percentiles = plot_detector_percentiles(
                detector,
                tiles,
                output,
                args.temperature_tolerance_k,
                args.assume_missing_operating_temperature,
                args.dpi,
            )
            trajectory_path, relation_path = write_percentile_summaries(
                args.output_dir,
                detector,
                tiles,
                days,
                rate_percentiles,
            )
            percentile_outputs.append(str(output.resolve()))
            print(f"Wrote {output}")
            print(f"Wrote {trajectory_path}")
            print(f"Wrote {relation_path}")
        else:
            output = (
                Path(args.output_dir)
                / f"{detector.lower()}_operating_temperature_ice_rate_all_tiles.png"
            )
            plot_detector_rate_panels(
                detector,
                tiles,
                output,
                args.temperature_tolerance_k,
                args.assume_missing_operating_temperature,
                args.dpi,
            )
            print(f"Wrote {output} using {statistics_path}")
        for tile in tiles:
            rates = tile["filtered_rates"]
            days = tile["filtered_days"]
            summary_rows.append(
                {
                    "detector": detector,
                    "tile_id": tile["tile_id"],
                    "reference_temperature_k": tile["reference_temperature_k"],
                    "temperature_tolerance_k": args.temperature_tolerance_k,
                    "total_nodes": tile["keep"].size,
                    **tile["filter_info"],
                    "filtered_day_min": float(np.min(days)) if days.size else np.nan,
                    "filtered_day_max": float(np.max(days)) if days.size else np.nan,
                    "filtered_rate_min_nm_day": float(np.min(rates)) if rates.size else np.nan,
                    "filtered_rate_max_nm_day": float(np.max(rates)) if rates.size else np.nan,
                }
            )
    summary_path = Path(args.output_dir) / "operating_temperature_ice_rate_summary.csv"
    write_summary(summary_path, summary_rows)
    metadata = {
        "production_root": str(Path(args.production_root).resolve()),
        "temperature_coefficients": str(Path(args.temperature_coefficients).resolve()),
        "tile_size": args.tile_size,
        "temperature_tolerance_k": args.temperature_tolerance_k,
        "assume_missing_operating_temperature": args.assume_missing_operating_temperature,
        "detectors": detectors,
        "detector_tile_counts": detector_counts,
        "total_tiles": len(summary_rows),
        "aggregate_only": args.aggregate_only,
        "percentile_outputs": percentile_outputs,
        **telemetry_info,
    }
    metadata_path = Path(args.output_dir) / "operating_temperature_ice_rate_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
