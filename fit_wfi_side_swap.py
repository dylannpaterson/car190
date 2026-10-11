#!/usr/bin/env python3
"""Fit local LED-dependent image and photodiode side-swap factors."""

import argparse
from collections import defaultdict
import csv
import sys
from pathlib import Path

import numpy as np

from wfi_side_swap import fit_log_step_spline


def _read_image_series(path):
    grouped = defaultdict(list)
    metadata = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            n_pair_valid = float(row["n_pair_valid"])
            if n_pair_valid <= 0.0:
                continue
            current_mean = float(row["sum_current_counts"]) / n_pair_valid
            if not np.isfinite(current_mean) or current_mean <= 0.0:
                continue
            key = (int(row["repeat"]), int(row["led"]), row["file_set_name"])
            grouped[key].append(current_mean)
            metadata[key] = float(row["days"])

    series = defaultdict(list)
    for (repeat, led, file_set_name), values in grouped.items():
        log_values = np.log(np.asarray(values, dtype=float))
        series[led].append(
            {
                "repeat": repeat,
                "days": metadata[(repeat, led, file_set_name)],
                "value": float(np.exp(np.median(log_values))),
                "n_tiles": len(values),
                "file_set_name": file_set_name,
            }
        )
    return {led: sorted(rows, key=lambda row: (row["days"], row["repeat"])) for led, rows in series.items()}


def _read_photodiode_series(path, signal_field):
    series = defaultdict(list)
    seen = set()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"led", "repeat", "days", signal_field}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Photodiode inventory is missing fields: {sorted(missing)}")
        for row in reader:
            if "photodiode_dark_subtracted" in fields:
                flag = str(row["photodiode_dark_subtracted"]).strip().lower()
                if flag not in {"true", "1"}:
                    raise ValueError(
                        f"Photodiode row is not dark-subtracted: {row['file_set_name']}"
                    )
            led = int(row["led"])
            repeat = int(row["repeat"])
            key = (led, repeat)
            if key in seen:
                raise ValueError(f"Duplicate photodiode inventory row: {key}")
            seen.add(key)
            value = float(row[signal_field])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Photodiode signal must be positive: {row}")
            series[led].append(
                {
                    "repeat": repeat,
                    "days": float(row["days"]),
                    "value": value,
                    "n_tiles": 1,
                    "file_set_name": row["file_set_name"],
                }
            )
    return {led: sorted(rows, key=lambda row: (row["days"], row["repeat"])) for led, rows in series.items()}


def _event_day(rows, event_repeat):
    before = [row["days"] for row in rows if row["repeat"] < event_repeat]
    after = [row["days"] for row in rows if row["repeat"] >= event_repeat]
    if not before or not after:
        raise ValueError(
            f"Event repeat {event_repeat} needs observations on both sides"
        )
    return 0.5 * (max(before) + min(after))


def _fit_channel(rows, event_day, event_repeat, pre_points, post_points):
    times = np.asarray([row["days"] for row in rows], dtype=float)
    values = np.asarray([row["value"] for row in rows], dtype=float)
    fit = fit_log_step_spline(
        times,
        values,
        event_day,
        pre_points=pre_points,
        post_points=post_points,
    )
    selected_rows = [rows[index] for index in fit.selected_indices]
    return {
        "event_repeat": event_repeat,
        "event_day": event_day,
        "n_pre": int(np.count_nonzero(fit.selected_times < event_day)),
        "n_post": int(np.count_nonzero(fit.selected_times >= event_day)),
        "selected_repeats": ";".join(str(row["repeat"]) for row in selected_rows),
        "knot_days": ";".join(f"{value:.16g}" for value in fit.knot_times),
        "step_log": fit.step_log,
        "correction_factor": fit.correction_factor,
        "rmse_log": fit.rmse_log,
        "r2": fit.r2,
    }


def fit_side_swap_series(
    raw_statistics_path,
    inventory_path,
    event_repeat,
    *,
    pre_points=5,
    post_points=5,
    signal_field="photodiode_mean",
):
    """Return image and dark-subtracted photodiode fits for each LED."""
    image_series = _read_image_series(raw_statistics_path)
    photodiode_series = _read_photodiode_series(inventory_path, signal_field)
    rows = []
    for led in sorted(set(image_series).intersection(photodiode_series)):
        event_day = _event_day(image_series[led], event_repeat)
        for channel, series in (
            ("image", image_series[led]),
            ("photodiode", photodiode_series[led]),
        ):
            result = _fit_channel(
                series,
                event_day,
                event_repeat,
                pre_points,
                post_points,
            )
            rows.append({"led": led, "channel": channel, **result})
    if not rows:
        raise ValueError("No LED is present in both the image and photodiode inputs")
    return rows


def available_side_swap_points(raw_statistics_path, inventory_path, event_repeat):
    """Return the minimum available pre/post counts across common LED series."""
    image_series = _read_image_series(raw_statistics_path)
    photodiode_series = _read_photodiode_series(inventory_path, "photodiode_mean")
    common_leds = sorted(set(image_series).intersection(photodiode_series))
    if not common_leds:
        raise ValueError("No LED is present in both the image and photodiode inputs")
    counts = []
    for led in common_leds:
        event_day = _event_day(image_series[led], event_repeat)
        for series in (image_series[led], photodiode_series[led]):
            counts.append(
                (
                    sum(row["days"] < event_day for row in series),
                    sum(row["days"] >= event_day for row in series),
                )
            )
    return min(count[0] for count in counts), min(count[1] for count in counts)


OUTPUT_FIELDS = (
    "led",
    "channel",
    "event_repeat",
    "event_day",
    "n_pre",
    "n_post",
    "selected_repeats",
    "knot_days",
    "step_log",
    "correction_factor",
    "rmse_log",
    "r2",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit local LED-dependent image and photodiode side-swap factors"
    )
    parser.add_argument("--raw-statistics", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--event-repeat", required=True, type=int)
    parser.add_argument("--pre-points", type=int, default=5)
    parser.add_argument("--post-points", type=int, default=5)
    parser.add_argument("--photodiode-signal-field", default="photodiode_mean")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = fit_side_swap_series(
        args.raw_statistics,
        args.inventory,
        args.event_repeat,
        pre_points=args.pre_points,
        post_points=args.post_points,
        signal_field=args.photodiode_signal_field,
    )
    handle = sys.stdout
    close_handle = False
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("w", newline="")
        close_handle = True
    try:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if close_handle:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
