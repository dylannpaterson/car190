"""Plot the inventory photodiode telemetry scale recorded by WFI reduction.

This is the per-exposure ``reference_signal / signal`` scale. It does not
include the later fitted photodiode denominator step from the side-swap
calibration, which can be supplied separately by the caller.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LED_COUNT = 6


def _timestamp(value):
    value = str(value or "").strip()
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _numeric(value, field, row_number):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field} on inventory row {row_number}: {value!r}"
        ) from error
    if not np.isfinite(parsed):
        raise ValueError(f"Non-finite {field} on inventory row {row_number}")
    return parsed


def load_inventory(path, detector=None):
    """Load per-exposure photodiode scale factors from one inventory CSV."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "detector",
            "led",
            "program",
            "repeat",
            "exposure_start_time",
            "photodiode_mean",
            "photodiode_reference",
            "photodiode_scale",
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Inventory is missing required fields: {', '.join(missing)}")
        source_rows = list(reader)

    detector_filter = detector.upper() if detector is not None else None
    selected = []
    for row_number, source in enumerate(source_rows, start=2):
        row_detector = str(source["detector"]).strip().upper()
        if detector_filter is not None and row_detector != detector_filter:
            continue
        led = int(source["led"])
        if led < 1 or led > LED_COUNT:
            raise ValueError(f"LED must be 1-{LED_COUNT} on inventory row {row_number}")
        scale = _numeric(source["photodiode_scale"], "photodiode_scale", row_number)
        if scale <= 0.0:
            raise ValueError(f"Photodiode scale must be positive on inventory row {row_number}")
        signal = _numeric(source["photodiode_mean"], "photodiode_mean", row_number)
        reference = _numeric(
            source["photodiode_reference"], "photodiode_reference", row_number
        )
        if signal <= 0.0 or reference <= 0.0:
            raise ValueError(
                f"Photodiode signal and reference must be positive on inventory row {row_number}"
            )
        selected.append(
            {
                "detector": row_detector,
                "led": led,
                "program": str(source["program"]).strip() or "unknown",
                "repeat": int(source["repeat"]),
                "file_set_name": str(source.get("file_set_name") or ""),
                "exposure_start_time": str(source["exposure_start_time"]),
                "days_value": source.get("days", ""),
                "timestamp": _timestamp(source["exposure_start_time"]),
                "photodiode_mean": signal,
                "photodiode_reference": reference,
                "photodiode_scale": scale,
                "correction_percent": 100.0 * (scale - 1.0),
                "photodiode_change_pct": source.get("photodiode_change_pct", ""),
                "photodiode_channel": str(source.get("photodiode_channel") or ""),
                "photodiode_source": str(source.get("photodiode_source") or ""),
            }
        )

    if not selected:
        label = detector_filter or "the inventory"
        raise ValueError(f"No inventory rows found for {label}")

    numeric_days = []
    for row in selected:
        try:
            day = float(row["days_value"])
        except (TypeError, ValueError):
            day = np.nan
        numeric_days.append(day)
    numeric_days = np.asarray(numeric_days, dtype=float)
    if np.all(np.isfinite(numeric_days)):
        days = numeric_days
    else:
        timestamps = [row["timestamp"] for row in selected]
        if any(timestamp is None for timestamp in timestamps):
            raise ValueError("Inventory needs finite days or exposure_start_time values")
        origin = min(timestamps)
        days = np.asarray(
            [(timestamp - origin).total_seconds() / 86400.0 for timestamp in timestamps],
            dtype=float,
        )
    for row, day in zip(selected, days):
        row["days"] = float(day)
    selected.sort(key=lambda row: (row["led"], row["days"], row["repeat"]))

    detectors = sorted({row["detector"] for row in selected})
    if len(detectors) != 1:
        raise ValueError(
            "An inventory plot must contain exactly one detector; "
            f"found {', '.join(detectors)}"
        )
    return selected, detectors[0]


def load_side_swap_events(path, detector=None):
    """Load per-LED side-swap event times from a calibration CSV."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"detector", "led", "channel", "event_repeat", "event_day"}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                f"Side-swap calibration is missing required fields: {', '.join(missing)}"
            )
        events = {}
        detector_filter = detector.upper() if detector is not None else None
        for row_number, row in enumerate(reader, start=2):
            row_detector = str(row["detector"]).strip().upper()
            if detector_filter is not None and row_detector != detector_filter:
                continue
            if str(row["channel"]).strip().lower() != "photodiode":
                continue
            led = int(row["led"])
            if led in events:
                raise ValueError(f"Duplicate photodiode side-swap LED {led}")
            events[led] = {
                "event_repeat": int(row["event_repeat"]),
                "event_day": _numeric(row["event_day"], "event_day", row_number),
            }
    if not events:
        label = detector_filter or "the calibration"
        raise ValueError(f"No photodiode side-swap events found for {label}")
    return events


def write_rows(path, rows):
    """Write the tidy correction data used by the plot."""
    fields = (
        "detector",
        "led",
        "program",
        "repeat",
        "days",
        "exposure_start_time",
        "file_set_name",
        "photodiode_mean",
        "photodiode_reference",
        "photodiode_scale",
        "correction_percent",
        "photodiode_change_pct",
        "photodiode_channel",
        "photodiode_source",
    )
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def plot_correction(path, rows, detector, quantity="factor", side_swap_events=None):
    """Plot one photodiode-correction panel for each LED."""
    if quantity not in {"factor", "percent"}:
        raise ValueError(f"Unsupported quantity: {quantity}")
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        sharex=True,
        constrained_layout=True,
    )
    axes = axes.reshape(-1)
    programs = sorted({row["program"] for row in rows})
    colors = plt.get_cmap("tab10", max(len(programs), 1))
    program_colors = {program: colors(index) for index, program in enumerate(programs)}
    reference = 1.0 if quantity == "factor" else 0.0
    y_label = (
        "Photodiode scale (reference / signal)"
        if quantity == "factor"
        else "Applied photodiode correction (%)"
    )

    for axis, led in zip(axes, range(1, LED_COUNT + 1)):
        led_rows = [row for row in rows if row["led"] == led]
        for program in programs:
            program_rows = [row for row in led_rows if row["program"] == program]
            if not program_rows:
                continue
            values = [
                row["photodiode_scale"]
                if quantity == "factor"
                else row["correction_percent"]
                for row in program_rows
            ]
            axis.scatter(
                [row["days"] for row in program_rows],
                values,
                color=program_colors[program],
                marker=".",
                s=24,
                edgecolors="none",
                label=f"Program {program}",
            )
        axis.axhline(
            reference,
            color="0.25",
            linestyle="--",
            linewidth=0.8,
            label="no correction" if led == 1 else None,
        )
        event = (side_swap_events or {}).get(led)
        if event is not None:
            axis.axvline(
                event["event_day"],
                color="#d95f02",
                linestyle=":",
                linewidth=1.0,
                label="side-swap boundary" if led == 1 else None,
            )
            axis.text(
                event["event_day"],
                0.02,
                f"swap r{event['event_repeat']}\n{event['event_day']:.2f} d",
                color="#a04000",
                fontsize=7,
                rotation=90,
                ha="right",
                va="bottom",
                transform=axis.get_xaxis_transform(),
            )
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
        if led == 1:
            axis.legend(frameon=False, fontsize=8)

    for axis in axes[-3:]:
        axis.set_xlabel("Days since first unified exposure")
    for axis in axes[::3]:
        axis.set_ylabel(y_label)
    figure.suptitle(
        f"{detector} inventory photodiode telemetry scale by LED\n"
        "Scale is reference signal divided by dark-subtracted signal"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--detector", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--side-swap-calibration",
        type=Path,
        default=None,
        help="Optional per-LED side-swap calibration CSV for event markers.",
    )
    parser.add_argument(
        "--quantity",
        choices=("factor", "percent"),
        default="factor",
        help="Plot the scale factor or its percent departure from unity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, detector = load_inventory(args.inventory, detector=args.detector)
    side_swap_events = (
        load_side_swap_events(args.side_swap_calibration, detector=detector)
        if args.side_swap_calibration is not None
        else None
    )
    output = args.output or args.inventory.with_name(
        f"{args.inventory.stem}_photodiode_correction.png"
    )
    csv_output = output.with_suffix(".csv")
    plot_correction(
        output,
        rows,
        detector,
        quantity=args.quantity,
        side_swap_events=side_swap_events,
    )
    write_rows(csv_output, rows)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
