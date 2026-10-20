"""Compare local photodiode factors with Eric's HDF5 photodiode factors.

The local reduction has two distinct photodiode-related factors:

* ``photodiode_scale`` from the exposure inventory, applied while building
  tile statistics; and
* the fitted post-event photodiode side-swap factor, applied at the configured
  side-swap repeat.  This factor corrects the photodiode denominator, so its
  contribution to the current-count multiplier is its inverse.

Eric's ``photodiode_multiplier`` is read from the HDF5 calibration table.  The
plots keep these local terms separate and also show their product for context.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DETECTORS = ("WFI01", "WFI10")
LEDS = tuple(range(1, 7))
PLOT_TILE = 136
SCA_BY_DETECTOR = {"WFI01": "0", "WFI10": "9"}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_inventory(root, detector):
    rows = read_csv(root / f"{detector.lower()}_rrn_inventory.csv")
    return {(row["file_set_name"]): row for row in rows}


def load_side_swap(root, detector):
    rows = read_csv(
        root / "corrections" / f"{detector.lower()}_side_swap_calibration.csv"
    )
    return {
        int(row["led"]): row
        for row in rows
        if row["channel"] == "photodiode"
    }


def build_rows(matched_rows, eric_rows, local_root):
    eric_by_key = {
        (
            row["detector"],
            row["run_label"],
            int(row["led"]),
            row["sca"],
        ): row
        for row in eric_rows
    }
    output = []
    inventories = {detector: load_inventory(local_root, detector) for detector in DETECTORS}
    side_swap = {detector: load_side_swap(local_root, detector) for detector in DETECTORS}

    for matched in matched_rows:
        detector = matched["detector"]
        if int(matched["tile_id"]) != PLOT_TILE:
            continue
        led = int(matched["led"])
        inventory = inventories[detector][matched["local_file_set_name"]]
        eric = eric_by_key[(detector, matched["run_label"], led, SCA_BY_DETECTOR[detector])]
        side = side_swap[detector][led]
        repeat = int(inventory["repeat"])
        event_repeat = int(side["event_repeat"])
        telemetry_factor = float(inventory["photodiode_scale"])
        step_factor = (
            float(side["correction_factor"])
            if repeat >= event_repeat
            else 1.0
        )
        # The fitted photodiode step corrects the denominator in the image /
        # photodiode ratio; write_side_swap_scaled_statistics therefore uses
        # telemetry_factor / step_factor for the effective count multiplier.
        output.append(
            {
                "detector": detector,
                "run_label": matched["run_label"],
                "timestamp": matched["timestamp"],
                "days": float(matched["days"]),
                "led": led,
                "repeat": repeat,
                "event_repeat": event_repeat,
                "telemetry_factor": telemetry_factor,
                "side_swap_factor": step_factor,
                "combined_local_factor": telemetry_factor / step_factor,
                "eric_photodiode_multiplier": float(eric["photodiode_multiplier"]),
            }
        )

    for detector in DETECTORS:
        for led in LEDS:
            series = sorted(
                [row for row in output if row["detector"] == detector and row["led"] == led],
                key=lambda row: row["days"],
            )
            if not series:
                raise ValueError(f"No photodiode comparison rows for {detector}, LED {led}")
            first = series[0]
            for row in series:
                for field in ("telemetry_factor", "combined_local_factor", "eric_photodiode_multiplier"):
                    row[f"{field}_percent"] = 100.0 * (row[field] - 1.0)
                    row[f"{field}_relative_percent"] = 100.0 * (row[field] / first[field] - 1.0)

    return output


def series(rows, detector, led):
    return sorted(
        [row for row in rows if row["detector"] == detector and row["led"] == led],
        key=lambda row: row["days"],
    )


def configure_axes(axes, title, ylabel):
    for axis, led in zip(axes.flat, LEDS):
        axis.set_title(f"LED {led}")
        axis.set_xlabel("Days since first Eric product")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].figure.suptitle(title, y=0.985)


def plot_factors(output, rows, detector, relative):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fields = (
        ("telemetry_factor_relative_percent" if relative else "telemetry_factor_percent", "#d95f02"),
        ("combined_local_factor_relative_percent" if relative else "combined_local_factor_percent", "#7570b3"),
        ("eric_photodiode_multiplier_relative_percent" if relative else "eric_photodiode_multiplier_percent", "#1b9e77"),
    )
    labels = {
        fields[0][0]: "Ours: inventory telemetry scale",
        fields[1][0]: "Ours: telemetry / fitted PD denominator step",
        fields[2][0]: "Eric: HDF5 photodiode multiplier",
    }
    for axis, led in zip(axes.flat, LEDS):
        current = series(rows, detector, led)
        for field, color in fields:
            axis.plot(
                [row["days"] for row in current],
                [row[field] for row in current],
                color=color,
                linewidth=1.0,
                label=labels[field],
            )
        axis.axhline(0.0, color="0.25", linewidth=0.7)
        event_day = next(
            (row["days"] for row in current if row["repeat"] >= row["event_repeat"]),
            None,
        )
        if event_day is not None:
            axis.axvline(event_day, color="0.45", linestyle=":", linewidth=0.8)

    ylabel = "Relative to first product (%)" if relative else "Factor minus 1 (%)"
    title = (
        f"{detector}: photodiode correction factors relative to first product"
        if relative
        else f"{detector}: photodiode correction factors"
    )
    configure_axes(axes, title, ylabel)
    handles = [
        Line2D([0], [0], color="#d95f02", linewidth=1.5, label=labels[fields[0][0]]),
        Line2D([0], [0], color="#7570b3", linewidth=1.5, label=labels[fields[1][0]]),
        Line2D([0], [0], color="#1b9e77", linewidth=1.5, label=labels[fields[2][0]]),
        Line2D([0], [0], color="0.45", linestyle=":", linewidth=1.0, label="Local repeat-686 step"),
    ]
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_csv(path, rows):
    fields = (
        "detector",
        "run_label",
        "timestamp",
        "days",
        "led",
        "repeat",
        "event_repeat",
        "telemetry_factor",
        "side_swap_factor",
        "combined_local_factor",
        "eric_photodiode_multiplier",
        "telemetry_factor_percent",
        "combined_local_factor_percent",
        "eric_photodiode_multiplier_percent",
        "telemetry_factor_relative_percent",
        "combined_local_factor_relative_percent",
        "eric_photodiode_multiplier_relative_percent",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-rows", type=Path, required=True)
    parser.add_argument("--eric-corrections", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        read_csv(args.matched_rows),
        read_csv(args.eric_corrections),
        args.local_root,
    )
    write_csv(args.output_dir / "eric_local_photodiode_correction_matched.csv", rows)
    for detector in DETECTORS:
        plot_factors(
            args.output_dir / f"{detector.lower()}_photodiode_factors.png",
            rows,
            detector,
            relative=False,
        )
        plot_factors(
            args.output_dir / f"{detector.lower()}_photodiode_factors_relative.png",
            rows,
            detector,
            relative=True,
        )
    print(f"matched_rows={len(rows)}")
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
