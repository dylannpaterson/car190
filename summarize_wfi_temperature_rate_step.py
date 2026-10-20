"""Summarize corrected tile rate-step departures by detector."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))


def _quantiles(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return np.percentile(values, (16.0, 50.0, 84.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detector",
        dest="detectors",
        action="append",
        default=None,
        help="Restrict the summary to the selected detector(s); may be repeated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input_csv.open(newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    detectors = tuple(detector.upper() for detector in (args.detectors or DETECTORS))
    unknown = set(detectors).difference(DETECTORS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    grouped = {detector: [] for detector in detectors}
    for row in input_rows:
        grouped[row["detector"]].append(row)
    if any(not rows for rows in grouped.values()):
        missing = [detector for detector, rows in grouped.items() if not rows]
        raise ValueError(f"Missing detector rows: {missing}")

    fields = (
        "detector",
        "tile_count",
        "step_utc",
        "step_edt",
        "window_hours",
        "baseline_rate_nm_day_p16",
        "baseline_rate_nm_day_p50",
        "baseline_rate_nm_day_p84",
        "first_post_rate_nm_day_p16",
        "first_post_rate_nm_day_p50",
        "first_post_rate_nm_day_p84",
        "difference_nm_day_p16",
        "difference_nm_day_p50",
        "difference_nm_day_p84",
        "difference_percent_p16",
        "difference_percent_p50",
        "difference_percent_p84",
    )
    output_rows = []
    for detector in detectors:
        rows = grouped[detector]
        baseline = _quantiles(rows, "baseline_rate_at_step_nm_day")
        first_post = _quantiles(rows, "first_post_rate_nm_day")
        difference = _quantiles(rows, "difference_nm_day")
        percentage = _quantiles(rows, "difference_percent")
        output_rows.append(
            {
                "detector": detector,
                "tile_count": len(rows),
                "step_utc": rows[0]["step_utc"],
                "step_edt": rows[0]["step_edt"],
                "window_hours": rows[0]["window_hours"],
                "baseline_rate_nm_day_p16": baseline[0],
                "baseline_rate_nm_day_p50": baseline[1],
                "baseline_rate_nm_day_p84": baseline[2],
                "first_post_rate_nm_day_p16": first_post[0],
                "first_post_rate_nm_day_p50": first_post[1],
                "first_post_rate_nm_day_p84": first_post[2],
                "difference_nm_day_p16": difference[0],
                "difference_nm_day_p50": difference[1],
                "difference_nm_day_p84": difference[2],
                "difference_percent_p16": percentage[0],
                "difference_percent_p50": percentage[1],
                "difference_percent_p84": percentage[2],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "wfi_temperature_rate_step_per_detector.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    x = np.arange(len(detectors))
    difference = np.asarray(
        [[
            float(row["difference_nm_day_p16"]),
            float(row["difference_nm_day_p50"]),
            float(row["difference_nm_day_p84"]),
        ] for row in output_rows]
    )
    percentage = np.asarray(
        [[
            float(row["difference_percent_p16"]),
            float(row["difference_percent_p50"]),
            float(row["difference_percent_p84"]),
        ] for row in output_rows]
    )
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, values, title, ylabel in (
        (axes[0], difference, "Rate-step departure by detector", "Observed - baseline (nm/day)"),
        (axes[1], percentage, "Rate-step departure by detector", "Observed - baseline (%)"),
    ):
        median = values[:, 1]
        axis.errorbar(
            x,
            median,
            yerr=(median - values[:, 0], values[:, 2] - median),
            fmt="o",
            color="#0072B2",
            ecolor="#0072B2",
            elinewidth=1.2,
            capsize=4,
            markersize=5,
        )
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, detectors)
    axes[1].set_xlabel("Detector; markers are p50 and bars are p16-p84 across 256 tiles")
    step_utc = output_rows[0]["step_utc"]
    step_edt = output_rows[0]["step_edt"]
    window_hours = output_rows[0]["window_hours"]
    figure.suptitle(
        "Per-detector distribution of the observed rate-step departure",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.5,
        0.955,
        f"Observed rate step: {step_utc} UTC / {step_edt} EDT; "
        f"pre-step fit window: {float(window_hours):g} hours",
        ha="center",
        va="top",
        fontsize=9,
        color="0.3",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_plot = args.output_dir / "wfi_temperature_rate_step_per_detector.png"
    figure.savefig(output_plot, dpi=180)
    plt.close(figure)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_plot}")


if __name__ == "__main__":
    main()
