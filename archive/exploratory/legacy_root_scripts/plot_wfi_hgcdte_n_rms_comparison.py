"""Compare residual RMS for baseline and relaxed HgCdTe-n fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _quantiles(values):
    return np.percentile(np.asarray(values, dtype=float), (16.0, 50.0, 84.0))


def _read_detector(comparison_root, detector):
    path = Path(comparison_root) / detector.lower() / "wfi_hgcdte_n_sensitivity_per_tile.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No comparison rows found in {path}")
    baseline = np.asarray(
        [float(row["baseline_residual_rms_log"]) for row in rows], dtype=float
    )
    sensitivity = np.asarray(
        [float(row["sensitivity_residual_rms_log"]) for row in rows], dtype=float
    )
    if not np.isfinite(baseline).all() or not np.isfinite(sensitivity).all():
        raise ValueError(f"Non-finite residual RMS found in {path}")
    delta = sensitivity - baseline
    percent_change = 100.0 * delta / baseline
    return {
        "detector": detector,
        "tile_count": len(rows),
        "baseline": _quantiles(baseline),
        "sensitivity": _quantiles(sensitivity),
        "delta": _quantiles(delta),
        "percent_change": _quantiles(percent_change),
    }


def _summary_row(summary):
    baseline = summary["baseline"]
    sensitivity = summary["sensitivity"]
    delta = summary["delta"]
    percent_change = summary["percent_change"]
    return {
        "detector": summary["detector"],
        "tile_count": summary["tile_count"],
        "baseline_rms_p16": baseline[0],
        "baseline_rms_p50": baseline[1],
        "baseline_rms_p84": baseline[2],
        "sensitivity_rms_p16": sensitivity[0],
        "sensitivity_rms_p50": sensitivity[1],
        "sensitivity_rms_p84": sensitivity[2],
        "rms_delta_p16": delta[0],
        "rms_delta_p50": delta[1],
        "rms_delta_p84": delta[2],
        "rms_change_percent_p16": percent_change[0],
        "rms_change_percent_p50": percent_change[1],
        "rms_change_percent_p84": percent_change[2],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", action="append", required=True)
    return parser.parse_args()


def _errorbars(quantiles):
    return np.asarray([[quantiles[1] - quantiles[0]], [quantiles[2] - quantiles[1]]])


def main():
    args = parse_args()
    detectors = tuple(detector.upper() for detector in args.detector)
    summaries = [_read_detector(args.comparison_root, detector) for detector in detectors]
    rows = [_summary_row(summary) for summary in summaries]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    positions = np.arange(len(rows), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    baseline = [summary["baseline"] for summary in summaries]
    sensitivity = [summary["sensitivity"] for summary in summaries]
    percent_change = [summary["percent_change"] for summary in summaries]
    baseline_medians = np.asarray([values[1] for values in baseline])
    sensitivity_medians = np.asarray([values[1] for values in sensitivity])
    percent_medians = np.asarray([values[1] for values in percent_change])

    axes[0].errorbar(
        positions - 0.12,
        baseline_medians,
        yerr=np.asarray([_errorbars(values).reshape(2) for values in baseline]).T,
        fmt="o",
        capsize=4,
        linewidth=1.2,
        label="Relative-response baseline",
    )
    axes[0].errorbar(
        positions + 0.12,
        sensitivity_medians,
        yerr=np.asarray([_errorbars(values).reshape(2) for values in sensitivity]).T,
        fmt="o",
        capsize=4,
        linewidth=1.2,
        label="Full-TMM HgCdTe n sigma 0.25",
    )
    axes[0].set_ylabel("Residual RMS (log)")
    axes[0].set_title("Residual RMS")
    axes[0].set_xticks(positions, detectors)
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].errorbar(
        positions,
        percent_medians,
        yerr=np.asarray([_errorbars(values).reshape(2) for values in percent_change]).T,
        fmt="o",
        color="#0072B2",
        capsize=4,
        linewidth=1.2,
    )
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("RMS change from baseline (%)")
    axes[1].set_title("Relaxed prior minus baseline")
    axes[1].set_xticks(positions, detectors)
    axes[1].grid(alpha=0.2, axis="y")

    figure.suptitle("Per-tile residual RMS: baseline versus relaxed HgCdTe-n prior", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output}")
    for row in rows:
        print(
            f"{row['detector']}: RMS p50 {row['baseline_rms_p50']:.9f} -> "
            f"{row['sensitivity_rms_p50']:.9f}; change "
            f"{row['rms_change_percent_p50']:+.2f}%"
        )


if __name__ == "__main__":
    main()
