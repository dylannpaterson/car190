"""Compare rate-step estimates across windows and pre-step baseline methods."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wfi_production_layout import corrected_statistics_path, positive_integer
from wfi_production_plot_annotations import load_time_annotations


UTC = ZoneInfo("UTC")
STEP_TIME_TOLERANCE_DAYS = 1.0e-9
DEFAULT_STEP_UTC = "2026-10-17T22:37:21.121000+00:00"
DEFAULT_WINDOWS_HOURS = (48.0, 72.0)
DEFAULT_METHODS = ("linear", "median")


def _parse_utc_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_tile_rates(path, step_day, window_hours):
    with np.load(path, allow_pickle=False) as source:
        days = np.asarray(source["observation_days"], dtype=float).reshape(-1)
        rates = np.asarray(source["tile_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        tile_id = int(np.asarray(source["tile_ids"]).reshape(-1)[0])
    finite = np.isfinite(days) & np.isfinite(rates)
    days = days[finite]
    rates = rates[finite]
    order = np.argsort(days, kind="stable")
    days = days[order]
    rates = rates[order]

    window_days = float(window_hours) / 24.0
    boundary = step_day - STEP_TIME_TOLERANCE_DAYS
    pre = (days >= step_day - window_days) & (days < boundary)
    post = np.flatnonzero(days >= boundary)
    if np.count_nonzero(pre) < 2 or post.size == 0:
        raise ValueError(f"Insufficient pre/post step samples in {path}")
    pre_days = days[pre]
    pre_rates = rates[pre]
    if np.unique(pre_days).size < 2:
        raise ValueError(f"Pre-step sample times are not distinct in {path}")

    slope, intercept = np.polyfit(pre_days, pre_rates, 1)
    post_index = int(post[0])
    return {
        "tile_id": tile_id,
        "pre_sample_count": int(np.count_nonzero(pre)),
        "pre_slope_nm_day_per_day": float(slope),
        "pre_median_rate_nm_day": float(np.median(pre_rates)),
        "linear_baseline_rate_nm_day": float(slope * step_day + intercept),
        "first_post_rate_nm_day": float(rates[post_index]),
    }


def _summary(values):
    return np.percentile(np.asarray(values, dtype=float), (16.0, 50.0, 84.0))


def _summary_row(detector, window_hours, method, rows):
    baseline = np.asarray(
        [row["baseline_rate_nm_day"] for row in rows], dtype=float
    )
    first_post = np.asarray(
        [row["first_post_rate_nm_day"] for row in rows], dtype=float
    )
    difference = first_post - baseline
    percentage = 100.0 * difference / baseline
    baseline_q = _summary(baseline)
    first_post_q = _summary(first_post)
    difference_q = _summary(difference)
    percentage_q = _summary(percentage)
    return {
        "detector": detector,
        "window_hours": float(window_hours),
        "baseline_method": method,
        "tile_count": len(rows),
        "baseline_rate_p16": baseline_q[0],
        "baseline_rate_p50": baseline_q[1],
        "baseline_rate_p84": baseline_q[2],
        "first_post_rate_p16": first_post_q[0],
        "first_post_rate_p50": first_post_q[1],
        "first_post_rate_p84": first_post_q[2],
        "difference_nm_day_p16": difference_q[0],
        "difference_nm_day_p50": difference_q[1],
        "difference_nm_day_p84": difference_q[2],
        "difference_percent_p16": percentage_q[0],
        "difference_percent_p50": percentage_q[1],
        "difference_percent_p84": percentage_q[2],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detector-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", action="append", required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument("--step-utc", default=DEFAULT_STEP_UTC)
    parser.add_argument(
        "--window-hours",
        type=float,
        action="append",
        dest="windows_hours",
        default=None,
        help="Pre-step windows to compare; may be repeated. Defaults to 48 and 72.",
    )
    parser.add_argument(
        "--baseline-method",
        choices=DEFAULT_METHODS,
        action="append",
        dest="baseline_methods",
        default=None,
        help="Baseline methods to compare; may be repeated. Defaults to linear and median.",
    )
    return parser.parse_args()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _errorbars(row, prefix):
    return np.asarray(
        [
            float(row[f"{prefix}_p50"]) - float(row[f"{prefix}_p16"]),
            float(row[f"{prefix}_p84"]) - float(row[f"{prefix}_p50"]),
        ]
    ).reshape(2, 1)


def _plot(output, rows, detectors, windows_hours, methods):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    positions = np.arange(len(detectors), dtype=float)
    styles = {
        (48.0, "linear"): ("#0072B2", "o", "48 h linear"),
        (72.0, "linear"): ("#56B4E9", "o", "72 h linear"),
        (48.0, "median"): ("#D55E00", "s", "48 h median"),
        (72.0, "median"): ("#E69F00", "s", "72 h median"),
    }
    offsets = np.linspace(-0.24, 0.24, len(windows_hours) * len(methods))
    for offset, window_hours, method in zip(
        offsets,
        (window for window in windows_hours for _method in methods),
        (method for _window in windows_hours for method in methods),
    ):
        key = (float(window_hours), method)
        color, marker, label = styles.get(
            key, (None, "o", f"{window_hours:g} h {method}")
        )
        selected = {
            row["detector"]: row
            for row in rows
            if float(row["window_hours"]) == float(window_hours)
            and row["baseline_method"] == method
        }
        ordered = [selected[detector] for detector in detectors]
        axes[0].errorbar(
            positions + offset,
            [float(row["difference_nm_day_p50"]) for row in ordered],
            yerr=np.hstack([_errorbars(row, "difference_nm_day") for row in ordered]),
            color=color,
            marker=marker,
            linestyle="none",
            capsize=4,
            label=label,
        )
        axes[1].errorbar(
            positions + offset,
            [float(row["difference_percent_p50"]) for row in ordered],
            yerr=np.hstack([_errorbars(row, "difference_percent") for row in ordered]),
            color=color,
            marker=marker,
            linestyle="none",
            capsize=4,
            label=label,
        )

    axes[0].set_title("Estimated rate increase")
    axes[0].set_ylabel("Post-step minus baseline (nm/day)")
    axes[1].set_title("Estimated relative increase")
    axes[1].set_ylabel("Post-step minus baseline (%)")
    for axis in axes:
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_xticks(positions, detectors)
        axis.grid(alpha=0.2, axis="y")
    figure.suptitle(
        "Rate-step systematic effect: window and pre-step baseline method",
        fontsize=14,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    windows_hours = tuple(args.windows_hours or DEFAULT_WINDOWS_HOURS)
    methods = tuple(args.baseline_methods or DEFAULT_METHODS)
    if any(window <= 0.0 for window in windows_hours):
        raise ValueError("window-hours must be positive")
    root = Path(args.root)
    detector_output_root = Path(args.detector_output_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detectors = tuple(detector.upper() for detector in args.detector)
    step_utc = _parse_utc_timestamp(args.step_utc)

    rows_by_condition = {}
    tile_rows = []
    summary_rows = []
    for detector in detectors:
        statistics_path = corrected_statistics_path(root, detector, args.tile_size)
        annotations = load_time_annotations(statistics_path)
        step_day = (step_utc - annotations.start_utc).total_seconds() / 86400.0
        for window_hours in windows_hours:
            condition_rows = {method: [] for method in methods}
            for tile_id in range(256):
                fit_path = (
                    detector_output_root
                    / detector.lower()
                    / "fits"
                    / f"tile_{tile_id:03d}"
                    / "posterior_samples.npz"
                )
                loaded = _load_tile_rates(fit_path, step_day, window_hours)
                for method in methods:
                    baseline = loaded[
                        "linear_baseline_rate_nm_day"
                        if method == "linear"
                        else "pre_median_rate_nm_day"
                    ]
                    difference = loaded["first_post_rate_nm_day"] - baseline
                    percentage = 100.0 * difference / baseline
                    row = {
                        "detector": detector,
                        "tile_id": loaded["tile_id"],
                        "window_hours": float(window_hours),
                        "baseline_method": method,
                        "pre_sample_count": loaded["pre_sample_count"],
                        "pre_slope_nm_day_per_day": loaded["pre_slope_nm_day_per_day"],
                        "pre_median_rate_nm_day": loaded["pre_median_rate_nm_day"],
                        "baseline_rate_nm_day": baseline,
                        "first_post_rate_nm_day": loaded["first_post_rate_nm_day"],
                        "difference_nm_day": difference,
                        "difference_percent": percentage,
                    }
                    condition_rows[method].append(row)
                    tile_rows.append(row)
            rows_by_condition[(detector, float(window_hours))] = condition_rows
            for method in methods:
                summary_rows.append(
                    _summary_row(
                        detector,
                        window_hours,
                        method,
                        condition_rows[method],
                    )
                )

    _write_csv(output_dir / "wfi_rate_step_systematic_per_tile.csv", tile_rows)
    _write_csv(output_dir / "wfi_rate_step_systematic_per_detector.csv", summary_rows)
    _plot(
        output_dir / "wfi_rate_step_systematic_comparison.png",
        summary_rows,
        detectors,
        windows_hours,
        methods,
    )
    print(f"Wrote {output_dir / 'wfi_rate_step_systematic_per_tile.csv'}")
    print(f"Wrote {output_dir / 'wfi_rate_step_systematic_per_detector.csv'}")
    print(f"Wrote {output_dir / 'wfi_rate_step_systematic_comparison.png'}")


if __name__ == "__main__":
    main()
