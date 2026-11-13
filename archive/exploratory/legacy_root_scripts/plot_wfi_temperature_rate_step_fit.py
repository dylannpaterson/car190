"""Plot fitted rate samples and the pre-event linear rate extrapolation."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_wfi_all_detector_trajectories import detector_output_dir
from wfi_production_layout import corrected_statistics_path, positive_integer
from wfi_production_plot_annotations import (
    ENCLOSURE_TEMPERATURE_CHANGE_UTC,
    latest_observation_label,
    load_time_annotations,
)


EASTERN = ZoneInfo("America/New_York")
STEP_TIME_TOLERANCE_DAYS = 1.0e-9


def _summary_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_tile(value):
    detector, tile_id = str(value).split(":", 1)
    detector = detector.upper()
    if not detector.startswith("WFI"):
        raise ValueError(f"Invalid detector in --tile: {value}")
    return detector, int(tile_id)


def _parse_utc_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(ZoneInfo("UTC"))


def _default_tiles(rows):
    differences = np.asarray([float(row["difference_nm_day"]) for row in rows])
    median_index = int(np.argsort(np.abs(differences - np.median(differences)))[0])
    selected = [
        ("largest positive", rows[int(np.argmax(differences))]),
        ("most negative", rows[int(np.argmin(differences))]),
        ("median departure", rows[median_index]),
    ]
    wfi01 = next(row for row in rows if row["detector"] == "WFI01" and row["tile_id"] == "0")
    selected.append(("WFI01 tile 000", wfi01))
    unique = []
    seen = set()
    for label, row in selected:
        key = (row["detector"], row["tile_id"])
        if key not in seen:
            unique.append((label, row))
            seen.add(key)
    return unique


def _load_series(fit_path, step_day, window_days):
    with np.load(fit_path, allow_pickle=False) as source:
        days = np.asarray(source["observation_days"], dtype=float).reshape(-1)
        rates = np.asarray(source["tile_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
    finite = np.isfinite(days) & np.isfinite(rates)
    order = np.argsort(days[finite], kind="stable")
    days = days[finite][order]
    rates = rates[finite][order]
    boundary = step_day - STEP_TIME_TOLERANCE_DAYS
    pre = (days >= step_day - window_days) & (days < boundary)
    post = days >= boundary
    if np.count_nonzero(pre) < 2 or not np.any(post):
        raise ValueError(f"Insufficient pre/post samples in {fit_path}")
    slope, intercept = np.polyfit(days[pre], rates[pre], 1)
    post_index = int(np.flatnonzero(post)[0])
    baseline = float(slope * step_day + intercept)
    post_rate = float(rates[post_index])
    return {
        "days": days,
        "rates": rates,
        "pre": pre,
        "post": post,
        "slope": float(slope),
        "intercept": float(intercept),
        "baseline": baseline,
        "post_index": post_index,
        "post_rate": post_rate,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Step-analysis CSV; defaults to the baseline relative-response CSV.",
    )
    parser.add_argument(
        "--detector-output-root",
        type=Path,
        default=None,
        help="Direct directory containing detector subdirectories (wfi01, wfi02, ...).",
    )
    parser.add_argument("--step-utc", default=None)
    parser.add_argument("--window-hours", type=float, default=48.0)
    parser.add_argument(
        "--tile",
        action="append",
        help="Optional DETECTOR:TILE selection; defaults to representative map tiles.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_hours <= 0.0:
        raise ValueError("window-hours must be positive")
    root = Path(args.root)
    statistics_path = corrected_statistics_path(root, "WFI01", args.tile_size)
    annotations = load_time_annotations(statistics_path)
    summary_path = args.summary_csv or (
        root
        / "production"
        / "relative_response"
        / "wfi_temperature_rate_step_extrapolation.csv"
    )
    summary = _summary_rows(summary_path)
    step_utc = (
        _parse_utc_timestamp(args.step_utc)
        if args.step_utc is not None
        else _parse_utc_timestamp(summary[0]["step_utc"])
    )
    step_day = (step_utc - annotations.start_utc).total_seconds() / 86400.0
    window_days = float(args.window_hours) / 24.0
    selected = (
        [(f"{detector} tile {tile_id:03d}", next(
            row
            for row in summary
            if row["detector"] == detector and int(row["tile_id"]) == tile_id
        )) for detector, tile_id in (_parse_tile(value) for value in args.tile)]
        if args.tile
        else _default_tiles(summary)
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    axes = axes.flat
    for axis, (label, row) in zip(axes, selected):
        detector = row["detector"]
        tile_id = int(row["tile_id"])
        if args.detector_output_root is not None:
            fit_root = Path(args.detector_output_root) / detector.lower() / "fits"
        else:
            fit_root = detector_output_dir(
                root,
                detector,
                "production",
                tile_size=args.tile_size,
                fit_subdir=Path("relative_response"),
            ) / "fits"
        series = _load_series(
            fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz",
            step_day,
            window_days,
        )
        hours = (series["days"] - step_day) * 24.0
        visible = (hours >= -float(args.window_hours)) & (hours <= 8.0)
        pre_visible = series["pre"] & visible
        post_visible = series["post"] & visible
        axis.plot(
            hours[pre_visible],
            series["rates"][pre_visible],
            color="#0072B2",
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            label="Pre-step fitted rate",
        )
        axis.plot(
            hours[post_visible],
            series["rates"][post_visible],
            color="#D55E00",
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            label="Post-step fitted rate",
        )
        fit_hours = np.linspace(-float(args.window_hours), 0.0, 100)
        fit_days = step_day + fit_hours / 24.0
        axis.plot(
            fit_hours,
            series["slope"] * fit_days + series["intercept"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Pre-step linear fit",
        )
        axis.axvline(0.0, color="#009E73", linestyle="-.", linewidth=1.0, label="Observed rate step")
        axis.scatter(
            [0.0],
            [series["baseline"]],
            color="black",
            marker="x",
            s=45,
            zorder=5,
            label="Baseline at step",
        )
        post_hours = hours[series["post_index"]]
        axis.scatter(
            [post_hours],
            [series["post_rate"]],
            color="#D55E00",
            edgecolor="black",
            linewidth=0.6,
            s=45,
            zorder=5,
            label="First post-step rate",
        )
        difference = float(row["difference_nm_day"])
        percentage = float(row["difference_percent"])
        axis.set_title(f"{label}: {detector} tile {tile_id:03d}\n"
                       f"departure {difference:+.4f} nm/day ({percentage:+.3f}%)")
        axis.set_xlabel("Hours relative to observed rate step")
        axis.set_ylabel("Accumulation rate (nm/day)")
        axis.set_xlim(-float(args.window_hours), 8.0)
        axis.grid(alpha=0.2)
        axis.text(
            0.02,
            0.97,
            f"baseline @ step = {series['baseline']:.4f}\n"
            f"first post = {series['post_rate']:.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    event_utc = ENCLOSURE_TEMPERATURE_CHANGE_UTC.astimezone(ZoneInfo("UTC"))
    event_edt = ENCLOSURE_TEMPERATURE_CHANGE_UTC.astimezone(EASTERN)
    figure.suptitle(
        "Pre-step rate trend and post-step fitted rate",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.5,
        0.958,
        f"Observed rate step: {step_utc:%Y-%m-%d %H:%M:%S.%f} UTC / "
        f"{step_utc.astimezone(EASTERN):%Y-%m-%d %H:%M:%S.%f} EDT; "
        f"nominal enclosure +4 K event: {event_utc:%Y-%m-%d %H:%M:%S} UTC / "
        f"{event_edt:%Y-%m-%d %H:%M:%S} EDT",
        ha="center",
        va="top",
        fontsize=7,
        color="0.3",
    )
    figure.text(
        0.5,
        0.941,
        f"{args.window_hours:g}-hour pre-step baseline; first post-step fitted sample; "
        f"{latest_observation_label(root, args.tile_size)}",
        ha="center",
        va="top",
        fontsize=7,
        color="0.3",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=5, frameon=False)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.91))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")
    print("Selected: " + ", ".join(f"{label} ({row['detector']}:{row['tile_id']})" for label, row in selected))


if __name__ == "__main__":
    main()
