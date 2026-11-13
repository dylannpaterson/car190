"""Plot window and baseline-method effects for one rate-step tile."""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wfi_production_layout import corrected_statistics_path, positive_integer
from wfi_production_plot_annotations import latest_observation_datetime, load_time_annotations


UTC = ZoneInfo("UTC")
STEP_TIME_TOLERANCE_DAYS = 1.0e-9
DEFAULT_STEP_UTC = "2026-10-17T22:37:21.121000+00:00"
DEFAULT_WINDOWS_HOURS = (48.0, 72.0)


def _parse_utc_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_tile(path):
    with np.load(path, allow_pickle=False) as source:
        days = np.asarray(source["observation_days"], dtype=float).reshape(-1)
        rates = np.asarray(source["tile_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
    finite = np.isfinite(days) & np.isfinite(rates)
    order = np.argsort(days[finite], kind="stable")
    return days[finite][order], rates[finite][order]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detector-output-root", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile", type=int, default=0)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    detector = args.detector.upper()
    if not 0 <= args.tile < 256:
        raise ValueError("tile must be between 0 and 255")
    windows_hours = tuple(args.windows_hours or DEFAULT_WINDOWS_HOURS)
    if any(window <= 0.0 for window in windows_hours):
        raise ValueError("window-hours must be positive")

    statistics_path = corrected_statistics_path(args.root, detector, args.tile_size)
    annotations = load_time_annotations(statistics_path)
    latest_observation = latest_observation_datetime(statistics_path, args.tile_size)
    step_utc = _parse_utc_timestamp(args.step_utc)
    step_day = (step_utc - annotations.start_utc).total_seconds() / 86400.0
    fit_path = (
        args.detector_output_root
        / detector.lower()
        / "fits"
        / f"tile_{args.tile:03d}"
        / "posterior_samples.npz"
    )
    days, rates = _load_tile(fit_path)
    hours = (days - step_day) * 24.0
    max_window = max(windows_hours)
    visible = (hours >= -max_window) & (hours <= 8.0)

    figure, axis = plt.subplots(figsize=(13, 7))
    axis.plot(
        hours[visible],
        rates[visible],
        color="0.35",
        marker="o",
        markersize=4,
        linewidth=1.1,
        label="Fitted rate samples",
    )
    post = np.flatnonzero(days >= step_day - STEP_TIME_TOLERANCE_DAYS)
    if post.size == 0:
        raise ValueError(f"No post-step sample found in {fit_path}")
    first_post = int(post[0])
    axis.scatter(
        [hours[first_post]],
        [rates[first_post]],
        color="#D55E00",
        edgecolor="black",
        linewidth=0.7,
        s=65,
        zorder=5,
        label="First post-step rate",
    )
    axis.axvline(0.0, color="#009E73", linestyle="-.", linewidth=1.2, label="Observed rate step")

    styles = {
        (48.0, "linear"): ("#0072B2", "-", "48 h linear"),
        (72.0, "linear"): ("#56B4E9", "--", "72 h linear"),
        (48.0, "median"): ("#D55E00", "-", "48 h median"),
        (72.0, "median"): ("#E69F00", "--", "72 h median"),
    }
    annotation_lines = []
    for window_hours in windows_hours:
        window_days = float(window_hours) / 24.0
        pre = (days >= step_day - window_days) & (
            days < step_day - STEP_TIME_TOLERANCE_DAYS
        )
        if np.count_nonzero(pre) < 2:
            raise ValueError(f"Insufficient pre-step samples for {window_hours:g} h")
        slope, intercept = np.polyfit(days[pre], rates[pre], 1)
        baselines = {
            "linear": float(slope * step_day + intercept),
            "median": float(np.median(rates[pre])),
        }
        fit_hours = np.linspace(-float(window_hours), 0.0, 100)
        for method in ("linear", "median"):
            color, linestyle, label = styles.get(
                (float(window_hours), method),
                (None, "-" if method == "linear" else "--", f"{window_hours:g} h {method}"),
            )
            baseline = baselines[method]
            if method == "linear":
                values = slope * (step_day + fit_hours / 24.0) + intercept
            else:
                values = np.full_like(fit_hours, baseline)
            difference = float(rates[first_post] - baseline)
            percentage = 100.0 * difference / baseline
            axis.plot(
                fit_hours,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
            )
            annotation_lines.append(
                f"{label}: baseline {baseline:.4f}; increase {difference:+.4f} nm/day "
                f"({percentage:+.2f}%)"
            )

    axis.set_xlim(-max_window, 8.0)
    axis.set_xlabel("Hours relative to observed rate step")
    axis.set_ylabel("Accumulation rate (nm/day)")
    axis.set_title(f"{detector} tile {args.tile:03d}: rate-step baseline systematic")
    axis.grid(alpha=0.2)
    axis.text(
        0.015,
        0.98,
        "\n".join(annotation_lines),
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    figure.suptitle(
        f"Observed rate step: {step_utc:%Y-%m-%d %H:%M:%S.%f} UTC; "
        "latest observation: "
        + (
            "unknown"
            if latest_observation is None
            else f"{latest_observation:%Y-%m-%d %H:%M:%S.%f} UTC"
        ),
        fontsize=10,
        y=0.98,
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
