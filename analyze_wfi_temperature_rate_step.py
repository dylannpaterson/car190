"""Map per-tile rate departures from a pre-event rate extrapolation."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from plot_wfi_full_focal_plane_rates import (
    COLUMN_SCA,
    COLUMN_Y_OFFSET,
    SCAS,
    SCA_PITCH,
    SCA_SIZE,
    sca_position,
)
from plot_wfi_all_detector_trajectories import detector_output_dir
from wfi_production_layout import corrected_statistics_path, positive_integer
from wfi_production_plot_annotations import (
    ENCLOSURE_TEMPERATURE_CHANGE_UTC,
    latest_observation_label,
    load_time_annotations,
)


UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")
STEP_TIME_TOLERANCE_DAYS = 1.0e-9


def _format_timestamp(value):
    return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")


def _parse_utc_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("step-utc must include a UTC offset")
    return parsed.astimezone(UTC)


def _load_tile_step(fit_path, step_day, window_days):
    with np.load(fit_path, allow_pickle=False) as source:
        days = np.asarray(source["observation_days"], dtype=float).reshape(-1)
        rates = np.asarray(source["tile_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        tile_id = int(np.asarray(source["tile_ids"]).reshape(-1)[0])
    finite = np.isfinite(days) & np.isfinite(rates)
    days = days[finite]
    rates = rates[finite]
    order = np.argsort(days, kind="stable")
    days = days[order]
    rates = rates[order]

    boundary = step_day - STEP_TIME_TOLERANCE_DAYS
    pre_mask = (days >= step_day - window_days) & (days < boundary)
    post_indices = np.flatnonzero(days >= boundary)
    if np.count_nonzero(pre_mask) < 2 or post_indices.size == 0:
        raise ValueError(f"Insufficient pre/post step samples in {fit_path}")

    pre_days = days[pre_mask]
    pre_rates = rates[pre_mask]
    if np.unique(pre_days).size < 2:
        raise ValueError(f"Pre-step sample times are not distinct in {fit_path}")
    slope, intercept = np.polyfit(pre_days, pre_rates, 1)
    baseline_rate = float(slope * step_day + intercept)
    post_index = int(post_indices[0])
    post_day = float(days[post_index])
    post_rate = float(rates[post_index])
    difference = post_rate - baseline_rate
    percentage = 100.0 * difference / baseline_rate
    return {
        "tile_id": tile_id,
        "pre_sample_count": int(np.count_nonzero(pre_mask)),
        "pre_start_day": float(pre_days[0]),
        "pre_end_day": float(pre_days[-1]),
        "pre_slope_nm_day_per_day": float(slope),
        "baseline_rate_at_step_nm_day": baseline_rate,
        "first_post_day": post_day,
        "first_post_rate_nm_day": post_rate,
        "difference_nm_day": float(difference),
        "difference_percent": float(percentage),
    }


def _map_values(rows, field):
    values = {}
    for detector in SCAS:
        grid = np.full((16, 16), np.nan, dtype=float)
        for row in rows.get(detector, ()):
            tile_id = int(row["tile_id"])
            grid[tile_id // 16, tile_id % 16] = float(row[field])
        values[detector] = grid
    return values


def _draw_map(axis, values, title, colorbar_label, limit):
    image = None
    for detector in SCAS:
        x, y = sca_position(detector)
        image = axis.imshow(
            values[detector],
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            extent=(x, x + SCA_SIZE, y, y + SCA_SIZE),
        )
        axis.add_patch(
            Rectangle(
                (x, y),
                SCA_SIZE,
                SCA_SIZE,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )
        axis.text(
            x + SCA_SIZE / 2,
            y + SCA_SIZE / 2,
            detector,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
        )
        for tile in range(1, 16):
            position = SCA_SIZE * tile / 16.0
            axis.plot(
                (x + position, x + position),
                (y, y + SCA_SIZE),
                color="white",
                alpha=0.35,
                linewidth=0.35,
            )
            axis.plot(
                (x, x + SCA_SIZE),
                (y + position, y + position),
                color="white",
                alpha=0.35,
                linewidth=0.35,
            )
    axis.set_aspect("equal")
    axis.set_xlim(-1.0, SCA_PITCH * 6 - 1.0)
    axis.set_ylim(min(COLUMN_Y_OFFSET) - 1.0, SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title)
    axis.set_xlabel("Roman WFI focal-plane mosaic")
    axis.plot(
        SCA_PITCH * 3 - 0.5,
        SCA_PITCH * 2 + COLUMN_Y_OFFSET[2] + 1.2,
        marker="x",
        color="#d62728",
        markersize=7,
        markeredgewidth=1.5,
    )
    figure = axis.figure
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label=colorbar_label)


def _write_rows(path, rows):
    fields = tuple(rows[SCAS[0]][0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector in SCAS:
            writer.writerows(rows[detector])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument(
        "--detector",
        dest="detectors",
        action="append",
        default=None,
        help="Restrict the analysis to the selected detector(s); may be repeated.",
    )
    parser.add_argument(
        "--detector-output-root",
        type=Path,
        default=None,
        help="Direct directory containing detector subdirectories (wfi01, wfi02, ...).",
    )
    parser.add_argument(
        "--step-utc",
        default=None,
        help="UTC timestamp defining the observed rate-step boundary; defaults to the enclosure event.",
    )
    parser.add_argument("--window-hours", type=float, default=48.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_hours <= 0.0:
        raise ValueError("window-hours must be positive")
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detectors = tuple(detector.upper() for detector in (args.detectors or SCAS))
    unknown = set(detectors).difference(SCAS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    statistics_path = corrected_statistics_path(root, detectors[0], args.tile_size)
    annotations = load_time_annotations(statistics_path)
    step_utc = (
        _parse_utc_timestamp(args.step_utc)
        if args.step_utc is not None
        else ENCLOSURE_TEMPERATURE_CHANGE_UTC
    )
    step_day = (
        step_utc - annotations.start_utc
    ).total_seconds() / 86400.0
    window_days = float(args.window_hours) / 24.0
    rows = {}
    for detector in detectors:
        if args.detector_output_root is not None:
            fit_root = Path(args.detector_output_root) / detector.lower() / "fits"
        else:
            fit_root = (
                detector_output_dir(
                    root,
                    detector,
                    "production",
                    tile_size=args.tile_size,
                    fit_subdir=Path("relative_response"),
                )
                / "fits"
            )
        detector_rows = []
        for tile_id in range(256):
            result = _load_tile_step(
                fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz",
                step_day,
                window_days,
            )
            result["detector"] = detector
            result["step_utc"] = _format_timestamp(step_utc)
            result["step_edt"] = _format_timestamp(step_utc.astimezone(EASTERN))
            result["window_hours"] = float(args.window_hours)
            detector_rows.append(result)
        rows[detector] = detector_rows

    all_rows = [row for detector in detectors for row in rows[detector]]
    fields = (
        "detector",
        "tile_id",
        "step_utc",
        "step_edt",
        "window_hours",
        "pre_sample_count",
        "pre_start_day",
        "pre_end_day",
        "pre_slope_nm_day_per_day",
        "baseline_rate_at_step_nm_day",
        "first_post_day",
        "first_post_rate_nm_day",
        "difference_nm_day",
        "difference_percent",
    )
    with (output_dir / "wfi_temperature_rate_step_extrapolation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    absolute = _map_values(rows, "difference_nm_day")
    percentage = _map_values(rows, "difference_percent")
    absolute_limit = max(
        float(np.max(np.abs(values[np.isfinite(values)])))
        for values in absolute.values()
        if np.isfinite(values).any()
    )
    percentage_limit = max(
        float(np.max(np.abs(values[np.isfinite(values)])))
        for values in percentage.values()
        if np.isfinite(values).any()
    )
    absolute_limit = max(absolute_limit, 1.0e-6)
    percentage_limit = max(percentage_limit, 1.0e-6)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    _draw_map(axes[0], absolute, "Rate departure (nm/day)", "Observed - extrapolated (nm/day)", absolute_limit)
    _draw_map(axes[1], percentage, "Rate departure (%)", "Observed - extrapolated (%)", percentage_limit)
    event_utc = ENCLOSURE_TEMPERATURE_CHANGE_UTC.astimezone(UTC)
    event_edt = ENCLOSURE_TEMPERATURE_CHANGE_UTC.astimezone(EASTERN)
    figure.suptitle(
        f"Rate departure at the observed step from a {args.window_hours:g}-hour "
        "pre-step linear extrapolation",
        fontsize=13,
        y=0.99,
    )
    figure.text(
        0.5,
        0.958,
        f"Observed rate step: {_format_timestamp(step_utc)} UTC / "
        f"{_format_timestamp(step_utc.astimezone(EASTERN))} EDT; "
        f"nominal enclosure +4 K event: {_format_timestamp(event_utc)} UTC / "
        f"{_format_timestamp(event_edt)} EDT",
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
    figure.savefig(output_dir / "wfi_temperature_rate_step_extrapolation.png", dpi=180)
    plt.close(figure)
    print(f"Wrote {output_dir / 'wfi_temperature_rate_step_extrapolation.png'}")
    print(f"Wrote {output_dir / 'wfi_temperature_rate_step_extrapolation.csv'}")
    print(f"Step day {step_day:.9f}; pre-step window {args.window_hours:g} h")


if __name__ == "__main__":
    main()
