"""Fit independent LED sinusoids and ACM steps directly to WFI data."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from analyze_wfi01_all_pixel_soc_model import DEFAULT_STEP_TIME, parse_mast_time


DATASET_COLORS = {"old": "tab:blue", "rrn": "tab:orange"}
DATASET_MARKERS = {"old": "o", "rrn": "s"}


def load_led_data(path, tile_id=0):
    values = {led: [] for led in range(1, 7)}
    event_time = parse_mast_time(DEFAULT_STEP_TIME)
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["tile_id"]) != tile_id:
                continue
            count = float(row["n_pair_valid"])
            if count <= 0.0:
                continue
            values[int(row["led"])].append(
                {
                    "repeat": int(row["repeat"]),
                    "days": float(row["days"]),
                    "mean_log_change": float(row["sum_log_change"]) / count,
                    "after_event": parse_mast_time(row["exposure_start_time"])
                    >= event_time,
                }
            )
    for led, rows in values.items():
        rows.sort(key=lambda row: row["days"])
        if len(rows) < 4:
            raise ValueError(f"LED {led} has too few observed points in {path}")
        baseline_after_event = float(rows[0]["after_event"])
        for row in rows:
            row["step_design"] = float(row["after_event"]) - baseline_after_event
        values[led] = {
            key: np.asarray([row[key] for row in rows])
            for key in ("repeat", "days", "mean_log_change", "step_design")
        }
    return values


def design_matrix(time_days, period_days, step_design):
    angular_frequency = 2.0 * np.pi / period_days
    return np.column_stack(
        (
            np.ones_like(time_days),
            np.sin(angular_frequency * time_days),
            np.cos(angular_frequency * time_days),
            step_design,
        )
    )


def fit_for_period(time_days, values, step_design, period_days):
    design = design_matrix(time_days, period_days, step_design)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = values - fitted
    return float(np.sum(residual * residual)), coefficients, fitted, residual


def fit_sinusoid_plus_acm(days, values, step_design):
    time_days = np.asarray(days, dtype=float) - float(days[0])
    values = np.asarray(values, dtype=float)
    step_design = np.asarray(step_design, dtype=float)
    span = float(np.ptp(time_days))
    if span <= 0.0:
        raise ValueError("LED data must span a positive time interval")
    minimum_period = max(2.0 * float(np.median(np.diff(time_days))), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)
    objective_values = np.asarray(
        [fit_for_period(time_days, values, step_design, period)[0] for period in period_grid]
    )
    best_index = int(np.argmin(objective_values))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(period_grid) - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        result = minimize_scalar(
            lambda log_period: fit_for_period(
                time_days,
                values,
                step_design,
                float(np.exp(log_period)),
            )[0],
            bounds=(
                np.log(period_grid[lower_index]),
                np.log(period_grid[upper_index]),
            ),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))
    sse, coefficients, fitted, residual = fit_for_period(
        time_days, values, step_design, period_days
    )
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    phase = float(np.arctan2(coefficients[2], coefficients[1]))
    return {
        "period_days": period_days,
        "offset_log_change": float(coefficients[0]),
        "amplitude_log_change": amplitude,
        "phase_rad": phase,
        "acm_step_log_change": float(coefficients[3]),
        "sse_log_change_squared": sse,
        "rmse_log_change": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares)
        if total_sum_squares > 0.0
        else np.nan,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
    }


def evaluate_components(time_days, fit, event_step_design):
    relative_days = np.asarray(time_days, dtype=float)
    period_days = fit["period_days"]
    angular_frequency = 2.0 * np.pi / period_days
    base = (
        fit["offset_log_change"]
        + fit["coefficients"][1] * np.sin(angular_frequency * relative_days)
        + fit["coefficients"][2] * np.cos(angular_frequency * relative_days)
    )
    full = base + fit["acm_step_log_change"] * np.asarray(event_step_design, dtype=float)
    return base, full


def fit_datasets(old_path, rrn_path, tile_id):
    loaded = {
        "old": load_led_data(old_path, tile_id),
        "rrn": load_led_data(rrn_path, tile_id),
    }
    fits = {dataset: {} for dataset in loaded}
    for dataset, values in loaded.items():
        for led, series in values.items():
            fits[dataset][led] = fit_sinusoid_plus_acm(
                series["days"],
                series["mean_log_change"],
                series["step_design"],
            )
    return loaded, fits


def plot_data_and_models(path, loaded, fits):
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        sharex=True,
        constrained_layout=True,
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        maximum_day = 0.0
        for dataset in ("old", "rrn"):
            series = loaded[dataset][led]
            fit = fits[dataset][led]
            days = series["days"]
            observed = 1000.0 * series["mean_log_change"]
            color = DATASET_COLORS[dataset]
            axis.scatter(
                days,
                observed,
                color=color,
                marker=DATASET_MARKERS[dataset],
                s=22,
                alpha=0.55,
                label=f"{dataset.upper()} data",
            )
            grid = np.linspace(days[0], days[-1], 600)
            grid_step = (grid >= days[series["step_design"] > 0.0][0]).astype(float)
            base, full = evaluate_components(grid - days[0], fit, grid_step)
            axis.plot(
                grid,
                1000.0 * full,
                color=color,
                linewidth=2.0,
                label=(
                    f"{dataset.upper()} full: P={fit['period_days']:.3g} d, "
                    f"step={1000.0 * fit['acm_step_log_change']:.2f} mlog, "
                    f"R2={fit['r2']:.3f}"
                ),
            )
            axis.plot(
                grid,
                1000.0 * base,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"{dataset.upper()} sinusoid only",
            )
            maximum_day = max(maximum_day, float(days[-1]))
        axis.set_title(f"LED {led}")
        axis.set_ylabel("Mean log change (millilog)")
        axis.set_xlim(left=0.0, right=maximum_day)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    for axis in axes[-1, :]:
        axis.set_xlabel("Days since first exposure")
    figure.suptitle(
        "Observed WFI01 tile-0 data versus per-LED sinusoid plus fitted ACM step"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary(path, loaded, fits):
    fields = (
        "dataset",
        "led",
        "n_points",
        "period_days",
        "offset_log_change",
        "amplitude_log_change",
        "phase_rad",
        "acm_step_log_change",
        "sse_log_change_squared",
        "rmse_log_change",
        "r2",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in ("old", "rrn"):
            for led in range(1, 7):
                fit = fits[dataset][led]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "led": led,
                        "n_points": loaded[dataset][led]["days"].size,
                        **{key: fit[key] for key in fields if key not in ("dataset", "led", "n_points")},
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--rrn", type=Path, required=True)
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    loaded, fits = fit_datasets(args.old, args.rrn, args.tile_id)
    plot_data_and_models(args.output, loaded, fits)
    write_summary(args.summary, loaded, fits)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary}")
    for dataset in ("old", "rrn"):
        for led in range(1, 7):
            fit = fits[dataset][led]
            print(
                f"{dataset} LED {led}: points={loaded[dataset][led]['days'].size}, "
                f"period={fit['period_days']:.8g} d, "
                f"ACM step={fit['acm_step_log_change']:.8g} log, "
                f"RMSE={fit['rmse_log_change']:.8g} log, R2={fit['r2']:.8g}"
            )


if __name__ == "__main__":
    main()
