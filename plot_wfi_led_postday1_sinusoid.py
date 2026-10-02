"""Fit one WFI LED's post-day-one ratios with a sinusoid and two ACM steps."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from analyze_wfi01_all_pixel_soc_model import DEFAULT_STEP_TIME, parse_mast_time


def load_led_data(path, led, tile_id, minimum_day):
    event_time = parse_mast_time(DEFAULT_STEP_TIME)
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["tile_id"]) != tile_id or int(row["led"]) != led:
                continue
            count = float(row["n_pair_valid"])
            if count <= 0.0:
                continue
            rows.append(
                {
                    "days": float(row["days"]),
                    "mean_log_change": float(row["sum_log_change"]) / count,
                    "repeat": int(row["repeat"]),
                    "after_event": parse_mast_time(row["exposure_start_time"])
                    >= event_time,
                }
            )
    rows.sort(key=lambda row: row["days"])
    fit_rows = [row for row in rows if row["days"] > minimum_day]
    if len(fit_rows) < 6:
        raise ValueError("Too few post-cutoff observations for a sinusoid fit")

    days = np.asarray([row["days"] for row in rows], dtype=float)
    values = np.asarray([row["mean_log_change"] for row in rows], dtype=float)
    first_step_design = np.asarray(
        [float(row["after_event"]) for row in rows], dtype=float
    )
    first_step_design -= first_step_design[0]

    late_repeat = max(row["repeat"] for row in rows)
    late_step_design = np.asarray(
        [float(row["repeat"] == late_repeat) for row in rows], dtype=float
    )
    late_step_design -= late_step_design[0]

    first_event_day = (
        float(days[np.flatnonzero(first_step_design > 0.0)[0]])
        if np.any(first_step_design > 0.0)
        else None
    )
    late_event_day = (
        float(days[np.flatnonzero(late_step_design > 0.0)[0]])
        if np.any(late_step_design > 0.0)
        else None
    )
    fit_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["days"] > minimum_day],
        dtype=int,
    )
    return (
        days,
        values,
        first_step_design,
        late_step_design,
        fit_indices,
        first_event_day,
        late_event_day,
    )


def design_matrix(time_days, period_days, first_step, late_step, additional_step=None):
    angular_frequency = 2.0 * np.pi / period_days
    columns = [
        np.ones_like(time_days),
        np.sin(angular_frequency * time_days),
        np.cos(angular_frequency * time_days),
        first_step,
    ]
    if additional_step is not None:
        columns.append(additional_step)
    columns.append(late_step)
    return np.column_stack(columns)


def fit_for_period(
    time_days,
    values,
    first_step,
    late_step,
    period_days,
    additional_step=None,
):
    design = design_matrix(
        time_days,
        period_days,
        first_step,
        late_step,
        additional_step,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = values - fitted
    return float(np.sum(residual * residual)), coefficients, fitted, residual


def fit_sinusoid_plus_two_acm(days, values, first_step, late_step):
    time_days = np.asarray(days, dtype=float) - float(days[0])
    values = np.asarray(values, dtype=float)
    first_step = np.asarray(first_step, dtype=float)
    late_step = np.asarray(late_step, dtype=float)
    span = float(np.ptp(time_days))
    if span <= 0.0:
        raise ValueError("LED data must span a positive time interval")

    minimum_period = max(2.0 * float(np.median(np.diff(time_days))), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)
    objective_values = np.asarray(
        [
            fit_for_period(
                time_days, values, first_step, late_step, period
            )[0]
            for period in period_grid
        ]
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
                first_step,
                late_step,
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
        time_days, values, first_step, late_step, period_days
    )
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    return {
        "period_days": period_days,
        "offset_log_change": float(coefficients[0]),
        "amplitude_log_change": float(np.hypot(coefficients[1], coefficients[2])),
        "phase_rad": float(np.arctan2(coefficients[2], coefficients[1])),
        "acm_step_log_change": float(coefficients[3]),
        "late_acm_step_log_change": float(coefficients[4]),
        "sse_log_change_squared": sse,
        "rmse_log_change": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares)
        if total_sum_squares > 0.0
        else np.nan,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
    }


def fit_sinusoid_plus_three_acm(
    days,
    values,
    first_step,
    additional_step,
    late_step,
):
    """Fit a sinusoid with the original, intermediate, and final ACM steps."""
    time_days = np.asarray(days, dtype=float) - float(days[0])
    values = np.asarray(values, dtype=float)
    first_step = np.asarray(first_step, dtype=float)
    additional_step = np.asarray(additional_step, dtype=float)
    late_step = np.asarray(late_step, dtype=float)
    span = float(np.ptp(time_days))
    if span <= 0.0:
        raise ValueError("LED data must span a positive time interval")

    minimum_period = max(2.0 * float(np.median(np.diff(time_days))), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)

    def objective(log_period):
        period = float(np.exp(log_period))
        return fit_for_period(
            time_days,
            values,
            first_step,
            late_step,
            period,
            additional_step,
        )[0]

    objective_values = np.asarray(
        [objective(np.log(period)) for period in period_grid]
    )
    best_index = int(np.argmin(objective_values))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(period_grid) - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        result = minimize_scalar(
            objective,
            bounds=(
                np.log(period_grid[lower_index]),
                np.log(period_grid[upper_index]),
            ),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))

    sse, coefficients, fitted, residual = fit_for_period(
        time_days,
        values,
        first_step,
        late_step,
        period_days,
        additional_step,
    )
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    return {
        "period_days": period_days,
        "offset_log_change": float(coefficients[0]),
        "amplitude_log_change": float(np.hypot(coefficients[1], coefficients[2])),
        "phase_rad": float(np.arctan2(coefficients[2], coefficients[1])),
        "acm_step_log_change": float(coefficients[3]),
        "additional_acm_step_log_change": float(coefficients[4]),
        "late_acm_step_log_change": float(coefficients[5]),
        "sse_log_change_squared": sse,
        "rmse_log_change": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares)
        if total_sum_squares > 0.0
        else np.nan,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
    }


def evaluate_components(
    time_days,
    fit,
    first_step,
    late_step,
    additional_step=None,
):
    time_days = np.asarray(time_days, dtype=float)
    angular_frequency = 2.0 * np.pi / fit["period_days"]
    sinusoid = (
        fit["offset_log_change"]
        + fit["coefficients"][1] * np.sin(angular_frequency * time_days)
        + fit["coefficients"][2] * np.cos(angular_frequency * time_days)
    )
    full = (
        sinusoid
        + fit["acm_step_log_change"] * np.asarray(first_step, dtype=float)
    )
    if additional_step is not None:
        full = full + fit["additional_acm_step_log_change"] * np.asarray(
            additional_step, dtype=float
        )
    full = full + fit["late_acm_step_log_change"] * np.asarray(
        late_step, dtype=float
    )
    return sinusoid, full


def write_summary(
    path,
    led,
    tile_id,
    minimum_day,
    days,
    first_event_day,
    late_event_day,
    fit,
):
    fields = (
        "led",
        "tile_id",
        "minimum_day_exclusive",
        "n_points",
        "first_day",
        "last_day",
        "period_days",
        "offset_log_change",
        "amplitude_log_change",
        "phase_rad",
        "acm_step_log_change",
        "late_acm_step_log_change",
        "first_acm_day",
        "late_acm_day",
        "rmse_log_change",
        "r2",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "led": led,
                "tile_id": tile_id,
                "minimum_day_exclusive": minimum_day,
                "n_points": days.size,
                "first_day": days[0],
                "last_day": days[-1],
                "period_days": fit["period_days"],
                "offset_log_change": fit["offset_log_change"],
                "amplitude_log_change": fit["amplitude_log_change"],
                "phase_rad": fit["phase_rad"],
                "acm_step_log_change": fit["acm_step_log_change"],
                "late_acm_step_log_change": fit["late_acm_step_log_change"],
                "first_acm_day": first_event_day,
                "late_acm_day": late_event_day,
                "rmse_log_change": fit["rmse_log_change"],
                "r2": fit["r2"],
            }
        )


def make_plot(
    path,
    led,
    tile_id,
    minimum_day,
    days,
    values,
    fit_indices,
    first_event_day,
    late_event_day,
    fit,
):
    fit_days = days[fit_indices]
    fit_values = values[fit_indices]
    grid = np.linspace(days[0], days[-1], 800)
    grid_relative_days = grid - fit_days[0]
    grid_first_step = (
        (grid >= first_event_day).astype(float)
        if first_event_day is not None
        else np.zeros_like(grid)
    )
    grid_late_step = (
        (grid >= late_event_day).astype(float)
        if late_event_day is not None
        else np.zeros_like(grid)
    )
    sinusoid, full = evaluate_components(
        grid_relative_days, fit, grid_first_step, grid_late_step
    )
    residual = fit_values - fit["fitted"]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    axis = axes[0]
    excluded = days <= minimum_day
    axis.scatter(
        days[excluded],
        1000.0 * values[excluded],
        facecolors="none",
        edgecolors="0.45",
        s=38,
        label="pre-day-1 data (excluded from fit)",
    )
    axis.scatter(
        fit_days,
        1000.0 * fit_values,
        color="tab:blue",
        s=32,
        label="post-day-1 data used in fit",
    )
    extrapolated = grid <= minimum_day
    axis.plot(
        grid[~extrapolated],
        1000.0 * full[~extrapolated],
        color="tab:red",
        linewidth=2.0,
        label=(
            f"sinusoid + 2 ACM: P={fit['period_days']:.4g} d, "
            f"A={1000.0 * fit['amplitude_log_change']:.3f} mlog, "
            f"R2={fit['r2']:.3f}"
        ),
    )
    axis.plot(
        grid[extrapolated],
        1000.0 * full[extrapolated],
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="back-extrapolated fit",
    )
    axis.plot(
        grid,
        1000.0 * sinusoid,
        color="tab:orange",
        linestyle="--",
        linewidth=1.4,
        label="sinusoid component",
    )
    if first_event_day is not None:
        axis.axvline(
            first_event_day,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="first ACM step",
        )
    if late_event_day is not None:
        axis.axvline(
            late_event_day,
            color="tab:green",
            linestyle=":",
            linewidth=1.0,
            label="late ACM step",
        )
    axis.axvline(
        minimum_day,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        label="fit boundary",
    )
    axis.set_ylabel("Mean log ratio (millilog)")
    axis.set_title(
        f"WFI01 tile {tile_id}, LED {led}: post-day-{minimum_day:g} fit and back-extrapolation"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[1]
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.scatter(fit_days, 1000.0 * residual, color="tab:purple", s=28)
    if first_event_day is not None:
        axis.axvline(first_event_day, color="black", linestyle=":", linewidth=1.0)
    if late_event_day is not None:
        axis.axvline(late_event_day, color="tab:green", linestyle=":", linewidth=1.0)
    axis.axvline(minimum_day, color="0.35", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Residual (millilog)")
    axis.set_title(f"Fit residual, RMSE={1000.0 * fit['rmse_log_change']:.3f} millilog")
    axis.grid(alpha=0.25)

    figure.suptitle(
        "Raw ratios fitted jointly with temperature-like sinusoid and two ACM steps"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("wfi_tile_log_statistics_512/wfi01_tile_log_statistics_512.csv"),
    )
    parser.add_argument("--led", type=int, default=1)
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--minimum-day", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wfi01_led1_postday1_sinusoid_plus_acm.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("wfi01_led1_postday1_sinusoid_plus_acm.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.led < 1 or args.led > 6:
        raise ValueError("LED must be between 1 and 6")
    if args.minimum_day < 0.0 or not np.isfinite(args.minimum_day):
        raise ValueError("Minimum day must be finite and non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    (
        days,
        values,
        first_step_design,
        late_step_design,
        fit_indices,
        first_event_day,
        late_event_day,
    ) = load_led_data(args.statistics, args.led, args.tile_id, args.minimum_day)
    fit = fit_sinusoid_plus_two_acm(
        days[fit_indices],
        values[fit_indices],
        first_step_design[fit_indices],
        late_step_design[fit_indices],
    )
    make_plot(
        args.output,
        args.led,
        args.tile_id,
        args.minimum_day,
        days,
        values,
        fit_indices,
        first_event_day,
        late_event_day,
        fit,
    )
    write_summary(
        args.summary,
        args.led,
        args.tile_id,
        args.minimum_day,
        days[fit_indices],
        first_event_day,
        late_event_day,
        fit,
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary}")
    print(
        f"LED {args.led}: n={fit_indices.size}, period={fit['period_days']:.8g} d, "
        f"amplitude={fit['amplitude_log_change']:.8g} log, "
        f"ACM steps={fit['acm_step_log_change']:.8g}, "
        f"{fit['late_acm_step_log_change']:.8g} log, "
        f"RMSE={fit['rmse_log_change']:.8g} log, R2={fit['r2']:.8g}"
    )


if __name__ == "__main__":
    main()
