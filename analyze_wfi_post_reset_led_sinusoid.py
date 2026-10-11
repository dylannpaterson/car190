"""Test LED-specific post-reset sinusoids against held-out Program 1063 data."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import linregress


ROOT = Path(__file__).resolve().parent
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
LEDS = tuple(range(1, 7))
DEFAULT_RESET_DAY = 12.907397928240743
DEFAULT_PROGRAM = 1063


def read_rows(path):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            count = float(row["n_pair_valid"])
            if count <= 0.0:
                continue
            rows.append(
                {
                    "detector": Path(path).name[:5].upper(),
                    "tile": int(row["tile_id"]),
                    "led": int(row["led"]),
                    "program": int(row["program"]),
                    "repeat": int(row["repeat"]),
                    "days": float(row["days"]),
                    "value": float(row["sum_log_change"]) / count,
                }
            )
    return rows


def select_rows(rows, reset_day, program=None, include_program=True):
    selected = []
    for row in rows:
        if row["days"] < reset_day:
            continue
        if program is not None and ((row["program"] == program) != include_program):
            continue
        selected.append(row)
    return selected


def pool_rows(rows, led):
    grouped = defaultdict(list)
    for row in rows:
        if row["led"] == led:
            grouped[row["repeat"]].append(row)
    pooled = []
    for repeat, repeat_rows in sorted(grouped.items()):
        values = [row["value"] for row in repeat_rows]
        pooled.append(
            {
                "led": led,
                "repeat": repeat,
                "days": float(np.mean([row["days"] for row in repeat_rows])),
                "value": float(np.mean(values)),
                "detector_count": len(values),
            }
        )
    return pooled


def design_matrix(time_days, period_days, harmonic_count=1):
    angular_frequency = 2.0 * np.pi / period_days
    columns = [np.ones_like(time_days)]
    for harmonic in range(1, harmonic_count + 1):
        angle = harmonic * angular_frequency * time_days
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def fit_at_period(time_days, values, period_days, harmonic_count=1):
    design = design_matrix(time_days, period_days, harmonic_count)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = values - fitted
    return float(np.sum(residual * residual)), coefficients, fitted, residual


def fit_sinusoid(time_days, values, harmonic_count=1):
    time_days = np.asarray(time_days, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(time_days, kind="stable")
    time_days = time_days[order]
    values = values[order]
    span = float(np.ptp(time_days))
    if span <= 0.0 or time_days.size < 6:
        raise ValueError("A sinusoid requires at least six distinct time samples")
    minimum_period = max(2.0 * float(np.median(np.diff(time_days))), 0.05)
    maximum_period = max(100.0 * span, 2.0 * minimum_period)
    period_grid = np.geomspace(minimum_period, maximum_period, 600)
    objectives = np.asarray(
        [
            fit_at_period(time_days, values, period, harmonic_count)[0]
            for period in period_grid
        ]
    )
    best_index = int(np.argmin(objectives))
    lower_index = max(0, best_index - 1)
    upper_index = min(period_grid.size - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        result = minimize_scalar(
            lambda log_period: fit_at_period(
                time_days,
                values,
                float(np.exp(log_period)),
                harmonic_count,
            )[0],
            bounds=(
                np.log(period_grid[lower_index]),
                np.log(period_grid[upper_index]),
            ),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))
    sse, coefficients, fitted, residual = fit_at_period(
        time_days, values, period_days, harmonic_count
    )
    n = time_days.size
    parameter_count = 1 + 2 * harmonic_count
    total_ss = float(np.sum((values - np.mean(values)) ** 2))
    return {
        "period_days": period_days,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
        "sse": sse,
        "rmse": float(np.sqrt(sse / n)),
        "r2": float(1.0 - sse / total_ss) if total_ss > 0.0 else np.nan,
        "bic": float(n * np.log(max(sse / n, np.finfo(float).tiny)) + parameter_count * np.log(n)),
        "n": n,
    }


def evaluate_model(days, fit, origin_day):
    relative_days = np.asarray(days, dtype=float) - origin_day
    return design_matrix(relative_days, fit["period_days"]) @ fit["coefficients"]


def latest_block_mask(days):
    days = np.asarray(days, dtype=float)
    sorted_days = np.sort(days)
    if sorted_days.size < 3:
        raise ValueError("A latest block requires at least three samples")
    split = int(np.argmax(np.diff(sorted_days))) + 1
    latest_start = float(sorted_days[split])
    mask = days >= latest_start
    if np.count_nonzero(mask) < 3:
        latest_start = float(sorted_days[-3])
        mask = days >= latest_start
    return mask, latest_start


def summarize_series(train_rows, test_rows, reset_day, detector_label, tile_id, led):
    train = [row for row in train_rows if row["led"] == led]
    test = [row for row in test_rows if row["led"] == led]
    if len(train) < 6 or len(test) < 3:
        raise ValueError(f"Insufficient samples for {detector_label} LED {led}")
    train_days = np.asarray([row["days"] for row in train], dtype=float)
    train_values = np.asarray([row["value"] for row in train], dtype=float)
    test_days = np.asarray([row["days"] for row in test], dtype=float)
    test_values = np.asarray([row["value"] for row in test], dtype=float)
    origin_day = float(reset_day)
    train_fit = fit_sinusoid(train_days - origin_day, train_values)
    train_pred_test = evaluate_model(test_days, train_fit, origin_day)
    test_residual = test_values - train_pred_test
    test_ss = float(np.sum((test_values - np.mean(test_values)) ** 2))
    late_mask, late_start = latest_block_mask(test_days)
    late_slope = linregress(test_days[late_mask], test_residual[late_mask])
    return {
        "detector": detector_label,
        "tile": tile_id,
        "led": led,
        "train_points": len(train),
        "test_points": len(test),
        "train_first_day": float(np.min(train_days)),
        "train_last_day": float(np.max(train_days)),
        "test_first_day": float(np.min(test_days)),
        "test_last_day": float(np.max(test_days)),
        "late_start_day": late_start,
        "train_period_days": train_fit["period_days"],
        "train_amplitude_mlog": float(1000.0 * np.hypot(train_fit["coefficients"][1], train_fit["coefficients"][2])),
        "train_rmse_mlog": 1000.0 * train_fit["rmse"],
        "train_r2": train_fit["r2"],
        "test_rmse_mlog": float(1000.0 * np.sqrt(np.mean(test_residual**2))),
        "test_bias_mlog": float(1000.0 * np.mean(test_residual)),
        "test_r2_vs_mean": float(1.0 - np.sum(test_residual**2) / test_ss) if test_ss > 0.0 else np.nan,
        "test_late_bias_mlog": float(1000.0 * np.mean(test_residual[late_mask])),
        "test_late_rmse_mlog": float(1000.0 * np.sqrt(np.mean(test_residual[late_mask] ** 2))),
        "test_late_slope_mlog_per_day": float(1000.0 * late_slope.slope),
        "test_late_slope_pvalue": float(late_slope.pvalue),
        "test_late_points": int(np.count_nonzero(late_mask)),
        "train_fit": train_fit,
        "test_prediction": train_pred_test,
        "test_days": test_days,
        "test_values": test_values,
    }


def write_summary(path, summaries):
    fields = [
        key
        for key, value in summaries[0].items()
        if np.isscalar(value) and key not in {"test_prediction", "test_days", "test_values"}
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})


def plot_pooled(output_path, train_rows, test_rows, reset_day):
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for axis, led in zip(axes.flat, LEDS):
        train = pool_rows(train_rows, led)
        test = pool_rows(test_rows, led)
        train_days = np.asarray([row["days"] for row in train], dtype=float)
        train_values = np.asarray([row["value"] for row in train], dtype=float)
        test_days = np.asarray([row["days"] for row in test], dtype=float)
        test_values = np.asarray([row["value"] for row in test], dtype=float)
        fit = fit_sinusoid(train_days - reset_day, train_values)
        grid = np.linspace(reset_day, max(test_days), 600)
        prediction = evaluate_model(grid, fit, reset_day)
        axis.scatter(train_days, 1000.0 * train_values, s=12, color="#0072B2", alpha=0.65, label="post-reset non-1063 train")
        axis.scatter(test_days, 1000.0 * test_values, s=16, color="#D55E00", alpha=0.7, label="Program 1063 holdout")
        axis.plot(grid, 1000.0 * prediction, color="black", linewidth=1.6, label=f"sine P={fit['period_days']:.2f} d")
        axis.axvline(reset_day, color="0.35", linestyle=":", linewidth=0.8)
        axis.axhline(0.0, color="0.35", linewidth=0.7)
        axis.set_title(f"LED {led}; train RMSE={1000.0 * fit['rmse']:.3f} mlog")
        axis.set_ylabel("Mean log change (millilog)")
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Days since first exposure")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Post-reset LED-specific sinusoid fit; Program 1063 held out")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_residuals(output_path, train_rows, test_rows, reset_day):
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, led in zip(axes.flat, LEDS):
        train = pool_rows(train_rows, led)
        test = pool_rows(test_rows, led)
        train_days = np.asarray([row["days"] for row in train], dtype=float)
        train_values = np.asarray([row["value"] for row in train], dtype=float)
        test_days = np.asarray([row["days"] for row in test], dtype=float)
        test_values = np.asarray([row["value"] for row in test], dtype=float)
        fit = fit_sinusoid(train_days - reset_day, train_values)
        residual = 1000.0 * (test_values - evaluate_model(test_days, fit, reset_day))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.scatter(test_days, residual, color="#D55E00", s=16)
        _, latest_start = latest_block_mask(test_days)
        axis.axvline(latest_start, color="#7F3C8D", linestyle="--", linewidth=0.9)
        axis.set_title(f"LED {led}; holdout RMS={np.sqrt(np.mean(residual**2)):.3f} mlog")
        axis.set_ylabel("Program 1063 residual (millilog)")
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Days since first exposure")
    figure.suptitle("Program 1063 departure from the post-reset non-1063 LED sinusoids")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_tile_diagnostics(output_dir, train_by_tile, test_by_tile, summaries, reset_day):
    summary_by_key = {
        (summary["detector"], summary["tile"], summary["led"]): summary
        for summary in summaries
    }
    output_dir = Path(output_dir)
    plot_count = 0
    for (detector, tile), train_rows in sorted(train_by_tile.items()):
        test_rows = test_by_tile[(detector, tile)]
        detector_dir = output_dir / detector.lower()
        detector_dir.mkdir(parents=True, exist_ok=True)
        plot_path = detector_dir / f"tile_{tile:03d}.png"
        if plot_path.exists():
            plot_count += 1
            continue
        figure, axes = plt.subplots(6, 2, figsize=(10, 14), sharex="col")
        for row_index, led in enumerate(LEDS):
            summary = summary_by_key[(detector, tile, led)]
            train = [row for row in train_rows if row["led"] == led]
            test = [row for row in test_rows if row["led"] == led]
            train_days = np.asarray([row["days"] for row in train], dtype=float)
            train_values = np.asarray([row["value"] for row in train], dtype=float)
            test_days = np.asarray([row["days"] for row in test], dtype=float)
            test_values = np.asarray([row["value"] for row in test], dtype=float)
            grid = np.linspace(reset_day, max(test_days), 500)
            prediction = evaluate_model(grid, summary["train_fit"], reset_day)
            residual = 1000.0 * (
                test_values - np.asarray(summary["test_prediction"], dtype=float)
            )
            fit_axis, residual_axis = axes[row_index]
            fit_axis.scatter(
                train_days,
                1000.0 * train_values,
                s=9,
                color="#0072B2",
                alpha=0.7,
                label="non-1063 train" if row_index == 0 else None,
            )
            fit_axis.scatter(
                test_days,
                1000.0 * test_values,
                s=13,
                color="#D55E00",
                alpha=0.8,
                label="1063 holdout" if row_index == 0 else None,
            )
            fit_axis.plot(
                grid,
                1000.0 * prediction,
                color="black",
                linewidth=1.3,
                label="fitted sine" if row_index == 0 else None,
            )
            fit_axis.axvline(reset_day, color="0.35", linestyle=":", linewidth=0.7)
            fit_axis.axvline(
                summary["late_start_day"],
                color="#7F3C8D",
                linestyle="--",
                linewidth=0.8,
            )
            fit_axis.axhline(0.0, color="0.35", linewidth=0.6)
            fit_axis.set_title(
                f"LED {led}; P={summary['train_period_days']:.2f} d; "
                f"latest bias={summary['test_late_bias_mlog']:+.2f} mlog"
            )
            fit_axis.set_ylabel("log change\n(millilog)")
            fit_axis.grid(alpha=0.2)
            residual_axis.axhline(0.0, color="black", linewidth=0.7)
            residual_axis.scatter(test_days, residual, color="#D55E00", s=13)
            residual_axis.axvline(
                summary["late_start_day"],
                color="#7F3C8D",
                linestyle="--",
                linewidth=0.8,
            )
            residual_axis.set_title(
                f"1063 residual; RMS={summary['test_rmse_mlog']:.2f} mlog"
            )
            residual_axis.set_ylabel("residual\n(millilog)")
            residual_axis.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("Days since first exposure")
        axes[-1, 1].set_xlabel("Days since first exposure")
        axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
        figure.suptitle(f"{detector} tile {tile:03d}: independent LED sinusoids")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        figure.savefig(plot_path, dpi=100)
        plt.close(figure)
        plot_count += 1
    return plot_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    parser.add_argument("--holdout-program", type=int, default=DEFAULT_PROGRAM)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for detector in DETECTORS:
        path = args.statistics_dir / f"{detector.lower()}_tile_log_statistics_512_count_corrected.csv"
        all_rows.extend(read_rows(path))
    train_rows = select_rows(
        all_rows,
        args.reset_day,
        program=args.holdout_program,
        include_program=False,
    )
    test_rows = select_rows(
        all_rows,
        args.reset_day,
        program=args.holdout_program,
        include_program=True,
    )
    train_by_tile = defaultdict(list)
    test_by_tile = defaultdict(list)
    for row in train_rows:
        train_by_tile[(row["detector"], row["tile"])].append(row)
    for row in test_rows:
        test_by_tile[(row["detector"], row["tile"])].append(row)
    summaries = []
    for detector, tile in sorted(train_by_tile):
        for led in LEDS:
            summaries.append(
                summarize_series(
                    train_by_tile[(detector, tile)],
                    test_by_tile[(detector, tile)],
                    args.reset_day,
                    detector,
                    tile,
                    led,
                )
            )
    write_summary(args.output_dir / "post_reset_led_sinusoid_tile_summary.csv", summaries)
    plot_count = plot_tile_diagnostics(
        args.output_dir / "post_reset_led_sinusoid_tile_plots",
        train_by_tile,
        test_by_tile,
        summaries,
        args.reset_day,
    )
    print(f"Training rows after reset, excluding Program {args.holdout_program}: {len(train_rows)}")
    print(f"Held-out Program {args.holdout_program} rows after reset: {len(test_rows)}")
    print(f"Independent tile/LED fits: {len(summaries)}")
    print(f"Tile diagnostic plots: {plot_count}")


if __name__ == "__main__":
    main()
