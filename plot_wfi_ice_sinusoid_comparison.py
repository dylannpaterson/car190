"""Fit and compare pure sinusoidal models to two MAP ice trajectories."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


def load_trajectory(output_dir):
    with np.load(Path(output_dir) / "posterior_samples.npz", allow_pickle=False) as archive:
        days = np.asarray(archive["observation_days"], dtype=float)
        thickness = np.asarray(archive["ice_thickness"], dtype=float)
    if thickness.ndim == 1:
        median = thickness
    else:
        median = np.median(thickness, axis=0)
    return days, median


def sinusoid_design(time_days, period_days):
    angular_frequency = 2.0 * np.pi / period_days
    return np.column_stack(
        (
            np.ones_like(time_days),
            np.sin(angular_frequency * time_days),
            np.cos(angular_frequency * time_days),
        )
    )


def fit_for_period(time_days, thickness, period_days):
    design = sinusoid_design(time_days, period_days)
    coefficients, _, _, _ = np.linalg.lstsq(design, thickness, rcond=None)
    fitted = design @ coefficients
    residual = thickness - fitted
    return float(np.sum(residual * residual)), coefficients, fitted


def fit_sinusoid(days, thickness):
    time_days = np.asarray(days, dtype=float) - float(days[0])
    thickness = np.asarray(thickness, dtype=float)
    span = float(np.ptp(time_days))
    if span <= 0.0:
        raise ValueError("Trajectory must span a positive time interval")
    minimum_period = max(2.0 * float(np.median(np.diff(time_days))), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)
    objective_values = np.asarray(
        [fit_for_period(time_days, thickness, period)[0] for period in period_grid]
    )
    best_index = int(np.argmin(objective_values))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(period_grid) - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        result = minimize_scalar(
            lambda log_period: fit_for_period(
                time_days, thickness, float(np.exp(log_period))
            )[0],
            bounds=(
                np.log(period_grid[lower_index]),
                np.log(period_grid[upper_index]),
            ),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))
    sse, coefficients, fitted = fit_for_period(time_days, thickness, period_days)
    residual = thickness - fitted
    total_sum_squares = float(np.sum((thickness - np.mean(thickness)) ** 2))
    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    phase = float(np.arctan2(coefficients[2], coefficients[1]))
    return {
        "period_days": period_days,
        "offset_nm": float(coefficients[0]),
        "amplitude_nm": amplitude,
        "phase_rad": phase,
        "sse_nm2": sse,
        "rmse_nm": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares)
        if total_sum_squares > 0.0
        else np.nan,
        "fitted": fitted,
        "residual": residual,
    }


def plot_comparison(old_output, rrn_output, output_path, summary_path):
    old_days, old_thickness = load_trajectory(old_output)
    rrn_days, rrn_thickness = load_trajectory(rrn_output)
    old_fit = fit_sinusoid(old_days, old_thickness)
    rrn_fit = fit_sinusoid(rrn_days, rrn_thickness)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    figure.suptitle("WFI01 single-tile MAP ice trajectories: pure sinusoid fits")

    axis = axes[0]
    axis.plot(old_days, old_thickness, color="tab:blue", linewidth=2, label="Old data")
    axis.plot(
        old_days,
        old_fit["fitted"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label=(
            f"Old sinusoid (P={old_fit['period_days']:.3g} d, "
            f"R2={old_fit['r2']:.3f})"
        ),
    )
    axis.plot(rrn_days, rrn_thickness, color="tab:orange", linewidth=2, label="RRN data")
    axis.plot(
        rrn_days,
        rrn_fit["fitted"],
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label=(
            f"RRN sinusoid (P={rrn_fit['period_days']:.3g} d, "
            f"R2={rrn_fit['r2']:.3f})"
        ),
    )
    axis.axvline(
        old_days[-1],
        color="tab:blue",
        linestyle=":",
        linewidth=0.8,
        alpha=0.7,
        label="Old-data endpoint",
    )
    axis.set_ylabel("Ice thickness (nm)")
    axis.set_title("Observed fitted trajectories and pure sinusoidal models")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    axis = axes[1]
    axis.plot(old_days, old_fit["residual"], color="tab:blue", linewidth=1.8, label="Old residual")
    axis.plot(rrn_days, rrn_fit["residual"], color="tab:orange", linewidth=1.8, label="RRN residual")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Observed - sinusoid (nm)")
    axis.set_title("Residuals from pure sinusoidal fits")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    with Path(summary_path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "period_days",
                "offset_nm",
                "amplitude_nm",
                "phase_rad",
                "sse_nm2",
                "rmse_nm",
                "r2",
            ),
        )
        writer.writeheader()
        for dataset, result in (("old", old_fit), ("rrn", rrn_fit)):
            writer.writerow(
                {
                    "dataset": dataset,
                    **{key: result[key] for key in writer.fieldnames if key != "dataset"},
                }
            )
    return old_fit, rrn_fit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--rrn-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    old_fit, rrn_fit = plot_comparison(
        args.old_output,
        args.rrn_output,
        args.output,
        args.summary,
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary}")
    print(
        f"old period={old_fit['period_days']:.8g} d, "
        f"RMSE={old_fit['rmse_nm']:.8g} nm"
    )
    print(
        f"rrn period={rrn_fit['period_days']:.8g} d, "
        f"RMSE={rrn_fit['rmse_nm']:.8g} nm"
    )


if __name__ == "__main__":
    main()
