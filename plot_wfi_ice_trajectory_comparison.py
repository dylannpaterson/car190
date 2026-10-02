"""Compare MAP ice trajectories from two NumPyro output directories."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(output_dir):
    with np.load(Path(output_dir) / "posterior_samples.npz", allow_pickle=False) as archive:
        days = np.asarray(archive["observation_days"], dtype=float)
        thickness = np.asarray(archive["ice_thickness"], dtype=float)
        mean_rate = np.asarray(archive["mean_rate_nm_day"], dtype=float)
    if thickness.ndim == 1:
        median = thickness
    else:
        median = np.median(thickness, axis=0)
    return days, median, float(np.median(mean_rate))


def _load_free_trajectory_pair(old_free_output, rrn_free_output):
    if (old_free_output is None) != (rrn_free_output is None):
        raise ValueError("Both free-d0 output directories are required together")
    if old_free_output is None:
        return None
    return load_trajectory(old_free_output), load_trajectory(rrn_free_output)


def plot_comparison(
    old_output,
    rrn_output,
    output_path,
    old_free_output=None,
    rrn_free_output=None,
):
    old_days, old_thickness, _ = load_trajectory(old_output)
    rrn_days, rrn_thickness, _ = load_trajectory(rrn_output)
    rrn_at_old_days = np.interp(old_days, rrn_days, rrn_thickness)
    difference = rrn_at_old_days - old_thickness
    free_pair = _load_free_trajectory_pair(old_free_output, rrn_free_output)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    title = "WFI01 single-tile MAP ice trajectory: old versus RRN"
    if free_pair is not None:
        title += "; fixed and free initial thickness"
    figure.suptitle(title)

    axis = axes[0]
    axis.plot(
        old_days,
        old_thickness,
        color="tab:blue",
        linewidth=2,
        label="Old data, d0=0",
    )
    axis.plot(
        rrn_days,
        rrn_thickness,
        color="tab:orange",
        linewidth=2,
        label="RRN data, d0=0",
    )
    if free_pair is not None:
        (old_free_days, old_free_thickness, _), (
            rrn_free_days,
            rrn_free_thickness,
            _,
        ) = free_pair
        axis.plot(
            old_free_days,
            old_free_thickness,
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
            label="Old data, free d0",
        )
        axis.plot(
            rrn_free_days,
            rrn_free_thickness,
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label="RRN data, free d0",
        )
    axis.axvline(
        old_days[-1],
        color="tab:blue",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="Old-data endpoint",
    )
    axis.set_ylabel("Ice thickness (nm)")
    axis.set_title("Fitted shared ice thickness")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    axis = axes[1]
    axis.plot(old_days, difference, color="black", linewidth=1.8, label="d0=0")
    if free_pair is not None:
        free_old_days, free_old_thickness, _ = free_pair[0]
        free_rrn_days, free_rrn_thickness, _ = free_pair[1]
        free_rrn_at_old_days = np.interp(
            free_old_days, free_rrn_days, free_rrn_thickness
        )
        free_difference = free_rrn_at_old_days - free_old_thickness
        axis.plot(
            free_old_days,
            free_difference,
            color="dimgray",
            linestyle="--",
            linewidth=1.8,
            label="free d0",
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("RRN - old (nm)")
    axis.set_title("Trajectory difference over the common time range")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_trend_residual_comparison(
    old_output,
    rrn_output,
    output_path,
    old_free_output=None,
    rrn_free_output=None,
):
    old_days, old_thickness, old_mean_rate = load_trajectory(old_output)
    rrn_days, rrn_thickness, rrn_mean_rate = load_trajectory(rrn_output)
    old_trend = old_thickness[0] + old_mean_rate * (old_days - old_days[0])
    rrn_trend = rrn_thickness[0] + rrn_mean_rate * (rrn_days - rrn_days[0])
    old_residual = old_thickness - old_trend
    rrn_residual = rrn_thickness - rrn_trend
    rrn_residual_at_old_days = np.interp(old_days, rrn_days, rrn_residual)
    residual_difference = rrn_residual_at_old_days - old_residual
    free_pair = _load_free_trajectory_pair(old_free_output, rrn_free_output)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    title = "WFI01 single-tile MAP ice trend minus mean-rate trend"
    if free_pair is not None:
        title += "; fixed and free initial thickness"
    figure.suptitle(title)

    axis = axes[0]
    axis.plot(
        old_days,
        old_residual,
        color="tab:blue",
        linewidth=2,
        label=f"Old data, d0=0 ({old_mean_rate:.3f} nm/day)",
    )
    axis.plot(
        rrn_days,
        rrn_residual,
        color="tab:orange",
        linewidth=2,
        label=f"RRN data, d0=0 ({rrn_mean_rate:.3f} nm/day)",
    )
    if free_pair is not None:
        (old_free_days, old_free_thickness, old_free_mean_rate), (
            rrn_free_days,
            rrn_free_thickness,
            rrn_free_mean_rate,
        ) = free_pair
        old_free_residual = old_free_thickness - (
            old_free_thickness[0]
            + old_free_mean_rate * (old_free_days - old_free_days[0])
        )
        rrn_free_residual = rrn_free_thickness - (
            rrn_free_thickness[0]
            + rrn_free_mean_rate * (rrn_free_days - rrn_free_days[0])
        )
        axis.plot(
            old_free_days,
            old_free_residual,
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
            label=f"Old data, free d0 ({old_free_mean_rate:.3f} nm/day)",
        )
        axis.plot(
            rrn_free_days,
            rrn_free_residual,
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label=f"RRN data, free d0 ({rrn_free_mean_rate:.3f} nm/day)",
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.axvline(
        old_days[-1],
        color="tab:blue",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="Old-data endpoint",
    )
    axis.set_ylabel("Thickness minus mean-rate trend (nm)")
    axis.set_title("Deviation from each fit's constant average-rate trend")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    axis = axes[1]
    axis.plot(
        old_days,
        residual_difference,
        color="black",
        linewidth=1.8,
        label="d0=0",
    )
    if free_pair is not None:
        free_old_days, free_old_thickness, free_old_mean_rate = free_pair[0]
        free_rrn_days, free_rrn_thickness, free_rrn_mean_rate = free_pair[1]
        free_old_residual = free_old_thickness - (
            free_old_thickness[0]
            + free_old_mean_rate * (free_old_days - free_old_days[0])
        )
        free_rrn_residual = free_rrn_thickness - (
            free_rrn_thickness[0]
            + free_rrn_mean_rate * (free_rrn_days - free_rrn_days[0])
        )
        free_residual_difference = np.interp(
            free_old_days, free_rrn_days, free_rrn_residual
        ) - free_old_residual
        axis.plot(
            free_old_days,
            free_residual_difference,
            color="dimgray",
            linestyle="--",
            linewidth=1.8,
            label="free d0",
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("RRN - old (nm)")
    axis.set_title("Detrended trajectory difference over the common time range")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--rrn-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--old-free-output",
        type=Path,
        help="Optional old-data fit with a free initial thickness.",
    )
    parser.add_argument(
        "--rrn-free-output",
        type=Path,
        help="Optional RRN fit with a free initial thickness.",
    )
    parser.add_argument(
        "--trend-residual-output",
        type=Path,
        help="Also write the comparison after subtracting each fit's mean-rate trend.",
    )
    args = parser.parse_args()
    if (args.old_free_output is None) != (args.rrn_free_output is None):
        parser.error("--old-free-output and --rrn-free-output must be supplied together")
    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(
        args.old_output,
        args.rrn_output,
        args.output,
        args.old_free_output,
        args.rrn_free_output,
    )
    print(f"wrote: {args.output}")
    if args.trend_residual_output is not None:
        args.trend_residual_output.parent.mkdir(parents=True, exist_ok=True)
        plot_trend_residual_comparison(
            args.old_output,
            args.rrn_output,
            args.trend_residual_output,
            args.old_free_output,
            args.rrn_free_output,
        )
        print(f"wrote: {args.trend_residual_output}")


if __name__ == "__main__":
    main()
