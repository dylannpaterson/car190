"""Plot combined WFI ice-growth trajectories from the production MAP fits."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_wfi_all_detector_trajectories import SCAS, load_trajectory, write_quantiles


ROOT = Path(__file__).resolve().parent
DEFAULT_FIT_PREFIX = "wfi_production_20261005_shared_hgcdte_prior10_smooth001"
DEFAULT_FIT_SUFFIX = "map10000_emcee1000"
DEFAULT_OUTPUT = (
    ROOT / "wfi_production_20261005_shared_hgcdte_prior10_smooth001_ice_growth.png"
)
DEFAULT_RESET_DAY = 15.064405023148149


def plot_ice_growth(path, trajectories, reset_day, start_date, updated_date, x_max, y_max):
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    colors = plt.get_cmap("tab20", len(SCAS))
    legend_handles = []
    maximum_thickness = 0.0

    for index, detector in enumerate(SCAS):
        days, ice, _ = trajectories[detector]
        thickness_microns = ice[1] / 1000.0
        maximum_thickness = max(maximum_thickness, float(np.max(thickness_microns)))
        color = colors(index)
        axis.plot(days, thickness_microns, color=color, linewidth=1.1)
        legend_handles.append(
            Line2D([], [], color=color, linewidth=1.5, label=detector)
        )

    axis.axvline(
        reset_day,
        color="0.35",
        linestyle=":",
        linewidth=0.9,
        alpha=0.8,
    )
    axis.set_xlim(0.0, x_max)
    axis.set_ylim(-0.01, y_max)
    axis.set_xlabel("time [days]", fontsize=15)
    axis.set_ylabel("thickness [microns]", fontsize=15)
    axis.set_title(f"Ice growth starting {start_date}", fontsize=17, pad=16)
    axis.text(
        0.25,
        0.88,
        f"updated {updated_date}",
        transform=axis.transAxes,
        fontsize=15,
    )
    axis.grid(True, linestyle=":", linewidth=0.8, color="0.35", alpha=0.75)
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=12,
        handlelength=3.5,
        handletextpad=0.6,
        labelspacing=0.25,
    )
    axis.text(
        0.99,
        0.02,
        f"reset day = {reset_day:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return maximum_thickness


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fit-prefix", default=DEFAULT_FIT_PREFIX)
    parser.add_argument("--fit-suffix", default=DEFAULT_FIT_SUFFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2026/09/12")
    parser.add_argument("--updated-date", default="2026/10/05")
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    parser.add_argument("--x-max", type=float, default=21.0)
    parser.add_argument("--y-max", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    trajectories = {}
    for detector in SCAS:
        output_dir = args.root / f"{args.fit_prefix}_{detector}_{args.fit_suffix}"
        posterior_path = output_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            raise FileNotFoundError(f"Missing production posterior: {posterior_path}")
        trajectories[detector] = load_trajectory(output_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    maximum_thickness = plot_ice_growth(
        args.output,
        trajectories,
        args.reset_day,
        args.start_date,
        args.updated_date,
        args.x_max,
        args.y_max,
    )
    quantile_path = args.output.with_name(args.output.stem + "_quantiles.csv")
    write_quantiles(quantile_path, trajectories)
    print(f"Wrote {args.output}")
    print(f"Wrote {quantile_path}")
    print(f"Maximum median tile-mean thickness: {maximum_thickness:.6g} microns")


if __name__ == "__main__":
    main()
