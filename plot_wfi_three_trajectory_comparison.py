"""Plot full-TMM, non-spline, and spline WFI ice trajectories."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = (
    ("full_tmm", "Full TMM", "#0072B2"),
    ("non_spline", "Relative response, non-spline", "#D55E00"),
    ("spline", "Relative response, 48 h spline", "#009E73"),
)
RESET_COLOR = "#555555"


def load_trajectory(root, tile_id):
    path = Path(root) / f"tile_{tile_id:03d}" / "posterior_samples.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
        thickness = np.asarray(archive["tile_ice_thickness"], dtype=float).reshape(-1)
        third_log_nm = np.asarray(
            archive["third_decon_initial_thickness_log_nm"], dtype=float
        ).reshape(-1)
        reset_days = np.asarray(archive["ice_reset_days"], dtype=float).reshape(-1)
    if days.size != thickness.size:
        raise ValueError(f"Trajectory lengths differ in {path}")
    return {
        "days": days,
        "thickness": thickness,
        "third_decon_initial_thickness_nm": float(np.exp(third_log_nm.item())),
        "reset_days": reset_days,
    }


def plot_trajectories(full_tmm_root, non_spline_root, spline_root, tile_ids, output):
    roots = {
        "full_tmm": full_tmm_root,
        "non_spline": non_spline_root,
        "spline": spline_root,
    }
    tile_ids = tuple(tile_ids)
    figure, axes = plt.subplots(
        len(tile_ids),
        1,
        figsize=(12.0, max(4.5, 4.0 * len(tile_ids))),
        sharex=False,
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, tile_id in enumerate(tile_ids):
        axis = axes[row_index, 0]
        trajectories = {
            name: load_trajectory(root, tile_id) for name, root in roots.items()
        }
        reset_days = trajectories["full_tmm"]["reset_days"]
        for name, label, color in MODEL_LABELS:
            trajectory = trajectories[name]
            axis.plot(
                trajectory["days"],
                trajectory["thickness"],
                color=color,
                linewidth=1.35,
                label=label,
            )
        for reset_day in reset_days:
            axis.axvline(
                reset_day,
                color=RESET_COLOR,
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )
        third_text = "\n".join(
            f"{label}: {trajectories[name]['third_decon_initial_thickness_nm']:.2f} nm"
            for name, label, _ in MODEL_LABELS
        )
        axis.text(
            0.99,
            0.04,
            "Third-Decon initial thickness\n" + third_text,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        axis.set_title(f"WFI18 tile {tile_id:03d}", loc="left", fontsize=11)
        axis.set_ylabel("Ice thickness (nm)")
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("Days since first observation")
    figure.suptitle(
        "WFI18 ice-thickness trajectory comparison\n"
        "Full TMM vs non-spline relative response vs 48 h spline; 20261109 data",
        fontsize=13,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-tmm-root", type=Path, required=True)
    parser.add_argument("--non-spline-root", type=Path, required=True)
    parser.add_argument("--spline-root", type=Path, required=True)
    parser.add_argument("--tile-id", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = plot_trajectories(
        args.full_tmm_root,
        args.non_spline_root,
        args.spline_root,
        args.tile_id,
        args.output,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
