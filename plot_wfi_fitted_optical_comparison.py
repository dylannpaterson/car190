"""Plot fitted optical properties and ice trajectories for WFI detectors."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "wfi01_wfi03_fitted_optical_comparison.png"
DEFAULT_INPUTS = (
    ROOT / "wfi01_absolute_substrate_power_weak_priors_map10000_tile_corrections_x64_free_tile_trajectories",
    ROOT / "wfi02_absolute_substrate_power_weak_priors_map10000_tile_corrections_x64_free_tile_trajectories_prod",
    ROOT / "wfi03_absolute_substrate_power_weak_priors_map10000_tile_corrections_x64_free_tile_trajectories",
)
COLORS = {"WFI01": "#0072B2", "WFI02": "#D55E00", "WFI03": "#009E73"}


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_detector(path):
    path = Path(path)
    ar_rows = read_rows(path / "tile_ar_posterior.csv")
    hgcdte_rows = read_rows(path / "tile_hgcdte_posterior.csv")
    detector = ar_rows[0]["detector"]
    ar = {
        (layer, parameter): np.asarray(
            [float(row[f"ar_layer{layer}_{parameter}_median"]) for row in ar_rows]
        )
        for layer in (1, 2, 3)
        for parameter in ("thickness", "n")
    }
    hgcdte = {
        parameter: np.asarray(
            [float(row[f"{parameter}_log_scale_median"]) for row in hgcdte_rows]
        )
        for parameter in ("n", "k")
    }
    with np.load(path / "posterior_samples.npz") as samples:
        days = np.asarray(samples["observation_days"], dtype=float)
        ice_thickness = np.asarray(samples["ice_thickness"], dtype=float)
        if ice_thickness.ndim == 3:
            ice_thickness = np.median(ice_thickness[0], axis=0)
        else:
            ice_thickness = np.asarray(ice_thickness[0], dtype=float)
        angle = float(np.asarray(samples["incidence_angle_deg"]).reshape(-1)[0])
        mean_rate = float(np.asarray(samples["mean_rate_nm_day"]).reshape(-1)[0])
    return {
        "detector": detector,
        "ar": ar,
        "hgcdte": hgcdte,
        "days": days,
        "ice_thickness": ice_thickness,
        "angle": angle,
        "mean_rate": mean_rate,
    }


def scatter_detector_values(axis, detector_data, values, ylabel, title, layers=None):
    detectors = [item["detector"] for item in detector_data]
    x = np.arange(len(detectors), dtype=float)
    layered = layers is not None
    if layers is None:
        layers = [None]
    for layer_index, layer in enumerate(layers):
        color = plt.cm.viridis(layer_index / max(len(layers) - 1, 1))
        for detector_index, item in enumerate(detector_data):
            tile_values = values(item, layer)
            jitter = np.linspace(-0.18, 0.18, len(tile_values))
            axis.scatter(
                x[detector_index] + jitter,
                tile_values,
                color=color if layer is not None else COLORS[item["detector"]],
                alpha=0.35,
                s=12,
                linewidths=0,
            )
            axis.plot(
                [x[detector_index] - 0.2, x[detector_index] + 0.2],
                [np.median(tile_values)] * 2,
                color=color if layer is not None else COLORS[item["detector"]],
                linewidth=2.4,
            )
        if layer is not None:
            axis.plot([], [], color=color, linewidth=2.4, label=f"Layer {layer}")
    axis.set_xticks(x, detectors)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    if layered:
        axis.legend(frameon=False, fontsize=8)


def make_plot(output, detector_data):
    figure = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1, 1, 1.25))
    thickness_axis = figure.add_subplot(grid[0, 0])
    index_axis = figure.add_subplot(grid[0, 1])
    hgcdte_n_axis = figure.add_subplot(grid[1, 0])
    hgcdte_k_axis = figure.add_subplot(grid[1, 1])
    trajectory_axis = figure.add_subplot(grid[2, :])

    scatter_detector_values(
        thickness_axis,
        detector_data,
        lambda item, layer: 100.0 * item["ar"][(layer, "thickness")],
        "Fitted thickness (nm)",
        "AR physical thickness",
        layers=(1, 2, 3),
    )
    scatter_detector_values(
        index_axis,
        detector_data,
        lambda item, layer: item["ar"][(layer, "n")],
        "Fitted refractive index n",
        "AR refractive index",
        layers=(1, 2, 3),
    )
    scatter_detector_values(
        hgcdte_n_axis,
        detector_data,
        lambda item, layer: np.exp(item["hgcdte"]["n"]),
        "n scale multiplier",
        "HgCdTe real-index scale",
    )
    scatter_detector_values(
        hgcdte_k_axis,
        detector_data,
        lambda item, layer: np.exp(item["hgcdte"]["k"]),
        "k scale multiplier",
        "HgCdTe extinction-index scale",
    )

    for item in detector_data:
        color = COLORS[item["detector"]]
        trajectory_axis.plot(
            item["days"],
            item["ice_thickness"],
            color=color,
            linewidth=2.2,
            label=(
                f"{item['detector']}: {item['mean_rate']:.2f} nm/day, "
                f"angle {item['angle']:.3f} deg"
            ),
        )
    trajectory_axis.set_xlabel("Days since first exposure")
    trajectory_axis.set_ylabel("Fitted ice thickness (nm; tile median)")
    trajectory_axis.set_title(
        "Fitted ice trajectories: the temporal phase driver in the TMM model"
    )
    trajectory_axis.grid(alpha=0.2)
    trajectory_axis.legend(frameon=False)

    figure.suptitle(
        "WFI01-WFI03 fitted optical properties and phase-driving trajectories\n"
        "AR points are individual tiles with detector medians; AR k is fixed to zero",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = tuple(args.inputs) if args.inputs else DEFAULT_INPUTS
    detector_data = [load_detector(path) for path in inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.output, detector_data)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
