"""Compare WFI01 tile ice trajectories under optical-model constraints."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = (
    (
        "Free AR n/k + free HgCdTe",
        ROOT
        / "wfi01_absolute_substrate_power_weak_priors_map10000_tile_corrections_x64_free_tile_trajectories",
    ),
    (
        "Shared AR n/k + free HgCdTe",
        ROOT
        / "wfi01_absolute_substrate_power_shared_ar_optical_properties_map10000_tile_corrections_x64_free_tile_trajectories",
    ),
    (
        "Shared AR n/k + fixed HgCdTe",
        ROOT
        / "wfi01_absolute_substrate_power_shared_ar_optical_properties_fixed_hgcdte_reference_map10000_tile_corrections_x64_free_tile_trajectories",
    ),
)


def load_model(path):
    with np.load(path / "posterior_samples.npz", allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float)
        thickness = np.asarray(samples["tile_ice_thickness"], dtype=float)
        rates = np.asarray(samples["tile_instantaneous_rate_nm_day"], dtype=float)
    if thickness.ndim == 3:
        thickness = np.median(thickness, axis=0)
    if rates.ndim == 3:
        rates = np.median(rates, axis=0)
    order = np.argsort(days, kind="stable")
    rows = []
    with (path / "tile_ar_posterior.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["tile_id"]), int(row["tile_row"]), int(row["tile_col"])))
    rows.sort()
    if thickness.shape != rates.shape or thickness.shape[0] != len(rows):
        raise ValueError(f"Tile posterior shapes do not match in {path}")
    return {
        "days": days[order],
        "thickness": thickness[:, order],
        "rates": rates[:, order],
        "tile_positions": rows,
    }


def format_range(values, unit):
    return f"{np.min(values):.2f}–{np.max(values):.2f} {unit}"


def make_trajectory_plot(output, models):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)
    for axis, (label, data) in zip(axes, models):
        thickness = data["thickness"]
        days = data["days"]
        for tile in thickness:
            axis.plot(days, tile, color="#777777", alpha=0.16, linewidth=0.7)
        axis.fill_between(
            days,
            np.min(thickness, axis=0),
            np.max(thickness, axis=0),
            color="#56B4E9",
            alpha=0.25,
            label="Tile min–max",
        )
        axis.plot(
            days,
            np.median(thickness, axis=0),
            color="#0072B2",
            linewidth=2.0,
            label="Tile median",
        )
        final = thickness[:, -1]
        axis.set_title(f"{label}\nfinal range {format_range(final, 'nm')}")
        axis.set_xlabel("Days since first exposure")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Ice thickness (nm)")
    figure.suptitle("WFI01 tile ice trajectories under optical-model constraints", fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_spatial_plot(output, models):
    thickness_maps = []
    rate_maps = []
    for _, data in models:
        thickness_map = np.full((8, 8), np.nan)
        rate_map = np.full((8, 8), np.nan)
        for tile, row, column in data["tile_positions"]:
            thickness_map[row, column] = data["thickness"][tile, -1]
            rate_map[row, column] = data["rates"][tile, -1]
        thickness_maps.append(thickness_map)
        rate_maps.append(rate_map)

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    thickness_min = min(np.nanmin(values) for values in thickness_maps)
    thickness_max = max(np.nanmax(values) for values in thickness_maps)
    rate_min = min(np.nanmin(values) for values in rate_maps)
    rate_max = max(np.nanmax(values) for values in rate_maps)
    thickness_images = []
    rate_images = []
    for column, ((label, _), thickness_map, rate_map) in enumerate(
        zip(models, thickness_maps, rate_maps)
    ):
        thickness_images.append(
            axes[0, column].imshow(
                thickness_map,
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=thickness_min,
                vmax=thickness_max,
            )
        )
        rate_images.append(
            axes[1, column].imshow(
                rate_map,
                origin="lower",
                interpolation="none",
                cmap="plasma",
                vmin=rate_min,
                vmax=rate_max,
            )
        )
        axes[0, column].set_title(label)
        axes[1, column].set_title(label)
        for axis in axes[:, column]:
            axis.set_xlabel("Tile column")
            axis.set_ylabel("Tile row")
            axis.set_xticks(range(8))
            axis.set_yticks(range(8))
    figure.colorbar(thickness_images[0], ax=axes[0, :], fraction=0.025, pad=0.02, label="Final ice thickness (nm)")
    figure.colorbar(rate_images[0], ax=axes[1, :], fraction=0.025, pad=0.02, label="Final instantaneous rate (nm/day)")
    figure.suptitle("WFI01 final tile ice variation under optical-model constraints", fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=ROOT / "wfi01_ice_trajectory_model_comparison.png",
    )
    parser.add_argument(
        "--spatial-output",
        type=Path,
        default=ROOT / "wfi01_ice_spatial_model_comparison.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = [(label, load_model(path)) for label, path in MODELS]
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.spatial_output.parent.mkdir(parents=True, exist_ok=True)
    make_trajectory_plot(args.trajectory_output, models)
    make_spatial_plot(args.spatial_output, models)
    print(f"Wrote {args.trajectory_output}")
    print(f"Wrote {args.spatial_output}")


if __name__ == "__main__":
    main()
