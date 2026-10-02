"""Compare tile-level MAP AR properties across WFI detector fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "wfi01_wfi03_ar_property_comparison.png"
DEFAULT_SUMMARY = ROOT / "wfi01_wfi03_ar_property_comparison.csv"
DEFAULT_INPUTS = tuple(
    ROOT / path
    for path in (
        "wfi01_absolute_substrate_power_weak_priors_map10000_tile_corrections_x64_free_tile_trajectories",
        "wfi02_absolute_substrate_power_weak_priors_map10000_tile_corrections_x64_free_tile_trajectories_prod",
        "wfi03_absolute_substrate_power_weak_priors_map10000_tile_corrections_x64_free_tile_trajectories",
    )
)


def load_ar_posterior(path):
    path = Path(path)
    posterior_path = path / "tile_ar_posterior.csv" if path.is_dir() else path
    with posterior_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No AR posterior rows found in {posterior_path}")
    detector = str(rows[0]["detector"])
    layers = []
    layer = 1
    while f"ar_layer{layer}_thickness_median" in rows[0]:
        layers.append(layer)
        layer += 1
    if not layers:
        raise ValueError(f"No AR layer columns found in {posterior_path}")
    values = {
        (layer, parameter): np.asarray(
            [float(row[f"ar_layer{layer}_{parameter}_median"]) for row in rows],
            dtype=float,
        )
        for layer in layers
        for parameter in ("thickness", "n", "k")
    }
    return detector, layers, values


def write_summary(path, records):
    fields = (
        "detector",
        "layer",
        "parameter",
        "tile_count",
        "tile_median",
        "tile_min",
        "tile_max",
        "tile_std",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def make_plot(output, detector_data):
    detectors = [item[0] for item in detector_data]
    layers = detector_data[0][1]
    parameters = ("thickness", "n")
    parameter_labels = {
        "thickness": "thickness scale",
        "n": "refractive index n",
    }
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    figure, axes = plt.subplots(
        len(layers), len(parameters),
        figsize=(12, 3.2 * len(layers)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    x = np.arange(len(detectors), dtype=float)
    for layer_index, layer in enumerate(layers):
        for parameter_index, parameter in enumerate(parameters):
            axis = axes[layer_index, parameter_index]
            for detector_index, (detector, _, values) in enumerate(detector_data):
                tile_values = values[(layer, parameter)]
                jitter = np.linspace(-0.2, 0.2, len(tile_values))
                axis.scatter(
                    x[detector_index] + jitter,
                    tile_values,
                    color=colors[detector_index % len(colors)],
                    alpha=0.45,
                    s=14,
                    linewidths=0,
                )
                median = float(np.median(tile_values))
                axis.plot(
                    [x[detector_index] - 0.22, x[detector_index] + 0.22],
                    [median, median],
                    color="black",
                    linewidth=2.2,
                )
            axis.set_title(f"AR layer {layer}: {parameter_labels[parameter]}")
            axis.grid(alpha=0.2)
            axis.set_xticks(x, detectors)
            if parameter_index == 0:
                axis.set_ylabel("Tile MAP value")
            if layer_index == len(layers) - 1:
                axis.set_xlabel("Detector")
    figure.suptitle(
        "WFI01-WFI03 tile-level MAP AR properties\n"
        "Points are individual tiles; black bars are detector tile medians; AR k is fixed to zero",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = tuple(args.inputs) if args.inputs else DEFAULT_INPUTS
    detector_data = [load_ar_posterior(path) for path in inputs]
    layers = detector_data[0][1]
    if any(item[1] != layers for item in detector_data):
        raise ValueError("All detector AR posteriors must have the same layer count")
    records = []
    for detector, _, values in detector_data:
        for (layer, parameter), tile_values in values.items():
            records.append(
                {
                    "detector": detector,
                    "layer": layer,
                    "parameter": parameter,
                    "tile_count": len(tile_values),
                    "tile_median": float(np.median(tile_values)),
                    "tile_min": float(np.min(tile_values)),
                    "tile_max": float(np.max(tile_values)),
                    "tile_std": float(np.std(tile_values)),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.output, detector_data)
    write_summary(args.summary, records)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
