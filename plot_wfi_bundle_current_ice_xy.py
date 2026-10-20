"""Plot bundled fitted ice thickness at tile focal-plane positions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wfi_wavelength_response_bundle import load_bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--detector")
    parser.add_argument("--mean-last-observations", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mean_last_observations < 1:
        parser.error("--mean-last-observations must be positive")
    return args


def main():
    args = parse_args()
    bundle = load_bundle(args.bundle)
    if not bundle.get("trajectory_included", False):
        raise ValueError("Bundle does not contain fitted ice trajectories")
    selected = np.ones(bundle["tile_ids"].size, dtype=bool)
    if args.detector is not None:
        selected &= bundle["detectors"] == str(args.detector).strip().upper()
    if not np.any(selected):
        raise ValueError(f"Detector {args.detector} is not present in the bundle")
    indices = np.flatnonzero(selected)
    count = args.mean_last_observations
    thickness = np.asarray(bundle["tile_ice_thickness"], dtype=float)
    if thickness.shape[1] < count:
        raise ValueError("Requested averaging window exceeds the observation count")
    values = np.mean(thickness[indices, -count:], axis=1)
    x = np.asarray(bundle["tile_center_x_mm"], dtype=float)[indices]
    y = np.asarray(bundle["tile_center_y_mm"], dtype=float)[indices]
    detectors = sorted({str(value) for value in bundle["detectors"][indices]})
    latest_time = str(np.asarray(bundle["observation_start_time"])[-1])
    latest_label = latest_time[:16].replace("T", " ") + " UTC"
    vmin = float(np.nanmin(values))
    vmax = float(np.nanpercentile(values, 99.0))

    figure, axis = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    scatter = axis.scatter(
        x,
        y,
        c=values,
        s=72,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.35,
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02, extend="max")
    colorbar.set_label("Fitted ice thickness (nm)")
    axis.set_xlabel("MPA focal-plane x (mm; positive left)")
    axis.set_ylabel("MPA focal-plane y (mm)")
    window_label = (
        "latest observation"
        if count == 1
        else f"mean of last {count} observations"
    )
    detector_label = ", ".join(detectors) if len(detectors) == 1 else "All WFI detectors"
    axis.set_title(
        f"{detector_label} bundled ice thickness ({window_label})\n"
        f"mean {np.mean(values):.3f} nm, median {np.median(values):.3f} nm; "
        f"vmax p99 {vmax:.3f} nm; "
        f"latest {latest_label}"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.invert_xaxis()
    axis.grid(True, alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(
        f"Wrote {args.output}; tiles={values.size}, mean={np.mean(values):.9g} nm, "
        f"median={np.median(values):.9g} nm",
        flush=True,
    )


if __name__ == "__main__":
    main()
