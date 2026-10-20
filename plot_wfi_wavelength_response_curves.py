"""Plot wavelength-space relative response versus ice thickness for one tile."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wfi_wavelength_response_bundle import evaluate, load_bundle, tile_record


DEFAULT_WAVELENGTHS_NM = (500.0, 800.0, 1100.0, 1400.0, 1800.0, 2400.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--detector", default="WFI02")
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        nargs="+",
        default=DEFAULT_WAVELENGTHS_NM,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_bundle(args.bundle)
    thickness = np.asarray(bundle["ice_thickness_nm"], dtype=float)
    wavelengths = np.asarray(args.wavelength_nm, dtype=float)
    _, response = evaluate(
        bundle,
        args.tile_id,
        thickness,
        wavelengths,
        detector=args.detector,
    )
    response = np.asarray(response, dtype=float).T
    tile = tile_record(bundle, args.tile_id, detector=args.detector)

    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, wavelengths.size))
    for color, wavelength, values in zip(colors, wavelengths, response):
        axis.plot(thickness, values, color=color, linewidth=1.8, label=f"{wavelength:g} nm")
    axis.set_xlabel("Ice thickness (nm)")
    axis.set_ylabel("Relative log response, log(coated / clean)")
    axis.set_title(
        f"{args.detector} tile {args.tile_id:03d} wavelength response\n"
        f"MPA position ({tile['tile_center_x_mm']:.2f}, {tile['tile_center_y_mm']:.2f}) mm"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(title="Wavelength", ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
