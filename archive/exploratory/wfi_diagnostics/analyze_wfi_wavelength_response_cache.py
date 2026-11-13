"""Plot generic wavelength/thickness WFI response caches."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wfi_relative_response import evaluate_spectral_response_cache, load_spectral_response_cache


REPRESENTATIVE_TILES = (0, 44, 123, 239, 255)
ROMAN_FILTER_RANGES_NM = {
    "F062": [480.0, 760.0],
    "F087": [760.0, 977.0],
    "F106": [927.0, 1192.0],
    "F129": [1131.0, 1454.0],
    "F146": [927.0, 2000.0],
    "F158": [1380.0, 1774.0],
    "F184": [1683.0, 2000.0],
    "F213": [1950.0, 2300.0],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cache_path(cache_root, tile_id):
    return Path(cache_root) / f"tile_{tile_id:03d}" / "spectral_response.npz"


def plot_tile(output, cache, tile_id):
    wavelengths = cache["wavelengths_nm"]
    thickness = cache["ice_thickness_nm"]
    _, response = evaluate_spectral_response_cache(cache, thickness, wavelengths)
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(
        response,
        origin="lower",
        aspect="auto",
        extent=(wavelengths[0], wavelengths[-1], thickness[0], thickness[-1]),
        cmap="coolwarm",
    )
    figure.colorbar(image, ax=axis, label="log relative response")
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Ice thickness (nm)")
    axis.set_title(f"WFI01 tile {tile_id:03d}: generic spectral response")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_curves(output, caches):
    figure, axes = plt.subplots(1, len(caches), figsize=(16, 4), sharey=True)
    for axis, (tile_id, cache) in zip(axes, caches.items()):
        wavelengths = cache["wavelengths_nm"]
        query_thickness = np.asarray([0.0, 50.0, 100.0, 250.0, 500.0])
        _, response = evaluate_spectral_response_cache(
            cache, query_thickness, wavelengths
        )
        for index, thickness in enumerate(query_thickness):
            axis.plot(wavelengths, response[index], label=f"{thickness:g} nm")
        axis.axhline(0.0, color="0.25", linewidth=0.6)
        axis.set_title(f"Tile {tile_id:03d}")
        axis.set_xlabel("Wavelength (nm)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("log relative response")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Generic spectral response across representative tiles")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.cache_root).glob("tile_*/spectral_response.npz"))
    if not paths:
        raise FileNotFoundError(f"No spectral caches found under {args.cache_root}")
    representative = {}
    for tile_id in REPRESENTATIVE_TILES:
        path = cache_path(args.cache_root, tile_id)
        representative[tile_id] = load_spectral_response_cache(path)
        plot_tile(args.output_dir / f"tile_{tile_id:03d}_response_heatmap.png", representative[tile_id], tile_id)
    plot_curves(args.output_dir / "representative_response_curves.png", representative)
    first = representative[REPRESENTATIVE_TILES[0]]
    summary = {
        "cache_root": str(Path(args.cache_root).resolve()),
        "cache_count": len(paths),
        "representative_tiles": list(REPRESENTATIVE_TILES),
        "wavelength_count": int(first["wavelengths_nm"].size),
        "wavelength_min_nm": float(first["wavelengths_nm"][0]),
        "wavelength_max_nm": float(first["wavelengths_nm"][-1]),
        "wavelength_step_nm": float(np.median(np.diff(first["wavelengths_nm"]))),
        "thickness_count": int(first["ice_thickness_nm"].size),
        "thickness_max_nm": float(first["ice_thickness_nm"][-1]),
        "thickness_step_nm": float(np.median(np.diff(first["ice_thickness_nm"]))),
        "roman_filter_ranges_nm": ROMAN_FILTER_RANGES_NM,
        "response_min": float(
            min(
                np.min(
                    evaluate_spectral_response_cache(
                        cache,
                        cache["ice_thickness_nm"],
                        cache["wavelengths_nm"],
                    )[1]
                )
                for cache in representative.values()
            )
        ),
        "response_max": float(
            max(
                np.max(
                    evaluate_spectral_response_cache(
                        cache,
                        cache["ice_thickness_nm"],
                        cache["wavelengths_nm"],
                    )[1]
                )
                for cache in representative.values()
            )
        ),
    }
    (args.output_dir / "spectral_response_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"Analyzed {len(paths)} spectral caches")
    print(f"Wrote diagnostics under {args.output_dir}")


if __name__ == "__main__":
    main()
