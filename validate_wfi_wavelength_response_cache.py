"""Validate wavelength-space caches against direct TMM evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
from wfi_relative_response import (
    build_spectral_response_lookup,
    evaluate_spectral_response_cache,
    load_optical_values,
    load_spectral_response_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-id", type=int, action="append")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def validate_tile(cache_root, fit_root, detector, tile_id, wavelengths, weights, thickness):
    cache = load_spectral_response_cache(
        cache_root / f"tile_{tile_id:03d}" / "spectral_response.npz",
        detector=detector,
        tile_id=tile_id,
    )
    optical_values = load_optical_values(
        fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz"
    )
    direct = build_spectral_response_lookup(optical_values, wavelengths, thickness)
    cached_clean, cached_relative = evaluate_spectral_response_cache(
        cache, thickness, wavelengths
    )
    relative_error = cached_relative - direct["relative_log_response"]
    clean_error = cached_clean - direct["clean_power"]
    direct_integrated = []
    cached_integrated = []
    for index in range(thickness.size):
        direct_integrated.append(
            np.log(
                np.sum(
                    weights
                    * direct["clean_power"][None, :]
                    * np.exp(direct["relative_log_response"][index]),
                    axis=1,
                )
                / np.sum(weights * direct["clean_power"][None, :], axis=1)
            )
        )
        cached_integrated.append(
            np.log(
                np.sum(
                    weights * cached_clean[None, :] * np.exp(cached_relative[index]),
                    axis=1,
                )
                / np.sum(weights * cached_clean[None, :], axis=1)
            )
        )
    integrated_error = np.asarray(cached_integrated) - np.asarray(direct_integrated)
    return {
        "tile_id": int(tile_id),
        "relative_rms_log": float(np.sqrt(np.mean(relative_error**2))),
        "relative_p99_abs_log": float(np.quantile(np.abs(relative_error), 0.99)),
        "relative_max_abs_log": float(np.max(np.abs(relative_error))),
        "clean_max_abs": float(np.max(np.abs(clean_error))),
        "integrated_rms_log": float(np.sqrt(np.mean(integrated_error**2))),
        "integrated_max_abs_log": float(np.max(np.abs(integrated_error))),
    }


def main():
    args = parse_args()
    tile_ids = tuple(args.tile_id or (0, 44, 123, 239, 255))
    context = production.load_response_context(np.asarray([args.detector]))
    wavelengths = np.asarray(context["wavelengths"], dtype=float)
    weights = np.asarray(context["model_weights"][0], dtype=float)
    thickness = np.asarray([1.0, 100.5, 333.3, 499.0])
    rows = [
        validate_tile(
            args.cache_root,
            args.fit_root,
            args.detector,
            tile_id,
            wavelengths,
            weights,
            thickness,
        )
        for tile_id in tile_ids
    ]
    summary = {
        "cache_root": str(args.cache_root.resolve()),
        "fit_root": str(args.fit_root.resolve()),
        "detector": args.detector,
        "tile_count": len(rows),
        "wavelength_count": int(wavelengths.size),
        "thickness_nm": thickness.tolist(),
        "tiles": rows,
        "max_relative_abs_log": float(max(row["relative_max_abs_log"] for row in rows)),
        "max_integrated_abs_log": float(
            max(row["integrated_max_abs_log"] for row in rows)
        ),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
