"""Build generic wavelength-space, low-rank WFI response caches."""

import argparse
from pathlib import Path
import time

import numpy as np

import fit_wfi_tmm_production as production
from wfi_production_layout import positive_integer, tile_directory_entries
from wfi_relative_response import (
    build_spectral_response_lookup,
    load_optical_values,
    write_spectral_response_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-id", type=int, action="append")
    parser.add_argument("--wavelength-min-nm", type=float, default=470.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=2700.0)
    parser.add_argument("--wavelength-step-nm", type=float, default=2.0)
    parser.add_argument("--max-thickness-nm", type=float, default=500.0)
    parser.add_argument("--thickness-step-nm", type=float, default=2.0)
    parser.add_argument("--rank", type=positive_integer, default=16)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse already-written spectral_response.npz tile caches.",
    )
    args = parser.parse_args()
    if args.wavelength_min_nm <= 0.0:
        raise ValueError("wavelength-min-nm must be positive")
    if args.wavelength_max_nm <= args.wavelength_min_nm:
        raise ValueError("wavelength-max-nm must exceed wavelength-min-nm")
    if args.wavelength_step_nm <= 0.0 or args.thickness_step_nm <= 0.0:
        raise ValueError("grid steps must be positive")
    if args.max_thickness_nm <= 0.0:
        raise ValueError("max-thickness-nm must be positive")
    return args


def main():
    args = parse_args()
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    entries = dict(tile_directory_entries(args.fit_root))
    tile_ids = sorted(entries) if args.tile_id is None else sorted(set(args.tile_id))
    wavelengths = np.arange(
        args.wavelength_min_nm,
        args.wavelength_max_nm + 0.5 * args.wavelength_step_nm,
        args.wavelength_step_nm,
    )
    thickness = np.arange(
        0.0,
        args.max_thickness_nm + 0.5 * args.thickness_step_nm,
        args.thickness_step_nm,
    )
    for tile_id in tile_ids:
        fit_dir = entries.get(tile_id)
        if fit_dir is None:
            raise FileNotFoundError(f"No fit archive for tile {tile_id}: {args.fit_root}")
        fit_archive = fit_dir / "posterior_samples.npz"
        output = (
            args.output_root / f"tile_{tile_id:03d}" / "spectral_response.npz"
        )
        if args.skip_existing and output.is_file():
            print(f"Skipping existing {output}", flush=True)
            continue
        optics = load_optical_values(fit_archive)
        started = time.perf_counter()
        spectral = build_spectral_response_lookup(optics, wavelengths, thickness)
        write_spectral_response_cache(
            output,
            args.detector,
            tile_id,
            spectral,
            fit_archive,
            args.rank,
        )
        print(
            f"Wrote {output} ({time.perf_counter() - started:.2f}s, "
            f"{wavelengths.size} wavelengths, {thickness.size} thickness nodes, "
            f"rank {args.rank})",
            flush=True,
        )


if __name__ == "__main__":
    main()
