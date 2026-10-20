"""Build per-tile fixed-optics relative-response caches."""

import argparse
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
from wfi_production_layout import (
    positive_integer,
    tile_directory_entries,
    tile_statistics_path,
)
from wfi_relative_response import (
    build_response_lookup,
    load_optical_values,
    write_response_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-root", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument("--tile-id", type=int, action="append")
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--max-thickness", type=float, default=400.0)
    args = parser.parse_args()
    if args.grid_step <= 0.0 or args.max_thickness <= 0.0:
        raise ValueError("grid-step and max-thickness must be positive")
    return args


def main():
    args = parse_args()
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    fit_entries = dict(tile_directory_entries(args.fit_root))
    tile_ids = sorted(fit_entries) if args.tile_id is None else sorted(set(args.tile_id))
    grid = np.linspace(
        0.0,
        float(args.max_thickness),
        int(round(args.max_thickness / args.grid_step)) + 1,
    )
    compiled_predictor = None
    for tile_id in tile_ids:
        fit_dir = fit_entries.get(tile_id)
        if fit_dir is None:
            raise FileNotFoundError(f"No fit archive for tile {tile_id}: {args.fit_root}")
        fit_archive = fit_dir / "posterior_samples.npz"
        statistics_path = tile_statistics_path(
            args.statistics_root,
            args.detector,
            args.tile_size,
            tile_id,
        )
        if not statistics_path.is_file():
            raise FileNotFoundError(statistics_path)
        data = production.load_tile_statistics(statistics_path)
        context = production.load_response_context(data["sca_names"])
        data["qe_columns"] = context["qe_columns"]
        data["response_wavelengths_nm"] = context["wavelengths"]
        data["response_wavelength_count"] = int(context["wavelengths"].size)
        optics = load_optical_values(fit_archive)
        curves, build_seconds, compiled_predictor = build_response_lookup(
            data,
            context,
            optics,
            grid,
            compiled_predictor=compiled_predictor,
        )
        output = args.output_root / f"tile_{tile_id:03d}" / "relative_response.npz"
        write_response_cache(
            output,
            data,
            context,
            optics,
            grid,
            curves,
            fit_archive,
            statistics_path,
            build_seconds,
        )
        print(
            f"Wrote {output} ({build_seconds:.2f}s response build)",
            flush=True,
        )


if __name__ == "__main__":
    main()
