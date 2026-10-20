"""Fit the reduced-knot spline relative-response experiment for WFI tiles."""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment_wfi_spline_trajectory as spline
import fit_wfi_relative_response as relative_fit
import fit_wfi_tmm_production as production
import run_wfi_relative_response_production as relative_runner
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    corrected_statistics_path,
    detector_output_root,
    positive_integer,
)


DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_WORKERS = 2
DEFAULT_OUTPUT_SUBDIR = "spline_relative_response"


def _configure_parallel_environment(workers):
    if workers <= 1:
        return
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _load_job(job, spacing_hours, extra_days, jax, jnp):
    tile_id, statistics_path, cache_path, current_fit, output_dir = job
    data = production.load_tile_statistics(statistics_path)
    data["statistics_path"] = str(statistics_path)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    cache = relative_fit.load_response_cache(cache_path, data, context)
    initial_fit, initial_statistics = relative_runner.cache_source_paths(cache_path)
    knot_days = spline.build_spline_knot_days(
        data,
        spacing_hours=spacing_hours,
        extra_days=extra_days,
    )
    current_values, rate_reference = spline._load_initial_and_current_values(
        data,
        current_fit,
        initial_fit,
        initial_statistics,
        knot_days,
    )
    return {
        "tile_id": tile_id,
        "statistics_path": statistics_path,
        "cache_path": cache_path,
        "output_dir": output_dir,
        "data": data,
        "cache": cache,
        "knot_days": knot_days,
        "current_values": current_values,
        "rate_reference": rate_reference,
    }


def _fit_job_chunk(arguments):
    jobs, spacing_hours, extra_days, max_nfev = arguments
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    loaded = [
        _load_job(job, spacing_hours, extra_days, jax, jnp)
        for job in jobs
    ]
    if not loaded:
        return []
    first = loaded[0]
    for item in loaded[1:]:
        if not np.array_equal(first["knot_days"], item["knot_days"]):
            raise ValueError("Spline knot layouts differ within a detector")
        for name in ("days", "groups", "led", "first_by_led"):
            if not np.array_equal(
                np.asarray(first["data"][name]),
                np.asarray(item["data"][name]),
            ):
                raise ValueError(f"Observation layouts differ in {name}")
    evaluator = spline.build_spline_evaluator(
        first["data"],
        first["current_values"],
        first["cache"],
        first["knot_days"],
        jax,
        jnp,
    )
    results = []
    for item in loaded:
        values, diagnostics = spline.fit_spline_reusable(
            item["data"],
            item["cache"],
            item["current_values"],
            item["rate_reference"],
            evaluator,
            jax,
            jnp,
            max_nfev,
        )
        state = spline.spline_relative_state(
            item["data"],
            values,
            item["cache"],
            item["knot_days"],
            jax,
            jnp,
        )
        relative_fit.write_outputs(
            item["output_dir"],
            item["data"],
            values,
            state,
            diagnostics,
            item["cache_path"],
            jax,
            jnp,
        )
        print(
            f"Wrote spline relative-response fit under {item['output_dir']} "
            f"({diagnostics['nfev']} evaluations)",
            flush=True,
        )
        results.append(diagnostics)
    return results


def run_detector(
    detector,
    statistics_dir,
    response_cache_base,
    *,
    tile_size=DEFAULT_TILE_SIZE,
    output_subdir=DEFAULT_OUTPUT_SUBDIR,
    current_subdir="relative_response",
    selected_tile_ids=None,
    force=False,
    workers=DEFAULT_WORKERS,
    spacing_hours=12.0,
    extra_days=(),
    max_nfev=production.PRODUCTION_MAP_STEPS,
):
    detector = detector.upper()
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {detector}")
    statistics_dir = Path(statistics_dir)
    statistics_path = corrected_statistics_path(statistics_dir, detector, tile_size)
    if not statistics_path.is_file():
        raise FileNotFoundError(statistics_path)
    detector_root = detector_output_root(statistics_dir, detector, tile_size)
    output_root = detector_root / output_subdir
    statistics_root = output_root / "tile_statistics"
    fit_root = output_root / "fits"
    paths = relative_runner.tile_statistics_paths(
        detector,
        statistics_path,
        statistics_root,
        tile_size,
    )
    all_tile_ids = sorted(paths)
    selected = all_tile_ids if selected_tile_ids is None else sorted(set(selected_tile_ids))
    missing = sorted(set(selected).difference(paths))
    if missing:
        raise ValueError(f"Unknown tile IDs for {detector}: {missing}")

    response_cache_root = Path(response_cache_base) / f"{detector.lower()}_2nm"
    current_root = detector_root / current_subdir
    jobs = []
    skipped = 0
    for tile_id in selected:
        cache_path = response_cache_root / f"tile_{tile_id:03d}" / "relative_response.npz"
        current_fit = current_root / "fits" / f"tile_{tile_id:03d}" / "posterior_samples.npz"
        output_dir = fit_root / f"tile_{tile_id:03d}"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        if not current_fit.is_file():
            raise FileNotFoundError(current_fit)
        relative_runner.validate_response_cache_provenance(cache_path)
        if not force and (output_dir / "posterior_samples.npz").is_file():
            skipped += 1
            continue
        jobs.append(
            (
                tile_id,
                paths[tile_id],
                cache_path,
                current_fit,
                output_dir,
            )
        )

    active_workers = min(max(1, int(workers)), len(jobs)) if jobs else 1
    _configure_parallel_environment(active_workers)
    output_root.mkdir(parents=True, exist_ok=True)
    fit_root.mkdir(parents=True, exist_ok=True)
    print(
        f"{detector}: fitting {len(jobs)} spline tile(s); skipped {skipped}; "
        f"workers={active_workers}, knots at {spacing_hours:g} hours",
        flush=True,
    )
    if jobs:
        chunks = [jobs[index::active_workers] for index in range(active_workers)]
        arguments = [
            (chunk, float(spacing_hours), tuple(extra_days), int(max_nfev))
            for chunk in chunks
            if chunk
        ]
        if len(arguments) == 1:
            _fit_job_chunk(arguments[0])
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=len(arguments),
                mp_context=context,
            ) as executor:
                list(executor.map(_fit_job_chunk, arguments))

    aggregate = relative_runner.write_detector_rate_posterior(fit_root, output_root)
    metadata = {
        "detector": detector,
        "fit_mode": "spline_relative_response_ice_only",
        "trajectory_parameterization": "natural_cubic_log_rate",
        "spline_spacing_hours": float(spacing_hours),
        "spline_extra_knot_days": [float(value) for value in extra_days],
        "response_cache_root": str(response_cache_root.resolve()),
        "current_fit_root": str(current_root.resolve()),
        "statistics": str(statistics_path.resolve()),
        "tile_size": int(tile_size),
        "workers": int(workers),
        "max_nfev": int(max_nfev),
        "force": bool(force),
    }
    (output_root / "spline_relative_response_mode.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    if aggregate is not None:
        print(f"Wrote {aggregate}", flush=True)
    return output_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--response-cache-base", type=Path, required=True)
    parser.add_argument("--detector", action="append")
    parser.add_argument("--all-detectors", action="store_true")
    parser.add_argument("--tile-id", type=int, action="append")
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--current-subdir", default="relative_response")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=positive_integer, default=DEFAULT_WORKERS)
    parser.add_argument("--spacing-hours", type=float, default=12.0)
    parser.add_argument("--extra-knot", type=float, action="append", default=[])
    parser.add_argument("--max-nfev", type=positive_integer, default=production.PRODUCTION_MAP_STEPS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detectors = DETECTORS if args.all_detectors or not args.detector else tuple(
        detector.upper() for detector in args.detector
    )
    for detector in detectors:
        run_detector(
            detector,
            args.statistics_dir,
            args.response_cache_base,
            tile_size=args.tile_size,
            output_subdir=args.output_subdir,
            current_subdir=args.current_subdir,
            selected_tile_ids=args.tile_id,
            force=args.force,
            workers=args.workers,
            spacing_hours=args.spacing_hours,
            extra_days=args.extra_knot,
            max_nfev=args.max_nfev,
        )


if __name__ == "__main__":
    main()
