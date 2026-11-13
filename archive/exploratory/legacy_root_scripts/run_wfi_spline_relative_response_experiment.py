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


def _response_cache_root(response_cache_base, detector):
    base = Path(response_cache_base)
    detector = detector.lower()
    candidates = (
        base,
        base / f"{detector}_2nm",
        base / f"{detector}_400nm",
        base / detector,
    )
    for candidate in candidates:
        if (candidate / "tile_000" / "relative_response.npz").is_file():
            return candidate
    raise FileNotFoundError(
        f"No spline response cache found below {base} for {detector}"
    )


def _detector_warmup_root(warmup_prior_root, detector):
    if warmup_prior_root is None:
        return None
    root = Path(warmup_prior_root)
    detector_root = root / detector.lower() / "relative_response_constant_rate_warmup_start40"
    if (detector_root / "fits").is_dir():
        return detector_root
    return root


def _incremental_source_root(incremental_from, detector, output_subdir):
    """Resolve a previous production root or detector spline-output root."""
    base = Path(incremental_from)
    detector = detector.lower()
    candidates = (
        base / detector / output_subdir,
        base / detector,
        base,
    )
    for candidate in candidates:
        if (candidate / "fits" / "tile_000" / "posterior_samples.npz").is_file():
            return candidate
    raise FileNotFoundError(
        f"No previous spline fit found below {base} for {detector}"
    )


def _source_string(source, name):
    if name not in source:
        return None
    return str(np.asarray(source[name]).reshape(-1)[0])


def _source_mode(source_root):
    mode_path = Path(source_root) / "spline_relative_response_mode.json"
    if not mode_path.is_file():
        return {}
    return json.loads(mode_path.read_text())


def _constant_rate_warmup_diagnostics(current_fit):
    if Path(current_fit).name != "constant_rate_map.npz":
        return None
    with np.load(Path(current_fit), allow_pickle=False) as source:
        required = (
            "constant_rate_map_rate_nm_day",
            "constant_rate_map_objective",
        )
        if not all(name in source for name in required):
            return None
        if "constant_rate_map_initial_objective" in source:
            initial_objective = float(
                np.asarray(
                    source["constant_rate_map_initial_objective"]
                ).reshape(-1)[0]
            )
        elif "map_objective_history" in source:
            initial_objective = float(
                np.asarray(source["map_objective_history"]).reshape(-1)[0]
            )
        else:
            initial_objective = 0.0
        nfev_name = (
            "constant_rate_map_nfev"
            if "constant_rate_map_nfev" in source
            else "nfev"
        )
        if nfev_name not in source:
            return None
        return {
            "constant_rate_map_rate_nm_day": float(
                np.asarray(source["constant_rate_map_rate_nm_day"]).reshape(-1)[0]
            ),
            "residual_cost": float(
                np.asarray(source["constant_rate_map_objective"]).reshape(-1)[0]
            ),
            "initial_residual_cost": initial_objective,
            "nfev": int(np.asarray(source[nfev_name]).reshape(-1)[0]),
        }


def _load_job(
    job,
    spacing_hours,
    extra_days,
    third_decon_day,
    fit_led_log_offsets,
    warmup_prior_root,
    thickness_prior_log_sigma,
    thickness_prior_stages,
    incremental,
    incremental_overlap_knots,
    jax,
    jnp,
):
    (
        tile_id,
        statistics_path,
        cache_path,
        current_fit,
        output_dir,
    ) = job
    data = production.load_tile_statistics(
        statistics_path,
        third_decon_day=third_decon_day,
    )
    data["statistics_path"] = str(statistics_path)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    cache = relative_fit.load_response_cache(cache_path, data, context)
    initial_fit, initial_statistics = relative_runner.cache_source_paths(cache_path)
    if incremental:
        with np.load(Path(current_fit), allow_pickle=False) as source:
            old_statistics_path = _source_string(source, "statistics_path")
            if old_statistics_path is None:
                raise ValueError(f"Incremental source is missing statistics_path: {current_fit}")
            source_has_led_offsets = "led_log_offsets" in source
            saved_old_knots = (
                np.asarray(source["spline_knot_days"], dtype=float)
                if "spline_knot_days" in source
                else None
            )
            saved_old_boundaries = (
                np.asarray(source["spline_knot_boundaries"], dtype=float)
                if "spline_knot_boundaries" in source
                else None
            )
            saved_old_segment_sizes = (
                np.asarray(source["spline_knot_segment_sizes"], dtype=int)
                if "spline_knot_segment_sizes" in source
                else None
            )
        old_statistics_path = Path(old_statistics_path)
        if not old_statistics_path.is_file():
            raise FileNotFoundError(
                f"Historical statistics for incremental fit are missing: {old_statistics_path}"
            )
        old_data = production.load_tile_statistics(
            old_statistics_path,
            third_decon_day=third_decon_day,
        )
        old_data["statistics_path"] = str(old_statistics_path)
        if saved_old_knots is not None and saved_old_boundaries is not None and saved_old_segment_sizes is not None:
            old_layout = spline.spline_knot_layout_from_metadata(
                saved_old_knots,
                saved_old_boundaries,
                saved_old_segment_sizes,
                spacing_hours=spacing_hours,
            )
        else:
            old_layout = spline.build_spline_knot_layout(
                old_data,
                spacing_hours=spacing_hours,
                extra_days=(
                    tuple(saved_old_knots.tolist())
                    if saved_old_knots is not None
                    else tuple(extra_days)
                ),
            )
        knot_layout = spline.build_incremental_knot_layout(
            old_data,
            data,
            old_layout,
            spacing_hours=spacing_hours,
            extra_days=extra_days,
            overlap_knots=incremental_overlap_knots,
        )
        current_values, rate_reference = (
            spline._load_incremental_initial_and_current_values(
                data,
                current_fit,
                initial_fit,
                initial_statistics,
                old_data,
                old_layout,
                knot_layout,
                fit_led_log_offsets=source_has_led_offsets,
            )
        )
        free_rate_mask = np.broadcast_to(
            spline.incremental_new_knot_mask(
                knot_layout,
                float(np.max(np.asarray(old_data["days"], dtype=float))),
            ),
            np.asarray(current_values["np_log_rate_nodes"]).shape,
        ).copy()
        thickness_prior_centers = None
        fit_thickness_prior_log_sigma = production.INITIAL_THICKNESS_LOG_SIGMA
        effective_fit_led_log_offsets = source_has_led_offsets
    else:
        knot_layout = spline.build_spline_knot_layout(
            data,
            spacing_hours=spacing_hours,
            extra_days=extra_days,
        )
        current_values, rate_reference = spline._load_initial_and_current_values(
            data,
            current_fit,
            initial_fit,
            initial_statistics,
            knot_layout,
            fit_led_log_offsets=fit_led_log_offsets,
        )
        free_rate_mask = None
        effective_fit_led_log_offsets = fit_led_log_offsets
    constant_rate_warmup_diagnostics = _constant_rate_warmup_diagnostics(current_fit)
    if not incremental and warmup_prior_root is None:
        thickness_prior_centers = None
        fit_thickness_prior_log_sigma = production.INITIAL_THICKNESS_LOG_SIGMA
    elif not incremental:
        thickness_prior_centers, fit_thickness_prior_log_sigma = (
            relative_fit.warmup_thickness_prior(
                warmup_prior_root,
                tile_id,
                data,
                thickness_prior_log_sigma,
                thickness_prior_stages,
            )
        )
    return {
        "tile_id": tile_id,
        "statistics_path": statistics_path,
        "cache_path": cache_path,
        "output_dir": output_dir,
        "data": data,
        "cache": cache,
        "knot_layout": knot_layout,
        "current_values": current_values,
        "rate_reference": rate_reference,
        "thickness_prior_centers": thickness_prior_centers,
        "thickness_prior_log_sigma": fit_thickness_prior_log_sigma,
        "free_rate_mask": free_rate_mask,
        "constant_rate_warmup_diagnostics": constant_rate_warmup_diagnostics,
        "incremental": bool(incremental),
        "incremental_overlap_knots": int(incremental_overlap_knots),
        "fit_led_log_offsets": bool(effective_fit_led_log_offsets),
    }


def _fit_job_chunk(arguments):
    (
        jobs,
        spacing_hours,
        extra_days,
        third_decon_day,
        fit_led_log_offsets,
        warmup_prior_root,
        thickness_prior_log_sigma,
        thickness_prior_stages,
        led_log_offset_prior_sigma_log,
        incremental,
        incremental_overlap_knots,
        max_nfev,
        joint_batch,
    ) = arguments
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    loaded = [
        _load_job(
            job,
            spacing_hours,
            extra_days,
            third_decon_day,
            fit_led_log_offsets,
            warmup_prior_root,
            thickness_prior_log_sigma,
            thickness_prior_stages,
            incremental,
            incremental_overlap_knots,
            jax,
            jnp,
        )
        for job in jobs
    ]
    if not loaded:
        return []
    first = loaded[0]
    for item in loaded[1:]:
        if not np.array_equal(
            first["knot_layout"]["flat_knots"],
            item["knot_layout"]["flat_knots"],
        ):
            raise ValueError("Spline knot layouts differ within a detector")
        for name in ("days", "groups", "led", "first_by_led"):
            if not np.array_equal(
                np.asarray(first["data"][name]),
                np.asarray(item["data"][name]),
            ):
                raise ValueError(f"Observation layouts differ in {name}")
    results = []
    def write_item(item, values, diagnostics):
        state = spline.spline_relative_state(
            item["data"],
            values,
            item["cache"],
            item["knot_layout"],
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
            fit_led_log_offsets=item["fit_led_log_offsets"],
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
            fit_mode="spline_relative_response_ice_only",
            thickness_prior_log_sigma=(
                thickness_prior_log_sigma
                if warmup_prior_root is not None and not item["incremental"]
                else None
            ),
            thickness_prior_stages=(
                thickness_prior_stages
                if warmup_prior_root is not None and not item["incremental"]
                else None
            ),
            warmup_prior_root=warmup_prior_root,
            constant_rate_warmup_diagnostics=item[
                "constant_rate_warmup_diagnostics"
            ],
            spline_knot_days=item["knot_layout"]["flat_knots"],
            spline_knot_boundaries=item["knot_layout"]["boundaries"],
            spline_knot_segment_sizes=[
                len(segment) for segment in item["knot_layout"]["segments"]
            ],
        )
        print(
            f"Wrote spline relative-response fit under {item['output_dir']} "
            f"({diagnostics['nfev']} evaluations)",
            flush=True,
        )
        results.append(diagnostics)

    if joint_batch:
        values, batch_diagnostics = spline.fit_spline_batch(
            [item["data"] for item in loaded],
            [item["cache"] for item in loaded],
            [item["current_values"] for item in loaded],
            [item["rate_reference"] for item in loaded],
            first["knot_layout"],
            jax,
            jnp,
            max_nfev,
            thickness_prior_centers_list=[
                item["thickness_prior_centers"] for item in loaded
            ],
            thickness_prior_log_sigma=first["thickness_prior_log_sigma"],
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        )
        for index, item in enumerate(loaded):
            tile_values = dict(item["current_values"])
            for name in spline.relative.variable_sites(tile_values):
                source_shape = np.asarray(tile_values[name]).shape
                tile_values[name] = np.asarray(values[name])[index].reshape(source_shape)
            tile_diagnostics = {
                "converged": bool(batch_diagnostics["converged"]),
                "message": str(batch_diagnostics["message"]),
                "nfev": int(batch_diagnostics["nfev"]),
                "parameter_count": int(batch_diagnostics["parameter_count"]),
                "compile_seconds": float(batch_diagnostics["compile_seconds"]),
                "total_seconds": float(
                    batch_diagnostics["total_seconds"] / len(loaded)
                ),
                "residual_cost": float(batch_diagnostics["residual_costs"][index]),
                "initial_residual_cost": float(
                    batch_diagnostics["initial_residual_costs"][index]
                ),
                "trajectory_rate_initialization_nm_day": float(
                    batch_diagnostics["trajectory_rate_initialization_nm_day"][index]
                ),
            }
            write_item(item, tile_values, tile_diagnostics)
    else:
        evaluator = spline.build_spline_evaluator(
            first["data"],
            first["current_values"],
            first["cache"],
            first["knot_layout"],
            jax,
            jnp,
            thickness_prior_log_sigma=first["thickness_prior_log_sigma"],
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        )
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
                thickness_prior_centers=item["thickness_prior_centers"],
                thickness_prior_log_sigma=item["thickness_prior_log_sigma"],
                led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
                free_rate_mask=item["free_rate_mask"],
            )
            write_item(item, values, diagnostics)
    return results


def run_detector(
    detector,
    statistics_dir,
    response_cache_base,
    *,
    tile_size=DEFAULT_TILE_SIZE,
    output_subdir=DEFAULT_OUTPUT_SUBDIR,
    current_subdir="relative_response",
    constant_rate_warmup_root=None,
    selected_tile_ids=None,
    force=False,
    workers=DEFAULT_WORKERS,
    batch_size=1,
    joint_batch=False,
    spacing_hours=48.0,
    extra_days=(),
    third_decon_day=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    warmup_prior_root=None,
    thickness_prior_log_sigma=None,
    thickness_prior_stages=None,
    initial_from=None,
    incremental_from=None,
    incremental_overlap_knots=0,
    max_nfev=production.PRODUCTION_MAP_STEPS,
):
    detector = detector.upper()
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {detector}")
    if initial_from is not None and incremental_from is not None:
        raise ValueError("Choose either --initial-from or --incremental-from")
    if constant_rate_warmup_root is not None and (
        initial_from is not None or incremental_from is not None
    ):
        raise ValueError(
            "Constant-rate warmup initialization cannot be combined with "
            "--initial-from or --incremental-from"
        )
    incremental = incremental_from is not None
    if incremental and warmup_prior_root is not None:
        raise ValueError("Incremental spline fitting cannot use a warmup prior")
    if incremental and joint_batch:
        raise ValueError("Incremental spline fitting requires independent tile fits")
    incremental_overlap_knots = int(incremental_overlap_knots)
    if incremental_overlap_knots < 0:
        raise ValueError("Incremental overlap knot count must be non-negative")
    if not incremental and incremental_overlap_knots:
        raise ValueError("Incremental overlap knots require --incremental-from")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if warmup_prior_root is None:
        if thickness_prior_log_sigma is not None or thickness_prior_stages is not None:
            raise ValueError(
                "Thickness prior settings require warmup_prior_root"
            )
    else:
        if third_decon_day is None:
            raise ValueError(
                "Warmup-centered thickness priors require third_decon_day"
            )
        if thickness_prior_log_sigma is None:
            thickness_prior_log_sigma = (
                relative_fit.DEFAULT_WARMUP_THICKNESS_PRIOR_LOG_SIGMA
            )
        if thickness_prior_stages is None:
            thickness_prior_stages = relative_fit.DEFAULT_WARMUP_THICKNESS_PRIOR_STAGES
        warmup_prior_root = _detector_warmup_root(warmup_prior_root, detector)
    statistics_dir = Path(statistics_dir)
    statistics_path = corrected_statistics_path(statistics_dir, detector, tile_size)
    if not statistics_path.is_file():
        raise FileNotFoundError(statistics_path)
    detector_root = detector_output_root(statistics_dir, detector, tile_size)
    output_root = detector_root / output_subdir
    source_root = None
    source_mode = {}
    current_fit_name = "posterior_samples.npz"
    if initial_from is not None:
        source_root = _incremental_source_root(
            initial_from,
            detector,
            output_subdir,
        )
        if source_root.resolve() == output_root.resolve():
            raise ValueError("Initialization source and output roots must be different")
    elif incremental:
        source_root = _incremental_source_root(
            incremental_from,
            detector,
            output_subdir,
        )
        if source_root.resolve() == output_root.resolve():
            raise ValueError("Incremental source and output roots must be different")
        source_mode = _source_mode(source_root)
        source_extra_days = tuple(
            float(value) for value in source_mode.get("spline_extra_knot_days", ())
        )
        extra_days = tuple(dict.fromkeys(source_extra_days + tuple(extra_days)))
        source_spacing = source_mode.get("spline_spacing_hours")
        if source_spacing is not None and not np.isclose(
            float(source_spacing), float(spacing_hours), rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "Incremental spline spacing must match the previous fit"
            )
        source_third_decon = source_mode.get("third_decon_day")
        if third_decon_day is None and source_third_decon is not None:
            third_decon_day = float(source_third_decon)
        elif (
            source_third_decon is not None
            and third_decon_day is not None
            and not np.isclose(
                float(source_third_decon),
                float(third_decon_day),
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "Incremental third-decon day must match the previous fit"
            )
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

    response_cache_root = _response_cache_root(response_cache_base, detector)
    if constant_rate_warmup_root is not None:
        current_root = Path(constant_rate_warmup_root)
        current_fit_name = "constant_rate_map.npz"
        if not (current_root / "fits" / "tile_000" / current_fit_name).is_file():
            raise FileNotFoundError(
                "Constant-rate warmup archive is missing: "
                f"{current_root / 'fits' / 'tile_000' / current_fit_name}"
            )
    else:
        current_root = source_root if source_root is not None else detector_root / current_subdir
    jobs = []
    skipped = 0
    for tile_id in selected:
        cache_path = response_cache_root / f"tile_{tile_id:03d}" / "relative_response.npz"
        current_fit = current_root / "fits" / f"tile_{tile_id:03d}" / current_fit_name
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

    if joint_batch and jobs:
        job_groups = [
            jobs[start : start + batch_size] for start in range(0, len(jobs), batch_size)
        ]
        active_workers = min(max(1, int(workers)), len(job_groups))
    else:
        job_groups = None
        active_workers = min(max(1, int(workers)), len(jobs)) if jobs else 1
    _configure_parallel_environment(active_workers)
    output_root.mkdir(parents=True, exist_ok=True)
    fit_root.mkdir(parents=True, exist_ok=True)
    print(
        f"{detector}: fitting {len(jobs)} spline tile(s); skipped {skipped}; "
        f"workers={active_workers}, knots at {spacing_hours:g} hours, "
        f"mode={'joint batch' if joint_batch else 'independent'}, "
        f"batch_size={batch_size}",
        flush=True,
    )
    if jobs:
        chunks = (
            job_groups
            if joint_batch
            else [jobs[index::active_workers] for index in range(active_workers)]
        )
        arguments = [
            (
                chunk,
                float(spacing_hours),
                tuple(extra_days),
                third_decon_day,
                bool(fit_led_log_offsets),
                warmup_prior_root,
                thickness_prior_log_sigma,
                thickness_prior_stages,
                float(led_log_offset_prior_sigma_log),
                bool(incremental),
                incremental_overlap_knots,
                int(max_nfev),
                bool(joint_batch),
            )
            for chunk in chunks
            if chunk
        ]
        if len(arguments) == 1:
            _fit_job_chunk(arguments[0])
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=active_workers,
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
        "spline_reset_boundary_mode": "independent_piecewise_segments",
        "third_decon_day": (
            None if third_decon_day is None else float(third_decon_day)
        ),
        "response_cache_root": str(response_cache_root.resolve()),
        "current_fit_root": str(current_root.resolve()),
        "constant_rate_warmup_root": (
            None
            if constant_rate_warmup_root is None
            else str(Path(constant_rate_warmup_root).resolve())
        ),
        "statistics": str(statistics_path.resolve()),
        "tile_size": int(tile_size),
        "workers": int(workers),
        "batch_size": int(batch_size),
        "joint_batch": bool(joint_batch),
        "incremental": bool(incremental),
        "incremental_new_knots_only": bool(
            incremental and incremental_overlap_knots == 0
        ),
        "incremental_overlap_knots": int(incremental_overlap_knots),
        "incremental_from": (
            None
            if not incremental
            else str(source_root.resolve())
        ),
        "initial_from": (
            None
            if initial_from is None
            else str(Path(initial_from).resolve())
        ),
        "max_nfev": int(max_nfev),
        "force": bool(force),
        "fit_led_log_offsets": bool(
            source_mode.get("fit_led_log_offsets", fit_led_log_offsets)
            if incremental
            else fit_led_log_offsets
        ),
        "led_log_offset_prior_sigma_log": float(led_log_offset_prior_sigma_log),
        "warmup_prior_root": (
            None
            if warmup_prior_root is None
            else str(Path(warmup_prior_root).resolve())
        ),
        "thickness_prior_log_sigma": (
            None
            if thickness_prior_log_sigma is None
            else float(thickness_prior_log_sigma)
        ),
        "thickness_prior_stages": thickness_prior_stages,
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
    parser.add_argument(
        "--constant-rate-warmup-root",
        type=Path,
        default=None,
        help=(
            "Use fits/tile_###/constant_rate_map.npz from this separate "
            "constant-rate warmup as the spline initializer."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=positive_integer, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--joint-batch", action="store_true")
    parser.add_argument("--spacing-hours", type=float, default=48.0)
    parser.add_argument("--extra-knot", type=float, action="append", default=[])
    parser.add_argument("--third-decon-day", type=float, default=None)
    parser.add_argument("--fit-led-log-offsets", action="store_true")
    parser.add_argument(
        "--led-log-offset-prior-sigma-log",
        type=float,
        default=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    )
    parser.add_argument("--warmup-prior-root", type=Path, default=None)
    parser.add_argument("--thickness-prior-log-sigma", type=float, default=None)
    parser.add_argument(
        "--thickness-prior-stages",
        choices=("all", "third-decon"),
        default=None,
    )
    parser.add_argument(
        "--initial-from",
        type=Path,
        default=None,
        help=(
            "Previous spline production root used only for initialization; "
            "fit all knots against the current statistics"
        ),
    )
    parser.add_argument(
        "--incremental-from",
        type=Path,
        default=None,
        help=(
            "Previous spline output root or production root; fit only knots "
            "newer than the previous spline endpoint"
        ),
    )
    parser.add_argument(
        "--incremental-overlap-knots",
        type=int,
        default=0,
        help="Also refit this many historical knots at the old spline edge",
    )
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
            constant_rate_warmup_root=args.constant_rate_warmup_root,
            selected_tile_ids=args.tile_id,
            force=args.force,
            workers=args.workers,
            batch_size=args.batch_size,
            joint_batch=args.joint_batch,
            spacing_hours=args.spacing_hours,
            extra_days=args.extra_knot,
            third_decon_day=args.third_decon_day,
            fit_led_log_offsets=args.fit_led_log_offsets,
            led_log_offset_prior_sigma_log=args.led_log_offset_prior_sigma_log,
            warmup_prior_root=args.warmup_prior_root,
            thickness_prior_log_sigma=args.thickness_prior_log_sigma,
            thickness_prior_stages=args.thickness_prior_stages,
            initial_from=args.initial_from,
            incremental_from=args.incremental_from,
            incremental_overlap_knots=args.incremental_overlap_knots,
            max_nfev=args.max_nfev,
        )


if __name__ == "__main__":
    main()
