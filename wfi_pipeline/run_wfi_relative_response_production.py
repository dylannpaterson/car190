"""Run per-tile ice-only fits using saved fixed-optics response caches."""

import argparse
import csv
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import fit_wfi_relative_response as relative_fit
from reset_events import DEFAULT_RESET_CONFIG, load_reset_config
from wfi_relative_response import (
    CACHE_VERSION,
    PRODUCTION_ENTROPY_WEIGHT,
    PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
    portable_path_string,
    resolve_portable_path,
    variable_sites,
    load_response_cache,
    validate_response_cache_provenance,
)
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    corrected_statistics_path,
    detector_output_root,
    positive_integer,
    tile_statistics_path,
)


DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_WORKERS = 4


def normalize_detector(value):
    detector = value.upper()
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {value}")
    return detector


def tile_statistics_paths(detector, statistics, statistics_root, tile_size):
    with Path(statistics).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows_by_tile = {}
        for row in reader:
            rows_by_tile.setdefault(int(row["tile_id"]), []).append(row)
    if not fieldnames or not rows_by_tile:
        raise ValueError(f"No tile statistics found in {statistics}")
    statistics_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tile_id, rows in sorted(rows_by_tile.items()):
        path = tile_statistics_path(statistics_root, detector, tile_size, tile_id)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        paths[tile_id] = path
    return paths


def cache_source_paths(cache_path):
    import numpy as np

    with np.load(cache_path, allow_pickle=False) as cache:
        source_fit = resolve_portable_path(
            str(cache["source_fit"].reshape(-1)[0])
        )
        source_statistics = resolve_portable_path(
            str(cache["source_statistics"].reshape(-1)[0])
        )
    if not source_fit.is_file():
        raise FileNotFoundError(f"Cached source fit is missing: {source_fit}")
    if not source_statistics.is_file():
        raise FileNotFoundError(
            f"Cached source statistics are missing: {source_statistics}"
        )
    return source_fit, source_statistics


def detector_rate_initialization_nm_day(cache_root):
    """Return a robust detector-level constant-rate warm start from source fits."""
    import numpy as np

    tile_medians = []
    cache_paths = sorted(Path(cache_root).glob("tile_*/relative_response.npz"))
    if not cache_paths:
        raise ValueError(f"No response caches found under {cache_root}")
    for cache_path in cache_paths:
        source_fit, _ = cache_source_paths(cache_path)
        with np.load(source_fit, allow_pickle=False) as source:
            if "node_instantaneous_rate_nm_day" not in source:
                raise ValueError(f"Source fit lacks rate nodes: {source_fit}")
            source_rates = np.asarray(
                source["node_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
        if source_rates.size == 0 or not np.all(np.isfinite(source_rates)):
            raise ValueError(f"Source fit has invalid rate nodes: {source_fit}")
        tile_medians.append(float(np.median(source_rates)))
    initialization = float(np.median(np.asarray(tile_medians)))
    if not np.isfinite(initialization) or initialization <= 0.0:
        raise ValueError("Detector rate initialization must be positive and finite")
    return initialization


def cached_fit_matches(
    output_dir,
    statistics_path,
    cache_path,
    third_decon_day=None,
    third_decon_initial_thickness_start_nm=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    warmup_prior_root=None,
    thickness_prior_log_sigma=None,
    thickness_prior_stages=None,
    reset_events=None,
    reset_event_initial_thickness_starts_nm=None,
):
    import numpy as np
    from wfi_relative_response import resolve_portable_path

    try:
        with np.load(Path(output_dir) / "posterior_samples.npz", allow_pickle=False) as fit:
            fit_mode = str(np.asarray(fit["fit_mode"]).reshape(-1)[0])
            expected_fit_mode = (
                "relative_response_warmup_centered_thickness_prior"
                if warmup_prior_root is not None
                else "relative_response_production"
            )
            archived_statistics = str(
                np.asarray(fit["statistics_fingerprint"]).reshape(-1)[0]
            )
            archived_cache = resolve_portable_path(
                str(np.asarray(fit["response_cache"]).reshape(-1)[0])
            )
            archived_led = bool(
                int(np.asarray(fit.get("fit_led_log_offsets", 0)).item())
            )
            archived_constant_rate_warmup = bool(
                int(np.asarray(fit.get("constant_rate_map_warmup", 0)).item())
            )
            archived_led_sigma = float(
                np.asarray(
                    fit.get(
                        "led_log_offset_prior_sigma_log",
                        relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
                    )
                ).item()
            )
            archived_third = None
            if "third_decon_day" in fit:
                value = float(np.asarray(fit["third_decon_day"]).item())
                if np.isfinite(value):
                    archived_third = value
            archived_third_start = None
            if "third_decon_initial_thickness_start_nm" in fit:
                value = float(
                    np.asarray(fit["third_decon_initial_thickness_start_nm"]).item()
                )
                if np.isfinite(value):
                    archived_third_start = value
            archived_warmup_root = None
            if "warmup_prior_root" in fit:
                archived_warmup_root = resolve_portable_path(
                    str(np.asarray(fit["warmup_prior_root"]).reshape(-1)[0])
                )
            archived_prior_sigma = None
            if "thickness_prior_log_sigma" in fit:
                value = float(np.asarray(fit["thickness_prior_log_sigma"]).item())
                if np.isfinite(value):
                    archived_prior_sigma = value
            archived_prior_stages = None
            if "thickness_prior_stages" in fit:
                archived_prior_stages = str(
                    np.asarray(fit["thickness_prior_stages"]).reshape(-1)[0]
                )
            archived_reset_days = np.asarray(
                fit.get("reset_event_days", [fit.get("ice_reset_day", np.nan)]),
                dtype=float,
            ).reshape(-1)
            archived_reset_names = np.asarray(
                fit.get("reset_event_names", ["second_decon"])
            ).astype(str).reshape(-1)
            archived_reset_starts = np.asarray(
                fit.get("reset_event_initial_thickness_start_nm", []), dtype=float
            ).reshape(-1)
            archived_reset_prior_medians = np.asarray(
                fit.get("reset_event_thickness_prior_median_nm", []), dtype=float
            ).reshape(-1)
            archived_reset_prior_sigmas = np.asarray(
                fit.get("reset_event_thickness_prior_log_sigma", []), dtype=float
            ).reshape(-1)
            archived_reset_warmup_sigmas = np.asarray(
                fit.get("reset_event_relative_response_warmup_log_sigma", []),
                dtype=float,
            ).reshape(-1)
        if warmup_prior_root is None:
            prior_matches = (
                archived_warmup_root is None
                and archived_prior_sigma is None
                and archived_prior_stages is None
            )
        else:
            prior_matches = (
                archived_warmup_root == Path(warmup_prior_root).resolve()
                and archived_prior_stages == str(thickness_prior_stages)
            )
            if thickness_prior_stages == "configured":
                expected_warmup_sigmas = np.asarray(
                    [
                        event.get("relative_response_warmup", {}).get(
                            "log_sigma", np.nan
                        )
                        for event in (reset_events or ())
                    ],
                    dtype=float,
                )
                prior_matches = prior_matches and archived_prior_sigma is None and np.allclose(
                    archived_reset_warmup_sigmas,
                    expected_warmup_sigmas,
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                )
            else:
                prior_matches = prior_matches and archived_prior_sigma is not None and np.isclose(
                    archived_prior_sigma,
                    float(thickness_prior_log_sigma),
                    rtol=0.0,
                    atol=1.0e-12,
                )
        reset_layout_matches = True
        if reset_events is not None:
            expected_data = relative_fit.production.load_tile_statistics(
                statistics_path, reset_events=reset_events
            )
            reset_layout_matches = (
                np.array_equal(
                    archived_reset_names,
                    np.asarray(expected_data["reset_event_names"]).astype(str),
                )
                and np.allclose(
                    archived_reset_days,
                    np.asarray(expected_data["reset_days"], dtype=float),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            expected_medians = np.asarray(
                [
                    event.get("thickness_prior", {}).get(
                        "median_nm", relative_fit.production.INITIAL_THICKNESS_LOG_MEDIAN_NM
                    )
                    for event in reset_events
                ],
                dtype=float,
            )
            expected_sigmas = np.asarray(
                [
                    event.get("thickness_prior", {}).get(
                        "log_sigma", relative_fit.production.INITIAL_THICKNESS_LOG_SIGMA
                    )
                    for event in reset_events
                ],
                dtype=float,
            )
            expected_warmup_sigmas = np.asarray(
                [
                    event.get("relative_response_warmup", {}).get(
                        "log_sigma", np.nan
                    )
                    for event in reset_events
                ],
                dtype=float,
            )
            reset_layout_matches = reset_layout_matches and np.allclose(
                archived_reset_prior_medians,
                expected_medians,
                rtol=0.0,
                atol=1.0e-12,
            ) and np.allclose(
                archived_reset_prior_sigmas,
                expected_sigmas,
                rtol=0.0,
                atol=1.0e-12,
            )
            reset_layout_matches = reset_layout_matches and np.allclose(
                archived_reset_warmup_sigmas,
                expected_warmup_sigmas,
                rtol=0.0,
                atol=1.0e-12,
                equal_nan=True,
            )
            expected_starts = reset_event_initial_thickness_starts_nm or {}
            if expected_starts:
                expected_values = np.asarray(
                    [expected_starts.get(name, np.nan) for name in expected_data["reset_event_names"]],
                    dtype=float,
                )
                reset_layout_matches = reset_layout_matches and np.allclose(
                    archived_reset_starts,
                    expected_values,
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                )
        return (
            fit_mode == expected_fit_mode
            and archived_statistics == relative_fit.statistics_fingerprint(statistics_path)
            and archived_cache == Path(cache_path).resolve()
            and archived_led == bool(fit_led_log_offsets)
            and archived_constant_rate_warmup
            and np.isclose(
                archived_led_sigma,
                float(led_log_offset_prior_sigma_log),
                rtol=0.0,
                atol=1.0e-12,
            )
            and (
                reset_events is not None
                or (
                    (third_decon_day is None and archived_third is None)
                    or (
                        third_decon_day is not None
                        and archived_third is not None
                        and np.isclose(
                            float(third_decon_day),
                            archived_third,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                    )
                )
            )
            and (
                third_decon_initial_thickness_start_nm is None
                or (
                    archived_third_start is not None
                    and np.isclose(
                        float(third_decon_initial_thickness_start_nm),
                        archived_third_start,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
            )
            and prior_matches
            and reset_layout_matches
            and (Path(output_dir) / "tile_rate_posterior.csv").is_file()
        )
    except (KeyError, OSError, ValueError):
        return False


def fit_tile(
    tile_id,
    statistics_path,
    cache_path,
    output_dir,
    force,
    trajectory_rate_initialization_nm_day=PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
    third_decon_day=None,
    third_decon_initial_thickness_start_nm=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    warmup_prior_root=None,
    thickness_prior_log_sigma=None,
    thickness_prior_stages=None,
    reset_events=None,
    reset_event_initial_thickness_starts_nm=None,
):
    output_dir = Path(output_dir)
    (
        warmup_prior_root,
        thickness_prior_log_sigma,
        thickness_prior_stages,
    ) = relative_fit.resolve_warmup_prior_settings(
        third_decon_day,
        warmup_prior_root,
        thickness_prior_log_sigma,
        thickness_prior_stages,
        reset_events=reset_events,
    )
    posterior_path = output_dir / "posterior_samples.npz"
    if not force and cached_fit_matches(
        output_dir,
        statistics_path,
        cache_path,
        third_decon_day=third_decon_day,
        third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
        fit_led_log_offsets=fit_led_log_offsets,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        warmup_prior_root=warmup_prior_root,
        thickness_prior_log_sigma=thickness_prior_log_sigma,
        thickness_prior_stages=thickness_prior_stages,
        reset_events=reset_events,
        reset_event_initial_thickness_starts_nm=(
            reset_event_initial_thickness_starts_nm
        ),
    ):
        print(f"Skipping tile {tile_id}: cached fit already exists", flush=True)
        return output_dir
    if posterior_path.is_file() and not force:
        print(f"Refitting tile {tile_id}: cached fit metadata does not match", flush=True)
    initial_fit, initial_statistics = cache_source_paths(cache_path)
    relative_fit.fit_one(
        statistics_path,
        cache_path,
        initial_fit,
        initial_statistics,
        output_dir,
        trajectory_rate_initialization_nm_day=trajectory_rate_initialization_nm_day,
        third_decon_day=third_decon_day,
        fit_led_log_offsets=fit_led_log_offsets,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
        warmup_prior_root=warmup_prior_root,
        thickness_prior_log_sigma=thickness_prior_log_sigma,
        thickness_prior_stages=thickness_prior_stages,
        reset_events=reset_events,
        reset_event_initial_thickness_starts_nm=(
            reset_event_initial_thickness_starts_nm
        ),
    )
    return output_dir


def _fit_tile_worker(arguments):
    return fit_tile(*arguments)


def _run_constant_rate_warmups(
    data_list,
    cache_list,
    initial_values_list,
    jax,
    jnp,
    led_log_offset_prior_sigma_log,
):
    """Warm every tile with one scalar rate before the production MAP."""
    if not data_list:
        return [], []
    evaluator = relative_fit.build_ice_only_constant_rate_evaluator(
        data_list[0],
        initial_values_list[0],
        cache_list[0],
        jax,
        jnp,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    warm_values_list = []
    warmup_diagnostics = []
    for data, cache, initial_values in zip(
        data_list, cache_list, initial_values_list
    ):
        values, diagnostics = relative_fit.fit_ice_only_constant_rate_reusable(
            data,
            cache,
            initial_values,
            evaluator,
            jax,
            jnp,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        )
        warm_values_list.append(values)
        warmup_diagnostics.append(diagnostics)
    return warm_values_list, warmup_diagnostics


def fit_tiles_batch(jobs):
    """Fit a batch of independent tiles with one vectorized JAX objective."""
    import numpy as np

    if not jobs:
        return []
    active_jobs = []
    for job in jobs:
        (
            tile_id,
            statistics_path,
            cache_path,
            output_dir,
            force,
            trajectory_rate_initialization_nm_day,
            third_decon_day,
            third_decon_initial_thickness_start_nm,
            fit_led_log_offsets,
            led_log_offset_prior_sigma_log,
            warmup_prior_root,
            thickness_prior_log_sigma,
            thickness_prior_stages,
        ) = job
        output_dir = Path(output_dir)
        if not force and cached_fit_matches(
            output_dir,
            statistics_path,
            cache_path,
            third_decon_day=third_decon_day,
            third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
            fit_led_log_offsets=fit_led_log_offsets,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
            warmup_prior_root=warmup_prior_root,
            thickness_prior_log_sigma=thickness_prior_log_sigma,
            thickness_prior_stages=thickness_prior_stages,
        ):
            print(f"Skipping tile {tile_id}: cached fit already exists", flush=True)
            continue
        if output_dir.joinpath("posterior_samples.npz").is_file() and not force:
            print(
                f"Refitting tile {tile_id}: cached fit metadata does not match",
                flush=True,
            )
        active_jobs.append(job)
    if not active_jobs:
        return []

    jax, jnp, _, _ = relative_fit.production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data_list = []
    cache_list = []
    initial_values_list = []
    contexts = []
    for (
        tile_id,
        statistics_path,
        cache_path,
        output_dir,
        force,
        trajectory_rate_initialization_nm_day,
        third_decon_day,
         third_decon_initial_thickness_start_nm,
         fit_led_log_offsets,
         led_log_offset_prior_sigma_log,
         warmup_prior_root,
         thickness_prior_log_sigma,
         thickness_prior_stages,
     ) in active_jobs:
        initial_fit, initial_statistics = cache_source_paths(cache_path)
        data = relative_fit.production.load_tile_statistics(
            statistics_path,
            third_decon_day=third_decon_day,
        )
        data["statistics_path"] = str(statistics_path)
        context = relative_fit.production.load_response_context(data["sca_names"])
        data["qe_columns"] = context["qe_columns"]
        data["response_wavelengths_nm"] = context["wavelengths"]
        data["response_wavelength_count"] = int(context["wavelengths"].size)
        cache = load_response_cache(cache_path, data, context)
        values = relative_fit.load_warm_values(
            data,
            initial_fit,
            initial_statistics,
            trajectory_rate_initialization_nm_day=trajectory_rate_initialization_nm_day,
            fit_led_log_offsets=fit_led_log_offsets,
            third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
        )
        data_list.append(data)
        cache_list.append(cache)
        initial_values_list.append(values)
        contexts.append(context)
    reference_context = contexts[0]
    if any(
        relative_fit.context_fingerprint(context)
        != relative_fit.context_fingerprint(reference_context)
        for context in contexts[1:]
    ):
        raise ValueError("Cannot batch tiles with different response contexts")

    initial_values_list, warmup_diagnostics = _run_constant_rate_warmups(
        data_list,
        cache_list,
        initial_values_list,
        jax,
        jnp,
        led_log_offset_prior_sigma_log,
    )
    values, diagnostics = relative_fit.fit_ice_only_batch(
        data_list,
        cache_list,
        initial_values_list,
        jax,
        jnp,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    diagnostics_by_tile = {
        "converged": diagnostics["converged"],
        "message": diagnostics["message"],
        "nfev": diagnostics["nfev"],
        "parameter_count": diagnostics["parameter_count"],
        "compile_seconds": diagnostics["compile_seconds"],
        "total_seconds": diagnostics["total_seconds"] / len(active_jobs),
    }
    output_dirs = []
    for index, job in enumerate(active_jobs):
        (
            tile_id,
            statistics_path,
            cache_path,
            output_dir,
            force,
            trajectory_rate_initialization_nm_day,
            third_decon_day,
         third_decon_initial_thickness_start_nm,
         fit_led_log_offsets,
         led_log_offset_prior_sigma_log,
         warmup_prior_root,
         thickness_prior_log_sigma,
         thickness_prior_stages,
     ) = job
        tile_values = dict(initial_values_list[index])
        for name in variable_sites(tile_values):
            source = np.asarray(tile_values[name])
            target_shape = source.shape
            batch_value = np.asarray(values[name])
            tile_values[name] = batch_value[index].reshape(target_shape)
        state = relative_fit.relative_state(
            data_list[index],
            tile_values,
            cache_list[index],
            jax,
            jnp,
        )
        tile_diagnostics = dict(diagnostics_by_tile)
        tile_diagnostics["initial_residual_cost"] = float(
            diagnostics["initial_residual_costs"][index]
        )
        tile_diagnostics["residual_cost"] = float(
            diagnostics["residual_costs"][index]
        )
        tile_diagnostics["trajectory_rate_initialization_nm_day"] = float(
            diagnostics["trajectory_rate_initialization_nm_day"][index]
        )
        relative_fit.write_outputs(
            output_dir,
            data_list[index],
            tile_values,
            state,
            tile_diagnostics,
            cache_path,
            jax,
            jnp,
            fit_led_log_offsets=fit_led_log_offsets,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
            third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
            constant_rate_warmup_diagnostics=warmup_diagnostics[index],
        )
        print(
            f"Wrote relative-response fit under {output_dir} "
            f"(batch of {len(active_jobs)}; {diagnostics['nfev']} evaluations)",
            flush=True,
        )
        output_dirs.append(output_dir)
    return output_dirs


def _fit_tile_batch_worker(jobs):
    return fit_tiles_batch(jobs)


def fit_tiles_reusable(jobs):
    """Fit tiles sequentially with one reusable compiled evaluator."""
    import numpy as np

    if not jobs:
        return []
    active_jobs = []
    for job in jobs:
        (
            tile_id,
            statistics_path,
            cache_path,
            output_dir,
            force,
            trajectory_rate_initialization_nm_day,
            third_decon_day,
            third_decon_initial_thickness_start_nm,
            fit_led_log_offsets,
            led_log_offset_prior_sigma_log,
            warmup_prior_root,
            thickness_prior_log_sigma,
            thickness_prior_stages,
        ) = job
        output_dir = Path(output_dir)
        if not force and cached_fit_matches(
            output_dir,
            statistics_path,
            cache_path,
            third_decon_day=third_decon_day,
            third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
            fit_led_log_offsets=fit_led_log_offsets,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
            warmup_prior_root=warmup_prior_root,
            thickness_prior_log_sigma=thickness_prior_log_sigma,
            thickness_prior_stages=thickness_prior_stages,
        ):
            print(f"Skipping tile {tile_id}: cached fit already exists", flush=True)
            continue
        if output_dir.joinpath("posterior_samples.npz").is_file() and not force:
            print(
                f"Refitting tile {tile_id}: cached fit metadata does not match",
                flush=True,
            )
        active_jobs.append(job)
    if not active_jobs:
        return []

    jax, jnp, _, _ = relative_fit.production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data_list = []
    cache_list = []
    initial_values_list = []
    contexts = []
    for (
        tile_id,
        statistics_path,
        cache_path,
        output_dir,
        force,
        trajectory_rate_initialization_nm_day,
        third_decon_day,
         third_decon_initial_thickness_start_nm,
         fit_led_log_offsets,
         led_log_offset_prior_sigma_log,
         warmup_prior_root,
         thickness_prior_log_sigma,
         thickness_prior_stages,
     ) in active_jobs:
        initial_fit, initial_statistics = cache_source_paths(cache_path)
        data = relative_fit.production.load_tile_statistics(
            statistics_path,
            third_decon_day=third_decon_day,
        )
        data["statistics_path"] = str(statistics_path)
        context = relative_fit.production.load_response_context(data["sca_names"])
        data["qe_columns"] = context["qe_columns"]
        data["response_wavelengths_nm"] = context["wavelengths"]
        data["response_wavelength_count"] = int(context["wavelengths"].size)
        cache = load_response_cache(cache_path, data, context)
        values = relative_fit.load_warm_values(
            data,
            initial_fit,
            initial_statistics,
            trajectory_rate_initialization_nm_day=trajectory_rate_initialization_nm_day,
            fit_led_log_offsets=fit_led_log_offsets,
            third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
        )
        data_list.append(data)
        cache_list.append(cache)
        initial_values_list.append(values)
        contexts.append(context)
    reference_context = contexts[0]
    if any(
        relative_fit.context_fingerprint(context)
        != relative_fit.context_fingerprint(reference_context)
        for context in contexts[1:]
    ):
        raise ValueError("Cannot reuse an evaluator across different response contexts")

    shared_names = (
        "days",
        "node_days",
        "led",
        "repeat",
        "program",
        "groups",
        "first_by_led",
        "repeat_values",
        "repeat_index",
        "exposure_times",
        "reset_day",
        "reset_days",
    )
    for data in data_list[1:]:
        for name in shared_names:
            if not np.array_equal(np.asarray(data_list[0][name]), np.asarray(data[name])):
                raise ValueError(
                    f"Cannot reuse an evaluator across different {name} layouts"
                )
    first_grid = np.asarray(cache_list[0]["grid_nm"])
    if any(not np.array_equal(first_grid, cache["grid_nm"]) for cache in cache_list[1:]):
        raise ValueError("Cannot reuse an evaluator across different cache grids")
    initial_values_list, warmup_diagnostics = _run_constant_rate_warmups(
        data_list,
        cache_list,
        initial_values_list,
        jax,
        jnp,
        led_log_offset_prior_sigma_log,
    )
    evaluator = relative_fit.build_ice_only_evaluator(
        data_list[0],
        initial_values_list[0],
        cache_list[0],
        jax,
        jnp,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    output_dirs = []
    for index, job in enumerate(active_jobs):
        (
            tile_id,
            statistics_path,
            cache_path,
            output_dir,
            force,
            trajectory_rate_initialization_nm_day,
            third_decon_day,
            third_decon_initial_thickness_start_nm,
            fit_led_log_offsets,
            led_log_offset_prior_sigma_log,
            warmup_prior_root,
            thickness_prior_log_sigma,
            thickness_prior_stages,
        ) = job
        values, diagnostics = relative_fit.fit_ice_only_reusable(
            data_list[index],
            cache_list[index],
            initial_values_list[index],
            evaluator,
            jax,
            jnp,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        )
        state = relative_fit.relative_state(
            data_list[index],
            values,
            cache_list[index],
            jax,
            jnp,
        )
        relative_fit.write_outputs(
            output_dir,
            data_list[index],
            values,
            state,
            diagnostics,
            cache_path,
            jax,
            jnp,
            fit_led_log_offsets=fit_led_log_offsets,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
            third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
            constant_rate_warmup_diagnostics=warmup_diagnostics[index],
        )
        print(
            f"Wrote relative-response fit under {output_dir} "
            f"(reused evaluator; {diagnostics['nfev']} evaluations)",
            flush=True,
        )
        output_dirs.append(output_dir)
    return output_dirs


def _fit_tile_reusable_worker(jobs):
    return fit_tiles_reusable(jobs)


def _configure_parallel_environment(workers):
    if workers <= 1:
        return
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def write_detector_rate_posterior(fit_root, output_root):
    rows = []
    fieldnames = None
    for path in sorted(Path(fit_root).glob("tile_*/tile_rate_posterior.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Inconsistent tile-rate fields in {path}")
            rows.extend(reader)
    if not rows or not fieldnames:
        return None
    output = Path(output_root) / "tile_rate_posterior.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--response-cache-root", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-id", type=int, action="append")
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=positive_integer, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=8,
        help="Independent tiles per joint JAX fit batch when --joint-batch is set.",
    )
    parser.add_argument(
        "--joint-batch",
        action="store_true",
        help="Fit each batch jointly; default reuses one evaluator across independent tiles.",
    )
    parser.add_argument(
        "--allow-external-cache-source",
        action="store_true",
        help=(
            "Reuse the source fit/statistics embedded in a pinned cache from "
            "another production run instead of requiring current-run fits."
        ),
    )
    parser.add_argument("--third-decon-day", type=float, default=None)
    parser.add_argument("--reset-config", type=Path, default=DEFAULT_RESET_CONFIG)
    parser.add_argument(
        "--third-decon-initial-thickness-start-nm",
        type=float,
        default=None,
        help="Override the post-third-decon thickness warm start in nm.",
    )
    parser.add_argument("--fit-led-log-offsets", action="store_true")
    parser.add_argument(
        "--led-log-offset-prior-sigma-log",
        type=float,
        default=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    )
    parser.add_argument(
        "--warmup-prior-root",
        type=Path,
        default=None,
        help="Constant-rate warmup directory containing fits/tile_### archives.",
    )
    parser.add_argument(
        "--thickness-prior-log-sigma",
        type=float,
        default=None,
        help=(
            "Override the warmup-centered thickness-prior log sigma for legacy "
            "stage selectors; configured uses each event's YAML value."
        ),
    )
    parser.add_argument(
        "--thickness-prior-stages",
        default=None,
        help=(
            "Warmup-centered stages: configured (event YAML), all, third-decon "
            "(legacy), or reset-event:<name>."
        ),
    )
    return parser.parse_args(argv)


def run_detector(
    detector,
    statistics_dir,
    response_cache_root,
    selected_tile_ids=None,
    tile_size=DEFAULT_TILE_SIZE,
    output_root=None,
    force=False,
    workers=DEFAULT_WORKERS,
    batch_size=8,
    joint_batch=False,
    allow_external_cache_source=False,
    third_decon_day=None,
    third_decon_initial_thickness_start_nm=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    warmup_prior_root=None,
    thickness_prior_log_sigma=None,
    thickness_prior_stages=None,
    reset_events=None,
    reset_config_sha256=None,
):
    detector = normalize_detector(detector)
    statistics_dir = Path(statistics_dir)
    statistics = corrected_statistics_path(statistics_dir, detector, tile_size)
    if not statistics.is_file():
        raise FileNotFoundError(statistics)
    reset_events = None if reset_events is None else tuple(reset_events)
    reset_event_starts = {}
    if reset_events is not None:
        configured_third_day = next(
            (
                float(event["day"])
                for event in reset_events
                if event["name"] == "third_decon"
            ),
            None,
        )
        if (
            third_decon_day is not None
            and configured_third_day is not None
            and not np.isclose(
                float(third_decon_day), configured_third_day, rtol=0.0, atol=1.0e-12
            )
        ):
            raise ValueError(
                "third_decon_day conflicts with the shared reset configuration"
            )
        if configured_third_day is not None:
            third_decon_day = configured_third_day
        warmup_events = [
            event for event in reset_events if event.get("relative_response_warmup")
        ]
        reset_event_starts = {
            event["name"]: float(event["relative_response_warmup"]["start_nm"])
            for event in warmup_events
        }
        if third_decon_initial_thickness_start_nm is not None:
            reset_event_starts["third_decon"] = float(
                third_decon_initial_thickness_start_nm
            )
        if warmup_prior_root is not None and warmup_events:
            if thickness_prior_stages is None:
                thickness_prior_stages = "configured"
    if third_decon_day is not None:
        third_decon_day = float(third_decon_day)
        if not np.isfinite(third_decon_day):
            raise ValueError("third_decon_day must be finite")
    if third_decon_initial_thickness_start_nm is not None:
        third_decon_initial_thickness_start_nm = float(
            third_decon_initial_thickness_start_nm
        )
        if (
            not np.isfinite(third_decon_initial_thickness_start_nm)
            or third_decon_initial_thickness_start_nm <= 0.0
        ):
            raise ValueError(
                "third_decon_initial_thickness_start_nm must be positive and finite"
            )
    (
        warmup_prior_root,
        thickness_prior_log_sigma,
        thickness_prior_stages,
    ) = relative_fit.resolve_warmup_prior_settings(
        third_decon_day,
        warmup_prior_root,
        thickness_prior_log_sigma,
        thickness_prior_stages,
        reset_events=reset_events,
    )
    led_log_offset_prior_sigma_log = float(led_log_offset_prior_sigma_log)
    if (
        not np.isfinite(led_log_offset_prior_sigma_log)
        or led_log_offset_prior_sigma_log <= 0.0
    ):
        raise ValueError("LED log-offset prior sigma must be positive and finite")
    if warmup_prior_root is not None:
        if batch_size != 1 or joint_batch:
            raise ValueError(
                "Warmup-centered thickness priors require independent tile mode: "
                "batch_size=1, joint_batch=False"
            )
        warmup_prior_root = Path(warmup_prior_root)
    if reset_events is not None and (batch_size != 1 or joint_batch):
        raise ValueError(
            "Shared reset-event configurations require independent batch_size=1 fits"
        )
    output_root = (
        detector_output_root(statistics_dir, detector, tile_size)
        / "relative_response"
        if output_root is None
        else Path(output_root)
    )
    statistics_root = output_root / "tile_statistics"
    fit_root = output_root / "fits"
    paths = tile_statistics_paths(detector, statistics, statistics_root, tile_size)
    complete_tile_ids(paths, label=f"{detector} statistics tile IDs")
    selected = (
        sorted(paths)
        if selected_tile_ids is None
        else sorted(set(selected_tile_ids))
    )
    missing = sorted(set(selected).difference(paths))
    if missing:
        raise ValueError(f"Unknown tile IDs for {detector}: {missing}")
    cache_root = Path(response_cache_root)
    rate_initialization_nm_day = detector_rate_initialization_nm_day(cache_root)
    print(
        f"Using {detector} detector-median rate warm start: "
        f"{rate_initialization_nm_day:.12g} nm/day",
        flush=True,
    )
    jobs = []
    production_root = detector_output_root(statistics_dir, detector, tile_size)
    source_fit_root = production_root / "fits"
    source_statistics_root = production_root / "tile_statistics"
    for tile_id in selected:
        cache_path = cache_root / f"tile_{tile_id:03d}" / "relative_response.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        if allow_external_cache_source:
            validate_response_cache_provenance(cache_path)
        else:
            validate_response_cache_provenance(
                cache_path,
                expected_fit=source_fit_root
                / f"tile_{tile_id:03d}"
                / "posterior_samples.npz",
                expected_statistics=tile_statistics_path(
                    source_statistics_root,
                    detector,
                    tile_size,
                    tile_id,
                ),
            )
        jobs.append(
            (
                tile_id,
                paths[tile_id],
                cache_path,
                fit_root / f"tile_{tile_id:03d}",
                force,
                rate_initialization_nm_day,
                third_decon_day,
                third_decon_initial_thickness_start_nm,
                bool(fit_led_log_offsets),
                led_log_offset_prior_sigma_log,
                warmup_prior_root,
                thickness_prior_log_sigma,
                thickness_prior_stages,
            ) + (
                (reset_events, reset_event_starts)
                if reset_events is not None
                else ()
            )
        )
    active_workers = min(workers, len(jobs)) if jobs else 1
    _configure_parallel_environment(active_workers)
    if batch_size <= 1 and not joint_batch:
        fit_mode_label = (
            "one-tile sequential"
            if active_workers == 1
            else "independent one-tile workers"
        )
    elif joint_batch:
        fit_mode_label = f"joint batch size {batch_size}"
    else:
        fit_mode_label = "reused evaluator"
    print(
        f"Fitting {len(jobs)} {detector} tile(s) with "
        f"{active_workers} cached-response worker(s), {fit_mode_label}",
        flush=True,
    )
    fit_root.mkdir(parents=True, exist_ok=True)
    if batch_size <= 1 and not joint_batch:
        if active_workers == 1:
            for job in jobs:
                _fit_tile_worker(job)
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=active_workers,
                mp_context=context,
            ) as executor:
                list(executor.map(_fit_tile_worker, jobs))
    elif joint_batch:
        batches = [jobs[start : start + batch_size] for start in range(0, len(jobs), batch_size)]
        if active_workers == 1:
            for batch in batches:
                _fit_tile_batch_worker(batch)
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=active_workers,
                mp_context=context,
            ) as executor:
                list(executor.map(_fit_tile_batch_worker, batches))
    else:
        chunks = [jobs[index::active_workers] for index in range(active_workers)]
        if active_workers == 1:
            _fit_tile_reusable_worker(chunks[0])
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=active_workers,
                mp_context=context,
            ) as executor:
                list(executor.map(_fit_tile_reusable_worker, chunks))
    output = write_detector_rate_posterior(fit_root, output_root)
    metadata = {
        "detector": detector,
        "fit_mode": (
            "relative_response_warmup_centered_thickness_prior"
            if warmup_prior_root is not None
            else "relative_response_production"
        ),
        "response_cache_version": int(CACHE_VERSION),
        "cache_policy": "pinned_current_run_rebuild_only_after_reprocessing_or_model_change",
        "temperature_dependent_gain_uncorrected_data_supported": False,
        "response_cache_root": portable_path_string(cache_root),
        "statistics": portable_path_string(statistics),
        "tile_size": int(tile_size),
        "batch_size": int(batch_size),
        "joint_batch": bool(joint_batch),
        "allow_external_cache_source": bool(allow_external_cache_source),
        "workers": int(workers),
        "trajectory_rate_prior_mean_nm_day": float(
            PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY
        ),
        "trajectory_rate_initialization_nm_day": float(rate_initialization_nm_day),
        "trajectory_rate_prior_sigma_log": float(
            relative_fit.production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG
        ),
        "trajectory_entropy_weight": float(PRODUCTION_ENTROPY_WEIGHT),
        "trajectory_regularization": "none",
        "constant_rate_warmup": True,
        "third_decon_day": third_decon_day,
        "third_decon_initial_thickness_start_nm": (
            None
            if third_decon_initial_thickness_start_nm is None
            else float(third_decon_initial_thickness_start_nm)
        ),
        "fit_led_log_offsets": bool(fit_led_log_offsets),
        "led_log_offset_prior_sigma_log": led_log_offset_prior_sigma_log,
        "warmup_prior_root": (
            None if warmup_prior_root is None else portable_path_string(warmup_prior_root)
        ),
        "thickness_prior_log_sigma": thickness_prior_log_sigma,
        "thickness_prior_stages": thickness_prior_stages,
        "reset_config_sha256": reset_config_sha256,
        "reset_event_names": (
            None
            if reset_events is None
            else [event["name"] for event in reset_events]
        ),
        "reset_event_days": (
            None
            if reset_events is None
            else [event.get("day") for event in reset_events]
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "relative_response_mode.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    if output is not None:
        print(f"Wrote {output}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    reset_config = load_reset_config(args.reset_config)
    reset_events = list(reset_config["events"])
    if args.third_decon_day is not None:
        third_event = next(
            (event for event in reset_events if event["name"] == "third_decon"),
            None,
        )
        if third_event is None:
            raise ValueError("--third-decon-day requires a third_decon event in the reset config")
        third_event["day"] = float(args.third_decon_day)
    run_detector(
        args.detector,
        args.statistics_dir,
        args.response_cache_root,
        selected_tile_ids=args.tile_id,
        tile_size=args.tile_size,
        output_root=args.output_root,
        force=args.force,
        workers=args.workers,
        batch_size=args.batch_size,
        joint_batch=args.joint_batch,
        allow_external_cache_source=args.allow_external_cache_source,
        third_decon_day=args.third_decon_day,
        third_decon_initial_thickness_start_nm=args.third_decon_initial_thickness_start_nm,
        fit_led_log_offsets=args.fit_led_log_offsets,
        led_log_offset_prior_sigma_log=args.led_log_offset_prior_sigma_log,
        warmup_prior_root=args.warmup_prior_root,
        thickness_prior_log_sigma=args.thickness_prior_log_sigma,
        thickness_prior_stages=args.thickness_prior_stages,
        reset_events=tuple(reset_events),
        reset_config_sha256=reset_config["sha256"],
    )


if __name__ == "__main__":
    main()
