"""Run the production WFI fit with one fresh EMCEE template per detector."""

import argparse
import csv
from collections import defaultdict, deque
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import fit_wfi_tmm_production as production
import numpy as np
import run_wfi_tmm_emcee_warmup as emcee_warmup
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    corrected_statistics_path,
    detector_output_root,
    positive_integer,
    tile_statistics_path,
)


ROOT = Path(__file__).resolve().parent
REFERENCE_DATA_DIR = ROOT / "external_reference_data"
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
PRODUCTION_MAXENT_WEIGHT = 0.0
# More than two independent JAX/ROCm MAP processes can contend for the device
# runtime and leave one worker blocked indefinitely. Keep production bounded.
MAX_MAP_WORKERS = 2
DEFAULT_MAP_WORKERS = 2
DEFAULT_TILE_TIMEOUT_SECONDS = 1800.0
PRODUCTION_MODEL_VERSION = (
    "fixed_ice_n125_normal_incidence_arhgcdte_no_penalty_no_maxent_rate12_v6"
)
PRODUCTION_FIT_ICE_N_CONSTANT = True
PRODUCTION_FIXED_ICE_N_LOG_CONSTANT = 0.0
PRODUCTION_ICE_N_CONSTANT_PRIOR_SIGMA_LOG = 0.01
PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 12.0
PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG = 0.25
PRODUCTION_THICKNESS_PENALTY_ENABLED = False
PRODUCTION_THICKNESS_PENALTY_LAMBDA_SHARED = 4.0e8
PRODUCTION_THICKNESS_PENALTY_LAMBDA_DISAGREEMENT = 1.0e8
PRODUCTION_THICKNESS_PENALTY_SMOOTH_SIGMA_NM = 1.0
PRODUCTION_THICKNESS_PENALTY_GRID_MIN_NM = 5.0
PRODUCTION_THICKNESS_PENALTY_GRID_MAX_NM = 80.0
PRODUCTION_THICKNESS_PENALTY_GRID_COUNT = 64
PRODUCTION_FIT_INCIDENCE_ANGLE = False
PRODUCTION_FIT_AR_HGCDTE_MODULATION = True
PRODUCTION_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 0.10
PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG = production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG


def production_model_version(fit_led_log_offsets=False, ice_n_porosity_parameters=None):
    suffix = "_ll_porosity" if ice_n_porosity_parameters is not None else ""
    if fit_led_log_offsets:
        return f"{PRODUCTION_MODEL_VERSION}_led_offsets{suffix}"
    return f"{PRODUCTION_MODEL_VERSION}{suffix}"


def normalize_detector(value):
    detector = value.upper()
    if not detector.startswith("WFI") or len(detector) != 5:
        raise ValueError(f"Invalid detector name: {value}")
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {value}")
    return detector


def detector_paths(
    detector,
    statistics_dir,
    output_root=None,
    tile_size=DEFAULT_TILE_SIZE,
):
    statistics_dir = Path(statistics_dir)
    statistics = corrected_statistics_path(statistics_dir, detector, tile_size)
    fit_output_root = (
        detector_output_root(statistics_dir, detector, tile_size)
        if output_root is None
        else Path(output_root)
    )
    return (
        statistics,
        fit_output_root,
        fit_output_root / "tile_statistics",
        fit_output_root / "fits",
        fit_output_root / "emcee",
    )


def tile_statistics_paths(
    detector,
    statistics,
    statistics_root,
    tile_size=DEFAULT_TILE_SIZE,
):
    with statistics.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows_by_tile = defaultdict(list)
        for row in reader:
            rows_by_tile[int(row["tile_id"])].append(row)
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


def statistics_row_fingerprints(statistics_path):
    with statistics_path.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["group"]))
    if not rows:
        raise ValueError(f"No tile statistics found in {statistics_path}")
    for field in ("n_pair_valid", "sum_log_change", "sum_squared_log_change"):
        if field not in rows[0]:
            raise ValueError(f"Missing {field} in {statistics_path}")
    digest = hashlib.sha256()
    for field in ("n_pair_valid", "sum_log_change", "sum_squared_log_change"):
        digest.update(
            np.asarray([float(row[field]) for row in rows], dtype=np.float64).tobytes()
        )
    return np.asarray([digest.hexdigest()])


def statistics_reset_day(statistics_path):
    """Return the physical reset day when corrected input records it."""
    with Path(statistics_path).open(newline="") as handle:
        rows = csv.DictReader(handle)
        reset_days = [
            float(row["days"])
            for row in rows
            if row.get("program") == "1063" and row.get("days", "")
        ]
    return min(reset_days) if reset_days else None


def archive_matches_statistics(tile_fit_dir, statistics_path, third_decon_day=None):
    posterior_path = tile_fit_dir / "posterior_samples.npz"
    if not posterior_path.is_file():
        return False
    try:
        with np.load(posterior_path, allow_pickle=False) as samples:
            if "statistics_fingerprint" in samples:
                archived = np.asarray(samples["statistics_fingerprint"]).astype(str)
                matches = np.array_equal(
                    archived.reshape(-1), statistics_row_fingerprints(statistics_path)
                )
            else:
                archived = np.asarray(samples["statistics_row_fingerprints"]).astype(str)
                matches = np.array_equal(
                    archived.reshape(-1), statistics_row_fingerprints(statistics_path)
                )
            expected_reset_day = statistics_reset_day(statistics_path)
            if expected_reset_day is not None:
                if "ice_reset_day" not in samples:
                    return False
                archived_reset_day = float(np.asarray(samples["ice_reset_day"]).item())
                matches = matches and np.isclose(
                    archived_reset_day,
                    expected_reset_day,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            third_field_names = (
                "third_decon_day",
                "third_decon_initial_thickness_log_nm",
                "post_third_decon_tile_log_offset",
            )
            third_fields_present = [name in samples for name in third_field_names]
            if any(third_fields_present) and not all(third_fields_present):
                return False
            archived_third_decon_day = None
            if "third_decon_day" in samples:
                value = float(np.asarray(samples["third_decon_day"]).item())
                if np.isfinite(value):
                    archived_third_decon_day = value
                elif all(third_fields_present):
                    return False
            if third_decon_day is None:
                matches = matches and archived_third_decon_day is None
            else:
                matches = matches and archived_third_decon_day is not None
                if archived_third_decon_day is not None:
                    matches = matches and np.isclose(
                        archived_third_decon_day,
                        float(third_decon_day),
                        rtol=0.0,
                        atol=1.0e-12,
                    )
            return matches
    except (KeyError, OSError, ValueError):
        return False


def archive_matches_production_model(
    tile_fit_dir,
    entropy_weight=PRODUCTION_MAXENT_WEIGHT,
    thickness_penalty_enabled=PRODUCTION_THICKNESS_PENALTY_ENABLED,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    ice_n_porosity_parameters=None,
):
    """Return whether an archive uses the current production configuration."""
    posterior_path = tile_fit_dir / "posterior_samples.npz"
    try:
        with np.load(posterior_path, allow_pickle=False) as samples:
            model_version = str(np.asarray(samples["production_model_version"]).item())
            regularization = str(np.asarray(samples["trajectory_regularization"]).item())
            archived_weight = float(np.asarray(samples["trajectory_entropy_weight"]).item())
            archived_penalty = bool(
                int(np.asarray(samples["thickness_penalty_enabled"]).item())
            )
            warmup = int(np.asarray(samples["constant_rate_map_warmup"]).item())
            archived_fit_led_log_offsets = bool(
                int(np.asarray(samples.get("fit_led_log_offsets", 0)).item())
            )
            archived_led_log_offset_prior_sigma_log = float(
                np.asarray(
                    samples.get(
                        "led_log_offset_prior_sigma_log",
                        PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
                    )
                ).item()
            )
        return (
            model_version
            == production_model_version(
                fit_led_log_offsets,
                ice_n_porosity_parameters=ice_n_porosity_parameters,
            )
            and regularization == ("maxent" if entropy_weight else "none")
            and np.isclose(archived_weight, entropy_weight, rtol=0.0, atol=1.0e-12)
            and archived_penalty == bool(thickness_penalty_enabled)
            and warmup == 1
            and archived_fit_led_log_offsets == bool(fit_led_log_offsets)
            and np.isclose(
                archived_led_log_offset_prior_sigma_log,
                led_log_offset_prior_sigma_log,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    except (KeyError, OSError, ValueError):
        return False


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_constant_rate_warmup(
    tile_fit_dir,
    statistics_path,
    map_values,
    diagnostics,
    evaluated,
    absolute_hgcdte_n_scale_prior_sigma_log,
    fixed_absolute_hgcdte_n_log_scale,
):
    """Persist the per-tile scalar-rate warmup without changing final fit outputs."""
    tile_fit_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tile_fit_dir / "constant_rate_map.npz",
        **{
            name: np.asarray(value)
            for name, value in {**map_values, **evaluated}.items()
        },
        objective_history=np.asarray(diagnostics["objective_history"], dtype=float),
    )
    rate = float(np.exp(np.asarray(map_values["log_rate_scale"]).reshape(-1)[0]))
    metadata = {
        "statistics": str(Path(statistics_path).resolve()),
        "constant_rate": True,
        "rate_nm_day": rate,
        "absolute_hgcdte_n_scale_prior_sigma_log": float(
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
        "absolute_hgcdte_n_log_scale_fixed": (
            fixed_absolute_hgcdte_n_log_scale is not None
        ),
        "fixed_absolute_hgcdte_n_log_scale": (
            None
            if fixed_absolute_hgcdte_n_log_scale is None
            else float(fixed_absolute_hgcdte_n_log_scale)
        ),
        "diagnostics": {
            key: _json_value(value)
            for key, value in diagnostics.items()
            if key != "objective_history"
        },
        "objective_history": _json_value(diagnostics["objective_history"]),
    }
    (tile_fit_dir / "constant_rate_map.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def fit_tile(
    tile_id,
    statistics_path,
    fit_root,
    emcee_output,
    force,
    entropy_weight=PRODUCTION_MAXENT_WEIGHT,
    ice_n_constant_reference=production.ICE_INDEX_DEFAULT,
    ice_n_constant_prior_sigma_log=PRODUCTION_ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    absolute_hgcdte_n_scale_prior_sigma_log=(
        PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG
    ),
    fixed_absolute_hgcdte_n_log_scale=None,
    thickness_penalty_enabled=PRODUCTION_THICKNESS_PENALTY_ENABLED,
    third_decon_day=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    ice_n_porosity_parameters=None,
):
    ice_n_porosity_parameters = production.validate_ice_n_porosity_parameters(
        ice_n_porosity_parameters
    )
    fit_ice_n_constant = (
        PRODUCTION_FIT_ICE_N_CONSTANT and ice_n_porosity_parameters is None
    )
    fixed_ice_n_log_constant = (
        None
        if ice_n_porosity_parameters is not None
        else PRODUCTION_FIXED_ICE_N_LOG_CONSTANT
    )
    tile_fit_dir = fit_root / f"tile_{tile_id:03d}"
    if not force and (tile_fit_dir / "posterior_samples.npz").is_file():
        if (
            archive_matches_statistics(
                tile_fit_dir, statistics_path, third_decon_day=third_decon_day
            )
            and archive_matches_production_model(
                tile_fit_dir,
                entropy_weight,
                thickness_penalty_enabled,
                fit_led_log_offsets,
                led_log_offset_prior_sigma_log,
                ice_n_porosity_parameters,
            )
        ):
            print(f"Skipping tile {tile_id}: matching archived fit exists", flush=True)
            return tile_fit_dir
        print(f"Refitting tile {tile_id}: archived statistics do not match", flush=True)
    elif force and (tile_fit_dir / "posterior_samples.npz").is_file():
        print(f"Refitting tile {tile_id}: forced production rerun", flush=True)

    warmup_config = production.ProductionConfig(
        tile_statistics=statistics_path,
        output_dir=tile_fit_dir,
        third_decon_day=third_decon_day,
        map_init_from=emcee_output,
        smoothness_weight=0.0,
        rate_prior_log_mean=np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
        trajectory_rate_prior_sigma_log=production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
        fixed_absolute_hgcdte_n_log_scale=fixed_absolute_hgcdte_n_log_scale,
        constant_rate=True,
        fit_incidence_angle=PRODUCTION_FIT_INCIDENCE_ANGLE,
        fit_ice_n_constant=fit_ice_n_constant,
        ice_n_constant_prior_sigma_log=ice_n_constant_prior_sigma_log,
        ice_n_constant_reference=ice_n_constant_reference,
        fixed_ice_n_log_constant=fixed_ice_n_log_constant,
        ice_n_porosity_parameters=ice_n_porosity_parameters,
        fit_ar_hgcdte_modulation=PRODUCTION_FIT_AR_HGCDTE_MODULATION,
        ar_hgcdte_modulation_prior_sigma_log=(
            PRODUCTION_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG
        ),
        fit_led_log_offsets=fit_led_log_offsets,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    warmup_data, _, warmup_values, warmup_diagnostics, warmup_evaluated = production.fit_map_state(
        warmup_config
    )
    warmup_metadata = write_constant_rate_warmup(
        tile_fit_dir,
        statistics_path,
        warmup_values,
        warmup_diagnostics,
        warmup_evaluated,
        absolute_hgcdte_n_scale_prior_sigma_log,
        fixed_absolute_hgcdte_n_log_scale,
    )
    refined_rate = np.asarray(warmup_values["log_rate_scale"], dtype=float)
    rate_reference = np.broadcast_to(
        refined_rate[..., None],
        (
            int(warmup_data["n_scas"]),
            int(warmup_data["n_tiles"]),
            len(warmup_data["node_days"]),
        ),
    ).copy()
    penalty_plan = None
    if thickness_penalty_enabled:
        from fit_wfi_tile136_thickness_penalty import _penalty_plan

        penalty_plan = _penalty_plan(
            warmup_data,
            [{"ice": np.asarray(warmup_evaluated["ice"])[0]}] * 3,
            PRODUCTION_THICKNESS_PENALTY_GRID_MIN_NM,
            PRODUCTION_THICKNESS_PENALTY_GRID_MAX_NM,
            PRODUCTION_THICKNESS_PENALTY_GRID_COUNT,
            PRODUCTION_THICKNESS_PENALTY_SMOOTH_SIGMA_NM,
            lock_grid=True,
        )
    rate_prior_log_mean = np.full_like(
        rate_reference,
        np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
    )
    final_initial_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in warmup_values.items()
        if name != "log_rate_scale"
    }
    final_initial_values["np_log_rate_nodes"] = rate_reference.copy()
    production.fit_one(
        production.ProductionConfig(
            tile_statistics=statistics_path,
            output_dir=tile_fit_dir,
            third_decon_day=third_decon_day,
            map_initial_values=final_initial_values,
            smoothness_weight=0.0,
            entropy_reference_log_rates=rate_reference,
            entropy_weight=entropy_weight,
            rate_prior_log_mean=rate_prior_log_mean,
            absolute_hgcdte_n_scale_prior_sigma_log=(
                absolute_hgcdte_n_scale_prior_sigma_log
            ),
            fixed_absolute_hgcdte_n_log_scale=fixed_absolute_hgcdte_n_log_scale,
            fit_incidence_angle=PRODUCTION_FIT_INCIDENCE_ANGLE,
            fit_ice_n_constant=fit_ice_n_constant,
            ice_n_constant_prior_sigma_log=ice_n_constant_prior_sigma_log,
            ice_n_constant_reference=ice_n_constant_reference,
            fixed_ice_n_log_constant=fixed_ice_n_log_constant,
            ice_n_porosity_parameters=ice_n_porosity_parameters,
            fit_ar_hgcdte_modulation=PRODUCTION_FIT_AR_HGCDTE_MODULATION,
            ar_hgcdte_modulation_prior_sigma_log=(
                PRODUCTION_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG
            ),
            fit_led_log_offsets=fit_led_log_offsets,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
            thickness_penalty_plan=penalty_plan,
            thickness_penalty_lambda_shared=(
                PRODUCTION_THICKNESS_PENALTY_LAMBDA_SHARED
                if thickness_penalty_enabled
                else 0.0
            ),
            thickness_penalty_lambda_disagreement=(
                PRODUCTION_THICKNESS_PENALTY_LAMBDA_DISAGREEMENT
                if thickness_penalty_enabled
                else 0.0
            ),
            model_metadata={
                "production_model_version": production_model_version(
                    fit_led_log_offsets,
                    ice_n_porosity_parameters=ice_n_porosity_parameters,
                ),
                "trajectory_regularization": (
                    "maxent" if entropy_weight else "none"
                ),
                "trajectory_smoothness_weight": 0.0,
                "trajectory_entropy_weight": float(entropy_weight),
                "trajectory_entropy_reference": "constant_rate_map_warmup",
                "trajectory_rate_prior_mean_nm_day": (
                    PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY
                ),
                "constant_rate_map_warmup": 1,
                "fit_incidence_angle": int(PRODUCTION_FIT_INCIDENCE_ANGLE),
                "fit_ice_n_constant": int(fit_ice_n_constant),
                "ice_n_constant_prior_sigma_log": (
                    float(ice_n_constant_prior_sigma_log)
                ),
                "ice_n_constant_reference": float(ice_n_constant_reference),
                "ice_n_fixed_log_constant": fixed_ice_n_log_constant,
                "ice_n_fixed_value": float(
                    production.ICE_INDEX_DEFAULT
                    * np.exp(fixed_ice_n_log_constant)
                )
                if fixed_ice_n_log_constant is not None
                else None,
                "ice_n_model": (
                    "lorentz_lorenz_porosity"
                    if ice_n_porosity_parameters is not None
                    else "constant"
                ),
                "ice_n_porosity_p0": (
                    None
                    if ice_n_porosity_parameters is None
                    else ice_n_porosity_parameters[0]
                ),
                "ice_n_porosity_pinf": (
                    None
                    if ice_n_porosity_parameters is None
                    else ice_n_porosity_parameters[1]
                ),
                "ice_n_porosity_length_nm": (
                    None
                    if ice_n_porosity_parameters is None
                    else ice_n_porosity_parameters[2]
                ),
                "ice_n_porosity_solid_n": (
                    None
                    if ice_n_porosity_parameters is None
                    else ice_n_porosity_parameters[3]
                ),
                "fit_ar_hgcdte_modulation": int(
                    PRODUCTION_FIT_AR_HGCDTE_MODULATION
                ),
                "ar_hgcdte_modulation_prior_sigma_log": (
                    PRODUCTION_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG
                ),
                "fit_led_log_offsets": int(fit_led_log_offsets),
                "led_log_offset_prior_sigma_log": float(
                    led_log_offset_prior_sigma_log
                ),
                "constant_rate_map_rate_nm_day": warmup_metadata["rate_nm_day"],
                "constant_rate_map_objective": warmup_diagnostics["objective"],
                "absolute_hgcdte_n_log_scale_fixed": int(
                    fixed_absolute_hgcdte_n_log_scale is not None
                ),
                "fixed_absolute_hgcdte_n_log_scale": (
                    None
                    if fixed_absolute_hgcdte_n_log_scale is None
                    else float(fixed_absolute_hgcdte_n_log_scale)
                ),
                "thickness_penalty_enabled": int(thickness_penalty_enabled),
                "thickness_penalty_lambda_shared": (
                    PRODUCTION_THICKNESS_PENALTY_LAMBDA_SHARED
                    if thickness_penalty_enabled
                    else 0.0
                ),
                "thickness_penalty_lambda_disagreement": (
                    PRODUCTION_THICKNESS_PENALTY_LAMBDA_DISAGREEMENT
                    if thickness_penalty_enabled
                    else 0.0
                ),
                "thickness_penalty_grid_lower_nm": (
                    None
                    if penalty_plan is None
                    else float(penalty_plan["grid"][0])
                ),
                "thickness_penalty_grid_upper_nm": (
                    None
                    if penalty_plan is None
                    else float(penalty_plan["grid"][-1])
                ),
                "thickness_penalty_grid_count": (
                    0 if penalty_plan is None else int(penalty_plan["grid"].size)
                ),
            },
        )
    )
    return tile_fit_dir


def _fit_tile_worker(arguments):
    return fit_tile(*arguments)


def _fit_tile_process(arguments, finished):
    """Fit one tile and bypass JAX runtime teardown after durable output."""
    try:
        _fit_tile_worker(arguments)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finished.set()
    # ROCm/JAX can keep non-daemon runtime threads alive after the fit returns.
    os._exit(0)


def _stop_tile_process(process):
    if process.is_alive():
        process.terminate()
    process.join(timeout=10.0)
    if process.is_alive():
        process.kill()
        process.join()
    process.close()


def _run_tile_batch(jobs, workers, timeout_seconds, context, process_target):
    """Run one batch and return jobs that need a retry."""
    pending = deque(enumerate(jobs))
    pending_exhausted = False
    active = {}
    retry_jobs = []
    while active or not pending_exhausted:
        while len(active) < workers:
            try:
                job_index, job = pending.popleft()
            except IndexError:
                pending_exhausted = True
                break
            finished = context.Event()
            process = context.Process(
                target=process_target,
                args=(job, finished),
            )
            try:
                process.start()
            except BaseException as error:
                process.close()
                retry_jobs.append(
                    (job, f"worker could not start: {type(error).__name__}: {error}")
                )
                continue
            active[job_index] = {
                "job": job,
                "process": process,
                "finished": finished,
                "started": time.monotonic(),
            }

        completed = []
        for job_index, entry in tuple(active.items()):
            process = entry["process"]
            if entry["finished"].is_set():
                _stop_tile_process(process)
                completed.append(job_index)
                continue
            if not process.is_alive():
                exitcode = process.exitcode
                _stop_tile_process(process)
                if exitcode != 0:
                    retry_jobs.append(
                        (
                            entry["job"],
                            f"worker exited with status {exitcode}",
                        )
                    )
                completed.append(job_index)
                continue
            elapsed = time.monotonic() - entry["started"]
            if elapsed > timeout_seconds:
                tile_id = entry["job"][0]
                _stop_tile_process(process)
                retry_jobs.append(
                    (
                        entry["job"],
                        f"exceeded the {timeout_seconds:g}-second worker timeout",
                    )
                )
                print(
                    f"Tile {tile_id} failed in parallel batch: "
                    f"{retry_jobs[-1][1]}; deferring for serial retry",
                    flush=True,
                )
                completed.append(job_index)
        for job_index in completed:
            del active[job_index]
        if active:
            time.sleep(0.25)

    return retry_jobs


def _run_tile_jobs(
    jobs,
    workers,
    timeout_seconds,
    *,
    process_target=None,
    retry_timeout_seconds=None,
):
    """Run tile fits without allowing one bad worker to abort the batch."""
    if not jobs:
        return []
    context = multiprocessing.get_context("spawn")
    process_target = _fit_tile_process if process_target is None else process_target
    retry_timeout_seconds = (
        max(timeout_seconds * 2.0, timeout_seconds)
        if retry_timeout_seconds is None
        else retry_timeout_seconds
    )
    retry_jobs = _run_tile_batch(
        jobs,
        workers,
        timeout_seconds,
        context,
        process_target,
    )
    if not retry_jobs:
        return []

    print(
        f"Retrying {len(retry_jobs)} tile(s) serially with a "
        f"{retry_timeout_seconds:g}-second timeout",
        flush=True,
    )
    final_failures = _run_tile_batch(
        [job for job, _reason in retry_jobs],
        1,
        retry_timeout_seconds,
        context,
        process_target,
    )
    for job, reason in final_failures:
        print(
            f"WARNING: tile {job[0]} failed after serial retry: {reason}",
            flush=True,
        )
    return final_failures


def _configure_parallel_environment(workers):
    # Avoid reserving most of the device before a one-tile or multi-worker fit
    # has established its actual working set.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if workers <= 1:
        return
    # Let independent ROCm workers share the device instead of oversubscribing
    # host-side numerical thread pools.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def write_detector_rate_posterior(
    fit_root,
    output_root,
    statistics_paths=None,
    entropy_weight=PRODUCTION_MAXENT_WEIGHT,
    thickness_penalty_enabled=PRODUCTION_THICKNESS_PENALTY_ENABLED,
    third_decon_day=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    ice_n_porosity_parameters=None,
):
    """Write an aggregate using only complete, current tile fits."""
    rows = []
    fieldnames = None
    output = output_root / "tile_rate_posterior.csv"
    if output.is_file():
        output.unlink()
    if statistics_paths is None:
        candidates = (
            (None, path.parent, path)
            for path in sorted(fit_root.glob("tile_*/tile_rate_posterior.csv"))
        )
    else:
        candidates = (
            (
                statistics_path,
                fit_root / f"tile_{tile_id:03d}",
                fit_root / f"tile_{tile_id:03d}" / "tile_rate_posterior.csv",
            )
            for tile_id, statistics_path in sorted(statistics_paths.items())
        )
    for statistics_path, tile_fit_dir, path in candidates:
        if statistics_path is not None and (
            not archive_matches_statistics(
                tile_fit_dir, statistics_path, third_decon_day=third_decon_day
            )
            or not archive_matches_production_model(
                tile_fit_dir,
                entropy_weight,
                thickness_penalty_enabled,
                fit_led_log_offsets,
                led_log_offset_prior_sigma_log,
                ice_n_porosity_parameters,
            )
        ):
            continue
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Inconsistent tile-rate fields in {path}")
            rows.extend(reader)
    if not rows or not fieldnames:
        return None
    output_root.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_detector_status(
    detector,
    output_root,
    statistics_paths,
    failures,
    entropy_weight,
    thickness_penalty_enabled,
    third_decon_day=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    ice_n_porosity_parameters=None,
):
    complete_tile_ids = []
    for tile_id, statistics_path in sorted(statistics_paths.items()):
        tile_fit_dir = output_root / "fits" / f"tile_{tile_id:03d}"
        if (
            archive_matches_statistics(
                tile_fit_dir, statistics_path, third_decon_day=third_decon_day
            )
            and archive_matches_production_model(
                tile_fit_dir,
                entropy_weight,
                thickness_penalty_enabled,
                fit_led_log_offsets,
                led_log_offset_prior_sigma_log,
                ice_n_porosity_parameters,
            )
            and (tile_fit_dir / "tile_rate_posterior.csv").is_file()
        ):
            complete_tile_ids.append(tile_id)
    failed_tile_ids = {int(job[0]) for job, _reason in failures}
    status = {
        "detector": detector,
        "model_version": production_model_version(
            fit_led_log_offsets,
            ice_n_porosity_parameters=ice_n_porosity_parameters,
        ),
        "fit_led_log_offsets": bool(fit_led_log_offsets),
        "led_log_offset_prior_sigma_log": float(led_log_offset_prior_sigma_log),
        "expected_tile_count": len(statistics_paths),
        "complete_tile_ids": complete_tile_ids,
        "failed_tile_ids": sorted(failed_tile_ids),
        "incomplete_tile_ids": sorted(
            set(statistics_paths).difference(complete_tile_ids)
        ),
        "complete": len(complete_tile_ids) == len(statistics_paths),
        "failures": [
            {"tile_id": int(job[0]), "reason": str(reason)}
            for job, reason in failures
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "production_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        required=True,
        help="Folder containing the corrected per-detector tile-statistics CSVs.",
    )
    parser.add_argument(
        "--detector",
        action="append",
        help="Run the selected detector; omit to run WFI01 through WFI18.",
    )
    parser.add_argument(
        "--tile-id",
        type=int,
        action="append",
        help="Run only the selected tile IDs for every selected detector.",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_integer,
        default=DEFAULT_TILE_SIZE,
        help=(
            "Tile edge length used in the corrected-statistics filename and split "
            f"files; default: {DEFAULT_TILE_SIZE}."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit selected tiles even when matching archives already exist.",
    )
    parser.add_argument(
        "--maxent-weight",
        type=float,
        default=PRODUCTION_MAXENT_WEIGHT,
        help="Relative-entropy weight for the final production MAP fit.",
    )
    parser.add_argument(
        "--ice-n-constant-reference",
        type=float,
        default=production.ICE_INDEX_DEFAULT,
        help="Reference ice refractive index for the positive constant-n prior.",
    )
    parser.add_argument(
        "--ice-n-constant-prior-sigma-log",
        type=float,
        default=PRODUCTION_ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
        help="Log sigma for the constant ice refractive-index prior.",
    )
    parser.add_argument(
        "--absolute-hgcdte-n-scale-prior-sigma-log",
        type=float,
        default=PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
        help=(
            "Log prior sigma for the independent absolute HgCdTe n scale; "
            "k retains the production prior sigma."
        ),
    )
    parser.add_argument(
        "--fixed-absolute-hgcdte-n-log-scale",
        type=float,
        default=None,
        help=(
            "Fix the absolute HgCdTe n log scale during MAP instead of fitting "
            "it; use 0 for the reference curve."
        ),
    )
    parser.add_argument(
        "--fit-led-log-offsets",
        action="store_true",
        help=(
            "Fit one time-independent post-reference log offset per LED."
        ),
    )
    parser.add_argument(
        "--led-log-offset-prior-sigma-log",
        type=float,
        default=PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
        help="Prior sigma for the experimental per-LED log offsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Isolated detector output root; requires exactly one --detector.",
    )
    parser.add_argument(
        "--reuse-emcee-from",
        type=Path,
        default=None,
        help="Reuse an existing EMCEE output directory instead of rerunning EMCEE.",
    )
    parser.add_argument(
        "--response-cache-root",
        type=Path,
        default=None,
        help=(
            "Use the six-LED fixed-optics relative-response cache instead of "
            "the full-TMM production model. Requires exactly one --detector."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=DEFAULT_MAP_WORKERS,
        help=f"Parallel tile MAP workers; default: {DEFAULT_MAP_WORKERS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=8,
        help=(
            "Cached-response tiles per joint JAX fit batch when --joint-batch is "
            "set; default: 8."
        ),
    )
    parser.add_argument(
        "--joint-batch",
        action="store_true",
        help=(
            "Fit cached-response tiles jointly in batches; by default each worker "
            "reuses one compiled evaluator across independent tiles."
        ),
    )
    parser.add_argument(
        "--allow-external-cache-source",
        action="store_true",
        help=(
            "Reuse the source fit/statistics embedded in a pinned response cache "
            "from another production run."
        ),
    )
    parser.add_argument(
        "--tile-timeout-seconds",
        type=float,
        default=DEFAULT_TILE_TIMEOUT_SECONDS,
        help=(
            "Maximum wall time for one isolated tile worker; default: "
            f"{DEFAULT_TILE_TIMEOUT_SECONDS:g}."
        ),
    )
    parser.add_argument(
        "--third-decon-day",
        type=float,
        default=None,
        help=(
            "Relative day of the authoritative physical third decon; current "
            "production value: 38.79784510416666."
        ),
    )
    parser.add_argument(
        "--warmup-prior-root",
        type=Path,
        default=None,
        help=(
            "Constant-rate warmup directory for the cached relative-response "
            "third-decon thickness prior."
        ),
    )
    parser.add_argument(
        "--thickness-prior-log-sigma",
        type=float,
        default=None,
        help=(
            "Warmup-centered thickness-prior log sigma; defaults to 0.10 when "
            "--warmup-prior-root is supplied."
        ),
    )
    parser.add_argument(
        "--thickness-prior-stages",
        choices=("all", "third-decon"),
        default=None,
        help=(
            "Warmup-centered prior stages; defaults to third-decon when "
            "--warmup-prior-root is supplied."
        ),
    )
    return parser.parse_args(argv)


def run_cached_response(selected_args, detectors):
    """Dispatch the explicit six-LED cache mode to its ice-only runner."""
    if len(detectors) != 1:
        raise ValueError("--response-cache-root requires exactly one --detector")
    if selected_args.reuse_emcee_from is not None:
        raise ValueError("--reuse-emcee-from is not valid with --response-cache-root")
    if selected_args.maxent_weight != PRODUCTION_MAXENT_WEIGHT:
        raise ValueError("--maxent-weight is not valid with --response-cache-root")
    if selected_args.ice_n_constant_reference != production.ICE_INDEX_DEFAULT:
        raise ValueError(
            "--ice-n-constant-reference is not valid with --response-cache-root"
        )
    if (
        selected_args.ice_n_constant_prior_sigma_log
        != PRODUCTION_ICE_N_CONSTANT_PRIOR_SIGMA_LOG
    ):
        raise ValueError(
            "--ice-n-constant-prior-sigma-log is not valid with --response-cache-root"
        )
    if (
        selected_args.absolute_hgcdte_n_scale_prior_sigma_log
        != PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG
    ):
        raise ValueError(
            "--absolute-hgcdte-n-scale-prior-sigma-log is not valid with "
            "--response-cache-root"
        )
    if selected_args.fixed_absolute_hgcdte_n_log_scale is not None:
        raise ValueError(
            "--fixed-absolute-hgcdte-n-log-scale is not valid with "
            "--response-cache-root"
        )
    import run_wfi_relative_response_production as relative_runner

    run_kwargs = dict(
        selected_tile_ids=selected_args.tile_id,
        tile_size=selected_args.tile_size,
        output_root=selected_args.output_root,
        force=selected_args.force,
        workers=selected_args.workers,
        batch_size=selected_args.batch_size,
        joint_batch=selected_args.joint_batch,
    )
    if selected_args.third_decon_day is not None:
        run_kwargs["third_decon_day"] = selected_args.third_decon_day
    if selected_args.allow_external_cache_source:
        run_kwargs["allow_external_cache_source"] = True
    if selected_args.fit_led_log_offsets:
        run_kwargs["fit_led_log_offsets"] = True
    if (
        selected_args.led_log_offset_prior_sigma_log
        != PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG
    ):
        run_kwargs["led_log_offset_prior_sigma_log"] = (
            selected_args.led_log_offset_prior_sigma_log
        )
    if selected_args.warmup_prior_root is not None:
        run_kwargs["warmup_prior_root"] = selected_args.warmup_prior_root
    if selected_args.thickness_prior_log_sigma is not None:
        run_kwargs["thickness_prior_log_sigma"] = (
            selected_args.thickness_prior_log_sigma
        )
    if selected_args.thickness_prior_stages is not None:
        run_kwargs["thickness_prior_stages"] = selected_args.thickness_prior_stages
    relative_runner.run_detector(
        detectors[0],
        selected_args.statistics_dir,
        selected_args.response_cache_root,
        **run_kwargs,
    )


def run_detector(
    detector,
    selected_tile_ids,
    statistics_dir,
    force,
    entropy_weight,
    ice_n_constant_reference=production.ICE_INDEX_DEFAULT,
    ice_n_constant_prior_sigma_log=PRODUCTION_ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    output_root=None,
    reuse_emcee_from=None,
    tile_size=DEFAULT_TILE_SIZE,
    workers=DEFAULT_MAP_WORKERS,
    tile_timeout_seconds=DEFAULT_TILE_TIMEOUT_SECONDS,
    absolute_hgcdte_n_scale_prior_sigma_log=(
        PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG
    ),
    fixed_absolute_hgcdte_n_log_scale=None,
    third_decon_day=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=PRODUCTION_LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    ice_n_porosity_parameters=None,
):
    ice_n_porosity_parameters = production.validate_ice_n_porosity_parameters(
        ice_n_porosity_parameters
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    if not np.isfinite(tile_timeout_seconds) or tile_timeout_seconds <= 0.0:
        raise ValueError("tile_timeout_seconds must be positive and finite")
    if (
        not np.isfinite(ice_n_constant_reference)
        or ice_n_constant_reference <= 0.0
    ):
        raise ValueError("ice_n_constant_reference must be positive and finite")
    if (
        not np.isfinite(ice_n_constant_prior_sigma_log)
        or ice_n_constant_prior_sigma_log <= 0.0
    ):
        raise ValueError("ice_n_constant_prior_sigma_log must be positive and finite")
    if (
        not np.isfinite(absolute_hgcdte_n_scale_prior_sigma_log)
        or absolute_hgcdte_n_scale_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "absolute_hgcdte_n_scale_prior_sigma_log must be positive and finite"
        )
    if fixed_absolute_hgcdte_n_log_scale is not None:
        fixed_absolute_hgcdte_n_log_scale = float(
            fixed_absolute_hgcdte_n_log_scale
        )
        if not np.isfinite(fixed_absolute_hgcdte_n_log_scale):
            raise ValueError("fixed_absolute_hgcdte_n_log_scale must be finite")
    if (
        not np.isfinite(led_log_offset_prior_sigma_log)
        or led_log_offset_prior_sigma_log <= 0.0
    ):
        raise ValueError("led_log_offset_prior_sigma_log must be positive and finite")
    if third_decon_day is not None:
        third_decon_day = float(third_decon_day)
        if not np.isfinite(third_decon_day):
            raise ValueError("third_decon_day must be finite")
    _configure_parallel_environment(min(workers, MAX_MAP_WORKERS))
    statistics, output_root, statistics_root, fit_root, emcee_output = detector_paths(
        detector,
        statistics_dir,
        output_root=output_root,
        tile_size=tile_size,
    )
    if not statistics.is_file():
        raise FileNotFoundError(statistics)
    for filename in (
        "WFI_RCS_LEDP_spectral_energy_distribution.ecsv",
        "WFI_Quantum_efficiency.ecsv",
    ):
        reference_path = REFERENCE_DATA_DIR / filename
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
    paths = tile_statistics_paths(
        detector,
        statistics,
        statistics_root,
        tile_size=tile_size,
    )
    complete_tile_ids(paths, label=f"{detector} statistics tile IDs")
    selected = (
        sorted(paths)
        if selected_tile_ids is None
        else sorted(set(selected_tile_ids))
    )
    missing = sorted(set(selected).difference(paths))
    if missing:
        raise ValueError(f"Unknown tile IDs for {detector}: {missing}")
    fit_root.mkdir(parents=True, exist_ok=True)
    for stale_name in ("tile_rate_posterior.csv", "production_status.json"):
        stale_path = output_root / stale_name
        if stale_path.is_file():
            stale_path.unlink()
    if reuse_emcee_from is None:
        print(f"Starting fresh one-tile EMCEE for {detector}", flush=True)
        emcee_warmup.run_emcee(
            paths[sorted(paths)[0]],
            emcee_output,
            trajectory="constant-rate",
            fit_incidence_angle=PRODUCTION_FIT_INCIDENCE_ANGLE,
            trajectory_rate_prior_mean_nm_day=(
                PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY
            ),
            trajectory_rate_prior_sigma_log=(
                production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG
            ),
            absolute_hgcdte_n_scale_prior_sigma_log=(
                PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG
            ),
            fixed_absolute_hgcdte_n_log_scale=fixed_absolute_hgcdte_n_log_scale,
            fit_ice_n_constant=PRODUCTION_FIT_ICE_N_CONSTANT,
            ice_n_constant_prior_sigma_log=ice_n_constant_prior_sigma_log,
            ice_n_constant_reference=ice_n_constant_reference,
            fixed_ice_n_log_constant=PRODUCTION_FIXED_ICE_N_LOG_CONSTANT,
            fit_ar_k_constant=False,
            third_decon_day=third_decon_day,
        )
    else:
        emcee_output = Path(reuse_emcee_from)
        warm_start = emcee_output / "posterior_samples.npz"
        if not warm_start.is_file():
            raise FileNotFoundError(warm_start)
        output_emcee = output_root / "emcee"
        output_emcee.mkdir(parents=True, exist_ok=True)
        for source in emcee_output.iterdir():
            destination = output_emcee / source.name
            if source.is_file() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        print(
            f"Reusing existing one-tile EMCEE warmup for {detector}: {emcee_output}",
            flush=True,
        )
    jobs = []
    for tile_id in selected:
        tile_fit_dir = fit_root / f"tile_{tile_id:03d}"
        if (
            not force
            and archive_matches_statistics(
                tile_fit_dir,
                paths[tile_id],
                third_decon_day=third_decon_day,
            )
            and archive_matches_production_model(
                tile_fit_dir,
                entropy_weight,
                PRODUCTION_THICKNESS_PENALTY_ENABLED,
                fit_led_log_offsets,
                led_log_offset_prior_sigma_log,
                ice_n_porosity_parameters,
            )
        ):
            print(f"Skipping tile {tile_id}: matching archived fit exists", flush=True)
            continue
        jobs.append(
            (
                tile_id,
                paths[tile_id],
                fit_root,
                emcee_output,
                force,
                entropy_weight,
                ice_n_constant_reference,
                ice_n_constant_prior_sigma_log,
                absolute_hgcdte_n_scale_prior_sigma_log,
                fixed_absolute_hgcdte_n_log_scale,
                PRODUCTION_THICKNESS_PENALTY_ENABLED,
                third_decon_day,
                fit_led_log_offsets,
                led_log_offset_prior_sigma_log,
                ice_n_porosity_parameters,
            )
        )
    active_workers = min(workers, MAX_MAP_WORKERS, len(jobs)) if jobs else 1
    if jobs and workers > MAX_MAP_WORKERS:
        print(
            f"Capping requested MAP workers from {workers} to "
            f"{MAX_MAP_WORKERS} for ROCm runtime stability",
            flush=True,
        )
    print(
        f"Fitting {len(jobs)} {detector} tile(s) with {active_workers} MAP worker(s)",
        flush=True,
    )
    _configure_parallel_environment(active_workers)
    failures = _run_tile_jobs(jobs, active_workers, tile_timeout_seconds)
    output = write_detector_rate_posterior(
        fit_root,
        output_root,
        statistics_paths=paths,
        entropy_weight=entropy_weight,
        thickness_penalty_enabled=PRODUCTION_THICKNESS_PENALTY_ENABLED,
        third_decon_day=third_decon_day,
        fit_led_log_offsets=fit_led_log_offsets,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        ice_n_porosity_parameters=ice_n_porosity_parameters,
    )
    if output is not None:
        print(f"Wrote {output}", flush=True)
    status_path = write_detector_status(
        detector,
        output_root,
        paths,
        failures,
        entropy_weight,
        PRODUCTION_THICKNESS_PENALTY_ENABLED,
        third_decon_day=third_decon_day,
        fit_led_log_offsets=fit_led_log_offsets,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        ice_n_porosity_parameters=ice_n_porosity_parameters,
    )
    if failures:
        print(
            f"WARNING: {detector} completed with {len(failures)} failed tile(s); "
            f"see {status_path}",
            flush=True,
        )


def main():
    selected_args = parse_args()
    if selected_args.maxent_weight < 0.0:
        raise ValueError("--maxent-weight must be non-negative")
    if (
        not np.isfinite(selected_args.ice_n_constant_reference)
        or selected_args.ice_n_constant_reference <= 0.0
    ):
        raise ValueError("--ice-n-constant-reference must be positive and finite")
    if (
        not np.isfinite(selected_args.ice_n_constant_prior_sigma_log)
        or selected_args.ice_n_constant_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "--ice-n-constant-prior-sigma-log must be positive and finite"
        )
    if (
        not np.isfinite(selected_args.led_log_offset_prior_sigma_log)
        or selected_args.led_log_offset_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "--led-log-offset-prior-sigma-log must be positive and finite"
        )
    detectors = (
        [normalize_detector(value) for value in selected_args.detector]
        if selected_args.detector
        else list(DETECTORS)
    )
    if selected_args.response_cache_root is None and (
        selected_args.warmup_prior_root is not None
        or selected_args.thickness_prior_log_sigma is not None
        or selected_args.thickness_prior_stages is not None
    ):
        raise ValueError(
            "Warmup-centered thickness-prior settings require "
            "--response-cache-root"
        )
    if selected_args.response_cache_root is not None:
        run_cached_response(selected_args, detectors)
        return
    if (
        (selected_args.output_root is not None or selected_args.reuse_emcee_from is not None)
        and len(detectors) != 1
    ):
        raise ValueError("--output-root and --reuse-emcee-from require exactly one detector")
    for detector in detectors:
        run_detector(
            detector,
            selected_args.tile_id,
            selected_args.statistics_dir,
            selected_args.force,
            selected_args.maxent_weight,
            ice_n_constant_reference=selected_args.ice_n_constant_reference,
            ice_n_constant_prior_sigma_log=(
                selected_args.ice_n_constant_prior_sigma_log
            ),
            output_root=selected_args.output_root,
            reuse_emcee_from=selected_args.reuse_emcee_from,
            tile_size=selected_args.tile_size,
            workers=selected_args.workers,
            tile_timeout_seconds=selected_args.tile_timeout_seconds,
            absolute_hgcdte_n_scale_prior_sigma_log=(
                selected_args.absolute_hgcdte_n_scale_prior_sigma_log
            ),
            fixed_absolute_hgcdte_n_log_scale=(
                selected_args.fixed_absolute_hgcdte_n_log_scale
            ),
            third_decon_day=selected_args.third_decon_day,
            fit_led_log_offsets=selected_args.fit_led_log_offsets,
            led_log_offset_prior_sigma_log=(
                selected_args.led_log_offset_prior_sigma_log
            ),
        )


if __name__ == "__main__":
    main()
