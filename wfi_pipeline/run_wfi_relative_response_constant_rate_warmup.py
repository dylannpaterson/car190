"""Run cached-response constant-rate warmups without changing production fits."""

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import fit_wfi_relative_response as relative_fit
from reset_events import DEFAULT_RESET_CONFIG, load_reset_config
import wfi_relative_response as cached_response
from run_wfi_relative_response_production import (
    cache_source_paths,
    detector_rate_initialization_nm_day,
    tile_statistics_paths,
)
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    corrected_statistics_path,
    detector_output_root,
    positive_integer,
    tile_statistics_path,
)
from wfi_relative_response import (
    CACHE_VERSION,
    load_response_cache,
    validate_response_cache_provenance,
)


DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_WORKERS = 2


def normalize_detector(value):
    detector = value.upper()
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {value}")
    return detector


def _physical_value(values, name):
    value = np.asarray(values[name], dtype=float).reshape(-1)
    return float(np.exp(value[0]))


def write_warmup_outputs(
    output_dir,
    data,
    values,
    state,
    diagnostics,
    cache_path,
    third_decon_initial_thickness_start_nm,
    reset_event_initial_thickness_starts_nm=None,
    reset_config_sha256=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_id = int(np.asarray(data["tile_ids"]).reshape(-1)[0])
    detector = str(np.asarray(data["sca_names"]).reshape(-1)[0])
    metadata = data["metadata"][tile_id]
    reset_days = np.asarray(data["reset_days"], dtype=float)
    thickness_names, offset_names = relative_fit.production.reset_site_names(data)
    reset_event_starts = reset_event_initial_thickness_starts_nm or {}
    archive_values = {
        "map_only": np.asarray(1),
        "tile_ids": np.asarray(data["tile_ids"], dtype=int),
        "sca_names": np.asarray(data["sca_names"]).astype(str),
        "observation_days": np.asarray(data["days"], dtype=float),
        "observation_led": np.asarray(data["led"], dtype=int),
        "observation_repeat": np.asarray(data["repeat"], dtype=int),
        "np_log_rate_nodes": np.asarray(values["np_log_rate_nodes"]),
        "initial_thickness_log_nm": np.asarray(values["initial_thickness_log_nm"]),
        "reset_initial_thickness_log_nm": np.asarray(
            values["reset_initial_thickness_log_nm"]
        ),
        "post_reset_tile_log_offset": np.asarray(values["post_reset_tile_log_offset"]),
        "tile_ice_thickness": np.asarray(state["ice"]),
        "tile_instantaneous_rate_nm_day": np.asarray(state["rates"]),
        "np_node_thickness": np.asarray(state["node_ice"]),
        "node_instantaneous_rate_nm_day": np.asarray(state["node_rates"]),
        "response_cache": np.asarray(str(Path(cache_path).resolve())),
        "statistics_path": np.asarray(str(Path(data["statistics_path"]).resolve())),
        "statistics_fingerprint": np.asarray(
            relative_fit.statistics_fingerprint(data["statistics_path"])
        ),
        "fit_mode": np.asarray("relative_response_constant_rate_warmup"),
        "constant_rate_map_warmup": np.asarray(1),
        "constant_rate_map_rate_nm_day": np.asarray(
            diagnostics["constant_rate_map_rate_nm_day"]
        ),
        "constant_rate_map_objective": np.asarray(diagnostics["residual_cost"]),
        "map_objective_history": np.asarray(
            [diagnostics["initial_residual_cost"], diagnostics["residual_cost"]]
        ),
        "trajectory_rate_initialization_nm_day": np.asarray(
            diagnostics["trajectory_rate_initialization_nm_day"]
        ),
        "response_cache_version": np.asarray(CACHE_VERSION),
        "third_decon_day": np.asarray(
            reset_days[data["reset_event_names"].index("third_decon")]
            if "third_decon" in data["reset_event_names"]
            else np.nan
        ),
        "third_decon_initial_thickness_start_nm": np.asarray(
            np.nan
            if third_decon_initial_thickness_start_nm is None
            else third_decon_initial_thickness_start_nm
        ),
        "converged": np.asarray(int(diagnostics["converged"])),
        "nfev": np.asarray(diagnostics["nfev"]),
    }
    if reset_days.size > 1:
        archive_values["ice_reset_days"] = reset_days
        if "third_decon" in data["reset_event_names"]:
            archive_values["post_third_decon_tile_log_offset"] = np.asarray(
                values["post_third_decon_tile_log_offset"]
            )
            archive_values["third_decon_initial_thickness_log_nm"] = np.asarray(
                values["third_decon_initial_thickness_log_nm"]
            )
    archive_values.update(
        {
            "reset_event_days": reset_days,
            "reset_event_names": np.asarray(data["reset_event_names"], dtype="U64"),
            "reset_event_thickness_site_names": np.asarray(
                thickness_names[1:], dtype="U96"
            ),
            "reset_event_offset_site_names": np.asarray(offset_names, dtype="U96"),
            "reset_event_initial_thickness_log_nm": np.stack(
                [np.asarray(values[name]) for name in thickness_names[1:]], axis=0
            ),
            "reset_event_tile_log_offsets": np.stack(
                [np.asarray(values[name]) for name in offset_names], axis=0
            ),
            "reset_event_initial_thickness_start_nm": np.asarray(
                [reset_event_starts.get(name, np.nan) for name in data["reset_event_names"]],
                dtype=np.float64,
            ),
            "reset_config_sha256": np.asarray(reset_config_sha256 or ""),
        }
    )
    if "led_log_offsets" in values:
        archive_values["led_log_offsets"] = np.asarray(values["led_log_offsets"])
        archive_values["fit_led_log_offsets"] = np.asarray(1)
    np.savez_compressed(output_dir / "constant_rate_map.npz", **archive_values)
    (output_dir / "constant_rate_map.json").write_text(
        json.dumps(
            {
                "detector": detector,
                "tile_id": tile_id,
                "statistics": cached_response.portable_path_string(
                    data["statistics_path"]
                ),
                "response_cache": cached_response.portable_path_string(cache_path),
                "constant_rate": True,
                "constant_rate_nm_day": diagnostics["constant_rate_map_rate_nm_day"],
                "third_decon_initial_thickness_start_nm": (
                    None
                    if third_decon_initial_thickness_start_nm is None
                    else float(third_decon_initial_thickness_start_nm)
                ),
                "reset_event_names": list(data["reset_event_names"]),
                "reset_event_days": reset_days.tolist(),
                "reset_event_initial_thickness_starts_nm": reset_event_starts,
                "reset_config_sha256": reset_config_sha256,
                "converged": bool(diagnostics["converged"]),
                "nfev": int(diagnostics["nfev"]),
                "runtime_seconds": diagnostics["total_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary_lines = [
        "# Cached-response constant-rate warmup",
        "",
        f"- Detector: `{detector}`.",
        f"- Tile: `{tile_id}`.",
        f"- Converged: `{diagnostics['converged']}`.",
        f"- Evaluations: `{diagnostics['nfev']}`.",
        f"- Runtime: `{diagnostics['total_seconds']:.6g}` seconds.",
        f"- Constant accumulation rate: `{diagnostics['constant_rate_map_rate_nm_day']:.12g}` nm/day.",
        f"- Initial thickness: `{_physical_value(values, 'initial_thickness_log_nm'):.12g}` nm.",
        f"- Post-Decon2 initial thickness: `{_physical_value(values, 'reset_initial_thickness_log_nm'):.12g}` nm.",
    ]
    if "third_decon_initial_thickness_log_nm" in values:
        summary_lines.append(
            f"- Post-third-decon initial thickness: `{_physical_value(values, 'third_decon_initial_thickness_log_nm'):.12g}` nm."
        )
    summary_lines.extend(
        (
            f"- Third-decon initial thickness start: `{third_decon_initial_thickness_start_nm:.12g}` nm.",
            "",
        )
    )
    (output_dir / "map_summary.md").write_text("\n".join(summary_lines))
    return {
        "detector": detector,
        "tile_id": tile_id,
        "tile_row": int(metadata["tile_row"]),
        "tile_col": int(metadata["tile_col"]),
        "converged": int(diagnostics["converged"]),
        "nfev": int(diagnostics["nfev"]),
        "runtime_seconds": float(diagnostics["total_seconds"]),
        "constant_rate_nm_day": diagnostics["constant_rate_map_rate_nm_day"],
        "initial_thickness_nm": _physical_value(values, "initial_thickness_log_nm"),
        "reset_initial_thickness_nm": _physical_value(
            values, "reset_initial_thickness_log_nm"
        ),
        "third_decon_initial_thickness_nm": (
            _physical_value(values, "third_decon_initial_thickness_log_nm")
            if "third_decon_initial_thickness_log_nm" in values
            else np.nan
        ),
    }


def _load_tile_context(statistics_path, cache_path, third_decon_day=None, reset_events=None):
    initial_fit, initial_statistics = cache_source_paths(cache_path)
    data = relative_fit.production.load_tile_statistics(
        statistics_path,
        third_decon_day=(None if reset_events is not None else third_decon_day),
        reset_events=reset_events,
    )
    data["statistics_path"] = str(statistics_path)
    context = relative_fit.production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    cache = load_response_cache(cache_path, data, context)
    return data, cache, initial_fit, initial_statistics


def _fit_chunk(jobs):
    if not jobs:
        return []
    jax, jnp, _, _ = relative_fit.production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    loaded = []
    for job in jobs:
        (
            tile_id,
            statistics_path,
            cache_path,
            output_dir,
            force,
            rate_initialization_nm_day,
            third_decon_day,
            third_decon_initial_thickness_start_nm,
            fit_led_log_offsets,
            led_log_offset_prior_sigma_log,
            reset_events,
            reset_event_initial_thickness_starts_nm,
            reset_config_sha256,
        ) = job
        output_dir = Path(output_dir)
        warmup_path = output_dir / "constant_rate_map.npz"
        if not force and warmup_path.is_file():
            expected_data = relative_fit.production.load_tile_statistics(
                statistics_path,
                third_decon_day=(None if reset_events is not None else third_decon_day),
                reset_events=reset_events,
            )
            with np.load(warmup_path, allow_pickle=False) as existing:
                same = (
                    str(np.asarray(existing.get("reset_config_sha256", "")).reshape(-1)[0])
                    == str(reset_config_sha256 or "")
                    and np.array_equal(
                        np.asarray(existing.get("reset_event_names", [] )).astype(str),
                        np.asarray(expected_data["reset_event_names"]).astype(str),
                    )
                    and np.allclose(
                        np.asarray(existing.get("reset_event_days", []), dtype=float),
                        np.asarray(expected_data["reset_days"], dtype=float),
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    and np.allclose(
                        np.asarray(
                            existing.get("reset_event_initial_thickness_start_nm", []),
                            dtype=float,
                        ),
                        np.asarray(
                            [
                                reset_event_initial_thickness_starts_nm.get(name, np.nan)
                                for name in expected_data["reset_event_names"]
                            ],
                            dtype=float,
                        ),
                        rtol=0.0,
                        atol=1.0e-12,
                        equal_nan=True,
                    )
                )
            if same:
                continue
        data, cache, initial_fit, initial_statistics = _load_tile_context(
            statistics_path, cache_path, third_decon_day, reset_events
        )
        values = relative_fit.load_warm_values(
            data,
            initial_fit,
            initial_statistics,
            trajectory_rate_initialization_nm_day=rate_initialization_nm_day,
            fit_led_log_offsets=fit_led_log_offsets,
            third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
            reset_event_initial_thickness_starts_nm=(
                reset_event_initial_thickness_starts_nm
            ),
        )
        loaded.append((job, data, cache, values))
    if not loaded:
        return []
    reference_data = loaded[0][1]
    reference_cache = loaded[0][2]
    evaluator = cached_response.build_ice_only_constant_rate_evaluator(
        reference_data,
        loaded[0][3],
        reference_cache,
        jax,
        jnp,
        led_log_offset_prior_sigma_log=loaded[0][0][9],
    )
    results = []
    for job, data, cache, values in loaded:
        diagnostics_sigma = job[9]
        values, diagnostics = cached_response.fit_ice_only_constant_rate_reusable(
            data,
            cache,
            values,
            evaluator,
            jax,
            jnp,
            led_log_offset_prior_sigma_log=diagnostics_sigma,
        )
        state = relative_fit.relative_state(data, values, cache, jax, jnp)
        results.append(
            write_warmup_outputs(
                job[3],
                data,
                values,
                state,
                diagnostics,
                job[2],
                job[7],
                reset_event_initial_thickness_starts_nm=job[11],
                reset_config_sha256=job[12],
            )
        )
    return results


def run_detector(
    detector,
    statistics_dir,
    response_cache_root,
    output_root=None,
    tile_size=DEFAULT_TILE_SIZE,
    force=False,
    workers=DEFAULT_WORKERS,
    allow_external_cache_source=False,
    third_decon_day=None,
    third_decon_initial_thickness_start_nm=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
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
            raise ValueError("third_decon_day conflicts with the reset configuration")
        if configured_third_day is not None:
            third_decon_day = configured_third_day
        reset_event_starts = {
            event["name"]: float(event["relative_response_warmup"]["start_nm"])
            for event in reset_events
            if event.get("relative_response_warmup")
        }
        if third_decon_initial_thickness_start_nm is not None:
            if configured_third_day is None:
                raise ValueError(
                    "third_decon_initial_thickness_start_nm requires a third_decon event"
                )
            reset_event_starts["third_decon"] = float(
                third_decon_initial_thickness_start_nm
            )
    if reset_events is None and third_decon_day is None:
        raise ValueError("Warmup requires reset-event configuration")
    for name, value in reset_event_starts.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} initial thickness start must be positive and finite")
    third_decon_initial_thickness_start_nm = reset_event_starts.get("third_decon")
    output_root = (
        detector_output_root(statistics_dir, detector, tile_size)
        / "relative_response_constant_rate_warmup"
        if output_root is None
        else Path(output_root)
    )
    paths = tile_statistics_paths(
        detector, statistics, output_root / "tile_statistics", tile_size
    )
    complete_tile_ids(paths, label=f"{detector} statistics tile IDs")
    cache_root = Path(response_cache_root)
    rate_initialization_nm_day = detector_rate_initialization_nm_day(cache_root)
    source_root = detector_output_root(statistics_dir, detector, tile_size)
    jobs = []
    for tile_id in sorted(paths):
        cache_path = cache_root / f"tile_{tile_id:03d}" / "relative_response.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        if allow_external_cache_source:
            validate_response_cache_provenance(cache_path)
        else:
            validate_response_cache_provenance(
                cache_path,
                expected_fit=source_root / "fits" / f"tile_{tile_id:03d}" / "posterior_samples.npz",
                expected_statistics=tile_statistics_path(
                    source_root / "tile_statistics", detector, tile_size, tile_id
                ),
            )
        jobs.append(
            (
                tile_id,
                paths[tile_id],
                cache_path,
                output_root / "fits" / f"tile_{tile_id:03d}",
                force,
                rate_initialization_nm_day,
                float(third_decon_day),
                third_decon_initial_thickness_start_nm,
                bool(fit_led_log_offsets),
                float(led_log_offset_prior_sigma_log),
                reset_events,
                reset_event_starts,
                reset_config_sha256,
            )
        )
    active_workers = min(int(workers), len(jobs)) if jobs else 1
    chunks = [jobs[index::active_workers] for index in range(active_workers)]
    if active_workers == 1:
        results = [_fit_chunk(chunks[0])]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=active_workers, mp_context=context) as executor:
            results = list(executor.map(_fit_chunk, chunks))
    rows = [row for chunk in results for row in chunk]
    rows.sort(key=lambda row: int(row["tile_id"]))
    summary_path = output_root / "tile_constant_rate_warmup.csv"
    if rows:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output_root / "relative_response_constant_rate_warmup_mode.json").write_text(
        json.dumps(
            {
                "detector": detector,
                "fit_mode": "relative_response_constant_rate_warmup",
                "response_cache_root": cached_response.portable_path_string(cache_root),
                "statistics": cached_response.portable_path_string(statistics),
                "tile_size": int(tile_size),
                "workers": int(workers),
                "third_decon_day": (
                    None if third_decon_day is None else float(third_decon_day)
                ),
                "third_decon_initial_thickness_start_nm": third_decon_initial_thickness_start_nm,
                "trajectory_rate_initialization_nm_day": rate_initialization_nm_day,
                "fit_led_log_offsets": bool(fit_led_log_offsets),
                "led_log_offset_prior_sigma_log": float(led_log_offset_prior_sigma_log),
                "allow_external_cache_source": bool(allow_external_cache_source),
                "reset_config_sha256": reset_config_sha256,
                "reset_event_names": [event["name"] for event in reset_events or ()],
                "reset_event_days": [event.get("day") for event in reset_events or ()],
                "reset_event_initial_thickness_starts_nm": reset_event_starts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {len(rows)} constant-rate warmups for {detector} under {output_root}")
    if rows:
        print(f"Wrote {summary_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--response-cache-root", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=positive_integer, default=DEFAULT_WORKERS)
    parser.add_argument("--allow-external-cache-source", action="store_true")
    parser.add_argument("--third-decon-day", type=float, default=None)
    parser.add_argument("--reset-config", type=Path, default=DEFAULT_RESET_CONFIG)
    parser.add_argument(
        "--third-decon-initial-thickness-start-nm", type=float, default=None
    )
    parser.add_argument("--fit-led-log-offsets", action="store_true")
    parser.add_argument(
        "--led-log-offset-prior-sigma-log",
        type=float,
        default=relative_fit.production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    )
    return parser.parse_args(argv)


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
        output_root=args.output_root,
        tile_size=args.tile_size,
        force=args.force,
        workers=args.workers,
        allow_external_cache_source=args.allow_external_cache_source,
        third_decon_day=args.third_decon_day,
        third_decon_initial_thickness_start_nm=args.third_decon_initial_thickness_start_nm,
        fit_led_log_offsets=args.fit_led_log_offsets,
        led_log_offset_prior_sigma_log=args.led_log_offset_prior_sigma_log,
        reset_events=tuple(reset_events),
        reset_config_sha256=reset_config["sha256"],
    )


if __name__ == "__main__":
    main()
