"""Run the production WFI fit with one fresh EMCEE template per detector."""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import fit_wfi_tmm_production as production
import numpy as np
import run_wfi_tmm_emcee_warmup as emcee_warmup


ROOT = Path(__file__).resolve().parent
REFERENCE_DATA_DIR = ROOT / "external_reference_data"
PRODUCTION_DIR_NAME = "production"
TILE_COUNT = 64
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
PRODUCTION_MAXENT_WEIGHT = 0.1
DEFAULT_MAP_WORKERS = 4


def normalize_detector(value):
    detector = value.upper()
    if not detector.startswith("WFI") or len(detector) != 5:
        raise ValueError(f"Invalid detector name: {value}")
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {value}")
    return detector


def detector_paths(detector, statistics_dir, output_root=None):
    detector_lower = detector.lower()
    statistics_dir = Path(statistics_dir)
    statistics = statistics_dir / (
        f"{detector_lower}_tile_log_statistics_512_count_corrected.csv"
    )
    detector_output_root = (
        statistics_dir / PRODUCTION_DIR_NAME / detector_lower
        if output_root is None
        else Path(output_root)
    )
    return (
        statistics,
        detector_output_root,
        detector_output_root / "tile_statistics",
        detector_output_root / "fits",
        detector_output_root / "emcee",
    )


def tile_statistics_paths(detector, statistics, statistics_root):
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
        path = (
            statistics_root
            / f"{detector.lower()}_tile_log_statistics_512_tile_{tile_id:03d}.csv"
        )
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


def archive_matches_statistics(tile_fit_dir, statistics_path):
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
            return matches
    except (KeyError, OSError, ValueError):
        return False


def archive_matches_production_model(tile_fit_dir, entropy_weight=PRODUCTION_MAXENT_WEIGHT):
    """Return whether an archive uses the current two-stage MaxEnt production fit."""
    posterior_path = tile_fit_dir / "posterior_samples.npz"
    try:
        with np.load(posterior_path, allow_pickle=False) as samples:
            regularization = str(np.asarray(samples["trajectory_regularization"]).item())
            archived_weight = float(np.asarray(samples["trajectory_entropy_weight"]).item())
            warmup = int(np.asarray(samples["constant_rate_map_warmup"]).item())
        return (
            regularization == "maxent"
            and np.isclose(archived_weight, entropy_weight, rtol=0.0, atol=1.0e-12)
            and warmup == 1
        )
    except (KeyError, OSError, ValueError):
        return False


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_constant_rate_warmup(tile_fit_dir, statistics_path, map_values, diagnostics, evaluated):
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
):
    tile_fit_dir = fit_root / f"tile_{tile_id:03d}"
    if not force and (tile_fit_dir / "posterior_samples.npz").is_file():
        if (
            archive_matches_statistics(tile_fit_dir, statistics_path)
            and archive_matches_production_model(tile_fit_dir, entropy_weight)
        ):
            print(f"Skipping tile {tile_id}: matching archived fit exists", flush=True)
            return tile_fit_dir
        print(f"Refitting tile {tile_id}: archived statistics do not match", flush=True)
    elif force and (tile_fit_dir / "posterior_samples.npz").is_file():
        print(f"Refitting tile {tile_id}: forced production rerun", flush=True)

    warmup_config = production.ProductionConfig(
        tile_statistics=statistics_path,
        output_dir=tile_fit_dir,
        map_init_from=emcee_output,
        smoothness_weight=0.0,
        constant_rate=True,
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
            map_initial_values=final_initial_values,
            smoothness_weight=0.0,
            entropy_reference_log_rates=rate_reference,
            entropy_weight=entropy_weight,
            rate_prior_log_mean=rate_reference,
            model_metadata={
                "trajectory_regularization": "maxent",
                "trajectory_smoothness_weight": 0.0,
                "trajectory_entropy_weight": float(entropy_weight),
                "trajectory_entropy_reference": "constant_rate_map_warmup",
                "constant_rate_map_warmup": 1,
                "constant_rate_map_rate_nm_day": warmup_metadata["rate_nm_day"],
                "constant_rate_map_objective": warmup_diagnostics["objective"],
            },
        )
    )
    return tile_fit_dir


def _fit_tile_worker(arguments):
    return fit_tile(*arguments)


def _configure_parallel_environment(workers):
    if workers <= 1:
        return
    # Let independent ROCm workers share the device instead of preallocating it.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def write_detector_rate_posterior(fit_root, output_root):
    rows = []
    fieldnames = None
    for path in sorted(fit_root.glob("tile_*/tile_rate_posterior.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Inconsistent tile-rate fields in {path}")
            rows.extend(reader)
    if not rows or not fieldnames:
        return None
    output = output_root / "tile_rate_posterior.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


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
        "--workers",
        type=int,
        default=DEFAULT_MAP_WORKERS,
        help=f"Parallel tile MAP workers; default: {DEFAULT_MAP_WORKERS}.",
    )
    return parser.parse_args(argv)


def run_detector(
    detector,
    selected_tile_ids,
    statistics_dir,
    force,
    entropy_weight,
    output_root=None,
    reuse_emcee_from=None,
    workers=DEFAULT_MAP_WORKERS,
):
    if workers < 1:
        raise ValueError("workers must be positive")
    statistics, output_root, statistics_root, fit_root, emcee_output = detector_paths(
        detector,
        statistics_dir,
        output_root=output_root,
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
    paths = tile_statistics_paths(detector, statistics, statistics_root)
    expected_tile_ids = set(range(TILE_COUNT))
    actual_tile_ids = set(paths)
    if actual_tile_ids != expected_tile_ids:
        raise ValueError(
            f"Expected tile IDs 0-{TILE_COUNT - 1} in {statistics}; "
            f"found {sorted(actual_tile_ids)}"
        )
    selected = (
        sorted(paths)
        if selected_tile_ids is None
        else sorted(set(selected_tile_ids))
    )
    missing = sorted(set(selected).difference(paths))
    if missing:
        raise ValueError(f"Unknown tile IDs for {detector}: {missing}")
    fit_root.mkdir(parents=True, exist_ok=True)
    if reuse_emcee_from is None:
        print(f"Starting fresh one-tile EMCEE for {detector}", flush=True)
        emcee_warmup.run_emcee(paths[sorted(paths)[0]], emcee_output)
    else:
        emcee_output = Path(reuse_emcee_from)
        warm_start = emcee_output / "posterior_samples.npz"
        if not warm_start.is_file():
            raise FileNotFoundError(warm_start)
        output_emcee = output_root / "emcee"
        output_emcee.mkdir(parents=True, exist_ok=True)
        for source in emcee_output.iterdir():
            if source.is_file():
                shutil.copy2(source, output_emcee / source.name)
        print(
            f"Reusing existing one-tile EMCEE warmup for {detector}: {emcee_output}",
            flush=True,
        )
    jobs = [
        (
            tile_id,
            paths[tile_id],
            fit_root,
            emcee_output,
            force,
            entropy_weight,
        )
        for tile_id in selected
    ]
    active_workers = min(workers, len(jobs)) if jobs else 1
    print(
        f"Fitting {len(jobs)} {detector} tile(s) with {active_workers} MAP worker(s)",
        flush=True,
    )
    _configure_parallel_environment(active_workers)
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
    output = write_detector_rate_posterior(fit_root, output_root)
    if output is not None:
        print(f"Wrote {output}", flush=True)


def main():
    selected_args = parse_args()
    if selected_args.maxent_weight < 0.0:
        raise ValueError("--maxent-weight must be non-negative")
    detectors = (
        [normalize_detector(value) for value in selected_args.detector]
        if selected_args.detector
        else list(DETECTORS)
    )
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
            output_root=selected_args.output_root,
            reuse_emcee_from=selected_args.reuse_emcee_from,
            workers=selected_args.workers,
        )


if __name__ == "__main__":
    main()
