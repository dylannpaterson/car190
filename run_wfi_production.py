"""Run the production WFI fit with one fresh EMCEE template per detector."""

import argparse
import csv
from collections import defaultdict
import hashlib
from pathlib import Path

import fit_wfi_tmm_production as production
import numpy as np
import run_wfi_tmm_emcee_warmup as emcee_warmup


ROOT = Path(__file__).resolve().parent
STATISTICS_ROOT = ROOT / "rrn_reduction_20261005"
STATISTICS_PATTERN = "{detector_lower}_tile_log_statistics_512_count_corrected.csv"
REFERENCE_DATA_DIR = ROOT / "external_reference_data"
FIT_SUFFIX = "absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares"
HGCDTE_PRIOR_SIGMA_LOG = 0.25
TILE_COUNT = 64
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))


def normalize_detector(value):
    detector = value.upper()
    if not detector.startswith("WFI") or len(detector) != 5:
        raise ValueError(f"Invalid detector name: {value}")
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {value}")
    return detector


def detector_paths(
    detector, statistics_root, output_root, statistics_pattern, emcee_steps
):
    detector_lower = detector.lower()
    statistics = statistics_root / statistics_pattern.format(
        detector=detector,
        detector_lower=detector_lower
    )
    detector_output_root = output_root / f"{detector_lower}_{FIT_SUFFIX}"
    return (
        statistics,
        detector_output_root,
        detector_output_root / "tile_statistics",
        detector_output_root / "fits",
        detector_output_root / f"one_tile_emcee{emcee_steps}",
    )


def tile_statistics_paths(detector, statistics, statistics_root, write_files=True):
    with statistics.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows_by_tile = defaultdict(list)
        for row in reader:
            rows_by_tile[int(row["tile_id"])].append(row)
    if not fieldnames or not rows_by_tile:
        raise ValueError(f"No tile statistics found in {statistics}")
    if write_files:
        statistics_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tile_id, rows in sorted(rows_by_tile.items()):
        path = (
            statistics_root
            / f"{detector.lower()}_tile_log_statistics_512_tile_{tile_id:03d}.csv"
        )
        if write_files:
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


def archive_matches_statistics(tile_fit_dir, statistics_path):
    posterior_path = tile_fit_dir / "posterior_samples.npz"
    if not posterior_path.is_file():
        return False
    try:
        with np.load(posterior_path, allow_pickle=False) as samples:
            archived = np.asarray(samples["statistics_row_fingerprints"]).astype(str)
    except (KeyError, OSError, ValueError):
        return False
    return np.array_equal(
        archived.reshape(-1), statistics_row_fingerprints(statistics_path)
    )


def fit_tile(
    tile_id,
    statistics_path,
    fit_root,
    emcee_output,
    reference_data_dir,
    map_steps,
):
    tile_fit_dir = fit_root / f"tile_{tile_id:03d}_least_squares"
    if (tile_fit_dir / "posterior_samples.npz").is_file():
        if archive_matches_statistics(tile_fit_dir, statistics_path):
            print(f"Skipping tile {tile_id}: matching archived fit exists", flush=True)
            return tile_fit_dir
        print(f"Refitting tile {tile_id}: archived statistics do not match", flush=True)

    production._PRODUCTION_STATISTICS_PATH = statistics_path
    production.SED_URL = str(
        reference_data_dir / "WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
    )
    production.QE_URL = str(reference_data_dir / "WFI_Quantum_efficiency.ecsv")
    args = production.parse_args()
    args.all_sca = False
    args.max_tiles = 1
    args.tile_statistics = statistics_path
    args.output_dir = tile_fit_dir
    args.map_init_from = emcee_output
    args.map_init_allow_tile_subset = True
    args.map_init_broadcast_tile_values = True
    args.map_init_allow_tile_template = True
    args.map_optimizer = "least-squares"
    args.map_steps = map_steps
    args.map_only = True
    args.skip_fixed_rate_warmup = True
    args.fixed_absolute_hgcdte_scales = False
    args.shared_absolute_hgcdte_scales = False
    args.absolute_hgcdte_scale_prior_sigma_log = HGCDTE_PRIOR_SIGMA_LOG
    args.shared_ar_optical_properties = False
    args.shared_ar_parameters_across_tiles = False
    args.independent_tile_ar_parameters = True
    args.tile_rate_variation = True
    args.tile_rate_trajectory_variation = True
    # The independent-tile trajectory branch is inactive for a one-tile model;
    # use its equivalent fully non-parametric node site instead.
    args.independent_tile_trajectories = False
    args.tile_rate_neighbor_smoothness = False
    args.whiten_np_trajectory = False
    args.fit_complex_q = False
    production._fit_one(args)
    return tile_fit_dir


def write_detector_rate_posterior(fit_root, output_root):
    rows = []
    fieldnames = None
    for path in sorted(fit_root.glob("tile_*_least_squares/tile_rate_posterior.csv")):
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics-dir",
        "--statistics-root",
        dest="statistics_dir",
        type=Path,
        default=STATISTICS_ROOT,
        help="Folder containing corrected per-detector tile-statistics CSVs.",
    )
    parser.add_argument(
        "--statistics-pattern",
        default=STATISTICS_PATTERN,
        help=(
            "Statistics filename pattern; supports {detector} and "
            "{detector_lower}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_dir",
        type=Path,
        default=ROOT,
        help="Parent folder for detector-specific production outputs.",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=REFERENCE_DATA_DIR,
        help="Pinned local SED and QE reference-data directory.",
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
    parser.add_argument("--emcee-walkers", type=int, default=52)
    parser.add_argument("--emcee-steps", type=int, default=1000)
    parser.add_argument("--emcee-seed", type=int, default=20261004)
    parser.add_argument("--map-steps", type=int, default=10000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate detector inputs and print output paths without fitting.",
    )
    return parser.parse_args()


def run_detector(detector, selected_tile_ids, config):
    statistics, output_root, statistics_root, fit_root, emcee_output = detector_paths(
        detector,
        config["statistics_dir"],
        config["output_dir"],
        config["statistics_pattern"],
        config["emcee_steps"],
    )
    if not statistics.is_file():
        raise FileNotFoundError(statistics)
    try:
        output_root.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "Production outputs must remain inside the repository workspace: "
            f"{output_root}"
        ) from error
    for filename in (
        "WFI_RCS_LEDP_spectral_energy_distribution.ecsv",
        "WFI_Quantum_efficiency.ecsv",
    ):
        reference_path = config["reference_data_dir"] / filename
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
    paths = tile_statistics_paths(
        detector,
        statistics,
        statistics_root,
        write_files=not config["dry_run"],
    )
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
    if config["dry_run"]:
        print(f"{detector}: statistics={statistics}; output={output_root}")
        return
    fit_root.mkdir(parents=True, exist_ok=True)
    print(f"Starting fresh one-tile EMCEE for {detector}", flush=True)
    emcee_warmup.run_emcee(
        paths[sorted(paths)[0]],
        emcee_output,
        walkers=config["emcee_walkers"],
        steps=config["emcee_steps"],
        seed=config["emcee_seed"],
        absolute_hgcdte_scale_prior_sigma_log=HGCDTE_PRIOR_SIGMA_LOG,
        fit_complex_q=False,
        reference_data_dir=config["reference_data_dir"],
    )
    for tile_id in selected:
        fit_tile(
            tile_id,
            paths[tile_id],
            fit_root,
            emcee_output,
            config["reference_data_dir"],
            config["map_steps"],
        )
    output = write_detector_rate_posterior(fit_root, output_root)
    if output is not None:
        print(f"Wrote {output}", flush=True)


def main():
    selected_args = parse_args()
    detectors = (
        [normalize_detector(value) for value in selected_args.detector]
        if selected_args.detector
        else list(DETECTORS)
    )
    config = {
        "statistics_dir": selected_args.statistics_dir,
        "statistics_pattern": selected_args.statistics_pattern,
        "output_dir": selected_args.output_dir,
        "reference_data_dir": selected_args.reference_data_dir,
        "emcee_walkers": selected_args.emcee_walkers,
        "emcee_steps": selected_args.emcee_steps,
        "emcee_seed": selected_args.emcee_seed,
        "map_steps": selected_args.map_steps,
        "dry_run": selected_args.dry_run,
    }
    for detector in detectors:
        run_detector(detector, selected_args.tile_id, config)


if __name__ == "__main__":
    main()
