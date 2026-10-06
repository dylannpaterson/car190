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
REFERENCE_DATA_DIR = ROOT / "external_reference_data"
PRODUCTION_DIR_NAME = "production"
TILE_COUNT = 64
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))


def normalize_detector(value):
    detector = value.upper()
    if not detector.startswith("WFI") or len(detector) != 5:
        raise ValueError(f"Invalid detector name: {value}")
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector name: {value}")
    return detector


def detector_paths(detector, statistics_dir):
    detector_lower = detector.lower()
    statistics_dir = Path(statistics_dir)
    statistics = statistics_dir / (
        f"{detector_lower}_tile_log_statistics_512_count_corrected.csv"
    )
    detector_output_root = statistics_dir / PRODUCTION_DIR_NAME / detector_lower
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


def archive_matches_statistics(tile_fit_dir, statistics_path):
    posterior_path = tile_fit_dir / "posterior_samples.npz"
    if not posterior_path.is_file():
        return False
    try:
        with np.load(posterior_path, allow_pickle=False) as samples:
            if "statistics_fingerprint" in samples:
                archived = np.asarray(samples["statistics_fingerprint"]).astype(str)
                return np.array_equal(
                    archived.reshape(-1), statistics_row_fingerprints(statistics_path)
                )
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
    force,
):
    tile_fit_dir = fit_root / f"tile_{tile_id:03d}"
    if not force and (tile_fit_dir / "posterior_samples.npz").is_file():
        if archive_matches_statistics(tile_fit_dir, statistics_path):
            print(f"Skipping tile {tile_id}: matching archived fit exists", flush=True)
            return tile_fit_dir
        print(f"Refitting tile {tile_id}: archived statistics do not match", flush=True)
    elif force and (tile_fit_dir / "posterior_samples.npz").is_file():
        print(f"Refitting tile {tile_id}: forced production rerun", flush=True)

    production.fit_one(
        production.ProductionConfig(
            tile_statistics=statistics_path,
            output_dir=tile_fit_dir,
            map_init_from=emcee_output,
        )
    )
    return tile_fit_dir


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


def parse_args():
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
    return parser.parse_args()


def run_detector(detector, selected_tile_ids, statistics_dir, force):
    statistics, output_root, statistics_root, fit_root, emcee_output = detector_paths(
        detector, statistics_dir
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
    print(f"Starting fresh one-tile EMCEE for {detector}", flush=True)
    emcee_warmup.run_emcee(paths[sorted(paths)[0]], emcee_output)
    for tile_id in selected:
        fit_tile(
            tile_id,
            paths[tile_id],
            fit_root,
            emcee_output,
            force,
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
    for detector in detectors:
        run_detector(
            detector,
            selected_args.tile_id,
            selected_args.statistics_dir,
            selected_args.force,
        )


if __name__ == "__main__":
    main()
