"""Run independent no-q WFI01 fits for every tile."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import fit_wfi_tmm_production as production
import run_wfi_tmm_emcee_warmup as emcee_warmup


ROOT = Path(__file__).resolve().parent
STATISTICS = (
    ROOT / "rrn_reduction_20261005/wfi01_tile_log_statistics_512_count_corrected.csv"
)
REFERENCE_DATA_DIR = ROOT / "external_reference_data"
OUTPUT_ROOT = (
    ROOT / "wfi01_absolute_substrate_power_hgcdte_prior025_independent_tiles"
)
STATISTICS_ROOT = OUTPUT_ROOT / "tile_statistics"
FIT_ROOT = OUTPUT_ROOT / "fits"
EXISTING_TILE_ZERO_OUTPUT = (
    ROOT
    / "wfi01_absolute_substrate_power_hgcdte_prior025_one_tile_map10000_emcee1000_warmstarted"
)
HGCDTE_PRIOR_SIGMA_LOG = 0.25


def tile_statistics_paths():
    with STATISTICS.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows_by_tile = defaultdict(list)
        for row in reader:
            rows_by_tile[int(row["tile_id"])].append(row)
    if not fieldnames or not rows_by_tile:
        raise ValueError(f"No tile statistics found in {STATISTICS}")
    STATISTICS_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tile_id, rows in sorted(rows_by_tile.items()):
        path = STATISTICS_ROOT / (
            f"wfi01_tile_log_statistics_512_tile_{tile_id:03d}.csv"
        )
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        paths[tile_id] = path
    return paths


def fit_tile(tile_id, statistics_path):
    tile_fit_dir = FIT_ROOT / f"tile_{tile_id:03d}_map10000_emcee1000_warmstarted"
    tile_emcee_dir = FIT_ROOT / f"tile_{tile_id:03d}_emcee1000"
    if (tile_fit_dir / "posterior_samples.npz").is_file():
        print(f"Skipping tile {tile_id}: archived fit already exists", flush=True)
        return tile_fit_dir

    emcee_warmup.run_emcee(
        statistics_path,
        tile_emcee_dir,
        walkers=52,
        steps=1000,
        seed=20261004 + tile_id,
        absolute_hgcdte_scale_prior_sigma_log=HGCDTE_PRIOR_SIGMA_LOG,
        fit_complex_q=False,
        reference_data_dir=REFERENCE_DATA_DIR,
    )
    production._PRODUCTION_STATISTICS_PATH = statistics_path
    production.SED_URL = str(
        REFERENCE_DATA_DIR / "WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
    )
    production.QE_URL = str(REFERENCE_DATA_DIR / "WFI_Quantum_efficiency.ecsv")
    args = production.parse_args()
    args.all_sca = False
    args.max_tiles = 1
    args.tile_statistics = statistics_path
    args.output_dir = tile_fit_dir
    args.map_init_from = tile_emcee_dir
    args.map_steps = 10000
    args.map_only = True
    args.map_init_allow_tile_subset = False
    args.map_init_broadcast_tile_values = False
    args.fixed_absolute_hgcdte_scales = False
    args.shared_absolute_hgcdte_scales = True
    args.absolute_hgcdte_scale_prior_sigma_log = HGCDTE_PRIOR_SIGMA_LOG
    args.fit_complex_q = False
    try:
        production._fit_one(args)
    except ValueError as error:
        if "non-finite objective" not in str(error):
            raise
        print(
            f"Tile {tile_id}: retrying MAP without the auxiliary fixed-rate warmup",
            flush=True,
        )
        saved_warmup_steps = production.WARMUP_MAP_STEPS
        production.WARMUP_MAP_STEPS = 0
        try:
            production._fit_one(args)
        finally:
            production.WARMUP_MAP_STEPS = saved_warmup_steps
    return tile_fit_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tile-id",
        type=int,
        action="append",
        help="Run only the selected tile IDs; omit to run every tile.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = tile_statistics_paths()
    selected = sorted(paths) if args.tile_id is None else sorted(set(args.tile_id))
    missing = sorted(set(selected).difference(paths))
    if missing:
        raise ValueError(f"Unknown tile IDs: {missing}")
    FIT_ROOT.mkdir(parents=True, exist_ok=True)
    for tile_id in selected:
        if tile_id == 0 and (EXISTING_TILE_ZERO_OUTPUT / "posterior_samples.npz").is_file():
            print(
                f"Using existing independent tile 0 fit: {EXISTING_TILE_ZERO_OUTPUT}",
                flush=True,
            )
            continue
        fit_tile(tile_id, paths[tile_id])


if __name__ == "__main__":
    main()
