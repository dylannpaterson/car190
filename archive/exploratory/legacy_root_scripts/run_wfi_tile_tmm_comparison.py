"""Run the diagnostic TMM extrapolation on representative WFI tiles."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIAGNOSTIC = ROOT / "fit_wfi01_clean_program1039_tmm.py"
TELEMETRY = ROOT / (
    "20261011/reduction_stage_comparison/"
    "clean_program1039_tmm_wfi01_tile000/telemetry/wfi_mce_adc_wtsm00.csv"
)
RESET_DAY = 12.907397928240743
DEFAULT_TILES = (0, 32, 63)


def input_paths(detector, tile_id):
    detector = detector.lower()
    statistics_root = ROOT / "20261011/reduction_stage_comparison/inputs" / detector
    production_root = ROOT / "20261011/corrected_statistics_program1063_4h/production"
    filename = f"{detector}_tile_log_statistics_512_tile_{tile_id:03d}.csv"
    return {
        "statistics": statistics_root / "scaled_full" / filename,
        "warm_statistics": statistics_root / "scaled_full_4h" / filename,
        "corrections": ROOT / f"20261011/corrections/{detector}_count_space_unified_temperature_corrections.csv",
        "warm_start": production_root / detector / "fits" / f"tile_{tile_id:03d}" / "posterior_samples.npz",
    }


def run_tile(detector, tile_id, output_root):
    paths = input_paths(detector, tile_id)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing diagnostic input(s): " + ", ".join(missing))
    output_dir = output_root / detector.lower() / f"tile_{tile_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    command = [
        sys.executable,
        str(DIAGNOSTIC),
        "--statistics",
        str(paths["statistics"]),
        "--corrections",
        str(paths["corrections"]),
        "--warm-start",
        str(paths["warm_start"]),
        "--warm-statistics",
        str(paths["warm_statistics"]),
        "--output-dir",
        str(output_dir),
        "--mpa-temperature-telemetry",
        str(TELEMETRY),
        "--tile-id",
        str(tile_id),
        "--reset-day",
        str(RESET_DAY),
    ]
    with log_path.open("w") as log_handle:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    with (output_dir / "metadata.json").open() as handle:
        return output_dir, json.load(handle)


def summary_row(detector, tile_id, metadata):
    extrapolation = metadata["program_1063_comparison"]
    assessment = metadata["correction_assessment"]
    return {
        "detector": detector,
        "tile_id": tile_id,
        "tile_row": tile_id // 8,
        "tile_col": tile_id % 8,
        "selected_program1039_rows": metadata["selected_rows"],
        "excluded_program1039_temperature_rows": metadata[
            "excluded_temperature_rows"
        ],
        "map_objective": metadata["map_objective"],
        "map_converged": metadata["map_converged"],
        "program1063_rows": extrapolation["comparison_rows"],
        "program1063_temperature_rows": extrapolation[
            "rows_with_existing_temperature_correction"
        ],
        "program1063_beyond_program1039_fit_time_rows": extrapolation[
            "rows_beyond_program1039_fit_time"
        ],
        "program1063_inferred_minus_existing_bias_mlog": extrapolation[
            "inferred_vs_existing_bias_mlog"
        ],
        "program1063_inferred_minus_existing_rmse_mlog": extrapolation[
            "inferred_vs_existing_rmse_mlog"
        ],
        "program1063_inferred_vs_existing_correlation": extrapolation[
            "inferred_vs_existing_correlation"
        ],
        "program1039_inferred_minus_existing_bias_mlog": assessment[
            "inferred_vs_existing_bias_mlog"
        ],
        "program1039_inferred_minus_existing_rmse_mlog": assessment[
            "inferred_vs_existing_rmse_mlog"
        ],
        "program1039_inferred_vs_existing_correlation": assessment[
            "inferred_vs_existing_correlation"
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=("WFI01", "WFI02"),
        default=["WFI01", "WFI02"],
    )
    parser.add_argument("--tiles", nargs="+", type=int, default=DEFAULT_TILES)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "20261011/reduction_stage_comparison/clean_program1039_tmm_tile_comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for detector in args.detectors:
        for tile_id in args.tiles:
            print(f"Running {detector} tile {tile_id:03d}", flush=True)
            output_dir, metadata = run_tile(detector, tile_id, args.output_root)
            row = summary_row(detector, tile_id, metadata)
            row["output_dir"] = str(output_dir)
            rows.append(row)
            print(
                f"  P1063 bias={row['program1063_inferred_minus_existing_bias_mlog']:+.4f} mlog, "
                f"RMSE={row['program1063_inferred_minus_existing_rmse_mlog']:.4f} mlog, "
                f"correlation={row['program1063_inferred_vs_existing_correlation']:.6f}",
                flush=True,
            )
    summary_path = args.output_root / "tile_comparison_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
