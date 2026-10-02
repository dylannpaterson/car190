"""Build spatial sufficient statistics for the WFI01 tile likelihood."""

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_wfi01_all_pixel_soc_model import (
    DATA_DIR,
    INDEX_PATH,
    ROOT,
    load_asdf_image,
    read_wfi01_index,
)


DEFAULT_OUTPUT = ROOT / "wfi01_tile_log_statistics_512.csv"
FIELDS = (
    "led",
    "repeat",
    "group",
    "file_set_name",
    "baseline_file_set_name",
    "exposure_start_time",
    "days",
    "tile_id",
    "tile_row",
    "tile_col",
    "row_start",
    "row_stop_exclusive",
    "col_start",
    "col_stop_exclusive",
    "tile_n_rows",
    "tile_n_cols",
    "n_total",
    "n_pair_valid",
    "sum_reference_counts",
    "sum_squared_reference_counts",
    "sum_current_counts",
    "sum_squared_current_counts",
    "sum_reference_current_counts",
    "min_reference_count",
    "max_reference_count",
    "min_current_count",
    "max_current_count",
    "sum_log_change",
    "sum_squared_log_change",
    "min_log_change",
    "max_log_change",
)


def tile_sufficient_statistics(
    baseline_data,
    baseline_dq,
    current_data,
    current_dq,
    tile_size,
):
    """Yield log-ratio and compact count-space statistics for each tile."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if baseline_data.shape != current_data.shape or baseline_data.shape != baseline_dq.shape:
        raise ValueError("Baseline/current image shapes must match")
    if baseline_data.shape != current_dq.shape:
        raise ValueError("Current data and DQ shapes must match")

    n_rows, n_cols = baseline_data.shape
    tile_row_count = (n_rows + tile_size - 1) // tile_size
    tile_col_count = (n_cols + tile_size - 1) // tile_size
    for tile_row in range(tile_row_count):
        row_start = tile_row * tile_size
        row_stop = min(row_start + tile_size, n_rows)
        for tile_col in range(tile_col_count):
            col_start = tile_col * tile_size
            col_stop = min(col_start + tile_size, n_cols)
            baseline = np.asarray(baseline_data[row_start:row_stop, col_start:col_stop])
            current = np.asarray(current_data[row_start:row_stop, col_start:col_stop])
            valid = (
                np.isfinite(baseline)
                & np.isfinite(current)
                & (np.asarray(baseline_dq[row_start:row_stop, col_start:col_stop]) == 0)
                & (np.asarray(current_dq[row_start:row_stop, col_start:col_stop]) == 0)
                & (baseline > 0.0)
                & (current > 0.0)
            )
            reference_values = baseline[valid].astype(np.float64)
            current_values = current[valid].astype(np.float64)
            values = np.log(current_values) - np.log(reference_values)
            if values.size:
                reference_total = float(np.sum(reference_values, dtype=np.float64))
                reference_squared = float(
                    np.sum(reference_values * reference_values, dtype=np.float64)
                )
                current_total = float(np.sum(current_values, dtype=np.float64))
                current_squared = float(
                    np.sum(current_values * current_values, dtype=np.float64)
                )
                reference_current = float(
                    np.sum(reference_values * current_values, dtype=np.float64)
                )
                reference_minimum = float(np.min(reference_values))
                reference_maximum = float(np.max(reference_values))
                current_minimum = float(np.min(current_values))
                current_maximum = float(np.max(current_values))
                minimum = float(np.min(values))
                maximum = float(np.max(values))
                total = float(np.sum(values, dtype=np.float64))
                total_squared = float(np.sum(values * values, dtype=np.float64))
            else:
                reference_total = 0.0
                reference_squared = 0.0
                current_total = 0.0
                current_squared = 0.0
                reference_current = 0.0
                reference_minimum = np.nan
                reference_maximum = np.nan
                current_minimum = np.nan
                current_maximum = np.nan
                minimum = np.nan
                maximum = np.nan
                total = 0.0
                total_squared = 0.0
            tile_id = tile_row * tile_col_count + tile_col
            yield {
                "tile_id": tile_id,
                "tile_row": tile_row,
                "tile_col": tile_col,
                "row_start": row_start,
                "row_stop_exclusive": row_stop,
                "col_start": col_start,
                "col_stop_exclusive": col_stop,
                "tile_n_rows": row_stop - row_start,
                "tile_n_cols": col_stop - col_start,
                "n_total": (row_stop - row_start) * (col_stop - col_start),
                "n_pair_valid": int(values.size),
                "sum_reference_counts": reference_total,
                "sum_squared_reference_counts": reference_squared,
                "sum_current_counts": current_total,
                "sum_squared_current_counts": current_squared,
                "sum_reference_current_counts": reference_current,
                "min_reference_count": reference_minimum,
                "max_reference_count": reference_maximum,
                "min_current_count": current_minimum,
                "max_current_count": current_maximum,
                "sum_log_change": total,
                "sum_squared_log_change": total_squared,
                "min_log_change": minimum,
                "max_log_change": maximum,
            }


def build_statistics(index_path, data_dir, output_path, tile_size):
    observations, _ = read_wfi01_index(index_path, data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for led in range(1, 7):
            led_rows = [row for row in observations if row["led"] == led]
            baseline_row = led_rows[0]
            baseline_data, baseline_dq = load_asdf_image(baseline_row["image_path"])
            for row in led_rows:
                if row["repeat"] == 0:
                    current_data, current_dq = baseline_data, baseline_dq
                else:
                    current_data, current_dq = load_asdf_image(row["image_path"])
                for tile in tile_sufficient_statistics(
                    baseline_data,
                    baseline_dq,
                    current_data,
                    current_dq,
                    tile_size,
                ):
                    writer.writerow(
                        {
                            "led": row["led"],
                            "repeat": row["repeat"],
                            "group": row["group"],
                            "file_set_name": row["file_set_name"],
                            "baseline_file_set_name": baseline_row["file_set_name"],
                            "exposure_start_time": row["exposure_start_time"],
                            "days": row["days"],
                            **tile,
                        }
                    )
                print(
                    f"LED {led}, repeat {row['repeat']:02d}: "
                    f"wrote {tile['tile_id'] + 1} tiles"
                )
                if row["repeat"] != 0:
                    del current_data, current_dq
            del baseline_data, baseline_dq
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_size < 1:
        raise ValueError("--tile-size must be positive")
    path = build_statistics(
        args.index_path,
        args.data_dir,
        args.output_path,
        args.tile_size,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
