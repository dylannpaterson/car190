"""Write count-corrected tile statistics without downloading images."""

import argparse
import csv
from pathlib import Path

import numpy as np

from fit_wfi_tmm_numpyro import apply_fixed_corrections, load_tile_statistics


def write_corrected_statistics(
    statistics_path,
    corrections_path,
    output_path,
    additional_acm_step_repeat=None,
    pre_reset_acm_step_repeat=None,
):
    data = load_tile_statistics(
        statistics_path,
        additional_acm_step_repeat=additional_acm_step_repeat,
        pre_reset_acm_step_repeat=pre_reset_acm_step_repeat,
    )
    corrected = apply_fixed_corrections(data, corrections_path)
    if corrected.get("fixed_correction_space") != "counts":
        raise ValueError("The correction table must be marked as count-space")
    with Path(statistics_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    tile_indices = {
        int(tile_id): index for index, tile_id in enumerate(data["tile_ids"])
    }
    group_indices = {
        int(group): index for index, group in enumerate(data["groups"])
    }
    count_fields = (
        "sum_reference_counts",
        "sum_squared_reference_counts",
        "sum_current_counts",
        "sum_squared_current_counts",
        "sum_reference_current_counts",
        "min_reference_count",
        "max_reference_count",
        "min_current_count",
        "max_current_count",
    )
    for row in rows:
        tile_index = tile_indices[int(row["tile_id"])]
        group_index = group_indices[int(row["group"])]
        delta = float(corrected["fixed_total_log_correction"][tile_index, group_index])
        row["sum_log_change"] = repr(float(corrected["sums"][tile_index, group_index]))
        row["sum_squared_log_change"] = repr(
            float(corrected["sum_squared"][tile_index, group_index])
        )
        row["min_log_change"] = repr(float(row["min_log_change"]) + delta)
        row["max_log_change"] = repr(float(row["max_log_change"]) + delta)
        for field in count_fields:
            row[field] = repr(float(corrected[field][tile_index, group_index]))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=None)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = write_corrected_statistics(
        args.statistics,
        args.corrections,
        args.output,
        args.additional_acm_step_repeat,
        args.pre_reset_acm_step_repeat,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
