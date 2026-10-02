"""Compare two WFI tile-statistics CSV files keyed by product and tile."""

import argparse
import csv
import math
from pathlib import Path


COUNT_FIELDS = (
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


def read_old_rows(path):
    rows = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["file_set_name"], int(row["tile_id"]))
            if key in rows:
                raise ValueError(f"Duplicate comparison key in {path}: {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def mean_and_rms(row):
    count = float(row["n_pair_valid"])
    if count <= 0.0:
        return math.nan, math.nan
    mean = float(row["sum_log_change"]) / count
    variance = max(float(row["sum_squared_log_change"]) / count - mean * mean, 0.0)
    return mean, math.sqrt(variance)


def count_means(row):
    count = float(row["n_pair_valid"])
    if count <= 0.0 or not set(COUNT_FIELDS).issubset(row):
        return math.nan, math.nan
    return (
        float(row["sum_reference_counts"]) / count,
        float(row["sum_current_counts"]) / count,
    )


def compare(old_path, new_path, output_path=None, common_new_output=None):
    old_rows = read_old_rows(old_path)
    matched = set()
    abs_mean_deltas = []
    abs_rms_deltas = []
    abs_reference_count_deltas = []
    abs_current_count_deltas = []
    valid_deltas = []
    count_statistics_compared = False
    output_handle = None
    writer = None
    common_output_handle = None
    common_writer = None
    try:
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", newline="")
            writer = csv.DictWriter(
                output_handle,
                fieldnames=(
                    "file_set_name",
                    "tile_id",
                    "old_n_pair_valid",
                    "new_n_pair_valid",
                    "delta_n_pair_valid",
                    "old_mean_log_change",
                    "new_mean_log_change",
                    "delta_mean_log_change",
                    "old_rms_log_change",
                    "new_rms_log_change",
                    "delta_rms_log_change",
                    "delta_sum_log_change",
                    "delta_sum_squared_log_change",
                    "old_reference_mean_counts",
                    "new_reference_mean_counts",
                    "delta_reference_mean_counts",
                    "old_current_mean_counts",
                    "new_current_mean_counts",
                    "delta_current_mean_counts",
                ),
            )
            writer.writeheader()

        new_rows = 0
        common_rows = 0
        seen_new = set()
        with Path(new_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if common_new_output is not None:
                common_new_output = Path(common_new_output)
                common_new_output.parent.mkdir(parents=True, exist_ok=True)
                common_output_handle = common_new_output.open("w", newline="")
                common_writer = csv.DictWriter(
                    common_output_handle,
                    fieldnames=reader.fieldnames,
                )
                common_writer.writeheader()
            for new_row in reader:
                new_rows += 1
                key = (new_row["file_set_name"], int(new_row["tile_id"]))
                if key in seen_new:
                    raise ValueError(f"Duplicate comparison key in {new_path}: {key}")
                seen_new.add(key)
                old_row = old_rows.get(key)
                if old_row is None:
                    continue
                for field in ("tile_n_rows", "tile_n_cols"):
                    if field in old_row and field in new_row and old_row[field] != new_row[field]:
                        raise ValueError(f"Tile geometry mismatch for {key}: {field}")
                matched.add(key)
                common_rows += 1
                if common_writer is not None:
                    common_writer.writerow(new_row)
                old_mean, old_rms = mean_and_rms(old_row)
                new_mean, new_rms = mean_and_rms(new_row)
                old_reference_counts, old_current_counts = count_means(old_row)
                new_reference_counts, new_current_counts = count_means(new_row)
                old_has_counts = math.isfinite(old_reference_counts)
                new_has_counts = math.isfinite(new_reference_counts)
                if old_has_counts != new_has_counts:
                    raise ValueError(
                        "Both statistics files must include all count moments "
                        "to compare counts"
                    )
                if old_has_counts:
                    count_statistics_compared = True
                    abs_reference_count_deltas.append(
                        abs(new_reference_counts - old_reference_counts)
                    )
                    abs_current_count_deltas.append(
                        abs(new_current_counts - old_current_counts)
                    )
                mean_delta = new_mean - old_mean
                rms_delta = new_rms - old_rms
                valid_delta = int(new_row["n_pair_valid"]) - int(old_row["n_pair_valid"])
                if math.isfinite(mean_delta):
                    abs_mean_deltas.append(abs(mean_delta))
                if math.isfinite(rms_delta):
                    abs_rms_deltas.append(abs(rms_delta))
                valid_deltas.append(valid_delta)
                if writer is not None:
                    writer.writerow(
                        {
                            "file_set_name": new_row["file_set_name"],
                            "tile_id": new_row["tile_id"],
                            "old_n_pair_valid": old_row["n_pair_valid"],
                            "new_n_pair_valid": new_row["n_pair_valid"],
                            "delta_n_pair_valid": valid_delta,
                            "old_mean_log_change": old_mean,
                            "new_mean_log_change": new_mean,
                            "delta_mean_log_change": mean_delta,
                            "old_rms_log_change": old_rms,
                            "new_rms_log_change": new_rms,
                            "delta_rms_log_change": rms_delta,
                            "delta_sum_log_change": float(new_row["sum_log_change"])
                            - float(old_row["sum_log_change"]),
                            "delta_sum_squared_log_change": float(
                                new_row["sum_squared_log_change"]
                            )
                            - float(old_row["sum_squared_log_change"]),
                            "old_reference_mean_counts": old_reference_counts,
                            "new_reference_mean_counts": new_reference_counts,
                            "delta_reference_mean_counts": (
                                new_reference_counts - old_reference_counts
                            ),
                            "old_current_mean_counts": old_current_counts,
                            "new_current_mean_counts": new_current_counts,
                            "delta_current_mean_counts": (
                                new_current_counts - old_current_counts
                            ),
                        }
                    )
    finally:
        if output_handle is not None:
            output_handle.close()
        if common_output_handle is not None:
            common_output_handle.close()

    old_only = len(old_rows) - len(matched)
    new_only = new_rows - common_rows
    summary = {
        "old_rows": len(old_rows),
        "new_rows": new_rows,
        "common_rows": common_rows,
        "old_only_rows": old_only,
        "new_only_rows": new_only,
        "max_abs_mean_delta": max(abs_mean_deltas) if abs_mean_deltas else math.nan,
        "mean_abs_mean_delta": (
            sum(abs_mean_deltas) / len(abs_mean_deltas) if abs_mean_deltas else math.nan
        ),
        "mean_abs_rms_delta": (
            sum(abs_rms_deltas) / len(abs_rms_deltas) if abs_rms_deltas else math.nan
        ),
        "valid_count_delta_min": min(valid_deltas) if valid_deltas else math.nan,
        "valid_count_delta_max": max(valid_deltas) if valid_deltas else math.nan,
        "count_statistics_compared": count_statistics_compared,
        "max_abs_reference_mean_count_delta": (
            max(abs_reference_count_deltas)
            if abs_reference_count_deltas
            else math.nan
        ),
        "mean_abs_reference_mean_count_delta": (
            sum(abs_reference_count_deltas) / len(abs_reference_count_deltas)
            if abs_reference_count_deltas
            else math.nan
        ),
        "max_abs_current_mean_count_delta": (
            max(abs_current_count_deltas)
            if abs_current_count_deltas
            else math.nan
        ),
        "mean_abs_current_mean_count_delta": (
            sum(abs_current_count_deltas) / len(abs_current_count_deltas)
            if abs_current_count_deltas
            else math.nan
        ),
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--common-new-output",
        type=Path,
        default=None,
        help="Write the new full-statistics rows matching the old product/tile keys.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = compare(args.old, args.new, args.output, args.common_new_output)
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.output is not None:
        print(f"wrote: {args.output}")
    if args.common_new_output is not None:
        print(f"wrote: {args.common_new_output}")


if __name__ == "__main__":
    main()
