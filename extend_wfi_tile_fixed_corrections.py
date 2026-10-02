"""Extend a tile correction table over new groups without refitting old offsets."""

import argparse
from bisect import bisect_left
import csv
from pathlib import Path


FIELDS = (
    "detector",
    "tile_id",
    "group",
    "days",
    "log_gain_correction",
    "acm_step_log_offset",
    "pre_reset_acm_step_log_offset",
    "additional_acm_step_log_offset",
    "late_acm_step_log_offset",
)
REQUIRED_FIELDS = {
    "detector",
    "tile_id",
    "group",
    "days",
    "log_gain_correction",
    "acm_step_log_offset",
    "late_acm_step_log_offset",
}


def detector_name(path):
    return path.name.split("_tile_log_statistics_", 1)[0].upper()


def read_corrections(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No correction rows found in {path}")
    missing = REQUIRED_FIELDS.difference(rows[0])
    if missing:
        raise ValueError(f"Correction file is missing columns: {sorted(missing)}")
    keyed = {}
    for row in rows:
        key = (row["detector"].strip().upper(), int(row["tile_id"]), int(row["group"]))
        if key in keyed:
            raise ValueError(f"Duplicate correction row for {key}")
        keyed[key] = {
            field: row.get(field, "0") or "0"
            for field in FIELDS
        }
    return keyed


def read_statistics(paths):
    rows = {}
    for path in sorted(Path(paths).glob("wfi*_tile_log_statistics_512.csv")):
        detector = detector_name(path)
        with path.open(newline="") as handle:
            for source_row in csv.DictReader(handle):
                key = (detector, int(source_row["tile_id"]), int(source_row["group"]))
                if key in rows:
                    raise ValueError(f"Duplicate statistics row for {key}")
                rows[key] = {
                    "days": float(source_row["days"]),
                }
    if not rows:
        raise ValueError(f"No tile statistics found in {paths}")
    return rows


def extend_corrections(base_path, statistics_dir, output_path, new_log_gain_correction=0.0):
    if not isinstance(new_log_gain_correction, (int, float)):
        raise ValueError("New log gain correction must be numeric")
    base = read_corrections(base_path)
    statistics = read_statistics(statistics_dir)
    output = dict(base)
    base_groups_by_tile = {}
    for detector, tile_id, group in base:
        base_groups_by_tile.setdefault((detector, tile_id), []).append(group)
    for groups in base_groups_by_tile.values():
        groups.sort()
    added = 0
    for key, statistics_row in statistics.items():
        detector, tile_id, group = key
        if key in base:
            if abs(float(base[key]["days"]) - statistics_row["days"]) > 1.0e-9:
                raise ValueError(f"Correction day does not match statistics for {key}")
            continue
        tile_groups = base_groups_by_tile.get((detector, tile_id), [])
        prior_index = bisect_left(tile_groups, group) - 1
        if prior_index < 0:
            raise ValueError(f"No prior correction exists for new group {key}")
        prior_group = tile_groups[prior_index]
        prior = base[(detector, tile_id, prior_group)]
        output[key] = {
            "detector": detector,
            "tile_id": str(tile_id),
            "group": str(group),
            "days": repr(statistics_row["days"]),
            "log_gain_correction": repr(float(new_log_gain_correction)),
            "acm_step_log_offset": prior["acm_step_log_offset"],
            "pre_reset_acm_step_log_offset": prior[
                "pre_reset_acm_step_log_offset"
            ],
            "additional_acm_step_log_offset": prior[
                "additional_acm_step_log_offset"
            ],
            "late_acm_step_log_offset": prior["late_acm_step_log_offset"],
        }
        added += 1

    missing = sorted(set(statistics).difference(output))
    if missing:
        raise ValueError(f"Failed to create corrections for statistics rows: {missing[:5]}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(output, key=lambda value: (value[0], value[1], value[2])):
            writer.writerow(output[key])
    return added, len(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-corrections",
        type=Path,
        default=Path("refit_corrections_tile/wfi_all_tile_fixed_led3_gain_acm_corrections.csv"),
    )
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=Path("wfi_tile_log_statistics_512"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "refit_corrections_tile/wfi_all_tile_fixed_led3_gain_acm_corrections_extended.csv"
        ),
    )
    parser.add_argument(
        "--new-log-gain-correction",
        type=float,
        default=0.0,
        help="Gain correction assigned to new groups whose temperature correction is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    added, total = extend_corrections(
        args.base_corrections,
        args.statistics_dir,
        args.output,
        args.new_log_gain_correction,
    )
    print(f"added {added} correction rows; wrote {total} rows to {args.output}")


if __name__ == "__main__":
    main()
