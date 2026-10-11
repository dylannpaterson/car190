"""Aggregate production model residuals into detector-level CSV files."""

import argparse
import csv
from pathlib import Path


DETECTORS = ("wfi01", "wfi02")
FIELDNAMES = (
    "detector",
    "tile_id",
    "group",
    "days",
    "led",
    "repeat",
    "observed_log_change",
    "model_log_change",
    "residual_log",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tile_sort_key(path):
    return int(path.parent.name.split("_", 1)[1])


def load_detector_rows(production_dir, detector):
    fits_dir = production_dir / detector / "fits"
    prediction_paths = sorted(
        fits_dir.glob("tile_*/tile_group_predictions.csv"),
        key=tile_sort_key,
    )
    if not prediction_paths:
        raise FileNotFoundError(f"No tile prediction files found under {fits_dir}")

    rows = []
    expected_detector = detector.upper()
    for prediction_path in prediction_paths:
        with prediction_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDNAMES:
                raise ValueError(
                    f"Unexpected columns in {prediction_path}: {reader.fieldnames}"
                )
            for row in reader:
                if row["detector"] != expected_detector:
                    raise ValueError(
                        f"Unexpected detector in {prediction_path}: {row['detector']}"
                    )
                rows.append({field: row[field] for field in FIELDNAMES})

    rows.sort(
        key=lambda row: (
            int(row["tile_id"]),
            int(row["repeat"]),
            int(row["led"]),
            int(row["group"]),
        )
    )
    return rows, len(prediction_paths)


def write_rows(rows, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for detector in DETECTORS:
        rows, tile_count = load_detector_rows(args.production_dir, detector)
        output_path = args.output_dir / f"{detector}_model_residuals.csv"
        write_rows(rows, output_path)
        all_rows.extend(rows)
        print(f"{detector.upper()}: {tile_count} tiles, {len(rows)} rows -> {output_path}")

    all_rows.sort(
        key=lambda row: (
            row["detector"],
            int(row["tile_id"]),
            int(row["repeat"]),
            int(row["led"]),
            int(row["group"]),
        )
    )
    write_rows(all_rows, args.output_dir / "wfi01_wfi02_model_residuals.csv")


if __name__ == "__main__":
    main()
