"""Audit focal-plane spatial trends before and after WFI corrections."""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


STAGES = (
    ("raw", "raw_statistics", ""),
    ("scaled", "statistics", ""),
    ("corrected", "corrected_statistics", "_count_corrected"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", required=True)
    parser.add_argument("--reset-day", type=float, required=True)
    return parser.parse_args()


def statistics_path(reduction_root, detector, directory, suffix):
    name = f"{detector.lower()}_tile_log_statistics_512{suffix}.csv"
    return reduction_root / directory / name


def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Statistics file is empty: {path}")
    by_tile = {}
    for row in rows:
        if float(row["n_pair_valid"]) <= 0.0:
            continue
        by_tile.setdefault(int(row["tile_id"]), []).append(row)
    if len(by_tile) != 64:
        raise ValueError(f"Expected 64 tiles in {path}, found {len(by_tile)}")
    return by_tile


def slope(rows, reset_day, post):
    days = np.asarray([float(row["days"]) for row in rows], dtype=float)
    values = np.asarray(
        [
            float(row["sum_log_change"]) / float(row["n_pair_valid"])
            for row in rows
        ],
        dtype=float,
    )
    mask = days >= reset_day if post else days < reset_day
    if np.count_nonzero(mask) < 2:
        raise ValueError("A reset segment has too few observations")
    return float(np.polyfit(days[mask], values[mask], 1)[0])


def baseline_mean(rows):
    first_day = min(float(row["days"]) for row in rows)
    first_rows = [row for row in rows if float(row["days"]) == first_day]
    return float(
        np.mean(
            [
                float(row["sum_current_counts"]) / float(row["n_pair_valid"])
                for row in first_rows
            ]
        )
    )


def tile_summary(detector, stage_rows, reset_day):
    raw_rows = stage_rows["raw"]
    rows = []
    for tile_id in sorted(raw_rows):
        metadata = raw_rows[tile_id][0]
        row = {
            "detector": detector,
            "tile_id": tile_id,
            "tile_row": int(metadata["tile_row"]),
            "tile_col": int(metadata["tile_col"]),
            "raw_baseline_mean_counts": baseline_mean(raw_rows[tile_id]),
        }
        for stage, by_tile in stage_rows.items():
            for segment, post in (("pre", False), ("post", True)):
                row[f"{stage}_{segment}_log_slope_per_day"] = slope(
                    by_tile[tile_id], reset_day, post
                )
        rows.append(row)
    return rows


def inventory_key(row):
    return (
        row["epoch_key"],
        row["led"],
        row["exposure_start_time"],
        row["exposure_end_time"],
        row["program"],
        row["repeat"],
    )


def validate_inventory(reduction_root, detectors):
    inventories = {}
    for detector in detectors:
        path = reduction_root / f"{detector.lower()}_rrn_inventory.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        inventories[detector] = {inventory_key(row) for row in rows}
    reference = inventories[detectors[0]]
    mismatches = {
        detector: len(reference.symmetric_difference(keys))
        for detector, keys in inventories.items()
    }
    return mismatches


def pair_correlations(summary_rows, detectors):
    by_detector = {}
    for row in summary_rows:
        by_detector.setdefault(row["detector"], []).append(row)
    output = []
    for detector_a, detector_b in combinations(detectors, 2):
        rows_a = sorted(by_detector[detector_a], key=lambda row: row["tile_id"])
        rows_b = sorted(by_detector[detector_b], key=lambda row: row["tile_id"])
        for stage in ("raw", "scaled", "corrected"):
            for segment in ("pre", "post"):
                field = f"{stage}_{segment}_log_slope_per_day"
                values_a = np.asarray([float(row[field]) for row in rows_a])
                values_b = np.asarray([float(row[field]) for row in rows_b])
                output.append(
                    {
                        "detector_a": detector_a,
                        "detector_b": detector_b,
                        "stage": stage,
                        "segment": segment,
                        "correlation": float(np.corrcoef(values_a, values_b)[0, 1]),
                        "rms_difference": float(np.sqrt(np.mean((values_a - values_b) ** 2))),
                    }
                )
    return output


def write_difference_map(path, summary_rows):
    by_detector = {}
    for row in summary_rows:
        by_detector.setdefault(row["detector"], []).append(row)
    if "WFI02" not in by_detector or "WFI03" not in by_detector:
        return
    figure, axes = plt.subplots(3, 2, figsize=(10, 13), constrained_layout=True)
    for row_index, stage in enumerate(("raw", "scaled", "corrected")):
        for column, segment in enumerate(("pre", "post")):
            field = f"{stage}_{segment}_log_slope_per_day"
            values = np.zeros((8, 8), dtype=float)
            for row_02, row_03 in zip(
                sorted(by_detector["WFI02"], key=lambda row: row["tile_id"]),
                sorted(by_detector["WFI03"], key=lambda row: row["tile_id"]),
            ):
                values[int(row_02["tile_row"]), int(row_02["tile_col"])] = (
                    float(row_03[field]) - float(row_02[field])
                )
            limit = max(float(np.max(np.abs(values))), 1.0e-12)
            image = axes[row_index, column].imshow(
                values,
                origin="lower",
                interpolation="none",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            axes[row_index, column].set_title(f"WFI03 - WFI02, {stage}, {segment}")
            axes[row_index, column].set_xlabel("Tile column")
            axes[row_index, column].set_ylabel("Tile row")
            axes[row_index, column].set_xticks(range(8))
            axes[row_index, column].set_yticks(range(8))
            figure.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.04)
    figure.suptitle("WFI02/WFI03 per-tile log-slope differences")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    detectors = [detector.upper() for detector in args.detectors]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = {}
    all_summary_rows = []
    for detector in detectors:
        detector_stages = {
            stage: read_rows(statistics_path(args.reduction_root, detector, directory, suffix))
            for stage, directory, suffix in STAGES
        }
        detector_rows = tile_summary(detector, detector_stages, args.reset_day)
        all_summary_rows.extend(detector_rows)
        stage_rows[detector] = detector_stages

    summary_fields = list(all_summary_rows[0])
    with (args.output_dir / "wfi_detector_tile_provenance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(all_summary_rows)

    correlations = pair_correlations(all_summary_rows, detectors)
    with (args.output_dir / "wfi_detector_pair_correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlations[0]))
        writer.writeheader()
        writer.writerows(correlations)

    write_difference_map(args.output_dir / "wfi02_wfi03_log_slope_differences.png", all_summary_rows)
    inventory_mismatches = validate_inventory(args.reduction_root, detectors)
    lines = [
        "# WFI detector provenance audit",
        "",
        f"- Detectors: `{', '.join(detectors)}`.",
        f"- Physical reset split: `{args.reset_day:.15g}` days.",
        "- Statistics stages: unscaled raw counts, photodiode-scaled counts, and count-corrected counts.",
        "- Inventory mismatch counts are symmetric differences against the first detector's observation keys.",
        "",
        "## Inventory",
    ]
    for detector in detectors:
        lines.append(f"- `{detector}` mismatch count: `{inventory_mismatches[detector]}`.")
    lines.extend(("", "## WFI02/WFI03", ""))
    for stage in ("raw", "scaled", "corrected"):
        for segment in ("pre", "post"):
            match = next(
                row
                for row in correlations
                if row["detector_a"] == "WFI02"
                and row["detector_b"] == "WFI03"
                and row["stage"] == stage
                and row["segment"] == segment
            )
            lines.append(
                f"- `{stage}` `{segment}` slope correlation: `{float(match['correlation']):.6f}`, "
                f"RMS difference: `{float(match['rms_difference']):.6g}`."
            )
    lines.extend(
        (
            "",
            "The raw-stage correlation is the key provenance check: similarity present there is not created by the TMM fit or count-correction pass.",
            "",
        )
    )
    (args.output_dir / "wfi_detector_provenance_summary.md").write_text("\n".join(lines))
    print(f"Wrote {args.output_dir / 'wfi_detector_tile_provenance.csv'}")
    print(f"Wrote {args.output_dir / 'wfi_detector_pair_correlations.csv'}")
    print(f"Wrote {args.output_dir / 'wfi_detector_provenance_summary.md'}")


if __name__ == "__main__":
    main()
