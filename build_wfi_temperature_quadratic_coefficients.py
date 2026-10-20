#!/usr/bin/env python3
"""Build reproducible WTSM00 quadratic coefficient tables for WFI tiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from build_wfi_quadratic_temperature_test import (
    fit_coefficients,
    interpolated_temperature,
    load_temperature,
    read_rows,
)


DEFAULT_DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_ARCHIVE_ROOT = Path("20261019_v2/extracted/20261019")
DEFAULT_TEMPERATURE = Path("20261019_v2/wfi_mce_adc_wtsm00.csv")
DEFAULT_OUTPUT = Path("temperature_quadratic_coefficients.csv")
DEFAULT_OPERATIONAL_TEMPERATURE_TIME = "2026-09-27T02:58:35.308Z"


def _fit_points(points):
    data = np.asarray(points, dtype=float)
    if data.shape[0] < 3:
        raise ValueError("A tile temperature correction needs at least three points")
    design = np.column_stack((np.ones(data.shape[0]), data[:, 0], data[:, 0] ** 2))
    coefficients, _, _, _ = np.linalg.lstsq(design, data[:, 1], rcond=None)
    normalization_offset = float(coefficients[0])
    coefficients = np.asarray(coefficients, dtype=float)
    coefficients[0] = 0.0
    fitted = design @ coefficients
    residual_log = data[:, 1] - fitted
    residual_percent = 100.0 * np.expm1(data[:, 1]) - 100.0 * np.expm1(fitted)
    total_sum_squares = float(np.sum((data[:, 1] - np.mean(data[:, 1])) ** 2))
    return coefficients, {
        "fit_points": int(data.shape[0]),
        "normalization_offset_log_correction": normalization_offset,
        "delta_temperature_min_k": float(np.min(data[:, 0])),
        "delta_temperature_max_k": float(np.max(data[:, 0])),
        "r2_log": (
            float(1.0 - np.sum(residual_log**2) / total_sum_squares)
            if total_sum_squares > 0.0
            else np.nan
        ),
        "rmse_percent": float(np.sqrt(np.mean(residual_percent**2))),
        "max_abs_percent": float(np.max(np.abs(residual_percent))),
    }


def aggregate_coefficients(
    statistics_rows,
    correction_rows,
    temperature_seconds,
    temperature_values,
    operational_temperature_time,
    source_tile_size=256,
    output_tile_size=512,
):
    """Fit output-tile curves from count-weighted source-tile corrections."""
    if output_tile_size % source_tile_size != 0:
        raise ValueError("Output tile size must be a multiple of source tile size")
    grouping = output_tile_size // source_tile_size
    output_tiles_per_axis = 4096 // output_tile_size
    operational_temperature = interpolated_temperature(
        operational_temperature_time,
        temperature_seconds,
        temperature_values,
    )
    current_counts = {}
    timestamp_by_key = {}
    geometry = {}
    for row in statistics_rows:
        tile_id = int(row["tile_id"])
        group = int(row["group"])
        key = (tile_id, group)
        current_counts[key] = current_counts.get(key, 0.0) + float(
            row["sum_current_counts"]
        )
        timestamp_by_key.setdefault(key, row["exposure_start_time"])
        geometry.setdefault(
            tile_id,
            (int(row["tile_row"]), int(row["tile_col"])),
        )
    temperature_corrections = {
        (int(row["tile_id"]), int(row["group"])): float(
            row["log_gain_correction"]
        )
        + float(row["temperature_gain_log_correction"])
        for row in correction_rows
    }
    groups_by_tile = {}
    for tile_id, group in temperature_corrections:
        groups_by_tile.setdefault(tile_id, set()).add(group)
    source_tiles_by_output = {}
    for source_tile, (tile_row, tile_col) in geometry.items():
        if tile_row < 0 or tile_col < 0:
            raise ValueError(f"Invalid tile geometry for source tile {source_tile}")
        output_row = tile_row // grouping
        output_col = tile_col // grouping
        if output_row >= output_tiles_per_axis or output_col >= output_tiles_per_axis:
            raise ValueError(f"Source tile {source_tile} is outside the detector")
        output_tile = output_row * output_tiles_per_axis + output_col
        source_tiles_by_output.setdefault(output_tile, []).append(source_tile)

    points_by_tile = {}
    for output_tile, source_tiles in source_tiles_by_output.items():
        common_groups = None
        for source_tile in source_tiles:
            groups = groups_by_tile.get(source_tile, set())
            common_groups = groups if common_groups is None else common_groups & groups
        for group in sorted(common_groups or ()):
            weighted_factors = []
            for source_tile in source_tiles:
                key = (source_tile, group)
                if key not in current_counts or key not in timestamp_by_key:
                    continue
                weight = current_counts[key]
                if not np.isfinite(weight) or weight <= 0.0:
                    continue
                weighted_factors.append(
                    (weight, temperature_corrections[key])
                )
            if len(weighted_factors) != len(source_tiles):
                continue
            weights = np.asarray([item[0] for item in weighted_factors], dtype=float)
            corrections = np.asarray(
                [item[1] for item in weighted_factors], dtype=float
            )
            effective_correction = float(
                np.log(np.sum(weights * np.exp(corrections)) / np.sum(weights))
            )
            if abs(effective_correction) <= 1.0e-14:
                continue
            source_key = (source_tiles[0], group)
            delta_temperature = (
                interpolated_temperature(
                    timestamp_by_key[source_key],
                    temperature_seconds,
                    temperature_values,
                )
                - operational_temperature
            )
            points_by_tile.setdefault(output_tile, []).append(
                (delta_temperature, effective_correction)
            )

    coefficients = {}
    diagnostics = {}
    for tile_id, points in sorted(points_by_tile.items()):
        coefficients[tile_id], diagnostics[tile_id] = _fit_points(points)
    return coefficients, diagnostics, operational_temperature


def _rows_for_detector(archive_root, detector, tile_size=256):
    detector_name = detector.lower()
    statistics_path = (
        archive_root / "statistics" / f"{detector_name}_tile_log_statistics_{tile_size}.csv"
    )
    corrections_path = (
        archive_root
        / "corrections"
        / f"{detector_name}_count_space_unified_temperature_corrections.csv"
    )
    return read_rows(statistics_path), read_rows(corrections_path)


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "detector",
        "tile_size",
        "tile_id",
        "reference_temperature_k",
        "a_log_correction",
        "b_log_correction_per_k",
        "c_log_correction_per_k2",
        "normalization_offset_log_correction",
        "fit_points",
        "delta_temperature_min_k",
        "delta_temperature_max_k",
        "r2_log",
        "rmse_percent",
        "max_abs_percent",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_table(
    archive_root,
    temperature_path,
    detectors,
    operational_temperature_time,
):
    temperature_seconds, temperature_values = load_temperature(temperature_path)
    rows = []
    for detector in detectors:
        statistics_rows, correction_rows = _rows_for_detector(archive_root, detector)
        coefficients, diagnostics = fit_coefficients(
            statistics_rows,
            correction_rows,
            temperature_seconds,
            temperature_values,
            operational_temperature_time,
        )
        reference_temperature = interpolated_temperature(
            operational_temperature_time,
            temperature_seconds,
            temperature_values,
        )
        for tile_id in sorted(coefficients):
            rows.append(
                {
                    "detector": detector,
                    "tile_size": 256,
                    "tile_id": tile_id,
                    "reference_temperature_k": reference_temperature,
                    "a_log_correction": coefficients[tile_id][0],
                    "b_log_correction_per_k": coefficients[tile_id][1],
                    "c_log_correction_per_k2": coefficients[tile_id][2],
                    **diagnostics[tile_id],
                }
            )
        coefficients, diagnostics, reference_temperature = aggregate_coefficients(
            statistics_rows,
            correction_rows,
            temperature_seconds,
            temperature_values,
            operational_temperature_time,
        )
        for tile_id in sorted(coefficients):
            rows.append(
                {
                    "detector": detector,
                    "tile_size": 512,
                    "tile_id": tile_id,
                    "reference_temperature_k": reference_temperature,
                    "a_log_correction": coefficients[tile_id][0],
                    "b_log_correction_per_k": coefficients[tile_id][1],
                    "c_log_correction_per_k2": coefficients[tile_id][2],
                    **diagnostics[tile_id],
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--temperature", type=Path, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detector", action="append", dest="detectors")
    parser.add_argument(
        "--operational-temperature-time",
        default=DEFAULT_OPERATIONAL_TEMPERATURE_TIME,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detectors = tuple(args.detectors or DEFAULT_DETECTORS)
    rows = build_table(
        args.archive_root,
        args.temperature,
        detectors,
        args.operational_temperature_time,
    )
    write_table(args.output, rows)
    print(f"Wrote {len(rows)} coefficient rows to {args.output}")


if __name__ == "__main__":
    main()
