#!/usr/bin/env python3
"""Build an isolated WFI per-tile quadratic temperature-correction test input."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REFERENCE_COUNT_FIELDS = (
    "sum_reference_counts",
    "min_reference_count",
    "max_reference_count",
)
CURRENT_COUNT_FIELDS = (
    "sum_current_counts",
    "min_current_count",
    "max_current_count",
)
REFERENCE_SQUARE_FIELDS = (
    "sum_squared_reference_counts",
)
CURRENT_SQUARE_FIELDS = (
    "sum_squared_current_counts",
)
MIXED_SQUARE_FIELDS = (
    "sum_reference_current_counts",
)


def parse_time(value: str) -> float:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_temperature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    seconds = []
    values = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            seconds.append(parse_time(row["ObsTime"]))
            values.append(float(row["EUValue"]))
    order = np.argsort(seconds)
    return np.asarray(seconds, dtype=float)[order], np.asarray(values, dtype=float)[order]


def interpolated_temperature(
    timestamp: str, temperature_seconds: np.ndarray, temperature_values: np.ndarray
) -> float:
    return float(
        np.interp(
            parse_time(timestamp),
            temperature_seconds,
            temperature_values,
        )
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fit_coefficients(
    statistics_rows: list[dict[str, str]],
    correction_rows: list[dict[str, str]],
    temperature_seconds: np.ndarray,
    temperature_values: np.ndarray,
    operational_temperature_time: str,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    timestamp_by_key = {
        (int(row["tile_id"]), int(row["group"])): row["exposure_start_time"]
        for row in statistics_rows
    }
    operational_temperature = interpolated_temperature(
        operational_temperature_time,
        temperature_seconds,
        temperature_values,
    )
    points_by_tile: dict[int, list[tuple[float, float]]] = {}
    for row in correction_rows:
        old_temperature_correction = float(row["log_gain_correction"]) + float(
            row["temperature_gain_log_correction"]
        )
        if abs(old_temperature_correction) <= 1.0e-14:
            continue
        key = (int(row["tile_id"]), int(row["group"]))
        timestamp = timestamp_by_key[key]
        delta_temperature = (
            interpolated_temperature(
                timestamp,
                temperature_seconds,
                temperature_values,
            )
            - operational_temperature
        )
        points_by_tile.setdefault(key[0], []).append(
            (delta_temperature, old_temperature_correction)
        )

    coefficients = {}
    diagnostics = {}
    for tile_id, points in sorted(points_by_tile.items()):
        data = np.asarray(points, dtype=float)
        design = np.column_stack(
            (np.ones(data.shape[0]), data[:, 0], data[:, 0] ** 2)
        )
        fitted_coefficients, _, _, _ = np.linalg.lstsq(
            design,
            data[:, 1],
            rcond=None,
        )
        normalization_offset = float(fitted_coefficients[0])
        fitted_coefficients = np.asarray(fitted_coefficients, dtype=float)
        fitted_coefficients[0] = 0.0
        fitted = design @ fitted_coefficients
        residual_percent = 100.0 * np.expm1(data[:, 1]) - 100.0 * np.expm1(fitted)
        residual_log = data[:, 1] - fitted
        total_sum_squares = np.sum((data[:, 1] - np.mean(data[:, 1])) ** 2)
        coefficients[tile_id] = fitted_coefficients
        diagnostics[tile_id] = {
            "fit_points": float(data.shape[0]),
            "normalization_offset_log_correction": normalization_offset,
            "delta_temperature_min_k": float(np.min(data[:, 0])),
            "delta_temperature_max_k": float(np.max(data[:, 0])),
            "r2_log": float(
                1.0 - np.sum(residual_log**2) / total_sum_squares
            ),
            "rmse_percent": float(np.sqrt(np.mean(residual_percent**2))),
            "max_abs_percent": float(np.max(np.abs(residual_percent))),
        }
    return coefficients, diagnostics


def write_coefficients(
    path: Path,
    coefficients: dict[int, np.ndarray],
    diagnostics: dict[int, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile_id",
        "a_log_correction",
        "b_log_correction_per_k",
        "c_log_correction_per_k2",
        *diagnostics[next(iter(diagnostics))].keys(),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tile_id in sorted(coefficients):
            row = {
                "tile_id": tile_id,
                "a_log_correction": coefficients[tile_id][0],
                "b_log_correction_per_k": coefficients[tile_id][1],
                "c_log_correction_per_k2": coefficients[tile_id][2],
                **diagnostics[tile_id],
            }
            writer.writerow(row)


def build_corrected_statistics(
    output_path: Path,
    correction_rows: list[dict[str, str]],
    raw_statistics_rows: list[dict[str, str]],
    selected_statistics_rows: list[dict[str, str]],
    coefficients: dict[int, np.ndarray],
    temperature_seconds: np.ndarray,
    temperature_values: np.ndarray,
    operational_temperature_time: str,
) -> dict[str, float]:
    correction_by_key = {
        (int(row["tile_id"]), int(row["group"])): row
        for row in correction_rows
    }
    timestamp_by_key = {
        (int(row["tile_id"]), int(row["group"])): row["exposure_start_time"]
        for row in raw_statistics_rows
    }
    operational_temperature = interpolated_temperature(
        operational_temperature_time,
        temperature_seconds,
        temperature_values,
    )
    first_group_by_led = {}
    for row in selected_statistics_rows:
        first_group_by_led.setdefault(int(row["led"]), int(row["group"]))

    new_absolute_correction = {}
    max_abs_absolute_change = 0.0
    for key, correction in correction_by_key.items():
        old_temperature_correction = float(
            correction["log_gain_correction"]
        ) + float(correction["temperature_gain_log_correction"])
        timestamp = timestamp_by_key[key]
        delta_temperature = (
            interpolated_temperature(
                timestamp,
                temperature_seconds,
                temperature_values,
            )
            - operational_temperature
        )
        a, b, c = coefficients[key[0]]
        new_temperature_correction = float(
            a + b * delta_temperature + c * delta_temperature**2
        )
        old_absolute_correction = float(correction["count_log_correction"])
        new_absolute = (
            old_absolute_correction
            - old_temperature_correction
            + new_temperature_correction
        )
        new_absolute_correction[key] = new_absolute
        max_abs_absolute_change = max(
            max_abs_absolute_change,
            abs(new_absolute - old_absolute_correction),
        )

    raw_by_key = {
        (int(row["tile_id"]), int(row["group"])): row
        for row in raw_statistics_rows
    }
    fieldnames = list(selected_statistics_rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    applied_rows = 0
    max_abs_relative_delta = 0.0
    with output_path.open("w", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        for selected_row in selected_statistics_rows:
            key = (int(selected_row["tile_id"]), int(selected_row["group"]))
            led = int(selected_row["led"])
            row = dict(raw_by_key[key])
            current_correction = new_absolute_correction[key]
            reference_key = (int(selected_row["tile_id"]), first_group_by_led[led])
            reference_correction = new_absolute_correction[reference_key]
            applied_rows += 1
            delta = current_correction - reference_correction
            reference_factor = float(np.exp(reference_correction))
            current_factor = float(np.exp(current_correction))
            old_sum = float(row["sum_log_change"])
            old_sum_squared = float(row["sum_squared_log_change"])
            n_pair_valid = float(row["n_pair_valid"])
            row["sum_log_change"] = repr(old_sum + n_pair_valid * delta)
            row["sum_squared_log_change"] = repr(
                old_sum_squared + 2.0 * delta * old_sum + n_pair_valid * delta * delta
            )
            row["min_log_change"] = repr(float(row["min_log_change"]) + delta)
            row["max_log_change"] = repr(float(row["max_log_change"]) + delta)
            for field in REFERENCE_COUNT_FIELDS:
                row[field] = repr(float(float(row[field]) * reference_factor))
            for field in CURRENT_COUNT_FIELDS:
                row[field] = repr(float(float(row[field]) * current_factor))
            for field in REFERENCE_SQUARE_FIELDS:
                row[field] = repr(float(float(row[field]) * reference_factor**2))
            for field in CURRENT_SQUARE_FIELDS:
                row[field] = repr(float(float(row[field]) * current_factor**2))
            for field in MIXED_SQUARE_FIELDS:
                row[field] = repr(
                    float(float(row[field]) * reference_factor * current_factor)
                )
            writer.writerow(row)
            max_abs_relative_delta = max(max_abs_relative_delta, abs(delta))
    return {
        "input_rows": float(len(selected_statistics_rows)),
        "applied_rows": float(applied_rows),
        "max_abs_relative_log_correction": max_abs_relative_delta,
        "max_abs_absolute_correction_change": max_abs_absolute_change,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="WFI02")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("20261019_v2/extracted/20261019"),
    )
    parser.add_argument(
        "--temperature",
        type=Path,
        default=Path("20261019_v2/wfi_mce_adc_wtsm00.csv"),
    )
    parser.add_argument(
        "--input-statistics",
        type=Path,
        default=Path("20261019/corrected_statistics_4h_thinned"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("20261019/quadratic_temperature_test/wfi02"),
    )
    parser.add_argument(
        "--operational-temperature-time",
        default="2026-09-27T02:58:35.308Z",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = args.detector.lower()
    archive_statistics = (
        args.archive_root / "statistics" / f"{detector}_tile_log_statistics_256.csv"
    )
    correction_path = (
        args.archive_root
        / "corrections"
        / f"{detector}_count_space_unified_temperature_corrections.csv"
    )
    input_statistics = (
        args.input_statistics / f"{detector}_tile_log_statistics_256_count_corrected.csv"
    )
    output_statistics = (
        args.output_root / f"{detector}_tile_log_statistics_256_count_corrected.csv"
    )
    temperature_seconds, temperature_values = load_temperature(args.temperature)
    archive_rows = read_rows(archive_statistics)
    correction_rows = read_rows(correction_path)
    coefficients, diagnostics = fit_coefficients(
        archive_rows,
        correction_rows,
        temperature_seconds,
        temperature_values,
        args.operational_temperature_time,
    )
    write_coefficients(args.output_root / "tile_quadratic_coefficients.csv", coefficients, diagnostics)
    input_rows = read_rows(input_statistics)
    application = build_corrected_statistics(
        output_statistics,
        correction_rows,
        archive_rows,
        input_rows,
        coefficients,
        temperature_seconds,
        temperature_values,
        args.operational_temperature_time,
    )
    summary = {
        "detector": args.detector,
        "operational_temperature_time": args.operational_temperature_time,
        "tile_count": len(coefficients),
        "coefficients": str(args.output_root / "tile_quadratic_coefficients.csv"),
        "statistics": str(output_statistics),
        **application,
    }
    with (args.output_root / "build_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
