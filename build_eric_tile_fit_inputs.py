#!/usr/bin/env python3
"""Build matched local and Eric one-tile production-fit statistics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


ERIC_RUN_RE = re.compile(
    r"^(?:CAR\d+_)?(?P<program>\d{5})_(?P<decon>\d{2})_"
    r"(?P<pass_number>\d{3})_(?P<segment>\d{3})(?:_|$)"
)

OUTPUT_FIELDS = (
    "tile_id",
    "source_tile_id",
    "tile_row",
    "tile_col",
    "group",
    "n_pair_valid",
    "sum_log_change",
    "sum_squared_log_change",
    "led",
    "repeat",
    "days",
    "program",
    "exposure_start_time",
    "source_run_label",
    "source_file_set_name",
)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def parse_timestamp(value):
    return datetime.fromisoformat(value.rstrip("0") if value.endswith("0000") else value)


def parse_program(run_label):
    match = ERIC_RUN_RE.match(run_label)
    if match is None:
        raise ValueError(f"Could not parse Eric run label: {run_label}")
    return int(match.group("program"))


def make_row(
    *,
    source_tile_id,
    local_row,
    run_label,
    file_set_name,
    repeat,
    days,
    led,
    mean,
    sum_log_change,
    sum_squared_log_change,
    group,
    timestamp,
):
    return {
        "tile_id": 0,
        "source_tile_id": source_tile_id,
        "tile_row": local_row.get("tile_row", source_tile_id // 8),
        "tile_col": local_row.get("tile_col", source_tile_id % 8),
        "group": group,
        "n_pair_valid": local_row["n_pair_valid"],
        "sum_log_change": repr(sum_log_change),
        "sum_squared_log_change": repr(sum_squared_log_change),
        "led": led,
        "repeat": repeat,
        "days": repr(days),
        "program": parse_program(run_label),
        "exposure_start_time": timestamp,
        "source_run_label": run_label,
        "source_file_set_name": file_set_name,
    }


def build_inputs(detector, source_tile_id, local_path, matched_path, eric_path, output_root):
    local_rows = {
        (row["file_set_name"], int(row["led"])): row
        for row in read_csv(local_path)
        if int(row["tile_id"]) == source_tile_id
    }
    matched_rows = {
        (row["run_label"], int(row["led"])): row
        for row in read_csv(matched_path)
        if row["detector"] == detector and int(row["tile_id"]) == source_tile_id
    }
    eric_rows = [
        row
        for row in read_csv(eric_path)
        if row["detector"] == detector and int(row["tile_id"]) == source_tile_id
    ]
    if not local_rows or not matched_rows or not eric_rows:
        raise ValueError(f"Missing matched rows for {detector} tile {source_tile_id}")

    run_times = {}
    for row in eric_rows:
        run_times.setdefault(row["run_label"], parse_timestamp(row["timestamp"]))
    ordered_runs = sorted(run_times, key=lambda run: (run_times[run], run))
    repeat_by_run = {run: repeat for repeat, run in enumerate(ordered_runs)}
    first_time = run_times[ordered_runs[0]]

    outputs = {"local": [], "eric_calibrated": [], "eric_no_tempgain": []}
    for eric in sorted(eric_rows, key=lambda row: (run_times[row["run_label"]], int(row["led"]))):
        run_label = eric["run_label"]
        led = int(eric["led"])
        matched = matched_rows.get((run_label, led))
        if matched is None:
            raise ValueError(f"No local match for {detector} {run_label} LED {led}")
        local = local_rows.get((matched["local_file_set_name"], led))
        if local is None:
            raise ValueError(
                f"No local statistics row for {matched['local_file_set_name']} LED {led}"
            )
        n = float(local["n_pair_valid"])
        local_sum = float(local["sum_log_change"])
        local_squared = float(local["sum_squared_log_change"])
        local_mean = local_sum / n
        within = max(local_squared - n * local_mean * local_mean, 0.0)
        repeat = repeat_by_run[run_label]
        days = (run_times[run_label] - first_time).total_seconds() / 86400.0
        group = repeat * 6 + led - 1
        common = {
            "source_tile_id": source_tile_id,
            "local_row": local,
            "run_label": run_label,
            "file_set_name": matched["local_file_set_name"],
            "repeat": repeat,
            "days": days,
            "led": led,
            "group": group,
            "timestamp": eric["timestamp"],
        }
        outputs["local"].append(
            make_row(
                **common,
                sum_log_change=local_sum,
                sum_squared_log_change=local_squared,
                mean=local_mean,
            )
        )
        for name, field in (
            ("eric_calibrated", "calibrated_log_change_mean"),
            ("eric_no_tempgain", "no_tempgain_log_change_mean"),
        ):
            if not eric.get(field, ""):
                raise ValueError(f"Eric row is missing {field}: {eric}")
            mean = float(eric[field])
            outputs[name].append(
                make_row(
                    **common,
                    sum_log_change=n * mean,
                    sum_squared_log_change=within + n * mean * mean,
                    mean=mean,
                )
            )

    detector_root = Path(output_root) / detector.lower()
    for manifest_path in detector_root.glob("*/input_manifest.json"):
        existing = json.loads(manifest_path.read_text())
        if int(existing.get("source_tile_id", source_tile_id)) != source_tile_id:
            raise ValueError(
                f"{detector_root} already contains tile {existing['source_tile_id']}; "
                f"use a separate output root for tile {source_tile_id}"
            )
    detector_root.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        branch = detector_root / name
        branch.mkdir(parents=True, exist_ok=True)
        path = branch / f"{detector.lower()}_tile_log_statistics_256_count_corrected.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "detector": detector,
            "source_tile_id": source_tile_id,
            "fit_tile_id": 0,
            "branch": name,
            "rows": len(rows),
            "products": len(ordered_runs),
            "construction": (
                "Eric means use local valid-pixel counts and local within-group variance; "
                "local branch preserves local sufficient statistics."
            ),
            "source_statistics": str(Path(local_path).resolve()),
            "source_eric_trends": str(Path(eric_path).resolve()),
            "source_matching": str(Path(matched_path).resolve()),
        }
        (branch / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--source-tile-id", required=True, type=int)
    parser.add_argument("--local-statistics", required=True, type=Path)
    parser.add_argument("--matched-rows", required=True, type=Path)
    parser.add_argument("--eric-trends", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    build_inputs(
        args.detector.upper(),
        args.source_tile_id,
        args.local_statistics,
        args.matched_rows,
        args.eric_trends,
        args.output_root,
    )


if __name__ == "__main__":
    main()
