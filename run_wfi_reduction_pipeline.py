#!/usr/bin/env python3
"""Fully automated CAR190 reduction entry point.

The entry point has no required arguments. It synchronizes calibrated ASDF
products from S3, refreshes the all-detector MAST supplement and photodiode
telemetry, reduces every detector, fits the count-space gain/ACM correction
table, and writes a second statistics CSV with that correction applied. The
raw statistics plus correction table remain the canonical inputs for the TMM
fit; the count-corrected CSV is an explicit inspection and downstream-
reduction product.
"""

import csv
from datetime import datetime, timezone
import json
import shlex
from pathlib import Path
import shutil
import subprocess
import sys

from apply_wfi_tile_count_corrections import write_corrected_statistics
from build_wfi_tile_count_corrections import (
    CORRECTION_FIELDS,
    SUMMARY_FIELDS,
    build_corrections,
    write_csv,
)
from build_wfi_tile_statistics_rrn import (
    attach_photodiode_corrections,
    build_observations,
    fetch_photodiode_telemetry,
    login_to_mast,
    mast_records_from_index,
    merge_product_records,
    read_optional_index,
    read_index_rows,
    resolve_record_times,
    scan_products,
    stream_mast_image,
    update_detector_statistics_incremental,
    write_detector_statistics,
    write_inventory,
)
from query_rcs_flats import DEFAULT_DETECTORS, refresh_index


ROOT = Path(__file__).resolve().parent
DEFAULT_S3_SOURCE = "s3://roman-s3-romcom/activities/CAR190/"
DEFAULT_TILE_SIZES = (512, 64)
DEFAULT_RUN_ROOT = ROOT / "CAR190_reduction"
DEFAULT_MAST_TOKEN = ROOT / "mast_api_token.txt"
DEFAULT_MAST_INDEX = ROOT / "all_wfi_rcs_flat_index.csv"
DEFAULT_PHOTODIODE_CHANNEL = "pd1"
DEFAULT_ICE_RESET_DAY = 15.064405023148149
DEFAULT_ADDITIONAL_ACM_STEP_REPEAT = 39
DEFAULT_PRE_RESET_ACM_STEP_REPEAT = 35


def utc_run_date(value=None):
    """Return a compact UTC date suitable for a dated output directory."""
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%d")
    parsed = datetime.strptime(str(value), "%Y%m%d")
    return parsed.strftime("%Y%m%d")


def is_s3_uri(value):
    return str(value).lower().startswith("s3://")


def sync_s3_source(source_uri, cache_dir, aws_binary="aws", aws_profile=None):
    """Synchronize only calibrated products into a persistent local cache."""
    source_uri = str(source_uri)
    if not is_s3_uri(source_uri):
        raise ValueError(f"Expected an s3:// source URI, got {source_uri!r}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [str(aws_binary)]
    if aws_profile:
        command.extend(("--profile", str(aws_profile)))
    command.extend(
        (
            "s3",
            "sync",
            source_uri.rstrip("/") + "/",
            str(cache_dir),
            "--exclude",
            "*",
            "--include",
            "*_cal.asdf",
            "--delete",
            "--no-progress",
            "--only-show-errors",
        )
    )
    print("Synchronizing S3 source:", shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    return cache_dir


def find_previous_statistics(output_root, run_dir, detector, tile_size):
    """Find the latest dated raw statistics file for incremental reuse."""
    filename = f"{detector.lower()}_tile_log_statistics_{tile_size}.csv"
    candidates = sorted(
        path
        for path in Path(output_root).glob(f"*/statistics/{filename}")
        if path.is_file() and path.parent.parent != Path(run_dir)
    )
    return candidates[-1] if candidates else None


def find_previous_inventory(output_root, run_dir, detector, tile_size=512):
    """Find the inventory paired with the latest reusable statistics file."""
    statistics = find_previous_statistics(output_root, run_dir, detector, tile_size)
    if statistics is None:
        return None
    inventory = statistics.parent.parent / f"{detector.lower()}_rrn_inventory.csv"
    return inventory if inventory.is_file() else None


def _normalized_scale(value):
    """Normalize inventory CSV values for comparison with observation metadata."""
    if value in (None, ""):
        return "1"
    return repr(float(value))


def source_changes_require_rebuild(inventory_path, records, observations):
    """Detect source or photodiode changes not represented by file-set names."""
    inventory_path = Path(inventory_path) if inventory_path is not None else None
    if inventory_path is None or not inventory_path.is_file():
        return True
    with inventory_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_set_name", "source_signature", "photodiode_scale"}
        if not required.issubset(reader.fieldnames or ()):
            return True
        previous = {
            row["file_set_name"]: (
                row["source_signature"],
                _normalized_scale(row["photodiode_scale"]),
            )
            for row in reader
        }
    records_by_file_set = {
        record["file_set_name"]: record for record in records
    }
    current = {
        observation["file_set_name"]: (
            records_by_file_set[observation["file_set_name"]].get(
                "source_signature", ""
            ),
            _normalized_scale(observation.get("photodiode_scale")),
        )
        for observation in observations
    }
    if set(previous).difference(current):
        return True
    return any(previous[name] != current[name] for name in set(previous) & set(current))


def remove_incremental_state(path):
    parts = path.with_name(path.name + ".incremental.parts")
    if parts.is_dir():
        shutil.rmtree(parts)


def reduce_statistics(
    observations,
    records,
    output_path,
    tile_size,
    output_root,
    run_dir,
    no_resume,
    fetch_image=None,
):
    """Build or incrementally update one raw tile-statistics CSV."""
    output_path = Path(output_path)
    if no_resume:
        remove_incremental_state(output_path)
        write_detector_statistics(
            observations,
            output_path,
            tile_size,
            resume=False,
            fetch_image=fetch_image,
        )
        return "rebuilt"

    if not output_path.exists():
        previous = find_previous_statistics(
            output_root, run_dir, records[0]["detector"], tile_size
        )
        if previous is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous, output_path)
            print(f"Seeding incremental reduction from {previous}")
        else:
            write_detector_statistics(
                observations,
                output_path,
                tile_size,
                resume=True,
                fetch_image=fetch_image,
            )
            return "rebuilt"

    try:
        update_detector_statistics_incremental(
            observations,
            output_path,
            tile_size,
            fetch_image=fetch_image,
        )
    except ValueError as error:
        if not str(error).startswith("Incremental update"):
            raise
        print(f"Incremental state is incompatible; rebuilding {output_path}: {error}")
        output_path.unlink(missing_ok=True)
        remove_incremental_state(output_path)
        write_detector_statistics(
            observations,
            output_path,
            tile_size,
            resume=False,
            fetch_image=fetch_image,
        )
        return "rebuilt"
    return "incremental"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_telemetry(path, telemetry):
    """Persist the raw photodiode query used for this reduction."""
    rows = [
        {
            "channel": telemetry["channel"],
            "mnemonic": telemetry["mnemonic"],
            "query_start": telemetry["query_start"].isoformat() + "Z",
            "query_end": telemetry["query_end"].isoformat() + "Z",
            "observation_time": timestamp.isoformat() + "Z",
            "value": value,
        }
        for timestamp, value in telemetry["values"]
    ]
    write_csv(
        path,
        (
            "channel",
            "mnemonic",
            "query_start",
            "query_end",
            "observation_time",
            "value",
        ),
        rows,
    )


def run_detector(
    source_dir,
    run_dir,
    output_root,
    detector,
    index_metadata,
    index_rows,
    telemetry,
    fetch_image,
    manifest,
):
    local_records = scan_products(source_dir, detector=detector, allow_empty=True)
    mast_records = mast_records_from_index(index_rows, detector, allow_empty=True)
    records = merge_product_records(local_records, mast_records)
    if not records:
        raise ValueError(f"No local or MAST products found for {detector}")
    resolve_record_times(records, index_metadata, allow_ordinal_days=False)
    observations = build_observations(records, allow_ordinal_days=False)
    attach_photodiode_corrections(observations, telemetry)

    tile_sizes = DEFAULT_TILE_SIZES
    statistics_dir = run_dir / "statistics"
    inventory_path = run_dir / f"{detector.lower()}_rrn_inventory.csv"
    previous_inventory = inventory_path if inventory_path.is_file() else None
    if previous_inventory is None:
        previous_inventory = find_previous_inventory(output_root, run_dir, detector)
    source_changed = source_changes_require_rebuild(
        previous_inventory, records, observations
    )
    write_inventory(inventory_path, records, observations)
    detector_manifest = {
        "detector": detector,
        "local_products": len(local_records),
        "mast_products": len(mast_records),
        "products": len(records),
        "observation_epochs": len({row["repeat"] for row in observations}),
        "statistics": {},
        "inventory": str(inventory_path),
        "source_change_rebuild": source_changed,
    }
    for tile_size in tile_sizes:
        output_path = statistics_dir / (
            f"{detector.lower()}_tile_log_statistics_{tile_size}.csv"
        )
        mode = reduce_statistics(
            observations,
            records,
            output_path,
            tile_size,
            output_root,
            run_dir,
            source_changed,
            fetch_image=fetch_image,
        )
        detector_manifest["statistics"][str(tile_size)] = {
            "path": str(output_path),
            "mode": mode,
        }

    correction_dir = run_dir / "corrections"
    corrected_dir = run_dir / "corrected_statistics"
    raw_512 = statistics_dir / f"{detector.lower()}_tile_log_statistics_512.csv"
    correction_path = correction_dir / (
        f"{detector.lower()}_count_space_fixed_corrections_three_events.csv"
    )
    correction_summary_path = correction_dir / (
        f"{detector.lower()}_count_space_fit_summary_three_events.csv"
    )
    correction_rows, summary_rows = build_corrections(
        raw_512,
        3,
        1.0,
        DEFAULT_ADDITIONAL_ACM_STEP_REPEAT,
        DEFAULT_ICE_RESET_DAY,
        DEFAULT_PRE_RESET_ACM_STEP_REPEAT,
    )
    write_csv(correction_path, CORRECTION_FIELDS, correction_rows)
    write_csv(correction_summary_path, SUMMARY_FIELDS, summary_rows)
    corrected_path = corrected_dir / (
        f"{detector.lower()}_tile_log_statistics_512_count_corrected.csv"
    )
    write_corrected_statistics(
        raw_512,
        correction_path,
        corrected_path,
        additional_acm_step_repeat=DEFAULT_ADDITIONAL_ACM_STEP_REPEAT,
        pre_reset_acm_step_repeat=DEFAULT_PRE_RESET_ACM_STEP_REPEAT,
    )
    detector_manifest["corrections"] = {
        "table": str(correction_path),
        "fit_summary": str(correction_summary_path),
        "count_corrected_statistics": str(corrected_path),
        "correction_space": "counts",
    }
    manifest["detectors"][detector] = detector_manifest
    write_json(run_dir / "pipeline_manifest.json", manifest)


def main():
    source = DEFAULT_S3_SOURCE
    output_root = DEFAULT_RUN_ROOT
    run_date = utc_run_date()
    run_dir = output_root / run_date
    cache_dir = output_root / "_s3_cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pipeline": "run_wfi_reduction_pipeline.py",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "source": source,
        "source_directory": str(cache_dir),
        "output_directory": str(run_dir),
        "command": [str(item) for item in sys.argv],
        "mast_token": str(DEFAULT_MAST_TOKEN),
        "mast_index": str(DEFAULT_MAST_INDEX),
        "mast_supplement": True,
        "photodiode_channel": DEFAULT_PHOTODIODE_CHANNEL,
        "tile_sizes": list(DEFAULT_TILE_SIZES),
        "correction_settings": {
            "reference_led": 3,
            "minimum_day": 1.0,
            "additional_acm_step_repeat": DEFAULT_ADDITIONAL_ACM_STEP_REPEAT,
            "pre_reset_acm_step_repeat": DEFAULT_PRE_RESET_ACM_STEP_REPEAT,
            "ice_reset_day": DEFAULT_ICE_RESET_DAY,
        },
        "detectors": {},
    }
    write_json(run_dir / "pipeline_manifest.json", manifest)

    sync_s3_source(source, cache_dir)
    refresh_index(
        DEFAULT_MAST_INDEX,
        token_path=DEFAULT_MAST_TOKEN,
        detectors=DEFAULT_DETECTORS,
    )
    index_rows = read_index_rows(DEFAULT_MAST_INDEX, require_program=True)
    index_metadata = read_optional_index(DEFAULT_MAST_INDEX)
    shutil.copy2(DEFAULT_MAST_INDEX, run_dir / DEFAULT_MAST_INDEX.name)

    telemetry = fetch_photodiode_telemetry(
        index_rows,
        DEFAULT_MAST_TOKEN,
        DEFAULT_PHOTODIODE_CHANNEL,
    )
    telemetry_path = run_dir / "telemetry" / "photodiode_pd1.csv"
    write_telemetry(telemetry_path, telemetry)
    manifest["telemetry"] = {
        "source": telemetry["source"],
        "channel": telemetry["channel"],
        "mnemonic": telemetry["mnemonic"],
        "path": str(telemetry_path),
        "samples": len(telemetry["values"]),
    }

    missions = login_to_mast(DEFAULT_MAST_TOKEN)
    fetch_image = lambda uri: stream_mast_image(missions, uri)
    local_records = scan_products(cache_dir, allow_empty=True)
    local_detectors = {record["detector"] for record in local_records}
    mast_detectors = {row["detector"].strip() for row in index_rows}
    detectors = sorted(local_detectors | mast_detectors)
    if not detectors:
        raise ValueError("No local or MAST calibrated products were found")
    manifest["detectors_requested"] = detectors
    manifest["detectors_available"] = detectors
    write_json(run_dir / "pipeline_manifest.json", manifest)
    print(
        f"Reducing {', '.join(detectors)} from {cache_dir} with the MAST supplement "
        f"into {run_dir}",
        flush=True,
    )
    for detector in detectors:
        run_detector(
            cache_dir,
            run_dir,
            output_root,
            detector,
            index_metadata,
            index_rows,
            telemetry,
            fetch_image,
            manifest,
        )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "pipeline_manifest.json", manifest)
    print(f"Wrote pipeline manifest {run_dir / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    main()
