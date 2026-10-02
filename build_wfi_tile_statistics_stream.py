"""Stream Roman WFI calibrated products from MAST into tile statistics.

Only the current LED baseline and one current exposure are held in memory at
once. The downloaded ASDF payload is kept in a BytesIO buffer and is released
after its data and DQ arrays have been copied out. No raw MAST product is
written to disk. Completed exposure statistics are checkpointed in
per-detector part files so interrupted runs can resume without redownloading
them.
"""

import argparse
import csv
import io
import os
from pathlib import Path
import time
from urllib.parse import quote
import warnings

import numpy as np
from requests.exceptions import RequestException

from analyze_wfi01_all_pixel_soc_model import parse_mast_time
from build_wfi01_tile_statistics import FIELDS, tile_sufficient_statistics


COUNT_FIELDS = tuple(
    field
    for field in FIELDS
    if field
    in {
        "sum_reference_counts",
        "sum_squared_reference_counts",
        "sum_current_counts",
        "sum_squared_current_counts",
        "sum_reference_current_counts",
        "min_reference_count",
        "max_reference_count",
        "min_current_count",
        "max_current_count",
    }
)
LEGACY_FIELDS = tuple(field for field in FIELDS if field not in COUNT_FIELDS)


ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "all_wfi_rcs_flat_index.csv"
DEFAULT_OUTPUT_DIR = ROOT / "wfi_tile_log_statistics_512"
DEFAULT_TOKEN = Path.home() / "Downloads" / "mast_api_token.txt"
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
RCS_PROGRAM = "1039"


def read_index(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No MAST index rows found in {path}")
    required = {
        "fileSetName",
        "detector",
        "led",
        "exposure_start_time",
        "visit_id",
        "program",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"MAST index is missing columns: {sorted(missing)}")
    programs = sorted({str(row["program"]).strip() for row in rows})
    if programs != [RCS_PROGRAM]:
        raise ValueError(
            f"MAST index must contain only RCS program {RCS_PROGRAM}; "
            f"found programs: {programs}"
        )
    return rows


def detector_observations(rows, detector):
    selected = [row for row in rows if row["detector"] == detector]
    if not selected:
        raise ValueError(f"No MAST index rows found for {detector}")
    programs = sorted(
        {str(row.get("program", RCS_PROGRAM)).strip() for row in selected}
    )
    if programs != [RCS_PROGRAM]:
        raise ValueError(
            f"{detector} observations must use RCS program {RCS_PROGRAM}; "
            f"found programs: {programs}"
        )
    global_start = min(
        parse_mast_time(row["exposure_start_time"]) for row in rows
    )

    visit_groups = {}
    for row in selected:
        visit_id = str(row.get("visit_id", "")).strip()
        if len(visit_id) <= 4:
            raise ValueError(
                f"{detector} has an invalid visit_id for {row['fileSetName']}: "
                f"{visit_id!r}"
            )
        # The final four visit-id digits identify the LED/exposure product;
        # the prefix is the shared temporal visit sequence.
        sequence_key = visit_id[:-4]
        visit_groups.setdefault(sequence_key, []).append(row)

    sequence_groups = []
    for sequence_key, sequence_rows in visit_groups.items():
        sequence_rows.sort(
            key=lambda row: (
                parse_mast_time(row["exposure_start_time"]),
                int(row["led"]),
                row["fileSetName"],
            )
        )
        sequence_days = np.mean(
            [
                (
                    parse_mast_time(row["exposure_start_time"]) - global_start
                ).total_seconds()
                / 86400.0
                for row in sequence_rows
            ]
        )
        sequence_groups.append((sequence_days, sequence_key, sequence_rows))
    sequence_groups.sort(key=lambda item: (item[0], item[1]))

    observations = []
    led_counts = {led: 0 for led in range(1, 7)}
    for repeat, (_, _, sequence_rows) in enumerate(sequence_groups):
        for row in sequence_rows:
            led = int(row["led"])
            if led not in led_counts:
                raise ValueError(f"{detector} has invalid LED number: {led}")
            led_counts[led] += 1
            observations.append(
                {
                    "led": led,
                    "repeat": repeat,
                    "group": len(observations),
                    "file_set_name": row["fileSetName"],
                    "baseline_file_set_name": "",
                    "exposure_start_time": row["exposure_start_time"],
                    "days": (
                        parse_mast_time(row["exposure_start_time"]) - global_start
                    ).total_seconds()
                    / 86400.0,
                    "cal_uri": row.get("cal_uri")
                    or f'{row["fileSetName"]}/{row["fileSetName"]}_cal.asdf',
                }
            )
    for led, count in led_counts.items():
        if count < 3:
            raise ValueError(f"{detector} LED {led} has too few epochs")
    baselines = {}
    for row in observations:
        baselines.setdefault(row["led"], row["file_set_name"])
    for row in observations:
        row["baseline_file_set_name"] = baselines[row["led"]]
    return observations


def load_asdf_buffer(buffer, source_name):
    try:
        import asdf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Streaming MAST statistics require the 'asdf' package"
        ) from error

    buffer.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with asdf.open(
            buffer,
            lazy_load=True,
            ignore_unrecognized_tag=True,
        ) as af:
            model = af.tree["roman"]
            data = np.array(model["data"], copy=True)
            dq = np.array(model["dq"], copy=True)
    if data.shape != dq.shape:
        raise ValueError(f"Data/DQ shape mismatch in {source_name}")
    return data, dq


def stream_mast_image(
    missions,
    uri,
    chunk_size=DEFAULT_CHUNK_SIZE,
    retries=3,
    request_timeout=600,
):
    """Download one MAST ASDF product into memory and return data plus DQ."""
    if chunk_size < 1 or retries < 0 or request_timeout <= 0:
        raise ValueError("Invalid streaming download settings")
    base_url = (
        missions._service_api_connection.MISSIONS_DOWNLOAD_URL
        + "roman/api/v0.1/retrieve_product"
    )
    # Roman's retrieve endpoint accepts the product filename, not its
    # directory-qualified MAST URI.
    product_name = Path(uri).name
    url = f"{base_url}?product_name={quote(product_name, safe='')}"
    for attempt in range(retries + 1):
        try:
            with missions._session.get(
                url,
                stream=True,
                timeout=request_timeout,
            ) as response:
                response.raise_for_status()
                with io.BytesIO() as buffer:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            buffer.write(chunk)
                    return load_asdf_buffer(buffer, uri)
        except RequestException:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 16))


def login_to_mast(token_path):
    try:
        from astroquery.mast import MastMissions
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Streaming MAST statistics require the 'astroquery' package"
        ) from error
    token = token_path.read_text().strip()
    if not token:
        raise ValueError(f"MAST token file is empty: {token_path}")
    missions = MastMissions(mission="roman")
    missions.login(token=token)
    return missions


def detector_part_path(parts_dir, group):
    return parts_dir / f"group_{int(group):04d}.csv"


def write_detector_statistics(
    observations,
    fetch_image,
    output_path,
    tile_size,
    resume=True,
):
    """Build one detector's CSV using an injected image-fetch function."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and output_path.exists():
        with output_path.open(newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
        missing = set(FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(
                f"Existing statistics file is missing fields {sorted(missing)}; "
                "rerun with --no-resume to rebuild it"
            )
        print(f"Skipping completed detector output: {output_path}")
        return output_path

    parts_dir = output_path.with_name(output_path.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    for led in range(1, 7):
        led_rows = [row for row in observations if row["led"] == led]
        pending_rows = [
            row
            for row in led_rows
            if not (resume and detector_part_path(parts_dir, row["group"]).exists())
        ]
        if not pending_rows:
            print(f"Skipping completed LED {led} in {output_path}")
            continue

        baseline_data, baseline_dq = fetch_image(led_rows[0]["cal_uri"])
        try:
            for row in pending_rows:
                current_data = current_dq = None
                if row["file_set_name"] == row["baseline_file_set_name"]:
                    current_data, current_dq = baseline_data, baseline_dq
                else:
                    current_data, current_dq = fetch_image(row["cal_uri"])
                part_path = detector_part_path(parts_dir, row["group"])
                temporary_part = part_path.with_name(part_path.name + ".partial")
                try:
                    with temporary_part.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                        tiles = tile_sufficient_statistics(
                            baseline_data,
                            baseline_dq,
                            current_data,
                            current_dq,
                            tile_size,
                        )
                        last_tile = None
                        for tile in tiles:
                            last_tile = tile
                            writer.writerow(
                                {
                                    "led": row["led"],
                                    "repeat": row["repeat"],
                                    "group": row["group"],
                                    "file_set_name": row["file_set_name"],
                                    "baseline_file_set_name": row["baseline_file_set_name"],
                                    "exposure_start_time": row["exposure_start_time"],
                                    "days": row["days"],
                                    **tile,
                                }
                            )
                        if last_tile is None:
                            raise RuntimeError(
                                "Tile statistic generator returned no tiles"
                            )
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary_part.replace(part_path)
                finally:
                    if (
                        row["file_set_name"] != row["baseline_file_set_name"]
                        and current_data is not None
                    ):
                        del current_data, current_dq
                print(
                    f"{row['file_set_name']}: wrote {last_tile['tile_id'] + 1} tiles"
                )
        finally:
            del baseline_data, baseline_dq

    missing = [
        row["group"]
        for row in observations
        if not detector_part_path(parts_dir, row["group"]).exists()
    ]
    if missing:
        raise RuntimeError(f"Missing completed exposure parts: {missing}")

    temporary_path = output_path.with_name(output_path.name + ".partial")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for observation in observations:
            part_path = detector_part_path(parts_dir, observation["group"])
            with part_path.open(newline="") as part_handle:
                writer.writerows(csv.DictReader(part_handle))
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(output_path)
    return output_path


def incremental_part_path(parts_dir, file_set_name):
    """Return a safe resumable part path keyed by the MAST file-set name."""
    if not file_set_name or Path(file_set_name).name != file_set_name:
        raise ValueError(f"Unsafe MAST file-set name: {file_set_name!r}")
    return parts_dir / f"{file_set_name}.csv"


def read_statistics_rows(path, return_fieldnames=False):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Statistics file has no header: {path}")
        fieldnames = tuple(reader.fieldnames)
        missing = set(LEGACY_FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(
                f"Statistics file is missing fields {sorted(missing)}: {path}"
            )
        present_count_fields = set(fieldnames).intersection(COUNT_FIELDS)
        if present_count_fields and present_count_fields != set(COUNT_FIELDS):
            raise ValueError(
                f"Statistics file has an incomplete count-moment schema: {path}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Statistics file has no rows: {path}")
    if return_fieldnames:
        return rows, fieldnames
    return rows


def update_statistics_metadata(row, observation):
    updated = dict(row)
    for field in (
        "led",
        "repeat",
        "group",
        "file_set_name",
        "baseline_file_set_name",
        "exposure_start_time",
        "days",
    ):
        updated[field] = observation[field]
    return updated


def write_statistics_rows(output_path, rows, fieldnames=None):
    """Atomically replace a statistics file with already-computed rows."""
    fieldnames = tuple(FIELDS if fieldnames is None else fieldnames)
    temporary_path = output_path.with_name(output_path.name + ".partial")
    try:
        with temporary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_incremental_observation(
    observation,
    fetch_image,
    part_path,
    tile_size,
    baseline_data,
    baseline_dq,
):
    """Write one newly observed exposure without touching old statistics."""
    current_data = current_dq = None
    if observation["file_set_name"] == observation["baseline_file_set_name"]:
        current_data, current_dq = baseline_data, baseline_dq
    else:
        current_data, current_dq = fetch_image(observation["cal_uri"])

    temporary_part = part_path.with_name(part_path.name + ".partial")
    try:
        with temporary_part.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            last_tile = None
            tiles = tile_sufficient_statistics(
                baseline_data,
                baseline_dq,
                current_data,
                current_dq,
                tile_size,
            )
            for tile in tiles:
                last_tile = tile
                writer.writerow(
                    {
                        "led": observation["led"],
                        "repeat": observation["repeat"],
                        "group": observation["group"],
                        "file_set_name": observation["file_set_name"],
                        "baseline_file_set_name": observation[
                            "baseline_file_set_name"
                        ],
                        "exposure_start_time": observation["exposure_start_time"],
                        "days": observation["days"],
                        **tile,
                    }
                )
            if last_tile is None:
                raise RuntimeError("Tile statistic generator returned no tiles")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_part.replace(part_path)
    finally:
        if temporary_part.exists():
            temporary_part.unlink()
        if (
            observation["file_set_name"] != observation["baseline_file_set_name"]
            and current_data is not None
        ):
            del current_data, current_dq
    return last_tile


def update_detector_statistics_incremental(
    observations,
    fetch_image,
    output_path,
    tile_size,
):
    """Add only file sets absent from an existing detector statistics file.

    Existing sufficient statistics are reused by file-set name.  This matters
    when MAST inserts a newly discovered exposure into the middle of a LED's
    time sequence and therefore changes every later numeric group label.
    """
    if not output_path.exists():
        raise ValueError(
            f"Incremental update requires an existing statistics file: {output_path}"
        )
    existing_rows, existing_fieldnames = read_statistics_rows(
        output_path, return_fieldnames=True
    )
    existing_by_file_set = {}
    for row in existing_rows:
        file_set_name = row["file_set_name"]
        tile_id = row["tile_id"]
        by_tile = existing_by_file_set.setdefault(file_set_name, {})
        if tile_id in by_tile:
            raise ValueError(
                f"Duplicate tile {tile_id} for {file_set_name} in {output_path}"
            )
        by_tile[tile_id] = row

    current_by_file_set = {row["file_set_name"]: row for row in observations}
    if len(current_by_file_set) != len(observations):
        raise ValueError("The MAST index contains duplicate file-set names")
    removed = set(existing_by_file_set).difference(current_by_file_set)
    if removed:
        raise ValueError(
            "Incremental update refuses to discard existing observations: "
            + ", ".join(sorted(removed)[:3])
        )

    new_observations = [
        row for row in observations if row["file_set_name"] not in existing_by_file_set
    ]
    if not new_observations:
        output_rows = []
        for observation in observations:
            for row in existing_by_file_set[observation["file_set_name"]].values():
                output_rows.append(update_statistics_metadata(row, observation))
        write_statistics_rows(output_path, output_rows, existing_fieldnames)
        print(
            f"No new observations for {output_path}; refreshed metadata for "
            f"{len(observations)} existing observations"
        )
        return output_path

    parts_dir = output_path.with_name(output_path.name + ".incremental.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    new_rows_by_file_set = {}
    for observation in new_observations:
        part_path = incremental_part_path(parts_dir, observation["file_set_name"])
        if part_path.exists():
            rows = read_statistics_rows(part_path)
            new_rows_by_file_set[observation["file_set_name"]] = rows

    for led in range(1, 7):
        led_rows = [row for row in new_observations if row["led"] == led]
        pending_rows = [
            row
            for row in led_rows
            if row["file_set_name"] not in new_rows_by_file_set
        ]
        if not pending_rows:
            continue

        all_led_rows = [row for row in observations if row["led"] == led]
        baseline_data, baseline_dq = fetch_image(all_led_rows[0]["cal_uri"])
        try:
            for observation in pending_rows:
                part_path = incremental_part_path(parts_dir, observation["file_set_name"])
                last_tile = write_incremental_observation(
                    observation,
                    fetch_image,
                    part_path,
                    tile_size,
                    baseline_data,
                    baseline_dq,
                )
                new_rows_by_file_set[observation["file_set_name"]] = read_statistics_rows(
                    part_path
                )
                print(
                    f"{observation['file_set_name']}: wrote {last_tile['tile_id'] + 1} tiles"
                )
        finally:
            del baseline_data, baseline_dq

    output_rows = []
    for observation in observations:
        file_set_name = observation["file_set_name"]
        if file_set_name in existing_by_file_set:
            rows = existing_by_file_set[file_set_name].values()
        else:
            rows = new_rows_by_file_set.get(file_set_name)
            if rows is None:
                raise RuntimeError(
                    f"Missing incremental statistics for {file_set_name}"
                )
        for row in rows:
            output_rows.append(update_statistics_metadata(row, observation))

    output_rows = [
        {field: row[field] for field in existing_fieldnames} for row in output_rows
    ]
    write_statistics_rows(output_path, output_rows, existing_fieldnames)
    print(
        f"Updated {output_path} with {len(new_observations)} new observations; "
        f"reused {len(observations) - len(new_observations)} existing observations"
    )
    return output_path


def output_path_for_detector(output_dir, detector, tile_size):
    return output_dir / f"{detector.lower()}_tile_log_statistics_{tile_size}.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--all-detectors", action="store_true")
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Recompute existing exposure parts instead of resuming them.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse existing detector statistics and download only new MAST file sets.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_size < 1 or args.chunk_size < 1:
        raise ValueError("Tile and download chunk sizes must be positive")
    if args.retries < 0 or args.request_timeout <= 0.0:
        raise ValueError("Retries must be non-negative and timeout must be positive")
    if args.all_detectors and args.output_path is not None:
        raise ValueError("--output-path cannot be combined with --all-detectors")
    if args.incremental and not args.resume:
        raise ValueError("--incremental cannot be combined with --no-resume")
    rows = read_index(args.index_path)
    available = sorted({row["detector"] for row in rows})
    if args.all_detectors:
        detectors = available
    else:
        detectors = [args.detector]
        if args.detector not in available:
            raise ValueError(
                f"Unknown detector {args.detector}; available detectors: {available}"
            )

    jobs = []
    for detector in detectors:
        observations = detector_observations(rows, detector)
        output_path = args.output_path or output_path_for_detector(
            args.output_dir,
            detector,
            args.tile_size,
        )
        if args.incremental and not output_path.exists():
            raise ValueError(
                f"Incremental update requires an existing statistics file: {output_path}"
            )
        if args.resume and output_path.exists() and not args.incremental:
            print(f"Skipping completed detector output: {output_path}")
            continue
        jobs.append((detector, observations, output_path))

    if not jobs:
        print("All requested detector outputs are complete; nothing to download.")
        return

    missions = login_to_mast(args.token_file)
    for detector, observations, output_path in jobs:
        print(
            f"Streaming {detector}: {len(observations)} exposures, "
            f"output={output_path}"
        )
        fetch_image = lambda uri: stream_mast_image(
            missions,
            uri,
            args.chunk_size,
            args.retries,
            args.request_timeout,
        )
        if args.incremental:
            update_detector_statistics_incremental(
                observations,
                fetch_image,
                output_path,
                args.tile_size,
            )
        else:
            write_detector_statistics(
                observations,
                fetch_image,
                output_path,
                args.tile_size,
                resume=args.resume,
            )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
