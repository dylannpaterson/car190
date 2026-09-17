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


ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "all_wfi_rcs_flat_index.csv"
DEFAULT_OUTPUT_DIR = ROOT / "wfi_tile_log_statistics_512"
DEFAULT_TOKEN = Path.home() / "Downloads" / "mast_api_token.txt"
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


def read_index(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No MAST index rows found in {path}")
    required = {"fileSetName", "detector", "led", "exposure_start_time"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"MAST index is missing columns: {sorted(missing)}")
    return rows


def detector_observations(rows, detector):
    selected = [row for row in rows if row["detector"] == detector]
    if not selected:
        raise ValueError(f"No MAST index rows found for {detector}")
    global_start = min(
        parse_mast_time(row["exposure_start_time"]) for row in rows
    )
    observations = []
    counts = []
    for led in range(1, 7):
        led_rows = sorted(
            [row for row in selected if int(row["led"]) == led],
            key=lambda row: parse_mast_time(row["exposure_start_time"]),
        )
        counts.append(len(led_rows))
        if len(led_rows) < 3:
            raise ValueError(f"{detector} LED {led} has too few repeats")
        for repeat, row in enumerate(led_rows):
            observations.append(
                {
                    "led": led,
                    "repeat": repeat,
                    "group": (led - 1) * len(led_rows) + repeat,
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
    if len(set(counts)) != 1:
        raise ValueError(f"{detector} LEDs have unequal repeat counts: {counts}")
    repeats = counts[0]
    for row in observations:
        baseline = observations[(row["led"] - 1) * repeats]
        row["baseline_file_set_name"] = baseline["file_set_name"]
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
    url = f"{base_url}?product_name={quote(uri, safe='')}"
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
                if row["repeat"] == 0:
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
                    if row["repeat"] != 0 and current_data is not None:
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
        if args.resume and output_path.exists():
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
        write_detector_statistics(
            observations,
            lambda uri: stream_mast_image(
                missions,
                uri,
                args.chunk_size,
                args.retries,
                args.request_timeout,
            ),
            output_path,
            args.tile_size,
            resume=args.resume,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
