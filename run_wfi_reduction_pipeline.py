#!/usr/bin/env python3
"""Fully automated CAR190 reduction entry point.

The entry point has no required arguments. It lists calibrated ASDF products
from S3, streams each image into memory, refreshes the all-detector MAST
supplement and photodiode telemetry, reduces every detector, fits the
count-space gain/ACM correction table, and writes a second statistics CSV with
that correction applied. The raw statistics plus correction table remain the
canonical inputs for the TMM fit; the count-corrected CSV is an explicit
inspection and downstream-reduction product.
"""

import atexit
import csv
from datetime import datetime, timezone
import json
import os
import shlex
import io
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote, urlparse
import warnings

import numpy as np

try:
    from requests.exceptions import RequestException
except ModuleNotFoundError:
    class RequestException(Exception):
        pass


FIELDS = (
    "led",
    "repeat",
    "group",
    "file_set_name",
    "baseline_file_set_name",
    "exposure_start_time",
    "days",
    "tile_id",
    "tile_row",
    "tile_col",
    "row_start",
    "row_stop_exclusive",
    "col_start",
    "col_stop_exclusive",
    "tile_n_rows",
    "tile_n_cols",
    "n_total",
    "n_pair_valid",
    "sum_reference_counts",
    "sum_squared_reference_counts",
    "sum_current_counts",
    "sum_squared_current_counts",
    "sum_reference_current_counts",
    "min_reference_count",
    "max_reference_count",
    "min_current_count",
    "max_current_count",
    "sum_log_change",
    "sum_squared_log_change",
    "min_log_change",
    "max_log_change",
)

FILE_NAME_PATTERN = re.compile(
    r"^r(?P<visit_id>[0-9]+)_[^_]+_wfi(?P<detector>[0-9]{2})_"
    r"(?P<optical_element>[^_]+)_cal\.asdf$",
    re.IGNORECASE,
)
PHOTODIODE_MNEMONICS = {
    "pd1": "WFI_MCE_SRCS_PD1_V",
    "pd2": "WFI_MCE_SRCS_PD2_V",
}

INDEX_FIELDS = [
    "fileSetName",
    "exposure_type",
    "optical_element",
    "detector",
    "exposure_start_time",
    "exposure_end_time",
    "exposure_time",
    "effective_exposure_time",
    "led",
    "bank",
    "counts",
    "program",
    "observation_id",
    "visit_id",
    "ArchiveFileID",
    "product_type",
    "productLevel",
    "image_median",
    "image_rms",
    "good_pixel_fraction",
    "target_aperture",
    "access",
    "cal_uri",
]
INDEX_SELECT_COLUMNS = INDEX_FIELDS[:-1]
DEFAULT_DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_PROGRAM = "1039"

CORRECTION_FIELDS = (
    "detector",
    "tile_id",
    "group",
    "days",
    "correction_space",
    "count_log_correction",
    "log_gain_correction",
    "acm_step_log_offset",
    "pre_reset_acm_step_log_offset",
    "additional_acm_step_log_offset",
    "late_acm_step_log_offset",
)
SUMMARY_FIELDS = (
    "detector",
    "tile_id",
    "reference_led",
    "minimum_day_exclusive",
    "reference_period_days",
    "reference_rmse_counts",
    "reference_r2",
    "post_reset_period_days",
    "post_reset_rmse_counts",
    "post_reset_r2",
    "reference_acm_step_counts",
    "reference_pre_reset_acm_step_counts",
    "reference_additional_acm_step_counts",
    "reference_late_acm_step_counts",
)


ROOT = Path(__file__).resolve().parent
DEFAULT_S3_SOURCE = "s3://roman-s3-romcom/activities/CAR190/"
# A 4096x4096 detector produces 64 512x512 tiles. Do not emit 64x64 tiles.
DEFAULT_TILE_SIZES = (512,)
DEFAULT_RUN_ROOT = ROOT / "CAR190_reduction"
DEFAULT_MAST_TOKEN = ROOT / "mast_api_token.txt"
DEFAULT_MAST_INDEX = ROOT / "all_wfi_rcs_flat_index.csv"
DEFAULT_PHOTODIODE_CHANNEL = "pd1"
DEFAULT_ROMAN_TELEM_BINARY = "roman-telem"
DEFAULT_ICE_RESET_DAY = 15.064405023148149
DEFAULT_ADDITIONAL_ACM_STEP_REPEAT = 39
DEFAULT_PRE_RESET_ACM_STEP_REPEAT = 35
_LOG_HANDLE = None


def _close_pipeline_log():
    global _LOG_HANDLE
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.close()
        _LOG_HANDLE = None


def configure_pipeline_log(path):
    global _LOG_HANDLE
    _close_pipeline_log()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_HANDLE = path.open("a", buffering=1)
    return path


def status(message):
    """Print and persist a timestamped progress message."""
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.write(line + "\n")
        _LOG_HANDLE.flush()


atexit.register(_close_pipeline_log)


def utc_run_date(value=None):
    """Return a compact UTC date suitable for a dated output directory."""
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%d")
    parsed = datetime.strptime(str(value), "%Y%m%d")
    return parsed.strftime("%Y%m%d")


def is_s3_uri(value):
    return str(value).lower().startswith("s3://")


def _split_s3_uri(source_uri):
    source_uri = str(source_uri)
    if not is_s3_uri(source_uri):
        raise ValueError(f"Expected an s3:// source URI, got {source_uri!r}")
    parsed = urlparse(source_uri)
    if not parsed.netloc:
        raise ValueError(f"S3 URI is missing a bucket: {source_uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def list_s3_products(source_uri, aws_binary="aws", aws_profile=None):
    """List calibrated S3 products without downloading any image data."""
    bucket, prefix = _split_s3_uri(source_uri)
    command = [str(aws_binary)]
    if aws_profile:
        command.extend(("--profile", str(aws_profile)))
    command.extend(
        (
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--output",
            "json",
        )
    )
    status(f"Listing S3 products without downloading images: {shlex.join(command)}")
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as error:
        raise RuntimeError("AWS returned invalid JSON while listing S3 products") from error
    records = []
    seen = {}
    for item in payload.get("Contents", []):
        key = str(item.get("Key", ""))
        record = parse_product_name(Path(key))
        if record is None:
            continue
        previous = seen.get(record["file_set_name"])
        if previous is not None:
            raise ValueError(
                f"Duplicate S3 file-set name for {record['file_set_name']}: "
                f"{previous} and {key}"
            )
        record.update(
            {
                "path": None,
                "cal_uri": f"s3://{bucket}/{key}",
                "s3_uri": f"s3://{bucket}/{key}",
                "source": "s3",
                "source_signature": ":".join(
                    str(item.get(field, ""))
                    for field in ("Size", "ETag", "LastModified")
                ),
            }
        )
        seen[record["file_set_name"]] = key
        records.append(record)
    if not records:
        raise ValueError(f"No calibrated S3 products found under {source_uri}")
    status(f"Listed {len(records)} calibrated S3 products; image data remains remote")
    return records


def parse_product_name(path):
    """Derive detector, LED, and epoch metadata from a calibrated filename."""
    match = FILE_NAME_PATTERN.match(path.name)
    if match is None:
        return None
    visit_id = match.group("visit_id")
    if len(visit_id) < 5 or visit_id[-3:] != "001":
        raise ValueError(f"Unexpected visit suffix in {path.name}: {visit_id}")
    led_code = visit_id[-4]
    if not led_code.isdigit() or not 2 <= int(led_code) <= 7:
        raise ValueError(f"Cannot derive LED from {path.name}")
    epoch_key = visit_id[:-4]
    if not epoch_key:
        raise ValueError(f"Cannot derive observation epoch from {path.name}")
    file_set_name = path.name[: -len("_cal.asdf")]
    return {
        "path": path,
        "file_name": path.name,
        "file_set_name": file_set_name,
        "visit_id": visit_id,
        "detector": f"WFI{int(match.group('detector')):02d}",
        "led": int(led_code) - 1,
        "epoch_key": epoch_key,
        "source": "rrn",
    }


def read_index_rows(path, require_program=False):
    """Read and validate the refreshed MAST supplement index."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    required = {"fileSetName", "detector", "led", "visit_id", "exposure_start_time"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Index is missing columns {sorted(missing)}: {path}")
    if not rows:
        raise ValueError(f"Index contains no rows: {path}")
    seen = set()
    for row in rows:
        file_set_name = row["fileSetName"].strip()
        if not file_set_name:
            raise ValueError("Index contains an empty fileSetName")
        if file_set_name in seen:
            raise ValueError(f"Index contains duplicate fileSetName: {file_set_name}")
        seen.add(file_set_name)
    if require_program:
        programs = sorted({str(row.get("program", "")).strip() for row in rows})
        if programs != [DEFAULT_PROGRAM]:
            raise ValueError(
                f"MAST index must contain only RCS program {DEFAULT_PROGRAM}; "
                f"found programs: {programs}"
            )
    return rows


def read_optional_index(path):
    """Index metadata keyed by both file-set name and calibrated filename."""
    if path is None:
        return {}
    metadata = {}
    for row in read_index_rows(path):
        file_set_name = row["fileSetName"].strip()
        file_name = Path(row.get("cal_uri", "")).name
        if not file_name:
            file_name = f"{file_set_name}_cal.asdf"
        for key in (file_set_name, file_name):
            if key in metadata and metadata[key]["fileSetName"] != file_set_name:
                raise ValueError(f"Index contains duplicate product key {key}")
            metadata[key] = row
    return metadata


def mast_records_from_index(rows, detector=None, allow_empty=False):
    """Create remote product records for missing MAST supplement products."""
    records = []
    seen = set()
    for row in rows:
        row_detector = row["detector"].strip()
        if detector is not None and row_detector != detector:
            continue
        file_set_name = row["fileSetName"].strip()
        file_name = Path(row.get("cal_uri", "")).name
        if not file_name:
            file_name = f"{file_set_name}_cal.asdf"
        visit_id = str(row["visit_id"]).strip()
        led = int(row["led"])
        if not visit_id or led not in range(1, 7):
            raise ValueError(f"Invalid MAST metadata for {file_set_name}")
        if file_set_name in seen:
            raise ValueError(f"Duplicate MAST supplement product {file_set_name}")
        seen.add(file_set_name)
        records.append(
            {
                "path": None,
                "file_name": file_name,
                "file_set_name": file_set_name,
                "visit_id": visit_id,
                "detector": row_detector,
                "led": led,
                "epoch_key": visit_id[:-4],
                "cal_uri": row.get("cal_uri", "").strip()
                or f"{file_set_name}/{file_name}",
                "source": "mast",
            }
        )
    if detector is not None and not records and not allow_empty:
        raise ValueError(f"No MAST supplement products found for {detector}")
    return records


def merge_product_records(local_records, mast_records):
    """Prefer the S3 product when a product exists in both remote sources."""
    merged = []
    by_file_set = {}
    for record in local_records:
        key = record["file_set_name"]
        if key in by_file_set:
            raise ValueError(f"Duplicate S3 product {key}")
        by_file_set[key] = record
        merged.append(record)
    for record in mast_records:
        key = record["file_set_name"]
        local = by_file_set.get(key)
        if local is not None:
            if (
                local["detector"] != record["detector"]
                or local["led"] != record["led"]
                or local["visit_id"] != record["visit_id"]
            ):
                raise ValueError(f"S3/MAST metadata mismatch for {key}")
            continue
        by_file_set[key] = record
        merged.append(record)
    return merged


def load_asdf_buffer(buffer, source_name):
    try:
        import asdf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Remote product streaming requires the 'asdf' package") from error
    buffer.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with asdf.open(buffer, lazy_load=True, ignore_unrecognized_tag=True) as handle:
            model = handle.tree["roman"]
            data = np.array(model["data"], copy=True)
            dq = np.array(model["dq"], copy=True)
    if data.shape != dq.shape:
        raise ValueError(f"Data/DQ shape mismatch in {source_name}")
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D detector image in {source_name}, found {data.shape}")
    return data, dq


def login_to_mast(token_path):
    status("Logging in to MAST for product downloads")
    try:
        from astroquery.mast import MastMissions
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("MAST supplementation requires astroquery") from error
    token = _read_mast_token(token_path)
    missions = MastMissions(mission="roman")
    missions.login(token=token)
    return missions


def stream_mast_image(missions, uri, chunk_size=8 * 1024 * 1024, retries=3, request_timeout=600):
    service = missions._service_api_connection
    missions_url = getattr(service, "MISSIONS_DOWNLOAD_URL", None)
    if missions_url is not None:
        base_url = missions_url.rstrip("/") + "/roman/api/v0.1/retrieve_product"
    else:
        download_url = getattr(service, "MAST_DOWNLOAD_URL", None)
        if download_url is None:
            raise RuntimeError("MAST service exposes no Roman product download URL")
        service_root = str(download_url).split("/api/v0.1/", 1)[0].rstrip("/")
        base_url = service_root + "/search/roman/api/v0.1/retrieve_product"
    url = f"{base_url}?product_name={quote(Path(uri).name, safe='')}"
    for attempt in range(retries + 1):
        try:
            with missions._session.get(url, stream=True, timeout=request_timeout) as response:
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
    raise RuntimeError(f"MAST download failed for {uri}")


def stream_s3_image(uri, aws_binary="aws", aws_profile=None):
    """Stream one S3 product to memory and never create a local image file."""
    command = [str(aws_binary)]
    if aws_profile:
        command.extend(("--profile", str(aws_profile)))
    command.extend(("s3", "cp", str(uri), "-", "--only-show-errors"))
    status(f"Streaming S3 product to memory: {uri}")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"S3 product download failed for {uri}: {error}")
    return load_asdf_buffer(io.BytesIO(result.stdout), str(uri))


def parse_iso_time(value):
    value = str(value).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _read_mast_token(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"MAST token file does not exist: {path}")
    token = path.read_text().strip()
    if token.startswith("MAST_API_TOKEN="):
        token = token.split("=", 1)[1].strip()
    if not token:
        raise ValueError(f"MAST token file is empty: {path}")
    return token


def load_ice_monitor_photodiode_summary(path, channel):
    """Load and validate dark-subtracted per-exposure ``roman-telem`` output."""
    if channel not in PHOTODIODE_MNEMONICS:
        raise ValueError(f"Unsupported photodiode channel {channel!r}")
    path = Path(path)
    required = {
        "mnemonic",
        "step",
        "start",
        "end",
        "signal",
        "dark_median",
        "dark_subtracted",
        "n_samples",
        "ref_pass",
        "ref_signal",
        "change_pct",
    }
    values = {}
    mnemonic = PHOTODIODE_MNEMONICS[channel]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or ()
        missing = required.difference(fields)
        led_field = "led" if "led" in fields else "led_band"
        if led_field not in fields:
            missing.add("led or led_band")
        if missing:
            raise ValueError(f"Ice-monitor summary is missing fields: {sorted(missing)}")
        for row in reader:
            if row["mnemonic"] != mnemonic or not row["step"].startswith("band"):
                continue
            if str(row["dark_subtracted"]).strip().lower() not in {"true", "1"}:
                raise ValueError(f"Ice-monitor row is not dark-subtracted: {row}")
            start = parse_iso_time(row["start"])
            end = parse_iso_time(row["end"])
            led = int(float(row[led_field]))
            signal = float(row["signal"])
            dark = float(row["dark_median"])
            reference = float(row["ref_signal"])
            change_pct = float(row["change_pct"])
            if not np.isfinite(signal) or signal <= 0.0:
                raise ValueError(f"Dark-subtracted photodiode signal must be positive: {row}")
            if not np.isfinite(reference) or reference <= 0.0:
                raise ValueError(f"Ice-monitor reference signal must be positive: {row}")
            expected_change_pct = 100.0 * (signal / reference - 1.0)
            if not np.isclose(change_pct, expected_change_pct, rtol=1e-6, atol=1e-9):
                raise ValueError(f"Ice-monitor change_pct does not match signal/ref_signal: {row}")
            key = (start, end, led)
            if key in values:
                raise ValueError(f"Duplicate ice-monitor exposure: {key}")
            values[key] = {
                "signal": signal,
                "dark_median": dark,
                "n_samples": int(row["n_samples"]),
                "reference_pass": row["ref_pass"],
                "reference_signal": reference,
                "change_pct": change_pct,
            }
    if not values:
        raise ValueError(f"No {mnemonic} LED rows found in {path}")
    return {
        "source": "roman-telem-ice-monitor",
        "summary_path": str(path),
        "channel": channel,
        "mnemonic": mnemonic,
        "values": values,
    }


def run_ice_monitor_telemetry(index_rows, token_path, channel, output_path):
    """Run ``roman-telem`` and return its validated dark-subtracted summary."""
    if channel not in PHOTODIODE_MNEMONICS:
        raise ValueError(f"Unsupported photodiode channel {channel!r}")
    windows = []
    for row in index_rows:
        start_value = str(row.get("exposure_start_time", "")).strip()
        end_value = str(row.get("exposure_end_time", "")).strip()
        if not start_value or not end_value:
            raise ValueError("Photodiode correction requires indexed exposure windows")
        start = parse_iso_time(start_value)
        end = parse_iso_time(end_value)
        if end < start:
            raise ValueError(f"Exposure ends before it starts: {start_value} to {end_value}")
        windows.append((start, end))
    if not windows:
        raise ValueError("Photodiode correction has no exposure windows")
    query_start = min(start for start, _ in windows)
    query_end = max(end for _, end in windows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = Path(sys.executable).with_name(DEFAULT_ROMAN_TELEM_BINARY)
    binary = str(binary if binary.exists() else DEFAULT_ROMAN_TELEM_BINARY)
    command = [
        binary,
        "--ice-monitor",
        "--ice-program",
        DEFAULT_PROGRAM,
        "--start",
        query_start.isoformat(timespec="milliseconds") + "Z",
        "--end",
        query_end.isoformat(timespec="milliseconds") + "Z",
        "--output",
        str(output_path),
        "--quiet",
    ]
    token = _read_mast_token(token_path)
    environment = os.environ.copy()
    environment["MAST_API_TOKEN"] = token
    status(
        f"Running dark-subtracted ice-monitor telemetry for "
        f"{query_start.isoformat()} to {query_end.isoformat()}: {shlex.join(command)}"
    )
    subprocess.run(command, check=True, env=environment)
    telemetry = load_ice_monitor_photodiode_summary(output_path, channel)
    status(f"Validated {len(telemetry['values'])} dark-subtracted photodiode exposure rows")
    return telemetry


def attach_photodiode_corrections(observations, telemetry):
    """Apply the tool-selected dark-subtracted reference-pass scale per LED."""
    if telemetry.get("source") != "roman-telem-ice-monitor":
        raise ValueError("Production photodiode correction requires roman-telem --ice-monitor")
    values = telemetry["values"]
    for observation in observations:
        key = (
            parse_iso_time(observation["exposure_start_time"]),
            parse_iso_time(observation["exposure_end_time"]),
            int(observation["led"]),
        )
        try:
            value = values[key]
        except KeyError as error:
            raise ValueError(
                f"No dark-subtracted ice-monitor row for {observation['file_set_name']}: {key}"
            ) from error
        scale = value["reference_signal"] / value["signal"]
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"Invalid ice-monitor photodiode scale for {observation['file_set_name']}: {scale}"
            )
        observation.update(
            {
                "photodiode_channel": telemetry["channel"],
                "photodiode_source": telemetry["source"],
                "photodiode_mean": value["signal"],
                "photodiode_reference": value["reference_signal"],
                "photodiode_reference_pass": value["reference_pass"],
                "photodiode_scale": float(scale),
                "photodiode_n_samples": value["n_samples"],
                "photodiode_dark_median": value["dark_median"],
                "photodiode_dark_subtracted": True,
                "photodiode_change_pct": value["change_pct"],
            }
        )
    return observations


def query_mast_detector(missions, detector, page_size=5000):
    """Query one detector's current Roman L2 dark-flat products."""
    try:
        from astropy.table import vstack
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("MAST index refresh requires astropy") from error
    pages = []
    offset = 0
    while True:
        page = missions.query_criteria(
            exposure_type="WFI_FLAT",
            optical_element="DARK",
            detector=detector,
            program=DEFAULT_PROGRAM,
            product_type="l2",
            select_cols=INDEX_SELECT_COLUMNS,
            limit=page_size,
            offset=offset,
        )
        pages.append(page)
        if len(page) < page_size:
            break
        offset += len(page)
    if not pages or sum(len(page) for page in pages) == 0:
        return None
    return vstack(pages, metadata_conflicts="silent")


def table_to_index_rows(table):
    rows = []
    for row in table:
        file_set = str(row["fileSetName"])
        rows.append(
            {
                **{column: str(row[column]) for column in INDEX_SELECT_COLUMNS},
                "cal_uri": f"{file_set}/{file_set}_cal.asdf",
            }
        )
    return rows


def merge_index_rows(existing_rows, fresh_rows):
    merged = {}
    for row in existing_rows:
        file_set = row["fileSetName"].strip()
        if file_set in merged:
            raise ValueError(f"Duplicate existing MAST fileSetName: {file_set}")
        merged[file_set] = row
    added = 0
    updated = 0
    for row in fresh_rows:
        file_set = row["fileSetName"].strip()
        previous = merged.get(file_set)
        if previous is None:
            added += 1
        elif previous != row:
            updated += 1
        merged[file_set] = row
    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            row["exposure_start_time"],
            row["detector"],
            row["led"],
            row["fileSetName"],
        )
    )
    return rows, added, updated


def write_index_rows_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def refresh_index(output_path, token_path, detectors=DEFAULT_DETECTORS, page_size=5000):
    """Refresh the persistent all-detector MAST supplement index."""
    if page_size <= 0:
        raise ValueError("MAST index page size must be positive")
    status(f"Refreshing MAST index for {len(detectors)} detectors")
    token = _read_mast_token(token_path)
    try:
        from astroquery.mast import MastMissions
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("MAST index refresh requires astroquery") from error
    missions = MastMissions(mission="roman")
    missions.login(token=token)
    fresh_rows = []
    for detector in detectors:
        status(f"Querying MAST index for {detector}")
        table = query_mast_detector(missions, detector, page_size)
        count = 0 if table is None else len(table)
        status(f"Found {count} {detector} L2 flat candidates")
        if table is not None:
            fresh_rows.extend(table_to_index_rows(table))
    if not fresh_rows:
        raise RuntimeError("MAST returned no WFI L2 dark-flat candidates")
    output_path = Path(output_path)
    if output_path.exists():
        existing_rows = read_index_rows(output_path)
        rows, added, updated = merge_index_rows(existing_rows, fresh_rows)
    else:
        rows, added, updated = merge_index_rows([], fresh_rows)
    write_index_rows_atomic(output_path, rows)
    status(
        f"Checked {len(fresh_rows)} current MAST rows; added {added}, "
        f"updated {updated}, retained {len(rows) - added - updated} unchanged"
    )
    status(f"Wrote {len(rows)} rows to {output_path}")
    return rows


def resolve_record_times(records, index_metadata):
    """Resolve physical timestamps from the refreshed MAST index."""
    if not records:
        raise ValueError("No products available for timestamp resolution")
    for record in records:
        row = index_metadata.get(record["file_set_name"]) or index_metadata.get(
            record["file_name"]
        )
        if row is None:
            raise ValueError(f"Missing MAST timestamp for {record['file_name']}")
        if row["detector"].strip() != record["detector"]:
            raise ValueError(f"Index detector mismatch for {record['file_name']}")
        if int(row["led"]) != record["led"]:
            raise ValueError(f"Index LED mismatch for {record['file_name']}")
        timestamp = row["exposure_start_time"].strip()
        if not timestamp:
            raise ValueError(f"Empty MAST timestamp for {record['file_name']}")
        record["exposure_start_time"] = timestamp
        record["exposure_end_time"] = row.get("exposure_end_time", "").strip()
        record["timestamp_source"] = "mast_index"
    return records


def write_inventory(path, records, observations):
    """Write the authoritative product and reduction inventory."""
    record_by_file_set = {record["file_set_name"]: record for record in records}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "file_name",
        "file_path",
        "source_signature",
        "file_set_name",
        "detector",
        "source",
        "cal_uri",
        "led",
        "epoch_key",
        "repeat",
        "exposure_start_time",
        "exposure_end_time",
        "days",
        "timestamp_source",
        "photodiode_channel",
        "photodiode_source",
        "photodiode_mean",
        "photodiode_reference",
        "photodiode_reference_pass",
        "photodiode_scale",
        "photodiode_n_samples",
        "photodiode_dark_median",
        "photodiode_dark_subtracted",
        "photodiode_change_pct",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for observation in observations:
            record = record_by_file_set[observation["file_set_name"]]
            writer.writerow(
                {
                    "file_name": record["file_name"],
                    "file_path": str(record["path"]) if record.get("path") else "",
                    "source_signature": record.get("source_signature", ""),
                    "file_set_name": record["file_set_name"],
                    "detector": record["detector"],
                    "source": record.get("source", "rrn"),
                    "cal_uri": record.get("cal_uri") or record["file_name"],
                    "led": observation["led"],
                    "epoch_key": record["epoch_key"],
                    "repeat": observation["repeat"],
                    "exposure_start_time": observation["exposure_start_time"],
                    "exposure_end_time": observation.get("exposure_end_time") or "",
                    "days": observation["days"],
                    "timestamp_source": record.get("timestamp_source", ""),
                    "photodiode_channel": observation.get("photodiode_channel", ""),
                    "photodiode_source": observation.get("photodiode_source", ""),
                    "photodiode_mean": observation.get("photodiode_mean", ""),
                    "photodiode_reference": observation.get("photodiode_reference", ""),
                    "photodiode_reference_pass": observation.get(
                        "photodiode_reference_pass", ""
                    ),
                    "photodiode_scale": observation.get("photodiode_scale", ""),
                    "photodiode_n_samples": observation.get("photodiode_n_samples", ""),
                    "photodiode_dark_median": observation.get(
                        "photodiode_dark_median", ""
                    ),
                    "photodiode_dark_subtracted": observation.get(
                        "photodiode_dark_subtracted", ""
                    ),
                    "photodiode_change_pct": observation.get(
                        "photodiode_change_pct", ""
                    ),
                }
            )


def tile_sufficient_statistics(
    baseline_data,
    baseline_dq,
    current_data,
    current_dq,
    tile_size,
    current_scale=1.0,
):
    """Yield compact count and log-ratio moments for each detector tile."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if not np.isfinite(current_scale) or current_scale <= 0.0:
        raise ValueError("current_scale must be positive and finite")
    if baseline_data.shape != current_data.shape or baseline_data.shape != baseline_dq.shape:
        raise ValueError("Baseline/current image shapes must match")
    if baseline_data.shape != current_dq.shape:
        raise ValueError("Current data and DQ shapes must match")
    n_rows, n_cols = baseline_data.shape
    tile_row_count = (n_rows + tile_size - 1) // tile_size
    tile_col_count = (n_cols + tile_size - 1) // tile_size
    for tile_row in range(tile_row_count):
        row_start = tile_row * tile_size
        row_stop = min(row_start + tile_size, n_rows)
        for tile_col in range(tile_col_count):
            col_start = tile_col * tile_size
            col_stop = min(col_start + tile_size, n_cols)
            baseline = np.asarray(baseline_data[row_start:row_stop, col_start:col_stop])
            current = np.asarray(
                current_data[row_start:row_stop, col_start:col_stop], dtype=np.float64
            ) * float(current_scale)
            valid = (
                np.isfinite(baseline)
                & np.isfinite(current)
                & (np.asarray(baseline_dq[row_start:row_stop, col_start:col_stop]) == 0)
                & (np.asarray(current_dq[row_start:row_stop, col_start:col_stop]) == 0)
                & (baseline > 0.0)
                & (current > 0.0)
            )
            reference_values = baseline[valid].astype(np.float64)
            current_values = current[valid].astype(np.float64)
            values = np.log(current_values) - np.log(reference_values)
            if values.size:
                reference_total = float(np.sum(reference_values, dtype=np.float64))
                reference_squared = float(np.sum(reference_values * reference_values, dtype=np.float64))
                current_total = float(np.sum(current_values, dtype=np.float64))
                current_squared = float(np.sum(current_values * current_values, dtype=np.float64))
                reference_current = float(np.sum(reference_values * current_values, dtype=np.float64))
                reference_minimum = float(np.min(reference_values))
                reference_maximum = float(np.max(reference_values))
                current_minimum = float(np.min(current_values))
                current_maximum = float(np.max(current_values))
                minimum = float(np.min(values))
                maximum = float(np.max(values))
                total = float(np.sum(values, dtype=np.float64))
                total_squared = float(np.sum(values * values, dtype=np.float64))
            else:
                reference_total = reference_squared = current_total = current_squared = 0.0
                reference_current = 0.0
                reference_minimum = reference_maximum = current_minimum = current_maximum = np.nan
                minimum = maximum = np.nan
                total = total_squared = 0.0
            yield {
                "tile_id": tile_row * tile_col_count + tile_col,
                "tile_row": tile_row,
                "tile_col": tile_col,
                "row_start": row_start,
                "row_stop_exclusive": row_stop,
                "col_start": col_start,
                "col_stop_exclusive": col_stop,
                "tile_n_rows": row_stop - row_start,
                "tile_n_cols": col_stop - col_start,
                "n_total": (row_stop - row_start) * (col_stop - col_start),
                "n_pair_valid": int(values.size),
                "sum_reference_counts": reference_total,
                "sum_squared_reference_counts": reference_squared,
                "sum_current_counts": current_total,
                "sum_squared_current_counts": current_squared,
                "sum_reference_current_counts": reference_current,
                "min_reference_count": reference_minimum,
                "max_reference_count": reference_maximum,
                "min_current_count": current_minimum,
                "max_current_count": current_maximum,
                "sum_log_change": total,
                "sum_squared_log_change": total_squared,
                "min_log_change": minimum,
                "max_log_change": maximum,
            }


def build_observations(records):
    """Order products into shared epochs and assign elapsed physical days."""
    epoch_groups = {}
    for record in records:
        epoch_groups.setdefault(record["epoch_key"], []).append(record)
    for epoch_key, group in epoch_groups.items():
        led_values = [record["led"] for record in group]
        if len(led_values) != len(set(led_values)):
            raise ValueError(f"Duplicate LED product in observation epoch {epoch_key}")
    if not all(record.get("exposure_start_time") for record in records):
        raise ValueError("All products require exposure timestamps")
    for record in records:
        record["sort_time"] = parse_iso_time(record["exposure_start_time"])
    epoch_order = sorted(
        epoch_groups,
        key=lambda key: (
            min(record["sort_time"] for record in epoch_groups[key]),
            key,
        ),
    )
    global_start = min(record["sort_time"] for record in records)
    observations = []
    for repeat, epoch_key in enumerate(epoch_order):
        group = sorted(epoch_groups[epoch_key], key=lambda record: (record["sort_time"], record["led"]))
        for record in group:
            observations.append(
                {
                    "led": record["led"],
                    "repeat": repeat,
                    "group": len(observations),
                    "file_set_name": record["file_set_name"],
                    "baseline_file_set_name": "",
                    "exposure_start_time": record["exposure_start_time"],
                    "exposure_end_time": record.get("exposure_end_time"),
                    "days": (record["sort_time"] - global_start).total_seconds() / 86400.0,
                    "cal_uri": record.get("cal_uri")
                    or record.get("s3_uri")
                    or (str(record["path"]) if record.get("path") else ""),
                    "s3_uri": record.get("s3_uri"),
                    "source": record.get("source", "rrn"),
                    "path": record.get("path"),
                }
            )
    led_counts = {led: 0 for led in range(1, 7)}
    for row in observations:
        led_counts[row["led"]] += 1
    for led, count in led_counts.items():
        if count < 3:
            raise ValueError(f"LED {led} has too few observation epochs: {count}")
    baselines = {}
    for row in observations:
        baselines.setdefault(row["led"], row["file_set_name"])
    for row in observations:
        row["baseline_file_set_name"] = baselines[row["led"]]
    return observations


def load_observation_image(observation, fetch_image=None):
    if observation.get("source", "rrn") in {"mast", "s3"}:
        if fetch_image is None:
            raise RuntimeError(
                f"No remote image fetcher configured for {observation['file_set_name']}"
            )
        status(f"Streaming {observation['source']} product {observation['file_set_name']}")
        return fetch_image(observation)
    raise ValueError(f"Unsupported non-remote product source: {observation.get('source')!r}")


def detector_part_path(parts_dir, group):
    return parts_dir / f"group_{int(group):04d}.csv"


def write_detector_statistics(observations, output_path, tile_size, resume=True, fetch_image=None):
    """Write one detector's statistics with per-observation checkpoints."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and output_path.exists():
        with output_path.open(newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
        missing = set(FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(f"Existing statistics file is missing fields {sorted(missing)}")
        status(f"Skipping completed output: {output_path}")
        return
    parts_dir = output_path.with_name(output_path.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    for led in range(1, 7):
        led_rows = [row for row in observations if row["led"] == led]
        pending_rows = [
            row for row in led_rows
            if not (resume and detector_part_path(parts_dir, row["group"]).exists())
        ]
        if not pending_rows:
            continue
        baseline_data, baseline_dq = load_observation_image(led_rows[0], fetch_image)
        try:
            for row in pending_rows:
                if row["file_set_name"] == row["baseline_file_set_name"]:
                    current_data, current_dq = baseline_data, baseline_dq
                else:
                    current_data, current_dq = load_observation_image(row, fetch_image)
                part_path = detector_part_path(parts_dir, row["group"])
                temporary_part = part_path.with_name(part_path.name + ".partial")
                try:
                    with temporary_part.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                        last_tile = None
                        for tile in tile_sufficient_statistics(
                            baseline_data,
                            baseline_dq,
                            current_data,
                            current_dq,
                            tile_size,
                            current_scale=float(row.get("photodiode_scale", 1.0)),
                        ):
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
                            raise RuntimeError("Tile statistic generator returned no tiles")
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary_part.replace(part_path)
                finally:
                    if temporary_part.exists():
                        temporary_part.unlink()
                if current_data is not baseline_data:
                    del current_data, current_dq
                status(f"{row['file_set_name']}: wrote {last_tile['tile_id'] + 1} tiles")
        finally:
            del baseline_data, baseline_dq
    missing = [
        row["group"]
        for row in observations
        if not detector_part_path(parts_dir, row["group"]).exists()
    ]
    if missing:
        raise RuntimeError(f"Missing completed observation parts: {missing[:5]}")
    temporary_path = output_path.with_name(output_path.name + ".partial")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for observation in observations:
            with detector_part_path(parts_dir, observation["group"]).open(newline="") as part:
                writer.writerows(csv.DictReader(part))
    temporary_path.replace(output_path)


def incremental_part_path(parts_dir, file_set_name):
    if not file_set_name or Path(file_set_name).name != file_set_name:
        raise ValueError(f"Unsafe file-set name: {file_set_name!r}")
    return parts_dir / f"{file_set_name}.csv"


def read_statistics_rows(path, return_fieldnames=False):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = set(FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(f"Statistics file is missing fields {sorted(missing)}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Statistics file has no rows: {path}")
    return (rows, fieldnames) if return_fieldnames else rows


def write_statistics_rows(output_path, rows, fieldnames=FIELDS):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def update_detector_statistics_incremental(observations, output_path, tile_size, fetch_image=None):
    """Reuse old file sets and fetch only new S3 or MAST products."""
    output_path = Path(output_path)
    if not output_path.exists():
        raise ValueError(f"Incremental update requires an existing statistics file: {output_path}")
    existing_rows, existing_fieldnames = read_statistics_rows(output_path, return_fieldnames=True)
    existing_by_file_set = {}
    existing_baselines = {}
    for row in existing_rows:
        file_set_name = row["file_set_name"]
        tile_rows = existing_by_file_set.setdefault(file_set_name, {})
        tile_id = row["tile_id"]
        if tile_id in tile_rows:
            raise ValueError(f"Duplicate tile {tile_id} for {file_set_name}")
        tile_rows[tile_id] = row
        led = int(row["led"])
        baseline = row["baseline_file_set_name"]
        previous = existing_baselines.setdefault(led, baseline)
        if previous != baseline:
            raise ValueError(f"Existing statistics have inconsistent LED {led} baselines")
    current_by_file_set = {row["file_set_name"]: row for row in observations}
    if len(current_by_file_set) != len(observations):
        raise ValueError("The source contains duplicate file-set names")
    removed = set(existing_by_file_set).difference(current_by_file_set)
    if removed:
        raise ValueError("Incremental update refuses to discard existing observations")
    current_baselines = {}
    for observation in observations:
        current_baselines.setdefault(int(observation["led"]), observation["baseline_file_set_name"])
    for led, baseline in existing_baselines.items():
        if current_baselines.get(led) != baseline:
            raise ValueError(f"Incremental update cannot change LED {led} baseline")
    new_observations = [
        row for row in observations if row["file_set_name"] not in existing_by_file_set
    ]
    parts_dir = output_path.with_name(output_path.name + ".incremental.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    new_rows_by_file_set = {}
    for observation in new_observations:
        part_path = incremental_part_path(parts_dir, observation["file_set_name"])
        if part_path.exists():
            new_rows_by_file_set[observation["file_set_name"]] = read_statistics_rows(part_path)
    for led in range(1, 7):
        pending_rows = [
            row for row in new_observations
            if int(row["led"]) == led and row["file_set_name"] not in new_rows_by_file_set
        ]
        if not pending_rows:
            continue
        led_rows = [row for row in observations if int(row["led"]) == led]
        baseline_data, baseline_dq = load_observation_image(led_rows[0], fetch_image)
        try:
            for observation in pending_rows:
                part_path = incremental_part_path(parts_dir, observation["file_set_name"])
                if observation["file_set_name"] == observation["baseline_file_set_name"]:
                    current_data, current_dq = baseline_data, baseline_dq
                else:
                    current_data, current_dq = load_observation_image(observation, fetch_image)
                temporary_part = part_path.with_name(part_path.name + ".partial")
                try:
                    with temporary_part.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                        last_tile = None
                        for tile in tile_sufficient_statistics(
                            baseline_data,
                            baseline_dq,
                            current_data,
                            current_dq,
                            tile_size,
                            current_scale=float(observation.get("photodiode_scale", 1.0)),
                        ):
                            last_tile = tile
                            writer.writerow(
                                {
                                    "led": observation["led"],
                                    "repeat": observation["repeat"],
                                    "group": observation["group"],
                                    "file_set_name": observation["file_set_name"],
                                    "baseline_file_set_name": observation["baseline_file_set_name"],
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
                new_rows_by_file_set[observation["file_set_name"]] = read_statistics_rows(part_path)
                if current_data is not baseline_data:
                    del current_data, current_dq
                status(
                    f"{observation['file_set_name']}: wrote {last_tile['tile_id'] + 1} tiles"
                )
        finally:
            del baseline_data, baseline_dq
    output_rows = []
    for observation in observations:
        file_set_name = observation["file_set_name"]
        rows = (
            existing_by_file_set[file_set_name].values()
            if file_set_name in existing_by_file_set
            else new_rows_by_file_set.get(file_set_name)
        )
        if rows is None:
            raise RuntimeError(f"Missing incremental statistics for {file_set_name}")
        for row in rows:
            output_rows.append(update_statistics_metadata(row, observation))
    output_rows = [{field: row[field] for field in existing_fieldnames} for row in output_rows]
    write_statistics_rows(output_path, output_rows, existing_fieldnames)
    status(
        f"Updated {output_path} with {len(new_observations)} new observations; "
        f"reused {len(observations) - len(new_observations)} existing observations"
    )
    return output_path


DEFAULT_STEP_TIME = "2026-09-14T23:38:00Z"


def _design_matrix(time_days, period_days, first_step, late_step, additional_step=None):
    angular_frequency = 2.0 * np.pi / period_days
    columns = [
        np.ones_like(time_days),
        np.sin(angular_frequency * time_days),
        np.cos(angular_frequency * time_days),
        first_step,
    ]
    if additional_step is not None:
        columns.append(additional_step)
    columns.append(late_step)
    return np.column_stack(columns)


def _fit_for_period(time_days, values, first_step, late_step, period_days, additional_step=None):
    design = _design_matrix(time_days, period_days, first_step, late_step, additional_step)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = values - fitted
    return float(np.sum(residual * residual)), coefficients, fitted, residual


def fit_sinusoid_plus_two_acm(days, values, first_step, late_step):
    from scipy.optimize import minimize_scalar

    time_days = np.asarray(days, dtype=float) - float(days[0])
    values = np.asarray(values, dtype=float)
    first_step = np.asarray(first_step, dtype=float)
    late_step = np.asarray(late_step, dtype=float)
    span = float(np.ptp(time_days))
    if span <= 0.0:
        raise ValueError("LED data must span a positive time interval")
    minimum_period = max(2.0 * float(np.median(np.diff(time_days))), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)
    objective_values = np.asarray(
        [_fit_for_period(time_days, values, first_step, late_step, period)[0] for period in period_grid]
    )
    best_index = int(np.argmin(objective_values))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(period_grid) - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        result = minimize_scalar(
            lambda log_period: _fit_for_period(
                time_days, values, first_step, late_step, float(np.exp(log_period))
            )[0],
            bounds=(np.log(period_grid[lower_index]), np.log(period_grid[upper_index])),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))
    sse, coefficients, fitted, residual = _fit_for_period(
        time_days, values, first_step, late_step, period_days
    )
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    return {
        "period_days": period_days,
        "offset_log_change": float(coefficients[0]),
        "amplitude_log_change": float(np.hypot(coefficients[1], coefficients[2])),
        "phase_rad": float(np.arctan2(coefficients[2], coefficients[1])),
        "acm_step_log_change": float(coefficients[3]),
        "late_acm_step_log_change": float(coefficients[4]),
        "sse_log_change_squared": sse,
        "rmse_log_change": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares) if total_sum_squares > 0.0 else np.nan,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
    }


def fit_sinusoid_plus_three_acm(days, values, first_step, additional_step, late_step):
    from scipy.optimize import minimize_scalar

    time_days = np.asarray(days, dtype=float) - float(days[0])
    values = np.asarray(values, dtype=float)
    first_step = np.asarray(first_step, dtype=float)
    additional_step = np.asarray(additional_step, dtype=float)
    late_step = np.asarray(late_step, dtype=float)
    span = float(np.ptp(time_days))
    if span <= 0.0:
        raise ValueError("LED data must span a positive time interval")
    minimum_period = max(2.0 * float(np.median(np.diff(time_days))), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)

    def objective(log_period):
        return _fit_for_period(
            time_days,
            values,
            first_step,
            late_step,
            float(np.exp(log_period)),
            additional_step,
        )[0]

    objective_values = np.asarray([objective(np.log(period)) for period in period_grid])
    best_index = int(np.argmin(objective_values))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(period_grid) - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        result = minimize_scalar(
            objective,
            bounds=(np.log(period_grid[lower_index]), np.log(period_grid[upper_index])),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))
    sse, coefficients, fitted, residual = _fit_for_period(
        time_days, values, first_step, late_step, period_days, additional_step
    )
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    return {
        "period_days": period_days,
        "offset_log_change": float(coefficients[0]),
        "amplitude_log_change": float(np.hypot(coefficients[1], coefficients[2])),
        "phase_rad": float(np.arctan2(coefficients[2], coefficients[1])),
        "acm_step_log_change": float(coefficients[3]),
        "additional_acm_step_log_change": float(coefficients[4]),
        "late_acm_step_log_change": float(coefficients[5]),
        "sse_log_change_squared": sse,
        "rmse_log_change": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares) if total_sum_squares > 0.0 else np.nan,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
    }


def evaluate_components(time_days, fit, first_step, late_step, additional_step=None):
    time_days = np.asarray(time_days, dtype=float)
    angular_frequency = 2.0 * np.pi / fit["period_days"]
    sinusoid = (
        fit["offset_log_change"]
        + fit["coefficients"][1] * np.sin(angular_frequency * time_days)
        + fit["coefficients"][2] * np.cos(angular_frequency * time_days)
    )
    full = sinusoid + fit["acm_step_log_change"] * np.asarray(first_step, dtype=float)
    if additional_step is not None:
        full = full + fit["additional_acm_step_log_change"] * np.asarray(
            additional_step, dtype=float
        )
    full = full + fit["late_acm_step_log_change"] * np.asarray(late_step, dtype=float)
    return sinusoid, full


def load_count_statistics(path, additional_acm_step_repeat, pre_reset_acm_step_repeat):
    event_time = parse_iso_time(DEFAULT_STEP_TIME)
    rows_by_tile = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sum_current_counts", "n_pair_valid"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Count-space correction fitting requires fields: {sorted(missing)}")
        for row in reader:
            if float(row["n_pair_valid"]) > 0.0:
                rows_by_tile.setdefault(int(row["tile_id"]), []).append(row)
    if not rows_by_tile:
        raise ValueError(f"No valid tile rows found in {path}")
    tiles = {}
    for tile_id, rows in rows_by_tile.items():
        rows = sorted(rows, key=lambda row: int(row["group"]))
        by_led = {}
        for row in rows:
            by_led.setdefault(int(row["led"]), []).append(row)
        led_data = {}
        for led in range(1, 7):
            led_rows = sorted(by_led[led], key=lambda row: float(row["days"]))
            days = np.asarray([float(row["days"]) for row in led_rows], dtype=float)
            values = np.asarray(
                [float(row["sum_current_counts"]) / float(row["n_pair_valid"]) for row in led_rows],
                dtype=float,
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"Non-positive count means for LED {led}, tile {tile_id}")
            after_event = np.asarray(
                [parse_iso_time(row["exposure_start_time"]) >= event_time for row in led_rows],
                dtype=float,
            )
            first_step = after_event - after_event[0]
            late_repeat = max(int(row["repeat"]) for row in led_rows)
            late_step = np.asarray(
                [float(int(row["repeat"]) == late_repeat) for row in led_rows], dtype=float
            )
            late_step -= late_step[0]
            additional_step = None
            if additional_acm_step_repeat is not None:
                additional_step = np.asarray(
                    [float(int(row["repeat"]) >= additional_acm_step_repeat) for row in led_rows],
                    dtype=float,
                )
                additional_step -= additional_step[0]
            pre_reset_step = None
            if pre_reset_acm_step_repeat is not None:
                pre_reset_step = np.asarray(
                    [float(int(row["repeat"]) >= pre_reset_acm_step_repeat) for row in led_rows],
                    dtype=float,
                )
                pre_reset_step -= pre_reset_step[0]
            led_data[led] = {
                "rows": led_rows,
                "days": days,
                "values": values,
                "first_step": first_step,
                "additional_step": additional_step,
                "pre_reset_step": pre_reset_step,
                "late_step": late_step,
                "repeat": np.asarray([int(row["repeat"]) for row in led_rows]),
            }
        tiles[tile_id] = {"rows": rows, "led": led_data}
    return tiles


def _fit_count_segment(values, days, first_step, additional_step, late_step, fit_mask):
    if not np.any(fit_mask):
        raise ValueError("Count data has no observations in the requested fit segment")
    if additional_step is None:
        return fit_sinusoid_plus_two_acm(
            days[fit_mask], values[fit_mask], first_step[fit_mask], late_step[fit_mask]
        )
    return fit_sinusoid_plus_three_acm(
        days[fit_mask],
        values[fit_mask],
        first_step[fit_mask],
        additional_step[fit_mask],
        late_step[fit_mask],
    )


def _fit_count_curve(led, minimum_day, ice_reset_day):
    days = led["days"]
    values = led["values"]
    zero = np.zeros_like(days)
    if ice_reset_day is None:
        fit_mask = days > minimum_day
        fit = _fit_count_segment(
            values,
            days,
            led["first_step"],
            led["additional_step"],
            led["late_step"],
            fit_mask,
        )
        sinusoid, full = evaluate_components(
            days - days[fit_mask][0],
            fit,
            led["first_step"],
            led["late_step"],
            led["additional_step"],
        )
        return fit, None, sinusoid, full, None
    pre_mask = (days > minimum_day) & (days < ice_reset_day)
    post_mask = days >= ice_reset_day
    pre_fit = fit_sinusoid_plus_three_acm(
        days[pre_mask],
        values[pre_mask],
        led["first_step"][pre_mask],
        led["pre_reset_step"][pre_mask],
        zero[pre_mask],
    )
    post_fit = _fit_count_segment(
        values,
        days,
        zero,
        led["additional_step"],
        led["late_step"],
        post_mask,
    )
    pre_sinusoid, pre_full = evaluate_components(
        days - days[pre_mask][0],
        pre_fit,
        led["first_step"],
        zero,
        led["pre_reset_step"],
    )
    post_sinusoid, post_full = evaluate_components(
        days - days[post_mask][0],
        post_fit,
        zero,
        led["late_step"],
        led["additional_step"],
    )
    sinusoid = np.where(post_mask, post_sinusoid, pre_sinusoid)
    full = np.where(post_mask, post_full, pre_full)
    return pre_fit, post_fit, sinusoid, full, post_mask


def fit_count_corrections(
    tile,
    reference_led,
    minimum_day,
    additional_acm_step_repeat,
    ice_reset_day,
    pre_reset_acm_step_repeat,
):
    reference = tile["led"][reference_led]
    reference_fit, reference_post_fit, reference_sinusoid, _, _ = _fit_count_curve(
        reference, minimum_day, ice_reset_day
    )
    early = reference["days"] <= minimum_day
    gain_factor_days = reference["days"][early]
    gain_factors = reference_sinusoid[early] / reference["values"][early]
    if np.any(~np.isfinite(gain_factors)) or np.any(gain_factors <= 0.0):
        raise ValueError("Reference count fit produced invalid early gain factors")
    step_factors_by_repeat = {}
    led_fits = {}
    for led in range(1, 7):
        current = tile["led"][led]
        gain_factor = np.ones_like(current["days"])
        early_current = current["days"] <= minimum_day
        gain_factor[early_current] = np.interp(
            current["days"][early_current], gain_factor_days, gain_factors
        )
        corrected_values = current["values"] * gain_factor
        corrected_led = dict(current)
        corrected_led["values"] = corrected_values
        pre_fit, post_fit, sinusoid, full, led_post_mask = _fit_count_curve(
            corrected_led, minimum_day, ice_reset_day
        )
        step_factor = sinusoid / full
        if np.any(~np.isfinite(step_factor)) or np.any(step_factor <= 0.0):
            raise ValueError(f"LED {led} count fit produced invalid ACM factors")
        for repeat, factor in zip(current["repeat"], step_factor):
            step_factors_by_repeat.setdefault(int(repeat), []).append(float(factor))
        led_fits[led] = {
            "gain_factor": gain_factor,
            "step_factor": step_factor,
            "pre_fit": pre_fit,
            "post_fit": post_fit,
        }
    shared_step_factor = {
        repeat: float(np.exp(np.median(np.log(factors))))
        for repeat, factors in step_factors_by_repeat.items()
    }
    correction_rows = []
    rows_by_key = {}
    for led in range(1, 7):
        current = tile["led"][led]
        for index, row in enumerate(current["rows"]):
            factor = led_fits[led]["gain_factor"][index] * shared_step_factor[int(row["repeat"])]
            correction_rows.append(
                {
                    "detector": "",
                    "tile_id": int(row["tile_id"]),
                    "group": int(row["group"]),
                    "days": float(row["days"]),
                    "correction_space": "counts",
                    "count_log_correction": float(np.log(factor)),
                    "log_gain_correction": float(np.log(led_fits[led]["gain_factor"][index])),
                    "acm_step_log_offset": 0.0,
                    "pre_reset_acm_step_log_offset": 0.0,
                    "additional_acm_step_log_offset": 0.0,
                    "late_acm_step_log_offset": 0.0,
                }
            )
            rows_by_key[(int(row["tile_id"]), int(row["group"]))] = row
    summary = {
        "tile_id": int(next(iter(rows_by_key.values()))["tile_id"]),
        "reference_period_days": reference_fit["period_days"],
        "reference_rmse_counts": reference_fit["rmse_log_change"],
        "reference_r2": reference_fit["r2"],
        "post_reset_period_days": (
            np.nan if reference_post_fit is None else reference_post_fit["period_days"]
        ),
        "post_reset_rmse_counts": (
            np.nan if reference_post_fit is None else reference_post_fit["rmse_log_change"]
        ),
        "post_reset_r2": np.nan if reference_post_fit is None else reference_post_fit["r2"],
        "reference_acm_step_counts": reference_fit["acm_step_log_change"],
        "reference_pre_reset_acm_step_counts": 0.0,
        "reference_additional_acm_step_counts": (
            0.0 if reference_post_fit is None else reference_post_fit["additional_acm_step_log_change"]
        ),
        "reference_late_acm_step_counts": reference_fit["late_acm_step_log_change"],
    }
    if ice_reset_day is not None:
        summary["reference_pre_reset_acm_step_counts"] = reference_fit[
            "additional_acm_step_log_change"
        ]
    return correction_rows, summary


def build_corrections(path, reference_led, minimum_day, additional_acm_step_repeat, ice_reset_day, pre_reset_acm_step_repeat):
    detector = Path(path).name.split("_tile_log_statistics_", 1)[0].upper()
    tiles = load_count_statistics(path, additional_acm_step_repeat, pre_reset_acm_step_repeat)
    correction_rows = []
    summary_rows = []
    for tile_id in sorted(tiles):
        rows, summary = fit_count_corrections(
            tiles[tile_id],
            reference_led,
            minimum_day,
            additional_acm_step_repeat,
            ice_reset_day,
            pre_reset_acm_step_repeat,
        )
        for row in rows:
            row["detector"] = detector
        correction_rows.extend(rows)
        summary_rows.append(
            {
                "detector": detector,
                "tile_id": summary["tile_id"],
                "reference_led": reference_led,
                "minimum_day_exclusive": minimum_day,
                **{key: value for key, value in summary.items() if key != "tile_id"},
            }
        )
    return correction_rows, summary_rows


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_corrected_statistics(statistics_path, corrections_path, output_path):
    """Apply the generated count-space correction without the TMM module."""
    with Path(statistics_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    with Path(corrections_path).open(newline="") as handle:
        correction_rows = list(csv.DictReader(handle))
    if not rows or not correction_rows:
        raise ValueError("Statistics and correction files must both contain rows")
    if {row.get("correction_space", "") for row in correction_rows} != {"counts"}:
        raise ValueError("The correction table must be marked as count-space")
    detector = Path(statistics_path).name.split("_tile_log_statistics_", 1)[0].upper()
    by_key = {}
    for row in correction_rows:
        row_detector = str(row.get("detector", detector)).strip().upper() or detector
        key = (row_detector, int(row["tile_id"]), int(row["group"]))
        if key in by_key:
            raise ValueError(f"Duplicate count correction for {key}")
        by_key[key] = row
    first_group_by_led = {}
    for row in rows:
        first_group_by_led.setdefault(int(row["led"]), int(row["group"]))
    absolute = {}
    for row in rows:
        key = (detector, int(row["tile_id"]), int(row["group"]))
        correction = by_key.get(key)
        if correction is None:
            raise ValueError(f"Missing count correction for {key}")
        if not np.isclose(float(correction["days"]), float(row["days"]), atol=1.0e-9, rtol=0.0):
            raise ValueError(f"Correction days do not match statistics for {key}")
        absolute[(int(row["tile_id"]), int(row["group"]))] = float(
            correction["count_log_correction"]
        )
    count_fields = (
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
    for row in rows:
        tile_id = int(row["tile_id"])
        group = int(row["group"])
        led = int(row["led"])
        correction = absolute[(tile_id, group)]
        reference_correction = absolute[(tile_id, first_group_by_led[led])]
        delta = correction - reference_correction
        reference_factor = float(np.exp(reference_correction))
        current_factor = float(np.exp(correction))
        old_sum = float(row["sum_log_change"])
        old_sum_squared = float(row["sum_squared_log_change"])
        n_pair_valid = float(row["n_pair_valid"])
        row["sum_log_change"] = repr(old_sum + n_pair_valid * delta)
        row["sum_squared_log_change"] = repr(
            old_sum_squared + 2.0 * delta * old_sum + n_pair_valid * delta * delta
        )
        row["min_log_change"] = repr(float(row["min_log_change"]) + delta)
        row["max_log_change"] = repr(float(row["max_log_change"]) + delta)
        for field in count_fields:
            value = float(row[field])
            if field in {"sum_reference_counts", "min_reference_count", "max_reference_count"}:
                value *= reference_factor
            elif field in {"sum_current_counts", "min_current_count", "max_current_count"}:
                value *= current_factor
            elif field == "sum_squared_reference_counts":
                value *= reference_factor * reference_factor
            elif field == "sum_squared_current_counts":
                value *= current_factor * current_factor
            elif field == "sum_reference_current_counts":
                value *= reference_factor * current_factor
            row[field] = repr(value)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


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
            status(f"Seeding incremental reduction from {previous}")
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
        status(f"Incremental state is incompatible; rebuilding {output_path}: {error}")
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


def run_detector(
    source_records,
    run_dir,
    output_root,
    detector,
    index_metadata,
    index_rows,
    telemetry,
    fetch_image,
    manifest,
):
    status(f"Starting reduction for {detector}")
    s3_records = [
        dict(record)
        for record in source_records
        if record["detector"] == detector
    ]
    mast_records = mast_records_from_index(index_rows, detector, allow_empty=True)
    records = merge_product_records(s3_records, mast_records)
    if not records:
        raise ValueError(f"No S3 or MAST products found for {detector}")
    resolve_record_times(records, index_metadata)
    observations = build_observations(records)
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
        "s3_products": len(s3_records),
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
    status(f"Fitting count-space corrections for {detector}")
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
    )
    status(f"Wrote corrected statistics for {detector}: {corrected_path}")
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
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = configure_pipeline_log(run_dir / "pipeline.log")
    status(f"Starting CAR190 reduction; log: {log_path}")
    manifest = {
        "pipeline": "run_wfi_reduction_pipeline.py",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "source": source,
        "output_directory": str(run_dir),
        "log_path": str(log_path),
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

    s3_records = list_s3_products(source)
    refresh_index(
        DEFAULT_MAST_INDEX,
        token_path=DEFAULT_MAST_TOKEN,
        detectors=DEFAULT_DETECTORS,
    )
    status("MAST index refresh complete")
    index_rows = read_index_rows(DEFAULT_MAST_INDEX, require_program=True)
    index_metadata = read_optional_index(DEFAULT_MAST_INDEX)
    shutil.copy2(DEFAULT_MAST_INDEX, run_dir / DEFAULT_MAST_INDEX.name)

    telemetry_path = run_dir / "telemetry" / "photodiode_pd1_ice_monitor.csv"
    telemetry = run_ice_monitor_telemetry(
        index_rows,
        DEFAULT_MAST_TOKEN,
        DEFAULT_PHOTODIODE_CHANNEL,
        telemetry_path,
    )
    status(f"Received {len(telemetry['values'])} dark-subtracted photodiode rows")
    manifest["telemetry"] = {
        "source": telemetry["source"],
        "channel": telemetry["channel"],
        "mnemonic": telemetry["mnemonic"],
        "path": telemetry["summary_path"],
        "dark_subtracted": True,
        "exposure_rows": len(telemetry["values"]),
    }

    missions = login_to_mast(DEFAULT_MAST_TOKEN)

    def fetch_image(observation):
        if observation["source"] == "s3":
            return stream_s3_image(observation["s3_uri"])
        return stream_mast_image(missions, observation["cal_uri"])

    s3_detectors = {record["detector"] for record in s3_records}
    mast_detectors = {row["detector"].strip() for row in index_rows}
    detectors = sorted(s3_detectors | mast_detectors)
    if not detectors:
        raise ValueError("No S3 or MAST calibrated products were found")
    manifest["detectors_requested"] = detectors
    manifest["detectors_available"] = detectors
    write_json(run_dir / "pipeline_manifest.json", manifest)
    status(
        f"Reducing {', '.join(detectors)} from streamed S3/MAST products "
        f"into {run_dir}",
    )
    for detector in detectors:
        run_detector(
            s3_records,
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
    status(f"Wrote pipeline manifest {run_dir / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    main()
