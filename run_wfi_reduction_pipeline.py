#!/usr/bin/env python3
"""Fully automated CAR190 reduction entry point.

The entry point has no required arguments. It lists calibrated ASDF products
from S3, streams each image into memory, refreshes one unified Program 1039
plus Program 1063 MAST supplement and per-program photodiode telemetry, and
reduces every detector chronologically. Photodiode scales use the first
non-warm-up dark-subtracted exposure for each LED as one shared reference
across the unified programs. Each detector receives an independent local
side-swap calibration with separate LED-dependent image and photodiode factors;
the calibrated photodiode-scaled statistics are then passed through the
remaining count-space temperature/ACM corrections. The late shared ACM step
is not fitted again because it is replaced by the side-swap calibration.
The statistics directory contains side-swap-corrected photodiode-scaled
moments, raw_statistics contains truly uncorrected detector-count moments,
and corrected_statistics contains the fully count-corrected product.
"""

import atexit
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
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
import threading
import time
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
import warnings

import numpy as np

from fit_wfi_side_swap import (
    available_side_swap_points,
    fit_side_swap_series,
)

try:
    from requests.exceptions import RequestException
except ModuleNotFoundError:
    class RequestException(Exception):
        pass


FIELDS = (
    "led",
    "program",
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
    r"^r(?P<visit_id>[0-9]+)_(?P<exposure_id>[^_]+)_wfi(?P<detector>[0-9]{2})_"
    r"(?P<optical_element>[^_]+)_cal\.asdf$",
    re.IGNORECASE,
)
PRODUCT_EXPOSURE_PATTERN = re.compile(
    r"^r[0-9]+_(?P<exposure_id>[^_]+)_wfi[0-9]{2}_",
    re.IGNORECASE,
)
PHOTODIODE_MNEMONICS = {
    "pd1": "WFI_MCE_SRCS_PD1_V",
    "pd2": "WFI_MCE_SRCS_PD2_V",
}
MPA_TEMPERATURE_MNEMONIC = "WFI_MCE_ADC_WTSM00"
MPA_TEMPERATURE_BUCKET_SECONDS = 300

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
UNIFIED_PROGRAMS = ("1039", "1063")
REFERENCE_PROGRAM = "1039"
TEMPERATURE_CORRECTION_PROGRAM = "1063"

CORRECTION_FIELDS = (
    "detector",
    "tile_id",
    "group",
    "days",
    "program",
    "correction_space",
    "count_log_correction",
    "log_gain_correction",
    "temperature_gain_log_correction",
    "acm_step_log_offset",
    "pre_reset_acm_step_log_offset",
    "additional_acm_step_log_offset",
    "day22_acm_step_log_offset",
    "day26_acm_step_log_offset",
    "late_acm_step_log_offset",
    "temperature_k",
    "temperature_reference_k",
    "temperature_quadratic_a_log_correction",
    "temperature_quadratic_b_log_correction_per_k",
    "temperature_quadratic_c_log_correction_per_k2",
    "temperature_quadratic_normalization_offset_log_correction",
    "temperature_quadratic_fit_r2",
    "temperature_quadratic_fit_points",
    "decon_step_log_correction",
)
SUMMARY_FIELDS = (
    "detector",
    "tile_id",
    "reference_led",
    "minimum_day_exclusive",
    "reference_program",
    "temperature_correction_program",
    "pre_operational_temperature_period_days",
    "pre_operational_temperature_rmse_counts",
    "pre_operational_temperature_r2",
    "post_operational_temperature_period_days",
    "post_operational_temperature_rmse_counts",
    "post_operational_temperature_r2",
    "pre_operational_temperature_fit_points",
    "pre_operational_temperature_fit_start_repeat",
    "pre_operational_temperature_fit_end_repeat",
    "post_operational_temperature_fit_points",
    "post_operational_temperature_fit_start_repeat",
    "post_acm_fit_start_repeat",
    "post_operational_temperature_fit_end_repeat",
    "post_acm_fit_end_repeat",
    "reference_acm_step_counts",
    "reference_pre_reset_acm_step_counts",
    "reference_additional_acm_step_counts",
    "additional_acm_step_fit_scope",
    "additional_acm_step_led_1_counts",
    "additional_acm_step_led_2_counts",
    "additional_acm_step_led_3_counts",
    "additional_acm_step_led_4_counts",
    "additional_acm_step_led_5_counts",
    "additional_acm_step_led_6_counts",
    "reference_day22_acm_step_counts",
    "day22_acm_step_fit_scope",
    "day22_acm_step_led_1_counts",
    "day22_acm_step_led_2_counts",
    "day22_acm_step_led_3_counts",
    "day22_acm_step_led_4_counts",
    "day22_acm_step_led_5_counts",
    "day22_acm_step_led_6_counts",
    "reference_day26_acm_step_counts",
    "day26_acm_step_fit_scope",
    "reference_late_acm_step_counts",
    "temperature_model",
    "temperature_mnemonic",
    "temperature_reference_k",
    "temperature_quadratic_a_log_correction",
    "temperature_quadratic_b_log_correction_per_k",
    "temperature_quadratic_c_log_correction_per_k2",
    "temperature_quadratic_normalization_offset_log_correction",
    "temperature_quadratic_fit_r2",
    "temperature_quadratic_fit_points",
    "temperature_quadratic_rmse_percent",
    "temperature_quadratic_max_abs_percent",
    "pre_operational_temperature_slope_log_per_k",
    "post_operational_temperature_slope_log_per_k",
    "decon_step_time",
    "decon_step_log_correction",
    "decon_step_fit_points",
    "decon_step_pre_points",
    "decon_step_post_points",
    "decon_step_rmse_log",
)
SIDE_SWAP_FIELDS = (
    "detector",
    "led",
    "channel",
    "event_repeat",
    "event_day",
    "n_pre",
    "n_post",
    "selected_repeats",
    "knot_days",
    "step_log",
    "correction_factor",
    "rmse_log",
    "r2",
)


ROOT = Path(__file__).resolve().parent
DEFAULT_S3_SOURCE = "s3://roman-s3-romcom/activities/CAR190/"
# A 4096x4096 detector produces 64 512x512 tiles by default. The CLI can
# override this for separate tile-resolution reductions.
DEFAULT_TILE_SIZES = (512,)
DEFAULT_RUN_ROOT = ROOT / "CAR190_reduction"
DEFAULT_MAST_TOKEN = ROOT / "mast_api_token.txt"
DEFAULT_MAST_INDEX = ROOT / "all_wfi_rcs_flat_index.csv"
DEFAULT_TEMPERATURE_QUADRATIC_COEFFICIENTS = (
    ROOT / "temperature_quadratic_coefficients.csv"
)
DEFAULT_PHOTODIODE_CHANNEL = "pd2"
DEFAULT_ROMAN_TELEM_BINARY = "roman-telem"
DEFAULT_MAST_EDP_TEMPERATURE_URL = (
    "https://mast.stsci.edu/edp/api/v0.1/mnemonics/spa/roman/data"
)
# A daily request stays below the EDP point limit for this approximately 1 Hz
# stream while avoiding hundreds of slow per-hour requests.
DEFAULT_TEMPERATURE_QUERY_CHUNK_HOURS = 24
# This is the first post-reset observation in the original 1039 timeline. In
# the unified timeline it marks the operational-temperature transition, not
# the physical ice reset used by the downstream TMM.
DEFAULT_OPERATIONAL_TEMPERATURE_TIME = "2026-09-27T02:58:35.308Z"
DEFAULT_ADDITIONAL_ACM_STEP_REPEAT = 39
DEFAULT_PRE_RESET_ACM_STEP_REPEAT = 35
DEFAULT_FIRST_ACM_STEP_REPEAT = 14
# The legacy 1039-only timeline identifies the side-swap event at repeat 56.
# Later unified files append observations after it, so the event must not be
# inferred from max(repeat); run_detector maps it to the current timeline.
DEFAULT_LATE_ACM_STEP_REPEAT = 56
DEFAULT_FIRST_POST_OPERATIONAL_REPEAT = 36
# These are legacy post-operational repeats and are mapped onto each unified
# timeline in run_detector. The legacy temperature application stops at day 22; the day-22
# per-LED step and later day-26 shared-grey step are fitted separately.
DEFAULT_TEMPERATURE_FIT_END_REPEAT = 52
DEFAULT_DAY22_ACM_STEP_REPEAT = 52
# The next ACM adjustment is the first post-adjustment epoch in the unified
# timeline; WFI02 maps legacy repeat 61 to effective repeat 691.
DEFAULT_DAY26_ACM_STEP_REPEAT = 61
DEFAULT_SIDE_SWAP_PRE_POINTS = 5
DEFAULT_SIDE_SWAP_POST_POINTS = 5
DEFAULT_DECON_FIT_WINDOW_DAYS = 2.0
BASE_STATISTICS_FIELDS = tuple(field for field in FIELDS if field != "program")
_LOG_HANDLE = None
_LOG_LOCK = threading.Lock()
_PLOT_LOCK = threading.Lock()


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
    with _LOG_LOCK:
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
        "exposure_id": match.group("exposure_id"),
        "detector": f"WFI{int(match.group('detector')):02d}",
        "led": int(led_code) - 1,
        "epoch_key": epoch_key,
        "source": "rrn",
    }


def read_index_rows(path, require_program=False, programs=UNIFIED_PROGRAMS):
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
        allowed_programs = sorted({str(program).strip() for program in programs})
        found_programs = sorted({str(row.get("program", "")).strip() for row in rows})
        if found_programs != allowed_programs:
            raise ValueError(
                f"MAST index must contain unified programs {allowed_programs}; "
                f"found programs: {found_programs}"
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
                "exposure_id": product_exposure_id(
                    {"file_name": file_name, "file_set_name": file_set_name}
                ),
                "program": str(row.get("program", "")).strip(),
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


def product_exposure_id(record):
    """Extract Roman's exposure identifier from a product name."""
    exposure_id = str(record.get("exposure_id", "")).strip()
    if exposure_id:
        return exposure_id
    for value in (record.get("file_name"), record.get("file_set_name")):
        if not value:
            continue
        match = PRODUCT_EXPOSURE_PATTERN.match(Path(str(value)).name)
        if match is not None:
            return match.group("exposure_id")
    return ""


def product_identity(record):
    """Return the physical product identity shared by S3 and MAST records."""
    visit_id = str(record["visit_id"])
    if len(visit_id) <= 4:
        raise ValueError(f"Invalid visit ID for product identity: {visit_id!r}")
    exposure_id = product_exposure_id(record)
    if not exposure_id:
        exposure_id = str(record.get("file_set_name", ""))
    return (
        record["detector"],
        visit_id,
        exposure_id,
        int(record["led"]),
    )


def merge_product_records(local_records, mast_records):
    """Prefer S3 when both remote sources describe one physical product."""
    merged = []
    by_file_set = {}
    by_identity = {}
    for record in local_records:
        key = record["file_set_name"]
        if key in by_file_set:
            raise ValueError(f"Duplicate S3 product {key}")
        identity = product_identity(record)
        if identity in by_identity:
            previous = by_identity[identity]
            raise ValueError(
                "Duplicate S3 products for physical observation "
                f"{identity}: {previous['file_set_name']} and {key}"
            )
        by_file_set[key] = record
        by_identity[identity] = record
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
            if local.get("program") and record.get("program") and local["program"] != record["program"]:
                raise ValueError(f"S3/MAST program mismatch for {key}")
            continue
        identity = product_identity(record)
        existing = by_identity.get(identity)
        if existing is not None:
            if existing.get("source") == "s3":
                # The S3 basename and MAST fileSetName are not always identical,
                # but the S3 object is the preferred copy of this observation.
                existing.setdefault("mast_file_set_names", []).append(key)
                continue
            raise ValueError(
                "Duplicate MAST products for physical observation "
                f"{identity}: {existing['file_set_name']} and {key}"
            )
        by_file_set[key] = record
        by_identity[identity] = record
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


def _utc_timestamp(value):
    return parse_iso_time(value).replace(tzinfo=timezone.utc).timestamp()


def load_temperature_series(
    path,
    mnemonic=MPA_TEMPERATURE_MNEMONIC,
    bucket_seconds=MPA_TEMPERATURE_BUCKET_SECONDS,
):
    """Load long-form MAST/roman-telem temperature samples.

    The MAST engineering and ``roman-telem`` exports use ``ObsTime``,
    ``EUValue``, and ``mnemonic`` columns. Samples are averaged into fixed
    UTC buckets before they are interpolated to exposure timestamps.
    """
    path = Path(path)
    required = {"ObsTime", "EUValue", "mnemonic"}
    buckets = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Temperature telemetry is missing {sorted(missing)}: {path}"
            )
        for row in reader:
            if str(row.get("mnemonic", "")).strip() != mnemonic:
                continue
            try:
                value = float(row["EUValue"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value) or value <= 0.0:
                continue
            second = _utc_timestamp(row["ObsTime"])
            bucket = int(second) // int(bucket_seconds)
            total, count = buckets.get(bucket, (0.0, 0))
            buckets[bucket] = (total + value, count + 1)
    if not buckets:
        raise ValueError(f"No telemetry samples found for {mnemonic}: {path}")
    ordered = sorted(buckets.items())
    seconds = np.asarray(
        [(bucket + 0.5) * int(bucket_seconds) for bucket, _ in ordered],
        dtype=float,
    )
    values = np.asarray(
        [total / count for _, (total, count) in ordered],
        dtype=float,
    )
    if seconds.size < 2 or np.ptp(seconds) <= 0.0:
        raise ValueError(f"Temperature telemetry has no time span: {path}")
    return seconds, values


def interpolate_temperature(exposure_times, temperature_series):
    """Interpolate WTSM00 without silently extrapolating beyond telemetry."""
    seconds, values = temperature_series
    exposure_seconds = np.asarray(
        [_utc_timestamp(value) for value in exposure_times],
        dtype=float,
    )
    if np.any(exposure_seconds < seconds[0]) or np.any(exposure_seconds > seconds[-1]):
        raise ValueError(
            "Temperature telemetry does not cover all exposure timestamps: "
            f"[{exposure_seconds.min()}, {exposure_seconds.max()}] outside "
            f"[{seconds[0]}, {seconds[-1]}]"
        )
    return np.interp(exposure_seconds, seconds, values)


def temperature_series_covers(exposure_times, temperature_series):
    """Return whether telemetry brackets every requested exposure time."""
    seconds, _ = temperature_series
    exposure_seconds = np.asarray(
        [_utc_timestamp(value) for value in exposure_times],
        dtype=float,
    )
    return bool(
        exposure_seconds.size
        and exposure_seconds.min() >= seconds[0]
        and exposure_seconds.max() <= seconds[-1]
    )


def _mast_edp_time(value):
    # MAST EDP accepts ISO timestamps without a trailing UTC designator.
    return parse_iso_time(value).isoformat(timespec="milliseconds")


def fetch_temperature_telemetry(
    output_path,
    token_path,
    start_time,
    end_time,
    mnemonic=MPA_TEMPERATURE_MNEMONIC,
    bucket_seconds=MPA_TEMPERATURE_BUCKET_SECONDS,
    chunk_hours=DEFAULT_TEMPERATURE_QUERY_CHUNK_HOURS,
):
    """Fetch sporadic Roman engineering telemetry into the local CSV format."""
    if chunk_hours <= 0:
        raise ValueError("Temperature query chunk size must be positive")
    start = parse_iso_time(start_time) - timedelta(seconds=bucket_seconds)
    end = parse_iso_time(end_time) + timedelta(seconds=bucket_seconds)
    if end <= start:
        raise ValueError("Temperature telemetry query must span positive time")

    token = _read_mast_token(token_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".partial")
    last_second = -np.inf
    row_count = 0
    cursor = start
    try:
        with temporary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("ObsTime", "EUValue", "mnemonic"))
            writer.writeheader()
            while cursor < end:
                chunk_end = min(
                    cursor + timedelta(hours=int(chunk_hours)),
                    end,
                )
                query = urlencode(
                    {
                        "mnemonic": mnemonic,
                        "s_time": _mast_edp_time(cursor),
                        "e_time": _mast_edp_time(chunk_end),
                        "result_format": "json",
                    }
                )
                request = Request(
                    f"{DEFAULT_MAST_EDP_TEMPERATURE_URL}?{query}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                try:
                    with urlopen(request, timeout=120) as response:
                        payload = json.load(response)
                except Exception as error:
                    raise RuntimeError(
                        f"MAST temperature query failed for {mnemonic} "
                        f"from {_mast_edp_time(cursor)} to {_mast_edp_time(chunk_end)}"
                    ) from error
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("Data"), list
                ):
                    raise ValueError(
                        f"MAST temperature response has no Data list for {mnemonic}"
                    )
                chunk_rows = []
                for row in payload["Data"]:
                    timestamp = str(row.get("ObsTime", "")).strip()
                    try:
                        second = _utc_timestamp(timestamp)
                        value = float(row["EUValue"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not np.isfinite(value):
                        continue
                    chunk_rows.append((second, timestamp, value))
                for second, timestamp, value in sorted(chunk_rows):
                    if second <= last_second:
                        continue
                    writer.writerow(
                        {
                            "ObsTime": timestamp,
                            "EUValue": repr(value),
                            "mnemonic": mnemonic,
                        }
                    )
                    last_second = second
                    row_count += 1
                cursor = chunk_end
        if row_count < 2:
            raise ValueError(
                f"MAST returned fewer than two temperature samples for {mnemonic}"
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    status(f"Fetched {row_count} {mnemonic} samples to {output_path}")
    return output_path


def prepare_temperature_telemetry(args, index_rows, run_dir):
    """Load supplied WTSM00 data or fetch a covered local copy from MAST."""
    missing_argument = object()
    requested_path = getattr(args, "mpa_temperature_telemetry", missing_argument)
    if requested_path is missing_argument:
        # Keep embedded/unit-test callers that construct a partial argument object
        # on the legacy path; the CLI always defines this option.
        return None, None
    exposure_times = [
        row["exposure_start_time"]
        for row in index_rows
        if str(row.get("exposure_start_time", "")).strip()
    ]
    if not exposure_times:
        raise ValueError("Temperature telemetry requires indexed exposure times")
    if requested_path is None:
        telemetry_path = Path(run_dir) / "telemetry" / "wfi_mce_adc_wtsm00.csv"
        temperature_series = None
        if telemetry_path.is_file():
            temperature_series = load_temperature_series(telemetry_path)
        if temperature_series is None or not temperature_series_covers(
            exposure_times, temperature_series
        ):
            fetch_temperature_telemetry(
                telemetry_path,
                DEFAULT_MAST_TOKEN,
                min(exposure_times, key=parse_iso_time),
                max(exposure_times, key=parse_iso_time),
            )
    else:
        telemetry_path = Path(requested_path)
    temperature_series = load_temperature_series(telemetry_path)
    if not temperature_series_covers(exposure_times, temperature_series):
        raise ValueError(
            f"Temperature telemetry does not cover all indexed exposures: {telemetry_path}"
        )
    interpolate_temperature(exposure_times, temperature_series)
    return temperature_series, telemetry_path


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
        "warmup",
        "ref_pass",
        "ref_signal",
        "change_pct",
    }
    values = {}
    identity_values = {}
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
            warmup_text = str(row["warmup"]).strip().lower()
            if warmup_text not in {"true", "false", "1", "0"}:
                raise ValueError(f"Ice-monitor warmup flag is invalid: {row}")
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
            value = {
                "signal": signal,
                "dark_median": dark,
                "n_samples": int(row["n_samples"]),
                "pass_id": str(row.get("pass_id", "")).strip(),
                "warmup": warmup_text in {"true", "1"},
                "reference_pass": row["ref_pass"],
                "reference_signal": reference,
                "change_pct": change_pct,
            }
            values[key] = value
            pass_id = str(row.get("pass_id", "")).strip()
            observations = {
                observation.strip()
                for observation in str(row.get("observation", "")).split(",")
                if observation.strip()
            }
            for observation in observations:
                if pass_id:
                    identity_values.setdefault((pass_id, observation, led), []).append(
                        (start, end, value)
                    )
    if not values:
        raise ValueError(f"No {mnemonic} LED rows found in {path}")
    return {
        "source": "roman-telem-ice-monitor",
        "summary_path": str(path),
        "channel": channel,
        "mnemonic": mnemonic,
        "values": values,
        "identity_values": identity_values,
    }


def apply_first_non_warmup_references(telemetry):
    """Use one earliest non-warm-up dark-subtracted reference per LED."""
    if telemetry.get("reference_mode") == "first_non_warmup_per_led":
        return telemetry
    values = telemetry.get("values", {})
    if not values:
        raise ValueError("Cannot select photodiode references from empty telemetry")
    led_values = {}
    for key, value in values.items():
        if "warmup" not in value:
            raise ValueError("Photodiode telemetry is missing the warmup flag")
        led_values.setdefault(int(key[2]), []).append((key, value))
    references = {}
    for led, candidates in led_values.items():
        non_warmup = [item for item in candidates if not item[1]["warmup"]]
        if not non_warmup:
            raise ValueError(
                f"Photodiode telemetry has no non-warm-up reference for LED {led}"
            )
        reference_key, reference_value = min(
            non_warmup,
            key=lambda item: (item[0][0], item[0][1]),
        )
        references[led] = {
            "start": reference_key[0].isoformat(),
            "end": reference_key[1].isoformat(),
            "pass_id": reference_value.get("pass_id", "")
            or reference_value.get("reference_pass", ""),
            "signal": float(reference_value["signal"]),
        }

    for key, value in values.items():
        reference = references[int(key[2])]
        signal = float(value["signal"])
        reference_signal = reference["signal"]
        value["reference_pass"] = reference["pass_id"]
        value["reference_signal"] = reference_signal
        value["change_pct"] = 100.0 * (signal / reference_signal - 1.0)

    telemetry["reference_mode"] = "first_non_warmup_per_led"
    telemetry["reference_values"] = {
        str(led): reference for led, reference in sorted(references.items())
    }
    return telemetry


def run_ice_monitor_telemetry(index_rows, token_path, channel, program, output_path):
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
        str(program),
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


def merge_telemetry_summaries(telemetries):
    """Combine telemetry and select shared first non-warm-up LED references."""
    if not telemetries:
        raise ValueError("At least one photodiode telemetry summary is required")
    merged_values = {}
    channels = set()
    sources = set()
    paths = []
    for telemetry in telemetries:
        channels.add(telemetry["channel"])
        sources.add(telemetry["source"])
        paths.append(telemetry["summary_path"])
        for key, value in telemetry["values"].items():
            previous = merged_values.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"Conflicting photodiode telemetry rows: {key}")
            merged_values[key] = value
    if len(channels) != 1 or len(sources) != 1:
        raise ValueError("Photodiode telemetry summaries use inconsistent sources")
    merged = {
        "source": next(iter(sources)),
        "summary_path": ";".join(paths),
        "channel": next(iter(channels)),
        "mnemonic": telemetries[0]["mnemonic"],
        "values": merged_values,
        "identity_values": {
            key: values
            for telemetry in telemetries
            for key, values in telemetry.get("identity_values", {}).items()
        },
    }
    return apply_first_non_warmup_references(merged)


def telemetry_identity(observation):
    """Return roman-telem's pass/observation identity for one image visit."""
    visit_id = str(observation.get("visit_id", "")).strip()
    if len(visit_id) != 19 or not visit_id.isdigit():
        return None
    return (
        visit_id[:10],
        ".".join((visit_id[10:13], visit_id[13:16], visit_id[16:19])),
        int(observation["led"]),
    )


def attach_photodiode_corrections(observations, telemetry):
    """Apply the shared first non-warm-up dark-subtracted scale per LED."""
    if telemetry.get("source") != "roman-telem-ice-monitor":
        raise ValueError("Production photodiode correction requires roman-telem --ice-monitor")
    if telemetry.get("reference_mode") != "first_non_warmup_per_led":
        raise ValueError(
            "Production photodiode correction requires merged telemetry with "
            "first non-warm-up LED references"
        )
    values = telemetry["values"]
    for observation in observations:
        key = (
            parse_iso_time(observation["exposure_start_time"]),
            parse_iso_time(observation["exposure_end_time"]),
            int(observation["led"]),
        )
        value = values.get(key)
        if value is None:
            identity = telemetry_identity(observation)
            candidates = telemetry.get("identity_values", {}).get(identity, [])
            if len(candidates) == 1:
                value = candidates[0][2]
        if value is None:
            raise ValueError(
                f"No unambiguous dark-subtracted ice-monitor row for "
                f"{observation['file_set_name']}: {key}; "
                f"telemetry_identity={telemetry_identity(observation)}"
            )
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


def query_mast_detector(missions, detector, program, page_size=5000):
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
            program=str(program),
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


def refresh_index(
    output_path,
    token_path,
    detectors=DEFAULT_DETECTORS,
    programs=UNIFIED_PROGRAMS,
    page_size=5000,
):
    """Refresh the persistent all-detector MAST supplement index."""
    if page_size <= 0:
        raise ValueError("MAST index page size must be positive")
    status(
        f"Refreshing unified MAST index for {len(detectors)} detectors and "
        f"programs {','.join(str(program) for program in programs)}"
    )
    token = _read_mast_token(token_path)
    try:
        from astroquery.mast import MastMissions
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("MAST index refresh requires astroquery") from error
    missions = MastMissions(mission="roman")
    missions.login(token=token)
    fresh_rows = []
    for program in programs:
        for detector in detectors:
            status(f"Querying MAST index for program {program}, {detector}")
            table = query_mast_detector(missions, detector, program, page_size)
            count = 0 if table is None else len(table)
            status(f"Found {count} program {program} {detector} L2 flat candidates")
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
    metadata_by_identity = {}
    for row in index_metadata.values():
        if not row.get("fileSetName") or not row.get("visit_id"):
            continue
        identity = product_identity(
            {
                "detector": row["detector"].strip(),
                "visit_id": row["visit_id"].strip(),
                "file_set_name": row.get("fileSetName", "").strip(),
                "led": row["led"],
            }
        )
        metadata_by_identity[identity] = row
    for record in records:
        row = index_metadata.get(record["file_set_name"]) or index_metadata.get(
            record["file_name"]
        )
        if row is None:
            row = metadata_by_identity.get(product_identity(record))
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
        record["program"] = str(row.get("program", "")).strip()
        if record["program"] not in UNIFIED_PROGRAMS:
            raise ValueError(
                f"Unsupported program for {record['file_name']}: {record['program']!r}"
            )
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
        "program",
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
                    "program": observation["program"],
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


def _tile_statistics(tile_id, tile_row, tile_col, row_start, row_stop, col_start, col_stop,
                     baseline_values, current_values):
    """Return one tile's moments for a supplied current-count scale."""
    values = np.log(current_values) - np.log(baseline_values)
    if values.size:
        reference_total = float(np.sum(baseline_values, dtype=np.float64))
        reference_squared = float(np.sum(baseline_values * baseline_values, dtype=np.float64))
        current_total = float(np.sum(current_values, dtype=np.float64))
        current_squared = float(np.sum(current_values * current_values, dtype=np.float64))
        reference_current = float(np.sum(baseline_values * current_values, dtype=np.float64))
        reference_minimum = float(np.min(baseline_values))
        reference_maximum = float(np.max(baseline_values))
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
    return {
        "tile_id": tile_id,
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


def _iter_tile_statistics(
    baseline_data,
    baseline_dq,
    current_data,
    current_dq,
    tile_size,
    current_scale=1.0,
    include_raw=False,
):
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
            current_raw = np.asarray(
                current_data[row_start:row_stop, col_start:col_stop], dtype=np.float64
            )
            base_valid = (
                np.isfinite(baseline)
                & (np.asarray(baseline_dq[row_start:row_stop, col_start:col_stop]) == 0)
                & (np.asarray(current_dq[row_start:row_stop, col_start:col_stop]) == 0)
                & (baseline > 0.0)
            )
            scaled_current = current_raw * float(current_scale)
            raw_valid = base_valid & np.isfinite(current_raw) & (current_raw > 0.0)
            scaled_valid = (
                base_valid
                & np.isfinite(scaled_current)
                & (scaled_current > 0.0)
            )
            tile_id = tile_row * tile_col_count + tile_col
            raw_tile = None
            if include_raw:
                raw_tile = _tile_statistics(
                    tile_id,
                    tile_row,
                    tile_col,
                    row_start,
                    row_stop,
                    col_start,
                    col_stop,
                    baseline[raw_valid].astype(np.float64),
                    current_raw[raw_valid],
                )
            scaled_values = scaled_current[scaled_valid]
            scaled_tile = _tile_statistics(
                tile_id,
                tile_row,
                tile_col,
                row_start,
                row_stop,
                col_start,
                col_stop,
                baseline[scaled_valid].astype(np.float64),
                scaled_values,
            )
            yield (raw_tile, scaled_tile) if include_raw else scaled_tile


def tile_sufficient_statistics(
    baseline_data,
    baseline_dq,
    current_data,
    current_dq,
    tile_size,
    current_scale=1.0,
):
    """Yield compact count and log-ratio moments for each detector tile."""
    yield from _iter_tile_statistics(
        baseline_data,
        baseline_dq,
        current_data,
        current_dq,
        tile_size,
        current_scale=current_scale,
    )


def tile_sufficient_statistics_pair(
    baseline_data,
    baseline_dq,
    current_data,
    current_dq,
    tile_size,
    current_scale=1.0,
):
    """Yield unscaled and photodiode-scaled moments from one image pair."""
    yield from _iter_tile_statistics(
        baseline_data,
        baseline_dq,
        current_data,
        current_dq,
        tile_size,
        current_scale=current_scale,
        include_raw=True,
    )


def build_observations(records):
    """Order products into shared epochs and assign elapsed physical days."""
    if not all(record.get("exposure_start_time") for record in records):
        raise ValueError("All products require exposure timestamps")
    for record in records:
        program = str(record.get("program", REFERENCE_PROGRAM))
        record["program"] = program
        record["sort_time"] = parse_iso_time(record["exposure_start_time"])

    # The historical RCS visit prefix groups the six LED observations, but
    # newer programs can reuse that prefix for another LED cycle. Split only
    # when an LED repeats in chronological order so both cycles remain data.
    records_by_prefix = {}
    for record in records:
        prefix = (record["program"], record["epoch_key"])
        records_by_prefix.setdefault(prefix, []).append(record)
    epoch_groups = {}
    for prefix, prefix_records in records_by_prefix.items():
        cycle = 0
        cycle_leds = set()
        for record in sorted(
            prefix_records,
            key=lambda row: (row["sort_time"], row["led"], row["file_set_name"]),
        ):
            if record["led"] in cycle_leds:
                cycle += 1
                cycle_leds = set()
            cycle_leds.add(record["led"])
            epoch_groups.setdefault((*prefix, cycle), []).append(record)
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
                    "program": str(record["program"]),
                    "repeat": repeat,
                    "group": len(observations),
                    "file_set_name": record["file_set_name"],
                    "visit_id": record["visit_id"],
                    "exposure_id": record.get("exposure_id", ""),
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


def statistics_file_fieldnames(path):
    """Validate a completed statistics CSV and return its field names."""
    path = Path(path)
    with path.open(newline="") as handle:
        fieldnames = tuple(csv.DictReader(handle).fieldnames or ())
    missing = set(BASE_STATISTICS_FIELDS).difference(fieldnames)
    if missing:
        raise ValueError(f"Existing statistics file is missing fields {sorted(missing)}")
    return fieldnames


def statistics_row(observation, tile):
    return {
        "led": observation["led"],
        "program": observation["program"],
        "repeat": observation["repeat"],
        "group": observation["group"],
        "file_set_name": observation["file_set_name"],
        "baseline_file_set_name": observation["baseline_file_set_name"],
        "exposure_start_time": observation["exposure_start_time"],
        "days": observation["days"],
        **tile,
    }


def write_statistics_parts(
    observation,
    baseline_data,
    baseline_dq,
    current_data,
    current_dq,
    tile_size,
    raw_part_path=None,
    scaled_part_path=None,
):
    """Atomically write whichever raw/scaled observation parts are missing."""
    targets = []
    if raw_part_path is not None:
        targets.append((Path(raw_part_path), "raw"))
    if scaled_part_path is not None:
        targets.append((Path(scaled_part_path), "scaled"))
    if not targets:
        raise ValueError("At least one statistics part path is required")
    temporary_paths = []
    try:
        with ExitStack() as stack:
            writers = {}
            for part_path, kind in targets:
                part_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = part_path.with_name(part_path.name + ".partial")
                temporary_paths.append((temporary, part_path))
                handle = stack.enter_context(temporary.open("w", newline=""))
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writers[kind] = (handle, writer)

            scale = float(observation.get("photodiode_scale", 1.0))
            if "raw" in writers and "scaled" in writers:
                tiles = tile_sufficient_statistics_pair(
                    baseline_data,
                    baseline_dq,
                    current_data,
                    current_dq,
                    tile_size,
                    current_scale=scale,
                )
            elif "raw" in writers:
                tiles = (
                    (tile, None)
                    for tile in tile_sufficient_statistics(
                        baseline_data,
                        baseline_dq,
                        current_data,
                        current_dq,
                        tile_size,
                        current_scale=1.0,
                    )
                )
            else:
                tiles = (
                    (None, tile)
                    for tile in tile_sufficient_statistics(
                        baseline_data,
                        baseline_dq,
                        current_data,
                        current_dq,
                        tile_size,
                        current_scale=scale,
                    )
                )
            last_tile = None
            for raw_tile, scaled_tile in tiles:
                last_tile = raw_tile if raw_tile is not None else scaled_tile
                if raw_tile is not None:
                    writers["raw"][1].writerow(statistics_row(observation, raw_tile))
                if scaled_tile is not None:
                    writers["scaled"][1].writerow(statistics_row(observation, scaled_tile))
            if last_tile is None:
                raise RuntimeError("Tile statistic generator returned no tiles")
            for handle, _ in writers.values():
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, part_path in temporary_paths:
            temporary.replace(part_path)
    finally:
        for temporary, _ in temporary_paths:
            if temporary.exists():
                temporary.unlink()
    return last_tile


def assemble_statistics_output(output_path, observations, parts_dir):
    output_path = Path(output_path)
    missing = [
        row["group"]
        for row in observations
        if not detector_part_path(parts_dir, row["group"]).exists()
    ]
    if missing:
        raise RuntimeError(f"Missing completed observation parts: {missing[:5]}")
    temporary_path = output_path.with_name(output_path.name + ".partial")
    try:
        with temporary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for observation in observations:
                with detector_part_path(parts_dir, observation["group"]).open(
                    newline=""
                ) as part:
                    writer.writerows(csv.DictReader(part))
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_detector_statistics(
    observations,
    output_path,
    tile_size,
    resume=True,
    fetch_image=None,
    raw_output_path=None,
):
    """Write photodiode-scaled and optional unscaled statistics together."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = Path(raw_output_path) if raw_output_path is not None else None
    if raw_output_path is not None:
        if raw_output_path == output_path:
            raise ValueError("Raw and photodiode-scaled statistics paths must differ")
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled_complete = resume and output_path.exists()
    raw_complete = raw_output_path is None or (resume and raw_output_path.exists())
    if scaled_complete:
        statistics_file_fieldnames(output_path)
    if raw_output_path is not None and raw_complete:
        statistics_file_fieldnames(raw_output_path)
    if scaled_complete and raw_complete:
        status(f"Skipping completed statistics outputs: {output_path}")
        return

    scaled_parts_dir = output_path.with_name(output_path.name + ".parts")
    raw_parts_dir = (
        raw_output_path.with_name(raw_output_path.name + ".parts")
        if raw_output_path is not None
        else None
    )
    scaled_parts_dir.mkdir(parents=True, exist_ok=True)
    if raw_parts_dir is not None:
        raw_parts_dir.mkdir(parents=True, exist_ok=True)
    for led in range(1, 7):
        led_rows = [row for row in observations if row["led"] == led]
        pending_rows = [
            row for row in led_rows
            if (
                not scaled_complete
                and not (resume and detector_part_path(scaled_parts_dir, row["group"]).exists())
            )
            or (
                raw_output_path is not None
                and not raw_complete
                and not (resume and detector_part_path(raw_parts_dir, row["group"]).exists())
            )
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
                scaled_part_path = (
                    None
                    if scaled_complete
                    or (resume and detector_part_path(scaled_parts_dir, row["group"]).exists())
                    else detector_part_path(scaled_parts_dir, row["group"])
                )
                raw_part_path = (
                    None
                    if raw_output_path is None
                    or raw_complete
                    or (resume and detector_part_path(raw_parts_dir, row["group"]).exists())
                    else detector_part_path(raw_parts_dir, row["group"])
                )
                last_tile = write_statistics_parts(
                    row,
                    baseline_data,
                    baseline_dq,
                    current_data,
                    current_dq,
                    tile_size,
                    raw_part_path=raw_part_path,
                    scaled_part_path=scaled_part_path,
                )
                if current_data is not baseline_data:
                    del current_data, current_dq
                status(f"{row['file_set_name']}: wrote {last_tile['tile_id'] + 1} tiles")
        finally:
            del baseline_data, baseline_dq
    if not scaled_complete:
        assemble_statistics_output(output_path, observations, scaled_parts_dir)
    if raw_output_path is not None and not raw_complete:
        assemble_statistics_output(raw_output_path, observations, raw_parts_dir)


def write_raw_detector_statistics(
    observations,
    output_path,
    tile_size,
    resume=True,
    fetch_image=None,
):
    """Write only truly raw detector statistics with resumable parts."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and output_path.exists():
        statistics_file_fieldnames(output_path)
        status(f"Skipping completed raw statistics output: {output_path}")
        return

    parts_dir = output_path.with_name(output_path.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    for led in range(1, 7):
        led_rows = [row for row in observations if int(row["led"]) == led]
        pending_rows = [
            row
            for row in led_rows
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
                last_tile = write_statistics_parts(
                    row,
                    baseline_data,
                    baseline_dq,
                    current_data,
                    current_dq,
                    tile_size,
                    raw_part_path=part_path,
                )
                if current_data is not baseline_data:
                    del current_data, current_dq
                status(f"{row['file_set_name']}: wrote {last_tile['tile_id'] + 1} raw tiles")
        finally:
            del baseline_data, baseline_dq
    assemble_statistics_output(output_path, observations, parts_dir)


def incremental_part_path(parts_dir, file_set_name):
    if not file_set_name or Path(file_set_name).name != file_set_name:
        raise ValueError(f"Unsafe file-set name: {file_set_name!r}")
    return parts_dir / f"{file_set_name}.csv"


def read_statistics_rows(path, return_fieldnames=False):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = set(BASE_STATISTICS_FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(f"Statistics file is missing fields {sorted(missing)}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Statistics file has no rows: {path}")
    if "program" not in fieldnames:
        fieldnames = (*fieldnames, "program")
    for row in rows:
        row.setdefault("program", "")
    return (rows, fieldnames) if return_fieldnames else rows


def post_temperature_repeats_by_tile(
    path,
    start_repeat,
    end_repeat,
    reference_led=3,
):
    """Return the usable reference-program repeats present in each tile."""
    repeats_by_tile = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["led"]) == int(reference_led)
                and str(row.get("program") or REFERENCE_PROGRAM) == REFERENCE_PROGRAM
                and int(row["repeat"]) >= int(start_repeat)
                and int(row["repeat"]) < int(end_repeat)
                and float(row["n_pair_valid"]) > 0.0
            ):
                repeats_by_tile.setdefault(int(row["tile_id"]), set()).add(
                    int(row["repeat"])
                )
    return {
        tile_id: tuple(sorted(repeats))
        for tile_id, repeats in repeats_by_tile.items()
    }


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
        "program",
        "repeat",
        "group",
        "file_set_name",
        "baseline_file_set_name",
        "exposure_start_time",
        "days",
    ):
        updated[field] = observation[field]
    return updated


def update_detector_statistics_incremental(
    observations,
    output_path,
    tile_size,
    fetch_image=None,
    raw_output_path=None,
    force_unscaled=False,
):
    """Reuse old file sets and fetch only new S3 or MAST products."""
    if raw_output_path is not None:
        if force_unscaled:
            raise ValueError("force_unscaled cannot be combined with raw_output_path")
        return update_detector_statistics_incremental_dual(
            observations,
            output_path,
            raw_output_path,
            tile_size,
            fetch_image=fetch_image,
        )
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
                            current_scale=(
                                1.0
                                if force_unscaled
                                else float(observation.get("photodiode_scale", 1.0))
                            ),
                        ):
                            last_tile = tile
                            writer.writerow(
                                {
                                    "led": observation["led"],
                                    "program": observation["program"],
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


def statistics_rows_by_file_set(rows):
    by_file_set = {}
    baselines = {}
    for row in rows:
        file_set_name = row["file_set_name"]
        tile_rows = by_file_set.setdefault(file_set_name, {})
        tile_id = row["tile_id"]
        if tile_id in tile_rows:
            raise ValueError(f"Duplicate tile {tile_id} for {file_set_name}")
        tile_rows[tile_id] = row
        led = int(row["led"])
        baseline = row["baseline_file_set_name"]
        previous = baselines.setdefault(led, baseline)
        if previous != baseline:
            raise ValueError(f"Existing statistics have inconsistent LED {led} baselines")
    return by_file_set, baselines


def update_detector_statistics_incremental_dual(
    observations,
    output_path,
    raw_output_path,
    tile_size,
    fetch_image=None,
):
    """Incrementally update scaled and unscaled products from shared fetches."""
    output_path = Path(output_path)
    raw_output_path = Path(raw_output_path)
    if not output_path.exists():
        raise ValueError(f"Incremental update requires an existing statistics file: {output_path}")
    if not raw_output_path.exists():
        raise ValueError(
            f"Incremental update requires an existing raw statistics file: {raw_output_path}"
        )
    scaled_rows, scaled_fieldnames = read_statistics_rows(
        output_path, return_fieldnames=True
    )
    raw_rows, raw_fieldnames = read_statistics_rows(
        raw_output_path, return_fieldnames=True
    )
    scaled_by_file_set, scaled_baselines = statistics_rows_by_file_set(scaled_rows)
    raw_by_file_set, raw_baselines = statistics_rows_by_file_set(raw_rows)
    if set(scaled_by_file_set) != set(raw_by_file_set):
        raise ValueError("Incremental update has mismatched raw and scaled observations")
    for file_set_name in scaled_by_file_set:
        if set(scaled_by_file_set[file_set_name]) != set(raw_by_file_set[file_set_name]):
            raise ValueError(
                f"Incremental update has mismatched raw/scaled tiles for {file_set_name}"
            )
    if scaled_baselines != raw_baselines:
        raise ValueError("Incremental update has inconsistent raw/scaled baselines")

    current_by_file_set = {row["file_set_name"]: row for row in observations}
    if len(current_by_file_set) != len(observations):
        raise ValueError("The source contains duplicate file-set names")
    removed = set(scaled_by_file_set).difference(current_by_file_set)
    if removed:
        raise ValueError("Incremental update refuses to discard existing observations")
    current_baselines = {}
    for observation in observations:
        current_baselines.setdefault(
            int(observation["led"]), observation["baseline_file_set_name"]
        )
    for led, baseline in scaled_baselines.items():
        if current_baselines.get(led) != baseline:
            raise ValueError(f"Incremental update cannot change LED {led} baseline")

    new_observations = [
        row for row in observations if row["file_set_name"] not in scaled_by_file_set
    ]
    scaled_parts_dir = output_path.with_name(output_path.name + ".incremental.parts")
    raw_parts_dir = raw_output_path.with_name(
        raw_output_path.name + ".incremental.parts"
    )
    scaled_parts_dir.mkdir(parents=True, exist_ok=True)
    raw_parts_dir.mkdir(parents=True, exist_ok=True)
    new_scaled_rows = {}
    new_raw_rows = {}
    for observation in new_observations:
        file_set_name = observation["file_set_name"]
        scaled_part = incremental_part_path(scaled_parts_dir, file_set_name)
        raw_part = incremental_part_path(raw_parts_dir, file_set_name)
        if scaled_part.exists():
            new_scaled_rows[file_set_name] = read_statistics_rows(scaled_part)
        if raw_part.exists():
            new_raw_rows[file_set_name] = read_statistics_rows(raw_part)

    for led in range(1, 7):
        pending_rows = [
            row
            for row in new_observations
            if int(row["led"]) == led
            and (
                row["file_set_name"] not in new_scaled_rows
                or row["file_set_name"] not in new_raw_rows
            )
        ]
        if not pending_rows:
            continue
        led_rows = [row for row in observations if int(row["led"]) == led]
        baseline_data, baseline_dq = load_observation_image(led_rows[0], fetch_image)
        try:
            for observation in pending_rows:
                file_set_name = observation["file_set_name"]
                need_scaled = file_set_name not in new_scaled_rows
                need_raw = file_set_name not in new_raw_rows
                if observation["file_set_name"] == observation["baseline_file_set_name"]:
                    current_data, current_dq = baseline_data, baseline_dq
                else:
                    current_data, current_dq = load_observation_image(
                        observation, fetch_image
                    )
                scaled_part = (
                    incremental_part_path(scaled_parts_dir, file_set_name)
                    if need_scaled
                    else None
                )
                raw_part = (
                    incremental_part_path(raw_parts_dir, file_set_name)
                    if need_raw
                    else None
                )
                last_tile = write_statistics_parts(
                    observation,
                    baseline_data,
                    baseline_dq,
                    current_data,
                    current_dq,
                    tile_size,
                    raw_part_path=raw_part,
                    scaled_part_path=scaled_part,
                )
                if need_scaled:
                    new_scaled_rows[file_set_name] = read_statistics_rows(scaled_part)
                if need_raw:
                    new_raw_rows[file_set_name] = read_statistics_rows(raw_part)
                if current_data is not baseline_data:
                    del current_data, current_dq
                status(f"{file_set_name}: wrote {last_tile['tile_id'] + 1} tiles")
        finally:
            del baseline_data, baseline_dq

    scaled_output_rows = []
    raw_output_rows = []
    for observation in observations:
        file_set_name = observation["file_set_name"]
        scaled_rows_for_observation = (
            scaled_by_file_set[file_set_name].values()
            if file_set_name in scaled_by_file_set
            else new_scaled_rows.get(file_set_name)
        )
        raw_rows_for_observation = (
            raw_by_file_set[file_set_name].values()
            if file_set_name in raw_by_file_set
            else new_raw_rows.get(file_set_name)
        )
        if scaled_rows_for_observation is None or raw_rows_for_observation is None:
            raise RuntimeError(f"Missing incremental statistics for {file_set_name}")
        scaled_output_rows.extend(
            update_statistics_metadata(row, observation)
            for row in scaled_rows_for_observation
        )
        raw_output_rows.extend(
            update_statistics_metadata(row, observation)
            for row in raw_rows_for_observation
        )
    scaled_output_rows = [
        {field: row[field] for field in scaled_fieldnames} for row in scaled_output_rows
    ]
    raw_output_rows = [
        {field: row[field] for field in raw_fieldnames} for row in raw_output_rows
    ]
    write_statistics_rows(output_path, scaled_output_rows, scaled_fieldnames)
    write_statistics_rows(raw_output_path, raw_output_rows, raw_fieldnames)
    status(
        f"Updated {output_path} and {raw_output_path} with {len(new_observations)} "
        f"new observations; reused {len(observations) - len(new_observations)} existing observations"
    )
    return output_path


DEFAULT_STEP_TIME = "2026-09-14T23:38:00Z"


def _shared_design_matrix(
    time_days,
    led_ids,
    period_days,
    step_arrays,
    led_step_arrays=(),
):
    """Build a pooled LED design with optional LED-specific ACM steps."""
    time_days = np.asarray(time_days, dtype=float)
    led_ids = np.asarray(led_ids, dtype=int)
    unique_leds = tuple(int(value) for value in np.unique(led_ids))
    angular_frequency = 2.0 * np.pi / period_days
    sine = np.sin(angular_frequency * time_days)
    cosine = np.cos(angular_frequency * time_days)
    columns = []
    for led in unique_leds:
        columns.append((led_ids == led).astype(float))
    for led in unique_leds:
        columns.append((led_ids == led).astype(float) * sine)
    for led in unique_leds:
        columns.append((led_ids == led).astype(float) * cosine)
    columns.extend(np.asarray(step, dtype=float) for step in step_arrays)
    for step in led_step_arrays:
        step = np.asarray(step, dtype=float)
        for led in unique_leds:
            columns.append((led_ids == led).astype(float) * step)
    return np.column_stack(columns), unique_leds


def _fit_shared_for_period(
    time_days,
    log_values,
    led_ids,
    step_arrays,
    period_days,
    led_step_arrays=(),
):
    design, unique_leds = _shared_design_matrix(
        time_days,
        led_ids,
        period_days,
        step_arrays,
        led_step_arrays=led_step_arrays,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, log_values, rcond=None)
    fitted = design @ coefficients
    residual = log_values - fitted
    return (
        float(np.sum(residual * residual)),
        coefficients,
        fitted,
        residual,
        unique_leds,
    )


def _fit_shared_log_curve(
    entries,
    masks,
    step_names,
    per_led_step_names=(),
):
    """Fit one sinusoid and shared ACM steps to all supplied LED rows.

    LED-specific intercepts and sinusoid amplitudes are nuisance terms only.
    Each ACM step has one common log-throughput coefficient for the pooled fit,
    unless its name is listed in ``per_led_step_names``.
    """
    if len(entries) != len(masks):
        raise ValueError("Shared curve entries and masks must have equal lengths")
    days_parts = []
    log_values_parts = []
    led_parts = []
    step_parts = {name: [] for name in step_names}
    per_led_step_parts = {name: [] for name in per_led_step_names}
    for entry, mask in zip(entries, masks):
        days = np.asarray(entry["days"], dtype=float)
        values = np.asarray(entry["values"], dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != days.shape or values.shape != days.shape:
            raise ValueError("Shared curve mask and data shapes must match")
        if not np.any(mask):
            continue
        selected_values = values[mask]
        if np.any(~np.isfinite(selected_values)) or np.any(selected_values <= 0.0):
            raise ValueError("Shared curve values must be finite and positive")
        days_parts.append(days[mask])
        log_values_parts.append(np.log(selected_values))
        led_parts.append(np.full(np.count_nonzero(mask), int(entry["led"]), dtype=int))
        for name in step_names:
            step = entry.get(name)
            if step is None:
                raise ValueError(f"Missing shared ACM step array: {name}")
            step = np.asarray(step, dtype=float)
            if step.shape != days.shape:
                raise ValueError(f"Shared ACM step shape does not match data: {name}")
            step_parts[name].append(step[mask])
        for name in per_led_step_names:
            step = entry.get(name)
            if step is None:
                raise ValueError(f"Missing LED-specific ACM step array: {name}")
            step = np.asarray(step, dtype=float)
            if step.shape != days.shape:
                raise ValueError(
                    f"LED-specific ACM step shape does not match data: {name}"
                )
            per_led_step_parts[name].append(step[mask])
    if not days_parts:
        raise ValueError("Shared curve has no observations in the requested fit segment")
    days = np.concatenate(days_parts)
    log_values = np.concatenate(log_values_parts)
    led_ids = np.concatenate(led_parts)
    step_arrays = [np.concatenate(step_parts[name]) for name in step_names]
    per_led_step_arrays = [
        np.concatenate(per_led_step_parts[name]) for name in per_led_step_names
    ]
    origin_day = float(np.min(days))
    time_days = days - origin_day
    unique_days = np.unique(np.sort(days))
    differences = np.diff(unique_days)
    differences = differences[differences > 0.0]
    if differences.size == 0:
        raise ValueError("Shared curve data must span a positive time interval")
    span = float(np.ptp(time_days))
    minimum_period = max(2.0 * float(np.median(differences)), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)

    def objective(log_period):
        return _fit_shared_for_period(
            time_days,
            log_values,
            led_ids,
            step_arrays,
            float(np.exp(log_period)),
            led_step_arrays=per_led_step_arrays,
        )[0]

    objective_values = np.asarray([objective(np.log(period)) for period in period_grid])
    best_index = int(np.argmin(objective_values))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(period_grid) - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            objective,
            bounds=(np.log(period_grid[lower_index]), np.log(period_grid[upper_index])),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))
    sse, coefficients, fitted, residual, unique_leds = _fit_shared_for_period(
        time_days,
        log_values,
        led_ids,
        step_arrays,
        period_days,
        led_step_arrays=per_led_step_arrays,
    )
    led_count = len(unique_leds)
    step_start = 3 * led_count
    step_coefficients = {
        name: float(coefficients[step_start + index])
        for index, name in enumerate(step_names)
    }
    per_led_step_start = step_start + len(step_names)
    per_led_step_coefficients = {}
    for name_index, name in enumerate(per_led_step_names):
        start = per_led_step_start + name_index * led_count
        per_led_step_coefficients[name] = {
            led: float(coefficients[start + index])
            for index, led in enumerate(unique_leds)
        }
    total_sum_squares = float(np.sum((log_values - np.mean(log_values)) ** 2))
    return {
        "period_days": period_days,
        "origin_day": origin_day,
        "led_order": unique_leds,
        "led_offsets": {
            led: float(coefficients[index]) for index, led in enumerate(unique_leds)
        },
        "led_sine_coefficients": {
            led: float(coefficients[led_count + index])
            for index, led in enumerate(unique_leds)
        },
        "led_cosine_coefficients": {
            led: float(coefficients[2 * led_count + index])
            for index, led in enumerate(unique_leds)
        },
        "step_coefficients": step_coefficients,
        "led_step_coefficients": per_led_step_coefficients,
        "acm_step_log_change": step_coefficients.get("first_step", 0.0),
        "additional_acm_step_log_change": step_coefficients.get(
            "additional_step", 0.0
        ),
        "late_acm_step_log_change": step_coefficients.get("late_step", 0.0),
        "pre_reset_acm_step_log_change": step_coefficients.get(
            "pre_reset_step", 0.0
        ),
        "sse_log_change_squared": sse,
        "rmse_log_change": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares)
        if total_sum_squares > 0.0
        else np.nan,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": residual,
    }


def _fit_count_sinusoid(days, values):
    """Fit the count-space temperature sinusoid for the reference LED."""
    from scipy.optimize import minimize_scalar

    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    if days.shape != values.shape or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Temperature-fit count data must be finite and positive")
    time_days = days - float(days[0])
    span = float(np.ptp(time_days))
    if span <= 0.0:
        raise ValueError("Temperature-fit count data must span a positive time interval")
    differences = np.diff(np.unique(np.sort(days)))
    differences = differences[differences > 0.0]
    if differences.size == 0:
        raise ValueError("Temperature-fit count data must span a positive time interval")
    minimum_period = max(2.0 * float(np.median(differences)), 0.05)
    maximum_period = 100.0 * span
    period_grid = np.geomspace(minimum_period, maximum_period, 600)

    def fit_for_period(period_days):
        angular_frequency = 2.0 * np.pi / period_days
        design = np.column_stack(
            (
                np.ones_like(time_days),
                np.sin(angular_frequency * time_days),
                np.cos(angular_frequency * time_days),
            )
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ coefficients
        residual = values - fitted
        return float(np.sum(residual * residual)), coefficients, fitted, residual

    objective_values = np.asarray(
        [fit_for_period(period)[0] for period in period_grid]
    )
    best_index = int(np.argmin(objective_values))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(period_grid) - 1, best_index + 1)
    if lower_index == upper_index:
        period_days = float(period_grid[best_index])
    else:
        result = minimize_scalar(
            lambda log_period: fit_for_period(float(np.exp(log_period)))[0],
            bounds=(np.log(period_grid[lower_index]), np.log(period_grid[upper_index])),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        period_days = float(np.exp(result.x))
    sse, coefficients, fitted, residual = fit_for_period(period_days)
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    return {
        "period_days": period_days,
        "origin_day": float(days[0]),
        "offset": float(coefficients[0]),
        "sine": float(coefficients[1]),
        "cosine": float(coefficients[2]),
        "sse": sse,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - sse / total_sum_squares)
        if total_sum_squares > 0.0
        else np.nan,
        "fitted": fitted,
        "residual": residual,
    }


def _evaluate_count_sinusoid(fit, days):
    time_days = np.asarray(days, dtype=float) - fit["origin_day"]
    angular_frequency = 2.0 * np.pi / fit["period_days"]
    return (
        fit["offset"]
        + fit["sine"] * np.sin(angular_frequency * time_days)
        + fit["cosine"] * np.cos(angular_frequency * time_days)
    )


def load_temperature_quadratic_coefficients(path):
    """Load frozen per-detector/per-tile WTSM00 correction parabolas."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Temperature quadratic coefficient table does not exist: {path}"
        )
    required = {
        "detector",
        "tile_size",
        "tile_id",
        "reference_temperature_k",
        "a_log_correction",
        "b_log_correction_per_k",
        "c_log_correction_per_k2",
    }
    coefficients = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Temperature quadratic table is missing fields: {sorted(missing)}"
            )
        for row in reader:
            detector = str(row["detector"]).strip().upper()
            tile_size = int(row["tile_size"])
            if tile_size < 1:
                raise ValueError(
                    f"Invalid temperature quadratic tile size for "
                    f"{(detector, tile_size, row['tile_id'])}"
                )
            tile_id = int(row["tile_id"])
            detector_coefficients = coefficients.setdefault(detector, {})
            tile_coefficients = detector_coefficients.setdefault(tile_size, {})
            if tile_id in tile_coefficients:
                raise ValueError(
                    f"Duplicate temperature quadratic coefficient for "
                    f"{(detector, tile_size, tile_id)}"
                )
            values = {
                "reference_temperature_k": float(row["reference_temperature_k"]),
                "a_log_correction": float(row["a_log_correction"]),
                "b_log_correction_per_k": float(row["b_log_correction_per_k"]),
                "c_log_correction_per_k2": float(row["c_log_correction_per_k2"]),
                "normalization_offset_log_correction": float(
                    row.get("normalization_offset_log_correction", "0") or 0.0
                ),
                "fit_r2": float(row.get("r2_log", "nan")),
                "fit_points": int(float(row.get("fit_points", "0"))),
                "rmse_percent": float(row.get("rmse_percent", "nan")),
                "max_abs_percent": float(row.get("max_abs_percent", "nan")),
            }
            if not all(
                np.isfinite(values[name])
                for name in (
                    "reference_temperature_k",
                    "a_log_correction",
                    "b_log_correction_per_k",
                    "c_log_correction_per_k2",
                )
            ):
                raise ValueError(
                    "Non-finite temperature quadratic coefficient for "
                    f"{(detector, tile_size, tile_id)}"
                )
            tile_coefficients[tile_id] = values
    if not coefficients:
        raise ValueError(f"Temperature quadratic coefficient table is empty: {path}")
    return coefficients


def _temperature_quadratic_fit(coefficients):
    """Return the frozen correction parabola in the local fit schema."""
    raw_intercept = float(coefficients["a_log_correction"])
    return {
        "reference_temperature_k": float(coefficients["reference_temperature_k"]),
        # The temperature correction is normalized to zero at the operating
        # temperature; retain the original intercept only for provenance.
        "a_log_correction": 0.0,
        "b_log_correction_per_k": float(coefficients["b_log_correction_per_k"]),
        "c_log_correction_per_k2": float(coefficients["c_log_correction_per_k2"]),
        "normalization_offset_log_correction": float(
            coefficients.get("normalization_offset_log_correction", raw_intercept)
        ),
        "r2": float(coefficients.get("fit_r2", np.nan)),
        "fit_points": int(coefficients.get("fit_points", 0)),
        "rmse_percent": float(coefficients.get("rmse_percent", np.nan)),
        "max_abs_percent": float(coefficients.get("max_abs_percent", np.nan)),
        # These legacy keys are retained for summary compatibility only.
        "slope_log_per_k": float(coefficients["b_log_correction_per_k"]),
        "rmse_counts": np.nan,
    }


def _evaluate_temperature_quadratic(fit, temperatures):
    temperatures = np.asarray(temperatures, dtype=float)
    delta_temperature = temperatures - fit["reference_temperature_k"]
    return (
        fit["a_log_correction"]
        + fit["b_log_correction_per_k"] * delta_temperature
        + fit["c_log_correction_per_k2"] * delta_temperature**2
    )


def _fit_count_temperature(temperatures, values, reference_temperature_k=None):
    """Fit log count response as a linear function of detector temperature."""
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    if (
        temperatures.shape != values.shape
        or temperatures.size < 2
        or np.any(~np.isfinite(temperatures))
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("Temperature-fit data must be finite and positive")
    if np.ptp(temperatures) <= 0.0:
        raise ValueError("Temperature-fit data must span a positive temperature interval")
    if reference_temperature_k is None:
        reference_temperature_k = float(np.median(temperatures))
    reference_temperature_k = float(reference_temperature_k)
    design = np.column_stack(
        (np.ones_like(temperatures), temperatures - reference_temperature_k)
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, np.log(values), rcond=None
    )
    fitted_log = design @ coefficients
    residual_log = np.log(values) - fitted_log
    fitted = np.exp(fitted_log)
    total_sum_squares = float(
        np.sum((np.log(values) - np.mean(np.log(values))) ** 2)
    )
    return {
        "reference_temperature_k": reference_temperature_k,
        "offset_log_counts": float(coefficients[0]),
        "slope_log_per_k": float(coefficients[1]),
        "sse_log_squared": float(np.sum(residual_log * residual_log)),
        "rmse_counts": float(np.sqrt(np.mean((values - fitted) ** 2))),
        "r2": float(
            1.0 - np.sum(residual_log * residual_log) / total_sum_squares
        )
        if total_sum_squares > 0.0
        else np.nan,
        "fitted": fitted,
        "residual": values - fitted,
    }


def _evaluate_count_temperature(fit, temperatures):
    temperatures = np.asarray(temperatures, dtype=float)
    return np.exp(
        fit["offset_log_counts"]
        + fit["slope_log_per_k"]
        * (temperatures - fit["reference_temperature_k"])
    )


def _fit_free_count_step(
    entries,
    acm_log_by_led,
    decon_day,
    fit_window_days=DEFAULT_DECON_FIT_WINDOW_DAYS,
):
    """Fit a local count-space step without imposing a post-step baseline."""
    if decon_day is None:
        return None
    if fit_window_days <= 0.0:
        raise ValueError("The decon fit window must be positive")
    days_parts = []
    log_values_parts = []
    led_parts = []
    temperature_parts = []
    step_parts = []
    for entry in entries:
        step = entry.get("decon_step")
        if step is None:
            continue
        days = np.asarray(entry["days"], dtype=float)
        values = np.asarray(entry["values"], dtype=float)
        mask = np.abs(days - float(decon_day)) <= float(fit_window_days)
        if not np.any(mask):
            continue
        days_parts.append(days[mask])
        log_values_parts.append(
            np.log(values[mask]) + np.asarray(acm_log_by_led[int(entry["led"])])[mask]
        )
        led_parts.append(np.full(np.count_nonzero(mask), int(entry["led"]), dtype=int))
        temperature = entry.get("temperature_k")
        if temperature is None:
            temperature_parts.append(None)
        else:
            temperature_parts.append(np.asarray(temperature, dtype=float)[mask])
        step_parts.append(np.asarray(step, dtype=float)[mask])
    if not days_parts:
        return None
    days = np.concatenate(days_parts)
    log_values = np.concatenate(log_values_parts)
    led_ids = np.concatenate(led_parts)
    steps = np.concatenate(step_parts)
    pre_points = int(np.count_nonzero(steps < 0.5))
    post_points = int(np.count_nonzero(steps >= 0.5))
    if pre_points < 2 or post_points < 2:
        return None

    unique_leds = tuple(int(value) for value in np.unique(led_ids))
    columns = [
        (led_ids == led).astype(float) for led in unique_leds
    ]
    centered_days = days - float(decon_day)
    columns.append(centered_days)
    has_temperature = bool(
        temperature_parts
        and all(part is not None for part in temperature_parts)
    )
    if has_temperature:
        temperatures = np.concatenate(temperature_parts)
        if np.any(~np.isfinite(temperatures)) or np.ptp(temperatures) <= 0.0:
            has_temperature = False
        else:
            columns.append(temperatures - float(np.median(temperatures)))
    columns.append(steps)
    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(design, log_values, rcond=None)
    residual = log_values - design @ coefficients
    step_coefficient = float(coefficients[-1])
    return {
        "step_log_change": step_coefficient,
        "step_log_correction": -step_coefficient,
        "fit_points": int(log_values.size),
        "pre_points": pre_points,
        "post_points": post_points,
        "rmse_log": float(np.sqrt(np.mean(residual * residual))),
        "has_temperature": has_temperature,
    }


def load_count_statistics(
    path,
    additional_acm_step_repeat,
    pre_reset_acm_step_repeat,
    late_acm_step_repeat=None,
    disable_late_acm_step=False,
    day22_acm_step_repeat=None,
    day26_acm_step_repeat=None,
    temperature_series=None,
    decon_time=None,
):
    event_time = parse_iso_time(DEFAULT_STEP_TIME)
    decon_event_time = None if decon_time is None else parse_iso_time(decon_time)
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
            temperature_k = None
            if temperature_series is not None:
                temperature_k = interpolate_temperature(
                    [row["exposure_start_time"] for row in led_rows],
                    temperature_series,
                )
            decon_step = None
            if decon_event_time is not None:
                decon_step = np.asarray(
                    [
                        float(parse_iso_time(row["exposure_start_time"]) >= decon_event_time)
                        for row in led_rows
                    ],
                    dtype=float,
                )
                decon_step -= decon_step[0]
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
            if disable_late_acm_step:
                late_step = None
            elif late_acm_step_repeat is None:
                late_repeat = max(int(row["repeat"]) for row in led_rows)
                late_step = np.asarray(
                    [float(int(row["repeat"]) == late_repeat) for row in led_rows],
                    dtype=float,
                )
            else:
                late_step = np.asarray(
                    [
                        float(int(row["repeat"]) >= int(late_acm_step_repeat))
                        for row in led_rows
                    ],
                    dtype=float,
                )
            if late_step is not None:
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
            day22_step = None
            if day22_acm_step_repeat is not None:
                day22_step = np.asarray(
                    [float(int(row["repeat"]) >= day22_acm_step_repeat) for row in led_rows],
                    dtype=float,
                )
                day22_step -= day22_step[0]
            day26_step = None
            if day26_acm_step_repeat is not None:
                day26_step = np.asarray(
                    [float(int(row["repeat"]) >= day26_acm_step_repeat) for row in led_rows],
                    dtype=float,
                )
                day26_step -= day26_step[0]
            led_data[led] = {
                "rows": led_rows,
                "days": days,
                "values": values,
                "program": np.asarray(
                    [str(row.get("program") or REFERENCE_PROGRAM) for row in led_rows],
                    dtype=object,
                ),
                "first_step": first_step,
                "additional_step": additional_step,
                "pre_reset_step": pre_reset_step,
                "day22_step": day22_step,
                "day26_step": day26_step,
                "late_step": late_step,
                "repeat": np.asarray([int(row["repeat"]) for row in led_rows]),
                "temperature_k": temperature_k,
                "decon_step": decon_step,
            }
        tiles[tile_id] = {"rows": rows, "led": led_data}
    return tiles


def fit_count_corrections(
    tile,
    reference_led,
    minimum_day,
    additional_acm_step_repeat,
    operational_temperature_day,
    pre_reset_acm_step_repeat,
    late_acm_step_repeat=None,
    disable_late_acm_step=False,
    post_temperature_fit_start_repeat=None,
    post_acm_fit_start_repeat=None,
    per_led_acm_step_names=(),
    post_temperature_fit_end_repeat=None,
    pre_temperature_fit_start_repeat=None,
    pre_temperature_fit_end_repeat=None,
    day22_acm_step_repeat=None,
    day26_acm_step_repeat=None,
    post_acm_fit_end_repeat=None,
    temperature_mnemonic=MPA_TEMPERATURE_MNEMONIC,
    temperature_quadratic_coefficients=None,
    decon_time=None,
    decon_day=None,
    decon_fit_window_days=DEFAULT_DECON_FIT_WINDOW_DAYS,
):
    """Fit grey ACM steps jointly and apply the frozen temperature correction.

    The ACM fit pools all six LEDs with one coefficient per throughput step.
    LED-specific offsets and time-sinusoid terms are nuisance terms and are never
    emitted as corrections. When WTSM00 data are present in ``tile``, the
    temperature term is instead the supplied per-tile WTSM00 quadratic in log
    correction versus detector temperature. The frozen coefficients are applied
    identically to every LED.
    The frozen WTSM00 quadratic is applied to every finite-temperature row. The
    repeat boundaries only control the legacy time-sinusoid fallback and the
    ACM fit partitions.
    Names in ``per_led_acm_step_names`` receive separate step coefficients for
    each LED in the pooled ACM fit.
    ``post_temperature_fit_end_repeat`` and ``post_acm_fit_end_repeat`` limit
    the legacy temperature and pooled ACM fits respectively.
    When ``day22_acm_step_repeat`` is supplied, its per-LED coefficient is
    estimated locally from the neighboring reference rows.
    When ``day26_acm_step_repeat`` is supplied, its shared grey coefficient is
    estimated locally from the neighboring all-program rows.
    The legacy post-operational gain extrapolation is applied only to the first
    Program 1063 batch, not to later 1063 batches.
    When temperature telemetry is omitted, the legacy pre-operational sine is
    fitted only on
    ``[pre_temperature_fit_start_repeat, pre_temperature_fit_end_repeat)``.
    When ``decon_day`` is supplied, the independently fitted decon step remains
    additive in log-correction space.
    """
    led_entries = [
        {"led": led, **tile["led"][led]} for led in range(1, 7)
    ]
    reference = tile["led"][reference_led]
    use_mpa_temperature = reference.get("temperature_k") is not None
    tile_id = int(tile["rows"][0]["tile_id"])
    quadratic_fit = None
    if use_mpa_temperature:
        if temperature_quadratic_coefficients is None:
            raise ValueError(
                "WTSM00 temperature data requires frozen per-tile quadratic "
                "coefficients"
            )
        tile_coefficients = temperature_quadratic_coefficients.get(tile_id)
        if tile_coefficients is None:
            raise ValueError(
                f"Missing WTSM00 quadratic coefficients for tile {tile_id}"
            )
        quadratic_fit = _temperature_quadratic_fit(tile_coefficients)
    if use_mpa_temperature and any(
        entry.get("temperature_k") is None for entry in led_entries
    ):
        raise ValueError("WTSM00 temperature data must be present for every LED")
    if use_mpa_temperature and any(
        np.any(~np.isfinite(np.asarray(entry["temperature_k"], dtype=float)))
        for entry in led_entries
    ):
        raise ValueError("WTSM00 temperature data must be finite for every LED")
    temperature_model = (
        "mpa_wtsm00_quadratic_per_tile"
        if use_mpa_temperature
        else "legacy_time_sinusoid"
    )
    if post_acm_fit_start_repeat is None:
        post_acm_fit_start_repeat = post_temperature_fit_start_repeat
    if post_temperature_fit_end_repeat is None:
        post_temperature_fit_end_repeat = late_acm_step_repeat
    if post_acm_fit_end_repeat is None:
        post_acm_fit_end_repeat = post_temperature_fit_end_repeat
    post_temperature_application_end_repeat = (
        post_acm_fit_start_repeat
        if post_acm_fit_start_repeat is not None
        else post_temperature_fit_start_repeat
    )

    def first_temperature_batch_mask(entry):
        mask = entry["program"] == TEMPERATURE_CORRECTION_PROGRAM
        if post_temperature_application_end_repeat is not None:
            return mask & (
                entry["repeat"] < int(post_temperature_application_end_repeat)
            )
        if operational_temperature_day is None:
            return np.zeros_like(mask, dtype=bool)
        return mask & (entry["days"] < float(operational_temperature_day))

    def available_step_names(names):
        return tuple(
            name
            for name in names
            if all(entry.get(name) is not None for entry in led_entries)
        )

    source_masks = [entry["program"] == REFERENCE_PROGRAM for entry in led_entries]
    if operational_temperature_day is None:
        shared_step_names = available_step_names(
            ("first_step", "additional_step", "late_step")
        )
        shared_masks = [
            mask & (entry["days"] > minimum_day)
            for entry, mask in zip(led_entries, source_masks)
        ]
        pre_acm_fit = _fit_shared_log_curve(
            led_entries, shared_masks, shared_step_names
        )
        post_acm_fit = None
        pre_step_names = shared_step_names
        post_step_names = ()
    else:
        pre_step_names = available_step_names(("first_step", "pre_reset_step"))
        post_step_names = available_step_names(("additional_step", "late_step"))
        per_led_post_step_names = tuple(
            name for name in per_led_acm_step_names if name in post_step_names
        )
        shared_post_step_names = tuple(
            name for name in post_step_names if name not in per_led_post_step_names
        )
        if post_acm_fit_start_repeat is None:
            pre_operational = lambda entry: entry["days"] < operational_temperature_day
            post_operational = lambda entry: entry["days"] >= operational_temperature_day
        else:
            pre_operational = lambda entry: entry["repeat"] < int(
                post_acm_fit_start_repeat
            )
            post_operational = lambda entry: entry["repeat"] >= int(
                post_acm_fit_start_repeat
            )
        pre_masks = [
            mask & (entry["days"] > minimum_day) & pre_operational(entry)
            for entry, mask in zip(led_entries, source_masks)
        ]
        post_masks = [
            mask & post_operational(entry)
            for entry, mask in zip(led_entries, source_masks)
        ]
        # Keep the legacy temperature cutoff separate from the ACM fit cutoff.
        if post_acm_fit_end_repeat is not None:
            post_masks = [
                mask
                & (entry["repeat"] < int(post_acm_fit_end_repeat))
                for entry, mask in zip(led_entries, post_masks)
            ]
        pre_acm_fit = _fit_shared_log_curve(
            led_entries, pre_masks, pre_step_names
        )
        post_acm_fit = _fit_shared_log_curve(
            led_entries,
            post_masks,
            shared_post_step_names,
            per_led_step_names=per_led_post_step_names,
        )

    component_fields = {
        "first_step": "acm_step_log_offset",
        "pre_reset_step": "pre_reset_acm_step_log_offset",
        "additional_step": "additional_acm_step_log_offset",
        "day22_step": "day22_acm_step_log_offset",
        "day26_step": "day26_acm_step_log_offset",
        "late_step": "late_acm_step_log_offset",
    }

    def local_day22_step_coefficient(entry):
        if day22_acm_step_repeat is None:
            return 0.0
        repeat = np.asarray(entry["repeat"], dtype=int)
        reference_mask = entry["program"] == REFERENCE_PROGRAM
        event_indices = np.flatnonzero(
            reference_mask & (repeat == int(day22_acm_step_repeat))
        )
        before_indices = np.flatnonzero(
            reference_mask & (repeat < int(day22_acm_step_repeat))
        )
        after_mask = reference_mask & (repeat > int(day22_acm_step_repeat))
        if late_acm_step_repeat is not None:
            after_mask &= repeat < int(late_acm_step_repeat)
        elif post_acm_fit_end_repeat is not None:
            after_mask &= repeat < int(post_acm_fit_end_repeat)
        after_indices = np.flatnonzero(after_mask)
        if (
            not event_indices.size
            or before_indices.size < 2
            or after_indices.size < 2
        ):
            return 0.0
        event_index = int(event_indices[0])
        before_indices = before_indices[-4:]
        after_indices = after_indices[:4]
        event_day = float(entry["days"][event_index])
        before_days = np.asarray(entry["days"][before_indices], dtype=float) - event_day
        after_days = np.asarray(entry["days"][after_indices], dtype=float) - event_day
        if np.ptp(before_days) <= 0.0 or np.ptp(after_days) <= 0.0:
            return 0.0
        before_logs = np.log(np.asarray(entry["values"][before_indices], dtype=float))
        after_logs = np.log(np.asarray(entry["values"][after_indices], dtype=float))
        before_fit = np.polyfit(before_days, before_logs, 1)
        after_fit = np.polyfit(after_days, after_logs, 1)
        return float(np.polyval(after_fit, 0.0) - np.polyval(before_fit, 0.0))

    def local_day26_step_coefficient():
        if day26_acm_step_repeat is None:
            return 0.0
        coefficients = []
        for entry in led_entries:
            repeat = np.asarray(entry["repeat"], dtype=int)
            event_indices = np.flatnonzero(
                repeat == int(day26_acm_step_repeat)
            )
            before_indices = np.flatnonzero(
                repeat < int(day26_acm_step_repeat)
            )
            after_indices = np.flatnonzero(
                repeat > int(day26_acm_step_repeat)
            )
            if (
                not event_indices.size
                or before_indices.size < 2
                or after_indices.size < 2
            ):
                continue
            before_indices = before_indices[-4:]
            after_indices = after_indices[:4]
            event_day = float(entry["days"][int(event_indices[0])])
            before_days = (
                np.asarray(entry["days"][before_indices], dtype=float) - event_day
            )
            after_days = (
                np.asarray(entry["days"][after_indices], dtype=float) - event_day
            )
            if np.ptp(before_days) <= 0.0 or np.ptp(after_days) <= 0.0:
                continue
            before_logs = np.log(
                np.asarray(entry["values"][before_indices], dtype=float)
            )
            after_logs = np.log(
                np.asarray(entry["values"][after_indices], dtype=float)
            )
            before_fit = np.polyfit(before_days, before_logs, 1)
            after_fit = np.polyfit(after_days, after_logs, 1)
            coefficients.append(
                float(np.polyval(after_fit, 0.0) - np.polyval(before_fit, 0.0))
            )
        return float(np.median(coefficients)) if coefficients else 0.0

    day22_step_coefficients = {
        int(entry["led"]): local_day22_step_coefficient(entry)
        for entry in led_entries
    }
    day26_step_coefficient = local_day26_step_coefficient()
    acm_components_by_led = {}
    acm_log_by_led = {}
    def fit_step_coefficient(fit, name, entry=None):
        if entry is not None:
            led_coefficients = fit.get("led_step_coefficients", {}).get(name)
            if led_coefficients is not None:
                return led_coefficients.get(int(entry["led"]), 0.0)
        return fit["step_coefficients"].get(name, 0.0)

    for entry in led_entries:
        components = {
            field: np.zeros_like(entry["days"], dtype=float)
            for field in component_fields.values()
        }
        for fit, step_names in (
            (pre_acm_fit, pre_step_names),
            (post_acm_fit, post_step_names),
        ):
            if fit is None:
                continue
            for name in step_names:
                coefficient = fit_step_coefficient(fit, name, entry)
                components[component_fields[name]] = -coefficient * np.asarray(
                    entry[name], dtype=float
                )
        if entry.get("day22_step") is not None:
            components[component_fields["day22_step"]] = (
                -day22_step_coefficients[int(entry["led"])]
                * np.asarray(entry["day22_step"], dtype=float)
            )
        if entry.get("day26_step") is not None:
            components[component_fields["day26_step"]] = (
                -day26_step_coefficient
                * np.asarray(entry["day26_step"], dtype=float)
            )
        acm_components_by_led[int(entry["led"])] = components
        acm_log_by_led[int(entry["led"])] = sum(components.values())

    reference_acm_values = reference["values"] * np.exp(
        acm_log_by_led[int(reference_led)]
    )
    reference_source_mask = reference["program"] == REFERENCE_PROGRAM
    decon_fit = _fit_free_count_step(
        led_entries,
        acm_log_by_led,
        decon_day,
        fit_window_days=decon_fit_window_days,
    )

    if operational_temperature_day is None:
        pre_temperature_mask = reference_source_mask & (
            reference["days"] > minimum_day
        )
        pre_temperature_fit_start = None
        pre_temperature_fit_end = None
    else:
        if post_temperature_fit_start_repeat is None:
            post_temperature_start_mask = reference["days"] >= operational_temperature_day
        else:
            post_temperature_start_mask = reference["repeat"] >= int(
                post_temperature_fit_start_repeat
            )
        pre_temperature_mask = (
            reference_source_mask
            & (reference["days"] > minimum_day)
            & (reference["days"] < operational_temperature_day)
        )
        if pre_temperature_fit_start_repeat is not None:
            pre_temperature_mask &= reference["repeat"] >= int(
                pre_temperature_fit_start_repeat
            )
        if pre_temperature_fit_end_repeat is not None:
            pre_temperature_mask &= reference["repeat"] < int(
                pre_temperature_fit_end_repeat
            )
        selected_repeats = reference["repeat"][pre_temperature_mask]
        pre_temperature_fit_start = (
            int(np.min(selected_repeats))
            if selected_repeats.size
            else pre_temperature_fit_start_repeat
        )
        pre_temperature_fit_end = (
            int(pre_temperature_fit_end_repeat)
            if pre_temperature_fit_end_repeat is not None
            else (
                int(np.max(selected_repeats)) + 1
                if selected_repeats.size
                else None
            )
        )
    pre_temperature_fit_points = int(np.count_nonzero(pre_temperature_mask))
    if use_mpa_temperature:
        pre_temperature_fit = quadratic_fit
    else:
        pre_temperature_fit = _fit_count_sinusoid(
            reference["days"][pre_temperature_mask],
            reference_acm_values[pre_temperature_mask],
        )

    early_reference_mask = reference_source_mask & (
        reference["days"] <= minimum_day
    )
    early_reference_repeats = reference["repeat"][early_reference_mask]
    if early_reference_repeats.size:
        if use_mpa_temperature:
            early_reference_factor = np.exp(
                _evaluate_temperature_quadratic(
                    pre_temperature_fit,
                    reference["temperature_k"][early_reference_mask],
                )
            )
        else:
            fitted_early = _evaluate_count_sinusoid(
                pre_temperature_fit,
                reference["days"][early_reference_mask],
            )
            early_reference_log_factor = np.log(
                fitted_early / reference_acm_values[early_reference_mask]
            )
            early_reference_factor = np.exp(early_reference_log_factor)
    else:
        early_reference_factor = np.asarray([], dtype=float)

    post_temperature_fit = quadratic_fit if use_mpa_temperature else None
    post_reference_repeats = np.asarray([], dtype=int)
    post_reference_factor = np.asarray([], dtype=float)
    post_temperature_fit_points = 0
    if operational_temperature_day is not None:
        post_temperature_mask = reference_source_mask & post_temperature_start_mask
        # The late ACM coefficient may be disabled because side-swap handles
        # that discontinuity, but its event still bounds the temperature fit.
        if post_temperature_fit_end_repeat is None:
            raise ValueError(
                "The post-operational fit requires an end boundary"
            )
        post_temperature_mask &= reference["repeat"] < int(
            post_temperature_fit_end_repeat
        )
        post_temperature_fit_points = int(np.count_nonzero(post_temperature_mask))
        if use_mpa_temperature:
            post_temperature_fit = quadratic_fit
        else:
            post_temperature_fit = _fit_count_sinusoid(
                reference["days"][post_temperature_mask],
                reference_acm_values[post_temperature_mask],
            )
        reference_temperature_rows = first_temperature_batch_mask(reference)
        post_reference_repeats = reference["repeat"][reference_temperature_rows]
        if post_reference_repeats.size:
            if use_mpa_temperature:
                post_reference_factor = np.exp(
                    _evaluate_temperature_quadratic(
                        post_temperature_fit,
                        reference["temperature_k"][reference_temperature_rows],
                    )
                )
            else:
                fitted_post = _evaluate_count_sinusoid(
                    post_temperature_fit,
                    reference["days"][reference_temperature_rows],
                )
                post_reference_factor = fitted_post / reference_acm_values[
                    reference_temperature_rows
                ]

    def shared_step_coefficient(name, entry=None):
        if name in pre_acm_fit["step_coefficients"]:
            return fit_step_coefficient(pre_acm_fit, name, entry)
        if post_acm_fit is not None and (
            name in post_acm_fit["step_coefficients"]
            or name in post_acm_fit.get("led_step_coefficients", {})
        ):
            return fit_step_coefficient(post_acm_fit, name, entry)
        return 0.0

    correction_rows = []
    for entry in led_entries:
        led = int(entry["led"])
        early_log_factor = np.zeros_like(entry["days"], dtype=float)
        post_log_factor = np.zeros_like(entry["days"], dtype=float)
        if use_mpa_temperature:
            temperature_log_factor = _evaluate_temperature_quadratic(
                quadratic_fit,
                entry["temperature_k"],
            )
            reference_rows = entry["program"] == REFERENCE_PROGRAM
            early_log_factor[reference_rows] = temperature_log_factor[reference_rows]
            post_log_factor[~reference_rows] = temperature_log_factor[~reference_rows]
        else:
            early_rows = (entry["days"] <= minimum_day) & (
                entry["program"] == REFERENCE_PROGRAM
            )
            if early_reference_repeats.size and np.any(early_rows):
                early_factors = np.interp(
                    entry["repeat"][early_rows],
                    early_reference_repeats,
                    early_reference_factor,
                )
                if np.any(~np.isfinite(early_factors)) or np.any(early_factors <= 0.0):
                    raise ValueError("LED 3 early temperature factors are invalid")
                early_log_factor[early_rows] = np.log(early_factors)

            temperature_rows = first_temperature_batch_mask(entry)
            if np.any(temperature_rows):
                if post_reference_repeats.size:
                    post_factors = np.interp(
                        entry["repeat"][temperature_rows],
                        post_reference_repeats,
                        post_reference_factor,
                    )
                    if np.any(~np.isfinite(post_factors)) or np.any(
                        post_factors <= 0.0
                    ):
                        raise ValueError("LED 3 post-temperature factors are invalid")
                    post_log_factor[temperature_rows] = np.log(post_factors)

        total_log_correction = (
            acm_log_by_led[led] + early_log_factor + post_log_factor
        )
        decon_log_factor = np.zeros_like(entry["days"], dtype=float)
        if decon_fit is not None and entry.get("decon_step") is not None:
            decon_log_factor = (
                decon_fit["step_log_correction"]
                * np.asarray(entry["decon_step"], dtype=float)
            )
            total_log_correction += decon_log_factor
        for index, row in enumerate(entry["rows"]):
            correction_rows.append(
                {
                    "detector": "",
                    "tile_id": int(row["tile_id"]),
                    "group": int(row["group"]),
                    "days": float(row["days"]),
                    "program": str(row.get("program") or REFERENCE_PROGRAM),
                    "correction_space": "counts",
                    "count_log_correction": float(total_log_correction[index]),
                    "log_gain_correction": float(early_log_factor[index]),
                    "temperature_gain_log_correction": float(post_log_factor[index]),
                    "acm_step_log_offset": float(
                        shared_step_coefficient("first_step", entry)
                    ),
                    "pre_reset_acm_step_log_offset": float(
                        shared_step_coefficient("pre_reset_step", entry)
                    ),
                    "additional_acm_step_log_offset": float(
                        shared_step_coefficient("additional_step", entry)
                    ),
                    "day22_acm_step_log_offset": float(
                        day22_step_coefficients[led]
                    ),
                    "day26_acm_step_log_offset": float(
                        day26_step_coefficient
                    ),
                    "late_acm_step_log_offset": float(
                        shared_step_coefficient("late_step", entry)
                    ),
                    "temperature_k": (
                        float(entry["temperature_k"][index])
                        if use_mpa_temperature
                        else np.nan
                    ),
                    "temperature_reference_k": (
                        float(
                            pre_temperature_fit["reference_temperature_k"]
                        )
                        if use_mpa_temperature
                        else np.nan
                    ),
                    "temperature_quadratic_a_log_correction": (
                        float(pre_temperature_fit["a_log_correction"])
                        if use_mpa_temperature
                        else np.nan
                    ),
                    "temperature_quadratic_b_log_correction_per_k": (
                        float(pre_temperature_fit["b_log_correction_per_k"])
                        if use_mpa_temperature
                        else np.nan
                    ),
                    "temperature_quadratic_c_log_correction_per_k2": (
                        float(pre_temperature_fit["c_log_correction_per_k2"])
                        if use_mpa_temperature
                        else np.nan
                    ),
                    "temperature_quadratic_normalization_offset_log_correction": (
                        float(
                            pre_temperature_fit[
                                "normalization_offset_log_correction"
                            ]
                        )
                        if use_mpa_temperature
                        else np.nan
                    ),
                    "temperature_quadratic_fit_r2": (
                        float(pre_temperature_fit["r2"])
                        if use_mpa_temperature
                        else np.nan
                    ),
                    "temperature_quadratic_fit_points": (
                        int(pre_temperature_fit["fit_points"])
                        if use_mpa_temperature
                        else 0
                    ),
                    "decon_step_log_correction": float(decon_log_factor[index]),
                }
            )

    tile_id = int(tile["rows"][0]["tile_id"])
    summary = {
        "tile_id": tile_id,
        "reference_program": REFERENCE_PROGRAM,
        "temperature_correction_program": TEMPERATURE_CORRECTION_PROGRAM,
        "pre_operational_temperature_period_days": pre_temperature_fit.get(
            "period_days", np.nan
        ),
        "pre_operational_temperature_rmse_counts": pre_temperature_fit.get(
            "rmse", pre_temperature_fit.get("rmse_counts", np.nan)
        ),
        "pre_operational_temperature_r2": pre_temperature_fit["r2"],
        "pre_operational_temperature_fit_points": pre_temperature_fit_points,
        "pre_operational_temperature_fit_start_repeat": pre_temperature_fit_start,
        "pre_operational_temperature_fit_end_repeat": pre_temperature_fit_end,
        "post_operational_temperature_period_days": (
            np.nan
            if post_temperature_fit is None
            else post_temperature_fit.get("period_days", np.nan)
        ),
        "post_operational_temperature_rmse_counts": (
            np.nan
            if post_temperature_fit is None
            else post_temperature_fit.get(
                "rmse", post_temperature_fit.get("rmse_counts", np.nan)
            )
        ),
        "post_operational_temperature_r2": (
            np.nan if post_temperature_fit is None else post_temperature_fit["r2"]
        ),
        "post_operational_temperature_fit_points": post_temperature_fit_points,
        "post_operational_temperature_fit_start_repeat": (
            None
            if post_temperature_fit_start_repeat is None
            else int(post_temperature_fit_start_repeat)
        ),
        "post_acm_fit_start_repeat": (
            None
            if post_acm_fit_start_repeat is None
            else int(post_acm_fit_start_repeat)
        ),
        "post_operational_temperature_fit_end_repeat": int(
            post_temperature_fit_end_repeat
        ),
        "post_acm_fit_end_repeat": (
            None if post_acm_fit_end_repeat is None else int(post_acm_fit_end_repeat)
        ),
        "reference_acm_step_counts": shared_step_coefficient("first_step"),
        "reference_pre_reset_acm_step_counts": shared_step_coefficient(
            "pre_reset_step"
        ),
        "reference_additional_acm_step_counts": shared_step_coefficient(
            "additional_step", {"led": reference_led}
        ),
        "additional_acm_step_fit_scope": (
            "per_led" if "additional_step" in per_led_acm_step_names else "shared"
        ),
        "reference_day22_acm_step_counts": day22_step_coefficients.get(
            int(reference_led), 0.0
        ),
        "day22_acm_step_fit_scope": (
            "per_led" if day22_acm_step_repeat is not None else "none"
        ),
        "reference_day26_acm_step_counts": day26_step_coefficient,
        "day26_acm_step_fit_scope": (
            "shared_grey" if day26_acm_step_repeat is not None else "none"
        ),
        "reference_late_acm_step_counts": shared_step_coefficient("late_step"),
        "temperature_model": temperature_model,
        "temperature_mnemonic": temperature_mnemonic if use_mpa_temperature else "",
        "temperature_reference_k": (
            float(pre_temperature_fit["reference_temperature_k"])
            if use_mpa_temperature
            else np.nan
        ),
        "temperature_quadratic_a_log_correction": (
            float(pre_temperature_fit["a_log_correction"])
            if use_mpa_temperature
            else np.nan
        ),
        "temperature_quadratic_b_log_correction_per_k": (
            float(pre_temperature_fit["b_log_correction_per_k"])
            if use_mpa_temperature
            else np.nan
        ),
        "temperature_quadratic_c_log_correction_per_k2": (
            float(pre_temperature_fit["c_log_correction_per_k2"])
            if use_mpa_temperature
            else np.nan
        ),
        "temperature_quadratic_normalization_offset_log_correction": (
            float(pre_temperature_fit["normalization_offset_log_correction"])
            if use_mpa_temperature
            else np.nan
        ),
        "temperature_quadratic_fit_r2": (
            float(pre_temperature_fit["r2"])
            if use_mpa_temperature
            else np.nan
        ),
        "temperature_quadratic_fit_points": (
            int(pre_temperature_fit["fit_points"])
            if use_mpa_temperature
            else 0
        ),
        "temperature_quadratic_rmse_percent": (
            float(pre_temperature_fit["rmse_percent"])
            if use_mpa_temperature
            else np.nan
        ),
        "temperature_quadratic_max_abs_percent": (
            float(pre_temperature_fit["max_abs_percent"])
            if use_mpa_temperature
            else np.nan
        ),
        "pre_operational_temperature_slope_log_per_k": (
            float(pre_temperature_fit["slope_log_per_k"])
            if use_mpa_temperature
            else np.nan
        ),
        "post_operational_temperature_slope_log_per_k": (
            np.nan
            if post_temperature_fit is None or not use_mpa_temperature
            else float(post_temperature_fit["slope_log_per_k"])
        ),
        "decon_step_time": (
            ""
            if decon_time is None
            else parse_iso_time(decon_time).isoformat() + "Z"
        ),
        "decon_step_log_correction": (
            np.nan
            if decon_fit is None
            else float(decon_fit["step_log_correction"])
        ),
        "decon_step_fit_points": (
            0 if decon_fit is None else decon_fit["fit_points"]
        ),
        "decon_step_pre_points": (
            0 if decon_fit is None else decon_fit["pre_points"]
        ),
        "decon_step_post_points": (
            0 if decon_fit is None else decon_fit["post_points"]
        ),
        "decon_step_rmse_log": (
            np.nan if decon_fit is None else decon_fit["rmse_log"]
        ),
    }
    if post_acm_fit is not None:
        additional_coefficients = post_acm_fit.get("led_step_coefficients", {}).get(
            "additional_step"
        )
    else:
        additional_coefficients = None
    for led in range(1, 7):
        if additional_coefficients is None:
            coefficient = shared_step_coefficient("additional_step")
        else:
            coefficient = additional_coefficients.get(led, 0.0)
        summary[f"additional_acm_step_led_{led}_counts"] = coefficient
    for led in range(1, 7):
        coefficient = day22_step_coefficients.get(led, 0.0)
        summary[f"day22_acm_step_led_{led}_counts"] = coefficient
    return correction_rows, summary


def build_corrections(
    path,
    reference_led,
    minimum_day,
    additional_acm_step_repeat,
    operational_temperature_day,
    pre_reset_acm_step_repeat,
    late_acm_step_repeat,
    disable_late_acm_step=False,
    post_temperature_fit_start_repeat=None,
    post_acm_fit_start_repeat=None,
    per_led_acm_step_names=(),
    post_temperature_fit_end_repeat=None,
    pre_temperature_fit_start_repeat=None,
    pre_temperature_fit_end_repeat=None,
    day22_acm_step_repeat=None,
    day26_acm_step_repeat=None,
    post_acm_fit_end_repeat=None,
    temperature_series=None,
    temperature_mnemonic=MPA_TEMPERATURE_MNEMONIC,
    temperature_quadratic_coefficients=None,
    decon_time=None,
    decon_day=None,
    decon_fit_window_days=DEFAULT_DECON_FIT_WINDOW_DAYS,
):
    detector = Path(path).name.split("_tile_log_statistics_", 1)[0].upper()
    tiles = load_count_statistics(
        path,
        additional_acm_step_repeat,
        pre_reset_acm_step_repeat,
        late_acm_step_repeat,
        disable_late_acm_step=disable_late_acm_step,
        day22_acm_step_repeat=day22_acm_step_repeat,
        day26_acm_step_repeat=day26_acm_step_repeat,
        temperature_series=temperature_series,
        decon_time=decon_time,
    )
    correction_rows = []
    summary_rows = []
    for tile_id in sorted(tiles):
        rows, summary = fit_count_corrections(
            tiles[tile_id],
            reference_led,
            minimum_day,
            additional_acm_step_repeat,
            operational_temperature_day,
            pre_reset_acm_step_repeat,
            late_acm_step_repeat,
            disable_late_acm_step=disable_late_acm_step,
            post_temperature_fit_start_repeat=post_temperature_fit_start_repeat,
            post_acm_fit_start_repeat=post_acm_fit_start_repeat,
            per_led_acm_step_names=per_led_acm_step_names,
            post_temperature_fit_end_repeat=post_temperature_fit_end_repeat,
            pre_temperature_fit_start_repeat=pre_temperature_fit_start_repeat,
            pre_temperature_fit_end_repeat=pre_temperature_fit_end_repeat,
            day22_acm_step_repeat=day22_acm_step_repeat,
            day26_acm_step_repeat=day26_acm_step_repeat,
            post_acm_fit_end_repeat=post_acm_fit_end_repeat,
            temperature_mnemonic=temperature_mnemonic,
            temperature_quadratic_coefficients=temperature_quadratic_coefficients,
            decon_time=decon_time,
            decon_day=decon_day,
            decon_fit_window_days=decon_fit_window_days,
        )
        if operational_temperature_day is not None:
            reference = tiles[tile_id]["led"][reference_led]
            if post_temperature_fit_start_repeat is None:
                expected_start = reference["days"] >= operational_temperature_day
            else:
                expected_start = reference["repeat"] >= int(
                    post_temperature_fit_start_repeat
                )
            expected_points = int(
                np.count_nonzero(
                    (reference["program"] == REFERENCE_PROGRAM)
                    & expected_start
                    & (
                        reference["repeat"]
                        < int(
                            late_acm_step_repeat
                            if post_temperature_fit_end_repeat is None
                            else post_temperature_fit_end_repeat
                        )
                    )
                )
            )
            actual_points = int(summary["post_operational_temperature_fit_points"])
            if actual_points != expected_points:
                raise ValueError(
                    f"Post-temperature fit used {actual_points} points for tile "
                    f"{tile_id}, expected {expected_points} through repeat "
                    f"{late_acm_step_repeat if post_temperature_fit_end_repeat is None else post_temperature_fit_end_repeat}"
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


def _side_swap_factor(factors, channel, led, repeat, event_repeat):
    factor = factors.get((channel, int(led)))
    if factor is None:
        raise ValueError(f"Missing side-swap {channel} factor for LED {led}")
    return float(factor) if int(repeat) >= int(event_repeat) else 1.0


def write_side_swap_scaled_statistics(
    raw_statistics_path,
    output_path,
    observations,
    side_swap_rows,
    event_repeat,
):
    """Build corrected photodiode-scaled statistics while preserving raw rows."""
    raw_statistics_path = Path(raw_statistics_path)
    output_path = Path(output_path)
    with raw_statistics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"Raw statistics are empty: {raw_statistics_path}")
    if not fields:
        raise ValueError(f"Raw statistics have no fields: {raw_statistics_path}")

    observation_by_file_set = {
        row["file_set_name"]: row for row in observations
    }
    factors = {}
    for side_swap_row in side_swap_rows:
        key = (str(side_swap_row["channel"]), int(side_swap_row["led"]))
        if key in factors:
            raise ValueError(f"Duplicate side-swap factor for {key}")
        factor = float(side_swap_row["correction_factor"])
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"Invalid side-swap factor for {key}: {factor}")
        factors[key] = factor
    if {(channel, led) for channel in ("image", "photodiode") for led in range(1, 7)} != set(
        factors
    ):
        raise ValueError("Side-swap calibration must contain image and photodiode factors for all LEDs")

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
    missing = set(count_fields + ("sum_log_change", "sum_squared_log_change", "min_log_change", "max_log_change")) - set(fields)
    if missing:
        raise ValueError(f"Raw statistics are missing side-swap fields: {sorted(missing)}")

    corrected_rows = []
    for raw_row in rows:
        current = observation_by_file_set.get(raw_row["file_set_name"])
        baseline = observation_by_file_set.get(raw_row["baseline_file_set_name"])
        if current is None or baseline is None:
            raise ValueError(
                f"Missing observation metadata for raw statistics row {raw_row['file_set_name']}"
            )
        led = int(raw_row["led"])
        if int(baseline["led"]) != led or int(current["led"]) != led:
            raise ValueError(f"Raw statistics baseline/current LED mismatch for {raw_row['file_set_name']}")
        image_baseline = _side_swap_factor(
            factors, "image", led, baseline["repeat"], event_repeat
        )
        image_current = _side_swap_factor(
            factors, "image", led, current["repeat"], event_repeat
        )
        pd_baseline = _side_swap_factor(
            factors, "photodiode", led, baseline["repeat"], event_repeat
        )
        pd_current = _side_swap_factor(
            factors, "photodiode", led, current["repeat"], event_repeat
        )
        baseline_photodiode_scale = float(baseline.get("photodiode_scale", 1.0))
        current_photodiode_scale = float(current.get("photodiode_scale", 1.0))
        if (
            not np.isfinite(baseline_photodiode_scale)
            or baseline_photodiode_scale <= 0.0
            or not np.isfinite(current_photodiode_scale)
            or current_photodiode_scale <= 0.0
        ):
            raise ValueError(f"Invalid photodiode scale for {raw_row['file_set_name']}")
        reference_factor = image_baseline * baseline_photodiode_scale
        current_factor = (
            image_current
            * current_photodiode_scale
            * pd_baseline
            / pd_current
        )
        delta = float(np.log(current_factor / reference_factor))
        updated = dict(raw_row)
        n_pair_valid = float(raw_row["n_pair_valid"])
        old_sum = float(raw_row["sum_log_change"])
        old_sum_squared = float(raw_row["sum_squared_log_change"])
        updated["sum_log_change"] = repr(old_sum + n_pair_valid * delta)
        updated["sum_squared_log_change"] = repr(
            old_sum_squared
            + 2.0 * delta * old_sum
            + n_pair_valid * delta * delta
        )
        updated["min_log_change"] = repr(float(raw_row["min_log_change"]) + delta)
        updated["max_log_change"] = repr(float(raw_row["max_log_change"]) + delta)
        for field in count_fields:
            value = float(raw_row[field])
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
            updated[field] = repr(value)
        corrected_rows.append(updated)
    write_statistics_rows(output_path, corrected_rows, fields)
    return output_path


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


def plot_count_diagnostic(
    output_path,
    statistics_path,
    corrections_path,
    detector,
    operational_temperature_day,
    physical_reset_day,
    day22_step_day=None,
    day26_step_day=None,
):
    """Plot raw and corrected current-count means for every LED."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The count diagnostic plot requires matplotlib"
        ) from error

    with Path(statistics_path).open(newline="") as handle:
        statistics_rows = list(csv.DictReader(handle))
    with Path(corrections_path).open(newline="") as handle:
        correction_rows = list(csv.DictReader(handle))
    corrections = {
        (int(row["tile_id"]), int(row["group"])): float(row["count_log_correction"])
        for row in correction_rows
    }
    rows = []
    for row in statistics_rows:
        key = (int(row["tile_id"]), int(row["group"]))
        if key not in corrections:
            raise ValueError(f"Missing count correction for diagnostic row {key}")
        n_pair_valid = float(row["n_pair_valid"])
        if n_pair_valid <= 0.0:
            continue
        raw_mean = float(row["sum_current_counts"]) / n_pair_valid
        correction = corrections[key]
        rows.append(
            {
                "led": int(row["led"]),
                "tile_id": int(row["tile_id"]),
                "group": int(row["group"]),
                "days": float(row["days"]),
                "program": str(row.get("program") or REFERENCE_PROGRAM),
                "raw_mean": raw_mean,
                "corrected_mean": raw_mean * float(np.exp(correction)),
            }
        )
    if not rows:
        raise ValueError(f"No valid count rows available for diagnostic plot: {statistics_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        sharex=True,
        constrained_layout=True,
    )
    raw_color = "0.55"
    corrected_color = "tab:blue"
    temperature_color = "tab:orange"
    for axis, led in zip(axes.flat, range(1, 7)):
        led_rows = [row for row in rows if row["led"] == led]
        tile_ids = sorted({row["tile_id"] for row in led_rows})
        for tile_id in tile_ids:
            tile_rows = sorted(
                [row for row in led_rows if row["tile_id"] == tile_id],
                key=lambda row: row["days"],
            )
            days = [row["days"] for row in tile_rows]
            axis.plot(days, [row["raw_mean"] for row in tile_rows],
                      color=raw_color, alpha=0.12, linewidth=0.45)
            axis.plot(days, [row["corrected_mean"] for row in tile_rows],
                      color=corrected_color, alpha=0.16, linewidth=0.45)

        groups = sorted({row["group"] for row in led_rows})
        median_days = []
        raw_medians = []
        corrected_medians = []
        median_programs = []
        for group in groups:
            group_rows = [row for row in led_rows if row["group"] == group]
            median_days.append(float(np.median([row["days"] for row in group_rows])))
            raw_medians.append(float(np.median([row["raw_mean"] for row in group_rows])))
            corrected_medians.append(
                float(np.median([row["corrected_mean"] for row in group_rows]))
            )
            median_programs.append(group_rows[0]["program"])
        axis.plot(
            median_days,
            raw_medians,
            color="0.2",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="raw median",
        )
        axis.plot(
            median_days,
            corrected_medians,
            color=corrected_color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="corrected median",
        )
        program_1063 = [index for index, program in enumerate(median_programs) if program == TEMPERATURE_CORRECTION_PROGRAM]
        if program_1063:
            axis.scatter(
                [median_days[index] for index in program_1063],
                [corrected_medians[index] for index in program_1063],
                color=temperature_color,
                marker="^",
                s=28,
                zorder=4,
                label="corrected Program 1063",
            )
        if operational_temperature_day is not None:
            axis.axvline(
                operational_temperature_day,
                color="tab:green",
                linestyle="--",
                linewidth=0.9,
                label="operational temperature",
            )
        if physical_reset_day is not None:
            axis.axvline(
                physical_reset_day,
                color=temperature_color,
                linestyle=":",
                linewidth=1.0,
                label="physical ice reset",
            )
        if day22_step_day is not None:
            axis.axvline(
                day22_step_day,
                color="tab:red",
                linestyle=":",
                linewidth=1.0,
                label="per-LED day-22 ACM step",
            )
        if day26_step_day is not None:
            axis.axvline(
                day26_step_day,
                color="tab:purple",
                linestyle=":",
                linewidth=1.0,
                label="shared-grey day-26 ACM step",
            )
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Days since first unified exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean current counts")
    figure.suptitle(
        f"{detector} raw versus corrected current-count means\n"
        "Photodiode-scaled statistics and final count-space correction"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_reset_aligned_counts(
    output_path,
    corrected_statistics_path,
    detector,
    physical_reset_day,
):
    """Plot pre/post-reset corrected counts on a common reset-relative time axis."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The reset-aligned count plot requires matplotlib"
        ) from error

    with Path(corrected_statistics_path).open(newline="") as handle:
        statistics_rows = list(csv.DictReader(handle))
    raw_rows = []
    for row in statistics_rows:
        n_pair_valid = float(row["n_pair_valid"])
        if n_pair_valid <= 0.0:
            continue
        raw_rows.append(
            {
                "led": int(row["led"]),
                "tile_id": int(row["tile_id"]),
                "group": int(row["group"]),
                "absolute_days": float(row["days"]),
                "corrected_mean": float(row["sum_current_counts"])
                / n_pair_valid,
            }
        )
    if not raw_rows:
        raise ValueError(
            "No valid count rows available for reset-aligned plot: "
            f"{corrected_statistics_path}"
        )
    post_reset_origin = min(
        row["absolute_days"]
        for row in raw_rows
        if row["absolute_days"] >= physical_reset_day
    )
    rows = []
    for row in raw_rows:
        post_reset = row["absolute_days"] >= physical_reset_day
        rows.append(
            {
                **row,
                "post_reset": post_reset,
                "days": (
                    row["absolute_days"] - post_reset_origin
                    if post_reset
                    else row["absolute_days"]
                ),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        sharex=True,
        constrained_layout=True,
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        led_rows = [row for row in rows if row["led"] == led]
        for tile_id in sorted({row["tile_id"] for row in led_rows}):
            tile_rows = sorted(
                [row for row in led_rows if row["tile_id"] == tile_id],
                key=lambda row: row["days"],
            )
            pre_rows = [row for row in tile_rows if not row["post_reset"]]
            post_rows = [row for row in tile_rows if row["post_reset"]]
            if pre_rows:
                axis.plot(
                    [row["days"] for row in pre_rows],
                    [row["corrected_mean"] for row in pre_rows],
                    color="tab:blue",
                    alpha=0.12,
                    linewidth=0.45,
                )
            if post_rows:
                axis.plot(
                    [row["days"] for row in post_rows],
                    [row["corrected_mean"] for row in post_rows],
                    color="tab:orange",
                    alpha=0.12,
                    linewidth=0.45,
                )

        for reset_side, color, label in (
            ("pre", "tab:blue", "pre-reset"),
            ("post", "tab:orange", "post-reset"),
        ):
            side_rows = [
                row
                for row in led_rows
                if (
                    not row["post_reset"]
                    if reset_side == "pre"
                    else row["post_reset"]
                )
            ]
            by_group = {}
            for row in side_rows:
                by_group.setdefault(row["group"], []).append(row)
            median_rows = [
                (
                    float(np.median([row["days"] for row in group_rows])),
                    float(np.median([row["corrected_mean"] for row in group_rows])),
                )
                for group_rows in by_group.values()
            ]
            median_rows.sort()
            if median_rows:
                axis.plot(
                    [item[0] for item in median_rows],
                    [item[1] for item in median_rows],
                    color=color,
                    linewidth=1.8,
                    marker="o",
                    markersize=2.5,
                    label=label,
                )
        axis.axvline(
            0.0,
            color="0.2",
            linestyle="--",
            linewidth=1.0,
            label="series start",
        )
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Days since each series start")
    for axis in axes[:, 0]:
        axis.set_ylabel("Corrected mean current counts")
    figure.suptitle(
        f"{detector} corrected counts before and after the physical reset\n"
        "Post-reset time shifted back to t = 0 for direct overlap"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def select_post_temperature_fit_rows(
    rows,
    operational_temperature_day,
    post_temperature_fit_end_repeat,
    post_temperature_fit_start_repeat=None,
):
    """Select reference-program rows for the post-operational fit window."""
    if post_temperature_fit_start_repeat is None:
        start_rows = [
            row
            for row in rows
            if row["program"] == REFERENCE_PROGRAM
            and row["days"] >= operational_temperature_day
        ]
    else:
        start_rows = [
            row
            for row in rows
            if row["program"] == REFERENCE_PROGRAM
            and row["repeat"] >= int(post_temperature_fit_start_repeat)
        ]
    if post_temperature_fit_end_repeat is None:
        return start_rows
    return [
        row
        for row in start_rows
        if row["repeat"] < int(post_temperature_fit_end_repeat)
    ]


def _load_led_temperature_plot_rows(
    statistics_path,
    corrections_path,
    tile_id,
    reference_led,
    removed_temperature_field,
):
    with Path(statistics_path).open(newline="") as handle:
        statistics_rows = list(csv.DictReader(handle))
    with Path(corrections_path).open(newline="") as handle:
        correction_rows = list(csv.DictReader(handle))
    corrections = {
        (int(row["tile_id"]), int(row["group"])): row
        for row in correction_rows
    }
    def optional_float(value):
        if value is None or str(value).strip() == "":
            return np.nan
        return float(value)

    rows = []
    for row in statistics_rows:
        if int(row["tile_id"]) != int(tile_id) or int(row["led"]) != int(reference_led):
            continue
        n_pair_valid = float(row["n_pair_valid"])
        if n_pair_valid <= 0.0:
            continue
        correction = corrections.get((int(row["tile_id"]), int(row["group"])))
        if correction is None:
            raise ValueError(
                f"Missing count correction for LED {reference_led} diagnostic row "
                f"{(tile_id, row['group'])}"
            )
        raw_mean = float(row["sum_current_counts"]) / n_pair_valid
        count_log_correction = float(correction["count_log_correction"])
        removed_temperature_log_correction = float(
            correction.get(removed_temperature_field) or 0.0
        )
        quadratic_coefficients = tuple(
            optional_float(correction.get(field))
            for field in (
                "temperature_quadratic_a_log_correction",
                "temperature_quadratic_b_log_correction_per_k",
                "temperature_quadratic_c_log_correction_per_k2",
            )
        )
        if all(np.isfinite(value) for value in quadratic_coefficients):
            temperature_log_correction = float(
                correction.get("log_gain_correction") or 0.0
            ) + float(correction.get("temperature_gain_log_correction") or 0.0)
        else:
            temperature_log_correction = removed_temperature_log_correction
        rows.append(
            {
                "days": float(row["days"]),
                "repeat": int(row["repeat"]),
                "program": str(row.get("program") or REFERENCE_PROGRAM),
                "temperature_k": optional_float(correction.get("temperature_k")),
                "temperature_reference_k": optional_float(
                    correction.get("temperature_reference_k")
                ),
                "temperature_log_correction": temperature_log_correction,
                "temperature_quadratic_a": quadratic_coefficients[0],
                "temperature_quadratic_b": quadratic_coefficients[1],
                "temperature_quadratic_c": quadratic_coefficients[2],
                "acm_corrected_mean": raw_mean
                * np.exp(count_log_correction - removed_temperature_log_correction),
            }
        )
    if not rows:
        raise ValueError(f"No valid LED {reference_led} rows found for {statistics_path}")
    return rows


def _plot_quadratic_temperature_fit(
    output_path,
    detector,
    tile_id,
    reference_led,
    fit_rows,
    regime_label,
):
    """Plot the frozen log-correction parabola without refitting a sinusoid."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The quadratic temperature-fit plot requires matplotlib"
        ) from error
    valid_rows = [
        row
        for row in fit_rows
        if all(
            np.isfinite(row[field])
            for field in (
                "temperature_k",
                "temperature_reference_k",
                "temperature_log_correction",
                "temperature_quadratic_a",
                "temperature_quadratic_b",
                "temperature_quadratic_c",
            )
        )
    ]
    if not valid_rows:
        raise ValueError("No quadratic temperature rows are available for plotting")
    reference_temperature = valid_rows[0]["temperature_reference_k"]
    coefficient = tuple(
        valid_rows[0][field]
        for field in (
            "temperature_quadratic_a",
            "temperature_quadratic_b",
            "temperature_quadratic_c",
        )
    )
    delta_temperature = np.asarray(
        [row["temperature_k"] - reference_temperature for row in valid_rows],
        dtype=float,
    )
    log_correction = np.asarray(
        [row["temperature_log_correction"] for row in valid_rows],
        dtype=float,
    )
    curve_delta_temperature = np.linspace(
        float(np.min(delta_temperature)),
        float(np.max(delta_temperature)),
        500,
    )
    curve_log_correction = (
        coefficient[0]
        + coefficient[1] * curve_delta_temperature
        + coefficient[2] * curve_delta_temperature**2
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.scatter(
        delta_temperature,
        log_correction,
        color="tab:blue",
        s=32,
        label=f"LED {reference_led} {regime_label} correction ({len(valid_rows)})",
        zorder=4,
    )
    axis.plot(
        curve_delta_temperature,
        curve_log_correction,
        color="tab:red",
        linewidth=2.0,
        label="frozen per-tile quadratic",
        zorder=3,
    )
    axis.axvline(0.0, color="0.2", linestyle="--", linewidth=1.0)
    axis.set_title(
        f"{detector} tile {tile_id} LED {reference_led} "
        f"{regime_label} WTSM00 correction"
    )
    axis.set_xlabel("WFI_MCE_ADC_WTSM00 - operational temperature (K)")
    axis.set_ylabel("Log count correction")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_led3_pre_temperature_fit(
    output_path,
    statistics_path,
    corrections_path,
    detector,
    first_acm_step_repeat,
    second_acm_step_repeat,
    tile_id=0,
    reference_led=3,
):
    """Plot the pre-operational sine and only the points between ACM steps."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The pre-temperature fit plot requires matplotlib"
        ) from error

    rows = _load_led_temperature_plot_rows(
        statistics_path,
        corrections_path,
        tile_id,
        reference_led,
        "log_gain_correction",
    )
    fit_rows = [
        row
        for row in rows
        if row["program"] == REFERENCE_PROGRAM
        and int(first_acm_step_repeat) <= row["repeat"] < int(second_acm_step_repeat)
    ]
    before_rows = [
        row
        for row in rows
        if row["program"] == REFERENCE_PROGRAM
        and row["repeat"] < int(first_acm_step_repeat)
    ]
    after_rows = [
        row
        for row in rows
        if row["program"] == REFERENCE_PROGRAM
        and row["repeat"] >= int(second_acm_step_repeat)
    ]
    temperature_rows = [
        row for row in rows if row["program"] == TEMPERATURE_CORRECTION_PROGRAM
    ]
    if not fit_rows:
        raise ValueError("No pre-operational rows between the ACM step boundaries")
    if any(
        np.isfinite(row["temperature_quadratic_a"])
        for row in fit_rows
    ):
        return _plot_quadratic_temperature_fit(
            output_path,
            detector,
            tile_id,
            reference_led,
            fit_rows,
            "pre-operational",
        )
    fit_rows.sort(key=lambda row: row["days"])
    fit = _fit_count_sinusoid(
        np.asarray([row["days"] for row in fit_rows], dtype=float),
        np.asarray([row["acm_corrected_mean"] for row in fit_rows], dtype=float),
    )
    curve_days = np.linspace(fit_rows[0]["days"], fit_rows[-1]["days"], 500)
    curve_values = _evaluate_count_sinusoid(fit, curve_days)
    first_step_day = min(row["days"] for row in fit_rows)
    second_step_day = min(
        row["days"] for row in rows if row["repeat"] >= int(second_acm_step_repeat)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for excluded_rows, label in (
        (before_rows, "1039 points before first ACM step (excluded)"),
        (after_rows, "1039 points at/after second ACM step (excluded)"),
    ):
        if excluded_rows:
            axis.scatter(
                [row["days"] for row in excluded_rows],
                [row["acm_corrected_mean"] for row in excluded_rows],
                facecolors="none",
                edgecolors="0.45",
                s=32,
                label=label,
            )
    axis.scatter(
        [row["days"] for row in fit_rows],
        [row["acm_corrected_mean"] for row in fit_rows],
        color="tab:blue",
        s=32,
        label=f"LED {reference_led} 1039 points used ({len(fit_rows)})",
        zorder=4,
    )
    axis.plot(
        curve_days,
        curve_values,
        color="tab:red",
        linewidth=2.0,
        label=f"fitted sine, period={fit['period_days']:.4g} days",
        zorder=3,
    )
    if temperature_rows:
        axis.scatter(
            [row["days"] for row in temperature_rows],
            [row["acm_corrected_mean"] for row in temperature_rows],
            color="tab:orange",
            marker="^",
            s=28,
            alpha=0.8,
            label="Program 1063 ACM-corrected points",
        )
    axis.axvline(
        first_step_day,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"first ACM step (repeat {first_acm_step_repeat})",
    )
    axis.axvline(
        second_step_day,
        color="tab:green",
        linestyle=":",
        linewidth=1.0,
        label=f"second ACM step (repeat {second_acm_step_repeat})",
    )
    axis.set_title(
        f"{detector} tile {tile_id} LED {reference_led} pre-operational temperature fit"
    )
    axis.set_xlabel("Days since first unified exposure")
    axis.set_ylabel("ACM-corrected mean current counts")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_led3_post_temperature_fit(
    output_path,
    statistics_path,
    corrections_path,
    detector,
    operational_temperature_day,
    post_temperature_fit_end_repeat,
    post_temperature_fit_start_repeat=None,
    tile_id=0,
    reference_led=3,
    post_acm_step_repeat=None,
):
    """Plot the LED 3 post-temperature correction and its selected points."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The LED 3 temperature-fit plot requires matplotlib"
        ) from error

    rows = _load_led_temperature_plot_rows(
        statistics_path,
        corrections_path,
        tile_id,
        reference_led,
        "temperature_gain_log_correction",
    )

    fit_rows = select_post_temperature_fit_rows(
        rows,
        operational_temperature_day,
        post_temperature_fit_end_repeat,
        post_temperature_fit_start_repeat,
    )
    excluded_rows = [
        row
        for row in rows
        if row["program"] == REFERENCE_PROGRAM
        and (
            row["repeat"] >= int(post_temperature_fit_start_repeat)
            if post_temperature_fit_start_repeat is not None
            else row["days"] >= operational_temperature_day
        )
        and row["repeat"] >= int(post_temperature_fit_end_repeat)
    ]
    temperature_rows = [
        row
        for row in rows
        if row["program"] == TEMPERATURE_CORRECTION_PROGRAM
        and (
            row["repeat"] < int(post_temperature_fit_start_repeat)
            if post_temperature_fit_start_repeat is not None
            else row["days"] < operational_temperature_day
        )
    ]
    if not fit_rows:
        raise ValueError("LED 3 has no post-operational pre-step points for the fit")
    if any(
        np.isfinite(row["temperature_quadratic_a"])
        for row in fit_rows
    ):
        return _plot_quadratic_temperature_fit(
            output_path,
            detector,
            tile_id,
            reference_led,
            fit_rows,
            "post-operational",
        )
    fit_rows.sort(key=lambda row: row["days"])
    fit = _fit_count_sinusoid(
        np.asarray([row["days"] for row in fit_rows], dtype=float),
        np.asarray([row["acm_corrected_mean"] for row in fit_rows], dtype=float),
    )
    curve_days = np.linspace(
        fit_rows[0]["days"], fit_rows[-1]["days"], 500
    )
    curve_values = _evaluate_count_sinusoid(fit, curve_days)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.scatter(
        [row["days"] for row in fit_rows],
        [row["acm_corrected_mean"] for row in fit_rows],
        color="tab:blue",
        s=32,
        label=f"LED {reference_led} 1039 points used ({len(fit_rows)})",
        zorder=4,
    )
    axis.plot(
        curve_days,
        curve_values,
        color="tab:red",
        linewidth=2.0,
        label=f"fitted sine, period={fit['period_days']:.4g} days",
        zorder=3,
    )
    if excluded_rows:
        axis.scatter(
            [row["days"] for row in excluded_rows],
            [row["acm_corrected_mean"] for row in excluded_rows],
            color="0.45",
            s=18,
            alpha=0.7,
            label="1039 points after later discontinuity (excluded)",
        )
    if temperature_rows:
        axis.scatter(
            [row["days"] for row in temperature_rows],
            [row["acm_corrected_mean"] for row in temperature_rows],
            color="tab:orange",
            marker="^",
            s=28,
            alpha=0.8,
            label="first 1063 batch with ACM/temp correction",
        )
    axis.axvline(
        operational_temperature_day,
        color="tab:green",
        linestyle="--",
        linewidth=1.0,
        label="temperature-fit start",
    )
    next_step_repeat = (
        int(post_acm_step_repeat)
        if post_acm_step_repeat is not None
        else int(post_temperature_fit_end_repeat)
    )
    next_step_day = min(
        (
            row["days"]
            for row in rows
            if row["repeat"] >= next_step_repeat
        ),
        default=None,
    )
    if next_step_day is not None:
        axis.axvline(
            next_step_day,
            color="0.25",
            linestyle=":",
            linewidth=1.2,
            label=(
                f"per-LED day-22 ACM step (repeat {next_step_repeat})"
                if post_acm_step_repeat is not None
                else f"temperature-fit cutoff (repeat {next_step_repeat})"
            ),
        )
    axis.set_title(
        f"{detector} tile {tile_id} LED {reference_led} post-temperature fit"
    )
    axis.set_xlabel("Days since first unified exposure")
    axis.set_ylabel("ACM-corrected mean current counts")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def find_previous_raw_statistics(output_root, run_dir, detector, tile_size):
    """Find the latest dated raw statistics file independently of scaled data."""
    filename = f"{detector.lower()}_tile_log_statistics_{tile_size}.csv"
    candidates = sorted(
        path
        for path in Path(output_root).glob(f"*/raw_statistics/{filename}")
        if path.is_file() and path.parent.parent != Path(run_dir)
    )
    return candidates[-1] if candidates else None


def find_previous_inventory(output_root, run_dir, detector, tile_size=512):
    """Find the inventory paired with the latest reusable raw statistics."""
    raw = find_previous_raw_statistics(output_root, run_dir, detector, tile_size)
    if raw is None:
        return None
    inventory = raw.parent.parent / f"{detector.lower()}_rrn_inventory.csv"
    return inventory if inventory.is_file() else None


def source_changes_require_rebuild(inventory_path, records, observations):
    """Detect image-source changes that invalidate raw statistics reuse."""
    inventory_path = Path(inventory_path) if inventory_path is not None else None
    if inventory_path is None or not inventory_path.is_file():
        return True
    with inventory_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_set_name", "source_signature"}
        if not required.issubset(reader.fieldnames or ()):
            return True
        previous = {
            row["file_set_name"]: row["source_signature"] for row in reader
        }
    records_by_file_set = {
        record["file_set_name"]: record for record in records
    }
    current = {
        observation["file_set_name"]: records_by_file_set[
            observation["file_set_name"]
        ].get("source_signature", "")
        for observation in observations
    }
    if set(previous).difference(current):
        return True
    return any(previous[name] != current[name] for name in set(previous) & set(current))


def remove_incremental_state(path):
    path = Path(path)
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
    raw_output_path=None,
):
    """Build or incrementally update raw statistics only.

    Photodiode-scaled statistics are intentionally rebuilt later from this raw
    output so changes to telemetry or correction models cannot reuse stale
    scaled values.
    """
    output_path = Path(output_path)
    if raw_output_path is None:
        raise ValueError("Raw statistics output is required for production reduction")
    raw_output_path = Path(raw_output_path)
    if no_resume:
        output_path.unlink(missing_ok=True)
        remove_incremental_state(output_path)
        raw_output_path.unlink(missing_ok=True)
        remove_incremental_state(raw_output_path)
        write_raw_detector_statistics(
            observations,
            raw_output_path,
            tile_size,
            resume=False,
            fetch_image=fetch_image,
        )
        return "rebuilt"

    previous_raw = find_previous_raw_statistics(
        output_root, run_dir, records[0]["detector"], tile_size
    )
    if not raw_output_path.exists() and previous_raw is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous_raw, raw_output_path)
        status(f"Seeding raw statistics from {previous_raw}")

    if not raw_output_path.exists():
        write_raw_detector_statistics(
            observations,
            raw_output_path,
            tile_size,
            resume=False,
            fetch_image=fetch_image,
        )
        return "rebuilt"

    try:
        update_detector_statistics_incremental(
            observations,
            raw_output_path,
            tile_size,
            fetch_image=fetch_image,
            force_unscaled=True,
        )
    except ValueError as error:
        if not str(error).startswith("Incremental update"):
            raise
        status(f"Incremental raw state is incompatible; rebuilding {raw_output_path}: {error}")
        raw_output_path.unlink(missing_ok=True)
        remove_incremental_state(raw_output_path)
        write_raw_detector_statistics(
            observations,
            raw_output_path,
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


def day_for_timestamp(observations, timestamp):
    """Return elapsed days for an absolute timestamp on the unified timeline."""
    if not observations:
        raise ValueError("Cannot locate an event on an empty observation timeline")
    start = min(parse_iso_time(row["exposure_start_time"]) for row in observations)
    return (parse_iso_time(timestamp) - start).total_seconds() / 86400.0


def first_program_observation(observations, program):
    """Return the first unified observation belonging to one program."""
    matching = [row for row in observations if str(row.get("program")) == str(program)]
    if not matching:
        raise ValueError(f"No observations found for unified program {program}")
    return min(matching, key=lambda row: parse_iso_time(row["exposure_start_time"]))


def previous_repeat_time(inventory_path, repeat, program=None):
    """Recover an event timestamp from the prior unified inventory when available."""
    if inventory_path is None or not Path(inventory_path).is_file():
        return None
    with Path(inventory_path).open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("repeat", "") != "" and int(row["repeat"]) == int(repeat)
            and (
                program is None
                or not row.get("program")
                or str(row.get("program")) == str(program)
            )
        ]
    if not rows:
        return None
    return min(rows, key=lambda row: parse_iso_time(row["exposure_start_time"]))[
        "exposure_start_time"
    ]


def effective_repeat_for_time(observations, timestamp, fallback):
    """Translate an absolute event time into the current unified repeat index."""
    if timestamp is None:
        return fallback
    event_time = parse_iso_time(timestamp)
    matching = [
        row
        for row in observations
        if parse_iso_time(row["exposure_start_time"]) >= event_time
    ]
    if not matching:
        return fallback
    return min(int(row["repeat"]) for row in matching)


def effective_post_operational_repeat(
    observations,
    legacy_repeat,
    fallback=None,
    include_all_programs=False,
):
    """Map a legacy post-operational repeat onto the current unified timeline."""
    operational_day = day_for_timestamp(
        observations, DEFAULT_OPERATIONAL_TEMPERATURE_TIME
    )
    post_repeats = sorted(
        {
            int(row["repeat"])
            for row in observations
            if (
                include_all_programs
                or str(row.get("program") or REFERENCE_PROGRAM) == REFERENCE_PROGRAM
            )
            and float(row["days"]) >= operational_day
        }
    )
    offset = int(legacy_repeat) - DEFAULT_FIRST_POST_OPERATIONAL_REPEAT
    if 0 <= offset < len(post_repeats):
        return post_repeats[offset]
    return int(legacy_repeat) if fallback is None else fallback


def temperature_fit_end_repeat(
    additional_repeat,
    late_repeat,
    requested_end_repeat=None,
):
    """Validate the post-temperature fit boundary without adding an ACM step."""
    if late_repeat is None:
        raise ValueError("The post-temperature fit requires a late ACM boundary")
    late_repeat = int(late_repeat)
    if late_repeat <= int(additional_repeat):
        raise ValueError(
            "The post-temperature fit boundary must be after the additional ACM "
            f"boundary: {late_repeat} <= {int(additional_repeat)}"
        )
    if requested_end_repeat is None:
        return late_repeat
    requested_end_repeat = int(requested_end_repeat)
    if requested_end_repeat <= int(additional_repeat):
        raise ValueError(
            "The post-temperature fit end must be after the additional ACM "
            f"boundary: {requested_end_repeat} <= {int(additional_repeat)}"
        )
    if requested_end_repeat >= late_repeat:
        raise ValueError(
            "The post-temperature fit end must be before the late ACM boundary: "
            f"{requested_end_repeat} >= {late_repeat}"
        )
    return requested_end_repeat


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
    write_manifest=True,
    tile_size=None,
    temperature_series=None,
    temperature_telemetry_path=None,
    temperature_quadratic_coefficients=None,
    temperature_quadratic_coefficients_path=None,
    decon_time=None,
    decon_fit_window_days=DEFAULT_DECON_FIT_WINDOW_DAYS,
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

    selected_tile_size = (
        DEFAULT_TILE_SIZES[0] if tile_size is None else int(tile_size)
    )
    tile_sizes = (selected_tile_size,)
    statistics_dir = run_dir / "statistics"
    raw_statistics_dir = run_dir / "raw_statistics"
    inventory_path = run_dir / f"{detector.lower()}_rrn_inventory.csv"
    previous_inventory = inventory_path if inventory_path.is_file() else None
    if previous_inventory is None:
        previous_inventory = find_previous_inventory(
            output_root,
            run_dir,
            detector,
            tile_size=selected_tile_size,
        )
    previous_raw_statistics = find_previous_raw_statistics(
        output_root,
        run_dir,
        detector,
        tile_size=selected_tile_size,
    )
    operational_temperature_day = day_for_timestamp(
        observations,
        DEFAULT_OPERATIONAL_TEMPERATURE_TIME,
    )
    decon_day = (
        None
        if decon_time is None
        else day_for_timestamp(observations, decon_time)
    )
    physical_reset = first_program_observation(
        observations,
        TEMPERATURE_CORRECTION_PROGRAM,
    )
    pre_reset_event_time = previous_repeat_time(
        previous_inventory,
        DEFAULT_PRE_RESET_ACM_STEP_REPEAT,
        REFERENCE_PROGRAM,
    )
    additional_event_time = previous_repeat_time(
        previous_inventory,
        DEFAULT_ADDITIONAL_ACM_STEP_REPEAT,
        REFERENCE_PROGRAM,
    )
    late_event_time = previous_repeat_time(
        previous_inventory,
        DEFAULT_LATE_ACM_STEP_REPEAT,
        REFERENCE_PROGRAM,
    )
    effective_pre_reset_repeat = effective_repeat_for_time(
        observations,
        pre_reset_event_time,
        DEFAULT_PRE_RESET_ACM_STEP_REPEAT,
    )
    effective_first_acm_repeat = effective_repeat_for_time(
        observations,
        DEFAULT_STEP_TIME,
        DEFAULT_FIRST_ACM_STEP_REPEAT,
    )
    if additional_event_time is None:
        effective_additional_repeat = effective_post_operational_repeat(
            observations,
            DEFAULT_ADDITIONAL_ACM_STEP_REPEAT,
        )
    else:
        effective_additional_repeat = effective_repeat_for_time(
            observations,
            additional_event_time,
            DEFAULT_ADDITIONAL_ACM_STEP_REPEAT,
        )
    if late_event_time is None:
        effective_late_repeat = effective_post_operational_repeat(
            observations,
            DEFAULT_LATE_ACM_STEP_REPEAT,
        )
    else:
        effective_late_repeat = effective_repeat_for_time(
            observations,
            late_event_time,
            DEFAULT_LATE_ACM_STEP_REPEAT,
        )
    effective_post_acm_fit_start_repeat = effective_post_operational_repeat(
        observations,
        DEFAULT_FIRST_POST_OPERATIONAL_REPEAT,
    )
    effective_post_temperature_start_repeat = effective_additional_repeat
    effective_post_temperature_end_repeat = effective_post_operational_repeat(
        observations,
        DEFAULT_TEMPERATURE_FIT_END_REPEAT,
    )
    effective_day22_step_repeat = effective_post_operational_repeat(
        observations,
        DEFAULT_DAY22_ACM_STEP_REPEAT,
    )
    effective_day26_step_repeat = effective_post_operational_repeat(
        observations,
        DEFAULT_DAY26_ACM_STEP_REPEAT,
        # The reference-program timeline ends at repeat 690; this event is
        # the first subsequent unified Program 1063 epoch.
        include_all_programs=True,
    )
    post_temperature_end_repeat = temperature_fit_end_repeat(
        effective_additional_repeat,
        effective_late_repeat,
        effective_post_temperature_end_repeat,
    )
    post_acm_fit_end_repeat = post_temperature_end_repeat
    expected_post_temperature_repeats = tuple(
        sorted(
            {
                int(row["repeat"])
                for row in observations
                if int(row["led"]) == 3
                and str(row.get("program") or REFERENCE_PROGRAM) == REFERENCE_PROGRAM
                and int(row["repeat"]) >= effective_post_temperature_start_repeat
                and int(row["repeat"]) < post_temperature_end_repeat
            }
        )
    )
    status(
        f"{detector}: post-temperature fit repeats="
        f"[{effective_post_temperature_start_repeat}, {post_temperature_end_repeat}); "
        f"ACM post-fit split at repeat {effective_post_acm_fit_start_repeat}; "
        f"day-22 per-LED step at repeat {effective_day22_step_repeat}; "
        f"day-26 shared-grey step at repeat {effective_day26_step_repeat}; "
        f"ACM fit ends at repeat {post_acm_fit_end_repeat}; late ACM coefficient disabled"
    )
    source_changed = source_changes_require_rebuild(
        previous_inventory, records, observations
    )
    status(
        f"{detector}: raw reuse candidate="
        f"{previous_raw_statistics or 'none'}; source rebuild={source_changed}"
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
        "previous_raw_statistics": (
            str(previous_raw_statistics) if previous_raw_statistics is not None else None
        ),
        "programs": list(UNIFIED_PROGRAMS),
        "operational_temperature_transition": {
            "time": DEFAULT_OPERATIONAL_TEMPERATURE_TIME,
            "day": operational_temperature_day,
        },
        "physical_ice_reset": {
            "program": TEMPERATURE_CORRECTION_PROGRAM,
            "time": physical_reset["exposure_start_time"],
            "day": physical_reset["days"],
            "repeat": physical_reset["repeat"],
            "file_set_name": physical_reset["file_set_name"],
        },
        "effective_correction_repeats": {
            "first_acm_step_repeat": effective_first_acm_repeat,
            "pre_reset_acm_step_repeat": effective_pre_reset_repeat,
            "additional_acm_step_repeat": effective_additional_repeat,
            "late_acm_step_repeat": effective_late_repeat,
            "post_acm_fit_start_repeat": effective_post_acm_fit_start_repeat,
            "post_temperature_fit_start_repeat": effective_post_temperature_start_repeat,
            "post_temperature_fit_end_repeat": post_temperature_end_repeat,
            "day22_acm_step_repeat": effective_day22_step_repeat,
            "day26_acm_step_repeat": effective_day26_step_repeat,
            "post_acm_fit_end_repeat": post_acm_fit_end_repeat,
        },
        "temperature_telemetry": {
            "enabled": temperature_series is not None,
            "mnemonic": (
                MPA_TEMPERATURE_MNEMONIC if temperature_series is not None else None
            ),
            "path": (
                str(temperature_telemetry_path)
                if temperature_telemetry_path is not None
                else None
            ),
            "quadratic_coefficients": (
                str(
                    temperature_quadratic_coefficients_path
                    or DEFAULT_TEMPERATURE_QUADRATIC_COEFFICIENTS
                )
                if temperature_series is not None
                else None
            ),
            "tile_size": selected_tile_size,
            "model": (
                "mpa_wtsm00_quadratic_per_tile"
                if temperature_series is not None
                else "legacy_time_sinusoid"
            ),
        },
        "decon_step": {
            "enabled": decon_time is not None,
            "time": str(decon_time) if decon_time is not None else None,
            "fit_window_days": decon_fit_window_days,
            "baseline_assumption": "none",
        },
    }
    for tile_size in tile_sizes:
        output_path = statistics_dir / (
            f"{detector.lower()}_tile_log_statistics_{tile_size}.csv"
        )
        raw_output_path = raw_statistics_dir / output_path.name
        mode = reduce_statistics(
            observations,
            records,
            output_path,
            tile_size,
            output_root,
            run_dir,
            source_changed,
            fetch_image=fetch_image,
            raw_output_path=raw_output_path,
        )
        detector_manifest["statistics"][str(tile_size)] = {
            "path": str(output_path),
            "space": "photodiode_scaled_side_swap_corrected_counts",
            "raw_path": str(raw_output_path),
            "raw_space": "unscaled_detector_counts",
            "mode": mode,
        }

    raw_statistics = raw_statistics_dir / (
        f"{detector.lower()}_tile_log_statistics_{selected_tile_size}.csv"
    )
    actual_post_temperature_repeats = post_temperature_repeats_by_tile(
        raw_statistics,
        effective_post_temperature_start_repeat,
        post_temperature_end_repeat,
    )
    status(
        f"{detector}: raw post-temperature repeat sets="
        f"{sorted(set(actual_post_temperature_repeats.values()))}"
    )
    if (
        not actual_post_temperature_repeats
        or any(
            repeats != expected_post_temperature_repeats
            for repeats in actual_post_temperature_repeats.values()
        )
    ):
        status(
            f"{detector}: reused raw statistics do not contain expected "
            f"post-temperature repeats {list(expected_post_temperature_repeats)}; "
            "forcing a full raw rebuild"
        )
        mode = reduce_statistics(
            observations,
            records,
            statistics_dir
            / f"{detector.lower()}_tile_log_statistics_{selected_tile_size}.csv",
            selected_tile_size,
            output_root,
            run_dir,
            True,
            fetch_image=fetch_image,
            raw_output_path=raw_statistics,
        )
        detector_manifest["statistics"][str(selected_tile_size)]["mode"] = mode

    correction_dir = run_dir / "corrections"
    corrected_dir = run_dir / "corrected_statistics"
    correction_dir.mkdir(parents=True, exist_ok=True)
    scaled_statistics = statistics_dir / (
        f"{detector.lower()}_tile_log_statistics_{selected_tile_size}.csv"
    )
    side_swap_pre_points_available, side_swap_post_points_available = (
        available_side_swap_points(
            raw_statistics,
            inventory_path,
            effective_late_repeat,
        )
    )
    side_swap_pre_points = min(
        DEFAULT_SIDE_SWAP_PRE_POINTS,
        side_swap_pre_points_available,
    )
    side_swap_post_points = min(
        DEFAULT_SIDE_SWAP_POST_POINTS,
        side_swap_post_points_available,
    )
    if side_swap_pre_points < 2 or side_swap_post_points < 2:
        raise ValueError(
            f"Side-swap calibration needs at least two points on each side; "
            f"available pre/post counts are {side_swap_pre_points_available}/"
            f"{side_swap_post_points_available} for {detector}"
        )
    side_swap_rows = fit_side_swap_series(
        raw_statistics,
        inventory_path,
        effective_late_repeat,
        pre_points=side_swap_pre_points,
        post_points=side_swap_post_points,
    )
    side_swap_calibration_path = correction_dir / (
        f"{detector.lower()}_side_swap_calibration.csv"
    )
    side_swap_output_rows = [
        {"detector": detector, **row} for row in side_swap_rows
    ]
    write_csv(side_swap_calibration_path, SIDE_SWAP_FIELDS, side_swap_output_rows)
    write_side_swap_scaled_statistics(
        raw_statistics,
        scaled_statistics,
        observations,
        side_swap_rows,
        effective_late_repeat,
    )
    detector_manifest["side_swap"] = {
        "calibration": str(side_swap_calibration_path),
        "event_repeat": effective_late_repeat,
        "pre_points": side_swap_pre_points,
        "post_points": side_swap_post_points,
        "fit_scope": "per_detector_per_led_separate_image_photodiode",
        "raw_statistics_unchanged": True,
    }
    status(f"Fitting count-space corrections for {detector}")
    correction_path = correction_dir / (
        f"{detector.lower()}_count_space_unified_temperature_corrections.csv"
    )
    correction_summary_path = correction_dir / (
        f"{detector.lower()}_count_space_unified_temperature_fit_summary.csv"
    )
    day22_step_day = min(
        float(row["days"])
        for row in observations
        if int(row["repeat"]) >= effective_day22_step_repeat
    ) if any(
        int(row["repeat"]) >= effective_day22_step_repeat for row in observations
    ) else None
    day26_step_day = min(
        float(row["days"])
        for row in observations
        if int(row["repeat"]) >= effective_day26_step_repeat
    ) if any(
        int(row["repeat"]) >= effective_day26_step_repeat for row in observations
    ) else None
    correction_rows, summary_rows = build_corrections(
        scaled_statistics,
        3,
        1.0,
        effective_additional_repeat,
        operational_temperature_day,
        effective_pre_reset_repeat,
        effective_late_repeat,
        post_temperature_fit_start_repeat=effective_post_temperature_start_repeat,
        post_acm_fit_start_repeat=effective_post_acm_fit_start_repeat,
        per_led_acm_step_names=("additional_step",),
        post_temperature_fit_end_repeat=post_temperature_end_repeat,
        pre_temperature_fit_start_repeat=effective_first_acm_repeat,
        pre_temperature_fit_end_repeat=effective_pre_reset_repeat,
        day22_acm_step_repeat=effective_day22_step_repeat,
        day26_acm_step_repeat=effective_day26_step_repeat,
        post_acm_fit_end_repeat=post_acm_fit_end_repeat,
        disable_late_acm_step=True,
        temperature_series=temperature_series,
        temperature_mnemonic=MPA_TEMPERATURE_MNEMONIC,
        temperature_quadratic_coefficients=(
            temperature_quadratic_coefficients.get(detector, {})
            .get(selected_tile_size, {})
            if temperature_quadratic_coefficients is not None
            else None
        ),
        decon_time=decon_time,
        decon_day=decon_day,
        decon_fit_window_days=decon_fit_window_days,
    )
    status(
        f"{detector}: post-temperature fit points="
        f"{sorted({int(row['post_operational_temperature_fit_points']) for row in summary_rows})} "
        f"for repeats [{effective_post_temperature_start_repeat}, "
        f"{post_temperature_end_repeat})"
    )
    write_csv(correction_path, CORRECTION_FIELDS, correction_rows)
    write_csv(correction_summary_path, SUMMARY_FIELDS, summary_rows)
    corrected_path = corrected_dir / (
        f"{detector.lower()}_tile_log_statistics_{selected_tile_size}_count_corrected.csv"
    )
    write_corrected_statistics(
        scaled_statistics,
        correction_path,
        corrected_path,
    )
    diagnostic_plot = run_dir / "diagnostics" / (
        f"{detector.lower()}_counts_before_after_correction.png"
    )
    with _PLOT_LOCK:
        plot_count_diagnostic(
            diagnostic_plot,
            scaled_statistics,
            correction_path,
            detector,
            operational_temperature_day,
            physical_reset["days"],
            day22_step_day=day22_step_day,
            day26_step_day=day26_step_day,
        )
    reset_aligned_plot = run_dir / "diagnostics" / (
        f"{detector.lower()}_reset_aligned_counts.png"
    )
    led3_temperature_fit_plot = run_dir / "diagnostics" / (
        f"{detector.lower()}_led3_post_temperature_fit.png"
    )
    led3_pre_temperature_fit_plot = run_dir / "diagnostics" / (
        f"{detector.lower()}_led3_pre_temperature_fit.png"
    )
    with _PLOT_LOCK:
        plot_reset_aligned_counts(
            reset_aligned_plot,
            corrected_path,
            detector,
            physical_reset["days"],
        )
        plot_led3_post_temperature_fit(
            led3_temperature_fit_plot,
            scaled_statistics,
            correction_path,
            detector,
            operational_temperature_day,
            post_temperature_end_repeat,
            effective_post_temperature_start_repeat,
            tile_id=0,
            reference_led=3,
            post_acm_step_repeat=effective_day22_step_repeat,
        )
        plot_led3_pre_temperature_fit(
            led3_pre_temperature_fit_plot,
            scaled_statistics,
            correction_path,
            detector,
            effective_first_acm_repeat,
            effective_pre_reset_repeat,
            tile_id=0,
            reference_led=3,
        )
    status(f"Wrote corrected statistics for {detector}: {corrected_path}")
    detector_manifest["corrections"] = {
        "table": str(correction_path),
        "fit_summary": str(correction_summary_path),
        "count_corrected_statistics": str(corrected_path),
        "count_diagnostic_plot": str(diagnostic_plot),
        "reset_aligned_count_plot": str(reset_aligned_plot),
        "led3_post_temperature_fit_plot": str(led3_temperature_fit_plot),
        "led3_pre_temperature_fit_plot": str(led3_pre_temperature_fit_plot),
        "correction_space": "counts",
        "temperature_mnemonic": (
            MPA_TEMPERATURE_MNEMONIC if temperature_series is not None else ""
        ),
        "temperature_model": (
            "mpa_wtsm00_quadratic_per_tile"
            if temperature_series is not None
            else "legacy_time_sinusoid"
        ),
        "temperature_quadratic_coefficients": (
            str(
                temperature_quadratic_coefficients_path
                or DEFAULT_TEMPERATURE_QUADRATIC_COEFFICIENTS
            )
            if temperature_series is not None
            else ""
        ),
        "temperature_quadratic_tile_size": selected_tile_size,
        "decon_step_time": str(decon_time) if decon_time is not None else "",
        "decon_step_baseline_assumption": "none",
        "input_statistics": str(scaled_statistics),
        "raw_statistics": str(raw_statistics),
        "additional_acm_step_fit_scope": "per_led",
        "day22_acm_step_fit_scope": "per_led",
        "day26_acm_step_fit_scope": "shared_grey",
        "late_acm_step_replaced_by_side_swap": True,
    }
    manifest["detectors"][detector] = detector_manifest
    if write_manifest:
        write_json(run_dir / "pipeline_manifest.json", manifest)
    return detector_manifest


def make_image_fetcher(missions):
    def fetch_image(observation):
        if observation["source"] == "s3":
            return stream_s3_image(observation["s3_uri"])
        return stream_mast_image(missions, observation["cal_uri"])

    return fetch_image


def run_detector_worker(
    source_records,
    run_dir,
    output_root,
    detector,
    index_metadata,
    index_rows,
    telemetry,
    mast_token,
    tile_size,
    temperature_series=None,
    temperature_telemetry_path=None,
    temperature_quadratic_coefficients=None,
    temperature_quadratic_coefficients_path=None,
    decon_time=None,
    decon_fit_window_days=DEFAULT_DECON_FIT_WINDOW_DAYS,
):
    """Run one detector with an isolated MAST session for threaded execution."""
    missions = login_to_mast(mast_token)
    return run_detector(
        source_records,
        run_dir,
        output_root,
        detector,
        index_metadata,
        index_rows,
        telemetry,
        make_image_fetcher(missions),
        {"detectors": {}},
        write_manifest=False,
        tile_size=tile_size,
        temperature_series=temperature_series,
        temperature_telemetry_path=temperature_telemetry_path,
        temperature_quadratic_coefficients=temperature_quadratic_coefficients,
        temperature_quadratic_coefficients_path=(
            temperature_quadratic_coefficients_path
        ),
        decon_time=decon_time,
        decon_fit_window_days=decon_fit_window_days,
    )


def positive_integer(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reduce unified CAR190 Program 1039 and Program 1063 products."
    )
    parser.add_argument(
        "--detector",
        choices=DEFAULT_DETECTORS,
        help="Reduce one detector instead of all WFI detectors.",
    )
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=1,
        help="Run up to this many detectors concurrently; default: 1.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=(
            "Dated reduction root to reuse and extend; default: "
            f"{DEFAULT_RUN_ROOT}."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=positive_integer,
        default=DEFAULT_TILE_SIZES[0],
        help=(
            "Tile edge length in detector pixels; default: "
            f"{DEFAULT_TILE_SIZES[0]}. A 4096-pixel detector produces "
            "64 tiles at 512 or 256 tiles at 256."
        ),
    )
    parser.add_argument(
        "--mpa-temperature-telemetry",
        type=Path,
        default=None,
        help=(
            "Long-form CSV for WFI_MCE_ADC_WTSM00. When omitted, fetch the "
            "required sporadic MAST engineering telemetry automatically. "
            "The telemetry is evaluated through the frozen per-tile quadratic "
            "correction table."
        ),
    )
    parser.add_argument(
        "--temperature-quadratic-coefficients",
        type=Path,
        default=DEFAULT_TEMPERATURE_QUADRATIC_COEFFICIENTS,
        help=(
            "Frozen detector/tile WTSM00 log-correction parabolas; default: "
            f"{DEFAULT_TEMPERATURE_QUADRATIC_COEFFICIENTS}."
        ),
    )
    parser.add_argument(
        "--decon-time",
        default=None,
        help=(
            "Optional UTC timestamp for the expected decon. Fit its local "
            "count-space step as a free offset without assuming a post-decon "
            "baseline; omit until the event time is known."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selected_detectors = (
        [args.detector] if args.detector is not None else list(DEFAULT_DETECTORS)
    )
    source = DEFAULT_S3_SOURCE
    output_root = Path(getattr(args, "run_root", DEFAULT_RUN_ROOT))
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
        "detectors_requested": selected_detectors,
        "mast_token": str(DEFAULT_MAST_TOKEN),
        "mast_index": str(DEFAULT_MAST_INDEX),
        "mast_supplement": True,
        "programs": list(UNIFIED_PROGRAMS),
        "photodiode_channel": DEFAULT_PHOTODIODE_CHANNEL,
        "tile_sizes": [args.tile_size],
        "correction_settings": {
            "reference_led": 3,
            "acm_fit_scope": (
                "all_leds_shared_grey_except_additional_and_day22_per_led"
            ),
            "additional_acm_step_fit_scope": "per_led",
            "temperature_fit_led": 3,
            "minimum_day": 1.0,
            "additional_acm_step_repeat": DEFAULT_ADDITIONAL_ACM_STEP_REPEAT,
            "pre_reset_acm_step_repeat": DEFAULT_PRE_RESET_ACM_STEP_REPEAT,
            "late_acm_step_repeat": DEFAULT_LATE_ACM_STEP_REPEAT,
            "late_acm_step_replaced_by_side_swap": True,
            "side_swap_fit_scope": "per_detector_per_led_separate_image_photodiode",
            "side_swap_pre_points": DEFAULT_SIDE_SWAP_PRE_POINTS,
            "side_swap_post_points": DEFAULT_SIDE_SWAP_POST_POINTS,
            "operational_temperature_time": DEFAULT_OPERATIONAL_TEMPERATURE_TIME,
            "temperature_model": "mpa_wtsm00_quadratic_per_tile",
            "temperature_quadratic_coefficients": str(
                getattr(
                    args,
                    "temperature_quadratic_coefficients",
                    DEFAULT_TEMPERATURE_QUADRATIC_COEFFICIENTS,
                )
            ),
            "temperature_quadratic_tile_size": args.tile_size,
            "post_temperature_fit_starts_after_additional_acm": True,
            "temperature_fit_end_legacy_repeat": DEFAULT_TEMPERATURE_FIT_END_REPEAT,
            "day22_acm_step_legacy_repeat": DEFAULT_DAY22_ACM_STEP_REPEAT,
            "day22_acm_step_fit_scope": "per_led",
            "day26_acm_step_legacy_repeat": DEFAULT_DAY26_ACM_STEP_REPEAT,
            "day26_acm_step_fit_scope": "shared_grey",
            "physical_ice_reset": "first_program_1063_observation",
            "decon_step": "optional_free_count_space_step",
        },
        "detectors": {},
    }
    write_json(run_dir / "pipeline_manifest.json", manifest)

    s3_records = list_s3_products(source)
    refresh_index(
        DEFAULT_MAST_INDEX,
        token_path=DEFAULT_MAST_TOKEN,
        detectors=selected_detectors,
        programs=UNIFIED_PROGRAMS,
    )
    status("MAST index refresh complete")
    index_rows = read_index_rows(
        DEFAULT_MAST_INDEX,
        require_program=True,
        programs=UNIFIED_PROGRAMS,
    )
    index_rows = [
        row
        for row in index_rows
        if str(row["detector"]).strip() in selected_detectors
    ]
    if not index_rows:
        raise ValueError(
            "The refreshed MAST index contains no products for the selected detector(s)"
        )
    for detector in selected_detectors:
        detector_programs = {
            str(row["program"]).strip()
            for row in index_rows
            if str(row["detector"]).strip() == detector
        }
        missing_programs = sorted(set(UNIFIED_PROGRAMS) - detector_programs)
        if missing_programs:
            raise ValueError(
                f"Selected detector {detector} is missing unified program(s): "
                f"{missing_programs}"
            )
    temperature_series, temperature_telemetry_path = prepare_temperature_telemetry(
        args,
        index_rows,
        run_dir,
    )
    temperature_quadratic_coefficients_path = getattr(
        args,
        "temperature_quadratic_coefficients",
        DEFAULT_TEMPERATURE_QUADRATIC_COEFFICIENTS,
    )
    temperature_quadratic_coefficients = (
        load_temperature_quadratic_coefficients(
            temperature_quadratic_coefficients_path
        )
        if temperature_series is not None
        else {}
    )
    if temperature_series is not None:
        missing_detectors = sorted(
            detector
            for detector in selected_detectors
            if not temperature_quadratic_coefficients.get(detector, {}).get(
                args.tile_size
            )
        )
        if missing_detectors:
            raise ValueError(
                "Temperature quadratic coefficients are missing detector/tile-size "
                f"entries for {missing_detectors} at tile size {args.tile_size}"
            )
    decon_time = getattr(args, "decon_time", None)
    index_metadata = read_optional_index(DEFAULT_MAST_INDEX)
    shutil.copy2(DEFAULT_MAST_INDEX, run_dir / DEFAULT_MAST_INDEX.name)

    telemetry_summaries = []
    for program in UNIFIED_PROGRAMS:
        program_rows = [row for row in index_rows if str(row["program"]).strip() == program]
        telemetry_path = (
            run_dir / "telemetry" / f"photodiode_{DEFAULT_PHOTODIODE_CHANNEL}_"
            f"ice_monitor_program_{program}.csv"
        )
        telemetry_summaries.append(
            run_ice_monitor_telemetry(
                program_rows,
                DEFAULT_MAST_TOKEN,
                DEFAULT_PHOTODIODE_CHANNEL,
                program,
                telemetry_path,
            )
        )
    telemetry = merge_telemetry_summaries(telemetry_summaries)
    status(
        f"Received {len(telemetry['values'])} dark-subtracted photodiode rows; "
        "using first non-warm-up reference per LED"
    )
    manifest["telemetry"] = {
        "source": telemetry["source"],
        "channel": telemetry["channel"],
        "mnemonic": telemetry["mnemonic"],
        "path": telemetry["summary_path"],
        "programs": list(UNIFIED_PROGRAMS),
        "dark_subtracted": True,
        "exposure_rows": len(telemetry["values"]),
        "reference_mode": telemetry["reference_mode"],
        "reference_values": telemetry["reference_values"],
    }
    manifest["temperature_telemetry"] = {
        "enabled": temperature_series is not None,
        "mnemonic": (
            MPA_TEMPERATURE_MNEMONIC if temperature_series is not None else None
        ),
        "path": (
            str(temperature_telemetry_path)
            if temperature_telemetry_path is not None
            else None
        ),
        "quadratic_coefficients": (
            str(temperature_quadratic_coefficients_path)
            if temperature_series is not None
            else None
        ),
        "tile_size": args.tile_size,
        "source": "mast-edp-sporadic" if temperature_series is not None else None,
        "bucket_seconds": MPA_TEMPERATURE_BUCKET_SECONDS,
        "model": (
            "mpa_wtsm00_quadratic_per_tile"
            if temperature_series is not None
            else "legacy_time_sinusoid"
        ),
    }

    s3_detectors = {
        record["detector"]
        for record in s3_records
        if record["detector"] in selected_detectors
    }
    mast_detectors = {row["detector"].strip() for row in index_rows}
    detectors = sorted(s3_detectors | mast_detectors)
    if not detectors:
        raise ValueError("No S3 or MAST calibrated products were found")
    manifest["detectors_requested"] = selected_detectors
    manifest["detectors_available"] = detectors
    workers = min(args.workers, len(detectors))
    manifest["detector_workers"] = {
        "requested": args.workers,
        "used": workers,
        "implementation": "thread_pool_with_isolated_mast_sessions",
    }
    write_json(run_dir / "pipeline_manifest.json", manifest)
    status(
        f"Reducing {', '.join(detectors)} from streamed S3/MAST products "
        f"into {run_dir}",
    )
    if workers == 1:
        missions = login_to_mast(DEFAULT_MAST_TOKEN)
        fetch_image = make_image_fetcher(missions)
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
                tile_size=args.tile_size,
                temperature_series=temperature_series,
                temperature_telemetry_path=temperature_telemetry_path,
                temperature_quadratic_coefficients=(
                    temperature_quadratic_coefficients
                ),
                temperature_quadratic_coefficients_path=(
                    temperature_quadratic_coefficients_path
                ),
                decon_time=decon_time,
            )
    else:
        status(f"Running detectors concurrently with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                detector: executor.submit(
                    run_detector_worker,
                    s3_records,
                    run_dir,
                    output_root,
                    detector,
                    index_metadata,
                    index_rows,
                    telemetry,
                    DEFAULT_MAST_TOKEN,
                    args.tile_size,
                    temperature_series,
                    temperature_telemetry_path,
                    temperature_quadratic_coefficients,
                    temperature_quadratic_coefficients_path,
                    decon_time,
                )
                for detector in detectors
            }
            for detector in detectors:
                detector_manifest = futures[detector].result()
                manifest["detectors"][detector] = detector_manifest
                write_json(run_dir / "pipeline_manifest.json", manifest)
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "pipeline_manifest.json", manifest)
    status(f"Wrote pipeline manifest {run_dir / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    main()
