"""Fully reduce local RRN products plus missing MAST products into tile statistics.

The RRN copy contains files but no archive manifest.  This script derives the
detector, LED, and observation-epoch grouping from the standard file name.  An
optional archive index supplies exact exposure timestamps and, with
``--mast-supplement``, defines the MAST products to add when they are absent
from the RRN directory.

This is a pixel-level reduction, not a merge of precomputed statistics.  Local
RRN ASDFs are read from disk; missing MAST ASDFs are downloaded into a BytesIO
buffer, copied into NumPy arrays, and released without ever being written to
disk.  The first product in the combined chronological sequence for each LED
is used as that LED's divisor.

Only one LED baseline and one current image are held in memory at a time.
Per-observation checkpoint files make an interrupted local-only run resumable;
MAST-supplemented runs require ``--no-resume`` for a true full rerun.

Passing ``--photodiode-channel pd1 --photodiode-summary PATH`` loads the
per-exposure CSV produced by ``roman-telem --ice-monitor``.  Its signal is the
in-exposure LED median minus the same-pass dark median; that signal is
normalized independently per LED to the tool-provided reference pass (the
first non-warm-up pass in each reset segment) and applied to each current image
before statistics are accumulated.
"""

import argparse
import ast
import bisect
import csv
import io
import os
from datetime import date, datetime, timedelta, timezone
import json
from numbers import Real
from pathlib import Path
import re
import time
from typing import Mapping
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import warnings

import numpy as np

try:
    from requests.exceptions import RequestException
except ModuleNotFoundError:  # Keep local-only RRN reductions dependency-light.
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
DEFAULT_TILE_SIZES = (512, 64)
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_MAST_TOKEN = Path(__file__).resolve().parent / "mast_api_token.txt"
RCS_PROGRAM = "1039"
PHOTODIODE_DATA_URI = (
    "https://mast.stsci.edu/edp/api/v0.1/mnemonics/spa/roman/data"
)
PHOTODIODE_MNEMONICS = {
    "pd1": "WFI_MCE_SRCS_PD1_V",
    "pd2": "WFI_MCE_SRCS_PD2_V",
}
PHOTODIODE_QUERY_PADDING = timedelta(minutes=1)


def parse_product_name(path):
    """Return filename-derived metadata, or None for unrelated ASDF files."""
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


def scan_products(input_dir, detector=None, allow_empty=False):
    """Find local calibrated products and validate their derived identities."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    records = []
    seen = {}
    for path in sorted(input_dir.rglob("*_cal.asdf")):
        record = parse_product_name(path)
        if record is None:
            continue
        if detector is not None and record["detector"] != detector:
            continue
        previous = seen.get(record["file_set_name"])
        if previous is not None:
            raise ValueError(
                f"Duplicate file-set name for {record['file_set_name']}: "
                f"{previous} and {path}"
            )
        stat = path.stat()
        record["source_signature"] = f"{stat.st_size}:{stat.st_mtime_ns}"
        seen[record["file_set_name"]] = path
        records.append(record)
    if not records and not allow_empty:
        suffix = f" for {detector}" if detector else ""
        raise ValueError(f"No matching calibrated ASDF files found{suffix} in {input_dir}")
    return records


def read_index_rows(path, require_program=False):
    """Read and validate rows used for timestamps and MAST supplementation."""
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Index file does not exist: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Index contains no rows: {path}")
    required = {"fileSetName", "detector", "led", "visit_id", "exposure_start_time"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Index is missing columns {sorted(missing)}: {path}")
    if require_program:
        if "program" not in rows[0]:
            raise ValueError(f"MAST supplement index is missing column 'program': {path}")
        programs = sorted({str(row["program"]).strip() for row in rows})
        if programs != [RCS_PROGRAM]:
            raise ValueError(
                f"MAST supplement index must contain only RCS program {RCS_PROGRAM}; "
                f"found programs: {programs}"
            )
    return rows


def read_optional_index(path):
    """Read optional reference metadata keyed by file-set and product name."""
    metadata = {}
    for row in read_index_rows(path):
        file_set_name = row["fileSetName"].strip()
        if not file_set_name:
            raise ValueError(f"Index contains an empty fileSetName")
        file_name = Path(row.get("cal_uri", "")).name
        if not file_name:
            file_name = f"{file_set_name}_cal.asdf"
        for key in (file_set_name, file_name):
            if key in metadata and metadata[key]["fileSetName"] != file_set_name:
                raise ValueError(f"Index contains duplicate product key {key}")
            metadata[key] = row
    return metadata


def mast_records_from_index(rows, detector=None, allow_empty=False):
    """Create remote product records for the MAST side of a combined run."""
    records = []
    seen = {}
    for row in rows:
        row_detector = row["detector"].strip()
        if detector is not None and row_detector != detector:
            continue
        file_set_name = row["fileSetName"].strip()
        if not file_set_name:
            raise ValueError("MAST supplement index contains an empty fileSetName")
        file_name = Path(row.get("cal_uri", "")).name
        if not file_name:
            file_name = f"{file_set_name}_cal.asdf"
        visit_id = str(row["visit_id"]).strip()
        if len(visit_id) <= 4:
            raise ValueError(f"Invalid visit_id for MAST product {file_set_name}: {visit_id!r}")
        led = int(row["led"])
        if led not in range(1, 7):
            raise ValueError(f"Invalid LED {led} for MAST product {file_set_name}")
        record = {
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
        previous = seen.get(file_set_name)
        if previous is not None:
            raise ValueError(f"Duplicate MAST product {file_set_name} in supplement index")
        seen[file_set_name] = record
        records.append(record)
    if detector is not None and not records and not allow_empty:
        raise ValueError(f"No MAST supplement products found for {detector}")
    return records


def merge_product_records(local_records, mast_records):
    """Union local and MAST products, preferring a local RRN copy."""
    merged = []
    by_file_set = {}
    for record in local_records:
        key = record["file_set_name"]
        if key in by_file_set:
            raise ValueError(f"Duplicate local product {key}")
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
                raise ValueError(f"Local/MAST metadata mismatch for {key}")
            continue
        by_file_set[key] = record
        merged.append(record)
    return merged


def _get_path(value, path):
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _named_values(value, names):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in names:
                yield child
            yield from _named_values(child, names)


def _format_time(value):
    """Convert common ASDF time representations to an ISO UTC string."""
    if value is None:
        return None
    if hasattr(value, "isot"):
        value = value.isot
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
    if isinstance(value, date):
        return value.isoformat() + "T00:00:00.000000Z"
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        try:
            from astropy.time import Time
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Numeric ASDF timestamps require astropy; install it or provide an index"
            ) from error
        if 30000.0 < numeric < 100000.0:
            return Time(numeric, format="mjd", scale="utc").isot + "Z"
        if 1.0e9 < numeric < 4.0e9:
            return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat(
                timespec="microseconds"
            ).replace("+00:00", "Z")
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _header_timestamp(tree):
    """Find an exposure start timestamp in a Roman ASDF tree."""
    meta = _get_path(tree, ("meta",))
    if meta is None:
        return None
    for path in (
        ("exposure", "start_time"),
        ("exposure", "date_beg"),
        ("exposure", "date_obs"),
        ("observation", "date_beg"),
        ("observation", "date_obs"),
    ):
        timestamp = _format_time(_get_path(meta, path))
        if timestamp is not None:
            return timestamp

    observation = _get_path(meta, ("observation",))
    date_value = _get_path(observation, ("date",))
    time_value = _get_path(observation, ("time",))
    if date_value is not None and time_value is not None:
        timestamp = _format_time(f"{date_value}T{time_value}")
        if timestamp is not None:
            return timestamp

    for value in _named_values(meta, {"start_time", "date_beg", "date_obs"}):
        timestamp = _format_time(value)
        if timestamp is not None:
            return timestamp
    return None


def read_asdf_timestamp(path):
    try:
        import asdf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Reading local Roman products requires the 'asdf' package"
        ) from error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with asdf.open(
            path,
            lazy_load=True,
            ignore_unrecognized_tag=True,
        ) as handle:
            tree = handle.tree["roman"]
            timestamp = _header_timestamp(tree)
    if timestamp is None:
        raise ValueError(f"No exposure start timestamp found in {path}")
    return timestamp


def load_local_asdf(path):
    try:
        import asdf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Reading local Roman products requires the 'asdf' package"
        ) from error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with asdf.open(
            path,
            lazy_load=True,
            ignore_unrecognized_tag=True,
        ) as handle:
            model = handle.tree["roman"]
            data = np.array(model["data"], copy=True)
            dq = np.array(model["dq"], copy=True)
    if data.shape != dq.shape:
        raise ValueError(f"Data/DQ shape mismatch in {path}")
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D detector image in {path}, found {data.shape}")
    return data, dq


def load_asdf_buffer(buffer, source_name):
    """Copy Roman image and DQ arrays from an in-memory ASDF payload."""
    try:
        import asdf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MAST supplementation requires the 'asdf' package"
        ) from error
    buffer.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with asdf.open(
            buffer,
            lazy_load=True,
            ignore_unrecognized_tag=True,
        ) as handle:
            model = handle.tree["roman"]
            data = np.array(model["data"], copy=True)
            dq = np.array(model["dq"], copy=True)
    if data.shape != dq.shape:
        raise ValueError(f"Data/DQ shape mismatch in {source_name}")
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D detector image in {source_name}, found {data.shape}")
    return data, dq


def login_to_mast(token_path):
    """Authenticate once; image downloads remain in memory only."""
    try:
        from astroquery.mast import MastMissions
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MAST supplementation requires the 'astroquery' package"
        ) from error
    token_path = Path(token_path)
    if not token_path.exists():
        raise ValueError(f"MAST token file does not exist: {token_path}")
    token = token_path.read_text().strip()
    if not token:
        raise ValueError(f"MAST token file is empty: {token_path}")
    missions = MastMissions(mission="roman")
    missions.login(token=token)
    return missions


def stream_mast_image(
    missions,
    uri,
    chunk_size=DEFAULT_CHUNK_SIZE,
    retries=3,
    request_timeout=600,
):
    """Download one MAST ASDF product to memory and return data plus DQ."""
    if chunk_size < 1 or retries < 0 or request_timeout <= 0:
        raise ValueError("Invalid MAST download settings")
    service = missions._service_api_connection
    missions_url = getattr(service, "MISSIONS_DOWNLOAD_URL", None)
    if missions_url is not None:
        base_url = missions_url.rstrip("/") + "/roman/api/v0.1/retrieve_product"
    else:
        # Some RRN astroquery builds expose only MAST_DOWNLOAD_URL.  Derive
        # the Roman service endpoint without changing the in-memory download.
        download_url = getattr(service, "MAST_DOWNLOAD_URL", None)
        if download_url is None:
            raise RuntimeError(
                "MAST service exposes neither MISSIONS_DOWNLOAD_URL nor "
                "MAST_DOWNLOAD_URL"
            )
        service_root = str(download_url).split("/api/v0.1/", 1)[0].rstrip("/")
        base_url = service_root + "/search/roman/api/v0.1/retrieve_product"
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
    raise RuntimeError(f"MAST download failed for {uri}")


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


def _query_photodiode_mnemonic(token, mnemonic, start, end):
    query = urlencode(
        {
            "mnemonic": mnemonic,
            "s_time": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "e_time": end.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    request = Request(
        f"{PHOTODIODE_DATA_URI}?{query}",
        headers={"Authorization": f"token {token}", "Accept": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        raw = response.read().decode()
    payload = json.loads(raw) if raw.lstrip().startswith(("{", "[")) else ast.literal_eval(raw)
    data = payload.get("Data", []) if isinstance(payload, Mapping) else []
    values = []
    for item in data:
        try:
            value = float(item.get("EUValue"))
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append((parse_iso_time(item["ObsTime"]), value))
    if not values:
        raise ValueError(f"No numeric telemetry returned for {mnemonic}")
    return sorted(values)


def _photodiode_values_in_window(values, start, end):
    times = [item[0] for item in values]
    lower = bisect.bisect_left(times, start)
    upper = bisect.bisect_right(times, end)
    selected = values[lower:upper]
    if not selected:
        raise ValueError(
            f"No photodiode samples overlap {start.isoformat()} to {end.isoformat()}"
        )
    return np.asarray([value for _, value in selected], dtype=float)


def fetch_photodiode_telemetry(index_rows, token_path, channel):
    """Fetch one flight photodiode channel for all indexed exposure windows."""
    if channel not in PHOTODIODE_MNEMONICS:
        raise ValueError(
            f"Unsupported photodiode channel {channel!r}; "
            f"choose from {sorted(PHOTODIODE_MNEMONICS)}"
        )
    windows = []
    for row in index_rows:
        start_value = str(row.get("exposure_start_time", "")).strip()
        end_value = str(row.get("exposure_end_time", "")).strip()
        if not start_value or not end_value:
            raise ValueError(
                "Photodiode correction requires exposure_start_time and "
                "exposure_end_time in the archive index"
            )
        start = parse_iso_time(start_value)
        end = parse_iso_time(end_value)
        if end < start:
            raise ValueError(f"Exposure ends before it starts: {start_value} to {end_value}")
        windows.append((start, end))
    if not windows:
        raise ValueError("Photodiode correction has no archive exposure windows")
    query_start = min(start for start, _ in windows) - PHOTODIODE_QUERY_PADDING
    query_end = max(end for _, end in windows) + PHOTODIODE_QUERY_PADDING
    token = _read_mast_token(token_path)
    mnemonic = PHOTODIODE_MNEMONICS[channel]
    values = _query_photodiode_mnemonic(token, mnemonic, query_start, query_end)
    return {
        "source": "raw-edb-mnemonic",
        "channel": channel,
        "mnemonic": mnemonic,
        "query_start": query_start,
        "query_end": query_end,
        "values": values,
    }


def load_ice_monitor_photodiode_summary(path, channel):
    """Load dark-subtracted per-exposure output from ``roman-telem``."""
    if channel not in PHOTODIODE_MNEMONICS:
        raise ValueError(
            f"Unsupported photodiode channel {channel!r}; "
            f"choose from {sorted(PHOTODIODE_MNEMONICS)}"
        )
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
        missing = required.difference(reader.fieldnames or ())
        led_field = "led" if "led" in (reader.fieldnames or ()) else "led_band"
        if led_field not in (reader.fieldnames or ()):
            missing.add("led or led_band")
        if missing:
            raise ValueError(
                f"Ice-monitor photodiode summary is missing fields: {sorted(missing)}"
            )
        for row in reader:
            if row["mnemonic"] != mnemonic or not row["step"].startswith("band"):
                continue
            if str(row["dark_subtracted"]).strip().lower() not in {"true", "1"}:
                raise ValueError(
                    f"Ice-monitor summary row is not dark-subtracted: {row}"
                )
            start = parse_iso_time(row["start"])
            end = parse_iso_time(row["end"])
            led = int(float(row[led_field]))
            signal = float(row["signal"])
            dark = float(row["dark_median"])
            reference = float(row["ref_signal"])
            change_pct = float(row["change_pct"])
            if not np.isfinite(signal) or signal <= 0.0:
                raise ValueError(
                    f"Dark-subtracted photodiode signal must be positive: {row}"
                )
            if not np.isfinite(reference) or reference <= 0.0:
                raise ValueError(
                    f"Ice-monitor reference signal must be positive: {row}"
                )
            expected_change_pct = 100.0 * (signal / reference - 1.0)
            if not np.isclose(change_pct, expected_change_pct, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"Ice-monitor change_pct does not match signal/ref_signal: {row}"
                )
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


def attach_photodiode_corrections(observations, telemetry):
    """Attach per-LED current-image scales derived from exposure-window means."""
    if telemetry.get("source") == "roman-telem-ice-monitor":
        return _attach_ice_monitor_corrections(observations, telemetry)
    channel = telemetry["channel"]
    values = telemetry["values"]
    by_led = {}
    for observation in observations:
        start_value = observation.get("exposure_start_time")
        end_value = observation.get("exposure_end_time")
        if not start_value or not end_value:
            raise ValueError(
                "Photodiode correction requires exposure start and end times for "
                f"{observation['file_set_name']}"
            )
        start = parse_iso_time(start_value)
        end = parse_iso_time(end_value)
        samples = _photodiode_values_in_window(values, start, end)
        mean = float(np.mean(samples))
        if not np.isfinite(mean) or mean <= 0.0:
            raise ValueError(
                f"Photodiode mean must be positive for {observation['file_set_name']}: {mean}"
            )
        by_led.setdefault(int(observation["led"]), []).append(
            (start, observation, mean, int(samples.size))
        )

    references = {
        led: min(rows, key=lambda item: (item[0], item[1]["group"]))[2]
        for led, rows in by_led.items()
    }
    for led, rows in by_led.items():
        reference = references[led]
        for _, observation, mean, sample_count in rows:
            scale = reference / mean
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Invalid photodiode scale for {observation['file_set_name']}: {scale}"
                )
            observation.update(
                {
                    "photodiode_channel": channel,
                    "photodiode_source": telemetry.get("source", "raw-edb-mnemonic"),
                    "photodiode_mean": mean,
                    "photodiode_reference": reference,
                    "photodiode_scale": float(scale),
                    "photodiode_n_samples": sample_count,
                }
            )
    return observations


def _attach_ice_monitor_corrections(observations, telemetry):
    values = telemetry["values"]
    for observation in observations:
        start = parse_iso_time(observation["exposure_start_time"])
        end = parse_iso_time(observation["exposure_end_time"])
        key = (start, end, int(observation["led"]))
        try:
            value = values[key]
        except KeyError as error:
            raise ValueError(
                f"No dark-subtracted ice-monitor row for {observation['file_set_name']}: "
                f"{key}"
            ) from error
        scale = value["reference_signal"] / value["signal"]
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"Invalid ice-monitor photodiode scale for "
                f"{observation['file_set_name']}: {scale}"
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


def resolve_record_times(records, index_metadata, allow_ordinal_days=False):
    """Use ASDF header times, with optional index values for known products."""
    missing = []
    for record in records:
        row = index_metadata.get(record["file_set_name"]) or index_metadata.get(
            record["file_name"]
        )
        if row is not None:
            if row["detector"].strip() != record["detector"]:
                raise ValueError(f"Index detector mismatch for {record['file_name']}")
            if int(row["led"]) != record["led"]:
                raise ValueError(f"Index LED mismatch for {record['file_name']}")
            timestamp = row["exposure_start_time"].strip()
            if timestamp:
                record["exposure_start_time"] = timestamp
                record["exposure_end_time"] = row.get("exposure_end_time", "").strip()
                record["timestamp_source"] = "optional_index"
                continue
        record["exposure_start_time"] = None
        record["exposure_end_time"] = None
        record["timestamp_source"] = None
        missing.append(record)

    for record in missing:
        if record.get("path") is None:
            raise ValueError(
                f"Missing MAST timestamp for {record['file_name']}; "
                "the supplement index must provide exposure_start_time"
            )
        try:
            record["exposure_start_time"] = read_asdf_timestamp(record["path"])
            record["timestamp_source"] = "asdf_header"
        except ValueError:
            if not allow_ordinal_days:
                raise
            record["exposure_start_time"] = None
            record["timestamp_source"] = "ordinal"

    if not records and not allow_ordinal_days:
        raise ValueError("No products available for timestamp resolution")
    return records


def write_inventory(path, records, observations):
    """Write the authoritative inventory discovered from the RRN directory."""
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
                    "timestamp_source": record["timestamp_source"],
                    "photodiode_channel": observation.get("photodiode_channel", ""),
                    "photodiode_source": observation.get("photodiode_source", ""),
                    "photodiode_mean": observation.get("photodiode_mean", ""),
                    "photodiode_reference": observation.get(
                        "photodiode_reference", ""
                    ),
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
    """Yield log-ratio and compact count-space statistics for every tile."""
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
                reference_squared = float(
                    np.sum(reference_values * reference_values, dtype=np.float64)
                )
                current_total = float(np.sum(current_values, dtype=np.float64))
                current_squared = float(
                    np.sum(current_values * current_values, dtype=np.float64)
                )
                reference_current = float(
                    np.sum(reference_values * current_values, dtype=np.float64)
                )
                reference_minimum = float(np.min(reference_values))
                reference_maximum = float(np.max(reference_values))
                current_minimum = float(np.min(current_values))
                current_maximum = float(np.max(current_values))
                minimum = float(np.min(values))
                maximum = float(np.max(values))
                total = float(np.sum(values, dtype=np.float64))
                total_squared = float(np.sum(values * values, dtype=np.float64))
            else:
                reference_total = 0.0
                reference_squared = 0.0
                current_total = 0.0
                current_squared = 0.0
                reference_current = 0.0
                reference_minimum = np.nan
                reference_maximum = np.nan
                current_minimum = np.nan
                current_maximum = np.nan
                minimum = np.nan
                maximum = np.nan
                total = 0.0
                total_squared = 0.0
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


def build_observations(records, allow_ordinal_days=False):
    """Order products into shared observation epochs and assign elapsed days."""
    epoch_groups = {}
    for record in records:
        epoch_groups.setdefault(record["epoch_key"], []).append(record)
    for epoch_key, group in epoch_groups.items():
        led_values = [record["led"] for record in group]
        if len(led_values) != len(set(led_values)):
            raise ValueError(f"Duplicate LED product in observation epoch {epoch_key}")

    if all(record.get("exposure_start_time") for record in records):
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
    elif allow_ordinal_days:
        epoch_order = sorted(epoch_groups)
        global_start = datetime(2000, 1, 1)
        for index, key in enumerate(epoch_order):
            for record in epoch_groups[key]:
                record["sort_time"] = global_start + timedelta(days=index)
                record["exposure_start_time"] = record["sort_time"].isoformat() + "Z"
        print("WARNING: using ordinal epoch days; do not use this output for physical rates")
    else:
        missing = [record["file_name"] for record in records if not record.get("exposure_start_time")]
        raise ValueError(
            "Missing exposure timestamps; provide an index or use --allow-ordinal-days: "
            + ", ".join(missing[:3])
        )

    led_counts = {led: 0 for led in range(1, 7)}
    observations = []
    for repeat, epoch_key in enumerate(epoch_order):
        group = sorted(epoch_groups[epoch_key], key=lambda record: (record["sort_time"], record["led"]))
        for record in group:
            led = record["led"]
            if led not in led_counts:
                raise ValueError(f"Invalid LED {led} in {record['file_name']}")
            led_counts[led] += 1
            observations.append(
                {
                    "led": led,
                    "repeat": repeat,
                    "group": len(observations),
                    "file_set_name": record["file_set_name"],
                    "baseline_file_set_name": "",
                    "exposure_start_time": record["exposure_start_time"],
                    "exposure_end_time": record.get("exposure_end_time"),
                    "days": (record["sort_time"] - global_start).total_seconds() / 86400.0,
                    "cal_uri": record.get("cal_uri")
                    or (str(record["path"]) if record.get("path") else record["file_name"]),
                    "source": record.get("source", "rrn"),
                    "path": record.get("path"),
                }
            )
    for led, count in led_counts.items():
        if count < 3:
            raise ValueError(f"LED {led} has too few observation epochs: {count}")
    baselines = {}
    for row in observations:
        baselines.setdefault(row["led"], row["file_set_name"])
    for row in observations:
        row["baseline_file_set_name"] = baselines[row["led"]]
    return observations


def detector_part_path(parts_dir, group):
    return parts_dir / f"group_{int(group):04d}.csv"


def load_observation_image(observation, fetch_image=None):
    """Load a local RRN image or stream a MAST image without persisting it."""
    if observation.get("source", "rrn") == "mast":
        if fetch_image is None:
            raise RuntimeError(
                f"No MAST image fetcher configured for {observation['file_set_name']}"
            )
        return fetch_image(observation["cal_uri"])
    path = observation.get("path")
    if path is None:
        path = observation["cal_uri"]
    return load_local_asdf(path)


def write_detector_statistics(
    observations,
    output_path,
    tile_size,
    resume=True,
    fetch_image=None,
):
    """Write one detector output with resumable per-observation parts."""
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
        print(f"Skipping completed output: {output_path}")
        return

    parts_dir = output_path.with_name(output_path.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    for led in range(1, 7):
        led_rows = [row for row in observations if row["led"] == led]
        pending_rows = [
            row
            for row in led_rows
            if not (
                resume and detector_part_path(parts_dir, row["group"]).exists()
            )
        ]
        if not pending_rows:
            continue
        baseline_data, baseline_dq = load_observation_image(led_rows[0], fetch_image)
        try:
            for row in pending_rows:
                current_data = current_dq = None
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
                    temporary_part.replace(part_path)
                finally:
                    if temporary_part.exists():
                        temporary_part.unlink()
                    if (
                        row["file_set_name"] != row["baseline_file_set_name"]
                        and current_data is not None
                    ):
                        del current_data, current_dq
                print(f"{row['file_set_name']}: wrote {last_tile['tile_id'] + 1} tiles")
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
    """Return a safe checkpoint path keyed by the source file-set name."""
    if not file_set_name or Path(file_set_name).name != file_set_name:
        raise ValueError(f"Unsafe file-set name: {file_set_name!r}")
    return parts_dir / f"{file_set_name}.csv"


def read_statistics_rows(path, return_fieldnames=False):
    """Read a complete statistics CSV for file-set keyed incremental updates."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = set(FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(
                f"Statistics file is missing fields {sorted(missing)}: {path}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Statistics file has no rows: {path}")
    if return_fieldnames:
        return rows, fieldnames
    return rows


def write_statistics_rows(output_path, rows, fieldnames=FIELDS):
    """Atomically write an incrementally rebuilt statistics CSV."""
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
    """Refresh chronological metadata while preserving computed tile moments."""
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


def update_detector_statistics_incremental(
    observations, output_path, tile_size, fetch_image=None
):
    """Reuse existing file-set statistics and reduce only newly discovered files.

    Existing observations must retain their per-LED baseline file. If a new
    earlier baseline appears, the caller should rebuild the detector output
    because the old ratios were formed against a different image.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        raise ValueError(
            f"Incremental update requires an existing statistics file: {output_path}"
        )
    existing_rows, existing_fieldnames = read_statistics_rows(
        output_path, return_fieldnames=True
    )
    existing_by_file_set = {}
    existing_baselines = {}
    for row in existing_rows:
        file_set_name = row["file_set_name"]
        tile_rows = existing_by_file_set.setdefault(file_set_name, {})
        tile_id = row["tile_id"]
        if tile_id in tile_rows:
            raise ValueError(
                f"Duplicate tile {tile_id} for {file_set_name} in {output_path}"
            )
        tile_rows[tile_id] = row
        led = int(row["led"])
        baseline = row["baseline_file_set_name"]
        previous = existing_baselines.setdefault(led, baseline)
        if previous != baseline:
            raise ValueError(
                f"Existing statistics have inconsistent LED {led} baselines: "
                f"{previous!r} and {baseline!r}"
            )

    current_by_file_set = {row["file_set_name"]: row for row in observations}
    if len(current_by_file_set) != len(observations):
        raise ValueError("The source contains duplicate file-set names")
    removed = set(existing_by_file_set).difference(current_by_file_set)
    if removed:
        raise ValueError(
            "Incremental update refuses to discard existing observations: "
            + ", ".join(sorted(removed)[:3])
        )
    current_baselines = {}
    for observation in observations:
        current_baselines.setdefault(
            int(observation["led"]), observation["baseline_file_set_name"]
        )
    for led, baseline in existing_baselines.items():
        if current_baselines.get(led) != baseline:
            raise ValueError(
                f"Incremental update cannot change LED {led} baseline from "
                f"{baseline!r} to {current_baselines.get(led)!r}"
            )

    new_observations = [
        row
        for row in observations
        if row["file_set_name"] not in existing_by_file_set
    ]
    parts_dir = output_path.with_name(output_path.name + ".incremental.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    new_rows_by_file_set = {}
    for observation in new_observations:
        part_path = incremental_part_path(parts_dir, observation["file_set_name"])
        if part_path.exists():
            new_rows_by_file_set[observation["file_set_name"]] = read_statistics_rows(
                part_path
            )

    for led in range(1, 7):
        pending_rows = [
            row
            for row in new_observations
            if int(row["led"]) == led
            and row["file_set_name"] not in new_rows_by_file_set
        ]
        if not pending_rows:
            continue
        led_rows = [row for row in observations if int(row["led"]) == led]
        baseline_data, baseline_dq = load_observation_image(led_rows[0], fetch_image)
        try:
            for observation in pending_rows:
                part_path = incremental_part_path(
                    parts_dir, observation["file_set_name"]
                )
                current_data = current_dq = None
                if (
                    observation["file_set_name"]
                    == observation["baseline_file_set_name"]
                ):
                    current_data, current_dq = baseline_data, baseline_dq
                else:
                    current_data, current_dq = load_observation_image(
                        observation, fetch_image
                    )
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
                            current_scale=float(
                                observation.get("photodiode_scale", 1.0)
                            ),
                        ):
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
                                    "exposure_start_time": observation[
                                        "exposure_start_time"
                                    ],
                                    "days": observation["days"],
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
                    if temporary_part.exists():
                        temporary_part.unlink()
                    if current_data is not baseline_data and current_data is not None:
                        del current_data, current_dq
                new_rows_by_file_set[observation["file_set_name"]] = (
                    read_statistics_rows(part_path)
                )
                print(
                    f"{observation['file_set_name']}: wrote "
                    f"{last_tile['tile_id'] + 1} tiles"
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
                raise RuntimeError(f"Missing incremental statistics for {file_set_name}")
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


def choose_detectors(records, requested, all_detectors):
    available = sorted({record["detector"] for record in records})
    if all_detectors:
        return available
    if requested not in available:
        raise ValueError(f"Detector {requested} is not present; available: {available}")
    return [requested]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help=(
            "Optional archive CSV used for timestamps; with --mast-supplement it "
            "also defines products to fetch from MAST."
        ),
    )
    parser.add_argument(
        "--mast-supplement",
        action="store_true",
        help=(
            "Add products listed by --index-path that are absent locally, downloading "
            "each MAST ASDF only into memory. Requires --no-resume."
        ),
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_MAST_TOKEN,
        help=f"MAST API token file (default: {DEFAULT_MAST_TOKEN}).",
    )
    parser.add_argument(
        "--download-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Streaming MAST download chunk size in bytes.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="Number of retries for a failed MAST product download.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=600,
        help="Per-request MAST download timeout in seconds.",
    )
    parser.add_argument(
        "--inventory-path",
        type=Path,
        default=None,
        help="Path for the authoritative inventory generated from the RRN folder.",
    )
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--all-detectors", action="store_true")
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs="+",
        default=list(DEFAULT_TILE_SIZES),
        help="One or more tile sizes; default: 512 64.",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--allow-ordinal-days",
        action="store_true",
        help="Use one synthetic day per epoch if no timestamp can be found; not suitable for physical rates.",
    )
    parser.add_argument(
        "--photodiode-channel",
        choices=sorted(PHOTODIODE_MNEMONICS),
        default=None,
        help=(
            "Select the flight source channel in --photodiode-summary and normalize "
            "each current image per LED; PD1 is recommended. Requires --index-path "
            "and --no-resume."
        ),
    )
    parser.add_argument(
        "--photodiode-summary",
        type=Path,
        default=None,
        help=(
            "Per-exposure dark-subtracted CSV produced by "
            "roman-telem --ice-monitor."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(size < 1 for size in args.tile_size):
        raise ValueError("Tile sizes must be positive")
    if args.all_detectors and args.inventory_path is not None:
        raise ValueError("--inventory-path requires a single selected detector")
    if args.mast_supplement:
        if args.index_path is None:
            raise ValueError("--mast-supplement requires --index-path")
        if args.resume:
            raise ValueError(
                "--mast-supplement requires --no-resume for a full pixel-level rerun"
            )
        if args.download_chunk_size < 1 or args.download_retries < 0:
            raise ValueError("Invalid MAST download settings")
        if args.download_timeout <= 0:
            raise ValueError("--download-timeout must be positive")
    if args.photodiode_channel is not None:
        if args.index_path is None:
            raise ValueError("--photodiode-channel requires --index-path")
        if args.photodiode_summary is None:
            raise ValueError(
                "--photodiode-channel requires --photodiode-summary produced by "
                "roman-telem --ice-monitor"
            )
        if args.resume:
            raise ValueError(
                "--photodiode-channel requires --no-resume so uncorrected "
                "checkpoint parts cannot be reused"
            )
    elif args.photodiode_summary is not None:
        raise ValueError("--photodiode-summary requires --photodiode-channel")
    tile_sizes = list(dict.fromkeys(args.tile_size))
    scan_detector = None if args.all_detectors else args.detector
    local_records = scan_products(
        args.input_dir,
        scan_detector,
        allow_empty=args.mast_supplement,
    )
    index_rows = read_index_rows(args.index_path, require_program=args.mast_supplement)
    index_metadata = read_optional_index(args.index_path)
    mast_records = (
        mast_records_from_index(index_rows, scan_detector) if args.mast_supplement else []
    )
    records = merge_product_records(local_records, mast_records)
    selected_detectors = choose_detectors(records, args.detector, args.all_detectors)
    photodiode_telemetry = None
    if args.photodiode_channel is not None and not args.dry_run:
        photodiode_telemetry = load_ice_monitor_photodiode_summary(
            args.photodiode_summary,
            args.photodiode_channel,
        )
        print(
            f"Loaded {photodiode_telemetry['mnemonic']} dark-subtracted telemetry "
            f"from {photodiode_telemetry['summary_path']} "
            f"({len(photodiode_telemetry['values'])} exposures)"
        )
    fetch_image = None
    if args.mast_supplement and not args.dry_run:
        missions = login_to_mast(args.token_file)
        fetch_image = lambda uri: stream_mast_image(
            missions,
            uri,
            chunk_size=args.download_chunk_size,
            retries=args.download_retries,
            request_timeout=args.download_timeout,
        )
    print(
        f"Found {len(local_records)} local RRN products and "
        f"{len(mast_records)} MAST-indexed products; combined inventory has "
        f"{len(records)} products for {', '.join(selected_detectors)}"
    )

    for detector in selected_detectors:
        detector_records = [record for record in records if record["detector"] == detector]
        resolve_record_times(detector_records, index_metadata, args.allow_ordinal_days)
        observations = build_observations(detector_records, args.allow_ordinal_days)
        if photodiode_telemetry is not None:
            attach_photodiode_corrections(observations, photodiode_telemetry)
        inventory_path = args.inventory_path or args.output_dir / (
            f"{detector.lower()}_rrn_inventory.csv"
        )
        write_inventory(inventory_path, detector_records, observations)
        epoch_leds = {}
        for row in observations:
            epoch_leds.setdefault(row["repeat"], set()).add(row["led"])
        partial_epochs = {
            epoch: sorted(leds)
            for epoch, leds in epoch_leds.items()
            if len(leds) != 6
        }
        print(
            f"{detector}: {len(observations)} products across "
            f"{len({row['repeat'] for row in observations})} observation epochs"
        )
        if partial_epochs:
            print(f"WARNING: partial observation epochs: {partial_epochs}")
        print(f"Wrote authoritative inventory {inventory_path}")
        if args.dry_run:
            continue
        for tile_size in tile_sizes:
            output_path = args.output_dir / (
                f"{detector.lower()}_tile_log_statistics_{tile_size}.csv"
            )
            print(f"Writing {output_path}")
            write_detector_statistics(
                observations,
                output_path,
                tile_size,
                args.resume,
                fetch_image=fetch_image,
            )


if __name__ == "__main__":
    main()
