"""Export a compact comparison bundle from Eric's RCS Trend v4 products.

The v4 ``cal/*.h5`` and matching ``h5/*_raw.h5`` files contain one epoch of
16x-binned arrays for all 18 SCAs and six LEDs.  This script reads those files
where they are available on RRN and writes only the metadata, calibration
factors, and spatial trend summaries needed for comparison with the local
MAST/tile reduction.  The HDF5 files are not copied.

The default export selects the full available time series for WFI01 and WFI10
and local tiles 136 and 137.  Use ``--tile-ids`` for a different one- or
two-tile comparison.  Full-SCA summaries are opt-in with
``--include-sca-summary``.

Example on RRN (the default source is this S3 prefix)::

    python export_eric_rcs_trend_comparison.py \
        --output /path/to/eric_rcs_trend_comparison

The S3 mode requires the AWS CLI, in addition to ``numpy`` and ``h5py``.  A
local ``--source-root`` can still be supplied explicitly.  The script is
standalone and does not require ``rcs_trend_reader.py``.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

try:
    import h5py
except ModuleNotFoundError:  # Allow --help without the optional dependency.
    h5py = None

import numpy as np


CALIBRATION_STEPS = (
    "photodiode",
    "acm",
    "acm_bridge",
    "side_bridge",
    "tempgain",
)
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/smdc-rom-com-ops/rrn-ops/activities/"
    "car-190-rcs-spectro-trending/rcs_trend_v4"
)
DEFAULT_S3_URI = (
    "s3://smdc-rom-com-ops/rrn-ops/activities/"
    "car-190-rcs-spectro-trending/rcs_trend_v4/"
)
DEFAULT_DETECTORS = ("WFI01", "WFI10")
DEFAULT_TILE_IDS = (136, 137)

COMMON_FIELDS = (
    "source_file",
    "source_relpath",
    "timestamp",
    "run_label",
    "car_label",
    "pass_number",
    "decon_cycle",
    "mce_side",
    "acm_segment",
    "calibration_schema",
    "bin_factor",
)

FACTOR_FIELDS = tuple(
    field
    for step in CALIBRATION_STEPS
    for field in (f"{step}_multiplier",)
) + (
    "total_multiplier_no_tempgain",
    "total_multiplier",
) + tuple(
    field
    for step in CALIBRATION_STEPS
    for field in (f"log_{step}_ratio_to_first",)
) + (
    "log_total_no_tempgain_ratio_to_first",
    "log_total_calibrated_ratio_to_first",
)

RATE_FIELDS = (
    "raw_mean_dn_s",
    "raw_median_dn_s",
    "raw_valid_fraction",
    "raw_log_change_mean",
    "raw_log_change_median",
    "raw_log_pair_valid_fraction",
    "no_tempgain_mean_dn_s",
    "no_tempgain_median_dn_s",
    "no_tempgain_log_change_mean",
    "no_tempgain_log_change_median",
    "calibrated_mean_dn_s",
    "calibrated_median_dn_s",
    "calibrated_log_change_mean",
    "calibrated_log_change_median",
)

PRODUCT_FIELDS = (
    "source_file",
    "source_relpath",
    "source_size_bytes",
    "source_sha256",
    "timestamp",
    "run_label",
    "car_label",
    "pass_number",
    "decon_cycle",
    "mce_side",
    "acm_segment",
    "calibration_schema",
    "bin_factor",
    "n_sca",
    "n_superpixels",
)

CORRECTION_FIELDS = COMMON_FIELDS + ("sca", "detector", "led") + FACTOR_FIELDS
SCA_TREND_FIELDS = COMMON_FIELDS + ("sca", "detector", "led") + RATE_FIELDS
TILE_TREND_FIELDS = (
    COMMON_FIELDS
    + (
        "sca",
        "detector",
        "led",
        "tile_id",
        "tile_row",
        "tile_col",
        "local_row_start",
        "local_row_stop_exclusive",
        "local_col_start",
        "local_col_stop_exclusive",
        "superpixel_row_start",
        "superpixel_row_stop_exclusive",
        "superpixel_col_start",
        "superpixel_col_stop_exclusive",
    )
    + RATE_FIELDS
)


def _require_h5py():
    if h5py is None:
        raise RuntimeError(
            "This export requires h5py. Install it in the RRN environment "
            "with 'python -m pip install h5py' or load the site's HDF5 module."
        )


def _decode(value):
    """Convert an HDF5 attribute to ordinary Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _decode(value.item())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode(value.item())
        return [_decode(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_decode(item) for item in value]
    return value


def _text(value):
    value = _decode(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(_text(item) for item in value)
    return str(value)


def _attribute(attrs, name):
    """Look up an attribute with a case-insensitive fallback."""
    if name in attrs:
        return attrs[name]
    wanted = name.lower()
    for key in attrs:
        if str(key).lower() == wanted:
            return attrs[key]
    return None


def _numeric_attribute(attrs, name, default=None):
    value = _attribute(attrs, name)
    if value is None:
        return default
    values = np.asarray(_decode(value)).reshape(-1)
    if values.size != 1:
        raise ValueError(f"HDF5 attribute {name!r} is not scalar: {value!r}")
    return float(values[0])


def _metadata(handle, path, source_root):
    attrs = handle.attrs
    timestamp = _text(_attribute(attrs, "timestamp"))
    if not timestamp:
        raise ValueError(f"{path} has no timestamp HDF5 attribute")
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        relative = path.name
    return {
        "source_file": path.name,
        "source_relpath": str(relative),
        "source_path": str(path),
        "source_size_bytes": path.stat().st_size,
        "timestamp": timestamp,
        "run_label": _text(_attribute(attrs, "run_label")),
        "car_label": _text(_attribute(attrs, "car_label")),
        "pass_number": _text(_attribute(attrs, "pass_number")),
        "decon_cycle": _text(_attribute(attrs, "decon_cycle")),
        "mce_side": _text(_attribute(attrs, "mce_side")),
        "acm_segment": _text(_attribute(attrs, "acm_segment")),
        "calibration_schema": _text(_attribute(attrs, "calibration_schema")),
        "bin_factor": _text(_attribute(attrs, "bin_factor")),
    }


def _raw_product_path(source_root, calibrated_path):
    """Return the matching local h5 raw product when one is present."""
    source_root = Path(source_root)
    calibrated_path = Path(calibrated_path)
    name = calibrated_path.name
    raw_name = None
    for calibrated_suffix, raw_suffix in (
        ("_cal.h5", "_raw.h5"),
        ("_cal.hdf5", "_raw.hdf5"),
    ):
        if name.endswith(calibrated_suffix):
            raw_name = name[: -len(calibrated_suffix)] + raw_suffix
            break
    if raw_name is None:
        return None

    candidates = [calibrated_path.with_name(raw_name)]
    try:
        relative = calibrated_path.relative_to(source_root)
    except ValueError:
        relative = None
    if relative is not None and relative.parts and relative.parts[0] == "cal":
        candidates.insert(0, source_root / "h5" / raw_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _timestamp_sort_key(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return (1, value)
    return (0, parsed.timestamp())


def _find_dataset(handle, led, group, required=True):
    """Find an LED dataset across the v4 reader's accepted layouts."""
    names = {
        f"led{led}",
        f"led_{led}",
        str(led),
    }
    candidates = (
        f"{group}/LED{led}",
        f"{group}/led{led}",
        f"{group}/LED_{led}",
        f"{group}/led_{led}",
        f"{group}/raw_LED{led}",
        f"{group}/raw_led{led}",
        f"sca_arrays/LED{led}",
        f"sca_arrays/led{led}",
        f"sca_arrays/LED_{led}",
        f"sca_arrays/led_{led}",
        f"LED{led}",
        f"led{led}",
    )
    for candidate in candidates:
        if candidate in handle and isinstance(handle[candidate], h5py.Dataset):
            return handle[candidate]

    found = []

    def visitor(name, item):
        if not isinstance(item, h5py.Dataset):
            return
        if not (name == group or name.startswith(f"{group}/")):
            return
        leaf = name.rsplit("/", 1)[-1].lower().replace("-", "_")
        if leaf in names or leaf.removeprefix("raw_") in names:
            found.append(item)

    handle.visititems(visitor)
    if len(found) == 1:
        return found[0]
    if not found:
        if not required:
            return None
        raise KeyError(f"Could not find {group} dataset for LED {led}")
    raise ValueError(f"Found multiple {group} datasets for LED {led}")


def _sca_stack(dataset, source_name):
    values = np.asarray(dataset, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            f"{source_name} must be a 3-D SCA stack; got shape {values.shape}"
        )
    if values.shape[0] == 18:
        return values
    if values.shape[-1] == 18:
        return np.moveaxis(values, -1, 0)
    raise ValueError(
        f"{source_name} has no 18-SCA axis; got shape {values.shape}"
    )


def _dataset_layout(dataset, source_name):
    """Return (n_superpixels, axis_order) without loading the dataset."""
    shape = tuple(int(value) for value in dataset.shape)
    if len(shape) != 3:
        raise ValueError(
            f"{source_name} must be a 3-D SCA stack; got shape {shape}"
        )
    if shape[0] == 18:
        n_rows, n_cols = shape[1:]
        axis_order = "sca_first"
    elif shape[-1] == 18:
        n_rows, n_cols = shape[:2]
        axis_order = "sca_last"
    else:
        raise ValueError(
            f"{source_name} has no 18-SCA axis; got shape {shape}"
        )
    if n_rows != n_cols:
        raise ValueError(
            f"{source_name} must contain square superpixel arrays; got {shape}"
        )
    return n_rows, axis_order


def _dataset_block(
    dataset,
    sca,
    row_start,
    row_stop,
    col_start,
    col_stop,
    source_name,
):
    """Read one SCA/tile block without materializing the full SCA stack."""
    _n_superpixels, axis_order = _dataset_layout(dataset, source_name)
    if axis_order == "sca_first":
        values = dataset[sca, row_start:row_stop, col_start:col_stop]
    else:
        values = dataset[row_start:row_stop, col_start:col_stop, sca]
    return np.asarray(values, dtype=float)


def _stack(handle, led, group, required=True):
    dataset = _find_dataset(handle, led, group, required=required)
    if dataset is None:
        return None
    return _sca_stack(dataset, dataset.name)


def _raw_stack(handle, led):
    return _stack(handle, led, "raw")


def _calibrated_stack(handle, led):
    return _stack(handle, led, "cal", required=False)


def _product_datasets(handle, led):
    """Return raw/cal datasets; v4 exports may contain only the cal dataset."""
    raw_dataset = _find_dataset(handle, led, "raw", required=False)
    calibrated_dataset = _find_dataset(handle, led, "cal", required=False)
    if raw_dataset is None and calibrated_dataset is None:
        raise KeyError(f"Could not find raw or calibrated dataset for LED {led}")
    return raw_dataset, calibrated_dataset


def _paired_product_datasets(raw_handle, calibrated_handle, led):
    """Find raw and calibrated LED datasets across paired HDF5 files."""
    if raw_handle is calibrated_handle:
        return _product_datasets(calibrated_handle, led)

    raw_dataset = _find_dataset(raw_handle, led, "raw", required=False)
    if raw_dataset is None:
        raw_dataset = _find_dataset(raw_handle, led, "cal", required=False)
    calibrated_dataset = _find_dataset(
        calibrated_handle,
        led,
        "cal",
        required=False,
    )
    if calibrated_dataset is None:
        calibrated_dataset = _find_dataset(
            calibrated_handle,
            led,
            "raw",
            required=False,
        )
    if raw_dataset is None and calibrated_dataset is None:
        raise KeyError(f"Could not find raw or calibrated dataset for LED {led}")
    return raw_dataset, calibrated_dataset


@contextmanager
def _open_product_pair(calibrated_path, raw_path=None):
    """Open a calibrated product and its optional separate raw product."""
    calibrated_path = Path(calibrated_path)
    raw_path = Path(raw_path) if raw_path else None
    with h5py.File(calibrated_path, "r") as calibrated_handle:
        if raw_path is not None and raw_path.resolve() != calibrated_path.resolve():
            with h5py.File(raw_path, "r") as raw_handle:
                yield raw_handle, calibrated_handle
        else:
            yield calibrated_handle, calibrated_handle


def _total_calibration_factor(factors):
    total = np.asarray(1.0)
    for factor in factors.values():
        total = total * np.asarray(factor, dtype=float)
    if total.ndim == 0:
        return float(total)
    return total


def _factor_for_sca(factors, sca):
    selected = {}
    for step, value in factors.items():
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            selected[step] = float(values)
            continue
        values = values.reshape(-1)
        if values.size != 18:
            raise ValueError(
                f"{step} calibration factor must have 18 SCA values; "
                f"got shape {np.asarray(value).shape}"
            )
        selected[step] = float(values[sca])
    return selected


def _raw_stack_from_datasets(raw_dataset, calibrated_dataset, factors):
    if raw_dataset is not None:
        return _sca_stack(raw_dataset, raw_dataset.name)
    values = _sca_stack(calibrated_dataset, calibrated_dataset.name)
    total = _total_calibration_factor(factors)
    if np.asarray(total).ndim:
        total = np.asarray(total, dtype=float).reshape(18, 1, 1)
    return values / total


def _raw_dataset_block(
    raw_dataset,
    calibrated_dataset,
    factors,
    sca,
    row_start,
    row_stop,
    col_start,
    col_stop,
):
    dataset = raw_dataset if raw_dataset is not None else calibrated_dataset
    values = _dataset_block(
        dataset,
        sca,
        row_start,
        row_stop,
        col_start,
        col_stop,
        dataset.name,
    )
    if raw_dataset is None:
        values = values / _total_calibration_factor(_factor_for_sca(factors, sca))
    return values


def _calibration_factors(handle, led):
    factors = {}
    for step in CALIBRATION_STEPS:
        value = _attribute(handle.attrs, f"cal_{step}_LED{led}")
        if value is None:
            value = _attribute(handle.attrs, f"cal_{step}_led{led}")
        if value is None:
            dataset_path = f"calibration/{step}/multiplier"
            if dataset_path not in handle:
                raise KeyError(
                    f"{handle.filename} has no cal_{step}_LED{led} attribute "
                    f"or {dataset_path} dataset"
                )
            dataset = handle[dataset_path]
            value = dataset[led - 1]
        values = np.asarray(_decode(value), dtype=float).reshape(-1)
        if values.size not in {1, 18}:
            raise ValueError(
                f"cal_{step}_LED{led} must be scalar or have 18 SCA values; "
                f"got shape {np.asarray(value).shape}"
            )
        factor = float(values[0]) if values.size == 1 else values
        if not np.all(np.isfinite(factor)) or np.any(np.asarray(factor) <= 0.0):
            raise ValueError(
                f"cal_{step}_LED{led} is not a positive finite factor: {factor}"
            )
        factors[step] = factor
    return factors


def _paired_calibration_factors(calibrated_handle, raw_handle, led):
    """Read factors from the calibrated file, with a raw-file fallback."""
    try:
        return _calibration_factors(calibrated_handle, led)
    except KeyError:
        if raw_handle is calibrated_handle:
            raise
        return _calibration_factors(raw_handle, led)


def _factor_values(factors, baseline):
    row = {}
    for step in CALIBRATION_STEPS:
        row[f"{step}_multiplier"] = factors[step]
    total_no_tempgain = float(
        np.prod([factors[step] for step in CALIBRATION_STEPS if step != "tempgain"])
    )
    total = float(np.prod([factors[step] for step in CALIBRATION_STEPS]))
    row["total_multiplier_no_tempgain"] = total_no_tempgain
    row["total_multiplier"] = total
    for step in CALIBRATION_STEPS:
        row[f"log_{step}_ratio_to_first"] = float(
            np.log(factors[step] / baseline[step])
        )
    baseline_no_tempgain = float(
        np.prod([baseline[step] for step in CALIBRATION_STEPS if step != "tempgain"])
    )
    baseline_total = float(np.prod([baseline[step] for step in CALIBRATION_STEPS]))
    row["log_total_no_tempgain_ratio_to_first"] = float(
        np.log(total_no_tempgain / baseline_no_tempgain)
    )
    row["log_total_calibrated_ratio_to_first"] = float(
        np.log(total / baseline_total)
    )
    return row


def _common_values(metadata):
    return {field: metadata.get(field, "") for field in COMMON_FIELDS}


def _rate_stats(values):
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan, np.nan, 0.0
    selected = values[finite]
    return (
        float(np.mean(selected)),
        float(np.median(selected)),
        float(np.count_nonzero(finite) / finite.size),
    )


def _log_stats(current, baseline):
    valid = (
        np.isfinite(current)
        & np.isfinite(baseline)
        & (current > 0.0)
        & (baseline > 0.0)
    )
    if not np.any(valid):
        return np.nan, np.nan, 0.0
    changes = np.log(current[valid] / baseline[valid])
    return (
        float(np.mean(changes)),
        float(np.median(changes)),
        float(np.count_nonzero(valid) / valid.size),
    )


def _rate_values(
    current,
    baseline,
    factors,
    baseline_factors,
    calibrated_current=None,
    calibrated_baseline=None,
):
    no_tempgain = float(
        np.prod([factors[step] for step in CALIBRATION_STEPS if step != "tempgain"])
    )
    baseline_no_tempgain = float(
        np.prod(
            [
                baseline_factors[step]
                for step in CALIBRATION_STEPS
                if step != "tempgain"
            ]
        )
    )
    calibrated = float(np.prod(list(factors.values())))
    baseline_calibrated = float(np.prod(list(baseline_factors.values())))
    if calibrated_current is None:
        calibrated_current = current * calibrated
    if calibrated_baseline is None:
        calibrated_baseline = baseline * baseline_calibrated

    raw_mean, raw_median, raw_fraction = _rate_stats(current)
    raw_log_mean, raw_log_median, raw_log_fraction = _log_stats(
        current, baseline
    )
    no_mean, no_median, _ = _rate_stats(current * no_tempgain)
    no_log_mean, no_log_median, _ = _log_stats(
        current * no_tempgain,
        baseline * baseline_no_tempgain,
    )
    cal_mean, cal_median, _ = _rate_stats(calibrated_current)
    cal_log_mean, cal_log_median, _ = _log_stats(
        calibrated_current,
        calibrated_baseline,
    )
    return {
        "raw_mean_dn_s": raw_mean,
        "raw_median_dn_s": raw_median,
        "raw_valid_fraction": raw_fraction,
        "raw_log_change_mean": raw_log_mean,
        "raw_log_change_median": raw_log_median,
        "raw_log_pair_valid_fraction": raw_log_fraction,
        "no_tempgain_mean_dn_s": no_mean,
        "no_tempgain_median_dn_s": no_median,
        "no_tempgain_log_change_mean": no_log_mean,
        "no_tempgain_log_change_median": no_log_median,
        "calibrated_mean_dn_s": cal_mean,
        "calibrated_median_dn_s": cal_median,
        "calibrated_log_change_mean": cal_log_mean,
        "calibrated_log_change_median": cal_log_median,
    }


def _tile_geometry(n_superpixels, bin_factor, tile_size):
    if bin_factor <= 0 or tile_size <= 0:
        raise ValueError("bin factor and tile size must be positive")
    if tile_size % bin_factor:
        raise ValueError(
            f"tile size {tile_size} is not divisible by bin factor {bin_factor}"
        )
    detector_pixels = n_superpixels * bin_factor
    if detector_pixels % tile_size:
        raise ValueError(
            f"{n_superpixels} superpixels at bin factor {bin_factor} do not "
            f"form an integral {tile_size}-pixel tile grid"
        )
    tiles_per_axis = detector_pixels // tile_size
    block_size = tile_size // bin_factor
    return tiles_per_axis, block_size


def _tile_slices(tile_id, tiles_per_axis, block_size):
    if tile_id < 0 or tile_id >= tiles_per_axis * tiles_per_axis:
        raise ValueError(
            f"tile {tile_id} is outside the {tiles_per_axis}x{tiles_per_axis} grid"
        )
    tile_row, tile_col = divmod(tile_id, tiles_per_axis)
    row_start = tile_row * block_size
    col_start = tile_col * block_size
    return (
        tile_row,
        tile_col,
        row_start,
        row_start + block_size,
        col_start,
        col_start + block_size,
    )


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_s3_uri(uri):
    parsed = urlparse(str(uri))
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def _s3_object_uri(bucket, key):
    return f"s3://{bucket}/{key}"


def _run_aws(arguments, aws_cli="aws", aws_profile=None):
    command = [str(aws_cli)]
    if aws_profile:
        command.extend(("--profile", str(aws_profile)))
    command.extend(str(argument) for argument in arguments)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"AWS CLI executable was not found: {aws_cli!r}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"AWS command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{detail}"
        )
    return completed.stdout


def _list_s3_objects(s3_uri, aws_cli="aws", aws_profile=None):
    bucket, prefix = _parse_s3_uri(s3_uri)
    output = _run_aws(
        (
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--output",
            "json",
        ),
        aws_cli=aws_cli,
        aws_profile=aws_profile,
    )
    payload = json.loads(output)
    objects = list(payload.get("Contents", ()))
    if not objects:
        raise ValueError(f"No objects found below {s3_uri}")
    return bucket, prefix, objects


def _relative_s3_key(key, prefix):
    if not key.startswith(prefix):
        raise ValueError(f"S3 object {key!r} is outside prefix {prefix!r}")
    relative = key[len(prefix) :]
    if not relative:
        return ""
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe relative S3 object key: {key!r}")
    return relative


def _raw_relative_s3_key(calibrated_relative):
    """Map cal/foo_cal.h5 to its paired h5/foo_raw.h5 object."""
    path = Path(calibrated_relative)
    if not path.parts or path.parts[0] != "cal":
        raise ValueError(
            f"Expected a cal/ relative key, got {calibrated_relative!r}"
        )
    for calibrated_suffix, raw_suffix in (
        ("_cal.h5", "_raw.h5"),
        ("_cal.hdf5", "_raw.hdf5"),
    ):
        if path.name.endswith(calibrated_suffix):
            raw_name = path.name[: -len(calibrated_suffix)] + raw_suffix
            return (Path("h5") / raw_name).as_posix()
    raise ValueError(
        f"Calibrated HDF5 key has no recognized suffix: {calibrated_relative!r}"
    )


def _prepare_s3_source(
    s3_uri,
    source_root,
    inventory_path,
    max_products=None,
    aws_cli="aws",
    aws_profile=None,
):
    """Select S3 products and fetch only the small metadata sidecars."""
    if max_products is not None and max_products < 1:
        raise ValueError("max_products must be positive")
    bucket, prefix, objects = _list_s3_objects(
        s3_uri,
        aws_cli=aws_cli,
        aws_profile=aws_profile,
    )
    source_root.mkdir(parents=True, exist_ok=True)
    relative_objects = [
        (item, _relative_s3_key(str(item["Key"]), prefix))
        for item in objects
        if item.get("Key")
    ]
    relative_objects = [
        (item, relative) for item, relative in relative_objects if relative
    ]
    cal_objects = [
        (item, relative)
        for item, relative in relative_objects
        if relative.startswith("cal/")
        and Path(relative).suffix.lower() in {".h5", ".hdf5"}
    ]
    cal_objects.sort(key=lambda item: item[1])
    if max_products is not None:
        cal_objects = cal_objects[:max_products]
    if not cal_objects:
        raise ValueError(f"No cal/*.h5 objects found below {s3_uri}")

    objects_by_relative = {relative: item for item, relative in relative_objects}
    product_objects = []
    missing_raw = []
    for calibrated_item, calibrated_relative in cal_objects:
        raw_relative = _raw_relative_s3_key(calibrated_relative)
        raw_item = objects_by_relative.get(raw_relative)
        if raw_item is None:
            missing_raw.append(
                {
                    "calibrated": calibrated_relative,
                    "expected_raw": raw_relative,
                }
            )
            continue
        product_objects.append(
            (calibrated_item, calibrated_relative, raw_item, raw_relative)
        )
    if missing_raw:
        examples = ", ".join(item["calibrated"] for item in missing_raw[:3])
        raise ValueError(
            "Missing matching h5 raw objects for cal products: "
            f"{examples}"
        )

    known_metadata = {
        "rcs_reference_card.ecsv",
        "rcs_bridge_solution.ecsv",
        "tempgain_model.ecsv",
    }
    metadata_objects = [
        (item, relative)
        for item, relative in relative_objects
        if not relative.endswith("/")
        and int(item.get("Size", 0)) > 0
        if relative in known_metadata
        or relative.startswith("digests_v4_10_12_26/")
        or relative.startswith("digests/digests_v4_10_12_26/")
    ]
    inventory = {
        "s3_uri": str(s3_uri),
        "bucket": bucket,
        "prefix": prefix,
        "objects": objects,
        "calibration_objects_selected": [
            relative for _item, relative, _raw_item, _raw_relative in product_objects
        ],
        "raw_objects_selected": [
            relative for _item, _cal_relative, _raw_item, relative in product_objects
        ],
        "calibration_objects_downloaded_one_at_a_time": True,
        "raw_objects_downloaded_one_at_a_time": True,
    }
    with inventory_path.open("w") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for item, relative in metadata_objects:
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_aws(
            (
                "s3",
                "cp",
                _s3_object_uri(bucket, str(item["Key"])),
                str(destination),
                "--only-show-errors",
            ),
            aws_cli=aws_cli,
            aws_profile=aws_profile,
        )
    return bucket, prefix, product_objects, inventory


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _copy_metadata(source_root, output_dir):
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    known_files = (
        "rcs_reference_card.ecsv",
        "rcs_bridge_solution.ecsv",
        "tempgain_model.ecsv",
    )
    for name in known_files:
        source = source_root / name
        if not source.is_file():
            missing.append(str(source))
            continue
        destination = metadata_dir / name
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "output": str(destination.relative_to(output_dir)),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    digest_roots = (
        source_root / "digests" / "digests_v4_10_12_26",
        source_root / "digests_v4_10_12_26",
    )
    copied_digest_root = False
    for digest_root in digest_roots:
        if not digest_root.is_dir():
            continue
        copied_digest_root = True
        for source in sorted(path for path in digest_root.rglob("*") if path.is_file()):
            relative = source.relative_to(source_root)
            destination = metadata_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(
                {
                    "source": str(source),
                    "output": str(destination.relative_to(output_dir)),
                    "sha256": _sha256(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )
    if not copied_digest_root:
        missing.append(str(digest_roots[0]))
    return copied, missing


def _inspect_products(source_root, cal_dir, start_timestamp=None, end_timestamp=None):
    products = []
    paths = sorted(
        path
        for path in cal_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not paths:
        raise ValueError(f"No HDF5 products found below {cal_dir}")
    for path in paths:
        with h5py.File(path, "r") as handle:
            metadata = _metadata(handle, path, source_root)
            raw_path = _raw_product_path(source_root, path)
            metadata["raw_source_path"] = str(raw_path) if raw_path else ""
            if (
                start_timestamp is not None
                and _timestamp_sort_key(metadata["timestamp"])
                < _timestamp_sort_key(start_timestamp)
            ):
                continue
            if (
                end_timestamp is not None
                and _timestamp_sort_key(metadata["timestamp"])
                > _timestamp_sort_key(end_timestamp)
            ):
                continue
        products.append(metadata)
    products.sort(
        key=lambda item: (
            _timestamp_sort_key(item["timestamp"]),
            item["source_relpath"],
        )
    )
    if not products:
        raise ValueError("No HDF5 products remain after the timestamp filter")
    return products


def _normalize_detectors(detectors, all_detectors):
    if all_detectors:
        return tuple(f"WFI{index:02d}" for index in range(1, 19))
    normalized = []
    for detector in detectors:
        detector = detector.upper()
        if not detector.startswith("WFI"):
            detector = f"WFI{int(detector):02d}"
        try:
            number = int(detector[3:])
        except ValueError as error:
            raise ValueError(f"Invalid detector: {detector}") from error
        if not 1 <= number <= 18:
            raise ValueError(f"Invalid detector: {detector}")
        normalized.append(f"WFI{number:02d}")
    if not normalized:
        raise ValueError("At least one detector is required")
    return tuple(dict.fromkeys(normalized))


def _product_manifest_row(metadata, n_superpixels, sha256=""):
    row = {field: metadata.get(field, "") for field in PRODUCT_FIELDS}
    row.update(
        {
            "source_size_bytes": metadata["source_size_bytes"],
            "source_sha256": sha256,
            "n_sca": 18,
            "n_superpixels": n_superpixels,
        }
    )
    return row


class _StreamingBundleWriter:
    """Accumulate selected blocks while source HDF5 files are short-lived."""

    def __init__(
        self,
        source_root,
        output_dir,
        detectors,
        tile_ids,
        all_tiles,
        tile_size,
        bin_factor_override,
        include_sca_summary=False,
    ):
        self.source_root = Path(source_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.detectors = tuple(detectors)
        self.scas = tuple(int(detector[3:]) - 1 for detector in detectors)
        self.tile_ids = tuple(tile_ids)
        self.all_tiles = bool(all_tiles)
        self.tile_size = int(tile_size)
        self.bin_factor_override = bin_factor_override
        self.include_sca_summary = bool(include_sca_summary)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.product_manifest_rows = []
        self.correction_rows = []
        self.sca_rows = []
        self.tile_rows = []
        self.baseline_raw = {}
        self.baseline_sca_raw = {}
        self.baseline_factors = {}
        self.initialized = False

    def _initialize(self, source_path, raw_source_path=None):
        with _open_product_pair(source_path, raw_source_path) as (
            raw_handle,
            calibrated_handle,
        ):
            first_raw_dataset, first_calibrated_dataset = _paired_product_datasets(
                raw_handle,
                calibrated_handle,
                1,
            )
            first_dataset = (
                first_raw_dataset
                if first_raw_dataset is not None
                else first_calibrated_dataset
            )
            n_superpixels, _axis_order = _dataset_layout(
                first_dataset,
                first_dataset.name,
            )
            first_bin_factor = _numeric_attribute(
                calibrated_handle.attrs,
                "bin_factor",
                _numeric_attribute(raw_handle.attrs, "bin_factor", 16.0),
            )
        self.n_sca = 18
        self.n_superpixels = n_superpixels
        self.bin_factor = (
            int(self.bin_factor_override)
            if self.bin_factor_override is not None
            else int(round(first_bin_factor))
        )
        if self.n_sca != 18:
            raise ValueError(f"Expected 18 SCAs; first product has {self.n_sca}")
        self.tiles_per_axis, self.block_size = _tile_geometry(
            self.n_superpixels,
            self.bin_factor,
            self.tile_size,
        )
        selected_tiles = (
            tuple(range(self.tiles_per_axis * self.tiles_per_axis))
            if self.all_tiles
            else self.tile_ids
        )
        self.tile_descriptions = [
            (tile_id, _tile_slices(tile_id, self.tiles_per_axis, self.block_size))
            for tile_id in selected_tiles
        ]
        self.selected_tiles = selected_tiles
        self.initialized = True

    def process(
        self,
        source_path,
        metadata,
        raw_source_path=None,
        hash_products=False,
    ):
        source_path = Path(source_path)
        if not self.initialized:
            self._initialize(source_path, raw_source_path=raw_source_path)
        elif _timestamp_sort_key(metadata["timestamp"]) < _timestamp_sort_key(
            self.product_manifest_rows[-1]["timestamp"]
        ):
            raise ValueError(
                "S3 calibration objects are not timestamp ordered: "
                f"{metadata['timestamp']} follows "
                f"{self.product_manifest_rows[-1]['timestamp']}"
            )
        source_sha256 = _sha256(source_path) if hash_products else ""
        with _open_product_pair(source_path, raw_source_path) as (
            raw_handle,
            calibrated_handle,
        ):
            product_datasets = {}
            product_calibrated_datasets = {}
            product_factors = {}
            for led in range(1, 7):
                raw_dataset, calibrated_dataset = _paired_product_datasets(
                    raw_handle,
                    calibrated_handle,
                    led,
                )
                dataset = raw_dataset if raw_dataset is not None else calibrated_dataset
                current_n_superpixels, _axis_order = _dataset_layout(
                    dataset,
                    dataset.name,
                )
                if current_n_superpixels != self.n_superpixels:
                    raise ValueError(
                        f"{source_path} LED {led} has {current_n_superpixels} "
                        f"superpixels; expected {self.n_superpixels}"
                    )
                product_datasets[led] = raw_dataset
                if calibrated_dataset is not None:
                    calibrated_n_superpixels, _axis_order = _dataset_layout(
                        calibrated_dataset,
                        calibrated_dataset.name,
                    )
                    if calibrated_n_superpixels != self.n_superpixels:
                        raise ValueError(
                            f"{source_path} LED {led} calibrated data has "
                            f"{calibrated_n_superpixels} superpixels; expected "
                            f"{self.n_superpixels}"
                        )
                product_calibrated_datasets[led] = calibrated_dataset
                product_factors[led] = _paired_calibration_factors(
                    calibrated_handle,
                    raw_handle,
                    led,
                )

            self.product_manifest_rows.append(
                _product_manifest_row(metadata, self.n_superpixels, source_sha256)
            )
            for sca in self.scas:
                detector = f"WFI{sca + 1:02d}"
                for led in range(1, 7):
                    full_factors = product_factors[led]
                    factors = _factor_for_sca(full_factors, sca)
                    factor_key = (sca, led)
                    if factor_key not in self.baseline_factors:
                        self.baseline_factors[factor_key] = dict(factors)
                    factor_row = _common_values(metadata)
                    factor_row.update(
                        {
                            "sca": sca,
                            "detector": detector,
                            "led": led,
                            **_factor_values(
                                factors,
                                self.baseline_factors[factor_key],
                            ),
                        }
                    )
                    self.correction_rows.append(factor_row)
                    if self.include_sca_summary:
                        current = _raw_stack_from_datasets(
                            product_datasets[led],
                            product_calibrated_datasets[led],
                            full_factors,
                        )[sca]
                        sca_key = (sca, led)
                        if sca_key not in self.baseline_sca_raw:
                            self.baseline_sca_raw[sca_key] = current.copy()
                        baseline = self.baseline_sca_raw[sca_key]
                        metrics = _rate_values(
                            current,
                            baseline,
                            factors,
                            self.baseline_factors[factor_key],
                        )
                        sca_row = _common_values(metadata)
                        sca_row.update(
                            {
                                "sca": sca,
                                "detector": detector,
                                "led": led,
                                **metrics,
                            }
                        )
                        self.sca_rows.append(sca_row)

                    for tile_id, geometry in self.tile_descriptions:
                        (
                            tile_row,
                            tile_col,
                            row_start,
                            row_stop,
                            col_start,
                            col_stop,
                        ) = geometry
                        current_block = _raw_dataset_block(
                            product_datasets[led],
                            product_calibrated_datasets[led],
                            full_factors,
                            sca,
                            row_start,
                            row_stop,
                            col_start,
                            col_stop,
                        )
                        tile_key = (sca, led, tile_id)
                        if tile_key not in self.baseline_raw:
                            self.baseline_raw[tile_key] = current_block.copy()
                        baseline_block = self.baseline_raw[tile_key]
                        metrics = _rate_values(
                            current_block,
                            baseline_block,
                            factors,
                            self.baseline_factors[factor_key],
                        )
                        tile_row_data = _common_values(metadata)
                        tile_row_data.update(
                            {
                                "sca": sca,
                                "detector": detector,
                                "led": led,
                                "tile_id": tile_id,
                                "tile_row": tile_row,
                                "tile_col": tile_col,
                                "local_row_start": row_start * self.bin_factor,
                                "local_row_stop_exclusive": row_stop * self.bin_factor,
                                "local_col_start": col_start * self.bin_factor,
                                "local_col_stop_exclusive": col_stop * self.bin_factor,
                                "superpixel_row_start": row_start,
                                "superpixel_row_stop_exclusive": row_stop,
                                "superpixel_col_start": col_start,
                                "superpixel_col_stop_exclusive": col_stop,
                                **metrics,
                            }
                        )
                        self.tile_rows.append(tile_row_data)

    def finish(self, copy_metadata=True):
        if not self.product_manifest_rows:
            raise ValueError("No products were processed")
        output_paths = {
            "product_manifest": self.output_dir / "eric_v4_product_manifest.csv",
            "corrections": self.output_dir / "eric_v4_corrections.csv",
            "sca_trends": self.output_dir / "eric_v4_sca_trends.csv",
            "tile_trends": self.output_dir / "eric_v4_tile_trends.csv",
        }
        _write_csv(
            output_paths["product_manifest"],
            PRODUCT_FIELDS,
            self.product_manifest_rows,
        )
        _write_csv(output_paths["corrections"], CORRECTION_FIELDS, self.correction_rows)
        _write_csv(output_paths["sca_trends"], SCA_TREND_FIELDS, self.sca_rows)
        _write_csv(output_paths["tile_trends"], TILE_TREND_FIELDS, self.tile_rows)
        copied_metadata = []
        missing_metadata = []
        if copy_metadata:
            copied_metadata, missing_metadata = _copy_metadata(
                self.source_root,
                self.output_dir,
            )
        manifest = {
            "schema_version": 1,
            "script": Path(__file__).name,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "source_root": str(self.source_root),
            "calibration_directory": str(self.source_root / "cal"),
            "output_directory": str(self.output_dir),
            "detectors": list(self.detectors),
            "scas_zero_based": list(self.scas),
            "products_selected": len(self.product_manifest_rows),
            "first_timestamp": self.product_manifest_rows[0]["timestamp"],
            "last_timestamp": self.product_manifest_rows[-1]["timestamp"],
            "bin_factor": self.bin_factor,
            "n_superpixels": self.n_superpixels,
            "tile_size_pixels": self.tile_size,
            "tiles_per_axis": self.tiles_per_axis,
            "tile_block_size_superpixels": self.block_size,
            "tile_ids": list(self.selected_tiles),
            "include_sca_summary": self.include_sca_summary,
            "calibration_steps": list(CALIBRATION_STEPS),
            "digest_definition": "total_multiplier_no_tempgain excludes tempgain",
            "products": self.product_manifest_rows,
            "metadata_copied": copied_metadata,
            "metadata_missing": missing_metadata,
            "outputs": {
                key: str(path.relative_to(self.output_dir))
                for key, path in output_paths.items()
            },
        }
        with (self.output_dir / "manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        readme = (
            "Eric RCS Trend v4 comparison export\n"
            "===================================\n\n"
            "The HDF5 source products were read one at a time.\n"
            "Only the selected local tile blocks were materialized by default.\n"
            + (
                "The optional SCA summary was also exported.\n"
                if self.include_sca_summary
                else "Full-SCA summaries were not materialized.\n"
            )
            + "The no-tempgain columns apply photodiode, ACM, ACM bridge, and side\n"
            "bridge only. The calibrated columns also apply temperature gain.\n"
            "All log-change columns are relative to the first selected product for\n"
            "the same SCA and LED.\n\n"
            "For a local 256-pixel tile and bin_factor=16, each tile row/column\n"
            "is the corresponding 16x16 block in the v4 superpixel array.\n\n"
            "Use manifest.json for source metadata and copied reference files.\n"
        )
        (self.output_dir / "README.txt").write_text(readme)
        return manifest


def export_bundle(
    source_root,
    output_dir,
    detectors=DEFAULT_DETECTORS,
    tile_ids=DEFAULT_TILE_IDS,
    all_tiles=False,
    tile_size=256,
    bin_factor_override=None,
    start_timestamp=None,
    end_timestamp=None,
    max_products=None,
    hash_products=False,
    copy_metadata=True,
    include_sca_summary=False,
):
    """Build the comparison bundle and return its JSON manifest."""
    _require_h5py()
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    cal_dir = source_root / "cal"
    if not source_root.is_dir():
        raise ValueError(f"Source root does not exist: {source_root}")
    if not cal_dir.is_dir():
        raise ValueError(f"Calibration directory does not exist: {cal_dir}")
    detectors = tuple(detectors)
    scas = tuple(int(detector[3:]) - 1 for detector in detectors)
    products = _inspect_products(
        source_root,
        cal_dir,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    if max_products is not None:
        if max_products < 1:
            raise ValueError("max_products must be positive")
        products = products[:max_products]
    if not products:
        raise ValueError("No products selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    first_raw_source_path = products[0].get("raw_source_path") or None
    with _open_product_pair(products[0]["source_path"], first_raw_source_path) as (
        raw_handle,
        calibrated_handle,
    ):
        first_raw_dataset, first_calibrated_dataset = _paired_product_datasets(
            raw_handle,
            calibrated_handle,
            1,
        )
        first_dataset = (
            first_raw_dataset
            if first_raw_dataset is not None
            else first_calibrated_dataset
        )
        n_superpixels, _axis_order = _dataset_layout(
            first_dataset,
            first_dataset.name,
        )
        n_sca = 18
        first_bin_factor = _numeric_attribute(
            calibrated_handle.attrs,
            "bin_factor",
            _numeric_attribute(raw_handle.attrs, "bin_factor", 16.0),
        )
    bin_factor = (
        int(bin_factor_override)
        if bin_factor_override is not None
        else int(round(first_bin_factor))
    )
    if n_sca != 18:
        raise ValueError(f"Expected 18 SCAs; first product has {n_sca}")
    tiles_per_axis, block_size = _tile_geometry(
        n_superpixels,
        bin_factor,
        tile_size,
    )
    selected_tiles = (
        tuple(range(tiles_per_axis * tiles_per_axis))
        if all_tiles
        else tuple(tile_ids)
    )
    tile_descriptions = []
    for tile_id in selected_tiles:
        tile_descriptions.append(
            (tile_id, _tile_slices(tile_id, tiles_per_axis, block_size))
        )

    product_manifest_rows = []
    correction_rows = []
    sca_rows = []
    tile_rows = []
    baseline_raw = {}
    baseline_sca_raw = {}
    baseline_factors = {}

    for metadata in products:
        source_path = Path(metadata["source_path"])
        raw_source_path = metadata.get("raw_source_path") or None
        source_sha256 = _sha256(source_path) if hash_products else ""
        with _open_product_pair(source_path, raw_source_path) as (
            raw_handle,
            calibrated_handle,
        ):
            product_datasets = {}
            product_calibrated_datasets = {}
            product_factors = {}
            for led in range(1, 7):
                raw_dataset, calibrated_dataset = _paired_product_datasets(
                    raw_handle,
                    calibrated_handle,
                    led,
                )
                dataset = raw_dataset if raw_dataset is not None else calibrated_dataset
                current_n_superpixels, _axis_order = _dataset_layout(
                    dataset,
                    dataset.name,
                )
                if current_n_superpixels != n_superpixels:
                    raise ValueError(
                        f"{source_path} LED {led} has {current_n_superpixels} "
                        f"superpixels; expected {n_superpixels}"
                    )
                product_datasets[led] = raw_dataset
                if calibrated_dataset is not None:
                    calibrated_n_superpixels, _axis_order = _dataset_layout(
                        calibrated_dataset,
                        calibrated_dataset.name,
                    )
                    if calibrated_n_superpixels != n_superpixels:
                        raise ValueError(
                            f"{source_path} LED {led} calibrated data has "
                            f"{calibrated_n_superpixels} superpixels; "
                            f"expected {n_superpixels}"
                        )
                product_calibrated_datasets[led] = calibrated_dataset
                product_factors[led] = _paired_calibration_factors(
                    calibrated_handle,
                    raw_handle,
                    led,
                )

            product_manifest_rows.append(
                _product_manifest_row(metadata, n_superpixels, source_sha256)
            )
            for sca in scas:
                detector = f"WFI{sca + 1:02d}"
                for led in range(1, 7):
                    full_factors = product_factors[led]
                    factors = _factor_for_sca(full_factors, sca)
                    factor_key = (sca, led)
                    if factor_key not in baseline_factors:
                        baseline_factors[factor_key] = dict(factors)
                    factor_row = _common_values(metadata)
                    factor_row.update(
                        {
                            "sca": sca,
                            "detector": detector,
                            "led": led,
                            **_factor_values(
                                factors,
                                baseline_factors[factor_key],
                            ),
                        }
                    )
                    correction_rows.append(factor_row)
                    if include_sca_summary:
                        current = _raw_stack_from_datasets(
                            product_datasets[led],
                            product_calibrated_datasets[led],
                            full_factors,
                        )[sca]
                        sca_key = (sca, led)
                        if sca_key not in baseline_sca_raw:
                            baseline_sca_raw[sca_key] = current.copy()
                        baseline = baseline_sca_raw[sca_key]
                        metrics = _rate_values(
                            current,
                            baseline,
                            factors,
                            baseline_factors[factor_key],
                        )
                        sca_row = _common_values(metadata)
                        sca_row.update(
                            {
                                "sca": sca,
                                "detector": detector,
                                "led": led,
                                **metrics,
                            }
                        )
                        sca_rows.append(sca_row)

                    for tile_id, geometry in tile_descriptions:
                        (
                            tile_row,
                            tile_col,
                            row_start,
                            row_stop,
                            col_start,
                            col_stop,
                        ) = geometry
                        current_block = _raw_dataset_block(
                            product_datasets[led],
                            product_calibrated_datasets[led],
                            full_factors,
                            sca,
                            row_start,
                            row_stop,
                            col_start,
                            col_stop,
                        )
                        tile_key = (sca, led, tile_id)
                        if tile_key not in baseline_raw:
                            baseline_raw[tile_key] = current_block.copy()
                        baseline_block = baseline_raw[tile_key]
                        metrics = _rate_values(
                            current_block,
                            baseline_block,
                            factors,
                            baseline_factors[factor_key],
                        )
                        tile_row_data = _common_values(metadata)
                        tile_row_data.update(
                            {
                                "sca": sca,
                                "detector": detector,
                                "led": led,
                                "tile_id": tile_id,
                                "tile_row": tile_row,
                                "tile_col": tile_col,
                                "local_row_start": row_start * bin_factor,
                                "local_row_stop_exclusive": row_stop * bin_factor,
                                "local_col_start": col_start * bin_factor,
                                "local_col_stop_exclusive": col_stop * bin_factor,
                                "superpixel_row_start": row_start,
                                "superpixel_row_stop_exclusive": row_stop,
                                "superpixel_col_start": col_start,
                                "superpixel_col_stop_exclusive": col_stop,
                                **metrics,
                            }
                        )
                        tile_rows.append(tile_row_data)

    output_paths = {
        "product_manifest": output_dir / "eric_v4_product_manifest.csv",
        "corrections": output_dir / "eric_v4_corrections.csv",
        "sca_trends": output_dir / "eric_v4_sca_trends.csv",
        "tile_trends": output_dir / "eric_v4_tile_trends.csv",
    }
    _write_csv(output_paths["product_manifest"], PRODUCT_FIELDS, product_manifest_rows)
    _write_csv(output_paths["corrections"], CORRECTION_FIELDS, correction_rows)
    _write_csv(output_paths["sca_trends"], SCA_TREND_FIELDS, sca_rows)
    _write_csv(output_paths["tile_trends"], TILE_TREND_FIELDS, tile_rows)

    copied_metadata = []
    missing_metadata = []
    if copy_metadata:
        copied_metadata, missing_metadata = _copy_metadata(source_root, output_dir)

    manifest = {
        "schema_version": 1,
        "script": Path(__file__).name,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "source_root": str(source_root),
        "calibration_directory": str(cal_dir),
        "output_directory": str(output_dir),
        "detectors": list(detectors),
        "scas_zero_based": list(scas),
        "products_selected": len(products),
        "first_timestamp": products[0]["timestamp"],
        "last_timestamp": products[-1]["timestamp"],
        "bin_factor": bin_factor,
        "n_superpixels": n_superpixels,
        "tile_size_pixels": tile_size,
        "tiles_per_axis": tiles_per_axis,
        "tile_block_size_superpixels": block_size,
        "tile_ids": list(selected_tiles),
        "include_sca_summary": include_sca_summary,
        "calibration_steps": list(CALIBRATION_STEPS),
        "digest_definition": "total_multiplier_no_tempgain excludes tempgain",
        "products": product_manifest_rows,
        "metadata_copied": copied_metadata,
        "metadata_missing": missing_metadata,
        "outputs": {
            key: str(path.relative_to(output_dir))
            for key, path in output_paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "Eric RCS Trend v4 comparison export\n"
        "===================================\n\n"
        "The HDF5 source products were read in place and were not copied.\n"
        "Rates are DN/s medians/means over the selected local tile blocks.\n"
        + (
            "The optional SCA summary was also exported.\n"
            if include_sca_summary
            else "Full-SCA summaries were not materialized.\n"
        )
        + "The no-tempgain columns apply photodiode, ACM, ACM bridge, and side\n"
        "bridge only. The calibrated columns also apply temperature gain.\n"
        "All log-change columns are relative to the first selected product for\n"
        "the same SCA and LED.\n\n"
        "For a local 256-pixel tile and bin_factor=16, each tile row/column\n"
        "is the corresponding 16x16 block in the v4 superpixel array.\n\n"
        "Use manifest.json for source metadata and copied reference files.\n"
    )
    return manifest


def export_s3_bundle(
    s3_uri,
    output_dir,
    detectors=DEFAULT_DETECTORS,
    tile_ids=DEFAULT_TILE_IDS,
    all_tiles=False,
    tile_size=256,
    bin_factor_override=None,
    start_timestamp=None,
    end_timestamp=None,
    max_products=None,
    hash_products=False,
    copy_metadata=True,
    include_sca_summary=False,
    aws_cli="aws",
    aws_profile=None,
):
    """Stream the selected v4 epoch files from S3 into the compact bundle."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "s3_inventory.json"
    with tempfile.TemporaryDirectory(
        prefix=".eric_v4_s3_source_",
        dir=output_dir,
    ) as cache_name:
        cache_root = Path(cache_name)
        bucket, prefix, cal_objects, _inventory = _prepare_s3_source(
            s3_uri,
            cache_root,
            inventory_path,
            max_products=max_products,
            aws_cli=aws_cli,
            aws_profile=aws_profile,
        )
        writer = _StreamingBundleWriter(
            source_root=cache_root,
            output_dir=output_dir,
            detectors=detectors,
            tile_ids=tile_ids,
            all_tiles=all_tiles,
            tile_size=tile_size,
            bin_factor_override=bin_factor_override,
            include_sca_summary=include_sca_summary,
        )
        ordered_products = []
        for (
            calibrated_item,
            calibrated_relative,
            raw_item,
            raw_relative,
        ) in cal_objects:
            source_path = cache_root / calibrated_relative
            raw_source_path = cache_root / raw_relative
            source_path.parent.mkdir(parents=True, exist_ok=True)
            _run_aws(
                (
                    "s3",
                    "cp",
                    _s3_object_uri(bucket, str(calibrated_item["Key"])),
                    str(source_path),
                    "--only-show-errors",
                ),
                aws_cli=aws_cli,
                aws_profile=aws_profile,
            )
            try:
                with h5py.File(source_path, "r") as handle:
                    metadata = _metadata(handle, source_path, cache_root)
                timestamp = _timestamp_sort_key(metadata["timestamp"])
                if (
                    start_timestamp is not None
                    and timestamp < _timestamp_sort_key(start_timestamp)
                ) or (
                    end_timestamp is not None
                    and timestamp > _timestamp_sort_key(end_timestamp)
                ):
                    continue
                ordered_products.append(
                    (
                        timestamp,
                        calibrated_relative,
                        calibrated_item,
                        raw_item,
                        raw_relative,
                        metadata,
                    )
                )
            finally:
                source_path.unlink(missing_ok=True)
                raw_source_path.unlink(missing_ok=True)

        ordered_products.sort(key=lambda item: (item[0], item[1]))
        for (
            _timestamp,
            calibrated_relative,
            calibrated_item,
            raw_item,
            raw_relative,
            metadata,
        ) in ordered_products:
            source_path = cache_root / calibrated_relative
            raw_source_path = cache_root / raw_relative
            source_path.parent.mkdir(parents=True, exist_ok=True)
            _run_aws(
                (
                    "s3",
                    "cp",
                    _s3_object_uri(bucket, str(calibrated_item["Key"])),
                    str(source_path),
                    "--only-show-errors",
                ),
                aws_cli=aws_cli,
                aws_profile=aws_profile,
            )
            try:
                raw_source_path.parent.mkdir(parents=True, exist_ok=True)
                _run_aws(
                    (
                        "s3",
                        "cp",
                        _s3_object_uri(bucket, str(raw_item["Key"])),
                        str(raw_source_path),
                        "--only-show-errors",
                    ),
                    aws_cli=aws_cli,
                    aws_profile=aws_profile,
                )
                writer.process(
                    source_path,
                    metadata,
                    raw_source_path=raw_source_path,
                    hash_products=hash_products,
                )
            finally:
                source_path.unlink(missing_ok=True)
                raw_source_path.unlink(missing_ok=True)
        manifest = writer.finish(copy_metadata=copy_metadata)

    s3_root = str(s3_uri).rstrip("/")
    manifest["source_mode"] = "s3"
    manifest["source_uri"] = str(s3_uri)
    manifest["source_root"] = s3_root
    manifest["calibration_directory"] = f"{s3_root}/cal"
    manifest["s3_inventory"] = inventory_path.name
    for metadata in manifest["metadata_copied"]:
        relative = Path(metadata["output"]).relative_to("metadata")
        metadata["source"] = f"{s3_root}/{relative.as_posix()}"
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        readme_path.read_text()
        + "\nThe source was fetched from S3 one HDF5 object at a time; each object "
        "was removed after its selected tile blocks were exported. See "
        "s3_inventory.json for the remote object inventory.\n"
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source-root",
        "--root",
        type=Path,
        default=None,
        help=f"Explicit local Eric v4 root (legacy/default path: {DEFAULT_SOURCE_ROOT})",
    )
    source_group.add_argument(
        "--s3-uri",
        default=DEFAULT_S3_URI,
        help=f"Eric v4 S3 prefix (default: {DEFAULT_S3_URI})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eric_rcs_trend_comparison"),
        help="Output directory (default: eric_rcs_trend_comparison)",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=list(DEFAULT_DETECTORS),
        help="Detector names/numbers to export (default: WFI01 WFI10)",
    )
    parser.add_argument(
        "--all-detectors",
        action="store_true",
        help="Export all 18 SCA detector arrays",
    )
    parser.add_argument(
        "--tile-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_TILE_IDS),
        help="Local 256-pixel tile IDs (default: 136 137)",
    )
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="Export all tiles in the local tile grid",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Local tile size in full-resolution pixels (default: 256)",
    )
    parser.add_argument(
        "--bin-factor",
        type=int,
        default=None,
        help="Override the HDF5 bin_factor instead of reading its attribute",
    )
    parser.add_argument("--start-timestamp", help="Inclusive ISO timestamp filter")
    parser.add_argument("--end-timestamp", help="Inclusive ISO timestamp filter")
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Limit the chronological product count; default is the full series",
    )
    parser.add_argument(
        "--include-sca-summary",
        action="store_true",
        help="Also materialize full-SCA summaries; off by default",
    )
    parser.add_argument(
        "--hash-products",
        action="store_true",
        help="Compute SHA256 for each source HDF5; this rereads all source data",
    )
    parser.add_argument(
        "--no-copy-metadata",
        action="store_true",
        help="Do not copy the reference card, bridge solution, or digest files",
    )
    parser.add_argument(
        "--aws-cli",
        default="aws",
        help="AWS CLI executable (default: aws)",
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
        help="Optional AWS CLI profile",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detectors = _normalize_detectors(args.detectors, args.all_detectors)
    if args.source_root is not None:
        manifest = export_bundle(
            source_root=args.source_root,
            output_dir=args.output,
            detectors=detectors,
            tile_ids=args.tile_ids,
            all_tiles=args.all_tiles,
            tile_size=args.tile_size,
            bin_factor_override=args.bin_factor,
            start_timestamp=args.start_timestamp,
            end_timestamp=args.end_timestamp,
            max_products=args.max_products,
            hash_products=args.hash_products,
            copy_metadata=not args.no_copy_metadata,
            include_sca_summary=args.include_sca_summary,
        )
    else:
        manifest = export_s3_bundle(
            s3_uri=args.s3_uri,
            output_dir=args.output,
            detectors=detectors,
            tile_ids=args.tile_ids,
            all_tiles=args.all_tiles,
            tile_size=args.tile_size,
            bin_factor_override=args.bin_factor,
            start_timestamp=args.start_timestamp,
            end_timestamp=args.end_timestamp,
            max_products=args.max_products,
            hash_products=args.hash_products,
            copy_metadata=not args.no_copy_metadata,
            include_sca_summary=args.include_sca_summary,
            aws_cli=args.aws_cli,
            aws_profile=args.aws_profile,
        )
    print(
        f"Exported {manifest['products_selected']} products, "
        f"{len(manifest['detectors'])} detectors, "
        f"tiles={manifest['tile_ids']} to {manifest['output_directory']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
