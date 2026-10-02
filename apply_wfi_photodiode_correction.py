"""Apply a per-LED photodiode normalization to saved tile statistics."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path


LOG_FIELDS = (
    "sum_log_change",
    "sum_squared_log_change",
    "min_log_change",
    "max_log_change",
)
LINEAR_COUNT_FIELDS = (
    "sum_current_counts",
    "sum_reference_current_counts",
    "min_current_count",
    "max_current_count",
)
SQUARED_COUNT_FIELDS = ("sum_squared_current_counts",)
COUNT_FIELDS = (*LINEAR_COUNT_FIELDS, *SQUARED_COUNT_FIELDS)
REQUIRED_LOG_FIELDS = {
    "led",
    "exposure_start_time",
    "n_pair_valid",
    *LOG_FIELDS,
}
CHANNEL_FIELDS = {
    "pd1": "pd1_v_mean",
    "pd2": "pd2_v_mean",
}


def parse_time(value):
    """Parse timestamp formats used by the archive summary and RRN CSVs."""
    return datetime.fromisoformat(str(value).strip().rstrip("Z"))


def load_exposure_scales(summary_path, channel):
    """Return positive current-image factors keyed by timestamp and LED."""
    if channel not in CHANNEL_FIELDS:
        raise ValueError(f"Unsupported photodiode channel: {channel}")
    field = CHANNEL_FIELDS[channel]
    rows_by_led = {}
    with Path(summary_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"led", "exposure_start_time", field}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Photodiode summary is missing fields: {sorted(missing)}")
        for row in reader:
            led = int(row["led"])
            timestamp = parse_time(row["exposure_start_time"])
            value = float(row[field])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Photodiode value must be positive and finite: LED {led}, "
                    f"{row['exposure_start_time']}, {value!r}"
                )
            key = (timestamp, led)
            if key in rows_by_led:
                raise ValueError(f"Duplicate photodiode exposure: {key}")
            rows_by_led[key] = value
    if not rows_by_led:
        raise ValueError(f"Photodiode summary is empty: {summary_path}")

    references = {
        led: min(timestamp for timestamp, row_led in rows_by_led if row_led == led)
        for led in {row_led for _, row_led in rows_by_led}
    }
    scales = {}
    for key, value in rows_by_led.items():
        reference_time = references[key[1]]
        reference = rows_by_led[(reference_time, key[1])]
        factor = reference / value
        scales[key] = (factor, math.log(factor))
    return scales


def load_direct_exposure_scales(summary_path, scale_field):
    """Load already-computed multiplicative factors keyed by exposure."""
    rows_by_key = {}
    with Path(summary_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"led", "exposure_start_time", scale_field}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Photodiode scale summary is missing fields: {sorted(missing)}"
            )
        for row in reader:
            key = (parse_time(row["exposure_start_time"]), int(row["led"]))
            factor = float(row[scale_field])
            if not math.isfinite(factor) or factor <= 0.0:
                raise ValueError(
                    f"Photodiode scale must be positive and finite: {key}, {factor!r}"
                )
            if key in rows_by_key:
                raise ValueError(f"Duplicate photodiode exposure: {key}")
            rows_by_key[key] = factor
    if not rows_by_key:
        raise ValueError(f"Photodiode scale summary is empty: {summary_path}")
    return {key: (factor, math.log(factor)) for key, factor in rows_by_key.items()}


def _apply_scales(
    statistics_path,
    summary_path,
    output_path,
    scales,
    *,
    channel=None,
    scale_field=None,
    overwrite=False,
):
    """Apply keyed multiplicative factors to saved sufficient statistics."""
    statistics_path = Path(statistics_path)
    summary_path = Path(summary_path)
    output_path = Path(output_path)
    if statistics_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite the input statistics file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    count_statistics_available = None
    used_keys = set()
    temporary_path = output_path.with_name(output_path.name + ".partial")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_path}")
    try:
        with statistics_path.open(newline="") as source, temporary_path.open(
            "w", newline=""
        ) as target:
            reader = csv.DictReader(source)
            fieldnames = set(reader.fieldnames or ())
            missing = REQUIRED_LOG_FIELDS.difference(fieldnames)
            if missing:
                raise ValueError(f"Tile statistics are missing fields: {sorted(missing)}")
            present_count_fields = fieldnames.intersection(COUNT_FIELDS)
            if present_count_fields and present_count_fields != set(COUNT_FIELDS):
                raise ValueError(
                    "Tile statistics contain an incomplete set of count moments: "
                    f"{sorted(present_count_fields)}"
                )
            count_statistics_available = bool(present_count_fields)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = (parse_time(row["exposure_start_time"]), int(row["led"]))
                if key not in scales:
                    raise ValueError(
                        f"No photodiode exposure match for LED {key[1]}, "
                        f"{row['exposure_start_time']}"
                    )
                factor, log_factor = scales[key]
                count = float(row["n_pair_valid"])
                old_log_sum = float(row["sum_log_change"])
                old_log_squared = float(row["sum_squared_log_change"])
                row["sum_log_change"] = repr(old_log_sum + count * log_factor)
                row["sum_squared_log_change"] = repr(
                    old_log_squared
                    + 2.0 * log_factor * old_log_sum
                    + count * log_factor * log_factor
                )
                row["min_log_change"] = repr(
                    float(row["min_log_change"]) + log_factor
                )
                row["max_log_change"] = repr(
                    float(row["max_log_change"]) + log_factor
                )
                if count_statistics_available:
                    for field in LINEAR_COUNT_FIELDS:
                        row[field] = repr(float(row[field]) * factor)
                    for field in SQUARED_COUNT_FIELDS:
                        row[field] = repr(float(row[field]) * factor * factor)
                writer.writerow(row)
                row_count += 1
                used_keys.add(key)

        if used_keys != set(scales):
            unused = sorted(set(scales).difference(used_keys))
            raise ValueError(
                f"Statistics did not cover {len(unused)} photodiode exposures; "
                f"first missing exposure is {unused[0] if unused else 'unknown'}"
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    result = {
        "output": str(output_path.resolve()),
        "input": str(statistics_path.resolve()),
        "summary": str(summary_path.resolve()),
        "n_rows": row_count,
        "n_exposures": len(used_keys),
        "count_statistics_available": count_statistics_available,
    }
    if channel is not None:
        result["channel"] = channel
    if scale_field is not None:
        result["scale_field"] = scale_field
    return result


def apply_correction(statistics_path, summary_path, output_path, channel, overwrite=False):
    """Write statistics with the selected source normalization applied exactly."""
    return _apply_scales(
        statistics_path,
        summary_path,
        output_path,
        load_exposure_scales(summary_path, channel),
        channel=channel,
        overwrite=overwrite,
    )


def apply_scale_correction(
    statistics_path, summary_path, output_path, scale_field, overwrite=False
):
    """Apply a direct per-exposure replacement factor from a scale table."""
    return _apply_scales(
        statistics_path,
        summary_path,
        output_path,
        load_direct_exposure_scales(summary_path, scale_field),
        scale_field=scale_field,
        overwrite=overwrite,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--channel", choices=sorted(CHANNEL_FIELDS))
    source.add_argument(
        "--scale-field",
        help="Use a direct positive factor column from the scale summary.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.channel is not None:
        result = apply_correction(
            args.statistics,
            args.summary,
            args.output,
            args.channel,
            overwrite=args.overwrite,
        )
    else:
        result = apply_scale_correction(
            args.statistics,
            args.summary,
            args.output,
            args.scale_field,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
