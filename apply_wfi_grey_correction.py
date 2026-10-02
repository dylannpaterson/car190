"""Add a post-reset grey correction to a count-space correction table."""

import argparse
import csv
import math
from pathlib import Path


REQUIRED_FIELDS = {
    "days",
    "count_log_correction",
    "log_gain_correction",
}


def write_grey_corrected_corrections(
    corrections_path,
    output_path,
    grey_scale,
    reset_day,
):
    if not math.isfinite(grey_scale) or not 0.0 < grey_scale:
        raise ValueError("grey_scale must be positive and finite")
    if not math.isfinite(reset_day):
        raise ValueError("reset_day must be finite")
    offset = -math.log(grey_scale)
    with Path(corrections_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = REQUIRED_FIELDS.difference(fields)
        if missing:
            raise ValueError(f"Correction table is missing fields: {sorted(missing)}")
        rows = list(reader)
    changed = 0
    for row in rows:
        if float(row["days"]) < reset_day:
            continue
        row["count_log_correction"] = repr(
            float(row["count_log_correction"]) + offset
        )
        row["log_gain_correction"] = repr(
            float(row["log_gain_correction"]) + offset
        )
        changed += 1
    if changed == 0:
        raise ValueError("No correction rows occur on or after reset_day")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return offset, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grey-scale", type=float, required=True)
    parser.add_argument("--reset-day", type=float, required=True)
    args = parser.parse_args()
    offset, changed = write_grey_corrected_corrections(
        args.corrections,
        args.output,
        args.grey_scale,
        args.reset_day,
    )
    print(
        f"Wrote {args.output}: added {offset:.12g} log correction "
        f"to {changed} post-reset rows"
    )


if __name__ == "__main__":
    main()
