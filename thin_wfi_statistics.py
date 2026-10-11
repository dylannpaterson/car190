#!/usr/bin/env python3
"""Create a cadence-thinned derived WFI statistics directory.

Program 1039 is copied unchanged.  Program 1063 is reduced to complete
six-LED cycles separated by at least the requested amount of time.  The
canonical corrected statistics are never modified.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


TARGET_PROGRAM = "1063"
EXPECTED_LEDS = frozenset(range(1, 7))
STATISTICS_GLOB = "*_tile_log_statistics_512_count_corrected.csv"
REQUIRED_FIELDS = frozenset(
    {"program", "repeat", "led", "exposure_start_time"}
)


def parse_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def positive_hours(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def complete_program_cycles(rows, program=TARGET_PROGRAM):
    """Return complete six-LED cycles as ``(first_time, last_time, repeat)``."""
    by_repeat = defaultdict(list)
    for row in rows:
        if str(row.get("program", "")) == str(program):
            by_repeat[int(row["repeat"])].append(row)

    cycles = []
    for repeat, repeat_rows in by_repeat.items():
        leds = {int(row["led"]) for row in repeat_rows}
        if leds != EXPECTED_LEDS:
            continue
        times = [parse_timestamp(row["exposure_start_time"]) for row in repeat_rows]
        cycles.append((min(times), max(times), repeat))
    return sorted(cycles)


def select_program_1063_repeats(rows, spacing_hours):
    """Select complete Program 1063 cycles with a minimum time separation."""
    cycles = complete_program_cycles(rows)
    if not cycles:
        raise ValueError("No complete six-LED Program 1063 cycles found")

    spacing = timedelta(hours=float(spacing_hours))
    selected = []
    for cycle in cycles:
        if not selected or cycle[0] >= selected[-1][1] + spacing:
            selected.append(cycle)
    return selected


def _read_rows(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Statistics file has no header: {path}")
        missing = REQUIRED_FIELDS.difference(fieldnames)
        if missing:
            raise ValueError(
                f"Statistics file is missing fields {sorted(missing)}: {path}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Statistics file is empty: {path}")
    return fieldnames, rows


def _write_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def thin_statistics_file(source, output, spacing_hours):
    source = Path(source)
    output = Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Source and output statistics files must differ")

    fieldnames, rows = _read_rows(source)
    selected_cycles = select_program_1063_repeats(rows, spacing_hours)
    selected_repeats = {cycle[2] for cycle in selected_cycles}
    retained = [
        row
        for row in rows
        if str(row.get("program", "")) != TARGET_PROGRAM
        or int(row["repeat"]) in selected_repeats
    ]
    _write_rows(output, fieldnames, retained)

    return {
        "source": str(source),
        "output": str(output),
        "source_rows": len(rows),
        "retained_rows": len(retained),
        "excluded_rows": len(rows) - len(retained),
        "source_program_1063_rows": sum(
            str(row.get("program", "")) == TARGET_PROGRAM for row in rows
        ),
        "retained_program_1063_rows": sum(
            str(row.get("program", "")) == TARGET_PROGRAM for row in retained
        ),
        "selected_repeats": [cycle[2] for cycle in selected_cycles],
        "selected_cycle_times": {
            str(cycle[2]): cycle[0].isoformat() for cycle in selected_cycles
        },
    }


def thin_statistics_directory(source_dir, output_dir, spacing_hours, force=False):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Source and output statistics directories must differ")
    if output_dir.exists() and not force:
        existing = list(output_dir.glob(STATISTICS_GLOB))
        if existing:
            raise FileExistsError(
                f"Output directory already contains statistics: {output_dir}; "
                "use --force to replace them"
            )

    source_paths = sorted(source_dir.glob(STATISTICS_GLOB))
    if not source_paths:
        raise FileNotFoundError(
            f"No corrected statistics files matching {STATISTICS_GLOB} in {source_dir}"
        )

    files = {}
    for source in source_paths:
        output = output_dir / source.name
        detector = source.name.split("_tile_log_statistics_", 1)[0].upper()
        files[detector] = thin_statistics_file(source, output, spacing_hours)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "policy": {
            "program_1039": "retain all rows unchanged",
            "program_1063": (
                "retain complete six-LED cycles separated by at least the requested "
                "cadence"
            ),
            "target_program": TARGET_PROGRAM,
            "spacing_hours": float(spacing_hours),
            "corrections": "preserve corrected sufficient statistics; apply no additional correction",
        },
        "files": files,
    }
    manifest_path = output_dir / "program1063_cadence_thinning_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Canonical corrected-statistics directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Derived directory to create.",
    )
    parser.add_argument(
        "--spacing-hours",
        type=positive_hours,
        default=1.0,
        help="Minimum spacing between complete Program 1063 cycles; default: 1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing derived statistics files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = thin_statistics_directory(
        args.source_dir,
        args.output_dir,
        args.spacing_hours,
        force=args.force,
    )
    print(
        f"Wrote {len(manifest['files'])} detector files to "
        f"{manifest['output_directory']}"
    )


if __name__ == "__main__":
    main()
