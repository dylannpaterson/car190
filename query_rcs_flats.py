import argparse
import csv
import os
from pathlib import Path
import tempfile

from astropy.table import vstack
from astroquery.mast import MastMissions


ROOT = Path(__file__).resolve().parent
token_path = Path(
    os.environ.get("MAST_TOKEN_FILE", Path.home() / "Downloads/mast_api_token.txt")
)
DEFAULT_OUTPUT = ROOT / "wfi01_rcs_flat_index.csv"
DEFAULT_PROGRAM = "1039"

select_cols = [
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
]
INDEX_FIELDS = [*select_cols, "cal_uri"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        default="WFI01",
        help="Detector to query; ignored with --all-detectors (default: WFI01)",
    )
    parser.add_argument(
        "--all-detectors",
        action="store_true",
        help="Query each of the 18 WFI detectors separately",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default depends on detector selection)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=5000,
        help="MAST page size per detector (default: 5000)",
    )
    parser.add_argument(
        "--program",
        default=DEFAULT_PROGRAM,
        help=f"MAST program to query (default: {DEFAULT_PROGRAM})",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help=(
            "Merge the fresh MAST query into the existing output indexed by "
            "fileSetName instead of overwriting it"
        ),
    )
    return parser.parse_args()


def query_detector(missions, detector, page_size, program):
    rows = []
    offset = 0
    while True:
        page = missions.query_criteria(
            exposure_type="WFI_FLAT",
            optical_element="DARK",
            detector=detector,
            program=program,
            product_type="l2",
            select_cols=select_cols,
            limit=page_size,
            offset=offset,
        )
        rows.append(page)
        if len(page) < page_size:
            break
        offset += len(page)
    if not rows or sum(len(page) for page in rows) == 0:
        return None
    return vstack(rows, metadata_conflicts="silent")


def table_to_index_rows(table):
    """Convert a MAST table to the stable CSV index schema."""
    rows = []
    for row in table:
        file_set = str(row["fileSetName"])
        rows.append(
            {
                **{column: str(row[column]) for column in select_cols},
                "cal_uri": f"{file_set}/{file_set}_cal.asdf",
            }
        )
    return rows


def read_index_rows(path):
    """Read an existing index while preserving its CSV values."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = set(INDEX_FIELDS).difference(fieldnames)
        if missing:
            raise ValueError(f"Existing index is missing columns: {sorted(missing)}")
        rows = []
        seen = set()
        for row in reader:
            file_set = row["fileSetName"].strip()
            if not file_set:
                raise ValueError("Existing index contains an empty fileSetName")
            if file_set in seen:
                raise ValueError(f"Existing index contains duplicate fileSetName: {file_set}")
            seen.add(file_set)
            rows.append({column: row.get(column, "") for column in INDEX_FIELDS})
    return rows


def merge_index_rows(existing_rows, fresh_rows):
    """Replace refreshed products and append products not in the existing index."""
    merged = {}
    for row in existing_rows:
        file_set = row["fileSetName"].strip()
        if file_set in merged:
            raise ValueError(f"Existing index contains duplicate fileSetName: {file_set}")
        merged[file_set] = row

    added = 0
    updated = 0
    for row in fresh_rows:
        file_set = row["fileSetName"].strip()
        if not file_set:
            raise ValueError("Fresh MAST query contains an empty fileSetName")
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
    """Write an updated index without leaving a partially written target."""
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


def main():
    args = parse_args()
    if args.page_size <= 0:
        raise ValueError("Page size must be positive")
    if args.all_detectors:
        detectors = [f"WFI{index:02d}" for index in range(1, 19)]
        output_path = args.output or ROOT / "all_wfi_rcs_flat_index.csv"
    else:
        if not args.detector.startswith("WFI"):
            raise ValueError("Detector must look like WFI01")
        detectors = [args.detector]
        output_path = args.output or DEFAULT_OUTPUT

    if args.update_existing and not output_path.exists():
        raise ValueError(
            f"--update-existing requires an existing index: {output_path}"
        )

    token = token_path.read_text().strip()
    missions = MastMissions(mission="roman")
    missions.login(token=token)

    tables = []
    for detector in detectors:
        table = query_detector(missions, detector, args.page_size, args.program)
        if table is not None:
            tables.append(table)
        print(f"Found {0 if table is None else len(table)} {detector} L2 flat candidates")
    if not tables:
        raise RuntimeError("MAST returned no WFI L2 dark-flat candidates")

    if args.update_existing:
        fresh_rows = []
        for table in tables:
            fresh_rows.extend(table_to_index_rows(table))
        existing_rows = read_index_rows(output_path)
        rows, added, updated = merge_index_rows(existing_rows, fresh_rows)
        write_index_rows_atomic(output_path, rows)
        print(
            f"Checked {len(fresh_rows)} current MAST rows; added {added}, "
            f"updated {updated}, retained {len(rows) - added - updated} unchanged"
        )
        print(f"Wrote {len(rows)} rows to {output_path}")
        return

    flats = vstack(tables, metadata_conflicts="silent")
    flats.sort("exposure_start_time")
    flats["cal_uri"] = [
        f"{file_set}/{file_set}_cal.asdf" for file_set in flats["fileSetName"]
    ]
    flats.write(output_path, format="ascii.csv", overwrite=True)
    print(f"Found {len(flats)} WFI L2 flat candidates total")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
