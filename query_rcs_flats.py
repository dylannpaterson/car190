import argparse
import os
from pathlib import Path

from astropy.table import vstack
from astroquery.mast import MastMissions


ROOT = Path(__file__).resolve().parent
token_path = Path(
    os.environ.get("MAST_TOKEN_FILE", Path.home() / "Downloads/mast_api_token.txt")
)
DEFAULT_OUTPUT = ROOT / "wfi01_rcs_flat_index.csv"

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
    return parser.parse_args()


def query_detector(missions, detector, page_size):
    rows = []
    offset = 0
    while True:
        page = missions.query_criteria(
            exposure_type="WFI_FLAT",
            optical_element="DARK",
            detector=detector,
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

    token = token_path.read_text().strip()
    missions = MastMissions(mission="roman")
    missions.login(token=token)

    tables = []
    for detector in detectors:
        table = query_detector(missions, detector, args.page_size)
        if table is not None:
            tables.append(table)
        print(f"Found {0 if table is None else len(table)} {detector} L2 flat candidates")
    if not tables:
        raise RuntimeError("MAST returned no WFI L2 dark-flat candidates")

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
