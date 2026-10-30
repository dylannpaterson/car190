"""Standalone reader for portable WFI wavelength-response bundles.

This module intentionally depends only on NumPy so it can be shared with the
compressed bundle without the production TMM or JAX environment.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BUNDLE_VERSION = 2
LEGACY_BUNDLE_VERSION = 1
DETECTOR_PIXELS = 4096
ACTIVE_REFERENCE_PIXELS = 4
ACTIVE_PIXELS = DETECTOR_PIXELS - 2 * ACTIVE_REFERENCE_PIXELS
PIXEL_PITCH_MM = 0.01
MPA_SCA_WIDTH_MM = 40.88

# Coordinates are nominal MPA focal-plane millimetres. The MPA +x axis points
# left in the focal-plane view supplied with MPA_SCA_info.
WFI_SCA_LAYOUT = {
    1: (-22.14, 12.15, 0.0),
    2: (-22.29, -37.03, 0.0),
    3: (-22.44, -82.06, 0.0),
    4: (-66.42, 20.90, 0.0),
    5: (-66.92, -28.28, 0.0),
    6: (-67.42, -73.06, 0.0),
    7: (-110.70, 42.20, 0.0),
    8: (-111.48, -6.98, 0.0),
    9: (-112.64, -51.06, 0.0),
    10: (22.14, 12.15, 0.0),
    11: (22.29, -37.03, 0.0),
    12: (22.44, -82.06, 0.0),
    13: (66.42, 20.90, 0.0),
    14: (66.92, -28.28, 0.0),
    15: (67.42, -73.06, 0.0),
    16: (110.70, 42.20, 0.0),
    17: (111.48, -6.98, 0.0),
    18: (112.64, -51.06, 0.0),
}


def detector_to_sca_number(detector):
    detector = str(detector).strip().upper()
    if not detector.startswith("WFI") or not detector[3:].isdigit():
        raise ValueError(f"Expected WFI01-WFI18 detector label, got {detector!r}")
    sca_number = int(detector[3:])
    if sca_number not in WFI_SCA_LAYOUT:
        raise ValueError(f"Unknown WFI SCA number {sca_number}")
    return sca_number


def _active_interval(start, stop):
    active_start = max(int(start), ACTIVE_REFERENCE_PIXELS)
    active_stop = min(int(stop), DETECTOR_PIXELS - ACTIVE_REFERENCE_PIXELS)
    if active_stop <= active_start:
        raise ValueError(f"Tile interval [{start}, {stop}) has no active pixels")
    return active_start, active_stop


def tile_geometry(
    detector,
    tile_id,
    tile_row,
    tile_col,
    row_start,
    row_stop_exclusive,
    col_start,
    col_stop_exclusive,
):
    """Return nominal MPA geometry for one zero-based science-frame tile."""
    sca_number = detector_to_sca_number(detector)
    sca_x_mm, sca_y_mm, rotation_deg = WFI_SCA_LAYOUT[sca_number]
    active_row_start, active_row_stop = _active_interval(
        row_start, row_stop_exclusive
    )
    active_col_start, active_col_stop = _active_interval(
        col_start, col_stop_exclusive
    )
    row_center_pixel = 0.5 * (float(row_start) + float(row_stop_exclusive))
    col_center_pixel = 0.5 * (float(col_start) + float(col_stop_exclusive))
    active_row_center_pixel = 0.5 * (active_row_start + active_row_stop)
    active_col_center_pixel = 0.5 * (active_col_start + active_col_stop)

    # Science-frame columns increase to the right, while MPA +x points left.
    local_x_mm = -(active_col_center_pixel - DETECTOR_PIXELS / 2.0) * PIXEL_PITCH_MM
    local_y_mm = (active_row_center_pixel - DETECTOR_PIXELS / 2.0) * PIXEL_PITCH_MM
    angle = np.deg2rad(rotation_deg)
    rotated_x_mm = np.cos(angle) * local_x_mm - np.sin(angle) * local_y_mm
    rotated_y_mm = np.sin(angle) * local_x_mm + np.cos(angle) * local_y_mm
    return {
        "detector": str(detector).upper(),
        "sca_number": int(sca_number),
        "tile_id": int(tile_id),
        "tile_row": int(tile_row),
        "tile_col": int(tile_col),
        "row_start": int(row_start),
        "row_stop_exclusive": int(row_stop_exclusive),
        "col_start": int(col_start),
        "col_stop_exclusive": int(col_stop_exclusive),
        "tile_n_rows": int(row_stop_exclusive) - int(row_start),
        "tile_n_cols": int(col_stop_exclusive) - int(col_start),
        "pixel_center_row": row_center_pixel,
        "pixel_center_col": col_center_pixel,
        "active_pixel_center_row": active_row_center_pixel,
        "active_pixel_center_col": active_col_center_pixel,
        "sca_center_x_mm": float(sca_x_mm),
        "sca_center_y_mm": float(sca_y_mm),
        "sca_rotation_deg": float(rotation_deg),
        "tile_local_x_mm": float(rotated_x_mm),
        "tile_local_y_mm": float(rotated_y_mm),
        "tile_center_x_mm": float(sca_x_mm + rotated_x_mm),
        "tile_center_y_mm": float(sca_y_mm + rotated_y_mm),
    }


def load_bundle(path):
    """Load a bundle into memory and validate its shared dimensions."""
    with np.load(Path(path), allow_pickle=False) as source:
        version = int(np.asarray(source["bundle_version"]).reshape(-1)[0])
        if version not in (LEGACY_BUNDLE_VERSION, BUNDLE_VERSION):
            raise ValueError(f"Unsupported wavelength-response bundle version {version}")
        bundle = {name: np.asarray(source[name]) for name in source.files}
    metadata = json.loads(str(np.asarray(bundle["metadata_json"]).reshape(-1)[0]))
    bundle["metadata"] = metadata
    required = (
        "wavelengths_nm",
        "ice_thickness_nm",
        "clean_power",
        "response_left",
        "response_right",
        "tile_ids",
        "tile_center_x_mm",
        "tile_center_y_mm",
    )
    missing = [name for name in required if name not in bundle]
    if missing:
        raise ValueError(f"Bundle is missing required arrays: {missing}")
    tile_count = bundle["tile_ids"].size
    if version == LEGACY_BUNDLE_VERSION:
        detector = str(np.asarray(bundle["detector"]).reshape(-1)[0]).upper()
        sca_number = int(np.asarray(bundle["sca_number"]).reshape(-1)[0])
        bundle["detectors"] = np.full(tile_count, detector, dtype="U5")
        bundle["sca_numbers"] = np.full(tile_count, sca_number, dtype=np.int16)
    else:
        for name in ("detectors", "sca_numbers"):
            if name not in bundle:
                raise ValueError(f"Bundle is missing required array: {name}")
        if bundle["detectors"].size != tile_count:
            raise ValueError("Bundle detectors has an inconsistent tile count")
        if bundle["sca_numbers"].size != tile_count:
            raise ValueError("Bundle sca_numbers has an inconsistent tile count")
    if bundle["clean_power"].shape[0] != tile_count:
        raise ValueError("Bundle tile arrays have inconsistent lengths")
    if bundle["response_left"].shape[0] != tile_count:
        raise ValueError("Bundle response_left has an inconsistent tile count")
    if bundle["response_right"].shape[0] != tile_count:
        raise ValueError("Bundle response_right has an inconsistent tile count")
    trajectory_fields = (
        "observation_days",
        "observation_start_time",
        "observation_led",
        "observation_repeat",
        "observation_program",
        "groups",
        "tile_ice_thickness",
        "tile_instantaneous_rate_nm_day",
    )
    grey_fields = (
        "post_reset_tile_log_offset",
        "post_reset_tile_throughput_factor",
        "ice_reset_day",
    )
    third_grey_fields = (
        "post_third_decon_tile_log_offset",
        "post_third_decon_tile_throughput_factor",
        "third_decon_day",
    )
    present = [name for name in trajectory_fields if name in bundle]
    if present and len(present) != len(trajectory_fields):
        missing = [name for name in trajectory_fields if name not in bundle]
        raise ValueError(f"Bundle has incomplete trajectory fields: {missing}")
    if len(present) == len(trajectory_fields):
        days = np.asarray(bundle["observation_days"], dtype=float)
        thickness = np.asarray(bundle["tile_ice_thickness"], dtype=float)
        rates = np.asarray(bundle["tile_instantaneous_rate_nm_day"], dtype=float)
        if days.ndim != 1 or days.size == 0:
            raise ValueError("Bundle has an invalid observation time axis")
        if thickness.shape != (tile_count, days.size):
            raise ValueError("Bundle tile_ice_thickness has an inconsistent shape")
        if rates.shape != thickness.shape:
            raise ValueError(
                "Bundle tile_instantaneous_rate_nm_day has an inconsistent shape"
            )
        if np.asarray(bundle["observation_start_time"]).shape != days.shape:
            raise ValueError("Bundle observation_start_time has an inconsistent shape")
        for name in ("observation_led", "observation_repeat", "observation_program", "groups"):
            if np.asarray(bundle[name]).shape != days.shape:
                raise ValueError(f"Bundle {name} has an inconsistent shape")
        bundle["trajectory_included"] = True
        grey_present = [name for name in grey_fields if name in bundle]
        if grey_present and len(grey_present) != len(grey_fields):
            missing = [name for name in grey_fields if name not in bundle]
            raise ValueError(f"Bundle has incomplete grey-throughput fields: {missing}")
        if len(grey_present) == len(grey_fields):
            offsets = np.asarray(bundle["post_reset_tile_log_offset"], dtype=float)
            factors = np.asarray(
                bundle["post_reset_tile_throughput_factor"], dtype=float
            )
            reset_day = np.asarray(bundle["ice_reset_day"], dtype=float).reshape(-1)
            if offsets.shape != (tile_count,) or factors.shape != (tile_count,):
                raise ValueError("Bundle grey-throughput arrays have an inconsistent shape")
            if reset_day.size != 1 or not np.isfinite(reset_day[0]):
                raise ValueError("Bundle has an invalid ice reset day")
            if not np.allclose(factors, np.exp(offsets), rtol=1.0e-6, atol=1.0e-7):
                raise ValueError("Bundle grey-throughput factors do not match log offsets")
            bundle["grey_throughput_included"] = True
            third_present = [name for name in third_grey_fields if name in bundle]
            if third_present and len(third_present) != len(third_grey_fields):
                missing = [name for name in third_grey_fields if name not in bundle]
                raise ValueError(
                    f"Bundle has incomplete third-decon grey-throughput fields: {missing}"
                )
            if len(third_present) == len(third_grey_fields):
                third_offsets = np.asarray(
                    bundle["post_third_decon_tile_log_offset"], dtype=float
                )
                third_factors = np.asarray(
                    bundle["post_third_decon_tile_throughput_factor"], dtype=float
                )
                third_day = np.asarray(bundle["third_decon_day"], dtype=float).reshape(-1)
                if (
                    third_offsets.shape != (tile_count,)
                    or third_factors.shape != (tile_count,)
                ):
                    raise ValueError(
                        "Bundle third-decon grey-throughput arrays have an inconsistent shape"
                    )
                if (
                    third_day.size != 1
                    or not np.isfinite(third_day[0])
                    or third_day[0] <= reset_day[0]
                ):
                    raise ValueError("Bundle has an invalid third-decon day")
                if not np.allclose(
                    third_factors,
                    np.exp(third_offsets),
                    rtol=1.0e-6,
                    atol=1.0e-7,
                ):
                    raise ValueError(
                        "Bundle third-decon throughput factors do not match log offsets"
                    )
                bundle["third_decon_included"] = True
            else:
                bundle["third_decon_included"] = False
        else:
            bundle["grey_throughput_included"] = False
            bundle["third_decon_included"] = False
    else:
        bundle["trajectory_included"] = False
        bundle["grey_throughput_included"] = False
    return bundle


def bundle_summary(bundle):
    detectors = sorted({str(value) for value in bundle["detectors"]})
    summary = {
        "bundle_version": int(np.asarray(bundle["bundle_version"]).reshape(-1)[0]),
        "detectors": detectors,
        "detector_count": len(detectors),
        "tile_count": int(bundle["tile_ids"].size),
        "wavelength_count": int(bundle["wavelengths_nm"].size),
        "wavelength_min_nm": float(bundle["wavelengths_nm"][0]),
        "wavelength_max_nm": float(bundle["wavelengths_nm"][-1]),
        "thickness_count": int(bundle["ice_thickness_nm"].size),
        "thickness_min_nm": float(bundle["ice_thickness_nm"][0]),
        "thickness_max_nm": float(bundle["ice_thickness_nm"][-1]),
        "response_rank": int(np.asarray(bundle["response_rank"]).reshape(-1)[0]),
        "metadata": bundle["metadata"],
    }
    if bundle.get("trajectory_included", False):
        summary.update(
            {
                "trajectory_included": True,
                "observation_count": int(bundle["observation_days"].size),
                "observation_day_min": float(bundle["observation_days"][0]),
                "observation_day_max": float(bundle["observation_days"][-1]),
            }
        )
    else:
        summary["trajectory_included"] = False
        summary["grey_throughput_included"] = False
    if bundle.get("grey_throughput_included", False):
        summary["grey_throughput_included"] = True
        summary["ice_reset_day"] = float(
            np.asarray(bundle["ice_reset_day"], dtype=float).reshape(-1)[0]
        )
    if bundle.get("third_decon_included", False):
        summary["third_decon_day"] = float(
            np.asarray(bundle["third_decon_day"], dtype=float).reshape(-1)[0]
        )
    return summary


def tile_index(bundle, tile_id, detector=None):
    matches = bundle["tile_ids"] == int(tile_id)
    if detector is not None:
        matches &= bundle["detectors"] == str(detector).strip().upper()
    matches = np.flatnonzero(matches)
    if matches.size != 1:
        if matches.size > 1:
            raise KeyError(f"Tile {tile_id} is ambiguous; specify a detector")
        label = f"{detector} tile {tile_id}" if detector else f"Tile {tile_id}"
        raise KeyError(f"{label} is not present in the bundle")
    return int(matches[0])


def tile_record(bundle, tile_id, detector=None):
    index = tile_index(bundle, tile_id, detector=detector)
    array_fields = {
        "tile_id": "tile_ids",
        "tile_row": "tile_rows",
        "tile_col": "tile_cols",
        "row_start": "row_starts",
        "row_stop_exclusive": "row_stops_exclusive",
        "col_start": "col_starts",
        "col_stop_exclusive": "col_stops_exclusive",
        "tile_n_rows": "tile_n_rows",
        "tile_n_cols": "tile_n_cols",
        "pixel_center_row": "pixel_center_rows",
        "pixel_center_col": "pixel_center_cols",
        "active_pixel_center_row": "active_pixel_center_rows",
        "active_pixel_center_col": "active_pixel_center_cols",
        "sca_center_x_mm": "sca_center_x_mm",
        "sca_center_y_mm": "sca_center_y_mm",
        "sca_rotation_deg": "sca_rotation_deg",
        "tile_local_x_mm": "tile_local_x_mm",
        "tile_local_y_mm": "tile_local_y_mm",
        "tile_center_x_mm": "tile_center_x_mm",
        "tile_center_y_mm": "tile_center_y_mm",
    }
    record = {
        name: bundle[array_name][index].item()
        if isinstance(bundle[array_name][index], np.generic)
        else bundle[array_name][index]
        for name, array_name in array_fields.items()
    }
    record["detector"] = str(bundle["detectors"][index])
    record["sca_number"] = int(bundle["sca_numbers"][index])
    return record


def tile_trajectory(bundle, tile_id, detector=None):
    """Return one tile's fitted thickness and rate trajectory."""
    if not bundle.get("trajectory_included", False):
        raise ValueError("Bundle does not contain fitted ice trajectories")
    index = tile_index(bundle, tile_id, detector=detector)
    record = {
        "detector": str(bundle["detectors"][index]),
        "tile_id": int(bundle["tile_ids"][index]),
        "observation_start_time": np.asarray(bundle["observation_start_time"]),
        "observation_days": np.asarray(bundle["observation_days"], dtype=float),
        "ice_thickness_nm": np.asarray(bundle["tile_ice_thickness"][index], dtype=float),
        "instantaneous_rate_nm_day": np.asarray(
            bundle["tile_instantaneous_rate_nm_day"][index], dtype=float
        ),
    }
    if bundle.get("grey_throughput_included", False):
        record["post_reset_tile_log_offset"] = float(
            bundle["post_reset_tile_log_offset"][index]
        )
        record["post_reset_tile_throughput_factor"] = float(
            bundle["post_reset_tile_throughput_factor"][index]
        )
        record["ice_reset_day"] = float(
            np.asarray(bundle["ice_reset_day"], dtype=float).reshape(-1)[0]
        )
        if bundle.get("third_decon_included", False):
            record["post_third_decon_tile_log_offset"] = float(
                bundle["post_third_decon_tile_log_offset"][index]
            )
            record["post_third_decon_tile_throughput_factor"] = float(
                bundle["post_third_decon_tile_throughput_factor"][index]
            )
            record["third_decon_day"] = float(
                np.asarray(bundle["third_decon_day"], dtype=float).reshape(-1)[0]
            )
    return record


def all_tile_records(bundle):
    return [
        tile_record(bundle, tile_id, detector=detector)
        for tile_id, detector in zip(bundle["tile_ids"], bundle["detectors"])
    ]


def nearest_tile(bundle, x_mm, y_mm, detector=None):
    selected = np.ones(bundle["tile_ids"].size, dtype=bool)
    if detector is not None:
        selected &= bundle["detectors"] == str(detector).strip().upper()
    if not np.any(selected):
        raise KeyError(f"Detector {detector} is not present in the bundle")
    indices = np.flatnonzero(selected)
    distances = np.hypot(
        bundle["tile_center_x_mm"][indices] - float(x_mm),
        bundle["tile_center_y_mm"][indices] - float(y_mm),
    )
    index = int(indices[np.argmin(distances)])
    return tile_record(
        bundle,
        int(bundle["tile_ids"][index]),
        detector=str(bundle["detectors"][index]),
    )


def _query_grid(source, query, label):
    source = np.asarray(source, dtype=float)
    query = np.asarray(query, dtype=float).reshape(-1)
    if np.any(~np.isfinite(query)) or np.any(query < source[0]) or np.any(query > source[-1]):
        raise ValueError(f"{label} is outside the bundle grid")
    return query


def evaluate(bundle, tile_id, ice_thickness_nm, wavelengths_nm, detector=None):
    """Evaluate clean power and relative log response for one tile."""
    index = tile_index(bundle, tile_id, detector=detector)
    thickness = _query_grid(
        bundle["ice_thickness_nm"], ice_thickness_nm, "ice thickness"
    )
    wavelengths = _query_grid(bundle["wavelengths_nm"], wavelengths_nm, "wavelength")
    left = np.asarray(bundle["response_left"][index], dtype=float)
    right = np.asarray(bundle["response_right"][index], dtype=float)
    interpolated_left = np.stack(
        [np.interp(thickness, bundle["ice_thickness_nm"], left[:, rank]) for rank in range(left.shape[1])],
        axis=1,
    )
    interpolated_right = np.stack(
        [np.interp(wavelengths, bundle["wavelengths_nm"], right[rank]) for rank in range(right.shape[0])],
        axis=0,
    )
    clean_power = np.interp(
        wavelengths, bundle["wavelengths_nm"], bundle["clean_power"][index]
    )
    return clean_power, interpolated_left @ interpolated_right


def integrate(
    bundle, tile_id, ice_thickness_nm, wavelengths_nm, weights, detector=None
):
    """Integrate one tile against one or more wavelength weights."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.ndim != 2 or weights.shape[1] != wavelengths.size:
        raise ValueError("weights must have shape (filter_count, wavelength_count)")
    clean_power, relative = evaluate(
        bundle, tile_id, ice_thickness_nm, wavelengths, detector=detector
    )
    clean_integral = np.trapezoid(weights * clean_power[None, :], wavelengths, axis=1)
    coated_integral = np.trapezoid(
        weights[None, :, :] * clean_power[None, None, :] * np.exp(relative[:, None, :]),
        wavelengths,
        axis=2,
    )
    if np.any(clean_integral <= 0.0) or np.any(coated_integral <= 0.0):
        raise ValueError("filter weights must produce positive integrated power")
    return np.log(coated_integral / clean_integral[None, :])


def _write_tile_csv(path, records):
    if not records:
        raise ValueError("No tile records to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--list-tiles", action="store_true")
    parser.add_argument("--tile-map-csv", type=Path)
    parser.add_argument("--tile-id", type=int)
    parser.add_argument("--detector")
    parser.add_argument("--x-mm", type=float)
    parser.add_argument("--y-mm", type=float)
    parser.add_argument("--ice-thickness-nm", type=float, nargs="+")
    parser.add_argument("--wavelength-nm", type=float, nargs="+")
    parser.add_argument("--trajectory", action="store_true")
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    if args.summary:
        print(json.dumps(bundle_summary(bundle), indent=2, sort_keys=True))
    records = all_tile_records(bundle)
    if args.list_tiles:
        print(json.dumps(records, indent=2, sort_keys=True))
    if args.tile_map_csv is not None:
        _write_tile_csv(args.tile_map_csv, records)
    if (args.x_mm is None) != (args.y_mm is None):
        parser.error("--x-mm and --y-mm must be supplied together")
    tile_id = args.tile_id
    if args.x_mm is not None:
        tile = nearest_tile(bundle, args.x_mm, args.y_mm, detector=args.detector)
        tile_id = int(tile["tile_id"])
        args.detector = tile["detector"]
    if tile_id is not None:
        if args.trajectory:
            record = tile_trajectory(bundle, tile_id, detector=args.detector)
            print(
                json.dumps(
                    {
                        key: value.tolist()
                        if isinstance(value, np.ndarray)
                        else value
                        for key, value in record.items()
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.ice_thickness_nm is None or args.wavelength_nm is None:
            parser.error("tile queries require --ice-thickness-nm and --wavelength-nm")
        else:
            clean_power, relative = evaluate(
                bundle,
                tile_id,
                args.ice_thickness_nm,
                args.wavelength_nm,
                detector=args.detector,
            )
            print(
                json.dumps(
                    {
                        "tile": tile_record(bundle, tile_id, detector=args.detector),
                        "ice_thickness_nm": args.ice_thickness_nm,
                        "wavelength_nm": args.wavelength_nm,
                        "clean_power": clean_power.tolist(),
                        "relative_log_response": relative.tolist(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    if not (args.summary or args.list_tiles or args.tile_map_csv or tile_id is not None):
        print(json.dumps(bundle_summary(bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
