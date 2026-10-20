"""Pack all discovered detector caches into one NPZ file."""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from wfi_wavelength_response_bundle import (
    BUNDLE_VERSION,
    DETECTOR_PIXELS,
    MPA_SCA_WIDTH_MM,
    PIXEL_PITCH_MM,
    ACTIVE_REFERENCE_PIXELS,
    ACTIVE_PIXELS,
    WFI_SCA_LAYOUT,
    tile_geometry,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--fit-root",
        type=Path,
        required=False,
        help="Current relative-response fit root containing tile posterior archives.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", help="Restrict discovery to one detector.")
    parser.add_argument(
        "--all-detectors",
        action="store_true",
        help="Require WFI01 through WFI18 to be present below --cache-root.",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()
    if args.all_detectors and args.detector is not None:
        parser.error("--all-detectors cannot be combined with --detector")
    return args


def load_cache(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def cache_detector(cache_root):
    paths = sorted(Path(cache_root).glob("tile_*/spectral_response.npz"))
    if not paths:
        return None
    cache = load_cache(paths[0])
    return str(np.asarray(cache["detector"]).reshape(-1)[0]).upper()


def discover_cache_roots(root, detector=None, require_all=False):
    """Discover one direct cache root or all detector roots below a parent."""
    root = Path(root)
    direct_detector = cache_detector(root)
    if direct_detector is not None:
        candidates = [(direct_detector, root)]
    else:
        candidates = []
        for child in sorted(root.iterdir()):
            if child.is_dir():
                current_detector = cache_detector(child)
                if current_detector is not None:
                    candidates.append((current_detector, child))
    if detector is not None:
        requested = str(detector).strip().upper()
        candidates = [item for item in candidates if item[0] == requested]
    if not candidates:
        raise FileNotFoundError(
            f"No detector spectral caches found under {root}"
        )
    if len({item[0] for item in candidates}) != len(candidates):
        raise ValueError(
            "Multiple cache roots were discovered for one detector; "
            "use a parent containing one cache root per detector"
        )
    if require_all:
        expected = {f"WFI{index:02d}" for index in range(1, 19)}
        found = {item[0] for item in candidates}
        missing = sorted(expected - found)
        if missing:
            raise FileNotFoundError(
                "--all-detectors requested but caches are missing: "
                + ", ".join(missing)
            )
    return sorted(candidates)


def validate_tile_ids(detector, tile_ids, tile_size, require_complete=False):
    grid_size = int(np.ceil(DETECTOR_PIXELS / tile_size))
    expected = set(range(grid_size * grid_size))
    found = [int(tile_id) for tile_id in tile_ids]
    invalid = sorted(set(found) - expected)
    if invalid:
        raise ValueError(
            f"{detector} has tile IDs outside the {grid_size}x{grid_size} grid: "
            + ", ".join(str(tile_id) for tile_id in invalid)
        )
    if len(set(found)) != len(found):
        raise ValueError(f"{detector} has duplicate tile IDs")
    if require_complete and set(found) != expected:
        missing = sorted(expected - set(found))
        raise ValueError(
            f"{detector} is incomplete: expected {len(expected)} tiles, "
            f"found {len(found)}; missing "
            + ", ".join(str(tile_id) for tile_id in missing)
        )


def fit_root_for_detector(root, detector):
    root = Path(root)
    candidates = (
        root / detector.lower() / "relative_response" / "fits",
        root / detector.lower() / "fits",
        root / "relative_response" / "fits",
        root,
    )
    for candidate in candidates:
        if (candidate / "tile_000" / "posterior_samples.npz").is_file():
            return candidate
    raise FileNotFoundError(
        f"No relative-response fit archives found for {detector} below {root}"
    )


def string_array(values):
    values = [str(value) for value in values]
    return np.asarray(values, dtype=f"U{max(1, max(map(len, values)))}")


def load_observation_times(fit, days):
    statistics_path = Path(str(np.asarray(fit["statistics_path"]).reshape(-1)[0]))
    with statistics_path.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["group"]))
    if len(rows) != days.size:
        raise ValueError(f"Statistics row count differs from fit in {statistics_path}")
    statistics_days = np.asarray([float(row["days"]) for row in rows], dtype=float)
    if not np.allclose(statistics_days, days, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Statistics days differ from fit in {statistics_path}")
    return np.asarray([row["exposure_start_time"] for row in rows])


def load_fit_trajectory(path):
    fit = load_cache(path)
    days = np.asarray(fit["observation_days"], dtype=float).reshape(-1)
    thickness = np.asarray(fit["tile_ice_thickness"], dtype=float).reshape(-1)
    rates = np.asarray(
        fit["tile_instantaneous_rate_nm_day"], dtype=float
    ).reshape(-1)
    if thickness.shape != days.shape or rates.shape != days.shape:
        raise ValueError(f"Fit trajectory shape mismatch in {path}")
    grey_offset = float(
        np.asarray(fit["post_reset_tile_log_offset"]).reshape(-1)[0]
    )
    reset_day = float(np.asarray(fit["ice_reset_day"]).reshape(-1)[0])
    return (
        fit,
        days,
        thickness,
        rates,
        load_observation_times(fit, days),
        grey_offset,
        reset_day,
    )


def main():
    args = parse_args()
    if args.tile_size < 1:
        raise ValueError("tile-size must be positive")
    cache_roots = discover_cache_roots(
        args.cache_root,
        detector=args.detector,
        require_all=args.all_detectors,
    )
    records = []
    wavelengths = None
    thickness = None
    rank = None
    observation_days = None
    observation_start_time = None
    ice_reset_day = None
    observation_fields = {}
    fit_roots = {}
    for detector, cache_root in cache_roots:
        paths = sorted(Path(cache_root).glob("tile_*/spectral_response.npz"))
        if not paths:
            raise FileNotFoundError(f"No spectral caches found under {cache_root}")
        fit_root = None
        fit_by_tile = {}
        if args.fit_root is not None:
            fit_root = fit_root_for_detector(args.fit_root, detector)
            fit_roots[detector] = fit_root
            fit_paths = sorted(fit_root.glob("tile_*/posterior_samples.npz"))
            fit_by_tile = {
                int(path.parent.name.split("_")[-1]): path for path in fit_paths
            }
        detector_tile_ids = []
        for path in paths:
            cache = load_cache(path)
            tile_id = int(np.asarray(cache["tile_id"]).reshape(-1)[0])
            detector_tile_ids.append(tile_id)
            fit = None
            if args.fit_root is not None:
                fit_path = fit_by_tile.get(tile_id)
                if fit_path is None:
                    raise FileNotFoundError(
                        f"No fit archive for {detector} tile {tile_id}"
                    )
                (
                    fit,
                    days,
                    tile_ice,
                    tile_rate,
                    start_times,
                    grey_offset,
                    reset_day,
                ) = load_fit_trajectory(fit_path)
                fit_detector = str(np.asarray(fit["sca_names"]).reshape(-1)[0]).upper()
                fit_tile_id = int(np.asarray(fit["tile_ids"]).reshape(-1)[0])
                if (fit_detector, fit_tile_id) != (detector, tile_id):
                    raise ValueError(f"Fit identity differs from cache in {fit_path}")
                if observation_days is None:
                    observation_days = days
                    observation_start_time = start_times
                    ice_reset_day = reset_day
                    for name in (
                        "observation_led",
                        "observation_repeat",
                        "observation_program",
                        "groups",
                    ):
                        observation_fields[name] = np.asarray(fit[name])
                elif not np.array_equal(days, observation_days):
                    raise ValueError(f"Observation days differ in {fit_path}")
                elif not np.isclose(reset_day, ice_reset_day, rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"Ice reset day differs in {fit_path}")
                if not np.array_equal(start_times, observation_start_time):
                    raise ValueError(f"Observation start times differ in {fit_path}")
                for name, reference in observation_fields.items():
                    if not np.array_equal(np.asarray(fit[name]), reference):
                        raise ValueError(f"Observation field {name} differs in {fit_path}")
            current_detector = str(np.asarray(cache["detector"]).reshape(-1)[0]).upper()
            if current_detector != detector:
                raise ValueError(f"Cache detector differs from {detector} in {path}")
            current_wavelengths = np.asarray(cache["wavelengths_nm"], dtype=np.float32)
            current_thickness = np.asarray(cache["ice_thickness_nm"], dtype=np.float32)
            current_rank = int(np.asarray(cache["response_rank"]).reshape(-1)[0])
            if wavelengths is None:
                wavelengths = current_wavelengths
                thickness = current_thickness
                rank = current_rank
            elif not np.array_equal(current_wavelengths, wavelengths):
                raise ValueError(f"Wavelength grid differs in {path}")
            elif not np.array_equal(current_thickness, thickness):
                raise ValueError(f"Thickness grid differs in {path}")
            elif current_rank != rank:
                raise ValueError(f"Response rank differs in {path}")
            grid_size = int(np.ceil(DETECTOR_PIXELS / args.tile_size))
            tile_row, tile_col = divmod(tile_id, grid_size)
            row_start = tile_row * args.tile_size
            col_start = tile_col * args.tile_size
            row_stop = min(row_start + args.tile_size, DETECTOR_PIXELS)
            col_stop = min(col_start + args.tile_size, DETECTOR_PIXELS)
            geometry = tile_geometry(
                detector,
                tile_id,
                tile_row,
                tile_col,
                row_start,
                row_stop,
                col_start,
                col_stop,
            )
            record = {
                "detector": detector,
                "tile_id": tile_id,
                "geometry": geometry,
                "clean_power": np.asarray(cache["clean_power"], dtype=np.float32),
                "response_left": np.asarray(cache["response_left"], dtype=np.float32),
                "response_right": np.asarray(cache["response_right"], dtype=np.float32),
            }
            if fit is not None:
                record.update(
                    {
                        "tile_ice_thickness": tile_ice.astype(np.float32),
                        "tile_instantaneous_rate_nm_day": tile_rate.astype(np.float32),
                        "post_reset_tile_log_offset": np.float32(grey_offset),
                        "fit_path": str(Path(fit_path).resolve()),
                        "statistics_path": str(
                            np.asarray(fit["statistics_path"]).reshape(-1)[0]
                        ),
                        "fit_statistics_fingerprint": str(
                            np.asarray(fit["statistics_fingerprint"]).reshape(-1)[0]
                        ),
                    }
                )
            records.append(record)
        validate_tile_ids(
            detector,
            detector_tile_ids,
            args.tile_size,
            require_complete=args.all_detectors,
        )
    keys = [(record["detector"], record["tile_id"]) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate detector/tile identifiers found")
    records.sort(key=lambda record: (record["detector"], record["tile_id"]))
    tile_ids = np.asarray([record["tile_id"] for record in records], dtype=np.int32)
    tile_rows = np.asarray(
        [record["geometry"]["tile_row"] for record in records], dtype=np.int16
    )
    tile_cols = np.asarray(
        [record["geometry"]["tile_col"] for record in records], dtype=np.int16
    )
    detectors = np.asarray([record["detector"] for record in records], dtype="U5")
    sca_numbers = np.asarray(
        [record["geometry"]["sca_number"] for record in records], dtype=np.int16
    )
    geometries = [record["geometry"] for record in records]
    clean_power = np.asarray(
        [record["clean_power"] for record in records], dtype=np.float32
    )
    response_left = np.asarray(
        [record["response_left"] for record in records], dtype=np.float32
    )
    response_right = np.asarray(
        [record["response_right"] for record in records], dtype=np.float32
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "bundle_format": "NPZ low-rank spectral response bundle",
        "bundle_version": BUNDLE_VERSION,
        "detectors": sorted(set(detectors.tolist())),
        "detector_count": len(set(detectors.tolist())),
        "source_cache_roots": {
            detector: str(cache_root) for detector, cache_root in cache_roots
        },
        "source_cache_count": int(tile_ids.size),
        "tile_size_pixels": int(args.tile_size),
        "detector_size_pixels": DETECTOR_PIXELS,
        "wavelength_grid_definition": "shared wavelength grid in nm",
        "thickness_grid_definition": "shared ice-thickness grid in nm",
        "response_definition": "log(coated_power / clean_power) before filter/SED/QE weighting",
        "response_representation": "response_left @ response_right with linear interpolation of factors",
        "coordinate_source": "MPA_SCA_info ECSV supplied by project; nominal geometry",
        "coordinate_convention": "MPA focal-plane x points left; science-frame row increases up and column increases right",
        "sca_width_mm": MPA_SCA_WIDTH_MM,
        "pixel_pitch_mm": PIXEL_PITCH_MM,
        "active_reference_pixels_each_edge": ACTIVE_REFERENCE_PIXELS,
        "active_pixels_per_axis": ACTIVE_PIXELS,
        "sca_layout": {str(number): list(values) for number, values in WFI_SCA_LAYOUT.items()},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    trajectory_payload = {}
    if observation_days is not None:
        first_fit = load_cache(records[0]["fit_path"])
        metadata.update(
            {
                "trajectory_included": True,
                "trajectory_definition": "MAP tile_ice_thickness from relative_response_production posterior_samples.npz",
                "trajectory_rate_definition": "MAP tile_instantaneous_rate_nm_day",
                "trajectory_uncertainties": "not included; this is a deterministic MAP export",
                "grey_throughput_included": True,
                "grey_throughput_definition": "post-reset multiplicative tile throughput factor exp(post_reset_tile_log_offset)",
                "grey_throughput_application": "add post_reset_tile_log_offset to log response for observation_days >= ice_reset_day",
                "trajectory_fit_roots": {
                    detector: str(fit_root.resolve())
                    for detector, fit_root in fit_roots.items()
                },
                "time_definition": "observation_days from the corrected tile statistics",
                "utc_time_definition": "exposure_start_time from the corrected tile statistics",
                "fit_settings": {
                    name: np.asarray(first_fit[name]).reshape(-1)[0].item()
                    for name in (
                        "fit_mode",
                        "ice_reset_day",
                        "trajectory_rate_prior_mean_nm_day",
                        "trajectory_rate_prior_sigma_log",
                        "trajectory_entropy_weight",
                        "trajectory_regularization",
                        "map_optimizer",
                        "map_steps",
                        "map_parameter_scaling",
                    )
                    if name in first_fit
                },
            }
        )
        trajectory_payload = {
            "observation_days": np.asarray(observation_days, dtype=np.float64),
            "observation_start_time": string_array(observation_start_time),
            "observation_led": np.asarray(
                observation_fields["observation_led"], dtype=np.int16
            ),
            "observation_repeat": np.asarray(
                observation_fields["observation_repeat"], dtype=np.int16
            ),
            "observation_program": np.asarray(
                observation_fields["observation_program"], dtype=np.int16
            ),
            "groups": np.asarray(observation_fields["groups"], dtype=np.int32),
            "tile_ice_thickness": np.asarray(
                [record["tile_ice_thickness"] for record in records],
                dtype=np.float32,
            ),
            "tile_instantaneous_rate_nm_day": np.asarray(
                [record["tile_instantaneous_rate_nm_day"] for record in records],
                dtype=np.float32,
            ),
            "post_reset_tile_log_offset": np.asarray(
                [record["post_reset_tile_log_offset"] for record in records],
                dtype=np.float32,
            ),
            "post_reset_tile_throughput_factor": np.exp(
                np.asarray(
                    [record["post_reset_tile_log_offset"] for record in records],
                    dtype=np.float64,
                )
            ).astype(np.float32),
            "ice_reset_day": np.asarray(ice_reset_day, dtype=np.float64),
            "fit_paths": string_array([record["fit_path"] for record in records]),
            "statistics_paths": string_array(
                [record["statistics_path"] for record in records]
            ),
            "fit_statistics_fingerprints": string_array(
                [record["fit_statistics_fingerprint"] for record in records]
            ),
        }
    else:
        metadata["trajectory_included"] = False
    np.savez_compressed(
        args.output,
        bundle_version=np.asarray(BUNDLE_VERSION, dtype=np.int16),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        detectors=detectors,
        sca_numbers=sca_numbers,
        response_rank=np.asarray(rank, dtype=np.int16),
        wavelengths_nm=wavelengths,
        ice_thickness_nm=thickness,
        tile_ids=tile_ids,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        row_starts=np.asarray([item["row_start"] for item in geometries], dtype=np.int32),
        row_stops_exclusive=np.asarray(
            [item["row_stop_exclusive"] for item in geometries], dtype=np.int32
        ),
        col_starts=np.asarray([item["col_start"] for item in geometries], dtype=np.int32),
        col_stops_exclusive=np.asarray(
            [item["col_stop_exclusive"] for item in geometries], dtype=np.int32
        ),
        tile_n_rows=np.asarray(
            [item["tile_n_rows"] for item in geometries], dtype=np.int32
        ),
        tile_n_cols=np.asarray(
            [item["tile_n_cols"] for item in geometries], dtype=np.int32
        ),
        pixel_center_rows=np.asarray(
            [item["pixel_center_row"] for item in geometries], dtype=np.float32
        ),
        pixel_center_cols=np.asarray(
            [item["pixel_center_col"] for item in geometries], dtype=np.float32
        ),
        active_pixel_center_rows=np.asarray(
            [item["active_pixel_center_row"] for item in geometries], dtype=np.float32
        ),
        active_pixel_center_cols=np.asarray(
            [item["active_pixel_center_col"] for item in geometries], dtype=np.float32
        ),
        sca_center_x_mm=np.asarray(
            [item["sca_center_x_mm"] for item in geometries], dtype=np.float64
        ),
        sca_center_y_mm=np.asarray(
            [item["sca_center_y_mm"] for item in geometries], dtype=np.float64
        ),
        sca_rotation_deg=np.asarray(
            [item["sca_rotation_deg"] for item in geometries], dtype=np.float64
        ),
        tile_local_x_mm=np.asarray(
            [item["tile_local_x_mm"] for item in geometries], dtype=np.float64
        ),
        tile_local_y_mm=np.asarray(
            [item["tile_local_y_mm"] for item in geometries], dtype=np.float64
        ),
        tile_center_x_mm=np.asarray(
            [item["tile_center_x_mm"] for item in geometries], dtype=np.float64
        ),
        tile_center_y_mm=np.asarray(
            [item["tile_center_y_mm"] for item in geometries], dtype=np.float64
        ),
        clean_power=clean_power,
        response_left=response_left,
        response_right=response_right,
        **trajectory_payload,
    )
    print(
        f"Wrote {args.output} with {tile_ids.size} tiles across "
        f"{len(set(detectors.tolist()))} detectors"
    )


if __name__ == "__main__":
    main()
