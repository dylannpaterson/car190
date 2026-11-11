"""Bundle the production WFI fits into one self-describing FITS product."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import subprocess

from astropy.io import fits
from astropy.time import Time
import numpy as np

from fit_wfi_tmm_production import (
    HGCDTE_N_TABLE_WAVELENGTHS_NM,
    HGCDTE_REFERENCE_COMPOSITION,
    HGCDTE_REFERENCE_TEMPERATURE_K,
    hgcdte_reference_optical_constants,
)
from plot_wfi_full_focal_plane_rates import (
    COLUMN_SCA,
    COLUMN_Y_OFFSET,
    SCAS,
    SCA_PITCH,
    SCA_SIZE,
    sca_position,
)
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    corrected_statistics_path,
    detector_output_root,
    positive_integer,
    production_output_root,
    tile_directory_entries,
)


ROOT = Path(__file__).resolve().parent
LEGACY_MODEL_NAME = (
    "absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares"
)
# Current production decon2 boundary used by the independent MAP fits.
RESET_DAY = 12.907397928240743
AR_LAYERS = 3
OBSERVATION_FIELDS = (
    "group",
    "led",
    "repeat",
    "file_set_name",
    "baseline_file_set_name",
    "exposure_start_time",
    "days",
)
POSITION_FIELDS = (
    "tile_row",
    "tile_col",
    "row_start",
    "row_stop_exclusive",
    "col_start",
    "col_stop_exclusive",
    "tile_n_rows",
    "tile_n_cols",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def statistics_fingerprint(rows):
    rows = sorted(rows, key=lambda row: int(row["group"]))
    digest = hashlib.sha256()
    for field in ("n_pair_valid", "sum_log_change", "sum_squared_log_change"):
        digest.update(
            np.asarray([float(row[field]) for row in rows], dtype=np.float64).tobytes()
        )
    return digest.hexdigest()


def canonical_utc(value):
    value = str(value).strip()
    if not value:
        raise ValueError("Exposure timestamp is empty")
    return value if value.endswith("Z") else f"{value}Z"


def timestamp_mjd_utc(value):
    return float(Time(canonical_utc(value)[:-1], format="isot", scale="utc").mjd)


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def detector_mosaic_position(detector):
    for column, detectors in enumerate(COLUMN_SCA):
        if detector in detectors:
            row = len(detectors) - 1 - detectors.index(detector)
            x, y = sca_position(detector)
            return column, row, x, y
    raise ValueError(f"Unknown detector: {detector}")


def load_statistics(path, detector):
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No statistics rows found in {path}")

    rows_by_tile = {}
    for row in rows:
        rows_by_tile.setdefault(int(row["tile_id"]), []).append(row)
    tile_ids = complete_tile_ids(
        rows_by_tile,
        label=f"{detector} statistics tile IDs",
    )

    sorted_rows = {}
    for tile_id, tile_rows in rows_by_tile.items():
        sorted_rows[tile_id] = sorted(tile_rows, key=lambda row: int(row["group"]))
    reference_rows = sorted_rows[0]
    reference_keys = [
        _observation_key(row, detector) for row in reference_rows
    ]
    for tile_id, tile_rows in sorted_rows.items():
        keys = [_observation_key(row, detector) for row in tile_rows]
        if keys != reference_keys:
            raise ValueError(f"Observation metadata differs for {detector} tile {tile_id}")

    observations = []
    for observation_index, row in enumerate(reference_rows):
        observations.append(
            {
                "observation_index": observation_index,
                "group": int(row["group"]),
                "led": int(row["led"]),
                "repeat": int(row["repeat"]),
                "time_days_since_first_exposure": float(row["days"]),
                "time_mjd_utc": timestamp_mjd_utc(row["exposure_start_time"]),
                "exposure_start_utc": canonical_utc(row["exposure_start_time"]),
                "file_set_name": row["file_set_name"],
                "baseline_file_set_name": row["baseline_file_set_name"],
            }
        )
    first_mjd = observations[0]["time_mjd_utc"]
    for observation in observations:
        observation["time_seconds_since_first_exposure"] = (
            observation["time_days_since_first_exposure"] * 86400.0
        )
        observation["post_reset"] = (
            observation["time_days_since_first_exposure"] >= RESET_DAY
        )

    tile_metadata = {}
    for tile_id, tile_rows in sorted_rows.items():
        first = tile_rows[0]
        metadata = {field: int(first[field]) for field in POSITION_FIELDS}
        for row in tile_rows[1:]:
            if any(int(row[field]) != metadata[field] for field in POSITION_FIELDS):
                raise ValueError(f"Tile position changes across observations for {detector} tile {tile_id}")
        tile_metadata[tile_id] = metadata

    repeat_values = sorted({observation["repeat"] for observation in observations})
    node_days = np.asarray(
        [
            np.mean(
                [
                    observation["time_days_since_first_exposure"]
                    for observation in observations
                    if observation["repeat"] == repeat
                ]
            )
            for repeat in repeat_values
        ],
        dtype=float,
    )
    if node_days.size < 3 or np.any(np.diff(node_days) <= 0.0):
        raise ValueError(f"{detector} repeat nodes are not strictly increasing")

    return {
        "path": path,
        "sha256": sha256_file(path),
        "rows_by_tile": sorted_rows,
        "tile_ids": tile_ids,
        "tile_grid_size": math.isqrt(len(tile_ids)),
        "observations": observations,
        "tile_metadata": tile_metadata,
        "node_days": node_days,
        "first_mjd": first_mjd,
    }


def _observation_key(row, detector):
    return (
        int(row["group"]),
        int(row["led"]),
        int(row["repeat"]),
        row["file_set_name"],
        row["baseline_file_set_name"],
        canonical_utc(row["exposure_start_time"]),
        float(row["days"]),
        detector,
    )


def find_fit_archives(
    fit_root,
    detector,
    tile_size=DEFAULT_TILE_SIZE,
    expected_tile_ids=None,
):
    fit_root = Path(fit_root)
    candidates = (
        (
            "production",
            detector_output_root(fit_root, detector, tile_size) / "fits",
        ),
        (
            "direct_production",
            fit_root / detector.lower() / "fits",
        ),
        (
            "relative_response_production",
            detector_output_root(fit_root, detector, tile_size)
            / "relative_response"
            / "fits",
        ),
        (
            "direct_relative_response",
            fit_root / detector.lower() / "relative_response" / "fits",
        ),
        (
            "legacy_independent_production",
            fit_root / f"{detector.lower()}_{LEGACY_MODEL_NAME}" / "fits",
        ),
    )
    partial = []
    for layout, candidate in candidates:
        found = {}
        for tile_id, path in tile_directory_entries(candidate):
            posterior = path / "posterior_samples.npz"
            if posterior.is_file():
                found[tile_id] = path
        if not found:
            continue
        expected = None if expected_tile_ids is None else set(expected_tile_ids)
        if expected is None:
            try:
                complete_tile_ids(found, label=f"tile IDs in {candidate}")
            except ValueError:
                partial.append(f"{candidate} (incomplete tile IDs)")
                continue
        if expected is None or set(found) == expected:
            return found, layout
        partial.append(f"{candidate} ({len(found)}/{len(expected)} tiles)")
    details = "; ".join(partial) if partial else "no fit archives found"
    raise FileNotFoundError(f"Complete production fits unavailable for {detector}: {details}")


def _scalar(samples, names, default=np.nan):
    if isinstance(names, str):
        names = (names,)
    for name in names:
        if name in samples:
            value = np.asarray(samples[name]).reshape(-1)
            if value.size:
                return value[0].item()
    return default


def _string_scalar(samples, names, default=""):
    value = _scalar(samples, names, default)
    return str(value)


def _series(samples, names, count, required=True):
    if isinstance(names, str):
        names = (names,)
    for name in names:
        if name not in samples:
            continue
        value = np.asarray(samples[name], dtype=float).reshape(-1)
        if value.size == count:
            return value
    if required:
        raise ValueError(f"None of {names} contains {count} values")
    return np.full(count, np.nan)


def response_wavelengths(samples, path):
    if "response_wavelengths_nm" not in samples:
        raise ValueError(f"Missing response wavelength grid in {path}")
    return validate_response_wavelengths(samples["response_wavelengths_nm"], path)


def validate_response_wavelengths(values, path):
    wavelengths = np.asarray(values, dtype=float).reshape(-1)
    if (
        wavelengths.size < 2
        or np.any(~np.isfinite(wavelengths))
        or np.any(wavelengths <= 0.0)
        or np.any(np.diff(wavelengths) <= 0.0)
    ):
        raise ValueError(f"Invalid response wavelength grid in {path}")
    return wavelengths.copy()


def load_response_grid_cache(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        if "wavelengths_nm" not in cache:
            raise ValueError(f"Missing wavelengths_nm in response grid cache {path}")
        return validate_response_wavelengths(cache["wavelengths_nm"], path)


def load_model_predictions(path, source_rows, count):
    prediction_path = Path(path) / "tile_group_predictions.csv"
    if not prediction_path.is_file():
        raise ValueError(
            f"Missing _map_least_squares_prediction and tile_group_predictions.csv in {path}"
        )
    with prediction_path.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["group"]))
    expected_groups = [int(row["group"]) for row in source_rows]
    actual_groups = [int(row["group"]) for row in rows]
    if len(rows) != count or actual_groups != expected_groups:
        raise ValueError(f"Prediction groups differ from statistics in {prediction_path}")
    return np.asarray([float(row["model_log_change"]) for row in rows], dtype=float)


def reconstruct_hgcdte(reference_n, reference_k, n_log_scale, k_log_scale):
    reference_n = np.asarray(reference_n, dtype=float)
    reference_k = np.asarray(reference_k, dtype=float)
    if reference_n.shape != reference_k.shape:
        raise ValueError("HgCdTe reference n/k grids must have the same shape")
    physical_n = reference_n * np.exp(float(n_log_scale))
    physical_k = reference_k * np.exp(float(k_log_scale))
    if np.any(~np.isfinite(physical_n)) or np.any(~np.isfinite(physical_k)):
        raise ValueError("Reconstructed HgCdTe values must be finite")
    return physical_n, physical_k


def read_fit_archive(path, statistics, tile_id, fit_root, response_grid=None):
    source_rows = statistics["rows_by_tile"][tile_id]
    source_fingerprint = statistics_fingerprint(source_rows)
    with np.load(path / "posterior_samples.npz", allow_pickle=False) as samples:
        archive_fingerprint = _string_scalar(samples, "statistics_fingerprint")
        if archive_fingerprint != source_fingerprint:
            raise ValueError(
                f"Statistics fingerprint mismatch for {path}: "
                f"archive={archive_fingerprint}, source={source_fingerprint}"
            )
        if "response_wavelengths_nm" in samples:
            fit_wavelengths = response_wavelengths(samples, path)
        elif response_grid is not None:
            fit_wavelengths = validate_response_wavelengths(response_grid, path)
        else:
            raise ValueError(
                f"Missing response wavelength grid in {path}; pass --response-grid-cache"
            )
        observation_count = len(statistics["observations"])
        thickness = _series(
            samples,
            ("tile_ice_thickness", "ice_thickness"),
            observation_count,
        )
        rates = _series(
            samples,
            ("tile_instantaneous_rate_nm_day", "instantaneous_rate_nm_day"),
            observation_count,
        )
        if "_map_least_squares_prediction" in samples:
            predictions = _series(
                samples, "_map_least_squares_prediction", observation_count
            )
        else:
            predictions = load_model_predictions(
                path, source_rows, observation_count
            )
        node_count = statistics["node_days"].size
        node_thickness = _series(
            samples,
            ("np_node_thickness", "tile_node_thickness", "node_ice_thickness"),
            node_count,
        )
        node_rates = _series(
            samples,
            (
                "node_instantaneous_rate_nm_day",
                "tile_node_instantaneous_rate_nm_day",
                "node_instantaneous_rate_nm_day",
            ),
            node_count,
        )
        node_multipliers = _series(
            samples,
            ("np_node_multipliers", "np_log_node_multipliers"),
            node_count,
            required=False,
        )
        ar_parameters = np.asarray(samples["ar_parameters"], dtype=float).reshape(-1)
        if ar_parameters.size != AR_LAYERS * 3:
            raise ValueError(f"Unexpected AR parameter count in {path}")
        initial_log = float(_scalar(samples, "initial_thickness_log_nm"))
        reset_log = float(_scalar(samples, "reset_initial_thickness_log_nm"))
        third_field_names = (
            "third_decon_day",
            "third_decon_initial_thickness_log_nm",
            "post_third_decon_tile_log_offset",
        )
        third_fields_present = [name in samples for name in third_field_names]
        if any(third_fields_present) and not all(third_fields_present):
            raise ValueError(f"Incomplete third-decon fields in {path}")
        third_day = float(_scalar(samples, "third_decon_day", np.nan))
        third_log = float(
            _scalar(samples, "third_decon_initial_thickness_log_nm", np.nan)
        )
        third_offset = float(
            _scalar(samples, "post_third_decon_tile_log_offset", np.nan)
        )
        third_included = np.isfinite(third_day)
        if third_included and (
            not np.isfinite(third_log)
            or not np.isfinite(third_offset)
        ):
            raise ValueError(f"Incomplete third-decon fields in {path}")
        map_history = np.asarray(samples["map_objective_history"], dtype=float).reshape(-1)
        initial_rate = float(
            _scalar(samples, ("tile_initial_rate_nm_day", "initial_rate_nm_day"))
        )
        final_rate = float(
            _scalar(samples, ("tile_final_rate_nm_day", "final_rate_nm_day"))
        )
        mean_rate = float(
            _scalar(samples, ("tile_mean_rate_nm_day", "mean_rate_nm_day"))
        )
        if not np.isfinite(initial_rate):
            initial_rate = float(rates[0])
        if not np.isfinite(final_rate):
            final_rate = float(rates[-1])
        if not np.isfinite(mean_rate):
            mean_rate = float(np.mean(rates))
        fit_values = {
            "ar_parameters": ar_parameters,
            "initial_thickness_nm": np.exp(initial_log),
            "reset_initial_thickness_nm": np.exp(reset_log),
            "third_decon_initial_thickness_nm": (
                np.exp(third_log) if third_included else np.nan
            ),
            "incidence_angle_deg": float(_scalar(samples, "incidence_angle_deg")),
            "post_reset_tile_log_offset": float(
                _scalar(samples, "post_reset_tile_log_offset")
            ),
            "post_third_decon_tile_log_offset": third_offset,
            "absolute_hgcdte_n_log_scale": float(
                _scalar(samples, "absolute_hgcdte_n_log_scale")
            ),
            "absolute_hgcdte_k_log_scale": float(
                _scalar(samples, "absolute_hgcdte_k_log_scale")
            ),
            "initial_rate_nm_day": initial_rate,
            "final_rate_nm_day": final_rate,
            "mean_rate_nm_day": mean_rate,
            "trajectory_rate_prior_mean_nm_day": float(
                _scalar(samples, "trajectory_rate_prior_mean_nm_day")
            ),
            "trajectory_rate_prior_log_sigma": float(
                _scalar(
                    samples,
                    ("trajectory_rate_prior_sigma_log", "trajectory_rate_prior_log_sigma"),
                )
            ),
            "map_objective": float(map_history[-1]) if map_history.size else np.nan,
            "map_steps": int(_scalar(samples, "map_steps", 10000)),
            "fit_incidence_angle": bool(_scalar(samples, "fit_incidence_angle", 1)),
            "ice_reset_day": float(_scalar(samples, "ice_reset_day", RESET_DAY)),
            "third_decon_day": third_day,
            "third_decon_included": third_included,
            "map_optimizer": _string_scalar(samples, "map_optimizer", "least-squares"),
            "thickness_nm": thickness,
            "rate_nm_day": rates,
            "model_log_change": predictions,
            "node_thickness_nm": node_thickness,
            "node_rate_nm_day": node_rates,
            "node_multiplier": node_multipliers,
            "source_fingerprint": source_fingerprint,
            "archive_fingerprint": archive_fingerprint,
            "archive_sha256": sha256_file(path / "posterior_samples.npz"),
            "archive_relative_path": str(
                (path / "posterior_samples.npz").relative_to(fit_root)
            ),
            "response_wavelengths_nm": fit_wavelengths,
            "fit_mode": _string_scalar(samples, "fit_mode", "unknown"),
            "response_cache": _string_scalar(samples, "response_cache", ""),
            "thickness_prior_log_sigma": float(
                _scalar(samples, "thickness_prior_log_sigma", np.nan)
            ),
            "thickness_prior_stages": _string_scalar(
                samples, "thickness_prior_stages", ""
            ),
        }
    return fit_values


def _string_array(values, width):
    return np.asarray([str(value)[:width] for value in values], dtype=f"S{width}")


def make_table_hdu(name, specs, rows):
    columns = []
    for column_name, fmt, unit, _description in specs:
        values = [row[column_name] for row in rows]
        if fmt.startswith("A"):
            array = _string_array(values, int(fmt[1:]))
        elif fmt == "L":
            array = np.asarray(values, dtype=bool)
        elif fmt == "I":
            array = np.asarray(values, dtype=np.int16)
        elif fmt == "J":
            array = np.asarray(values, dtype=np.int32)
        elif fmt == "K":
            array = np.asarray(values, dtype=np.int64)
        elif fmt == "D":
            array = np.asarray(values, dtype=np.float64)
        elif fmt.endswith("D") and fmt[:-1].isdigit():
            vector_length = int(fmt[:-1])
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (len(values), vector_length):
                raise ValueError(
                    f"Expected {column_name} to have shape "
                    f"({len(values)}, {vector_length}), got {array.shape}"
                )
        else:
            raise ValueError(f"Unsupported FITS format {fmt}")
        columns.append(
            fits.Column(
                name=column_name,
                format=fmt,
                unit=unit or None,
                array=array,
            )
        )
    hdu = fits.BinTableHDU.from_columns(columns, name=name)
    return hdu


def schema_hdu(schemas):
    rows = []
    for extension, specs in schemas.items():
        for column_name, fmt, unit, description in specs:
            rows.append(
                {
                    "extension": extension,
                    "column": column_name,
                    "format": fmt,
                    "unit": unit,
                    "description": description,
                }
            )
    specs = (
        ("extension", "A24", "", "FITS extension containing the column"),
        ("column", "A40", "", "FITS column name"),
        ("format", "A8", "", "FITS binary-table format"),
        ("unit", "A32", "", "Physical unit, if applicable"),
        ("description", "A180", "", "Unambiguous column definition"),
    )
    return make_table_hdu("SCHEMA", specs, rows)


def model_hdu(model_rows):
    specs = (
        ("key", "A64", "", "Metadata key"),
        ("value", "A256", "", "Metadata value"),
        ("unit", "A24", "", "Physical unit, if applicable"),
        ("description", "A180", "", "Meaning of the metadata value"),
    )
    return make_table_hdu("MODEL", specs, model_rows)


def readme_hdu(lines):
    specs = (("text", "A200", "", "Human-readable bundle guidance"),)
    return make_table_hdu("README", specs, [{"text": line} for line in lines])


def build_bundle(
    statistics_dir,
    fit_root,
    tile_size=DEFAULT_TILE_SIZE,
    response_grid=None,
    response_grid_source=None,
):
    statistics_dir = Path(statistics_dir)
    fit_root = Path(fit_root).resolve()
    if response_grid is not None:
        response_grid = validate_response_wavelengths(
            response_grid, response_grid_source or "response_grid"
        )
    detector_data = {}
    fit_data = {}
    fit_layouts = {}
    for detector in SCAS:
        stats_path = corrected_statistics_path(statistics_dir, detector, tile_size)
        statistics = load_statistics(stats_path, detector)
        archives, layout = find_fit_archives(
            fit_root,
            detector,
            tile_size=tile_size,
            expected_tile_ids=statistics["tile_ids"],
        )
        detector_data[detector] = statistics
        fit_layouts[detector] = layout
        fit_data[detector] = {
            tile_id: read_fit_archive(
                path,
                statistics,
                tile_id,
                fit_root,
                response_grid=response_grid,
            )
            for tile_id, path in sorted(archives.items())
        }

    response_grid = None
    for detector in SCAS:
        for fit in fit_data[detector].values():
            wavelengths = fit["response_wavelengths_nm"]
            if response_grid is None:
                response_grid = wavelengths
            elif not np.allclose(response_grid, wavelengths, rtol=0.0, atol=1.0e-10):
                raise ValueError("Production fit archives use different response wavelength grids")
    reference_n, reference_k = hgcdte_reference_optical_constants(response_grid)
    reference_rows = [
        {
            "wavelength_nm": wavelength,
            "reference_n": real_index,
            "reference_k": extinction_index,
        }
        for wavelength, real_index, extinction_index in zip(
            response_grid, reference_n, reference_k
        )
    ]

    detector_rows = []
    tile_rows = []
    observation_rows = []
    timeseries_rows = []
    node_rows = []
    parameter_rows = []
    provenance_rows = []
    for sca_index, detector in enumerate(SCAS):
        statistics = detector_data[detector]
        detector_fits = fit_data[detector]
        detector_third_days = {
            float(fit["third_decon_day"])
            for fit in detector_fits.values()
            if fit["third_decon_included"]
        }
        if len(detector_third_days) > 1:
            raise ValueError(f"Third-decon day differs across {detector} fit archives")
        detector_third_day = next(iter(detector_third_days), None)
        if detector_third_day is not None and not all(
            fit["third_decon_included"] for fit in detector_fits.values()
        ):
            raise ValueError(f"Third-decon metadata is incomplete for {detector}")
        column, row, mosaic_x, mosaic_y = detector_mosaic_position(detector)
        detector_rows.append(
            {
                "detector": detector,
                "sca_index": sca_index,
                "mosaic_column": column,
                "mosaic_row": row,
                "mosaic_x_pitch_units": mosaic_x,
                "mosaic_y_pitch_units": mosaic_y,
                "first_exposure_mjd_utc": statistics["first_mjd"],
                "observation_count": len(statistics["observations"]),
                "tile_count": len(statistics["tile_ids"]),
            }
        )
        for observation in statistics["observations"]:
            observation_rows.append(
                {
                    "detector": detector,
                    "sca_index": sca_index,
                    "post_third_decon": (
                        detector_third_day is not None
                        and observation["time_days_since_first_exposure"]
                        >= detector_third_day
                    ),
                    **observation,
                }
            )
        provenance_rows.append(
            {
                "detector": detector,
                "statistics_file": statistics["path"].name,
                "statistics_sha256": statistics["sha256"],
                "fit_layout": fit_layouts[detector],
                "fit_root": str(fit_root),
                "fit_archive_count": len(fit_data[detector]),
                "observation_count": len(statistics["observations"]),
            }
        )
        for tile_id in statistics["tile_ids"]:
            metadata = statistics["tile_metadata"][tile_id]
            fit = fit_data[detector][tile_id]
            tile_rows.append(
                {
                    "detector": detector,
                    "sca_index": sca_index,
                    "tile_id": tile_id,
                    **metadata,
                    "pixel_center_row": 0.5
                    * (metadata["row_start"] + metadata["row_stop_exclusive"]),
                    "pixel_center_col": 0.5
                    * (metadata["col_start"] + metadata["col_stop_exclusive"]),
                    "fit_status": "MAP",
                    "statistics_fingerprint": fit["source_fingerprint"],
                    "fit_archive_sha256": fit["archive_sha256"],
                    "fit_archive": fit["archive_relative_path"],
                }
            )
            parameters = {
                "detector": detector,
                "sca_index": sca_index,
                "tile_id": tile_id,
                "tile_row": metadata["tile_row"],
                "tile_col": metadata["tile_col"],
                "incidence_angle_deg": fit["incidence_angle_deg"],
                "initial_thickness_nm": fit["initial_thickness_nm"],
                "reset_initial_thickness_nm": fit["reset_initial_thickness_nm"],
                "post_reset_tile_log_offset": fit["post_reset_tile_log_offset"],
                "third_decon_initial_thickness_nm": fit[
                    "third_decon_initial_thickness_nm"
                ],
                "post_third_decon_tile_log_offset": fit[
                    "post_third_decon_tile_log_offset"
                ],
                "absolute_hgcdte_n_log_scale": fit["absolute_hgcdte_n_log_scale"],
                "absolute_hgcdte_k_log_scale": fit["absolute_hgcdte_k_log_scale"],
                "absolute_hgcdte_n": None,
                "absolute_hgcdte_k": None,
                "initial_rate_nm_day": fit["initial_rate_nm_day"],
                "final_rate_nm_day": fit["final_rate_nm_day"],
                "mean_rate_nm_day": fit["mean_rate_nm_day"],
                "map_objective": fit["map_objective"],
                "map_steps": fit["map_steps"],
                "fit_incidence_angle": fit["fit_incidence_angle"],
                "ice_reset_day": fit["ice_reset_day"],
                "third_decon_day": fit["third_decon_day"],
                "map_optimizer": fit["map_optimizer"],
            }
            physical_n, physical_k = reconstruct_hgcdte(
                reference_n,
                reference_k,
                fit["absolute_hgcdte_n_log_scale"],
                fit["absolute_hgcdte_k_log_scale"],
            )
            parameters["absolute_hgcdte_n"] = physical_n
            parameters["absolute_hgcdte_k"] = physical_k
            for layer in range(AR_LAYERS):
                ar_thickness_scale = fit["ar_parameters"][3 * layer]
                parameters[f"ar_layer{layer + 1}_thickness_scale_100nm"] = ar_thickness_scale
                parameters[f"ar_layer{layer + 1}_thickness_nm"] = 100.0 * ar_thickness_scale
                parameters[f"ar_layer{layer + 1}_n"] = fit["ar_parameters"][3 * layer + 1]
                parameters[f"ar_layer{layer + 1}_k"] = fit["ar_parameters"][3 * layer + 2]
            parameter_rows.append(parameters)

            source_rows = statistics["rows_by_tile"][tile_id]
            for observation_index, (observation, source_row) in enumerate(
                zip(statistics["observations"], source_rows)
            ):
                observed = float(source_row["sum_log_change"]) / float(
                    source_row["n_pair_valid"]
                )
                model_value = fit["model_log_change"][observation_index]
                timeseries_rows.append(
                    {
                        "detector": detector,
                        "sca_index": sca_index,
                        "tile_id": tile_id,
                        "tile_row": metadata["tile_row"],
                        "tile_col": metadata["tile_col"],
                        "observation_index": observation_index,
                        "group": observation["group"],
                        "led": observation["led"],
                        "repeat": observation["repeat"],
                        "exposure_start_utc": observation["exposure_start_utc"],
                        "time_mjd_utc": observation["time_mjd_utc"],
                        "time_days_since_first_exposure": observation[
                            "time_days_since_first_exposure"
                        ],
                        "n_pair_valid": int(source_row["n_pair_valid"]),
                        "sum_log_change": float(source_row["sum_log_change"]),
                        "sum_squared_log_change": float(
                            source_row["sum_squared_log_change"]
                        ),
                        "observed_log_change": observed,
                        "model_log_change": model_value,
                        "residual_log": observed - model_value,
                        "ice_thickness_nm": fit["thickness_nm"][observation_index],
                        "instantaneous_rate_nm_day": fit["rate_nm_day"][observation_index],
                        "post_reset": observation["post_reset"],
                        "post_third_decon": (
                            fit["third_decon_included"]
                            and observation["time_days_since_first_exposure"]
                            >= fit["third_decon_day"]
                        ),
                    }
                )
            for node_index, node_day in enumerate(statistics["node_days"]):
                node_rows.append(
                    {
                        "detector": detector,
                        "sca_index": sca_index,
                        "tile_id": tile_id,
                        "tile_row": metadata["tile_row"],
                        "tile_col": metadata["tile_col"],
                        "node_index": node_index,
                        "node_day_since_first_exposure": node_day,
                        "node_mjd_utc": statistics["first_mjd"] + node_day,
                        "ice_thickness_nm": fit["node_thickness_nm"][node_index],
                        "instantaneous_rate_nm_day": fit["node_rate_nm_day"][node_index],
                        "rate_multiplier": fit["node_multiplier"][node_index],
                    }
                )

    schemas = bundle_schemas(response_grid.size)
    model_metadata_rows = model_rows(
        detector_data,
        fit_data,
        fit_layouts,
        fit_root,
        len(observation_rows),
        len(timeseries_rows),
        len(node_rows),
        response_grid,
        response_grid_source=response_grid_source,
    )
    primary = fits.PrimaryHDU()
    header = primary.header
    header["BUNDLE"] = "CAR190_WFI_PRODUCTION_PRIOR"
    header["BVER"] = "1.2"
    header["ESTIM"] = "MAP"
    header["UNCERT"] = "NONE"
    header["TIMESYS"] = "UTC"
    header["TIMEDEF"] = "EXPOSURE_START"
    header["RELDEF"] = "DAYS_FROM_DETECTOR_FIRST_EXPOSURE"
    header["POSDEF"] = "ZERO_BASED_SCIENCE_FRAME_PIXELS"
    grid_sizes = {statistics["tile_grid_size"] for statistics in detector_data.values()}
    header["TILEGRID"] = (
        f"{next(iter(grid_sizes))}X{next(iter(grid_sizes))}_PER_DETECTOR"
        if len(grid_sizes) == 1
        else "VARIABLE_PER_DETECTOR"
    )
    header["NDETECT"] = len(SCAS)
    header["NTILES"] = len(tile_rows)
    header["NOBS"] = len(observation_rows)
    header["NTSROWS"] = len(timeseries_rows)
    header["NNODES"] = len(node_rows)
    header["CREATED"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header["GITREV"] = git_revision()[:40]
    header["MODEL"] = "FIXED_PRODUCTION_TMM"
    header["CORRECT"] = "INPUT_ALREADY_CORRECTED"
    header["COMMENT"] = "See README, MODEL, and SCHEMA extensions before use."
    header.add_comment("All fit values in this product are deterministic MAP point estimates.")
    header.add_comment("MJD columns are UTC exposure-start timestamps, not midpoints.")

    hdus = [
        primary,
        readme_hdu(readme_lines()),
        model_hdu(model_metadata_rows),
        schema_hdu(schemas),
        make_table_hdu("HGCDTE_REFERENCE", schemas["HGCDTE_REFERENCE"], reference_rows),
        make_table_hdu("DETECTORS", schemas["DETECTORS"], detector_rows),
        make_table_hdu("TILES", schemas["TILES"], tile_rows),
        make_table_hdu("OBSERVATIONS", schemas["OBSERVATIONS"], observation_rows),
        make_table_hdu("TILE_TIMESERIES", schemas["TILE_TIMESERIES"], timeseries_rows),
        make_table_hdu("NODE_TRAJECTORIES", schemas["NODE_TRAJECTORIES"], node_rows),
        make_table_hdu("PARAMETERS", schemas["PARAMETERS"], parameter_rows),
        make_table_hdu("PROVENANCE", schemas["PROVENANCE"], provenance_rows),
    ]
    return fits.HDUList(hdus)


def readme_lines():
    return [
        "CAR190 WFI production prior bundle, schema version 1.2.",
        "Read the primary header and MODEL/SCHEMA extensions before consuming values.",
        "All parameter and trajectory estimates are deterministic least-squares MAP outputs.",
        "No posterior uncertainty is represented by the fitted values in this product.",
        "OBSERVATIONS has one row per detector and observation group.",
        "TILE_TIMESERIES joins detector, tile, and observation_index to fitted state and residuals.",
        "NODE_TRAJECTORIES contains the non-parametric repeat-node state used by the fit.",
        "exposure_start_utc is the source exposure start in ISO-8601 UTC with a trailing Z.",
        "time_mjd_utc is the same exposure start in UTC Modified Julian Date.",
        "time_days_since_first_exposure is the source statistics days value for that detector.",
        "tile_row/tile_col are zero-based tile indices in the detector science frame; grid dimensions are recorded per detector.",
        "row/col bounds are zero-based half-open pixel intervals: [start, stop_exclusive).",
        "The source statistics are corrected products; no correction table is applied here.",
        "The bundle contains checksums and per-tile statistics fingerprints for provenance.",
        "HGCDTE_REFERENCE gives the wavelength-dependent internal reference n and k used by the fit.",
        "PARAMETERS absolute_hgcdte_n and absolute_hgcdte_k are physical per-tile vectors in HGCDTE_REFERENCE row order; for cached relative-response fits this is a reconstruction grid.",
        "Physical HgCdTe values are reference_n/k multiplied by exp of the corresponding fitted log scale.",
        "The HgCdTe reference is an internal 80 K, x=0.445 sensitivity model, not a detector-specific validation table.",
        "Absolute-substrate-power fits omit multiplication by independent measured QE values; cached relative-response fits retain their response-cache semantics.",
        "Third-decon fields are NaN/false when the optional third-decon boundary is unset.",
        "post_third_decon marks observations on/after the optional physical third-decon boundary.",
    ]


def model_rows(
    detector_data,
    fit_data,
    fit_layouts,
    fit_root,
    observation_count,
    timeseries_count,
    node_count,
    response_grid,
    response_grid_source=None,
):
    prior_values = [
        (
            fit["trajectory_rate_prior_mean_nm_day"],
            fit["trajectory_rate_prior_log_sigma"],
        )
        for detector_fits in fit_data.values()
        for fit in detector_fits.values()
        if np.isfinite(fit["trajectory_rate_prior_mean_nm_day"])
        and np.isfinite(fit["trajectory_rate_prior_log_sigma"])
    ]
    if prior_values:
        prior_array = np.asarray(prior_values, dtype=float)
        if not np.allclose(prior_array, prior_array[0], rtol=0.0, atol=1.0e-12):
            raise ValueError("Trajectory-rate prior metadata differs across fit archives")
        trajectory_rate_prior_mean, trajectory_rate_prior_log_sigma = prior_array[0]
    else:
        trajectory_rate_prior_mean = 5.0
        trajectory_rate_prior_log_sigma = 1.5
    third_days = {
        float(fit["third_decon_day"])
        for detector_fits in fit_data.values()
        for fit in detector_fits.values()
        if fit["third_decon_included"]
    }
    if len(third_days) > 1:
        raise ValueError("Third-decon day differs across fit archives")
    third_day = next(iter(third_days), None)
    fit_modes = {
        fit["fit_mode"]
        for detector_fits in fit_data.values()
        for fit in detector_fits.values()
    }
    response_caches = {
        str(Path(fit["response_cache"]).parent.parent)
        for detector_fits in fit_data.values()
        for fit in detector_fits.values()
        if fit["response_cache"]
    }
    response_cache_parents = {str(Path(path).parent) for path in response_caches}
    response_cache_source = (
        next(iter(response_cache_parents))
        if len(response_cache_parents) == 1
        else f"{len(response_cache_parents)} distinct detector cache parent roots"
    )
    prior_sigmas = {
        float(fit["thickness_prior_log_sigma"])
        for detector_fits in fit_data.values()
        for fit in detector_fits.values()
        if np.isfinite(fit["thickness_prior_log_sigma"])
    }
    prior_stages = {
        fit["thickness_prior_stages"]
        for detector_fits in fit_data.values()
        for fit in detector_fits.values()
        if fit["thickness_prior_stages"]
    }
    relative_response_fit = fit_modes and all(
        mode.startswith("relative_response") for mode in fit_modes
    )
    response_mode = (
        "cached six-LED relative response; fixed-optics HgCdTe reconstruction grid"
        if relative_response_fit
        else "absolute substrate power; semi-infinite HgCdTe"
    )
    initial_thickness_prior = "lognormal median 10, log sigma 1.5"
    if len(prior_sigmas) == 1:
        initial_thickness_prior = f"lognormal median 10, log sigma {next(iter(prior_sigmas)):g}"
    if len(prior_stages) == 1:
        initial_thickness_prior += f"; stages {next(iter(prior_stages))}"

    rows = [
        ("bundle_schema_version", "1.2", "", "Version of the embedded bundle schema."),
        ("product_type", "CAR190_WFI_PRODUCTION_PRIOR", "", "Product purpose."),
        ("fit_estimator", "deterministic least-squares MAP", "", "Estimator used for fitted values."),
        ("fit_uncertainty", "not represented; MAP point estimates only", "", "Uncertainty limitation."),
        ("response_mode", response_mode, "", "Production optical response."),
        ("fit_modes", "; ".join(sorted(fit_modes)), "", "Fit mode values recorded by the source archives."),
        ("response_cache_root", response_cache_source, "", "Common response-cache parent recorded by the source archives."),
        ("response_grid_source", str(response_grid_source or "fit archives"), "", "Source of the bundled response wavelength grid."),
        ("ar_layers", "3 independent lossless layers", "", "AR model structure."),
        ("sed_weighting", "energy", "", "SED weighting mode."),
        ("ice_index", "1.25", "", "Fixed lossless ice refractive index."),
        ("ice_reset_day", str(RESET_DAY), "day", "Fixed reset day in relative time."),
        (
            "third_decon_day",
            "unset" if third_day is None else str(third_day),
            "day",
            "Optional physical third-decon boundary; unset for legacy two-decon fits.",
        ),
        ("initial_thickness_prior", initial_thickness_prior, "nm", "Initial/reset thickness prior."),
        (
            "trajectory_rate_prior",
            f"lognormal mean {trajectory_rate_prior_mean:g}, log sigma {trajectory_rate_prior_log_sigma:g}",
            "nm/day",
            "Trajectory rate prior recorded by the fit archives.",
        ),
        ("post_reset_offset_prior_sigma", "0.1", "log", "Post-reset tile offset prior sigma."),
        ("post_third_decon_offset_prior_sigma", "0.1", "log", "Optional post-third-decon tile offset prior sigma."),
        ("absolute_hgcdte_scale_prior_sigma", "0.25", "log", "HgCdTe n/k scale prior sigma."),
        ("hgcdte_reference_model", "fit_wfi_tmm_production.hgcdte_reference_optical_constants", "", "Function that generated HGCDTE_REFERENCE."),
        ("hgcdte_reference_status", "internal sensitivity model; not detector-specific validation", "", "Interpretation of the bundled HgCdTe reference."),
        ("hgcdte_reference_composition_x", str(HGCDTE_REFERENCE_COMPOSITION), "", "Cadmium composition in Hg(1-x)Cd(x)Te."),
        ("hgcdte_reference_temperature_k", str(HGCDTE_REFERENCE_TEMPERATURE_K), "K", "Reference temperature."),
        ("hgcdte_native_n_table_range_nm", f"{HGCDTE_N_TABLE_WAVELENGTHS_NM[0]:.1f}-{HGCDTE_N_TABLE_WAVELENGTHS_NM[-1]:.1f}", "nm", "Native range of the tabulated real-index reference."),
        ("hgcdte_bundle_curve_range_nm", f"{response_grid[0]:.9f}-{response_grid[-1]:.9f}", "nm", "Wavelength range of the bundled response-grid reference and physical vectors."),
        ("hgcdte_physical_reconstruction", "n=reference_n*exp(n_log_scale); k=reference_k*exp(k_log_scale)", "", "Per-tile physical HgCdTe reconstruction."),
        (
            "qe_policy",
            "not applicable to cached relative-response fit"
            if relative_response_fit
            else "independent measured QE values are not multiplied",
            "",
            "Absolute-substrate-power response policy.",
        ),
        ("system_effective_area", "not used by the production RCS fit", "", "Roman effective-area curves must replace, not multiply, independent QE in a downstream system-throughput model."),
        ("ar_thickness_parameterization", "raw production coordinate * 100 = physical nm", "nm", "AR thickness conversion used by the production TMM."),
        ("time_scale", "UTC", "", "Absolute timestamp scale."),
        ("absolute_time_definition", "exposure_start_utc and time_mjd_utc", "", "Timestamp semantics."),
        ("relative_time_definition", "source days from first exposure per detector", "day", "Relative time semantics."),
        ("tile_position_definition", "zero-based science-frame half-open pixel bounds", "pixel", "Tile position semantics."),
        ("wfi_mosaic_position", "display pitch coordinates from the Roman six-column layout", "pitch", "SCA mosaic position semantics."),
        ("statistics_correction", "already applied in input corrected CSV", "", "No correction is reapplied."),
        ("source_fit_root", str(fit_root), "", "Fit archive root used to construct the bundle."),
        ("detector_count", str(len(detector_data)), "", "Number of bundled detectors."),
        (
            "tile_count",
            str(sum(len(statistics["tile_ids"]) for statistics in detector_data.values())),
            "",
            "Number of bundled tiles.",
        ),
        ("observation_row_count", str(observation_count), "", "Rows in OBSERVATIONS."),
        ("tile_timeseries_row_count", str(timeseries_count), "", "Rows in TILE_TIMESERIES."),
        ("node_trajectory_row_count", str(node_count), "", "Rows in NODE_TRAJECTORIES."),
        ("fit_layouts", "; ".join(sorted(set(fit_layouts.values()))), "", "Archive layout(s) used."),
    ]
    return [
        {"key": key, "value": value, "unit": unit, "description": description}
        for key, value, unit, description in rows
    ]


def bundle_schemas(reference_wavelength_count=567):
    return {
        "HGCDTE_REFERENCE": (
            ("wavelength_nm", "D", "nm", "Production response-grid wavelength shared by all fit archives."),
            ("reference_n", "D", "", "Internal reference HgCdTe real refractive index at this wavelength."),
            ("reference_k", "D", "", "Internal reference HgCdTe extinction coefficient at this wavelength."),
        ),
        "DETECTORS": (
            ("detector", "A5", "", "Roman WFI SCA identifier."),
            ("sca_index", "I", "", "Zero-based detector index in WFI01-WFI18 order."),
            ("mosaic_column", "I", "", "Zero-based display column in the six-column WFI layout."),
            ("mosaic_row", "I", "", "Zero-based display row, bottom-to-top."),
            ("mosaic_x_pitch_units", "D", "pitch", "Display x origin; not a physical focal-plane coordinate."),
            ("mosaic_y_pitch_units", "D", "pitch", "Display y origin; not a physical focal-plane coordinate."),
            ("first_exposure_mjd_utc", "D", "d", "MJD UTC of the detector's first exposure start."),
            ("observation_count", "J", "", "Number of observation groups for this detector."),
            ("tile_count", "J", "", "Number of tiles for this detector."),
        ),
        "TILES": (
            ("detector", "A5", "", "Roman WFI SCA identifier."),
            ("sca_index", "I", "", "Zero-based detector index."),
            ("tile_id", "I", "", "Zero-based tile identifier, with tile-row/column layout recorded per detector."),
            ("tile_row", "I", "", "Zero-based science-frame tile row."),
            ("tile_col", "I", "", "Zero-based science-frame tile column."),
            ("row_start", "J", "pixel", "Inclusive zero-based detector row start."),
            ("row_stop_exclusive", "J", "pixel", "Exclusive detector row stop."),
            ("col_start", "J", "pixel", "Inclusive zero-based detector column start."),
            ("col_stop_exclusive", "J", "pixel", "Exclusive detector column stop."),
            ("tile_n_rows", "J", "pixel", "Tile height."),
            ("tile_n_cols", "J", "pixel", "Tile width."),
            ("pixel_center_row", "D", "pixel", "Center of the half-open row interval."),
            ("pixel_center_col", "D", "pixel", "Center of the half-open column interval."),
            ("fit_status", "A8", "", "Fit status; MAP means a completed deterministic fit."),
            ("statistics_fingerprint", "A64", "", "SHA-256 fingerprint of tile sufficient-statistic values."),
            ("fit_archive_sha256", "A64", "", "SHA-256 of the source posterior_samples.npz archive."),
            ("fit_archive", "A256", "", "Fit archive path relative to the fit root."),
        ),
        "OBSERVATIONS": (
            ("detector", "A5", "", "Roman WFI SCA identifier."),
            ("sca_index", "I", "", "Zero-based detector index."),
            ("observation_index", "J", "", "Zero-based join key within one detector."),
            ("group", "J", "", "Source sufficient-statistics group identifier."),
            ("led", "I", "", "LED identifier 1-6."),
            ("repeat", "I", "", "Source repeat/epoch identifier."),
            ("exposure_start_utc", "A32", "", "ISO-8601 UTC exposure start with trailing Z."),
            ("time_mjd_utc", "D", "d", "UTC Modified Julian Date of exposure start."),
            ("time_days_since_first_exposure", "D", "day", "Source relative time from first detector exposure."),
            ("time_seconds_since_first_exposure", "D", "s", "Same relative time expressed in seconds."),
            ("file_set_name", "A96", "", "Source current exposure file-set identifier."),
            ("baseline_file_set_name", "A96", "", "Source baseline exposure file-set identifier."),
            ("post_reset", "L", "", "Whether relative time is on/after the fixed reset day."),
            ("post_third_decon", "L", "", "Whether relative time is on/after the optional third-decon day."),
        ),
        "TILE_TIMESERIES": (
            ("detector", "A5", "", "Roman WFI SCA identifier."),
            ("sca_index", "I", "", "Zero-based detector index."),
            ("tile_id", "I", "", "Zero-based tile identifier."),
            ("tile_row", "I", "", "Zero-based science-frame tile row."),
            ("tile_col", "I", "", "Zero-based science-frame tile column."),
            ("observation_index", "J", "", "Join key into OBSERVATIONS for this detector."),
            ("group", "J", "", "Source sufficient-statistics group identifier."),
            ("led", "I", "", "LED identifier 1-6."),
            ("repeat", "I", "", "Source repeat/epoch identifier."),
            ("exposure_start_utc", "A32", "", "ISO-8601 UTC exposure start with trailing Z."),
            ("time_mjd_utc", "D", "d", "UTC Modified Julian Date of exposure start."),
            ("time_days_since_first_exposure", "D", "day", "Source relative time."),
            ("n_pair_valid", "J", "pixel", "Valid pixel-pair count in the corrected statistics."),
            ("sum_log_change", "D", "", "Corrected sufficient-statistics sum of log changes."),
            ("sum_squared_log_change", "D", "", "Corrected sufficient-statistics sum of squared log changes."),
            ("observed_log_change", "D", "", "sum_log_change / n_pair_valid."),
            ("model_log_change", "D", "", "Production MAP model prediction."),
            ("residual_log", "D", "", "observed_log_change - model_log_change."),
            ("ice_thickness_nm", "D", "nm", "Fitted ice thickness at this observation."),
            ("instantaneous_rate_nm_day", "D", "nm/day", "Fitted instantaneous accumulation rate."),
            ("post_reset", "L", "", "Whether relative time is on/after the fixed reset day."),
            ("post_third_decon", "L", "", "Whether relative time is on/after the optional third-decon day."),
        ),
        "NODE_TRAJECTORIES": (
            ("detector", "A5", "", "Roman WFI SCA identifier."),
            ("sca_index", "I", "", "Zero-based detector index."),
            ("tile_id", "I", "", "Zero-based tile identifier."),
            ("tile_row", "I", "", "Zero-based science-frame tile row."),
            ("tile_col", "I", "", "Zero-based science-frame tile column."),
            ("node_index", "I", "", "Zero-based non-parametric optimization node."),
            ("node_day_since_first_exposure", "D", "day", "Mean relative day of the source repeat node."),
            ("node_mjd_utc", "D", "d", "Derived UTC MJD: detector first MJD plus node day."),
            ("ice_thickness_nm", "D", "nm", "Fitted ice thickness at the node."),
            ("instantaneous_rate_nm_day", "D", "nm/day", "Fitted instantaneous rate at the node."),
            ("rate_multiplier", "D", "", "Fitted rate multiplier relative to the trajectory-rate prior mean recorded in MODEL."),
        ),
        "PARAMETERS": (
            ("detector", "A5", "", "Roman WFI SCA identifier."),
            ("sca_index", "I", "", "Zero-based detector index."),
            ("tile_id", "I", "", "Zero-based tile identifier."),
            ("tile_row", "I", "", "Zero-based science-frame tile row."),
            ("tile_col", "I", "", "Zero-based science-frame tile column."),
            ("ar_layer1_thickness_scale_100nm", "D", "", "Raw production AR thickness coordinate; physical nm equals 100 times this value."),
            ("ar_layer1_thickness_nm", "D", "nm", "Physical MAP AR layer 1 thickness."),
            ("ar_layer1_n", "D", "", "MAP AR layer 1 real refractive index."),
            ("ar_layer1_k", "D", "", "MAP AR layer 1 extinction coefficient."),
            ("ar_layer2_thickness_scale_100nm", "D", "", "Raw production AR thickness coordinate; physical nm equals 100 times this value."),
            ("ar_layer2_thickness_nm", "D", "nm", "Physical MAP AR layer 2 thickness."),
            ("ar_layer2_n", "D", "", "MAP AR layer 2 real refractive index."),
            ("ar_layer2_k", "D", "", "MAP AR layer 2 extinction coefficient."),
            ("ar_layer3_thickness_scale_100nm", "D", "", "Raw production AR thickness coordinate; physical nm equals 100 times this value."),
            ("ar_layer3_thickness_nm", "D", "nm", "Physical MAP AR layer 3 thickness."),
            ("ar_layer3_n", "D", "", "MAP AR layer 3 real refractive index."),
            ("ar_layer3_k", "D", "", "MAP AR layer 3 extinction coefficient."),
            ("incidence_angle_deg", "D", "deg", "Fitted non-negative chief-ray incidence angle."),
            ("initial_thickness_nm", "D", "nm", "MAP initial ice thickness."),
            ("reset_initial_thickness_nm", "D", "nm", "MAP ice thickness immediately after reset."),
            ("post_reset_tile_log_offset", "D", "log", "MAP post-reset tile log offset."),
            ("third_decon_initial_thickness_nm", "D", "nm", "Optional MAP ice thickness immediately after the third decon; NaN when unset."),
            ("post_third_decon_tile_log_offset", "D", "log", "Optional MAP post-third-decon tile log offset; NaN when unset."),
            ("absolute_hgcdte_n_log_scale", "D", "log", "MAP HgCdTe n log scale."),
            ("absolute_hgcdte_k_log_scale", "D", "log", "MAP HgCdTe k log scale."),
            ("absolute_hgcdte_n", f"{reference_wavelength_count}D", "", "Physical per-tile HgCdTe n vector indexed by HGCDTE_REFERENCE row."),
            ("absolute_hgcdte_k", f"{reference_wavelength_count}D", "", "Physical per-tile HgCdTe k vector indexed by HGCDTE_REFERENCE row."),
            ("initial_rate_nm_day", "D", "nm/day", "MAP initial instantaneous rate."),
            ("final_rate_nm_day", "D", "nm/day", "MAP final instantaneous rate."),
            ("mean_rate_nm_day", "D", "nm/day", "MAP time-mean rate."),
            ("map_objective", "D", "", "Final MAP objective value recorded by the fit."),
            ("map_steps", "J", "", "Configured MAP evaluation limit recorded by the fit."),
            ("fit_incidence_angle", "L", "", "Whether incidence_angle_deg was fitted."),
            ("ice_reset_day", "D", "day", "Fixed reset day used by this fit."),
            ("third_decon_day", "D", "day", "Optional third-decon day used by this fit; NaN when unset."),
            ("map_optimizer", "A32", "", "MAP optimizer name."),
        ),
        "PROVENANCE": (
            ("detector", "A5", "", "Roman WFI SCA identifier."),
            ("statistics_file", "A96", "", "Corrected source statistics filename."),
            ("statistics_sha256", "A64", "", "SHA-256 of the complete source statistics file."),
            ("fit_layout", "A40", "", "Fit archive layout selected by the bundler."),
            ("fit_root", "A256", "", "Absolute fit archive root used to build the bundle."),
            ("fit_archive_count", "J", "", "Number of fit archives included for this detector."),
            ("observation_count", "J", "", "Number of detector observation rows."),
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        required=True,
        help="Directory containing corrected WFI tile-statistics CSVs.",
    )
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=ROOT,
        help="Root containing production or legacy per-tile fit archives.",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_integer,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile edge length used by the production reduction; default: {DEFAULT_TILE_SIZE}.",
    )
    parser.add_argument(
        "--response-grid-cache",
        type=Path,
        default=None,
        help=(
            "Spectral response cache containing wavelengths_nm; used when the fit "
            "archives do not carry their own response grid."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or (
        production_output_root(args.statistics_dir, args.tile_size)
        / "wfi_production_map_prior_bundle.fits"
    )
    response_grid = None
    response_grid_source = None
    if args.response_grid_cache is not None:
        response_grid = load_response_grid_cache(args.response_grid_cache)
        response_grid_source = str(args.response_grid_cache.resolve())
    hdul = build_bundle(
        args.statistics_dir,
        args.fit_root,
        tile_size=args.tile_size,
        response_grid=response_grid,
        response_grid_source=response_grid_source,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(output, overwrite=True, checksum=True)
    hdul.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
