"""Plot per-tile ice rates in the Roman WFI focal-plane mosaic."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from wfi_production_plot_annotations import (
    AOM_TEMPERATURE_BUMP_LABEL,
    latest_observation_label,
    load_time_annotations,
)
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    detector_output_root as production_detector_output_root,
    positive_integer,
    production_output_root,
    tile_directory_entries,
    tile_grid_size_for_tile_size,
)


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_FIT_SUFFIX = "production"
METRICS = (
    ("Initial rate", "initial_rate_nm_day_median"),
    ("Final rate", "final_rate_nm_day_median"),
    ("Mean rate", "mean_rate_nm_day_median"),
)
RESET_METRICS = (
    ("Mean before Decon2", "pre_reset_mean_rate_nm_day"),
    ("Mean after Decon2", "post_reset_mean_rate_nm_day"),
)
TREND_METRICS = (
    ("Linear trend before Decon2", "pre_reset_trend_rate_nm_day"),
    ("Linear trend after Decon2", "post_reset_trend_rate_nm_day"),
)
TREND_RATE_COLOR_LIMITS = (7.9, 10.9)
RMS_METRICS = (("RMS residual", "rms_residual_millilog"),)
AOM_JUMP_METRICS = (
    ("Rate jump", "aom_rate_jump_nm_day"),
    ("Percentage change", "aom_rate_jump_percent"),
)
AOM_JUMP_VMAX_PERCENTILE = 99.0
LAST_48H_METRICS = (
    ("Mean accumulation rate, last 48 h", "last_48h_mean_rate_nm_day"),
)
CURRENT_THICKNESS_METRICS = (("Current ice thickness", "current_ice_thickness_nm"),)
CURRENT_THICKNESS_VMAX_PERCENTILE = 99.0
INITIAL_THICKNESS_METRICS = (
    ("Before Decon2", "initial_thickness_nm"),
    ("After Decon2", "reset_initial_thickness_nm"),
    ("After third decon", "third_decon_initial_thickness_nm"),
)
INITIAL_THICKNESS_VMAX_PERCENTILE = 99.0

# The Roman WFI layout is a staggered six-column by three-row mosaic.  The
# offsets reproduce the smile-shaped focal plane shown in the Roman reference
# frame figure; y increases toward the top of the plot.
COLUMN_SCA = (
    ("WFI16", "WFI17", "WFI18"),
    ("WFI13", "WFI14", "WFI15"),
    ("WFI10", "WFI11", "WFI12"),
    ("WFI01", "WFI02", "WFI03"),
    ("WFI04", "WFI05", "WFI06"),
    ("WFI07", "WFI08", "WFI09"),
)
COLUMN_Y_OFFSET = (0.0, -4.0, -6.0, -6.0, -4.0, 0.0)
SCA_SIZE = 8.0
SCA_PITCH = 9.0


def sca_position(sca):
    for column, scas in enumerate(COLUMN_SCA):
        if sca in scas:
            # COLUMN_SCA is listed top-to-bottom, while plot y increases up.
            row = len(scas) - 1 - scas.index(sca)
            return SCA_PITCH * column, SCA_PITCH * row + COLUMN_Y_OFFSET[column]
    raise ValueError(f"Unknown SCA: {sca}")


def load_tile_rows(path, expected_sca):
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {row.get("detector") for row in rows} != {expected_sca}:
        return None
    return rows


def load_rate_map(path, expected_sca):
    rows = load_tile_rows(path, expected_sca)
    if rows is None:
        fit_root = path.parent / "fits"
        rows = []
        for tile_path in sorted(fit_root.glob("tile_*/tile_rate_posterior.csv")):
            tile_rows = load_tile_rows(tile_path, expected_sca)
            if tile_rows is not None:
                rows.extend(tile_rows)
        if not rows:
            rows = None
    if rows is None:
        return None

    coordinates = [
        (int(row["tile_row"]), int(row["tile_col"])) for row in rows
    ]
    tile_grid_size = max(max(row, column) for row, column in coordinates) + 1
    maps = {}
    for _, field in METRICS:
        values = np.full((tile_grid_size, tile_grid_size), np.nan)
        for row in rows:
            values[int(row["tile_row"]), int(row["tile_col"])] = float(row[field])
        # MAST L1/L2 arrays are already in the common WFI science frame.
        # Tile row 0 is the science-frame bottom, so no physical-detector flip
        # belongs here.
        maps[field] = values
    return maps


def load_aom_rate_jump_map(output_dir, expected_sca, event_day):
    """Load latest-post-AOM versus last-pre-AOM rate changes by tile."""
    rows = load_tile_rows(Path(output_dir) / "tile_rate_posterior.csv", expected_sca)
    if rows is None:
        return None
    row_tile_ids = [int(row["tile_id"]) for row in rows]
    try:
        tile_ids = complete_tile_ids(row_tile_ids)
    except ValueError:
        return None
    fit_dirs = dict(tile_directory_entries(Path(output_dir) / "fits"))
    tile_grid_size = int(len(tile_ids) ** 0.5)
    jump_values = np.full((tile_grid_size, tile_grid_size), np.nan, dtype=float)
    percent_values = np.full((tile_grid_size, tile_grid_size), np.nan, dtype=float)
    for row in rows:
        tile_id = int(row["tile_id"])
        fit_dir = fit_dirs.get(tile_id)
        if fit_dir is None:
            return None
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        with np.load(posterior_path, allow_pickle=False) as samples:
            days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            rates = np.asarray(
                samples["tile_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
        if days.size != rates.size:
            raise ValueError(f"Unexpected tile-rate shape in {posterior_path}")
        finite = np.isfinite(days) & np.isfinite(rates)
        if not np.any(finite):
            return None
        days = days[finite]
        rates = rates[finite]
        order = np.argsort(days)
        days = days[order]
        rates = rates[order]
        before = np.flatnonzero(days < float(event_day))
        after = np.flatnonzero(days > float(event_day))
        if before.size == 0 or after.size == 0:
            return None
        pre_rate = rates[before[-1]]
        post_rate = rates[after[-1]]
        if not np.isfinite(pre_rate) or not np.isfinite(post_rate) or pre_rate == 0.0:
            return None
        jump = post_rate - pre_rate
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        jump_values[tile_row, tile_col] = jump
        percent_values[tile_row, tile_col] = 100.0 * jump / abs(pre_rate)
    return {
        "aom_rate_jump_nm_day": jump_values,
        "aom_rate_jump_percent": percent_values,
    }


def maxent_fit_set_complete(output_dir):
    """Return whether all tile archives are current MaxEnt production fits."""
    fit_root = Path(output_dir) / "fits"
    entries = tile_directory_entries(fit_root)
    try:
        complete_tile_ids(tile_id for tile_id, _path in entries)
    except ValueError:
        return False
    for _tile_id, fit_dir in entries:
        try:
            with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
                if (
                    str(np.asarray(samples["trajectory_regularization"]).item()) != "maxent"
                    or not np.isclose(
                        float(np.asarray(samples["trajectory_entropy_weight"]).item()),
                        0.1,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or int(np.asarray(samples["constant_rate_map_warmup"]).item()) != 1
                ):
                    return False
        except (KeyError, OSError, ValueError):
            return False
    return True


def load_reset_rate_values(posterior_path):
    with np.load(posterior_path, allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
        reset_values = np.asarray(samples["ice_reset_day"], dtype=float).reshape(-1)
        thickness = np.asarray(samples["tile_ice_thickness"], dtype=float)
        tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)

    if reset_values.size != 1 or not np.isfinite(reset_values[0]):
        return None
    if thickness.ndim < 2 or thickness.size % (tile_ids.size * days.size) != 0:
        raise ValueError(f"Unexpected tile-thickness shape in {posterior_path}")
    thickness = thickness.reshape(-1, tile_ids.size, days.size)
    if thickness.shape[1] != tile_ids.size:
        raise ValueError(f"Tile IDs do not match tile thickness in {posterior_path}")

    reset_day = float(reset_values[0])
    before = days < reset_day
    after = days >= reset_day
    if not np.any(before) or not np.any(after):
        raise ValueError(f"Reset day does not split observations in {posterior_path}")
    before_indices = np.flatnonzero(before)
    after_indices = np.flatnonzero(after)
    pre_start, pre_end = before_indices[[0, -1]]
    post_start, post_end = after_indices[[0, -1]]
    pre_values = np.median(
        (thickness[:, :, pre_end] - thickness[:, :, pre_start])
        / (days[pre_end] - days[pre_start]),
        axis=0,
    )
    post_values = np.median(
        (thickness[:, :, post_end] - thickness[:, :, post_start])
        / (days[post_end] - days[post_start]),
        axis=0,
    )
    return tile_ids, pre_values, post_values, reset_day


def reset_rate_maps_from_rows(
    rows, tile_ids, pre_values, post_values, posterior_path, tile_grid_size=None
):
    tile_indices = {int(tile_id): index for index, tile_id in enumerate(tile_ids)}
    if tile_grid_size is None:
        tile_grid_size = max(
            max(int(row["tile_row"]), int(row["tile_col"])) for row in rows
        ) + 1
    maps = {
        field: np.full((tile_grid_size, tile_grid_size), np.nan)
        for _, field in RESET_METRICS
    }
    for row in rows:
        tile_id = int(row["tile_id"])
        if tile_id not in tile_indices:
            raise ValueError(f"Tile {tile_id} is missing from {posterior_path}")
        tile_index = tile_indices[tile_id]
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        maps[RESET_METRICS[0][1]][tile_row, tile_col] = pre_values[tile_index]
        maps[RESET_METRICS[1][1]][tile_row, tile_col] = post_values[tile_index]
    return maps


def load_reset_rate_map(output_dir, expected_sca):
    rows = load_tile_rows(output_dir / "tile_rate_posterior.csv", expected_sca)
    posterior_path = output_dir / "posterior_samples.npz"
    if rows is not None and posterior_path.is_file():
        loaded = load_reset_rate_values(posterior_path)
        if loaded is None:
            return None
        tile_ids, pre_mean, post_mean, reset_day = loaded
        maps = reset_rate_maps_from_rows(
            rows,
            tile_ids,
            pre_mean,
            post_mean,
            posterior_path,
        )
        return maps, reset_day

    # Independent MAP runs archive one posterior per tile rather than one
    # detector-level posterior. Reconstruct the detector map from those files.
    entries = tile_directory_entries(output_dir / "fits")
    try:
        tile_ids = complete_tile_ids(tile_id for tile_id, _path in entries)
    except ValueError:
        return None
    tile_fit_dirs = [path for _tile_id, path in entries]
    tile_grid_size = int(len(tile_ids) ** 0.5)
    maps = {
        field: np.full((tile_grid_size, tile_grid_size), np.nan)
        for _, field in RESET_METRICS
    }
    reset_days = []
    for tile_fit_dir in tile_fit_dirs:
        tile_rows = load_tile_rows(
            tile_fit_dir / "tile_rate_posterior.csv", expected_sca
        )
        tile_posterior = tile_fit_dir / "posterior_samples.npz"
        if tile_rows is None or not tile_posterior.is_file():
            return None
        loaded = load_reset_rate_values(tile_posterior)
        if loaded is None:
            return None
        tile_ids, pre_mean, post_mean, reset_day = loaded
        tile_maps = reset_rate_maps_from_rows(
            tile_rows,
            tile_ids,
            pre_mean,
            post_mean,
            tile_posterior,
            tile_grid_size=tile_grid_size,
        )
        for field in tile_maps:
            finite = np.isfinite(tile_maps[field])
            maps[field][finite] = tile_maps[field][finite]
        reset_days.append(reset_day)
    if not np.allclose(reset_days, reset_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"Independent outputs use inconsistent reset days: {reset_days}"
        )
    return maps, reset_days[0]


def _load_tile_thickness(posterior_path):
    with np.load(posterior_path, allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
        thickness = np.asarray(samples["tile_ice_thickness"], dtype=float)
        reset_day = float(np.asarray(samples["ice_reset_day"]).reshape(-1)[0])
    if thickness.ndim == 4:
        thickness = thickness[0, 0]
    elif thickness.ndim == 3:
        thickness = thickness[0]
    elif thickness.ndim == 2:
        thickness = thickness[0]
    thickness = thickness.reshape(-1)
    if thickness.size != days.size:
        raise ValueError(f"Unexpected tile thickness shape in {posterior_path}")
    return days, thickness, reset_day


def mean_rate_over_interval(days, thickness, duration_days=1.0):
    """Return the time-mean fitted accumulation rate over a trailing interval."""
    if duration_days <= 0.0:
        raise ValueError("duration_days must be positive")
    days = np.asarray(days, dtype=float).reshape(-1)
    thickness = np.asarray(thickness, dtype=float).reshape(-1)
    if days.size != thickness.size:
        raise ValueError("Days and thickness must have the same size")
    finite = np.isfinite(days) & np.isfinite(thickness)
    days = days[finite]
    thickness = thickness[finite]
    if days.size < 2:
        return None
    order = np.argsort(days)
    days = days[order]
    thickness = thickness[order]
    end_day = float(days[-1])
    start_day = end_day - float(duration_days)
    if start_day < days[0]:
        return None
    start_thickness = float(np.interp(start_day, days, thickness))
    end_thickness = float(np.interp(end_day, days, thickness))
    return (end_thickness - start_thickness) / float(duration_days)


def load_trend_rate_map(output_dir, expected_sca):
    rows = load_tile_rows(output_dir / "tile_rate_posterior.csv", expected_sca)
    fit_root = Path(output_dir) / "fits"
    if rows is None:
        return None
    row_tile_ids = [int(row["tile_id"]) for row in rows]
    if len(set(row_tile_ids)) != len(row_tile_ids):
        return None
    try:
        tile_ids = complete_tile_ids(row_tile_ids)
    except ValueError:
        return None
    fit_dirs = dict(tile_directory_entries(fit_root))
    tile_grid_size = int(len(tile_ids) ** 0.5)
    maps = {
        field: np.full((tile_grid_size, tile_grid_size), np.nan)
        for _, field in TREND_METRICS
    }
    reset_days = []
    for row in rows:
        tile_id = int(row["tile_id"])
        fit_dir = fit_dirs.get(tile_id)
        if fit_dir is None:
            return None
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        days, thickness, reset_day = _load_tile_thickness(posterior_path)
        before = days < reset_day
        after = days >= reset_day
        if not np.any(before) or not np.any(after):
            raise ValueError(f"Reset day does not split observations in {posterior_path}")
        pre_rate = np.polyfit(days[before], thickness[before], 1)[0]
        post_rate = np.polyfit(days[after], thickness[after], 1)[0]
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        maps["pre_reset_trend_rate_nm_day"][tile_row, tile_col] = pre_rate
        maps["post_reset_trend_rate_nm_day"][tile_row, tile_col] = post_rate
        reset_days.append(reset_day)
    if not np.allclose(reset_days, reset_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Independent outputs use inconsistent reset days: {reset_days}")
    return maps, reset_days[0]


def load_last_48h_rate_map(output_dir, expected_sca):
    rows = load_tile_rows(output_dir / "tile_rate_posterior.csv", expected_sca)
    if rows is None:
        return None
    row_tile_ids = [int(row["tile_id"]) for row in rows]
    if len(set(row_tile_ids)) != len(row_tile_ids):
        return None
    try:
        tile_ids = complete_tile_ids(row_tile_ids)
    except ValueError:
        return None
    fit_dirs = dict(tile_directory_entries(Path(output_dir) / "fits"))
    tile_grid_size = int(len(tile_ids) ** 0.5)
    maps = {
        field: np.full((tile_grid_size, tile_grid_size), np.nan)
        for _, field in LAST_48H_METRICS
    }
    for row in rows:
        tile_id = int(row["tile_id"])
        fit_dir = fit_dirs.get(tile_id)
        if fit_dir is None:
            return None
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        days, thickness, _reset_day = _load_tile_thickness(posterior_path)
        rate = mean_rate_over_interval(days, thickness, duration_days=2.0)
        if rate is None:
            return None
        maps["last_48h_mean_rate_nm_day"][
            int(row["tile_row"]), int(row["tile_col"])
        ] = rate
    return maps


def load_rms_map(output_dir, expected_sca):
    rows = load_tile_rows(output_dir / "tile_rate_posterior.csv", expected_sca)
    if rows is None:
        return None
    row_tile_ids = [int(row["tile_id"]) for row in rows]
    if len(set(row_tile_ids)) != len(row_tile_ids):
        return None
    try:
        tile_ids = complete_tile_ids(row_tile_ids)
    except ValueError:
        return None
    fit_dirs = dict(tile_directory_entries(Path(output_dir) / "fits"))
    tile_grid_size = int(len(tile_ids) ** 0.5)
    values = np.full((tile_grid_size, tile_grid_size), np.nan)
    for row in rows:
        tile_id = int(row["tile_id"])
        fit_dir = fit_dirs.get(tile_id)
        if fit_dir is None:
            return None
        prediction_path = fit_dir / "tile_group_predictions.csv"
        if not prediction_path.is_file():
            return None
        with prediction_path.open(newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
        if not prediction_rows:
            return None
        repeats = np.asarray(
            [int(prediction["repeat"]) for prediction in prediction_rows],
            dtype=int,
        )
        residuals = np.asarray(
            [float(prediction["residual_log"]) for prediction in prediction_rows],
            dtype=float,
        )
        valid = (repeats > np.min(repeats)) & np.isfinite(residuals)
        if not np.any(valid):
            return None
        values[int(row["tile_row"]), int(row["tile_col"])] = (
            1000.0 * np.sqrt(np.mean(residuals[valid] ** 2))
        )
    return {"rms_residual_millilog": values}


def load_current_ice_thickness_map(output_dir, expected_sca, tile_size):
    """Load the latest fitted thickness from each complete tile archive."""
    entries = tile_directory_entries(Path(output_dir) / "fits")
    try:
        complete_tile_ids(tile_id for tile_id, _path in entries)
    except ValueError:
        return None
    tile_grid_size = tile_grid_size_for_tile_size(tile_size)
    values = np.full((tile_grid_size, tile_grid_size), np.nan, dtype=float)
    for tile_id, fit_dir in entries:
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        with np.load(posterior_path, allow_pickle=False) as samples:
            days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            thickness = np.asarray(samples["tile_ice_thickness"], dtype=float).reshape(-1)
        if days.size != thickness.size or days.size == 0:
            raise ValueError(f"Unexpected thickness trajectory in {posterior_path}")
        finite = np.isfinite(days) & np.isfinite(thickness)
        if not np.any(finite):
            continue
        latest = int(np.flatnonzero(finite)[np.argmax(days[finite])])
        values[tile_id // tile_grid_size, tile_id % tile_grid_size] = thickness[latest]
    if not np.isfinite(values).any():
        return None
    return {"current_ice_thickness_nm": values}


def _archive_scalar(samples, names):
    for name in names:
        if name not in samples:
            continue
        values = np.asarray(samples[name], dtype=float).reshape(-1)
        if values.size == 1 and np.isfinite(values[0]):
            return float(values[0])
    return None


def _reset_event_display_name(name):
    known = {"second_decon": "Decon2", "third_decon": "Decon3"}
    return known.get(str(name), str(name).replace("_", " ").capitalize())


def initial_thickness_metric_specs(reset_event_names):
    names = tuple(str(name) for name in reset_event_names)
    metrics = [(f"Before {_reset_event_display_name(names[0])}", "initial_thickness_nm")]
    metrics.extend(
        (
            f"After {_reset_event_display_name(name)}",
            f"reset_event_{index + 1}_initial_thickness_nm",
        )
        for index, name in enumerate(names)
    )
    return tuple(metrics)


def load_initial_thickness_map(
    output_dir, expected_sca, tile_size, *, return_metrics=False
):
    """Load initial thickness for the starting state and every reset event."""
    entries = tile_directory_entries(Path(output_dir) / "fits")
    try:
        complete_tile_ids(tile_id for tile_id, _path in entries)
    except ValueError:
        return None
    tile_grid_size = tile_grid_size_for_tile_size(tile_size)
    first_archive_path = entries[0][1] / "posterior_samples.npz"
    if not first_archive_path.is_file():
        return None
    with np.load(first_archive_path, allow_pickle=False) as samples:
        if "reset_event_names" in samples:
            reset_names = np.asarray(samples["reset_event_names"]).astype(str).reshape(-1)
        else:
            reset_names = np.asarray(
                ["second_decon"]
                + (["third_decon"] if "third_decon_day" in samples else [])
            )
    metrics = initial_thickness_metric_specs(reset_names)
    values = {
        field: np.full((tile_grid_size, tile_grid_size), np.nan, dtype=float)
        for _title, field in metrics
    }
    legacy_log_fields = (
        "reset_initial_thickness_log_nm",
        "third_decon_initial_thickness_log_nm",
    )
    for tile_id, fit_dir in entries:
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        with np.load(posterior_path, allow_pickle=False) as samples:
            stage_values = [_archive_scalar(samples, ("initial_thickness_nm",))]
            if stage_values[0] is None:
                initial_log = _archive_scalar(samples, ("initial_thickness_log_nm",))
                stage_values[0] = None if initial_log is None else float(np.exp(initial_log))
            if "reset_event_initial_thickness_log_nm" in samples:
                reset_logs = np.asarray(
                    samples["reset_event_initial_thickness_log_nm"], dtype=float
                ).reshape(-1)
                stage_values.extend(np.exp(reset_logs).tolist())
            else:
                for index, name in enumerate(reset_names):
                    alias = (
                        legacy_log_fields[index]
                        if index < len(legacy_log_fields)
                        else f"reset_event_{index + 1}_initial_thickness_log_nm"
                    )
                    value = _archive_scalar(samples, (alias,))
                    stage_values.append(None if value is None else float(np.exp(value)))
            if len(stage_values) != len(metrics):
                return None
            for (_title, field), value in zip(metrics, stage_values, strict=True):
                if value is not None and np.isfinite(value):
                    values[field][tile_id // tile_grid_size, tile_id % tile_grid_size] = value
    result = values if any(np.isfinite(value).any() for value in values.values()) else None
    return (result, metrics) if return_metrics else result


def load_all_current_ice_thickness_maps(
    root,
    fit_suffix,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    maps = {}
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        maps[sca] = next(
            (
                thickness_map
                for output_dir in output_dirs
                if (thickness_map := load_current_ice_thickness_map(
                    output_dir, sca, tile_size
                )) is not None
            ),
            None,
        )
    return maps


def load_all_initial_thickness_maps(
    root,
    fit_suffix,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
    return_metrics=False,
):
    maps = {}
    metrics = None
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        maps[sca] = None
        for output_dir in output_dirs:
            loaded = load_initial_thickness_map(
                output_dir, sca, tile_size, return_metrics=True
            )
            if loaded is not None and loaded[0] is not None:
                maps[sca], detector_metrics = loaded
                if metrics is None:
                    metrics = detector_metrics
                elif detector_metrics != metrics:
                    raise ValueError("Detector fits disagree on reset-event initial-thickness fields")
                break
    if return_metrics:
        return maps, (INITIAL_THICKNESS_METRICS if metrics is None else metrics)
    return maps


def percentile_color_limits(maps, field, upper_percentile):
    finite_values = np.concatenate(
        [
            values[field][np.isfinite(values[field])]
            for values in maps.values()
            if values is not None and np.isfinite(values[field]).any()
        ]
    )
    lower = float(np.min(finite_values))
    upper = float(np.percentile(finite_values, upper_percentile))
    if not lower < upper:
        upper = lower + 1.0
    return lower, upper


def initial_thickness_color_limits(
    maps,
    upper_percentile=INITIAL_THICKNESS_VMAX_PERCENTILE,
    metrics=INITIAL_THICKNESS_METRICS,
):
    finite_values = [
        values[field][np.isfinite(values[field])]
        for values in maps.values()
        if values is not None
        for _title, field in metrics
        if np.isfinite(values[field]).any()
    ]
    if not finite_values:
        return 0.0, 1.0
    finite_values = np.concatenate(finite_values)
    lower = float(np.min(finite_values))
    upper = float(np.percentile(finite_values, upper_percentile))
    if not lower < upper:
        upper = lower + 1.0
    return lower, upper


def detector_output_dirs(
    root,
    sca,
    fit_suffix,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    root = Path(root)
    if fit_root is not None:
        return (Path(fit_root) / sca.lower(),)
    if fit_suffix == DEFAULT_FIT_SUFFIX:
        output_dirs = (production_detector_output_root(root, sca, tile_size),)
    else:
        output_dirs = (
            root / f"{sca.lower()}_{fit_suffix}_prod",
            root / f"{sca.lower()}_{fit_suffix}",
            root / f"wfi_{fit_suffix}_{sca}_map10000_emcee1000",
        )
    if fit_subdir is None:
        return output_dirs
    return tuple(output_dir / fit_subdir for output_dir in output_dirs)


def load_all_maps(
    root,
    fit_suffix,
    maxent_only=False,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    maps = {}
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        maps[sca] = next(
            (
                rate_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (rate_map := load_rate_map(output_dir / "tile_rate_posterior.csv", sca))
                is not None
            ),
            None,
        )
    return maps


def _aom_event_day(statistics_path, expected_sca, tile_size):
    paths = [
        path
        for path in _statistics_paths(Path(statistics_path), tile_size)
        if path.name.lower().startswith(expected_sca.lower() + "_")
    ]
    if not paths:
        return None
    annotations = load_time_annotations(paths[0])
    for event in annotations.events:
        if event.label == AOM_TEMPERATURE_BUMP_LABEL:
            return event.day
    return None


def load_all_aom_rate_jump_maps(
    root,
    fit_suffix,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    maps = {}
    for sca in SCAS:
        event_day = _aom_event_day(root, sca, tile_size)
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        maps[sca] = next(
            (
                jump_map
                for output_dir in output_dirs
                if event_day is not None
                if (jump_map := load_aom_rate_jump_map(
                    output_dir, sca, event_day
                )) is not None
            ),
            None,
        )
    return maps


def load_all_reset_maps(
    root,
    fit_suffix,
    maxent_only=False,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    maps = {}
    reset_days = []
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        loaded = next(
            (
                reset_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (reset_map := load_reset_rate_map(output_dir, sca)) is not None
            ),
            None,
        )
        maps[sca] = None if loaded is None else loaded[0]
        if loaded is not None:
            reset_days.append(loaded[1])
    if reset_days and not np.allclose(
        reset_days, reset_days[0], rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"Production outputs use inconsistent reset days: {reset_days}")
    return maps, (reset_days[0] if reset_days else None)


def load_all_trend_maps(
    root,
    fit_suffix,
    maxent_only=False,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    maps = {}
    reset_days = []
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        loaded = next(
            (
                trend_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (trend_map := load_trend_rate_map(output_dir, sca)) is not None
            ),
            None,
        )
        maps[sca] = None if loaded is None else loaded[0]
        if loaded is not None:
            reset_days.append(loaded[1])
    if reset_days and not np.allclose(
        reset_days, reset_days[0], rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"Production outputs use inconsistent reset days: {reset_days}")
    return maps, (reset_days[0] if reset_days else None)


def load_all_last_48h_maps(
    root,
    fit_suffix,
    maxent_only=False,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    maps = {}
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        maps[sca] = next(
            (
                rate_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (rate_map := load_last_48h_rate_map(output_dir, sca)) is not None
            ),
            None,
        )
    return maps


def load_all_rms_maps(
    root,
    fit_suffix,
    maxent_only=False,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    maps = {}
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        maps[sca] = next(
            (
                rms_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (rms_map := load_rms_map(output_dir, sca)) is not None
            ),
            None,
        )
    return maps


def load_rate_histogram(output_dir, expected_sca):
    fit_root = Path(output_dir) / "fits"
    entries = tile_directory_entries(fit_root)
    try:
        complete_tile_ids(tile_id for tile_id, _path in entries)
    except ValueError:
        return None
    fit_dirs = [path for _tile_id, path in entries]

    pre_values = []
    post_values = []
    current_values = []
    reset_days = []
    for fit_dir in fit_dirs:
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        with np.load(posterior_path, allow_pickle=False) as samples:
            days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            rates = np.asarray(
                samples["tile_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
            reset_values = np.asarray(samples["ice_reset_day"], dtype=float).reshape(-1)
        if reset_values.size != 1 or not np.isfinite(reset_values[0]):
            return None
        if rates.size != days.size:
            raise ValueError(f"Unexpected tile-rate shape in {posterior_path}")
        reset_day = float(reset_values[0])
        before = days < reset_day
        after = days >= reset_day
        if not np.any(before) or not np.any(after):
            raise ValueError(f"Reset day does not split observations in {posterior_path}")
        pre_values.append(rates[before])
        post_values.append(rates[after])
        current_values.append(rates[int(np.argmax(days))])
        reset_days.append(reset_day)

    if not np.allclose(reset_days, reset_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Independent outputs use inconsistent reset days in {output_dir}")
    pre_values = np.concatenate(pre_values)
    post_values = np.concatenate(post_values)
    current_values = np.asarray(current_values, dtype=float)
    pre_values = pre_values[np.isfinite(pre_values)]
    post_values = post_values[np.isfinite(post_values)]
    current_values = current_values[np.isfinite(current_values)]
    if pre_values.size == 0 or post_values.size == 0 or current_values.size == 0:
        return None
    return {
        "detector": expected_sca,
        "pre_rates": pre_values,
        "post_rates": post_values,
        "current_rate": float(np.median(current_values)),
        "reset_day": reset_days[0],
    }


def load_all_rate_histograms(
    root,
    fit_suffix,
    maxent_only=False,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
    fit_root=None,
):
    histograms = {}
    reset_days = []
    for sca in SCAS:
        output_dirs = detector_output_dirs(
            root, sca, fit_suffix, tile_size, fit_subdir, fit_root
        )
        loaded = next(
            (
                histogram
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (histogram := load_rate_histogram(output_dir, sca)) is not None
            ),
            None,
        )
        histograms[sca] = loaded
        if loaded is not None:
            reset_days.append(loaded["reset_day"])
    if reset_days and not np.allclose(
        reset_days, reset_days[0], rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"Production outputs use inconsistent reset days: {reset_days}")
    return histograms, (reset_days[0] if reset_days else None)


def _statistics_paths(statistics_path, tile_size=None):
    if statistics_path.is_file():
        return [statistics_path]
    elif statistics_path.is_dir():
        pattern = "wfi*_tile_log_statistics_*.csv"
        if tile_size is not None:
            pattern = f"wfi*_tile_log_statistics_{tile_size}*.csv"
        return sorted(statistics_path.glob(pattern))
    else:
        return []


def latest_statistics_observation(statistics_path, tile_size=None):
    paths = _statistics_paths(statistics_path, tile_size)
    if not paths:
        return "unknown"
    latest_observation = None
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                value = str(row.get("exposure_start_time", "")).strip()
                if not value:
                    continue
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                try:
                    observation = datetime.fromisoformat(value)
                except ValueError:
                    continue
                if observation.tzinfo is None:
                    observation = observation.replace(tzinfo=timezone.utc)
                observation = observation.astimezone(timezone.utc)
                if latest_observation is None or observation > latest_observation:
                    latest_observation = observation
    if latest_observation is None:
        return "unknown"
    formatted = latest_observation.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    return f"{formatted} UTC"


def latest_statistics_date(statistics_path, tile_size=None):
    observation = latest_statistics_observation(statistics_path, tile_size)
    return observation[:10] if observation != "unknown" else "unknown"


def draw_rate_histogram_mosaic(
    path,
    histograms,
    model_label,
    pass_label,
    date_label,
    reset_day,
    observation_label="unknown",
):
    available = [sca for sca in SCAS if histograms[sca] is not None]
    if not available:
        raise ValueError("No completed detector histograms are available")
    all_values = np.concatenate(
        [
            np.concatenate((histograms[sca]["pre_rates"], histograms[sca]["post_rates"]))
            for sca in available
        ]
    )
    rate_min = float(np.min(all_values))
    rate_max = float(np.max(all_values))
    if np.isclose(rate_min, rate_max):
        half_width = max(abs(rate_min) * 0.05, 0.1)
        bins = np.linspace(rate_min - half_width, rate_max + half_width, 21)
    else:
        bins = np.histogram_bin_edges(all_values, bins=28)

    density_max = 0.0
    for sca in available:
        histogram = histograms[sca]
        for values in (histogram["pre_rates"], histogram["post_rates"]):
            density, _ = np.histogram(values, bins=bins, density=True)
            density_max = max(density_max, float(np.max(density)))
    density_max = max(1.0, density_max * 1.12)

    figure = plt.figure(figsize=(16, 10))
    x_min = -1.0
    x_max = SCA_PITCH * 6 - 1.0
    y_min = min(COLUMN_Y_OFFSET) - 1.0
    y_max = SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0
    left, right = 0.055, 0.985
    bottom, top = 0.09, 0.78
    bottom_scas = {column[-1] for column in COLUMN_SCA}
    left_scas = {column[0] for column in COLUMN_SCA}

    for sca in SCAS:
        x, y = sca_position(sca)
        axis = figure.add_axes(
            (
                left + (x - x_min) / (x_max - x_min) * (right - left),
                bottom + (y - y_min) / (y_max - y_min) * (top - bottom),
                SCA_SIZE / (x_max - x_min) * (right - left),
                SCA_SIZE / (y_max - y_min) * (top - bottom),
            )
        )
        histogram = histograms[sca]
        if histogram is None:
            axis.set_facecolor("#e6e6e6")
            axis.patch.set_hatch("///")
            axis.patch.set_edgecolor("#333333")
            axis.text(
                0.5,
                0.5,
                f"{sca}\npending",
                ha="center",
                va="center",
                fontsize=8,
                color="#333333",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            continue

        axis.hist(
            histogram["pre_rates"],
            bins=bins,
            density=True,
            color="#0072B2",
            alpha=0.60,
        )
        axis.hist(
            histogram["post_rates"],
            bins=bins,
            density=True,
            color="#D55E00",
            alpha=0.55,
        )
        axis.axvline(histogram["current_rate"], color="black", linewidth=1.4)
        axis.text(
            0.96,
            0.94,
            f"now {histogram['current_rate']:.2f}",
            ha="right",
            va="top",
            fontsize=6.5,
            color="black",
            transform=axis.transAxes,
        )
        axis.set_xlim(rate_min, rate_max)
        axis.set_ylim(0.0, density_max)
        axis.set_title(sca, fontsize=8, pad=1.0)
        axis.tick_params(labelsize=6, length=2)
        if sca not in bottom_scas:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel("nm/day", fontsize=6, labelpad=0.0)
        if sca not in left_scas:
            axis.set_yticklabels([])

    figure.legend(
        handles=(
            Patch(facecolor="#0072B2", alpha=0.60, label="Pre-Decon2"),
            Patch(facecolor="#D55E00", alpha=0.55, label="Post-Decon2"),
            Line2D([0], [0], color="black", linewidth=1.8, label="Latest-epoch fitted median"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"Focal-plane rate distributions\n{observation_label}",
        fontsize=14,
        y=0.985,
    )
    figure.text(
        0.015,
        0.43,
        "Probability density",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.025,
        "Instantaneous ice accumulation rate (nm/day)",
        ha="center",
        fontsize=9,
    )
    figure.text(
        0.985,
        0.012,
        observation_label,
        ha="right",
        fontsize=8,
        color="0.3",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def latest_pass_label(statistics_path, tile_size=None):
    paths = _statistics_paths(statistics_path, tile_size)
    if not paths:
        return "Pass unavailable"
    latest_repeat = -1
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                latest_repeat = max(latest_repeat, int(row["repeat"]))
    if latest_repeat < 0:
        return "Pass unavailable"
    return f"Pass {latest_repeat + 1} (repeat index {latest_repeat})"


def draw_rate_mosaic(
    path,
    maps,
    model_label,
    pass_label,
    local_contrast=False,
    metrics=METRICS,
    title_prefix="Focal-plane accumulation rates",
    observation_label="unknown",
    suptitle_fontsize=14,
    axis_title_fontsize=12,
    color_limits=None,
    color_limits_by_field=None,
    colorbar_label=None,
    colorbar_labels=None,
    metric_unit_label="nm/day",
    metric_unit_labels=None,
):
    available_by_field = {
        field: [
            sca
            for sca in SCAS
            if maps[sca] is not None
            and field in maps[sca]
            and np.isfinite(maps[sca][field]).any()
        ]
        for _title, field in metrics
    }
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(6 * len(metrics), 6),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    if not local_contrast:
        if color_limits is not None and color_limits_by_field is not None:
            raise ValueError("Specify color_limits or color_limits_by_field, not both")
        if color_limits_by_field is not None:
            limits_by_field = {}
            for _title, field in metrics:
                if field not in color_limits_by_field:
                    raise ValueError(f"Missing color limits for {field}")
                lower, upper = map(float, color_limits_by_field[field])
                if not lower < upper:
                    raise ValueError(f"color limits must increase for {field}")
                limits_by_field[field] = (lower, upper)
        elif color_limits is None:
            all_values = [
                maps[sca][field][np.isfinite(maps[sca][field])]
                for _title, field in metrics
                for sca in available_by_field[field]
            ]
            if all_values:
                all_values = np.concatenate(all_values)
                global_min = float(np.min(all_values))
                global_max = float(np.max(all_values))
            else:
                global_min, global_max = 0.0, 1.0
            if global_min == global_max:
                global_max = global_min + 1.0
        else:
            global_min, global_max = map(float, color_limits)
            if not global_min < global_max:
                raise ValueError("color_limits must be an increasing pair")

    for axis, (title, field) in zip(axes, metrics):
        available = available_by_field[field]
        plotted = {}
        if local_contrast:
            for sca in available:
                values = maps[sca][field]
                finite = values[np.isfinite(values)]
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                plotted[sca] = np.zeros_like(values) + 0.5 if vmin == vmax else (values - vmin) / (vmax - vmin)
            color_min, color_max = 0.0, 1.0
        elif color_limits_by_field is not None:
            color_min, color_max = limits_by_field[field]
            plotted = {sca: maps[sca][field] for sca in available}
        else:
            color_min = global_min
            color_max = global_max
            plotted = {sca: maps[sca][field] for sca in available}
        image = None
        for sca in SCAS:
            x, y = sca_position(sca)
            if sca not in available:
                axis.add_patch(
                    Rectangle(
                        (x, y),
                        SCA_SIZE,
                        SCA_SIZE,
                        facecolor="#e6e6e6",
                        edgecolor="#333333",
                        hatch="///",
                        linewidth=1.2,
                    )
                )
                axis.text(
                    x + SCA_SIZE / 2,
                    y + SCA_SIZE / 2,
                    f"{sca}\npending",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#333333",
                )
                continue
            image = axis.imshow(
                plotted[sca],
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                extent=(x, x + SCA_SIZE, y, y + SCA_SIZE),
            )
            axis.add_patch(
                Rectangle(
                    (x, y),
                    SCA_SIZE,
                    SCA_SIZE,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.0,
                )
            )
            axis.text(
                x + SCA_SIZE / 2,
                y + SCA_SIZE / 2,
                sca,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                path_effects=None,
            )
            tile_grid_size = plotted[sca].shape[0]
            for tile in range(1, tile_grid_size):
                position = SCA_SIZE * tile / tile_grid_size
                axis.plot(
                    (x + position, x + position),
                    (y, y + SCA_SIZE),
                    color="white",
                    alpha=0.35,
                    linewidth=0.35,
                )
                axis.plot(
                    (x, x + SCA_SIZE),
                    (y + position, y + position),
                    color="white",
                    alpha=0.35,
                    linewidth=0.35,
                )
        if image is not None:
            if local_contrast:
                label = "Within-SCA relative rate (min to max)"
            elif colorbar_labels is not None:
                label = colorbar_labels.get(field, colorbar_label or "nm/day")
            else:
                label = colorbar_label or "nm/day"
            figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label=label)
        axis.set_aspect("equal")
        axis.set_xlim(-1.0, SCA_PITCH * 6 - 1.0)
        min_y = min(COLUMN_Y_OFFSET) - 1.0
        max_y = SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0
        axis.set_ylim(min_y, max_y)
        axis.set_xticks([])
        axis.set_yticks([])
        if local_contrast:
            axis_title = f"{title} (within-SCA contrast)"
        else:
            unit = (
                metric_unit_labels.get(field, metric_unit_label)
                if metric_unit_labels is not None
                else metric_unit_label
            )
            axis_title = f"{title} ({unit})" if unit else title
        axis.set_title(axis_title, fontsize=axis_title_fontsize)
        axis.set_xlabel("Roman WFI focal-plane mosaic")
        center_x = SCA_PITCH * 3 - 0.5
        center_y = SCA_PITCH * 2 + COLUMN_Y_OFFSET[2] + 1.2
        axis.plot(
            center_x,
            center_y,
            marker="x",
            color="#d62728",
            markersize=7,
            markeredgewidth=1.5,
        )
        axis.text(
            center_x + 0.3,
            center_y + 0.5,
            "WFI_CEN",
            color="#d62728",
            fontsize=7,
            va="bottom",
        )

    title = title_prefix
    if observation_label != "unknown":
        title = f"{title_prefix}\n{observation_label}"
    figure.suptitle(title, fontsize=suptitle_fontsize)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--fit-suffix",
        default=DEFAULT_FIT_SUFFIX,
        help="Use 'production' for the fixed input-directory layout or a legacy suffix.",
    )
    parser.add_argument(
        "--fit-subdir",
        type=Path,
        default=None,
        help="Read fits from a subdirectory below each detector output root.",
    )
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=None,
        help="Directory containing detector subdirectories for an isolated fit tree.",
    )
    parser.add_argument(
        "--tile-statistics",
        type=Path,
        default=None,
        help="Statistics CSV or directory used to derive the latest pass label.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--aom-rate-jump-output",
        type=Path,
        default=None,
        help=(
            "Also write a focal-plane map of latest post-AOM versus last "
            "pre-AOM rate changes in percent."
        ),
    )
    parser.add_argument(
        "--maxent-only",
        action="store_true",
        help="Use only complete per-tile MaxEnt production archives; reject stale aggregates.",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_integer,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile edge length used by the production reduction; default: {DEFAULT_TILE_SIZE}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fit_root is not None and args.fit_subdir is not None:
        raise ValueError("--fit-root and --fit-subdir cannot be combined")
    fit_root = args.fit_root.resolve() if args.fit_root is not None else None
    fit_subdir = args.fit_subdir
    output_root = (
        fit_root
        if fit_root is not None
        else production_output_root(args.root, args.tile_size)
        / (fit_subdir if fit_subdir is not None else Path())
    )
    output = args.output or output_root / "wfi_full_focal_plane_rate_maps.png"
    maps = load_all_maps(
        args.root,
        args.fit_suffix,
        args.maxent_only,
        args.tile_size,
        fit_subdir,
        fit_root,
    )
    current_thickness_maps = load_all_current_ice_thickness_maps(
        args.root,
        args.fit_suffix,
        args.tile_size,
        fit_subdir,
        fit_root,
    )
    initial_thickness_maps, initial_thickness_metrics = load_all_initial_thickness_maps(
        args.root,
        args.fit_suffix,
        args.tile_size,
        fit_subdir,
        fit_root,
        return_metrics=True,
    )
    reset_maps, reset_day = load_all_reset_maps(
        args.root,
        args.fit_suffix,
        args.maxent_only,
        args.tile_size,
        fit_subdir,
        fit_root,
    )
    trend_maps, trend_day = load_all_trend_maps(
        args.root,
        args.fit_suffix,
        args.maxent_only,
        args.tile_size,
        fit_subdir,
        fit_root,
    )
    last_48h_maps = load_all_last_48h_maps(
        args.root,
        args.fit_suffix,
        args.maxent_only,
        args.tile_size,
        fit_subdir,
        fit_root,
    )
    rms_maps = load_all_rms_maps(
        args.root,
        args.fit_suffix,
        args.maxent_only,
        args.tile_size,
        fit_subdir,
        fit_root,
    )
    statistics_path = args.tile_statistics
    if statistics_path is None:
        statistics_path = args.root
    elif not statistics_path.is_absolute():
        statistics_path = args.root / statistics_path
    pass_label = latest_pass_label(statistics_path, args.tile_size)
    date_label = latest_statistics_date(statistics_path, args.tile_size)
    observation_label = latest_observation_label(statistics_path, args.tile_size)
    if fit_root is not None:
        model_label = "isolated independent per-tile production fits"
    elif fit_subdir is not None:
        model_label = "cached fixed-optics independent ice-only fits"
    elif args.fit_suffix == DEFAULT_FIT_SUFFIX:
        model_label = (
            "independent per-tile MaxEnt MAP fits"
            if args.maxent_only
            else "independent per-tile non-parametric MAP fits"
        )
    elif "production_sharedAR_tileRate_twoCommonACM" in args.fit_suffix:
        model_label = (
            "constant-rate modulation; tile-specific rate scales; "
            "shared AR and common ACM steps"
        )
    elif "np_ledacm" in args.fit_suffix:
        model_label = "square-root + NP modulation"
    else:
        model_label = "fully nonparametric trajectory"
    draw_rate_mosaic(
        output,
        maps,
        model_label,
        pass_label,
        observation_label=observation_label,
    )
    if args.aom_rate_jump_output is not None:
        aom_jump_maps = load_all_aom_rate_jump_maps(
            args.root,
            args.fit_suffix,
            args.tile_size,
            fit_subdir,
            fit_root,
        )
        if any(value is not None for value in aom_jump_maps.values()):
            draw_rate_mosaic(
                args.aom_rate_jump_output,
                aom_jump_maps,
                model_label,
                pass_label,
                metrics=AOM_JUMP_METRICS,
                title_prefix="AOM rate change after 01:43 UTC bump",
                observation_label=observation_label,
                color_limits_by_field={
                    field: percentile_color_limits(
                        aom_jump_maps, field, AOM_JUMP_VMAX_PERCENTILE
                    )
                    for _title, field in AOM_JUMP_METRICS
                },
                colorbar_labels={
                    "aom_rate_jump_nm_day": (
                        "Rate jump (nm/day; "
                        f"vmax={AOM_JUMP_VMAX_PERCENTILE:g}th percentile)"
                    ),
                    "aom_rate_jump_percent": (
                        "Rate change (%; "
                        f"vmax={AOM_JUMP_VMAX_PERCENTILE:g}th percentile)"
                    ),
                },
                metric_unit_labels={
                    "aom_rate_jump_nm_day": "nm/day",
                    "aom_rate_jump_percent": "%",
                },
                suptitle_fontsize=9,
                axis_title_fontsize=9,
            )
            print(f"Wrote {args.aom_rate_jump_output}")
        else:
            print("No complete AOM rate-jump maps found")
    initial_thickness_output = output.with_name(
        "wfi_full_focal_plane_initial_thicknesses.png"
    )
    draw_rate_mosaic(
        initial_thickness_output,
        initial_thickness_maps,
        model_label,
        pass_label,
        metrics=initial_thickness_metrics,
        title_prefix="Initial ice thickness by decon stage",
        observation_label=observation_label,
        color_limits=initial_thickness_color_limits(
            initial_thickness_maps, metrics=initial_thickness_metrics
        ),
        colorbar_label="Initial ice thickness (nm)",
        metric_unit_label="nm",
        suptitle_fontsize=10,
        axis_title_fontsize=10,
    )
    print(f"Wrote {initial_thickness_output}")
    contrast_output = output.with_name(output.stem + "_tile_contrast.png")
    draw_rate_mosaic(
        contrast_output,
        maps,
        model_label,
        pass_label,
        local_contrast=True,
        observation_label=observation_label,
    )
    if any(value is not None for value in current_thickness_maps.values()):
        current_thickness_output = output.with_name(
            "wfi_full_focal_plane_current_ice_thickness.png"
        )
        draw_rate_mosaic(
            current_thickness_output,
            current_thickness_maps,
            model_label,
            pass_label,
            metrics=CURRENT_THICKNESS_METRICS,
            title_prefix=(
                "Current fitted ice thickness "
                f"(vmax={CURRENT_THICKNESS_VMAX_PERCENTILE:g}th percentile)"
            ),
            observation_label=observation_label,
            color_limits=percentile_color_limits(
                current_thickness_maps,
                "current_ice_thickness_nm",
                CURRENT_THICKNESS_VMAX_PERCENTILE,
            ),
            colorbar_label=(
                "Ice thickness (nm; "
                f"vmax={CURRENT_THICKNESS_VMAX_PERCENTILE:g}th percentile)"
            ),
            metric_unit_label="nm",
            suptitle_fontsize=10,
            axis_title_fontsize=10,
        )
        print(f"Wrote {current_thickness_output}")
    else:
        print("No complete current ice-thickness maps found")
    histogram_maps, histogram_reset_day = load_all_rate_histograms(
        args.root,
        args.fit_suffix,
        args.maxent_only,
        args.tile_size,
        fit_subdir,
        fit_root,
    )
    if histogram_reset_day is not None:
        histogram_output = output.with_name(
            f"wfi_full_focal_plane_rate_histograms_{date_label}.png"
        )
        draw_rate_histogram_mosaic(
            histogram_output,
            histogram_maps,
            model_label,
            pass_label,
            date_label,
            histogram_reset_day,
            observation_label=observation_label,
        )
        print(f"Wrote {histogram_output}")
    else:
        print("No complete detector rate histograms found")
    if reset_day is not None:
        reset_output = output.with_name(output.stem + "_pre_post_reset.png")
        draw_rate_mosaic(
            reset_output,
            reset_maps,
            model_label,
            pass_label,
            metrics=RESET_METRICS,
            observation_label=observation_label,
            title_prefix="Rate by Decon2 epoch",
        )
        reset_contrast_output = reset_output.with_name(
            reset_output.stem + "_tile_contrast.png"
        )
        draw_rate_mosaic(
            reset_contrast_output,
            reset_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=RESET_METRICS,
            observation_label=observation_label,
            title_prefix="Rate by Decon2 epoch",
        )
        print(f"Wrote {reset_output}")
        print(f"Wrote {reset_contrast_output}")
    else:
        print("No reset-aware posterior maps found")
    if trend_day is not None:
        trend_output = output.with_name(
            "wfi_full_focal_plane_trend_rates_pre_post.png"
        )
        draw_rate_mosaic(
            trend_output,
            trend_maps,
            model_label,
            pass_label,
            metrics=TREND_METRICS,
            observation_label=observation_label,
            title_prefix="Trend rate by Decon2 epoch",
            color_limits=TREND_RATE_COLOR_LIMITS,
            colorbar_label="Growth rate nm/day",
        )
        trend_contrast_output = trend_output.with_name(
            trend_output.stem + "_tile_contrast.png"
        )
        draw_rate_mosaic(
            trend_contrast_output,
            trend_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=TREND_METRICS,
            observation_label=observation_label,
            title_prefix="Trend rate by Decon2 epoch",
            color_limits=TREND_RATE_COLOR_LIMITS,
            colorbar_label="Growth rate nm/day",
        )
        print(f"Wrote {trend_output}")
        print(f"Wrote {trend_contrast_output}")
    else:
        print("No per-tile trend-rate maps found")
    last_48h_available = [sca for sca in SCAS if last_48h_maps[sca] is not None]
    if last_48h_available:
        last_48h_output = output.with_name(
            "wfi_full_focal_plane_mean_accumulation_rate_last_48h.png"
        )
        draw_rate_mosaic(
            last_48h_output,
            last_48h_maps,
            model_label,
            pass_label,
            metrics=LAST_48H_METRICS,
            observation_label=observation_label,
            title_prefix=(
                "WFI focal-plane mean accumulation rate, last 48 h"
            ),
            suptitle_fontsize=10,
        )
        last_48h_contrast_output = last_48h_output.with_name(
            last_48h_output.stem + "_tile_contrast.png"
        )
        draw_rate_mosaic(
            last_48h_contrast_output,
            last_48h_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=LAST_48H_METRICS,
            observation_label=observation_label,
            title_prefix=(
                "WFI focal-plane mean accumulation rate, last 48 h"
            ),
            suptitle_fontsize=10,
        )
        print(f"Wrote {last_48h_output}")
        print(f"Wrote {last_48h_contrast_output}")
    else:
        print("No complete last-48-hour rate maps found")
    if any(rms_maps[sca] is not None for sca in SCAS):
        rms_output = output.with_name("wfi_full_focal_plane_rms_residual.png")
        draw_rate_mosaic(
            rms_output,
            rms_maps,
            model_label,
            pass_label,
            metrics=RMS_METRICS,
            observation_label=observation_label,
            title_prefix="Focal-plane RMS residual",
            colorbar_label="RMS residual (millilog)",
            metric_unit_label="millilog",
        )
        print(f"Wrote {rms_output}")
    else:
        print("No complete RMS residual maps found")
    available = [sca for sca in SCAS if maps[sca] is not None]
    pending = [sca for sca in SCAS if maps[sca] is None]
    print(f"Wrote {output}")
    print(f"Wrote {contrast_output}")
    print(f"Available SCAs: {', '.join(available)}")
    if pending:
        print(f"Pending or invalid SCAs: {', '.join(pending)}")


if __name__ == "__main__":
    main()
