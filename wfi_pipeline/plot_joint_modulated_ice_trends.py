"""Plot focal-plane and detector ice trends from the joint-modulated bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PLOT_ROOT = ROOT / "archive" / "exploratory" / "legacy_root_scripts"
for import_root in (ROOT / "wfi_pipeline", ROOT, ARCHIVE_PLOT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import plot_wfi_full_focal_plane_rates as focal_plotter
import plot_wfi_all_detector_trajectories as trajectory_plotter
from plot_wfi_independent_production import (
    save_independent_trajectory_plot,
    save_rate_histogram,
)
from wfi_independent_plot_helpers import save_np_node_plot
from wfi_pipeline_defaults import DETECTORS, TILE_SIZE_PIXELS
from wfi_production_layout import corrected_statistics_path
from wfi_production_plot_annotations import load_time_annotations
from wfi_wavelength_response_bundle import MODULATED_BUNDLE_VERSION


JOINT_TRAJECTORY_PREFIX = "joint CAR190 + Amber MAP thickness"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_joint_modulated_bundle(bundle):
    metadata = bundle.get("metadata", {})
    trajectory_definition = str(metadata.get("trajectory_definition", ""))
    if not trajectory_definition.startswith(JOINT_TRAJECTORY_PREFIX):
        raise ValueError(
            "Trend plots require the joint CAR190 + Amber ice trajectory; "
            f"bundle records {trajectory_definition!r}"
        )
    if not metadata.get("joint_ice_history_fit_roots"):
        raise ValueError("Joint-modulated bundle has no joint ice-history provenance")
    if not bool(bundle.get("clean_reflectance_modulation_included", False)):
        raise ValueError("Trend plots require a clean-reflectance modulated bundle")

    detector_values = np.asarray(bundle["detectors"]).astype(str)
    tile_ids = np.asarray(bundle["tile_ids"], dtype=int).reshape(-1)
    days = np.asarray(bundle["observation_days"], dtype=float).reshape(-1)
    thickness = np.asarray(bundle["tile_ice_thickness"], dtype=float)
    rates = np.asarray(bundle["tile_instantaneous_rate_nm_day"], dtype=float)
    if detector_values.shape != tile_ids.shape:
        raise ValueError("Bundle detector and tile ID arrays have different shapes")
    if thickness.shape != (tile_ids.size, days.size) or rates.shape != thickness.shape:
        raise ValueError("Joint-modulated bundle has inconsistent trajectory dimensions")
    if set(detector_values.tolist()) != set(DETECTORS):
        raise ValueError("Joint-modulated bundle must contain all 18 detectors")
    for detector in DETECTORS:
        mask = detector_values == detector
        if int(np.count_nonzero(mask)) != 256:
            raise ValueError(f"{detector}: expected 256 joint-modulated tile histories")
        if set(tile_ids[mask].tolist()) != set(range(256)):
            raise ValueError(f"{detector}: joint-modulated tile IDs are incomplete")
    if days.size < 2 or np.any(np.diff(days) < 0.0):
        raise ValueError("Joint-modulated bundle has an invalid observation time axis")
    if not np.isfinite(thickness).all() or not np.isfinite(rates).all():
        raise ValueError("Joint-modulated histories contain nonfinite values")
    return metadata


def load_joint_modulated_plot_bundle(bundle_path):
    """Load only trajectory/geometry fields, avoiding the large spectral tensors."""
    required = (
        "metadata_json",
        "bundle_version",
        "detectors",
        "tile_ids",
        "tile_rows",
        "tile_cols",
        "observation_days",
        "observation_led",
        "observation_repeat",
        "tile_ice_thickness",
        "tile_instantaneous_rate_nm_day",
        "ice_reset_day",
    )
    optional = (
        "reset_event_days",
        "reset_event_names",
        "third_decon_day",
    )
    with np.load(bundle_path, allow_pickle=False) as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"Joint-modulated bundle is missing fields: {missing}")
        version = int(np.asarray(source["bundle_version"]).reshape(-1)[0])
        if version != MODULATED_BUNDLE_VERSION:
            raise ValueError(
                f"Expected modulated bundle version {MODULATED_BUNDLE_VERSION}, got {version}"
            )
        bundle = {name: np.asarray(source[name]) for name in required}
        bundle.update(
            {name: np.asarray(source[name]) for name in optional if name in source}
        )
    bundle["metadata"] = json.loads(
        str(np.asarray(bundle.pop("metadata_json")).reshape(-1)[0])
    )
    bundle["clean_reflectance_modulation_included"] = True
    return bundle


def extract_detector_history(bundle, detector):
    detector = str(detector).upper()
    if detector not in DETECTORS:
        raise ValueError(f"Unknown WFI detector: {detector}")
    detector_values = np.asarray(bundle["detectors"]).astype(str)
    selected = np.flatnonzero(detector_values == detector)
    order = np.argsort(np.asarray(bundle["tile_ids"])[selected])
    selected = selected[order]
    return {
        "detector": detector,
        "tile_ids": np.asarray(bundle["tile_ids"], dtype=int)[selected],
        "tile_rows": np.asarray(bundle["tile_rows"], dtype=int)[selected],
        "tile_cols": np.asarray(bundle["tile_cols"], dtype=int)[selected],
        "days": np.asarray(bundle["observation_days"], dtype=float).reshape(-1),
        "leds": np.asarray(bundle["observation_led"], dtype=int).reshape(-1),
        "repeats": np.asarray(bundle["observation_repeat"], dtype=int).reshape(-1),
        "thickness": np.asarray(bundle["tile_ice_thickness"], dtype=float)[selected],
        "rates": np.asarray(bundle["tile_instantaneous_rate_nm_day"], dtype=float)[selected],
    }


def _reset_metadata(bundle):
    if "reset_event_days" in bundle:
        reset_days = np.asarray(bundle["reset_event_days"], dtype=float).reshape(-1)
    else:
        reset_days = [float(np.asarray(bundle["ice_reset_day"]).reshape(-1)[0])]
        if "third_decon_day" in bundle:
            reset_days.append(float(np.asarray(bundle["third_decon_day"]).reshape(-1)[0]))
        reset_days = np.asarray(reset_days, dtype=float)
    if "reset_event_names" in bundle:
        reset_names = np.asarray(bundle["reset_event_names"]).astype(str).reshape(-1)
    else:
        reset_names = np.asarray(
            ["second_decon"] + ["third_decon"] * max(0, reset_days.size - 1)
        )
    if (
        reset_days.size == 0
        or reset_names.shape != reset_days.shape
        or np.any(np.diff(reset_days) <= 0.0)
    ):
        raise ValueError("Joint-modulated bundle has invalid reset metadata")
    return reset_days, reset_names


def _tile_map(values, rows, cols):
    grid_size = max(int(np.max(rows)), int(np.max(cols))) + 1
    result = np.full((grid_size, grid_size), np.nan, dtype=float)
    result[rows, cols] = np.asarray(values, dtype=float)
    return result


def _segment_mask(days, left, right=None):
    mask = np.asarray(days, dtype=float) >= float(left)
    if right is not None:
        mask &= np.asarray(days, dtype=float) < float(right)
    return mask


def _segment_rate(days, history, mask):
    selected_days = np.asarray(days, dtype=float)[mask]
    selected_history = np.asarray(history, dtype=float)[:, mask]
    if selected_days.size < 2:
        return np.full(selected_history.shape[0], np.nan)
    duration = float(selected_days[-1] - selected_days[0])
    if duration <= 0.0:
        return np.full(selected_history.shape[0], np.nan)
    return (selected_history[:, -1] - selected_history[:, 0]) / duration


def _segment_trend(days, history, mask):
    selected_days = np.asarray(days, dtype=float)[mask]
    selected_history = np.asarray(history, dtype=float)[:, mask]
    if selected_days.size < 2:
        return np.full(selected_history.shape[0], np.nan)
    return np.polyfit(selected_days, selected_history.T, 1)[0]


def _episode_start_thickness(days, history, reset_days):
    starts = [0]
    for reset_day in reset_days:
        candidates = np.flatnonzero(np.asarray(days, dtype=float) >= reset_day)
        if candidates.size:
            starts.append(int(candidates[0]))
    values = np.full((history.shape[0], 3), np.nan, dtype=float)
    for stage, index in enumerate(starts[: values.shape[1]]):
        values[:, stage] = history[:, index]
    return values


def _aom_event_day(statistics_path):
    annotations = load_time_annotations(statistics_path)
    return next(
        (
            float(event.day)
            for event in annotations.events
            if event.label == focal_plotter.AOM_TEMPERATURE_BUMP_LABEL
        ),
        None,
    )


def _detector_focal_maps(data, reset_days, aom_day=None):
    days = data["days"]
    history = data["thickness"]
    rates = data["rates"]
    rows = data["tile_rows"]
    cols = data["tile_cols"]
    reset_day = float(reset_days[0])
    next_reset = float(reset_days[1]) if reset_days.size > 1 else None
    pre_reset = days < reset_day
    post_reset = _segment_mask(days, reset_day, next_reset)

    result = {
        "rates": {
            "initial_rate_nm_day_median": _tile_map(rates[:, 0], rows, cols),
            "final_rate_nm_day_median": _tile_map(rates[:, -1], rows, cols),
            "mean_rate_nm_day_median": _tile_map(np.mean(rates, axis=1), rows, cols),
        },
        "current_thickness": {
            "current_ice_thickness_nm": _tile_map(history[:, -1], rows, cols),
        },
        "initial_thickness": {},
        "reset_rates": {},
        "trend_rates": {},
        "last_48h": {},
        "aom_jump": None,
        "rate_histogram": {
            "detector": data["detector"],
            "pre_rates": rates[:, pre_reset].reshape(-1),
            "post_rates": rates[:, ~pre_reset].reshape(-1),
            "current_rate": float(np.median(rates[:, -1])),
            "reset_day": reset_day,
        },
    }
    initial_values = _episode_start_thickness(days, history, reset_days)
    for column, (_label, field) in enumerate(focal_plotter.INITIAL_THICKNESS_METRICS):
        result["initial_thickness"][field] = _tile_map(
            initial_values[:, column], rows, cols
        )
    result["reset_rates"] = {
        "pre_reset_mean_rate_nm_day": _tile_map(
            _segment_rate(days, history, pre_reset), rows, cols
        ),
        "post_reset_mean_rate_nm_day": _tile_map(
            _segment_rate(days, history, post_reset), rows, cols
        ),
    }
    result["trend_rates"] = {
        "pre_reset_trend_rate_nm_day": _tile_map(
            _segment_trend(days, history, pre_reset), rows, cols
        ),
        "post_reset_trend_rate_nm_day": _tile_map(
            _segment_trend(days, history, post_reset), rows, cols
        ),
    }
    result["last_48h"] = {
        "last_48h_mean_rate_nm_day": _tile_map(
            [
                focal_plotter.mean_rate_over_interval(days, tile_history, 2.0)
                for tile_history in history
            ],
            rows,
            cols,
        )
    }
    if aom_day is not None:
        before = np.flatnonzero(days < aom_day)
        after = np.flatnonzero(days > aom_day)
        if before.size and after.size:
            pre_rates = rates[:, before[-1]]
            post_rates = rates[:, after[-1]]
            jump = post_rates - pre_rates
            percent = np.divide(
                100.0 * jump,
                np.abs(pre_rates),
                out=np.full_like(jump, np.nan),
                where=np.abs(pre_rates) > 0.0,
            )
            result["aom_jump"] = {
                "aom_rate_jump_nm_day": _tile_map(jump, rows, cols),
                "aom_rate_jump_percent": _tile_map(percent, rows, cols),
            }
    return result


def _write_detector_trend_plots(data, reset_days, statistics_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_time_annotations(statistics_path)
    source_label = "joint-modulated CAR190 + Amber fit"
    plot_data = {
        "detector": data["detector"],
        "days": data["days"],
        "thickness": data["thickness"],
        "rates": data["rates"],
    }
    save_independent_trajectory_plot(
        output_dir / "numpyro_ice_trajectory.png",
        plot_data,
        time_annotations=annotations,
        reset_days=reset_days,
        source_label=source_label,
        rate_y_limit=30.0,
        date_tick_interval_days=7,
    )
    save_rate_histogram(
        output_dir / "numpyro_rate_histogram_pre_post_reset.png",
        plot_data,
        reset_day=float(reset_days[0]),
        source_label=source_label,
    )

    repeat_values = np.unique(data["repeats"])
    node_days = np.asarray(
        [np.mean(data["days"][data["repeats"] == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    node_thickness = np.stack(
        [
            np.mean(data["thickness"][:, data["repeats"] == repeat], axis=1)
            for repeat in repeat_values
        ],
        axis=1,
    )
    node_rates = np.stack(
        [
            np.mean(data["rates"][:, data["repeats"] == repeat], axis=1)
            for repeat in repeat_values
        ],
        axis=1,
    )
    node_samples = {
        "sca_names": np.asarray([data["detector"]]),
        "fully_np_trajectory": np.asarray(1),
        "node_instantaneous_rate_nm_day": node_rates,
        "np_node_thickness": node_thickness,
        "np_node_multipliers": np.ones_like(node_rates),
    }
    save_np_node_plot(
        output_dir / "numpyro_np_trajectory.png",
        node_samples,
        data["days"],
        data["repeats"],
        time_annotations=annotations,
        reset_days=reset_days,
        source_label=source_label,
        node_title="Joint-modulated fitted rate sampled at CAR190 observation epochs",
        thickness_title="Joint-modulated ice thickness at CAR190 observation epochs",
        percentile_label="Tile 16th-84th percentile",
        rate_y_limit=30.0,
        date_tick_interval_days=7,
    )
    save_joint_tile_rate_map(
        output_dir / f"numpyro_tile_rate_maps_{data['detector']}.png", data
    )
    return [
        str((output_dir / name).resolve())
        for name in (
            "numpyro_ice_trajectory.png",
            "numpyro_rate_histogram_pre_post_reset.png",
            "numpyro_np_trajectory.png",
            f"numpyro_tile_rate_maps_{data['detector']}.png",
        )
    ]


def save_joint_tile_rate_map(path, data):
    metrics = (
        ("Initial rate", data["rates"][:, 0]),
        ("Final rate", data["rates"][:, -1]),
        ("Mean rate", np.mean(data["rates"], axis=1)),
    )
    grid_size = max(int(data["tile_rows"].max()), int(data["tile_cols"].max())) + 1
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(4.8 * len(metrics), 4.6), constrained_layout=True
    )
    for axis, (title, values) in zip(np.asarray(axes).reshape(-1), metrics, strict=True):
        image = axis.imshow(
            _tile_map(values, data["tile_rows"], data["tile_cols"]),
            origin="lower",
            interpolation="none",
            cmap="viridis",
        )
        axis.set_xlabel("Tile column")
        axis.set_ylabel("Tile row")
        axis.set_title(title)
        axis.set_xticks(range(grid_size))
        axis.set_yticks(range(grid_size))
        axis.grid(color="white", alpha=0.25, linewidth=0.5)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="nm/day")
    figure.suptitle(f"{data['detector']} joint-modulated tile accumulation rates")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _draw_focal_plane_products(bundle, statistics_dir, output_dir, reset_days):
    output_dir = Path(output_dir)
    focal_root = output_dir / "focal_plane"
    trajectory_root = output_dir / "trajectories"
    recent_root = output_dir / "trajectories_last_7d"
    for directory in (focal_root, trajectory_root, recent_root):
        directory.mkdir(parents=True, exist_ok=True)
    model_label = "joint-modulated CAR190 + Amber ice histories"
    pass_label = focal_plotter.latest_pass_label(statistics_dir, TILE_SIZE_PIXELS)
    date_label = focal_plotter.latest_statistics_date(statistics_dir, TILE_SIZE_PIXELS)
    observation_label = (
        "Joint-modulated CAR190 + Amber fit; "
        + focal_plotter.latest_observation_label(statistics_dir, TILE_SIZE_PIXELS)
    )

    rate_maps = {}
    current_maps = {}
    initial_maps = {}
    reset_maps = {}
    trend_maps = {}
    recent_maps = {}
    aom_maps = {}
    histograms = {}
    trajectories = {}
    annotations = None
    reset_day = float(reset_days[0])
    for detector in DETECTORS:
        data = extract_detector_history(bundle, detector)
        statistics_path = corrected_statistics_path(
            statistics_dir, detector, TILE_SIZE_PIXELS
        )
        annotations = load_time_annotations(statistics_path)
        aom_day = next(
            (
                float(event.day)
                for event in annotations.events
                if event.label == focal_plotter.AOM_TEMPERATURE_BUMP_LABEL
            ),
            None,
        )
        maps = _detector_focal_maps(data, reset_days, aom_day)
        rate_maps[detector] = maps["rates"]
        current_maps[detector] = maps["current_thickness"]
        initial_maps[detector] = maps["initial_thickness"]
        reset_maps[detector] = maps["reset_rates"]
        trend_maps[detector] = maps["trend_rates"]
        recent_maps[detector] = maps["last_48h"]
        aom_maps[detector] = maps["aom_jump"]
        histograms[detector] = maps["rate_histogram"]
        trajectories[detector] = (
            data["days"],
            np.percentile(data["thickness"], [16, 50, 84], axis=0),
            np.percentile(data["rates"], [16, 50, 84], axis=0),
        )

    outputs = []
    primary_rate_map = focal_root / "wfi_full_focal_plane_rate_maps.png"
    focal_plotter.draw_rate_mosaic(
        primary_rate_map,
        rate_maps,
        model_label,
        pass_label,
        observation_label=observation_label,
    )
    rate_contrast_output = primary_rate_map.with_name(
        primary_rate_map.stem + "_tile_contrast.png"
    )
    focal_plotter.draw_rate_mosaic(
        rate_contrast_output,
        rate_maps,
        model_label,
        pass_label,
        local_contrast=True,
        observation_label=observation_label,
    )
    outputs.extend(
        (
            primary_rate_map,
            rate_contrast_output,
        )
    )
    current_output = focal_root / "wfi_full_focal_plane_current_ice_thickness.png"
    focal_plotter.draw_rate_mosaic(
        current_output,
        current_maps,
        model_label,
        pass_label,
        metrics=focal_plotter.CURRENT_THICKNESS_METRICS,
        title_prefix="Current joint-modulated ice thickness",
        observation_label=observation_label,
        color_limits=focal_plotter.percentile_color_limits(
            current_maps,
            "current_ice_thickness_nm",
            focal_plotter.CURRENT_THICKNESS_VMAX_PERCENTILE,
        ),
        colorbar_label="nm",
        metric_unit_label="nm",
    )
    outputs.append(current_output)
    initial_output = focal_root / "wfi_full_focal_plane_initial_thicknesses.png"
    focal_plotter.draw_rate_mosaic(
        initial_output,
        initial_maps,
        model_label,
        pass_label,
        metrics=focal_plotter.INITIAL_THICKNESS_METRICS,
        title_prefix="Joint-modulated thickness at each reset episode start",
        observation_label=observation_label,
        color_limits=focal_plotter.initial_thickness_color_limits(initial_maps),
        colorbar_label="nm",
        metric_unit_label="nm",
    )
    outputs.append(initial_output)
    reset_output = focal_root / "wfi_full_focal_plane_rate_maps_pre_post_reset.png"
    focal_plotter.draw_rate_mosaic(
        reset_output,
        reset_maps,
        model_label,
        pass_label,
        metrics=focal_plotter.RESET_METRICS,
        title_prefix="Joint-modulated accumulation rates around Decon2",
        observation_label=observation_label,
    )
    reset_contrast_output = reset_output.with_name(
        reset_output.stem + "_tile_contrast.png"
    )
    focal_plotter.draw_rate_mosaic(
        reset_contrast_output,
        reset_maps,
        model_label,
        pass_label,
        local_contrast=True,
        metrics=focal_plotter.RESET_METRICS,
        title_prefix="Joint-modulated accumulation rates around Decon2",
        observation_label=observation_label,
    )
    outputs.extend(
        (
            reset_output,
            reset_contrast_output,
        )
    )
    trend_output = focal_root / "wfi_full_focal_plane_trend_rates_pre_post.png"
    focal_plotter.draw_rate_mosaic(
        trend_output,
        trend_maps,
        model_label,
        pass_label,
        metrics=focal_plotter.TREND_METRICS,
        title_prefix="Joint-modulated ice-thickness trends around Decon2",
        observation_label=observation_label,
    )
    trend_contrast_output = trend_output.with_name(
        trend_output.stem + "_tile_contrast.png"
    )
    focal_plotter.draw_rate_mosaic(
        trend_contrast_output,
        trend_maps,
        model_label,
        pass_label,
        local_contrast=True,
        metrics=focal_plotter.TREND_METRICS,
        title_prefix="Joint-modulated ice-thickness trends around Decon2",
        observation_label=observation_label,
    )
    outputs.extend(
        (
            trend_output,
            trend_contrast_output,
        )
    )
    last48_output = focal_root / "wfi_full_focal_plane_mean_accumulation_rate_last_48h.png"
    focal_plotter.draw_rate_mosaic(
        last48_output,
        recent_maps,
        model_label,
        pass_label,
        metrics=focal_plotter.LAST_48H_METRICS,
        title_prefix="Joint-modulated mean accumulation rate, last 48 h",
        observation_label=observation_label,
    )
    last48_contrast_output = last48_output.with_name(
        last48_output.stem + "_tile_contrast.png"
    )
    focal_plotter.draw_rate_mosaic(
        last48_contrast_output,
        recent_maps,
        model_label,
        pass_label,
        local_contrast=True,
        metrics=focal_plotter.LAST_48H_METRICS,
        title_prefix="Joint-modulated mean accumulation rate, last 48 h",
        observation_label=observation_label,
    )
    outputs.extend(
        (
            last48_output,
            last48_contrast_output,
        )
    )
    if any(item is not None for item in aom_maps.values()):
        aom_output = focal_root / "wfi_full_focal_plane_aom_rate_jump_percent.png"
        focal_plotter.draw_rate_mosaic(
            aom_output,
            aom_maps,
            model_label,
            pass_label,
            metrics=focal_plotter.AOM_JUMP_METRICS,
            title_prefix="Joint-modulated rate change after the AOM temperature bump",
            observation_label=observation_label,
            color_limits_by_field={
                field: focal_plotter.percentile_color_limits(
                    aom_maps, field, focal_plotter.AOM_JUMP_VMAX_PERCENTILE
                )
                for _title, field in focal_plotter.AOM_JUMP_METRICS
            },
            metric_unit_labels={
                "aom_rate_jump_nm_day": "nm/day",
                "aom_rate_jump_percent": "%",
            },
        )
        outputs.append(aom_output)

    histogram_output = focal_root / f"wfi_joint_modulated_rate_histograms_{date_label}.png"
    focal_plotter.draw_rate_histogram_mosaic(
        histogram_output,
        histograms,
        model_label,
        pass_label,
        date_label,
        reset_day,
        observation_label=observation_label,
    )
    outputs.append(histogram_output)

    trajectory_output = trajectory_root / "wfi_all_detector_ice_rate_trajectories.png"
    trajectory_plotter.plot_all_detector_trajectories(
        trajectory_output,
        trajectories,
        reset_day=reset_day,
        break_days=reset_days[1:].tolist(),
        time_annotations=annotations,
        observation_label=observation_label,
        rate_ylim_top=30.0,
        date_tick_interval_days=7,
    )
    quantiles_output = trajectory_output.with_name(
        "wfi_all_detector_ice_rate_trajectories_quantiles.csv"
    )
    trajectory_plotter.write_quantiles(quantiles_output, trajectories)
    latest_day = float(np.max(np.asarray(bundle["observation_days"], dtype=float)))
    recent_output = recent_root / "wfi_all_detector_ice_rate_trajectories_last_7d.png"
    trajectory_plotter.plot_all_detector_trajectories(
        recent_output,
        trajectories,
        reset_day=reset_day,
        break_days=reset_days[1:].tolist(),
        time_annotations=annotations,
        observation_label=observation_label,
        xlim_days=(latest_day - 7.0, latest_day),
        rate_ylim_top=30.0,
        date_tick_interval_days=2,
    )
    outputs.extend((trajectory_output, quantiles_output, recent_output))
    return [str(Path(path).resolve()) for path in outputs]


def plot_joint_modulated_trends(bundle_path, statistics_dir, output_dir, tile_size=256):
    bundle_path = Path(bundle_path).expanduser().resolve()
    statistics_dir = Path(statistics_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    bundle = load_joint_modulated_plot_bundle(bundle_path)
    metadata = validate_joint_modulated_bundle(bundle)
    reset_days, reset_names = _reset_metadata(bundle)
    plot_outputs = _draw_focal_plane_products(
        bundle, statistics_dir, output_dir, reset_days
    )
    detector_outputs = {}
    for detector in DETECTORS:
        data = extract_detector_history(bundle, detector)
        statistics_path = corrected_statistics_path(statistics_dir, detector, tile_size)
        detector_outputs[detector] = _write_detector_trend_plots(
            data,
            reset_days,
            statistics_path,
            output_dir / "detectors" / detector.lower(),
        )
    all_outputs = plot_outputs + [
        path for paths in detector_outputs.values() for path in paths
    ]
    missing = [path for path in all_outputs if not Path(path).is_file()]
    if missing:
        raise RuntimeError("Joint-modulated trend plotting omitted: " + ", ".join(missing))
    report = {
        "schema": "joint_modulated_ice_trend_plots/1",
        "source_bundle": str(bundle_path),
        "source_bundle_sha256": sha256_file(bundle_path),
        "source_modulation_fit_family": metadata["modulation_fit_family"],
        "source_trajectory_definition": metadata["trajectory_definition"],
        "source_trajectory_rate_definition": metadata.get("trajectory_rate_definition"),
        "joint_ice_history_fit_roots": metadata["joint_ice_history_fit_roots"],
        "reset_days": reset_days.tolist(),
        "reset_event_names": reset_names.tolist(),
        "tile_count": int(np.asarray(bundle["tile_ids"]).size),
        "observation_count": int(np.asarray(bundle["observation_days"]).size),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "global_plot_outputs": plot_outputs,
        "detector_plot_outputs": detector_outputs,
        "not_generated": [
            "CAR190 group-residual maps; residual predictions are not stored in the joint-modulated bundle"
        ],
    }
    summary_path = output_dir / "joint_modulated_ice_trend_plots.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["summary_path"] = str(summary_path.resolve())
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE_PIXELS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.tile_size != TILE_SIZE_PIXELS:
        raise ValueError(f"Joint-modulated trend plotting requires {TILE_SIZE_PIXELS}-pixel tiles")
    plot_joint_modulated_trends(
        args.bundle_path, args.statistics_dir, args.output_dir, tile_size=args.tile_size
    )


if __name__ == "__main__":
    main()
