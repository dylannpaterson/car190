"""Plot focal-plane fields using only explicitly selected current detectors."""

import argparse
from pathlib import Path

import numpy as np

from plot_wfi_full_focal_plane_rates import (
    LAST_48H_METRICS,
    METRICS,
    RESET_METRICS,
    RMS_METRICS,
    SCAS,
    TREND_RATE_COLOR_LIMITS,
    TREND_METRICS,
    draw_rate_histogram_mosaic,
    draw_rate_mosaic,
    latest_pass_label,
    latest_statistics_date,
    latest_statistics_observation,
    load_last_48h_rate_map,
    load_rate_histogram,
    load_rate_map,
    load_rms_map,
    load_reset_rate_map,
    load_trend_rate_map,
)
from wfi_production_layout import (
    complete_tile_ids,
    production_output_root,
    positive_integer,
    tile_directory_entries,
    tile_grid_size_for_tile_size,
)


CURRENT_THICKNESS_VMAX_PERCENTILE = 99.0


def selected_maps(detectors, loader):
    return {detector: loader(detector) if detector in detectors else None for detector in SCAS}


def load_current_ice_thickness_map(output_dir, expected_sca, tile_size):
    entries = tile_directory_entries(Path(output_dir) / "fits")
    complete_tile_ids(tile_id for tile_id, _path in entries)
    tile_grid_size = tile_grid_size_for_tile_size(tile_size)
    values = np.full((tile_grid_size, tile_grid_size), np.nan, dtype=float)
    for tile_id, fit_dir in entries:
        posterior_path = fit_dir / "posterior_samples.npz"
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument("--detector", action="append", required=True)
    parser.add_argument(
        "--detector-output-root",
        type=Path,
        default=None,
        help="Direct directory containing detector subdirectories (wfi01, wfi02, ...).",
    )
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detectors = {value.upper() for value in args.detector}
    unknown = detectors.difference(SCAS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector_output_root = (
        Path(args.detector_output_root)
        if args.detector_output_root is not None
        else production_output_root(args.root, args.tile_size)
    )

    def detector_root(detector):
        return detector_output_root / detector.lower()

    def rate_map(detector):
        return load_rate_map(detector_root(detector) / "tile_rate_posterior.csv", detector)

    def reset_map(detector):
        loaded = load_reset_rate_map(detector_root(detector), detector)
        return None if loaded is None else loaded[0]

    def trend_map(detector):
        loaded = load_trend_rate_map(detector_root(detector), detector)
        return None if loaded is None else loaded[0]

    def rms_map(detector):
        return load_rms_map(detector_root(detector), detector)

    def current_ice_thickness_map(detector):
        return load_current_ice_thickness_map(
            detector_root(detector), detector, args.tile_size
        )

    pass_label = latest_pass_label(args.statistics_dir, args.tile_size)
    date_label = latest_statistics_date(args.statistics_dir, args.tile_size)
    observation_label = latest_statistics_observation(args.statistics_dir, args.tile_size)
    model_label = args.model_label or (
        "Fresh sigma=0.05 production MAP fits; "
        f"{len(detectors)} current detector(s)"
    )

    draw_rate_mosaic(
        args.output_dir / "wfi_full_focal_plane_rate_maps_current.png",
        selected_maps(detectors, rate_map),
        model_label,
        pass_label,
        observation_label=observation_label,
    )
    draw_rate_mosaic(
        args.output_dir / "wfi_full_focal_plane_rate_maps_current_tile_contrast.png",
        selected_maps(detectors, rate_map),
        model_label,
        pass_label,
        local_contrast=True,
        observation_label=observation_label,
    )

    current_thickness_values = {
        detector: current_ice_thickness_map(detector) for detector in detectors
    }
    if any(value is not None for value in current_thickness_values.values()):
        current_thickness_maps = {
            detector: current_thickness_values.get(detector) for detector in SCAS
        }
        draw_rate_mosaic(
            args.output_dir / "wfi_full_focal_plane_current_ice_thickness.png",
            current_thickness_maps,
            model_label,
            pass_label,
            metrics=(("Current ice thickness", "current_ice_thickness_nm"),),
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
        )

    reset_values = {detector: load_reset_rate_map(detector_root(detector), detector) for detector in detectors}
    reset_values = {detector: value for detector, value in reset_values.items() if value is not None}
    if reset_values:
        reset_day = next(iter(reset_values.values()))[1]
        reset_maps = {
            detector: value[0] if detector in reset_values else None for detector, value in
            ((detector, reset_values.get(detector)) for detector in SCAS)
        }
        reset_title = "Rate by Decon2 epoch"
        draw_rate_mosaic(
            args.output_dir / "wfi_full_focal_plane_rate_maps_current_pre_post_reset.png",
            reset_maps,
            model_label,
            pass_label,
            metrics=RESET_METRICS,
            title_prefix=reset_title,
            observation_label=observation_label,
        )
        draw_rate_mosaic(
            args.output_dir / "wfi_full_focal_plane_rate_maps_current_pre_post_reset_tile_contrast.png",
            reset_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=RESET_METRICS,
            title_prefix=reset_title,
            observation_label=observation_label,
        )

    trend_values = {detector: trend_map(detector) for detector in detectors}
    if any(value is not None for value in trend_values.values()):
        trend_maps = {detector: trend_values.get(detector) for detector in SCAS}
        draw_rate_mosaic(
            args.output_dir / "wfi_full_focal_plane_trend_rates_current_pre_post.png",
            trend_maps,
            model_label,
            pass_label,
            metrics=TREND_METRICS,
            title_prefix="Fitted trend rates",
            observation_label=observation_label,
            color_limits=TREND_RATE_COLOR_LIMITS,
            colorbar_label="Growth rate nm/day",
        )

    rms_values = {detector: rms_map(detector) for detector in detectors}
    if any(value is not None for value in rms_values.values()):
        draw_rate_mosaic(
            args.output_dir / "wfi_full_focal_plane_rms_residual_current.png",
            {detector: rms_values.get(detector) for detector in SCAS},
            model_label,
            pass_label,
            metrics=RMS_METRICS,
            title_prefix="Focal-plane RMS residual",
            observation_label=observation_label,
            colorbar_label="RMS residual (millilog)",
            metric_unit_label="millilog",
        )

    last_48_values = {
        detector: load_last_48h_rate_map(detector_root(detector), detector)
        for detector in detectors
    }
    if any(value is not None for value in last_48_values.values()):
        draw_rate_mosaic(
            args.output_dir / "wfi_full_focal_plane_mean_accumulation_rate_current_last_48h.png",
            {detector: last_48_values.get(detector) for detector in SCAS},
            model_label,
            pass_label,
            metrics=LAST_48H_METRICS,
            title_prefix="Mean accumulation rate: last 48 h",
            observation_label=observation_label,
            suptitle_fontsize=10,
        )

    histograms = {
        detector: load_rate_histogram(detector_root(detector), detector)
        for detector in detectors
    }
    histograms = {detector: value for detector, value in histograms.items() if value is not None}
    if histograms:
        reset_day = next(iter(histograms.values()))["reset_day"]
        draw_rate_histogram_mosaic(
            args.output_dir / f"wfi_full_focal_plane_rate_histograms_current_{date_label}.png",
            {detector: histograms.get(detector) for detector in SCAS},
            model_label,
            pass_label,
            date_label,
            reset_day,
            observation_label=observation_label,
        )
    print(f"Wrote current-detector focal-plane plots to {args.output_dir}")


if __name__ == "__main__":
    main()
