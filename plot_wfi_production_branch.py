"""Generate canonical focal-plane plots for one production detector branch."""

import argparse
from pathlib import Path

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


def detector_maps(value):
    return {detector: value if detector == "WFI01" else None for detector in SCAS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-label", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detector = args.detector.upper()
    if detector != "WFI01":
        raise ValueError("This branch plotter currently expects WFI01")
    branch_dir = args.branch_dir
    output_dir = args.output_dir or branch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = args.model_label or f"{detector} production MAP branch"
    pass_label = latest_pass_label(args.statistics_dir, args.tile_size)
    date_label = latest_statistics_date(args.statistics_dir, args.tile_size)
    observation_label = latest_statistics_observation(
        args.statistics_dir, args.tile_size
    )

    rate_maps = detector_maps(
        load_rate_map(branch_dir / "tile_rate_posterior.csv", detector)
    )
    draw_rate_mosaic(
        output_dir / "wfi01_rate_maps.png",
        rate_maps,
        model_label,
        pass_label,
        observation_label=observation_label,
    )
    draw_rate_mosaic(
        output_dir / "wfi01_rate_maps_tile_contrast.png",
        rate_maps,
        model_label,
        pass_label,
        local_contrast=True,
        observation_label=observation_label,
    )

    reset_loaded = load_reset_rate_map(branch_dir, detector)
    if reset_loaded is not None:
        reset_maps, reset_day = reset_loaded
        reset_maps = detector_maps(reset_maps)
        reset_title = "Rate by Decon2 epoch"
        draw_rate_mosaic(
            output_dir / "wfi01_rate_maps_pre_post_reset.png",
            reset_maps,
            model_label,
            pass_label,
            metrics=RESET_METRICS,
            title_prefix=reset_title,
            observation_label=observation_label,
        )
        draw_rate_mosaic(
            output_dir / "wfi01_rate_maps_pre_post_reset_tile_contrast.png",
            reset_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=RESET_METRICS,
            title_prefix=reset_title,
            observation_label=observation_label,
        )

    trend_loaded = load_trend_rate_map(branch_dir, detector)
    if trend_loaded is not None:
        trend_maps, trend_day = trend_loaded
        trend_maps = detector_maps(trend_maps)
        trend_title = "Trend rate by Decon2 epoch"
        draw_rate_mosaic(
            output_dir / "wfi01_trend_rates_pre_post.png",
            trend_maps,
            model_label,
            pass_label,
            metrics=TREND_METRICS,
            title_prefix=trend_title,
            observation_label=observation_label,
            color_limits=TREND_RATE_COLOR_LIMITS,
            colorbar_label="Growth rate nm/day",
        )
        draw_rate_mosaic(
            output_dir / "wfi01_trend_rates_pre_post_tile_contrast.png",
            trend_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=TREND_METRICS,
            title_prefix=trend_title,
            observation_label=observation_label,
            color_limits=TREND_RATE_COLOR_LIMITS,
            colorbar_label="Growth rate nm/day",
        )

    last_48h = load_last_48h_rate_map(branch_dir, detector)
    if last_48h is not None:
        draw_rate_mosaic(
            output_dir / "wfi01_mean_accumulation_rate_last_48h.png",
            detector_maps(last_48h),
            model_label,
            pass_label,
            metrics=LAST_48H_METRICS,
            title_prefix="WFI focal-plane mean accumulation rate, last 48 h",
            observation_label=observation_label,
            suptitle_fontsize=10,
        )

    rms_map = load_rms_map(branch_dir, detector)
    if rms_map is not None:
        draw_rate_mosaic(
            output_dir / "wfi01_rms_residual.png",
            detector_maps(rms_map),
            model_label,
            pass_label,
            metrics=RMS_METRICS,
            title_prefix="Focal-plane RMS residual",
            observation_label=observation_label,
            colorbar_label="RMS residual (millilog)",
            metric_unit_label="millilog",
        )

    histogram = load_rate_histogram(branch_dir, detector)
    if histogram is not None:
        draw_rate_histogram_mosaic(
            output_dir / f"wfi01_rate_histograms_{date_label}.png",
            detector_maps(histogram),
            model_label,
            pass_label,
            date_label,
            histogram["reset_day"],
            observation_label=observation_label,
        )

    print(f"Wrote production plots to {output_dir}")


if __name__ == "__main__":
    main()
