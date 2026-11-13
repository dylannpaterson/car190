"""Generate canonical trajectory plots for explicitly selected detectors."""

import argparse
from pathlib import Path

from plot_wfi_all_detector_trajectories import (
    DEFAULT_FIT_SUFFIX,
    DEFAULT_RESET_DAY,
    SCAS,
    detrend_ice_trajectories,
    detrend_tile_trajectories,
    detector_output_dir,
    fractional_detrended_ice_trajectories,
    fractional_tile_trajectories,
    independent_fit_dirs,
    load_independent_tile_thickness,
    load_trajectory,
    plot_all_detector_trajectories,
    plot_all_tile_detrended_ice_thickness_stacked,
    plot_detrended_ice_thickness,
    write_quantiles,
)
from plot_wfi_all_detector_trajectories import load_time_annotations
from wfi_production_layout import corrected_statistics_path, positive_integer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    parser.add_argument("--detector", action="append", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detectors = tuple(value.upper() for value in args.detector)
    unknown = set(detectors).difference(SCAS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = {}
    tile_trajectories = {}
    for detector in detectors:
        output_dir = detector_output_dir(
            args.root,
            detector,
            DEFAULT_FIT_SUFFIX,
            args.tile_size,
        )
        independent_fit_dirs(output_dir)
        trajectories[detector] = load_trajectory(output_dir)
        tile_trajectories[detector] = load_independent_tile_thickness(output_dir)

    statistics_path = corrected_statistics_path(args.root, "WFI01", args.tile_size)
    time_annotations = load_time_annotations(statistics_path)
    output = args.output_dir / "wfi_all_detector_ice_rate_trajectories_current.png"
    plot_all_detector_trajectories(
        output,
        trajectories,
        reset_day=args.reset_day,
        independent_fit=True,
        time_annotations=time_annotations,
    )

    detrended, pre_rate, post_rate = detrend_ice_trajectories(
        trajectories,
        reset_day=args.reset_day,
    )
    plot_detrended_ice_thickness(
        args.output_dir / "wfi_all_detector_detrended_ice_thickness_current.png",
        detrended,
        pre_rate,
        post_rate,
        reset_day=args.reset_day,
        time_annotations=time_annotations,
    )
    plot_detrended_ice_thickness(
        args.output_dir / "wfi_all_detector_detrended_ice_thickness_fraction_current.png",
        fractional_detrended_ice_trajectories(detrended, trajectories),
        pre_rate,
        post_rate,
        reset_day=args.reset_day,
        time_annotations=time_annotations,
        fractional=True,
    )

    tile_detrended = detrend_tile_trajectories(tile_trajectories, args.reset_day)
    plot_all_tile_detrended_ice_thickness_stacked(
        args.output_dir / "wfi_all_tile_detrended_ice_thickness_current.png",
        tile_detrended,
        fractional_tile_trajectories(tile_detrended, tile_trajectories),
        reset_day=args.reset_day,
        time_annotations=time_annotations,
        alias_path=args.output_dir / "wfi_all_tile_detrended_ice_thickness_fraction_current.png",
    )
    write_quantiles(
        args.output_dir / "wfi_all_detector_ice_rate_trajectories_current_quantiles.csv",
        trajectories,
    )
    print(f"Wrote selected-detector trajectory plots to {args.output_dir}")


if __name__ == "__main__":
    main()
