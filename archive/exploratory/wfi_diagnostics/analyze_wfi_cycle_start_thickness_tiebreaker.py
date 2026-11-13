"""Test whether cycle-start thickness predicts reset-relative rate changes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr, spearmanr

import plot_wfi_operating_temperature_rate_only as temperature_filter


DEFAULT_AGES = (3.0, 5.0, 7.0)
CYCLE_NAMES = ("cycle1", "cycle2", "cycle3")
CYCLE_RESETS = (0.0, 12.907397928240743, 38.79784510416666)
TRANSITIONS = (
    ("cycle1_to_cycle2", 0, 1),
    ("cycle2_to_cycle3", 1, 2),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--telemetry-path", type=Path, required=True)
    parser.add_argument("--temperature-coefficients", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-subdir", default="relative_response")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--temperature-tolerance-k", type=float, default=0.1)
    parser.add_argument(
        "--assume-missing-operating-temperature",
        action="store_true",
        help="Retain repeats with missing telemetry when observed values are in-window.",
    )
    parser.add_argument(
        "--age-days",
        dest="ages",
        type=float,
        nargs="+",
        default=list(DEFAULT_AGES),
        help="Reset-relative ages at which to compare rates; default: 3 5 7.",
    )
    parser.add_argument(
        "--detector",
        dest="detectors",
        action="append",
        default=None,
        help="Restrict the analysis to selected detector(s); may be repeated.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_initial_thicknesses(path):
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(
            [
                np.exp(np.asarray(archive["initial_thickness_log_nm"]).reshape(-1)[0]),
                np.exp(np.asarray(archive["reset_initial_thickness_log_nm"]).reshape(-1)[0]),
                np.exp(
                    np.asarray(archive["third_decon_initial_thickness_log_nm"])
                    .reshape(-1)[0]
                ),
            ],
            dtype=float,
        )


def cycle_rate_at_age(tile, cycle_index, age_days):
    lower = CYCLE_RESETS[cycle_index]
    upper = (
        CYCLE_RESETS[cycle_index + 1]
        if cycle_index + 1 < len(CYCLE_RESETS)
        else np.inf
    )
    mask = (tile["filtered_days"] >= lower) & (tile["filtered_days"] < upper)
    cycle_age = tile["filtered_days"][mask] - lower
    cycle_rates = tile["filtered_rates"][mask]
    if cycle_age.size == 0 or age_days < np.min(cycle_age) or age_days > np.max(cycle_age):
        return np.nan
    order = np.argsort(cycle_age, kind="stable")
    return float(np.interp(age_days, cycle_age[order], cycle_rates[order]))


def collect_data(args, detectors):
    telemetry_temperatures, telemetry_info = temperature_filter.load_telemetry(
        args.telemetry_path
    )
    operating_temperatures = temperature_filter.load_operating_temperatures(
        args.temperature_coefficients,
        args.tile_size,
    )
    records = []
    tile_counts = {}
    for detector in detectors:
        detector_root = Path(args.production_root) / detector.lower()
        tiles, _statistics_path = temperature_filter.load_detector_data(
            detector,
            detector_root,
            args.tile_size,
            telemetry_temperatures,
            operating_temperatures,
            args.temperature_tolerance_k,
            args.assume_missing_operating_temperature,
            args.fit_subdir,
        )
        tile_counts[detector] = len(tiles)
        for tile in tiles:
            tile_id = tile["tile_id"]
            posterior_path = (
                detector_root
                / args.fit_subdir
                / "fits"
                / f"tile_{tile_id:03d}"
                / "posterior_samples.npz"
            )
            initial_thickness = load_initial_thicknesses(posterior_path)
            rates = {
                (cycle_index, age): cycle_rate_at_age(tile, cycle_index, age)
                for cycle_index in range(len(CYCLE_NAMES))
                for age in args.ages
            }
            records.append(
                {
                    "detector": detector,
                    "tile_id": tile_id,
                    "initial_thickness_nm": initial_thickness,
                    "rates_nm_day": rates,
                }
            )
    return records, telemetry_info, tile_counts


def finite_pair(values_x, values_y):
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def regression_summary(x, y):
    x, y = finite_pair(x, y)
    result = {
        "n": int(x.size),
        "slope": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
    }
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return result
    result["slope"] = float(np.polyfit(x, y, 1)[0])
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    result["pearson_r"] = float(pearson.statistic)
    result["pearson_p"] = float(pearson.pvalue)
    result["spearman_r"] = float(spearman.statistic)
    result["spearman_p"] = float(spearman.pvalue)
    return result


def paired_values(records, detector, transition, age):
    _transition_name, start_cycle, end_cycle = transition
    selected = [record for record in records if record["detector"] == detector]
    delta_thickness = []
    delta_rate = []
    for record in selected:
        start_rate = record["rates_nm_day"][(start_cycle, age)]
        end_rate = record["rates_nm_day"][(end_cycle, age)]
        delta_thickness.append(
            record["initial_thickness_nm"][end_cycle]
            - record["initial_thickness_nm"][start_cycle]
        )
        delta_rate.append(end_rate - start_rate)
    return finite_pair(delta_thickness, delta_rate)


def write_cycle_rate_summary(path, records, detectors, ages):
    fields = (
        "detector",
        "cycle",
        "age_days",
        "n",
        "initial_thickness_p16_nm",
        "initial_thickness_median_nm",
        "initial_thickness_p84_nm",
        "rate_p16_nm_day",
        "rate_median_nm_day",
        "rate_p84_nm_day",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector in detectors:
            selected = [record for record in records if record["detector"] == detector]
            for cycle_index, cycle in enumerate(CYCLE_NAMES):
                thickness = np.asarray(
                    [record["initial_thickness_nm"][cycle_index] for record in selected]
                )
                for age in ages:
                    rates = np.asarray(
                        [record["rates_nm_day"][(cycle_index, age)] for record in selected]
                    )
                    rates = rates[np.isfinite(rates)]
                    writer.writerow(
                        {
                            "detector": detector,
                            "cycle": cycle,
                            "age_days": age,
                            "n": rates.size,
                            "initial_thickness_p16_nm": np.percentile(thickness, 16),
                            "initial_thickness_median_nm": np.percentile(thickness, 50),
                            "initial_thickness_p84_nm": np.percentile(thickness, 84),
                            "rate_p16_nm_day": np.percentile(rates, 16)
                            if rates.size
                            else np.nan,
                            "rate_median_nm_day": np.percentile(rates, 50)
                            if rates.size
                            else np.nan,
                            "rate_p84_nm_day": np.percentile(rates, 84)
                            if rates.size
                            else np.nan,
                        }
                    )


def write_pair_summary(path, records, detectors, ages):
    fields = (
        "scope",
        "detector",
        "transition",
        "age_days",
        "n",
        "delta_initial_thickness_median_nm",
        "delta_initial_thickness_p16_nm",
        "delta_initial_thickness_p84_nm",
        "delta_rate_median_nm_day",
        "delta_rate_p16_nm_day",
        "delta_rate_p84_nm_day",
        "slope_delta_rate_per_delta_thickness",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
    )
    rows = []
    for transition in TRANSITIONS:
        transition_name = transition[0]
        for age in ages:
            all_x = []
            all_y = []
            for detector in detectors:
                x, y = paired_values(records, detector, transition, age)
                all_x.append(x)
                all_y.append(y)
                regression = regression_summary(x, y)
                rows.append(
                    {
                        "scope": "detector",
                        "detector": detector,
                        "transition": transition_name,
                        "age_days": age,
                        "n": regression["n"],
                        "delta_initial_thickness_median_nm": np.percentile(x, 50)
                        if x.size
                        else np.nan,
                        "delta_initial_thickness_p16_nm": np.percentile(x, 16)
                        if x.size
                        else np.nan,
                        "delta_initial_thickness_p84_nm": np.percentile(x, 84)
                        if x.size
                        else np.nan,
                        "delta_rate_median_nm_day": np.percentile(y, 50)
                        if y.size
                        else np.nan,
                        "delta_rate_p16_nm_day": np.percentile(y, 16)
                        if y.size
                        else np.nan,
                        "delta_rate_p84_nm_day": np.percentile(y, 84)
                        if y.size
                        else np.nan,
                        "slope_delta_rate_per_delta_thickness": regression["slope"],
                        "pearson_r": regression["pearson_r"],
                        "pearson_p": regression["pearson_p"],
                        "spearman_r": regression["spearman_r"],
                        "spearman_p": regression["spearman_p"],
                    }
                )

            x = np.concatenate(all_x)
            y = np.concatenate(all_y)
            centered_x = []
            centered_y = []
            for detector in detectors:
                detector_x, detector_y = paired_values(records, detector, transition, age)
                centered_x.append(detector_x - np.mean(detector_x))
                centered_y.append(detector_y - np.mean(detector_y))
            centered_x = np.concatenate(centered_x)
            centered_y = np.concatenate(centered_y)
            fixed_effects = regression_summary(centered_x, centered_y)
            rows.append(
                {
                    "scope": "within_detector_fixed_effect",
                    "detector": "ALL",
                    "transition": transition_name,
                    "age_days": age,
                    "n": fixed_effects["n"],
                    "delta_initial_thickness_median_nm": np.percentile(x, 50),
                    "delta_initial_thickness_p16_nm": np.percentile(x, 16),
                    "delta_initial_thickness_p84_nm": np.percentile(x, 84),
                    "delta_rate_median_nm_day": np.percentile(y, 50),
                    "delta_rate_p16_nm_day": np.percentile(y, 16),
                    "delta_rate_p84_nm_day": np.percentile(y, 84),
                    "slope_delta_rate_per_delta_thickness": fixed_effects["slope"],
                    "pearson_r": fixed_effects["pearson_r"],
                    "pearson_p": fixed_effects["pearson_p"],
                    "spearman_r": fixed_effects["spearman_r"],
                    "spearman_p": fixed_effects["spearman_p"],
                }
            )

    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_pair_diagnostic(path, records, detectors, ages, summary_rows, dpi):
    figure, axes = plt.subplots(2, len(ages), figsize=(5.0 * len(ages), 8.5), squeeze=False)
    colors = plt.get_cmap("tab20", len(detectors))
    color_by_detector = {detector: colors(index) for index, detector in enumerate(detectors)}
    summary_by_key = {
        (row["transition"], float(row["age_days"])): row
        for row in summary_rows
        if row["scope"] == "within_detector_fixed_effect"
    }
    for row_index, transition in enumerate(TRANSITIONS):
        transition_name = transition[0]
        for column, age in enumerate(ages):
            axis = axes[row_index, column]
            all_x = []
            all_y = []
            for detector in detectors:
                x, y = paired_values(records, detector, transition, age)
                all_x.append(x)
                all_y.append(y)
                axis.scatter(
                    x,
                    y,
                    s=9,
                    alpha=0.35,
                    color=color_by_detector[detector],
                )
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            row = summary_by_key[(transition_name, float(age))]
            beta = float(row["slope_delta_rate_per_delta_thickness"])
            x_line = np.linspace(float(np.min(all_x)), float(np.max(all_x)), 100)
            axis.plot(
                x_line,
                np.median(all_y) + beta * (x_line - np.median(all_x)),
                color="black",
                linestyle="--",
                linewidth=1.3,
            )
            axis.axhline(0.0, color="0.4", linewidth=0.7)
            axis.axvline(0.0, color="0.4", linewidth=0.7)
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{transition_name.replace('_', ' ')}; age {age:g} d\n"
                f"within-detector beta={beta:.3f}, r={float(row['pearson_r']):.3f}"
            )
            axis.set_xlabel("delta initial thickness [nm]")
            axis.set_ylabel("delta rate [nm/day]")

    figure.legend(
        handles=[
            Line2D([], [], color=color_by_detector[detector], marker="o", linestyle="", label=detector)
            for detector in detectors
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Cycle-start thickness tie-breaker\n"
        "Paired rate changes at equal time since reset; dashed line is the within-detector fit",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.92), pad=1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def write_readme(path, ages, telemetry_info, tile_counts):
    lines = [
        "# Cycle-start thickness tie-breaker",
        "",
        "This analysis uses the independently fitted cycle-start thickness parameters, "
        "not thickness reconstructed from the current cycle's rate nodes.",
        "",
        "- Rates are compared at equal elapsed days after each reset.",
        "- `delta rate` is the later-cycle rate minus the earlier-cycle rate.",
        "- `delta initial thickness` is the later-cycle fitted start thickness minus the earlier-cycle fitted start thickness.",
        "- A negative within-detector slope supports lower rate at larger starting thickness.",
        "- Detector-fixed slopes remove detector-level rate offsets; repeated ages are still correlated and exploratory.",
        f"- Matched ages: `{', '.join(f'{age:g}' for age in ages)}` days.",
        f"- Telemetry: `{telemetry_info['telemetry_path']}`.",
        f"- Telemetry coverage: `{telemetry_info['telemetry_first_utc']}` to `{telemetry_info['telemetry_last_utc']}`.",
        f"- Tile counts: `{json.dumps(tile_counts, sort_keys=True)}`.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.tile_size <= 0 or args.temperature_tolerance_k <= 0 or args.dpi <= 0:
        raise ValueError("tile-size, temperature-tolerance-k, and dpi must be positive")
    if any(age <= 0.0 for age in args.ages):
        raise ValueError("All matched ages must be positive")
    detectors = tuple(detector.upper() for detector in (args.detectors or temperature_filter.DETECTORS))
    unknown = set(detectors).difference(temperature_filter.DETECTORS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, telemetry_info, tile_counts = collect_data(args, detectors)
    pair_summary_path = args.output_dir / "cycle_start_thickness_pair_summary.csv"
    summary_rows = write_pair_summary(pair_summary_path, records, detectors, args.ages)
    cycle_summary_path = args.output_dir / "cycle_rate_at_matched_age_summary.csv"
    write_cycle_rate_summary(cycle_summary_path, records, detectors, args.ages)
    plot_path = args.output_dir / "cycle_start_thickness_tiebreaker.png"
    plot_pair_diagnostic(plot_path, records, detectors, args.ages, summary_rows, args.dpi)
    metadata = {
        "detectors": detectors,
        "tile_counts": tile_counts,
        "matched_ages_days": args.ages,
        "temperature_tolerance_k": args.temperature_tolerance_k,
        "assume_missing_operating_temperature": args.assume_missing_operating_temperature,
        "fit_subdir": args.fit_subdir,
        "telemetry": telemetry_info,
        "pair_summary": str(pair_summary_path.resolve()),
        "cycle_summary": str(cycle_summary_path.resolve()),
        "plot": str(plot_path.resolve()),
    }
    metadata_path = args.output_dir / "cycle_start_thickness_tiebreaker_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    write_readme(args.output_dir / "README.md", args.ages, telemetry_info, tile_counts)
    print(f"Wrote {plot_path}")
    print(f"Wrote {pair_summary_path}")
    print(f"Wrote {cycle_summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
