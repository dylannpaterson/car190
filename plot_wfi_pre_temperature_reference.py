#!/usr/bin/env python3
"""Plot one detector tile's pre-operational temperature-reference fit.

This is a diagnostic only.  It reconstructs the ACM-corrected count means
from the scaled sufficient statistics, fits each LED independently over the
same Program 1039 window used by the reduction, and shows the early points
that receive the backward temperature-gain correction.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_wfi_reduction_pipeline as reduction


DEFAULT_OPERATIONAL_DAY = 15.064405023148149
DEFAULT_PHYSICAL_RESET_DAY = 12.907397928240743
DEFAULT_MINIMUM_DAY = 1.0
DEFAULT_ADDITIONAL_ACM_REPEAT = 669
DEFAULT_PRE_RESET_ACM_REPEAT = 35
DEFAULT_LATE_ACM_REPEAT = 686


def read_corrections(path):
    with Path(path).open(newline="") as handle:
        return {
            (int(row["tile_id"]), int(row["group"])): row
            for row in csv.DictReader(handle)
        }


def count_mean(row):
    valid = float(row["n_pair_valid"])
    if valid <= 0.0:
        raise ValueError("Statistics row has no valid pixel pairs")
    return float(row["sum_current_counts"]) / valid


def acm_corrected_mean(row, correction):
    """Remove temperature terms so the sine is fit to ACM-corrected counts."""
    log_correction = float(correction["count_log_correction"])
    log_correction -= float(correction.get("log_gain_correction") or 0.0)
    log_correction -= float(
        correction.get("temperature_gain_log_correction") or 0.0
    )
    return count_mean(row) * np.exp(log_correction)


def load_tile_series(
    statistics_path,
    corrections_path,
    tile_id,
    minimum_day,
    operational_day,
    additional_acm_repeat,
    pre_reset_acm_repeat,
    late_acm_repeat,
):
    tiles = reduction.load_count_statistics(
        statistics_path,
        additional_acm_repeat,
        pre_reset_acm_repeat,
        late_acm_repeat,
    )
    if tile_id not in tiles:
        raise ValueError(f"Tile {tile_id} is not present in {statistics_path}")
    corrections = read_corrections(corrections_path)
    tile = tiles[tile_id]
    series = {}
    for led in range(1, 7):
        fit_days = []
        fit_values = []
        early_days = []
        early_values = []
        early_corrected_values = []
        early_repeats = []
        early_table_factors = []
        for row, day, program, repeat in zip(
            tile["led"][led]["rows"],
            tile["led"][led]["days"],
            tile["led"][led]["program"],
            tile["led"][led]["repeat"],
        ):
            if program != reduction.REFERENCE_PROGRAM:
                continue
            correction = corrections[(tile_id, int(row["group"]))]
            day = float(day)
            acm_mean = acm_corrected_mean(row, correction)
            if minimum_day < day < operational_day:
                fit_days.append(day)
                fit_values.append(acm_mean)
            elif day <= minimum_day:
                log_gain = float(correction.get("log_gain_correction") or 0.0)
                early_days.append(day)
                early_values.append(acm_mean)
                early_corrected_values.append(acm_mean * np.exp(log_gain))
                early_repeats.append(int(repeat))
                early_table_factors.append(log_gain)
        fit_days = np.asarray(fit_days, dtype=float)
        fit_values = np.asarray(fit_values, dtype=float)
        if fit_days.size < 4:
            raise ValueError(f"LED {led} has too few fit points")
        fit = reduction._fit_count_sinusoid(fit_days, fit_values)
        alternative_factors = np.log(
            reduction._evaluate_count_sinusoid(
                fit, np.asarray(early_days, dtype=float)
            )
            / np.asarray(early_values, dtype=float)
        )
        series[led] = {
            "fit_days": fit_days,
            "fit_values": fit_values,
            "fit": fit,
            "early_days": np.asarray(early_days, dtype=float),
            "early_values": np.asarray(early_values, dtype=float),
            "early_corrected_values": np.asarray(
                early_corrected_values, dtype=float
            ),
            "early_repeats": np.asarray(early_repeats, dtype=int),
            "early_table_factors": np.asarray(early_table_factors, dtype=float),
            "early_fit_factors": np.asarray(alternative_factors, dtype=float),
        }
    return series


def fit_pooled_period(series):
    """Fit one period to all LED fit-window data as a diagnostic only."""
    entries = []
    masks = []
    for led in range(1, 7):
        item = series[led]
        entries.append(
            {
                "led": led,
                "days": item["fit_days"],
                "values": item["fit_values"],
            }
        )
        masks.append(np.ones(item["fit_days"].shape, dtype=bool))
    return reduction._fit_shared_log_curve(entries, masks, ())


def evaluate_pooled_log_curve(fit, led, days):
    days = np.asarray(days, dtype=float)
    angular_frequency = 2.0 * np.pi / fit["period_days"]
    time_days = days - fit["origin_day"]
    return (
        fit["led_offsets"][led]
        + fit["led_sine_coefficients"][led]
        * np.sin(angular_frequency * time_days)
        + fit["led_cosine_coefficients"][led]
        * np.cos(angular_frequency * time_days)
    )


def pooled_early_factors(series, pooled_fit, led):
    item = series[led]
    return np.exp(
        evaluate_pooled_log_curve(pooled_fit, led, item["early_days"])
        - np.log(item["early_values"])
    )


def plot_sine_fits(series, output_path, detector, tile_id, minimum_day, reset_day):
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.07,
        top=0.78,
        wspace=0.20,
        hspace=0.30,
    )
    xmax = reset_day + 0.45
    for axis, led in zip(axes.flat, range(1, 7)):
        item = series[led]
        fit = item["fit"]
        fit_days = item["fit_days"]
        curve_days = np.linspace(0.0, xmax, 600)
        curve_values = reduction._evaluate_count_sinusoid(fit, curve_days)
        axis.scatter(
            fit_days,
            item["fit_values"],
            color="tab:blue",
            s=25,
            label="fit-window 1039",
            zorder=3,
        )
        if item["early_days"].size:
            axis.scatter(
                item["early_days"],
                item["early_values"],
                color="0.35",
                marker="o",
                s=30,
                label="early before correction",
                zorder=3,
            )
            axis.scatter(
                item["early_days"],
                item["early_corrected_values"],
                color="tab:orange",
                marker="x",
                s=38,
                label="early after current correction",
                zorder=4,
            )
        axis.plot(
            curve_days,
            curve_values,
            color="tab:red",
            linewidth=1.8,
            label="fitted sine",
            zorder=2,
        )
        axis.axvline(
            minimum_day,
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            label="fit starts after day 1",
        )
        axis.axvline(
            reset_day,
            color="tab:green",
            linestyle=":",
            linewidth=1.0,
            label="physical reset",
        )
        title = (
            f"LED {led}: P={fit['period_days']:.4g} d, "
            f"R2={fit['r2']:.3f}, RMSE={fit['rmse']:.3f}"
        )
        axis.set_title(title)
        if led == 3:
            axis.patch.set_edgecolor("tab:purple")
            axis.patch.set_linewidth(2.0)
            axis.set_title(title + "  CURRENT REFERENCE", color="tab:purple")
        axis.set_xlabel("Days since first unified exposure")
        axis.set_ylabel("ACM-corrected mean counts")
        axis.set_xlim(0.0, xmax)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=5,
        frameon=False,
    )
    figure.text(
        0.5,
        0.99,
        f"{detector} tile {tile_id:03d} pre-operational temperature-reference fits\n"
        "Program 1039 only; early points are shown before and after the current correction",
        ha="center",
        va="top",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_correction_factors(
    series, pooled_fit, output_path, detector, tile_id, minimum_day
):
    reference = series[3]
    alternative = series[2]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    axes[0].plot(
        reference["early_repeats"],
        1000.0 * reference["early_table_factors"],
        "o-",
        color="tab:purple",
        label="current applied factor (LED 3 reference)",
    )
    axes[0].plot(
        alternative["early_repeats"],
        1000.0 * alternative["early_fit_factors"],
        "s--",
        color="tab:blue",
        label="LED 2 refit factor",
    )
    axes[0].plot(
        reference["early_repeats"],
        1000.0 * reference["early_fit_factors"],
        "x:",
        color="tab:red",
        label="LED 3 refit factor",
    )
    for led, color in ((2, "tab:blue"), (3, "tab:red")):
        item = series[led]
        axes[0].plot(
            item["early_repeats"],
            1000.0 * np.log(pooled_early_factors(series, pooled_fit, led)),
            "^-." if led == 2 else "v-.",
            color=color,
            alpha=0.65,
            label=f"pooled-period factor (LED {led})",
        )
    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_title("Early log gain correction")
    axes[0].set_xlabel("Program 1039 repeat")
    axes[0].set_ylabel("Correction [millilog]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for led, color in ((2, "tab:blue"), (3, "tab:red")):
        item = series[led]
        residual = item["fit_values"] - reduction._evaluate_count_sinusoid(
            item["fit"], item["fit_days"]
        )
        axes[1].plot(
            item["fit_days"],
            residual,
            "o-",
            color=color,
            markersize=3,
            label=f"LED {led} fit residual",
        )
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].axvline(
        minimum_day,
        color="0.25",
        linestyle="--",
        linewidth=0.9,
        label="fit starts after day 1",
    )
    axes[1].set_title("Fit-window residuals")
    axes[1].set_xlabel("Days since first unified exposure")
    axes[1].set_ylabel("Count residual")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{detector} tile {tile_id:03d} pre-temperature correction diagnostics"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_pooled_fit(
    series, pooled_fit, output_path, detector, tile_id, minimum_day, reset_day
):
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.07,
        top=0.78,
        wspace=0.20,
        hspace=0.30,
    )
    xmax = reset_day + 0.45
    curve_days = np.linspace(0.0, xmax, 600)
    for axis, led in zip(axes.flat, range(1, 7)):
        item = series[led]
        pooled_values = np.exp(
            evaluate_pooled_log_curve(pooled_fit, led, curve_days)
        )
        individual_values = reduction._evaluate_count_sinusoid(
            item["fit"], curve_days
        )
        axis.scatter(
            item["fit_days"],
            item["fit_values"],
            color="tab:blue",
            s=25,
            label="fit-window 1039",
            zorder=3,
        )
        if item["early_days"].size:
            axis.scatter(
                item["early_days"],
                item["early_values"],
                color="0.35",
                marker="o",
                s=30,
                label="early before correction",
                zorder=3,
            )
            axis.scatter(
                item["early_days"],
                item["early_corrected_values"],
                color="tab:orange",
                marker="x",
                s=38,
                label="early after current correction",
                zorder=4,
            )
        axis.plot(
            curve_days,
            individual_values,
            color="tab:red",
            linewidth=1.5,
            label="LED-specific count sine",
            zorder=2,
        )
        axis.plot(
            curve_days,
            pooled_values,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="all-LED pooled period",
            zorder=2,
        )
        axis.axvline(
            minimum_day,
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            label="fit starts after day 1",
        )
        axis.axvline(
            reset_day,
            color="tab:green",
            linestyle=":",
            linewidth=1.0,
            label="physical reset",
        )
        title = f"LED {led}: pooled P={pooled_fit['period_days']:.4g} d"
        if led == 3:
            axis.patch.set_edgecolor("tab:purple")
            axis.patch.set_linewidth(2.0)
            title += "  CURRENT REFERENCE"
            axis.set_title(title, color="tab:purple")
        else:
            axis.set_title(title)
        axis.set_xlabel("Days since first unified exposure")
        axis.set_ylabel("ACM-corrected mean counts")
        axis.set_xlim(0.0, xmax)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=5,
        frameon=False,
    )
    figure.text(
        0.5,
        0.99,
        f"{detector} tile {tile_id:03d} pooled pre-temperature diagnostic\n"
        "One period is pooled across LEDs; LED-specific log sine terms remain nuisance fits",
        ha="center",
        va="top",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detector", default="WFI03")
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-day", type=float, default=DEFAULT_MINIMUM_DAY)
    parser.add_argument(
        "--operational-day",
        type=float,
        default=DEFAULT_OPERATIONAL_DAY,
    )
    parser.add_argument(
        "--physical-reset-day",
        type=float,
        default=DEFAULT_PHYSICAL_RESET_DAY,
    )
    parser.add_argument(
        "--additional-acm-repeat",
        type=int,
        default=DEFAULT_ADDITIONAL_ACM_REPEAT,
    )
    parser.add_argument(
        "--pre-reset-acm-repeat",
        type=int,
        default=DEFAULT_PRE_RESET_ACM_REPEAT,
    )
    parser.add_argument(
        "--late-acm-repeat",
        type=int,
        default=DEFAULT_LATE_ACM_REPEAT,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detector = args.detector.upper()
    detector_lower = detector.lower()
    statistics_path = (
        args.root / "statistics" / f"{detector_lower}_tile_log_statistics_512.csv"
    )
    corrections_path = (
        args.root
        / "corrections"
        / f"{detector_lower}_count_space_unified_temperature_corrections.csv"
    )
    output_dir = args.output_dir or args.root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    series = load_tile_series(
        statistics_path,
        corrections_path,
        args.tile_id,
        args.minimum_day,
        args.operational_day,
        args.additional_acm_repeat,
        args.pre_reset_acm_repeat,
        args.late_acm_repeat,
    )
    pooled_fit = fit_pooled_period(series)
    sine_path = output_dir / (
        f"{detector_lower}_tile_{args.tile_id:03d}_pre_temperature_sine_fits.png"
    )
    factor_path = output_dir / (
        f"{detector_lower}_tile_{args.tile_id:03d}_pre_temperature_correction_factors.png"
    )
    pooled_path = output_dir / (
        f"{detector_lower}_tile_{args.tile_id:03d}_pre_temperature_pooled_fit.png"
    )
    plot_sine_fits(
        series,
        sine_path,
        detector,
        args.tile_id,
        args.minimum_day,
        args.physical_reset_day,
    )
    plot_correction_factors(
        series,
        pooled_fit,
        factor_path,
        detector,
        args.tile_id,
        args.minimum_day,
    )
    plot_pooled_fit(
        series,
        pooled_fit,
        pooled_path,
        detector,
        args.tile_id,
        args.minimum_day,
        args.physical_reset_day,
    )
    print(sine_path)
    print(factor_path)
    print(pooled_path)
    print(
        f"Pooled all-LED period: n={sum(item['fit_days'].size for item in series.values())} "
        f"period_days={pooled_fit['period_days']:.8g} "
        f"r2_log={pooled_fit['r2']:.8g} "
        f"rmse_log={pooled_fit['rmse_log_change']:.8g}"
    )
    for led in range(1, 7):
        fit = series[led]["fit"]
        print(
            f"LED {led}: n={series[led]['fit_days'].size} "
            f"period_days={fit['period_days']:.8g} "
            f"r2={fit['r2']:.8g} rmse_counts={fit['rmse']:.8g}"
        )
        if series[led]["early_days"].size:
            pooled_log_factor = np.log(pooled_early_factors(series, pooled_fit, led))
            delta = 1000.0 * (
                pooled_log_factor - series[led]["early_table_factors"]
            )
            print(
                f"  pooled-minus-current-early-factor_millilog: "
                f"median={np.median(delta):.8g} "
                f"min={np.min(delta):.8g} max={np.max(delta):.8g}"
            )


if __name__ == "__main__":
    main()
