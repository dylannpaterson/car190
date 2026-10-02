"""Apply LED 3's early gain correction to every LED and refit sinusoids."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_wfi_led_postday1_sinusoid import (
    evaluate_components,
    fit_sinusoid_plus_two_acm,
    load_led_data,
)


def fit_reference_correction(statistics, reference_led, tile_id, minimum_day):
    (
        days,
        values,
        first_step,
        late_step,
        fit_indices,
        first_event_day,
        late_event_day,
    ) = load_led_data(statistics, reference_led, tile_id, -1.0)
    fit_mask = days > minimum_day
    fit = fit_sinusoid_plus_two_acm(
        days[fit_mask],
        values[fit_mask],
        first_step[fit_mask],
        late_step[fit_mask],
    )
    _, model = evaluate_components(
        days - days[fit_mask][0], fit, first_step, late_step
    )
    early = days <= minimum_day
    correction = model[early] - values[early]
    return {
        "days": days,
        "values": values,
        "early_days": days[early],
        "early_correction": correction,
        "fit": fit,
        "first_event_day": first_event_day,
        "late_event_day": late_event_day,
    }


def apply_common_correction(days, values, reference, minimum_day):
    early = days <= minimum_day
    correction = np.zeros_like(values)
    correction[early] = np.interp(
        days[early], reference["early_days"], reference["early_correction"]
    )
    return values + correction, correction


def fit_all_leds(statistics, tile_id, reference_led, minimum_day):
    reference = fit_reference_correction(
        statistics, reference_led, tile_id, minimum_day
    )
    results = {}
    for led in range(1, 7):
        (
            days,
            values,
            first_step,
            late_step,
            _,
            first_event_day,
            late_event_day,
        ) = load_led_data(statistics, led, tile_id, -1.0)
        corrected_values, correction = apply_common_correction(
            days, values, reference, minimum_day
        )
        raw_fit = fit_sinusoid_plus_two_acm(
            days, values, first_step, late_step
        )
        corrected_fit = fit_sinusoid_plus_two_acm(
            days, corrected_values, first_step, late_step
        )
        results[led] = {
            "days": days,
            "raw_values": values,
            "corrected_values": corrected_values,
            "correction": correction,
            "raw_fit": raw_fit,
            "fit": corrected_fit,
            "first_event_day": first_event_day,
            "late_event_day": late_event_day,
        }
    return reference, results


def write_summary(path, reference_led, tile_id, minimum_day, reference, results):
    fields = (
        "reference_led",
        "led",
        "tile_id",
        "minimum_day_exclusive",
        "n_points",
        "raw_period_days",
        "corrected_period_days",
        "raw_amplitude_log_change",
        "corrected_amplitude_log_change",
        "raw_acm_step_log_change",
        "corrected_acm_step_log_change",
        "raw_late_acm_step_log_change",
        "corrected_late_acm_step_log_change",
        "raw_rmse_log_change",
        "corrected_rmse_log_change",
        "raw_r2",
        "corrected_r2",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for led in range(1, 7):
            result = results[led]
            raw_fit = result["raw_fit"]
            corrected_fit = result["fit"]
            writer.writerow(
                {
                    "reference_led": reference_led,
                    "led": led,
                    "tile_id": tile_id,
                    "minimum_day_exclusive": minimum_day,
                    "n_points": result["days"].size,
                    "raw_period_days": raw_fit["period_days"],
                    "corrected_period_days": corrected_fit["period_days"],
                    "raw_amplitude_log_change": raw_fit["amplitude_log_change"],
                    "corrected_amplitude_log_change": corrected_fit[
                        "amplitude_log_change"
                    ],
                    "raw_acm_step_log_change": raw_fit["acm_step_log_change"],
                    "corrected_acm_step_log_change": corrected_fit[
                        "acm_step_log_change"
                    ],
                    "raw_late_acm_step_log_change": raw_fit[
                        "late_acm_step_log_change"
                    ],
                    "corrected_late_acm_step_log_change": corrected_fit[
                        "late_acm_step_log_change"
                    ],
                    "raw_rmse_log_change": raw_fit["rmse_log_change"],
                    "corrected_rmse_log_change": corrected_fit["rmse_log_change"],
                    "raw_r2": raw_fit["r2"],
                    "corrected_r2": corrected_fit["r2"],
                }
            )


def plot_results(path, reference_led, tile_id, minimum_day, reference, results):
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 13),
        sharex=True,
        constrained_layout=True,
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        result = results[led]
        days = result["days"]
        corrected_values = result["corrected_values"]
        fit = result["fit"]
        grid = np.linspace(days[0], days[-1], 700)
        grid_first_step = (grid >= result["first_event_day"]).astype(float)
        grid_late_step = (grid >= result["late_event_day"]).astype(float)
        _, model = evaluate_components(
            grid - days[0], fit, grid_first_step, grid_late_step
        )
        early = days <= minimum_day
        axis.scatter(
            days,
            1000.0 * corrected_values,
            color="tab:blue",
            s=18,
            label="corrected ratios",
        )
        axis.scatter(
            days[early],
            1000.0 * result["raw_values"][early],
            facecolors="none",
            edgecolors="0.45",
            s=32,
            label="raw early ratios",
        )
        axis.plot(
            grid,
            1000.0 * model,
            color="tab:red",
            linewidth=1.8,
            label=(
                f"P={fit['period_days']:.3g} d, "
                f"RMSE={1000.0 * fit['rmse_log_change']:.3f} mlog, "
                f"R2={fit['r2']:.3f}"
            ),
        )
        axis.axvline(minimum_day, color="0.35", linestyle="--", linewidth=0.8)
        axis.axvline(
            result["first_event_day"], color="black", linestyle=":", linewidth=0.8
        )
        axis.axvline(
            result["late_event_day"], color="tab:green", linestyle=":", linewidth=0.8
        )
        axis.set_title(f"LED {led}")
        axis.set_ylabel("Mean log ratio (millilog)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    for axis in axes[-1, :]:
        axis.set_xlabel("Days since first exposure")
    figure.suptitle(
        f"Common LED {reference_led} early gain correction applied to WFI01 tile {tile_id}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("wfi_tile_log_statistics_512/wfi01_tile_log_statistics_512.csv"),
    )
    parser.add_argument("--reference-led", type=int, default=3)
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--minimum-day", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wfi01_common_led3_gain_corrected_sinusoid_fits.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("wfi01_common_led3_gain_corrected_sinusoid_fits.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference_led < 1 or args.reference_led > 6:
        raise ValueError("Reference LED must be between 1 and 6")
    if args.minimum_day < 0.0 or not np.isfinite(args.minimum_day):
        raise ValueError("Minimum day must be finite and non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    reference, results = fit_all_leds(
        args.statistics, args.tile_id, args.reference_led, args.minimum_day
    )
    plot_results(
        args.output,
        args.reference_led,
        args.tile_id,
        args.minimum_day,
        reference,
        results,
    )
    write_summary(
        args.summary,
        args.reference_led,
        args.tile_id,
        args.minimum_day,
        reference,
        results,
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary}")
    for led in range(1, 7):
        fit = results[led]["fit"]
        print(
            f"LED {led}: corrected period={fit['period_days']:.8g} d, "
            f"RMSE={fit['rmse_log_change']:.8g} log, R2={fit['r2']:.8g}"
        )


if __name__ == "__main__":
    main()
