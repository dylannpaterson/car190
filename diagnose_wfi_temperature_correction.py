"""Diagnose the WFI count-space temperature correction without changing products."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_wfi_reduction_pipeline import (
    _evaluate_count_sinusoid,
    _fit_count_sinusoid,
)


REFERENCE_PROGRAM = "1039"
TEMPERATURE_PROGRAM = "1063"
DEFAULT_OPERATIONAL_DAY = 15.064405023148149
DEFAULT_PHYSICAL_RESET_DAY = 12.907397928240743
DEFAULT_POST_FIT_START = 669
DEFAULT_POST_FIT_END = 682
DEFAULT_POST_ACM_START = 666
DEFAULT_APPLICATION_END = 666


def load_rows(run_dir, detector):
    run_dir = Path(run_dir)
    statistics_path = run_dir / "statistics" / (
        f"{detector.lower()}_tile_log_statistics_512.csv"
    )
    raw_path = run_dir / "raw_statistics" / statistics_path.name
    corrections_path = run_dir / "corrections" / (
        f"{detector.lower()}_count_space_unified_temperature_corrections.csv"
    )
    with statistics_path.open(newline="") as handle:
        statistics = list(csv.DictReader(handle))
    with raw_path.open(newline="") as handle:
        raw = {
            (int(row["tile_id"]), int(row["group"])): row
            for row in csv.DictReader(handle)
        }
    with corrections_path.open(newline="") as handle:
        corrections = {
            (int(row["tile_id"]), int(row["group"])): row
            for row in csv.DictReader(handle)
        }

    rows = []
    for row in statistics:
        key = (int(row["tile_id"]), int(row["group"]))
        correction = corrections[key]
        raw_row = raw[key]
        n_pair_valid = float(row["n_pair_valid"])
        if n_pair_valid <= 0.0:
            continue
        scaled_mean = float(row["sum_current_counts"]) / n_pair_valid
        raw_mean = float(raw_row["sum_current_counts"]) / float(
            raw_row["n_pair_valid"]
        )
        total_log = float(correction["count_log_correction"])
        pre_temperature_log = float(correction["log_gain_correction"])
        post_temperature_log = float(
            correction["temperature_gain_log_correction"]
        )
        rows.append(
            {
                "tile": int(row["tile_id"]),
                "group": int(row["group"]),
                "led": int(row["led"]),
                "program": str(row.get("program") or REFERENCE_PROGRAM),
                "repeat": int(row["repeat"]),
                "days": float(row["days"]),
                "raw_mean": raw_mean,
                "scaled_mean": scaled_mean,
                "acm_mean": scaled_mean
                * math.exp(total_log - pre_temperature_log - post_temperature_log),
                "final_mean": scaled_mean * math.exp(total_log),
                "pre_temperature_log": pre_temperature_log,
                "post_temperature_log": post_temperature_log,
                "total_log": total_log,
            }
        )
    return rows


def fit_window(rows, tile, led, start_repeat, end_repeat):
    selected = sorted(
        [
            row
            for row in rows
            if row["tile"] == tile
            and row["led"] == led
            and row["program"] == REFERENCE_PROGRAM
            and row["repeat"] >= start_repeat
            and row["repeat"] < end_repeat
        ],
        key=lambda row: row["days"],
    )
    if len(selected) < 6:
        raise ValueError(
            f"Insufficient fit rows for tile {tile}, LED {led}, "
            f"repeats [{start_repeat}, {end_repeat})"
        )
    days = np.asarray([row["days"] for row in selected], dtype=float)
    values = np.asarray([row["acm_mean"] for row in selected], dtype=float)
    return selected, _fit_count_sinusoid(days, values)


def inferred_log_factor(fit, rows):
    days = np.asarray([row["days"] for row in rows], dtype=float)
    values = np.asarray([row["acm_mean"] for row in rows], dtype=float)
    return np.log(_evaluate_count_sinusoid(fit, days) / values)


def write_fit_summary(path, rows, tiles, post_fit_start, post_fit_end, wide_start):
    output = []
    for tile in tiles:
        for led in range(1, 7):
            narrow_rows, narrow_fit = fit_window(
                rows, tile, led, post_fit_start, post_fit_end
            )
            wide_rows, wide_fit = fit_window(
                rows, tile, led, wide_start, post_fit_end
            )
            early = [
                row
                for row in rows
                if row["tile"] == tile
                and row["led"] == led
                and row["program"] == TEMPERATURE_PROGRAM
                and row["repeat"] < DEFAULT_APPLICATION_END
            ]
            narrow_log = inferred_log_factor(narrow_fit, early)
            wide_log = inferred_log_factor(wide_fit, early)
            original_log = np.asarray(
                [row["post_temperature_log"] for row in early], dtype=float
            )
            output.append(
                {
                    "detector": rows[0].get("detector", "WFI01"),
                    "tile": tile,
                    "led": led,
                    "narrow_points": len(narrow_rows),
                    "wide_points": len(wide_rows),
                    "narrow_period_days": narrow_fit["period_days"],
                    "wide_period_days": wide_fit["period_days"],
                    "narrow_rmse_counts": narrow_fit["rmse"],
                    "wide_rmse_counts": wide_fit["rmse"],
                    "narrow_r2": narrow_fit["r2"],
                    "wide_r2": wide_fit["r2"],
                    "original_temperature_log_mean": float(np.mean(original_log)),
                    "narrow_all_led_log_mean": float(np.mean(narrow_log)),
                    "wide_all_led_log_mean": float(np.mean(wide_log)),
                    "narrow_minus_original_mlog": float(
                        1000.0 * np.mean(narrow_log - original_log)
                    ),
                    "wide_minus_original_mlog": float(
                        1000.0 * np.mean(wide_log - original_log)
                    ),
                }
            )
    fields = list(output[0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return output


def plot_stage_comparison(path, rows, tile, operational_day, reset_day):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for axis, led in zip(axes.flat, range(1, 7)):
        selected = sorted(
            [row for row in rows if row["tile"] == tile and row["led"] == led],
            key=lambda row: row["days"],
        )
        days = np.asarray([row["days"] for row in selected], dtype=float)
        for field, color, label in (
            ("raw_mean", "0.55", "raw detector counts"),
            ("scaled_mean", "tab:gray", "photodiode/side-swap scaled"),
            ("acm_mean", "tab:blue", "ACM-only"),
            ("final_mean", "tab:orange", "final with temperature"),
        ):
            axis.plot(
                days,
                [row[field] for row in selected],
                color=color,
                linewidth=0.8 if field in {"raw_mean", "scaled_mean"} else 1.25,
                alpha=0.75,
                label=label,
            )
        early_1063 = [
            row
            for row in selected
            if row["program"] == TEMPERATURE_PROGRAM and row["repeat"] < DEFAULT_APPLICATION_END
        ]
        axis.scatter(
            [row["days"] for row in early_1063],
            [row["final_mean"] for row in early_1063],
            color="tab:red",
            s=8,
            label="early 1063 final",
            zorder=4,
        )
        axis.axvline(reset_day, color="tab:red", linestyle=":", linewidth=0.8)
        axis.axvline(operational_day, color="tab:green", linestyle="--", linewidth=0.8)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Days since first unified exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean current counts")
    figure.suptitle(f"WFI01 tile {tile}: correction stages")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_temperature_fits(
    path,
    rows,
    tile,
    post_fit_start,
    post_fit_end,
    wide_start,
    application_end,
):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for axis, led in zip(axes.flat, range(1, 7)):
        narrow_rows, narrow_fit = fit_window(
            rows, tile, led, post_fit_start, post_fit_end
        )
        wide_rows, wide_fit = fit_window(rows, tile, led, wide_start, post_fit_end)
        selected = sorted(
            [row for row in rows if row["tile"] == tile and row["led"] == led],
            key=lambda row: row["days"],
        )
        fit_days = np.asarray([row["days"] for row in selected], dtype=float)
        fit_values = np.asarray([row["acm_mean"] for row in selected], dtype=float)
        reference = [row for row in selected if row["program"] == REFERENCE_PROGRAM]
        early = [
            row
            for row in selected
            if row["program"] == TEMPERATURE_PROGRAM
            and row["repeat"] < application_end
        ]
        curve_days = np.linspace(min(fit_days), max(fit_days), 600)
        extrapolation_days = np.linspace(
            min([row["days"] for row in early] + [min(fit_days)]),
            max(fit_days),
            800,
        )
        axis.scatter(
            [row["days"] for row in reference],
            [row["acm_mean"] for row in reference],
            color="0.55",
            s=8,
            label="1039 ACM-only",
        )
        axis.scatter(
            [row["days"] for row in narrow_rows],
            [row["acm_mean"] for row in narrow_rows],
            color="tab:blue",
            s=18,
            label=f"narrow fit {post_fit_start}:{post_fit_end}",
        )
        axis.scatter(
            [row["days"] for row in early],
            [row["acm_mean"] for row in early],
            color="tab:orange",
            s=8,
            label="early 1063 ACM-only",
        )
        axis.plot(
            extrapolation_days,
            _evaluate_count_sinusoid(narrow_fit, extrapolation_days),
            color="tab:blue",
            linewidth=1.4,
            label=f"narrow sine P={narrow_fit['period_days']:.2f} d",
        )
        axis.plot(
            extrapolation_days,
            _evaluate_count_sinusoid(wide_fit, extrapolation_days),
            color="tab:red",
            linestyle="--",
            linewidth=1.4,
            label=f"wide sine P={wide_fit['period_days']:.2f} d",
        )
        axis.axvline(
            min(row["days"] for row in early), color="tab:orange", linestyle=":", linewidth=0.8
        )
        axis.axvline(
            min(row["days"] for row in narrow_rows), color="tab:blue", linestyle=":", linewidth=0.8
        )
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Days since first unified exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("ACM-only mean counts")
    figure.suptitle(f"WFI01 tile {tile}: temperature-fit window and extrapolation")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_factor_comparison(
    path,
    rows,
    tile,
    post_fit_start,
    post_fit_end,
    wide_start,
    application_end,
):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for axis, led in zip(axes.flat, range(1, 7)):
        _, narrow_fit = fit_window(rows, tile, led, post_fit_start, post_fit_end)
        _, wide_fit = fit_window(rows, tile, led, wide_start, post_fit_end)
        early = sorted(
            [
                row
                for row in rows
                if row["tile"] == tile
                and row["led"] == led
                and row["program"] == TEMPERATURE_PROGRAM
                and row["repeat"] < application_end
            ],
            key=lambda row: row["days"],
        )
        days = np.asarray([row["days"] for row in early], dtype=float)
        narrow_log = inferred_log_factor(narrow_fit, early) * 1000.0
        wide_log = inferred_log_factor(wide_fit, early) * 1000.0
        original_log = np.asarray(
            [row["post_temperature_log"] for row in early], dtype=float
        ) * 1000.0
        axis.plot(days, original_log, color="black", linewidth=1.5, label="current LED3-derived")
        axis.plot(days, narrow_log, color="tab:blue", linewidth=1.1, label="independent LED narrow")
        axis.plot(days, wide_log, color="tab:red", linestyle="--", linewidth=1.1, label="independent LED wide")
        axis.axhline(0.0, color="0.4", linewidth=0.7)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Days since first unified exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("temperature log correction (millilog)")
    figure.suptitle(f"WFI01 tile {tile}: current versus all-LED inferred temperature factors")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_temperature_correction_stages(
    path,
    rows,
    tile,
    post_fit_start,
    post_fit_end,
    application_end,
):
    """Show the reference fit and the early-1063 correction before and after it."""
    selected = sorted(
        [row for row in rows if row["tile"] == tile and row["led"] == 3],
        key=lambda row: row["days"],
    )
    fit_rows, fit = fit_window(rows, tile, 3, post_fit_start, post_fit_end)
    reference_rows = [row for row in selected if row["program"] == REFERENCE_PROGRAM]
    early_rows = [
        row
        for row in selected
        if row["program"] == TEMPERATURE_PROGRAM
        and row["repeat"] < application_end
    ]
    if not early_rows:
        raise ValueError(f"No early {TEMPERATURE_PROGRAM} rows for tile {tile}")

    curve_start = min(
        min(row["days"] for row in early_rows),
        min(row["days"] for row in fit_rows),
    )
    curve_end = max(
        max(row["days"] for row in early_rows),
        max(row["days"] for row in fit_rows),
    )
    curve_days = np.linspace(curve_start, curve_end, 1200)
    curve_values = _evaluate_count_sinusoid(fit, curve_days)

    def predicted(row_list):
        return _evaluate_count_sinusoid(
            fit, np.asarray([row["days"] for row in row_list], dtype=float)
        )

    fit_prediction = predicted(fit_rows)
    early_prediction = predicted(early_rows)
    fit_residual = 1000.0 * np.log(
        np.asarray([row["acm_mean"] for row in fit_rows], dtype=float)
        / fit_prediction
    )
    early_pre_residual = 1000.0 * np.log(
        np.asarray([row["acm_mean"] for row in early_rows], dtype=float)
        / early_prediction
    )
    early_final_residual = 1000.0 * np.log(
        np.asarray([row["final_mean"] for row in early_rows], dtype=float)
        / early_prediction
    )

    figure, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    axis = axes[0]
    axis.plot(
        [row["days"] for row in selected],
        [row["raw_mean"] for row in selected],
        color="0.75",
        linewidth=0.7,
        label="raw detector",
    )
    axis.plot(
        [row["days"] for row in selected],
        [row["scaled_mean"] for row in selected],
        color="0.4",
        linewidth=0.8,
        label="photodiode/side-swap scaled",
    )
    axis.plot(
        [row["days"] for row in selected],
        [row["acm_mean"] for row in selected],
        color="tab:blue",
        linewidth=1.0,
        label="ACM-only",
    )
    axis.plot(
        [row["days"] for row in selected],
        [row["final_mean"] for row in selected],
        color="tab:orange",
        linewidth=1.0,
        label="final with temperature gain",
    )
    axis.scatter(
        [row["days"] for row in early_rows],
        [row["final_mean"] for row in early_rows],
        color="tab:green",
        s=8,
        label="early 1063 final",
        zorder=4,
    )
    axis.set_ylabel("mean current counts")
    axis.set_title(f"WFI01 tile {tile} LED 3: successive correction stages")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    axis.scatter(
        [row["days"] for row in reference_rows],
        [row["acm_mean"] for row in reference_rows],
        color="0.55",
        s=8,
        label="1039 ACM-only",
    )
    axis.scatter(
        [row["days"] for row in fit_rows],
        [row["acm_mean"] for row in fit_rows],
        color="tab:blue",
        s=26,
        label=f"fit points [{post_fit_start}, {post_fit_end})",
        zorder=4,
    )
    axis.scatter(
        [row["days"] for row in early_rows],
        [row["acm_mean"] for row in early_rows],
        color="tab:orange",
        s=8,
        label="early 1063 before temperature gain",
    )
    axis.scatter(
        [row["days"] for row in early_rows],
        [row["final_mean"] for row in early_rows],
        color="tab:green",
        s=8,
        label="early 1063 after temperature gain",
        zorder=4,
    )
    axis.plot(
        curve_days,
        curve_values,
        color="tab:red",
        linewidth=1.6,
        label=f"same fitted sine, P={fit['period_days']:.4g} d",
    )
    axis.axvline(min(row["days"] for row in early_rows), color="0.3", linestyle=":")
    axis.axvline(min(row["days"] for row in fit_rows), color="tab:blue", linestyle=":")
    axis.set_ylabel("mean current counts")
    axis.set_title("Fit window and extrapolation")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2]
    axis.scatter(
        [row["days"] for row in fit_rows],
        fit_residual,
        color="tab:blue",
        s=24,
        label="fit-point residual",
    )
    axis.scatter(
        [row["days"] for row in early_rows],
        early_pre_residual,
        color="tab:orange",
        s=8,
        label="early 1063 before temperature gain",
    )
    axis.scatter(
        [row["days"] for row in early_rows],
        early_final_residual,
        color="tab:green",
        s=8,
        label="early 1063 after temperature gain",
    )
    axis.axhline(0.0, color="0.3", linewidth=0.8)
    axis.set_xlabel("Days since first unified exposure")
    axis.set_ylabel("log residual to fitted sine (millilog)")
    axis.set_title("Temperature-fit residuals")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_tiles(value):
    return tuple(sorted({int(item) for item in value.split(",") if item.strip()}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tiles", type=parse_tiles, default=(0, 63))
    parser.add_argument("--post-fit-start-repeat", type=int, default=DEFAULT_POST_FIT_START)
    parser.add_argument("--post-fit-end-repeat", type=int, default=DEFAULT_POST_FIT_END)
    parser.add_argument("--wide-fit-start-repeat", type=int, default=DEFAULT_POST_ACM_START)
    parser.add_argument("--application-end-repeat", type=int, default=DEFAULT_APPLICATION_END)
    parser.add_argument("--operational-day", type=float, default=DEFAULT_OPERATIONAL_DAY)
    parser.add_argument("--physical-reset-day", type=float, default=DEFAULT_PHYSICAL_RESET_DAY)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.run_dir, args.detector)
    tiles = sorted({row["tile"] for row in rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_fit_summary(
        args.output_dir / f"{args.detector.lower()}_all_led_temperature_fit_summary.csv",
        rows,
        tiles,
        args.post_fit_start_repeat,
        args.post_fit_end_repeat,
        args.wide_fit_start_repeat,
    )
    for tile in args.tiles:
        if tile not in tiles:
            raise ValueError(f"Tile {tile} is not present in {args.run_dir}")
        plot_stage_comparison(
            args.output_dir / f"{args.detector.lower()}_tile_{tile:03d}_stages.png",
            rows,
            tile,
            args.operational_day,
            args.physical_reset_day,
        )
        plot_temperature_fits(
            args.output_dir / f"{args.detector.lower()}_tile_{tile:03d}_temperature_windows.png",
            rows,
            tile,
            args.post_fit_start_repeat,
            args.post_fit_end_repeat,
            args.wide_fit_start_repeat,
            args.application_end_repeat,
        )
        plot_factor_comparison(
            args.output_dir / f"{args.detector.lower()}_tile_{tile:03d}_temperature_factors.png",
            rows,
            tile,
            args.post_fit_start_repeat,
            args.post_fit_end_repeat,
            args.wide_fit_start_repeat,
            args.application_end_repeat,
        )
        plot_temperature_correction_stages(
            args.output_dir
            / f"{args.detector.lower()}_tile_{tile:03d}_led3_temperature_stages.png",
            rows,
            tile,
            args.post_fit_start_repeat,
            args.post_fit_end_repeat,
            args.application_end_repeat,
        )
    print(f"Loaded {len(rows)} valid scaled rows for {args.detector}")
    print(f"Wrote all-tile fit summary with {len(summary)} rows")
    print(f"Wrote diagnostic plots for tiles {args.tiles}")


if __name__ == "__main__":
    main()
