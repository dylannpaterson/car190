"""Fit independent pure sinusoids to observed WFI LED trend data."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_wfi_ice_sinusoid_comparison import fit_sinusoid


LED_COLORS = {"old": "tab:blue", "rrn": "tab:orange"}
LED_MARKERS = {"old": "o", "rrn": "s"}


def load_acm_correction(output_dir):
    with np.load(Path(output_dir) / "posterior_samples.npz", allow_pickle=False) as archive:
        acm_step = np.asarray(archive["acm_step_log_offset"], dtype=float)
        observation_led = np.asarray(archive["observation_led"], dtype=int)
        observation_repeat = np.asarray(archive["observation_repeat"], dtype=int)
        observation_step_design = np.asarray(
            archive["observation_step_design"], dtype=float
        )
    if acm_step.ndim == 1:
        median_acm_step = acm_step
    else:
        median_acm_step = np.median(acm_step, axis=0)
    if median_acm_step.shape != (6,):
        raise ValueError("Expected six LED-specific ACM step offsets")
    return {
        (int(led), int(repeat)): float(median_acm_step[led - 1] * step_design)
        for led, repeat, step_design in zip(
            observation_led, observation_repeat, observation_step_design
        )
    }, median_acm_step


def load_led_data(path, tile_id=0, acm_output_dir=None):
    correction = None
    acm_steps = None
    if acm_output_dir is not None:
        correction, acm_steps = load_acm_correction(acm_output_dir)
    values = {led: {"days": [], "mean_log_change": []} for led in range(1, 7)}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["tile_id"]) != tile_id:
                continue
            count = float(row["n_pair_valid"])
            if count <= 0.0:
                continue
            led = int(row["led"])
            repeat = int(row["repeat"])
            mean_log_change = float(row["sum_log_change"]) / count
            if correction is not None:
                correction_key = (led, repeat)
                if correction_key not in correction:
                    raise ValueError(
                        f"No archived ACM correction for LED {led}, repeat {repeat}"
                    )
                mean_log_change -= correction[correction_key]
            values[led]["days"].append(float(row["days"]))
            values[led]["mean_log_change"].append(mean_log_change)
    for led, series in values.items():
        order = np.argsort(series["days"])
        series["days"] = np.asarray(series["days"], dtype=float)[order]
        series["mean_log_change"] = np.asarray(
            series["mean_log_change"], dtype=float
        )[order]
        if series["days"].size < 4:
            raise ValueError(f"LED {led} has too few observed points in {path}")
    return values, acm_steps


def evaluate_sinusoid(time_days, fit):
    relative_days = np.asarray(time_days, dtype=float) - float(time_days[0])
    return fit["offset_nm"] + fit["amplitude_nm"] * np.sin(
        2.0 * np.pi * relative_days / fit["period_days"] + fit["phase_rad"]
    )


def fit_datasets(old_path, rrn_path, old_acm_output, rrn_acm_output, tile_id):
    old_values, old_acm_steps = load_led_data(old_path, tile_id, old_acm_output)
    rrn_values, rrn_acm_steps = load_led_data(rrn_path, tile_id, rrn_acm_output)
    loaded = {"old": old_values, "rrn": rrn_values}
    acm_steps = {"old": old_acm_steps, "rrn": rrn_acm_steps}
    fits = {}
    for dataset, values in loaded.items():
        fits[dataset] = {}
        for led, series in values.items():
            fits[dataset][led] = fit_sinusoid(
                series["days"], series["mean_log_change"]
            )
    return loaded, fits, acm_steps


def write_summary(path, loaded, fits, acm_steps):
    fields = (
        "dataset",
        "led",
        "n_points",
        "acm_step_log_offset",
        "period_days",
        "offset_log_change",
        "amplitude_log_change",
        "phase_rad",
        "sse_log_change_squared",
        "rmse_log_change",
        "r2",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in ("old", "rrn"):
            for led in range(1, 7):
                fit = fits[dataset][led]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "led": led,
                        "n_points": loaded[dataset][led]["days"].size,
                        "acm_step_log_offset": acm_steps[dataset][led - 1],
                        "period_days": fit["period_days"],
                        "offset_log_change": fit["offset_nm"],
                        "amplitude_log_change": fit["amplitude_nm"],
                        "phase_rad": fit["phase_rad"],
                        "sse_log_change_squared": fit["sse_nm2"],
                        "rmse_log_change": fit["rmse_nm"],
                        "r2": fit["r2"],
                    }
                )


def plot_data_and_models(path, loaded, fits):
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        sharex=True,
        constrained_layout=True,
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        maximum_day = 0.0
        for dataset in ("old", "rrn"):
            series = loaded[dataset][led]
            fit = fits[dataset][led]
            days = series["days"]
            observed = 1000.0 * series["mean_log_change"]
            axis.scatter(
                days,
                observed,
                color=LED_COLORS[dataset],
                marker=LED_MARKERS[dataset],
                s=22,
                alpha=0.55,
                label=f"{dataset.upper()} data",
            )
            grid = np.linspace(days[0], days[-1], 500)
            model = 1000.0 * evaluate_sinusoid(grid, fit)
            axis.plot(
                grid,
                model,
                color=LED_COLORS[dataset],
                linewidth=2.0,
                label=(
                    f"{dataset.upper()} sinusoid: "
                    f"P={fit['period_days']:.3g} d, "
                    f"R2={fit['r2']:.3f}"
                ),
            )
            maximum_day = max(maximum_day, float(days[-1]))
        axis.set_title(f"LED {led}")
        axis.set_ylabel("ACM-corrected mean log change (millilog)")
        axis.set_xlim(left=0.0, right=maximum_day)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Days since first exposure")
    figure.suptitle(
        "ACM-corrected observed WFI01 tile-0 data versus independent LED sinusoids"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--rrn", type=Path, required=True)
    parser.add_argument("--old-acm-output", type=Path, required=True)
    parser.add_argument("--rrn-acm-output", type=Path, required=True)
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    loaded, fits, acm_steps = fit_datasets(
        args.old,
        args.rrn,
        args.old_acm_output,
        args.rrn_acm_output,
        args.tile_id,
    )
    plot_data_and_models(args.output, loaded, fits)
    write_summary(args.summary, loaded, fits, acm_steps)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary}")
    for dataset in ("old", "rrn"):
        for led in range(1, 7):
            fit = fits[dataset][led]
            print(
                f"{dataset} LED {led}: points={loaded[dataset][led]['days'].size}, "
                f"period={fit['period_days']:.8g} d, "
                f"RMSE={fit['rmse_nm']:.8g} log, R2={fit['r2']:.8g}"
            )


if __name__ == "__main__":
    main()
