"""Plot all saved WFI02 regularization comparison fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_wfi_tmm_regularization_data_model import load_predictions, nonreference_rms
from plot_wfi_tmm_regularization_experiment import load_variant


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = (
    ROOT
    / "20261008/corrected_statistics_program1063_4h/production/wfi02/"
    "regularization_experiment"
)


def default_specs(root):
    return (
        ("baseline", "Baseline curvature", root / "tile_001/baseline"),
        ("curvature_off", "Curvature off", root / "tile_001/curvature_off"),
        ("maxent_0p01", "MaxEnt 0.01, cross-tile", root / "maxent_weight_sweep/weight_0p01/maxent"),
        ("maxent_0p03", "MaxEnt 0.03, cross-tile", root / "maxent_weight_sweep/weight_0p03/maxent"),
        ("maxent_0p05", "MaxEnt 0.05, cross-tile", root / "maxent_weight_sweep/weight_0p05/maxent"),
        ("maxent_0p10", "MaxEnt 0.10, cross-tile", root / "tile_001/maxent"),
        ("maxent_0p03_target_map", "MaxEnt 0.03, target constant MAP", root / "tile_001_constant_map_prior/weight_0p03/maxent"),
        ("maxent_0p10_target_map", "MaxEnt 0.10, target constant MAP", root / "tile_001_constant_map_prior/weight_0p1/maxent"),
    )


def load_fit(label, path):
    predictions = load_predictions(path)
    days, thickness, rate, reset_day = load_variant(path)
    if not np.array_equal(days, predictions["days"]):
        raise ValueError(f"Trajectory and prediction epochs differ for {label}")
    valid = predictions["repeat"] > np.min(predictions["repeat"])
    pre = days < reset_day
    post = days >= reset_day
    pre_rate = (thickness[pre][-1] - thickness[pre][0]) / (days[pre][-1] - days[pre][0])
    post_rate = (thickness[post][-1] - thickness[post][0]) / (days[post][-1] - days[post][0])
    return {
        "label": label,
        "path": Path(path),
        "predictions": predictions,
        "days": days,
        "thickness": thickness,
        "rate": rate,
        "reset_day": reset_day,
        "rms_nonreference": float(np.sqrt(np.mean(predictions["residual_log"][valid] ** 2))),
        "rms_all": float(np.sqrt(np.mean(predictions["residual_log"] ** 2))),
        "max_abs_residual": float(np.max(np.abs(predictions["residual_log"]))),
        "pre_rate": float(pre_rate),
        "post_rate": float(post_rate),
        "rate_difference": float(post_rate - pre_rate),
    }


def _grid(figure, axes, fits, plotter):
    for axis, fit in zip(axes.flat, fits):
        plotter(axis, fit)
    for axis in axes.flat[len(fits):]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))


def plot_data_model(fits, output_path):
    all_values = np.concatenate(
        [
            np.concatenate(
                (
                    fit["predictions"]["observed_log_change"],
                    fit["predictions"]["model_log_change"],
                )
            )
            for fit in fits
        ]
    )
    limit = max(float(np.max(np.abs(all_values))), 1.0e-6) * 1.08
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    def draw(axis, fit):
        values = fit["predictions"]
        axis.scatter(
            values["observed_log_change"],
            values["model_log_change"],
            s=8,
            alpha=0.62,
            color="tab:green" if "MaxEnt" in fit["label"] else "tab:blue",
            linewidths=0.0,
        )
        axis.plot([-limit, limit], [-limit, limit], "k--", linewidth=0.8)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_title(f"{fit['label']}\nRMS={fit['rms_nonreference']:.6f}", fontsize=10)
        axis.set_xlabel("Observed log change")
        axis.set_ylabel("Model log change")
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    _grid(figure, axes, fits, draw)
    figure.suptitle("WFI02 tile 001 observed versus model comparison", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_residuals(fits, output_path):
    residuals = np.concatenate([fit["predictions"]["residual_log"] for fit in fits])
    limit = max(float(np.max(np.abs(residuals))), 1.0e-6) * 1.12
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)

    def draw(axis, fit):
        values = fit["predictions"]
        axis.scatter(
            values["days"],
            values["residual_log"],
            s=8,
            alpha=0.62,
            color="tab:green" if "MaxEnt" in fit["label"] else "tab:blue",
            linewidths=0.0,
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(fit["reset_day"], color="0.25", linestyle=":", linewidth=0.9)
        axis.set_ylim(-limit, limit)
        axis.set_title(fit["label"], fontsize=10)
        axis.set_xlabel("Days")
        axis.set_ylabel("Data - model")
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    _grid(figure, axes, fits, draw)
    figure.suptitle("WFI02 tile 001 residuals; vertical line is decon2", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_trajectories(fits, output_path, values_key, ylabel, title):
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10", len(fits))
    for index, fit in enumerate(fits):
        values = fit[values_key]
        linestyle = "--" if "target constant MAP" in fit["label"] else "-"
        axis.plot(
            fit["days"],
            values,
            color=colors(index),
            linestyle=linestyle,
            linewidth=1.8,
            label=fit["label"],
        )
    axis.axvline(fits[0]["reset_day"], color="0.25", linestyle=":", linewidth=1.0)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, ncol=2, fontsize=9)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(fits, output_path):
    fields = (
        "label",
        "path",
        "rms_nonreference",
        "rms_all",
        "max_abs_residual",
        "pre_rate_nm_day",
        "post_rate_nm_day",
        "post_minus_pre_nm_day",
    )
    with Path(output_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fit in fits:
            writer.writerow(
                {
                    "label": fit["label"],
                    "path": str(fit["path"]),
                    "rms_nonreference": fit["rms_nonreference"],
                    "rms_all": fit["rms_all"],
                    "max_abs_residual": fit["max_abs_residual"],
                    "pre_rate_nm_day": fit["pre_rate"],
                    "post_rate_nm_day": fit["post_rate"],
                    "post_minus_pre_nm_day": fit["rate_difference"],
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.experiment_root / "all_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = default_specs(args.experiment_root)
    fits = [load_fit(label, path) for _, label, path in specs]
    plot_data_model(fits, output_dir / "data_vs_model_all_variants.png")
    plot_residuals(fits, output_dir / "residuals_all_variants.png")
    plot_trajectories(
        fits,
        output_dir / "ice_trajectories_all_variants.png",
        "thickness",
        "Ice thickness [nm]",
        "WFI02 tile 001 ice trajectories across regularization choices",
    )
    plot_trajectories(
        fits,
        output_dir / "instantaneous_rates_all_variants.png",
        "rate",
        "Instantaneous ice accumulation rate [nm/day]",
        "WFI02 tile 001 instantaneous rates across regularization choices",
    )
    write_summary(fits, output_dir / "comparison_summary.csv")
    print(f"Wrote comparison plots and summary under {output_dir}")


if __name__ == "__main__":
    main()
