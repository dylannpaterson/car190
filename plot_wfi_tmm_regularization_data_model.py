"""Plot observed WFI tile statistics against regularization experiment models."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = (
    ROOT
    / "20261008/corrected_statistics_program1063_4h/production/wfi02/"
    "regularization_experiment/tile_001"
)
VARIANTS = ("baseline", "curvature_off", "maxent")
LABELS = {
    "baseline": "Baseline curvature",
    "curvature_off": "Curvature off",
    "maxent": "MaxEnt prior",
}
COLORS = {
    "baseline": "tab:blue",
    "curvature_off": "tab:orange",
    "maxent": "tab:green",
}


def load_predictions(output_dir):
    with (Path(output_dir) / "tile_group_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows found in {output_dir}")
    values = {
        field: np.asarray([float(row[field]) for row in rows], dtype=float)
        for field in (
            "days",
            "observed_log_change",
            "model_log_change",
            "residual_log",
        )
    }
    values["repeat"] = np.asarray([int(row["repeat"]) for row in rows], dtype=int)
    with np.load(
        Path(output_dir) / "posterior_samples.npz",
        allow_pickle=False,
    ) as archive:
        values["reset_day"] = float(np.asarray(archive["ice_reset_day"]).item())
    return values


def nonreference_rms(values):
    valid = values["repeat"] > np.min(values["repeat"])
    return float(np.sqrt(np.mean(values["residual_log"][valid] ** 2)))


def plot_comparison(experiment_root, variants, output_path):
    loaded = {
        variant: load_predictions(Path(experiment_root) / variant)
        for variant in variants
    }
    reset_days = {round(values["reset_day"], 12) for values in loaded.values()}
    if len(reset_days) != 1:
        raise ValueError("Experiment variants have inconsistent reset days")
    reset_day = next(iter(reset_days))

    observed_and_model = np.concatenate(
        [
            np.concatenate((values["observed_log_change"], values["model_log_change"]))
            for values in loaded.values()
        ]
    )
    residuals = np.concatenate([values["residual_log"] for values in loaded.values()])
    data_limit = max(float(np.max(np.abs(observed_and_model))), 1.0e-6) * 1.08
    residual_limit = max(float(np.max(np.abs(residuals))), 1.0e-6) * 1.12

    figure, axes = plt.subplots(
        len(variants),
        2,
        figsize=(14, 4.0 * len(variants)),
        sharex="col",
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, variant in enumerate(variants):
        values = loaded[variant]
        days = values["days"]
        data_axis, residual_axis = axes[row_index]
        data_axis.scatter(
            values["observed_log_change"],
            values["model_log_change"],
            color=COLORS[variant],
            s=9,
            alpha=0.65,
            linewidths=0.0,
        )
        data_axis.plot(
            [-data_limit, data_limit],
            [-data_limit, data_limit],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="1:1",
        )
        data_axis.set_ylim(-data_limit, data_limit)
        data_axis.set_xlim(-data_limit, data_limit)
        data_axis.set_xlabel("Observed log change")
        data_axis.set_ylabel("Model log change")
        data_axis.set_title(
            f"{LABELS[variant]}: observed versus model; "
            f"non-reference RMS = {nonreference_rms(values):.6f}"
        )
        data_axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        data_axis.legend(frameon=False, loc="upper right")

        residual_axis.scatter(
            days,
            values["residual_log"],
            color=COLORS[variant],
            s=7,
            alpha=0.65,
            linewidths=0.0,
        )
        residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        residual_axis.axvline(reset_day, color="0.25", linestyle=":", linewidth=1.0)
        residual_axis.set_ylim(-residual_limit, residual_limit)
        residual_axis.set_ylabel("data - model")
        residual_axis.set_title("Residuals")
        residual_axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    axes[-1, 1].set_xlabel("Days since first exposure")
    figure.suptitle(
        f"WFI02 tile 001 observed statistics versus fitted model; decon2 day {reset_day:.3f}",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    return parser.parse_args()


def main():
    args = parse_args()
    variants = tuple(args.variant) if args.variant else VARIANTS
    output = args.output or args.experiment_root / "data_vs_model_comparison.png"
    plot_comparison(args.experiment_root, variants, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
