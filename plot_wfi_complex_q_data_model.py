"""Plot current WFI01 complex-q data/model curves and residuals."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = (
    ROOT / "wfi01_tile0_complex_q_production_fit" / "predictions.csv"
)
DEFAULT_SUMMARY = ROOT / "wfi01_tile0_complex_q_production_fit" / "fit_summary.json"
DEFAULT_DATA_OUTPUT = (
    ROOT / "wfi01_tile0_complex_q_production_fit" / "complex_q_data_vs_model.png"
)
DEFAULT_RESIDUAL_OUTPUT = (
    ROOT / "wfi01_tile0_complex_q_production_fit" / "complex_q_residuals.png"
)
RESET_DAY = 15.064405023148149


def load_predictions(path):
    required = {
        "led",
        "days",
        "observed_log_change",
        "production_log_change",
        "complex_q_log_change",
        "production_residual_log",
        "complex_q_residual_log",
    }
    by_led = {led: [] for led in range(1, 7)}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"Prediction table is missing columns: {missing}")
        for row in reader:
            led = int(row["led"])
            if led not in by_led:
                raise ValueError(f"Unexpected LED in prediction table: {led}")
            by_led[led].append(
                {
                    key: float(row[key])
                    for key in (
                        "days",
                        "observed_log_change",
                        "production_log_change",
                        "complex_q_log_change",
                        "production_residual_log",
                        "complex_q_residual_log",
                    )
                }
            )
    if any(not rows for rows in by_led.values()):
        raise ValueError("Prediction table must contain all six LEDs")
    result = {}
    for led, rows in by_led.items():
        rows.sort(key=lambda row: row["days"])
        result[led] = {
            key: np.asarray([row[key] for row in rows], dtype=float)
            for key in rows[0]
        }
    return result


def load_summary(path):
    if not Path(path).is_file():
        return None
    with Path(path).open() as handle:
        return json.load(handle)


def add_common_axis_format(axis, title):
    axis.set_title(title)
    axis.axvline(RESET_DAY, color="0.35", linestyle=":", linewidth=0.8, alpha=0.8)
    axis.grid(alpha=0.2)


def plot_data_vs_model(path, predictions):
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(13, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        values = predictions[led]
        days = values["days"]
        axis.scatter(
            days,
            100.0 * np.expm1(values["observed_log_change"]),
            color="black",
            s=13,
            alpha=0.75,
            label="data",
            zorder=3,
        )
        axis.plot(
            days,
            100.0 * np.expm1(values["production_log_change"]),
            color="tab:orange",
            linestyle="--",
            linewidth=1.4,
            label="production model",
        )
        axis.plot(
            days,
            100.0 * np.expm1(values["complex_q_log_change"]),
            color="tab:blue",
            linewidth=1.5,
            label="complex-q model",
        )
        add_common_axis_format(axis, f"LED {led}")

    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.supxlabel("Days since first exposure")
    figure.supylabel("Reference-normalized change (%)")
    figure.suptitle("WFI01 tile 0: data versus production and fitted complex-q models")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_residuals(path, predictions, summary):
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(13, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        values = predictions[led]
        days = values["days"]
        production_residual = 1000.0 * values["production_residual_log"]
        complex_q_residual = 1000.0 * values["complex_q_residual_log"]
        axis.plot(
            days,
            production_residual,
            color="tab:orange",
            linestyle="--",
            linewidth=1.4,
            label="production residual",
        )
        axis.plot(
            days,
            complex_q_residual,
            color="tab:blue",
            linewidth=1.5,
            label="complex-q residual",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        add_common_axis_format(axis, f"LED {led}")

    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.supxlabel("Days since first exposure")
    figure.supylabel("Residual [10^-3 log]")
    title = "WFI01 tile 0: data-model residuals"
    if summary is not None:
        baseline = 1000.0 * summary["baseline"]["unweighted_rms_log"]
        complex_q = 1000.0 * summary["complex_q"]["unweighted_rms_log"]
        title += f"; RMS production={baseline:.4f}, complex-q={complex_q:.4f}"
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument(
        "--residual-output", type=Path, default=DEFAULT_RESIDUAL_OUTPUT
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = load_predictions(args.predictions)
    summary = load_summary(args.summary)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.residual_output.parent.mkdir(parents=True, exist_ok=True)
    plot_data_vs_model(args.data_output, predictions)
    plot_residuals(args.residual_output, predictions, summary)
    print(f"Wrote {args.data_output}")
    print(f"Wrote {args.residual_output}")


if __name__ == "__main__":
    main()
