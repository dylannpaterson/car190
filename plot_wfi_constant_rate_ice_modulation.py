"""Plot residuals and correlations for constant-rate ice-index diagnostics."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production
from run_wfi_constant_rate_ice_modulation import (
    load_constant_rate_initial_values,
)


DETECTORS = ("wfi01", "wfi02")
LED_COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#f0e442", "#56b4e9")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--constant-rate-dir", type=Path, required=True)
    parser.add_argument("--modulation-dir", type=Path, required=True)
    parser.add_argument("--tight-modulation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_fit_predictions(statistics_path, constant_rate_map, modulation_dir):
    data = production.load_tile_statistics(statistics_path)
    context = production.load_response_context(data["sca_names"])

    control_values = load_constant_rate_initial_values(
        constant_rate_map,
        data,
        production.ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    )
    control_values.pop("ice_n_log_modulation")
    control_model = production.build_model(data, context, constant_rate=True)
    control_state = control_model.production_state(control_values)

    modulation_initial = production.initial_values_for_data(
        data,
        constant_rate=True,
        fit_ice_n_modulation=True,
    )
    modulation_values = production.load_map_warm_start(
        modulation_dir,
        modulation_initial,
        data,
    )
    modulation_model = production.build_model(
        data,
        context,
        constant_rate=True,
        fit_ice_n_modulation=True,
    )
    modulation_state = modulation_model.production_state(modulation_values)

    observed = np.asarray(data["sums"], dtype=float)[0] / np.asarray(
        data["counts"], dtype=float
    )[0]
    post_reset = np.asarray(data["days"]) >= data["reset_day"]
    offset = float(np.asarray(control_values["post_reset_tile_log_offset"]).reshape(-1)[0])
    observed = observed + offset * post_reset

    return {
        "data": data,
        "observed": observed,
        "control": {
            "prediction": np.asarray(control_state["prediction"], dtype=float)[0],
            "values": control_values,
        },
        "modulation": {
            "prediction": np.asarray(modulation_state["prediction"], dtype=float)[0],
            "values": modulation_values,
        },
    }


def lag_correlation(values):
    values = np.asarray(values, dtype=float)
    if values.size < 3 or np.std(values[:-1]) == 0.0 or np.std(values[1:]) == 0.0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def correlation_rows(fit, detector):
    data = fit["data"]
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    repeats = np.asarray(data["repeat"], dtype=int)
    rows = []
    for model_name in ("control", "modulation"):
        residual = fit[model_name]["prediction"] - fit["observed"]
        for led in range(1, 7):
            mask = leds == led
            order = np.argsort(days[mask])
            values = residual[mask][order]
            ordered_days = days[mask][order]
            rows.append(
                {
                    "detector": detector,
                    "model": model_name,
                    "series": f"LED{led}",
                    "n": values.size,
                    "lag1_correlation": lag_correlation(values),
                    "day_correlation": float(np.corrcoef(values, ordered_days)[0, 1]),
                    "rms": float(np.sqrt(np.mean(values**2))),
                    "mean": float(np.mean(values)),
                }
            )
        common = np.asarray(
            [
                np.mean(residual[repeats == repeat])
                for repeat in np.unique(repeats)
            ]
        )
        rows.append(
            {
                "detector": detector,
                "model": model_name,
                "series": "common_repeat_mean",
                "n": common.size,
                "lag1_correlation": lag_correlation(common),
                "day_correlation": np.nan,
                "rms": float(np.sqrt(np.mean(common**2))),
                "mean": float(np.mean(common)),
            }
        )
    return rows


def plot_residuals(fit, detector, output_path):
    data = fit["data"]
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(f"{detector.upper()} tile 000: constant-rate residuals")

    for axis, model_name, title in (
        (axes[0, 0], "control", "Fixed n = 1.25"),
        (axes[0, 1], "modulation", "Six-node n(lambda) modulation"),
    ):
        residual = fit[model_name]["prediction"] - fit["observed"]
        for led, color in zip(range(1, 7), LED_COLORS):
            mask = leds == led
            order = np.argsort(days[mask])
            axis.plot(
                days[mask][order],
                1e3 * residual[mask][order],
                marker=".",
                linewidth=0.8,
                markersize=3,
                color=color,
                label=f"LED {led}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("prediction - observed [10^-3]")
        axis.grid(alpha=0.2)

    rms = {model: [] for model in ("control", "modulation")}
    lag = {model: [] for model in ("control", "modulation")}
    for model_name in rms:
        residual = fit[model_name]["prediction"] - fit["observed"]
        for led in range(1, 7):
            values = residual[leds == led]
            values = values[np.argsort(days[leds == led])]
            rms[model_name].append(1e3 * np.sqrt(np.mean(values**2)))
            lag[model_name].append(lag_correlation(values))

    led_positions = np.arange(1, 7)
    width = 0.38
    axes[1, 0].bar(
        led_positions - width / 2,
        rms["control"],
        width,
        label="Fixed n",
        color="#999999",
    )
    axes[1, 0].bar(
        led_positions + width / 2,
        rms["modulation"],
        width,
        label="n(lambda) modulation",
        color="#0072b2",
    )
    axes[1, 0].set_title("Residual RMS by LED")
    axes[1, 0].set_xlabel("LED")
    axes[1, 0].set_ylabel("RMS [10^-3]")
    axes[1, 0].set_xticks(led_positions)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].bar(
        led_positions - width / 2,
        lag["control"],
        width,
        label="Fixed n",
        color="#999999",
    )
    axes[1, 1].bar(
        led_positions + width / 2,
        lag["modulation"],
        width,
        label="n(lambda) modulation",
        color="#d55e00",
    )
    axes[1, 1].set_title("Lag-1 residual correlation")
    axes[1, 1].set_xlabel("LED")
    axes[1, 1].set_ylabel("corr(residual[t], residual[t+1])")
    axes[1, 1].set_xticks(led_positions)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.2)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_modulation_curves(fits, output_path):
    wavelengths = np.linspace(605.0, 1744.0, 400)
    basis = production.ice_n_modulation_basis(wavelengths)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for detector, fit in fits.items():
        color = "#0072b2" if detector == "wfi01" else "#d55e00"
        for modulation_dir_name, linestyle, label_suffix in (
            ("modulation", "-", "sigma=0.05"),
            ("tight_modulation", "--", "sigma=0.02"),
        ):
            values = fit[modulation_dir_name]["values"]
            delta = np.asarray(values["ice_n_log_modulation"]).reshape(-1)
            factor = np.exp(basis @ delta)
            axes[0].plot(
                wavelengths,
                factor,
                linestyle=linestyle,
                color=color,
                label=f"{detector.upper()} {label_suffix}",
            )
    axes[0].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_title("Fitted real-index modulation")
    axes[0].set_xlabel("wavelength [nm]")
    axes[0].set_ylabel("n(lambda) / 1.25")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    labels = []
    control_rates = []
    modulation_rates = []
    tight_rates = []
    for detector, fit in fits.items():
        labels.append(detector.upper())
        control_rates.append(
            np.exp(float(np.asarray(fit["control"]["values"]["log_rate_scale"]).reshape(-1)[0]))
        )
        modulation_rates.append(
            np.exp(float(np.asarray(fit["modulation"]["values"]["log_rate_scale"]).reshape(-1)[0]))
        )
        tight_rates.append(
            np.exp(
                float(
                    np.asarray(fit["tight_modulation"]["values"]["log_rate_scale"])
                    .reshape(-1)[0]
                )
            )
        )
    positions = np.arange(len(labels))
    width = 0.25
    axes[1].bar(positions - width, control_rates, width, label="Fixed n", color="#999999")
    axes[1].bar(positions, modulation_rates, width, label="sigma=0.05", color="#0072b2")
    axes[1].bar(positions + width, tight_rates, width, label="sigma=0.02", color="#d55e00")
    axes[1].set_title("Constant-rate MAP")
    axes[1].set_ylabel("rate [nm/day]")
    axes[1].set_xticks(positions, labels)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits = {}
    summary_rows = []
    for detector in DETECTORS:
        statistics_path = args.statistics_dir / detector / "tile_statistics" / (
            f"{detector}_tile_log_statistics_512_tile_000.csv"
        )
        constant_rate_map = (
            args.constant_rate_dir / detector / "fits" / "tile_000" / "constant_rate_map.npz"
        )
        modulation_dir = args.modulation_dir / f"{detector}_tile_000"
        tight_modulation_dir = args.tight_modulation_dir / f"{detector}_tile_000"
        fit = load_fit_predictions(statistics_path, constant_rate_map, modulation_dir)
        tight = load_fit_predictions(statistics_path, constant_rate_map, tight_modulation_dir)
        fit["tight_modulation"] = tight["modulation"]
        fits[detector] = fit
        summary_rows.extend(correlation_rows(fit, detector))
        plot_residuals(
            fit,
            detector,
            args.output_dir / f"{detector}_tile_000_residuals.png",
        )

    plot_modulation_curves(fits, args.output_dir / "constant_rate_modulation_summary.png")
    summary_path = args.output_dir / "residual_correlations.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "detector",
                "model",
                "series",
                "n",
                "lag1_correlation",
                "day_correlation",
                "rms",
                "mean",
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote plots and correlation summary to {args.output_dir}")


if __name__ == "__main__":
    main()
