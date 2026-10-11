"""Plot residuals for the optional AR/HgCdTe spectral-modulation diagnostic."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production
from plot_wfi_constant_rate_ice_modulation import LED_COLORS, lag_correlation
from run_wfi_constant_rate_ar_hgcdte_modulation import (
    load_constant_rate_initial_values,
)


DETECTORS = ("wfi01", "wfi02")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--constant-rate-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_fit(statistics_path, constant_rate_map, extension_dir):
    data = production.load_tile_statistics(statistics_path)
    context = production.load_response_context(data["sca_names"])

    control_values = production.initial_values_for_data(data, constant_rate=True)
    with np.load(constant_rate_map, allow_pickle=False) as archived:
        for name, target in control_values.items():
            value = np.asarray(archived[name], dtype=float)
            target_array = np.asarray(target)
            while value.ndim > target_array.ndim and value.shape[0] == 1:
                value = value[0]
            control_values[name] = value
    control_model = production.build_model(data, context, constant_rate=True)
    control_state = control_model.production_state(control_values)

    extension_initial = production.initial_values_for_data(
        data,
        constant_rate=True,
        fit_ar_hgcdte_modulation=True,
    )
    extension_values = production.load_map_warm_start(
        extension_dir,
        extension_initial,
        data,
    )
    extension_model = production.build_model(
        data,
        context,
        constant_rate=True,
        fit_ar_hgcdte_modulation=True,
    )
    extension_state = extension_model.production_state(extension_values)

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
        "extension": {
            "prediction": np.asarray(extension_state["prediction"], dtype=float)[0],
            "values": extension_values,
        },
    }


def summary_rows(fit, detector):
    data = fit["data"]
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    repeats = np.asarray(data["repeat"], dtype=int)
    rows = []
    for model_name in ("control", "extension"):
        residual = fit[model_name]["prediction"] - fit["observed"]
        for led in range(1, 7):
            mask = leds == led
            order = np.argsort(days[mask])
            values = residual[mask][order]
            rows.append(
                {
                    "detector": detector,
                    "model": model_name,
                    "series": f"LED{led}",
                    "n": int(values.size),
                    "lag1_correlation": lag_correlation(values),
                    "rms": float(np.sqrt(np.mean(values**2))),
                    "mean": float(np.mean(values)),
                }
            )
        common = np.asarray(
            [np.mean(residual[repeats == repeat]) for repeat in np.unique(repeats)]
        )
        rows.append(
            {
                "detector": detector,
                "model": model_name,
                "series": "common_repeat_mean",
                "n": int(common.size),
                "lag1_correlation": lag_correlation(common),
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
    figure.suptitle(f"{detector.upper()} tile 000: AR/HgCdTe modulation")
    for axis, model_name, title in (
        (axes[0, 0], "control", "Fixed AR/HgCdTe reference"),
        (axes[0, 1], "extension", "Anchored spectral modulation"),
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

    rms = {name: [] for name in ("control", "extension")}
    lag = {name: [] for name in ("control", "extension")}
    for model_name in rms:
        residual = fit[model_name]["prediction"] - fit["observed"]
        for led in range(1, 7):
            mask = leds == led
            values = residual[mask]
            values = values[np.argsort(days[mask])]
            rms[model_name].append(1e3 * np.sqrt(np.mean(values**2)))
            lag[model_name].append(lag_correlation(values))
    positions = np.arange(1, 7)
    width = 0.38
    axes[1, 0].bar(
        positions - width / 2,
        rms["control"],
        width,
        label="Fixed reference",
        color="#999999",
    )
    axes[1, 0].bar(
        positions + width / 2,
        rms["extension"],
        width,
        label="Spectral modulation",
        color="#0072b2",
    )
    axes[1, 0].set_title("Residual RMS by LED")
    axes[1, 0].set_xlabel("LED")
    axes[1, 0].set_ylabel("RMS [10^-3]")
    axes[1, 0].set_xticks(positions)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)
    axes[1, 1].bar(
        positions - width / 2,
        lag["control"],
        width,
        label="Fixed reference",
        color="#999999",
    )
    axes[1, 1].bar(
        positions + width / 2,
        lag["extension"],
        width,
        label="Spectral modulation",
        color="#d55e00",
    )
    axes[1, 1].set_title("Lag-1 residual correlation")
    axes[1, 1].set_xlabel("LED")
    axes[1, 1].set_ylabel("corr(residual[t], residual[t+1])")
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_modulations(fits, output_path):
    wavelengths = np.linspace(605.0, 1744.0, 400)
    basis = production.spectral_modulation_basis(wavelengths)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for detector, fit in fits.items():
        color = "#0072b2" if detector == "wfi01" else "#d55e00"
        values = fit["extension"]["values"]
        ar_delta = np.asarray(values["ar_n_log_modulation"]).reshape(
            production.AR_LAYERS, -1
        )
        for layer in range(production.AR_LAYERS):
            factor = np.exp(basis @ ar_delta[layer])
            axes[0, 0].plot(
                wavelengths,
                factor,
                color=color,
                linestyle=("-", "--", ":")[layer],
                label=f"{detector.upper()} AR {layer + 1}",
            )
        hg_n = np.asarray(values["hgcdte_n_log_modulation"]).reshape(-1)
        hg_k = np.asarray(values["hgcdte_k_log_modulation"]).reshape(-1)
        axes[0, 1].plot(
            wavelengths,
            np.exp(basis @ hg_n),
            color=color,
            label=f"{detector.upper()} HgCdTe n",
        )
        axes[0, 1].plot(
            wavelengths,
            np.exp(basis @ hg_k),
            color=color,
            linestyle="--",
            label=f"{detector.upper()} HgCdTe k",
        )
    for axis, title in (
        (axes[0, 0], "AR n(lambda) factors"),
        (axes[0, 1], "HgCdTe n/k factors"),
    ):
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("wavelength [nm]")
        axis.set_ylabel("factor at fixed scale")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)

    labels = []
    rate_control = []
    rate_extension = []
    rms_control = []
    rms_extension = []
    for detector, fit in fits.items():
        labels.append(detector.upper())
        rate_control.append(
            np.exp(float(np.asarray(fit["control"]["values"]["log_rate_scale"]).reshape(-1)[0]))
        )
        rate_extension.append(
            np.exp(float(np.asarray(fit["extension"]["values"]["log_rate_scale"]).reshape(-1)[0]))
        )
        rms_control.append(
            np.sqrt(
                np.mean(
                    (fit["control"]["prediction"] - fit["observed"]) ** 2
                )
            )
        )
        rms_extension.append(
            np.sqrt(
                np.mean(
                    (fit["extension"]["prediction"] - fit["observed"]) ** 2
                )
            )
        )
    positions = np.arange(len(labels))
    width = 0.35
    axes[1, 0].bar(positions - width / 2, rate_control, width, label="Fixed", color="#999999")
    axes[1, 0].bar(positions + width / 2, rate_extension, width, label="Extension", color="#0072b2")
    axes[1, 0].set_title("Constant-rate MAP")
    axes[1, 0].set_ylabel("rate [nm/day]")
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)
    axes[1, 1].bar(positions - width / 2, np.asarray(rms_control) * 1e3, width, label="Fixed", color="#999999")
    axes[1, 1].bar(positions + width / 2, np.asarray(rms_extension) * 1e3, width, label="Extension", color="#d55e00")
    axes[1, 1].set_title("Overall residual RMS")
    axes[1, 1].set_ylabel("RMS [10^-3]")
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits = {}
    rows = []
    for detector in DETECTORS:
        statistics_path = args.statistics_dir / detector / "tile_statistics" / (
            f"{detector}_tile_log_statistics_512_tile_000.csv"
        )
        map_path = (
            args.constant_rate_dir / detector / "fits" / "tile_000" / "constant_rate_map.npz"
        )
        extension_dir = args.extension_dir / f"{detector}_tile_000"
        fit = load_fit(statistics_path, map_path, extension_dir)
        fits[detector] = fit
        rows.extend(summary_rows(fit, detector))
        plot_residuals(
            fit,
            detector,
            args.output_dir / f"{detector}_tile_000_residuals.png",
        )
    plot_modulations(fits, args.output_dir / "ar_hgcdte_modulation_summary.png")
    with (args.output_dir / "residual_correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("detector", "model", "series", "n", "lag1_correlation", "rms", "mean"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote plots and correlation summary to {args.output_dir}")


if __name__ == "__main__":
    main()
