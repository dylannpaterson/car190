"""Plot tile-136 data/model trajectories and LED residual diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import fit_wfi_tmm_production as production
from assess_rcs_spectral_identifiability import load_qe, load_sed


RESET_DAY = 12.907397928240743
DETECTORS = ("wfi01", "wfi10")
LED_COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#f0e442", "#56b4e9")
VARIANT_NAMES = (
    ("baseline", "Production baseline"),
    ("mod002", "AR/HgCdTe modulation, sigma=0.02"),
    ("mod010", "AR/HgCdTe modulation, sigma=0.10"),
    ("all025", "All n + HgCdTe k modulation, sigma=0.25"),
    ("all050", "All n + HgCdTe k modulation, sigma=0.50"),
    ("ice001", "Ice n sigma=0.01; AR/HgCdTe sigma=0.25"),
    (
        "ice001_hn5",
        "Ice n sigma=0.01; half-normal d0 sigma=5 nm",
    ),
    (
        "iceconstant_hn5",
        "Constant ice n; angle fixed; half-normal d0 sigma=5 nm",
    ),
    (
        "iceconstant_hn5_freeangle",
        "Constant ice n; angle fitted; half-normal d0 sigma=5 nm",
    ),
    (
        "iceconstant005_fixedangle",
        "Constant ice n; prior sigma=0.05; angle fixed",
    ),
)


def effective_led_wavelengths():
    spectra = load_sed(production.SED_URL)
    qe_wavelengths, _, _ = load_qe(production.QE_URL)
    effective = []
    for band in range(1, 7):
        spectrum = spectra[band]
        wavelengths = np.asarray(spectrum["wavelength"], dtype=float)
        output = np.where(
            (wavelengths >= qe_wavelengths[0])
            & (wavelengths <= qe_wavelengths[-1]),
            production._sed_output(spectrum),
            0.0,
        )
        weights = output * production._trapezoid_sample_weights(wavelengths)
        effective.append(float(np.sum(weights * wavelengths) / np.sum(weights)))
    return np.asarray(effective)


def load_predictions(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    return {
        "group": np.asarray([int(row["group"]) for row in rows]),
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "repeat": np.asarray([int(row["repeat"]) for row in rows]),
        "observed": np.asarray(
            [float(row["observed_log_change"]) for row in rows]
        ),
        "model": np.asarray([float(row["model_log_change"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }


def variant_path(root, detector, variant):
    if variant == "baseline":
        return (
            root
            / detector
            / "n_prior_sigma_0p05"
            / "fits"
            / "tile_136"
            / "tile_group_predictions.csv"
        )
    if variant == "all025":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_modulation_prior025"
            / "tile_group_predictions.csv"
        )
    if variant == "all050":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_modulation_prior050"
            / "tile_group_predictions.csv"
        )
    if variant == "ice001":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_iceprior001_arhgcdteprior025"
            / "tile_group_predictions.csv"
        )
    if variant == "ice001_hn5":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_iceprior001_arhgcdteprior025_initialhalfnormal5_correct"
            / "tile_group_predictions.csv"
        )
    if variant == "iceconstant_hn5":
        directory = "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fixedangle"
        return root / detector / "diagnostics" / directory / "tile_group_predictions.csv"
    if variant == "iceconstant_hn5_freeangle":
        directory = {
            "wfi01": "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fromconstant",
            "wfi10": "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fromemcee",
        }[detector]
        return root / detector / "diagnostics" / directory / "tile_group_predictions.csv"
    if variant == "iceconstant005_fixedangle":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_sig005_fixedangle"
            / "tile_group_predictions.csv"
        )
    suffix = "" if variant == "mod002" else "_modsigma010"
    return (
        root
        / detector
        / "diagnostics"
        / f"tile_136_hgcdte_modulation_matched_np_nprior005{suffix}"
        / "tile_group_predictions.csv"
    )


def load_all(root):
    fits = {}
    for detector in DETECTORS:
        fits[detector] = {
            variant: load_predictions(variant_path(root, detector, variant))
            for variant, _ in VARIANT_NAMES
        }
    return fits


def add_common_axis_format(axis):
    axis.axvline(RESET_DAY, color="0.45", linestyle="--", linewidth=0.8)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.grid(alpha=0.2)


def plot_data_model(fits, output):
    variant_count = len(VARIANT_NAMES)
    figure, axes = plt.subplots(
        2,
        variant_count,
        figsize=(5 * variant_count, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for detector_index, detector in enumerate(DETECTORS):
        for variant_index, (variant, title) in enumerate(VARIANT_NAMES):
            axis = axes[detector_index, variant_index]
            values = fits[detector][variant]
            for led, color in enumerate(LED_COLORS, start=1):
                mask = values["led"] == led
                order = np.argsort(values["days"][mask])
                axis.scatter(
                    values["days"][mask],
                    1000.0 * values["observed"][mask],
                    color=color,
                    s=7,
                    alpha=0.45,
                    edgecolors="none",
                )
                axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["model"][mask][order],
                    color=color,
                    linewidth=0.9,
                )
            add_common_axis_format(axis)
            axis.set_title(title, fontsize=10)
            if detector_index == 1:
                axis.set_xlabel("Days since first exposure")
            if variant_index == 0:
                axis.set_ylabel(f"{detector.upper()}\nlog change [10^-3]")
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="data"),
        Line2D([], [], color="black", linewidth=1.2, label="model"),
        Line2D([], [], color="0.45", linestyle="--", linewidth=0.8, label="ice reset"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("WFI tile 136: corrected data versus model by LED", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_residuals(fits, output):
    variant_count = len(VARIANT_NAMES)
    figure, axes = plt.subplots(
        2,
        variant_count,
        figsize=(5 * variant_count, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for detector_index, detector in enumerate(DETECTORS):
        for variant_index, (variant, title) in enumerate(VARIANT_NAMES):
            axis = axes[detector_index, variant_index]
            values = fits[detector][variant]
            for led, color in enumerate(LED_COLORS, start=1):
                mask = values["led"] == led
                order = np.argsort(values["days"][mask])
                axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["residual"][mask][order],
                    color=color,
                    marker=".",
                    markersize=2.5,
                    linewidth=0.65,
                )
            add_common_axis_format(axis)
            axis.set_title(title, fontsize=10)
            if detector_index == 1:
                axis.set_xlabel("Days since first exposure")
            if variant_index == 0:
                axis.set_ylabel(f"{detector.upper()}\nobserved - model [10^-3]")
    handles = [
        Line2D([], [], color=color, linewidth=1.2, label=f"LED {led}")
        for led, color in enumerate(LED_COLORS, start=1)
    ]
    axes[0, 0].legend(handles=handles, ncol=3, frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("WFI tile 136: residuals versus time", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(fits, effective_wavelengths, output):
    fieldnames = [
        "detector",
        "variant",
        "group_rms_log",
        "nonreference_rms_log",
        "residual_wavelength_pearson",
        "led_mean_wavelength_pearson",
        *[f"led_{led}_mean_residual_log" for led in range(1, 7)],
        *[f"led_{led}_rms_log" for led in range(1, 7)],
    ]
    with Path(output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for detector in DETECTORS:
            for variant, _ in VARIANT_NAMES:
                values = fits[detector][variant]
                nonreference = values["repeat"] != 0
                residual = values["residual"]
                led = values["led"]
                nonreference_residual = residual[nonreference]
                wavelengths = effective_wavelengths[led[nonreference] - 1]
                led_means = np.asarray(
                    [
                        np.mean(nonreference_residual[led[nonreference] == value])
                        for value in range(1, 7)
                    ]
                )
                row = {
                    "detector": detector.upper(),
                    "variant": variant,
                    "group_rms_log": float(np.sqrt(np.mean(residual**2))),
                    "nonreference_rms_log": float(
                        np.sqrt(np.mean(nonreference_residual**2))
                    ),
                    "residual_wavelength_pearson": float(
                        np.corrcoef(nonreference_residual, wavelengths)[0, 1]
                    ),
                    "led_mean_wavelength_pearson": float(
                        np.corrcoef(led_means, effective_wavelengths)[0, 1]
                    ),
                }
                row.update(
                    {
                        f"led_{led}_mean_residual_log": float(led_means[led - 1])
                        for led in range(1, 7)
                    }
                )
                row.update(
                    {
                        f"led_{led_number}_rms_log": float(
                            np.sqrt(
                                np.mean(
                                    nonreference_residual[
                                        led[nonreference] == led_value
                                    ]
                                    ** 2
                                )
                            )
                        )
                        for led_number, led_value in enumerate(range(1, 7), start=1)
                    }
                )
                writer.writerow(row)


def plot_led_summary(fits, effective_wavelengths, output):
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 8), constrained_layout=True
    )
    for detector_index, detector in enumerate(DETECTORS):
        mean_axis = axes[detector_index, 0]
        rms_axis = axes[detector_index, 1]
        positions = np.arange(1, 7)
        width = 0.25
        for variant_index, (variant, title) in enumerate(VARIANT_NAMES):
            values = fits[detector][variant]
            nonreference = values["repeat"] != 0
            led = values["led"][nonreference]
            residual = values["residual"][nonreference]
            means = np.asarray([np.mean(residual[led == value]) for value in positions])
            rms = np.asarray([np.sqrt(np.mean(residual[led == value] ** 2)) for value in positions])
            mean_axis.plot(
                effective_wavelengths,
                1000.0 * means,
                marker="o",
                linewidth=1.2,
                label=title,
            )
            rms_axis.bar(
                positions + (variant_index - 1) * width,
                1000.0 * rms,
                width,
                label=title,
            )
        mean_axis.axhline(0.0, color="black", linewidth=0.7)
        mean_axis.set_title(f"{detector.upper()} LED mean residual")
        mean_axis.set_ylabel("observed - model [10^-3]")
        mean_axis.set_xlabel("Effective LED wavelength [nm]")
        mean_axis.grid(alpha=0.2)
        rms_axis.set_title(f"{detector.upper()} residual RMS by LED")
        rms_axis.set_ylabel("RMS [10^-3]")
        rms_axis.set_xlabel("LED number")
        rms_axis.set_xticks(positions)
        rms_axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("WFI tile 136: LED residual structure", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits = load_all(args.root)
    effective_wavelengths = effective_led_wavelengths()
    plot_data_model(fits, args.output_dir / "tile136_data_vs_model.png")
    plot_residuals(fits, args.output_dir / "tile136_residuals_vs_time.png")
    plot_led_summary(
        fits,
        effective_wavelengths,
        args.output_dir / "tile136_residual_led_summary.png",
    )
    summarize(
        fits,
        effective_wavelengths,
        args.output_dir / "tile136_model_residual_summary.csv",
    )
    for path in sorted(args.output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
