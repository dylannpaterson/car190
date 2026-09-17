"""Plot the RCS spectral leverage and all-detector metadata fits."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from assess_rcs_spectral_identifiability import (
    BAND_FILTERS,
    QE_URL,
    SED_URL,
    interaction_residual,
    load_qe,
    load_sed,
    response_features,
)
from analyze_rcs_metadata_spectral_fit import (
    METADATA_PATH,
    fit_features,
    load_metadata,
)


ROOT = Path(__file__).resolve().parent
SED_PLOT_PATH = ROOT / "rcs_sed_response_models.png"
INTERACTION_PLOT_PATH = ROOT / "rcs_spectral_interaction_heatmaps.png"
FIT_PLOT_PATH = ROOT / "rcs_metadata_model_fits.png"
MEASURED_CENTERS = {1: 624, 2: 853, 3: 1039, 4: 1270, 5: 1481, 6: 1670}
MODES = ("signed", "clipped", "core")
MODE_COLORS = {"signed": "#2f5597", "clipped": "#c55a11", "core": "#548235"}
BASIS_COLORS = {"linear slope": "#4472c4", "quadratic curvature": "#ed7d31"}


def sed_output(spectrum, mode, threshold_percent=5.0):
    raw_output = spectrum["raw_output"]
    if mode == "signed":
        return raw_output
    if mode == "clipped":
        return np.maximum(raw_output, 0.0)
    threshold = np.max(raw_output) * threshold_percent / 100.0
    return np.where(raw_output >= threshold, raw_output, 0.0)


def plot_sed_models(spectra, features_by_mode):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.06, right=0.98, wspace=0.25, hspace=0.38)
    for band, axis in enumerate(axes.ravel(), start=1):
        spectrum = spectra[band]
        wavelengths = spectrum["wavelength"]
        raw_output = spectrum["raw_output"]
        axis.plot(
            wavelengths,
            raw_output,
            color="0.45",
            linewidth=0.55,
            alpha=0.8,
            label="published values",
        )
        for mode in ("clipped", "core"):
            axis.plot(
                wavelengths,
                sed_output(spectrum, mode),
                color=MODE_COLORS[mode],
                linewidth=0.9,
                alpha=0.9,
                label=mode,
            )
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.axhline(5.0, color=MODE_COLORS["core"], linestyle=":", linewidth=0.8)
        axis.axvline(
            MEASURED_CENTERS[band],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="published peak",
        )
        for mode in MODES:
            mean_wavelength = np.mean(
                features_by_mode[mode]["effective_wavelength"][:, band - 1]
            )
            axis.axvline(
                mean_wavelength,
                color=MODE_COLORS[mode],
                linestyle="-",
                linewidth=0.8,
                alpha=0.8,
            )
        axis.set_title(f"Band {band} / {BAND_FILTERS[band]}")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Relative output (%)")
        axis.set_ylim(-6, 106)
        axis.grid(color="0.9", linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "RCS LED SEDs and response-weighted means\n"
        "Solid colored vertical lines show signed, clipped, and 5%-core means; "
        "the 5% horizontal line defines the core diagnostic",
        fontsize=14,
        y=0.915,
    )
    fig.text(
        0.5,
        0.035,
        "The long low-level wings are measurement-noise sensitive, especially for the broad-band scans.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(SED_PLOT_PATH, dpi=180)
    plt.close(fig)


def symmetric_norm(values):
    limit = float(np.max(np.abs(values)))
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit if limit else 1.0)


def plot_interaction_heatmaps(features_by_mode):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.94, wspace=0.45, hspace=0.42)
    for column, mode in enumerate(MODES):
        raw = features_by_mode[mode]["slope"]
        residual = interaction_residual(raw)
        for row, (values, title) in enumerate(
            (
                (raw, "raw linear spectral proxy"),
                (residual, "after free SCU + LED offsets"),
            )
        ):
            axis = axes[row, column]
            image = axis.imshow(
                values,
                aspect="auto",
                cmap="coolwarm",
                norm=symmetric_norm(values),
                interpolation="nearest",
            )
            axis.set_xticks(range(6), [f"LED {index}" for index in range(1, 7)])
            axis.set_yticks(range(18), [f"SCU {index}" for index in range(1, 19)])
            axis.tick_params(labelsize=7)
            axis.set_title(f"{mode}: {title}", fontsize=10)
            if row == 1:
                axis.set_xlabel("LED band")
            if column == 0:
                axis.set_ylabel("Detector")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Detector-by-LED spectral interaction\n"
        "The bottom row is the only part available when source and detector offsets are unconstrained",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.035,
        "Values are response-weighted expectations of (wavelength - 1200 nm) / 1000 nm;"
        " the bottom row is much smaller than the raw proxy.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(INTERACTION_PLOT_PATH, dpi=180)
    plt.close(fig)


def plot_fit_results(observations, fits):
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.35), hspace=0.38, wspace=0.32)
    coefficient_axis = fig.add_subplot(grid[0, :2])
    rms_axis = fig.add_subplot(grid[0, 2])

    x = np.arange(len(MODES))
    offsets = (-0.18, 0.18)
    for offset, basis_index in zip(offsets, range(2)):
        values = np.asarray([fits[mode]["coefficients"][basis_index] for mode in MODES])
        errors = np.asarray([fits[mode]["standard_errors"][basis_index] for mode in MODES])
        basis = "linear slope" if basis_index == 0 else "quadratic curvature"
        coefficient_axis.errorbar(
            x + offset,
            values,
            yerr=errors,
            fmt="o",
            capsize=3,
            color=BASIS_COLORS[basis],
            label=basis,
        )
    coefficient_axis.axhline(0.0, color="black", linewidth=0.7)
    coefficient_axis.set_xticks(x, MODES)
    coefficient_axis.set_ylabel("Coefficient per day\n(arbitrary log-response basis)")
    coefficient_axis.set_title("Joint spectral-proxy coefficients")
    coefficient_axis.legend(frameon=False)
    coefficient_axis.grid(axis="y", color="0.9", linewidth=0.5)

    null_rms = [fits[mode]["null_residual_rms"] for mode in MODES]
    fit_rms = [fits[mode]["residual_rms"] for mode in MODES]
    width = 0.34
    rms_axis.bar(x - width / 2, null_rms, width, color="0.7", label="null model")
    rms_axis.bar(x + width / 2, fit_rms, width, color=[MODE_COLORS[mode] for mode in MODES], label="spectral fit")
    for index, mode in enumerate(MODES):
        improvement = 1.0 - fit_rms[index] / null_rms[index]
        rms_axis.text(
            index + width / 2,
            fit_rms[index] + 0.00008,
            f"{improvement:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    rms_axis.set_xticks(x, MODES)
    rms_axis.set_ylabel("Residual RMS in log median")
    rms_axis.set_title("Null versus fitted residual")
    rms_axis.legend(frameon=False, fontsize=9)
    rms_axis.grid(axis="y", color="0.9", linewidth=0.5)

    for column, mode in enumerate(MODES):
        axis = fig.add_subplot(grid[1, column])
        fit = fits[mode]
        observed = fit["observed_residualized"]
        predicted = fit["fitted_residualized"]
        limit = max(float(np.max(np.abs(observed))), float(np.max(np.abs(predicted)))) * 1.05
        axis.scatter(observed, predicted, s=7, alpha=0.18, color=MODE_COLORS[mode], edgecolors="none")
        axis.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=0.8)
        r_squared = 1.0 - np.sum(fit["fit_residualized"] ** 2) / np.sum(observed**2)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{mode}: observed vs fitted\n$R^2$={r_squared:.3f}")
        axis.set_xlabel("Observed residualized log change")
        if column == 0:
            axis.set_ylabel("Fitted residualized log change")
        axis.grid(color="0.9", linewidth=0.5)

    fig.suptitle(
        "All-detector metadata spectral-proxy fits\n"
        "Nominal errors are statistical only; coefficients are not nm/day",
        fontsize=15,
    )
    fig.savefig(FIT_PLOT_PATH, dpi=180)
    plt.close(fig)


def main():
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    features_by_mode = {
        mode: response_features(
            spectra,
            qe_wavelengths,
            qe_values,
            weighting="photon",
            sed_mode=mode,
            core_threshold_percent=5.0,
        )
        for mode in MODES
    }
    observations, _, _ = load_metadata(METADATA_PATH)
    fits = {
        mode: fit_features(
            observations,
            [features["slope"], features["curvature"]],
        )
        for mode, features in features_by_mode.items()
    }
    plot_sed_models(spectra, features_by_mode)
    plot_interaction_heatmaps(features_by_mode)
    plot_fit_results(observations, fits)
    print(f"Wrote {SED_PLOT_PATH}")
    print(f"Wrote {INTERACTION_PLOT_PATH}")
    print(f"Wrote {FIT_PLOT_PATH}")


if __name__ == "__main__":
    main()
