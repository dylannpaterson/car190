"""Plot the RCS LED SEDs and proposed WFI02 ice-modulation nodes."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from assess_rcs_spectral_identifiability import (
    BAND_FILTERS,
    QE_URL,
    SED_URL,
    load_qe,
    load_sed,
)
from fit_wfi_tmm_numpyro import load_response_context


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "wfi02_rcs_sed_proposed_node_placement.png"


def wfi02_effective_centers():
    context = load_response_context(
        None,
        "signed",
        1,
        ["WFI02"],
        sed_weighting="energy",
        absolute_substrate_power=True,
    )
    wavelengths = np.asarray(context["wavelengths"], dtype=float)
    weights = np.asarray(context["model_weights"], dtype=float)[0]
    return (weights @ wavelengths) / np.sum(weights, axis=1)


def sed_peak_wavelengths(spectra):
    return np.asarray(
        [
            spectra[band]["wavelength"][np.argmax(spectra[band]["raw_output"])]
            for band in range(1, 7)
        ],
        dtype=float,
    )


def plot_node_placement(output_path, placement):
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    node_wavelengths = (
        sed_peak_wavelengths(spectra)
        if placement == "max"
        else wfi02_effective_centers()
    )
    colors = plt.get_cmap("tab10").colors

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )
    for band in range(1, 7):
        wavelength = np.asarray(spectra[band]["wavelength"], dtype=float)
        raw_output = np.asarray(spectra[band]["raw_output"], dtype=float)
        color = colors[band - 1]
        raw_scale = np.max(np.abs(raw_output))
        axes[0].plot(
            wavelength,
            raw_output / raw_scale,
            color=color,
            linewidth=1.3,
            label=f"LED {band} / {BAND_FILTERS[band]}",
        )

        qe = np.interp(
            wavelength,
            qe_wavelengths,
            qe_values[:, 1],
            left=0.0,
            right=0.0,
        )
        weighted_output = raw_output * qe
        weighted_scale = np.max(np.abs(weighted_output))
        axes[1].plot(
            wavelength,
            weighted_output / weighted_scale,
            color=color,
            linewidth=1.3,
            label=f"LED {band} / {BAND_FILTERS[band]}",
        )

    for band, node_wavelength in enumerate(node_wavelengths, start=1):
        for axis in axes:
            axis.axvline(
                node_wavelength, color="#D55E00", alpha=0.45, linewidth=1.0
            )
        axes[0].text(
            node_wavelength,
            1.04,
            f"{band}: {node_wavelength:.0f}",
            color="#A64000",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            transform=axes[0].get_xaxis_transform(),
        )

    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Relative SED / peak")
    axes[1].set_ylabel("SED x WFI02 QE / peak")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].set_title("RCS LED spectral energy distributions")
    axes[1].set_title(
        (
            "Energy-weighted SED x WFI02 QE shown for context; nodes are SED maxima"
            if placement == "max"
            else "Energy-weighted SED x WFI02 QE shown for context; centers are SED-only"
        )
    )
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[1].legend(frameon=False, ncol=3, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(350.0, 2700.0)
    figure.suptitle(
        (
            "WFI02 proposed ice-modulation nodes: SED maxima\n"
            if placement == "max"
            else "WFI02 proposed ice-modulation nodes: absolute-power SED centers\n"
        )
        + ", ".join(f"{node:.1f} nm" for node in node_wavelengths),
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {output_path}")
    print(
        f"Node wavelengths ({placement}, nm):",
        ", ".join(f"{node:.9f}" for node in node_wavelengths),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--placement",
        choices=("max", "effective"),
        default="max",
        help="Use SED maxima or absolute-power first-moment centers.",
    )
    args = parser.parse_args()
    plot_node_placement(args.output, args.placement)


if __name__ == "__main__":
    main()
