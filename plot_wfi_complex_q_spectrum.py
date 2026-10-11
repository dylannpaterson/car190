"""Plot the fitted effective complex-q spectrum from the current-data test."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SPECTRUM = (
    ROOT / "wfi01_tile0_complex_q_production_fit" / "complex_q_spectrum.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "wfi01_tile0_complex_q_production_fit" / "complex_q_spectrum.png"
)


def load_spectrum(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Complex-q spectrum is empty: {path}")
    wavelengths = np.asarray([float(row["wavelength_nm"]) for row in rows])
    q_real = np.asarray([float(row["q_real"]) for row in rows])
    q_imag = np.asarray([float(row["q_imag"]) for row in rows])
    q_magnitude = np.asarray([float(row["q_magnitude"]) for row in rows])
    q_phase = np.asarray([float(row["q_phase_rad"]) for row in rows])
    absorption = np.asarray(
        [float(row["q_absorption_per_nm"]) for row in rows]
    )
    order = np.argsort(wavelengths, kind="stable")
    values = tuple(
        value[order]
        for value in (wavelengths, q_real, q_imag, q_magnitude, q_phase, absorption)
    )
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError(f"Complex-q spectrum contains non-finite values: {path}")
    return values


def plot_spectrum(path, values):
    wavelengths, q_real, q_imag, q_magnitude, q_phase, absorption = values
    unwrapped_phase = np.unwrap(q_phase)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    complex_axis = axes[0, 0]
    scatter = complex_axis.scatter(
        q_real,
        q_imag,
        c=wavelengths,
        cmap="viridis",
        s=8,
        linewidths=0,
        vmin=wavelengths.min(),
        vmax=wavelengths.max(),
    )
    complex_axis.plot(q_real, q_imag, color="0.45", linewidth=0.7, alpha=0.7)
    complex_axis.scatter(
        [q_real[0], q_real[-1]],
        [q_imag[0], q_imag[-1]],
        color=("tab:blue", "tab:red"),
        s=30,
        zorder=3,
    )
    complex_axis.axhline(0.0, color="0.5", linewidth=0.7)
    complex_axis.axvline(0.0, color="0.5", linewidth=0.7)
    complex_axis.set_xlabel("Re(q)")
    complex_axis.set_ylabel("Im(q)")
    complex_axis.set_title("Complex-plane path")
    complex_axis.ticklabel_format(
        style="sci", axis="both", scilimits=(-3, -3), useMathText=True
    )
    complex_axis.grid(alpha=0.2)
    colorbar = figure.colorbar(scatter, ax=complex_axis)
    colorbar.set_label("Wavelength (nm)")

    components_axis = axes[0, 1]
    components_axis.plot(wavelengths, q_real, label="Re(q)")
    components_axis.plot(wavelengths, q_imag, label="Im(q)")
    components_axis.axhline(0.0, color="0.5", linewidth=0.7)
    components_axis.set_xlabel("Wavelength (nm)")
    components_axis.set_ylabel("Effective q")
    components_axis.set_title("Real and imaginary components")
    components_axis.legend(frameon=False)
    components_axis.grid(alpha=0.2)

    magnitude_axis = axes[1, 0]
    magnitude_axis.plot(wavelengths, q_magnitude, color="tab:purple")
    magnitude_axis.set_xlabel("Wavelength (nm)")
    magnitude_axis.set_ylabel("|q|")
    magnitude_axis.set_title("Effective amplitude")
    magnitude_axis.grid(alpha=0.2)

    phase_axis = axes[1, 1]
    phase_axis.plot(wavelengths, unwrapped_phase, color="tab:orange")
    phase_axis.set_xlabel("Wavelength (nm)")
    phase_axis.set_ylabel("Unwrapped phase (rad)")
    phase_axis.set_title("Effective phase")
    phase_axis.grid(alpha=0.2)

    absorption_value = float(np.median(absorption))
    figure.suptitle(
        "WFI01 tile 0 fitted effective complex-q spectrum\n"
        f"attenuation = {absorption_value:.5g} per nm; q is an effective boundary correction",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = load_spectrum(args.spectrum)
    plot_spectrum(args.output, values)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
