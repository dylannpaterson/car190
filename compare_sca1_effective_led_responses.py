"""Compare an effective SCA1 reflectance model with the production table model."""

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_rcs_soc_ice_model import (
    DEFAULT_MODEL_PATH,
    MODES,
    fit_periodic_model,
    read_model_table,
    sed_output,
    transmission_factor,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed


ROOT = Path(__file__).resolve().parent
DEFAULT_EFFECTIVE_TABLE = ROOT / "effective_sca1_reflectance_physical" / "effective_sca1_reflectance_table.csv"
DEFAULT_OUTPUT = ROOT / "effective_sca1_led_response_comparison.csv"


def load_effective_table(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No effective-model rows found in {path}")
    wavelengths = np.asarray([float(row["wavelength_nm"]) for row in rows])
    rho = np.asarray([float(row["rho"]) for row in rows])
    phase = np.asarray([float(row["phase_offset_rad"]) for row in rows])
    alpha = np.asarray([float(row["alpha_per_nm"]) for row in rows])
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("Effective-model wavelengths must be strictly increasing")
    return wavelengths, rho * np.exp(1j * phase), alpha


def effective_factor(
    wavelengths,
    thicknesses,
    model_wavelengths,
    q_values,
    alpha_values,
    n_ice,
):
    wavelengths = np.asarray(wavelengths, dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    supported = (wavelengths >= model_wavelengths[0]) & (wavelengths <= model_wavelengths[-1])
    result = np.ones((len(thicknesses), len(wavelengths)), dtype=float)
    if not np.any(supported):
        return result

    q_real = np.interp(wavelengths[supported], model_wavelengths, q_values.real)
    q_imag = np.interp(wavelengths[supported], model_wavelengths, q_values.imag)
    alpha = np.interp(wavelengths[supported], model_wavelengths, alpha_values)
    q = q_real + 1j * q_imag
    phase = 4.0 * np.pi * n_ice * thicknesses[:, None] / wavelengths[supported][None, :]
    denominator = np.abs(1.0 - q[None, :] * np.exp(1j * phase)) ** 2
    numerator = np.abs(1.0 - q[None, :]) ** 2
    result[:, supported] = (
        np.exp(-alpha[None, :] * thicknesses[:, None])
        * numerator
        / denominator
    )
    return result


def response_curves(spectra, qe_wavelengths, qe_values, model_factors, thicknesses):
    responses = {mode: np.zeros((18, 6, len(thicknesses))) for mode in MODES}
    for band in range(1, 7):
        spectrum = spectra[band]
        wavelengths = spectrum["wavelength"]
        factors = model_factors(band, wavelengths, thicknesses)
        photon_weight = wavelengths
        for mode in MODES:
            output = sed_output(spectrum, mode)
            qe = np.asarray(
                [
                    np.interp(
                        wavelengths,
                        qe_wavelengths,
                        qe_values[:, scu],
                        left=0.0,
                        right=0.0,
                    )
                    for scu in range(18)
                ]
            ).T
            weighted_output = output[:, None] * photon_weight[:, None] * qe
            clean = np.asarray(
                [np.trapezoid(weighted_output[:, scu], wavelengths) for scu in range(18)]
            )
            numerator = np.asarray(
                [
                    np.trapezoid(
                        factors * weighted_output[:, scu][None, :],
                        wavelengths,
                        axis=1,
                    )
                    for scu in range(18)
                ]
            )
            responses[mode][:, band - 1] = numerator / clean[:, None]
    return responses


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--effective-table", type=Path, default=DEFAULT_EFFECTIVE_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thickness-steps", type=int, default=301)
    parser.add_argument("--n-ice", type=float, default=1.25)
    return parser.parse_args()


def main():
    args = parse_args()
    model_wavelengths, thicknesses, ratios = read_model_table(args.model_path)
    coefficients, omegas = fit_periodic_model(model_wavelengths, thicknesses, ratios)
    model = (model_wavelengths, thicknesses, coefficients, omegas)
    effective_wavelengths, q_values, alpha_values = load_effective_table(args.effective_table)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    d_grid = np.linspace(0.0, 300.0, args.thickness_steps)

    def production_factors(band, wavelengths, thicknesses):
        supported = (wavelengths >= model_wavelengths[0]) & (wavelengths <= model_wavelengths[-1])
        factors = np.ones((len(thicknesses), len(wavelengths)), dtype=float)
        if np.any(supported):
            factors[:, supported] = transmission_factor(
                model,
                wavelengths[supported],
                thicknesses,
            )
        return factors

    def effective_factors(band, wavelengths, thicknesses):
        return effective_factor(
            wavelengths,
            thicknesses,
            effective_wavelengths,
            q_values,
            alpha_values,
            args.n_ice,
        )

    production = response_curves(spectra, qe_wavelengths, qe_values, production_factors, d_grid)
    effective = response_curves(spectra, qe_wavelengths, qe_values, effective_factors, d_grid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sed_mode",
        "scu",
        "band",
        "max_abs_response_difference",
        "rms_response_difference",
        "response_difference_at_50nm",
        "response_difference_at_100nm",
        "response_difference_at_300nm",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            for scu in range(18):
                for band in range(6):
                    difference = effective[mode][scu, band] - production[mode][scu, band]
                    writer.writerow(
                        {
                            "sed_mode": mode,
                            "scu": scu + 1,
                            "band": band + 1,
                            "max_abs_response_difference": f"{np.max(np.abs(difference)):.12g}",
                            "rms_response_difference": f"{np.sqrt(np.mean(difference**2)):.12g}",
                            "response_difference_at_50nm": f"{difference[50]:.12g}",
                            "response_difference_at_100nm": f"{difference[100]:.12g}",
                            "response_difference_at_300nm": f"{difference[-1]:.12g}",
                        }
                    )

    all_difference = np.concatenate(
        [
            (effective[mode] - production[mode]).ravel()
            for mode in MODES
        ]
    )
    print(f"Response comparison RMS: {np.sqrt(np.mean(all_difference**2)):.8g}")
    print(f"Response comparison max absolute difference: {np.max(np.abs(all_difference)):.8g}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
