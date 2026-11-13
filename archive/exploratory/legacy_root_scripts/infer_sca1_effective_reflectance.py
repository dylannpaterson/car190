"""Infer an effective complex fringe coefficient from the SCA1 table.

The empirical table contains intensity ratios only. This script therefore
fits a compact effective model rather than claiming a unique AR stack:

    F(d) = exp(-alpha*d) * |1-q|^2 / |1-q*exp(i*theta(d))|^2

where q is a wavelength-dependent complex round-trip reflectance coefficient
and theta(d) = 4*pi*n*d/lambda. The phase index defaults to the value in the
table metadata, n=1.25. Results are diagnostics, not production constants.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_TABLE = Path(
    "/Users/paterson.35/Documents/"
    "rgespit-comissioning-ice-simulations/data/ice_optical_constants/"
    "spectral_response_model_SCA1.ecsv"
)
DEFAULT_OUTPUT = ROOT / "effective_sca1_reflectance"
THICKNESSES_NM = np.arange(0.0, 301.0, 50.0)


def load_empirical_table(path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    values = np.asarray([[float(value) for value in line.split()] for line in lines[1:]])
    return values[:, 0], values[:, 1:]


def harmonic_fit(wavelengths_nm, empirical, n_ice):
    phase = 4.0 * np.pi * n_ice * THICKNESSES_NM[None, :] / wavelengths_nm[:, None]
    design = np.stack(
        [np.ones_like(phase), np.cos(phase), np.sin(phase)],
        axis=-1,
    )
    coefficients = np.empty((len(wavelengths_nm), 3))
    predicted = np.empty_like(empirical)
    for index in range(len(wavelengths_nm)):
        coefficients[index] = np.linalg.lstsq(
            design[index], empirical[index], rcond=None
        )[0]
        predicted[index] = design[index] @ coefficients[index]
    return coefficients, predicted


def harmonic_fit_vectorized(wavelengths_nm, empirical, n_ice, thicknesses_nm):
    phase = 4.0 * np.pi * n_ice * thicknesses_nm[None, :] / wavelengths_nm[:, None]
    design = np.stack(
        [np.ones_like(phase), np.cos(phase), np.sin(phase)],
        axis=-1,
    )
    gram = np.einsum("nki,nkj->nij", design, design)
    rhs = np.einsum("nki,nk->ni", design, empirical)
    coefficients = np.linalg.solve(gram, rhs)
    predicted = np.einsum("nki,ni->nk", design, coefficients)
    return coefficients, predicted


def scan_phase_index(wavelengths_nm, empirical):
    candidates = np.linspace(1.10, 1.40, 301)
    rms_values = []
    for candidate in candidates:
        coefficients, predicted = harmonic_fit_vectorized(
            wavelengths_nm,
            empirical,
            candidate,
            THICKNESSES_NM,
        )
        predicted /= (coefficients[:, 0] + coefficients[:, 1])[:, None]
        rms_values.append(np.sqrt(np.mean((predicted - empirical) ** 2)))
    best_index = int(np.argmin(rms_values))
    return candidates[best_index], float(rms_values[best_index]), candidates, np.asarray(rms_values)


def compare_continuous_models(wavelengths_nm, empirical, harmonic_coefficients, parameters, n_ice):
    thicknesses = np.linspace(0.0, 300.0, 301)
    phase = 4.0 * np.pi * n_ice * thicknesses[None, :] / wavelengths_nm[:, None]
    harmonic = (
        harmonic_coefficients[:, 0, None]
        + harmonic_coefficients[:, 1, None] * np.cos(phase)
        + harmonic_coefficients[:, 2, None] * np.sin(phase)
    )
    harmonic /= (harmonic_coefficients[:, 0] + harmonic_coefficients[:, 1])[:, None]

    q = parameters[:, 0] * np.exp(1j * parameters[:, 1])
    denominator = np.abs(1.0 - q[:, None] * np.exp(1j * phase)) ** 2
    numerator = np.abs(1.0 - q) ** 2
    reflectance = (
        np.exp(-parameters[:, 2, None] * thicknesses[None, :])
        * numerator[:, None]
        / denominator
    )
    difference = reflectance - harmonic
    return thicknesses, harmonic, reflectance, float(np.sqrt(np.mean(difference**2))), float(np.max(np.abs(difference)))


def reflectance_model(parameters, wavelength_nm, n_ice, thicknesses_nm=THICKNESSES_NM):
    rho, phase_offset, alpha = parameters
    q = rho * np.exp(1j * phase_offset)
    phase = 4.0 * np.pi * n_ice * thicknesses_nm / wavelength_nm
    denominator = np.abs(1.0 - q * np.exp(1j * phase)) ** 2
    numerator = np.abs(1.0 - q) ** 2
    return np.exp(-alpha * thicknesses_nm) * numerator / denominator


def fit_one_wavelength(wavelength_nm, observed, n_ice, starts, rng, allow_negative_alpha):
    from scipy.optimize import least_squares

    def objective(parameters):
        return reflectance_model(parameters, wavelength_nm, n_ice) - observed

    alpha_lower = -0.02 if allow_negative_alpha else 0.0
    lower = np.asarray([0.0, -np.pi, alpha_lower])
    upper = np.asarray([0.999, np.pi, 0.02])
    candidates = [np.asarray([0.2, 0.0, 0.0])]
    for _ in range(max(starts - 1, 0)):
        candidates.append(
            np.asarray(
                [
                    rng.uniform(0.01, 0.9),
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(alpha_lower, 0.003),
                ]
            )
        )
    results = [
        least_squares(
            objective,
            candidate,
            bounds=(lower, upper),
            max_nfev=1000,
            x_scale="jac",
        )
        for candidate in candidates
    ]
    return min(results, key=lambda result: result.cost)


def fit_reflectance(wavelengths_nm, empirical, n_ice, starts, allow_negative_alpha):
    rng = np.random.default_rng(20260915)
    parameters = np.empty((len(wavelengths_nm), 3))
    predicted = np.empty_like(empirical)
    for index, (wavelength, observed) in enumerate(zip(wavelengths_nm, empirical)):
        result = fit_one_wavelength(
            wavelength,
            observed,
            n_ice,
            starts,
            rng,
            allow_negative_alpha,
        )
        parameters[index] = result.x
        predicted[index] = reflectance_model(result.x, wavelength, n_ice)
        if (index + 1) % 100 == 0 or index == len(wavelengths_nm) - 1:
            print(f"reflectance fits: {index + 1}/{len(wavelengths_nm)}")
    return parameters, predicted


def write_outputs(
    output_dir,
    wavelengths,
    empirical,
    harmonic,
    normalized_harmonic,
    parameters,
    predicted,
    n_ice,
    best_n,
    best_n_rms,
    continuous_rms,
    continuous_max,
    allow_negative_alpha,
    fit_phase_index,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "effective_sca1_reflectance_table.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm",
                "rho",
                "phase_offset_rad",
                "alpha_per_nm",
                "k_eff",
                "reflectance_rms",
                *[f"empirical_d{int(d)}nm" for d in THICKNESSES_NM],
                *[f"model_d{int(d)}nm" for d in THICKNESSES_NM],
            ]
        )
        for wave, fit, observed, modeled in zip(wavelengths, parameters, empirical, predicted):
            rho, phase_offset, alpha = fit
            k_eff = alpha * wave / (4.0 * np.pi)
            rms = np.sqrt(np.mean((modeled - observed) ** 2))
            writer.writerow(
                [
                    wave,
                    rho,
                    phase_offset,
                    alpha,
                    k_eff,
                    rms,
                    *observed,
                    *modeled,
                ]
            )

    reflectance_rms = float(np.sqrt(np.mean((predicted - empirical) ** 2)))
    harmonic_rms = float(np.sqrt(np.mean((harmonic - empirical) ** 2)))
    normalized_harmonic_rms = float(
        np.sqrt(np.mean((normalized_harmonic - empirical) ** 2))
    )
    k_eff = parameters[:, 2] * wavelengths / (4.0 * np.pi)
    negative_k_fraction = float(np.mean(k_eff < 0.0))
    summary_path = output_dir / "effective_sca1_reflectance.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Effective SCA1 reflectance inference",
                "",
                "The fit uses intensity ratios only. `q` is an effective complex round-trip coefficient, not a uniquely identified coating-layer reflectance.",
                "",
                f"Phase index used: `{n_ice:.12g}`",
                f"Phase index fitted from table: `{fit_phase_index}`",
                f"Negative absorption allowed: `{allow_negative_alpha}`",
                f"Harmonic reconstruction RMS: `{harmonic_rms:.12g}`",
                f"Normalized harmonic reconstruction RMS: `{normalized_harmonic_rms:.12g}`",
                f"Best phase index in [1.10, 1.40]: `{best_n:.12g}`",
                f"Best phase-index harmonic RMS: `{best_n_rms:.12g}`",
                f"Effective reflectance-model RMS: `{reflectance_rms:.12g}`",
                f"Effective reflectance-model maximum absolute residual: `{np.max(np.abs(predicted - empirical)):.12g}`",
                f"Effective-vs-normalized-harmonic RMS over d=0..300 nm: `{continuous_rms:.12g}`",
                f"Effective-vs-normalized-harmonic maximum absolute difference over d=0..300 nm: `{continuous_max:.12g}`",
                f"|q| range: `{np.min(parameters[:, 0]):.12g}` to `{np.max(parameters[:, 0]):.12g}`",
                f"k_eff range: `{np.min(k_eff):.12g}` to `{np.max(k_eff):.12g}`",
                f"Fraction of wavelengths with k_eff < 0: `{negative_k_fraction:.12g}`",
                "",
                "`k_eff = alpha * wavelength / (4*pi)` is the imaginary-index equivalent of the fitted intensity attenuation term.",
                "",
                f"Fit table: `{table_path}`",
            ]
        )
        + "\n"
    )

    rho = parameters[:, 0]
    phase_offset = np.unwrap(parameters[:, 1])
    k_eff = parameters[:, 2] * wavelengths / (4.0 * np.pi)
    per_wave_rms = np.sqrt(np.mean((predicted - empirical) ** 2, axis=1))
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(wavelengths, rho)
    axes[0].set_ylabel("|q|")
    axes[0].set_title("Effective complex round-trip reflectance")
    axes[1].plot(wavelengths, phase_offset)
    axes[1].set_ylabel("Unwrapped arg(q) (rad)")
    axes[2].plot(wavelengths, k_eff, label="k_eff")
    axes[2].plot(wavelengths, per_wave_rms, label="per-wavelength RMS")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].legend(frameon=False)
    plot_path = output_dir / "effective_sca1_reflectance.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    selected = [0, 1, 2, 3, 4, 5, 6]
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for index in selected:
        axes[0].plot(wavelengths, empirical[:, index], label=f"Empirical {int(THICKNESSES_NM[index])} nm")
        axes[0].plot(wavelengths, predicted[:, index], "--", linewidth=1.0)
        axes[1].plot(wavelengths, predicted[:, index] - empirical[:, index], label=f"{int(THICKNESSES_NM[index])} nm")
    axes[0].set_ylabel("T(d) / T(0)")
    axes[0].set_title("Empirical SCA1 table and effective reflectance reconstruction")
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Model - empirical")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].legend(ncol=2, frameon=False, fontsize=8)
    comparison_path = output_dir / "effective_sca1_reflectance_comparison.png"
    figure.savefig(comparison_path, dpi=180)
    plt.close(figure)

    return table_path, summary_path, plot_path, comparison_path, harmonic_rms, reflectance_rms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-ice", type=float, default=1.25)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument(
        "--fit-phase-index",
        action="store_true",
        help="Use the best global phase index from the table profile scan.",
    )
    parser.add_argument(
        "--allow-negative-alpha",
        action="store_true",
        help="Allow effective gain (negative k_eff); diagnostic only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wavelengths, empirical = load_empirical_table(args.table)
    best_n, best_n_rms, _, _ = scan_phase_index(wavelengths, empirical)
    fit_n = best_n if args.fit_phase_index else args.n_ice
    coefficients, harmonic = harmonic_fit(wavelengths, empirical, fit_n)
    normalized_harmonic = harmonic / (coefficients[:, 0] + coefficients[:, 1])[:, None]
    parameters, predicted = fit_reflectance(
        wavelengths,
        empirical,
        fit_n,
        args.starts,
        args.allow_negative_alpha,
    )
    _, _, _, continuous_rms, continuous_max = compare_continuous_models(
        wavelengths,
        empirical,
        coefficients,
        parameters,
        fit_n,
    )
    paths = write_outputs(
        args.output_dir,
        wavelengths,
        empirical,
        harmonic,
        normalized_harmonic,
        parameters,
        predicted,
        fit_n,
        best_n,
        best_n_rms,
        continuous_rms,
        continuous_max,
        args.allow_negative_alpha,
        args.fit_phase_index,
    )
    print(f"Best phase index in [1.10, 1.40]: {best_n:.8g}")
    print(f"Best phase-index harmonic RMS: {best_n_rms:.8g}")
    print(f"Harmonic reconstruction RMS: {paths[-2]:.8g}")
    print(f"Effective reflectance-model RMS: {paths[-1]:.8g}")
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    print(f"Wrote {paths[2]}")
    print(f"Wrote {paths[3]}")


if __name__ == "__main__":
    main()
