"""Fit an effective thin-film stack to the empirical SCA1 table.

This is a diagnostic inversion only. It does not alter the production ice
model. The fitted stack is air / ASW / AR / absorbing-substrate, evaluated at
normal incidence with a coherent transfer matrix and normalized to its clean
detector transmission.
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
DEFAULT_OUTPUT = ROOT / "effective_sca1_tmm_fit"
THICKNESSES_NM = np.arange(0.0, 301.0, 50.0)


def load_empirical_table(path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    columns = lines[0].split()
    values = np.asarray([[float(value) for value in line.split()] for line in lines[1:]])
    return values[:, 0], values[:, 1:]


def multiply_matrix(left, right):
    result = np.empty_like(left)
    result[..., 0, 0] = left[..., 0, 0] * right[..., 0, 0] + left[..., 0, 1] * right[..., 1, 0]
    result[..., 0, 1] = left[..., 0, 0] * right[..., 0, 1] + left[..., 0, 1] * right[..., 1, 1]
    result[..., 1, 0] = left[..., 1, 0] * right[..., 0, 0] + left[..., 1, 1] * right[..., 1, 0]
    result[..., 1, 1] = left[..., 1, 0] * right[..., 0, 1] + left[..., 1, 1] * right[..., 1, 1]
    return result


def layer_matrix(index, thickness_nm, wavelengths_nm):
    phase = 2.0 * np.pi * index * thickness_nm / wavelengths_nm
    cosine = np.cos(phase)
    sine = 1j * np.sin(phase)
    matrix = np.zeros((len(wavelengths_nm), 2, 2), dtype=complex)
    matrix[:, 0, 0] = cosine
    matrix[:, 1, 1] = cosine
    matrix[:, 0, 1] = sine / index
    matrix[:, 1, 0] = sine * index
    return matrix


def coherent_transmission(wavelengths_nm, layers, substrate_index, incident_index=1.0):
    matrix = np.broadcast_to(
        np.eye(2, dtype=complex), (len(wavelengths_nm), 2, 2)
    ).copy()
    for index, thickness_nm in layers:
        matrix = multiply_matrix(
            matrix,
            layer_matrix(index, thickness_nm, wavelengths_nm),
        )
    denominator = (
        incident_index * matrix[:, 0, 0]
        + incident_index * substrate_index * matrix[:, 0, 1]
        + matrix[:, 1, 0]
        + substrate_index * matrix[:, 1, 1]
    )
    amplitude = 2.0 * incident_index / denominator
    return np.real(substrate_index / incident_index) * np.abs(amplitude) ** 2


def transmission_ratios(wavelengths_nm, thicknesses_nm, parameters):
    n_ice, k_ice, ar_thickness, n_ar, k_ar, n_sub, k_sub = parameters
    ice_index = n_ice + 1j * k_ice
    ar_index = n_ar + 1j * k_ar
    substrate_index = n_sub + 1j * k_sub
    clean = coherent_transmission(
        wavelengths_nm,
        [(ar_index, ar_thickness)],
        substrate_index,
    )
    ratios = []
    for thickness in thicknesses_nm:
        coated = coherent_transmission(
            wavelengths_nm,
            [
                (ice_index, thickness),
                (ar_index, ar_thickness),
            ],
            substrate_index,
        )
        ratios.append(coated / clean)
    return np.asarray(ratios).T


def residuals(parameters, wavelengths_nm, thicknesses_nm, empirical):
    predicted = transmission_ratios(wavelengths_nm, thicknesses_nm, parameters)
    return (predicted[:, 1:] - empirical[:, 1:]).ravel()


def fit_stack(wavelengths_nm, thicknesses_nm, empirical, starts):
    from scipy.optimize import least_squares

    lower = np.asarray([1.05, 0.0, 1.0, 1.0, 0.0, 1.2, 0.0])
    upper = np.asarray([1.5, 0.2, 1000.0, 4.0, 0.2, 6.0, 0.2])
    initial = np.asarray([1.25, 0.0, 200.0, 2.0, 0.0, 3.8, 0.0])
    rng = np.random.default_rng(20260915)
    candidates = [initial]
    for _ in range(max(starts - 1, 0)):
        candidate = initial.copy()
        candidate[0] = rng.uniform(1.15, 1.35)
        candidate[1] = rng.uniform(0.0, 0.03)
        candidate[2] = rng.uniform(50.0, 500.0)
        candidate[3] = rng.uniform(1.4, 3.0)
        candidate[4] = rng.uniform(0.0, 0.03)
        candidate[5] = rng.uniform(2.5, 5.0)
        candidate[6] = rng.uniform(0.0, 0.03)
        candidates.append(candidate)

    results = []
    for index, start in enumerate(candidates):
        result = least_squares(
            residuals,
            start,
            bounds=(lower, upper),
            args=(wavelengths_nm, thicknesses_nm, empirical),
            x_scale="jac",
            max_nfev=5000,
        )
        results.append(result)
        print(
            f"start {index + 1}/{len(candidates)}: cost={result.cost:.8g}; "
            f"optimality={result.optimality:.8g}; status={result.status}"
        )
    return min(results, key=lambda result: result.cost)


def write_outputs(output_dir, wavelengths, empirical, predicted, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_csv = output_dir / "effective_sca1_tmm_table.csv"
    with fit_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm",
                *[f"empirical_d{int(d)}nm" for d in THICKNESSES_NM],
                *[f"model_d{int(d)}nm" for d in THICKNESSES_NM],
            ]
        )
        for wave, empirical_row, predicted_row in zip(wavelengths, empirical, predicted):
            writer.writerow([wave, *empirical_row, *predicted_row])

    parameters = result.x
    summary = output_dir / "effective_sca1_tmm_fit.md"
    rms = float(np.sqrt(np.mean((predicted - empirical) ** 2)))
    max_abs = float(np.max(np.abs(predicted - empirical)))
    summary.write_text(
        "\n".join(
            [
                "# Effective SCA1 transfer-matrix fit",
                "",
                "This is an effective normal-incidence air/ASW/AR/substrate fit to the empirical SCA1 transmission table. It is not a unique reconstruction of the manufacturer coating stack.",
                "",
                "| Parameter | Value |",
                "|---|---:|",
                f"| ASW real index | {parameters[0]:.12g} |",
                f"| ASW imaginary index | {parameters[1]:.12g} |",
                f"| AR thickness (nm) | {parameters[2]:.12g} |",
                f"| AR real index | {parameters[3]:.12g} |",
                f"| AR imaginary index | {parameters[4]:.12g} |",
                f"| Substrate real index | {parameters[5]:.12g} |",
                f"| Substrate imaginary index | {parameters[6]:.12g} |",
                f"| Table RMS residual | {rms:.12g} |",
                f"| Table maximum absolute residual | {max_abs:.12g} |",
                "",
                f"Optimizer status: `{result.status}`; `{result.message}`",
                "",
                f"Empirical table: `{fit_csv}`",
            ]
        )
        + "\n"
    )

    selected = [0, 1, 2, 3, 4, 5, 6]
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for index in selected:
        axes[0].plot(wavelengths, empirical[:, index], label=f"Empirical {int(THICKNESSES_NM[index])} nm")
        axes[0].plot(wavelengths, predicted[:, index], "--", linewidth=1.0)
        axes[1].plot(wavelengths, predicted[:, index] - empirical[:, index], label=f"{int(THICKNESSES_NM[index])} nm")
    axes[0].set_ylabel("T(d) / T(0)")
    axes[0].set_title("Empirical SCA1 table and effective TMM reconstruction")
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Model - empirical")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].legend(ncol=2, frameon=False, fontsize=8)
    plot_path = output_dir / "effective_sca1_tmm_fit.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return fit_csv, summary, plot_path, rms, max_abs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--starts", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    wavelengths, empirical = load_empirical_table(args.table)
    result = fit_stack(wavelengths, THICKNESSES_NM, empirical, args.starts)
    predicted = transmission_ratios(wavelengths, THICKNESSES_NM, result.x)
    paths = write_outputs(args.output_dir, wavelengths, empirical, predicted, result)
    print("Best parameters:", result.x)
    print(f"Table RMS residual: {paths[-2]:.8g}")
    print(f"Table maximum absolute residual: {paths[-1]:.8g}")
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    print(f"Wrote {paths[2]}")


if __name__ == "__main__":
    main()
