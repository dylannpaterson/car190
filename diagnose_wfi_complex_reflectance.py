"""Diagnose complex reflection for one archived WFI MAP tile."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fit_wfi_tmm_production import load_response_context


ROOT = Path(__file__).resolve().parent
DEFAULT_FIT_DIR = (
    ROOT
    / "wfi_production_20261005_shared_hgcdte_prior10_smooth001_WFI01_map10000_emcee1000"
)
DEFAULT_Q_TABLE = (
    ROOT / "effective_sca1_reflectance_physical" / "effective_sca1_reflectance_table.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "wfi01_tile0_complex_reflectance_diagnostic"
ICE_INDEX = 1.25 + 0.0j


def scalar_from_samples(samples, key, default=None):
    if key not in samples:
        return default
    value = np.asarray(samples[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar metadata for {key}")
    return value.reshape(-1)[0].item()


def load_q_table(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No effective reflectance rows found in {path}")
    wavelengths = np.asarray([float(row["wavelength_nm"]) for row in rows])
    q = np.asarray(
        [
            float(row["rho"]) * np.exp(1j * float(row["phase_offset_rad"]))
            for row in rows
        ]
    )
    alpha = np.asarray([float(row["alpha_per_nm"]) for row in rows])
    return wavelengths, q, alpha


def load_fit_state(fit_dir, tile_id, q_table_path):
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        detector_names = np.asarray(samples["sca_names"]).astype(str)
        if detector_names.size != 1:
            raise ValueError("The diagnostic expects one detector output directory")
        detector = detector_names[0]
        posterior_tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
        tile_indices = np.flatnonzero(posterior_tile_ids == tile_id)
        if tile_indices.size != 1:
            raise ValueError(f"Tile {tile_id} is not uniquely present in the posterior")
        tile_index = int(tile_indices[0])

        ar_parameters = np.asarray(samples["ar_parameters"], dtype=float)
        if ar_parameters.ndim == 4:
            ar_parameters = ar_parameters.reshape(
                ar_parameters.shape[0], -1, ar_parameters.shape[-1]
            )
        if ar_parameters.ndim != 3:
            raise ValueError("Unexpected AR posterior shape")
        ar_tile = np.median(ar_parameters[:, tile_index, :], axis=0)

        wavelengths = np.asarray(samples["response_wavelengths_nm"], dtype=float)
        sed_mode = str(scalar_from_samples(samples, "sed_mode", "signed"))
        sed_weighting = str(scalar_from_samples(samples, "sed_weighting", "photon"))
        wavelength_stride = int(scalar_from_samples(samples, "wavelength_stride", 1))
        archived_step = scalar_from_samples(samples, "wavelength_step_nm", np.nan)
        wavelength_step_nm = None if np.isnan(archived_step) else float(archived_step)
        absolute_substrate_power = bool(
            scalar_from_samples(samples, "absolute_substrate_power", 0)
        )
        use_hgcdte_optical_constants = bool(
            scalar_from_samples(samples, "use_hgcdte_optical_constants", 0)
        )
        if not absolute_substrate_power or not use_hgcdte_optical_constants:
            raise ValueError("This diagnostic currently expects production HgCdTe power mode")

        n_log_scale = np.asarray(
            samples["absolute_hgcdte_n_log_scale"], dtype=float
        )
        k_log_scale = np.asarray(
            samples["absolute_hgcdte_k_log_scale"], dtype=float
        )
        n_log_scale = float(np.median(n_log_scale))
        k_log_scale = float(np.median(k_log_scale))
        incidence_values = (
            np.asarray(samples["incidence_angle_deg"], dtype=float)
            if "incidence_angle_deg" in samples
            else np.asarray(0.0)
        )
        incidence_angle = float(np.median(incidence_values))

    context = load_response_context(
        "/dev/null",
        sed_mode,
        wavelength_stride,
        np.asarray([detector]),
        wavelength_step_nm,
        use_hgcdte_optical_constants=True,
        sed_weighting=sed_weighting,
        absolute_substrate_power=True,
    )
    context_wavelengths = np.asarray(context["wavelengths"], dtype=float)
    if context_wavelengths.shape != wavelengths.shape or not np.allclose(
        context_wavelengths, wavelengths, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("Reconstructed wavelength grid differs from the archived MAP grid")
    substrate_index = (
        np.asarray(context["hgcdte_n"], dtype=float) * np.exp(n_log_scale)
        + 1j
        * np.asarray(context["hgcdte_k"], dtype=float)
        * np.exp(k_log_scale)
    )
    q_wavelengths, q_values, q_alpha = load_q_table(q_table_path)
    q_real = np.interp(wavelengths, q_wavelengths, q_values.real, left=np.nan, right=np.nan)
    q_imag = np.interp(wavelengths, q_wavelengths, q_values.imag, left=np.nan, right=np.nan)
    alpha = np.interp(wavelengths, q_wavelengths, q_alpha, left=np.nan, right=np.nan)
    q_values = q_real + 1j * q_imag
    q_values[~np.isfinite(q_real) | ~np.isfinite(q_imag)] = np.nan + 1j * np.nan
    alpha[~np.isfinite(alpha)] = np.nan
    return {
        "detector": detector,
        "tile_id": tile_id,
        "wavelengths": wavelengths,
        "ar_parameters": ar_tile,
        "substrate_index": substrate_index,
        "incidence_angle_deg": incidence_angle,
        "q_values": q_values,
        "q_alpha": alpha,
    }


def layer_matrix(index, thickness_nm, wavelengths, sine_incidence, polarization):
    cosine = np.sqrt(1.0 - (sine_incidence / index) ** 2)
    phase = 2.0 * np.pi * index * thickness_nm * cosine / wavelengths
    cosine_phase = np.cos(phase)
    sine_phase = -1j * np.sin(phase)
    admittance = (
        index * cosine if polarization == "s" else index / cosine
    )
    return (
        cosine_phase,
        sine_phase / admittance,
        sine_phase * admittance,
        cosine_phase,
    )


def apply_layer(boundary, elements):
    first, second = boundary
    m00, m01, m10, m11 = elements
    return (
        m00 * first + m01 * second,
        m10 * first + m11 * second,
    )


def reflection_and_power(state, thicknesses_nm):
    wavelengths = state["wavelengths"]
    substrate_index = state["substrate_index"]
    ar_parameters = state["ar_parameters"].reshape(-1, 3)
    sine_incidence = np.sin(np.deg2rad(state["incidence_angle_deg"]))
    thicknesses_nm = np.asarray(thicknesses_nm, dtype=float)
    results = {}
    for polarization in ("s", "p"):
        cosine_incidence = np.sqrt(1.0 - sine_incidence**2)
        incident_admittance = (
            cosine_incidence if polarization == "s" else 1.0 / cosine_incidence
        )
        substrate_cosine = np.sqrt(
            1.0 - (sine_incidence / substrate_index) ** 2
        )
        substrate_admittance = (
            substrate_index * substrate_cosine
            if polarization == "s"
            else substrate_index / substrate_cosine
        )
        clean_boundary = (np.ones_like(wavelengths, dtype=complex), substrate_admittance)
        for scale, index_n, index_k in ar_parameters[::-1]:
            clean_boundary = apply_layer(
                clean_boundary,
                layer_matrix(
                    index_n + 1j * index_k,
                    100.0 * scale,
                    wavelengths,
                    sine_incidence,
                    polarization,
                ),
            )
        clean_denominator = (
            incident_admittance * clean_boundary[0] + clean_boundary[1]
        )
        clean_r = (
            incident_admittance * clean_boundary[0] - clean_boundary[1]
        ) / clean_denominator
        clean_power = (
            np.real(substrate_admittance)
            / np.real(incident_admittance)
            * np.abs(2.0 * incident_admittance / clean_denominator) ** 2
        )

        coated_r = np.empty((len(thicknesses_nm), len(wavelengths)), dtype=complex)
        coated_power = np.empty_like(coated_r.real)
        for thickness_index, thickness_nm in enumerate(thicknesses_nm):
            coated_boundary = apply_layer(
                clean_boundary,
                layer_matrix(
                    ICE_INDEX,
                    thickness_nm,
                    wavelengths,
                    sine_incidence,
                    polarization,
                ),
            )
            coated_denominator = (
                incident_admittance * coated_boundary[0] + coated_boundary[1]
            )
            coated_r[thickness_index] = (
                incident_admittance * coated_boundary[0] - coated_boundary[1]
            ) / coated_denominator
            coated_power[thickness_index] = (
                np.real(substrate_admittance)
                / np.real(incident_admittance)
                * np.abs(2.0 * incident_admittance / coated_denominator) ** 2
            )
        results[polarization] = {
            "clean_r": clean_r,
            "clean_power": clean_power,
            "coated_r": coated_r,
            "coated_power": coated_power,
        }
    results["mean_power_ratio"] = (
        0.5 * (results["s"]["coated_power"] + results["p"]["coated_power"])
    ) / (
        0.5 * (results["s"]["clean_power"] + results["p"]["clean_power"])
    )[None, :]
    return results


def effective_response(q_values, alpha, wavelengths, thicknesses_nm):
    phase = 4.0 * np.pi * ICE_INDEX.real * thicknesses_nm[:, None] / wavelengths[None, :]
    numerator = np.abs(1.0 - q_values[None, :]) ** 2
    denominator = np.abs(1.0 - q_values[None, :] * np.exp(1j * phase)) ** 2
    return np.exp(-alpha[None, :] * thicknesses_nm[:, None]) * numerator / denominator


def write_spectral_csv(path, state, results, diagnostic_thicknesses):
    wavelengths = state["wavelengths"]
    fields = [
        "wavelength_nm",
        "q_effective_real",
        "q_effective_imag",
        "q_effective_magnitude",
        "q_effective_phase_rad",
        "q_effective_alpha_per_nm",
        "clean_r_s_real",
        "clean_r_s_imag",
        "clean_reflectance_s",
        "clean_phase_s_rad",
        "clean_r_p_real",
        "clean_r_p_imag",
        "clean_reflectance_p",
        "clean_phase_p_rad",
    ]
    for thickness_nm in diagnostic_thicknesses:
        label = f"{thickness_nm:g}nm"
        fields.extend(
            (
                f"r_s_{label}_real",
                f"r_s_{label}_imag",
                f"reflectance_s_{label}",
                f"phase_s_{label}_rad",
                f"r_p_{label}_real",
                f"r_p_{label}_imag",
                f"reflectance_p_{label}",
                f"phase_p_{label}_rad",
            )
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wavelength_index, wavelength in enumerate(wavelengths):
            row = {
                "wavelength_nm": wavelength,
                "q_effective_real": state["q_values"].real[wavelength_index],
                "q_effective_imag": state["q_values"].imag[wavelength_index],
                "q_effective_magnitude": abs(state["q_values"][wavelength_index]),
                "q_effective_phase_rad": np.angle(state["q_values"][wavelength_index]),
                "q_effective_alpha_per_nm": state["q_alpha"][wavelength_index],
                "clean_r_s_real": results["s"]["clean_r"][wavelength_index].real,
                "clean_r_s_imag": results["s"]["clean_r"][wavelength_index].imag,
                "clean_reflectance_s": abs(results["s"]["clean_r"][wavelength_index]) ** 2,
                "clean_phase_s_rad": np.angle(results["s"]["clean_r"][wavelength_index]),
                "clean_r_p_real": results["p"]["clean_r"][wavelength_index].real,
                "clean_r_p_imag": results["p"]["clean_r"][wavelength_index].imag,
                "clean_reflectance_p": abs(results["p"]["clean_r"][wavelength_index]) ** 2,
                "clean_phase_p_rad": np.angle(results["p"]["clean_r"][wavelength_index]),
            }
            for thickness_index, thickness_nm in enumerate(diagnostic_thicknesses):
                label = f"{thickness_nm:g}nm"
                for polarization in ("s", "p"):
                    value = results[polarization]["coated_r"][
                        thickness_index, wavelength_index
                    ]
                    row[f"r_{polarization}_{label}_real"] = value.real
                    row[f"r_{polarization}_{label}_imag"] = value.imag
                    row[f"reflectance_{polarization}_{label}"] = abs(value) ** 2
                    row[f"phase_{polarization}_{label}_rad"] = np.angle(value)
            writer.writerow(row)


def write_response_csv(path, wavelengths, thicknesses, physical, effective, selected):
    fields = ["thickness_nm"]
    for wavelength_index in selected:
        wavelength = wavelengths[wavelength_index]
        fields.extend(
            (
                f"physical_ratio_{wavelength:.3f}nm",
                f"effective_q_ratio_{wavelength:.3f}nm",
                f"difference_{wavelength:.3f}nm",
            )
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for thickness_index, thickness_nm in enumerate(thicknesses):
            row = {"thickness_nm": thickness_nm}
            for wavelength_index in selected:
                wavelength = wavelengths[wavelength_index]
                row[f"physical_ratio_{wavelength:.3f}nm"] = physical[
                    thickness_index, wavelength_index
                ]
                row[f"effective_q_ratio_{wavelength:.3f}nm"] = effective[
                    thickness_index, wavelength_index
                ]
                row[f"difference_{wavelength:.3f}nm"] = (
                    physical[thickness_index, wavelength_index]
                    - effective[thickness_index, wavelength_index]
                )
            writer.writerow(row)


def plot_diagnostic(
    path,
    state,
    results,
    response_results,
    thicknesses,
    response_thicknesses,
    selected,
):
    wavelengths = state["wavelengths"]
    q_supported = np.isfinite(state["q_values"].real)
    mean_reflectance = 0.5 * (
        np.abs(results["s"]["clean_r"]) ** 2
        + np.abs(results["p"]["clean_r"]) ** 2
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for thickness_index, thickness_nm in enumerate(thicknesses):
        reflectance = 0.5 * (
            np.abs(results["s"]["coated_r"][thickness_index]) ** 2
            + np.abs(results["p"]["coated_r"][thickness_index]) ** 2
        )
        axes[0, 0].plot(wavelengths, reflectance, label=f"ice {thickness_nm:g} nm")
    axes[0, 0].plot(wavelengths, mean_reflectance, "k--", label="clean")
    axes[0, 0].set_ylabel("Mean reflectance |r|²")
    axes[0, 0].set_title("Explicit TMM reflectance")
    axes[0, 0].legend(fontsize=8)

    for polarization, color in (("s", "tab:blue"), ("p", "tab:orange")):
        axes[0, 1].plot(
            wavelengths,
            np.unwrap(np.angle(results[polarization]["clean_r"])),
            color=color,
            linestyle="--",
            label=f"clean {polarization}",
        )
        axes[0, 1].plot(
            wavelengths,
            np.unwrap(np.angle(results[polarization]["coated_r"][2])),
            color=color,
            label=f"ice 50 nm {polarization}",
        )
    axes[0, 1].set_ylabel("Unwrapped reflection phase (rad)")
    axes[0, 1].set_title("Complex reflection phase")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        wavelengths[q_supported],
        np.abs(state["q_values"][q_supported]),
        label="|q| effective table",
    )
    axes[1, 0].set_ylabel("Magnitude")
    axes[1, 0].set_title("Effective round-trip coefficient q")
    q_phase_axis = axes[1, 0].twinx()
    q_phase_axis.plot(
        wavelengths[q_supported],
        np.unwrap(np.angle(state["q_values"][q_supported])),
        color="tab:orange",
        label="arg(q)",
    )
    q_phase_axis.set_ylabel("Unwrapped phase (rad)")
    axes[1, 0].set_xlabel("Wavelength (nm)")

    physical = response_results["mean_power_ratio"]
    effective = effective_response(
        state["q_values"],
        state["q_alpha"],
        wavelengths,
        response_thicknesses,
    )
    colors = plt.get_cmap("viridis", len(selected))
    for color_index, wavelength_index in enumerate(selected):
        axes[1, 1].plot(
            response_thicknesses,
            physical[:, wavelength_index],
            color=colors(color_index),
            label=f"TMM {wavelengths[wavelength_index]:.0f} nm",
        )
        axes[1, 1].plot(
            response_thicknesses,
            effective[:, wavelength_index],
            color=colors(color_index),
            linestyle="--",
        )
    axes[1, 1].set_xlabel("Ice thickness (nm)")
    axes[1, 1].set_ylabel("Coated / clean terminal power")
    axes[1, 1].set_title("TMM response vs effective-q response")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{state['detector']} tile {state['tile_id']} complex-reflectance diagnostic; "
        f"incidence angle {state['incidence_angle_deg']:.4f} deg",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--q-table", type=Path, default=DEFAULT_Q_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = load_fit_state(args.fit_dir, args.tile_id, args.q_table)
    diagnostic_thicknesses = np.asarray([0.0, 10.0, 50.0, 100.0])
    response_thicknesses = np.linspace(0.0, 300.0, 301)
    results = reflection_and_power(state, diagnostic_thicknesses)
    response_results = reflection_and_power(state, response_thicknesses)
    q_supported = np.isfinite(state["q_values"].real)
    candidate_wavelengths = np.asarray([600.0, 800.0, 1000.0, 1300.0, 1600.0, 2000.0])
    selected = []
    for wavelength in candidate_wavelengths:
        differences = np.abs(state["wavelengths"] - wavelength)
        differences[~q_supported] = np.inf
        index = int(np.argmin(differences))
        if differences[index] < 5.0:
            selected.append(index)
    selected = np.asarray(selected, dtype=int)
    if selected.size == 0:
        selected = np.flatnonzero(q_supported)[:: max(np.count_nonzero(q_supported) // 6, 1)]
    effective = effective_response(
        state["q_values"],
        state["q_alpha"],
        state["wavelengths"],
        response_thicknesses,
    )
    spectral_path = args.output_dir / "complex_reflectance_spectrum.csv"
    response_path = args.output_dir / "tmm_vs_effective_q_response.csv"
    plot_path = args.output_dir / "complex_reflectance_diagnostic.png"
    write_spectral_csv(spectral_path, state, results, diagnostic_thicknesses)
    write_response_csv(
        response_path,
        state["wavelengths"],
        response_thicknesses,
        response_results["mean_power_ratio"],
        effective,
        selected,
    )
    plot_diagnostic(
        plot_path,
        state,
        results,
        response_results,
        diagnostic_thicknesses,
        response_thicknesses,
        selected,
    )
    supported_response = np.broadcast_to(
        q_supported[None, :], effective.shape
    ) & np.isfinite(effective)
    difference = response_results["mean_power_ratio"] - effective
    print(f"Detector: {state['detector']}; tile: {state['tile_id']}")
    print(f"Incidence angle: {state['incidence_angle_deg']:.9f} deg")
    print(f"Wavelengths: {len(state['wavelengths'])}; q table support: {np.count_nonzero(q_supported)}")
    print(
        "TMM/effective-q response RMS over supported thickness/wavelength grid: "
        f"{np.sqrt(np.mean(difference[supported_response] ** 2)):.8g}"
    )
    print(f"Wrote {spectral_path}")
    print(f"Wrote {response_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
