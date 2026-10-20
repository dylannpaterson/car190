"""Compare production TMM complex reflection for selected tile-136 MAP states."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


AR_LAYERS = production.AR_LAYERS
DEFAULT_THICKNESSES_NM = (0.0, 10.0, 50.0, 100.0)


def _scalar(archive, name, default):
    if name not in archive:
        return float(default)
    values = np.asarray(archive[name], dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one value for {name}")
    return float(values[0])


def load_state(path, wavelengths, base_n, base_k, spectral_basis):
    with np.load(path, allow_pickle=False) as archive:
        ar = np.asarray(archive["ar_parameters"], dtype=float).reshape(-1, AR_LAYERS, 3)[0]
        ar_modulation = np.asarray(
            archive["ar_n_log_modulation"], dtype=float
        ).reshape(AR_LAYERS, -1)
        hgcdte_n_modulation = np.asarray(
            archive["hgcdte_n_log_modulation"], dtype=float
        ).reshape(-1)
        hgcdte_k_modulation = np.asarray(
            archive["hgcdte_k_log_modulation"], dtype=float
        ).reshape(-1)
        ice_log = _scalar(
            archive,
            "ice_n_log_constant",
            np.log(_scalar(archive, "ice_n_constant", production.ICE_INDEX_DEFAULT)
                   / production.ICE_INDEX_DEFAULT),
        )
        n_log_scale = _scalar(archive, "absolute_hgcdte_n_log_scale", 0.0)
        k_log_scale = _scalar(archive, "absolute_hgcdte_k_log_scale", 0.0)
        incidence_angle_deg = _scalar(archive, "incidence_angle_deg", 0.0)

    ar_index = ar[:, 1, None] * np.exp(ar_modulation @ spectral_basis.T)
    ar_index = ar_index + 1j * ar[:, 2, None]
    hgcdte_n = base_n * np.exp(
        n_log_scale + hgcdte_n_modulation @ spectral_basis.T
    )
    hgcdte_k = base_k * np.exp(
        k_log_scale + hgcdte_k_modulation @ spectral_basis.T
    )
    return {
        "ar_parameters": ar,
        "ar_index": ar_index,
        "hgcdte_n": hgcdte_n,
        "hgcdte_k": hgcdte_k,
        "ice_index": production.ICE_INDEX_DEFAULT * np.exp(ice_log),
        "incidence_angle_deg": incidence_angle_deg,
    }


def _layer_matrix(index, thickness_nm, wavelengths, sine_incidence, polarization):
    cosine = np.sqrt(1.0 - (sine_incidence / index) ** 2)
    admittance = index * cosine if polarization == "s" else index / cosine
    phase = 2.0 * np.pi * index * thickness_nm * cosine / wavelengths
    cosine_phase = np.cos(phase)
    sine_phase = -1j * np.sin(phase)
    return (
        cosine_phase,
        sine_phase / admittance,
        sine_phase * admittance,
        cosine_phase,
    )


def _apply_layer(boundary, elements):
    first, second = boundary
    m00, m01, m10, m11 = elements
    return (
        m00 * first + m01 * second,
        m10 * first + m11 * second,
    )


def reflection(state, wavelengths, thickness_nm, polarization):
    sine_incidence = np.sin(np.deg2rad(state["incidence_angle_deg"]))
    incident_cosine = np.sqrt(1.0 - sine_incidence**2)
    incident_admittance = (
        incident_cosine
        if polarization == "s"
        else 1.0 / incident_cosine
    )
    substrate = state["hgcdte_n"] + 1j * state["hgcdte_k"]
    substrate_cosine = np.sqrt(1.0 - (sine_incidence / substrate) ** 2)
    substrate_admittance = (
        substrate * substrate_cosine
        if polarization == "s"
        else substrate / substrate_cosine
    )
    boundary = (np.ones_like(wavelengths, dtype=complex), substrate_admittance)
    for layer_index in range(AR_LAYERS - 1, -1, -1):
        boundary = _apply_layer(
            boundary,
            _layer_matrix(
                state["ar_index"][layer_index],
                100.0 * state["ar_parameters"][layer_index, 0],
                wavelengths,
                sine_incidence,
                polarization,
            ),
        )
    clean_denominator = incident_admittance * boundary[0] + boundary[1]
    clean_reflection = (
        incident_admittance * boundary[0] - boundary[1]
    ) / clean_denominator
    coated_boundary = _apply_layer(
        boundary,
        _layer_matrix(
            state["ice_index"],
            thickness_nm,
            wavelengths,
            sine_incidence,
            polarization,
        ),
    )
    coated_denominator = (
        incident_admittance * coated_boundary[0] + coated_boundary[1]
    )
    coated_reflection = (
        incident_admittance * coated_boundary[0] - coated_boundary[1]
    ) / coated_denominator
    return clean_reflection, coated_reflection


def load_wavelengths(reference_archive):
    with np.load(reference_archive, allow_pickle=False) as archive:
        return np.asarray(archive["response_wavelengths_nm"], dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--state", action="append", nargs=2, metavar=("NAME", "ARCHIVE"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thickness-nm",
        type=float,
        nargs="*",
        default=list(DEFAULT_THICKNESSES_NM),
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = load_wavelengths(args.reference_archive)
    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths)
    spectral_basis = production.spectral_modulation_basis(wavelengths)
    states = {
        name: load_state(Path(path), wavelengths, base_n, base_k, spectral_basis)
        for name, path in args.state
    }
    rows = []
    spectra = {}
    for name, state in states.items():
        spectra[name] = {}
        for thickness_nm in args.thickness_nm:
            spectra[name][thickness_nm] = {}
            for polarization in ("s", "p"):
                clean, coated = reflection(state, wavelengths, thickness_nm, polarization)
                spectra[name][thickness_nm][polarization] = (clean, coated)
                for wavelength_index, wavelength_nm in enumerate(wavelengths):
                    value = coated[wavelength_index]
                    rows.append(
                        {
                            "state": name,
                            "thickness_nm": thickness_nm,
                            "polarization": polarization,
                            "wavelength_nm": wavelength_nm,
                            "reflection_real": value.real,
                            "reflection_imag": value.imag,
                            "reflection_magnitude": abs(value),
                            "reflectance": abs(value) ** 2,
                            "phase_rad": np.angle(value),
                        }
                    )

    csv_path = args.output_dir / "complex_reflectance_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reference_name = next(iter(states))
    summary = {"reference_state": reference_name, "thicknesses_nm": args.thickness_nm, "states": {}}
    for name in states:
        state_summary = {}
        for thickness_nm in args.thickness_nm:
            state_summary[str(thickness_nm)] = {}
            for polarization in ("s", "p"):
                value = spectra[name][thickness_nm][polarization][1]
                state_summary[str(thickness_nm)][polarization] = {
                    "mean_reflectance": float(np.mean(abs(value) ** 2)),
                    "mean_magnitude": float(np.mean(abs(value))),
                }
                if name != reference_name:
                    reference = spectra[reference_name][thickness_nm][polarization][1]
                    delta = value - reference
                    state_summary[str(thickness_nm)][polarization].update(
                        {
                            "mean_complex_delta": float(np.mean(abs(delta))),
                            "max_complex_delta": float(np.max(abs(delta))),
                            "mean_reflectance_delta": float(
                                np.mean(abs(value) ** 2 - abs(reference) ** 2)
                            ),
                        }
                    )
        summary["states"][name] = state_summary
    json_path = args.output_dir / "complex_reflectance_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
