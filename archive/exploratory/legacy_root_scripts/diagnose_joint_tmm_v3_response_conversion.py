"""Validate joint-TMM to Eric-v3 response conversion on selected WFI tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import compare_eric_v3_amber_car201 as comparison
import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production
import mock_fit_wfi_star_reflectance_modulation as mock
import wfi_relative_response as tmm_response
import wfi_wavelength_response_bundle as bundle_api


ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = (
    ROOT / "20261111/corrected_statistics_4h_thinned/production/wavelength_response_bundle"
)
MODE_PATH = BUNDLE_DIR / "tmm_joint_modulated_reflectance_modes_v1.h5"
RESPONSE_BUNDLE = BUNDLE_DIR / "wfi_all_2nm_rank16_modulated.npz"
V3_MODEL = BUNDLE_DIR / "ice_response_model_tmm_joint_iceside_20261111_prototype.h5"


def error_summary(actual, expected):
    error = np.abs(np.asarray(actual) - np.asarray(expected))
    return {
        "rms": float(np.sqrt(np.mean(error**2))),
        "p95_abs": float(np.quantile(np.abs(error), 0.95)),
        "max_abs": float(np.max(np.abs(error))),
    }


def check_tile(detector, tile_id, model, core, bundle, mode_file, thicknesses):
    det_number = int(detector[3:])
    superpixels = mode_file["superpixels"]
    selected = np.flatnonzero(
        (np.asarray(superpixels["sca"], dtype=int) == det_number)
        & (np.asarray(superpixels["tile_id"], dtype=int) == tile_id)
    )
    if selected.size != 1:
        raise ValueError(f"{detector}/tile {tile_id}: expected one TMM mode row")
    mode_row = int(selected[0])
    source_fit_path = Path(superpixels["source_fit_path"].asstr()[mode_row])
    sca_row = int(np.flatnonzero(
        np.asarray(mode_file["joint_modulation/detectors"]).astype(str) == detector
    )[0])
    mode_wavelengths = np.asarray(mode_file["joint_modulation/wavelength_nm"], dtype=float)
    ratio = np.asarray(mode_file["joint_modulation/ratio"][sca_row], dtype=complex)

    sca = det_number
    tile_row, tile_col = divmod(tile_id, 16)
    x = tile_col * 256 + 128
    y = tile_row * 256 + 128
    v3_wavelengths = np.asarray(model.wavelength_nm, dtype=float)
    mod_ratio = (
        np.interp(v3_wavelengths, mode_wavelengths, ratio.real)
        + 1j * np.interp(v3_wavelengths, mode_wavelengths, ratio.imag)
    )

    base_n, base_k = production.hgcdte_reference_optical_constants(v3_wavelengths)
    state = complex_model.load_state(
        source_fit_path,
        v3_wavelengths,
        base_n,
        base_k,
        production.spectral_modulation_basis(v3_wavelengths),
    )
    first, second, incident_y, substrate_y, incidence_sine = mock.clean_stack_boundary(
        state, v3_wavelengths
    )
    if abs(incidence_sine) > 1e-12:
        raise ValueError(f"{detector}/tile {tile_id}: source TMM state is not normal incidence")
    source_air_reflection = (incident_y * first - second) / (incident_y * first + second)
    source_modulated_air_reflection = source_air_reflection * mod_ratio
    h5_reflection = model.reflectance(sca, [x], [y], v3_wavelengths)[0]

    cache_path = (
        Path(bundle["metadata"]["tile_optical_cache_roots"][detector])
        / f"tile_{tile_id:03d}/relative_response.npz"
    )
    with np.load(cache_path, allow_pickle=False) as cache:
        optical_values = {
            name: np.asarray(cache[name]).copy()
            for name in (
                "ar_tile_raw",
                "absolute_hgcdte_n_log_scale",
                "absolute_hgcdte_k_log_scale",
                "incidence_angle_deg",
            )
        }
        ice_n_log_constant = float(
            np.asarray(cache["ice_n_log_constant"], dtype=float).reshape(-1)[0]
        )
    ice_index = production.ICE_INDEX_DEFAULT * np.exp(ice_n_log_constant)
    if abs(ice_n_log_constant) > 1e-12:
        raise ValueError(
            f"{detector}/tile {tile_id}: v3 schema conversion currently requires "
            "the shared TMM ice index"
        )
    source_input_admittance = incident_y * (1.0 - source_modulated_air_reflection) / (
        1.0 + source_modulated_air_reflection
    )
    source_ice_side_reflection = (
        ice_index * incident_y - source_input_admittance
    ) / (ice_index * incident_y + source_input_admittance)
    ice_fresnel = (incident_y - ice_index * incident_y) / (
        incident_y + ice_index * incident_y
    )

    direct_response = np.exp(
        tmm_response.build_spectral_response_lookup(
            optical_values,
            v3_wavelengths,
            thicknesses,
            complex_reflection_modulation_ratio=mod_ratio,
        )["relative_log_response"]
    )

    source_clean_denominator = incident_y * first + first * source_input_admittance
    source_clean_power = (
        np.real(substrate_y / incident_y)
        * np.abs(2.0 * incident_y / source_clean_denominator) ** 2
    )
    source_coated_power, _ = mock.coated_substrate_power(
        source_modulated_air_reflection,
        first,
        v3_wavelengths,
        np.asarray(thicknesses, dtype=float)[:, None],
        ice_index,
        incident_y,
        substrate_y,
        incidence_sine,
    )
    source_state_response = source_coated_power / source_clean_power[None, :]

    mode_clean_reflection_error = error_summary(h5_reflection, source_ice_side_reflection)
    thickness_grid = np.asarray(thicknesses, dtype=float)
    sp = int(model.superpixel(sca, [x], [y])[0])
    h5_current_response = core.relative_response(
        h5_reflection[None, :],
        v3_wavelengths,
        thickness_grid[None, :],
        model._cos_theta[[sp]],
        model.wavelength_nm,
        model._n,
        model._kappa,
        model._b,
        model._n_ref,
        model._snell,
    )[0].T
    source_transformed_response = core.relative_response(
        source_ice_side_reflection[None, :],
        v3_wavelengths,
        thickness_grid[None, :],
        np.ones(1),
        model.wavelength_nm,
        np.full_like(model._n, ice_index.real),
        np.zeros_like(model._kappa),
        np.full_like(model._b, ice_fresnel, dtype=complex),
        ice_index.real,
        False,
    )[0].T

    return {
        "detector": detector,
        "tile_id": tile_id,
        "ice_index": [float(ice_index), 0.0],
        "mode_clean_reflection_error": mode_clean_reflection_error,
        "copied_v3_film_n_range": [float(np.min(model._n)), float(np.max(model._n))],
        "direct_tmm_vs_canonical_bundle_error": error_summary(
            direct_response,
            np.exp(bundle_api.evaluate(
                bundle, tile_id, thickness_grid, v3_wavelengths, detector=detector
            )[1]),
        ),
        "direct_tmm_vs_h5_current_response_error": error_summary(
            h5_current_response, direct_response
        ),
        "direct_tmm_vs_h5_ice_side_response_error": error_summary(
            h5_current_response, direct_response
        ),
        "sourcefit_tmm_vs_sourcefit_core_error": error_summary(
            source_transformed_response, source_state_response
        ),
        "sourcefit_tmm_vs_canonical_bundle_error": error_summary(
            source_state_response,
            np.exp(bundle_api.evaluate(
                bundle, tile_id, thickness_grid, v3_wavelengths, detector=detector
            )[1]),
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", default=["WFI01"])
    parser.add_argument("--tile", action="append", type=int, default=[0, 136, 255])
    parser.add_argument(
        "--thickness-nm", type=float, nargs="+", default=[0.0, 7.3, 50.0, 123.4, 250.0]
    )
    args = parser.parse_args(argv)
    bundle = bundle_api.load_bundle(RESPONSE_BUNDLE)
    rows = []
    with comparison.load_eric_v3_reader(comparison.DEFAULT_SOURCE_ZIP) as reader:
        model = reader.IceResponseModel(V3_MODEL)
        with h5py.File(MODE_PATH, "r") as mode_file:
            for detector in dict.fromkeys(args.detector):
                for tile_id in dict.fromkeys(args.tile):
                    rows.append(check_tile(
                        detector, tile_id, model, reader.core, bundle, mode_file,
                        args.thickness_nm,
                    ))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
