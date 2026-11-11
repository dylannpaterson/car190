"""Validate the joint CAR190/Amber modulated bundle against its source fits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_modulated_wfi_wavelength_response_bundle as builder
import wfi_relative_response as response
from wfi_wavelength_response_bundle import evaluate, load_bundle, tile_index


ROOT = Path(__file__).resolve().parent
PRODUCTION = ROOT / "20261106/corrected_statistics_4h_thinned/production"
BUNDLE_DIR = PRODUCTION / "wavelength_response_bundle"
CANDIDATE = BUNDLE_DIR / "wfi_all_2nm_rank16_joint_car190_amber_2day_modulated.npz"
PREVIOUS = BUNDLE_DIR / "wfi_all_2nm_rank16_amber_strong_red_led_mask.npz"
FIT_KIND = builder.JOINT_MODULATION_FIT


def main():
    bundle = load_bundle(CANDIDATE)
    previous = load_bundle(PREVIOUS)
    metadata = bundle["metadata"]
    if metadata.get("modulation_fit_family") != FIT_KIND:
        raise ValueError("Candidate bundle does not identify the joint 2-day modulation family")
    if metadata.get("modulation_zero_at_led_centers") is not False:
        raise ValueError("Joint-fit bundle must accurately report free LED-center modulation")
    if int(bundle["bundle_version"]) != int(previous["bundle_version"]):
        raise ValueError("Candidate changes the existing bundle version")

    retained = (
        "wavelengths_nm", "ice_thickness_nm", "tile_ids", "detectors", "sca_numbers",
        "tile_rows", "tile_cols", "row_starts", "row_stops_exclusive", "col_starts",
        "col_stops_exclusive", "tile_center_x_mm", "tile_center_y_mm", "observation_days",
        "observation_start_time", "observation_led", "observation_repeat", "observation_program",
        "groups", "post_reset_tile_log_offset", "post_reset_tile_throughput_factor",
        "ice_reset_day", "post_third_decon_tile_log_offset", "post_third_decon_tile_throughput_factor",
        "third_decon_day",
    )
    for name in retained:
        if name in previous or name in bundle:
            if name not in previous or name not in bundle:
                raise ValueError(f"Candidate changed retained field presence: {name}")
            np.testing.assert_array_equal(bundle[name], previous[name], err_msg=f"Changed retained field {name}")

    wavelengths = np.asarray(bundle["wavelengths_nm"], dtype=float)
    detectors = np.asarray(bundle["detectors"]).astype(str)
    tile_ids = np.asarray(bundle["tile_ids"], dtype=int)
    modulation_detectors = np.asarray(bundle["modulation_detectors"]).astype(str)
    records = []
    history_errors = []
    response_checks = []

    for detector in builder.DETECTORS:
        item = builder._load_detector_modulation(PRODUCTION, detector, wavelengths, fit_kind=FIT_KIND)
        modulation_index = int(np.flatnonzero(modulation_detectors == detector)[0])
        stored_ratio = (np.asarray(bundle["modulation_ratio_real"][modulation_index], dtype=float)
                        + 1j * np.asarray(bundle["modulation_ratio_imag"][modulation_index], dtype=float))
        np.testing.assert_allclose(stored_ratio, item["ratio"], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(bundle["modulation_log_amplitude_coefficients"][modulation_index],
                                   item["log_amplitudes"], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(bundle["modulation_phase_coefficients"][modulation_index],
                                   item["phases"], rtol=2e-6, atol=2e-7)

        joint = builder._load_joint_ice_history(PRODUCTION, detector, item)
        indices = np.searchsorted(joint["times"], bundle["observation_days"])
        if not np.allclose(joint["times"][indices], bundle["observation_days"], rtol=0, atol=1e-10):
            raise ValueError(f"Bundle CAR190 day axis is absent from the joint trajectory for {detector}")
        detector_indices = np.flatnonzero(detectors == detector)
        detector_tiles = tile_ids[detector_indices]
        if not np.array_equal(np.sort(detector_tiles), np.arange(256)):
            raise ValueError(f"Unexpected tile coverage for {detector}")
        for global_index in detector_indices:
            tile = int(tile_ids[global_index])
            expected_thickness = joint["thickness"][tile, indices]
            expected_rate = joint["rates"][tile, indices]
            thickness_error = float(np.max(np.abs(bundle["tile_ice_thickness"][global_index] - expected_thickness)))
            rate_error = float(np.max(np.abs(bundle["tile_instantaneous_rate_nm_day"][global_index] - expected_rate)))
            if thickness_error > 2e-5 or rate_error > 2e-5:
                raise ValueError(f"Joint ice history mismatch for {detector} tile {tile}: {thickness_error}, {rate_error}")
            history_errors.append((thickness_error, rate_error))

        for tile in (0, 136, 255):
            index = tile_index(bundle, tile, detector=detector)
            optical = builder._load_latest_tmm_optical_values(str(bundle["optical_fit_paths"][index]))
            final_joint_thickness = float(bundle["tile_ice_thickness"][index, -1])
            test_thickness = np.asarray([0.0, 7.3, 50.0, 123.4, 250.0, final_joint_thickness])
            direct = response.build_spectral_response_lookup(
                optical, wavelengths, test_thickness,
                complex_reflection_modulation_ratio=item["ratio"],
            )
            clean, relative = evaluate(bundle, tile, test_thickness, wavelengths, detector=detector)
            log_error = float(np.max(np.abs(relative - direct["relative_log_response"])))
            clean_error = float(np.max(np.abs(clean - direct["clean_power"])) /
                                np.max(np.abs(direct["clean_power"])))
            zero_error = float(np.max(np.abs(relative[0])))
            if log_error > 5e-4 or clean_error > 1e-6 or zero_error > 1e-12:
                raise RuntimeError(f"Direct TMM check failed for {detector} tile {tile}: "
                                   f"{log_error}, {clean_error}, {zero_error}")
            response_checks.append({
                "detector": detector,
                "tile_id": tile,
                "max_abs_log_response_error": log_error,
                "max_relative_clean_power_error": clean_error,
                "zero_thickness_max_abs_log_response": zero_error,
            })
        print(f"Validated {detector}: joint coefficients/history and 3 direct-TMM tiles", flush=True)

    response_table = pd.DataFrame(response_checks)
    response_table.to_csv(BUNDLE_DIR / "joint_car190_amber_2day_bundle_direct_tmm_checks.csv", index=False)
    result = {
        "candidate_bundle": str(CANDIDATE),
        "candidate_bundle_sha256": builder._sha256(CANDIDATE),
        "comparison_bundle": str(PREVIOUS),
        "bundle_version_preserved": int(bundle["bundle_version"]),
        "detectors": 18,
        "tiles": int(bundle["tile_ids"].size),
        "bundle_observation_count_preserved": int(bundle["observation_days"].size),
        "joint_ice_history_source": "joint CAR190 + Amber CAR201 photometry 2-day per-tile fit",
        "joint_ice_history_samples_compared": int(len(history_errors)),
        "max_abs_thickness_history_error_nm": float(max(x[0] for x in history_errors)),
        "max_abs_rate_history_error_nm_day": float(max(x[1] for x in history_errors)),
        "modulation_zero_at_led_centers": False,
        "max_led_center_complex_ratio_deviation": float(max(
            metadata["led_center_max_abs_complex_deviation_by_detector"].values()
        )),
        "direct_tmm_tile_checks": len(response_checks),
        "max_sampled_abs_log_response_error": float(response_table["max_abs_log_response_error"].max()),
        "max_sampled_relative_clean_power_error": float(response_table["max_relative_clean_power_error"].max()),
        "max_sampled_zero_thickness_log_response": float(response_table["zero_thickness_max_abs_log_response"].max()),
        "bundle_array_keys_preserved": set(bundle) - set(previous) == set(),
        "roll_zeropoints_embedded": False,
        "s3_uploaded": False,
    }
    summary_path = BUNDLE_DIR / "joint_car190_amber_2day_bundle_validation.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
