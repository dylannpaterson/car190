"""Check Amber provenance, retained trajectories, and sampled direct responses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_modulated_wfi_wavelength_response_bundle as builder
import wfi_relative_response as response
from wfi_wavelength_response_bundle import load_bundle, evaluate

ROOT=Path(__file__).resolve().parent
PRODUCTION=ROOT/"20261106/corrected_statistics_4h_thinned/production"
DIRECTORY=PRODUCTION/"wavelength_response_bundle"
NEW=DIRECTORY/"wfi_all_2nm_rank16_amber_strong_red_led_mask.npz"
OLD=DIRECTORY/"wfi_all_2nm_rank16_amber_support_aligned_endpoints.npz"
FAMILY="amber-student-t-strong-red-led-mask"


def main():
    bundle=load_bundle(NEW)
    metadata=bundle["metadata"]
    assert metadata["modulation_fit_family"]==FAMILY
    assert metadata["roll_zeropoints_embedded"] is False
    for name in ("clean_power","response_left","response_right","tile_ice_thickness","tile_instantaneous_rate_nm_day"):
        if not np.isfinite(bundle[name]).all():raise ValueError(f"Nonfinite array: {name}")
    preserved=("tile_ice_thickness","tile_instantaneous_rate_nm_day","observation_days",
               "post_reset_tile_log_offset","post_reset_tile_throughput_factor",
               "post_third_decon_tile_log_offset","post_third_decon_tile_throughput_factor")
    with np.load(OLD,allow_pickle=False) as old:
        for name in preserved:
            np.testing.assert_array_equal(bundle[name],old[name],err_msg=f"Changed retained field: {name}")
    wavelengths=np.asarray(bundle["wavelengths_nm"],float)
    detectors=np.asarray(bundle["detectors"]).astype(str)
    tile_ids=np.asarray(bundle["tile_ids"],int)
    modulation_detectors=np.asarray(bundle["modulation_detectors"]).astype(str)
    records=[]
    observations=0
    stars=0
    for detector in builder.DETECTORS:
        source=builder._load_detector_modulation(PRODUCTION,detector,wavelengths,fit_kind=FAMILY)
        entry=metadata["modulation_fit_sources"][detector]
        assert entry["photometry"]=="Amber" and entry["likelihood"]=="Student-t" and entry["nu"]==4.
        assert entry["summary_sha256"]==source["summary_sha256"]
        assert entry["parameters_sha256"]==source["parameters_sha256"]
        assert entry["effective_area_sha256"]==builder._sha256(entry["effective_area_file"])
        assert Path(entry["effective_area_file"]).name==f"Roman_effarea_v8_SCA{int(detector[3:]):02d}_20240301.ecsv"
        observations+=entry["used_observations"]
        stars+=entry["used_stars"]
        mi=int(np.flatnonzero(modulation_detectors==detector)[0])
        np.testing.assert_allclose(bundle["modulation_log_amplitude_coefficients"][mi],source["log_amplitudes"],rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(bundle["modulation_phase_coefficients"][mi],source["phases"],rtol=1e-6,atol=1e-8)
        ratio=(np.asarray(bundle["modulation_ratio_real"][mi],float)
               +1j*np.asarray(bundle["modulation_ratio_imag"][mi],float))
        np.testing.assert_allclose(ratio,source["ratio"],rtol=1e-6,atol=1e-7)
        for tile in (0,136,255):
            index=int(np.flatnonzero((detectors==detector)&(tile_ids==tile))[0])
            optical=builder._load_latest_tmm_optical_values(str(bundle["optical_fit_paths"][index]))
            thickness=np.array([0.,7.3,50.,123.4,250.])
            direct=response.build_spectral_response_lookup(optical,wavelengths,thickness,
                         complex_reflection_modulation_ratio=source["ratio"])
            clean,relative=evaluate(bundle,tile,thickness,wavelengths,detector=detector)
            error=float(np.max(np.abs(relative-direct["relative_log_response"])))
            clean_error=float(np.max(np.abs(clean-direct["clean_power"]))/np.max(np.abs(direct["clean_power"])))
            zero_error=float(np.max(np.abs(relative[0])))
            if error>5e-4 or clean_error>1e-6 or zero_error>1e-12:
                raise RuntimeError(f"Direct response check failed: {detector}/{tile}: {error}, {clean_error}, {zero_error}")
            records.append({"detector":detector,"tile_id":tile,"max_abs_log_response_error":error,
                            "max_relative_clean_power_error":clean_error,"zero_thickness_max_abs_log_response":zero_error})
        print(f"Validated {detector}: fit coefficients, provenance, and 3 direct-TMM tile comparisons",flush=True)
    table=pd.DataFrame(records)
    table.to_csv(DIRECTORY/"amber_strong_red_mask_bundle_validation.csv",index=False)
    result={"bundle":str(NEW),"bundle_sha256":builder._sha256(NEW),"detectors":18,"tiles":len(tile_ids),
            "photometry":"Amber","likelihood":"Student-t","nu":4.,"fit_measurements":observations,
            "fit_detector_star_pairs":stars,"sampled_direct_tile_checks":len(records),
            "sampled_thicknesses_nm":[0.,7.3,50.,123.4,250.],
            "max_sampled_abs_log_response_error":float(table.max_abs_log_response_error.max()),
            "max_sampled_relative_clean_power_error":float(table.max_relative_clean_power_error.max()),
            "zero_thickness_max_abs_log_response":float(table.zero_thickness_max_abs_log_response.max()),
            "retained_arrays_equal_previous_modulated_bundle":list(preserved),
            "global_grid_max_svd_error":max(metadata["response_svd_max_abs_error_by_detector"].values()),
            "bundle_fields_exclude_car201_roll_offsets":not metadata["roll_zeropoints_embedded"]}
    (DIRECTORY/"amber_strong_red_mask_bundle_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
