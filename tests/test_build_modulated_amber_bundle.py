import json

import numpy as np
import pandas as pd
import pytest

import build_modulated_wfi_wavelength_response_bundle as builder
import mock_fit_wfi_star_reflectance_modulation as mock


def write_amber_fixture(tmp_path, detector="WFI02", success=True):
    directory = (tmp_path / detector.lower() / "spline_relative_response_normal/experiments"
                 / f"car201_amber_{detector.lower()}_ep3v2_ep4_common_zane_quality_student_t_nu4")
    directory.mkdir(parents=True)
    summary = {"detector":detector,"photometry":"Amber","likelihood":"Student-t","nu":4.,
               "optimizer_success":success,"optical_state_source":str(tmp_path/"pinned_optics"),
               "amber_input":str(tmp_path/"amber.parquet"),"maxent_weight":1000.,"phase_weight":100.,
               "used_stars":100,"used_observations":800,"extra_student_t_scale_mmag":1.2}
    (directory/"fit_summary.json").write_text(json.dumps(summary))
    wave = np.arange(470.,2321.,5.)
    basis,names,*_ = mock.modulation_basis(wave,grouping="per-gap")
    amplitude=np.linspace(-.1,.1,len(names))
    phase=np.linspace(.05,-.05,len(names))
    pd.DataFrame({"mode":names,"fit_log_amplitude":amplitude,"fit_phase_rad":phase}).to_csv(
        directory/"student_t_modulation_parameters.csv",index=False)
    return wave,amplitude,phase,basis


def test_amber_loader_uses_student_t_coefficients_and_records_provenance(tmp_path):
    wave,amplitude,phase,basis=write_amber_fixture(tmp_path)
    result=builder._load_detector_modulation(tmp_path,"WFI02",wave,fit_kind="amber-student-t")
    np.testing.assert_allclose(result["ratio"],np.exp(amplitude@basis+1j*(phase@basis)))
    assert result["fit_provenance"]["photometry"]=="Amber"
    assert result["fit_provenance"]["likelihood"]=="Student-t"
    assert result["fit_provenance"]["scatter_scale_applied_to_optical_response"] is False
    assert result["anchor_error"]<1e-12
    assert len(result["parameters_sha256"])==64


def test_amber_loader_rejects_unconverged_fit(tmp_path):
    wave,*_=write_amber_fixture(tmp_path,success=False)
    with pytest.raises(ValueError,match="unsuccessful Amber"):
        builder._load_detector_modulation(tmp_path,"WFI02",wave,fit_kind="amber-student-t")


def test_main_accepts_amber_selection(monkeypatch):
    captured=[]
    monkeypatch.setattr(builder,"build_bundle",lambda args:captured.append(args) or 0)
    assert builder.main(["--base-bundle","base.npz","--modulation-fit","amber-student-t",
                         "--output","new.npz","--detector","WFI01"])==0
    assert captured[0].modulation_fit=="amber-student-t"


def test_joint_modulation_path_can_be_pinned_to_full_tmm_route(tmp_path):
    joint_name = "car190_amber_joint_2day_reader_compatible"
    relative = (
        tmp_path / "wfi01/relative_response/experiments" / joint_name
    )
    full_tmm = tmp_path / "wfi01/full_tmm/experiments" / joint_name
    relative.mkdir(parents=True)
    full_tmm.mkdir(parents=True)

    assert builder._joint_experiment_dir(
        tmp_path, "WFI01", joint_name, fit_subdir="full_tmm"
    ) == full_tmm


def test_detector_area_family_requires_matching_table_provenance(tmp_path):
    wave,*_=write_amber_fixture(tmp_path)
    old=(tmp_path/"wfi02/spline_relative_response_normal/experiments"
         /"car201_amber_wfi02_ep3v2_ep4_common_zane_quality_student_t_nu4")
    current=old.with_name(old.name+"_sca02_effective_area")
    old.rename(current)
    area=tmp_path/"Roman_effarea_v8_SCA02_20240301.ecsv"
    area.write_text("pinned detector-specific table")
    area.with_suffix(".provenance.json").write_text(json.dumps({"detector":"WFI02","commit":"test"}))
    summary=json.loads((current/"fit_summary.json").read_text())
    summary.update(effective_area_file=str(area),effective_area_sha256=builder._sha256(area))
    (current/"fit_summary.json").write_text(json.dumps(summary))
    result=builder._load_detector_modulation(tmp_path,"WFI02",wave,
        fit_kind="amber-student-t-detector-effective-area")
    assert result["fit_provenance"]["effective_area_sha256"]==builder._sha256(area)
    assert result["fit_provenance"]["effective_area_applied_to_exported_response"] is False
    area.write_text("changed")
    with pytest.raises(ValueError,match="Incorrect or changed"):
        builder._load_detector_modulation(tmp_path,"WFI02",wave,
            fit_kind="amber-student-t-detector-effective-area")


def test_support_aligned_loader_reconstructs_fit_basis_and_anchors(tmp_path):
    import hashlib
    from support_aligned_modulation import configure_basis
    from fit_car201_zane_star_reflectance_modulation import RealPhotometryFit
    # Use the actual fit's basis construction with deliberately different
    # detector-specific endpoint values on a bracketing effective-area grid.
    root=tmp_path/"wfi02/spline_relative_response_normal/experiments"
    previous=root/"car201_amber_wfi02_ep3v2_ep4_common_zane_quality_student_t_nu4_sca02_effective_area"
    current=previous.with_name(previous.name+"_support_aligned_free_endpoints_bound0p60")
    current.mkdir(parents=True)
    previous.mkdir()
    area=tmp_path/"Roman_effarea_v8_SCA02_20240301.ecsv"
    header=",".join(["Wave",*mock.FILTERS])
    rows=["0.470,"+",".join(["0"]*8),"0.474,"+",".join(["1"]*8),
          "2.333,"+",".join(["1"]*8),"2.335,"+",".join(["0"]*8)]
    area.write_text("\n".join([header,*rows])+"\n")
    area.with_suffix(".provenance.json").write_text(json.dumps({"detector":"WFI02"}))
    (previous/"fit_summary.json").write_text(json.dumps({"optical_state_source":"optics","amber_input":"amber"}))
    summary={"detector":"WFI02","photometry":"Amber","likelihood":"Student-t","nu":4.,
             "optimizer_success":True,"endpoint_values_free":True,"LED_values_free":False,
             "endpoint_amplitude_bound":.6,"modulation_boundaries_nm":[474.,2333.],
             "effective_area_file":str(area),"effective_area_sha256":hashlib.sha256(area.read_bytes()).hexdigest(),
             "extra_student_t_scale_mmag":1.,"used_stars":2,"used_observations":20,
             "maxent_weight":1000.,"phase_weight":100.,"parameters_at_bounds":[]}
    (current/"fit_summary.json").write_text(json.dumps(summary))
    wave=np.arange(470.,2701.,2.)
    shape=(1,len(wave))
    model=RealPhotometryFit(wave,np.ones((8,len(wave))),np.ones((2,len(wave))),
        np.ones((1,8,2)),np.zeros((2,8,2),int),np.full(shape,.2+.05j),np.ones(shape,complex),
        np.full(shape,.7),np.ones(1),np.full(shape,4.+.1j),np.zeros(1),np.full(1,1.3),"per-gap")
    configure_basis(model,(474.,2333.),free_endpoints=True)
    amp=np.linspace(-.2,.2,9); phase=np.linspace(.1,-.1,9)
    pd.DataFrame({"mode":model.basis_names,"fit_log_amplitude":amp,"fit_phase_rad":phase}).to_csv(current/"modulation_parameters.csv",index=False)
    result=builder._load_detector_modulation(tmp_path,"WFI02",wave,fit_kind="amber-student-t-support-aligned")
    np.testing.assert_allclose(result["ratio"],np.exp(amp@model.basis+1j*(phase@model.basis)))
    assert result["anchor_error"]<1e-12
    assert result["fit_provenance"]["modulation_boundaries_nm"]==[474.,2333.]
