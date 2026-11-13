"""Sequential Amber Student-t detector experiments with explicit selections."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from astropy.time import Time

import heatmap_car201_allstar_model_correction as workflow
from fit_amber_wfi01_student_t import StudentObjective, NU

ROOT = Path(__file__).resolve().parents[1]
CAR201_DATA_ROOT = ROOT / "car201_data"
AMBER = CAR201_DATA_ROOT / "ambers_photometry/photometry-all-star-epochs.parquet"
FILTERS, PASSES = workflow.FILTERS, workflow.PASSES
DETECTOR_EFFECTIVE_AREA_DIR=ROOT/"external_reference_data/roman_effarea_v8_20240301"


def paths(detector):
    lower = detector.lower()
    parent = workflow.PRODUCTION_ROOT / lower / "spline_relative_response_normal/experiments"
    old = parent / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched"
    new = parent / f"car201_amber_{lower}_ep3v2_ep4_common_zane_quality_student_t_nu4"
    return old, new


def canonical_configuration(detector,effective_area_path=None,output_dir=None):
    """Resolve future fit defaults without changing historical-reader paths."""
    if detector not in workflow.DETECTORS:
        raise ValueError(f"Expected explicit WFI01–WFI18 detector, got {detector}")
    number=int(detector[3:])
    area=Path(effective_area_path) if effective_area_path is not None else (
        DETECTOR_EFFECTIVE_AREA_DIR/f"Roman_effarea_v8_SCA{number:02d}_20240301.ecsv")
    if not area.is_file():
        raise FileNotFoundError(
            f"Detector-specific effective-area table missing: {area}. "
            f"Fetch it with fetch_roman_detector_effective_areas.py --detector {detector}; "
            "SCA05 is not used as a fallback."
        )
    _,historical=paths(detector)
    output=Path(output_dir) if output_dir is not None else historical.with_name(
        historical.name+f"_sca{number:02d}_effective_area")
    return area.resolve(),output.resolve()


def successful(detector,effective_area_path=None,output_dir=None):
    _, output = paths(detector)
    if output_dir is not None:output=Path(output_dir)
    file = output / "fit_summary.json"
    if not file.is_file():
        return False
    summary = json.loads(file.read_text())
    if effective_area_path is not None:
        expected=hashlib.sha256(Path(effective_area_path).read_bytes()).hexdigest()
        if summary.get("effective_area_sha256")!=expected:return False
    return (summary.get("optimizer_success",False) and summary.get("nu")==NU
            and (output / "predictions.npz").is_file())


def prepare(detector,write_inputs=True,effective_area_path=None,output_dir=None,
            full_effective_area_support=False,ice_history_fit_root=None,
            optical_cache_root=None):
    """Load inputs; the legacy area default supports historical diagnostics.

    All fitting entry points pass the explicitly resolved canonical table.
    """
    old, output = paths(detector)
    if output_dir is not None:output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    summary = json.loads((old / "car201_actual_star_modulation_fit_summary.json").read_text())
    zane = pd.read_csv(old / "car201_actual_star_modulation_measurements.csv",dtype={"source_id":str})
    amber = pd.read_parquet(AMBER)
    amber = amber.loc[amber["sca"]==detector].copy()
    epoch_map = {f"r010430300{i}00":f"ep3_pass{i}" for i in range(1,7)}
    epoch_map.update({f"r010430400{i}00":f"ep4_pass{i}" for i in range(1,6)})
    amber["pass"] = amber["epoch"].map(epoch_map)
    amber = amber.loc[amber["pass"].notna()].copy()
    common = zane.loc[zane["likelihood_used"]].merge(amber,on=["source_id","filter","pass"],
                  suffixes=("_zane","_amber"),validate="one_to_one")
    good = (np.isfinite(common["mag_obs_amber"]) & np.isfinite(common["mag_err_amber"])
            & (common["mag_err_amber"]>0) & np.isfinite(common["x_pixel_amber"])
            & np.isfinite(common["y_pixel_amber"]))
    common = common.loc[good].copy()
    common = common.loc[common.groupby(["source_id","filter"])["pass"].transform("size")>=2].copy()
    if common.empty:
        raise ValueError(f"No common repeated measurements for {detector}")
    ids = sorted(common["source_id"].unique())
    source_lookup={source_id:i for i,source_id in enumerate(ids)}
    filter_lookup={name:i for i,name in enumerate(FILTERS)}
    pass_lookup={name:i for i,name in enumerate(PASSES)}
    shape = (len(ids),len(FILTERS),len(PASSES))
    magnitude, error, zane_magnitude, zane_error = (np.full(shape,np.nan) for _ in range(4))
    valid = np.zeros(shape,dtype=bool)
    indices = (common["source_id"].map(source_lookup).to_numpy(int),
               common["filter"].map(filter_lookup).to_numpy(int),
               common["pass"].map(pass_lookup).to_numpy(int))
    magnitude[indices] = common["mag_obs_amber"].to_numpy(float)
    error[indices] = common["mag_err_amber"].to_numpy(float)
    zane_magnitude[indices] = common["mag_obs_zane"].to_numpy(float)
    zane_error[indices] = common["mag_err_zane"].to_numpy(float)
    valid[indices] = True
    common["mjd_amber"] = common["mjd"]
    frames = []
    for epoch in PASSES:
        rows = common.loc[common["pass"]==epoch]
        frames.append(rows[["source_id","filter","mjd_amber","x_pixel_amber","y_pixel_amber"]].rename(
            columns={"mjd_amber":"mjd","x_pixel_amber":"x_pixel","y_pixel_amber":"y_pixel"}))
    seds = workflow._load_successful_seds(((workflow.FAINT_SED_PATH,"faint_v2"),),ids)
    if len(seds)!=len(ids):
        raise ValueError("A common-fit source lacks a successful faint-v2 SED")
    wave,area = workflow._make_wavelength_grid(float(summary["modulation"]["wavelength_stride_nm"]),
                                               effective_area_path=effective_area_path,
                                               full_effective_area_support=full_effective_area_support)
    reference = Time(workflow.fit_model.ICE_REFERENCE_UTC,format="isot",scale="utc").mjd
    frozen,model = workflow._predict_detector(detector,frames,ids,seds,workflow.SED_GRID_PATH,
                                              old,wave,area,reference,return_model=True,
                                              effective_area_path=effective_area_path,
                                              full_effective_area_support=full_effective_area_support,
                                              ice_history_fit_root=ice_history_fit_root,
                                              optical_cache_root=optical_cache_root)
    model.entropy_weight=summary["modulation"]["maxent_regularization"]["weight"]
    model.phase_smoothness_weight=summary["modulation"]["phase_smoothness_regularization"]["weight"]
    coefficients,zero_points = workflow._load_fit_parameters(detector,old,summary)
    group_names=summary["modulation"]["zeropoint_model"]["group_names"]
    zpts=[]
    for band in FILTERS:
        for name in group_names[1:]:
            selected=zero_points.loc[(zero_points["filter"]==band)&(zero_points["offset_group"]==name),"fit_log_flux_offset"]
            if len(selected)!=1:raise ValueError("Saved roll offset missing")
            zpts.append(float(selected.iloc[0]))
    initial=np.concatenate([coefficients,np.asarray(zpts)])
    if write_inputs:
        np.savez_compressed(output/"fit_inputs.npz",source_ids=np.asarray(ids),magnitude=magnitude,
                            error=error,valid=valid,zane_magnitude=zane_magnitude,zane_error=zane_error,
                            frozen_zane_prediction_mag=frozen)
        common.to_csv(output/"common_measurements.csv",index=False)
    print(f"{detector}: {len(ids)} stars; {int(valid.sum())} common measurements",flush=True)
    return summary,common,ids,magnitude,error,valid,frozen,model,initial


def run_detector(detector,effective_area_path=None,output_dir=None):
    effective_area_path,output_dir=canonical_configuration(detector,effective_area_path,output_dir)
    if successful(detector,effective_area_path=effective_area_path,output_dir=output_dir):
        print(f"{detector}: reusing completed Student-t experiment",flush=True)
        return
    old,output=paths(detector)
    if output_dir is not None:output=Path(output_dir)
    old_summary,common,ids,magnitude,error,valid,frozen,model,parameters=prepare(
        detector,effective_area_path=effective_area_path,output_dir=output)
    objective=StudentObjective(model,magnitude,error,valid)
    initial=np.concatenate([parameters/objective.coordinate_scale,[np.log(6.)]])
    initial_source="original Zane Gaussian fit"
    _,baseline_dir=paths(detector)
    if (baseline_dir/"predictions.npz").is_file() and (baseline_dir/"fit_summary.json").is_file():
        with np.load(baseline_dir/"predictions.npz",allow_pickle=False) as previous:
            prior_parameters=previous["student_t_fit_parameters"]
            if prior_parameters.shape!=parameters.shape:
                raise ValueError("Warm-start modulation basis differs from current model")
            initial[:-1]=prior_parameters/objective.coordinate_scale
        baseline_summary=json.loads((baseline_dir/"fit_summary.json").read_text())
        initial[-1]=np.log(baseline_summary["extra_student_t_scale_mmag"])
        initial_source=str(baseline_dir)
    count=model.mode_count
    bounds=[(-3.,3.)]*count+[(-10.,10.)]*count+[(-25.,25.)]*model.zero_point_count
    bounds.append((np.log(1e-4),np.log(100.)))
    roll_support=[]
    groups=np.asarray(model.pass_zero_point_group_ids)
    for fi,band in enumerate(FILTERS):
        for gid in range(1,model.zero_point_group_count+1):
            connected=np.any(valid[:,fi,groups==0],axis=1)&np.any(valid[:,fi,groups==gid],axis=1)
            stars=int(connected.sum())
            index=2*count+fi*model.zero_point_group_count+gid-1
            if stars==0:
                initial[index]=0.
                bounds[index]=(0.,0.)
            roll_support.append({"filter":band,"roll_group_id":gid,"cross_roll_stars":stars,
                                 "offset_fitted":stars>0})
    # Initial gauge validation against the original forward calculation. It
    # excludes the roll term because unsupported roll offsets are fixed to zero.
    current=initial[:-1]*objective.coordinate_scale
    cached, _=objective.forward.evaluate(current[:2*count])
    original=model.modeled_log_response(current[:2*count])
    forward_error=float(np.max(np.abs(cached-original)))
    if forward_error>1e-10:
        raise RuntimeError(f"Cached optical response mismatch: {forward_error}")
    value,gradient=objective(initial)
    errors=[]
    for index in (0,count,len(initial)-2,len(initial)-1):
        if bounds[index][0]==bounds[index][1]:continue
        upper,lower=initial.copy(),initial.copy()
        upper[index]+=1e-4
        lower[index]-=1e-4
        numerical=(objective.state(upper)[0]-objective.state(lower)[0])/(2e-4)
        errors.append(abs(numerical-gradient[index])/(1+abs(numerical)))
    gradient_error=max(errors,default=0.)
    print(f"{detector}: response check={forward_error:.3g}; gradient check={gradient_error:.3g}",flush=True)
    if gradient_error>2e-3:
        raise RuntimeError(f"Student-t gradient verification failed: {gradient_error}")
    started=time.monotonic()
    iterations=0
    def checkpoint(coordinates):
        nonlocal iterations
        iterations+=1
        np.savez(output/"optimizer_checkpoint.npz",coordinates=coordinates,iterations=iterations)
    fit=minimize(objective,initial,method="L-BFGS-B",jac=True,bounds=bounds,callback=checkpoint,
                 options={"maxiter":400,"ftol":1e-11,"gtol":1e-5,"maxls":30})
    value,prediction,location,residual,score,scales=objective.state(fit.x)
    prediction/=1000.
    fitted=fit.x[:-1]*objective.coordinate_scale
    extra=float(np.exp(fit.x[-1]))
    final_gradient=objective(fit.x)[1]
    projected_gradient=final_gradient.copy()
    for index,(low,high) in enumerate(bounds):
        if (fit.x[index]<=low+1e-8 and final_gradient[index]>0) or (fit.x[index]>=high-1e-8 and final_gradient[index]<0) or low==high:
            projected_gradient[index]=0.
    summary={"detector":detector,"photometry":"Amber","likelihood":"Student-t","nu":NU,
             "extra_student_t_scale_mmag":extra,"scale_definition":"sqrt(catalog_error**2 + extra_scale**2); Student-t scale, not RMS",
             "used_stars":len(ids),"used_observations":int(valid.sum()),
             "star_filter_brightness_groups":int(objective.active_groups.sum()),
             "star_filter_locations_reestimated_under_student_t":True,
             "optimizer_success":bool(fit.success),"optimizer_message":str(fit.message),
             "objective":value,"iterations":int(fit.nit),"objective_gradient_evaluations":objective.calls,
             "max_projected_gradient":float(np.max(np.abs(projected_gradient))),
             "max_abs_profile_location_score_per_mmag":float(np.max(np.abs(score.sum(axis=1)))),
             "gradient_check_max_relative_discrepancy":gradient_error,
             "cached_vs_original_response_max_abs_log_difference":forward_error,
             "elapsed_seconds":time.monotonic()-started,
             "amplitude_bound":.3,"phase_bound_rad":1.,"maxent_weight":model.entropy_weight,
             "phase_weight":model.phase_smoothness_weight,"roll_identifiability":roll_support,
             "unsupported_roll_offset_policy":"fixed to zero if no common fit stars connect the roll groups",
             "selection":"common Amber/Zane measurements passing Zane saved likelihood mask; finite positive Amber errors; no extra clipping; only repeated source/filter groups",
             "amber_has_native_dq":False,"amber_input":str(AMBER),"native_magnitude_kind":"total corrected; EE applied; used as supplied",
             "nov6_visit_included":False,"sed_policy":"same faint-v2 successful posteriors as Zane fitting cohort",
             "source_zane_fit":str(old),"optical_state_source":old_summary["ice_history"]["optical_state_source"],
             "ice_history_root":str(workflow.PRODUCTION_ROOT/detector.lower()/"spline_relative_response_normal/fits"),
             "effective_area_file":str(Path(effective_area_path or workflow.mock_model.DEFAULT_EFFECTIVE_AREA).resolve()),
             "effective_area_sha256":hashlib.sha256(Path(effective_area_path or workflow.mock_model.DEFAULT_EFFECTIVE_AREA).read_bytes()).hexdigest(),
             "effective_area_provenance":(json.loads(effective_area_path.with_suffix(".provenance.json").read_text())
                                          if effective_area_path.with_suffix(".provenance.json").is_file() else None),
             "initial_parameter_source":initial_source,
             "code_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                 "run_amber_student_t_detectors.py","fit_amber_wfi01_student_t.py","car201_photometry_forward_jacobian.py")}}
    (output/"fit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(output/"predictions.npz",source_ids=np.asarray(ids),valid=valid,
                        magnitude=magnitude,error=error,frozen_zane_prediction_mag=frozen,
                        student_t_prediction_mag=prediction,student_t_fit_parameters=fitted,
                        robust_locations_mmag=location,robust_residuals_mmag=residual,
                        robust_scales_mmag=scales,active_groups=objective.active_groups)
    pd.DataFrame({"mode":model.basis_names,"fit_log_amplitude":fitted[:count],
                  "fit_phase_rad":fitted[count:2*count]}).to_csv(output/"student_t_modulation_parameters.csv",index=False)
    zpt=fitted[2*count:].reshape(len(FILTERS),model.zero_point_group_count)
    pd.DataFrame({"filter":FILTERS,"fit_log_flux_offset":zpt[:,0],
                  "equivalent_magnitude_offset_mmag":-1000*zpt[:,0]/workflow.LOG_FLUX_PER_MAG,
                  "cross_roll_fit_star_count":[row["cross_roll_stars"] for row in roll_support],
                  "offset_fitted":[row["offset_fitted"] for row in roll_support]}).to_csv(
                      output/"student_t_roll_offsets.csv",index=False)
    print(f"{detector}: success={fit.success}; extra t-scale={extra:.4f} mmag; iterations={fit.nit}; "
          f"max projected gradient={np.max(np.abs(projected_gradient)):.3g}",flush=True)
    if not fit.success:
        raise RuntimeError(f"{detector}: optimizer failed: {fit.message}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",action="append",required=True,choices=workflow.DETECTORS)
    parser.add_argument("--single-process",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--effective-area",type=Path,help="Override the default detector-specific SCA01–SCA18 curve; requires a separate output directory")
    parser.add_argument("--output-dir",type=Path,help="Separate experiment directory for one explicit detector")
    parser.add_argument("--check-configuration",action="store_true",help="Show selected tables and outputs without running fits")
    parser.add_argument("--legacy-anchored",action="store_true",help="Use the historical detector-area anchored-endpoint model")
    args=parser.parse_args()
    if args.output_dir is not None and len(args.detector)!=1:
        parser.error("--output-dir requires one explicit detector")
    if args.effective_area is not None and args.output_dir is None:
        parser.error("--effective-area requires --output-dir to preserve the baseline experiment")
    if not args.single_process and not args.legacy_anchored:
        if args.effective_area is not None or args.output_dir is not None:
            parser.error("Support-aligned production uses detector-specific default areas and output paths; use --legacy-anchored for explicit historical overrides")
        for detector in args.detector:
            area, old = canonical_configuration(detector)
            directory = old.with_name(old.name + "_support_aligned_free_endpoints_bound0p60_soft_led_sed_red_power_0p05")
            if args.check_configuration:
                print(json.dumps({"detector": detector, "effective_area_file": str(area),
                                  "output_dir": str(directory), "model": "support-aligned-free-endpoints-strong-red-led-mask",
                                  "completed_fit": (directory / "production_fit_summary.json").is_file()}), flush=True)
                continue
            subprocess.run([sys.executable, str(ROOT / "wfi_pipeline" / "run_amber_strong_red_production.py"),
                            "--detector", detector], cwd=ROOT, check=True)
        return
    if args.single_process:
        if len(args.detector)!=1:raise ValueError("One child detector per process")
        run_detector(args.detector[0],effective_area_path=args.effective_area,output_dir=args.output_dir)
        return
    for detector in args.detector:
        area,directory=canonical_configuration(detector,args.effective_area,args.output_dir)
        if args.check_configuration:
            print(json.dumps({"detector":detector,"effective_area_file":str(area),
                              "effective_area_sha256":hashlib.sha256(area.read_bytes()).hexdigest(),
                              "output_dir":str(directory),
                              "completed_matching_fit":successful(detector,effective_area_path=area,output_dir=directory)}),flush=True)
            continue
        if successful(detector,effective_area_path=area,output_dir=directory):
            print(f"{detector}: completed experiment already present",flush=True)
            continue
        directory.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),"--detector",detector,"--single-process"]
        command.extend(["--effective-area",str(area),"--output-dir",str(directory)])
        (directory/"command.json").write_text(json.dumps(command,indent=2)+"\n")
        print(f"Starting {detector}; log={directory/'student_t_fit.log'}",flush=True)
        with (directory/"student_t_fit.log").open("w") as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((directory/"fit_summary.json").read_text())
        print(f"Completed {detector}: scale={summary['extra_student_t_scale_mmag']:.4f} mmag",flush=True)


if __name__=="__main__":
    main()
