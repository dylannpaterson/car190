"""Controlled WFI01 comparison of Amber and Zane, with a fitted-floor refit."""

import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

import fit_car201_zane_star_reflectance_modulation as fitter
import heatmap_car201_allstar_model_correction as workflow

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = workflow.PRODUCTION_ROOT
DETECTOR = "WFI01"
BASE = PRODUCTION / "wfi01/spline_relative_response_normal/experiments/car201_zane_wfi01_ep3v2_ep4_faint_v2_tile_matched"
OUTPUT = BASE.with_name("car201_amber_wfi01_ep3v2_ep4_common_zane_quality_fitted_floor")
FILTERS, PASSES = workflow.FILTERS, workflow.PASSES


def profile_residual(magnitude, prediction, error, valid):
    raw = magnitude - prediction
    weight = np.where(valid, 1. / np.square(error), 0.)
    weighted = np.where(valid, raw, 0.) * weight
    norm = weight.sum(axis=2, keepdims=True)
    center = np.divide(weighted.sum(axis=2, keepdims=True), norm,
                       out=np.zeros_like(norm), where=norm > 0)
    return np.where(valid, raw-center, np.nan)


def prediction(model, coefficients, zero_points, group_names):
    response = model.modeled_log_response(coefficients)
    for pass_index, group_id in enumerate(model.pass_zero_point_group_ids):
        if group_id == 0:
            continue
        for band_index, band in enumerate(FILTERS):
            offset = zero_points.loc[(zero_points["filter"] == band)
                                     & (zero_points["offset_group"] == group_names[group_id]), "fit_log_flux_offset"]
            if len(offset) != 1:
                raise ValueError("Missing unique roll offset")
            response[:, band_index, pass_index] += float(offset.iloc[0])
    return -response / fitter.LOG_FLUX_PER_MAG


def summarize_metrics(variants, valid, matched, source_ids):
    output = []
    for name, (magnitude, predicted, errors) in variants.items():
        residual = profile_residual(magnitude, predicted, errors, valid)
        flux = 100 * np.expm1(-fitter.LOG_FLUX_PER_MAG * residual)
        for band_index, band in enumerate(FILTERS):
            values = residual[:, band_index][valid[:, band_index]] * 1000
            flux_values = flux[:, band_index][valid[:, band_index]]
            output.append({"variant": name, "filter": band, "observations": len(values),
                           "profiled_rms_mmag": np.sqrt(np.mean(values**2)),
                           "residual_p16_mmag": np.quantile(values,.16),
                           "residual_median_mmag": np.median(values),
                           "residual_p84_mmag": np.quantile(values,.84),
                           "fraction_within_0p5_percent": np.mean(np.abs(flux_values)<=.5)})
    return pd.DataFrame(output)


def plot_diagnostics(matched, variants, arrays, valid, prefix):
    # Measured reduction differences: remove only the per-star/filter constant
    # using a common set of Zane catalog weights, not an ice model.
    zmag, amag, zerr, aerr = arrays
    difference = profile_residual(amag, zmag, zerr, valid) * 1000.
    days = matched.groupby("pass")["mjd_amber"].median().reindex(PASSES).to_numpy(float)
    dates = mdates.date2num(Time(days,format="mjd",scale="utc").to_datetime())
    fig, axes = plt.subplots(4,2,figsize=(13,11),sharex=True)
    for band_index, (band, axis) in enumerate(zip(FILTERS,axes.flat,strict=True)):
        for epoch_index, date in enumerate(dates):
            values = difference[:,band_index,epoch_index]
            values = values[np.isfinite(values)]
            if not len(values): continue
            low,median,high = np.quantile(values,[.16,.5,.84])
            axis.errorbar(date,median,yerr=[[median-low],[high-median]],fmt="o",color="#3478a8",capsize=2)
        axis.axhline(0,color="0.5",linewidth=.7)
        axis.set_title(band)
        axis.grid(alpha=.2)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.supylabel("Amber − Zane after subtracting one constant per star/filter (mmag)")
    fig.suptitle("WFI01: directly measured photometry differences; matched measurements",y=.99)
    fig.tight_layout(rect=(.03,.04,1,.95))
    fig.savefig(prefix / "amber_vs_zane_photometry.png",dpi=180)
    plt.close(fig)

    # A common measured EP3 reference, with no invented baseline for stars
    # lacking that observation. The likelihood metrics above retain all repeats.
    fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
    colors=("#555555","#267da5","#d4782e","#619649")
    for band_index,(band,axis) in enumerate(zip(FILTERS,axes.flat,strict=True)):
        for (name,(magnitude,predicted,errors)),color in zip(variants.items(),colors,strict=True):
            corrected = magnitude-predicted
            delta = corrected-corrected[:,:,:1]
            flux=100*np.expm1(-fitter.LOG_FLUX_PER_MAG*delta)
            for epoch_index,date in enumerate(dates):
                select=valid[:,band_index,epoch_index]&valid[:,band_index,0]
                values=flux[select,band_index,epoch_index]
                if not len(values) or epoch_index==0: continue
                low,median,high=np.quantile(values,[.16,.5,.84])
                offset=(list(variants).index(name)-1.5)/24.
                axis.errorbar(date+offset,median,yerr=[[median-low],[high-median]],fmt="o",
                              color=color,capsize=1,markersize=3,label=name if epoch_index==1 else None)
        axis.axhspan(-.5,.5,color="#65b86b",alpha=.15)
        axis.axhline(0,color="0.5",linewidth=.7)
        axis.set_title(band)
        axis.grid(alpha=.2)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.95),ncol=2,fontsize=8)
    fig.supylabel("Corrected flux change from the star's measured EP3 pass 1 (%)")
    fig.suptitle("WFI01: frozen-model transfer and Amber fitted-floor refit",y=.995)
    fig.tight_layout(rect=(.03,.04,1,.89))
    fig.savefig(prefix / "amber_vs_zane_model_corrections.png",dpi=180)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    zane=pd.read_csv(BASE/"car201_actual_star_modulation_measurements.csv",dtype={"source_id":str})
    amber=pd.read_parquet(ROOT/"car201_data/ambers_photometry/photometry-all-star-epochs.parquet")
    amber=amber.loc[amber["sca"]==DETECTOR].copy()
    epoch_map={f"r010430300{i}00":f"ep3_pass{i}" for i in range(1,7)}
    epoch_map.update({f"r010430400{i}00":f"ep4_pass{i}" for i in range(1,6)})
    amber["pass"]=amber["epoch"].map(epoch_map)
    amber=amber.loc[amber["pass"].notna()].copy()
    keys=["source_id","filter","pass"]
    common=zane.loc[zane["likelihood_used"]].merge(amber,on=keys,suffixes=("_zane","_amber"),validate="one_to_one")
    valid_amber=(np.isfinite(common["mag_obs_amber"])&np.isfinite(common["mag_err_amber"])
                 &(common["mag_err_amber"]>0)&np.isfinite(common["x_pixel_amber"])
                 &np.isfinite(common["y_pixel_amber"]))
    common=common.loc[valid_amber].copy()
    # Use precisely the same repeated source/filter groups for both reductions.
    count=common.groupby(["source_id","filter"])["pass"].transform("size")
    common=common.loc[count>=2].copy()
    ids=sorted(common["source_id"].unique())
    source_lookup={s:i for i,s in enumerate(ids)}
    filter_lookup={f:i for i,f in enumerate(FILTERS)}
    pass_lookup={p:i for i,p in enumerate(PASSES)}
    shape=(len(ids),len(FILTERS),len(PASSES))
    zmag,amag,zerr,aerr,zpred=(np.full(shape,np.nan) for _ in range(5))
    valid=np.zeros(shape,bool)
    common["mjd_amber"]=common["mjd"]
    for row in common.rename(columns={"pass":"pass_name"}).itertuples(index=False):
        index=(source_lookup[row.source_id],filter_lookup[row.filter],pass_lookup[row.pass_name])
        zmag[index],amag[index]=row.mag_obs_zane,row.mag_obs_amber
        zerr[index],aerr[index]=row.mag_err_zane,row.mag_err_amber
        zpred[index]=row.fitted_delta_mag_including_zeropoint
        valid[index]=True
    frames=[]
    for epoch in PASSES:
        rows=common.loc[common["pass"]==epoch]
        frame=rows[["source_id","filter","mjd_amber","x_pixel_amber","y_pixel_amber"]].rename(
            columns={"mjd_amber":"mjd","x_pixel_amber":"x_pixel","y_pixel_amber":"y_pixel"})
        frames.append(frame)
    sed=workflow._load_successful_seds(((workflow.FAINT_SED_PATH,"faint_v2"),),ids)
    if len(sed)!=len(ids): raise ValueError("Common fit source lacks a successful faint-v2 SED")
    wave,area=workflow._make_wavelength_grid(5.)
    reference=Time(fitter.ICE_REFERENCE_UTC,format="isot",scale="utc").mjd
    frozen,model=workflow._predict_detector(DETECTOR,frames,ids,sed,workflow.SED_GRID_PATH,
                                           BASE,wave,area,reference,return_model=True)
    floor_dir=BASE.with_name(BASE.name+"_fitted_floor")
    floor_summary=json.loads((floor_dir/"car201_actual_star_modulation_fit_summary.json").read_text())
    floor_coefficients,floor_zero_points=workflow._load_fit_parameters(DETECTOR,floor_dir,floor_summary)
    frozen_floor=prediction(model,floor_coefficients,floor_zero_points,
                            floor_summary["modulation"]["zeropoint_model"]["group_names"])
    model.entropy_weight=1000.
    model.phase_smoothness_weight=100.
    common.to_csv(OUTPUT/"common_measurements.csv",index=False)
    print(f"Common sample: {len(ids)} stars, {valid.sum()} measurements; fitting Amber floor",flush=True)
    fit_result=fitter.fit_actual_photometry_with_profiled_systematic_floor(
        model,amag,aerr,valid,.30,1.,100.)
    fit,chi2,obs,sigma,mask,nobs,noffsets,floor,nll=fit_result
    if not fit.success: raise RuntimeError(f"Amber optimizer failed: {fit.message}")
    refit_response=model.modeled_log_response(fit.x[:2*model.mode_count])
    zpts=fit.x[2*model.mode_count:].reshape((len(FILTERS),model.zero_point_group_count))
    for pi,gid in enumerate(model.pass_zero_point_group_ids):
        if gid: refit_response[:,:,pi]+=zpts[:,gid-1][None,:]
    refitted=-refit_response/fitter.LOG_FLUX_PER_MAG
    variants={"Zane: original frozen fit":(zmag,zpred,zerr),
              "Amber: original frozen fit":(amag,frozen,aerr),
              "Amber: frozen Zane fitted-floor fit":(amag,frozen_floor,aerr),
              "Amber: own fitted-floor refit":(amag,refitted,aerr)}
    metrics=summarize_metrics(variants,valid,common,ids)
    metrics.to_csv(OUTPUT/"residual_metrics.csv",index=False)
    params=pd.DataFrame({"mode":model.basis_names,"fit_log_amplitude":fit.x[:model.mode_count],
                         "fit_phase_rad":fit.x[model.mode_count:2*model.mode_count]})
    params.to_csv(OUTPUT/"amber_modulation_parameters.csv",index=False)
    pd.DataFrame({"filter":FILTERS,"fit_log_flux_offset":zpts[:,0],
                  "equivalent_magnitude_offset_mmag":-1000*zpts[:,0]/fitter.LOG_FLUX_PER_MAG}).to_csv(
                      OUTPUT/"amber_roll_zero_points.csv",index=False)
    summary={"detector":DETECTOR,"matched_stars":len(ids),"matched_observations":int(valid.sum()),
             "systematic_floor_fitted":True,"systematic_floor_mmag":floor,
             "chi2":chi2,"degrees_of_freedom":nobs-noffsets-len(fit.x)-1,
             "reduced_chi2":chi2/(nobs-noffsets-len(fit.x)-1),"profile_nll":nll,
             "success":bool(fit.success),"message":str(fit.message),
             "selection":"Common Amber/Zane finite positive-error measurements passing Zane's saved likelihood mask; Amber has no native DQ column; no extra dithers/SED-quality cut",
             "amber_input":str(ROOT/"car201_data/ambers_photometry/photometry-all-star-epochs.parquet"),
             "native_magnitude_kind":"total corrected; ee_applied=True; used as supplied",
             "nov6_visit_included":False,"basis":"per-gap","maxent_weight":1000.,"phase_weight":100.,
             "sed_policy":"same faint-v2 successful posteriors as Zane fitting cohort; not Amber sed_catalogue labels",
             "amplitude_bound":.30,"phase_bound_rad":1.,"systematic_floor_max_mmag":100.,
             "frozen_model_source":str(BASE),"frozen_floor_model_source":str(floor_dir)}
    (OUTPUT/"comparison_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(OUTPUT/"predictions.npz",source_ids=np.asarray(ids),valid=valid,
                        zane_magnitude=zmag,amber_magnitude=amag,zane_error=zerr,amber_error=aerr,
                        zane_frozen=zpred,amber_frozen=frozen,amber_frozen_floor=frozen_floor,
                        amber_refitted_floor=refitted,fit_parameters=fit.x)
    plot_diagnostics(common,variants,(zmag,amag,zerr,aerr),valid,OUTPUT)
    print(json.dumps(summary,indent=2),flush=True)
    print(metrics.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
