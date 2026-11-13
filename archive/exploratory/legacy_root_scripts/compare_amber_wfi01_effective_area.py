"""Isolate fixed-parameter and refitted effects of SCA01 versus SCA05 areas."""

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

import run_amber_student_t_detectors as run
from check_amber_wfi01_photometry import profile_residual
from fit_amber_wfi01_student_t import profile_student_locations

BASE=run.paths("WFI01")[1]
NEW=BASE.with_name(BASE.name+"_sca01_effective_area")
CONTROL=BASE.with_name(BASE.name+"_sca05_control")
AREA=run.ROOT/"external_reference_data/roman_effarea_v8_20240301/Roman_effarea_v8_SCA01_20240301.ecsv"


def load(path):
    with np.load(path/"predictions.npz",allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def model_prediction(model,parameters):
    response=model.modeled_log_response(parameters[:2*model.mode_count])
    offsets=parameters[2*model.mode_count:].reshape(len(run.FILTERS),model.zero_point_group_count)
    for pi,gid in enumerate(model.pass_zero_point_group_ids):
        if gid:response[:,:,pi]+=offsets[:,gid-1][None,:]
    return -response/run.workflow.LOG_FLUX_PER_MAG


def main():
    old,control,new=load(BASE),load(CONTROL),load(NEW)
    for candidate in (old,control):
        np.testing.assert_array_equal(candidate["source_ids"],new["source_ids"])
        np.testing.assert_array_equal(candidate["valid"],new["valid"])
    np.testing.assert_array_equal(control["magnitude"],new["magnitude"])
    np.testing.assert_array_equal(control["error"],new["error"])
    old_summary=json.loads((CONTROL/"fit_summary.json").read_text())
    new_summary=json.loads((NEW/"fit_summary.json").read_text())
    _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare(
        "WFI01",write_inputs=False,effective_area_path=AREA,output_dir=NEW)
    np.testing.assert_array_equal(new["source_ids"].astype(str),np.asarray(ids))
    # Rebuild the final predictions independently before evaluating the
    # counterfactual that changes only the effective area, keeping coefficients.
    np.testing.assert_allclose(model_prediction(model,new["student_t_fit_parameters"])[valid],
                               new["student_t_prediction_mag"][valid],atol=1e-11,rtol=0.)
    fixed_sca01=model_prediction(model,control["student_t_fit_parameters"])
    predictions={"SCA05 baseline":control["student_t_prediction_mag"],
                 "SCA01 fixed coefficients":fixed_sca01,
                 "SCA01 refit":new["student_t_prediction_mag"]}
    active=valid.reshape(-1,valid.shape[-1]).sum(axis=1)>=2
    group_valid=valid.reshape(-1,valid.shape[-1])[active]
    group_error=error.reshape(-1,valid.shape[-1])[active]*1000
    scale=np.hypot(group_error,old_summary["extra_student_t_scale_mmag"])
    metrics=[]
    filter_changes=[]
    corrected={}
    primary=valid&valid[:,:,:1]
    primary[:,:,0]=False
    for name,prediction in predictions.items():
        values=(magnitude-prediction).reshape(-1,valid.shape[-1])[active]*1000
        _,residual,_=profile_student_locations(values,scale,group_valid)
        r=residual[group_valid]
        flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*r/1000)
        corrected[name]=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*((magnitude-prediction)
                                      -(magnitude-prediction)[:,:,:1]))
        metrics.append({"variant":name,"observations":int(valid.sum()),
                        "common_robust_centering_rms_mmag":np.sqrt(np.mean(r*r)),
                        "common_robust_centering_p16_mmag":np.quantile(r,.16),
                        "common_robust_centering_median_mmag":np.median(r),
                        "common_robust_centering_p84_mmag":np.quantile(r,.84),
                        "common_robust_centering_fraction_within_0p5_percent":np.mean(np.abs(flux)<=.5),
                        "measured_EP3_reference_fraction_within_0p5_percent":np.mean(np.abs(corrected[name][primary])<=.5)})
        if name=="SCA05 baseline":continue
        change=profile_residual(prediction,predictions["SCA05 baseline"],error,valid)*1000.
        for fi,band in enumerate(run.FILTERS):
            x=change[:,fi][valid[:,fi]]
            filter_changes.append({"filter":band,"variant":name,
                                   "prediction_change_rms_mmag":np.sqrt(np.mean(x*x)),
                                   "prediction_change_p95_absolute_mmag":np.quantile(np.abs(x),.95),
                                   "prediction_change_max_absolute_mmag":np.max(np.abs(x))})
    def change_stats(prediction,reference):
        x=profile_residual(prediction,reference,error,valid)[valid]*1000.
        return {"rms_mmag":float(np.sqrt(np.mean(x*x))),"p95_absolute_mmag":float(np.quantile(np.abs(x),.95)),
                "max_absolute_mmag":float(np.max(np.abs(x)))}
    report={"detector":"WFI01","stars":len(ids),"observations":int(valid.sum()),
            "same_measurements_and_errors":True,
            "extra_student_t_scale_sca05_mmag":old_summary["extra_student_t_scale_mmag"],
            "extra_student_t_scale_sca01_mmag":new_summary["extra_student_t_scale_mmag"],
            "effective_area_only_fixed_coefficients":change_stats(fixed_sca01,predictions["SCA05 baseline"]),
            "effective_area_and_refit":change_stats(new["student_t_prediction_mag"],predictions["SCA05 baseline"]),
            "optimizer_control_restart_vs_previous_fit":change_stats(control["student_t_prediction_mag"],old["student_t_prediction_mag"]),
            "sca01_objective_minus_sca05_objective":new_summary["objective"]-old_summary["objective"],
            "prediction_comparison_reference":"same per-star/filter catalog-error weighted constant removed; changes describe time-dependent predictions",
            "input_sca01":new_summary["effective_area_file"],"input_sca05":old_summary["effective_area_file"],
            "sca01_sha256":new_summary["effective_area_sha256"],"sca05_sha256":old_summary["effective_area_sha256"]}
    (NEW/"effective_area_comparison_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    pd.DataFrame(metrics).to_csv(NEW/"effective_area_comparison_residuals.csv",index=False)
    pd.DataFrame(filter_changes).to_csv(NEW/"effective_area_comparison_by_filter.csv",index=False)
    np.savez_compressed(NEW/"effective_area_fixed_coefficient_predictions.npz",prediction_mag=fixed_sca01,valid=valid,source_ids=np.asarray(ids))
    params=pd.read_csv(CONTROL/"student_t_modulation_parameters.csv").merge(
        pd.read_csv(NEW/"student_t_modulation_parameters.csv"),on="mode",suffixes=("_sca05","_sca01"),validate="one_to_one")
    params.to_csv(NEW/"effective_area_modulation_parameter_comparison.csv",index=False)
    dates=mdates.date2num(Time(common.groupby("pass")["mjd_amber"].median().reindex(run.PASSES).to_numpy(),
                              format="mjd",scale="utc").to_datetime())
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    colors=("#555555","#357fa1","#d47d2d")
    for fi,(band,axis) in enumerate(zip(run.FILTERS,axes.flat,strict=True)):
        axis.axhspan(-.5,.5,color="#65ac68",alpha=.15)
        for vi,(name,color) in enumerate(zip(predictions,colors,strict=True)):
            for pi,date in enumerate(dates):
                x=corrected[name][:,fi,pi][primary[:,fi,pi]]
                if not len(x):continue
                low,med,high=np.quantile(x,[.16,.5,.84])
                axis.errorbar(date+(vi-1)*.065,med,yerr=[[med-low],[high-med]],color=color,fmt="o",
                              capsize=1.5,markersize=3,label=name if pi==1 else None)
        axis.axhline(0,color="0.5",linewidth=.6)
        axis.set_title(band)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x",labelrotation=45,labelsize=8)
        axis.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.94),ncol=3)
    fig.suptitle("WFI01 Amber Student-t: effect of detector-specific effective area",fontsize=15,y=.99)
    fig.supylabel("Corrected flux change from measured EP3 pass 1 (%)")
    fig.text(.5,.025,"Identical stars, measurements, errors and fit settings. SCA01 fixed coefficients changes only the spectral weighting; SCA01 refit re-optimizes the same parameters.\n"
             "Points: star medians; bars: 16th–84th percentiles. Cumulative plot requires measured EP3 references; full fits use all common repeated sources.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.12,1,.88))
    fig.savefig(NEW/"sca01_vs_sca05_corrections.png",dpi=180)
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(filter_changes).to_string(index=False))


if __name__=="__main__":
    main()
