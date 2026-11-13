"""Compare per-tile modulated ice histories with the fixed-ice joint fit."""
import argparse
import json
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

import run_amber_student_t_detectors as run
from experiment_joint_wfi01_fixed_ice import add_full_basis,fixed_output
from experiment_joint_wfi01_weekly_ice import RESETS
from fit_amber_wfi01_student_t import profile_student_locations
from car201_photometry_forward_jacobian import CachedPhotometryForward
from support_aligned_modulation import effective_area_support
from check_amber_wfi01_photometry import profile_residual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",required=True,choices=run.workflow.DETECTORS)
    parser.add_argument("--fit-dir",type=Path,required=True)
    args=parser.parse_args()
    detector=args.detector
    fit_dir=args.fit_dir
    fit_summary=json.loads((fit_dir/"summary.json").read_text())
    cadence=fit_summary["rate_control_spacing_days"]
    ice_label=f"{cadence:g}-day per-tile ice"
    area,_=run.canonical_configuration(detector)
    _,common,ids,magnitude,error,valid,_,model,_=run.prepare(detector,write_inputs=False,
        effective_area_path=area,output_dir=fit_dir,full_effective_area_support=True)
    add_full_basis(model,effective_area_support(area))
    with np.load(fit_dir/"joint_weekly_ice_fit.npz",allow_pickle=False) as a:
        times=a["times"];baseline=a["baseline_thickness_nm"];thickness=a["fitted_thickness_nm"];theta=a["shared_parameters"]
    fixed=fixed_output(detector)
    with np.load(fixed/"joint_fit.npz",allow_pickle=False) as a:
        old=a["stellar_prediction_mag"];np.testing.assert_array_equal(a["stellar_valid"],valid)
    ref=Time(run.workflow.fit_model.ICE_REFERENCE_UTC,format="isot",scale="utc").mjd
    days=np.array([[common.loc[(common["filter"]==band)&(common["pass"]==epoch),"mjd_amber"].median()-ref
                    for epoch in run.PASSES] for band in run.FILTERS])
    indices=np.searchsorted(times,days)
    np.testing.assert_allclose(times[indices],days,atol=1e-10)
    model.tile_thickness_by_id=thickness[:,indices]
    forward=CachedPhotometryForward(model);count=model.mode_count
    response,_=forward.evaluate(theta[:2*count])
    roll=theta[2*count:].reshape(len(run.FILTERS),model.zero_point_group_count)
    for pi,gid in enumerate(model.pass_zero_point_group_ids):
        if gid:response[:,:,pi]+=roll[:,gid-1][None,:]
    new=-response/run.workflow.LOG_FLUX_PER_MAG
    active=valid.reshape(-1,valid.shape[-1]).sum(axis=1)>=2
    mask=valid.reshape(-1,valid.shape[-1])[active]
    old_scale=json.loads((fixed/"joint_fit_summary.json").read_text())["extra_student_t_scale_mmag"]
    scales=np.hypot(error.reshape(-1,valid.shape[-1])[active]*1000,old_scale)
    metrics=[]
    for label,prediction in (("fixed ice",old),(ice_label,new)):
        y=(magnitude-prediction).reshape(-1,valid.shape[-1])[active]*1000
        _,residual,_=profile_student_locations(y,scales,mask)
        values=residual[mask];flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*values/1000)
        metrics.append({"model":label,"observations":len(values),"residual_rms_mmag":np.sqrt(np.mean(values**2)),
                        "p16_mmag":np.quantile(values,.16),"median_mmag":np.median(values),"p84_mmag":np.quantile(values,.84),
                        "fraction_within_0p5_percent":np.mean(np.abs(flux)<=.5)})
    change=profile_residual(new,old,error,valid)[valid]*1000
    metrics_table=pd.DataFrame(metrics);metrics_table.to_csv(fit_dir/"fixed_vs_ice_stellar_metrics.csv",index=False)
    delta=thickness-baseline
    stats=np.percentile(delta,[16,50,84],axis=0)
    dates=Time(ref+times,format="mjd",scale="utc").to_datetime()
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for stage in range(3):
        selected=np.searchsorted(RESETS,times,side="right")==stage
        axes[0].fill_between(dates[selected],stats[0,selected],stats[2,selected],color="#357fa1",alpha=.2)
        axes[0].plot(np.asarray(dates)[selected],stats[1,selected],color="#357fa1",linewidth=1.7)
        for tile in (0,89,136,255):axes[1].plot(np.asarray(dates)[selected],delta[tile,selected],label=f"Tile {tile}" if stage==0 else None,linewidth=1.2)
    for axis in axes:
        axis.axhline(0,color=".4",linewidth=.7);axis.grid(alpha=.2)
        for day in RESETS:axis.axvline(Time(ref+day,format="mjd",scale="utc").to_datetime(),color=".5",linestyle=":")
        for event in ("2026-10-17T03:30:00","2026-10-17T23:00:00","2026-10-30T01:43:00"):
            axis.axvline(Time(event,format="isot",scale="utc").to_datetime(),color="#b66b2e",linestyle="--",linewidth=.7)
    axes[0].set_ylabel("Ice thickness change (nm)");axes[1].set_ylabel("Selected tile changes (nm)")
    axes[1].set_xlabel("UTC date");axes[1].legend(ncol=4)
    fig.suptitle(f"{detector} joint fit: {ice_label} history adjustments")
    fig.text(.5,.015,"Top: median and 16th–84th percentile across 256 tiles. Bottom: selected individual tiles. "
             "Dotted: reset boundaries; dashed: confirmed temperature events.",ha="center",fontsize=9)
    fig.tight_layout(rect=(.02,.055,1,.96));fig.savefig(fit_dir/"ice_history_changes.png",dpi=180);plt.close(fig)
    result={"prediction_change_rms_mmag":float(np.sqrt(np.mean(change**2))),
            "prediction_change_p95_abs_mmag":float(np.quantile(np.abs(change),.95)),
            "stellar_residual_normalization":"same Student-t location rule and fixed-ice fitted scale for both models",
            "thickness_change_p16_median_p84_nm":np.percentile(delta,[16,50,84]).tolist(),
            "max_abs_thickness_change_nm":float(np.max(np.abs(delta))),
             "model_comparison":f"per-tile {cadence:g}-day rate controls compared against detector-wide fixed-ice joint fit"}
    (fit_dir/"comparison_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2));print(metrics_table.to_string(index=False))


if __name__=="__main__":main()
