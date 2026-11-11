"""WFI01 Amber 12-visit sensitivity fit using the Nov 9 fixed ice histories."""

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

import run_amber_student_t_detectors as run
from experiment_support_aligned_endpoints import solve
from support_aligned_modulation import configure_basis, effective_area_support
from check_car201_star_modulated_bundle import _weighted_center

DETECTOR="WFI01"
HISTORY=run.workflow.PRODUCTION_ROOT.parents[1]/"20261109/corrected_statistics_4h_thinned/production/wfi01/relative_response/fits"
# PRODUCTION_ROOT is .../20261106/corrected_statistics_4h_thinned/production.
HISTORY=run.ROOT/"20261109/corrected_statistics_4h_thinned/production/wfi01/relative_response/fits"
PASSES=run.PASSES+("ep4_pass6",)
GROUPS=np.array([0,0,0,0,0,0,0,1,1,1,1,1])


def main():
    from run_amber_support_aligned_production import directory
    area,_=run.canonical_configuration(DETECTOR)
    baseline=directory(DETECTOR)
    output=baseline.with_name(baseline.name+"_amber_12epochs_nov09_ice_test")
    output.mkdir(parents=True,exist_ok=True)
    old_summary=json.loads((baseline/"fit_summary.json").read_text())
    with np.load(baseline/"predictions.npz",allow_pickle=False) as a:
        theta=a["parameters"]
        old_prediction=a["prediction_mag"]
        old_valid=a["valid"]
    old_dir=run.paths(DETECTOR)[1].with_name(run.paths(DETECTOR)[1].name+"_sca01_effective_area")
    old_common=pd.read_csv(old_dir/"common_measurements.csv",dtype={"source_id":str})
    old_common=old_common.rename(columns={"mag_obs_amber":"magnitude","mag_err_amber":"error",
        "x_pixel_amber":"x_pixel","y_pixel_amber":"y_pixel","mjd_amber":"mjd_visit"})
    cols=["source_id","filter","pass","magnitude","error","x_pixel","y_pixel","mjd_visit"]
    old_common=old_common[cols].copy()
    original_keys=old_common[["source_id","filter"]].drop_duplicates()
    amber=pd.read_parquet(run.AMBER)
    extra=amber.loc[(amber["sca"]==DETECTOR)&(amber["epoch"]=="r010430400600")].copy()
    extra=extra.merge(original_keys,on=["source_id","filter"],validate="many_to_one")
    mask=(np.isfinite(extra.mag_obs)&np.isfinite(extra.mag_err)&(extra.mag_err>0)
          &np.isfinite(extra.x_pixel)&np.isfinite(extra.y_pixel)
          &(extra.x_pixel>=0)&(extra.x_pixel<4096)&(extra.y_pixel>=0)&(extra.y_pixel<4096))
    extra=extra.loc[mask].copy()
    extra["pass"]="ep4_pass6"
    extra=extra.rename(columns={"mag_obs":"magnitude","mag_err":"error","mjd":"mjd_visit"})[cols]
    allrows=pd.concat([old_common,extra],ignore_index=True)
    ids=sorted(old_common.source_id.unique())
    si={v:i for i,v in enumerate(ids)}; fi={v:i for i,v in enumerate(run.FILTERS)}; pi={v:i for i,v in enumerate(PASSES)}
    shape=(len(ids),len(run.FILTERS),len(PASSES))
    mag=np.full(shape,np.nan); error=np.full(shape,np.nan); valid=np.zeros(shape,bool)
    indices=(allrows.source_id.map(si).to_numpy(int),allrows["filter"].map(fi).to_numpy(int),allrows["pass"].map(pi).to_numpy(int))
    mag[indices]=allrows.magnitude; error[indices]=allrows.error; valid[indices]=True
    np.testing.assert_array_equal(valid[:,:,:11],old_valid)
    frames=[allrows.loc[allrows["pass"]==epoch,["source_id","filter","x_pixel","y_pixel"]].copy() for epoch in PASSES]
    old_passes=run.workflow.PASSES
    run.workflow.PASSES=PASSES
    try: tiles=run.workflow._source_tile_ids(ids,frames)
    finally: run.workflow.PASSES=old_passes
    reference=Time(run.workflow.fit_model.ICE_REFERENCE_UTC,format="isot",scale="utc").mjd
    days=np.empty((len(run.FILTERS),len(PASSES)))
    for band_index,band in enumerate(run.FILTERS):
        for pass_index,epoch in enumerate(PASSES):
            days[band_index,pass_index]=allrows.loc[(allrows["filter"]==band)&(allrows["pass"]==epoch),"mjd_visit"].median()-reference
    histories=np.empty((256,len(run.FILTERS),len(PASSES)))
    history_max=[]
    for tile in range(256):
        path=HISTORY/f"tile_{tile:03d}/posterior_samples.npz"
        with np.load(path,allow_pickle=False) as archive:
            d=np.asarray(archive["observation_days"],float).reshape(-1)
            thickness=np.asarray(archive["tile_ice_thickness"],float).reshape(-1,len(d))[0]
            resets=np.asarray(archive["ice_reset_days"],float).reshape(-1)
        if days.max()>d.max() or days.min()<d.min():raise ValueError(f"History does not bracket target times: {path}")
        histories[tile]=run.workflow.mock_model.interpolate_fixed_history(d,thickness,resets,days.ravel()).reshape(days.shape)
        history_max.append(d.max())
    seds=run.workflow._load_successful_seds(((run.workflow.FAINT_SED_PATH,"faint_v2"),),ids)
    wave,area_values=run.workflow._make_wavelength_grid(5.,effective_area_path=area,full_effective_area_support=True)
    shapes,_,_=run.workflow.fit_model.reconstruct_sed_photon_shapes(ids,seds,run.workflow.SED_GRID_PATH,wave)
    optical_root=json.loads((old_dir/"fit_summary.json").read_text())["optical_state_source"]
    optical=run.workflow._optical_tile_arrays(DETECTOR,wave,Path(optical_root))
    model=run.workflow.fit_model.RealPhotometryFit(wave,area_values,shapes,histories,tiles,*optical,"per-gap",
        entropy_weight=1000.,phase_smoothness_weight=100.,pass_zero_point_group_ids=GROUPS)
    bounds=effective_area_support(area)
    configure_basis(model,bounds,free_endpoints=True)
    from fit_amber_wfi01_student_t import StudentObjective
    fixed_objective=StudentObjective(model,mag,error,valid)
    fixed_coordinates=theta/fixed_objective.coordinate_scale
    frozen_prediction=fixed_objective.prediction(fixed_coordinates)/1000.
    del fixed_objective
    print(f"WFI01: {len(ids)} existing stars; {len(extra)} additional observations from {extra.source_id.nunique()} stars",flush=True)
    result=solve(model,mag,error,valid,theta,old_summary["extra_student_t_scale_mmag"],.6,output)
    np.savez_compressed(output/"comparison_inputs.npz",source_ids=np.asarray(ids),magnitude=mag,error=error,valid=valid,
                        new_prediction=result["prediction"],old_prediction=old_prediction)
    extra.to_csv(output/"added_nov06_measurements.csv",index=False)
    # Relative-prediction comparison on the same original visits, after
    # eliminating one arbitrary constant per star/filter from both models.
    from check_amber_wfi01_photometry import profile_residual
    change=profile_residual(result["prediction"][:,:,:11],old_prediction,error[:,:,:11],old_valid)[old_valid]*1000
    metrics=[]
    for label,prediction,mask in (("11-epoch original",old_prediction,old_valid),
                                  ("12-epoch updated on original visits",result["prediction"][:,:,:11],old_valid)):
        data=mag[:,:,:11]; errors=error[:,:,:11]
        residual=profile_residual(data,prediction,errors,mask)[mask]*1000
        metrics.append({"variant":label,"observations":len(residual),"catalog_weighted_rms_mmag":float(np.sqrt(np.mean(residual**2))),
                        "p16_mmag":float(np.quantile(residual,.16)),"median_mmag":float(np.median(residual)),
                        "p84_mmag":float(np.quantile(residual,.84))})
    ep3=valid&valid[:,:,:1];ep3[:,:,0]=False
    corrected=mag-result["prediction"]
    flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*(corrected-corrected[:,:,:1]))
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    plotrows=[]
    for band_index,(band,axis) in enumerate(zip(run.FILTERS,axes.flat,strict=True)):
        axis.axhspan(-.5,.5,color="#65ac68",alpha=.15)
        for pass_index,epoch in enumerate(PASSES[1:],start=1):
            selected=ep3[:,band_index,pass_index]
            values=flux[selected,band_index,pass_index]
            if not len(values):continue
            low,med,high=np.quantile(values,[.16,.5,.84])
            date=mdates.date2num(Time(reference+days[band_index,pass_index],format="mjd",scale="utc").to_datetime())
            axis.errorbar(date,med,yerr=[[med-low],[high-med]],fmt="o",color="#d47727",capsize=2,markersize=3)
            plotrows.append({"filter":band,"pass":epoch,"stars":len(values),"p16_percent":low,"median_percent":med,"p84_percent":high})
        axis.set_title(band);axis.axhline(0,color=".5",linewidth=.7);axis.grid(alpha=.2)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"));axis.tick_params(axis="x",labelrotation=45,labelsize=8)
    fig.suptitle("WFI01 Amber: 12-epoch Student-t fit with Nov 9 ice history",y=.99)
    fig.supylabel("Corrected flux change from measured EP3 pass 1 (%)")
    fig.text(.5,.025,"Points: medians; bars: individual-star 16th–84th percentiles; green: ±0.5%.\n"
             "Nov 6 uses Amber-only measurements for previously selected star/filter groups; no matching Zane DQ flag exists.",ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.1,1,.95));fig.savefig(output/"12_epoch_corrected_photometry.png",dpi=180);plt.close(fig)
    pd.DataFrame(plotrows).to_csv(output/"12_epoch_plot_summary.csv",index=False)
    pd.DataFrame(metrics).to_csv(output/"11_vs_12_epoch_residuals.csv",index=False)
    # Anchor each source/filter level using only its original 11 measured visits.
    # This makes the Nov 6 comparison prospective for the frozen correction;
    # the 12-epoch model itself was trained on that visit.
    heldout=[]
    for label,prediction in (("frozen 11-epoch parameters on Nov 9 history",frozen_prediction),
                             ("12-epoch refit",result["prediction"])):
        raw=mag[:,:,:11]-prediction[:,:,:11]
        weight=np.where(old_valid,1/np.square(error[:,:,:11]),0.)
        denominator=weight.sum(axis=2)
        mean=np.divide(np.sum(np.where(old_valid,raw,0.)*weight,axis=2),denominator,
                       out=np.full(denominator.shape,np.nan),where=denominator>0)
        residual=(mag[:,:,11]-prediction[:,:,11]-mean)*1000
        for index,band in enumerate(run.FILTERS):
            selected=valid[:,index,11]&np.isfinite(residual[:,index])
            values=residual[selected,index]
            flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*values/1000)
            heldout.append({"variant":label,"filter":band,"observations":len(values),
                "median_residual_mmag":float(np.median(values)),"p16_residual_mmag":float(np.quantile(values,.16)),
                "p84_residual_mmag":float(np.quantile(values,.84)),
                "within_0p5_percent_fraction":float(np.mean(np.abs(flux)<=.5))})
    pd.DataFrame(heldout).to_csv(output/"nov06_residuals_11visit_stellar_baselines.csv",index=False)
    report={"detector":DETECTOR,"ice_history_root":str(HISTORY.resolve()),"stellar_input":str(run.AMBER),
        "added_visit":"r010430400600","added_visit_utc":"2026-11-06T01:01–01:48 UTC",
        "original_stars":len(ids),"original_measurements":int(old_valid.sum()),"added_stars":int(extra.source_id.nunique()),
        "added_measurements":len(extra),"total_measurements":int(valid.sum()),
        "new_visit_selection":"existing original fit star/filter group; finite Amber mag, positive error, valid coordinates; Amber has no native DQ field",
        "new_visit_zane_DQ_available":False,"ice_history_extrapolated":False,
        "last_fixed_history_utc":Time(reference+min(history_max),format="mjd",scale="utc").isot,
        "old_sigma_extra_mmag":old_summary["extra_student_t_scale_mmag"],
        "new_sigma_extra_mmag":result["summary"]["extra_student_t_scale_mmag"],
        "prediction_change_original_visits_rms_mmag":float(np.sqrt(np.mean(change**2))),
        "prediction_change_original_visits_p95_abs_mmag":float(np.quantile(np.abs(change),.95)),
        "comparison_centering":"same catalog-error weighted star/filter constant removed",
        "optimizer_success":result["summary"]["optimizer_success"],
        "endpoint_parameters_at_bounds":result["summary"]["parameters_at_bounds"],
        "production_bundle_changed":False}
    (output/"comparison_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2));print(pd.DataFrame(metrics).to_string(index=False));
    print("Nov 6 residuals with stellar levels estimated only from the original visits:");print(pd.DataFrame(heldout).to_string(index=False));print(output)


if __name__=="__main__":main()
