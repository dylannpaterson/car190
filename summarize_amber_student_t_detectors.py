"""Validate and summarize the completed Amber detector fits; no optimization."""

import json
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time

import run_amber_student_t_detectors as run
from fit_amber_wfi01_student_t import profile_student_locations
from check_amber_wfi01_photometry import profile_residual

DETECTORS=run.workflow.DETECTORS
FILTERS,PASSES=run.FILTERS,run.PASSES
OUTPUT=run.workflow.PRODUCTION_ROOT/"wavelength_response_bundle/amber_student_t_all_detectors"


def predicted_with_roll(model,parameters):
    response=model.modeled_log_response(parameters[:2*model.mode_count])
    zpt=parameters[2*model.mode_count:].reshape(len(FILTERS),model.zero_point_group_count)
    for pi,gid in enumerate(model.pass_zero_point_group_ids):
        if gid:response[:,:,pi]+=zpt[:,gid-1][None,:]
    return -response/run.workflow.LOG_FLUX_PER_MAG


def summarize_values(values):
    values=values[np.isfinite(values)]
    flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*values)
    return {"observations":len(values),"rms_mmag":1000*np.sqrt(np.mean(values**2)),
            "p16_mmag":1000*np.quantile(values,.16),"median_mmag":1000*np.median(values),
            "p84_mmag":1000*np.quantile(values,.84),
            "fraction_within_0p5_percent":np.mean(np.abs(flux)<=.5)}


def plot_all(cells,scales):
    colors=plt.get_cmap("tab20")(np.arange(len(DETECTORS)))
    lower,upper=float(cells.p16_percent.min()),float(cells.p84_percent.max())
    margin=.05*max(upper-lower,1.)
    for variant in ("roll_adjusted_no_ice","unmodulated","student_t_modulated"):
        fig,axes=plt.subplots(2,4,figsize=(20,10),sharex=True,sharey=True)
        for band,axis in zip(FILTERS,axes.flat,strict=True):
            axis.axhspan(-.5,.5,color="#65ac68",alpha=.15)
            axis.axhline(0,color="0.4",linewidth=.65)
            for index,(detector,color) in enumerate(zip(DETECTORS,colors,strict=True)):
                rows=cells.loc[(cells["variant"]==variant)&(cells["filter"]==band)&(cells["detector"]==detector)]
                axis.errorbar(rows["utc_date_number"]+(index-8.5)/96.,rows["median_percent"],
                              yerr=np.array([rows["median_percent"]-rows["p16_percent"],
                                             rows["p84_percent"]-rows["median_percent"]]),
                              fmt="o",color=color,markersize=2.5,elinewidth=.65,capsize=1.2)
            axis.set_title(band)
            axis.set_ylim(lower-margin,upper+margin)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=6))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            axis.tick_params(axis="x",labelrotation=50,labelsize=8)
            axis.grid(axis="y",alpha=.2)
        for axis in axes[-1]:axis.set_xlabel("UTC date")
        handles=[Line2D([0],[0],color=color,label=detector) for detector,color in zip(DETECTORS,colors,strict=True)]
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.945),ncol=9,frameon=False)
        title={"roll_adjusted_no_ice":"Roll-adjusted Amber data; no ice correction",
               "unmodulated":"Amber: unmodulated ice correction",
               "student_t_modulated":"Amber: Student-t fitted ice correction"}[variant]
        fig.suptitle(title,fontsize=16,y=.985)
        fig.supylabel("Relative flux change from measured EP3 pass 1 (%)")
        fig.text(.5,.025,"Points: detector medians; error bars: individual-star 16th–84th percentiles; green: ±0.5%. "
                 "Continuous UTC dates with small detector offsets.\n"
                 "Identical matched-quality sample in all panels; measured EP3-pass-1 references only. "
                 "Saved Student-t roll offsets applied consistently; unconstrained offsets fixed to zero.\n"
                 "Fits use all common repeated stars, including post-roll-only stars. All three figures share y limits.",
                 ha="center",fontsize=9)
        fig.tight_layout(rect=(.025,.12,1,.87))
        fig.savefig(OUTPUT/f"amber_{variant}_epoch_errorbars.png",dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    axes[0].hist(scales.extra_student_t_scale_mmag,bins=np.linspace(0,3.5,11),color="#3787a5",edgecolor="white")
    axes[0].set_xlabel("Fitted extra Student-t scale (mmag)")
    axes[0].set_ylabel("Detector count")
    axes[1].bar(scales.detector,scales.extra_student_t_scale_mmag,color="#3787a5")
    axes[1].set_ylabel("Fitted extra Student-t scale (mmag)")
    axes[1].tick_params(axis="x",labelrotation=60)
    fig.suptitle("Amber photometry: Student-t scales, ν=4 (scale is not RMS)")
    fig.savefig(OUTPUT/"amber_student_t_scales.png",dpi=180)
    plt.close(fig)


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-effective-areas",action="store_true",help="Summarize the canonical per-detector-area fits")
    parser.add_argument("--support-aligned-endpoints",action="store_true",help="Summarize support-aligned free-endpoint fits")
    parser.add_argument("--strong-red-led-mask",action="store_true",help="Summarize strong-red LED-mask production fits")
    args=parser.parse_args()
    if args.strong_red_led_mask:
        OUTPUT=OUTPUT.with_name("amber_student_t_strong_red_led_mask")
    elif args.support_aligned_endpoints:
        OUTPUT=OUTPUT.with_name("amber_student_t_support_aligned_free_endpoints")
    elif args.detector_effective_areas:
        OUTPUT=OUTPUT.with_name("amber_student_t_detector_effective_areas")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    detector_records,metric_records,cell_records,observation_frames=[],[],[],[]
    for detector in DETECTORS:
        old,directory=run.paths(detector)
        area=None
        if args.strong_red_led_mask:
            from run_amber_strong_red_production import directory as masked_directory
            area,_=run.canonical_configuration(detector)
            directory=masked_directory(detector)
        elif args.support_aligned_endpoints:
            from run_amber_support_aligned_production import directory as support_directory
            area,_=run.canonical_configuration(detector)
            directory=support_directory(detector)
        elif args.detector_effective_areas:
            area,directory=run.canonical_configuration(detector)
        if not (args.support_aligned_endpoints or args.strong_red_led_mask) and not run.successful(detector,effective_area_path=area,output_dir=directory):raise RuntimeError(f"Missing converged {detector} result")
        summary=json.loads((directory/("production_fit_summary.json" if args.strong_red_led_mask else "fit_summary.json")).read_text())
        if not summary.get("optimizer_success"):raise RuntimeError(f"Missing converged {detector} result")
        with np.load(directory/"predictions.npz",allow_pickle=False) as archive:
            stored={key:archive[key] for key in archive.files}
        if args.strong_red_led_mask:
            with np.load(directory/"production_predictions.npz",allow_pickle=False) as archive:
                stored.update({key:archive[key] for key in archive.files})
        # Reconstruct the fixed inputs once to evaluate the unmodulated optical
        # baseline and independently verify the saved Student-t prediction.
        _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare(
            detector,write_inputs=False,effective_area_path=area,output_dir=directory,
            full_effective_area_support=args.support_aligned_endpoints or args.strong_red_led_mask)
        if args.support_aligned_endpoints or args.strong_red_led_mask:
            from support_aligned_modulation import configure_basis,effective_area_support
            configure_basis(model,effective_area_support(area),free_endpoints=True)
            if args.strong_red_led_mask:
                from experiment_led_sed_soft_mask import sed_mask
                model.basis*=sed_mask(model.wavelengths_nm,red_exponent=.05)[None,:]
            stored["student_t_fit_parameters"]=stored["parameters"]
            stored["student_t_prediction_mag"]=stored["prediction_mag"]
            stored["active_groups"]=valid.reshape(-1,valid.shape[-1]).sum(axis=1)>=2
        if ("source_ids" in stored and not np.array_equal(stored["source_ids"].astype(str),np.asarray(ids))) or not np.array_equal(stored["valid"],valid):
            raise ValueError(f"Saved source/measurement selection changed for {detector}")
        parameters=stored["student_t_fit_parameters"]
        predicted=predicted_with_roll(model,parameters)
        check=float(np.max(np.abs(predicted[valid]-stored["student_t_prediction_mag"][valid])))
        if check>1e-9:raise RuntimeError(f"{detector}: saved prediction disagreement {check}")
        unmod_parameters=parameters.copy()
        unmod_parameters[:2*model.mode_count]=0.
        unmod=predicted_with_roll(model,unmod_parameters)
        zpt=parameters[2*model.mode_count:].reshape(len(FILTERS),model.zero_point_group_count)
        roll=np.zeros(predicted.shape)
        for pi,gid in enumerate(model.pass_zero_point_group_ids):
            if gid:roll[:,:,pi]=-zpt[:,gid-1][None,:]/run.workflow.LOG_FLUX_PER_MAG
        active=stored["active_groups"]
        robust_residual=stored["robust_residuals_mmag"]
        robust_mask=valid.reshape(-1,valid.shape[-1])[active]
        robust_values=robust_residual[robust_mask]/1000.
        # Independently evaluate the frozen original correction with the same
        # robust location rule and the same Student-t scale used for the new fit.
        y=(magnitude-frozen).reshape(-1,valid.shape[-1])[active]*1000.
        scales=np.hypot(error.reshape(-1,valid.shape[-1])[active]*1000.,summary["extra_student_t_scale_mmag"])
        _,frozen_residual,_=profile_student_locations(y,scales,robust_mask)
        new_metrics=summarize_values(robust_values)
        old_metrics=summarize_values(frozen_residual[robust_mask]/1000.)
        change=profile_residual(predicted,frozen,error,valid)[valid]*1000.
        detector_records.append({"detector":detector,"used_stars":len(ids),"used_observations":int(valid.sum()),
                                 "extra_student_t_scale_mmag":summary["extra_student_t_scale_mmag"],
                                 "optimizer_success":summary["optimizer_success"],"iterations":summary["iterations"],
                                 "max_saved_prediction_error_mag":check,
                                 "frozen_original_fraction_within_0p5_percent":old_metrics["fraction_within_0p5_percent"],
                                 "student_t_fraction_within_0p5_percent":new_metrics["fraction_within_0p5_percent"],
                                 "frozen_original_robust_residual_rms_mmag":old_metrics["rms_mmag"],
                                 "student_t_robust_residual_rms_mmag":new_metrics["rms_mmag"],
                                 "prediction_change_rms_mmag":np.sqrt(np.mean(change**2)),
                                 "max_abs_location_score":summary.get("max_abs_profile_location_score_per_mmag",summary.get("max_profile_location_score"))})
        si={source_id:i for i,source_id in enumerate(ids)}
        fi={band:i for i,band in enumerate(FILTERS)}
        pi={epoch:i for i,epoch in enumerate(PASSES)}
        indices=(common["source_id"].map(si).to_numpy(int),common["filter"].map(fi).to_numpy(int),common["pass"].map(pi).to_numpy(int))
        records=common[["source_id","filter","pass","mjd_amber","mag_obs_amber","mag_err_amber"]].copy()
        records["detector"]=detector
        records["student_t_prediction_mag"]=predicted[indices]
        records["unmodulated_prediction_mag"]=unmod[indices]
        records["roll_offset_mag"]=roll[indices]
        records["mag_roll_adjusted_no_ice"]=magnitude[indices]-roll[indices]
        records["mag_corrected_unmodulated"]=magnitude[indices]-unmod[indices]
        records["mag_corrected_student_t"]=magnitude[indices]-predicted[indices]
        records["frozen_zane_prediction_mag"]=frozen[indices]
        observation_frames.append(records)
        reference=valid[:,:,:1]
        plot_valid=valid&reference
        plot_valid[:,:,0]=False
        variants={"roll_adjusted_no_ice":magnitude-roll,
                  "unmodulated":magnitude-unmod,
                  "student_t_modulated":magnitude-predicted}
        for variant,values in variants.items():
            delta=values-values[:,:,:1]
            metric_records.append({"detector":detector,"variant":variant,
                                   "reference":"measured EP3 pass 1; reference measurements excluded",
                                   **summarize_values(delta[plot_valid])})
            flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*delta)
            for band_index,band in enumerate(FILTERS):
                for epoch_index,epoch in enumerate(PASSES[1:],start=1):
                    mask=plot_valid[:,band_index,epoch_index]
                    if not np.any(mask):continue
                    values_here=flux[mask,band_index,epoch_index]
                    lo,med,hi=np.quantile(values_here,[.16,.5,.84])
                    matched=common.loc[(common["filter"]==band)&(common["pass"]==epoch)]
                    date=float(mdates.date2num(Time(float(matched["mjd_amber"].median()),format="mjd",scale="utc").to_datetime()))
                    cell_records.append({"detector":detector,"filter":band,"pass":epoch,"variant":variant,
                                         "matched_stars":int(mask.sum()),"p16_percent":lo,"median_percent":med,
                                         "p84_percent":hi,"utc_date_number":date})
        print(f"{detector}: validated saved predictions to {check:.3g} mag",flush=True)
    scales=pd.DataFrame(detector_records)
    cells=pd.DataFrame(cell_records)
    scales.to_csv(OUTPUT/"amber_student_t_detector_summary.csv",index=False)
    pd.DataFrame(metric_records).to_csv(OUTPUT/"amber_student_t_ep3_reference_metrics.csv",index=False)
    cells.to_csv(OUTPUT/"amber_student_t_epoch_percentiles.csv",index=False)
    pd.concat(observation_frames,ignore_index=True).to_csv(OUTPUT/"amber_all_detector_corrected_magnitudes.csv",index=False)
    plot_all(cells,scales)
    (OUTPUT/"README.md").write_text("""# Amber Student-t detector fits

All 18 detectors use fixed Student-t nu=4, one fitted extra scale per detector,
and robust per-star/filter constant brightness levels. The scale is not RMS.

Selection is the same controlled comparison used for the initial WFI01 test:
Amber counterparts of Zane's saved likelihood-used measurements, finite positive
Amber errors, and at least two repeated common measurements per star/filter.
Amber's catalog has no native DQ column; the matching inherits Zane's saved DQ,
saturation and outlier selection. No additional clipping is performed. Successful
faint-v2 SEDs are retained as in the Zane fits. Amber's magnitudes have their
supplied encircled-energy correction; no additional EE correction is applied.
The first 11 visits through Oct 31 are used; Nov 6 is not included.

The LED ice histories and optical states are fixed. Modulation is detector-wide,
with the existing amplitude and phase regularization weights 1000 and 100. Roll
offsets without cross-roll source support are fixed to zero, not inferred.

WFI01 began as the verified single-detector experiment. The detector runs use
the same likelihood with a cached analytic derivative of the identical optical
forward calculation. Derivatives are checked numerically; saved predictions are
independently reconstructed with the original forward code.

Plots show a subset with measured EP3-pass-1 references; no model baseline is
invented for post-roll-only stars. Each point is a detector median with star
16th/84th percentile bars. All three plots use identical samples and y limits.
The fits themselves include post-roll-only repeated sources. The corrected
magnitude CSV includes all common fit measurements.

Detector-summary residual fractions compare the original frozen correction with
the Student-t refit using the same robust stellar-location rule and scale.
EP3-reference metrics use measured reference differences instead. These are
in-sample diagnostics, not held-out predictive validation. Existing bundles and
the S3 upload stage are handled by the separate bundle builder and stager.
"""+("\nThese diagnostics use support-aligned free endpoints (±0.60 log amplitude) with all LED anchors fixed.\n"
       if args.support_aligned_endpoints or args.strong_red_led_mask else "")+("\nThese fits suppress modulation using the measured LED spectra, with exponent 0.05 for LEDs 5 and 6.\n"
       if args.strong_red_led_mask else "")+("\nThese canonical diagnostics use each detector's own SCA01–SCA18 effective-area curve. "
       "Selected files and SHA256 hashes are recorded in the source fit summaries. "
       "The frozen original coefficients are evaluated with the same detector-specific weights for this diagnostic.\n"
       if args.detector_effective_areas or args.support_aligned_endpoints or args.strong_red_led_mask else "\nThese historical diagnostics use the shared SCA05 reference effective-area curve.\n"))
    print(scales.to_string(index=False))
    print("Output directory:",OUTPUT)


if __name__=="__main__":
    main()
