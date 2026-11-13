"""Nested free-LED/edge modulation sensitivity experiment for one detector."""

import argparse
import json
from pathlib import Path
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from astropy.time import Time

import run_amber_student_t_detectors as run
import mock_fit_wfi_star_reflectance_modulation as mock
from fit_amber_wfi01_student_t import StudentObjective, profile_student_locations, NU
from check_amber_wfi01_photometry import profile_residual


def free_value_basis(wavelengths,boundaries=(400.,2321.)):
    """C1 nodal modes at both edges and all six LEDs, clamped outside edges."""
    wavelengths=np.asarray(wavelengths,float)
    if not boundaries[0]<np.min(mock.LED_CENTERS_NM) or not boundaries[1]>np.max(mock.LED_CENTERS_NM):
        raise ValueError("Modulation boundaries must enclose all LED wavelengths")
    knots=np.concatenate(([boundaries[0]],mock.LED_CENTERS_NM,[boundaries[1]]))
    basis=np.zeros((len(knots),len(wavelengths)))
    weights=[]
    bounds=[]
    for index,center in enumerate(knots):
        left=knots[index-1] if index else center
        right=knots[index+1] if index<len(knots)-1 else center
        if index:
            mask=(wavelengths>=left)&(wavelengths<=center)
            u=(wavelengths[mask]-left)/(center-left)
            basis[index,mask]=.5*(1-np.cos(np.pi*u))
        else:
            basis[index,wavelengths<=center]=1.
        if index<len(knots)-1:
            mask=(wavelengths>=center)&(wavelengths<=right)
            u=(wavelengths[mask]-center)/(right-center)
            basis[index,mask]=.5*(1+np.cos(np.pi*u))
        else:
            basis[index,wavelengths>=center]=1.
        weights.append(.5*(right-left)/(knots[-1]-knots[0]))
        bounds.append((left,right))
    return basis,knots,np.asarray(weights),bounds


def augment_basis(model,endpoints_only=False):
    old=model.mode_count
    extra,knots,weights,bounds=free_value_basis(model.wavelengths_nm)
    if endpoints_only:
        selection=[0,len(knots)-1]
        extra,knots,weights=extra[selection],knots[selection],weights[selection]
        bounds=[bounds[index] for index in selection]
    model.basis=np.concatenate([model.basis,extra],axis=0)
    model.basis_names=list(model.basis_names)+[f"free_value_{value:.6f}nm" for value in knots]
    model.basis_centers_nm=np.concatenate([model.basis_centers_nm,knots])
    # Do not renormalize the old weights: setting all added coefficients to
    # zero must retain exactly the old prediction AND regularization objective.
    model.node_weights=np.concatenate([model.node_weights,weights])
    model.node_groups=list(model.node_groups)+[np.asarray([old+i]) for i in range(len(knots))]
    model.group_bounds=list(model.group_bounds)+bounds
    model.mode_count=len(model.basis)
    model.basis_grouping="per-gap-plus-free-endpoints" if endpoints_only else "per-gap-plus-free-led-and-edges"
    return old,len(knots)


def expand_parameters(parameters,old_modes,extra_modes):
    return np.concatenate([parameters[:old_modes],np.zeros(extra_modes),
                           parameters[old_modes:2*old_modes],np.zeros(extra_modes),parameters[2*old_modes:]])


def metrics(residual):
    values=residual[np.isfinite(residual)]
    flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*values/1000.)
    low,med,high=np.quantile(values,[.16,.5,.84])
    return {"observations":len(values),"rms_mmag":float(np.sqrt(np.mean(values**2))),
            "p16_mmag":float(low),"median_mmag":float(med),"p84_mmag":float(high),
            "fraction_within_0p5_percent":float(np.mean(np.abs(flux)<=.5))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",required=True,choices=run.workflow.DETECTORS)
    parser.add_argument("--free-endpoints-only",action="store_true",help="Keep all six LED values fixed; free only 400/2321-nm boundary values")
    parser.add_argument("--endpoint-amplitude-bound",type=float,default=.30,
                        help="Absolute log-amplitude bound for endpoint-only added modes; other bounds stay fixed")
    args=parser.parse_args()
    if not np.isfinite(args.endpoint_amplitude_bound) or args.endpoint_amplitude_bound<=0:
        parser.error("Endpoint amplitude bound must be finite and positive")
    if not args.free_endpoints_only and args.endpoint_amplitude_bound!=.30:
        parser.error("--endpoint-amplitude-bound requires --free-endpoints-only")
    area,baseline=run.canonical_configuration(args.detector)
    if not run.successful(args.detector,effective_area_path=area,output_dir=baseline):
        raise ValueError("A completed canonical detector fit is required")
    suffix="free_endpoints" if args.free_endpoints_only else "free_led"
    if args.free_endpoints_only and args.endpoint_amplitude_bound!=.30:
        suffix+=f"_bound{args.endpoint_amplitude_bound:.2f}".replace(".","p")
    test_label="Endpoints free; LEDs anchored" if args.free_endpoints_only else "LED free"
    output=baseline.with_name(baseline.name+f"_{suffix}_test")
    output.mkdir(parents=True,exist_ok=True)
    base_summary=json.loads((baseline/"fit_summary.json").read_text())
    with np.load(baseline/"predictions.npz",allow_pickle=False) as archive:
        previous={key:archive[key] for key in ("source_ids","valid","student_t_fit_parameters","student_t_prediction_mag")}
    _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare(args.detector,write_inputs=False,
                                                               effective_area_path=area,output_dir=output)
    np.testing.assert_array_equal(previous["source_ids"].astype(str),np.asarray(ids))
    np.testing.assert_array_equal(previous["valid"],valid)
    old_objective=StudentObjective(model,magnitude,error,valid)
    original_coordinates=np.concatenate([previous["student_t_fit_parameters"]/old_objective.coordinate_scale,
                                         [np.log(base_summary["extra_student_t_scale_mmag"])]])
    original_value,original_prediction,*_=old_objective.state(original_coordinates)
    old_modes=model.mode_count
    del old_objective
    _,extra_modes=augment_basis(model,endpoints_only=args.free_endpoints_only)
    initial_parameters=expand_parameters(previous["student_t_fit_parameters"],old_modes,extra_modes)
    objective=StudentObjective(model,magnitude,error,valid)
    initial=np.concatenate([initial_parameters/objective.coordinate_scale,[original_coordinates[-1]]])
    null_value,null_prediction,*_=objective.state(initial)
    null_prediction_error=float(np.max(np.abs(null_prediction-original_prediction)))
    null_objective_error=float(abs(null_value-original_value))
    if null_prediction_error>1e-8 or null_objective_error>1e-6:
        raise RuntimeError("Free-mode null model does not reproduce the canonical fit")
    value,gradient=objective(initial)
    gradient_errors=[]
    for index in (old_modes,model.mode_count+old_modes,len(initial)-2,len(initial)-1):
        upper,lower=initial.copy(),initial.copy()
        upper[index]+=1e-4
        lower[index]-=1e-4
        numerical=(objective.state(upper)[0]-objective.state(lower)[0])/(2e-4)
        gradient_errors.append(abs(numerical-gradient[index])/(1+abs(numerical)))
    if max(gradient_errors)>2e-3:raise RuntimeError("Free-mode gradient check failed")
    count=model.mode_count
    bounds=[(-3.,3.)]*count+[(-10.,10.)]*count+[(-25.,25.)]*model.zero_point_count
    bounds.append((np.log(1e-4),np.log(100.)))
    initial_source=str(baseline)
    if args.free_endpoints_only:
        for index in range(old_modes,count):
            bounds[index]=(-args.endpoint_amplitude_bound/.1,args.endpoint_amplitude_bound/.1)
        if args.endpoint_amplitude_bound>.30:
            previous_endpoint=baseline.with_name(baseline.name+"_free_endpoints_test")
            with np.load(previous_endpoint/"predictions.npz",allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["source_ids"].astype(str),np.asarray(ids))
                np.testing.assert_array_equal(archive["valid"],valid)
                initial[:-1]=archive["student_t_fit_parameters"]/objective.coordinate_scale
            previous_endpoint_summary=json.loads((previous_endpoint/"fit_summary.json").read_text())
            initial[-1]=np.log(previous_endpoint_summary["free_led_extra_scale_mmag"])
            initial_source=str(previous_endpoint)
    groups=np.asarray(model.pass_zero_point_group_ids)
    for fi in range(model.filter_count):
        for gid in range(1,model.zero_point_group_count+1):
            connected=np.any(valid[:,fi,groups==0],axis=1)&np.any(valid[:,fi,groups==gid],axis=1)
            if not connected.any():
                index=2*count+fi*model.zero_point_group_count+gid-1
                initial[index]=0.
                bounds[index]=(0.,0.)
    started=time.monotonic()
    fit=minimize(objective,initial,method="L-BFGS-B",jac=True,bounds=bounds,
                 options={"maxiter":600,"ftol":1e-11,"gtol":1e-5,"maxls":30})
    final_gradient=objective(fit.x)[1]
    projected_gradient=final_gradient.copy()
    parameter_names=(["amplitude:"+name for name in model.basis_names]
                     +["phase:"+name for name in model.basis_names]
                     +[f"roll:{i}" for i in range(model.zero_point_count)]+["log_extra_scale"])
    at_bounds=[]
    for index,(low,high) in enumerate(bounds):
        at_low=fit.x[index]<=low+1e-7
        at_high=fit.x[index]>=high-1e-7
        if at_low or at_high:
            at_bounds.append(parameter_names[index])
        if (at_low and final_gradient[index]>0) or (at_high and final_gradient[index]<0) or low==high:
            projected_gradient[index]=0.
    value,prediction,location,residual,score,scales=objective.state(fit.x)
    parameters=fit.x[:-1]*objective.coordinate_scale
    prediction/=1000.
    # Compare both predictions with the same robust-location rule and the same
    # canonical scale, and separately with measured EP3 reference differences.
    old_group_values=(magnitude-previous["student_t_prediction_mag"]).reshape(-1,valid.shape[-1])[objective.active_groups]*1000.
    new_group_values=(magnitude-prediction).reshape(-1,valid.shape[-1])[objective.active_groups]*1000.
    common_scales=np.hypot(objective.errors,base_summary["extra_student_t_scale_mmag"])
    _,old_residual,_=profile_student_locations(old_group_values,common_scales,objective.valid)
    _,new_residual,_=profile_student_locations(new_group_values,common_scales,objective.valid)
    active_indices=np.flatnonzero(objective.active_groups)
    band_indices=active_indices%model.filter_count
    filter_records=[]
    for fi,band in enumerate(run.FILTERS):
        mask=(band_indices==fi)[:,None]&objective.valid
        filter_records.append({"filter":band,"variant":"LED anchored",**metrics(old_residual[mask])})
        filter_records.append({"filter":band,"variant":test_label,**metrics(new_residual[mask])})
    metrics_table=pd.DataFrame(filter_records)
    metrics_table.to_csv(output/"filter_residual_comparison.csv",index=False)
    primary=valid&valid[:,:,:1]
    primary[:,:,0]=False
    fluxes={}
    ep3_metrics=[]
    for label,pred in (("LED anchored",previous["student_t_prediction_mag"]),(test_label,prediction)):
        corrected=magnitude-pred
        delta=corrected-corrected[:,:,:1]
        fluxes[label]=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*delta)
        ep3_metrics.append({"variant":label,**metrics(delta[primary]*1000.)})
    pd.DataFrame(ep3_metrics).to_csv(output/"ep3_reference_metrics.csv",index=False)
    change=profile_residual(prediction,previous["student_t_prediction_mag"],error,valid)[valid]*1000.
    summary={"detector":args.detector,"likelihood":"Student-t","nu":NU,
             "optimizer_success":bool(fit.success),"optimizer_message":str(fit.message),
             "iterations":int(fit.nit),"elapsed_seconds":time.monotonic()-started,
             "basis":("per-gap plus smooth free boundary values; six LED anchors retained" if args.free_endpoints_only
                      else "per-gap plus smooth free values at both boundaries and six LED wavelengths"),
             "LED_values_free":not args.free_endpoints_only,"boundary_values_free":True,
             "endpoint_log_amplitude_bound":args.endpoint_amplitude_bound,
             "original_gap_log_amplitude_bound":.30,"phase_bound_rad":1.,
             "initial_parameter_source":initial_source,
             "old_modulation_modes":old_modes,"new_modulation_modes":count,
             "added_real_parameters":2*extra_modes,"ice_histories_refitted":False,
             "used_stars":len(ids),"used_observations":int(valid.sum()),
             "canonical_extra_scale_mmag":base_summary["extra_student_t_scale_mmag"],
             "free_led_extra_scale_mmag":float(np.exp(fit.x[-1])),
             "canonical_objective":original_value,"free_led_objective":value,
             "objective_improvement":original_value-value,
             "null_prediction_max_difference_mmag":null_prediction_error,
             "null_objective_difference":null_objective_error,
             "max_gradient_check_error":max(gradient_errors),
             "max_abs_location_score":float(np.max(np.abs(score.sum(axis=1)))),
             "max_projected_gradient":float(np.max(np.abs(projected_gradient))),
             "parameters_at_bounds":at_bounds,
             "prediction_change_rms_mmag":float(np.sqrt(np.mean(change**2))),
             "prediction_change_p95_absolute_mmag":float(np.quantile(np.abs(change),.95)),
             "canonical_metrics_common_robust_normalization":metrics(old_residual[objective.valid]),
             "free_led_metrics_common_robust_normalization":metrics(new_residual[objective.valid]),
             "baseline_directory":str(baseline),"effective_area_file":str(area),
             "maxent_weight":model.entropy_weight,"phase_weight":model.phase_smoothness_weight,
             "prior_note":"Original gap-mode penalties unchanged; added knot-value modes have wavelength-support-weighted amplitude shrinkage and the same phase shrinkage convention",
             "scope":"separate sensitivity experiment; no production bundle changes"}
    (output/"fit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(output/"predictions.npz",source_ids=np.asarray(ids),valid=valid,
                        student_t_fit_parameters=parameters,student_t_prediction_mag=prediction,
                        basis=model.basis,basis_names=np.asarray(model.basis_names),wavelengths_nm=model.wavelengths_nm,
                        robust_residuals_mmag=residual,robust_locations_mmag=location)
    pd.DataFrame({"mode":model.basis_names,"fit_log_amplitude":parameters[:count],
                  "fit_phase_rad":parameters[count:2*count]}).to_csv(output/"modulation_parameters.csv",index=False)
    knots=np.concatenate(([400.],mock.LED_CENTERS_NM,[2321.]))
    knot_basis,*_=free_value_basis(knots)
    if args.free_endpoints_only:knot_basis=knot_basis[[0,-1]]
    knot_values=parameters[old_modes:count]@knot_basis
    knot_phases=parameters[count+old_modes:2*count]@knot_basis
    if args.free_endpoints_only:
        anchor_error=float(np.max(np.abs(np.exp(knot_values[1:-1]+1j*knot_phases[1:-1])-1.)))
        if anchor_error>1e-12:raise RuntimeError("Endpoint-only experiment changed an LED anchor")
    pd.DataFrame({"wavelength_nm":knots,"amplitude_multiplier":np.exp(knot_values),
                  "amplitude_modulation_percent":100*np.expm1(knot_values),"phase_modulation_rad":knot_phases,
                  "kind":["boundary"]+["LED"]*6+["boundary"]}).to_csv(output/"led_and_boundary_modulation.csv",index=False)
    dates=mdates.date2num(Time(common.groupby("pass")["mjd_amber"].median().reindex(run.PASSES).to_numpy(),
                              format="mjd",scale="utc").to_datetime())
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    colors={"LED anchored":"#397fa1",test_label:"#379663" if args.free_endpoints_only else "#d77c2e"}
    for fi,(band,axis) in enumerate(zip(run.FILTERS,axes.flat,strict=True)):
        axis.axhspan(-.5,.5,color="#65ac68",alpha=.15)
        for label,offset in (("LED anchored",-.06),(test_label,.06)):
            for pi,date in enumerate(dates):
                x=fluxes[label][:,fi,pi][primary[:,fi,pi]]
                if not len(x):continue
                low,med,high=np.quantile(x,[.16,.5,.84])
                axis.errorbar(date+offset,med,yerr=[[med-low],[high-med]],fmt="o",color=colors[label],
                              markersize=3,capsize=1.5,label=label if pi==1 else None)
        axis.set_title(band)
        axis.axhline(0,color="0.5",linewidth=.6)
        axis.grid(alpha=.2)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x",labelrotation=45,labelsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.94),ncol=2)
    fig.suptitle(f"{args.detector} Amber Student-t: canonical versus {test_label.lower()}",fontsize=15,y=.99)
    fig.supylabel("Corrected flux change from measured EP3 pass 1 (%)")
    fig.text(.5,.025,"Same photometry, quality cuts, detector effective area, fixed ice history and original regularization. "
             +("Added smooth modes free only the boundaries; all six LED anchors retained.\n" if args.free_endpoints_only
               else "Added smooth modes free the six LEDs and both boundaries.\n")
             +"Points: star medians; bars: 16th–84th percentiles; green: ±0.5%. Fits use all repeated stars; cumulative plot requires a measured EP3 reference.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.12,1,.88))
    fig.savefig(output/f"anchored_vs_{suffix}_corrections.png",dpi=180)
    plt.close(fig)
    wave=np.arange(470.,2321.,2.)
    old_basis,*_=mock.modulation_basis(wave,grouping="per-gap")
    extra,*_=free_value_basis(wave)
    if args.free_endpoints_only:extra=extra[[0,-1]]
    new_basis=np.concatenate([old_basis,extra])
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
    for label,theta,basis in (("LED anchored",previous["student_t_fit_parameters"],old_basis),
                              (test_label,parameters,new_basis)):
        modes=len(basis)
        axes[0].plot(wave,100*np.expm1(theta[:modes]@basis),color=colors[label],label=label)
        axes[1].plot(wave,theta[modes:2*modes]@basis,color=colors[label])
    for axis in axes:
        for center in mock.LED_CENTERS_NM:axis.axvline(center,color="0.6",linestyle=":",linewidth=.7)
        axis.axhline(0,color="0.5",linewidth=.6)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Clean-reflection amplitude modulation (%)")
    axes[1].set_ylabel("Clean-reflection phase modulation (rad)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].legend()
    fig.suptitle(f"{args.detector}: {test_label.lower()}")
    fig.savefig(output/f"anchored_vs_{suffix}_modulation.png",dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    print(metrics_table.to_string(index=False))
    print(pd.DataFrame(ep3_metrics).to_string(index=False))
    if not fit.success:raise RuntimeError(f"Free-LED optimizer failed: {fit.message}")


if __name__=="__main__":
    main()
