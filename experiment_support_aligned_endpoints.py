"""Compare anchored and free endpoints aligned to a detector's actual band support."""

import argparse
import json
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from astropy.time import Time

import run_amber_student_t_detectors as run
from fit_amber_wfi01_student_t import StudentObjective,profile_student_locations,NU
from experiment_free_led_modulation import expand_parameters,free_value_basis,metrics
from support_aligned_modulation import effective_area_support,configure_basis,gap_basis
from plot_wfi01_free_led_complex_reflection import aligned_phase,reciprocal


def solve(model,magnitude,error,valid,theta,scale,bound,output):
    objective=StudentObjective(model,magnitude,error,valid)
    initial=np.concatenate([theta/objective.coordinate_scale,[np.log(scale)]])
    count=model.mode_count
    bounds=[(-3.,3.)]*count+[(-10.,10.)]*count+[(-25.,25.)]*model.zero_point_count
    bounds.append((np.log(1e-4),np.log(100.)))
    if count==9:
        for index in (7,8):bounds[index]=(-bound/.1,bound/.1)
    groups=np.asarray(model.pass_zero_point_group_ids)
    for fi in range(model.filter_count):
        for gid in range(1,model.zero_point_group_count+1):
            connected=np.any(valid[:,fi,groups==0],axis=1)&np.any(valid[:,fi,groups==gid],axis=1)
            if not connected.any():
                index=2*count+fi*model.zero_point_group_count+gid-1
                initial[index]=0.
                bounds[index]=(0.,0.)
    value,gradient=objective(initial)
    checks=[]
    for index in (0,count,count-1,2*count-1,len(initial)-1):
        upper,lower=initial.copy(),initial.copy()
        upper[index]+=1e-4
        lower[index]-=1e-4
        numerical=(objective.state(upper)[0]-objective.state(lower)[0])/(2e-4)
        checks.append(abs(numerical-gradient[index])/(1+abs(numerical)))
    if max(checks)>2e-3:raise RuntimeError("Support-aligned gradient check failed")
    started=time.monotonic()
    result=minimize(objective,initial,method="L-BFGS-B",jac=True,bounds=bounds,
                    options={"maxiter":600,"ftol":1e-11,"gtol":1e-5,"maxls":30})
    value,prediction,location,residual,score,scales=objective.state(result.x)
    gradient=objective(result.x)[1]
    projected=gradient.copy()
    bound_names=[]
    names=(["amplitude:"+name for name in model.basis_names]
           +["phase:"+name for name in model.basis_names]
           +[f"roll:{i}" for i in range(model.zero_point_count)]+["log_extra_scale"])
    for index,(lower,upper) in enumerate(bounds):
        at_lower=result.x[index]<=lower+1e-7
        at_upper=result.x[index]>=upper-1e-7
        if at_lower or at_upper:bound_names.append(names[index])
        if (at_lower and gradient[index]>0) or (at_upper and gradient[index]<0) or lower==upper:
            projected[index]=0.
    parameters=result.x[:-1]*objective.coordinate_scale
    summary={"optimizer_success":bool(result.success),"optimizer_message":str(result.message),
             "iterations":int(result.nit),"objective":value,"extra_student_t_scale_mmag":float(np.exp(result.x[-1])),
             "endpoint_amplitude_bound":bound if count==9 else None,
             "endpoint_values_free":count==9,"LED_values_free":False,
             "modulation_modes":count,"original_gap_amplitude_bound":.3,"phase_bound_rad":1.,
             "parameters_at_bounds":bound_names,"max_projected_gradient":float(np.max(np.abs(projected))),
             "max_gradient_check_error":max(checks),"max_profile_location_score":float(np.max(np.abs(score.sum(axis=1)))),
             "elapsed_seconds":time.monotonic()-started,
             "modulation_boundaries_nm":list(model.modulation_boundaries_nm),
             "integration_limits_nm":[float(model.wavelengths_nm[0]),float(model.wavelengths_nm[-1])],
             "maxent_weight":model.entropy_weight,"phase_weight":model.phase_smoothness_weight}
    output.mkdir(parents=True,exist_ok=True)
    (output/"fit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(output/"predictions.npz",parameters=parameters,prediction_mag=prediction/1000.,valid=valid,
                        basis=model.basis,node_weights=model.node_weights,wavelengths_nm=model.wavelengths_nm,
                        robust_residuals_mmag=residual,robust_locations_mmag=location)
    pd.DataFrame({"mode":model.basis_names,"fit_log_amplitude":parameters[:count],
                  "fit_phase_rad":parameters[count:2*count]}).to_csv(output/"modulation_parameters.csv",index=False)
    if not result.success:raise RuntimeError(f"Support-aligned fit failed: {result.message}")
    return {"theta":parameters,"prediction":prediction/1000.,"summary":summary,"basis":model.basis.copy()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",required=True,choices=run.workflow.DETECTORS)
    args=parser.parse_args()
    area,canonical=run.canonical_configuration(args.detector)
    output=canonical.with_name(canonical.name+"_support_aligned_endpoints_test")
    bounds=effective_area_support(area)
    source_summary=json.loads((canonical/"fit_summary.json").read_text())
    with np.load(canonical/"predictions.npz",allow_pickle=False) as archive:
        old_theta=archive["student_t_fit_parameters"]
        old_ids=archive["source_ids"].astype(str)
        old_valid=archive["valid"]
    _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare(args.detector,write_inputs=False,
        effective_area_path=area,output_dir=output,full_effective_area_support=True)
    np.testing.assert_array_equal(old_ids,np.asarray(ids))
    np.testing.assert_array_equal(old_valid,valid)
    print(f"Support-aligned nodes: {bounds}; integration: {model.wavelengths_nm[0]}–{model.wavelengths_nm[-1]} nm",flush=True)
    trials={}
    configure_basis(model,bounds,free_endpoints=False)
    trials["Aligned anchored"]=solve(model,magnitude,error,valid,old_theta,
        source_summary["extra_student_t_scale_mmag"],None,output/"anchored")
    configure_basis(model,bounds,free_endpoints=True)
    initial=expand_parameters(trials["Aligned anchored"]["theta"],7,2)
    # The augmented model must contain the aligned anchored fit exactly.
    check=StudentObjective(model,magnitude,error,valid)
    coordinates=np.concatenate([initial/check.coordinate_scale,
        [np.log(trials["Aligned anchored"]["summary"]["extra_student_t_scale_mmag"])]])
    null_value,null_prediction,*_=check.state(coordinates)
    np.testing.assert_allclose(null_prediction/1000.,trials["Aligned anchored"]["prediction"],atol=1e-12)
    if abs(null_value-trials["Aligned anchored"]["summary"]["objective"])>1e-6:
        raise ValueError("Endpoint null changed the aligned anchored objective")
    del check
    trials["Endpoints ±0.30"]=solve(model,magnitude,error,valid,initial,
        trials["Aligned anchored"]["summary"]["extra_student_t_scale_mmag"],.3,output/"bound0p30")
    trials["Endpoints ±0.60"]=solve(model,magnitude,error,valid,trials["Endpoints ±0.30"]["theta"],
        trials["Endpoints ±0.30"]["summary"]["extra_student_t_scale_mmag"],.6,output/"bound0p60")
    active=valid.reshape(-1,valid.shape[-1]).sum(axis=1)>=2
    mask=valid.reshape(-1,valid.shape[-1])[active]
    scale=np.hypot(error.reshape(-1,valid.shape[-1])[active]*1000.,
                    trials["Aligned anchored"]["summary"]["extra_student_t_scale_mmag"])
    primary=valid&valid[:,:,:1]
    primary[:,:,0]=False
    summaries=[]
    endpoint_records=[]
    fluxes={}
    for label,trial in trials.items():
        group_values=(magnitude-trial["prediction"]).reshape(-1,valid.shape[-1])[active]*1000.
        _,residual,_=profile_student_locations(group_values,scale,mask)
        corrected=magnitude-trial["prediction"]
        delta=corrected-corrected[:,:,:1]
        fluxes[label]=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*delta)
        summaries.append({"variant":label,**trial["summary"],
                          "fraction_within_0p5_common_robust_normalization":metrics(residual[mask])["fraction_within_0p5_percent"],
                          "fraction_within_0p5_measured_ep3_reference":metrics(delta[primary]*1000.)["fraction_within_0p5_percent"]})
        nodes=np.concatenate(([bounds[0]],run.workflow.mock_model.LED_CENTERS_NM,[bounds[1]]))
        node_basis,_=gap_basis(nodes,bounds)
        if trial["summary"]["endpoint_values_free"]:
            extra,*_=free_value_basis(nodes,boundaries=bounds)
            node_basis=np.concatenate([node_basis,extra[[0,-1]]])
        count=len(node_basis)
        theta=trial["theta"]
        amplitude=theta[:count]@node_basis
        phase=theta[count:2*count]@node_basis
        ratio=np.exp(amplitude+1j*phase)
        if np.max(np.abs(ratio[1:-1]-1))>1e-12:raise RuntimeError("LED anchor changed")
        for ni,node in enumerate(nodes):endpoint_records.append({"variant":label,"wavelength_nm":node,
            "kind":"boundary" if ni in (0,len(nodes)-1) else "LED","log_amplitude":amplitude[ni],
            "amplitude_multiplier":float(np.exp(amplitude[ni])),"phase_rad":phase[ni]})
    pd.DataFrame(summaries).to_csv(output/"support_aligned_comparison.csv",index=False)
    pd.DataFrame(endpoint_records).to_csv(output/"support_aligned_node_values.csv",index=False)
    colors={"Aligned anchored":"#357fa1","Endpoints ±0.30":"#379663","Endpoints ±0.60":"#d77c2e"}
    dates=mdates.date2num(Time(common.groupby("pass")["mjd_amber"].median().reindex(run.PASSES).to_numpy(),
                              format="mjd",scale="utc").to_datetime())
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    for fi,(band,axis) in enumerate(zip(run.FILTERS,axes.flat,strict=True)):
        axis.axhspan(-.5,.5,color="#65ac68",alpha=.15)
        for li,label in enumerate(trials):
            for pi,date in enumerate(dates):
                values=fluxes[label][:,fi,pi][primary[:,fi,pi]]
                if not len(values):continue
                low,med,high=np.quantile(values,[.16,.5,.84])
                axis.errorbar(date+(li-1)*.07,med,yerr=[[med-low],[high-med]],fmt="o",color=colors[label],
                              markersize=3,capsize=1.5,label=label if pi==1 else None)
        axis.set_title(band)
        axis.grid(alpha=.2)
        axis.axhline(0,color="0.5",linewidth=.6)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x",labelrotation=45,labelsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.945),ncol=3)
    fig.suptitle(f"{args.detector}: support-aligned endpoints at {bounds[0]:g}/{bounds[1]:g} nm",y=.99,fontsize=15)
    fig.supylabel("Corrected flux change from measured EP3 pass 1 (%)")
    fig.text(.5,.025,"Same data, Student-t likelihood, fixed ice history and regularization. All six LED values remain anchored. Only endpoint amplitude bounds differ.\n"
             "Integration includes full imaging effective-area support with zero-throughput brackets. Points: medians; bars: star 16th–84th percentiles.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.12,1,.88))
    fig.savefig(output/"support_aligned_corrected_photometry.png",dpi=180)
    plt.close(fig)
    clean=model.clean_reflection_by_tile
    reference=np.angle(clean[:,-1])
    reference+=2*np.pi*np.round((np.median(reference)-reference)/(2*np.pi))
    order=np.argsort(1000/model.wavelengths_nm)
    x=1000/model.wavelengths_nm[order]
    fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
    for label,trial in trials.items():
        count=len(trial["basis"])
        theta=trial["theta"]
        ratio=np.exp(theta[:count]@trial["basis"]+1j*(theta[count:2*count]@trial["basis"]))
        reflection=clean*ratio[None,:]
        for axis,values in zip(axes,(np.abs(reflection),aligned_phase(reflection,reference)),strict=True):
            stats=np.percentile(values,[16,50,84],axis=0)
            axis.fill_between(x,stats[0,order],stats[2,order],color=colors[label],alpha=.12,linewidth=0)
            axis.plot(x,stats[1,order],color=colors[label],label=label,linewidth=1.7)
    for axis in axes:
        for led in run.workflow.mock_model.LED_CENTERS_NM:axis.axvline(1000/led,color="0.6",linestyle=":",linewidth=.7)
        for edge in bounds:axis.axvline(1000/edge,color="0.35",linestyle="--",linewidth=.7)
        axis.grid(alpha=.2)
        axis.set_xlim(x[0],x[-1])
    axes[0].set_ylabel(r"Clean reflection amplitude $|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped clean reflection phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top=axes[0].secondary_xaxis("top",functions=(reciprocal,reciprocal))
    top.set_xticks([2300,1700,1300,1000,800,600,500])
    top.set_xlabel("Wavelength (nm)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.945),ncol=3)
    fig.suptitle(f"{args.detector}: clean complex reflection with effective-area-aligned endpoints",y=.99,fontsize=15)
    fig.text(.5,.025,f"Endpoint nodes: {bounds[0]:g} and {bounds[1]:g} nm. All six LED anchors retained. Lines: tile medians; shading: tile 16th–84th percentiles.\n"
             "No ice coating shown. Phase branches share the same long-wavelength reference.",ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.08,1,.87))
    fig.savefig(output/"support_aligned_clean_complex_reflection.png",dpi=180)
    plt.close(fig)
    (output/"experiment_metadata.json").write_text(json.dumps({"detector":args.detector,"effective_area_file":str(area),
        "effective_area_sha256":run.hashlib.sha256(area.read_bytes()).hexdigest(),"support_nm":list(bounds),
        "integration_limits_nm":[float(model.wavelengths_nm[0]),float(model.wavelengths_nm[-1])],
        "grid_includes_zero_throughput_brackets":True,"six_LED_anchors_preserved":True,
        "observations":int(valid.sum()),"stars":len(ids),"published_models_changed":False},indent=2)+"\n")
    print(pd.DataFrame(summaries)[["variant","extra_student_t_scale_mmag","fraction_within_0p5_common_robust_normalization",
                                  "fraction_within_0p5_measured_ep3_reference","parameters_at_bounds"]].to_string(index=False))
    print(pd.DataFrame(endpoint_records).query("kind == 'boundary'").to_string(index=False))
    print(output)


if __name__=="__main__":
    main()
