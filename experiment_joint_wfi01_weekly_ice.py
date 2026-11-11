"""Joint detector CAR190 + Amber fit with independent per-tile rate modulations."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from astropy.time import Time

import run_amber_student_t_detectors as run
from experiment_joint_wfi01_fixed_ice import LedModel,add_full_basis,fixed_output,TILE_COUNT
from fit_amber_wfi01_student_t import StudentObjective
from car201_photometry_forward_jacobian import CachedPhotometryForward
from per_tile_rate_modulation import PerTileRateModulation
from support_aligned_modulation import effective_area_support

EXPERIMENTS=run.workflow.PRODUCTION_ROOT/"wfi01/spline_relative_response_normal/experiments"
OUTPUT=EXPERIMENTS/"car190_amber_joint_per_tile_weekly_rate_test"
RESETS=np.array([12.907397928240743,38.79784510416666])


def cadence_output(detector,spacing_days):
    name=("car190_amber_joint_per_tile_weekly_rate_test" if spacing_days==7.
          else f"car190_amber_joint_per_tile_{spacing_days:g}day_rate_test")
    return run.workflow.PRODUCTION_ROOT/f"{detector.lower()}/spline_relative_response_normal/experiments/{name}"


def led_loss(led,theta,thickness,jacobian,time_indices):
    mode=led.basis.shape[0]; grad_theta=np.zeros(2*mode); grad_ice=np.zeros((TILE_COUNT,jacobian.shape[-1]));cost=0.
    for band,tiles in enumerate(led.bands):
        for tile,item in enumerate(tiles):
            a0,b0,r,y0,factor,basis,saved,observed,sigma,days,original,wave,ice=item
            index=time_indices[band][tile]; new=thickness[tile,index]
            shift=2*np.pi*ice*(new-original)[:,None]/wave[None,:]
            c=np.cos(shift);s=-1j*np.sin(shift)
            a=c*a0+s*ice*b0;b=c*b0+s*a0/ice
            reflection=r*np.exp(theta[:mode]@basis+1j*(theta[mode:]@basis))
            y=y0*(1-reflection)/(1+reflection);dy=-2*y0*reflection/(1+reflection)**2
            den=a+b*y[None,:]; weighted=factor[None,:]/np.abs(den)**2
            rate=weighted.sum(axis=1); log=np.log(rate);old=led.references[band][tile]
            residual=(saved+(log-log[0])-(old-old[0])-observed)/sigma
            cost+=.5*float(residual@residual)
            sensitivity=b*dy[None,:]/den
            derivative=np.concatenate([(-2*weighted*sensitivity.real)@basis.T,
                                       (2*weighted*sensitivity.imag)@basis.T],axis=1)/rate[:,None]
            grad_theta+=(derivative-derivative[:1]).T@(residual/sigma)
            phase_derivative=2*np.pi*ice/wave[None,:]
            dl_dd=(-2*weighted*np.real((-1j*(ice*b+y[None,:]*a/ice))*phase_derivative/den)).sum(axis=1)/rate
            deriv=dl_dd[:,None]*jacobian[tile,index,:]
            grad_ice[tile]+=(deriv-deriv[:1]).T@(residual/sigma)
    return cost,grad_theta,grad_ice


def stellar_ice_gradient(objective,coordinates,jacobian,star_time_indices):
    model=objective.model
    _,_,_,_,scores,_=objective.state(coordinates)
    score=np.zeros((model.star_count*model.filter_count,model.pass_count))
    score[objective.active_groups]=scores
    score=score.reshape(model.star_count,model.filter_count,model.pass_count)
    mode=model.mode_count;theta=coordinates[:-1]*objective.coordinate_scale
    ratio=np.exp(theta[:mode]@model.basis+1j*(theta[mode:2*mode]@model.basis))
    y0=model.incident_admittance_by_tile[:,None];r=model.clean_reflection_by_tile*ratio[None,:]
    y=y0*(1-r)/(1+r);forward=objective.forward
    sensitivity=np.empty((model.star_count,model.filter_count,model.pass_count))
    for fi in range(model.filter_count):
        for pi in range(model.pass_count):
            a,b=forward.coefficients[fi][pi];den=a+b*y
            power=forward.power_numerator/np.abs(den)**2
            tile=model.source_tile_ids[:,fi,pi];kernel=forward.kernels[fi]*power[tile]
            dphase=2*np.pi*model.ice_index_by_tile[:,None]/model.wavelengths_nm[None,:]
            dl=(-2*np.real((-1j*(model.ice_index_by_tile[:,None]*b+y*a/model.ice_index_by_tile[:,None]))*dphase/den))[tile]
            sensitivity[:,fi,pi]=(kernel*dl).sum(axis=1)/kernel.sum(axis=1)
    grad=np.zeros((TILE_COUNT,jacobian.shape[-1]));factor=1000./run.workflow.LOG_FLUX_PER_MAG
    for fi in range(model.filter_count):
        ref_tile=model.source_tile_ids[:,fi,0];ref_index=star_time_indices[fi,0]
        for pi in range(model.pass_count):
            tile=model.source_tile_ids[:,fi,pi];index=star_time_indices[fi,pi]
            weight=(factor*score[:,fi,pi])[:,None]
            np.add.at(grad,tile,weight*sensitivity[:,fi,pi,None]*jacobian[tile,index,:])
            np.add.at(grad,ref_tile,-weight*sensitivity[:,fi,0,None]*jacobian[ref_tile,ref_index,:])
    return grad


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",required=True,choices=run.workflow.DETECTORS)
    parser.add_argument("--spacing-days",type=float,default=7.)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--initialize-from-weekly",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--maxiter",type=int,default=500)
    args=parser.parse_args()
    if args.spacing_days<=0:raise ValueError("--spacing-days must be positive")
    detector=args.detector
    fixed=fixed_output(detector)
    output=args.output_dir if args.output_dir is not None else cadence_output(detector,args.spacing_days)
    output.mkdir(parents=True,exist_ok=True)
    area,_=run.canonical_configuration(detector)
    _,common,ids,magnitude,error,valid,_,model,_=run.prepare(detector,write_inputs=False,
        effective_area_path=area,output_dir=output,full_effective_area_support=True)
    add_full_basis(model,effective_area_support(area))
    with np.load(fixed/"joint_fit.npz",allow_pickle=False) as a:
        theta=a["parameters"];np.testing.assert_array_equal(a["stellar_valid"],valid)
    fixed_summary=json.loads((fixed/"joint_fit_summary.json").read_text())
    led=LedModel(effective_area_support(area),detector=detector)
    reference=Time(run.workflow.fit_model.ICE_REFERENCE_UTC,format="isot",scale="utc").mjd
    star_days=np.array([[common.loc[(common["filter"]==band)&(common["pass"]==epoch),"mjd_amber"].median()-reference
                         for epoch in run.PASSES] for band in run.FILTERS])
    led_days=np.unique(np.concatenate([item[9] for tiles in led.bands for item in tiles]))
    times=np.unique(np.r_[led_days,star_days.ravel()])
    segment=np.searchsorted(RESETS,times,side="right")
    baseline=np.empty((TILE_COUNT,len(times)))
    for tile in range(TILE_COUNT):
        path=run.workflow.PRODUCTION_ROOT/f"{detector.lower()}/spline_relative_response_normal/fits/tile_{tile:03d}/posterior_samples.npz"
        with np.load(path,allow_pickle=False) as a:
            days=np.asarray(a["observation_days"],float).reshape(-1)
            values=np.asarray(a["tile_ice_thickness"],float).reshape(-1,len(days))[0]
        for stage in range(3):
            target=segment==stage;source=np.searchsorted(RESETS,days,side="right")==stage
            if times[target].min()<days[source].min()-1e-9 or times[target].max()>days[source].max()+1e-9:
                raise ValueError("Stellar or LED time is outside fixed history segment")
            baseline[tile,target]=np.interp(times[target],days[source],values[source])
    trajectory=PerTileRateModulation(times,baseline,RESETS,spacing_days=args.spacing_days)
    star_indices=np.searchsorted(times,star_days)
    led_indices=[[np.searchsorted(times,item[9]) for item in tiles] for tiles in led.bands]
    objective=StudentObjective(model,magnitude,error,valid)
    nopt=len(theta)+1
    shared_parameters=theta
    extra_scale=fixed_summary["extra_student_t_scale_mmag"]
    ice_parameters=np.zeros((TILE_COUNT,trajectory.parameter_count))
    if args.initialize_from_weekly:
        weekly_output=cadence_output(detector,7.)
        with np.load(weekly_output/"joint_weekly_ice_fit.npz",allow_pickle=False) as a:
            weekly_parameters=a["ice_parameters"]
            shared_parameters=a["shared_parameters"]
            extra_scale=float(a["extra_student_t_scale_mmag"])
        weekly_trajectory=PerTileRateModulation(times,baseline,RESETS,spacing_days=7.)
        if weekly_parameters.shape!=(TILE_COUNT,weekly_trajectory.parameter_count):
            raise ValueError("Weekly ice-control dimensions do not match the current trajectory")
        for current,weekly in zip(trajectory.specs,weekly_trajectory.specs):
            ice_parameters[:,current["initial_index"]]=weekly_parameters[:,weekly["initial_index"]]
            for tile in range(TILE_COUNT):
                ice_parameters[tile,current["rate_slice"]]=np.interp(
                    current["knots"],weekly["knots"],weekly_parameters[tile,weekly["rate_slice"]])
    initial=np.r_[shared_parameters/objective.coordinate_scale,np.log(extra_scale),ice_parameters.ravel()]
    mode=model.mode_count
    bounds=[(-3.,3.)]*mode+[(-10.,10.)]*mode+[(-25.,25.)]*model.zero_point_count+[(np.log(1e-4),np.log(100.))]
    bounds[7]=bounds[8]=(-6.,6.)
    bounds += [(-np.log(4.),np.log(4.))]*(TILE_COUNT*trajectory.parameter_count)
    sigma=np.full(trajectory.parameter_count,1.5)
    sigma[trajectory.specs[2]["initial_index"]]=.1
    if args.resume:
        with np.load(output/"optimizer_checkpoint.npz",allow_pickle=False) as checkpoint:
            initial=checkpoint["coordinates"]
        if initial.shape!=(nopt+TILE_COUNT*trajectory.parameter_count,):
            raise ValueError("Checkpoint dimensions do not match the selected ice cadence")
    calls=0
    def total(x):
        nonlocal calls
        ice_params=x[nopt:].reshape(TILE_COUNT,trajectory.parameter_count)
        thickness,jac=trajectory.evaluate(ice_params,with_jacobian=True)
        model.tile_thickness_by_id=thickness[:,star_indices]
        objective.forward=CachedPhotometryForward(model)
        stellar,stellar_grad=objective(x[:nopt])
        optical=x[:2*mode]*objective.coordinate_scale[:2*mode]
        led_cost,led_optical,led_ice=led_loss(led,optical,thickness,jac,led_indices)
        stellar_ice=stellar_ice_gradient(objective,x[:nopt],jac,star_indices)
        prior=.5*np.sum((ice_params/sigma[None,:])**2)
        grad=np.empty_like(x);grad[:nopt]=stellar_grad;grad[:2*mode]+=led_optical*objective.coordinate_scale[:2*mode]
        grad[nopt:]=(led_ice+stellar_ice+ice_params/sigma[None,:]**2).ravel()
        calls+=1
        if calls%10==0:print(f"evaluation {calls}: objective={stellar+led_cost+prior:.5f}; CAR190 chi2={2*led_cost:.4f}; ice prior={prior:.3f}",flush=True)
        return stellar+led_cost+prior,grad
    value,gradient=total(initial)
    checks=[]
    for idx in (0,mode,nopt,nopt+89*trajectory.parameter_count+2,nopt+136*trajectory.parameter_count+trajectory.specs[2]["initial_index"]):
        upper=initial.copy();lower=initial.copy();upper[idx]+=1e-4;lower[idx]-=1e-4
        numerical=(total(upper)[0]-total(lower)[0])/2e-4
        discrepancy=abs(gradient[idx]-numerical)/(1+abs(numerical));checks.append(discrepancy)
        print(f"gradient {idx}: analytic={gradient[idx]:.7g}; finite={numerical:.7g}; relative={discrepancy:.3g}",flush=True)
    if max(checks)>2e-3:raise ValueError(f"Joint weekly-ice gradient check failed: {checks}")
    started=time.monotonic()
    def checkpoint(x):
        np.savez(output/"optimizer_checkpoint.npz",coordinates=x)
    result=minimize(total,initial,jac=True,bounds=bounds,method="L-BFGS-B",callback=checkpoint,
                    options={"maxiter":args.maxiter,"ftol":1e-9,"gtol":1e-4,"maxls":30})
    final_value,gradient=total(result.x)
    parameters=result.x[nopt:].reshape(TILE_COUNT,trajectory.parameter_count)
    thickness=trajectory.evaluate(parameters)
    report={"detector":detector,"tiles":TILE_COUNT,"stellar_stars":len(ids),"stellar_measurements":int(valid.sum()),
            "rate_control_spacing_days":args.spacing_days,"ice_parameters_per_tile":trajectory.parameter_count,
            "total_ice_parameters":int(parameters.size),"reset_days":RESETS.tolist(),
            "trajectory_definition":"positive adjusted baseline growth increments; independent tile/segment initial thickness and time-varying log-rate controls",
            "initial_and_rate_log_prior_sigma":1.5,"third_decon_initial_log_prior_sigma":.1,
            "prior_note":"regularization on log multipliers around the existing history; CAR190 also contributed to the existing histories, so interpret this as a sensitivity fit, not an independent-prior posterior",
            "initialization":"weekly fitted controls interpolated to current knots" if args.initialize_from_weekly else "fixed-history joint fit",
            "resumed_from_checkpoint":bool(args.resume),
            "likelihood_multipliers":{"CAR190":1.,"Amber":1.},"initial_objective":float(value),"final_objective":float(final_value),
            "optimizer_success":bool(result.success),"optimizer_message":str(result.message),"iterations":int(result.nit),
            "gradient_check_max_relative_discrepancy":max(checks),"elapsed_seconds":time.monotonic()-started,
            "ice_change_nm_p16_median_p84":np.percentile(thickness-baseline,[16,50,84],axis=0).tolist(),
            "max_abs_ice_change_nm":float(np.max(np.abs(thickness-baseline))),"production_bundle_changed":False}
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(output/"joint_weekly_ice_fit.npz",times=times,baseline_thickness_nm=baseline,
                        fitted_thickness_nm=thickness,ice_parameters=parameters,shared_parameters=result.x[:nopt-1]*objective.coordinate_scale,
                        extra_student_t_scale_mmag=np.exp(result.x[nopt-1]),source_ids=np.asarray(ids))
    print(json.dumps({k:v for k,v in report.items() if k!="ice_change_nm_p16_median_p84"},indent=2),flush=True)
    if not result.success:raise RuntimeError(result.message)


if __name__=="__main__":main()
