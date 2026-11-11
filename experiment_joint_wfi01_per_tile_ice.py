"""Joint WFI01 fit with independent per-tile, per-growth-segment ice adjustments.

Each of 256 tiles has three multiplicative thickness adjustments. The shape of
each growth segment is retained and physical reset boundaries are never moved.
"""
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

import run_amber_student_t_detectors as run
from experiment_joint_wfi01_fixed_ice import LedModel,add_full_basis,OUTPUT as FIXED_OUTPUT
from fit_amber_wfi01_student_t import StudentObjective
from car201_photometry_forward_jacobian import CachedPhotometryForward
from support_aligned_modulation import effective_area_support

OUTPUT=FIXED_OUTPUT.with_name("car190_amber_joint_per_tile_segment_ice_test")
RESETS=np.array([12.907397928240743,38.79784510416666])
PRIOR_SIGMA=1.5
MAX_LOG_FACTOR=np.log(2.)


def led_objective(model,theta,log_factors):
    mode=model.basis.shape[0]
    grad_theta=np.zeros(2*mode);grad_ice=np.zeros((256,3));cost=0.;residuals=[]
    for band,tiles in enumerate(model.bands):
        for tile,item in enumerate(tiles):
            a0,b0,r,y0,factor,basis,saved,observed,sigma,days,thickness,wave,ice=item
            segment=np.searchsorted(RESETS,days,side="right")
            adjusted=thickness*np.exp(log_factors[tile,segment])
            shift=2*np.pi*ice*(adjusted-thickness)[:,None]/wave[None,:]
            c=np.cos(shift);s=-1j*np.sin(shift)
            a=c*a0+s*ice*b0;b=c*b0+s*a0/ice
            reflection=r*np.exp(theta[:mode]@basis+1j*(theta[mode:]@basis))
            y=y0*(1-reflection)/(1+reflection)
            dy=-2*y0*reflection/(1+reflection)**2
            den=a+b*y[None,:]
            weighted=factor[None,:]/np.abs(den)**2
            rate=weighted.sum(axis=1)
            log=np.log(rate)
            old=model.references[band][tile]
            prediction=saved+(log-log[0])-(old-old[0])
            normalized=(prediction-observed)/sigma
            cost+=.5*float(normalized@normalized);residuals.append(normalized)
            sensitivity=b*dy[None,:]/den
            derivatives=np.concatenate([(-2*weighted*sensitivity.real)@basis.T,
                                        (2*weighted*sensitivity.imag)@basis.T],axis=1)/rate[:,None]
            grad_theta+=(derivatives-derivatives[:1]).T@(normalized/sigma)
            phase=2*np.pi*ice*adjusted[:,None]/wave[None,:]
            dlog=(-2*weighted*np.real((-1j*(ice*b+y[None,:]*a/ice))*phase/den)).sum(axis=1)/rate
            source=normalized/sigma
            for stage in range(3):
                gradient_current=np.where(segment==stage,dlog,0.)
                grad_ice[tile,stage]+=np.dot(source,gradient_current-gradient_current[0])
    return cost,grad_theta,grad_ice,np.concatenate(residuals)


def stellar_ice_gradient(objective,coordinates,log_factors,segments):
    """Analytic derivative of Student-t negative log-likelihood w.r.t tile stages."""
    model=objective.model
    _,prediction,_,_,scores,scales=objective.state(coordinates,with_jacobian=False)
    scores_full=np.zeros((model.star_count*model.filter_count,model.pass_count))
    scores_full[objective.active_groups]=scores
    scores_full=scores_full.reshape(model.star_count,model.filter_count,model.pass_count)
    mode=model.mode_count;theta=coordinates[:-1]*objective.coordinate_scale
    ratio=np.exp(theta[:mode]@model.basis+1j*(theta[mode:2*mode]@model.basis))
    y0=model.incident_admittance_by_tile[:,None]
    reflect=model.clean_reflection_by_tile*ratio[None,:]
    y=y0*(1-reflect)/(1+reflect)
    forward=objective.forward
    sensitivity=np.zeros((model.star_count,model.filter_count,model.pass_count))
    for fi in range(model.filter_count):
        for pi in range(model.pass_count):
            a,b=forward.coefficients[fi][pi]
            den=a+b*y
            power=forward.power_numerator/np.abs(den)**2
            tile=model.source_tile_ids[:,fi,pi]
            kernel=forward.kernels[fi]*power[tile]
            phase=(2*np.pi*model.ice_index_by_tile[:,None]
                   *model.tile_thickness_by_id[:,fi,pi,None]/model.wavelengths_nm[None,:])
            derivative=(-2*np.real((-1j*(model.ice_index_by_tile[:,None]*b
                +y*a/model.ice_index_by_tile[:,None]))*phase/den))[tile]
            sensitivity[:,fi,pi]=(kernel*derivative).sum(axis=1)/kernel.sum(axis=1)
    grad=np.zeros((256,3))
    factor=1000./run.workflow.LOG_FLUX_PER_MAG
    for fi in range(model.filter_count):
        ref_tile=model.source_tile_ids[:,fi,0]
        ref_stage=segments[fi,0]
        for pi in range(model.pass_count):
            current_tile=model.source_tile_ids[:,fi,pi]
            stage=segments[fi,pi]
            contribution=factor*scores_full[:,fi,pi]
            np.add.at(grad,(current_tile,np.full(model.star_count,stage)),contribution*sensitivity[:,fi,pi])
            np.add.at(grad,(ref_tile,np.full(model.star_count,ref_stage)),-contribution*sensitivity[:,fi,0])
    return grad


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    area,_=run.canonical_configuration("WFI01")
    _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare("WFI01",write_inputs=False,
        effective_area_path=area,output_dir=OUTPUT,full_effective_area_support=True)
    add_full_basis(model,effective_area_support(area))
    with np.load(FIXED_OUTPUT/"joint_fit.npz",allow_pickle=False) as saved:
        parameters=saved["parameters"];np.testing.assert_array_equal(saved["stellar_valid"],valid)
    summary=json.loads((FIXED_OUTPUT/"joint_fit_summary.json").read_text())
    start_scale=summary["extra_student_t_scale_mmag"]
    original_thickness=model.tile_thickness_by_id.copy()
    from astropy.time import Time
    reference=Time(run.workflow.fit_model.ICE_REFERENCE_UTC,format="isot",scale="utc").mjd
    days=np.array([[common.loc[(common["filter"]==band)&(common["pass"]==epoch),"mjd_amber"].median()-reference
                    for epoch in run.PASSES] for band in run.FILTERS])
    segments=np.searchsorted(RESETS,days,side="right")
    objective=StudentObjective(model,magnitude,error,valid)
    led=LedModel(effective_area_support(area))
    nopt=len(parameters)+1
    initial=np.concatenate([parameters/objective.coordinate_scale,[np.log(start_scale)],np.zeros(256*3)])
    mode=model.mode_count
    bounds=[(-3.,3.)]*mode+[(-10.,10.)]*mode+[(-25.,25.)]*model.zero_point_count+[(np.log(1e-4),np.log(100.))]
    bounds[7]=bounds[8]=(-6.,6.)
    bounds += [(-MAX_LOG_FACTOR,MAX_LOG_FACTOR)]*(256*3)
    calls=0
    def total(x):
        nonlocal calls
        factors=x[nopt:].reshape(256,3)
        model.tile_thickness_by_id=original_thickness*np.exp(factors[:,segments])
        objective.forward=CachedPhotometryForward(model)
        stellar_value,stellar_gradient=objective(x[:nopt])
        theta=x[:2*mode]*objective.coordinate_scale[:2*mode]
        led_value,led_theta,led_ice,led_res=led_objective(led,theta,factors)
        stellar_ice=stellar_ice_gradient(objective,x[:nopt],factors,segments)
        prior=.5*np.sum((factors/PRIOR_SIGMA)**2)
        gradient=np.empty_like(x)
        gradient[:nopt]=stellar_gradient
        gradient[:2*mode]+=led_theta*objective.coordinate_scale[:2*mode]
        gradient[nopt:]=(led_ice+stellar_ice+factors/PRIOR_SIGMA**2).ravel()
        value=stellar_value+led_value+prior
        calls+=1
        if calls%10==0:print(f"eval {calls}: total={value:.4f}; LED chi2={2*led_value:.3f}; stellar={stellar_value:.4f}; max ice factor={np.exp(np.max(abs(factors))):.3f}",flush=True)
        return value,gradient
    value,gradient=total(initial)
    checks=[]
    for index in (0,model.mode_count,nopt,nopt+3*89+1,nopt+3*136+2):
        plus,minus=initial.copy(),initial.copy();plus[index]+=1e-4;minus[index]-=1e-4
        numerical=(total(plus)[0]-total(minus)[0])/2e-4
        discrepancy=abs(numerical-gradient[index])/(1+abs(numerical))
        checks.append(discrepancy)
        print(f"gradient {index}: analytic={gradient[index]:.7g}, finite={numerical:.7g}, relative discrepancy={discrepancy:.3g}",flush=True)
    if max(checks)>2e-3:raise RuntimeError("Joint per-tile ice gradient check failed")
    initial_led=led_objective(led,parameters[:2*mode],np.zeros((256,3)))
    # Explicit identity check: unchanged thickness must exactly reproduce the
    # previously verified detector-wide fixed-ice LED prediction.
    baseline=led.evaluate(parameters[:2*mode])[0]
    np.testing.assert_allclose(initial_led[3],baseline,atol=2e-10,rtol=1e-10)
    started=time.monotonic()
    result=minimize(total,initial,jac=True,bounds=bounds,method="L-BFGS-B",
                    options={"maxiter":600,"ftol":1e-10,"gtol":1e-5,"maxls":30})
    final_value,final_gradient=total(result.x)
    factors=result.x[nopt:].reshape(256,3)
    fitted=result.x[:nopt-1]*objective.coordinate_scale
    ledcost,_,_,led_residuals=led_objective(led,fitted[:2*mode],factors)
    report={"detector":"WFI01","tile_count":256,"ice_adjustment_parameters":768,
        "ice_adjustment":"independent multiplicative thickness factor per tile per decon segment; original shape and reset boundaries retained",
        "ice_log_factor_prior_sigma":PRIOR_SIGMA,"ice_factor_bounds":[.5,2.],
        "ice_prior_note":"independent zero-centered log-factor regularizer for this sensitivity experiment, not a posterior from the original LED fit",
        "LED_likelihood_multiplier":1.,"stellar_likelihood_multiplier":1.,
        "LED_chi2_before":float(initial_led[3]@initial_led[3]),"LED_chi2_after":float(led_residuals@led_residuals),
        "stellar_source_count":len(ids),"stellar_measurement_count":int(valid.sum()),
        "initial_objective":float(value),"final_objective":float(final_value),
        "extra_student_t_scale_mmag":float(np.exp(result.x[nopt-1])),
        "ice_factors_p16_median_p84":np.percentile(np.exp(factors),[16,50,84],axis=0).tolist(),
        "ice_parameters_at_bound":int(np.count_nonzero(np.isclose(np.abs(factors),MAX_LOG_FACTOR,atol=1e-5))),
        "optimizer_success":bool(result.success),"optimizer_message":str(result.message),"iterations":int(result.nit),
        "maximum_gradient_check_discrepancy":max(checks),"runtime_seconds":time.monotonic()-started,
        "production_bundle_changed":False}
    (OUTPUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(OUTPUT/"joint_per_tile_fit.npz",parameters=fitted,ice_log_factors=factors,valid=valid,source_ids=np.asarray(ids),
                        led_residuals=led_residuals)
    print(json.dumps(report,indent=2),flush=True)
    if not result.success:raise RuntimeError(f"Joint fit not converged: {result.message}")


if __name__=="__main__":main()
