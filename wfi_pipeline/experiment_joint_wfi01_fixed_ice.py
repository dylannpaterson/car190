"""Joint CAR190 + Amber detector fit with fixed per-tile ice histories.

CAR190 uses the production Gaussian group-weight convention; Amber uses the
existing Student-t likelihood. No arbitrary data-set reweighting is applied.
"""
import json
import argparse
from pathlib import Path
import os
import time

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_amber_student_t_detectors as run
import fit_wfi_tmm_production as production
import compare_wfi_tile136_complex_reflectance as complex_model
import mock_fit_wfi_star_reflectance_modulation as mock
from fit_amber_wfi01_student_t import StudentObjective
from experiment_free_led_modulation import free_value_basis
from support_aligned_modulation import effective_area_support,configure_basis,gap_basis
from run_amber_strong_red_production import directory as old_directory

ROOT=Path(__file__).resolve().parents[1]
PRODUCTION=run.workflow.PRODUCTION_ROOT
OUTPUT=PRODUCTION/"wfi01/spline_relative_response_normal/experiments/car190_amber_joint_fixed_ice_full_reflection_test"
NLED=6
TILE_COUNT=256


def fixed_output(detector):
    return PRODUCTION/f"{detector.lower()}/spline_relative_response_normal/experiments/car190_amber_joint_fixed_ice_full_reflection_test"


def full_basis(wave,bounds):
    gaps,_=gap_basis(wave,bounds)
    values,knots,weights,groups=free_value_basis(wave,boundaries=bounds)
    # Order gap, endpoints, interior LEDs, preserving the canonical 9 modes.
    order=[0,len(knots)-1]+list(range(1,len(knots)-1))
    return np.concatenate([gaps,values[order]]),knots,weights,groups,order


def add_full_basis(model,bounds):
    configure_basis(model,bounds,free_endpoints=True)
    base_count=model.mode_count
    basis,knots,weights,support,order=full_basis(model.wavelengths_nm,bounds)
    model.basis=basis
    model.basis_names += [f"free_LED_{i}" for i in range(1,7)]
    model.basis_centers_nm=np.concatenate([model.basis_centers_nm,knots[1:-1]])
    model.node_weights=np.concatenate([model.node_weights,weights[1:-1]])
    model.node_groups += [np.array([base_count+i]) for i in range(6)]
    model.group_bounds += [support[i] for i in range(1,7)]
    model.mode_count=len(basis)


class LedModel:
    def __init__(self,bounds,detector="WFI01",relative_response_root=None,
                 led_wavelength_nm=None,led_weights=None,optical_cache_root=None):
        self.detector=detector
        detector_root=(Path(relative_response_root) if relative_response_root is not None else
                       PRODUCTION/detector.lower()/"spline_relative_response_normal")
        if (led_wavelength_nm is None) != (led_weights is None):
            raise ValueError("LED quadrature wavelengths and weights must be supplied together")
        if led_wavelength_nm is None:
            ctx=production.load_response_context([detector],sed_weighting="energy",qe_response_mode="none")
            self.wave=np.asarray(ctx["wavelengths"],float)
            self.weights=np.asarray(ctx["model_weights"],float)[0]
        else:
            nodes=np.asarray(led_wavelength_nm,float)
            values=np.asarray(led_weights,float)
            if nodes.ndim!=2 or values.shape!=nodes.shape or nodes.shape[0]!=NLED:
                raise ValueError("Eric LED quadrature must have matching (6, node_count) arrays")
            self.wave,inverse=np.unique(nodes.reshape(-1),return_inverse=True)
            self.weights=np.zeros((NLED,self.wave.size),float)
            inverse=inverse.reshape(nodes.shape)
            for band in range(NLED):
                np.add.at(self.weights[band],inverse[band],values[band])
        self.basis=full_basis(self.wave,bounds)[0]
        nbase,kbase=production.hgcdte_reference_optical_constants(self.wave)
        spectral=production.spectral_modulation_basis(self.wave)
        self.bands=[]
        self.clean_reflection_by_tile=np.empty((TILE_COUNT,self.wave.size),dtype=complex)
        led_rows=[]
        by_band=[[] for _ in range(NLED)]
        for tile in range(TILE_COUNT):
            directory=detector_root/f"fits/tile_{tile:03d}"
            prediction=pd.read_csv(directory/"tile_group_predictions.csv")
            stats=pd.read_csv(detector_root/f"tile_statistics/wfi{detector[3:]}_tile_log_statistics_256_tile_{tile:03d}.csv")
            with np.load(directory/"posterior_samples.npz",allow_pickle=False) as a:
                d=np.asarray(a["tile_ice_thickness"],float).reshape(-1,len(prediction))[0]
                cache=(Path(str(np.asarray(a["response_cache"]).reshape(-1)[0]))
                       if "response_cache" in a else None)
                time_days=np.asarray(a["observation_days"],float).reshape(-1)
            if not np.array_equal(prediction["group"],stats["group"]):raise ValueError("CAR190 group order mismatch")
            np.testing.assert_allclose(prediction.days,time_days,atol=1e-10,rtol=0)
            if optical_cache_root is not None:
                cache=Path(optical_cache_root)/f"tile_{tile:03d}/relative_response.npz"
            if cache is not None:
                source=mock.source_fit_from_cache(cache,detector,tile)
            else:
                # Full-TMM production archives contain the fitted optical state
                # directly and need no six-LED cache for this in-memory model.
                source=directory/"posterior_samples.npz"
            state=complex_model.load_state(source,self.wave,nbase,kbase,spectral)
            first,second,y0,ysub,sine=mock.clean_stack_boundary(state,self.wave)
            if abs(sine)>1e-12:raise ValueError("Non-normal incidence")
            reflect=(y0*first-second)/(y0*first+second)
            self.clean_reflection_by_tile[tile]=reflect
            counts=stats["n_pair_valid"].to_numpy(float)
            sigma=.005*np.sqrt(np.median(counts)/counts)
            for band in range(1,7):
                mask=prediction.led.to_numpy()==band
                supported=self.weights[band-1]>0
                wave=self.wave[supported]; weight=self.weights[band-1,supported]
                ice=state["ice_index"]
                phase=2*np.pi*ice*d[mask,None]/wave[None,:]
                cosine=np.cos(phase); sinephase=-1j*np.sin(phase)
                a=first[None,supported]*(y0*cosine+sinephase*ice)
                b=first[None,supported]*(y0*sinephase/ice+cosine)
                factor=np.real(ysub[supported]/y0)*abs(2*y0)**2*weight
                by_band[band-1].append((a.astype(np.complex64),b.astype(np.complex64),
                    reflect[supported],float(y0),factor,self.basis[:,supported],
                    prediction.loc[mask,"model_log_change"].to_numpy(float),
                    prediction.loc[mask,"observed_log_change"].to_numpy(float),sigma[mask],
                    prediction.loc[mask,"days"].to_numpy(float),d[mask],wave,float(ice)))
            if (tile+1)%64==0:print(f"Prepared CAR190 tiles {tile+1}/256",flush=True)
        self.bands=by_band
        self.references=[]
        zero=np.zeros(2*self.basis.shape[0])
        _,_,rates=self.evaluate(zero,subtract_reference=False)
        self.references=rates

    def evaluate(self,theta,subtract_reference=True,thickness_log_scales=None):
        mode=self.basis.shape[0]
        scores=[]; jacobians=[]; raw_rates=[]
        for band,tiles in enumerate(self.bands):
            current=[]; jac=[]; raw=[]
            for item in tiles:
                a,b,r,y0,factor,basis,saved,observed,sigma,days=item[:10]
                if thickness_log_scales is not None:
                    thickness,wavelength,ice=item[10:]
                    segment=np.searchsorted([12.907397928240743,38.79784510416666],days,side="right")
                    scales=np.asarray(thickness_log_scales)
                    local_scale=scales[len(current)] if scales.ndim==2 else scales
                    extra_phase=(2*np.pi*ice*thickness[:,None]
                                 *np.expm1(local_scale[segment,None])
                                 /wavelength[None,:])
                    c=np.cos(extra_phase);s=-1j*np.sin(extra_phase)
                    a,b=c*a+s*ice*b,c*b+s*a/ice
                ratio=np.exp(theta[:mode]@basis+1j*(theta[mode:2*mode]@basis))
                reflected=r*ratio
                y=y0*(1-reflected)/(1+reflected)
                dy=-2*y0*reflected/(1+reflected)**2
                den=a+b*y[None,:]
                weighted=factor[None,:]/np.abs(den)**2
                rate=weighted.sum(axis=1)
                log=np.log(rate)
                sensitivity=b*dy[None,:]/den
                derivative=np.concatenate([(-2*weighted*sensitivity.real)@basis.T,
                                           (2*weighted*sensitivity.imag)@basis.T],axis=1)/rate[:,None]
                raw.append(log)
                if subtract_reference:
                    index=len(current)
                    old=self.references[band][index]
                    delta=(log-log[0])-(old-old[0])
                    res=(saved+delta-observed)/sigma
                    current.append(res)
                    jac.append((derivative-derivative[:1])/sigma[:,None])
            raw_rates.append(raw)
            if subtract_reference:
                scores.extend(current);jacobians.extend(jac)
        if not subtract_reference:return None,None,raw_rates
        return np.concatenate(scores),np.concatenate(jacobians),raw_rates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",required=True,choices=run.workflow.DETECTORS)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--maxiter",type=int,default=700)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--initial-fit-dir",type=Path,
                        help="Warm start from a completed joint fixed-ice fit")
    parser.add_argument("--ice-history-fit-root",type=Path,
                        help="Per-tile posterior_samples.npz directory")
    parser.add_argument("--optical-cache-root",type=Path,
                        help="Pinned per-tile relative-response cache directory")
    parser.add_argument("--relative-response-root",type=Path,
                        help="New production relative_response directory for CAR190")
    args=parser.parse_args()
    detector=args.detector
    output=args.output_dir or fixed_output(detector)
    output.mkdir(parents=True,exist_ok=True)
    area,_=run.canonical_configuration(detector,output_dir=output)
    if args.initial_fit_dir is None:
        prior=old_directory(detector)
        with np.load(prior/"production_predictions.npz",allow_pickle=False) as a:
            prior_theta=a["parameters"];prior_valid=a["valid"]
        prior_summary=json.loads((prior/"production_fit_summary.json").read_text())
        warm_start_is_joint=False
    else:
        prior=Path(args.initial_fit_dir)
        with np.load(prior/"joint_fit.npz",allow_pickle=False) as a:
            prior_theta=a["parameters"];prior_valid=a["stellar_valid"]
        prior_summary=json.loads((prior/"joint_fit_summary.json").read_text())
        warm_start_is_joint=True
    _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare(detector,write_inputs=False,
        effective_area_path=area,output_dir=output,full_effective_area_support=True,
        ice_history_fit_root=args.ice_history_fit_root,optical_cache_root=args.optical_cache_root)
    np.testing.assert_array_equal(prior_valid,valid)
    bounds=effective_area_support(area)
    add_full_basis(model,bounds)
    objective=StudentObjective(model,magnitude,error,valid)
    mode=model.mode_count
    if warm_start_is_joint:
        theta=np.asarray(prior_theta,dtype=float)
        if theta.shape!=objective.coordinate_scale.shape:
            raise ValueError("Joint warm-start parameter dimensions do not match the current model")
    else:
        # Expand [7 gaps + 2 endpoints] by six LED-value modes.
        theta=np.concatenate([prior_theta[:9],np.zeros(6),prior_theta[9:18],np.zeros(6),prior_theta[18:]])
    initial=np.concatenate([theta/objective.coordinate_scale,[np.log(prior_summary["extra_student_t_scale_mmag"])]])
    if args.resume:
        saved=np.load(output/"joint_fit.npz",allow_pickle=False)
        saved_summary=json.loads((output/"joint_fit_summary.json").read_text())
        initial=np.concatenate([saved["parameters"]/objective.coordinate_scale,
                                [np.log(saved_summary["extra_student_t_scale_mmag"])]])
    led=LedModel(bounds,detector=detector,relative_response_root=args.relative_response_root)
    weights=np.r_[objective.coordinate_scale[:2*mode],np.zeros(model.zero_point_count+1)]
    calls=0
    def total(coordinates):
        nonlocal calls
        star_value,star_gradient=objective(coordinates)
        physical=coordinates[:2*mode]*objective.coordinate_scale[:2*mode]
        led_res,led_jac,_=led.evaluate(physical)
        value=star_value+.5*float(led_res@led_res)
        gradient=star_gradient.copy()
        gradient[:2*mode]+=(led_jac.T@led_res)*objective.coordinate_scale[:2*mode]
        calls+=1
        if calls%10==0:print(f"evaluation {calls}: joint={value:.4f}; CAR190 chi2={led_res@led_res:.3f}; stellar objective={star_value:.3f}",flush=True)
        return value,gradient
    initial_value,initial_gradient=total(initial)
    checks=[]
    for index in (0,9,mode+9,len(initial)-1):
        h=1e-4;upper,lower=initial.copy(),initial.copy();upper[index]+=h;lower[index]-=h
        numerical=(total(upper)[0]-total(lower)[0])/(2*h)
        checks.append(abs(numerical-initial_gradient[index])/(1+abs(numerical)))
    if max(checks)>1e-2:raise RuntimeError(f"Joint gradient check failed: {checks}")
    fit_bounds=[(-3.,3.)]*mode+[(-10.,10.)]*mode+[(-25.,25.)]*model.zero_point_count+[(np.log(1e-4),np.log(100.))]
    for index in (7,8):fit_bounds[index]=(-6.,6.)
    started=time.monotonic()
    result=minimize(total,initial,method="L-BFGS-B",jac=True,bounds=fit_bounds,
                    options={"maxiter":args.maxiter,"ftol":1e-9,"gtol":1e-4,"maxls":30})
    final_value,final_gradient=total(result.x)
    star_value,prediction,_,_,_,_=objective.state(result.x)
    physical=result.x[:-1]*objective.coordinate_scale
    led_res,_,_=led.evaluate(physical[:2*mode])
    start_led,_,_=led.evaluate(theta[:2*mode])
    led_chi2=float(led_res@led_res);initial_led_chi2=float(start_led@start_led)
    summary={"detector":detector,"tile_count":TILE_COUNT,"stellar_sources":len(ids),"stellar_measurements":int(valid.sum()),
        "LED_observations":len(led_res),"ice_histories_refitted":False,"modulation":"unmasked free LED and endpoint complex reflection",
        "ice_history_fit_root":str(args.ice_history_fit_root) if args.ice_history_fit_root else None,
        "optical_cache_root":str(args.optical_cache_root) if args.optical_cache_root else None,
        "relative_response_root":str(args.relative_response_root) if args.relative_response_root else None,
        "initial_fit_directory":str(prior),
        "LED_weight":"existing CAR190 production: sigma_log=0.005*sqrt(median_pixel_count/current_pixel_count)",
        "stellar_likelihood":"Student-t nu=4, detector-level extra scale, robust star/filter levels",
        "dataset_likelihood_multiplier":1.,"CAR190_offsets":"fixed from each tile's existing CAR190 fit",
        "initial_LED_chi2":initial_led_chi2,"final_LED_chi2":led_chi2,"final_stellar_objective_including_regularization":star_value,
        "initial_total_objective":initial_value,"final_total_objective":final_value,
        "extra_student_t_scale_mmag":float(np.exp(result.x[-1])),"optimizer_success":bool(result.success),
        "optimizer_message":str(result.message),"iterations":int(result.nit),"seconds":time.monotonic()-started,
        "max_gradient_check_relative_difference":max(checks)}
    (output/"joint_fit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(output/"joint_fit.npz",parameters=physical,stellar_prediction_mag=prediction/1000.,
                        led_normalized_residual=led_res,led_initial_normalized_residual=start_led,
                        stellar_valid=valid,source_ids=np.asarray(ids),wavelength_nm=model.wavelengths_nm,basis=model.basis)
    pd.DataFrame({"mode":model.basis_names,"log_amplitude":physical[:mode],"phase_rad":physical[mode:2*mode]}).to_csv(output/"joint_modulation_parameters.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)
    if not result.success:raise RuntimeError(result.message)


if __name__=="__main__":main()
