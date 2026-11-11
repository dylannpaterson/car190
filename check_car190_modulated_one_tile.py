"""Compare fixed-history CAR190 data before/after Amber modulation for one tile."""
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

import build_modulated_wfi_wavelength_response_bundle as builder
import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production
import mock_fit_wfi_star_reflectance_modulation as mock

ROOT=Path(__file__).resolve().parent
PRODUCTION=ROOT/"20261106/corrected_statistics_4h_thinned/production"
OUTPUT=PRODUCTION/"wfi01/spline_relative_response_normal/experiments/car190_tile136_amber_modulation_check"
TILE=136


def main():
    global OUTPUT,TILE
    parser=argparse.ArgumentParser()
    parser.add_argument("--soft-led-sed-test",action="store_true")
    parser.add_argument("--red-exponent",type=float,default=1.)
    parser.add_argument("--detector",choices=[f"WFI{i:02d}" for i in range(1,19)],default="WFI01")
    parser.add_argument("--tile",type=int,default=136)
    args=parser.parse_args()
    if not 0<=args.tile<256:parser.error("tile must be in [0,255]")
    TILE=args.tile
    detector=args.detector
    OUTPUT=PRODUCTION/f"{detector.lower()}/spline_relative_response_normal/experiments/car190_tile{TILE:03d}_amber_modulation_check"
    if args.soft_led_sed_test:
        suffix="_soft_led_sed" if args.red_exponent==1. else f"_soft_led_sed_red_power_{args.red_exponent:g}".replace(".","p")
        OUTPUT=OUTPUT.with_name(OUTPUT.name+suffix)
    fit_dir=PRODUCTION/f"{detector.lower()}/spline_relative_response_normal/fits/tile_{TILE:03d}"
    rows=pd.read_csv(fit_dir/"tile_group_predictions.csv")
    with np.load(fit_dir/"posterior_samples.npz",allow_pickle=False) as a:
        thickness=np.asarray(a["tile_ice_thickness"],float).reshape(-1,len(rows))[0]
        days=np.asarray(a["observation_days"],float).reshape(-1)
        led=np.asarray(a["observation_led"],int).reshape(-1)
        cache=Path(str(np.asarray(a["response_cache"]).reshape(-1)[0]))
    np.testing.assert_allclose(rows.days,days,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(rows.led,led)
    with np.load(cache,allow_pickle=False) as a:
        source=Path(str(np.asarray(a["source_fit"]).reshape(-1)[0]))
        grid=np.asarray(a["grid_nm"],float)
        curves=np.asarray(a["relative_log_response"],float)
    context=production.load_response_context([detector],sed_weighting="energy",qe_response_mode="none")
    wave=np.asarray(context["wavelengths"],float)
    weights=np.asarray(context["model_weights"],float)[0]
    base_n,base_k=production.hgcdte_reference_optical_constants(wave)
    state=complex_model.load_state(source,wave,base_n,base_k,production.spectral_modulation_basis(wave))
    first,second,y0,ysub,sine=mock.clean_stack_boundary(state,wave)
    if abs(sine)>1e-12:raise ValueError("Diagnostic requires the production normal-incidence state")
    reflection=(y0*first-second)/(y0*first+second)
    modulation=builder._load_detector_modulation(PRODUCTION,detector,wave,fit_kind="amber-student-t-support-aligned")
    if args.soft_led_sed_test:
        from run_amber_support_aligned_production import directory
        from support_aligned_modulation import gap_basis
        from experiment_free_led_modulation import free_value_basis
        from experiment_led_sed_soft_mask import sed_mask
        suffix="_soft_led_sed_test" if args.red_exponent==1. else f"_soft_led_sed_red_power_{args.red_exponent:g}".replace(".","p")
        experiment=directory(detector).with_name(directory(detector).name+suffix)
        prediction_path=experiment/"soft_mask_definition.npz"
        if not prediction_path.exists():prediction_path=experiment/"production_predictions.npz"
        with np.load(prediction_path,allow_pickle=False) as data:
            theta=data["parameters"]
            bounds=tuple(data["bounds"]) if "bounds" in data else tuple(json.loads((experiment/"production_fit_summary.json").read_text())["modulation_boundaries_nm"])
            if "red_exponent" in data and not np.isclose(float(data["red_exponent"]),args.red_exponent):
                raise ValueError("Saved soft-mask exponent does not match requested experiment")
        gaps,_=gap_basis(wave,bounds); extra,*_=free_value_basis(wave,boundaries=bounds)
        basis=np.concatenate([gaps,extra[[0,-1]]])*sed_mask(wave,red_exponent=args.red_exponent)[None,:]
        count=len(basis)
        modulation["ratio"]=np.exp(theta[:count]@basis+1j*(theta[count:2*count]@basis))
    results={}
    for name,ratio in (("baseline",np.ones_like(wave,dtype=complex)),("modulated",modulation["ratio"])):
        reflected=reflection*ratio
        modified_y=y0*(1-reflected)/(1+reflected)
        boundary_second=first*modified_y
        phase=2*np.pi*state["ice_index"]*thickness[:,None]/wave[None,:]
        cosine=np.cos(phase); sinephase=-1j*np.sin(phase)
        denominator=y0*(cosine*first[None,:]+sinephase*boundary_second[None,:]/state["ice_index"])
        denominator+=sinephase*state["ice_index"]*first[None,:]+cosine*boundary_second[None,:]
        power=np.real(ysub/y0)[None,:]*np.abs(2*y0/denominator)**2
        count=np.sum(power*weights[led-1],axis=1)
        if np.any(count<=0) or not np.isfinite(count).all():raise ValueError("Invalid LED integral")
        log_count=np.log(count)
        for band in range(1,7):
            mask=led==band
            log_count[mask]-=log_count[np.flatnonzero(mask)[0]]
        results[name]=log_count
    correction=results["modulated"]-results["baseline"]
    rows["ice_thickness_nm"]=thickness
    rows["modulated_model_log_change"]=rows["model_log_change"]+correction
    rows["modulated_minus_baseline_log"]=correction
    rows["modulated_residual_log"]=rows["observed_log_change"]-rows["modulated_model_log_change"]
    rows["direct_unmodulated_relative_log"]=results["baseline"]
    cached=np.empty(len(rows))
    reference=np.zeros(len(rows),bool)
    for band in range(1,7):
        mask=led==band; indices=np.flatnonzero(mask); reference[indices[0]]=True
        raw=np.interp(thickness[mask],grid,curves[band-1])
        cached[mask]=raw-raw[0]
    rows["cached_baseline_relative_log"]=cached
    cache_error=float(np.max(np.abs(results["baseline"]-cached)))
    rows["is_led_reference"]=reference
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows.to_csv(OUTPUT/"car190_tile136_modulation_comparison.csv",index=False)
    bands=[]
    for band in range(1,7):
        selected=rows.loc[(rows.led==band)&~rows.is_led_reference]
        bands.append({"led":band,"observations":len(selected),
                      "baseline_residual_rms_percent":100*np.sqrt(np.mean(selected.residual_log**2)),
                      "modulated_residual_rms_percent":100*np.sqrt(np.mean(selected.modulated_residual_log**2)),
                      "baseline_residual_median_percent":100*np.median(selected.residual_log),
                      "modulated_residual_median_percent":100*np.median(selected.modulated_residual_log),
                      "response_change_rms_percent":100*np.sqrt(np.mean(selected.modulated_minus_baseline_log**2)),
                      "response_change_max_abs_percent":100*np.max(np.abs(selected.modulated_minus_baseline_log))})
    pd.DataFrame(bands).to_csv(OUTPUT/"car190_tile136_led_summary.csv",index=False)
    reference_mjd=Time("2026-09-12T01:25:50.714",format="isot",scale="utc").mjd
    dates=Time(reference_mjd+days,format="mjd",scale="utc").to_datetime()
    fig,axes=plt.subplots(3,2,figsize=(15,11),sharex=True)
    for band,axis in enumerate(axes.flat,start=1):
        mask=(led==band)&~reference
        axis.scatter(dates[mask],100*rows.loc[mask,"observed_log_change"],s=7,color="0.45",alpha=.5,label="CAR190 corrected data")
        axis.plot(dates[mask],100*rows.loc[mask,"model_log_change"],color="#357fa1",linewidth=1.1,label="Original CAR190 model")
        axis.plot(dates[mask],100*rows.loc[mask,"modulated_model_log_change"],color="#d47727",linewidth=1.1,label="Amber-modulated response")
        for event in ("2026-10-17T03:30:00","2026-10-17T23:00:00","2026-10-30T01:43:00"):
            axis.axvline(Time(event,format="isot",scale="utc").to_datetime(),color="0.65",linestyle=":",linewidth=.6)
        axis.set_title(f"LED {band}");axis.grid(alpha=.2)
    for axis in axes[:,0]:axis.set_ylabel("Log response change ×100 (%)")
    for axis in axes[-1]:axis.set_xlabel("UTC date")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.95),ncol=3)
    fig.suptitle(f"{detector} tile {TILE}: CAR190 data versus Amber-modulated optical response",y=.99)
    fig.text(.5,.025,"Same fixed CAR190 ice history, LED offsets and reset offsets in both models; only complex reflection is modulated.\n"
             "LED energy spectra are integrated over wavelength. Points are corrected CAR190 observations; vertical dotted lines mark temperature events.",ha="center",fontsize=9)
    fig.tight_layout(rect=(.03,.08,1,.90));fig.savefig(OUTPUT/"car190_tile136_modulation_comparison.png",dpi=180);plt.close(fig)
    if cache_error>5e-4:raise RuntimeError(f"Independent baseline does not reproduce pinned LED cache: {cache_error}")
    report={"detector":detector,"tile_id":TILE,"input_fit":str(fit_dir),"input_optics":str(source),
            "ice_history_refit":False,"nuisance_offsets_refit":False,"sed_weighting":"energy","reference_qe_applied":False,
            "max_direct_vs_pinned_LED_cache_abs_log_difference":cache_error,
            "baseline_residual_rms_percent":float(100*np.sqrt(np.mean(rows.loc[~reference,"residual_log"]**2))),
            "modulated_residual_rms_percent":float(100*np.sqrt(np.mean(rows.loc[~reference,"modulated_residual_log"]**2))),
            "model_change_rms_percent":float(100*np.sqrt(np.mean(correction[~reference]**2))),
            "model_change_max_abs_percent":float(100*np.max(np.abs(correction[~reference]))),
            "interpretation":"100 times log response, approximately percent for small differences; fixed offsets inherited from original CAR190 fit"}
    (OUTPUT/"car190_tile136_modulation_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2));print(pd.DataFrame(bands).to_string(index=False));print(OUTPUT)


if __name__=="__main__":main()
