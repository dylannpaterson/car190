"""Test suppressing reflection modulation in proportion to LED spectral intensity."""
import json
import argparse
import numpy as np
import pandas as pd

import run_amber_student_t_detectors as run
import fit_wfi_tmm_production as production
from support_aligned_modulation import configure_basis,effective_area_support
from experiment_support_aligned_endpoints import solve
from fit_amber_wfi01_student_t import profile_student_locations
from experiment_free_led_modulation import metrics


def sed_mask(wavelengths,spectra=None,red_exponent=1.):
    spectra=production.load_sed(production.SED_URL) if spectra is None else spectra
    wave=np.asarray(wavelengths,float)
    normalized=[]
    for band in range(1,7):
        source=spectra[band]
        values=np.maximum(np.asarray(source["raw_output"],float),0.)
        if np.max(values)<=0:raise ValueError("LED spectrum has no positive signal")
        relative=np.interp(wave,source["wavelength"],values/np.max(values),left=0.,right=0.)
        normalized.append(relative**red_exponent if band in (5,6) else relative)
    return np.clip(1.-np.max(normalized,axis=0),0.,1.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--red-exponent",type=float,default=1.,help="Power on normalized LED 5/6 SED; values below one suppress more strongly")
    args=parser.parse_args()
    if not 0 < args.red_exponent <= 1:parser.error("red-exponent must be in (0,1]")
    from run_amber_support_aligned_production import directory
    detector="WFI01"; baseline=directory(detector)
    suffix="_soft_led_sed_test" if args.red_exponent==1. else f"_soft_led_sed_red_power_{args.red_exponent:g}".replace(".","p")
    output=baseline.with_name(baseline.name+suffix)
    area,_=run.canonical_configuration(detector)
    summary=json.loads((baseline/"fit_summary.json").read_text())
    with np.load(baseline/"predictions.npz",allow_pickle=False) as archive:
        old_prediction=archive["prediction_mag"];theta=archive["parameters"];old_valid=archive["valid"]
    _,common,ids,magnitude,error,valid,_,model,_=run.prepare(detector,write_inputs=False,
        effective_area_path=area,output_dir=output,full_effective_area_support=True)
    np.testing.assert_array_equal(valid,old_valid)
    bounds=effective_area_support(area);configure_basis(model,bounds,free_endpoints=True)
    model.basis*=sed_mask(model.wavelengths_nm,red_exponent=args.red_exponent)[None,:]
    result=solve(model,magnitude,error,valid,theta,summary["extra_student_t_scale_mmag"],.6,output)
    np.savez_compressed(output/"soft_mask_definition.npz",wavelengths_nm=model.wavelengths_nm,
                        mask=sed_mask(model.wavelengths_nm,red_exponent=args.red_exponent),parameters=result["theta"],
                        bounds=np.asarray(bounds),red_exponent=np.asarray(args.red_exponent))
    summary_path=output/"fit_summary.json"
    fit_summary=json.loads(summary_path.read_text())
    fit_summary.update({"LED_sed_red_exponent":args.red_exponent,
                        "LED_sed_mask":"1 - maximum of normalized LED SED heights; LEDs 5 and 6 raised to red exponent",
                        "reference_fit":str(baseline)})
    summary_path.write_text(json.dumps(fit_summary,indent=2)+"\n")
    active=valid.reshape(-1,valid.shape[-1]).sum(axis=1)>=2;mask=valid.reshape(-1,valid.shape[-1])[active]
    scales=np.hypot(error.reshape(-1,valid.shape[-1])[active]*1000.,summary["extra_student_t_scale_mmag"])
    comparisons=[]
    for label,pred in (("current",old_prediction),("LED-SED soft mask",result["prediction"])):
        group=(magnitude-pred).reshape(-1,valid.shape[-1])[active]*1000.
        _,residual,_=profile_student_locations(group,scales,mask)
        comparisons.append({"variant":label,**metrics(residual[mask])})
    pd.DataFrame(comparisons).to_csv(output/"stellar_residual_comparison.csv",index=False)
    print(pd.DataFrame(comparisons).to_string(index=False));print(output)


if __name__=="__main__":main()
