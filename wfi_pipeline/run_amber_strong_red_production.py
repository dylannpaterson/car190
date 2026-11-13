"""Sequential Amber Student-t reruns with the strong red LED-SED soft mask."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
import run_amber_student_t_detectors as run
from experiment_led_sed_soft_mask import sed_mask
from experiment_support_aligned_endpoints import solve
from support_aligned_modulation import configure_basis,effective_area_support
from run_amber_support_aligned_production import directory as baseline_directory

RED_EXPONENT=.05


def directory(detector):
    old=baseline_directory(detector)
    return old.with_name(old.name+"_soft_led_sed_red_power_0p05")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",action="append",required=True,choices=run.workflow.DETECTORS)
    args=parser.parse_args()
    for detector in args.detector:
        old=baseline_directory(detector);output=directory(detector)
        summary_path=output/"production_fit_summary.json"
        if summary_path.is_file() and json.loads(summary_path.read_text()).get("optimizer_success"):
            print(f"{detector}: completed strong-red-mask fit already present",flush=True);continue
        area,_=run.canonical_configuration(detector)
        old_summary=json.loads((old/"fit_summary.json").read_text())
        with np.load(old/"predictions.npz",allow_pickle=False) as a:
            parameters=a["parameters"];old_valid=a["valid"]
        _,common,ids,magnitude,error,valid,_,model,_=run.prepare(detector,write_inputs=False,
            effective_area_path=area,output_dir=output,full_effective_area_support=True)
        np.testing.assert_array_equal(valid,old_valid)
        bounds=effective_area_support(area)
        configure_basis(model,bounds,free_endpoints=True)
        model.basis*=sed_mask(model.wavelengths_nm,red_exponent=RED_EXPONENT)[None,:]
        result=solve(model,magnitude,error,valid,parameters,old_summary["extra_student_t_scale_mmag"],.6,output)
        np.savez_compressed(output/"production_predictions.npz",source_ids=np.asarray(ids),valid=valid,
            magnitude=magnitude,error=error,parameters=result["theta"],prediction_mag=result["prediction"],
            wavelengths_nm=model.wavelengths_nm,basis=model.basis)
        summary=dict(result["summary"])
        summary.update({"detector":detector,"photometry":"Amber","likelihood":"Student-t","nu":4.,
            "used_stars":len(ids),"used_observations":int(valid.sum()),"LED_sed_red_exponent":RED_EXPONENT,
            "LED_sed_mask":"1 - maximum normalized LED SED height; LED 5/6 heights raised to 0.05",
            "led_sed_file":str(Path(production.SED_URL).resolve()),
            "led_sed_sha256":hashlib.sha256(Path(production.SED_URL).read_bytes()).hexdigest(),
            "effective_area_file":str(area),"effective_area_sha256":hashlib.sha256(area.read_bytes()).hexdigest(),
            "reference_fit":str(old),"optical_state_source":json.loads((run.paths(detector)[0]/"car201_actual_star_modulation_fit_summary.json").read_text())["ice_history"]["optical_state_source"],
            "selection":"same common Amber/Zane quality-selected repeated measurements as previous production fit",
            "ice_histories_refitted":False,"LED_integrated_response_guaranteed_unchanged":False})
        summary_path.write_text(json.dumps(summary,indent=2)+"\n")
        print(f"{detector}: complete; extra Student-t scale {summary['extra_student_t_scale_mmag']:.4f} mmag",flush=True)


if __name__=="__main__":main()
