"""Compare all canonical per-SCA refits with the historical SCA05-weighted fits."""

import json

import numpy as np
import pandas as pd

import run_amber_student_t_detectors as run
from check_amber_wfi01_photometry import profile_residual


def main():
    detector_records=[]
    filter_records=[]
    for detector in run.workflow.DETECTORS:
        _,old_dir=run.paths(detector)
        area,new_dir=run.canonical_configuration(detector)
        summaries=[json.loads((directory/"fit_summary.json").read_text()) for directory in (old_dir,new_dir)]
        if not summaries[1]["optimizer_success"]:raise RuntimeError(f"Unconverged {detector}")
        with np.load(old_dir/"predictions.npz",allow_pickle=False) as archive:
            old={name:archive[name] for name in ("source_ids","valid","student_t_prediction_mag")}
        with np.load(new_dir/"predictions.npz",allow_pickle=False) as archive:
            new={name:archive[name] for name in ("source_ids","valid","student_t_prediction_mag","magnitude","error")}
        np.testing.assert_array_equal(old["source_ids"],new["source_ids"])
        np.testing.assert_array_equal(old["valid"],new["valid"])
        valid=new["valid"]
        change=1000*profile_residual(new["student_t_prediction_mag"],old["student_t_prediction_mag"],new["error"],valid)
        x=change[valid]
        primary=valid&valid[:,:,:1]
        primary[:,:,0]=False
        fractions=[]
        for pred in (old["student_t_prediction_mag"],new["student_t_prediction_mag"]):
            corrected=new["magnitude"]-pred
            delta=(corrected-corrected[:,:,:1])[primary]
            flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*delta)
            fractions.append(float(np.mean(np.abs(flux)<=.5)))
        detector_records.append({"detector":detector,"used_stars":len(new["source_ids"]),
                                 "used_observations":int(valid.sum()),
                                 "previous_extra_scale_mmag":summaries[0]["extra_student_t_scale_mmag"],
                                 "detector_area_extra_scale_mmag":summaries[1]["extra_student_t_scale_mmag"],
                                 "prediction_change_rms_mmag":float(np.sqrt(np.mean(x*x))),
                                 "prediction_change_p95_absolute_mmag":float(np.quantile(np.abs(x),.95)),
                                 "ep3_reference_observations":int(primary.sum()),
                                 "previous_ep3_fraction_within_0p5_percent":fractions[0],
                                 "detector_area_ep3_fraction_within_0p5_percent":fractions[1],
                                 "effective_area_file":str(area),"effective_area_sha256":summaries[1]["effective_area_sha256"]})
        for fi,band in enumerate(run.FILTERS):
            values=change[:,fi][valid[:,fi]]
            filter_records.append({"detector":detector,"filter":band,
                                   "prediction_change_rms_mmag":float(np.sqrt(np.mean(values**2))),
                                   "prediction_change_p95_absolute_mmag":float(np.quantile(np.abs(values),.95))})
    result=pd.DataFrame(detector_records)
    output=run.workflow.PRODUCTION_ROOT/"wavelength_response_bundle"
    result.to_csv(output/"amber_detector_effective_area_fit_comparison.csv",index=False)
    pd.DataFrame(filter_records).to_csv(output/"amber_detector_effective_area_filter_comparison.csv",index=False)
    report={"converged_detectors":len(result),"fit_detector_star_pairs":int(result.used_stars.sum()),
            "fit_observations":int(result.used_observations.sum()),
            "median_extra_student_t_scale_mmag":float(result.detector_area_extra_scale_mmag.median()),
            "min_extra_student_t_scale_mmag":float(result.detector_area_extra_scale_mmag.min()),
            "max_extra_student_t_scale_mmag":float(result.detector_area_extra_scale_mmag.max()),
            "median_detector_prediction_change_rms_mmag":float(result.prediction_change_rms_mmag.median()),
            "max_detector_prediction_change_rms_mmag":float(result.prediction_change_rms_mmag.max()),
            "previous_ep3_fraction_within_0p5_percent":float(np.average(result.previous_ep3_fraction_within_0p5_percent,weights=result.ep3_reference_observations)),
            "detector_area_ep3_fraction_within_0p5_percent":float(np.average(result.detector_area_ep3_fraction_within_0p5_percent,weights=result.ep3_reference_observations)),
            "comparison":"same stars and masks; time-dependent prediction changes after removing a common catalog-error-weighted constant per star/filter"}
    (output/"amber_detector_effective_area_fit_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(result[["detector","detector_area_extra_scale_mmag","prediction_change_rms_mmag"]].to_string(index=False))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
