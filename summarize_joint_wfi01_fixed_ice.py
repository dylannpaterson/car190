"""Compare joint WFI01 fit to the previous stellar-only masked fit."""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_joint_wfi01_fixed_ice import OUTPUT,LedModel
from experiment_led_sed_soft_mask import sed_mask
from fit_amber_wfi01_student_t import profile_student_locations
from run_amber_strong_red_production import directory
from support_aligned_modulation import effective_area_support
import run_amber_student_t_detectors as run


def main():
    previous=directory("WFI01")
    with np.load(previous/"production_predictions.npz",allow_pickle=False) as archive:
        old={k:archive[k] for k in archive.files}
    with np.load(OUTPUT/"joint_fit.npz",allow_pickle=False) as archive:
        new={k:archive[k] for k in archive.files}
    np.testing.assert_array_equal(old["valid"],new["stellar_valid"])
    np.testing.assert_array_equal(old["source_ids"],new["source_ids"])
    valid=old["valid"]; error=old["error"]; magnitude=old["magnitude"]
    old_summary=json.loads((previous/"production_fit_summary.json").read_text())
    new_summary=json.loads((OUTPUT/"joint_fit_summary.json").read_text())
    area,_=run.canonical_configuration("WFI01")
    led=LedModel(effective_area_support(area))
    mode=led.basis.shape[0]
    joint=led.evaluate(new["parameters"][:2*mode])[0]
    # Evaluate exactly the previous stellar-only SED mask at the raw CAR190
    # wavelengths, rather than interpolating the 2-nm export.
    original_bands=led.bands
    masked=[]
    for band,tiles in enumerate(original_bands):
        mask=sed_mask(led.wave[led.weights[band]>0],red_exponent=.05)
        replacement=[]
        for item in tiles:
            row=list(item);row[5]=row[5]*mask[None,:];replacement.append(tuple(row))
        masked.append(replacement)
    led.bands=masked
    oldtheta=old["parameters"]
    expanded=np.concatenate([oldtheta[:9],np.zeros(6),oldtheta[9:18],np.zeros(6)])
    old_led=led.evaluate(expanded)[0]
    led.bands=original_bands
    sigmas=[];bands=[];tileids=[]
    for band,tiles in enumerate(led.bands,start=1):
        for tile,item in enumerate(tiles):
            sigmas.extend(item[8]);bands.extend([band]*len(item[8]));tileids.extend([tile]*len(item[8]))
    sigma=np.asarray(sigmas);bands=np.asarray(bands);tileids=np.asarray(tileids)
    if len(joint)!=len(sigma):raise ValueError("CAR190 group alignment error")
    ledger=[]
    for band in range(1,7):
        select=bands==band
        for name,values in (("previous masked",old_led),("joint",joint)):
            residual=values[select]*sigma[select]
            ledger.append({"LED":band,"model":name,"group_count":int(select.sum()),
                           "residual_rms_percent":float(100*np.sqrt(np.mean(residual**2))),
                           "CAR190_chi2":float(np.sum(values[select]**2))})
    led_summary=pd.DataFrame(ledger)
    led_summary.to_csv(OUTPUT/"joint_vs_previous_car190_metrics.csv",index=False)
    active=valid.reshape(-1,valid.shape[-1]).sum(axis=1)>=2
    mask=valid.reshape(-1,valid.shape[-1])[active]
    scale=np.hypot(error.reshape(-1,valid.shape[-1])[active]*1000.,old_summary["extra_student_t_scale_mmag"])
    rows=[]
    for name,prediction in (("previous masked",old["prediction_mag"]),("joint",new["stellar_prediction_mag"])):
        y=(magnitude-prediction).reshape(-1,valid.shape[-1])[active]*1000.
        _,r,_=profile_student_locations(y,scale,mask)
        flux=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*r[mask]/1000.)
        rows.append({"model":name,"observations":len(flux),"fraction_within_0p5_percent":float(np.mean(np.abs(flux)<=.5)),
                     "p16_mmag":float(np.quantile(r[mask],.16)),"median_mmag":float(np.median(r[mask])),
                     "p84_mmag":float(np.quantile(r[mask],.84))})
    stellar=pd.DataFrame(rows);stellar.to_csv(OUTPUT/"joint_vs_previous_stellar_metrics.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    x=np.arange(6)
    for offset,name,color in ((-.18,"previous masked","#397fa1"),(.18,"joint","#d47727")):
        sub=led_summary[led_summary.model==name].sort_values("LED")
        axes[0].bar(x+offset,sub.residual_rms_percent,width=.36,color=color,label=name)
    axes[0].set_xticks(x,labels=[str(i) for i in range(1,7)]);axes[0].set_xlabel("LED")
    axes[0].set_ylabel("CAR190 group residual RMS (%)")
    axes[0].legend();axes[0].grid(axis="y",alpha=.2)
    axes[1].bar([0,1],stellar.fraction_within_0p5_percent*100,color=["#397fa1","#d47727"])
    axes[1].set_xticks([0,1],labels=stellar.model);axes[1].set_ylabel("Amber residuals within ±0.5% (%)")
    axes[1].set_ylim(0,100);axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("WFI01 joint fit: CAR190 and Amber, fixed ice histories")
    fig.savefig(OUTPUT/"joint_vs_previous_metrics.png",dpi=180);plt.close(fig)
    aggregate={"previous_masked_CAR190_chi2":float(old_led@old_led),"joint_CAR190_chi2":float(joint@joint),
               "previous_masked_CAR190_residual_rms_percent":float(100*np.sqrt(np.mean((old_led*sigma)**2))),
               "joint_CAR190_residual_rms_percent":float(100*np.sqrt(np.mean((joint*sigma)**2))),
               "previous_stellar_fraction_within_0p5":float(stellar.iloc[0].fraction_within_0p5_percent),
               "joint_stellar_fraction_within_0p5":float(stellar.iloc[1].fraction_within_0p5_percent),
               "comparison_stellar_location":"same Student-t nu=4 location and previous masked scale for both predictions",
               "optimizer_status":new_summary["optimizer_message"]}
    (OUTPUT/"joint_vs_previous_summary.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    print(json.dumps(aggregate,indent=2));print(led_summary.to_string(index=False));print(stellar.to_string(index=False))


if __name__=="__main__":main()
