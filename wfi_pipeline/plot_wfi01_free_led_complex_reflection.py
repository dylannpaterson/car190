"""Clean complex reflection for canonical and free-LED WFI01 experiments."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_amber_student_t_detectors as run
import mock_fit_wfi_star_reflectance_modulation as mock
from experiment_free_led_modulation import free_value_basis


def reciprocal(values):
    values=np.asarray(values,float)
    return np.divide(1000.,values,out=np.full(values.shape,np.nan),where=values>0)


def aligned_phase(reflection,reference):
    result=np.unwrap(np.angle(reflection[:,::-1]),axis=1)[:,::-1]
    result+=2*np.pi*np.round((reference-result[:,-1])/(2*np.pi))[:,None]
    return result


def main():
    detector="WFI01"
    area,baseline=run.canonical_configuration(detector)
    experiment=baseline.with_name(baseline.name+"_free_led_test")
    with np.load(baseline/"predictions.npz",allow_pickle=False) as archive:
        canonical_parameters=archive["student_t_fit_parameters"]
    with np.load(experiment/"predictions.npz",allow_pickle=False) as archive:
        free_parameters=archive["student_t_fit_parameters"]
    _,_,_,_,_,_,_,model,_=run.prepare(detector,write_inputs=False,effective_area_path=area,output_dir=baseline)
    wave=model.wavelengths_nm
    original_basis=model.basis
    extra,*_=free_value_basis(wave)
    free_basis=np.concatenate([original_basis,extra])
    count=model.mode_count
    free_count=len(free_basis)
    canonical_ratio=np.exp(canonical_parameters[:count]@original_basis
                            +1j*(canonical_parameters[count:2*count]@original_basis))
    free_ratio=np.exp(free_parameters[:free_count]@free_basis
                     +1j*(free_parameters[free_count:2*free_count]@free_basis))
    clean=model.clean_reflection_by_tile
    curves={"Unmodulated clean TMM":clean,
            "LED-anchored fit":clean*canonical_ratio[None,:],
            "Free LEDs + boundaries":clean*free_ratio[None,:]}
    reference=np.angle(clean[:,-1])
    reference+=2*np.pi*np.round((np.median(reference)-reference)/(2*np.pi))
    order=np.argsort(1000./wave)
    x=1000./wave[order]
    styles={"Unmodulated clean TMM":("0.4","--"),"LED-anchored fit":("#357fa1","-"),
            "Free LEDs + boundaries":("#d77c2e","-")}
    fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
    records=[]
    limits={}
    for label,values in curves.items():
        amplitude=np.percentile(np.abs(values),[16,50,84],axis=0)
        phase=np.percentile(aligned_phase(values,reference),[16,50,84],axis=0)
        color,style=styles[label]
        for axis,stats in zip(axes,(amplitude,phase),strict=True):
            axis.fill_between(x,stats[0,order],stats[2,order],color=color,alpha=.13,linewidth=0)
            axis.plot(x,stats[1,order],color=color,linestyle=style,linewidth=1.8,label=label)
        for index,wavelength in enumerate(wave):
            records.append({"variant":label,"wavelength_nm":wavelength,
                            **{f"amplitude_p{p}":amplitude[i,index] for i,p in enumerate((16,50,84))},
                            **{f"phase_p{p}_rad":phase[i,index] for i,p in enumerate((16,50,84))}})
        limits[label]={"max_amplitude":float(np.abs(values).max()),
                       "min_amplitude":float(np.abs(values).min())}
    for axis in axes:
        for index,center in enumerate(mock.LED_CENTERS_NM,start=1):
            position=1000./center
            axis.axvline(position,color="0.5",linestyle=":",linewidth=.8)
            if axis is axes[0]:
                axis.text(position,.98,f"LED{index}",transform=axis.get_xaxis_transform(),
                          rotation=90,ha="right",va="top",fontsize=8,color="0.4")
        axis.set_xlim(x[0],x[-1])
        axis.grid(alpha=.2)
    axes[0].set_ylabel(r"Clean reflection amplitude $|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped clean reflection phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top=axes[0].secondary_xaxis("top",functions=(reciprocal,reciprocal))
    top.set_xticks([2300,1700,1300,1000,800,600,500])
    top.set_xlabel("Wavelength (nm)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.945),ncol=3,frameon=False)
    fig.suptitle("WFI01 clean complex reflection: effect of freeing the LED anchors",fontsize=15,y=.99)
    fig.text(.5,.025,"Lines: medians across 256 tile-specific optical states; shading: tile 16th–84th percentiles. No ice layer is shown.\n"
             "Phase branches use the same baseline long-wavelength reference; fitted phase offsets are retained, not independently zeroed.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.08,1,.87))
    output=experiment/"anchored_vs_free_led_clean_complex_reflection.png"
    fig.savefig(output,dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output.with_suffix(".csv"),index=False)
    (experiment/"clean_reflection_amplitude_ranges.json").write_text(json.dumps(limits,indent=2)+"\n")
    print(output)
    print(json.dumps(limits,indent=2))


if __name__=="__main__":
    main()
