"""Compare canonical, endpoint-only and free-LED WFI01 fits."""

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

import run_amber_student_t_detectors as run
from experiment_free_led_modulation import free_value_basis
from plot_wfi01_free_led_complex_reflection import aligned_phase,reciprocal


def main():
    area,baseline=run.canonical_configuration("WFI01")
    endpoint=baseline.with_name(baseline.name+"_free_endpoints_test")
    free=baseline.with_name(baseline.name+"_free_led_test")
    directories={"Canonical anchors":baseline,"Free endpoints; LEDs anchored":endpoint,
                 "Free LEDs + endpoints":free}
    colors={"Canonical anchors":"#357fa1","Free endpoints; LEDs anchored":"#379663",
            "Free LEDs + endpoints":"#d77c2e"}
    loaded={}
    for label,directory in directories.items():
        with np.load(directory/"predictions.npz",allow_pickle=False) as archive:
            loaded[label]={key:archive[key] for key in ("source_ids","valid","student_t_fit_parameters","student_t_prediction_mag")}
    _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare("WFI01",write_inputs=False,
                                                               effective_area_path=area,output_dir=baseline)
    for item in loaded.values():
        np.testing.assert_array_equal(item["source_ids"].astype(str),np.asarray(ids))
        np.testing.assert_array_equal(item["valid"],valid)
    wave=model.wavelengths_nm
    extra,*_=free_value_basis(wave)
    bases={"Canonical anchors":model.basis,
           "Free endpoints; LEDs anchored":np.concatenate([model.basis,extra[[0,-1]]]),
           "Free LEDs + endpoints":np.concatenate([model.basis,extra])}
    clean=model.clean_reflection_by_tile
    reference=np.angle(clean[:,-1])
    reference+=2*np.pi*np.round((np.median(reference)-reference)/(2*np.pi))
    order=np.argsort(1000./wave)
    x=1000./wave[order]
    fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
    curves={"Unmodulated clean TMM":clean}
    for label,item in loaded.items():
        count=len(bases[label])
        theta=item["student_t_fit_parameters"]
        ratio=np.exp(theta[:count]@bases[label]+1j*(theta[count:2*count]@bases[label]))
        curves[label]=clean*ratio[None,:]
    records=[]
    for label,values in curves.items():
        amplitude=np.percentile(np.abs(values),[16,50,84],axis=0)
        phase=np.percentile(aligned_phase(values,reference),[16,50,84],axis=0)
        color=colors.get(label,"0.4")
        for axis,statistics in zip(axes,(amplitude,phase),strict=True):
            axis.fill_between(x,statistics[0,order],statistics[2,order],color=color,alpha=.1,linewidth=0)
            axis.plot(x,statistics[1,order],color=color,linestyle="--" if label not in colors else "-",
                      linewidth=1.6,label=label)
        for i,wavelength in enumerate(wave):
            records.append({"variant":label,"wavelength_nm":wavelength,
                            **{f"amplitude_p{q}":amplitude[j,i] for j,q in enumerate((16,50,84))},
                            **{f"phase_p{q}_rad":phase[j,i] for j,q in enumerate((16,50,84))}})
    for axis in axes:
        for center in run.workflow.mock_model.LED_CENTERS_NM:
            axis.axvline(1000./center,color="0.6",linestyle=":",linewidth=.8)
        axis.grid(alpha=.2)
        axis.set_xlim(x[0],x[-1])
    axes[0].set_ylabel(r"Clean reflection amplitude $|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped clean reflection phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top=axes[0].secondary_xaxis("top",functions=(reciprocal,reciprocal))
    top.set_xticks([2300,1700,1300,1000,800,600,500])
    top.set_xlabel("Wavelength (nm)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.945),ncol=2,frameon=False)
    fig.suptitle("WFI01 clean complex reflection: which wavelength anchors matter?",fontsize=15,y=.99)
    fig.text(.5,.025,"Lines: medians over 256 tiles; shading: tile 16th–84th percentiles. No ice layer. Same phase branch reference for all curves.\n"
             "Green frees only 400/2321-nm endpoints; its modulation remains unity at all six LED wavelengths.",ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.08,1,.85))
    fig.savefig(endpoint/"anchor_choices_clean_complex_reflection.png",dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(endpoint/"anchor_choices_clean_complex_reflection.csv",index=False)
    # The three fits share one measured reference population for this plot.
    primary=valid&valid[:,:,:1]
    primary[:,:,0]=False
    dates=mdates.date2num(Time(common.groupby("pass")["mjd_amber"].median().reindex(run.PASSES).to_numpy(),
                              format="mjd",scale="utc").to_datetime())
    fluxes={}
    for label,item in loaded.items():
        corrected=magnitude-item["student_t_prediction_mag"]
        fluxes[label]=100*np.expm1(-run.workflow.LOG_FLUX_PER_MAG*(corrected-corrected[:,:,:1]))
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    summaries=[]
    for fi,(band,axis) in enumerate(zip(run.FILTERS,axes.flat,strict=True)):
        axis.axhspan(-.5,.5,color="#65ac68",alpha=.15)
        for li,label in enumerate(directories):
            for pi,date in enumerate(dates):
                values=fluxes[label][:,fi,pi][primary[:,fi,pi]]
                if not len(values):continue
                low,med,high=np.quantile(values,[.16,.5,.84])
                axis.errorbar(date+(li-1)*.07,med,yerr=[[med-low],[high-med]],fmt="o",color=colors[label],
                              capsize=1.5,markersize=3,label=label if pi==1 else None)
        axis.set_title(band)
        axis.axhline(0,color="0.5",linewidth=.6)
        axis.grid(alpha=.2)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x",labelrotation=45,labelsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.945),ncol=3,fontsize=9)
    fig.suptitle("WFI01 Amber Student-t: endpoint-only versus free-LED sensitivity",y=.99,fontsize=15)
    fig.supylabel("Corrected flux change from measured EP3 pass 1 (%)")
    fig.text(.5,.025,"Identical data, quality cuts, detector effective area, likelihood, fixed ice history and original regularization. Bounds unchanged.\n"
             "Points: medians; bars: individual-star 16th–84th percentiles. Cumulative plot uses measured EP3 references; fits include all repeated sources.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.12,1,.88))
    fig.savefig(endpoint/"anchor_choices_corrected_photometry.png",dpi=180)
    plt.close(fig)
    for label,directory in directories.items():
        fit=json.loads((directory/"fit_summary.json").read_text())
        if label=="Canonical anchors":
            # Canonical common-normalization metrics are present in both
            # experiments and use exactly the same rule and scale.
            reference_summary=json.loads((endpoint/"fit_summary.json").read_text())
            scale=fit["extra_student_t_scale_mmag"]
            fraction=reference_summary["canonical_metrics_common_robust_normalization"]["fraction_within_0p5_percent"]
            added=0
        else:
            scale=fit["free_led_extra_scale_mmag"]
            fraction=fit["free_led_metrics_common_robust_normalization"]["fraction_within_0p5_percent"]
            added=fit["added_real_parameters"]
        summaries.append({"variant":label,"extra_student_t_scale_mmag":scale,
                          "fraction_within_0p5_percent_common_robust_normalization":fraction,
                          "added_real_parameters":added})
    pd.DataFrame(summaries).to_csv(endpoint/"anchor_choices_summary.csv",index=False)
    led=pd.read_csv(endpoint/"led_and_boundary_modulation.csv")
    np.testing.assert_allclose(led.loc[led["kind"]=="LED","amplitude_multiplier"],1.,atol=1e-14)
    np.testing.assert_allclose(led.loc[led["kind"]=="LED","phase_modulation_rad"],0.,atol=1e-14)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(endpoint/"anchor_choices_clean_complex_reflection.png")
    print(endpoint/"anchor_choices_corrected_photometry.png")


if __name__=="__main__":
    main()
