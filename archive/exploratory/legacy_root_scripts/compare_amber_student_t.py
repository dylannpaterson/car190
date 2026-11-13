"""Compare completed WFI01 Gaussian and Student-t fits without new fitting."""

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

from fit_amber_wfi01_student_t import OUTPUT, profile_student_locations, NU
from check_amber_wfi01_photometry import OUTPUT as GAUSSIAN_OUTPUT, FILTERS, PASSES, profile_residual
import mock_fit_wfi_star_reflectance_modulation as mock


def metrics(name, reference, residual):
    values = residual[np.isfinite(residual)] * 1000.
    flux = 100. * np.expm1(-mock.LOG_FLUX_PER_MAG * values / 1000.)
    low, med, high = np.quantile(values, [.16,.5,.84])
    return {"likelihood": name, "reference": reference, "observations": len(values),
            "residual_rms_mmag": float(np.sqrt(np.mean(values**2))),
            "p16_mmag": low, "median_mmag": med, "p84_mmag": high,
            "mad_sigma_mmag": float(1.4826*np.median(np.abs(values-med))),
            "fraction_within_0p5_percent": float(np.mean(np.abs(flux)<=.5))}


def main():
    gaussian = np.load(GAUSSIAN_OUTPUT / "predictions.npz", allow_pickle=False)
    student = np.load(OUTPUT / "predictions.npz", allow_pickle=False)
    summary = json.loads((OUTPUT / "fit_summary.json").read_text())
    old_summary = json.loads((GAUSSIAN_OUTPUT / "comparison_summary.json").read_text())
    valid = gaussian["valid"]
    if not np.array_equal(valid, student["valid"]):
        raise ValueError("Comparison measurement masks differ")
    observed = gaussian["amber_magnitude"]
    error = gaussian["amber_error"]
    predictions = {"Gaussian": gaussian["amber_refitted_floor"],
                   "Student-t": student["student_t_prediction_mag"]}
    primary = valid & valid[:,:,:1]
    primary[:,:,0] = False
    group_active = valid.reshape(-1,valid.shape[-1]).sum(axis=1)>=2
    group_valid = valid.reshape(-1,valid.shape[-1])[group_active]
    group_error = error.reshape(-1,valid.shape[-1])[group_active]*1000.
    common_scale = np.hypot(group_error,summary["extra_student_t_scale_mmag"])
    rows = []
    common_residuals = {}
    ep3_flux = {}
    for name, prediction in predictions.items():
        weighted_residual = profile_residual(observed,prediction,error,valid)
        rows.append(metrics(name,"same catalog-error weighted star/filter means",weighted_residual))
        corrected = observed-prediction
        differences = corrected-corrected[:,:,:1]
        ep3 = np.where(primary,differences,np.nan)
        rows.append(metrics(name,"measured EP3 pass 1; reference observations excluded",ep3))
        ep3_flux[name] = 100.*np.expm1(-mock.LOG_FLUX_PER_MAG*ep3)
        group_values = (observed-prediction).reshape(-1,valid.shape[-1])[group_active]*1000.
        _, robust_residual, _ = profile_student_locations(group_values,common_scale,group_valid)
        common_residuals[name] = robust_residual[group_valid]/1000.
        rows.append(metrics(name,"same Student-t star/filter locations and scale for both models",
                            common_residuals[name]))
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "likelihood_comparison_metrics.csv", index=False)
    change = profile_residual(predictions["Student-t"],predictions["Gaussian"],error,valid)[valid]*1000.
    report = {"gaussian_extra_sigma_mmag":old_summary["systematic_floor_mmag"],
              "student_t_extra_scale_mmag":summary["extra_student_t_scale_mmag"],
              "student_t_nu":NU,
              "scale_note":"Student-t scale is not RMS; for nu=4 the full Student-t distribution SD is sqrt(2) times its total scale",
              "prediction_change_rms_mmag":float(np.sqrt(np.mean(change**2))),
              "prediction_change_p95_absolute_mmag":float(np.quantile(np.abs(change),.95)),
              "selection_identical":True,"additional_clipping":False,
              "validation":"In-sample likelihood sensitivity experiment, not held-out validation"}
    (OUTPUT / "likelihood_comparison_summary.json").write_text(json.dumps(report,indent=2)+"\n")

    common = pd.read_csv(GAUSSIAN_OUTPUT / "common_measurements.csv",dtype={"source_id":str})
    days = common.groupby("pass")["mjd_amber"].median().reindex(PASSES).to_numpy(float)
    dates = mdates.date2num(Time(days,format="mjd",scale="utc").to_datetime())
    colors = {"Gaussian":"#3579a0","Student-t":"#d57930"}
    fig,axes = plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    for band_index,(band,axis) in enumerate(zip(FILTERS,axes.flat,strict=True)):
        axis.axhspan(-.5,.5,color="#68ac6c",alpha=.16)
        for name, offset in (("Gaussian",-.055),("Student-t",.055)):
            for epoch_index,date in enumerate(dates):
                values = ep3_flux[name][:,band_index,epoch_index]
                values = values[np.isfinite(values)]
                if not len(values): continue
                low,median,high = np.quantile(values,[.16,.5,.84])
                axis.errorbar(date+offset,median,yerr=[[median-low],[high-median]],fmt="o",
                              color=colors[name],capsize=2,markersize=3,
                              label=name if epoch_index==1 else None)
        axis.set_title(band)
        axis.axhline(0,color="0.5",linewidth=.7)
        axis.grid(alpha=.2)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x",labelrotation=45,labelsize=8)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.94),ncol=2)
    fig.suptitle("WFI01 Amber: Gaussian versus Student-t corrected star changes",y=.99,fontsize=15)
    fig.supylabel("Corrected flux change from measured EP3 pass 1 (%)")
    fig.text(.5,.025,"Identical matched measurements and existing quality cuts; no additional clipping. "
             "Points: medians; bars: star 16th–84th percentiles; green: ±0.5%.\n"
             "The full fits use 563 stars and 36,682 observations; this cumulative plot requires a measured EP3-pass-1 reference.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.12,1,.88))
    fig.savefig(OUTPUT/"student_t_vs_gaussian_corrections.png",dpi=180)
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name,residual in common_residuals.items():
        values = residual*1000.
        axes[0].hist(values,bins=np.linspace(-30,30,61),weights=np.full(len(values),100./len(values)),
                     histtype="step",linewidth=1.5,color=colors[name],label=name)
        absolute = np.sort(np.abs(values))
        axes[1].plot(absolute,np.arange(1,len(values)+1)/len(values),color=colors[name],label=name)
    axes[0].set_xlabel("Residual after common robust star/filter normalization (mmag)")
    axes[0].set_ylabel("Observations per 1-mmag bin (%)")
    axes[0].legend()
    axes[1].set_xlim(0,30)
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel("Absolute residual (mmag)")
    axes[1].set_ylabel("Cumulative fraction of all observations")
    axes[1].grid(alpha=.2)
    fig.suptitle("Central residual distributions; all tail observations retained in normalization")
    fig.savefig(OUTPUT/"student_t_vs_gaussian_residual_distribution.png",dpi=180)
    plt.close(fig)

    old = pd.read_csv(GAUSSIAN_OUTPUT/"amber_modulation_parameters.csv").set_index("mode")
    new = pd.read_csv(OUTPUT/"student_t_modulation_parameters.csv").set_index("mode")
    wave = np.arange(470.,2321.,2.)
    basis,names,*_ = mock.modulation_basis(wave,grouping="per-gap")
    fig,axes = plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
    for name,parameters in (("Gaussian",old),("Student-t",new)):
        parameters = parameters.loc[names]
        axes[0].plot(wave,100*np.expm1(parameters.fit_log_amplitude.to_numpy()@basis),color=colors[name],label=name)
        axes[1].plot(wave,parameters.fit_phase_rad.to_numpy()@basis,color=colors[name])
    axes[0].set_ylabel("Clean-reflection amplitude modulation (%)")
    axes[1].set_ylabel("Clean-reflection phase modulation (rad)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].legend()
    for axis in axes:
        axis.axhline(0,color="0.5",linewidth=.6)
        axis.grid(alpha=.2)
    fig.suptitle("WFI01 Amber: inferred optical modulation depends on the likelihood")
    fig.savefig(OUTPUT/"student_t_vs_gaussian_modulation.png",dpi=180)
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
