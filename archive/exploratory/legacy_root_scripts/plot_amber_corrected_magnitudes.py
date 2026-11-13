"""Export corrected magnitudes and plot measured-reference magnitude changes."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

from check_amber_wfi01_photometry import OUTPUT as GAUSSIAN_OUTPUT, FILTERS, PASSES
from fit_amber_wfi01_student_t import OUTPUT


def main():
    with np.load(GAUSSIAN_OUTPUT / "predictions.npz", allow_pickle=False) as archive:
        gaussian = {key:archive[key] for key in ("source_ids","valid","amber_magnitude","amber_error","amber_refitted_floor")}
    with np.load(OUTPUT / "predictions.npz", allow_pickle=False) as archive:
        student = {"student_t_prediction_mag":archive["student_t_prediction_mag"]}
    common = pd.read_csv(GAUSSIAN_OUTPUT / "common_measurements.csv", dtype={"source_id": str})
    ids = gaussian["source_ids"].astype(str).tolist()
    lookup = {source_id:index for index,source_id in enumerate(ids)}
    measurements = []
    for _, row in common.iterrows():
        index = (lookup[str(row["source_id"])], FILTERS.index(row["filter"]), PASSES.index(row["pass"]))
        if not gaussian["valid"][index]:
            continue
        observed = float(gaussian["amber_magnitude"][index])
        gaussian_change = float(gaussian["amber_refitted_floor"][index])
        student_change = float(student["student_t_prediction_mag"][index])
        measurements.append({
            "detector":"WFI01", "source_id":str(row["source_id"]),
            "filter":row["filter"], "pass":row["pass"],
            "mjd_utc":float(row["mjd_amber"]),
            "mag_observed":observed, "catalog_mag_error":float(gaussian["amber_error"][index]),
            "gaussian_model_delta_mag_including_roll":gaussian_change,
            "student_t_model_delta_mag_including_roll":student_change,
            "mag_corrected_gaussian":observed-gaussian_change,
            "mag_corrected_student_t":observed-student_change,
        })
    data = pd.DataFrame(measurements)
    reference_columns = ["mag_observed","mag_corrected_gaussian","mag_corrected_student_t"]
    references = data.loc[data["pass"]=="ep3_pass1",["source_id","filter"]+reference_columns].rename(
        columns={column:column+"_ep3_reference" for column in reference_columns})
    data = data.merge(references,on=["source_id","filter"],how="left",validate="many_to_one")
    data["has_measured_ep3_reference"] = data["mag_observed_ep3_reference"].notna()
    for column in reference_columns:
        data[column+"_change_mmag"] = 1000*(data[column]-data[column+"_ep3_reference"])
    csv = OUTPUT / "amber_observed_and_corrected_magnitudes.csv"
    data.to_csv(csv,index=False)
    plot_data = data.loc[data["has_measured_ep3_reference"] & (data["pass"]!="ep3_pass1")]
    fig,axes = plt.subplots(2,4,figsize=(19,9),sharex=True,sharey=True)
    settings = [("mag_observed_change_mmag","Observed","0.55",-.09),
                ("mag_corrected_gaussian_change_mmag","Gaussian correction","#3579a0",0.),
                ("mag_corrected_student_t_change_mmag","Student-t correction","#d57930",.09)]
    summaries = []
    all_low,all_high=[],[]
    for band,axis in zip(FILTERS,axes.flat,strict=True):
        rows=plot_data.loc[plot_data["filter"]==band]
        axis.axhspan(-5,5,color="#68ac6c",alpha=.14)
        axis.axhline(0,color="0.4",linewidth=.6)
        for column,label,color,offset in settings:
            for epoch in PASSES[1:]:
                selected=rows.loc[rows["pass"]==epoch]
                values=selected[column].to_numpy(float)
                values=values[np.isfinite(values)]
                if not len(values):continue
                low,median,high=np.quantile(values,[.16,.5,.84])
                date=float(mdates.date2num(Time(float(selected["mjd_utc"].median()),format="mjd",scale="utc").to_datetime()))
                axis.errorbar(date+offset,median,yerr=[[median-low],[high-median]],fmt="o",color=color,
                              markersize=3,capsize=1.5,alpha=.85,label=label if epoch=="ep3_pass2" else None)
                summaries.append({"filter":band,"pass":epoch,"variant":label,"stars":len(values),
                                  "p16_change_mmag":low,"median_change_mmag":median,"p84_change_mmag":high})
                all_low.append(low)
                all_high.append(high)
        axis.set_title(band)
        axis.grid(alpha=.2)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x",labelrotation=45,labelsize=8)
    extent=max(max(all_high),-min(all_low))
    axes[0,0].set_ylim(-1.08*extent,1.08*extent)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.94),ncol=3)
    fig.suptitle("WFI01 Amber: observed and corrected magnitude changes",fontsize=16,y=.99)
    fig.supylabel("Magnitude change from the star's measured EP3 pass 1 (mmag)")
    fig.text(.5,.025,"Points: star medians; bars: 16th–84th percentiles. Negative magnitude change means brighter. Green band: ±5 mmag.\n"
             "Same quality-filtered sample; corrections include the fitted roll term. Only stars with measured EP3 references are plotted; all fitted stars are in the CSV.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(.025,.12,1,.88))
    output=OUTPUT/"amber_observed_and_corrected_magnitude_changes.png"
    fig.savefig(output,dpi=180)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(OUTPUT/"amber_corrected_magnitude_changes_summary.csv",index=False)
    print(output)
    print(csv)


if __name__=="__main__":
    main()
