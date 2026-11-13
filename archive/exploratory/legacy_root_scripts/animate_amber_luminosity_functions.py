"""Animated observed and ice-corrected apparent-magnitude distributions."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from astropy.time import Time

import run_amber_student_t_detectors as run
from plot_amber_high_ice_cmd import OUTPUT, zero_ice_log_ratio


def corrected_catalog():
    cache=OUTPUT/"luminosity_function_corrected_magnitudes.csv"
    if cache.is_file():
        return pd.read_csv(cache,dtype={"source_id":str})
    frames=[]
    zero_error=0.
    for detector in run.workflow.DETECTORS:
        if not run.successful(detector):raise RuntimeError(f"Missing fit: {detector}")
        _,directory=run.paths(detector)
        with np.load(directory/"predictions.npz",allow_pickle=False) as archive:
            parameters=archive["student_t_fit_parameters"]
            expected_ids=archive["source_ids"].astype(str)
        _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare(detector,write_inputs=False)
        np.testing.assert_array_equal(expected_ids,np.asarray(ids))
        times=common.groupby(["filter","pass"])["mjd_amber"].median()
        ids=np.asarray(ids)
        for fi,band in enumerate(run.FILTERS):
            check=zero_ice_log_ratio(model,parameters,fi,0,zero_thickness=True)
            zero_error=max(zero_error,float(np.max(np.abs(check))))
            for pi,epoch in enumerate(run.PASSES):
                mask=valid[:,fi,pi]
                if not np.any(mask):continue
                effect=-zero_ice_log_ratio(model,parameters,fi,pi)/run.workflow.LOG_FLUX_PER_MAG
                tiles=model.source_tile_ids[:,fi,pi]
                ice=model.tile_thickness_by_id[tiles,fi,pi]
                frames.append(pd.DataFrame({
                    "detector":detector,"source_id":ids[mask],"filter":band,"pass":epoch,
                    "mjd_utc":float(times.loc[(band,epoch)]),"mag_observed":magnitude[mask,fi,pi],
                    "catalog_mag_error":error[mask,fi,pi],"ice_effect_mag":effect[mask],
                    "mag_zero_ice_corrected":magnitude[mask,fi,pi]-effect[mask],
                    "tile_id":tiles[mask],"ice_nm":ice[mask],
                }))
        print(f"Computed all-filter luminosity-function corrections: {detector}",flush=True)
    if zero_error>1e-12:raise RuntimeError(f"Zero-thickness correction check failed: {zero_error}")
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(cache,index=False)
    return result


def main():
    catalog=corrected_catalog()
    # Count a Gaia star once per filter/visit if a dither-edge source has
    # measurements on more than one SCA. The choice is identical in both curves.
    unique=catalog.sort_values(["catalog_mag_error","detector"]).drop_duplicates(
        ["source_id","filter","pass"],keep="first")
    duplicates=len(catalog)-len(unique)
    fixed=[]
    selections=[]
    for band in run.FILTERS:
        rows=unique.loc[unique["filter"]==band]
        sources=set.intersection(*(set(rows.loc[rows["pass"]==epoch,"source_id"]) for epoch in run.PASSES))
        if not sources:raise ValueError(f"No fixed all-visit population for {band}")
        fixed.append(rows.loc[rows["source_id"].isin(sources)].copy())
        selections.extend({"filter":band,"source_id":source_id} for source_id in sorted(sources))
    data=pd.concat(fixed,ignore_index=True)
    for (band,epoch),rows in data.groupby(["filter","pass"]):
        if rows.source_id.duplicated().any():raise ValueError("Repeated source in luminosity function")
    histograms=[]
    curves={}
    plot_limits={}
    bin_width=.025
    smoothing=1.
    counts={}
    for band in run.FILTERS:
        rows=data.loc[data["filter"]==band]
        counts[band]=int(rows.source_id.nunique())
        all_mags=np.concatenate([rows.mag_observed.to_numpy(),rows.mag_zero_ice_corrected.to_numpy()])
        if not np.isfinite(all_mags).all():raise ValueError("Nonfinite plotted magnitude")
        edges=np.arange(np.floor(all_mags.min()/bin_width)*bin_width-bin_width,
                        np.ceil(all_mags.max()/bin_width)*bin_width+2*bin_width,bin_width)
        centers=(edges[:-1]+edges[1:])/2
        lo,hi=np.quantile(all_mags,[.005,.995])
        padding=.08*max(hi-lo,.1)
        plot_limits[band]=(float(lo-padding),float(hi+padding))
        ymax=0.
        for epoch in run.PASSES:
            frame=rows.loc[rows["pass"]==epoch]
            if len(frame)!=counts[band]:raise ValueError("Luminosity-function frame population changed")
            for variant,column in (("Uncorrected","mag_observed"),("Ice-corrected","mag_zero_ice_corrected")):
                hist,_=np.histogram(frame[column],bins=edges)
                if hist.sum()!=counts[band]:raise ValueError("Histogram dropped stars")
                smoothed=gaussian_filter1d(hist.astype(float),sigma=smoothing,mode="reflect")/bin_width
                curves[(band,epoch,variant)]=(centers,smoothed)
                visible=(centers>=plot_limits[band][0])&(centers<=plot_limits[band][1])
                ymax=max(ymax,float(smoothed[visible].max()))
                histograms.append(pd.DataFrame({"filter":band,"pass":epoch,"variant":variant,
                    "bin_left_mag":edges[:-1],"bin_right_mag":edges[1:],"bin_center_mag":centers,
                    "raw_star_count":hist,"smoothed_stars_per_mag":smoothed,"total_stars":counts[band]}))
        plot_limits[band]=(*plot_limits[band],1.1*ymax)
    fig,axes=plt.subplots(2,4,figsize=(17,9),dpi=120)
    colors={"Uncorrected":"#555555","Ice-corrected":"#d47727"}
    artists={}
    for band,axis in zip(run.FILTERS,axes.flat,strict=True):
        for variant,color in colors.items():
            artists[(band,variant)],=axis.plot([],[],color=color,linewidth=1.7,label=variant)
        axis.set_title(f"{band} · {counts[band]:,} stars",fontsize=12)
        axis.set_xlim(plot_limits[band][:2])
        axis.set_ylim(0,plot_limits[band][2])
        axis.set_xlabel(f"{band} apparent magnitude")
        axis.grid(alpha=.2)
    for axis in axes[:,0]:axis.set_ylabel("Stars per magnitude")
    title=fig.suptitle("",fontsize=16,y=.985)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.94),ncol=2,frameon=False,fontsize=11)
    fig.text(.5,.025,"Within each filter, the same Gaia stars appear at all 11 visits. Fixed 0.025-mag bins; both curves smoothed with the same 0.025-mag Gaussian kernel.\n"
             "Ice-only correction to zero thickness using the Amber Student-t model; no roll offsets or stellar-level alignment. Fixed axes per panel.\n"
             "Visits advance at equal animation intervals, not proportional elapsed time. Display limits use pooled 0.5–99.5% quantiles plus margins; histogram CSV retains all stars.",
             ha="center",fontsize=9)
    fig.subplots_adjust(left=.06,right=.985,bottom=.17,top=.86,wspace=.23,hspace=.32)
    frames=[]
    visit_records=[]
    for epoch in run.PASSES:
        rows=data.loc[data["pass"]==epoch]
        date=Time(float(rows.mjd_utc.median()),format="mjd",scale="utc").strftime("%Y-%m-%d")
        ice=float(rows.ice_nm.median())
        title.set_text(f"Amber luminosity functions: {date} UTC · {epoch.replace('_',' ')} · median ice {ice:.1f} nm")
        for band in run.FILTERS:
            for variant in colors:
                x,y=curves[(band,epoch,variant)]
                artists[(band,variant)].set_data(x,y)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        visit_records.append({"pass":epoch,"utc_date":date,"median_ice_nm":ice})
    peak=max(visit_records,key=lambda row:row["median_ice_nm"])["pass"]
    peak_index=run.PASSES.index(peak)
    frames[peak_index].save(OUTPUT/"luminosity_functions_highest_ice.png")
    plt.close(fig)
    # Shared palette prevents spurious changes in line color between frames.
    thumbnails=[frame.resize((frame.width//2,frame.height//2)) for frame in frames]
    montage=Image.new("RGB",(thumbnails[0].width,thumbnails[0].height*len(thumbnails)))
    for index,frame in enumerate(thumbnails):montage.paste(frame,(0,index*frame.height))
    palette=montage.quantize(colors=256)
    indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    durations=[1000]*len(frames)
    durations[0]=1400
    durations[-1]=1800
    output=OUTPUT/"luminosity_functions_uncorrected_corrected_timelapse.gif"
    indexed[0].save(output,save_all=True,append_images=indexed[1:],duration=durations,loop=0,disposal=2)
    pd.concat(histograms,ignore_index=True).to_csv(OUTPUT/"luminosity_function_histograms.csv",index=False)
    pd.DataFrame(selections).to_csv(OUTPUT/"luminosity_function_star_ids.csv",index=False)
    pd.DataFrame(visit_records).to_csv(OUTPUT/"luminosity_function_visits.csv",index=False)
    metadata={"plot":str(output),"frames":len(frames),"stars_per_filter":counts,
              "duplicate_source_filter_visits_resolved":duplicates,
              "duplicate_policy":"lowest catalog-error observation; identical choice in both curves",
              "bin_width_mag":bin_width,"smoothing_sigma_mag":bin_width*smoothing,
              "fixed_sources_across_visits_per_filter":True,"roll_offsets_applied":False,
              "stellar_level_alignment_applied":False,"reference":"same tile at zero ice",
              "highest_ice_frame":peak,"panel_limits":plot_limits}
    (OUTPUT/"luminosity_function_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
