"""Side-by-side observed/corrected CMD evolution with one fixed star sample."""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from PIL import Image
from astropy.time import Time

from plot_amber_high_ice_cmd import OUTPUT
from heatmap_car201_allstar_model_correction import PASSES


def main():
    global OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    OUTPUT=args.input_dir
    data=pd.read_csv(OUTPUT/"cmd_all_visits_stars.csv",dtype={"source_id":str})
    blue=data["blue_filter"].iloc[0]
    red=data["red_filter"].iloc[0]
    magnitude_filter=data["magnitude_filter"].iloc[0] if "magnitude_filter" in data else red
    observed_column="vertical_observed_mag" if "vertical_observed_mag" in data else "red_observed_mag"
    corrected_column="vertical_zero_ice_corrected_mag" if "vertical_zero_ice_corrected_mag" in data else "red_zero_ice_corrected_mag"
    if data.duplicated(["source_id","pass"]).any():
        raise ValueError("A source appears on multiple detectors in the same visit")
    source_sets=[set(data.loc[data["pass"]==epoch,"source_id"]) for epoch in PASSES]
    sources=set.intersection(*source_sets)
    if not sources:raise ValueError("No stars have both CMD filters at every visit")
    data=data.loc[data["source_id"].isin(sources)].copy()
    for epoch in PASSES:
        if len(data.loc[data["pass"]==epoch])!=len(sources):raise ValueError("Frame sample changed")
    color=np.concatenate([data.observed_color_mag.to_numpy(),data.corrected_color_mag.to_numpy()])
    magnitude=np.concatenate([data[observed_column].to_numpy(),data[corrected_column].to_numpy()])
    xlo,xhi=np.quantile(color,[.005,.995])
    ylo,yhi=np.quantile(magnitude,[.005,.995])
    xpad=.08*max(xhi-xlo,.1)
    ypad=.06*max(yhi-ylo,.1)
    extent=(xlo-xpad,xhi+xpad,ylo-ypad,yhi+ypad)
    # Determine one hexagon count scale across both panels and all frames.
    scratch,axis=plt.subplots()
    maximum=1.
    for epoch in PASSES:
        stars=data.loc[data["pass"]==epoch]
        for x,y in ((stars.observed_color_mag,stars[observed_column]),
                    (stars.corrected_color_mag,stars[corrected_column])):
            image=axis.hexbin(x,y,gridsize=50,extent=extent,mincnt=1)
            maximum=max(maximum,float(image.get_array().max()))
            axis.clear()
    plt.close(scratch)
    norm=LogNorm(vmin=1,vmax=maximum)
    fig=plt.figure(figsize=(13.5,7.7),dpi=120)
    axes=[fig.add_axes([.065,.17,.385,.65]),fig.add_axes([.49,.17,.385,.65])]
    caxis=fig.add_axes([.91,.17,.015,.65])
    heading=fig.text(.47,.965,"",ha="center",fontsize=15)
    detail=fig.text(.47,.91,"",ha="center",fontsize=11)
    fig.text(.5,.035,f"Same {len(sources):,} Gaia stars in every frame; both filters measured at all 11 visits. Fixed axes, hexagons and density colors.\n"
             "Right panel: actual ice-only correction to the same tile at zero thickness, using Amber Student-t fits. No fitted roll offsets applied.\n"
             "Frames advance by observed visit; pauses do not represent the elapsed time between visits. Display uses pooled 0.5–99.5% limits plus margins.",
             ha="center",fontsize=9)
    frames=[]
    summaries=[]
    for frame_index,epoch in enumerate(PASSES):
        stars=data.loc[data["pass"]==epoch]
        arrays=((stars.observed_color_mag,stars[observed_column]),
                (stars.corrected_color_mag,stars[corrected_column]))
        for index,(axis,(x,y),title) in enumerate(zip(axes,arrays,("Observed — uncorrected","Corrected to zero ice"),strict=True)):
            axis.clear()
            image=axis.hexbin(x,y,gridsize=50,extent=extent,mincnt=1,cmap="viridis",norm=norm,linewidths=0)
            axis.set_xlim(extent[:2])
            axis.set_ylim(extent[3],extent[2])
            axis.set_xlabel(f"{blue} − {red} (mag)")
            axis.set_title(title,fontsize=13)
            if index==0:axis.set_ylabel(f"{magnitude_filter} magnitude")
            else:axis.tick_params(labelleft=False)
            axis.grid(alpha=.12)
        if frame_index==0:fig.colorbar(image,cax=caxis,label="Stars per hexagon (log scale)")
        date=Time(float(stars.mjd_utc.median()),format="mjd",scale="utc").strftime("%Y-%m-%d")
        ice=float(stars.mean_ice_nm.median())
        color_shift=float(stars.color_shift_mag.median())
        magnitude_shift=float(stars.vertical_shift_mag.median())
        heading.set_text(f"Amber CMD evolution: {date} UTC · {epoch.replace('_',' ')}")
        detail.set_text(f"Median ice: {ice:.1f} nm     |     Correction: Δcolor = {1000*color_shift:+.1f} mmag, Δ{magnitude_filter} = {1000*magnitude_shift:+.1f} mmag")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        summaries.append({"pass":epoch,"utc_date":date,"stars":len(stars),"median_ice_nm":ice,
                          "median_color_correction_mag":color_shift,"median_magnitude_correction_mag":magnitude_shift})
    plt.close(fig)
    # A single palette across all frames prevents independent quantization flicker.
    montage=Image.new("RGB",(frames[0].width,frames[0].height*len(frames)))
    for index,frame in enumerate(frames):montage.paste(frame,(0,index*frame.height))
    palette=montage.quantize(colors=256)
    indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    output=OUTPUT/"cmd_uncorrected_corrected_timelapse.gif"
    durations=[1000]*len(frames)
    durations[0]=1500
    durations[-1]=1800
    indexed[0].save(output,save_all=True,append_images=indexed[1:],duration=durations,loop=0,disposal=2)
    pd.DataFrame(summaries).to_csv(OUTPUT/"cmd_timelapse_visits.csv",index=False)
    pd.DataFrame({"gaia_dr3_source_id":sorted(sources)}).to_csv(OUTPUT/"cmd_timelapse_star_ids.csv",index=False)
    (OUTPUT/"cmd_timelapse_metadata.json").write_text(json.dumps({"plot":str(output),"stars_per_frame":len(sources),
        "frames":len(frames),"fixed_source_population":True,"fixed_density_scale_max_stars":maximum,
        "axis_extent":list(extent),"roll_offsets_applied":False,
        "blue_filter":blue,"red_filter":red,"magnitude_filter":magnitude_filter,
        "correction":"same-tile zero-ice count ratio using the fitted photon-SED spectral weights",
        "timing":"equal-duration observed visits; not proportional to elapsed time"},indent=2)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(output)


if __name__=="__main__":
    main()
