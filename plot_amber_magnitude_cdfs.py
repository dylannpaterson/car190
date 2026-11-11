"""Static apparent-magnitude CDFs colored by modeled ice thickness."""

import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from astropy.time import Time

from animate_amber_luminosity_functions import OUTPUT
from heatmap_car201_allstar_model_correction import FILTERS, PASSES


def main():
    catalog=pd.read_csv(OUTPUT/"luminosity_function_corrected_magnitudes.csv",dtype={"source_id":str})
    sources=pd.read_csv(OUTPUT/"luminosity_function_star_ids.csv",dtype={"source_id":str})
    catalog=catalog.sort_values(["catalog_mag_error","detector"]).drop_duplicates(
        ["source_id","filter","pass"],keep="first")
    data=catalog.merge(sources,on=["source_id","filter"],how="inner",validate="many_to_one")
    visit=data.groupby("pass").agg(median_ice_nm=("ice_nm","median"),mjd_utc=("mjd_utc","median")).reindex(PASSES)
    norm=Normalize(vmin=float(visit.median_ice_nm.min()),vmax=float(visit.median_ice_nm.max()))
    cmap=plt.get_cmap("viridis")
    lf_metadata=json.loads((OUTPUT/"luminosity_function_metadata.json").read_text())
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharey=True)
    exported=[]
    for band,axis in zip(FILTERS,axes.flat,strict=True):
        rows=data.loc[data["filter"]==band]
        expected=int(rows.source_id.nunique())
        for epoch in PASSES:
            frame=rows.loc[rows["pass"]==epoch]
            if len(frame)!=expected:raise ValueError(f"Changing CDF population: {band}/{epoch}")
            color=cmap(norm(visit.loc[epoch,"median_ice_nm"]))
            for variant,column,style in (("Uncorrected","mag_observed","--"),
                                          ("Ice-corrected","mag_zero_ice_corrected","-")):
                values=np.sort(frame[column].to_numpy(float))
                cdf=np.arange(1,len(values)+1)/len(values)
                axis.step(values,cdf,where="post",color=color,linestyle=style,
                          linewidth=.95 if variant=="Uncorrected" else 1.25,
                          alpha=.75 if variant=="Uncorrected" else .95)
                exported.append({"filter":band,"pass":epoch,"variant":variant,"stars":len(values),
                                 "median_magnitude":float(np.median(values)),
                                 "p16_magnitude":float(np.quantile(values,.16)),
                                 "p84_magnitude":float(np.quantile(values,.84)),
                                 "visit_median_ice_nm":float(visit.loc[epoch,"median_ice_nm"])})
        axis.set_title(f"{band} · {expected:,} stars")
        axis.set_xlim(lf_metadata["panel_limits"][band][:2])
        axis.set_ylim(0,1)
        axis.set_xlabel(f"{band} apparent magnitude")
        axis.grid(alpha=.2)
    for axis in axes[:,0]:axis.set_ylabel("Cumulative fraction of stars")
    handles=[Line2D([0],[0],color="0.25",linestyle="--",label="Uncorrected"),
             Line2D([0],[0],color="0.25",linestyle="-",label="Ice-corrected")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.47,.945),ncol=2,frameon=False,fontsize=11)
    fig.suptitle("Amber magnitude CDFs across visits — colored by ice thickness",fontsize=16,y=.99)
    caxis=fig.add_axes([.935,.2,.014,.61])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=caxis,label="Visit median modeled ice thickness (nm)")
    fig.text(.48,.025,"Same stars at all 11 visits within each filter. Empirical CDFs: no magnitude binning or smoothing. Each visit has one shared color.\n"
             "Solid lines apply the ice-only correction to zero thickness; dashed lines show observed magnitudes. No roll offsets or stellar-level alignment applied.",
             ha="center",fontsize=9)
    fig.subplots_adjust(left=.06,right=.91,bottom=.14,top=.86,wspace=.22,hspace=.32)
    output=OUTPUT/"magnitude_cdfs_colored_by_ice.png"
    fig.savefig(output,dpi=180)
    plt.close(fig)
    pd.DataFrame(exported).to_csv(OUTPUT/"magnitude_cdf_summary.csv",index=False)
    visit["utc_date"]=[Time(value,format="mjd",scale="utc").strftime("%Y-%m-%d") for value in visit.mjd_utc]
    visit.to_csv(OUTPUT/"magnitude_cdf_visit_colors.csv")
    print(output)
    print(visit.to_string())


if __name__=="__main__":
    main()
