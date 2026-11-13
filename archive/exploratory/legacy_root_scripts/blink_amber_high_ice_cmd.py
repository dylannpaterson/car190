"""Blink identical observed/corrected CMDs without exaggerating the correction."""

import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image

from plot_amber_high_ice_cmd import OUTPUT


def main():
    stars=pd.read_csv(OUTPUT/"cmd_highest_ice_stars.csv",dtype={"source_id":str})
    summary=json.loads((OUTPUT/"cmd_highest_ice_summary.json").read_text())
    xmin,xmax,ymin,ymax=summary["display_limits"]
    extent=(xmin,xmax,ymin,ymax)
    grids=[]
    for x,y in ((stars.observed_color_mag,stars.red_observed_mag),
                (stars.corrected_color_mag,stars.red_zero_ice_corrected_mag)):
        grid,_,_=np.histogram2d(x,y,bins=(90,100),range=((xmin,xmax),(ymin,ymax)))
        grids.append(np.log10(1+grid.T))
    norm=Normalize(vmin=0,vmax=max(float(grid.max()) for grid in grids))
    fig=plt.figure(figsize=(8,8),dpi=150)
    axis=fig.add_axes([.11,.15,.72,.73])
    caxis=fig.add_axes([.86,.15,.025,.73])
    image=axis.imshow(grids[0],origin="lower",extent=extent,aspect="auto",cmap="magma_r",norm=norm)
    fig.colorbar(image,cax=caxis,label="log₁₀(1 + stars per bin)")
    axis.set_xlim(xmin,xmax)
    axis.set_ylim(ymax,ymin)
    axis.set_xlabel("F087 − F146 (mag)")
    axis.set_ylabel("F146 magnitude")
    axis.grid(alpha=.12)
    title=fig.text(.47,.955,"",ha="center",va="center",fontsize=15)
    fig.text(.47,.915,f"{summary['utc_date']} UTC · median ice {summary['median_ice_nm']:.1f} nm · {len(stars):,} same stars",
             ha="center",fontsize=10)
    fig.text(.5,.035,"Identical axes, binning and density scale. Actual correction size—no amplification.\n"
             "Median shift: +30.1 mmag in F087−F146; +9.3 mmag in F146.",
             ha="center",fontsize=9)
    frames=[]
    for label,grid in zip(("Observed — no ice correction","Corrected to zero ice"),grids,strict=True):
        image.set_data(grid)
        title.set_text(label)
        fig.canvas.draw()
        rgba=np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:,:,:3].copy()))
    plt.close(fig)
    output=OUTPUT/"cmd_uncorrected_corrected_blink.gif"
    # Use one shared palette for both frames to avoid artificial color flicker.
    stack=Image.new("RGB",(frames[0].width,2*frames[0].height))
    for i,frame in enumerate(frames):stack.paste(frame,(0,i*frame.height))
    palette=stack.quantize(colors=256)
    indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(output,save_all=True,append_images=indexed[1:],duration=[900,900],loop=0,disposal=2)
    print(output)


if __name__=="__main__":
    main()
