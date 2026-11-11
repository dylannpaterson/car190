"""Hexagonal CMD densities and enclosed-population contours, static and blink."""

import json

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from plot_amber_high_ice_cmd import OUTPUT


def contour_grid(x,y,limits):
    grid,xe,ye=np.histogram2d(x,y,bins=(120,120),range=(limits[:2],limits[2:]))
    density=gaussian_filter(grid.astype(float),sigma=2.)
    ranked=np.sort(density.ravel())[::-1]
    cumulative=np.cumsum(ranked)/ranked.sum()
    levels=sorted(set(float(ranked[min(np.searchsorted(cumulative,fraction),len(ranked)-1)])
                      for fraction in (.5,.8)))
    return (xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,density.T,levels


def draw(axis,x,y,limits,norm,contours):
    artist=axis.hexbin(x,y,gridsize=55,extent=tuple(limits),mincnt=1,
                      cmap="viridis",norm=norm,linewidths=0)
    cx,cy,density,levels=contours
    axis.contour(cx,cy,density,levels=levels,colors="black",linewidths=1.,alpha=.85)
    axis.set_xlim(*limits[:2])
    axis.set_ylim(limits[3],limits[2])
    axis.set_xlabel("F087 − F146 (mag)")
    axis.grid(alpha=.12)
    return artist


def main():
    stars=pd.read_csv(OUTPUT/"cmd_highest_ice_stars.csv",dtype={"source_id":str})
    summary=json.loads((OUTPUT/"cmd_highest_ice_summary.json").read_text())
    limits=summary["display_limits"]
    samples=[(stars.observed_color_mag.to_numpy(),stars.red_observed_mag.to_numpy()),
             (stars.corrected_color_mag.to_numpy(),stars.red_zero_ice_corrected_mag.to_numpy())]
    contours=[contour_grid(x,y,limits) for x,y in samples]
    # Obtain the true shared maximum for this identical hexagonal grid.
    scratch,axis=plt.subplots()
    maxima=[]
    for x,y in samples:
        image=axis.hexbin(x,y,gridsize=55,extent=tuple(limits),mincnt=1)
        maxima.append(float(image.get_array().max()))
        axis.clear()
    plt.close(scratch)
    norm=LogNorm(vmin=1,vmax=max(maxima))
    titles=("Observed — no ice correction","Corrected to zero ice")
    fig,axes=plt.subplots(1,2,figsize=(13.5,7.8),sharex=True,sharey=True)
    for axis,(x,y),contour,title in zip(axes,samples,contours,titles,strict=True):
        artist=draw(axis,x,y,limits,norm,contour)
        axis.set_title(title)
    axes[0].set_ylabel("F146 magnitude")
    caxis=fig.add_axes([.92,.19,.015,.64])
    fig.colorbar(artist,cax=caxis,label="Stars per hexagon (log scale)")
    fig.suptitle(f"Amber CMD: {summary['utc_date']} UTC · {len(stars):,} same stars · median ice {summary['median_ice_nm']:.1f} nm",y=.98)
    fig.text(.5,.025,"Black contours enclose 50% and 80% of the smoothed displayed density in each panel. Identical axes, hexagons and color scale.\n"
             "Actual ice-only correction: median +30.1 mmag in F087−F146, +9.3 mmag in F146. No roll offsets or stellar-level alignment.",
             ha="center",fontsize=9)
    fig.subplots_adjust(left=.075,right=.89,bottom=.15,top=.88,wspace=.1)
    static=OUTPUT/"cmd_highest_ice_hex_contours.png"
    fig.savefig(static,dpi=180,bbox_inches="tight")
    plt.close(fig)

    fig=plt.figure(figsize=(8,8),dpi=150)
    axis=fig.add_axes([.11,.15,.72,.73])
    caxis=fig.add_axes([.86,.15,.025,.73])
    heading=fig.text(.47,.96,"",ha="center",fontsize=15)
    fig.text(.47,.915,f"{summary['utc_date']} UTC · {len(stars):,} same stars · {summary['median_ice_nm']:.1f} nm ice",
             ha="center",fontsize=10)
    fig.text(.5,.025,"Contours: central 50% and 80% of smoothed displayed density. Fixed axes and color scale.\n"
             "Actual correction size, without amplification: +30.1 mmag in color, +9.3 mmag in F146.",
             ha="center",fontsize=9)
    frames=[]
    for index,((x,y),contour,title) in enumerate(zip(samples,contours,titles,strict=True)):
        axis.clear()
        artist=draw(axis,x,y,limits,norm,contour)
        axis.set_ylabel("F146 magnitude")
        heading.set_text(title)
        if index==0:fig.colorbar(artist,cax=caxis,label="Stars per hexagon (log scale)")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    plt.close(fig)
    stack=Image.new("RGB",(frames[0].width,2*frames[0].height))
    for i,frame in enumerate(frames):stack.paste(frame,(0,i*frame.height))
    palette=stack.quantize(colors=256)
    indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    animated=OUTPUT/"cmd_highest_ice_hex_contours_blink.gif"
    indexed[0].save(animated,save_all=True,append_images=indexed[1:],duration=[1000,1000],loop=0,disposal=2)
    print(static)
    print(animated)


if __name__=="__main__":
    main()
