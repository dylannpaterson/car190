"""Paired observed and zero-ice-corrected CMDs at the highest-ice fitted visit."""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from astropy.time import Time

import run_amber_student_t_detectors as run
from fit_car201_zane_star_reflectance_modulation import trapezoid_weights

OUTPUT=run.workflow.PRODUCTION_ROOT/"wavelength_response_bundle/amber_student_t_all_detectors"
BLUE,RED="F087","F146"


def zero_ice_log_ratio(model,parameters,filter_index,pass_index,zero_thickness=False):
    """Same-tile coated/zero-ice count ratio with the stellar-fit spectral weights.

    Only the fitted reflection modulation enters; stellar constant magnitudes,
    roll offsets and scatter scales are not used.
    """
    count=model.mode_count
    ratio=np.exp(parameters[:count]@model.basis+1j*(parameters[count:2*count]@model.basis))
    r=model.clean_reflection_by_tile*ratio[None,:]
    y0=model.incident_admittance_by_tile[:,None]
    clean_y=y0*(1-r)/(1+r)
    first=model.clean_transfer_first_by_tile
    second=first*clean_y
    thickness=model.tile_thickness_by_id[:,filter_index,pass_index,None]
    if zero_thickness:thickness=np.zeros_like(thickness)
    ice=model.ice_index_by_tile[:,None]
    phase=2*np.pi*ice*thickness/model.wavelengths_nm[None,:]
    cosine=np.cos(phase)
    sine=-1j*np.sin(phase)
    denominator=y0*(cosine*first+sine*second/ice)+(sine*ice*first+cosine*second)
    clean_denominator=y0*first+second
    numerator=np.real(model.substrate_admittance_by_tile/y0)*np.abs(2*y0)**2
    coated_power=numerator/np.abs(denominator)**2/model.clean_substrate_power_by_tile
    clean_power=numerator/np.abs(clean_denominator)**2/model.clean_substrate_power_by_tile
    kernel=(model.star_photon_shapes
            *(model.effective_area[filter_index]*trapezoid_weights(model.wavelengths_nm))[None,:])
    tiles=model.source_tile_ids[:,filter_index,pass_index]
    coated_rate=(kernel*coated_power[tiles]).sum(axis=1)
    clean_rate=(kernel*clean_power[tiles]).sum(axis=1)
    return np.log(coated_rate/clean_rate)


def main():
    global BLUE,RED,OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--blue-filter",default=BLUE,choices=run.FILTERS)
    parser.add_argument("--red-filter",default=RED,choices=run.FILTERS)
    parser.add_argument("--magnitude-filter",default=None,choices=run.FILTERS)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    BLUE,RED,OUTPUT=args.blue_filter,args.red_filter,args.output_dir
    magnitude_filter=args.magnitude_filter or RED
    all_records=[]
    zero_error=0.
    blue_index,red_index=run.FILTERS.index(BLUE),run.FILTERS.index(RED)
    magnitude_index=run.FILTERS.index(magnitude_filter)
    for detector in run.workflow.DETECTORS:
        if not run.successful(detector):raise RuntimeError(f"Missing fit for {detector}")
        _,directory=run.paths(detector)
        with np.load(directory/"predictions.npz",allow_pickle=False) as archive:
            parameters=archive["student_t_fit_parameters"]
            saved_ids=archive["source_ids"].astype(str)
        _,common,ids,magnitude,error,valid,frozen,model,_=run.prepare(detector,write_inputs=False)
        np.testing.assert_array_equal(saved_ids,np.asarray(ids))
        times=common.groupby(["filter","pass"])["mjd_amber"].median()
        for pass_index,epoch in enumerate(run.PASSES):
            matched=valid[:,blue_index,pass_index]&valid[:,red_index,pass_index]&valid[:,magnitude_index,pass_index]
            if not matched.any():continue
            zero_corrections=[]
            thicknesses=[]
            tiles=[]
            for fi in (blue_index,red_index,magnitude_index):
                logratio=zero_ice_log_ratio(model,parameters,fi,pass_index)
                check=zero_ice_log_ratio(model,parameters,fi,pass_index,zero_thickness=True)
                zero_error=max(zero_error,float(np.max(np.abs(check[matched]))))
                zero_corrections.append(-logratio/run.workflow.LOG_FLUX_PER_MAG)
                tile=model.source_tile_ids[:,fi,pass_index]
                tiles.append(tile)
                thicknesses.append(model.tile_thickness_by_id[tile,fi,pass_index])
            for si in np.flatnonzero(matched):
                blue=float(magnitude[si,blue_index,pass_index])
                red=float(magnitude[si,red_index,pass_index])
                db,dr=float(zero_corrections[0][si]),float(zero_corrections[1][si])
                vertical=float(magnitude[si,magnitude_index,pass_index])
                dv=float(zero_corrections[2][si])
                all_records.append({
                    "detector":detector,"source_id":ids[si],"pass":epoch,
                    "mjd_utc":float((times.loc[(BLUE,epoch)]+times.loc[(RED,epoch)])/2),
                    "blue_filter":BLUE,"red_filter":RED,
                    "magnitude_filter":magnitude_filter,
                    "blue_tile_id":int(tiles[0][si]),"red_tile_id":int(tiles[1][si]),
                    "blue_ice_nm":float(thicknesses[0][si]),"red_ice_nm":float(thicknesses[1][si]),
                    "mean_ice_nm":float((thicknesses[0][si]+thicknesses[1][si])/2),
                    "blue_observed_mag":blue,"red_observed_mag":red,
                    "blue_error_mag":float(error[si,blue_index,pass_index]),
                    "red_error_mag":float(error[si,red_index,pass_index]),
                    "blue_ice_effect_mag":db,"red_ice_effect_mag":dr,
                    "blue_zero_ice_corrected_mag":blue-db,"red_zero_ice_corrected_mag":red-dr,
                    "vertical_observed_mag":vertical,"vertical_zero_ice_corrected_mag":vertical-dv,
                    "observed_color_mag":blue-red,"corrected_color_mag":blue-db-red+dr,
                    "color_shift_mag":-db+dr,"vertical_shift_mag":-dv,
                })
        print(f"Computed same-tile zero-ice CMD corrections: {detector}",flush=True)
    data=pd.DataFrame(all_records)
    visits=data.groupby("pass").agg(star_measurements=("source_id","size"),
                                    median_ice_nm=("mean_ice_nm","median"),
                                    p16_ice_nm=("mean_ice_nm",lambda x:x.quantile(.16)),
                                    p84_ice_nm=("mean_ice_nm",lambda x:x.quantile(.84)),
                                    mjd_utc=("mjd_utc","median"))
    highest=visits["median_ice_nm"].idxmax()
    stars=data.loc[data["pass"]==highest].copy()
    if zero_error>1e-12:raise RuntimeError(f"Zero-thickness correction is not zero: {zero_error}")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data.to_csv(OUTPUT/"cmd_all_visits_stars.csv",index=False)
    visits.to_csv(OUTPUT/"cmd_visit_ice_thickness.csv")
    stars.to_csv(OUTPUT/"cmd_highest_ice_stars.csv",index=False)
    # Identical axes and points in both panels. A common density color scale
    # makes the locus comparable without overlapping 18 detector colors.
    color=np.concatenate([stars.observed_color_mag.to_numpy(),stars.corrected_color_mag.to_numpy()])
    mag=np.concatenate([stars.vertical_observed_mag.to_numpy(),stars.vertical_zero_ice_corrected_mag.to_numpy()])
    xlo,xhi=np.quantile(color,[.005,.995])
    ylo,yhi=np.quantile(mag,[.005,.995])
    xpad=.08*max(xhi-xlo,.1)
    ypad=.06*max(yhi-ylo,.1)
    limits=[float(xlo-xpad),float(xhi+xpad),float(ylo-ypad),float(yhi+ypad)]
    fig,axes=plt.subplots(1,2,figsize=(13.5,7.5),sharex=True,sharey=True)
    extent=(limits[0],limits[1],limits[2],limits[3])
    arrays=[]
    for x,y in ((stars.observed_color_mag,stars.vertical_observed_mag),
                (stars.corrected_color_mag,stars.vertical_zero_ice_corrected_mag)):
        grid,_,_=np.histogram2d(x,y,bins=(90,100),range=(limits[:2],limits[2:]))
        arrays.append(grid)
    vmax=max(float(grid.max()) for grid in arrays)
    norm=Normalize(vmin=0,vmax=np.log10(1+vmax))
    for axis,grid,title in zip(axes,arrays,("Observed — no ice correction","Corrected to zero ice"),strict=True):
        image=axis.imshow(np.log10(1+grid.T),origin="lower",extent=extent,aspect="auto",cmap="magma_r",norm=norm)
        axis.set_title(title)
        axis.set_xlabel(f"{BLUE} − {RED} (mag)")
        axis.set_xlim(limits[:2])
        axis.set_ylim(limits[3],limits[2])
        axis.grid(alpha=.12)
    axes[0].set_ylabel(f"{magnitude_filter} magnitude")
    fig.colorbar(image,ax=axes.tolist(),fraction=.025,pad=.03,label="log₁₀(1 + stars per bin)")
    date=Time(float(visits.loc[highest,"mjd_utc"]),format="mjd",scale="utc").strftime("%Y-%m-%d")
    color_shift=float(stars.color_shift_mag.median())
    vertical_shift=float(stars.vertical_shift_mag.median())
    fig.suptitle(f"Amber photometry at highest observed modeled ice: {date} UTC ({highest})\n"
                 f"Same {len(stars):,} detector/star pairs; median ice {visits.loc[highest,'median_ice_nm']:.1f} nm",y=.99)
    fig.text(.5,.025,f"Median correction: Δ({BLUE}−{RED}) = {color_shift:+.4f} mag; Δ{magnitude_filter} = {vertical_shift:+.4f} mag.\n"
             "Ice-only correction to the same tile at zero thickness, using the fitted Student-t model and each star's SED. No roll offsets or stellar-level alignment applied.\n"
             "Shared axes/density scale. Display limits use the pooled 0.5–99.5% range plus margins; all selected stars are retained in CSV.",
             ha="center",fontsize=9)
    fig.subplots_adjust(left=.075,right=.88,bottom=.18,top=.85,wspace=.12)
    # Place colorbar explicitly after panel placement to avoid overlap.
    cax=fig.axes[-1]
    cax.set_position([.91,.2,.018,.61])
    output=OUTPUT/"cmd_uncorrected_vs_zero_ice_highest_thickness.png"
    fig.savefig(output,dpi=180,bbox_inches="tight")
    plt.close(fig)
    result={"visit":highest,"utc_date":date,"star_detector_pairs":len(stars),
            "unique_gaia_sources":int(stars.source_id.nunique()),"median_ice_nm":float(visits.loc[highest,"median_ice_nm"]),
             "filters":[BLUE,RED],"magnitude_filter":magnitude_filter,"median_color_shift_mag":color_shift,"median_vertical_shift_mag":vertical_shift,
            "p16_color_shift_mag":float(stars.color_shift_mag.quantile(.16)),
            "p84_color_shift_mag":float(stars.color_shift_mag.quantile(.84)),
            "zero_thickness_check_max_abs_log_ratio":zero_error,"roll_offsets_applied":False,
            "stellar_level_alignment_applied":False,"correction_reference":"same tile at zero ice",
            "spectral_weighting":"same as stellar fitter: photon SED * reference effective area / unmodulated clean substrate power",
            "selection":"matched-quality Student-t fit measurements with valid photometry in both CMD filters at the selected visit",
            "display_limits":limits,"plot":str(output)}
    (OUTPUT/"cmd_highest_ice_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(visits.to_string())
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
