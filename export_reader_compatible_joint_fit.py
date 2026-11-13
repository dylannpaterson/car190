"""Export OUR reader-compatible joint fit, with history compression only AFTER fitting.

Exports complex reflection modes and thickness/rate quantities, never AR
parameters. All H5 group/dataset paths match Eric's original release exactly.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.linalg import svd

import compare_eric_v3_amber_car201 as c
import run_joint_car190_amber_inverse_wavelength_maxent as joint
from assemble_eric_v3_ice_response_candidate import sha256_file, utc_mjd
from export_joint_ice_history_reference import deposition_factorizations_on_blocks
from experiment_joint_wfi01_fixed_ice import full_basis, LedModel
from support_aligned_modulation import effective_area_support
from package_joint_h5_unchanged_contract import inventory
from reader_compatible_joint_forward import ICE_N, FRESNEL, internal_reflection


def replace(h5, path, values):
    dtype=h5[path].dtype
    if dtype.kind=='S':
        values=np.asarray(values).astype(dtype)
    del h5[path]
    kwargs=dict(compression='gzip',shuffle=True) if np.asarray(values).ndim else {}
    return h5.create_dataset(path,data=values,**kwargs)


def metrics(values):
    values=np.asarray(values,float)
    return dict(rms=float(np.sqrt(np.mean(values**2))),p95_abs=float(np.quantile(np.abs(values),.95)),
                max_abs=float(np.max(np.abs(values))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector',required=True,choices=c.DETECTORS)
    parser.add_argument('--fit-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--mode-count',type=int,default=64)
    parser.add_argument('--history-spacing-days',type=float,default=2.)
    args=parser.parse_args()
    d=args.detector;sca=int(d[3:])
    output=args.output_dir or args.fit_dir/'eric_h5_export'
    output.mkdir(parents=True,exist_ok=True)
    model_path=output/'joint_modulated.h5'
    summary=json.loads((args.fit_dir/'summary.json').read_text())
    if summary.get('response_convention')!='eric-reader' or not summary.get('optimizer_success'):
        raise ValueError('Export requires a converged reader-compatible joint fit')
    with np.load(args.fit_dir/'joint_fit.npz') as archive:
        theta=np.asarray(archive['parameters'],float)
        saved=np.asarray(archive['stellar_prediction_mag'],float)
        valid=np.asarray(archive['stellar_valid'],bool)
        saved_ids=archive['source_ids'].astype(str)
        fit_wave=np.asarray(archive['wavelengths_nm'],float)
    with np.load(args.fit_dir/'joint_weekly_ice_fit.npz') as archive:
        times=np.asarray(archive['times'],float)
        fitted_history=np.asarray(archive['fitted_thickness_nm'],float)
    area_path,_=c.amber.canonical_configuration(d)
    bounds=effective_area_support(area_path)
    # Eric's reader requires the existing film and reflectance wavelength axes
    # to be identical. Project the frozen fit onto that exact released H5 grid.
    with h5py.File(c.DEFAULT_ERIC_V3_MODEL,'r') as original:
        wavelengths=np.asarray(original['reflectance/wavelength_nm'],float)
    optical=c.workflow._optical_tile_arrays(d,wavelengths,Path(summary['optical_cache_root']))
    air0=optical[0]
    np.testing.assert_allclose(optical[-1],ICE_N,rtol=0,atol=1e-12)
    basis=full_basis(wavelengths,bounds)[0]
    mode=len(basis)
    air=air0*np.exp(theta[:mode]@basis+1j*(theta[mode:2*mode]@basis))[None,:]
    reflection,_=internal_reflection(air)
    mean=reflection.mean(axis=0)
    left,singular,right=svd(reflection-mean[None,:],full_matrices=False,check_finite=False)
    rank=min(args.mode_count,len(singular))
    amplitude=left[:,:rank]*singular[:rank]
    modes=right[:rank]
    optical_error=metrics(np.abs(reflection-(mean[None,:]+amplitude@modes)))

    episodes=np.searchsorted(joint.RESETS,times,side='right')
    history_conversion=deposition_factorizations_on_blocks(
        times,episodes,fitted_history,spacing_days=args.history_spacing_days,multi_rank=1,
    )
    converted=history_conversion['rank1']
    log_a=converted['log_a'];d_off=converted['d_off_nm']
    blocks=history_conversion['blocks'];multipliers=converted['multipliers']
    reference=c.utc_mjd(c.joint_fit.run.workflow.fit_model.ICE_REFERENCE_UTC)
    timestamp=np.asarray(Time(reference+times,format='mjd',scale='utc',precision=6).isot).astype('U32')
    gray_offsets=np.asarray(summary['reader_gray_log_offsets_by_tile'],float)

    # Fit and export use Eric's original H5 LED quadrature verbatim.
    with h5py.File(c.DEFAULT_ERIC_V3_MODEL,'r') as original:
        led_wave=np.asarray(original['led_bands/wavelength_nm'][()],float)
        led_weights=np.asarray(original['led_bands/weight'][()],float)
    led=LedModel(
        bounds,detector=d,relative_response_root=Path(summary['relative_response_root']),
        led_wavelength_nm=led_wave,led_weights=led_weights,
    )

    with h5py.File(c.DEFAULT_ERIC_V3_MODEL) as base,h5py.File(model_path,'w') as out:
        required=inventory(base)
        for key,value in base.attrs.items():out.attrs[key]=value
        for key in base:base.copy(key,out)
        fine_sca=out['superpixels/sca'][()]
        fine_row=out['superpixels/row'][()].astype(int)
        fine_col=out['superpixels/col'][()].astype(int)
        selected=fine_sca==sca
        tile=(fine_row//16)*16+fine_col//16
        count=len(tile)
        replace(out,'reflectance/wavelength_nm',wavelengths)
        replace(out,'reflectance/mean',mean)
        replace(out,'reflectance/modes',modes)
        replace(out,'reflectance/singular_values',singular[:rank])
        replace(out,'reflectance/mode_labels',np.array([f'M{i:02d}' for i in range(rank)]))
        del out['reflectance/amplitudes']
        ds=out['reflectance'].create_dataset('amplitudes',shape=(count,rank),dtype=np.complex128,
            compression='gzip',shuffle=True,chunks=(2048,rank))
        for start in range(0,count,2048):
            stop=min(start+2048,count)
            block=np.full((stop-start,rank),np.nan+1j*np.nan)
            mask=selected[start:stop]
            block[mask]=amplitude[tile[start:stop][mask]]
            ds[start:stop]=block
        replace(out,'film/n',np.full(len(wavelengths),ICE_N))
        replace(out,'film/kappa',np.zeros(len(wavelengths)))
        replace(out,'film/b',np.full(len(wavelengths),FRESNEL,dtype=complex))
        out['film'].attrs['n_snell_ref']=ICE_N
        out['film'].attrs['chromatic_snell']=False
        out['film'].attrs['index']='our fixed TMM ice: lossless n=1.25'
        out['superpixels/cos_theta'][...]=1.
        out['superpixels/reportable'][...]=selected.astype(out['superpixels/reportable'].dtype)
        out['superpixels/dq_flags'][...]=0
        if 'quality_mask' in out['superpixels']:
            out['superpixels/quality_mask'][...]=0
        la=np.full(count,np.nan);la[selected]=log_a[tile[selected]]
        offsets=np.full((count,d_off.shape[1]),np.nan);offsets[selected]=d_off[tile[selected]]
        rates=np.full((count,len(multipliers)),np.nan)
        rates[selected]=converted['rate_nm_per_day'][tile[selected]]
        replace(out,'deposition/log_a',la)
        replace(out,'deposition/d_off',offsets)
        replace(out,'deposition/rate_nm_per_day',rates)
        # The source joint fit does not provide a rate-control covariance.
        # Preserve the contract field without misrepresenting Eric's old values.
        del out['deposition/rate_param_cov']
        out['deposition'].create_dataset(
            'rate_param_cov',
            data=np.full((len(multipliers)-1,len(multipliers)-1),np.nan),
            compression='gzip',shuffle=True,
        )
        replace(out,'deposition/blocks',blocks)
        replace(out,'deposition/multipliers',multipliers)
        replace(out,'deposition/time_hours',times*24.)
        replace(out,'deposition/episode',episodes.astype(np.int64))
        replace(out,'deposition/timestamp_utc',np.array([str(t)+'+00:00' for t in timestamp]))
        replace(out,'deposition/tau_h',np.empty(0))
        replace(out,'deposition/offset_episodes',np.empty(0,dtype=np.int64))
        replace(out,'deposition/rate_param_cov',np.full((len(multipliers)-1,)*2,np.nan))
        out['deposition'].attrs['t0_utc']=c.joint_fit.run.workflow.fit_model.ICE_REFERENCE_UTC+'+00:00'
        # The original H5 LED nodes and weights are deliberately retained.
        for key in ('steps','shared_gain_coef','thermal_gain_local'):
            out[f'instrument/{key}'][...]=0.
        for key in ('reference_level','ref_led_level'):
            out[f'instrument/{key}'][...]=1.
        out['instrument/temperature_k'][...]=np.nan
        g=np.full((count,2),np.nan)
        g[selected,0]=np.expm1(gray_offsets[tile[selected],1])
        g[selected,1]=np.exp(gray_offsets[tile[selected],2])-np.exp(gray_offsets[tile[selected],1])
        replace(out,'instrument/gray',g)
        base_t0=c.DEFAULT_ERIC_V3_MODEL
        with h5py.File(base_t0,'r') as original:
            gray_shift=(utc_mjd(original['deposition'].attrs['t0_utc'])
                        -reference)*24.
            step_hours=original['instrument/step_boundary_hours'][()]+gray_shift
        # These gray coefficients belong to OUR fit and switch at OUR fixed
        # reset days. Eric's release epochs are not their boundaries.
        replace(out,'instrument/gray_period_hours',joint.RESETS*24.)
        replace(out,'instrument/step_boundary_hours',step_hours)
        for label,first in [('first',True),('last',False)]:
            idx=np.array([np.flatnonzero(episodes==e)[0 if first else -1] for e in range(3)])
            replace(out,f'time/episode_{label}_hours',times[idx]*24.)
            replace(out,f'time/episode_{label}_utc',np.array([str(timestamp[i])+'+00:00' for i in idx]))
        out['provenance/model'][()] = json.dumps(dict(model='our joint TMM modulation',detector=d))
        out['provenance'].attrs['source_fit']=str(args.fit_dir.resolve())
        out['provenance'].attrs['chi2_dof']=np.nan
        out['provenance'].attrs['n_reportable']=int(selected.sum())
        out.attrs['release_version']=f'{d.lower()}_joint_reader_compatible'
        out.attrs['n_modes']=rank
        out.attrs['release_utc']=datetime.now(timezone.utc).isoformat()
        out.attrs['source_fit']=str(args.fit_dir.resolve())
        out.attrs['release_source']=str(args.fit_dir.resolve())
        out.attrs['release_source_sha256']=sha256_file(args.fit_dir/'joint_fit.npz')
        out.attrs['basis']='our fitted TMM modulation, ice-side complex reflection PCA'
        out.attrs['release_note']=(f'{d} ONLY. Joint TMM modulation with full independent tile histories '
            'in the fit; history compression performed only for this H5 export. Other detectors return NaN.')
        if inventory(out)!=required:
            diff=[(key,required.get(key),inventory(out).get(key)) for key in set(required)|set(inventory(out))
                  if required.get(key)!=inventory(out).get(key)]
            raise ValueError(f'Output contract differs: {diff}')

    # Test the exported file through the original source archive reader.
    ids,seds,mag,err,reconstructed_valid,common,frames=joint.raw_common_training_sample(d)
    np.testing.assert_array_equal(ids,saved_ids)
    np.testing.assert_array_equal(valid,reconstructed_valid)
    fit_grid,fit_areas=c.workflow._make_wavelength_grid(5.,effective_area_path=area_path,full_effective_area_support=True)
    np.testing.assert_array_equal(fit_grid,fit_wave)
    with h5py.File(model_path,'r') as h5:
        h5_grid=np.asarray(h5['reflectance/wavelength_nm'],float)
    represented=(fit_grid>=h5_grid[0])&(fit_grid<=h5_grid[-1])
    full_area_integral=np.trapezoid(fit_areas,fit_grid,axis=1)
    wave,areas=fit_grid[represented],fit_areas[:,represented]
    represented_fraction=np.trapezoid(areas,wave,axis=1)/full_area_integral
    photon,_,fallback=c.workflow.fit_model.reconstruct_sed_photon_shapes(ids,seds,c.workflow.SED_GRID_PATH,wave)
    if fallback:raise ValueError('SED fallback in verification')
    tiles=c.workflow._source_tile_ids(ids,frames)
    day=np.array([[joint.common_pass_mjd(frames,band,p)-reference for p in c.amber.PASSES] for band in c.amber.FILTERS])
    indices=np.searchsorted(times,day)
    np.testing.assert_allclose(times[indices],day,rtol=0,atol=1e-10)
    kernel=[photon*(area*c.workflow.fit_model.trapezoid_weights(wave))[None,:] for area in areas]
    roll=np.zeros_like(saved)
    groups=np.asarray(c.workflow.fit_model.PASS_SET_CONFIG['ep3v2_ep4']['zero_point_group_ids'])
    offsets=theta[2*mode:].reshape(8,int(groups.max()))
    for pi,group in enumerate(groups):
        if group:roll[:,:,pi]=-offsets[:,group-1][None,:]/c.workflow.LOG_FLUX_PER_MAG

    with c.load_eric_v3_reader(c.DEFAULT_SOURCE_ZIP) as reader:
        model=reader.IceResponseModel(model_path)
        tileids=np.arange(256);x=(tileids%16)*256+128;y=(tileids//16)*256+128
        stage_times=np.r_[0.,joint.RESETS+1e-7]*24.
        stage_gray=model.gray(sca,x,y,stage_times)
        expected_gray=np.exp(gray_offsets)
        np.testing.assert_allclose(stage_gray,expected_gray,rtol=1e-12,atol=1e-14)
        gray_max_error=float(np.max(np.abs(stage_gray-expected_gray)))
        internal=model.reflectance(sca,x,y,wave)
        clean=reader.core.no_ice_transmission(internal,np.full(len(wave),FRESNEL))
        exact_log=np.empty_like(saved);portable_log=np.empty_like(saved)
        for fi in range(8):
            for pi in range(len(c.amber.PASSES)):
                hours=day[fi,pi]*24.
                approx=model.response(sca,x,y,[hours],wave,gray=True)[:,0,:]
                depth=fitted_history[:,indices[fi,pi]]
                exact=reader.core.relative_response(internal,wave,depth[:,None],np.ones(256),
                    model.wavelength_nm,model._n,model._kappa,model._b,model._n_ref,model._snell)[:,:,0]
                exact*=model.gray(sca,x,y,[hours])
                selected=tiles[:,fi,pi]
                weight=kernel[fi]*clean[selected]
                denominator=weight.sum(axis=1)
                exact_log[:,fi,pi]=np.log((weight*exact[selected]).sum(axis=1)/denominator)
                portable_log[:,fi,pi]=np.log((weight*approx[selected]).sum(axis=1)/denominator)
        exact_prediction=-(exact_log-exact_log[:,:,:1])/c.workflow.LOG_FLUX_PER_MAG+roll
        portable_prediction=-(portable_log-portable_log[:,:,:1])/c.workflow.LOG_FLUX_PER_MAG+roll
        api_checks=[]
        for fi in range(8):
            si=int(np.flatnonzero(valid[:,fi,0]&valid[:,fi,-1])[0])
            signal=[]
            for pi in (0,len(c.amber.PASSES)-1):
                tile=tiles[si,fi,pi]
                signal.append(model.band_response(sca,[x[tile]],[y[tile]],
                    [day[fi,pi]*24.],wave,kernel[fi][si],gray=True,no_ice_weight=True)[0,0])
            predicted=-np.log(signal[1]/signal[0])/c.workflow.LOG_FLUX_PER_MAG+roll[si,fi,-1]
            api_checks.append(float(abs(predicted-portable_prediction[si,fi,-1])))
        if max(api_checks)>1e-10:raise ValueError('Public band_response verification failed')
        use=valid&valid[:,:,:1];use[:,:,0]=False
        rows=[]
        for fi,band in enumerate(c.amber.FILTERS):
            mask=use[:,fi]
            rows.append(dict(filter=band,comparisons=int(mask.sum()),
                optical_and_normalization_error_mmag=metrics(1000*(exact_prediction-saved)[:,fi][mask]),
                history_compression_error_mmag=metrics(1000*(portable_prediction-exact_prediction)[:,fi][mask]),
                total_export_error_mmag=metrics(1000*(portable_prediction-saved)[:,fi][mask])))
        np.savez_compressed(output/'verification_predictions.npz',source_ids=np.asarray(ids),
            magnitude=mag,valid=valid,roll_offset_mag=roll,fit_prediction_mag=saved,
            exact_history_h5_prediction_mag=exact_prediction,exported_h5_prediction_mag=portable_prediction)

        # Independently check CAR190 via the exact existing LED public API.
        led_checks=[]
        for bi,band in enumerate(led.bands):
            days=band[0][9]
            native_band=model.band_response(sca,x,y,days*24.,model.led_nodes_nm[bi],model.led_weight[bi],
                                           gray=True,no_ice_weight=True)
            log_change=np.log(native_band)-np.log(native_band[:,:1])
            observed=np.stack([item[7] for item in band])
            sigma=np.stack([item[8] for item in band])
            led_checks.append(dict(led=bi+1,exported_chi2=float(np.sum(((log_change-observed)/sigma)**2))))

    report=dict(detector=d,fit_converged=True,full_independent_ice_parameters=summary['total_ice_parameters'],
        export_only_history_approximation=True,history_fit_metrics=converted['metrics'],
        h5_grid_wavelength_min_nm=float(wave[0]),
        h5_grid_wavelength_max_nm=float(wave[-1]),
        represented_effective_area_integral_fraction_by_filter={
            band:float(value) for band,value in zip(c.amber.FILTERS,represented_fraction,strict=True)
        },
        optical_reflection_compression_error=optical_error,stellar_checks=rows,led_checks=led_checks,
        fit_car190_chi2=summary['car190_chi2'],unchanged_public_band_response_max_abs_error_mag=max(api_checks),
        own_fitted_gray_factors_public_reader_max_abs_error=gray_max_error,
        gray_decon_days=joint.RESETS.tolist(),gray_source=summary.get('reader_gray_source'),
        reader_modified=False,h5_contract_matches_original=True,AR_parameters_exported=False,
        source_fit=str(args.fit_dir.resolve()),h5=str(model_path.resolve()),h5_sha256=sha256_file(model_path))
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
