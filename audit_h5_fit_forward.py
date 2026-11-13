"""Compare ported response to fitted forward arrays without exporting optical fit parameters."""

import argparse
import json

import numpy as np
import pandas as pd

import compare_eric_v3_amber_car201 as c
from car201_photometry_forward_jacobian import CachedPhotometryForward
from astropy.time import Time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', required=True)
    args = parser.parse_args()
    d = args.detector
    ids, seds, mag, err, valid, common, frames = c.joint_fit.raw_common_training_sample(d)
    fit_dir = c.DEFAULT_STATISTICS / 'production' / d.lower() / c.FIT_NAME
    summary = json.loads((fit_dir / 'summary.json').read_text())
    area, _ = c.amber.canonical_configuration(d)
    model = c.joint_fit.model_for_raw_sample(
        d, ids, seds, valid, frames, area,
        c.Path(summary['ice_history_fit_root']), c.Path(summary['optical_cache_root']),
    )
    with np.load(fit_dir / 'joint_fit.npz') as archive:
        theta = archive['parameters']
        saved = archive['stellar_prediction_mag']
    with np.load(fit_dir / 'joint_weekly_ice_fit.npz') as archive:
        times = archive['times']
        history = archive['fitted_thickness_nm']
    reference = c.utc_mjd(c.joint_fit.run.workflow.fit_model.ICE_REFERENCE_UTC)
    days = np.array([[c.joint_fit.common_pass_mjd(frames, b, p)-reference
                      for p in c.amber.PASSES] for b in c.amber.FILTERS])
    indices = np.searchsorted(times, days)
    np.testing.assert_allclose(times[indices], days, atol=1e-10, rtol=0)
    model.tile_thickness_by_id = history[:, indices]
    forward = CachedPhotometryForward(model)
    si = common.source_id.map({s:i for i,s in enumerate(ids)}).to_numpy(int)
    fi_rows = common['filter'].map({s:i for i,s in enumerate(c.amber.FILTERS)}).to_numpy(int)
    pi_rows = common['pass'].map({s:i for i,s in enumerate(c.amber.PASSES)}).to_numpy(int)
    row_tiles = (common.y_pixel.to_numpy(float)//256).astype(int)*16+(common.x_pixel.to_numpy(float)//256).astype(int)
    print('tile discrepancies',np.count_nonzero(row_tiles!=model.source_tile_ids[si,fi_rows,pi_rows]),flush=True)
    fit_log, _ = forward.evaluate(theta[:2*model.mode_count])
    physical_fit_mag = -fit_log / c.workflow.LOG_FLUX_PER_MAG
    groups = model.pass_zero_point_group_ids
    zp = theta[2*model.mode_count:].reshape(8, model.zero_point_group_count)
    roll = np.zeros_like(saved)
    for pi, group in enumerate(groups):
        if group:
            roll[:, :, pi] = -zp[:, group-1]/c.workflow.LOG_FLUX_PER_MAG
    np.testing.assert_allclose(physical_fit_mag+roll, saved, atol=2e-6, rtol=2e-6)

    with c.load_eric_v3_reader(c.DEFAULT_SOURCE_ZIP) as reader:
        h5 = reader.IceResponseModel(c.BUNDLE_DIR/'ice_response_model_tmm_joint_iceside_20261111_prototype.h5')
        wave = model.wavelengths_nm
        supported = (wave >= h5.wavelength_nm[0]) & (wave <= h5.wavelength_nm[-1])
        w = wave[supported]
        tiles = np.arange(256)
        x = (tiles % 16)*256+128
        y = (tiles // 16)*256+128
        rice = h5.reflectance(int(d[3:]), x, y, w)
        adm = 1.25*(1-rice)/(1+rice)
        rair = (1-adm)/(1+adm)
        n = model.mode_count
        ratio = np.exp(theta[:n]@model.basis+1j*(theta[n:2*n]@model.basis))
        fitted_rair = (model.clean_reflection_by_tile*ratio)[..., supported]
        print('reflection error',np.max(np.abs(rair-fitted_rair)),flush=True)
        # Clean modulated / clean unmodulated factor from the fitted optical curves.
        fit_weight = np.abs((1+fitted_rair)/(1+model.clean_reflection_by_tile[:, supported]))**2
        logs = np.empty_like(saved)
        for fi in range(8):
            kernel = model.star_photon_shapes[:, supported]*(model.effective_area[fi]*c.workflow.fit_model.trapezoid_weights(wave))[None, supported]
            for pi in range(len(c.amber.PASSES)):
                thickness = model.tile_thickness_by_id[:, fi, pi]
                response = reader.core.relative_response(
                    rice, w, thickness[:, None], np.ones(256), h5.wavelength_nm,
                    h5._n,h5._kappa,h5._b,h5._n_ref,h5._snell,
                )[:, :, 0]
                st = model.source_tile_ids[:, fi, pi]
                logs[:,fi,pi] = np.log(np.sum(kernel*response[st]*fit_weight[st],axis=1))
        predicted = -(logs-logs[:,:,:1])/c.workflow.LOG_FLUX_PER_MAG
        rows=[]
        for fi,b in enumerate(c.amber.FILTERS):
            use=valid[:,fi,:]&valid[:,fi,:1];use[:,0]=False
            delta=(predicted[:,fi,:]-physical_fit_mag[:,fi,:])[use]
            rows.append(dict(filter=b,rms_mmag=1000*np.sqrt(np.mean(delta**2)),median_mmag=1000*np.median(delta),max_mmag=1000*np.max(np.abs(delta))))
        print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
