"""Isolate deposition approximation error while holding the ported H5 optics fixed."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_eric_v3_amber_car201 as compare
import run_joint_car190_amber_inverse_wavelength_maxent as joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', action='append', required=True)
    parser.add_argument('--fit-grid', action='store_true')
    args = parser.parse_args()
    root = compare.BUNDLE_DIR
    output = root / 'corrected_h5_car201_plots'
    baseline = pd.read_csv(
        output / 'amber_eric_v3_vs_joint_modulated_measurement_predictions.csv',
        dtype={'source_id': str},
    )
    records = []
    with compare.load_eric_v3_reader(compare.DEFAULT_SOURCE_ZIP) as reader:
        model = reader.IceResponseModel(
            root / 'ice_response_model_tmm_joint_iceside_20261111_prototype.h5'
        )
        original_response = model.response
        for detector in args.detector:
            ids, seds, mag, valid, common, fit_dir, summary, roll, fitted = (
                compare.check_joint_fit_sample(detector, compare.DEFAULT_STATISTICS)
            )
            with np.load(fit_dir / 'joint_weekly_ice_fit.npz') as archive:
                times = np.asarray(archive['times'], float)
                history = np.asarray(archive['fitted_thickness_nm'], float)
            reference_mjd = compare.utc_mjd(joint.run.workflow.fit_model.ICE_REFERENCE_UTC)
            time_shift = compare.utc_mjd(model.t0_utc) - reference_mjd
            resets = np.asarray(joint.RESETS, float)
            stages = np.searchsorted(resets, times, side='right')
            use_fit_epochs = False
            epoch_map = {}
            for _, cell in common.groupby(['filter', 'pass']):
                median_day = float(cell.mjd_amber.median()) - reference_mjd
                for mjd in cell.mjd_amber:
                    epoch_map[round((float(mjd)-compare.utc_mjd(model.t0_utc))*24, 8)] = median_day

            def exact_response(sca, x, y, t, wavelength_nm=None, gray=True):
                sp = model.superpixel(sca, x, y)
                hours = model.hours(t)
                days = hours / 24 + time_shift
                if use_fit_epochs:
                    days = np.array([epoch_map[round(float(h), 8)] for h in hours])
                target_stages = np.searchsorted(resets, days, side='right')
                tiles = (model._row[sp] // 16) * 16 + model._col[sp] // 16
                thickness = np.empty((len(sp), len(days)))
                for stage in np.unique(target_stages):
                    source = stages == stage
                    target = target_stages == stage
                    if (days[target].min() < times[source].min() - 1e-8
                            or days[target].max() > times[source].max() + 1e-8):
                        raise ValueError('Exact history interpolation would extrapolate')
                    for i, tile in enumerate(tiles):
                        thickness[i, target] = np.interp(
                            days[target], times[source], history[tile, source]
                        )
                wave = model.wavelength_nm if wavelength_nm is None else np.asarray(wavelength_nm)
                reflection = model.reflectance(sca, x, y, wave)
                response = reader.core.relative_response(
                    reflection, wave, thickness, model._cos_theta[sp],
                    model.wavelength_nm, model._n, model._kappa, model._b,
                    model._n_ref, model._snell,
                )
                response[:, :, ~model.valid(hours)] = np.nan
                if gray and not use_fit_epochs:
                    response *= model.gray(sca, x, y, hours)[:, None, :]
                return response.transpose(0, 2, 1)

            area_path, _ = compare.amber.canonical_configuration(detector)
            wave = np.asarray(model.wavelength_nm, float)
            areas, _ = compare.effective_area_on_model_grid(detector, area_path, wave)
            if args.fit_grid:
                wave, areas = compare.workflow._make_wavelength_grid(
                    5.0, effective_area_path=area_path, full_effective_area_support=True
                )
                inside = (wave >= model.wavelength_nm[0]) & (wave <= model.wavelength_nm[-1])
                wave, areas = wave[inside], areas[:, inside]
            photon, _, fallback = compare.workflow.fit_model.reconstruct_sed_photon_shapes(
                ids, seds, compare.workflow.SED_GRID_PATH, wave
            )
            if fallback:
                raise ValueError('Unexpected SED fallback')
            model.response = exact_response
            try:
                exact, _, check = compare.evaluate_model_on_sample(
                    model, reader.core, detector, ids, seds, common, wave, photon, areas
                )
                original_transmission = reader.core.no_ice_transmission
                reader.core.no_ice_transmission = lambda r, b: np.ones_like(r.real)
                try:
                    flat_weight, _, _ = compare.evaluate_model_on_sample(
                        model, reader.core, detector, ids, seds, common, wave, photon, areas
                    )
                finally:
                    reader.core.no_ice_transmission = original_transmission
                import h5py
                with h5py.File(model.path, 'r') as h5:
                    names = h5['joint_modulation_reference/detectors'].asstr()[()]
                    index = int(np.flatnonzero(names == detector)[0])
                    mw = h5['joint_modulation_reference/wavelength_nm'][()]
                    ratio = h5['joint_modulation_reference/ratio'][index]
                ratio = np.interp(wave, mw, ratio.real) + 1j*np.interp(wave, mw, ratio.imag)

                def fit_weight(r, b):
                    admittance = 1.25 * (1-r)/(1+r)
                    air = (1-admittance)/(1+admittance)
                    unmodulated_air = air / ratio[None, :]
                    return np.abs((1+air)/(1+unmodulated_air))**2

                reader.core.no_ice_transmission = fit_weight
                try:
                    fit_normalized, _, _ = compare.evaluate_model_on_sample(
                        model, reader.core, detector, ids, seds, common, wave, photon, areas
                    )
                    use_fit_epochs = True
                    matched_epochs, _, _ = compare.evaluate_model_on_sample(
                        model, reader.core, detector, ids, seds, common, wave, photon, areas,
                        normalize_band=False,
                    )
                    use_fit_epochs = False
                finally:
                    reader.core.no_ice_transmission = original_transmission
            finally:
                model.response = original_response
            rows = baseline.loc[baseline.detector == detector].copy()
            si = rows.source_id.map({s: i for i, s in enumerate(ids)}).to_numpy(int)
            fi = rows['filter'].map({s: i for i, s in enumerate(compare.amber.FILTERS)}).to_numpy(int)
            pi = rows['pass'].map({s: i for i, s in enumerate(compare.amber.PASSES)}).to_numpy(int)
            rows['exact_ice_prediction_mag'] = exact[si, fi, pi] + roll[si, fi, pi]
            rows['exact_ice_no_extra_transmission_prediction_mag'] = (
                flat_weight[si, fi, pi] + roll[si, fi, pi]
            )
            rows['fit_normalized_prediction_mag'] = fit_normalized[si, fi, pi] + roll[si, fi, pi]
            rows['matched_epochs_prediction_mag'] = matched_epochs[si, fi, pi] + roll[si, fi, pi]
            rows['approx_ice_prediction_mag'] = (
                rows.joint_modulated_model_delta_mag + rows.amber_roll_offset_mag
            )
            rows['original_fit_prediction_mag'] = fitted[si, fi, pi]
            for band, group in rows.groupby('filter'):
                a = group.approx_ice_prediction_mag.to_numpy()
                e = group.exact_ice_prediction_mag.to_numpy()
                f = group.original_fit_prediction_mag.to_numpy()
                flat = group.exact_ice_no_extra_transmission_prediction_mag.to_numpy()
                normalized = group.fit_normalized_prediction_mag.to_numpy()
                matched = group.matched_epochs_prediction_mag.to_numpy()
                record = dict(
                    detector=detector, filter=band, comparisons=len(group),
                    approx_vs_fit_rms_mmag=1000*np.sqrt(np.mean((a-f)**2)),
                    exact_vs_fit_rms_mmag=1000*np.sqrt(np.mean((e-f)**2)),
                    ice_approximation_rms_mmag=1000*np.sqrt(np.mean((a-e)**2)),
                    approx_vs_fit_median_mmag=1000*np.median(a-f),
                    exact_vs_fit_median_mmag=1000*np.median(e-f),
                    no_extra_transmission_vs_fit_rms_mmag=1000*np.sqrt(np.mean((flat-f)**2)),
                    fit_normalized_vs_fit_rms_mmag=1000*np.sqrt(np.mean((normalized-f)**2)),
                    matched_epochs_vs_fit_rms_mmag=1000*np.sqrt(np.mean((matched-f)**2)),
                )
                records.append(record)
            print(pd.DataFrame(records).query('detector == @detector').to_string(index=False), flush=True)
            suffix = '_fit_grid' if args.fit_grid else ''
            rows.to_csv(output / f'{detector.lower()}_exact_ice_diagnostic{suffix}.csv', index=False)
    suffix = '_fit_grid' if args.fit_grid else ''
    pd.DataFrame(records).to_csv(output / f'h5_exact_ice_error_attribution{suffix}.csv', index=False)


if __name__ == '__main__':
    main()
