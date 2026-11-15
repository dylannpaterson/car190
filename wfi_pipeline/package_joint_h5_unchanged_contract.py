"""Package the fitted joint response with Eric's original dataset contract.

No model fitting, extra H5 groups, custom band weights, or reader modifications.
Retains the existing rank-one deposition approximation and reports its error.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

import assemble_eric_v3_ice_response_candidate as assembly
import compare_eric_v3_amber_car201 as comparison


ROOT = comparison.BUNDLE_DIR
DEFAULT_SOURCE = ROOT / 'ice_response_model_tmm_joint_iceside_20261111_prototype.h5'
DEFAULT_OUTPUT = ROOT / 'unchanged_reader_port/joint_modulated.h5'


def inventory(h5):
    result = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            result[name] = ('dataset', obj.ndim, obj.dtype.kind)
        else:
            result[name] = ('group',)
    h5.visititems(visit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fitted_gray_source_bundle = None
    fitted_gray_by_detector = {}
    fitted_gray_reset_hours = None
    with h5py.File(comparison.DEFAULT_ERIC_V3_MODEL) as base, h5py.File(args.source) as source:
        required = inventory(base)
        fitted_gray_source_bundle = Path(
            source['instrument'].attrs['fitted_gray_source_bundle']
        )
        fitted_gray_reset_hours = np.asarray(
            source['instrument/gray_period_hours'], dtype=float
        )
        with np.load(fitted_gray_source_bundle, allow_pickle=False) as bundle:
            bundle_detectors = np.asarray(bundle['detectors']).astype(str)
            bundle_tile_ids = np.asarray(bundle['tile_ids'], dtype=int)
            if 'reset_event_tile_log_offsets' in bundle:
                event_offsets = np.asarray(bundle['reset_event_tile_log_offsets'], dtype=float)
                event_names = np.asarray(bundle.get('reset_event_names', [])).astype(str).reshape(-1)
            else:
                event_offsets = np.column_stack((
                    np.asarray(bundle['post_reset_tile_log_offset'], dtype=float),
                    np.asarray(bundle['post_third_decon_tile_log_offset'], dtype=float),
                ))
                event_names = np.asarray(['program1063_reset', 'third_decon'])
        if (
            bundle_detectors.shape != bundle_tile_ids.shape
            or event_offsets.ndim != 2
            or event_offsets.shape[0] != bundle_tile_ids.size
            or event_offsets.shape[1] != fitted_gray_reset_hours.size
        ):
            raise ValueError('Fitted gray source arrays do not share their tile axis')
        for detector in comparison.DETECTORS:
            selected = bundle_detectors == detector
            tile_ids = bundle_tile_ids[selected]
            if tile_ids.size != 256 or set(tile_ids.tolist()) != set(range(256)):
                raise ValueError(f'{detector}: fitted gray source does not contain all 256 tiles')
            order = np.argsort(tile_ids)
            fitted_gray_by_detector[detector] = event_offsets[selected][order]
        with h5py.File(args.output, 'w') as out:
            # Start with exactly the released contract. All replacements are
            # existing fields, not optional groups understood only by our code.
            for key, value in base.attrs.items():
                out.attrs[key] = value
            for key in base:
                base.copy(key, out)
            for group_name in ('reflectance', 'deposition'):
                for key, old in base[group_name].items():
                    if key not in source[group_name]:
                        # Covariance of the shared rate parameters is not
                        # available for the translated model; never reuse Eric's.
                        if key != 'rate_param_cov':
                            raise ValueError(f'Missing source field {group_name}/{key}')
                        count = max(0, source['deposition/multipliers'].size-1)
                        values = np.full((count, count), np.nan)
                    else:
                        values = source[group_name][key][()]
                        if old.dtype.kind == 'S':
                            values = np.asarray(values).astype(old.dtype)
                    del out[group_name][key]
                    if np.asarray(values).ndim:
                        out[group_name].create_dataset(
                            key, data=values, compression='gzip', shuffle=True,
                        )
                    else:
                        out[group_name].create_dataset(key, data=values)
                if group_name == 'deposition':
                    out[group_name].attrs['t0_utc'] = source[group_name].attrs['t0_utc']
            for key in ('n', 'kappa', 'b'):
                out[f'film/{key}'][...] = source[f'film/{key}'][()]
            for key in base['film'].attrs:
                if key in source['film'].attrs:
                    out['film'].attrs[key] = source['film'].attrs[key]
            out['film'].attrs['index'] = 'source joint TMM: lossless n=1.25'
            for key in ('cos_theta', 'reportable', 'dq_flags', 'quality_mask'):
                if key in source['superpixels'] and key in out['superpixels']:
                    out[f'superpixels/{key}'][...] = source[f'superpixels/{key}'][()]
            for key in ('wavelength_nm', 'weight', 'center_nm'):
                out[f'led_bands/{key}'][...] = source[f'led_bands/{key}'][()]

            # The replacement deposition and instrument arrays share the source
            # origin already. Shifting these a second time changes the model.
            for key in ('gray', 'gray_period_hours', 'step_boundary_hours'):
                source_values = source[f'instrument/{key}'][()]
                output_dataset = out[f'instrument/{key}']
                if output_dataset.shape != source_values.shape:
                    dtype = output_dataset.dtype
                    del out[f'instrument/{key}']
                    out['instrument'].create_dataset(
                        key,
                        data=np.asarray(source_values, dtype=dtype),
                        compression='gzip' if np.asarray(source_values).ndim else None,
                        shuffle=bool(np.asarray(source_values).ndim),
                    )
                else:
                    output_dataset[...] = source_values

            # Do not misrepresent Eric's unrelated fitted calibration values as
            # estimates for our model. These fields are stored, not applied by
            # IceResponseModel.response(), but are part of the fixed contract.
            for key in ('steps', 'shared_gain_coef', 'thermal_gain_local'):
                out[f'instrument/{key}'][...] = 0.
            for key in ('reference_level', 'ref_led_level'):
                out[f'instrument/{key}'][...] = 1.
            out['instrument/temperature_k'][...] = np.nan

            hours = out['deposition/time_hours'][()]
            episodes = out['deposition/episode'][()]
            utc = out['deposition/timestamp_utc'][()]
            unique = np.unique(episodes)
            for prefix, use_first in [('first', True), ('last', False)]:
                indices = np.array([
                    np.flatnonzero(episodes == e)[0 if use_first else -1] for e in unique
                ])
                for suffix, values in (
                    ('hours', hours[indices]),
                    ('utc', utc[indices]),
                ):
                    dataset_path = f'time/episode_{prefix}_{suffix}'
                    target = out[dataset_path]
                    values = np.asarray(values, dtype=target.dtype)
                    if target.shape != values.shape:
                        dtype = target.dtype
                        del out[dataset_path]
                        out['time'].create_dataset(dataset_path.split('/')[-1], data=values, dtype=dtype)
                    else:
                        target[...] = values

            out.attrs['n_modes'] = out['reflectance/modes'].shape[0]
            out.attrs['release_version'] = 'joint_modulated_unchanged_reader'
            out.attrs['release_utc'] = datetime.now(timezone.utc).isoformat()
            out.attrs['basis'] = 'joint TMM clean reflection transformed to ice-side modal basis'
            out.attrs['release_note'] = (
                'Deterministic packaging of fitted joint optics. Original rcs_response_model/1 '
                'dataset contract; no extra weighting fields. Ice histories retain the '
                'documented rank-one approximation. Not a new fit.'
            )
            out.attrs['source_fit'] = str(args.source.resolve())
            out.attrs['release_source'] = str(args.source.resolve())
            out.attrs['release_source_sha256'] = assembly.sha256_file(args.source)
            out['provenance'].attrs['source_fit'] = str(args.source.resolve())
            out['provenance'].attrs['chi2_dof'] = np.nan
            if inventory(out) != required:
                raise ValueError('Output dataset contract differs from Eric original H5')

    report = dict(
        output=str(args.output.resolve()),
        output_sha256=assembly.sha256_file(args.output),
        source=str(args.source.resolve()),
        base_contract=str(comparison.DEFAULT_ERIC_V3_MODEL.resolve()),
        contract_matches_original_paths_ranks_and_dtype_kinds=True,
        model_refitted=False,
        custom_weighting=False,
        extra_h5_groups=False,
        gray_policy='source-fit fixed tile-gray offsets copied into the original instrument fields',
        outstanding_limitations=[
            'Rank-one ice histories approximate the original independent tile histories.',
            'The frozen source model was fit on its native 5 nm spectral grid; the Eric H5 uses its original 1 nm grid.',
        ],
    )
    with comparison.load_eric_v3_reader(comparison.DEFAULT_SOURCE_ZIP) as reader:
        model = reader.IceResponseModel(args.output)
        wave = np.array([500., 622., 845., 1300., 2200.])
        tile_ids = np.arange(256)
        x = (tile_ids % 16) * 256 + 128
        y = (tile_ids // 16) * 256 + 128
        stage_hours = np.r_[0.0, fitted_gray_reset_hours / 24.0 + 1e-7] * 24.0
        gray_errors = {}
        for detector in comparison.DETECTORS:
            sca = int(detector[3:])
            actual_gray = np.asarray(model.gray(sca, x, y, stage_hours), dtype=float)
            log_offsets = fitted_gray_by_detector[detector]
            if event_names.size == 2 and event_names[1] == 'third_decon':
                expected_gray = np.exp(
                    np.column_stack((np.zeros(256), log_offsets))
                )
            else:
                cumulative_log_offsets = np.cumsum(log_offsets, axis=1)
                expected_gray = np.exp(
                    np.column_stack((np.zeros(256), cumulative_log_offsets))
                )
            np.testing.assert_allclose(
                actual_gray, expected_gray, rtol=2e-8, atol=2e-10,
                err_msg=f'{detector}: unchanged-reader gray factors differ from current fitted tiles',
            )
            gray_errors[detector] = float(np.max(np.abs(actual_gray - expected_gray)))
            response = np.asarray(
                model.response(sca, x, y, stage_hours, wave, gray=True), dtype=float
            )
            if not np.isfinite(response).all():
                raise ValueError(f'{detector}: unchanged reader returned a nonfinite response')
        report['fitted_gray_source_bundle'] = str(fitted_gray_source_bundle.resolve())
        report['fitted_gray_public_reader_max_abs_error_by_detector'] = gray_errors
        report['all_18_detectors_public_reader_responses_finite'] = True
        report['unchanged_reader_load_and_response_finite'] = True
        report['reader_model_repr'] = repr(model)
    args.output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
