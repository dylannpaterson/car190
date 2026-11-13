"""Numerical model-to-model projection into the unchanged Eric H5 contract.

The target is ONLY the frozen source model's predicted magnitude changes.
Measured magnitudes, photometric uncertainties and a data likelihood are never
used in the projection objective. Original model parameters and histories are
not modified. Only the exported effective-reflectance representation changes.
"""

import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import svd

import compare_eric_v3_amber_car201 as c
from assemble_eric_v3_ice_response_candidate import sha256_file
from experiment_joint_wfi01_fixed_ice import full_basis
from package_joint_h5_unchanged_contract import DEFAULT_OUTPUT as SOURCE, inventory
from support_aligned_modulation import effective_area_support


class ResponseProjection:
    def __init__(self, model, reader, detector, ids, seds, common, valid, target, area_path, tile_local=False):
        self.detector = detector
        self.valid = valid & valid[:, :, :1]
        self.valid[:, :, 0] = False
        self.target = np.asarray(target, float)
        wave, area = c.workflow._make_wavelength_grid(
            5., effective_area_path=area_path, full_effective_area_support=True,
        )
        inside = (wave >= model.wavelength_nm[0]) & (wave <= model.wavelength_nm[-1])
        self.wave, area = wave[inside], area[:, inside]
        self.bounds = effective_area_support(area_path)
        self.basis = full_basis(self.wave, self.bounds)[0]
        self.count = len(self.basis)
        self.tile_local = tile_local
        self.local_basis = blue_basis(self.wave) if tile_local else np.empty((0,len(self.wave)))
        self.local_count = len(self.local_basis)
        self.parameter_count = 2*self.count+256*2*self.local_count
        self.sca = int(detector[3:])
        tiles = np.arange(256)
        self.x, self.y = (tiles % 16)*256+128, (tiles // 16)*256+128
        self.r0 = model.reflectance(self.sca, self.x, self.y, self.wave)
        frames = [common.loc[common['pass'] == p] for p in c.amber.PASSES]
        self.source_tiles = c.workflow._source_tile_ids(ids, frames)
        photon, _, fallback = c.workflow.fit_model.reconstruct_sed_photon_shapes(
            ids, seds, c.workflow.SED_GRID_PATH, self.wave,
        )
        if fallback:
            raise ValueError('Unexpected SED fallback')
        weights = c.workflow.fit_model.trapezoid_weights(self.wave)
        n, kappa, self.b = reader.core.optics_at(
            model.wavelength_nm, model._n, model._kappa, model._b, self.wave,
        )
        np.testing.assert_allclose(kappa, 0., atol=0., rtol=0.)
        self.cells = []
        self.hours = []
        reference = c.utc_mjd(model.t0_utc)
        for fi, band in enumerate(c.amber.FILTERS):
            supported = np.flatnonzero(area[fi] > 0.)
            kernel = photon[:, supported]*(area[fi, supported]*weights[supported])[None, :]
            local_phase, local_hours = [], []
            for pi, p in enumerate(c.amber.PASSES):
                day = float(common.loc[
                    (common['filter'] == band) & (common['pass'] == p), 'mjd_amber'
                ].median())
                hours = (day-reference)*24.
                if not model.valid([hours])[0]:
                    raise ValueError(f'{detector}/{band}/{p}: invalid model time')
                thickness = model.thickness(self.sca, self.x, self.y, [hours])[:, 0]
                local_phase.append(np.exp(
                    4j*np.pi*n[None, supported]*thickness[:, None]/self.wave[None, supported]
                ))
                local_hours.append(hours)
            self.cells.append((supported, kernel, local_phase))
            self.hours.append(local_hours)
        self.evaluations = 0

    def predict(self, theta, jacobian=True):
        count = self.count
        log_multiplier = theta[:count]@self.basis + 1j*(theta[count:2*count]@self.basis)
        log_multiplier = np.broadcast_to(log_multiplier,self.r0.shape).copy()
        if self.tile_local:
            local = theta[2*count:].reshape(256,2*self.local_count)
            log_multiplier += local[:,:self.local_count]@self.local_basis + 1j*(local[:,self.local_count:]@self.local_basis)
        multiplier = np.exp(log_multiplier)
        reflection = self.r0*multiplier
        rho2 = np.abs(reflection)**2
        if np.any(rho2 >= 0.99):
            raise ValueError('Exported effective reflection exceeds the passive domain')
        q0 = reflection*self.b[None, :]
        clean = (1-rho2)/np.abs(1+q0)**2
        dclean_amp = -2*rho2/(1-rho2)-2*(q0/(1+q0)).real
        dclean_phase = 2*(q0/(1+q0)).imag
        logs = np.empty_like(self.target)
        jac = np.empty((*logs.shape, 2*count)) if jacobian else None
        self.local_jac = np.empty((*logs.shape,2*self.local_count)) if jacobian and self.tile_local else None
        for fi, (supported, kernel, phases) in enumerate(self.cells):
            basis = self.basis[:, supported]
            local_rho2 = rho2[:, supported]
            for pi, phase in enumerate(phases):
                q = q0[:, supported]*phase
                power = (1-local_rho2)/np.abs(1+q)**2
                tiles = self.source_tiles[:, fi, pi]
                numerator_kernel = kernel*power[tiles]
                denominator_kernel = kernel*clean[tiles][:, supported]
                numerator = numerator_kernel.sum(axis=1)
                denominator = denominator_kernel.sum(axis=1)
                logs[:, fi, pi] = np.log(numerator)-np.log(denominator)
                if jacobian:
                    d_amp = -2*local_rho2/(1-local_rho2)-2*(q/(1+q)).real
                    d_phase = 2*(q/(1+q)).imag
                    jac[:, fi, pi, :count] = (
                        ((numerator_kernel*d_amp[tiles])@basis.T)/numerator[:, None]
                        - ((denominator_kernel*dclean_amp[tiles][:, supported])@basis.T)/denominator[:, None]
                    )
                    jac[:, fi, pi, count:] = (
                        ((numerator_kernel*d_phase[tiles])@basis.T)/numerator[:, None]
                        - ((denominator_kernel*dclean_phase[tiles][:, supported])@basis.T)/denominator[:, None]
                    )
                    if self.tile_local:
                        local_basis=self.local_basis[:,supported]
                        self.local_jac[:,fi,pi,:self.local_count] = (
                            ((numerator_kernel*d_amp[tiles])@local_basis.T)/numerator[:,None]
                            - ((denominator_kernel*dclean_amp[tiles][:,supported])@local_basis.T)/denominator[:,None]
                        )
                        self.local_jac[:,fi,pi,self.local_count:] = (
                            ((numerator_kernel*d_phase[tiles])@local_basis.T)/numerator[:,None]
                            - ((denominator_kernel*dclean_phase[tiles][:,supported])@local_basis.T)/denominator[:,None]
                        )
        prediction = -(logs-logs[:, :, :1])/c.workflow.LOG_FLUX_PER_MAG
        if jacobian:
            jac = -(jac-jac[:, :, :1, :])/c.workflow.LOG_FLUX_PER_MAG
        return prediction, jac

    def objective(self, theta):
        prediction, jac = self.predict(theta)
        residual = 1000*(prediction-self.target)[self.valid]
        jac = 1000*jac[self.valid]
        # A small representation regularizer prevents large optical excursions.
        # It is not an observational likelihood or a source-model prior update.
        global_size=2*self.count
        regularizer = 0.001*np.sum((theta[:global_size]/0.2)**2)
        if self.tile_local:
            regularizer += 0.001*np.sum((theta[global_size:]/0.1)**2)/256
        value = 0.5*np.mean(residual**2)+regularizer
        gradient = jac.T@residual/len(residual)+0.002*theta[:global_size]/0.2**2
        if self.tile_local:
            residual_cube=np.zeros(self.target.shape)
            residual_cube[self.valid]=residual
            local_gradient=np.zeros((256,2*self.local_count))
            factor=-1000/c.workflow.LOG_FLUX_PER_MAG/len(residual)
            for fi in range(len(c.amber.FILTERS)):
                for pi in range(len(c.amber.PASSES)):
                    weighted=residual_cube[:,fi,pi,None]*factor
                    np.add.at(local_gradient,self.source_tiles[:,fi,pi],weighted*self.local_jac[:,fi,pi])
                    np.add.at(local_gradient,self.source_tiles[:,fi,0],-weighted*self.local_jac[:,fi,0])
            gradient=np.concatenate([gradient,local_gradient.ravel()+0.002*theta[global_size:]/(256*0.1**2)])
        self.evaluations += 1
        if self.evaluations % 20 == 0:
            print(f'{self.detector}: projection {self.evaluations}, model-output RMS {np.sqrt(np.mean(residual**2)):.4f} mmag', flush=True)
        return value, gradient


def blue_basis(wave):
    """Smooth blue-only representation corrections, not source-model parameters."""
    wave=np.asarray(wave,float)
    return np.exp(-0.5*((wave[None,:]-np.array([550.,700.,850.])[:,None])/100.)**2)


def export_projection(model, detector, theta, bounds, output, source, tile_local=False):
    wave = np.asarray(model.wavelength_nm, float)
    # Re-express all the existing curves together so only the familiar
    # mean/modes/amplitudes fields are needed by the unchanged reader.
    tiles = np.arange(256)
    x,y=(tiles%16)*256+128,(tiles//16)*256+128
    curves=[]
    for sca in range(1, 19):
        local=model.reflectance(sca,x,y,wave)
        if sca == int(detector[3:]):
            basis=full_basis(wave,bounds)[0]
            count=len(basis)
            log_change=theta[:count]@basis+1j*(theta[count:2*count]@basis)
            log_change=np.broadcast_to(log_change,local.shape).copy()
            if tile_local:
                lb=blue_basis(wave)
                lc=len(lb)
                coeff=theta[2*count:].reshape(256,2*lc)
                log_change+=coeff[:,:lc]@lb+1j*(coeff[:,lc:]@lb)
            local *= np.exp(log_change)
        curves.append(local)
    curves=np.concatenate(curves)
    mean=curves.mean(axis=0)
    left,singular,right=svd(curves-mean[None,:],full_matrices=False,check_finite=False)
    rank=64
    amp=left[:,:rank]*singular[:rank]
    modes=right[:rank]
    max_error=float(np.max(np.abs(curves-(mean[None,:]+amp@modes))))
    # Only copy the existing, already contract-checked file and replace its
    # existing reflectance fields. No new H5 dataset/group names are introduced.
    with h5py.File(source) as src,h5py.File(output,'w') as dst:
        for key,value in src.attrs.items():
            dst.attrs[key]=value
        for key in src:
            src.copy(key,dst)
        group=dst['reflectance']
        for key,values in [('mean',mean),('modes',modes),('singular_values',singular[:rank])]:
            del group[key]
            group.create_dataset(key,data=values,compression='gzip',shuffle=True)
        label_dtype=group['mode_labels'].dtype
        del group['mode_labels']
        group.create_dataset('mode_labels',data=np.array(
            [f'PC_{i+1:02d}' for i in range(rank)],dtype=label_dtype
        ))
        del group['amplitudes']
        ds=group.create_dataset('amplitudes',shape=(model._sca.size,rank),dtype=np.complex128,
                                compression='gzip',shuffle=True,chunks=(2048,rank))
        for start in range(0,model._sca.size,2048):
            stop=min(start+2048,model._sca.size)
            coarse=(model._sca[start:stop].astype(int)-1)*256+(model._row[start:stop]//16)*16+model._col[start:stop]//16
            ds[start:stop]=amp[coarse]
        dst.attrs['n_modes']=rank
        dst.attrs['release_version']='joint_modulated_model_projection'
        dst.attrs['release_note']='Model-to-model approximation of frozen joint predictions; source model not refitted.'
        if inventory(src)!=inventory(dst):
            raise ValueError('Projection changed the H5 contract')
    return max_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector',required=True,choices=c.DETECTORS)
    parser.add_argument('--maxiter',type=int,default=150)
    parser.add_argument('--tile-local',action='store_true',
                        help='Allow smooth tile-dependent blue response representation corrections')
    parser.add_argument('--reuse-coefficients',action='store_true',
                        help='Re-export and verify previously computed model-projection coefficients')
    parser.add_argument('--output-dir',type=Path,default=c.BUNDLE_DIR/'unchanged_reader_port/model_projection')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    d=args.detector
    # Source predictions are the numerical conversion target. The observed
    # magnitudes returned by cohort reconstruction are deliberately discarded.
    ids,seds,_observed,valid,common,fit_dir,summary,roll,saved=c.check_joint_fit_sample(d,c.DEFAULT_STATISTICS)
    source_paths=[SOURCE,fit_dir/'joint_fit.npz',fit_dir/'joint_weekly_ice_fit.npz']
    before_hashes={str(p):sha256_file(p) for p in source_paths}
    area,_=c.amber.canonical_configuration(d)
    with c.load_eric_v3_reader(c.DEFAULT_SOURCE_ZIP) as reader:
        model=reader.IceResponseModel(SOURCE)
        forward=ResponseProjection(model,reader,d,ids,seds,common,valid,saved-roll,area,args.tile_local)
        theta=np.zeros(forward.parameter_count)
        initial,jac=forward.predict(theta)
        si=int(np.flatnonzero(forward.valid[:,0,1])[0])
        values=[]
        for pi in (0,1):
            tile=forward.source_tiles[si,0,pi]
            supported,kernel,phases=forward.cells[0]
            weight=np.zeros(len(forward.wave));weight[supported]=kernel[si]
            values.append(model.band_response(
                forward.sca,[forward.x[tile]],[forward.y[tile]],
                [forward.hours[0][pi]],forward.wave,weight,
                gray=True,no_ice_weight=True,
            )[0,0])
        expected=-np.log(values[1]/values[0])/c.workflow.LOG_FLUX_PER_MAG
        np.testing.assert_allclose(initial[si,0,1],expected,atol=1e-12,rtol=1e-10)
        direction=np.random.default_rng(12).normal(size=len(theta));direction/=np.linalg.norm(direction)
        eps=1e-6
        numeric=(forward.objective(theta+eps*direction)[0]-forward.objective(theta-eps*direction)[0])/(2*eps)
        analytic=forward.objective(theta)[1]@direction
        np.testing.assert_allclose(numeric,analytic,rtol=2e-5,atol=1e-5)
        if args.reuse_coefficients:
            from scipy.optimize import OptimizeResult
            archive=args.output_dir/f'{d.lower()}_projection_coefficients.npz'
            with np.load(archive) as saved_projection:
                coefficients=saved_projection['coefficients']
                np.testing.assert_array_equal(saved_projection['wavelength_nm'],forward.wave)
            result=OptimizeResult(x=coefficients,success=False,
                                  message='Reused model-projection coefficients; convergence not asserted',nfev=0)
        else:
            variable_bounds=[(-0.12,0.12)]*forward.count+[(-0.4,0.4)]*forward.count
            if args.tile_local:
                variable_bounds += ([(-0.06,0.06)]*forward.local_count+[(-0.2,0.2)]*forward.local_count)*256
            result=minimize(forward.objective,theta,jac=True,method='L-BFGS-B',
                            bounds=variable_bounds,
                            options={'maxiter':args.maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':30})
        projected,_=forward.predict(result.x,False)
        metrics=[]
        for fi,band in enumerate(c.amber.FILTERS):
            mask=forward.valid[:,fi]
            initial_error=1000*(initial-forward.target)[:,fi][mask]
            projected_error=1000*(projected-forward.target)[:,fi][mask]
            metrics.append(dict(filter=band,source_comparisons=int(mask.sum()),
                before_rms_mmag=float(np.sqrt(np.mean(initial_error**2))),
                projected_rms_mmag=float(np.sqrt(np.mean(projected_error**2)))))
        print(pd.DataFrame(metrics).to_string(index=False),flush=True)
        np.savez_compressed(args.output_dir/f'{d.lower()}_projection_coefficients.npz',
                            coefficients=result.x,wavelength_nm=forward.wave)
        output=args.output_dir/f'{d.lower()}_joint_modulated.h5'
        reconstruction_error=export_projection(model,d,result.x,forward.bounds,output,SOURCE,args.tile_local)
        exported=reader.IceResponseModel(output)
        check=ResponseProjection(exported,reader,d,ids,seds,common,valid,saved-roll,area)
        exported_prediction,_=check.predict(np.zeros(2*check.count),False)
        export_error=float(1000*np.sqrt(np.mean((exported_prediction-projected)[forward.valid]**2)))
        if export_error>0.02:
            raise ValueError(f'Export compression changes predictions by {export_error:g} mmag RMS')
    after_hashes={str(p):sha256_file(p) for p in source_paths}
    if before_hashes!=after_hashes:
        raise ValueError('Source model files changed during conversion')
    report=dict(detector=d,conversion='frozen model-output projection, not an observational refit',
                observations_used_in_objective=False,uncertainties_used_in_objective=False,
                source_model_files_unchanged=True,source_hashes=after_hashes,
                ice_histories_changed=False,reader_changed=False,h5_contract_changed=False,
                tile_local_blue_representation=args.tile_local,
                optimizer_success=bool(result.success),optimizer_message=str(result.message),
                evaluations=int(result.nfev),metrics=metrics,
                exported_prediction_representation_error_rms_mmag=export_error,
                reflection_compression_max_complex_error=reconstruction_error,
                output=str(output.resolve()),output_sha256=sha256_file(output))
    (args.output_dir/f'{d.lower()}_projection_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
