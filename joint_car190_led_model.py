"""CAR190 six-LED likelihood and Jacobian for joint complex-response fits."""

from pathlib import Path

import numpy as np
import pandas as pd

import fit_wfi_tmm_production as production
import compare_wfi_tile136_complex_reflectance as complex_model
import mock_fit_wfi_star_reflectance_modulation as mock


class JointCAR190LED:
    """Original CAR190 group likelihood with an Amber-compatible modulation basis."""
    def __init__(self,detector,basis,time_grid,bundle_root):
        self.detector=detector
        self.basis=np.asarray(basis,float)
        self.mode_count=len(self.basis)
        self.times=np.asarray(time_grid,float)
        self.bands=[[] for _ in range(6)]
        context=production.load_response_context([detector],sed_weighting="energy",qe_response_mode="none")
        self.wave=np.asarray(context["wavelengths"],float)
        self.led_weights=np.asarray(context["model_weights"],float)[0]
        n,k=production.hgcdte_reference_optical_constants(self.wave)
        spectral=production.spectral_modulation_basis(self.wave)
        baseline_rates=[]
        for tile in range(256):
            fit_directory=production.ROOT/"20261106/corrected_statistics_4h_thinned/production"/detector.lower()/"spline_relative_response_normal/fits"/f"tile_{tile:03d}"
            predictions=pd.read_csv(fit_directory/"tile_group_predictions.csv")
            statistics=pd.read_csv(production.ROOT/"20261106/corrected_statistics_4h_thinned/production"/detector.lower()/"spline_relative_response_normal/tile_statistics"/f"wfi{detector[3:]}_tile_log_statistics_256_tile_{tile:03d}.csv")
            if not np.array_equal(predictions["group"],statistics["group"]):raise ValueError("CAR190 group alignment mismatch")
            with np.load(fit_directory/"posterior_samples.npz",allow_pickle=False) as archive:
                days=np.asarray(archive["observation_days"],float).reshape(-1)
                thickness=np.asarray(archive["tile_ice_thickness"],float).reshape(-1,len(days))[0]
                optical_cache=Path(str(np.asarray(archive["response_cache"]).reshape(-1)[0]))
            np.testing.assert_allclose(days,predictions["days"].to_numpy(float),atol=1e-10,rtol=0)
            fit_path=mock.source_fit_from_cache(optical_cache,detector,tile)
            state=complex_model.load_state(fit_path,self.wave,n,k,spectral)
            first,second,y0,ysub,sine=mock.clean_stack_boundary(state,self.wave)
            if abs(sine)>1e-12:raise ValueError("Joint LED model expects fixed normal incidence")
            clean_y=second/first
            reflection=(y0*first-second)/(y0*first+second)
            weight_sigma=.005*np.sqrt(np.median(statistics["n_pair_valid"])/statistics["n_pair_valid"].to_numpy(float))
            for band in range(1,7):
                select=predictions["led"].to_numpy(int)==band
                support=self.led_weights[band-1]>0
                local_wave=self.wave[support]
                ice=state["ice_index"]
                factor=(np.real(ysub[support]/y0)*abs(2*y0)**2*self.led_weights[band-1,support])
                indices=np.searchsorted(self.times,days[select])
                if not np.allclose(self.times[indices],days[select],atol=1e-10,rtol=0):
                    raise ValueError("Joint time grid does not include CAR190 data times")
                item={"tile":tile,"first":first[support],"clean_y":clean_y[support],
                      "reflection":reflection[support],"y0":float(y0),"factor":factor,
                      "basis":self.basis[:,support],"ice":float(ice),"wave":local_wave,
                      "thickness":thickness[select],"time_index":indices,
                      "saved":predictions.loc[select,"model_log_change"].to_numpy(float),
                      "observed":predictions.loc[select,"observed_log_change"].to_numpy(float),
                      "sigma":weight_sigma[select]}
                item["baseline_log_rate"]=self._response(np.zeros(2*self.mode_count),item,item["thickness"])[0]
                self.bands[band-1].append(item)
            if (tile+1)%64==0:print(f"Prepared joint CAR190 tile {tile+1}/256",flush=True)

    def _response(self,theta,item,thickness):
        wave=item["wave"];ice=item["ice"];y0=item["y0"];first=item["first"]
        clean_y=item["clean_y"];basis=item["basis"]
        modulation=np.exp(theta[:self.mode_count]@basis+1j*(theta[self.mode_count:]@basis))
        reflected=item["reflection"]*modulation
        y=y0*(1-reflected)/(1+reflected)
        dy_da=-2*y0*reflected/(1+reflected)**2
        dy_dp=1j*dy_da
        phase=2*np.pi*ice*np.asarray(thickness)[:,None]/wave[None,:]
        c=np.cos(phase);s=-1j*np.sin(phase)
        a=first[None,:]*(y0*c+s*ice)
        b=first[None,:]*(y0*s/ice+c)
        denominator=a+b*y[None,:]
        weight=item["factor"][None,:]/np.abs(denominator)**2
        rate=weight.sum(axis=1)
        log_rate=np.log(rate)
        da=np.concatenate([b[:,:,None]*dy_da[None,:,None]*basis.T[None,:,:],
                            b[:,:,None]*dy_dp[None,:,None]*basis.T[None,:,:]],axis=2)
        dlog_theta=-2*np.sum(weight[:,:,None]*np.real(da/denominator[:,:,None]),axis=1)/rate[:,None]
        k=2*np.pi*ice/wave
        d_a=first[None,:]*(-y0*k[None,:]*np.sin(phase)-1j*k[None,:]*c*ice)
        d_b=first[None,:]*(-1j*k[None,:]*c*y0/ice-k[None,:]*np.sin(phase))
        d_den=d_a+d_b*y[None,:]
        dlog_thickness=-2*np.sum(weight*np.real(d_den/denominator),axis=1)/rate
        return log_rate,dlog_theta,dlog_thickness

    def evaluate(self,theta,thickness_by_tile,with_gradient=True):
        residuals=[]
        grad_theta=np.zeros(2*self.mode_count)
        grad_thickness=np.zeros((256,len(self.times))) if with_gradient else None
        for band in self.bands:
            for item in band:
                tile=item["tile"];idx=item["time_index"]
                thickness=thickness_by_tile[tile,idx]
                log_rate,jtheta,jdepth=self._response(theta,item,thickness)
                ref=0
                correction=(log_rate-log_rate[ref])-(item["baseline_log_rate"]-item["baseline_log_rate"][ref])
                predicted=item["saved"]+correction
                residual=(predicted-item["observed"])/item["sigma"]
                residuals.append(residual)
                if with_gradient:
                    grad_theta+=((jtheta-jtheta[ref])/item["sigma"][:,None]).T@residual
                    for row_index,time_index in enumerate(idx):
                        grad_thickness[tile,time_index]+=residual[row_index]/item["sigma"][row_index]*jdepth[row_index]
                        grad_thickness[tile,idx[ref]]-=residual[row_index]/item["sigma"][row_index]*jdepth[ref]
        return np.concatenate(residuals),grad_theta,grad_thickness
