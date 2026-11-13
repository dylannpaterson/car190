"""Our modulated TMM response with Eric-reader-compatible band normalization.

The air-side TMM modulation and the full tile histories remain the fit model.
Only the clean reference/band integral convention changes. No AR parameters
are exported by this module.
"""

import numpy as np


ICE_N = 1.25
FRESNEL = (1.0-ICE_N)/(1.0+ICE_N)


class NonPassiveReflectionError(ValueError):
    """Optimizer proposal lies outside the physical reflectance domain."""


def internal_reflection(air_reflection):
    denominator = (ICE_N+1)+(ICE_N-1)*air_reflection
    internal = ((ICE_N-1)+(ICE_N+1)*air_reflection)/denominator
    derivative = 4*ICE_N*air_reflection/denominator**2
    return internal, derivative


def band_terms(air_reflection, wavelengths, thickness):
    """S(d), S(0), optical log-amplitude/phase and thickness derivatives.

    reflection: (tile, wavelength), thickness: (tile, time).
    Matches the unchanged ice_response_core film/no_ice_transmission formula.
    """
    r, dr = internal_reflection(air_reflection)
    rho2 = np.abs(r)**2
    if np.any(rho2 >= 1.-1e-12):
        raise NonPassiveReflectionError('Non-passive internal reflectance in joint forward model')
    wave = np.asarray(wavelengths, float)
    phase = np.exp(4j*np.pi*ICE_N*np.asarray(thickness)[:, :, None]/wave[None, None, :])
    q = FRESNEL*r[:, None, :]*phase
    dq = FRESNEL*dr[:, None, :]*phase
    base_amp = -2*np.real(np.conj(r)*dr)/(1-rho2)
    base_phase = 2*np.imag(np.conj(r)*dr)/(1-rho2)
    power = (1-rho2)[:, None, :]/np.abs(1+q)**2
    amplitude = base_amp[:, None, :]-2*np.real(dq/(1+q))
    angle = base_phase[:, None, :]+2*np.imag(dq/(1+q))
    depth = 8*np.pi*ICE_N/wave[None, None, :]*np.imag(q/(1+q))
    q0 = FRESNEL*r
    dq0 = FRESNEL*dr
    clean = (1-rho2)/np.abs(1+q0)**2
    clean_amp = base_amp-2*np.real(dq0/(1+q0))
    clean_phase = base_phase+2*np.imag(dq0/(1+q0))
    return power, clean, amplitude, angle, depth, clean_amp, clean_phase


class ReaderCompatiblePhotometryForward:
    """Identical interface to CachedPhotometryForward, using the reader integral."""
    def __init__(self, model):
        self.model = model
        self.source = model.clean_reflection_by_tile
        from fit_car201_zane_star_reflectance_modulation import trapezoid_weights
        quadrature = trapezoid_weights(model.wavelengths_nm)
        self.cells = []
        for fi, area in enumerate(model.effective_area):
            support = np.flatnonzero(area > 0)
            kernel = model.star_photon_shapes[:, support]*(area[support]*quadrature[support])[None, :]
            self.cells.append((support, kernel))

    def evaluate(self, parameters):
        model = self.model
        count = model.mode_count
        multiplier = np.exp(parameters[:count]@model.basis + 1j*(parameters[count:2*count]@model.basis))
        reflection = self.source*multiplier[None, :]
        logs = np.empty((model.star_count, model.filter_count, model.pass_count))
        jac = np.empty((*logs.shape, 2*count))
        self.depth_sensitivity = np.empty_like(logs)
        for fi, (support, kernel) in enumerate(self.cells):
            wave = model.wavelengths_nm[support]
            basis = model.basis[:, support]
            terms = band_terms(reflection[:, support], wave, model.tile_thickness_by_id[:, fi, :])
            power, clean, amp, phase, depth, clean_amp, clean_phase = terms
            for pi in range(model.pass_count):
                tiles = model.source_tile_ids[:, fi, pi]
                numerator_kernel = kernel*power[tiles, pi]
                denominator_kernel = kernel*clean[tiles]
                numerator = numerator_kernel.sum(axis=1)
                denominator = denominator_kernel.sum(axis=1)
                logs[:, fi, pi] = np.log(numerator)-np.log(denominator)
                jac[:, fi, pi, :count] = (
                    ((numerator_kernel*amp[tiles, pi])@basis.T)/numerator[:, None]
                    - ((denominator_kernel*clean_amp[tiles])@basis.T)/denominator[:, None]
                )
                jac[:, fi, pi, count:] = (
                    ((numerator_kernel*phase[tiles, pi])@basis.T)/numerator[:, None]
                    - ((denominator_kernel*clean_phase[tiles])@basis.T)/denominator[:, None]
                )
                self.depth_sensitivity[:, fi, pi] = (numerator_kernel*depth[tiles, pi]).sum(axis=1)/numerator
        if hasattr(model, 'reader_gray_log_by_tile'):
            for fi in range(model.filter_count):
                for pi in range(model.pass_count):
                    tiles = model.source_tile_ids[:, fi, pi]
                    logs[:, fi, pi] += model.reader_gray_log_by_tile[tiles, fi, pi]
        return logs-logs[:, :, :1], jac-jac[:, :, :1, :]


def stellar_ice_gradient(objective, coordinates, jacobian, star_indices):
    model = objective.model
    _, _, _, _, scores, _ = objective.state(coordinates)
    full_score = np.zeros((model.star_count*model.filter_count, model.pass_count))
    full_score[objective.active_groups] = scores
    full_score = full_score.reshape(model.star_count, model.filter_count, model.pass_count)
    sensitivity = objective.forward.depth_sensitivity
    from heatmap_car201_allstar_model_correction import LOG_FLUX_PER_MAG
    gradient = np.zeros((model.tile_count, jacobian.shape[-1]))
    for fi in range(model.filter_count):
        ref_tile = model.source_tile_ids[:, fi, 0]
        for pi in range(model.pass_count):
            tile = model.source_tile_ids[:, fi, pi]
            weight = 1000/LOG_FLUX_PER_MAG*full_score[:, fi, pi, None]
            np.add.at(gradient, tile, weight*sensitivity[:, fi, pi, None]*jacobian[tile, star_indices[fi, pi]])
            np.add.at(gradient, ref_tile, -weight*sensitivity[:, fi, 0, None]*jacobian[ref_tile, star_indices[fi, 0]])
    return gradient


class ReaderCompatibleLED:
    """Our source TMM modulation, integrated with the unchanged reader convention."""
    def __init__(self, source_led, gray_log_at_days):
        self.source = source_led
        self.basis = source_led.basis
        self.bands = source_led.bands
        self.gray = gray_log_at_days
        self.cells = []
        for bi, band in enumerate(source_led.bands):
            exemplar = band[0]
            wave = exemplar[11]
            support = source_led.weights[bi] > 0
            weight = source_led.weights[bi, support]
            days = exemplar[9]
            for item in band:
                np.testing.assert_allclose(item[9], days, atol=1e-10, rtol=0)
                np.testing.assert_array_equal(item[11], wave)
            self.cells.append(dict(
                wave=wave, weight=weight, basis=exemplar[5], days=days,
                reflection=np.stack([item[2] for item in band]),
                observed=np.stack([item[7] for item in band]),
                sigma=np.stack([item[8] for item in band]),
            ))

    def loss(self, theta, thickness, jacobian, time_indices):
        count = self.basis.shape[0]
        optical_gradient = np.zeros(2*count)
        ice_gradient = np.zeros((256, jacobian.shape[-1]))
        cost = 0.
        for bi, cell in enumerate(self.cells):
            basis = cell['basis']
            multiplier = np.exp(theta[:count]@basis+1j*(theta[count:]@basis))
            reflection = cell['reflection']*multiplier[None, :]
            indices = np.asarray(time_indices[bi][0])
            for local in time_indices[bi]:
                np.testing.assert_array_equal(indices, local)
            terms = band_terms(reflection, cell['wave'], thickness[:, indices])
            power, clean, amp, phase, depth, clean_amp, clean_phase = terms
            weighted = power*cell['weight'][None, None, :]
            rate = weighted.sum(axis=2)
            # At the same tile, the clean-band denominator cancels between visits.
            logs = np.log(rate)+self.gray(cell['days'])
            predicted = logs-logs[:, :1]
            residual = (predicted-cell['observed'])/cell['sigma']
            cost += 0.5*np.sum(residual**2)
            optical = np.concatenate([
                np.einsum('itw,mw->itm', weighted*amp, basis)/rate[:, :, None],
                np.einsum('itw,mw->itm', weighted*phase, basis)/rate[:, :, None],
            ], axis=2)
            optical_gradient += np.einsum(
                'itm,it->m', optical-optical[:, :1], residual/cell['sigma']
            )
            dl_dd = (weighted*depth).sum(axis=2)/rate
            derivative = dl_dd[:, :, None]*jacobian[:, indices, :]
            ice_gradient += np.einsum(
                'itp,it->ip', derivative-derivative[:, :1], residual/cell['sigma']
            )
        return float(cost), optical_gradient, ice_gradient
