"""Cached exact response and analytic shared-modulation derivatives.

Implements RealPhotometryFit's existing normal-incidence ice propagation, with
the identical clean-power normalization and first-visit subtraction. No optical
model assumptions are changed.
"""

import numpy as np

from fit_car201_zane_star_reflectance_modulation import trapezoid_weights


class CachedPhotometryForward:
    def __init__(self, model):
        self.model = model
        self.admittance = model.incident_admittance_by_tile[:, None]
        self.first = model.clean_transfer_first_by_tile
        self.power_numerator = (
            np.real(model.substrate_admittance_by_tile / self.admittance)
            * np.abs(2 * self.admittance)**2
            / model.clean_substrate_power_by_tile
        )
        self.coefficients = []
        for fi in range(model.filter_count):
            local = []
            for pi in range(model.pass_count):
                ice = model.ice_index_by_tile[:, None]
                phase = 2 * np.pi * ice * model.tile_thickness_by_id[:, fi, pi, None] / model.wavelengths_nm[None, :]
                cosine = np.cos(phase)
                sine = -1j * np.sin(phase)
                a = self.first * (self.admittance * cosine + sine * ice)
                b = self.first * (self.admittance * sine / ice + cosine)
                local.append((a, b))
            self.coefficients.append(local)
        weights = trapezoid_weights(model.wavelengths_nm)
        self.kernels = [model.star_photon_shapes * (area*weights)[None, :]
                        for area in model.effective_area]

    def evaluate(self, parameters):
        model = self.model
        count = model.mode_count
        ratio = np.exp(parameters[:count] @ model.basis
                       + 1j * (parameters[count:2*count] @ model.basis))
        reflection = model.clean_reflection_by_tile * ratio[None, :]
        clean_y = self.admittance * (1-reflection)/(1+reflection)
        derivative_y = -2*self.admittance*reflection/(1+reflection)**2
        log_rate = np.empty((model.star_count, model.filter_count, model.pass_count))
        jacobian = np.empty(log_rate.shape + (2*count,))
        for fi in range(model.filter_count):
            for pi in range(model.pass_count):
                a, b = self.coefficients[fi][pi]
                denominator = a+b*clean_y
                power = self.power_numerator / np.abs(denominator)**2
                relative_derivative = b*derivative_y/denominator
                tiles = model.source_tile_ids[:, fi, pi]
                kernel = self.kernels[fi] * power[tiles]
                rate = kernel.sum(axis=1)
                if np.any(rate <= 0) or np.any(~np.isfinite(rate)):
                    raise ValueError("Invalid integrated photometry rate")
                log_rate[:, fi, pi] = np.log(rate)
                amplitude = -2*relative_derivative.real[tiles]
                phase = 2*relative_derivative.imag[tiles]
                jacobian[:, fi, pi, :count] = ((kernel*amplitude) @ model.basis.T) / rate[:, None]
                jacobian[:, fi, pi, count:] = ((kernel*phase) @ model.basis.T) / rate[:, None]
        return log_rate-log_rate[:, :, :1], jacobian-jacobian[:, :, :1, :]
