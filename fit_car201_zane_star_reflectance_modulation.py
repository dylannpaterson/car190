"""Fit clean-reflectance modulation to CAR201 WFI stellar photometry.

The selected ep3/ep4 passes are matched by Gaia DR3 source ID and joined to
successful BT-Settl SED posteriors in CAR201/sed. Ice history is fixed. Since
the parquet catalogs have no detector x/y or tile column, this first SCA-level
fit averages a quantile sample of fixed detector tile histories and uses the pinned
tile-136 clean TMM optical state as representative input. Measurements marked
SATURATED or DO_NOT_USE in the Roman DQ bitmask are excluded by default.
Clean-reflectance log-amplitude modulation uses the synthetic fit's relative-
entropy MaxEnt penalty; phase uses its separate second-difference penalty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import extinction
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import least_squares, minimize_scalar

import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production
from analyze_car201_zane_stellar_photometry import (
    DEFAULT_PHOTOMETRY_DIR,
    DEFAULT_SED_POSTERIORS,
    ICE_REFERENCE_UTC,
    PASS_NAMES,
    WINDOW_END_UTC,
    WINDOW_START_UTC,
    load_pass,
    load_sed_posteriors,
)
from mock_fit_wfi_star_reflectance_modulation import (
    DEFAULT_EFFECTIVE_AREA,
    DEFAULT_ICE_HISTORY,
    DEFAULT_OPTICAL_CACHE,
    FILTERS,
    LOG_FLUX_PER_MAG,
    clean_stack_boundary,
    load_effective_area,
    modulation_basis,
    source_fit_from_cache,
)
import mock_fit_wfi_star_reflectance_modulation as mock_model


ROOT = Path(__file__).resolve().parent
CAR201_ROOT = ROOT.parent / "CAR201"
DEFAULT_PRODUCTION_ROOT = (
    ROOT / "20261105/corrected_statistics_4h_thinned/production"
)
DEFAULT_RESPONSE_CACHE_DIR = DEFAULT_OPTICAL_CACHE.parents[2]
DETECTOR_PIXELS = 4096
TILE_SIZE_PIXELS = 256
TILE_GRID_SIZE = DETECTOR_PIXELS // TILE_SIZE_PIXELS
DQ_DO_NOT_USE = np.uint32(1 << 0)
DQ_SATURATED = np.uint32(1 << 1)
PASS_SET_CONFIG = {
    "ep3": {
        "passes": PASS_NAMES,
        "start": WINDOW_START_UTC,
        "end": WINDOW_END_UTC,
        "label": "all_sed_sep28_oct14",
        "zero_point_group_ids": (0, 0, 0, 0, 0),
        "zero_point_group_names": ("ep3_reference",),
    },
    "ep4": {
        "passes": (
            "ep4_pass1",
            "ep4_pass2",
            "ep4_pass3",
            "ep4_pass4",
            "ep4_pass5",
        ),
        "start": "2026-10-18T00:00:00",
        "end": "2026-11-01T00:00:00",
        "label": "ep4_oct18_oct31",
        "zero_point_group_ids": (0, 0, 0, 0, 0),
        "zero_point_group_names": ("ep4_reference",),
    },
    "ep3_ep4": {
        "passes": PASS_NAMES
        + (
            "ep4_pass1",
            "ep4_pass2",
            "ep4_pass3",
            "ep4_pass4",
            "ep4_pass5",
        ),
        "start": WINDOW_START_UTC,
        "end": "2026-11-01T00:00:00",
        "label": "ep3_ep4_roll_after_ep4pass1",
        "zero_point_group_ids": (0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
        "zero_point_group_names": ("pre_roll_reference", "post_roll"),
    },
    "ep3v2_ep4": {
        "passes": (
            "ep3_pass1",
            "ep3_pass2",
            "ep3_pass3",
            "ep3_pass4",
            "ep3_pass5",
            "ep3_pass6",
            "ep4_pass1",
            "ep4_pass2",
            "ep4_pass3",
            "ep4_pass4",
            "ep4_pass5",
        ),
        "start": WINDOW_START_UTC,
        "end": "2026-11-01T00:00:00",
        "label": "ep3v2_ep4_20261106_roll_after_ep4pass1",
        "zero_point_group_ids": (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
        "zero_point_group_names": ("pre_roll_reference", "post_roll"),
    },
}


def detector_tile_fit_root(detector):
    return (
        DEFAULT_PRODUCTION_ROOT
        / detector.lower()
        / "spline_relative_response_normal/fits"
    )


def detector_optical_cache(detector, tile_id=136):
    return (
        DEFAULT_RESPONSE_CACHE_DIR
        / f"{detector.lower()}_400nm/tile_{int(tile_id):03d}/relative_response.npz"
    )


def detector_default_output_dir(
    detector, pass_set="ep3", *, allow_partial_source_coverage=False
):
    pass_label = PASS_SET_CONFIG[pass_set]["label"]
    coverage_suffix = "_partial_source_coverage" if allow_partial_source_coverage else ""
    return (
        DEFAULT_PRODUCTION_ROOT
        / detector.lower()
        / "spline_relative_response_normal/experiments"
        / (
            f"car201_zane_{detector.lower()}_{pass_label}{coverage_suffix}_"
            "modulation_exclude_any_saturated_source_maxent_1000_phase_100"
        )
    )


def trapezoid_weights(x):
    x = np.asarray(x, dtype=float)
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def load_photometry_and_seds(
    photometry_dir,
    sed_posterior_path,
    detector="WFI16",
    *,
    pass_names=PASS_NAMES,
    window_start_utc=WINDOW_START_UTC,
    window_end_utc=WINDOW_END_UTC,
    require_common_source=True,
    exclude_dq_flagged=True,
    exclude_any_saturated_source=True,
):
    start_mjd = float(Time(window_start_utc, format="isot", scale="utc").mjd)
    end_mjd = float(Time(window_end_utc, format="isot", scale="utc").mjd)
    pass_frames = {
        pass_name: load_pass(
            photometry_dir / f"{pass_name}_master.parquet",
            start_mjd,
            end_mjd,
            detector,
            include_invalid=True,
        )
        for pass_name in pass_names
    }
    for pass_name, frame in pass_frames.items():
        missing_coordinates = {"x_pixel", "y_pixel"} - set(frame.columns)
        if missing_coordinates:
            raise ValueError(
                f"{pass_name} catalog lacks detector coordinates: "
                f"{sorted(missing_coordinates)}"
            )
    source_sets = [set(frame["source_id"].unique()) for frame in pass_frames.values()]
    source_ids_in_pass_set = (
        set.intersection(*source_sets)
        if require_common_source
        else set.union(*source_sets)
    )
    sed_by_id = load_sed_posteriors(sed_posterior_path)
    source_ids = sorted(
        source_id
        for source_id in source_ids_in_pass_set
        if source_id in sed_by_id and sed_by_id[source_id]["state"] == 1
    )
    if not source_ids:
        raise ValueError(
            f"No {detector} sources in pass set have successful SED posteriors"
        )

    star_count = len(source_ids)
    pass_count = len(pass_names)
    filter_count = len(FILTERS)
    magnitudes = np.full((star_count, filter_count, pass_count), np.nan)
    magnitude_errors = np.full_like(magnitudes, np.nan)
    valid_measurements = np.zeros((star_count, filter_count, pass_count), dtype=bool)
    dq_flags = np.full((star_count, filter_count, pass_count), -1, dtype=np.int64)
    x_pixels = np.full((star_count, filter_count, pass_count), np.nan)
    y_pixels = np.full_like(x_pixels, np.nan)
    tile_ids_by_measurement = np.full(
        (star_count, filter_count, pass_count), -1, dtype=np.int16
    )
    mjd_by_filter = np.full((filter_count, pass_count), np.nan)

    for pass_index, pass_name in enumerate(pass_names):
        frame = pass_frames[pass_name]
        selected = frame.loc[frame["source_id"].isin(source_ids)]
        for filter_index, filter_name in enumerate(FILTERS):
            band = selected.loc[selected["filter"] == filter_name].set_index("source_id")
            band = band.reindex(source_ids)
            x_pixels[:, filter_index, pass_index] = pd.to_numeric(
                band["x_pixel"], errors="coerce"
            ).to_numpy(float)
            y_pixels[:, filter_index, pass_index] = pd.to_numeric(
                band["y_pixel"], errors="coerce"
            ).to_numpy(float)
            valid_coordinates = (
                np.isfinite(x_pixels[:, filter_index, pass_index])
                & np.isfinite(y_pixels[:, filter_index, pass_index])
            )
            out_of_bounds = valid_coordinates & (
                (x_pixels[:, filter_index, pass_index] < 0.0)
                | (x_pixels[:, filter_index, pass_index] >= DETECTOR_PIXELS)
                | (y_pixels[:, filter_index, pass_index] < 0.0)
                | (y_pixels[:, filter_index, pass_index] >= DETECTOR_PIXELS)
            )
            if np.any(out_of_bounds):
                raise ValueError(
                    f"{pass_name}/{filter_name} has WFI16 coordinates outside "
                    f"[0,{DETECTOR_PIXELS})"
                )
            coordinate_indices = np.flatnonzero(valid_coordinates)
            tile_rows = np.floor(
                y_pixels[coordinate_indices, filter_index, pass_index]
                / TILE_SIZE_PIXELS
            ).astype(np.int16)
            tile_cols = np.floor(
                x_pixels[coordinate_indices, filter_index, pass_index]
                / TILE_SIZE_PIXELS
            ).astype(np.int16)
            tile_ids_by_measurement[
                coordinate_indices, filter_index, pass_index
            ] = tile_rows * TILE_GRID_SIZE + tile_cols
            magnitudes[:, filter_index, pass_index] = band["mag_obs"].to_numpy(float)
            magnitude_errors[:, filter_index, pass_index] = band["mag_err"].to_numpy(float)
            dq_flags[:, filter_index, pass_index] = (
                pd.to_numeric(band["dq_flag"], errors="coerce")
                .fillna(-1)
                .to_numpy(np.int64)
            )
            mjd_by_filter[filter_index, pass_index] = float(
                frame.loc[frame["filter"] == filter_name, "mjd"].median()
            )
            valid_measurements[:, filter_index, pass_index] = (
                np.isfinite(magnitudes[:, filter_index, pass_index])
                & np.isfinite(magnitude_errors[:, filter_index, pass_index])
                & (magnitude_errors[:, filter_index, pass_index] > 0.0)
            )

    raw_measurements = valid_measurements.copy()
    dq_unsigned = dq_flags.astype(np.uint32)
    dq_recorded = dq_flags >= 0
    saturated = dq_recorded & ((dq_unsigned & DQ_SATURATED) != 0)
    do_not_use = dq_recorded & ((dq_unsigned & DQ_DO_NOT_USE) != 0)
    source_has_saturation = np.any(saturated, axis=(1, 2))
    if exclude_dq_flagged:
        valid_measurements &= ~saturated & ~do_not_use
    if exclude_any_saturated_source:
        valid_measurements[source_has_saturation, :, :] = False
    usable_groups = np.sum(valid_measurements, axis=2) >= 2
    if not np.any(usable_groups):
        raise ValueError("No SED-matched source/filter has repeated valid photometry")
    return (
        source_ids,
        sed_by_id,
        magnitudes,
        magnitude_errors,
        raw_measurements,
        valid_measurements,
        saturated,
        do_not_use,
        source_has_saturation,
        dq_flags,
        mjd_by_filter,
        x_pixels,
        y_pixels,
        tile_ids_by_measurement,
    )


def reconstruct_sed_photon_shapes(source_ids, sed_by_id, sed_grid_path, target_wavelengths_nm):
    """Interpolate BT-Settl grid spectra at posterior-median stellar parameters."""
    target_wave_um = np.asarray(target_wavelengths_nm, dtype=float) / 1000.0
    with h5py.File(sed_grid_path, "r") as grid:
        grid_wave_um = np.asarray(grid["wavelength_um"], dtype=float)
        grid_parameters = np.asarray(grid["grid_parameters"], dtype=float)
        surface_grid = np.asarray(grid["surface_flux"], dtype=np.float32)

    # Resample the atmosphere grid to the Roman effective-area grid first so
    # interpolation over stellar parameters stays compact.
    surface_on_target = np.empty(
        (surface_grid.shape[0], target_wave_um.size), dtype=np.float32
    )
    for model_index, spectrum in enumerate(surface_grid):
        surface_on_target[model_index] = np.interp(
            target_wave_um, grid_wave_um, spectrum
        )

    coordinate_scale = np.asarray([0.5, 100.0, 0.3])
    grid_coordinates = grid_parameters / coordinate_scale[None, :]
    interpolator = LinearNDInterpolator(
        grid_coordinates, surface_on_target, fill_value=np.nan
    )
    sed_parameters = np.asarray(
        [sed_by_id[source_id]["parameters_median"] for source_id in source_ids],
        dtype=float,
    )
    queries = sed_parameters[:, [1, 0, 2]] / coordinate_scale[None, :]
    surface_at_star = np.asarray(interpolator(queries), dtype=float)
    outside_hull = np.any(~np.isfinite(surface_at_star), axis=1)
    nearest_fallback_count = int(np.count_nonzero(outside_hull))
    if nearest_fallback_count:
        for star_index in np.flatnonzero(outside_hull):
            distance = np.sum((grid_coordinates - queries[star_index]) ** 2, axis=1)
            nearest = int(np.argmin(distance))
            surface_at_star[star_index] = surface_on_target[nearest]

    extinction_mag = np.asarray(
        [
            extinction.fitzpatrick99(
                np.asarray(target_wavelengths_nm) * 10.0,
                max(float(av), 0.0),
                r_v=3.1,
                unit="aa",
            )
            for av in sed_parameters[:, 4]
        ],
        dtype=float,
    )
    observed_flux = (
        surface_at_star
        * sed_parameters[:, 3, None]
        * 10.0 ** (-0.4 * extinction_mag)
    )
    # F_lambda * lambda is proportional to photon flux density; constants cancel
    # in the per-star, per-filter time ratios used by the likelihood.
    photon_shape = observed_flux * target_wave_um[None, :]
    if np.any(~np.isfinite(photon_shape)) or np.any(photon_shape <= 0.0):
        raise ValueError("Reconstructed stellar photon spectra are invalid")
    return photon_shape, sed_parameters, nearest_fallback_count


def load_fixed_sca_tile_thickness(fit_root, target_days):
    archive_paths = sorted(Path(fit_root).glob("tile_*/posterior_samples.npz"))
    if not archive_paths:
        raise FileNotFoundError(f"No per-tile fixed-history fits under {fit_root}")
    tile_curves = []
    reset_days = None
    for path in archive_paths:
        with np.load(path, allow_pickle=False) as archive:
            days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
            thickness = np.asarray(archive["tile_ice_thickness"], dtype=float)
            thickness = thickness.reshape((-1, days.size))[0]
            local_resets = np.asarray(
                archive.get("ice_reset_days", archive.get("ice_reset_day", [])),
                dtype=float,
            ).reshape(-1)
        if reset_days is None:
            reset_days = np.sort(local_resets)
        elif not np.allclose(reset_days, np.sort(local_resets), rtol=0.0, atol=1.0e-8):
            raise ValueError("WFI16 tile histories disagree on reset boundaries")
        tile_curves.append(
            mock_model.interpolate_fixed_history(
                days, thickness, reset_days, np.asarray(target_days, dtype=float)
            )
        )
    return np.asarray(tile_curves, dtype=float), reset_days, len(archive_paths)


class RealPhotometryFit:
    def __init__(
        self,
        wavelengths_nm,
        effective_area,
        star_photon_shapes,
        tile_thickness_by_id,
        source_tile_ids,
        clean_reflection_by_tile,
        clean_transfer_first_by_tile,
        clean_substrate_power_by_tile,
        incident_admittance_by_tile,
        substrate_admittance_by_tile,
        incidence_sine_by_tile,
        ice_index_by_tile,
        basis_grouping,
        entropy_weight=0.0,
        phase_smoothness_weight=0.0,
        pass_zero_point_group_ids=None,
    ):
        self.wavelengths_nm = wavelengths_nm
        self.effective_area = effective_area
        self.star_photon_shapes = star_photon_shapes
        self.tile_thickness_by_id = np.asarray(tile_thickness_by_id, dtype=float)
        self.source_tile_ids = np.asarray(source_tile_ids, dtype=int)
        self.clean_reflection_by_tile = np.asarray(
            clean_reflection_by_tile, dtype=complex
        )
        self.clean_transfer_first_by_tile = np.asarray(
            clean_transfer_first_by_tile, dtype=complex
        )
        self.clean_substrate_power_by_tile = np.asarray(
            clean_substrate_power_by_tile, dtype=float
        )
        self.incident_admittance_by_tile = np.asarray(
            incident_admittance_by_tile, dtype=float
        ).reshape(-1)
        self.substrate_admittance_by_tile = np.asarray(
            substrate_admittance_by_tile, dtype=complex
        )
        self.incidence_sine_by_tile = np.asarray(
            incidence_sine_by_tile, dtype=float
        ).reshape(-1)
        self.ice_index_by_tile = np.asarray(ice_index_by_tile, dtype=float).reshape(-1)
        self.basis_grouping = basis_grouping
        self.entropy_weight = float(entropy_weight)
        self.phase_smoothness_weight = float(phase_smoothness_weight)
        (
            self.basis,
            self.basis_names,
            self.basis_centers_nm,
            self.node_groups,
            self.node_weights,
            self.group_bounds,
        ) = mock_model.modulation_basis(wavelengths_nm, grouping=basis_grouping)
        self.mode_count = self.basis.shape[0]
        self.star_count = star_photon_shapes.shape[0]
        self.filter_count = len(FILTERS)
        self.tile_count = self.clean_reflection_by_tile.shape[0]
        self.pass_count = self.tile_thickness_by_id.shape[2]
        expected_tile_wave_shape = (self.tile_count, self.wavelengths_nm.size)
        if self.clean_reflection_by_tile.shape != expected_tile_wave_shape:
            raise ValueError("Per-tile clean reflection has an invalid shape")
        for name, values in (
            ("clean transfer", self.clean_transfer_first_by_tile),
            ("clean substrate power", self.clean_substrate_power_by_tile),
            ("substrate admittance", self.substrate_admittance_by_tile),
        ):
            if values.shape != expected_tile_wave_shape:
                raise ValueError(f"Per-tile {name} has an invalid shape")
        if self.tile_thickness_by_id.shape != (
            self.tile_count,
            self.filter_count,
            self.pass_count,
        ):
            raise ValueError("Per-tile ice history has an invalid shape")
        if self.source_tile_ids.shape != (
            self.star_count,
            self.filter_count,
            self.pass_count,
        ):
            raise ValueError("Source tile IDs must have shape (source, filter, pass)")
        if np.any(self.source_tile_ids < 0) or np.any(
            self.source_tile_ids >= self.tile_count
        ):
            raise ValueError("Source tile IDs fall outside available tile states")
        for name, values in (
            ("incident admittance", self.incident_admittance_by_tile),
            ("incidence sine", self.incidence_sine_by_tile),
            ("ice index", self.ice_index_by_tile),
        ):
            if values.size != self.tile_count:
                raise ValueError(f"Per-tile {name} has an invalid shape")
        if pass_zero_point_group_ids is None:
            pass_zero_point_group_ids = np.zeros(self.pass_count, dtype=int)
        self.pass_zero_point_group_ids = np.asarray(
            pass_zero_point_group_ids, dtype=int
        ).reshape(-1)
        if self.pass_zero_point_group_ids.size != self.pass_count:
            raise ValueError("Pass zeropoint group IDs must match pass count")
        if self.pass_zero_point_group_ids[0] != 0:
            raise ValueError("The first pass must define the zeropoint reference group")
        unique_groups = np.unique(self.pass_zero_point_group_ids)
        if not np.array_equal(unique_groups, np.arange(unique_groups.size)):
            raise ValueError("Pass zeropoint group IDs must be contiguous from zero")
        self.zero_point_group_count = int(unique_groups.size - 1)
        self.zero_point_count = self.filter_count * self.zero_point_group_count

    @property
    def parameter_count(self):
        return 2 * self.mode_count + self.zero_point_count

    def maxent_regularization_residuals(self, modulation_parameters):
        """Wavelength-weighted relative-entropy penalty against unit amplitude."""
        if self.entropy_weight <= 0.0:
            return np.empty(0, dtype=float)
        amplitude = modulation_parameters[: self.mode_count]
        signed_entropy_residual = (
            mock_model.StarCountModel._signed_relative_entropy_residual(amplitude)
        )
        return np.sqrt(self.entropy_weight * self.node_weights) * signed_entropy_residual

    def phase_smoothness_residuals(self, modulation_parameters):
        if self.phase_smoothness_weight <= 0.0:
            return np.empty(0, dtype=float)
        phase = modulation_parameters[self.mode_count : 2 * self.mode_count]
        residuals = []
        for indices in self.node_groups:
            sequence = np.concatenate(([0.0], phase[indices], [0.0]))
            if sequence.size >= 3:
                residuals.extend(
                    np.sqrt(self.phase_smoothness_weight) * np.diff(sequence, n=2)
                )
        return np.asarray(residuals, dtype=float)

    def regularization_residuals(self, modulation_parameters):
        return np.concatenate(
            (
                self.maxent_regularization_residuals(modulation_parameters),
                self.phase_smoothness_residuals(modulation_parameters),
            )
        )

    def modeled_log_response(self, modulation_parameters):
        modulation_parameters = np.asarray(modulation_parameters, dtype=float)
        if modulation_parameters.ndim == 1:
            if modulation_parameters.size != 2 * self.mode_count:
                raise ValueError("Global modulation vector has an invalid length")
            amplitude = modulation_parameters[: self.mode_count]
            phase = modulation_parameters[self.mode_count :]
            modulation_by_tile = np.broadcast_to(
                np.exp(amplitude @ self.basis + 1j * (phase @ self.basis)),
                (self.tile_count, self.wavelengths_nm.size),
            )
        elif modulation_parameters.shape == (self.tile_count, 2 * self.mode_count):
            amplitude = modulation_parameters[:, : self.mode_count]
            phase = modulation_parameters[:, self.mode_count :]
            modulation_by_tile = np.exp(
                amplitude @ self.basis + 1j * (phase @ self.basis)
            )
        else:
            raise ValueError(
                "Modulation parameters must be one shared vector or one vector per tile"
            )
        clean_modulated = self.clean_reflection_by_tile * modulation_by_tile
        incident_admittance = self.incident_admittance_by_tile[:, None]
        clean_admittance = incident_admittance * (1.0 - clean_modulated) / (
            1.0 + clean_modulated
        )
        boundary_second = self.clean_transfer_first_by_tile * clean_admittance
        integration_weights = trapezoid_weights(self.wavelengths_nm)
        log_response = np.empty(
            (self.star_count, self.filter_count, self.pass_count), dtype=float
        )
        for filter_index in range(self.filter_count):
            base_kernel = self.effective_area[filter_index] * integration_weights
            base_rate = self.star_photon_shapes @ base_kernel
            for pass_index in range(self.pass_count):
                thickness = self.tile_thickness_by_id[:, filter_index, pass_index]
                ice_index = self.ice_index_by_tile[:, None]
                phase_ice = (
                    2.0
                    * np.pi
                    * ice_index
                    * thickness[:, None]
                    / self.wavelengths_nm[None, :]
                )
                cosine = np.cos(phase_ice)
                sine = -1j * np.sin(phase_ice)
                first = (
                    cosine * self.clean_transfer_first_by_tile
                    + sine * boundary_second / ice_index
                )
                second = (
                    sine * ice_index * self.clean_transfer_first_by_tile
                    + cosine * boundary_second
                )
                denominator = incident_admittance * first + second
                substrate_power = (
                    np.real(
                        self.substrate_admittance_by_tile / incident_admittance
                    )
                    * np.abs(2.0 * incident_admittance / denominator)
                    ** 2
                )
                relative_power_by_tile = (
                    substrate_power / self.clean_substrate_power_by_tile
                )
                relative_power_by_source = relative_power_by_tile[
                    self.source_tile_ids[:, filter_index, pass_index]
                ]
                rate = np.sum(
                    self.star_photon_shapes
                    * base_kernel[None, :]
                    * relative_power_by_source,
                    axis=1,
                )
                log_response[:, filter_index, pass_index] = np.log(rate / base_rate)
        return log_response - log_response[:, :, :1]


def make_real_fit_residual(
    model, observed_log_flux, sigma_log_flux, valid_measurements, parameters
):
    modulation = parameters[: 2 * model.mode_count]
    zpt = parameters[2 * model.mode_count :].reshape(
        (model.filter_count, model.zero_point_group_count)
    )
    predicted = model.modeled_log_response(modulation).copy()
    for pass_index, group_id in enumerate(model.pass_zero_point_group_ids):
        if group_id > 0:
            predicted[:, :, pass_index] += zpt[:, group_id - 1][None, :]
    residuals = []
    for star_index in range(model.star_count):
        for filter_index in range(model.filter_count):
            epochs = np.flatnonzero(valid_measurements[star_index, filter_index])
            if epochs.size < 2:
                continue
            sigma = sigma_log_flux[star_index, filter_index, epochs]
            weights = 1.0 / sigma**2
            raw = (
                observed_log_flux[star_index, filter_index, epochs]
                - predicted[star_index, filter_index, epochs]
            )
            # Profile the unknown stellar/filter normalization for each source.
            raw = raw - np.sum(weights * raw) / np.sum(weights)
            residuals.extend(raw / sigma)
    return np.asarray(residuals, dtype=float)


def fit_actual_photometry(
    model,
    magnitudes,
    magnitude_errors,
    valid_measurements,
    amplitude_bound,
    phase_bound,
    systematic_floor_mmag=0.0,
    initial_parameters=None,
):
    sigma_stat_log_flux = LOG_FLUX_PER_MAG * magnitude_errors
    sigma_floor_log_flux = LOG_FLUX_PER_MAG * systematic_floor_mmag / 1000.0
    sigma_log_flux = np.hypot(sigma_stat_log_flux, sigma_floor_log_flux)
    observed_log_flux = -LOG_FLUX_PER_MAG * magnitudes
    repeated_groups = np.sum(valid_measurements, axis=2) >= 2
    likelihood_measurements = valid_measurements & repeated_groups[:, :, None]
    observations_used = int(np.count_nonzero(likelihood_measurements))
    source_filter_offsets = int(np.count_nonzero(repeated_groups))
    parameter_count = model.parameter_count
    lower = np.concatenate(
        (
            np.full(model.mode_count, -amplitude_bound),
            np.full(model.mode_count, -phase_bound),
            np.full(model.zero_point_count, -0.25),
        )
    )
    upper = np.concatenate(
        (
            np.full(model.mode_count, amplitude_bound),
            np.full(model.mode_count, phase_bound),
            np.full(model.zero_point_count, 0.25),
        )
    )

    def objective_residual(parameters):
        data_residual = make_real_fit_residual(
            model,
            observed_log_flux,
            sigma_log_flux,
            valid_measurements,
            parameters,
        )
        maxent_residual = model.regularization_residuals(
            parameters[: 2 * model.mode_count]
        )
        return np.concatenate((data_residual, maxent_residual))

    initial_values = (
        np.zeros(parameter_count, dtype=float)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=float).reshape(-1)
    )
    if initial_values.size != parameter_count:
        raise ValueError(
            f"Initial fit vector has {initial_values.size} values; "
            f"expected {parameter_count}"
        )
    fit = least_squares(
        objective_residual,
        initial_values,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=300,
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    residual = make_real_fit_residual(
        model,
        observed_log_flux,
        sigma_log_flux,
        valid_measurements,
        fit.x,
    )
    data_chi2 = float(np.sum(residual**2))
    return (
        fit,
        data_chi2,
        observed_log_flux,
        sigma_log_flux,
        valid_measurements,
        observations_used,
        source_filter_offsets,
    )


def fit_actual_photometry_with_profiled_systematic_floor(
    model,
    magnitudes,
    magnitude_errors,
    valid_measurements,
    amplitude_bound,
    phase_bound,
    maximum_floor_mmag,
):
    """Profile a detector-level Gaussian floor in quadrature with reported errors."""
    repeated_groups = np.sum(valid_measurements, axis=2) >= 2
    likelihood_measurements = valid_measurements & repeated_groups[:, :, None]
    cache = {}

    def evaluate(systematic_floor_mmag):
        key = float(systematic_floor_mmag)
        if key in cache:
            return cache[key]
        result = fit_actual_photometry(
            model,
            magnitudes,
            magnitude_errors,
            valid_measurements,
            amplitude_bound,
            phase_bound,
            systematic_floor_mmag=key,
        )
        fit, data_chi2, _, sigma_log_flux, _, _, _ = result
        regularization = model.regularization_residuals(
            fit.x[: 2 * model.mode_count]
        )
        log_variance = np.sum(
            np.log(np.square(sigma_log_flux[likelihood_measurements]))
        )
        profile_nll = 0.5 * (
            data_chi2 + float(np.sum(regularization**2)) + float(log_variance)
        )
        cache[key] = (profile_nll, result)
        return cache[key]

    optimum = minimize_scalar(
        lambda floor: evaluate(floor)[0],
        bounds=(0.0, maximum_floor_mmag),
        method="bounded",
        options={"xatol": 0.05, "maxiter": 40},
    )
    candidates = [0.0, float(optimum.x), float(maximum_floor_mmag)]
    best_floor = min(candidates, key=lambda floor: evaluate(floor)[0])
    profile_nll, fit_result = evaluate(best_floor)
    return (*fit_result, best_floor, profile_nll)


def profile_zero_modulation(
    model, observed_log_flux, sigma_log_flux, valid_measurements
):
    zero_modulation = np.zeros(2 * model.mode_count, dtype=float)

    if model.zero_point_count == 0:
        parameters = zero_modulation
        residual = make_real_fit_residual(
            model,
            observed_log_flux,
            sigma_log_flux,
            valid_measurements,
            parameters,
        )
        return parameters, float(np.sum(residual**2))

    def residual(zero_point_parameters):
        parameters = np.concatenate((zero_modulation, zero_point_parameters))
        return make_real_fit_residual(
            model,
            observed_log_flux,
            sigma_log_flux,
            valid_measurements,
            parameters,
        )

    fit = least_squares(
        residual,
        np.zeros(model.zero_point_count, dtype=float),
        bounds=(
            np.full(model.zero_point_count, -0.25),
            np.full(model.zero_point_count, 0.25),
        ),
        max_nfev=100,
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    parameters = np.concatenate((zero_modulation, fit.x))
    return parameters, float(np.sum(fit.fun**2))


def run_per_tile_modulation_experiment(
    output_dir,
    detector,
    model,
    global_fit,
    global_prediction_log,
    observed_log_flux,
    sigma_log_flux,
    magnitudes,
    magnitude_errors,
    valid_measurements,
    tile_ids_by_measurement,
    amplitude_bound,
    phase_bound,
    systematic_floor_mmag=0.0,
    min_sources=2,
    min_source_filter_groups=8,
    min_likelihood_rows=32,
):
    """Fit independent tile modulation vectors and interpolate unsupported tiles."""
    from scipy.optimize import least_squares

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_count = model.mode_count
    coefficient_count = 2 * mode_count
    global_coefficients = np.asarray(global_fit.x[:coefficient_count], dtype=float)
    tile_parameters = np.full(
        (model.tile_count, coefficient_count), np.nan, dtype=float
    )
    direct_fit_mask = np.zeros(model.tile_count, dtype=bool)
    records = []
    interpolation_neighbors = {}

    def data_chi2_for_mask(predicted, mask):
        chi2 = 0.0
        used_rows = 0
        for source_index in range(mask.shape[0]):
            for filter_index in range(mask.shape[1]):
                epochs = np.flatnonzero(mask[source_index, filter_index])
                if epochs.size < 2:
                    continue
                sigma = sigma_log_flux[source_index, filter_index, epochs]
                weights = 1.0 / np.square(sigma)
                raw = (
                    observed_log_flux[source_index, filter_index, epochs]
                    - predicted[source_index, filter_index, epochs]
                )
                raw = raw - np.sum(weights * raw) / np.sum(weights)
                chi2 += float(np.sum(np.square(raw / sigma)))
                used_rows += int(epochs.size)
        return chi2, used_rows

    for tile_id in range(model.tile_count):
        tile_valid = valid_measurements & (tile_ids_by_measurement == tile_id)
        repeated_groups = np.sum(tile_valid, axis=2) >= 2
        tile_likelihood = tile_valid & repeated_groups[:, :, None]
        source_indices = np.flatnonzero(np.any(repeated_groups, axis=1))
        source_count = int(source_indices.size)
        group_count = int(np.count_nonzero(repeated_groups))
        row_count = int(np.count_nonzero(tile_likelihood))
        global_tile_chi2, global_rows = data_chi2_for_mask(
            global_prediction_log, tile_likelihood
        )
        tile_row = tile_id // TILE_GRID_SIZE
        tile_col = tile_id % TILE_GRID_SIZE
        record = {
            "tile_id": tile_id,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "source_count": source_count,
            "source_filter_group_count": group_count,
            "likelihood_measurement_count": row_count,
            "global_model_chi2_on_tile_data": global_tile_chi2,
            "global_model_rows_on_tile_data": global_rows,
            "local_fit_attempted": False,
            "local_fit_success": False,
            "local_fit_jacobian_rank": 0,
            "local_fit_condition_number": float("inf"),
            "local_fit_chi2": float("nan"),
            "tile_modulation_source": "pending interpolation",
            "interpolation_neighbors": "",
        }

        enough_data = (
            source_count >= min_sources
            and group_count >= min_source_filter_groups
            and row_count >= min_likelihood_rows
        )
        if enough_data:
            record["local_fit_attempted"] = True
            local_valid = tile_valid[source_indices]
            local_model = RealPhotometryFit(
                model.wavelengths_nm,
                model.effective_area,
                model.star_photon_shapes[source_indices],
                model.tile_thickness_by_id[tile_id : tile_id + 1],
                np.zeros(local_valid.shape, dtype=int),
                model.clean_reflection_by_tile[tile_id : tile_id + 1],
                model.clean_transfer_first_by_tile[tile_id : tile_id + 1],
                model.clean_substrate_power_by_tile[tile_id : tile_id + 1],
                model.incident_admittance_by_tile[tile_id : tile_id + 1],
                model.substrate_admittance_by_tile[tile_id : tile_id + 1],
                model.incidence_sine_by_tile[tile_id : tile_id + 1],
                model.ice_index_by_tile[tile_id : tile_id + 1],
                model.basis_grouping,
                entropy_weight=model.entropy_weight,
                phase_smoothness_weight=model.phase_smoothness_weight,
                # No per-tile roll zero point is fitted; its absolute level is
                # profiled with each source/filter normalization.
                pass_zero_point_group_ids=np.zeros(model.pass_count, dtype=int),
            )
            local_model.active_source_count = source_count
            local_model.detector = detector
            (
                local_fit,
                local_chi2,
                _local_observed,
                _local_sigma,
                _local_valid,
                _local_rows,
                _local_groups,
            ) = fit_actual_photometry(
                local_model,
                magnitudes[source_indices],
                magnitude_errors[source_indices],
                local_valid,
                amplitude_bound,
                phase_bound,
                systematic_floor_mmag=systematic_floor_mmag,
                initial_parameters=global_coefficients,
            )
            rank = int(np.linalg.matrix_rank(local_fit.jac))
            condition = float(np.linalg.cond(local_fit.jac))
            record.update(
                {
                    "local_fit_success": bool(local_fit.success),
                    "local_fit_jacobian_rank": rank,
                    "local_fit_condition_number": condition,
                    "local_fit_chi2": float(local_chi2),
                }
            )
            accepted = (
                bool(local_fit.success)
                and rank == coefficient_count
                and np.isfinite(condition)
                and condition < 1.0e12
                and np.all(np.isfinite(local_fit.x[:coefficient_count]))
            )
            if accepted:
                tile_parameters[tile_id] = local_fit.x[:coefficient_count]
                direct_fit_mask[tile_id] = True
                record["tile_modulation_source"] = "direct tile fit"
            else:
                record["tile_modulation_source"] = "interpolated: local fit unsupported"
        else:
            record["tile_modulation_source"] = "interpolated: insufficient star coverage"
        records.append(record)

    direct_tile_ids = np.flatnonzero(direct_fit_mask)
    if direct_tile_ids.size == 0:
        raise RuntimeError("No tile modulation fit had sufficient rank and coverage")
    grid_coordinates = np.asarray(
        [(tile_id // TILE_GRID_SIZE, tile_id % TILE_GRID_SIZE)
         for tile_id in range(model.tile_count)],
        dtype=float,
    )
    for tile_id in np.flatnonzero(~direct_fit_mask):
        distances = np.linalg.norm(
            grid_coordinates[direct_tile_ids] - grid_coordinates[tile_id], axis=1
        )
        nearest_order = np.argsort(distances)[: min(4, direct_tile_ids.size)]
        neighbors = direct_tile_ids[nearest_order]
        neighbor_distances = distances[nearest_order]
        weights = 1.0 / np.square(np.maximum(neighbor_distances, 1.0e-6))
        weights /= np.sum(weights)
        tile_parameters[tile_id] = np.sum(
            tile_parameters[neighbors] * weights[:, None], axis=0
        )
        interpolation_neighbors[int(tile_id)] = {
            "tile_ids": neighbors.astype(int).tolist(),
            "weights": weights.tolist(),
        }
        records[tile_id]["interpolation_neighbors"] = ";".join(
            f"{int(neighbor)}:{weight:.6g}"
            for neighbor, weight in zip(neighbors, weights)
        )

    tile_rows = pd.DataFrame(records)
    summary_csv = output_dir / "per_tile_modulation_summary.csv"
    tile_rows.to_csv(summary_csv, index=False)

    coefficient_rows = []
    for tile_id in range(model.tile_count):
        row = records[tile_id]
        for mode_index, mode_name in enumerate(model.basis_names):
            coefficient_rows.append(
                {
                    "tile_id": tile_id,
                    "tile_row": tile_id // TILE_GRID_SIZE,
                    "tile_col": tile_id % TILE_GRID_SIZE,
                    "modulation_source": row["tile_modulation_source"],
                    "source_count": row["source_count"],
                    "source_filter_group_count": row["source_filter_group_count"],
                    "mode": mode_name,
                    "global_log_amplitude": global_coefficients[mode_index],
                    "tile_log_amplitude": tile_parameters[tile_id, mode_index],
                    "delta_log_amplitude_from_global": (
                        tile_parameters[tile_id, mode_index]
                        - global_coefficients[mode_index]
                    ),
                    "global_phase_rad": global_coefficients[
                        mode_count + mode_index
                    ],
                    "tile_phase_rad": tile_parameters[tile_id, mode_count + mode_index],
                    "delta_phase_from_global_rad": (
                        tile_parameters[tile_id, mode_count + mode_index]
                        - global_coefficients[mode_count + mode_index]
                    ),
                    "interpolation_neighbors": row["interpolation_neighbors"],
                }
            )
    coefficients_csv = output_dir / "per_tile_modulation_coefficients.csv"
    pd.DataFrame(coefficient_rows).to_csv(coefficients_csv, index=False)

    modulation_ratio = np.exp(
        tile_parameters[:, :mode_count] @ model.basis
        + 1j * (tile_parameters[:, mode_count:] @ model.basis)
    )
    coefficient_archive = output_dir / "per_tile_modulation_test.npz"
    np.savez_compressed(
        coefficient_archive,
        detector=np.asarray(detector),
        tile_ids=np.arange(model.tile_count, dtype=np.int16),
        tile_modulation_parameters=tile_parameters.astype(np.float32),
        modulation_ratio_real=modulation_ratio.real.astype(np.float32),
        modulation_ratio_imag=modulation_ratio.imag.astype(np.float32),
        wavelengths_nm=model.wavelengths_nm.astype(np.float32),
        global_modulation_parameters=global_coefficients.astype(np.float32),
        direct_fit_mask=direct_fit_mask,
        source_count=tile_rows["source_count"].to_numpy(dtype=np.int16),
        local_fit_chi2=tile_rows["local_fit_chi2"].to_numpy(dtype=np.float32),
        global_model_chi2=tile_rows["global_model_chi2_on_tile_data"].to_numpy(
            dtype=np.float32
        ),
    )

    fitted_tiles = tile_rows.loc[direct_fit_mask]
    interpolated_tiles = tile_rows.loc[~direct_fit_mask]
    summary = {
        "detector": detector,
        "global_fit_seed": "tile-aware global modulation MAP",
        "shared_global_parameter_count": int(coefficient_count),
        "tile_parameter_count": int(coefficient_count),
        "total_tile_count": int(model.tile_count),
        "directly_fitted_tile_count": int(direct_fit_mask.sum()),
        "interpolated_tile_count": int((~direct_fit_mask).sum()),
        "directly_fitted_tile_ids": direct_tile_ids.astype(int).tolist(),
        "interpolated_tile_ids": np.flatnonzero(~direct_fit_mask).astype(int).tolist(),
        "coverage_thresholds": {
            "minimum_sources": min_sources,
            "minimum_repeated_source_filter_groups": min_source_filter_groups,
            "minimum_likelihood_rows": min_likelihood_rows,
            "fit_rank_required": int(coefficient_count),
            "condition_number_max": 1.0e12,
        },
        "interpolation": {
            "method": "inverse-distance weighted nearest four directly fitted tile centers",
            "tiles": interpolation_neighbors,
        },
        "global_chi2_on_directly_fitted_tiles": float(
            fitted_tiles["global_model_chi2_on_tile_data"].sum()
        ),
        "tile_fit_chi2_on_directly_fitted_tiles": float(
            fitted_tiles["local_fit_chi2"].sum()
        ),
        "direct_fit_summary_csv": str(summary_csv),
        "coefficient_csv": str(coefficients_csv),
        "coefficient_archive": str(coefficient_archive),
        "interpolated_tile_count_with_no_stars": int(
            (interpolated_tiles["source_count"] == 0).sum()
        ),
    }
    summary_path = output_dir / "per_tile_modulation_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(
        mode_count, 2, figsize=(11.5, 2.0 * mode_count), squeeze=False
    )
    amplitude_grid = np.empty((mode_count, TILE_GRID_SIZE, TILE_GRID_SIZE))
    phase_grid = np.empty_like(amplitude_grid)
    interpolation_grid = ~direct_fit_mask.reshape(TILE_GRID_SIZE, TILE_GRID_SIZE)
    for tile_id in range(model.tile_count):
        row, col = divmod(tile_id, TILE_GRID_SIZE)
        amplitude_grid[:, row, col] = 100.0 * np.expm1(
            tile_parameters[tile_id, :mode_count]
        )
        phase_grid[:, row, col] = tile_parameters[tile_id, mode_count:]
    for mode_index, mode_name in enumerate(model.basis_names):
        for axis, values, title in (
            (axes[mode_index, 0], amplitude_grid[mode_index], "amplitude change (%)"),
            (axes[mode_index, 1], phase_grid[mode_index], "phase change (rad)"),
        ):
            image = axis.imshow(values, origin="lower", cmap="coolwarm", aspect="equal")
            if np.any(interpolation_grid):
                rows, cols = np.nonzero(interpolation_grid)
                axis.scatter(cols, rows, marker="x", s=12, color="black", linewidths=0.6)
            axis.set_title(f"{mode_name}: {title}")
            axis.set_xlabel("tile column")
            axis.set_ylabel("tile row")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        f"{detector} per-tile modulation test: "
        f"{int(direct_fit_mask.sum())} direct fits, {int((~direct_fit_mask).sum())} interpolated",
        y=1.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    map_path = output_dir / "per_tile_modulation_maps.png"
    figure.savefig(map_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    summary["coefficient_map_plot"] = str(map_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _phase_unwrapped_long_to_short(wavelengths_nm, reflection):
    phase = np.unwrap(np.angle(reflection)[::-1])[::-1]
    return phase - phase[-1]


def identify_repeat_photometry_outliers(
    source_ids,
    magnitudes,
    magnitude_errors,
    valid_measurements,
    pass_group_ids,
    pass_names,
    tile_ids_by_measurement=None,
    x_pixels=None,
    y_pixels=None,
    *,
    min_group_size=4,
    min_deviation_mag=0.25,
    sigma_threshold=4.0,
):
    """Flag isolated, large magnitude excursions within a source/filter/roll group."""
    outlier_mask = np.zeros_like(valid_measurements, dtype=bool)
    centered_from_median = np.full(magnitudes.shape, np.nan, dtype=float)
    records = []
    pass_group_ids = np.asarray(pass_group_ids, dtype=int)
    for source_index, source_id in enumerate(source_ids):
        for filter_index, filter_name in enumerate(FILTERS):
            for group_id in np.unique(pass_group_ids):
                group_valid = (
                    valid_measurements[source_index, filter_index]
                    & (pass_group_ids == group_id)
                )
                if tile_ids_by_measurement is None:
                    tile_values = np.zeros(pass_group_ids.shape, dtype=int)
                else:
                    tile_values = tile_ids_by_measurement[source_index, filter_index]
                for tile_id in np.unique(tile_values[group_valid]):
                    epochs = np.flatnonzero(group_valid & (tile_values == tile_id))
                    if epochs.size < 2:
                        continue
                    values = magnitudes[source_index, filter_index, epochs]
                    median = float(np.median(values))
                    deviations = np.abs(values - median)
                    mad = float(np.median(deviations))
                    centered_from_median[source_index, filter_index, epochs] = (
                        values - median
                    ) * 1000.0
                    if epochs.size < min_group_size:
                        continue
                    for local_index, pass_index in enumerate(epochs):
                        measurement_sigma = float(
                            magnitude_errors[source_index, filter_index, pass_index]
                        )
                        threshold = max(
                            min_deviation_mag, sigma_threshold * measurement_sigma
                        )
                        deviation = float(deviations[local_index])
                        if deviation <= threshold:
                            continue
                        outlier_mask[source_index, filter_index, pass_index] = True
                        records.append(
                            {
                                "source_id": source_id,
                                "filter": filter_name,
                                "pass": pass_names[pass_index],
                                "group_id": int(group_id),
                                "tile_id": int(tile_id),
                                "x_pixel": (
                                    float(x_pixels[source_index, filter_index, pass_index])
                                    if x_pixels is not None
                                    else float("nan")
                                ),
                                "y_pixel": (
                                    float(y_pixels[source_index, filter_index, pass_index])
                                    if y_pixels is not None
                                    else float("nan")
                                ),
                                "measured_mag": float(values[local_index]),
                                "group_median_mag": median,
                                "mag_err": measurement_sigma,
                                "group_mad_mag": mad,
                                "deviation_mmag": deviation * 1000.0,
                                "threshold_mmag": threshold * 1000.0,
                                "measurement_significance": deviation / measurement_sigma
                                if measurement_sigma > 0.0
                                else float("inf"),
                                "group_measurement_count": int(epochs.size),
                            }
                        )
    return outlier_mask, centered_from_median, records


def group_centered_magnitude_series(
    values_mag, magnitude_errors, likelihood_measurements, pass_group_ids
):
    """Center each source/filter within its own roll group for display."""
    centered = np.full(values_mag.shape, np.nan, dtype=float)
    pass_group_ids = np.asarray(pass_group_ids, dtype=int)
    for source_index in range(values_mag.shape[0]):
        for filter_index in range(values_mag.shape[1]):
            for group_id in np.unique(pass_group_ids):
                epochs = np.flatnonzero(
                    likelihood_measurements[source_index, filter_index]
                    & (pass_group_ids == group_id)
                )
                if epochs.size < 2:
                    continue
                weights = 1.0 / np.square(
                    magnitude_errors[source_index, filter_index, epochs]
                )
                group_center = np.sum(
                    weights * values_mag[source_index, filter_index, epochs]
                ) / np.sum(weights)
                centered[source_index, filter_index, epochs] = (
                    values_mag[source_index, filter_index, epochs] - group_center
                ) * 1000.0
    return centered


def plot_repeat_photometry_outliers(
    output_path,
    detector,
    target_days_by_filter,
    likelihood_measurements,
    centered_from_median_mmag,
    outlier_mask,
    active_source_count,
):
    """Diagnostic scatter of all repeated points, with flagged excursions highlighted."""
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 10.5), sharex=True)
    all_points = int(np.count_nonzero(likelihood_measurements))
    outlier_points = int(np.count_nonzero(outlier_mask))
    all_points_labeled = False
    outlier_labeled = False
    for filter_index, axis in enumerate(axes.flat):
        for pass_index, day in enumerate(target_days_by_filter[filter_index]):
            selected = likelihood_measurements[:, filter_index, pass_index]
            if np.any(selected):
                axis.scatter(
                    np.full(np.count_nonzero(selected), day),
                    centered_from_median_mmag[selected, filter_index, pass_index],
                    s=7,
                    alpha=0.22,
                    color="0.45",
                    rasterized=True,
                    label="all repeated measurements" if not all_points_labeled else None,
                )
                all_points_labeled = True
            flagged = outlier_mask[:, filter_index, pass_index]
            if np.any(flagged):
                axis.scatter(
                    np.full(np.count_nonzero(flagged), day),
                    centered_from_median_mmag[flagged, filter_index, pass_index],
                    s=24,
                    marker="x",
                    color="#c43c39",
                    linewidths=1.0,
                    zorder=5,
                    label="candidate outlier" if not outlier_labeled else None,
                )
                outlier_labeled = True
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.set_title(FILTERS[filter_index])
        axis.grid(True, alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Δmag (mmag)")
    axes[-1, 0].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    axes[-1, 1].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        fontsize=7,
    )
    figure.suptitle(
        f"CAR201 {detector} repeated photometry: {active_source_count} active sources, "
        f"{all_points:,} measurements; {outlier_points} isolated outlier candidates\n"
        "Values are relative to each source/filter/roll-group median",
        y=0.99,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.86))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _legacy_plot_high_cadence_photometry_model(
    output_path,
    model,
    fit_parameters,
    zero_parameters,
    representative_tile_indices,
    tile_days,
    tile_curves,
    reset_days,
    target_days_by_filter,
    summary_rows,
    group_centered_observed_mmag,
    group_centered_fit_mmag,
    group_centered_zero_mmag,
    likelihood_measurements,
    chi2,
    dof,
    dense_samples=241,
):
    lower_day = float(np.min(target_days_by_filter))
    upper_day = float(np.max(target_days_by_filter))
    dense_days = np.linspace(lower_day, upper_day, dense_samples)
    dense_tile_history = np.asarray(
        [
            mock_model.interpolate_fixed_history(
                tile_days[index], tile_curves[index], reset_days, dense_days
            )
            for index in representative_tile_indices
        ],
        dtype=float,
    )
    fitted_response = model.modeled_log_response_for_common_history(
        fit_parameters[: 2 * model.mode_count], dense_tile_history
    )
    zero_response = model.modeled_log_response_for_common_history(
        zero_parameters[: 2 * model.mode_count], dense_tile_history
    )
    fitted_zpt = fit_parameters[2 * model.mode_count :].reshape(
        model.filter_count, model.zero_point_group_count
    )
    zero_zpt = zero_parameters[2 * model.mode_count :].reshape(
        model.filter_count, model.zero_point_group_count
    )

    def expand_roll_offsets(offsets, pass_days):
        curve = np.zeros_like(dense_days)
        for group_id in range(1, model.zero_point_group_count + 1):
            before = pass_days[model.pass_zero_point_group_ids < group_id]
            after = pass_days[model.pass_zero_point_group_ids == group_id]
            if before.size and after.size:
                boundary_day = 0.5 * (float(np.max(before)) + float(np.min(after)))
                curve[dense_days >= boundary_day] += offsets[group_id - 1]
        return curve

    figure, axes = plt.subplots(4, 2, figsize=(14.0, 10.5), sharex=True)
    pass_group_ids = np.asarray(model.pass_zero_point_group_ids, dtype=int)
    post_pass_indices = np.flatnonzero(pass_group_ids > 0)
    for filter_index, axis in enumerate(axes.flat):
        pass_days = target_days_by_filter[filter_index]
        fitted_zpt_curve = expand_roll_offsets(fitted_zpt[filter_index], pass_days)
        zero_zpt_curve = expand_roll_offsets(zero_zpt[filter_index], pass_days)
        reference_day = float(pass_days[0])
        fitted_reference = np.asarray(
            [np.interp(reference_day, dense_days, curve) for curve in fitted_response[:, filter_index]]
        )
        zero_reference = np.asarray(
            [np.interp(reference_day, dense_days, curve) for curve in zero_response[:, filter_index]]
        )
        fitted_delta = (
            fitted_response[:, filter_index, :]
            - fitted_reference[:, None]
            + fitted_zpt_curve[None, :]
        )
        zero_delta = (
            zero_response[:, filter_index, :]
            - zero_reference[:, None]
            + zero_zpt_curve[None, :]
        )
        fitted_median = -np.median(fitted_delta, axis=0) / LOG_FLUX_PER_MAG * 1000.0
        zero_median = -np.median(zero_delta, axis=0) / LOG_FLUX_PER_MAG * 1000.0
        before_roll_days = pass_days[pass_group_ids == 0]
        after_roll_days = pass_days[pass_group_ids > 0]
        if before_roll_days.size and after_roll_days.size:
            roll_boundary_day = 0.5 * (
                float(np.max(before_roll_days)) + float(np.min(after_roll_days))
            )
            fitted_median[dense_days >= roll_boundary_day] = np.nan
            zero_median[dense_days >= roll_boundary_day] = np.nan
        values = [row for row in summary_rows if row["filter"] == FILTERS[filter_index]]
        observed_days = np.asarray(
            [row["day_from_fixed_ice_reference"] for row in values]
        )
        observed_median = np.asarray(
            [row["observed_median_delta_mmag"] for row in values]
        )
        lower = observed_median - np.asarray(
            [row["observed_p16_delta_mmag"] for row in values]
        )
        upper = np.asarray(
            [row["observed_p84_delta_mmag"] for row in values]
        ) - observed_median
        axis.errorbar(
            observed_days,
            observed_median,
            yerr=np.vstack((lower, upper)),
            fmt="o",
            linestyle="none",
            markersize=4.5,
            color="0.25",
            ecolor="0.6",
            capsize=2,
            label="observed median; p16–p84",
        )
        axis.plot(dense_days, fitted_median, color="#d47730", linewidth=1.5, label="fitted TMM (dense time grid)")
        axis.plot(
            dense_days,
            zero_median,
            color="0.15",
            linestyle="-.",
            linewidth=1.2,
            label="zero modulation",
        )
        if after_roll_days.size:
            post_days = []
            post_fit_medians = []
            post_zero_medians = []
            for post_index in np.flatnonzero(pass_group_ids > 0):
                selected = likelihood_measurements[:, filter_index, post_index]
                if not np.any(selected):
                    continue
                post_days.append(pass_days[post_index])
                post_fit_medians.append(
                    float(
                        np.median(
                            group_centered_fit_mmag[
                                selected, filter_index, post_index
                            ]
                        )
                    )
                )
                post_zero_medians.append(
                    float(
                        np.median(
                            group_centered_zero_mmag[
                                selected, filter_index, post_index
                            ]
                        )
                    )
                )
            axis.plot(
                post_days,
                post_fit_medians,
                color="#d47730",
                marker="s",
                linewidth=1.5,
                label="post-roll fitted TMM at passes (group-centered)",
            )
            axis.plot(
                post_days,
                post_zero_medians,
                color="0.15",
                linestyle="-.",
                marker="x",
                linewidth=1.2,
                label="post-roll zero modulation at passes",
            )
        for pass_index in np.flatnonzero(pass_group_ids > 0):
            selected = likelihood_measurements[:, filter_index, pass_index]
            if not np.any(selected):
                continue
            day = pass_days[pass_index]
            post_values = group_centered_observed_mmag[
                selected, filter_index, pass_index
            ]
            post_median = float(np.median(post_values))
            post_p16, post_p84 = np.percentile(post_values, [16.0, 84.0])
            axis.errorbar(
                [day],
                [post_median],
                yerr=np.asarray([[post_median - post_p16], [post_p84 - post_median]]),
                fmt="o",
                linestyle="none",
                markersize=4.5,
                color="#3478a8",
                ecolor="#3478a8",
                capsize=2,
                zorder=5,
                label=(
                    "post-roll median; p16–p84"
                    if filter_index == 0 and pass_index == post_pass_indices[0]
                    else None
                ),
            )
        for reset_day in reset_days:
            if lower_day <= reset_day <= upper_day:
                axis.axvline(reset_day, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_title(FILTERS[filter_index])
        axis.grid(True, alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Δ magnitude (mmag)")
    axes[-1, 0].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    axes[-1, 1].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        fontsize=7,
    )
    figure.suptitle(
        f"Actual CAR201 {model.detector} photometry; fixed ice history; dense TMM prediction\n"
        f"{model.active_source_count} active of {model.star_count} SED-matched sources; "
        "post-roll medians and p16–p84; group-centered model predictions shown\n"
        f"data χ²/dof={chi2:.1f}/{dof} "
        f"({chi2/dof:.2f}); MaxEnt λ={model.entropy_weight:g}; "
        f"phase λ={model.phase_smoothness_weight:g}",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.85))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _legacy_plot_high_cadence_photometry_model(
    output_path,
    model,
    pass_names,
    target_days_by_filter,
    group_centered_observed_mmag,
    group_centered_fit_mmag,
    group_centered_zero_mmag,
    likelihood_measurements,
    pass_group_ids,
    group_names,
    reset_days,
    chi2,
    dof,
):
    """Plot all-source data and tile-matched model medians at observed passes."""
    pass_group_ids = np.asarray(pass_group_ids, dtype=int)
    pretty_group_names = {
        group_id: ("pre-roll" if group_id == 0 else "post-roll")
        for group_id in range(len(group_names))
    }
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 10.5), sharex=True)
    for filter_index, axis in enumerate(axes.flat):
        pass_days = target_days_by_filter[filter_index]
        for group_id in np.unique(pass_group_ids):
            pass_indices = np.flatnonzero(pass_group_ids == group_id)
            days = []
            data_medians = []
            data_lower = []
            data_upper = []
            fit_medians = []
            zero_medians = []
            for pass_index in pass_indices:
                selected = likelihood_measurements[:, filter_index, pass_index]
                if not np.any(selected):
                    continue
                observed = group_centered_observed_mmag[
                    selected, filter_index, pass_index
                ]
                fitted = group_centered_fit_mmag[
                    selected, filter_index, pass_index
                ]
                zero = group_centered_zero_mmag[
                    selected, filter_index, pass_index
                ]
                p16, median, p84 = np.percentile(observed, [16.0, 50.0, 84.0])
                days.append(float(pass_days[pass_index]))
                data_medians.append(float(median))
                data_lower.append(float(median - p16))
                data_upper.append(float(p84 - median))
                fit_medians.append(float(np.median(fitted)))
                zero_medians.append(float(np.median(zero)))
            if not days:
                continue
            group_label = pretty_group_names.get(group_id, f"group {group_id}")
            data_color = "0.25" if group_id == 0 else "#3478a8"
            axis.errorbar(
                days,
                data_medians,
                yerr=np.asarray([data_lower, data_upper]),
                fmt="o",
                linestyle="none",
                markersize=4.5,
                color=data_color,
                ecolor=("0.6" if group_id == 0 else "#3478a8"),
                capsize=2,
                label=f"{group_label} data median; p16–p84",
            )
            axis.plot(
                days,
                fit_medians,
                color="#d47730",
                marker="s",
                linewidth=1.4,
                label=f"{group_label} tile-specific TMM median",
            )
            axis.plot(
                days,
                zero_medians,
                color="0.15",
                linestyle="-.",
                marker="x",
                linewidth=1.1,
                label=f"{group_label} zero modulation",
            )
        pre_days = pass_days[pass_group_ids == 0]
        post_days = pass_days[pass_group_ids > 0]
        if pre_days.size and post_days.size:
            axis.axvline(
                0.5 * (float(np.max(pre_days)) + float(np.min(post_days))),
                color="0.55",
                linestyle=":",
                linewidth=0.8,
            )
        for reset_day in reset_days:
            if np.min(pass_days) <= reset_day <= np.max(pass_days):
                axis.axvline(reset_day, color="0.75", linestyle=":", linewidth=0.7)
        axis.axhline(0.0, color="0.8", linewidth=0.7)
        axis.set_title(FILTERS[filter_index])
        axis.grid(True, alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Within-roll-group Δmag (mmag)")
    axes[-1, 0].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    axes[-1, 1].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        fontsize=7,
    )
    figure.suptitle(
        f"Actual CAR201 {model.detector}: global modulation, tile-specific TMM and ice history\n"
        f"{model.active_source_count} active of {model.star_count} SED-matched sources; "
        "data medians and p16–p84 with model predictions at observed passes\n"
        f"data χ²/dof={chi2:.1f}/{dof} ({chi2/dof:.2f}); "
        f"MaxEnt λ={model.entropy_weight:g}; phase λ={model.phase_smoothness_weight:g}",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.85))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_high_cadence_photometry_model(
    output_path,
    model,
    target_days_by_filter,
    group_centered_observed_mmag,
    group_centered_fit_mmag,
    group_centered_zero_mmag,
    likelihood_measurements,
    pass_group_ids,
    group_names,
    reset_days,
    chi2,
    dof,
):
    """Plot per-pass medians/intervals and the tile-matched model at all epochs."""
    pass_group_ids = np.asarray(pass_group_ids, dtype=int)
    group_labels = {
        group_id: ("pre-roll" if group_id == 0 else "post-roll")
        for group_id in range(len(group_names))
    }
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 10.5), sharex=True)
    for filter_index, axis in enumerate(axes.flat):
        pass_days = target_days_by_filter[filter_index]
        for group_id in np.unique(pass_group_ids):
            pass_indices = np.flatnonzero(pass_group_ids == group_id)
            days = []
            observed_medians = []
            observed_p16 = []
            observed_p84 = []
            fit_medians = []
            zero_medians = []
            for pass_index in pass_indices:
                selected = likelihood_measurements[:, filter_index, pass_index]
                if not np.any(selected):
                    continue
                observed = group_centered_observed_mmag[
                    selected, filter_index, pass_index
                ]
                fitted = group_centered_fit_mmag[
                    selected, filter_index, pass_index
                ]
                zero = group_centered_zero_mmag[
                    selected, filter_index, pass_index
                ]
                p16, median, p84 = np.percentile(observed, [16.0, 50.0, 84.0])
                days.append(float(pass_days[pass_index]))
                observed_medians.append(float(median))
                observed_p16.append(float(median - p16))
                observed_p84.append(float(p84 - median))
                fit_medians.append(float(np.median(fitted)))
                zero_medians.append(float(np.median(zero)))
            if not days:
                continue
            label = group_labels.get(group_id, f"group {group_id}")
            data_color = "0.25" if group_id == 0 else "#3478a8"
            error_color = "0.6" if group_id == 0 else "#3478a8"
            axis.errorbar(
                days,
                observed_medians,
                yerr=np.asarray([observed_p16, observed_p84]),
                fmt="o",
                linestyle="none",
                markersize=4.5,
                color=data_color,
                ecolor=error_color,
                capsize=2,
                label=f"{label} data median; p16–p84",
            )
            axis.plot(
                days,
                fit_medians,
                color="#d47730",
                marker="s",
                linewidth=1.5,
                label=f"{label} tile-specific TMM median",
            )
            axis.plot(
                days,
                zero_medians,
                color="0.15",
                linestyle="-.",
                marker="x",
                linewidth=1.2,
                label=f"{label} zero-modulation median",
            )
        pre_days = pass_days[pass_group_ids == 0]
        post_days = pass_days[pass_group_ids > 0]
        if pre_days.size and post_days.size:
            axis.axvline(
                0.5 * (float(np.max(pre_days)) + float(np.min(post_days))),
                color="0.55",
                linestyle=":",
                linewidth=0.8,
            )
        for reset_day in reset_days:
            if np.min(pass_days) <= reset_day <= np.max(pass_days):
                axis.axvline(reset_day, color="0.75", linestyle=":", linewidth=0.7)
        axis.axhline(0.0, color="0.8", linewidth=0.7)
        axis.set_title(FILTERS[filter_index])
        axis.grid(True, alpha=0.25)

    figure.supylabel("Within-roll-group Δ magnitude (mmag)", x=0.012)
    axes[-1, 0].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    axes[-1, 1].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        fontsize=7,
    )
    figure.suptitle(
        f"Actual CAR201 {model.detector}: global modulation; tile-specific TMM and ice history\n"
        f"{model.active_source_count} active of {model.star_count} SED-matched sources; "
        "population medians; p16–p84 is star-to-star spread, not fit uncertainty\n"
        "No individual-star trajectories shown; use per-source correction diagnostics\n"
        f"data χ²/dof={chi2:.1f}/{dof} ({chi2/dof:.2f}); "
        f"MaxEnt λ={model.entropy_weight:g}; phase λ={model.phase_smoothness_weight:g}",
        y=0.995,
    )
    figure.tight_layout(rect=(0.045, 0.02, 0.99, 0.82))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_clean_complex_reflection(
    output_path,
    detector,
    wavelengths_nm,
    clean_reflection,
    basis,
    log_amplitude_coefficients,
    phase_coefficients,
    chi2,
    dof,
    maxent_weight,
    phase_smoothness_weight,
):
    log_amplitude = log_amplitude_coefficients @ basis
    phase_modulation = phase_coefficients @ basis
    fitted_clean_reflection = clean_reflection * np.exp(
        log_amplitude + 1j * phase_modulation
    )
    inverse_wavelength_per_micron = 1000.0 / np.asarray(wavelengths_nm, dtype=float)
    order = np.argsort(inverse_wavelength_per_micron)
    x = inverse_wavelength_per_micron[order]
    phase_clean = _phase_unwrapped_long_to_short(wavelengths_nm, clean_reflection)[order]
    phase_fitted = _phase_unwrapped_long_to_short(
        wavelengths_nm, fitted_clean_reflection
    )[order]
    clean_reflection = clean_reflection[order]
    fitted_clean_reflection = fitted_clean_reflection[order]
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True)
    axes[0].plot(
        x,
        np.abs(clean_reflection),
        color="0.35",
        linestyle="--",
        label="clean TMM, zero-thickness ice",
    )
    axes[0].plot(
        x,
        np.abs(fitted_clean_reflection),
        color="#d47730",
        linewidth=1.8,
        label="fitted clean complex reflection",
    )
    axes[0].set_ylabel(r"$|r(\lambda)|$")
    axes[0].legend(fontsize=8, loc="best")
    axes[1].plot(
        x,
        phase_clean,
        color="0.35",
        linestyle="--",
        label="clean TMM, zero-thickness ice",
    )
    axes[1].plot(
        x,
        phase_fitted,
        color="#d47730",
        linewidth=1.8,
        label="fitted clean complex reflection",
    )
    axes[1].set_ylabel("Unwrapped reflection phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    axes[1].legend(fontsize=8, loc="best")
    for axis in axes:
        for led_index, led_center in enumerate(mock_model.LED_CENTERS_NM, start=1):
            led_inverse_wavelength = 1000.0 / led_center
            axis.axvline(
                led_inverse_wavelength,
                color="0.45",
                linestyle=":",
                linewidth=1.6,
            )
            if axis is axes[0]:
                axis.text(
                    led_inverse_wavelength,
                    0.98,
                    f"LED{led_index}",
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    color="0.4",
                    fontsize=7,
                )
        axis.grid(True, alpha=0.25)
        axis.set_xlim(x[0], x[-1])

    def inverse_micron_to_wavelength_nm(inverse_micron):
        values = np.asarray(inverse_micron, dtype=float)
        return np.divide(
            1000.0,
            values,
            out=np.full(values.shape, np.nan, dtype=float),
            where=values > 0.0,
        )

    def wavelength_nm_to_inverse_micron(wavelength_nm):
        values = np.asarray(wavelength_nm, dtype=float)
        return np.divide(
            1000.0,
            values,
            out=np.full(values.shape, np.nan, dtype=float),
            where=values > 0.0,
        )

    wavelength_axis = axes[0].secondary_xaxis(
        "top",
        functions=(
            inverse_micron_to_wavelength_nm,
            wavelength_nm_to_inverse_micron,
        ),
    )
    wavelength_ticks_nm = np.asarray([2300.0, 1700.0, 1300.0, 1000.0, 800.0, 600.0, 500.0])
    wavelength_axis.set_xticks(wavelength_ticks_nm)
    wavelength_axis.set_xticklabels([f"{value:g}" for value in wavelength_ticks_nm])
    wavelength_axis.set_xlabel("Wavelength (nm)")
    figure.suptitle(
        f"{detector} clean complex reflection: TMM baseline and fitted clean response\n"
        f"No ice coating shown; data χ²/dof={chi2:.1f}/{dof} ({chi2/dof:.2f}); "
        f"MaxEnt λ={maxent_weight:g}; phase λ={phase_smoothness_weight:g}",
        y=0.99,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_clean_reflectance_modulation_ratio(
    output_path,
    detector,
    wavelengths_nm,
    basis,
    log_amplitude_coefficients,
    phase_coefficients,
    maxent_weight,
    phase_smoothness_weight,
):
    """Plot the fitted multiplicative clean-reflection modulation alone."""
    log_amplitude = log_amplitude_coefficients @ basis
    phase_modulation = phase_coefficients @ basis
    ratio = np.exp(log_amplitude + 1j * phase_modulation)
    inverse_wavelength_per_micron = 1000.0 / np.asarray(wavelengths_nm, dtype=float)
    order = np.argsort(inverse_wavelength_per_micron)
    x = inverse_wavelength_per_micron[order]
    amplitude_ratio = np.abs(ratio)[order]
    phase_ratio = phase_modulation[order]

    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True)
    axes[0].plot(x, amplitude_ratio, color="#d47730", linewidth=1.8)
    axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel(r"$|r_{\rm mod}/r_{\rm TMM}|$")
    axes[1].plot(x, phase_ratio, color="#d47730", linewidth=1.8)
    axes[1].axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel(r"$\arg(r_{\rm mod}/r_{\rm TMM})$ (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")

    for axis in axes:
        for led_index, led_center in enumerate(mock_model.LED_CENTERS_NM, start=1):
            led_inverse_wavelength = 1000.0 / led_center
            axis.axvline(
                led_inverse_wavelength,
                color="0.45",
                linestyle=":",
                linewidth=1.6,
            )
            if axis is axes[0]:
                axis.text(
                    led_inverse_wavelength,
                    0.98,
                    f"LED{led_index}",
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    color="0.4",
                    fontsize=7,
                )
        axis.grid(True, alpha=0.25)
        axis.set_xlim(x[0], x[-1])

    def inverse_micron_to_wavelength_nm(inverse_micron):
        values = np.asarray(inverse_micron, dtype=float)
        return np.divide(
            1000.0,
            values,
            out=np.full(values.shape, np.nan, dtype=float),
            where=values > 0.0,
        )

    def wavelength_nm_to_inverse_micron(wavelength_nm):
        values = np.asarray(wavelength_nm, dtype=float)
        return np.divide(
            1000.0,
            values,
            out=np.full(values.shape, np.nan, dtype=float),
            where=values > 0.0,
        )

    wavelength_axis = axes[0].secondary_xaxis(
        "top",
        functions=(inverse_micron_to_wavelength_nm, wavelength_nm_to_inverse_micron),
    )
    wavelength_ticks_nm = np.asarray(
        [2300.0, 1700.0, 1300.0, 1000.0, 800.0, 600.0, 500.0]
    )
    wavelength_axis.set_xticks(wavelength_ticks_nm)
    wavelength_axis.set_xticklabels([f"{value:g}" for value in wavelength_ticks_nm])
    wavelength_axis.set_xlabel("Wavelength (nm)")
    figure.suptitle(
        f"{detector} clean-reflection modulation ratio $r_{{mod}}/r_{{TMM}}$\n"
        f"LED-anchored; MaxEnt λ={maxent_weight:g}; "
        f"phase λ={phase_smoothness_weight:g}",
        y=0.99,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photometry-dir", type=Path, default=DEFAULT_PHOTOMETRY_DIR)
    parser.add_argument("--sed-posteriors", type=Path, default=DEFAULT_SED_POSTERIORS)
    parser.add_argument(
        "--sed-grid",
        type=Path,
        default=CAR201_ROOT / "sed/btsettl-spectral-grid.h5",
    )
    parser.add_argument("--effective-area", type=Path, default=DEFAULT_EFFECTIVE_AREA)
    parser.add_argument("--optical-cache", type=Path, default=None)
    parser.add_argument(
        "--optical-cache-root",
        type=Path,
        default=None,
        help="Directory containing tile_###/relative_response.npz for the detector.",
    )
    parser.add_argument("--ice-history-fit-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--detector", default="WFI16")
    parser.add_argument(
        "--pass-set",
        choices=tuple(PASS_SET_CONFIG),
        default="ep3",
        help="Select ep3, ep4, or the combined ep3+ep4 time series.",
    )
    parser.add_argument(
        "--allow-partial-source-coverage",
        action="store_true",
        help="Use the union of SED sources across passes; each source/filter needs 2 usable epochs.",
    )
    parser.add_argument("--basis", choices=("per-gap", "coarse"), default="per-gap")
    parser.add_argument("--amplitude-bound", type=float, default=0.30)
    parser.add_argument("--phase-bound-rad", type=float, default=1.0)
    parser.add_argument(
        "--maxent-weight",
        type=float,
        default=1000.0,
        help="Relative-entropy MaxEnt weight on clean-reflection log amplitude.",
    )
    parser.add_argument(
        "--phase-smoothness-weight",
        type=float,
        default=100.0,
        help="Existing second-difference penalty weight for phase modulation.",
    )
    parser.add_argument(
        "--fit-systematic-floor",
        action="store_true",
        help="Profile a Gaussian sigma_sys added in quadrature to catalog errors.",
    )
    parser.add_argument(
        "--systematic-floor-max-mmag",
        type=float,
        default=100.0,
        help="Upper bound for the profiled systematic floor in millimagnitudes.",
    )
    parser.add_argument("--wavelength-stride-nm", type=float, default=5.0)
    parser.add_argument(
        "--include-dq-flagged",
        action="store_true",
        help="Keep all measurements and sources regardless of DQ flags.",
    )
    parser.add_argument(
        "--exclude-measurement",
        action="append",
        nargs=3,
        metavar=("SOURCE_ID", "FILTER", "PASS"),
        help="Manually omit one catalog row; repeat for sensitivity checks.",
    )
    parser.add_argument(
        "--keep-unflagged-measurements-from-saturated-sources",
        action="store_true",
        help="Use unflagged rows from saturated sources instead of dropping those sources entirely.",
    )
    parser.add_argument(
        "--exclude-repeat-outliers",
        action="store_true",
        help=(
            "Exclude isolated source/filter/pass measurements that differ from their "
            "same-roll median by at least 0.25 mag and 8 combined robust/error sigmas "
            "(only groups with at least four measurements are screened)."
        ),
    )
    parser.add_argument(
        "--fit-per-tile-modulation-test",
        action="store_true",
        help=(
            "Run an independent tile-level modulation sensitivity test after the "
            "global fit; unsupported tiles are interpolated from neighboring fits."
        ),
    )
    parser.add_argument(
        "--per-tile-modulation-output-dir",
        type=Path,
        default=None,
        help="Output directory for the optional per-tile modulation test.",
    )
    args = parser.parse_args(argv)
    args.detector = args.detector.upper()
    if (
        len(args.detector) != 5
        or not args.detector.startswith("WFI")
        or not args.detector[3:].isdigit()
    ):
        parser.error("Detector must be an explicit WFI## name, e.g. WFI10")
    if args.optical_cache_root is None:
        if args.optical_cache is None:
            args.optical_cache = detector_optical_cache(args.detector, tile_id=136)
        args.optical_cache_root = args.optical_cache.parent.parent
    elif args.optical_cache is None:
        args.optical_cache = args.optical_cache_root / "tile_136/relative_response.npz"
    if args.ice_history_fit_root is None:
        args.ice_history_fit_root = detector_tile_fit_root(args.detector)
    if args.output_dir is None:
        args.output_dir = detector_default_output_dir(
            args.detector,
            args.pass_set,
            allow_partial_source_coverage=args.allow_partial_source_coverage,
        )
    if args.amplitude_bound <= 0.0 or args.phase_bound_rad <= 0.0:
        parser.error("Modulation bounds must be positive")
    if (
        not np.isfinite(args.maxent_weight)
        or args.maxent_weight < 0.0
        or not np.isfinite(args.phase_smoothness_weight)
        or args.phase_smoothness_weight < 0.0
    ):
        parser.error("Regularization weights must be finite and non-negative")
    if args.wavelength_stride_nm <= 0.0:
        parser.error("Wavelength stride must be positive")
    if args.fit_systematic_floor and (
        not np.isfinite(args.systematic_floor_max_mmag)
        or args.systematic_floor_max_mmag <= 0.0
    ):
        parser.error("Systematic-floor maximum must be finite and positive")

    pass_config = PASS_SET_CONFIG[args.pass_set]
    pass_names = pass_config["passes"]
    window_start_utc = pass_config["start"]
    window_end_utc = pass_config["end"]
    ice_reference_mjd = float(Time(ICE_REFERENCE_UTC, format="isot", scale="utc").mjd)
    (
        source_ids,
        sed_by_id,
        magnitudes,
        magnitude_errors,
        raw_measurements,
        valid_measurements,
        saturated_measurements,
        do_not_use_measurements,
        source_has_saturation,
        dq_flags,
        mjd_by_filter,
        x_pixels,
        y_pixels,
        tile_ids_by_measurement,
    ) = load_photometry_and_seds(
        args.photometry_dir,
        args.sed_posteriors,
        detector=args.detector,
        pass_names=pass_names,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        require_common_source=not args.allow_partial_source_coverage,
        exclude_dq_flagged=not args.include_dq_flagged,
        exclude_any_saturated_source=(
            not args.include_dq_flagged
            and not args.keep_unflagged_measurements_from_saturated_sources
        ),
    )
    dq_valid_measurements = valid_measurements.copy()
    manual_exclusion_mask = np.zeros_like(valid_measurements)
    manual_exclusions = []
    for source_id, filter_name, pass_name in args.exclude_measurement or ():
        if source_id not in source_ids:
            parser.error(f"Manual-exclusion source is not in the selected cohort: {source_id}")
        if filter_name not in FILTERS:
            parser.error(f"Unknown filter in manual exclusion: {filter_name}")
        if pass_name not in pass_names:
            parser.error(f"Unknown pass in manual exclusion: {pass_name}")
        star_index = source_ids.index(source_id)
        filter_index = FILTERS.index(filter_name)
        pass_index = pass_names.index(pass_name)
        manual_exclusion_mask[star_index, filter_index, pass_index] = True
        valid_measurements[star_index, filter_index, pass_index] = False
        manual_exclusions.append(
            {"source_id": source_id, "filter": filter_name, "pass": pass_name}
        )
    pre_outlier_valid_measurements = valid_measurements.copy()
    (
        outlier_candidate_mask,
        outlier_centered_from_median_mmag,
        outlier_candidate_records,
    ) = identify_repeat_photometry_outliers(
        source_ids,
        magnitudes,
        magnitude_errors,
        pre_outlier_valid_measurements,
        pass_config["zero_point_group_ids"],
        pass_names,
        tile_ids_by_measurement,
        x_pixels,
        y_pixels,
    )
    outlier_exclusion_mask = (
        outlier_candidate_mask.copy()
        if args.exclude_repeat_outliers
        else np.zeros_like(outlier_candidate_mask)
    )
    if args.exclude_repeat_outliers:
        valid_measurements[outlier_exclusion_mask] = False
    pre_outlier_repeated_groups = np.sum(
        pre_outlier_valid_measurements, axis=2
    ) >= 2
    pre_outlier_active_source_count = int(
        np.count_nonzero(np.any(pre_outlier_repeated_groups, axis=1))
    )
    source_excluded_for_saturation = (
        source_has_saturation
        & (not args.include_dq_flagged)
        & (not args.keep_unflagged_measurements_from_saturated_sources)
    )
    star_count = len(source_ids)
    repeated_source_filter = np.sum(valid_measurements, axis=2) >= 2
    likelihood_measurements = valid_measurements & repeated_source_filter[:, :, None]
    missing_tile_for_likelihood = likelihood_measurements & (tile_ids_by_measurement < 0)
    if np.any(missing_tile_for_likelihood):
        raise ValueError("A likelihood-used observation has no detector tile assignment")
    source_tile_ids_for_model = tile_ids_by_measurement.copy()
    source_tile_ids_for_model[source_tile_ids_for_model < 0] = 136
    active_source_mask = np.any(repeated_source_filter, axis=1)
    active_source_count = int(np.count_nonzero(active_source_mask))
    active_source_ids = [
        source_id for source_id, active in zip(source_ids, active_source_mask) if active
    ]
    filter_count = len(FILTERS)
    pass_count = len(pass_names)

    wavelengths_nm, area_m2 = mock_model.load_effective_area(args.effective_area)
    wavelength_grid_nm = np.arange(
        400.0, 2321.0 + 0.5 * args.wavelength_stride_nm, args.wavelength_stride_nm
    )
    area_m2 = np.asarray(
        [np.interp(wavelength_grid_nm, wavelengths_nm, band) for band in area_m2]
    )
    wavelengths_nm = wavelength_grid_nm
    supported = np.any(area_m2 > 0.0, axis=0)
    wavelengths_nm = wavelengths_nm[supported]
    area_m2 = area_m2[:, supported]
    seds, sed_parameters, sed_fallback_count = reconstruct_sed_photon_shapes(
        source_ids, sed_by_id, args.sed_grid, wavelengths_nm
    )
    active_sed_parameters = sed_parameters[active_source_mask]

    target_days_by_filter = mjd_by_filter - ice_reference_mjd
    tile_paths = sorted(
        Path(args.ice_history_fit_root).glob("tile_*/posterior_samples.npz")
    )
    if not tile_paths:
        raise FileNotFoundError(
            f"No fixed {args.detector} per-tile histories under {args.ice_history_fit_root}"
        )
    tile_ids = []
    tile_days_by_id = [None] * (TILE_GRID_SIZE**2)
    tile_curves_by_id = [None] * (TILE_GRID_SIZE**2)
    tile_resets = []
    for path in tile_paths:
        tile_id = int(path.parent.name.split("_")[1])
        if tile_id < 0 or tile_id >= TILE_GRID_SIZE**2 or tile_id in tile_ids:
            raise ValueError(f"Invalid or repeated ice-history tile ID {tile_id}: {path}")
        with np.load(path, allow_pickle=False) as archive:
            days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
            thickness = np.asarray(archive["tile_ice_thickness"], dtype=float)
            thickness = thickness.reshape((-1, days.size))[0]
            reset = np.asarray(
                archive.get("ice_reset_days", archive.get("ice_reset_day", [])),
                dtype=float,
            ).reshape(-1)
        tile_ids.append(tile_id)
        tile_days_by_id[tile_id] = days
        tile_curves_by_id[tile_id] = thickness
        tile_resets.append(np.sort(reset))
    if sorted(tile_ids) != list(range(TILE_GRID_SIZE**2)):
        raise ValueError(
            f"Expected all {TILE_GRID_SIZE**2} {args.detector} ice histories; "
            f"found {len(tile_ids)}"
        )
    reset_days = tile_resets[0]
    if any(not np.allclose(item, reset_days, rtol=0.0, atol=1.0e-8) for item in tile_resets[1:]):
        raise ValueError(f"{args.detector} tile histories disagree on fixed reset days")
    target_flat = target_days_by_filter.reshape(-1)
    thickness_by_tile = np.asarray(
        [
            mock_model.interpolate_fixed_history(
                tile_days_by_id[tile_id],
                tile_curves_by_id[tile_id],
                reset_days,
                target_flat,
            )
            for tile_id in range(TILE_GRID_SIZE**2)
        ]
    ).reshape((TILE_GRID_SIZE**2, filter_count, pass_count))
    tile_count_available = int(thickness_by_tile.shape[0])
    tile_thickness_percentiles = np.percentile(
        thickness_by_tile, [16.0, 50.0, 84.0], axis=0
    )
    filter_indices = np.broadcast_to(
        np.arange(filter_count, dtype=int)[None, :, None],
        source_tile_ids_for_model.shape,
    )
    pass_indices = np.broadcast_to(
        np.arange(pass_count, dtype=int)[None, None, :],
        source_tile_ids_for_model.shape,
    )
    source_tile_thickness = thickness_by_tile[
        source_tile_ids_for_model, filter_indices, pass_indices
    ]

    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths_nm)
    spectral_basis = production.spectral_modulation_basis(wavelengths_nm)
    tile_clean_reflection = np.empty(
        (tile_count_available, wavelengths_nm.size), dtype=complex
    )
    tile_clean_transfer_first = np.empty_like(tile_clean_reflection)
    tile_clean_power = np.empty((tile_count_available, wavelengths_nm.size), dtype=float)
    tile_incident_admittance = np.empty(tile_count_available, dtype=float)
    tile_substrate_admittance = np.empty_like(tile_clean_reflection)
    tile_incidence_sine = np.empty(tile_count_available, dtype=float)
    tile_ice_index = np.empty(tile_count_available, dtype=float)
    source_fit_by_tile = {}
    for tile_id in range(tile_count_available):
        optical_cache = (
            args.optical_cache_root
            / f"tile_{tile_id:03d}/relative_response.npz"
        )
        source_fit_by_tile[tile_id] = mock_model.source_fit_from_cache(
            optical_cache, args.detector, tile_id
        )
        state = complex_model.load_state(
            source_fit_by_tile[tile_id],
            wavelengths_nm,
            base_n,
            base_k,
            spectral_basis,
        )
        clean_first, clean_second, incident_y, substrate_y, incidence_sine = (
            mock_model.clean_stack_boundary(state, wavelengths_nm)
        )
        clean_reflection = (incident_y * clean_first - clean_second) / (
            incident_y * clean_first + clean_second
        )
        clean_power = (
            np.real(substrate_y / incident_y)
            * np.abs(2.0 * incident_y / (incident_y * clean_first + clean_second)) ** 2
        )
        tile_clean_reflection[tile_id] = clean_reflection
        tile_clean_transfer_first[tile_id] = clean_first
        tile_clean_power[tile_id] = clean_power
        tile_incident_admittance[tile_id] = incident_y
        tile_substrate_admittance[tile_id] = substrate_y
        tile_incidence_sine[tile_id] = incidence_sine
        tile_ice_index[tile_id] = state["ice_index"]

    # Keep tile 136 as the named reference state for the optical-only panel;
    # all photometric likelihood rows use their own mapped tile state.
    source_fit = source_fit_by_tile[136]
    clean_reflection = tile_clean_reflection[136]
    basis, names, centers, _, _, _ = mock_model.modulation_basis(
        wavelengths_nm, grouping=args.basis
    )
    model = RealPhotometryFit(
        wavelengths_nm,
        area_m2,
        seds,
        thickness_by_tile,
        source_tile_ids_for_model,
        tile_clean_reflection,
        tile_clean_transfer_first,
        tile_clean_power,
        tile_incident_admittance,
        tile_substrate_admittance,
        tile_incidence_sine,
        tile_ice_index,
        args.basis,
        entropy_weight=args.maxent_weight,
        phase_smoothness_weight=args.phase_smoothness_weight,
        pass_zero_point_group_ids=pass_config["zero_point_group_ids"],
    )
    model.active_source_count = active_source_count
    model.detector = args.detector
    expected_data_modes = len(names)
    if expected_data_modes != model.mode_count:
        raise RuntimeError("Modulation basis construction is inconsistent")

    if args.fit_systematic_floor:
        (
            fit,
            data_chi2,
            observed_log_flux,
            sigma_log_flux,
            valid_measurements,
            observations_used,
            source_filter_offsets,
            systematic_floor_mmag,
            systematic_profile_nll,
        ) = fit_actual_photometry_with_profiled_systematic_floor(
            model,
            magnitudes,
            magnitude_errors,
            valid_measurements,
            args.amplitude_bound,
            args.phase_bound_rad,
            args.systematic_floor_max_mmag,
        )
    else:
        (
            fit,
            data_chi2,
            observed_log_flux,
            sigma_log_flux,
            valid_measurements,
            observations_used,
            source_filter_offsets,
        ) = fit_actual_photometry(
            model,
            magnitudes,
            magnitude_errors,
            valid_measurements,
            args.amplitude_bound,
            args.phase_bound_rad,
        )
        systematic_floor_mmag = 0.0
        systematic_profile_nll = None
    zero_parameters, zero_chi2 = profile_zero_modulation(
        model, observed_log_flux, sigma_log_flux, valid_measurements
    )
    fit_modulation = fit.x[: 2 * model.mode_count]
    maxent_residuals = model.maxent_regularization_residuals(fit_modulation)
    phase_residuals = model.phase_smoothness_residuals(fit_modulation)
    maxent_chi2_by_mode = (
        np.square(maxent_residuals)
        if maxent_residuals.size
        else np.zeros(model.mode_count, dtype=float)
    )
    maxent_chi2 = float(np.sum(maxent_chi2_by_mode))
    phase_smoothness_chi2 = float(np.sum(phase_residuals**2))
    fit_log_response = model.modeled_log_response(fit_modulation)
    fit_zpt = fit.x[2 * model.mode_count :].reshape(
        (filter_count, model.zero_point_group_count)
    )
    for pass_index, group_id in enumerate(model.pass_zero_point_group_ids):
        if group_id > 0:
            fit_log_response[:, :, pass_index] += fit_zpt[:, group_id - 1][None, :]
    zero_modulation = zero_parameters[: 2 * model.mode_count]
    zero_log_response = model.modeled_log_response(zero_modulation)
    zero_zpt = zero_parameters[2 * model.mode_count :].reshape(
        (filter_count, model.zero_point_group_count)
    )
    for pass_index, group_id in enumerate(model.pass_zero_point_group_ids):
        if group_id > 0:
            zero_log_response[:, :, pass_index] += zero_zpt[:, group_id - 1][None, :]
    fit_delta_mag = -fit_log_response / LOG_FLUX_PER_MAG
    zero_delta_mag = -zero_log_response / LOG_FLUX_PER_MAG
    observed_delta_mag = magnitudes - magnitudes[:, :, :1]
    likelihood_measurements = valid_measurements & (
        np.sum(valid_measurements, axis=2) >= 2
    )[:, :, None]
    group_centered_observed_mmag = group_centered_magnitude_series(
        magnitudes,
        magnitude_errors,
        likelihood_measurements,
        model.pass_zero_point_group_ids,
    )
    group_centered_fit_mmag = group_centered_magnitude_series(
        fit_delta_mag,
        magnitude_errors,
        likelihood_measurements,
        model.pass_zero_point_group_ids,
    )
    group_centered_zero_mmag = group_centered_magnitude_series(
        zero_delta_mag,
        magnitude_errors,
        likelihood_measurements,
        model.pass_zero_point_group_ids,
    )
    per_tile_modulation_test_summary = None
    if args.fit_per_tile_modulation_test:
        per_tile_output_dir = args.per_tile_modulation_output_dir or (
            args.output_dir / "per_tile_modulation_test"
        )
        per_tile_modulation_test_summary = run_per_tile_modulation_experiment(
            per_tile_output_dir,
            args.detector,
            model,
            fit,
            fit_log_response,
            observed_log_flux,
            sigma_log_flux,
            magnitudes,
            magnitude_errors,
            valid_measurements,
            tile_ids_by_measurement,
            args.amplitude_bound,
            args.phase_bound_rad,
            systematic_floor_mmag=systematic_floor_mmag,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outlier_candidates_csv = (
        args.output_dir / "car201_actual_star_outlier_candidates.csv"
    )
    outlier_candidate_frame = pd.DataFrame(outlier_candidate_records)
    if not outlier_candidate_frame.empty:
        outlier_candidate_frame = outlier_candidate_frame.sort_values(
            "deviation_mmag", ascending=False
        )
    outlier_candidate_frame.to_csv(outlier_candidates_csv, index=False)

    outlier_diagnostic_path = (
        args.output_dir / "car201_actual_star_outlier_diagnostic.png"
    )
    plot_repeat_photometry_outliers(
        outlier_diagnostic_path,
        args.detector,
        target_days_by_filter,
        pre_outlier_valid_measurements
        & (np.sum(pre_outlier_valid_measurements, axis=2) >= 2)[:, :, None],
        outlier_centered_from_median_mmag,
        outlier_candidate_mask,
        pre_outlier_active_source_count,
    )

    source_fit_csv = args.output_dir / "car201_actual_star_modulation_parameters.csv"
    with source_fit_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "center_nm",
                "fit_log_amplitude",
                "fit_amplitude_percent",
                "fit_phase_rad",
                "maxent_relative_entropy_contribution",
            ]
        )
        for index, name in enumerate(model.basis_names):
            writer.writerow(
                [
                    name,
                    model.basis_centers_nm[index],
                    fit_modulation[index],
                    100.0 * np.expm1(fit_modulation[index]),
                    fit_modulation[model.mode_count + index],
                    maxent_chi2_by_mode[index],
                ]
            )

    roll_zero_points_csv = args.output_dir / "car201_actual_roll_zero_points.csv"
    with roll_zero_points_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filter",
                "offset_group",
                "passes_in_group",
                "fit_log_flux_offset",
                "equivalent_magnitude_offset_mmag",
            ]
        )
        for filter_index, filter_name in enumerate(FILTERS):
            for group_id in range(1, model.zero_point_group_count + 1):
                log_flux_offset = float(fit_zpt[filter_index, group_id - 1])
                writer.writerow(
                    [
                        filter_name,
                        pass_config["zero_point_group_names"][group_id],
                        ";".join(
                            pass_names[index]
                            for index, local_group in enumerate(
                                model.pass_zero_point_group_ids
                            )
                            if local_group == group_id
                        ),
                        log_flux_offset,
                        -log_flux_offset / LOG_FLUX_PER_MAG * 1000.0,
                    ]
                )

    measurements_csv = args.output_dir / "car201_actual_star_modulation_measurements.csv"
    with measurements_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_id",
                "filter",
                "pass",
                "x_pixel",
                "y_pixel",
                "tile_id",
                "tile_ice_thickness_used_nm",
                "day_from_fixed_ice_reference",
                "median_sca_ice_thickness_nm",
                "tile_ice_p16_nm",
                "tile_ice_p84_nm",
                "mag_obs",
                "mag_err",
                "sigma_total_mmag_used",
                "dq_flag",
                "photometry_row_valid",
                "likelihood_used",
                "repeat_outlier_candidate",
                "repeat_outlier_excluded",
                "manual_measurement_excluded",
                "dq_saturated_bit",
                "dq_do_not_use_bit",
                "source_has_any_saturated_measurement",
                "source_excluded_for_saturation",
                "observed_delta_mag",
                "fitted_delta_mag_including_zeropoint",
                "zero_modulation_delta_mag_including_zeropoint",
                "sed_teff_k_median",
                "sed_av_mag_median",
                "sed_band_count",
                "sed_chi2",
            ]
        )
        for star_index, source_id in enumerate(source_ids):
            sed = sed_by_id[source_id]
            for filter_index, filter_name in enumerate(FILTERS):
                for pass_index, pass_name in enumerate(pass_names):
                    d_values = thickness_by_tile[:, filter_index, pass_index]
                    writer.writerow(
                        [
                            source_id,
                            filter_name,
                            pass_name,
                            x_pixels[star_index, filter_index, pass_index],
                            y_pixels[star_index, filter_index, pass_index],
                            int(tile_ids_by_measurement[star_index, filter_index, pass_index]),
                            source_tile_thickness[star_index, filter_index, pass_index],
                            target_days_by_filter[filter_index, pass_index],
                            tile_thickness_percentiles[1, filter_index, pass_index],
                            tile_thickness_percentiles[0, filter_index, pass_index],
                            tile_thickness_percentiles[2, filter_index, pass_index],
                            magnitudes[star_index, filter_index, pass_index],
                            magnitude_errors[star_index, filter_index, pass_index],
                            np.hypot(
                                magnitude_errors[star_index, filter_index, pass_index]
                                * 1000.0,
                                systematic_floor_mmag,
                            ),
                            dq_flags[star_index, filter_index, pass_index],
                            bool(valid_measurements[star_index, filter_index, pass_index]),
                            bool(likelihood_measurements[star_index, filter_index, pass_index]),
                            bool(outlier_candidate_mask[star_index, filter_index, pass_index]),
                            bool(outlier_exclusion_mask[star_index, filter_index, pass_index]),
                            bool(manual_exclusion_mask[star_index, filter_index, pass_index]),
                            bool(saturated_measurements[star_index, filter_index, pass_index]),
                            bool(do_not_use_measurements[star_index, filter_index, pass_index]),
                            bool(source_has_saturation[star_index]),
                            bool(source_excluded_for_saturation[star_index]),
                            observed_delta_mag[star_index, filter_index, pass_index],
                            fit_delta_mag[star_index, filter_index, pass_index],
                            zero_delta_mag[star_index, filter_index, pass_index],
                            sed["parameters_median"][0],
                            sed["parameters_median"][4],
                            sed["bands"],
                            sed["chi2"],
                        ]
                    )

    summary_rows = []
    for filter_index, filter_name in enumerate(FILTERS):
        for pass_index, pass_name in enumerate(pass_names):
            observed_values = observed_delta_mag[:, filter_index, pass_index]
            valid_star = (
                valid_measurements[:, filter_index, pass_index]
                & valid_measurements[:, filter_index, 0]
                & np.isfinite(observed_values)
            )
            obs = observed_values[valid_star] * 1000.0
            fit_values = fit_delta_mag[valid_star, filter_index, pass_index] * 1000.0
            zero_values = zero_delta_mag[valid_star, filter_index, pass_index] * 1000.0
            if obs.size:
                observed_median = float(np.median(obs))
                observed_p16, observed_p84 = np.percentile(obs, [16, 84])
                fit_median = float(np.median(fit_values))
                zero_median = float(np.median(zero_values))
            else:
                observed_median = observed_p16 = observed_p84 = float("nan")
                fit_median = zero_median = float("nan")
            summary_rows.append(
                {
                    "filter": filter_name,
                    "pass": pass_name,
                    "day_from_fixed_ice_reference": target_days_by_filter[
                        filter_index, pass_index
                    ],
                    "n_sources_with_valid_measurement": int(
                        np.count_nonzero(valid_measurements[:, filter_index, pass_index])
                    ),
                    "n_sources_with_likelihood_measurement": int(
                        np.count_nonzero(
                            likelihood_measurements[:, filter_index, pass_index]
                        )
                    ),
                    "n_sources_paired_to_first_pass": int(
                        np.count_nonzero(valid_star)
                    ),
                    "n_stars": int(np.count_nonzero(valid_star)),
                    "observed_median_delta_mmag": observed_median,
                    "observed_p16_delta_mmag": float(observed_p16),
                    "observed_p84_delta_mmag": float(observed_p84),
                    "fit_model_median_delta_mmag": fit_median,
                    "zero_modulation_median_delta_mmag": zero_median,
                }
            )
    summary_csv = args.output_dir / "car201_actual_star_modulation_filter_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    pass_summary_path = args.output_dir / "car201_actual_star_modulation_pass_summary.png"
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 10.5), sharex=True)
    post_pass_indices = np.flatnonzero(model.pass_zero_point_group_ids > 0)
    for filter_index, axis in enumerate(axes.flat):
        values = [row for row in summary_rows if row["filter"] == FILTERS[filter_index]]
        x = np.asarray([row["day_from_fixed_ice_reference"] for row in values])
        median = np.asarray([row["observed_median_delta_mmag"] for row in values])
        lower = median - np.asarray([row["observed_p16_delta_mmag"] for row in values])
        upper = np.asarray([row["observed_p84_delta_mmag"] for row in values]) - median
        fit_median = np.asarray([row["fit_model_median_delta_mmag"] for row in values])
        zero_median = np.asarray([row["zero_modulation_median_delta_mmag"] for row in values])
        axis.errorbar(
            x,
            median,
            yerr=np.vstack((lower, upper)),
            fmt="o",
            linestyle="none",
            markersize=4.5,
            color="#333333",
            ecolor="0.6",
            capsize=2,
            label="observed median; bars p16–p84",
        )
        axis.plot(x, fit_median, color="#d47730", marker="s", linewidth=1.2, label="fit")
        axis.plot(
            x,
            zero_median,
            color="0.15",
            linestyle="-.",
            marker="x",
            linewidth=1.0,
            label="zero modulation",
        )
        for post_index in post_pass_indices:
            selected = likelihood_measurements[:, filter_index, post_index]
            if not np.any(selected):
                continue
            day = target_days_by_filter[filter_index, post_index]
            post_values = group_centered_observed_mmag[
                selected, filter_index, post_index
            ]
            post_median = float(np.median(post_values))
            post_p16, post_p84 = np.percentile(post_values, [16.0, 84.0])
            axis.errorbar(
                [day],
                [post_median],
                yerr=np.asarray([[post_median - post_p16], [post_p84 - post_median]]),
                fmt="o",
                linestyle="none",
                markersize=4.5,
                color="#3478a8",
                ecolor="#3478a8",
                capsize=2,
                zorder=5,
                label=(
                    "post-roll median; p16–p84"
                    if filter_index == 0 and post_index == post_pass_indices[0]
                    else None
                ),
            )
            axis.scatter(
                [day],
                [float(np.median(group_centered_fit_mmag[selected, filter_index, post_index]))],
                marker="s",
                s=28,
                color="#3478a8",
                edgecolors="white",
                linewidths=0.5,
                zorder=6,
                label=(
                    "post-roll fit median"
                    if filter_index == 0 and post_index == post_pass_indices[0]
                    else None
                ),
            )
            axis.scatter(
                [day],
                [float(np.median(group_centered_zero_mmag[selected, filter_index, post_index]))],
                marker="x",
                s=27,
                color="#159487",
                zorder=6,
                label=(
                    "post-roll zero-modulation median"
                    if filter_index == 0 and post_index == post_pass_indices[0]
                    else None
                ),
            )
        axis.set_title(FILTERS[filter_index])
        axis.grid(True, alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Δ magnitude (mmag)")
    axes[-1, 0].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    axes[-1, 1].set_xlabel("Days from fixed-ice reference (2026-09-12)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        fontsize=7,
    )
    figure.suptitle(
        f"Actual CAR201 {args.detector} photometry, {active_source_count} active of "
        f"{star_count} SED-matched sources, "
        f"{pass_names[0]} through {pass_names[-1]}\n"
        f"Tile-specific fixed ice history; Δmag relative to {pass_names[0]}\n"
        "Δmag points require a valid first-pass reference; post-roll-only sources "
        "remain in the repeated-photometry fit",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.85))
    figure.savefig(pass_summary_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    dof = (
        observations_used
        - source_filter_offsets
        - fit.x.size
        - int(args.fit_systematic_floor)
    )
    high_cadence_path = args.output_dir / "car201_actual_star_modulation_high_cadence.png"
    plot_high_cadence_photometry_model(
        high_cadence_path,
        model,
        target_days_by_filter,
        group_centered_observed_mmag,
        group_centered_fit_mmag,
        group_centered_zero_mmag,
        likelihood_measurements,
        pass_config["zero_point_group_ids"],
        pass_config["zero_point_group_names"],
        reset_days,
        data_chi2,
        dof,
    )
    complex_reflection_path = (
        args.output_dir / "car201_actual_clean_complex_reflection_fit.png"
    )
    plot_clean_complex_reflection(
        complex_reflection_path,
        f"{args.detector} tile 136 reference",
        wavelengths_nm,
        clean_reflection,
        model.basis,
        fit_modulation[: model.mode_count],
        fit_modulation[model.mode_count :],
        data_chi2,
        dof,
        model.entropy_weight,
        model.phase_smoothness_weight,
    )
    modulation_ratio_path = (
        args.output_dir / "car201_actual_reflectance_modulation_ratio.png"
    )
    plot_clean_reflectance_modulation_ratio(
        modulation_ratio_path,
        args.detector,
        wavelengths_nm,
        model.basis,
        fit_modulation[: model.mode_count],
        fit_modulation[model.mode_count :],
        model.entropy_weight,
        model.phase_smoothness_weight,
    )
    source_tile_counts = np.asarray(
        [
            np.unique(
                source_tile_ids_for_model[source_index][
                    likelihood_measurements[source_index]
                ]
            ).size
            for source_index in range(star_count)
        ],
        dtype=int,
    )
    tile_ids_used = np.unique(source_tile_ids_for_model[likelihood_measurements])
    summary = {
        "detector": args.detector,
        "pass_set": args.pass_set,
        "sed_posteriors": str(args.sed_posteriors),
        "manual_measurement_exclusions": manual_exclusions,
        "source_coverage_policy": (
            "union across passes; retain sources with at least two usable epochs per filter"
            if args.allow_partial_source_coverage
            else "intersection across all selected passes"
        ),
        "date_window_utc": [window_start_utc, window_end_utc],
        "pass_files": [f"{name}_master.parquet" for name in pass_names],
        "matched_source_count": star_count,
        "active_source_count_with_usable_photometry": active_source_count,
        "repeat_photometry_outlier_screen": {
            "enabled_for_exclusion": bool(args.exclude_repeat_outliers),
            "candidate_measurement_count": int(len(outlier_candidate_records)),
            "candidate_source_count": int(
                len({row["source_id"] for row in outlier_candidate_records})
            ),
            "excluded_measurement_count": int(
                np.count_nonzero(outlier_exclusion_mask)
            ),
            "minimum_group_measurements": 4,
            "minimum_absolute_deviation_mag": 0.25,
            "minimum_measurement_significance": 4.0,
            "comparison": "same source, filter, roll group, and tile median",
            "uncertainty_scale": "catalog magnitude error only; group MAD is reported as context",
        },
        "per_tile_modulation_test": per_tile_modulation_test_summary,
        "all_successful_sed_matches_used": active_source_count == star_count,
        "sed_fit_quality": {
            "band_count_lt_5": int(
                sum(sed_by_id[sid]["bands"] < 5 for sid in active_source_ids)
            ),
            "chi2_gt_25": int(
                sum(sed_by_id[sid]["chi2"] > 25.0 for sid in active_source_ids)
            ),
            "posterior_median_teff_k": [
                float(np.percentile(active_sed_parameters[:, 0], 16)),
                float(np.median(active_sed_parameters[:, 0])),
                float(np.percentile(active_sed_parameters[:, 0], 84)),
            ],
            "nearest_atmosphere_grid_fallbacks": int(sed_fallback_count),
            "sed_draws_used": "posterior medians; SED posterior uncertainty not propagated",
        },
        "ice_history": {
            "fixed": True,
            "reference_utc": ICE_REFERENCE_UTC,
            "tile_history_archives_available": int(tile_count_available),
            "tile_histories_assigned_per_measurement": True,
            "ice_resets_days": reset_days.tolist(),
            "tile_thickness_distribution_p16_median_p84_nm": np.percentile(
                thickness_by_tile,
                [16.0, 50.0, 84.0],
                axis=0,
            ).tolist(),
            "per_measurement_tile_position_available": True,
            "sources_with_measurements_in_multiple_tiles": int(
                np.count_nonzero(source_tile_counts > 1)
            ),
            "tile_ids_used": tile_ids_used.astype(int).tolist(),
            "tile_assignment": {
                "coordinate_columns": ["x_pixel", "y_pixel"],
                "tile_size_pixels": TILE_SIZE_PIXELS,
                "grid_size": TILE_GRID_SIZE,
                "row_major_formula": "floor(y_pixel/256)*16 + floor(x_pixel/256)",
                "assignment_granularity": "each source/filter/pass measurement",
            },
            "optical_state_source": str(args.optical_cache_root),
            "optical_tile_state_count": int(len(source_fit_by_tile)),
            "optical_spectrum_reference_tile": 136,
        },
        "data_quality": {
            "dq_flagged_measurements_excluded": not args.include_dq_flagged,
            "exclude_entire_source_if_any_measurement_saturated": bool(
                np.any(source_excluded_for_saturation)
                or (
                    not args.include_dq_flagged
                    and not args.keep_unflagged_measurements_from_saturated_sources
                )
            ),
            "source_count_with_any_saturated_measurement": int(
                np.count_nonzero(source_has_saturation)
            ),
            "source_count_excluded_for_saturation": int(
                np.count_nonzero(source_excluded_for_saturation)
            ),
            "source_count_retained": active_source_count,
            "rows_from_sources_excluded_for_saturation": int(
                np.count_nonzero(
                    raw_measurements & source_excluded_for_saturation[:, None, None]
                )
            ),
            "manual_measurements_excluded": int(
                np.count_nonzero(manual_exclusion_mask & raw_measurements)
            ),
            "repeat_outlier_measurements_excluded": int(
                np.count_nonzero(outlier_exclusion_mask)
            ),
            "dq_saturated_bit_value": int(DQ_SATURATED),
            "dq_do_not_use_bit_value": int(DQ_DO_NOT_USE),
            "finite_photometry_rows_before_dq_filter": int(
                np.count_nonzero(raw_measurements)
            ),
            "saturated_rows_flagged": int(np.count_nonzero(saturated_measurements)),
            "do_not_use_rows_flagged": int(np.count_nonzero(do_not_use_measurements)),
            "rows_flagged_both_saturated_and_do_not_use": int(
                np.count_nonzero(saturated_measurements & do_not_use_measurements)
            ),
            "rows_used_in_likelihood": int(observations_used),
            "rows_excluded_by_quality_filter": int(
                np.count_nonzero(raw_measurements)
                - np.count_nonzero(dq_valid_measurements)
            ),
            "quality_valid_rows_not_used_for_single_epoch_groups": int(
                np.count_nonzero(valid_measurements)
                - np.count_nonzero(likelihood_measurements)
            ),
            "dq_zero_fraction_before_filter": float(
                np.mean(dq_flags[raw_measurements] == 0)
            ),
            "median_mag_error_before_filter_mmag": float(
                np.nanmedian(magnitude_errors[raw_measurements]) * 1000.0
            ),
            "median_mag_error_used_mmag": float(
                np.nanmedian(magnitude_errors[likelihood_measurements]) * 1000.0
            ),
            "error_model": "sigma_total^2 = sigma_stat^2 + sigma_sys^2",
            "systematic_floor_fitted": bool(args.fit_systematic_floor),
            "systematic_floor_mmag": float(systematic_floor_mmag),
            "systematic_floor_profile_nll_up_to_constant": (
                float(systematic_profile_nll)
                if systematic_profile_nll is not None
                else None
            ),
            "median_total_error_used_mmag": float(
                np.nanmedian(
                    np.hypot(
                        magnitude_errors[valid_measurements] * 1000.0,
                        systematic_floor_mmag,
                    )
                )
            ),
            "dq_note": (
                "Roman DQ bitmask; sources with any SATURATED=2 measurement are "
                "excluded entirely by default, DO_NOT_USE=1 rows are excluded, and "
                "other DQ bits are retained."
            ),
        },
        "modulation": {
            "basis": args.basis,
            "wavelength_stride_nm": args.wavelength_stride_nm,
            "zeropoint_model": {
                "scope": "one offset per filter per roll group; shared across all passes in the group",
                "group_names": list(pass_config["zero_point_group_names"]),
                "pass_group_ids": model.pass_zero_point_group_ids.tolist(),
                "fitted_log_flux_offsets": fit_zpt.tolist(),
            },
            "maxent_regularization": {
                "type": "wavelength-weighted relative entropy",
                "reference_amplitude_multiplier": 1.0,
                "weight": model.entropy_weight,
                "penalty_chi2": maxent_chi2,
                "penalty_by_mode": maxent_chi2_by_mode.tolist(),
                "objective": "lambda * sum_j(node_weight_j * 2*KL(exp(a_j) || 1))",
            },
            "phase_smoothness_regularization": {
                "type": "second difference within each LED-free interval",
                "weight": model.phase_smoothness_weight,
                "penalty_chi2": phase_smoothness_chi2,
            },
            "mode_names": model.basis_names,
            "fitted_log_amplitudes": fit_modulation[: model.mode_count].tolist(),
            "fitted_amplitude_fractions": np.expm1(
                fit_modulation[: model.mode_count]
            ).tolist(),
            "fitted_phase_rad": fit_modulation[model.mode_count :].tolist(),
            "zero_modulation_chi2": float(zero_chi2),
            "modulated_chi2": data_chi2,
            "delta_chi2_zero_minus_fit": float(zero_chi2 - data_chi2),
            "penalized_objective": (
                data_chi2 + maxent_chi2 + phase_smoothness_chi2
            ),
        },
        "fit": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "chi2": data_chi2,
            "maxent_penalty_chi2": maxent_chi2,
            "phase_smoothness_penalty_chi2": phase_smoothness_chi2,
            "penalized_objective_chi2": (
                data_chi2 + maxent_chi2 + phase_smoothness_chi2
            ),
            "degrees_of_freedom_approx": int(dof),
            "reduced_chi2_approx": float(data_chi2 / dof),
            "parameter_count": int(fit.x.size),
            "systematic_floor_parameter_count": int(args.fit_systematic_floor),
            "total_parameter_count": int(fit.x.size) + int(args.fit_systematic_floor),
            "jacobian_rank": int(np.linalg.matrix_rank(fit.jac)),
            "condition_number": float(np.linalg.cond(fit.jac)),
        },
        "products": {
            "high_cadence_data_vs_fit_plot": str(high_cadence_path),
            "fitted_complex_reflection_plot": str(complex_reflection_path),
            "clean_reflectance_modulation_ratio_plot": str(modulation_ratio_path),
            "complex_reflection_plot_contents": (
                "Clean TMM baseline and fitted clean reflection only; no ice-coated curves."
            ),
            "pass_summary_plot": str(pass_summary_path),
            "repeat_outlier_diagnostic_plot": str(outlier_diagnostic_path),
            "parameters_csv": str(source_fit_csv),
            "roll_zero_points_csv": str(roll_zero_points_csv),
            "star_measurements_csv": str(measurements_csv),
            "filter_summary_csv": str(summary_csv),
            "repeat_outlier_candidates_csv": str(outlier_candidates_csv),
            "tile_matched_optical_cache_root": str(args.optical_cache_root),
        },
        "caveat": (
            "SCA-level exploratory fit with per-measurement pixel-to-tile assignment and "
            "tile-specific complex TMM/ice histories; one wavelength-dependent modulation "
            "is shared globally. Tile 136 is used only for the optical spectrum diagnostic panel."
        ),
    }
    summary_path = args.output_dir / "car201_actual_star_modulation_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for path in (
        high_cadence_path,
        complex_reflection_path,
        modulation_ratio_path,
        pass_summary_path,
        outlier_diagnostic_path,
        outlier_candidates_csv,
        source_fit_csv,
        measurements_csv,
        summary_csv,
        summary_path,
    ):
        print(f"Wrote {path}")
    print(
        f"Sources={active_source_count}/{star_count}, "
        f"repeat outlier candidates={len(outlier_candidate_records)}, "
        f"excluded={int(np.count_nonzero(outlier_exclusion_mask))}, "
        f"dχ²(no modulation−fit)={zero_chi2-data_chi2:.1f}, "
        f"reduced χ²={data_chi2/dof:.2f}, MaxEnt penalty={maxent_chi2:.2f} "
        f"(λ={model.entropy_weight:g}), phase penalty={phase_smoothness_chi2:.2f} "
        f"(λ={model.phase_smoothness_weight:g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
