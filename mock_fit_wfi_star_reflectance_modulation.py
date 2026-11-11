"""Mock sensitivity test for clean-reflectance modulation using a G star.

The ice-thickness history is read from a production fit and held fixed. A
shared, ice-independent complex modulation is applied to the clean reflection
coefficient, then the fixed ice layer is propagated over it with the normal-
incidence transfer matrix. Synthetic broadband stellar counts use Roman system
effective area exactly once (no separate QE factor).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production


ROOT = Path(__file__).resolve().parent
DEFAULT_ICE_HISTORY = (
    ROOT
    / "20261105/corrected_statistics_4h_thinned/production/wfi16/"
    "spline_relative_response_normal/fits/tile_136/posterior_samples.npz"
)
DEFAULT_OPTICAL_CACHE = (
    ROOT
    / "20261020/corrected_statistics_4h_thinned/production/relative_response_cache/"
    "tile_size_256/sensitivity_hgcdte_n_sigma_0p25/wfi16_400nm/tile_136/"
    "relative_response.npz"
)
DEFAULT_EFFECTIVE_AREA = (
    ROOT / "external_reference_data/Roman_effarea_v8_SCA05_20240301.ecsv"
)
FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213")
LED_CENTERS_NM = np.asarray(
    [
        623.520885362797,
        852.802649193115,
        1038.5847494059,
        1269.7582849979,
        1481.24145201185,
        1670.16888078615,
    ],
    dtype=float,
)
ROMAN_AB_ZEROPOINT_JY = 3631.0
SPEED_OF_LIGHT_M_S = 299_792_458.0
PLANCK_J_S = 6.62607015e-34
BOLTZMANN_J_K = 1.380649e-23
LOG_FLUX_PER_MAG = np.log(10.0) / 2.5


def millimagnitude_to_log_flux_sigma(mmag):
    return LOG_FLUX_PER_MAG * float(mmag) / 1000.0


def read_ecsv_rows(path):
    with Path(path).open() as handle:
        return list(
            csv.DictReader(
                (
                    line
                    for line in handle
                    if line.strip() and not line.lstrip().startswith("#")
                ),
                skipinitialspace=True,
            )
        )


def load_effective_area(path):
    rows = read_ecsv_rows(path)
    missing = [name for name in ("Wave", *FILTERS) if name not in rows[0]]
    if missing:
        raise ValueError(f"Effective-area table is missing {missing}")
    wavelengths_nm = np.asarray([float(row["Wave"]) * 1000.0 for row in rows])
    areas_m2 = np.asarray(
        [[float(row[name]) for name in FILTERS] for row in rows], dtype=float
    ).T
    order = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[order]
    areas_m2 = areas_m2[:, order]
    if np.any(np.diff(wavelengths_nm) <= 0.0) or np.any(areas_m2 < 0.0):
        raise ValueError("Effective-area grid or values are invalid")
    return wavelengths_nm, areas_m2


def load_fixed_ice_history(path, detector, tile_id):
    with np.load(path, allow_pickle=False) as archive:
        tile_ids = np.asarray(archive["tile_ids"], dtype=int).reshape(-1)
        detectors = np.asarray(archive["sca_names"]).astype(str).reshape(-1)
        if tile_id not in tile_ids:
            raise ValueError(f"Ice history has no tile {tile_id}: {path}")
        tile_index = int(np.flatnonzero(tile_ids == tile_id)[0])
        if detectors[tile_index].upper() != detector.upper():
            raise ValueError(
                f"Ice history tile {tile_id} is {detectors[tile_index]}, "
                f"not {detector}"
            )
        days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
        thickness = np.asarray(archive["tile_ice_thickness"], dtype=float)
        thickness = thickness.reshape((tile_ids.size, days.size))[tile_index]
        reset_days = np.asarray(
            archive.get("ice_reset_days", archive.get("ice_reset_day", [])),
            dtype=float,
        ).reshape(-1)
    order = np.argsort(days)
    days = days[order]
    thickness = thickness[order]
    if days.size < 2 or np.any(np.diff(days) <= 0.0):
        raise ValueError("Ice-history days must be strictly increasing")
    return days, thickness, np.sort(reset_days)


def interpolate_fixed_history(source_days, source_thickness, reset_days, target_days):
    """Interpolate within decon segments, never across an ice reset."""
    target_days = np.asarray(target_days, dtype=float)
    result = np.empty_like(target_days)
    segment_ids = np.searchsorted(reset_days, target_days, side="right")
    source_segment_ids = np.searchsorted(reset_days, source_days, side="right")
    for segment in range(reset_days.size + 1):
        target_mask = segment_ids == segment
        source_mask = source_segment_ids == segment
        if not np.any(target_mask):
            continue
        if not np.any(source_mask):
            raise ValueError(f"No fixed-history samples in reset segment {segment}")
        result[target_mask] = np.interp(
            target_days[target_mask],
            source_days[source_mask],
            source_thickness[source_mask],
        )
    return result


def source_fit_from_cache(path, detector, tile_id):
    with np.load(path, allow_pickle=False) as archive:
        cache_detector = str(np.asarray(archive["detector"]).reshape(-1)[0])
        cache_tile = int(np.asarray(archive["tile_id"]).reshape(-1)[0])
        source_fit = Path(str(np.asarray(archive["source_fit"]).reshape(-1)[0]))
    if cache_detector.upper() != detector.upper() or cache_tile != tile_id:
        raise ValueError("The optical cache detector/tile does not match this test")
    if not source_fit.is_file():
        raise FileNotFoundError(f"Optical-state source fit is missing: {source_fit}")
    return source_fit


def blackbody_photon_flux_per_nm(wavelengths_nm, temperature_k, ab_magnitude):
    """Planck spectrum normalized to an AB magnitude at 550 nm."""
    wavelengths_m = np.asarray(wavelengths_nm, dtype=float) * 1.0e-9
    exponent = PLANCK_J_S * SPEED_OF_LIGHT_M_S / (
        wavelengths_m * BOLTZMANN_J_K * temperature_k
    )
    shape = 1.0 / (wavelengths_m**5 * np.expm1(exponent))
    reference_m = 550.0e-9
    reference_exponent = PLANCK_J_S * SPEED_OF_LIGHT_M_S / (
        reference_m * BOLTZMANN_J_K * temperature_k
    )
    reference_shape = 1.0 / (reference_m**5 * np.expm1(reference_exponent))

    f_nu = ROMAN_AB_ZEROPOINT_JY * 1.0e-26 * 10.0 ** (-0.4 * ab_magnitude)
    f_lambda_per_nm_at_reference = (
        f_nu * SPEED_OF_LIGHT_M_S / reference_m**2 * 1.0e-9
    )
    f_lambda_per_nm = (
        f_lambda_per_nm_at_reference * shape / reference_shape
    )
    return f_lambda_per_nm * wavelengths_m / (PLANCK_J_S * SPEED_OF_LIGHT_M_S)


def modulation_basis(
    wavelengths_nm, grouping="per-gap", fine_node_spacing_nm=100.0
):
    """Build modulation modes that are exactly zero at each LED centre."""
    edges = np.concatenate(([400.0], LED_CENTERS_NM, [2321.0]))
    basis = []
    names = []
    centers = []
    node_groups = []
    node_weights = []
    group_bounds = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if grouping == "fine":
            segment_count = max(
                2, int(np.ceil((upper - lower) / fine_node_spacing_nm))
            )
            knots = np.linspace(lower, upper, segment_count + 1)
            group_indices = []
            for knot_index in range(1, knots.size - 1):
                node = knots[knot_index]
                values = np.zeros_like(wavelengths_nm)
                left, right = knots[knot_index - 1], knots[knot_index + 1]
                left_mask = (wavelengths_nm >= left) & (wavelengths_nm <= node)
                right_mask = (wavelengths_nm > node) & (wavelengths_nm <= right)
                values[left_mask] = (wavelengths_nm[left_mask] - left) / (node - left)
                values[right_mask] = (right - wavelengths_nm[right_mask]) / (right - node)
                group_indices.append(len(basis))
                basis.append(values)
                names.append(f"node_{node:.1f}nm")
                centers.append(node)
                node_weights.append(0.5 * (right - left))
            node_groups.append(np.asarray(group_indices, dtype=int))
            group_bounds.append((lower, upper))
        else:
            inside = (wavelengths_nm >= lower) & (wavelengths_nm <= upper)
            values = np.zeros_like(wavelengths_nm)
            phase = np.pi * (wavelengths_nm[inside] - lower) / (upper - lower)
            values[inside] = np.sin(phase) ** 2
            gap_index = len(basis)
            basis.append(values)
            names.append(f"gap_{lower:.0f}_{upper:.0f}nm")
            centers.append(0.5 * (lower + upper))
            node_weights.append(upper - lower)
            node_groups.append(np.asarray([gap_index], dtype=int))
            group_bounds.append((lower, upper))
    basis = np.asarray(basis)
    centers = np.asarray(centers)
    if grouping == "per-gap":
        pass
    elif grouping == "coarse":
        groups = ((0, 1), (2, 3, 4), (5, 6))
        grouped_basis = np.asarray([np.sum(basis[list(group)], axis=0) for group in groups])
        grouped_names = (
            "short_400_853nm",
            "middle_853_1481nm",
            "long_1481_2321nm",
        )
        grouped_centers = np.asarray([626.0, 1154.0, 1785.0])
        basis = grouped_basis
        names = list(grouped_names)
        centers = grouped_centers
        node_groups = [np.asarray([index]) for index in range(len(groups))]
        group_bounds = [(400.0, 852.8), (852.8, 1481.2), (1481.2, 2321.0)]
        node_weights = [upper - lower for lower, upper in group_bounds]
    elif grouping != "fine":
        raise ValueError(f"Unknown modulation-basis grouping: {grouping}")
    node_weights = np.asarray(node_weights, dtype=float)
    node_weights /= np.sum(node_weights)
    return basis, names, centers, node_groups, node_weights, group_bounds


def clean_stack_boundary(state, wavelengths_nm):
    """Return the clean-stack transfer boundary and its s-polarized admittances."""
    incidence_sine = np.sin(np.deg2rad(state["incidence_angle_deg"]))
    incidence_cosine = np.sqrt(1.0 - incidence_sine**2)
    incident_admittance = incidence_cosine
    substrate = state["hgcdte_n"] + 1j * state["hgcdte_k"]
    substrate_cosine = np.sqrt(1.0 - (incidence_sine / substrate) ** 2)
    substrate_admittance = substrate * substrate_cosine

    first = np.ones_like(wavelengths_nm, dtype=complex)
    second = substrate_admittance
    for layer_index in range(production.AR_LAYERS - 1, -1, -1):
        elements = complex_model._layer_matrix(
            state["ar_index"][layer_index],
            100.0 * state["ar_parameters"][layer_index, 0],
            wavelengths_nm,
            incidence_sine,
            "s",
        )
        first, second = complex_model._apply_layer((first, second), elements)
    return first, second, incident_admittance, substrate_admittance, incidence_sine


def coated_substrate_power(
    clean_reflection,
    clean_transfer_first,
    wavelengths_nm,
    thickness_nm,
    ice_index,
    incident_admittance,
    substrate_admittance,
    incidence_sine,
):
    """TMM substrate power after a shared clean-reflection modulation.

    The complex clean reflection sets the effective input admittance. The
    underlying clean transfer normalization (the first transfer-boundary
    component) and terminal HgCdTe admittance are retained from the pinned
    clean TMM state; the fixed ice layer is then applied explicitly.
    """
    clean_input_admittance = incident_admittance * (1.0 - clean_reflection) / (
        1.0 + clean_reflection
    )
    boundary = (
        clean_transfer_first,
        clean_transfer_first * clean_input_admittance,
    )
    ice_elements = complex_model._layer_matrix(
        ice_index,
        thickness_nm,
        wavelengths_nm,
        incidence_sine,
        "s",
    )
    first, second = complex_model._apply_layer(boundary, ice_elements)
    denominator = incident_admittance * first + second
    substrate_power = (
        np.real(substrate_admittance / incident_admittance)
        * np.abs(2.0 * incident_admittance / denominator) ** 2
    )
    reflected = (incident_admittance * first - second) / denominator
    return substrate_power, reflected


class StarCountModel:
    def __init__(
        self,
        wavelengths_nm,
        filter_weights,
        base_rates_per_second,
        clean_reflection,
        clean_transfer_first,
        clean_substrate_power,
        incident_admittance,
        substrate_admittance,
        incidence_sine,
        ice_index,
        history_thickness_nm,
        basis_grouping="per-gap",
        fine_node_spacing_nm=100.0,
        entropy_weight=0.0,
        phase_smoothness_weight=0.0,
    ):
        self.wavelengths_nm = wavelengths_nm
        self.filter_weights = filter_weights
        self.base_rates_per_second = base_rates_per_second
        self.clean_reflection = clean_reflection
        self.clean_transfer_first = clean_transfer_first
        self.clean_substrate_power = clean_substrate_power
        self.incident_admittance = incident_admittance
        self.substrate_admittance = substrate_admittance
        self.incidence_sine = incidence_sine
        self.ice_index = ice_index
        self.history_thickness_nm = history_thickness_nm
        self.basis_grouping = basis_grouping
        self.fine_node_spacing_nm = fine_node_spacing_nm
        self.entropy_weight = entropy_weight
        self.phase_smoothness_weight = phase_smoothness_weight
        (
            self.basis,
            self.basis_names,
            self.basis_centers_nm,
            self.node_groups,
            self.node_weights,
            self.group_bounds,
        ) = modulation_basis(
            wavelengths_nm,
            grouping=basis_grouping,
            fine_node_spacing_nm=fine_node_spacing_nm,
        )
        self.filter_count = len(FILTERS)
        self.mode_count = self.basis.shape[0]

    @property
    def parameter_count(self):
        return 2 * self.mode_count + self.filter_count

    def response_ratios(self, parameters):
        amp_coefficients = parameters[: self.mode_count]
        phase_coefficients = parameters[self.mode_count : 2 * self.mode_count]
        log_amplitude = amp_coefficients @ self.basis
        phase_modulation = phase_coefficients @ self.basis
        clean_modulated = self.clean_reflection * np.exp(
            log_amplitude + 1j * phase_modulation
        )
        coated_power = np.asarray(
            [
                coated_substrate_power(
                    clean_modulated,
                    self.clean_transfer_first,
                    self.wavelengths_nm,
                    thickness_nm,
                    self.ice_index,
                    self.incident_admittance,
                    self.substrate_admittance,
                    self.incidence_sine,
                )[0]
                for thickness_nm in self.history_thickness_nm
            ]
        )
        weighted_rate = np.trapezoid(
            (coated_power / self.clean_substrate_power[None, :])[:, None, :]
            * self.filter_weights[None, :, :],
            self.wavelengths_nm,
            axis=2,
        )
        return weighted_rate / self.base_rates_per_second[None, :]

    def expected_log_counts(self, parameters, exposure_s):
        modulation_parameters = parameters[: 2 * self.mode_count]
        filter_log_offsets = parameters[2 * self.mode_count :]
        ratios = self.response_ratios(modulation_parameters)
        log_counts = (
            np.log(exposure_s * self.base_rates_per_second[None, :])
            + np.log(ratios)
            + filter_log_offsets[None, :]
        )
        return log_counts

    @staticmethod
    def _signed_relative_entropy_residual(log_ratio):
        entropy = np.exp(log_ratio) * log_ratio - np.expm1(log_ratio)
        entropy = np.where(np.abs(log_ratio) < 1.0e-4, 0.5 * log_ratio**2, entropy)
        magnitude = np.sqrt(2.0 * np.maximum(entropy, 0.0))
        residual = np.sign(log_ratio) * magnitude
        return np.where(np.abs(log_ratio) < 1.0e-6, log_ratio, residual)

    def regularization_residuals(self, modulation_parameters):
        amplitude = modulation_parameters[: self.mode_count]
        phase = modulation_parameters[self.mode_count :]
        residuals = []
        if self.entropy_weight > 0.0:
            residuals.extend(
                np.sqrt(self.entropy_weight * self.node_weights)
                * self._signed_relative_entropy_residual(amplitude)
            )
        if self.phase_smoothness_weight > 0.0:
            for indices in self.node_groups:
                sequence = np.concatenate(([0.0], phase[indices], [0.0]))
                if sequence.size >= 3:
                    residuals.extend(
                        np.sqrt(self.phase_smoothness_weight)
                        * np.diff(sequence, n=2)
                    )
        return np.asarray(residuals, dtype=float)


def measurement_sigma(mean_counts, fractional_systematic):
    if np.any(mean_counts <= 0.0):
        raise ValueError("Expected stellar counts must be positive")
    return np.sqrt(1.0 / mean_counts + fractional_systematic**2)


def fit_synthetic_counts(model, truth, exposure_s, systematic, seed):
    truth_with_offsets = np.concatenate(
        (truth, np.zeros(model.filter_count, dtype=float))
    )
    expected_log_counts = model.expected_log_counts(truth_with_offsets, exposure_s)
    mean_counts = np.exp(expected_log_counts)
    sigma = measurement_sigma(mean_counts, systematic)
    rng = np.random.default_rng(seed)
    observed_log_counts = expected_log_counts + rng.normal(size=sigma.shape) * sigma

    def residual(parameters):
        data_residual = (
            (model.expected_log_counts(parameters, exposure_s) - observed_log_counts)
            / sigma
        ).reshape(-1)
        regularization_residual = model.regularization_residuals(
            parameters[: 2 * model.mode_count]
        )
        return np.concatenate((data_residual, regularization_residual))

    n = model.mode_count
    amplitude_limit = max(0.15, 1.5 * float(np.max(np.abs(truth[:n]))) + 0.02)
    phase_limit = max(0.5, 1.5 * float(np.max(np.abs(truth[n:]))) + 0.1)
    lower = np.concatenate(
        (
            np.full(n, -amplitude_limit),
            np.full(n, -phase_limit),
            np.full(model.filter_count, -2.0),
        )
    )
    upper = np.concatenate(
        (
            np.full(n, amplitude_limit),
            np.full(n, phase_limit),
            np.full(model.filter_count, 2.0),
        )
    )
    fit = least_squares(
        residual,
        np.zeros(model.parameter_count, dtype=float),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=300,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    return fit, observed_log_counts, expected_log_counts, mean_counts, sigma


def expected_fisher_errors(
    model, exposure_s, systematic, include_regularization=True
):
    """Compute local clean-modulation errors for a given count/noise level."""
    zero_parameters = np.zeros(model.parameter_count, dtype=float)
    expected_log_counts = model.expected_log_counts(zero_parameters, exposure_s)
    sigma = measurement_sigma(np.exp(expected_log_counts), systematic)

    def whitened_prediction(parameters):
        data_residual = (
            (model.expected_log_counts(parameters, exposure_s) - expected_log_counts)
            / sigma
        ).reshape(-1)
        regularization_residual = (
            model.regularization_residuals(
                parameters[: 2 * model.mode_count]
            )
            if include_regularization
            else np.empty(0, dtype=float)
        )
        return np.concatenate((data_residual, regularization_residual))

    step = 1.0e-5
    residual_size = whitened_prediction(zero_parameters).size
    jacobian = np.empty((residual_size, model.parameter_count), dtype=float)
    for parameter_index in range(model.parameter_count):
        delta = np.zeros(model.parameter_count, dtype=float)
        delta[parameter_index] = step
        jacobian[:, parameter_index] = (
            whitened_prediction(delta) - whitened_prediction(-delta)
        ) / (2.0 * step)
    return fisher_uncertainties(jacobian)


def fisher_uncertainties(jacobian):
    _, singular_values_full, right_vectors = np.linalg.svd(
        jacobian, full_matrices=False
    )
    threshold = singular_values_full[0] * 1.0e-8
    rank = int(np.count_nonzero(singular_values_full > threshold))
    resolved = singular_values_full[:rank]
    vectors = right_vectors[:rank]
    covariance = (vectors.T / resolved**2) @ vectors if rank else np.full(
        (jacobian.shape[1], jacobian.shape[1]), np.nan
    )
    null_projection = (
        np.sum(right_vectors[rank:] ** 2, axis=0)
        if rank < jacobian.shape[1]
        else np.zeros(jacobian.shape[1])
    )
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    errors[null_projection > 1.0e-4] = np.inf
    condition = (
        float(singular_values_full[0] / singular_values_full[rank - 1])
        if rank
        else float("inf")
    )
    return errors, singular_values_full, rank, condition


def write_sensitivity_scenarios(
    output_path, reference_model, exposure_s, reference_magnitude
):
    rows = []
    target_gap = int(np.argmin(np.abs(reference_model.basis_centers_nm - 1154.0)))
    for magnitude in sorted({15.0, 18.0, 20.0, float(reference_magnitude)}):
        photon_scale = 10.0 ** (-0.4 * (magnitude - reference_magnitude))
        scenario_model = StarCountModel(
            reference_model.wavelengths_nm,
            reference_model.filter_weights * photon_scale,
            reference_model.base_rates_per_second * photon_scale,
            reference_model.clean_reflection,
            reference_model.clean_transfer_first,
            reference_model.clean_substrate_power,
            reference_model.incident_admittance,
            reference_model.substrate_admittance,
            reference_model.incidence_sine,
            reference_model.ice_index,
            reference_model.history_thickness_nm,
            basis_grouping=reference_model.basis_grouping,
            fine_node_spacing_nm=reference_model.fine_node_spacing_nm,
            entropy_weight=reference_model.entropy_weight,
            phase_smoothness_weight=reference_model.phase_smoothness_weight,
        )
        for systematic_mmag in (0.0, 0.5, 1.0, 5.0):
            systematic = millimagnitude_to_log_flux_sigma(systematic_mmag)
            data_errors, _, data_rank, data_condition = expected_fisher_errors(
                scenario_model,
                exposure_s,
                systematic,
                include_regularization=False,
            )
            errors, _, rank, condition = expected_fisher_errors(
                scenario_model,
                exposure_s,
                systematic,
                include_regularization=True,
            )
            for gap_index, gap_name in enumerate(scenario_model.basis_names):
                amp_sigma = errors[gap_index]
                phase_sigma = errors[scenario_model.mode_count + gap_index]
                rows.append(
                    {
                        "ab_magnitude_at_550nm": magnitude,
                        "exposure_seconds_per_filter": exposure_s,
                        "systematic_floor": systematic,
                        "systematic_floor_mmag": systematic_mmag,
                        "gap": gap_name,
                        "gap_center_nm": scenario_model.basis_centers_nm[gap_index],
                        "sigma_log_amplitude_including_regularization": amp_sigma,
                        "sigma_phase_rad_including_regularization": phase_sigma,
                        "sigma_log_amplitude_data_only": data_errors[gap_index],
                        "sigma_phase_rad_data_only": data_errors[
                            scenario_model.mode_count + gap_index
                        ],
                        "snr_for_3pct_amplitude_with_prior": np.log(1.03) / amp_sigma,
                        "snr_for_0p1rad_phase_with_prior": 0.1 / phase_sigma,
                        "snr_for_3pct_amplitude_data_only": np.log(1.03)
                        / data_errors[gap_index],
                        "snr_for_0p1rad_phase_data_only": 0.1
                        / data_errors[scenario_model.mode_count + gap_index],
                        "jacobian_rank": rank,
                        "parameter_count": scenario_model.parameter_count,
                        "condition_number": condition,
                        "data_only_jacobian_rank": data_rank,
                        "data_only_condition_number": data_condition,
                        "target_gap_1039_1270nm": gap_index == target_gap,
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_outputs(
    output_dir,
    model,
    fit,
    truth,
    errors,
    singular_values,
    rank,
    condition,
    days,
    thickness,
    area_path,
    history_path,
    optical_source,
    star_temperature_k,
    star_ab_magnitude,
    exposure_s,
    systematic,
    cadence_days,
    sensitivity_path,
    counts,
    sigma,
    observed_log_counts,
    expected_log_counts,
    history_reset_days,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_count = model.mode_count
    fitted = fit.x
    fit_path = output_dir / "gstar_modulation_fit.csv"
    with fit_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gap",
                "gap_center_nm",
                "true_log_amplitude",
                "fit_log_amplitude",
            "local_sigma_log_amplitude_including_regularization",
                "true_phase_rad",
                "fit_phase_rad",
            "local_sigma_phase_rad_including_regularization",
            ]
        )
        for index, name in enumerate(model.basis_names):
            writer.writerow(
                [
                    name,
                    model.basis_centers_nm[index],
                    truth[index],
                    fitted[index],
                    errors[index],
                    truth[mode_count + index],
                    fitted[mode_count + index],
                    errors[mode_count + index],
                ]
            )

    counts_path = output_dir / "gstar_mock_observations.csv"
    prediction = model.expected_log_counts(fitted, exposure_s)
    weights = 1.0 / sigma**2

    def profile_filter_offsets(values):
        means = np.sum(weights * values, axis=0) / np.sum(weights, axis=0)
        centered = values - means[None, :]
        centered_variance = sigma**2 - 1.0 / np.sum(weights, axis=0)[None, :]
        return centered, np.sqrt(np.maximum(centered_variance, 0.0))

    observed_profiled, profiled_sigma = profile_filter_offsets(observed_log_counts)
    fitted_profiled, _ = profile_filter_offsets(prediction)
    injected_profiled, _ = profile_filter_offsets(expected_log_counts)
    unmodulated_log_counts = model.expected_log_counts(
        np.zeros(model.parameter_count, dtype=float), exposure_s
    )
    unmodulated_profiled, _ = profile_filter_offsets(unmodulated_log_counts)
    standardized_residual = (observed_log_counts - prediction) / sigma
    data_chi2 = float(np.sum(standardized_residual**2))
    data_degrees_of_freedom = int(observed_log_counts.size - fitted.size)
    with counts_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "day",
                "ice_thickness_nm_fixed",
                "filter",
                "expected_counts_per_exposure",
                "log_count_sigma",
                "mock_observed_log_counts",
                "injected_expected_log_counts",
                "fitted_expected_log_counts",
                "unmodulated_expected_log_counts",
                "mock_observed_profiled_log_flux",
                "injected_true_profiled_log_flux",
                "fitted_profiled_log_flux",
                "unmodulated_profiled_log_flux",
                "profiled_log_flux_sigma",
                "standardized_raw_residual",
            ]
        )
        for epoch_index, day in enumerate(days):
            for filter_index, filter_name in enumerate(FILTERS):
                writer.writerow(
                    [
                        day,
                        thickness[epoch_index],
                        filter_name,
                        counts[epoch_index, filter_index],
                        sigma[epoch_index, filter_index],
                        observed_log_counts[epoch_index, filter_index],
                        expected_log_counts[epoch_index, filter_index],
                        prediction[epoch_index, filter_index],
                        unmodulated_log_counts[epoch_index, filter_index],
                        observed_profiled[epoch_index, filter_index],
                        injected_profiled[epoch_index, filter_index],
                        fitted_profiled[epoch_index, filter_index],
                        unmodulated_profiled[epoch_index, filter_index],
                        profiled_sigma[epoch_index, filter_index],
                        standardized_residual[epoch_index, filter_index],
                    ]
                )

    lightcurve_path = output_dir / "gstar_mock_photometry_data_vs_fit.png"
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 11.0), sharex=True)
    for filter_index, axis in enumerate(axes.flat):
        observed_percent = 100.0 * np.expm1(observed_profiled[:, filter_index])
        injected_percent = 100.0 * np.expm1(injected_profiled[:, filter_index])
        fitted_percent = 100.0 * np.expm1(fitted_profiled[:, filter_index])
        unmodulated_percent = 100.0 * np.expm1(
            unmodulated_profiled[:, filter_index]
        )
        error_percent = (
            100.0
            * np.exp(observed_profiled[:, filter_index])
            * profiled_sigma[:, filter_index]
        )
        axis.errorbar(
            days,
            observed_percent,
            yerr=error_percent,
            fmt="o",
            markersize=2.8,
            color="0.25",
            ecolor="0.65",
            elinewidth=0.6,
            capsize=0,
            alpha=0.75,
            label="mock data",
        )
        axis.plot(
            days,
            injected_percent,
            color="#356bb3",
            linestyle="--",
            linewidth=1.2,
            label="injected mean",
        )
        axis.plot(
            days,
            fitted_percent,
            color="#d47730",
            linewidth=1.4,
            label="fitted model",
        )
        axis.plot(
            days,
            unmodulated_percent,
            color="0.15",
            linestyle="-.",
            linewidth=1.1,
            label="model with zero modulation",
        )
        for reset_day in history_reset_days:
            axis.axvline(reset_day, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_title(FILTERS[filter_index])
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Day (fixed ice-history timeline)")
    axes[-1, 1].set_xlabel("Day (fixed ice-history timeline)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Relative counts from day 0 (%)")
    axes[0, 0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "SYNTHETIC G-star photometry used in the modulation fit\n"
        "Points are mock measurements; curves show injected, fitted, and zero-modulation models; "
        "each filter's profiled offset is removed",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.95))
    figure.savefig(lightcurve_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    residual_path = output_dir / "gstar_mock_fit_residuals.png"
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 10.0), sharex=True, sharey=True)
    for filter_index, axis in enumerate(axes.flat):
        axis.axhspan(-1.0, 1.0, color="#6aa57a", alpha=0.15)
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.scatter(
            days,
            standardized_residual[:, filter_index],
            s=12,
            color="#3f6fa8",
            alpha=0.8,
        )
        for reset_day in history_reset_days:
            axis.axvline(reset_day, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_title(FILTERS[filter_index])
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Day (fixed ice-history timeline)")
    axes[-1, 1].set_xlabel("Day (fixed ice-history timeline)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Standardized residual")
    figure.suptitle(
        f"Mock-data likelihood residuals; χ²/dof = "
        f"{data_chi2:.1f}/{data_degrees_of_freedom} "
        f"({data_chi2 / data_degrees_of_freedom:.2f})",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.96))
    figure.savefig(residual_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    singular_path = output_dir / "gstar_fisher_singular_values.csv"
    with singular_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "whitened_jacobian_singular_value"])
        writer.writerows(enumerate(singular_values, start=1))

    figure_path = output_dir / "gstar_clean_reflectance_modulation_mock.png"
    wavelengths_nm = model.wavelengths_nm
    basis, _, _ = model.basis, model.basis_names, model.basis_centers_nm
    true_log_amp = truth[:mode_count] @ basis
    fit_log_amp = fitted[:mode_count] @ basis
    true_phase = truth[mode_count : 2 * mode_count] @ basis
    fit_phase = fitted[mode_count : 2 * mode_count] @ basis
    # Use the resolved local covariance for the plotted modulation envelope.
    _, _, right_vectors = np.linalg.svd(fit.jac, full_matrices=False)
    svals = np.linalg.svd(fit.jac, compute_uv=False)
    resolved_rank = int(np.count_nonzero(svals > svals[0] * 1.0e-8))
    cov = (
        (right_vectors[:resolved_rank].T / svals[:resolved_rank] ** 2)
        @ right_vectors[:resolved_rank]
    )
    amp_basis_variance = np.einsum(
        "iw,ij,jw->w", basis, cov[:mode_count, :mode_count], basis
    )
    phase_basis_variance = np.einsum(
        "iw,ij,jw->w",
        basis,
        cov[mode_count : 2 * mode_count, mode_count : 2 * mode_count],
        basis,
    )
    amp_sigma_curve = np.sqrt(np.maximum(amp_basis_variance, 0.0))
    phase_sigma_curve = np.sqrt(np.maximum(phase_basis_variance, 0.0))

    figure, axes = plt.subplots(3, 1, figsize=(13.5, 11.0), sharex=True)
    for filter_index, filter_name in enumerate(FILTERS):
        area = model.filter_weights[filter_index] / max(
            float(np.max(model.filter_weights[filter_index])), 1.0e-300
        )
        axes[0].plot(
            wavelengths_nm,
            area,
            linewidth=1.2,
            label=filter_name,
            alpha=0.8,
        )
    axes[0].set_ylabel("G-star photon weight × effective area / peak")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(
        f"G-star blackbody ({star_temperature_k:.0f} K, AB {star_ab_magnitude:.1f} at 550 nm); "
        "Roman effective-area filters"
    )

    axes[1].plot(
        wavelengths_nm,
        true_log_amp,
        color="#305fa6",
        label="injected log-amplitude",
    )
    axes[1].plot(
        wavelengths_nm,
        fit_log_amp,
        color="#d47730",
        label="recovered log-amplitude",
    )
    axes[1].fill_between(
        wavelengths_nm,
        fit_log_amp - amp_sigma_curve,
        fit_log_amp + amp_sigma_curve,
        color="#d47730",
        alpha=0.22,
        label="local 1σ envelope (including regularization)",
    )
    axes[1].set_ylabel("clean-reflectance log-amplitude")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(
        wavelengths_nm, true_phase, color="#305fa6", label="injected phase"
    )
    axes[2].plot(
        wavelengths_nm, fit_phase, color="#d47730", label="recovered phase"
    )
    axes[2].fill_between(
        wavelengths_nm,
        fit_phase - phase_sigma_curve,
        fit_phase + phase_sigma_curve,
        color="#d47730",
        alpha=0.22,
        label="local 1σ envelope (including regularization)",
    )
    axes[2].set_ylabel("clean-reflectance phase (rad)")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].legend(fontsize=8, loc="best")
    axes[2].grid(True, alpha=0.25)
    for axis in axes:
        for led_center in LED_CENTERS_NM:
            axis.axvline(led_center, color="0.4", linestyle=":", linewidth=0.8)
        axis.set_xlim(wavelengths_nm[0], wavelengths_nm[-1])
    figure.suptitle(
        "Fixed WFI16 tile-136 ice history; shared clean-stack modulation fit\n"
        f"{days.size} multi-filter epochs, {exposure_s:.0f} s/filter, "
        f"{systematic / LOG_FLUX_PER_MAG * 1000.0:.2f} mmag systematic floor",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.96))
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    precision_mmag = (2.5 / np.log(10.0)) * 1000.0 * sigma
    summary = {
        "test": "fixed_ice_history_gstar_photometric_modulation_mock",
        "detector": "WFI16",
        "tile_id": 136,
        "ice_history_fixed": True,
        "ice_history_archive": str(history_path),
        "fixed_ice_history_days": [float(days[0]), float(days[-1])],
        "fixed_ice_history_thickness_range_nm": [
            float(np.min(thickness)),
            float(np.max(thickness)),
        ],
        "fixed_reset_days": [float(value) for value in history_reset_days],
        "optical_state_source_fit": str(optical_source),
        "roman_effective_area_file": str(area_path),
        "roman_effective_area_sca": "SCA05 curve used as available system-area proxy; no separate QE multiplied",
        "filters": list(FILTERS),
        "star_sed": {
            "type": "Planck blackbody",
            "temperature_k": float(star_temperature_k),
            "normalization": "AB magnitude at 550 nm",
            "ab_magnitude": float(star_ab_magnitude),
        },
        "observing_assumptions": {
            "cadence_days": float(cadence_days),
            "exact_reset_epochs_added": True,
            "epochs": int(days.size),
            "filters_per_epoch": len(FILTERS),
            "exposure_seconds_per_filter": float(exposure_s),
            "fractional_systematic_floor": float(systematic),
            "systematic_floor_mmag": float(
                systematic / LOG_FLUX_PER_MAG * 1000.0
            ),
            "noise_model": "independent Gaussian log-count errors from Poisson limit plus systematic floor",
            "star_normalization": "free per-filter offset profiled in the fit",
        },
        "total_single_exposure_precision_mmag": {
            "p10": float(np.percentile(precision_mmag, 10.0)),
            "median": float(np.median(precision_mmag)),
            "p90": float(np.percentile(precision_mmag, 90.0)),
            "per_filter_median": {
                name: float(np.median(precision_mmag[:, index]))
                for index, name in enumerate(FILTERS)
            },
        },
        "modulation_parameterization": {
            "form": "r_clean_mod = r_clean_TMM * exp(sum(a_j B_j) + i sum(phi_j B_j))",
            "number_of_gap_modes": int(mode_count),
            "basis_grouping": model.basis_grouping,
            "fine_node_spacing_nm": float(model.fine_node_spacing_nm),
            "maxent_relative_entropy_weight": float(model.entropy_weight),
            "phase_second_difference_weight": float(model.phase_smoothness_weight),
            "maxent_reference": "unit amplitude multiplier (zero log-amplitude)",
            "maxent_penalty": "wavelength-weighted KL(exp(log-amplitude) || 1)",
            "phase_penalty": "second differences within each LED-free interval",
            "gap_intervals_nm": [
                [400.0, float(LED_CENTERS_NM[0])],
                *[
                    [float(lower), float(upper)]
                    for lower, upper in zip(LED_CENTERS_NM[:-1], LED_CENTERS_NM[1:])
                ],
                [float(LED_CENTERS_NM[-1]), 2321.0],
            ],
            "modulation_zero_at_led_centres": True,
            "shared_for_all_fixed_thicknesses": True,
            "stellar_observable": "Roman effective area weighted by TMM substrate-power ratio",
            "clean_transfer_normalization_held_fixed": True,
            "ice_layer": "existing production n=1.25, k=0 transfer-matrix layer",
        },
        "injected_mock_signals": [
            {
                "mode": model.basis_names[index],
                "center_nm": float(model.basis_centers_nm[index]),
                "clean_reflection_amplitude_fraction": float(np.expm1(truth[index])),
                "clean_reflection_phase_rad": float(truth[mode_count + index]),
            }
            for index in range(mode_count)
            if abs(truth[index]) > 1.0e-12 or abs(truth[mode_count + index]) > 1.0e-12
        ],
        "fit": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "cost_chi2_over_2": float(fit.cost),
            "data_chi2": data_chi2,
            "data_degrees_of_freedom_approx": data_degrees_of_freedom,
            "data_reduced_chi2_approx": float(
                data_chi2 / data_degrees_of_freedom
            ),
            "nfev": int(fit.nfev),
            "jacobian_rank": int(rank),
            "parameter_count_including_filter_offsets": int(fit.x.size),
            "whitened_jacobian_condition_number": float(condition),
            "least_squares_formal_errors_are_local": True,
            "fit_covariance_includes_regularization": bool(
                model.entropy_weight > 0.0 or model.phase_smoothness_weight > 0.0
            ),
        },
        "photon_rate_per_filter_s": {
            name: float(rate) for name, rate in zip(FILTERS, model.base_rates_per_second)
        },
        "counts_per_exposure_at_first_epoch": {
            name: float(value)
            for name, value in zip(FILTERS, counts[0])
        },
        "figure": str(figure_path),
        "fit_csv": str(fit_path),
        "observation_csv": str(counts_path),
        "photometry_data_vs_fit_plot": str(lightcurve_path),
        "residual_plot": str(residual_path),
        "observations_are_synthetic": True,
        "photometry_plot_normalization": "per-filter weighted mean removed, matching profiled filter offsets",
        "singular_values_csv": str(singular_path),
        "sensitivity_scenarios_csv": str(sensitivity_path),
        "interpretation": (
            "Optimistic identifiability test: correct blackbody, exact effective area, "
            "fixed known ice history, all eight filters at every epoch, and no model mismatch."
        ),
    }
    summary_path = output_dir / "gstar_modulation_mock_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return (
        figure_path,
        fit_path,
        counts_path,
        lightcurve_path,
        residual_path,
        singular_path,
        sensitivity_path,
        summary_path,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ice-history-archive", type=Path, default=DEFAULT_ICE_HISTORY)
    parser.add_argument("--optical-cache-archive", type=Path, default=DEFAULT_OPTICAL_CACHE)
    parser.add_argument("--effective-area", type=Path, default=DEFAULT_EFFECTIVE_AREA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI16")
    parser.add_argument("--tile-id", type=int, default=136)
    parser.add_argument("--star-temperature-k", type=float, default=5772.0)
    parser.add_argument("--star-ab-magnitude", type=float, default=15.0)
    parser.add_argument("--exposure-seconds", type=float, default=100.0)
    parser.add_argument("--cadence-days", type=float, default=1.0)
    parser.add_argument(
        "--systematic-floor-mmag",
        type=float,
        default=0.5,
        help="Photometric systematic floor in millimagnitudes (default: 0.5 mmag).",
    )
    parser.add_argument(
        "--systematic-floor",
        type=float,
        default=None,
        help="Legacy override: systematic floor as fractional log-count sigma.",
    )
    parser.add_argument(
        "--modulation-basis",
        choices=("per-gap", "coarse", "fine"),
        default="per-gap",
        help="Seven per-gap modes, three broad groups, or fine LED-anchored nodes.",
    )
    parser.add_argument("--fine-node-spacing-nm", type=float, default=100.0)
    parser.add_argument("--maxent-weight", type=float, default=0.0)
    parser.add_argument("--phase-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--injected-amplitude-fraction", type=float, default=0.03)
    parser.add_argument("--injected-phase-rad", type=float, default=0.1)
    parser.add_argument(
        "--inject-mode",
        action="append",
        nargs=3,
        type=float,
        metavar=("CENTER_NM", "AMPLITUDE_FRACTION", "PHASE_RAD"),
        help="Inject an extra local mode; repeat for positive/negative spectral features.",
    )
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args(argv)
    if args.star_temperature_k <= 0.0 or args.exposure_seconds <= 0.0:
        parser.error("Star temperature and exposure must be positive")
    if args.cadence_days <= 0.0 or args.systematic_floor_mmag < 0.0:
        parser.error("Cadence must be positive and systematic floor non-negative")
    systematic = (
        float(args.systematic_floor)
        if args.systematic_floor is not None
        else millimagnitude_to_log_flux_sigma(args.systematic_floor_mmag)
    )
    if systematic < 0.0:
        parser.error("Systematic floor must be non-negative")
    if args.fine_node_spacing_nm <= 0.0:
        parser.error("Fine modulation-node spacing must be positive")
    if args.maxent_weight < 0.0 or args.phase_smoothness_weight < 0.0:
        parser.error("Regularization weights must be non-negative")
    if args.injected_amplitude_fraction <= -1.0:
        parser.error("Injected clean-reflection amplitude fraction must exceed -1")
    injection_specs = args.inject_mode or [
        (1154.0, args.injected_amplitude_fraction, args.injected_phase_rad)
    ]
    if any(amplitude <= -1.0 for _, amplitude, _ in injection_specs):
        parser.error("Every injected amplitude fraction must exceed -1")

    source_days, source_thickness, reset_days = load_fixed_ice_history(
        args.ice_history_archive, args.detector, args.tile_id
    )
    end_day = float(source_days[-1])
    sample_days = np.arange(source_days[0], end_day, args.cadence_days)
    epoch_days = np.unique(np.concatenate((sample_days, [end_day], reset_days)))
    epoch_days = epoch_days[(epoch_days >= source_days[0]) & (epoch_days <= end_day)]
    history_thickness = interpolate_fixed_history(
        source_days, source_thickness, reset_days, epoch_days
    )

    wavelengths_nm, areas_m2 = load_effective_area(args.effective_area)
    positive_area = np.any(areas_m2 > 0.0, axis=0) & (wavelengths_nm <= 2321.0)
    wavelengths_nm = wavelengths_nm[positive_area]
    areas_m2 = areas_m2[:, positive_area]
    source_fit = source_fit_from_cache(
        args.optical_cache_archive, args.detector, args.tile_id
    )
    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths_nm)
    spectral_basis = production.spectral_modulation_basis(wavelengths_nm)
    state = complex_model.load_state(
        source_fit, wavelengths_nm, base_n, base_k, spectral_basis
    )
    (
        clean_transfer_first,
        clean_transfer_second,
        incident_admittance,
        substrate_admittance,
        incidence_sine,
    ) = clean_stack_boundary(state, wavelengths_nm)
    clean_reflection = (
        incident_admittance * clean_transfer_first - clean_transfer_second
    ) / (incident_admittance * clean_transfer_first + clean_transfer_second)
    direct_clean_reflection, _ = complex_model.reflection(
        state, wavelengths_nm, 0.0, "s"
    )
    if np.max(np.abs(clean_reflection - direct_clean_reflection)) > 1.0e-10:
        raise RuntimeError("Clean-stack reflection disagrees with the reference TMM")
    clean_substrate_power = (
        np.real(substrate_admittance / incident_admittance)
        * np.abs(
            2.0
            * incident_admittance
            / (incident_admittance * clean_transfer_first + clean_transfer_second)
        )
        ** 2
    )
    _, direct_coated_reflection = complex_model.reflection(
        state, wavelengths_nm, 30.0, "s"
    )
    _, reconstructed_coated_reflection = coated_substrate_power(
        clean_reflection,
        clean_transfer_first,
        wavelengths_nm,
        30.0,
        state["ice_index"],
        incident_admittance,
        substrate_admittance,
        incidence_sine,
    )
    if np.max(np.abs(direct_coated_reflection - reconstructed_coated_reflection)) > 1.0e-10:
        raise RuntimeError("Clean-reflection coating disagrees with the reference TMM")

    photon_flux = blackbody_photon_flux_per_nm(
        wavelengths_nm, args.star_temperature_k, args.star_ab_magnitude
    )
    filter_weights = photon_flux[None, :] * areas_m2
    base_rates = np.trapezoid(filter_weights, wavelengths_nm, axis=1)
    if np.any(base_rates <= 0.0):
        raise ValueError("A Roman filter has zero integrated G-star photon rate")

    model = StarCountModel(
        wavelengths_nm,
        filter_weights,
        base_rates,
        clean_reflection,
        clean_transfer_first,
        clean_substrate_power,
        incident_admittance,
        substrate_admittance,
        incidence_sine,
        state["ice_index"],
        history_thickness,
        basis_grouping=args.modulation_basis,
        fine_node_spacing_nm=args.fine_node_spacing_nm,
        entropy_weight=args.maxent_weight,
        phase_smoothness_weight=args.phase_smoothness_weight,
    )
    mode_count = model.mode_count
    truth = np.zeros(2 * mode_count, dtype=float)
    injected_mode_indices = []
    for center_nm, amplitude_fraction, phase_rad in injection_specs:
        mode_index = int(np.argmin(np.abs(model.basis_centers_nm - center_nm)))
        truth[mode_index] += np.log1p(amplitude_fraction)
        truth[mode_count + mode_index] += phase_rad
        if mode_index not in injected_mode_indices:
            injected_mode_indices.append(mode_index)
    fit, observed, expected, counts, sigma = fit_synthetic_counts(
        model,
        truth,
        args.exposure_seconds,
        systematic,
        args.seed,
    )
    errors, singular_values, rank, condition = fisher_uncertainties(fit.jac)
    sensitivity_path = args.output_dir / "gstar_sensitivity_scenarios.csv"
    sensitivity_rows = write_sensitivity_scenarios(
        sensitivity_path,
        model,
        args.exposure_seconds,
        args.star_ab_magnitude,
    )
    paths = save_outputs(
        args.output_dir,
        model,
        fit,
        truth,
        errors,
        singular_values,
        rank,
        condition,
        epoch_days,
        history_thickness,
        args.effective_area,
        args.ice_history_archive,
        source_fit,
        args.star_temperature_k,
        args.star_ab_magnitude,
        args.exposure_seconds,
        systematic,
        args.cadence_days,
        sensitivity_path,
        counts,
        sigma,
        observed,
        expected,
        reset_days,
    )
    print(
        f"Fixed history {epoch_days.size} epochs, d={history_thickness.min():.2f}–"
        f"{history_thickness.max():.2f} nm; {len(FILTERS)} Roman filters; "
        f"fit rank {rank}/{fit.x.size}, condition {condition:.3g}."
    )
    for path in paths:
        print(f"Wrote {path}")
    for index in injected_mode_indices:
        reference_sensitivity = next(
            row
            for row in sensitivity_rows
            if row["ab_magnitude_at_550nm"] == args.star_ab_magnitude
            and row["systematic_floor"] == systematic
            and row["gap"] == model.basis_names[index]
        )
        print(
            f"Injected {model.basis_names[index]}: "
            f"amplitude {np.expm1(truth[index]) * 100.0:+.1f}%, "
            f"phase {truth[mode_count + index]:+.3f} rad; recovered "
            f"log-amplitude {fit.x[index]:+.4f} ± {errors[index]:.4f}, "
            f"phase {fit.x[mode_count + index]:+.4f} ± "
            f"{errors[mode_count + index]:.4f} rad. "
            "Data-only 3% / 0.1-rad reference SNR: "
            f"{reference_sensitivity['snr_for_3pct_amplitude_data_only']:.2f}σ / "
            f"{reference_sensitivity['snr_for_0p1rad_phase_data_only']:.2f}σ."
        )
    for index, name in enumerate(model.basis_names):
        print(
            f"{name}: amplitude {fit.x[index]:+.4f} ± {errors[index]:.4f}; "
            f"phase {fit.x[mode_count + index]:+.4f} ± "
            f"{errors[mode_count + index]:.4f} rad"
        )


if __name__ == "__main__":
    main()
