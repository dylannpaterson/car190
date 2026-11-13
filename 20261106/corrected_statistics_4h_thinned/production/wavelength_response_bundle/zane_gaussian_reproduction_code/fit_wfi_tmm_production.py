"""Fixed production TMM model for independent WFI tile fits.

Each invocation evaluates the corrected sufficient statistics with a fixed
three-layer lossless AR stack, semi-infinite HgCdTe, an energy-weighted SED,
and independent per-tile latent parameters.  EMCEE initialization is handled
by ``run_wfi_tmm_emcee_warmup.py``; this module owns the deterministic MAP fit
and its optional constant-rate warmup/MaxEnt regularization stages.
"""

import csv
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from assess_rcs_spectral_identifiability import load_qe, load_sed


ROOT = Path(__file__).resolve().parent
SED_URL = str(
    ROOT / "external_reference_data/WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
)
QE_URL = str(ROOT / "external_reference_data/WFI_Quantum_efficiency.ecsv")
# Physical decon2 boundary: first Program 1063 observation.  The later
# operational-temperature transition is not an ice-thickness reset.
PRODUCTION_RESET_DAY = 12.907397928240743
PRODUCTION_MAP_STEPS = 10000
PRODUCTION_MAP_FTOL = 1.0e-9
PRODUCTION_MAP_XTOL = 1.0e-9
AR_LAYERS = 3
AR_DESIGN_WAVELENGTH_NM = 1050.0
AR_DIELECTRIC_K_START = 0.0
AR_LOWER = np.asarray([0.01, 1.0, 0.0], dtype=float)
AR_UPPER = np.asarray([10.0, 4.0, 0.5], dtype=float)
AR_MEAN_PRIOR_SIGMA = 10.0
AR_K_PRIOR_SIGMA = 0.02
AR_K_INIT = 1.0e-6
ICE_INDEX_DEFAULT = 1.25
ICE_K_PRIOR_SIGMA = 0.02
ICE_K_INIT = 1.0e-6
ICE_N_MODULATION_NODE_WAVELENGTHS_NM = np.asarray(
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
ICE_N_MODULATION_WIDTH_NM = 500.0
ICE_N_MODULATION_PRIOR_SIGMA_LOG = 0.05
# Production uses the tight baseline constant-index prior. Diagnostic scripts
# can pass a different width explicitly when running a sensitivity study.
ICE_N_CONSTANT_PRIOR_SIGMA_LOG = 0.01
AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 0.02
AR_HGCDTE_MODULATION_ANCHOR_NM = 1050.0
INCIDENCE_ANGLE_MAX_DEG = 15.0
INCIDENCE_ANGLE_INIT_DEG = 7.5
PUPIL_RADIAL_NODES, PUPIL_RADIAL_WEIGHTS = np.polynomial.legendre.leggauss(4)
PUPIL_AZIMUTH_COUNT = 8
POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA = 0.1
INITIAL_THICKNESS_LOG_MEDIAN_NM = 10.0
INITIAL_THICKNESS_LOG_SIGMA = 1.5
INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM = 5.0
INITIAL_THICKNESS_PRIOR_LOGNORMAL = "lognormal"
INITIAL_THICKNESS_PRIOR_HALF_NORMAL = "half_normal"
TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 5.0
TRAJECTORY_RATE_PRIOR_SIGMA_LOG = 1.5
NP_SMOOTHNESS_WEIGHT = 0.1
ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG = 0.25
# Production uses the sigma-0.25 HgCdTe n sensitivity as the default.
ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG = 0.25
LED_LOG_OFFSET_PRIOR_SIGMA_LOG = 0.01
LOG_NOISE_SIGMA = 0.005
GAUSS_LEGENDRE_NODES, GAUSS_LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(8)

NOMINAL_HGCDTE = np.asarray([3.6, 0.7], dtype=float)
HGCDTE_REFERENCE_COMPOSITION = 0.445
HGCDTE_REFERENCE_TEMPERATURE_K = 80.0
HGCDTE_N_TABLE_WAVELENGTHS_NM = np.arange(1600.0, 6600.1, 200.0)
HGCDTE_N_TABLE_X0397 = np.asarray(
    [
        2.946,
        2.978,
        3.012,
        3.049,
        3.088,
        3.128,
        3.168,
        3.206,
        3.159,
        3.126,
        3.110,
        3.100,
        3.093,
        3.087,
        3.083,
        3.079,
        3.077,
        3.074,
        3.072,
        3.071,
        3.069,
        3.068,
        3.067,
        3.066,
        3.065,
        3.064,
    ],
    dtype=float,
)
HGCDTE_N_TABLE_X0459 = np.asarray(
    [
        2.895,
        2.940,
        2.987,
        3.036,
        3.086,
        3.030,
        3.003,
        2.989,
        2.981,
        2.975,
        2.970,
        2.967,
        2.965,
        2.962,
        2.961,
        2.959,
        2.958,
        2.957,
        2.956,
        2.955,
        2.955,
        2.954,
        2.954,
        2.953,
        2.953,
        2.952,
    ],
    dtype=float,
)


def _initial_thickness_settings(prior, sigma_nm):
    prior = str(prior).lower().replace("-", "_")
    if prior not in (
        INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        INITIAL_THICKNESS_PRIOR_HALF_NORMAL,
    ):
        raise ValueError(
            "Initial-thickness prior must be 'lognormal' or 'half_normal'"
        )
    sigma_nm = float(sigma_nm)
    if not np.isfinite(sigma_nm) or sigma_nm <= 0.0:
        raise ValueError("Initial-thickness prior sigma must be positive and finite")
    if prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL:
        names = ("initial_thickness_log_nm", "reset_initial_thickness_log_nm")
    else:
        names = ("initial_thickness_nm", "reset_initial_thickness_nm")
    return prior, sigma_nm, names


def reset_days_for_data(data):
    """Return the ordered physical reset boundaries represented by ``data``."""
    if "reset_days" in data:
        reset_days = np.asarray(data["reset_days"], dtype=float).reshape(-1)
    else:
        reset_days = np.asarray(
            [data.get("reset_day", PRODUCTION_RESET_DAY)], dtype=float
        )
    if reset_days.size == 0 or np.any(~np.isfinite(reset_days)):
        raise ValueError("Reset days must contain at least one finite value")
    if np.any(np.diff(reset_days) <= 0.0):
        raise ValueError("Reset days must be strictly increasing")
    return reset_days


def reset_initial_thickness_site_names(prior, reset_count=1):
    """Return the initial-thickness site followed by one site per reset."""
    reset_count = int(reset_count)
    if reset_count not in (1, 2):
        raise ValueError("Production supports the existing reset and one optional third decon")
    prior, _, names = _initial_thickness_settings(
        prior, INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM
    )
    suffix = "_log_nm" if prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL else "_nm"
    reset_names = [names[1]]
    if reset_count > 1:
        reset_names.append(f"third_decon_initial_thickness{suffix}")
    return (names[0], *reset_names)


def reset_offset_site_names(reset_count=1):
    """Return the per-tile grey-offset site for each physical reset."""
    reset_count = int(reset_count)
    if reset_count < 1:
        raise ValueError("At least one reset site is required")
    return (
        "post_reset_tile_log_offset",
        *("post_third_decon_tile_log_offset" for _ in range(reset_count - 1)),
    )


def reset_site_names(data, initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL):
    """Return trajectory and grey site names for the reset boundaries in data."""
    reset_count = reset_days_for_data(data).size
    return (
        reset_initial_thickness_site_names(initial_thickness_prior, reset_count),
        reset_offset_site_names(reset_count),
    )


def initial_thickness_site_names(prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL):
    """Return the MAP site names used by an initial-thickness prior."""
    return _initial_thickness_settings(prior, INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM)[2]


@dataclass(frozen=True)
class ProductionConfig:
    tile_statistics: Path
    output_dir: Path
    third_decon_day: float | None = None
    map_init_from: Path | None = None
    map_initial_values: dict | None = None
    smoothness_weight: float | None = None
    entropy_reference_log_rates: np.ndarray | None = None
    entropy_weight: float = 0.0
    rate_prior_log_mean: np.ndarray | None = None
    trajectory_rate_prior_sigma_log: float = TRAJECTORY_RATE_PRIOR_SIGMA_LOG
    trajectory_parameterization: str = "rate"
    initial_thickness_prior: str = INITIAL_THICKNESS_PRIOR_LOGNORMAL
    initial_thickness_prior_sigma_nm: float = INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM
    thickness_prior_centers: dict | None = None
    thickness_prior_sigmas: dict | None = None
    absolute_hgcdte_n_scale_prior_sigma_log: float = (
        ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG
    )
    fixed_absolute_hgcdte_n_log_scale: float | None = None
    constant_rate: bool = False
    fit_incidence_angle: bool = True
    incidence_pupil_half_angle_deg: float = 0.0
    fit_ice_n_modulation: bool = False
    ice_n_modulation_prior_sigma_log: float = ICE_N_MODULATION_PRIOR_SIGMA_LOG
    fit_ice_n_constant: bool = False
    ice_n_constant_prior_sigma_log: float = ICE_N_CONSTANT_PRIOR_SIGMA_LOG
    ice_n_constant_reference: float = ICE_INDEX_DEFAULT
    fixed_ice_n_log_constant: float | None = None
    ice_n_porosity_parameters: tuple[float, float, float, float] | None = None
    fit_ice_k_constant: bool = False
    ice_k_prior_sigma: float = ICE_K_PRIOR_SIGMA
    fit_ar_k_constant: bool = False
    ar_k_prior_sigma: float = AR_K_PRIOR_SIGMA
    ar_mean_prior_sigma: float = AR_MEAN_PRIOR_SIGMA
    ar_prior_center_raw: np.ndarray | None = None
    ar_prior_sigma_raw: np.ndarray | float | None = None
    fit_ar_hgcdte_modulation: bool = False
    ar_hgcdte_modulation_prior_sigma_log: float = AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG
    hgcdte_n_modulation_prior_sigma_log: float | None = None
    fit_led_log_offsets: bool = False
    led_log_offset_prior_sigma_log: float = LED_LOG_OFFSET_PRIOR_SIGMA_LOG
    thickness_penalty_plan: dict | None = None
    thickness_penalty_lambda_shared: float = 0.0
    thickness_penalty_lambda_disagreement: float = 0.0
    sed_weighting: str = "energy"
    qe_response_mode: str = "none"
    model_metadata: dict | None = None


def validate_ice_n_porosity_parameters(parameters):
    """Validate ``(p0, pinf, length_nm, n_solid)`` for the LL ice model."""
    if parameters is None:
        return None
    values = np.asarray(parameters, dtype=float).reshape(-1)
    if values.size != 4 or np.any(~np.isfinite(values)):
        raise ValueError(
            "Ice-n porosity parameters must be finite (p0, pinf, length_nm, n_solid)"
        )
    p0, pinf, length_nm, n_solid = values
    if not 0.0 <= p0 < 1.0:
        raise ValueError("Ice-n porosity p0 must be in [0, 1)")
    if not p0 <= pinf < 1.0:
        raise ValueError("Ice-n porosity pinf must satisfy p0 <= pinf < 1")
    if length_nm <= 0.0:
        raise ValueError("Ice-n porosity length must be positive")
    if n_solid <= 0.0:
        raise ValueError("Solid ice refractive index must be positive")
    return tuple(float(value) for value in values)


def require_numpyro():
    """Load the JAX/NumPyro pieces used for EMCEE likelihoods and MAP."""
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The production TMM fit requires jax and numpyro"
        ) from error
    return jax, jnp, numpyro, dist


def _sed_output(spectrum):
    raw_output = np.asarray(spectrum["raw_output"], dtype=float)
    threshold = np.max(raw_output) * 0.05
    return np.where(raw_output >= threshold, raw_output, 0.0)


def _trapezoid_sample_weights(wavelengths):
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("A wavelength grid must contain at least two samples")
    weights = np.empty_like(wavelengths)
    weights[0] = 0.5 * (wavelengths[1] - wavelengths[0])
    weights[-1] = 0.5 * (wavelengths[-1] - wavelengths[-2])
    weights[1:-1] = 0.5 * (wavelengths[2:] - wavelengths[:-2])
    return weights


def ice_n_modulation_basis(
    wavelengths_nm,
    node_wavelengths_nm=ICE_N_MODULATION_NODE_WAVELENGTHS_NM,
    width_nm=ICE_N_MODULATION_WIDTH_NM,
):
    """Return normalized broad basis functions for n(lambda) modulation."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    node_wavelengths_nm = np.asarray(node_wavelengths_nm, dtype=float).reshape(-1)
    if (
        wavelengths_nm.size == 0
        or np.any(~np.isfinite(wavelengths_nm))
        or np.any(wavelengths_nm <= 0.0)
    ):
        raise ValueError("Ice modulation wavelengths must be positive and finite")
    if (
        node_wavelengths_nm.size < 2
        or np.any(~np.isfinite(node_wavelengths_nm))
        or np.any(np.diff(node_wavelengths_nm) <= 0.0)
        or not np.isfinite(width_nm)
        or width_nm <= 0.0
    ):
        raise ValueError("Ice modulation nodes and width must be finite and ordered")
    basis = np.exp(
        -0.5
        * ((wavelengths_nm[:, None] - node_wavelengths_nm[None, :]) / width_nm) ** 2
    )
    return basis / np.sum(basis, axis=1, keepdims=True)


def spectral_modulation_basis(
    wavelengths_nm,
    anchor_wavelength_nm=AR_HGCDTE_MODULATION_ANCHOR_NM,
):
    """Return broad basis functions anchored to zero at one wavelength."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    if not np.isfinite(anchor_wavelength_nm) or anchor_wavelength_nm <= 0.0:
        raise ValueError("Spectral modulation anchor must be positive and finite")
    basis = ice_n_modulation_basis(wavelengths_nm)
    anchor = ice_n_modulation_basis(np.asarray([anchor_wavelength_nm]))[0]
    return basis - anchor[None, :]


def natural_cubic_spline_values(knots, values, targets, jnp):
    """Evaluate a natural cubic spline with a leading batch axis.

    ``values`` has shape ``(batch, knot_count)`` and ``targets`` has shape
    ``(batch, ...)``. Targets are clamped to the knot interval so an
    experiment can define a bounded optical-index model without extrapolating
    a cubic beyond its final knot.
    """
    knots = np.asarray(knots, dtype=float).reshape(-1)
    if (
        knots.size < 2
        or np.any(~np.isfinite(knots))
        or np.any(np.diff(knots) <= 0.0)
    ):
        raise ValueError("Spline knots must be finite and strictly increasing")
    values = jnp.asarray(values)
    if values.ndim < 1 or values.shape[-1] != knots.size:
        raise ValueError("Spline values must end with the knot axis")
    flat_values = values.reshape((-1, knots.size))
    targets = jnp.asarray(targets)
    if targets.ndim == 0:
        targets = jnp.broadcast_to(targets, (flat_values.shape[0],))
    elif targets.shape[0] != flat_values.shape[0]:
        raise ValueError("Spline targets must have the same leading batch axis")

    knots_jnp = jnp.asarray(knots)
    h = jnp.diff(knots_jnp)
    matrix = jnp.zeros((knots.size, knots.size), dtype=values.dtype)
    matrix = matrix.at[0, 0].set(1.0)
    matrix = matrix.at[-1, -1].set(1.0)
    for index in range(1, knots.size - 1):
        matrix = matrix.at[index, index - 1].set(h[index - 1])
        matrix = matrix.at[index, index].set(2.0 * (h[index - 1] + h[index]))
        matrix = matrix.at[index, index + 1].set(h[index])
    slopes = jnp.diff(flat_values, axis=1) / h[None, :]
    rhs = jnp.zeros_like(flat_values)
    rhs = rhs.at[:, 1:-1].set(6.0 * jnp.diff(slopes, axis=1))
    moments = jnp.linalg.solve(matrix, rhs.T).T
    coefficient_a = flat_values[:, :-1]
    coefficient_b = slopes - h[None, :] * (
        2.0 * moments[:, :-1] + moments[:, 1:]
    ) / 6.0
    coefficient_c = moments[:, :-1] / 2.0
    coefficient_d = jnp.diff(moments, axis=1) / (6.0 * h[None, :])

    target = jnp.clip(targets, knots_jnp[0], knots_jnp[-1])
    index = jnp.searchsorted(knots_jnp, target, side="right") - 1
    index = jnp.clip(index, 0, knots.size - 2)
    row = jnp.arange(flat_values.shape[0]).reshape(
        (flat_values.shape[0],) + (1,) * (target.ndim - 1)
    )
    delta = target - knots_jnp[index]
    return (
        coefficient_a[row, index]
        + coefficient_b[row, index] * delta
        + coefficient_c[row, index] * delta**2
        + coefficient_d[row, index] * delta**3
    )


def monotonic_cubic_spline_values(knots, values, targets, jnp):
    """Evaluate a decreasing cubic Hermite spline without overshoot.

    The knot values must be non-increasing.  Zero tangents at each knot make
    every interval monotonic when its two endpoint values are monotonic.
    """
    knots = np.asarray(knots, dtype=float).reshape(-1)
    if (
        knots.size < 2
        or np.any(~np.isfinite(knots))
        or np.any(np.diff(knots) <= 0.0)
    ):
        raise ValueError("Spline knots must be finite and strictly increasing")
    values = jnp.asarray(values)
    if values.ndim < 1 or values.shape[-1] != knots.size:
        raise ValueError("Spline values must end with the knot axis")
    flat_values = values.reshape((-1, knots.size))
    targets = jnp.asarray(targets)
    if targets.ndim == 0:
        targets = jnp.broadcast_to(targets, (flat_values.shape[0],))
    elif targets.shape[0] != flat_values.shape[0]:
        raise ValueError("Spline targets must have the same leading batch axis")

    knots_jnp = jnp.asarray(knots)
    h = jnp.diff(knots_jnp)
    target = jnp.clip(targets, knots_jnp[0], knots_jnp[-1])
    index = jnp.searchsorted(knots_jnp, target, side="right") - 1
    index = jnp.clip(index, 0, knots.size - 2)
    row = jnp.arange(flat_values.shape[0]).reshape(
        (flat_values.shape[0],) + (1,) * (target.ndim - 1)
    )
    delta = target - knots_jnp[index]
    fraction = delta / h[index]
    first = flat_values[row, index]
    second = flat_values[row, index + 1]
    first_basis = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
    second_basis = -2.0 * fraction**3 + 3.0 * fraction**2
    return first_basis * first + second_basis * second


def _prepare_response_integration(
    model_wavelengths,
    spectra,
    qe_wavelengths,
    *,
    qe_values=None,
    qe_column=None,
    sed_weighting="energy",
    include_qe=False,
):
    if sed_weighting not in ("energy", "photon"):
        raise ValueError("SED weighting must be 'energy' or 'photon'")
    if include_qe:
        if qe_values is None or qe_column is None:
            raise ValueError("QE values and a QE column are required")
        qe_values = np.asarray(qe_values, dtype=float)
        if qe_values.ndim != 2 or qe_values.shape[0] != len(qe_wavelengths):
            raise ValueError("QE values must align with the QE wavelength grid")
        qe_column = int(qe_column)
        if qe_column < 0 or qe_column >= qe_values.shape[1]:
            raise ValueError("QE column is outside the supplied table")
    model_wavelengths = np.asarray(model_wavelengths, dtype=float)
    model_weights = np.zeros((6, model_wavelengths.size), dtype=float)
    outside_output = np.zeros(6, dtype=float)
    clean_output = np.zeros(6, dtype=float)
    for band in range(1, 7):
        spectrum = spectra[band]
        wavelengths = np.asarray(spectrum["wavelength"], dtype=float)
        supported_qe = (wavelengths >= qe_wavelengths[0]) & (
            wavelengths <= qe_wavelengths[-1]
        )
        spectral_weight = (
            wavelengths if sed_weighting == "photon" else np.ones_like(wavelengths)
        )
        qe = (
            np.interp(
                wavelengths,
                qe_wavelengths,
                qe_values[:, qe_column],
                left=0.0,
                right=0.0,
            )
            if include_qe
            else np.ones_like(wavelengths)
        )
        weighted_output = np.where(
            supported_qe,
            _sed_output(spectrum) * spectral_weight * qe,
            0.0,
        )
        sample_weights = weighted_output * _trapezoid_sample_weights(wavelengths)
        clean_output[band - 1] = np.sum(sample_weights)
        supported = (wavelengths >= model_wavelengths[0]) & (
            wavelengths <= model_wavelengths[-1]
        )
        outside_output[band - 1] = np.sum(sample_weights[~supported])
        supported_wavelengths = wavelengths[supported]
        supported_weights = sample_weights[supported]
        lower = np.searchsorted(model_wavelengths, supported_wavelengths, side="right") - 1
        lower = np.clip(lower, 0, model_wavelengths.size - 2)
        upper = lower + 1
        fraction = (
            (supported_wavelengths - model_wavelengths[lower])
            / (model_wavelengths[upper] - model_wavelengths[lower])
        )
        np.add.at(model_weights[band - 1], lower, supported_weights * (1.0 - fraction))
        np.add.at(model_weights[band - 1], upper, supported_weights * fraction)
    if np.any(clean_output <= 0.0):
        raise ValueError("Each RCS LED requires positive clean SED output")
    return model_weights, outside_output, clean_output


def hgcdte_reference_optical_constants(wavelengths_nm):
    """Return the fixed 80 K wavelength-dependent HgCdTe reference."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths_nm.size == 0 or np.any(~np.isfinite(wavelengths_nm)):
        raise ValueError("HgCdTe wavelengths must be finite and non-empty")
    composition = HGCDTE_REFERENCE_COMPOSITION
    weight = (composition - 0.397) / (0.459 - 0.397)
    real_table = HGCDTE_N_TABLE_X0397 + weight * (
        HGCDTE_N_TABLE_X0459 - HGCDTE_N_TABLE_X0397
    )
    real_index = np.interp(
        wavelengths_nm,
        HGCDTE_N_TABLE_WAVELENGTHS_NM,
        real_table,
        left=real_table[0],
        right=real_table[-1],
    )

    wavelength_um = wavelengths_nm / 1000.0
    photon_energy = 1.239841984 / wavelength_um
    temperature = HGCDTE_REFERENCE_TEMPERATURE_K
    band_gap = (
        -0.302
        + 1.93 * composition
        - 0.810 * composition**2
        + 0.832 * composition**3
        + 5.35e-4
        * (1.0 - 2.0 * composition)
        * (temperature**3 - 1822.0)
        / (255.2 + temperature**2)
    )
    transition_temperature = 81.9
    transition_energy_reference = -0.3424 + 1.838 * composition
    absorption_slope = 32670.0 * (1.0 + composition)
    absorption_reference = np.exp(-18.88 + 53.61 * composition)
    transition_absorption = (
        3.9778 * temperature
        - 566.4
        + (
            -0.020366 * temperature**2
            - 0.46742 * temperature
            + 3878.9
        )
        * composition
    )
    transition_energy = transition_energy_reference + (
        (temperature + transition_temperature)
        / absorption_slope
        * np.log(transition_absorption / absorption_reference)
    )
    below_gap_absorption = np.zeros_like(photon_energy)
    below_gap = photon_energy < transition_energy
    below_exponent = absorption_slope * (
        photon_energy[below_gap] - transition_energy_reference
    ) / (temperature + transition_temperature)
    below_gap_absorption[below_gap] = absorption_reference * np.exp(
        np.clip(below_exponent, -745.0, 700.0)
    )
    above_gap_absorption = np.zeros_like(photon_energy)
    above_gap = photon_energy > band_gap
    absorption_amplitude = (
        -20060.0
        + 115750.0 * composition
        + 32.43 * temperature
        - 64170.0 * composition**2
        + 0.43231 * temperature**2
        - 101.92 * composition * temperature
    )
    absorption_exponent = (
        0.74487
        - 0.44513 * composition
        + (0.000799 - 0.000757 * composition) * temperature
    )
    above_gap_absorption[above_gap] = (
        absorption_amplitude
        * np.maximum(photon_energy[above_gap] - band_gap, 0.0) ** absorption_exponent
        / photon_energy[above_gap]
    )
    fraction = np.clip(
        (photon_energy - band_gap) / (transition_energy - band_gap), 0.0, 1.0
    )
    fraction = fraction**2 * (3.0 - 2.0 * fraction)
    absorption = (
        (1.0 - fraction) * below_gap_absorption
        + fraction * above_gap_absorption
    )
    extinction_index = wavelengths_nm * 1.0e-7 * absorption / (4.0 * np.pi)
    return real_index, extinction_index


def qe_columns_for_scas(sca_names, qe_fieldnames):
    field_to_column = {
        str(name).strip().upper(): index
        for index, name in enumerate(qe_fieldnames)
    }
    columns = []
    for value in np.asarray(sca_names).reshape(-1):
        detector = str(value).strip().upper()
        if not detector.startswith("WFI") or not detector[3:].isdigit():
            raise ValueError(f"Expected a WFI detector label, got {value!r}")
        field = f"SCU {int(detector[3:])}"
        if field not in field_to_column:
            raise ValueError(f"QE table does not contain {field}")
        columns.append(field_to_column[field])
    return np.asarray(columns, dtype=int)


def load_response_context(
    sca_names,
    *,
    sed_weighting="energy",
    qe_response_mode="none",
):
    """Load an absolute-power response context with optional diagnostic weighting."""
    if qe_response_mode not in ("none", "clean_relative"):
        raise ValueError("QE response mode must be 'none' or 'clean_relative'")
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, qe_fieldnames = load_qe(QE_URL)
    model_wavelengths = np.unique(
        np.concatenate(
            [np.asarray(spectrum["wavelength"], dtype=float) for spectrum in spectra.values()]
        )
    )
    qe_columns = qe_columns_for_scas(sca_names, qe_fieldnames)
    if qe_response_mode == "clean_relative":
        integrations = [
            _prepare_response_integration(
                model_wavelengths,
                spectra,
                qe_wavelengths,
                qe_values=qe_values,
                qe_column=column,
                sed_weighting=sed_weighting,
                include_qe=True,
            )
            for column in qe_columns
        ]
    else:
        integration = _prepare_response_integration(
            model_wavelengths,
            spectra,
            qe_wavelengths,
            sed_weighting=sed_weighting,
        )
        integrations = [integration] * len(qe_columns)
    model_weights = np.stack([value[0] for value in integrations])
    active_wavelengths = np.any(model_weights != 0.0, axis=(0, 1))
    model_wavelengths = model_wavelengths[active_wavelengths]
    model_weights = model_weights[:, :, active_wavelengths]
    hgcdte_n, hgcdte_k = hgcdte_reference_optical_constants(model_wavelengths)
    return {
        "wavelengths": model_wavelengths,
        "model_weights": model_weights,
        "outside_output": np.stack([value[1] for value in integrations]),
        "clean_output": np.stack([value[2] for value in integrations]),
        "qe_columns": qe_columns,
        "qe_fieldnames": np.asarray(qe_fieldnames),
        "hgcdte_n": hgcdte_n,
        "hgcdte_k": hgcdte_k,
        "ice_n_modulation_basis": ice_n_modulation_basis(model_wavelengths),
        "spectral_modulation_basis": spectral_modulation_basis(model_wavelengths),
        "absolute_substrate_power": True,
        "sed_weighting": sed_weighting,
        "qe_response_mode": qe_response_mode,
        "normalize_clean_response": qe_response_mode == "clean_relative",
    }


def detector_name_from_statistics_path(path):
    return Path(path).name.split("_tile_log_statistics_", 1)[0].upper()


def load_tile_statistics(path, third_decon_day=None):
    """Load corrected tile sufficient statistics without applying corrections."""
    path = Path(path)
    if path.is_dir():
        raise ValueError("Production statistics must be one per-tile CSV file")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    tile_ids = np.asarray(sorted({int(row["tile_id"]) for row in rows}), dtype=int)
    if tile_ids.size != 1:
        raise ValueError(
            f"Production statistics must contain exactly one tile; found {tile_ids.tolist()}"
        )
    groups = sorted({int(row["group"]) for row in rows})
    if not groups:
        raise ValueError("No observation groups found")
    group_to_index = {group: index for index, group in enumerate(groups)}
    counts = np.zeros((1, len(groups)), dtype=float)
    sums = np.zeros_like(counts)
    sum_squared = np.zeros_like(counts)
    led = np.zeros(len(groups), dtype=int)
    repeat = np.zeros(len(groups), dtype=int)
    program = np.full(len(groups), -1, dtype=int)
    days = np.zeros(len(groups), dtype=float)
    exposure_times = [""] * len(groups)
    metadata = {}
    for row in rows:
        group = int(row["group"])
        group_index = group_to_index[group]
        counts[0, group_index] = float(row["n_pair_valid"])
        sums[0, group_index] = float(row["sum_log_change"])
        sum_squared[0, group_index] = float(row["sum_squared_log_change"])
        led[group_index] = int(row["led"])
        repeat[group_index] = int(row["repeat"])
        if row.get("program", ""):
            program[group_index] = int(row["program"])
        days[group_index] = float(row["days"])
        exposure_times[group_index] = row.get("exposure_start_time", "")
        tile_id = int(row["tile_id"])
        metadata[tile_id] = {
            "tile_row": int(row.get("tile_row", tile_id // 8)),
            "tile_col": int(row.get("tile_col", tile_id % 8)),
            "row_start": int(row.get("row_start", 0)),
            "row_stop_exclusive": int(row.get("row_stop_exclusive", 0)),
            "col_start": int(row.get("col_start", 0)),
            "col_stop_exclusive": int(row.get("col_stop_exclusive", 0)),
        }
    if np.any(counts <= 0.0):
        raise ValueError("Tile statistics contain empty observation groups")
    first_by_led = []
    for led_value in range(1, 7):
        indices = np.flatnonzero(led == led_value)
        if not indices.size:
            raise ValueError(f"Statistics contain no reference group for LED {led_value}")
        first_by_led.append(indices[0])
    first_by_led = np.asarray(first_by_led, dtype=int)
    repeat_values = np.asarray(sorted(set(repeat.tolist())), dtype=int)
    repeat_to_index = {value: index for index, value in enumerate(repeat_values)}
    repeat_index = np.asarray([repeat_to_index[value] for value in repeat], dtype=int)
    node_days = np.asarray(
        [np.mean(days[repeat == value]) for value in repeat_values], dtype=float
    )
    if node_days.size < 3 or np.any(np.diff(node_days) <= 0.0):
        raise ValueError("Repeat nodes must contain at least three increasing days")
    program_1063 = program == 1063
    reset_day = (
        float(np.min(days[program_1063]))
        if np.any(program_1063)
        else float(PRODUCTION_RESET_DAY)
    )
    reset_days = np.asarray([reset_day], dtype=float)
    if third_decon_day is not None:
        third_decon_day = float(third_decon_day)
        if not np.isfinite(third_decon_day):
            raise ValueError("third_decon_day must be finite")
        if third_decon_day <= reset_day:
            raise ValueError("third_decon_day must be after the physical reset day")
        reset_days = np.asarray([reset_day, third_decon_day], dtype=float)
    result = {
        "tile_ids": tile_ids,
        "metadata": metadata,
        "counts": counts,
        "sums": sums,
        "sum_squared": sum_squared,
        "groups": np.asarray(groups, dtype=int),
        "led": led,
        "repeat": repeat,
        "program": program,
        "days": days,
        "first_by_led": first_by_led,
        "exposure_times": exposure_times,
        "repeat_values": repeat_values,
        "repeat_index": repeat_index,
        "node_days": node_days,
        "reset_day": reset_day,
        "reset_days": reset_days,
        "sca_names": np.asarray([detector_name_from_statistics_path(path)]),
        "sca_index": np.asarray([0], dtype=int),
        "n_scas": 1,
        "n_tiles": 1,
    }
    if third_decon_day is not None:
        result["third_decon_day"] = third_decon_day
    return result


def default_likelihood_weight(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ValueError("Likelihood counts must be positive and finite")
    return 1.0 / float(np.median(counts))


def statistics_fingerprint(data):
    digest = hashlib.sha256()
    for key in ("counts", "sums", "sum_squared"):
        digest.update(np.ascontiguousarray(data[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def statistics_row_fingerprints(data):
    fingerprints = []
    for row in range(data["counts"].shape[0]):
        digest = hashlib.sha256()
        for key in ("counts", "sums", "sum_squared"):
            digest.update(np.ascontiguousarray(data[key][row], dtype=np.float64).tobytes())
        fingerprints.append(digest.hexdigest())
    return np.asarray(fingerprints)


def dielectric_ar_reference():
    indices = NOMINAL_HGCDTE[0] ** (
        np.arange(1, AR_LAYERS + 1, dtype=float) / (AR_LAYERS + 1)
    )
    thickness = AR_DESIGN_WAVELENGTH_NM / (4.0 * indices * 100.0)
    return np.column_stack((thickness, indices, np.full(AR_LAYERS, AR_DIELECTRIC_K_START))).reshape(-1)


def ar_prior_arrays():
    reference = dielectric_ar_reference()
    return reference, np.tile(AR_LOWER, AR_LAYERS), np.tile(AR_UPPER, AR_LAYERS)


def ar_active_parameter_indices():
    return np.arange(3 * AR_LAYERS, dtype=int)[np.arange(3 * AR_LAYERS) % 3 != 2]


def ar_reference_raw(
    ar_reference,
    ar_lower,
    ar_upper,
    active_parameter_indices,
):
    raw = []
    for parameter_index in np.asarray(active_parameter_indices, dtype=int):
        lower = float(ar_lower[parameter_index])
        if parameter_index % 3 == 1 and parameter_index >= 4:
            lower = float(ar_reference[parameter_index - 3])
        upper = float(ar_upper[parameter_index])
        probability = (ar_reference[parameter_index] - lower) / (upper - lower)
        if not 0.0 < probability < 1.0:
            raise ValueError("AR reference is outside its raw-coordinate bounds")
        raw.append(np.log(probability / (1.0 - probability)))
    return np.asarray(raw, dtype=float)


def ordered_ar_indices_from_raw(ar_raw, jax_module, jnp_module):
    _, ar_lower, ar_upper = ar_prior_arrays()
    active_parameter_indices = ar_active_parameter_indices()
    positions = {int(index): position for position, index in enumerate(active_parameter_indices)}
    values = jnp_module.zeros(
        ar_raw.shape[:-1] + (3 * AR_LAYERS,), dtype=ar_raw.dtype
    )
    for parameter_index in active_parameter_indices:
        if parameter_index % 3 == 2:
            continue
        if parameter_index % 3 == 1 and parameter_index >= 4:
            lower = values[..., parameter_index - 3]
        else:
            lower = jnp_module.asarray(ar_lower[parameter_index], dtype=ar_raw.dtype)
        upper = jnp_module.asarray(ar_upper[parameter_index], dtype=ar_raw.dtype)
        position = positions[int(parameter_index)]
        value = lower + (upper - lower) * jax_module.nn.sigmoid(ar_raw[..., position])
        values = values.at[..., parameter_index].set(value)
    return values


def _physical_ar_parameters(ar_raw, jax, jnp):
    return ordered_ar_indices_from_raw(ar_raw, jax, jnp)


def rate_curvature_matrix(node_days):
    node_days = np.asarray(node_days, dtype=float).reshape(-1)
    if node_days.size < 3 or np.any(np.diff(node_days) <= 0.0):
        raise ValueError("At least three increasing trajectory nodes are required")
    scaled = (node_days - node_days[0]) / np.median(np.diff(node_days))
    matrix = np.zeros((node_days.size - 2, node_days.size), dtype=float)
    for column in range(node_days.size):
        basis = np.zeros(node_days.size, dtype=float)
        basis[column] = 1.0
        slopes = np.diff(basis) / np.diff(scaled)
        matrix[:, column] = 2.0 * np.diff(slopes) / (scaled[2:] - scaled[:-2])
    return matrix


def _relative_entropy(log_values, log_reference, jnp):
    """Return the generalized KL penalty for positive values versus a prior."""
    log_ratio = log_values - log_reference
    ratio = jnp.exp(log_ratio)
    regular = ratio * log_ratio - jnp.expm1(log_ratio)
    quadratic = 0.5 * log_ratio**2
    return jnp.where(jnp.abs(log_ratio) < 1.0e-4, quadratic, regular)


def _signed_entropy_residual(log_values, log_reference, jnp):
    """Return a differentiable signed square root of twice the entropy."""
    log_ratio = log_values - log_reference
    entropy = _relative_entropy(log_values, log_reference, jnp)
    magnitude = jnp.sqrt(2.0 * jnp.maximum(entropy, 0.0))
    safe_abs_ratio = jnp.maximum(jnp.abs(log_ratio), 1.0e-12)
    return jnp.where(
        jnp.abs(log_ratio) < 1.0e-6,
        log_ratio,
        log_ratio * magnitude / safe_abs_ratio,
    )


def integrate_rate_trajectory(
    jax,
    jnp,
    days,
    node_days,
    integration_times,
    log_rates,
    *,
    extra_times=None,
):
    gauss_nodes = jnp.asarray(GAUSS_LEGENDRE_NODES)
    gauss_weights = jnp.asarray(GAUSS_LEGENDRE_WEIGHTS)
    days = jnp.asarray(days)
    node_days = jnp.asarray(node_days)
    integration_times = jnp.asarray(integration_times)

    def instantaneous_rate(time_value):
        return jnp.exp(jnp.interp(time_value, node_days, log_rates))

    def integrate_interval(left, right):
        left = jnp.asarray(left)
        right = jnp.asarray(right)
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        return half_width * jnp.sum(
            gauss_weights * instantaneous_rate(midpoint + half_width * gauss_nodes)
        )

    interval_integrals = jax.vmap(integrate_interval)(
        integration_times[:-1], integration_times[1:]
    )
    cumulative = jnp.concatenate((jnp.zeros(1), jnp.cumsum(interval_integrals)))

    def integrated_at(target):
        index = jnp.searchsorted(integration_times, target, side="right") - 1
        index = jnp.clip(index, 0, integration_times.size - 2)
        return cumulative[index] + integrate_interval(integration_times[index], target)

    result = (
        jax.vmap(integrated_at)(days),
        instantaneous_rate(days),
        jax.vmap(integrated_at)(node_days),
        instantaneous_rate(node_days),
    )
    if extra_times is None:
        return result
    return result + (jax.vmap(integrated_at)(jnp.asarray(extra_times)),)


def _incidence_sine(angle_deg, jnp):
    return jnp.sin(jnp.asarray(angle_deg) * jnp.pi / 180.0)


def incidence_pupil_quadrature(half_angle_deg):
    """Return directions and weights for a uniformly illuminated circular pupil."""
    half_angle_deg = float(half_angle_deg)
    if (
        not np.isfinite(half_angle_deg)
        or half_angle_deg < 0.0
        or half_angle_deg >= 90.0
    ):
        raise ValueError("Incidence pupil half-angle must be in [0, 90) degrees")
    if half_angle_deg == 0.0:
        return np.asarray([[0.0, 0.0, 1.0]]), np.asarray([1.0])

    # Uniform pupil area is uniform in normalized radius squared.
    radius_squared = 0.5 * (1.0 + PUPIL_RADIAL_NODES)
    radial_weights = 0.5 * PUPIL_RADIAL_WEIGHTS
    radius = np.sqrt(radius_squared)
    marginal_tangent = np.tan(np.deg2rad(half_angle_deg))
    polar_angle = np.arctan(radius * marginal_tangent)
    azimuth = 2.0 * np.pi * np.arange(PUPIL_AZIMUTH_COUNT) / PUPIL_AZIMUTH_COUNT
    sin_polar = np.sin(polar_angle)[:, None]
    directions = np.stack(
        (
            sin_polar * np.cos(azimuth)[None, :],
            sin_polar * np.sin(azimuth)[None, :],
            np.broadcast_to(
                np.cos(polar_angle)[:, None],
                sin_polar.shape[:-1] + (PUPIL_AZIMUTH_COUNT,),
            ),
        ),
        axis=-1,
    ).reshape((-1, 3))
    weights = np.repeat(radial_weights / PUPIL_AZIMUTH_COUNT, PUPIL_AZIMUTH_COUNT)
    return directions, weights


def build_prediction_function(
    data,
    context,
    *,
    incidence_pupil_half_angle_deg=0.0,
    fixed_normal_incidence=False,
    fit_ice_n_modulation=False,
    fit_ice_n_constant=False,
    ice_n_thickness_knots_nm=None,
    ice_n_thickness_spline_mode="natural",
    ice_n_porosity_parameters=None,
    fit_ice_k_constant=False,
    fit_ar_hgcdte_modulation=False,
    fit_led_log_offsets=False,
    normalize_clean_response=False,
):
    """Build the production TE/TM absolute-power predictor."""
    jax, jnp, _, _ = require_numpyro()
    pupil_directions, pupil_weights = incidence_pupil_quadrature(
        incidence_pupil_half_angle_deg
    )
    pupil_mode = float(incidence_pupil_half_angle_deg) > 0.0
    pupil_local_x = jnp.asarray(pupil_directions[:, 0])
    pupil_local_y = jnp.asarray(pupil_directions[:, 1])
    pupil_local_z = jnp.asarray(pupil_directions[:, 2])
    pupil_weights = jnp.asarray(pupil_weights)
    pupil_ray_count = int(pupil_directions.shape[0])
    wavelengths = jnp.asarray(context["wavelengths"])
    phase_factor = 2.0 * jnp.pi / wavelengths
    all_sca_index = np.asarray(data["sca_index"], dtype=int)
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1)
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    nonreference_group = np.ones(len(data["led"]), dtype=bool)
    nonreference_group[np.asarray(data["first_by_led"], dtype=int)] = False
    nonreference_group = jnp.asarray(nonreference_group)
    selected_weights = jnp.asarray(context["model_weights"])[all_sca_index]
    selected_weights = selected_weights[:, led_index, :]
    baseline_counts = np.asarray(data["counts"], dtype=float)
    baseline_sums = np.asarray(data["sums"], dtype=float)
    baseline_offsets = np.asarray(
        [
            np.average(
                baseline_sums[:, group] / baseline_counts[:, group],
                weights=baseline_counts[:, group],
            )
            for group in np.asarray(data["first_by_led"], dtype=int)
        ],
        dtype=float,
    )
    baseline_offsets = jnp.asarray(baseline_offsets)
    reference_wavelength_count = wavelengths.size
    base_n = jnp.asarray(context["hgcdte_n"])
    base_k = jnp.asarray(context["hgcdte_k"])
    ice_modulation_basis_value = None
    spectral_modulation_basis_value = None
    ice_n_thickness_knots = None
    ice_n_porosity_parameters = validate_ice_n_porosity_parameters(
        ice_n_porosity_parameters
    )
    if ice_n_thickness_spline_mode not in ("natural", "monotonic"):
        raise ValueError(
            "Thickness-dependent ice n spline mode must be 'natural' or 'monotonic'"
        )
    if ice_n_thickness_knots_nm is not None:
        ice_n_thickness_knots = np.asarray(
            ice_n_thickness_knots_nm, dtype=float
        ).reshape(-1)
        if (
            ice_n_thickness_knots.size < 2
            or np.any(~np.isfinite(ice_n_thickness_knots))
            or np.any(np.diff(ice_n_thickness_knots) <= 0.0)
        ):
            raise ValueError(
                "Thickness-dependent ice n knots must be finite and strictly increasing"
            )
    if sum(
        bool(value)
        for value in (
            fit_ice_n_modulation,
            fit_ice_n_constant,
            ice_n_thickness_knots is not None,
            ice_n_porosity_parameters is not None,
        )
    ) > 1:
        raise ValueError("Ice n wavelength and thickness models are mutually exclusive")
    if fit_ice_n_modulation:
        if "ice_n_modulation_basis" not in context:
            raise ValueError("Ice n modulation requires a wavelength basis")
        ice_modulation_basis_value = jnp.asarray(
            context["ice_n_modulation_basis"]
        )
    if fit_ar_hgcdte_modulation:
        if "spectral_modulation_basis" not in context:
            raise ValueError("AR/HgCdTe modulation requires a wavelength basis")
        spectral_modulation_basis_value = jnp.asarray(
            context["spectral_modulation_basis"]
        )

    def layer_elements(index, thickness, polarization, angle_sine):
        if index.ndim == 2:
            index = index[:, :, None]
        cosine = jnp.sqrt(1.0 - (angle_sine[:, None, None] / index) ** 2)
        admittance = index * cosine if polarization == "s" else index / cosine
        phase = (
            phase_factor[None, None, :]
            * index
            * thickness[:, :, None]
            * cosine
        )
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        return (
            cosine_phase,
            sine_phase / admittance,
            sine_phase * admittance,
            cosine_phase,
        )

    def polarization_response(
        ar_parameters,
        hgcdte_n,
        hgcdte_k,
        ice_thickness,
        angle_sine,
        polarization,
        ice_index,
        *,
        ar_index=None,
    ):
        ar_parameters = ar_parameters.reshape((ar_parameters.shape[0], AR_LAYERS, 3))
        if ar_index is None:
            ar_index = ar_parameters[:, :, 1] + 1j * ar_parameters[:, :, 2]
        # Serialized AR thickness coordinates use 100 nm as their unit.
        ar_thickness = 100.0 * ar_parameters[:, :, 0]
        substrate = hgcdte_n + 1j * hgcdte_k
        substrate_cosine = jnp.sqrt(1.0 - (angle_sine[:, None] / substrate) ** 2)
        substrate_admittance_value = (
            substrate * substrate_cosine
            if polarization == "s"
            else substrate / substrate_cosine
        )
        incident_cosine = jnp.sqrt(1.0 - angle_sine**2)
        incident_admittance = (
            incident_cosine if polarization == "s" else 1.0 / incident_cosine
        )
        boundary_first = jnp.ones((ar_parameters.shape[0], reference_wavelength_count), dtype=jnp.complex128)
        boundary_second = substrate_admittance_value
        for layer_index in range(AR_LAYERS - 1, -1, -1):
            elements = layer_elements(
                ar_index[:, layer_index : layer_index + 1],
                ar_thickness[:, layer_index : layer_index + 1],
                polarization,
                angle_sine,
            )
            old_first, old_second = boundary_first, boundary_second
            boundary_first = elements[0][:, 0, :] * old_first + elements[1][:, 0, :] * old_second
            boundary_second = elements[2][:, 0, :] * old_first + elements[3][:, 0, :] * old_second
        clean_denominator = incident_admittance[:, None] * boundary_first + boundary_second
        clean = (
            jnp.real(substrate_admittance_value / incident_admittance[:, None])
            * jnp.abs(2.0 * incident_admittance[:, None] / clean_denominator) ** 2
        )

        ice_cosine = jnp.sqrt(1.0 - (angle_sine[:, None, None] / ice_index) ** 2)
        ice_admittance = (
            ice_index * ice_cosine if polarization == "s" else ice_index / ice_cosine
        )
        ice_phase = (
            phase_factor[None, None, :]
            * ice_thickness[:, :, None]
            * ice_index
            * ice_cosine
        )
        ice_cosine_phase = jnp.cos(ice_phase)
        ice_sine_phase = -1j * jnp.sin(ice_phase)
        ice_m01 = ice_sine_phase / ice_admittance
        ice_m10 = ice_sine_phase * ice_admittance
        coated_first = ice_cosine_phase * boundary_first[:, None, :] + ice_m01 * boundary_second[:, None, :]
        coated_second = ice_m10 * boundary_first[:, None, :] + ice_cosine_phase * boundary_second[:, None, :]
        coated_denominator = incident_admittance[:, None, None] * coated_first + coated_second
        coated = (
            jnp.real(substrate_admittance_value[:, None, :] / incident_admittance[:, None, None])
            * jnp.abs(2.0 * incident_admittance[:, None, None] / coated_denominator) ** 2
        )
        return clean, coated

    def prediction(
        ice_thickness,
        ar_parameters,
        hgcdte_parameters,
        incidence_angle_deg,
        ice_n_log_modulation=None,
        ice_n_log_constant=None,
        ice_n_log_values=None,
        ice_n_log_thickness_knots=None,
        ice_k_constant=None,
        ar_n_log_modulation=None,
        hgcdte_n_log_modulation=None,
        hgcdte_k_log_modulation=None,
        led_log_offsets=None,
    ):
        ar_parameters = jnp.asarray(ar_parameters)
        if ar_parameters.ndim == 1:
            ar_parameters = ar_parameters[None, :]
        ice_thickness = jnp.asarray(ice_thickness)
        if ice_thickness.ndim == 1:
            ice_thickness = jnp.broadcast_to(ice_thickness, (ar_parameters.shape[0], ice_thickness.shape[0]))
        hgcdte_parameters = jnp.asarray(hgcdte_parameters)
        if hgcdte_parameters.ndim == 1:
            hgcdte_parameters = jnp.broadcast_to(hgcdte_parameters, (ar_parameters.shape[0], 2))
        if fixed_normal_incidence:
            angle_sine = jnp.zeros((ar_parameters.shape[0],), dtype=jnp.float64)
        else:
            angle_sine = jnp.broadcast_to(
                _incidence_sine(incidence_angle_deg, jnp), (ar_parameters.shape[0],)
            )
        if ice_n_porosity_parameters is not None:
            p0, pinf, length_nm, n_solid = ice_n_porosity_parameters
            porosity = pinf - (pinf - p0) * jnp.exp(
                -jnp.maximum(ice_thickness, 0.0) / length_nm
            )
            solid_lorentz_lorenz = (n_solid**2 - 1.0) / (n_solid**2 + 2.0)
            effective_lorentz_lorenz = (1.0 - porosity) * solid_lorentz_lorenz
            effective_n = jnp.sqrt(
                (1.0 + 2.0 * effective_lorentz_lorenz)
                / (1.0 - effective_lorentz_lorenz)
            )
            ice_index = effective_n[..., None].astype(jnp.complex128)
        elif fit_ice_n_constant:
            if ice_n_log_constant is None:
                raise ValueError("Constant ice n values are required")
            ice_index = (
                ICE_INDEX_DEFAULT
                * jnp.exp(jnp.asarray(ice_n_log_constant).reshape((-1, 1, 1)))
            ).astype(jnp.complex128)
        elif fit_ice_n_modulation:
            if ice_n_log_modulation is None:
                raise ValueError("Ice n modulation values are required")
            modulation = jnp.asarray(ice_n_log_modulation).reshape(
                (ar_parameters.shape[0], ice_modulation_basis_value.shape[1])
            )
            ice_index = (
                ICE_INDEX_DEFAULT
                * jnp.exp(modulation @ ice_modulation_basis_value.T)
            )[:, None, :].astype(jnp.complex128)
        elif ice_n_log_values is not None:
            ice_index = (
                ICE_INDEX_DEFAULT
                * jnp.exp(jnp.asarray(ice_n_log_values).reshape((ar_parameters.shape[0], -1)))
            )[..., None].astype(jnp.complex128)
        elif ice_n_thickness_knots is not None:
            if ice_n_log_thickness_knots is None:
                raise ValueError("Thickness-dependent ice n values are required")
            knot_log_values = jnp.asarray(ice_n_log_thickness_knots).reshape(
                (ar_parameters.shape[0], ice_n_thickness_knots.size)
            )
            knot_values = ICE_INDEX_DEFAULT * jnp.exp(knot_log_values)
            spline_values = (
                monotonic_cubic_spline_values
                if ice_n_thickness_spline_mode == "monotonic"
                else natural_cubic_spline_values
            )
            thickness_values = spline_values(
                ice_n_thickness_knots,
                knot_values,
                ice_thickness,
                jnp,
            )
            ice_index = thickness_values[..., None].astype(jnp.complex128)
        else:
            ice_index = jnp.asarray(ICE_INDEX_DEFAULT + 0.0j)
        if fit_ice_k_constant:
            if ice_k_constant is None:
                raise ValueError("Constant ice k values are required")
            ice_index = ice_index + 1j * jnp.asarray(ice_k_constant).reshape(
                (-1, 1, 1)
            )
        if fit_ar_hgcdte_modulation:
            if (
                ar_n_log_modulation is None
                or hgcdte_n_log_modulation is None
                or hgcdte_k_log_modulation is None
            ):
                raise ValueError("AR/HgCdTe modulation values are required")
            ar_modulation = jnp.exp(
                jnp.einsum(
                    "bln,wn->blw",
                    ar_n_log_modulation,
                    spectral_modulation_basis_value,
                )
            )
            ar_parameters_reshaped = ar_parameters.reshape(
                (ar_parameters.shape[0], AR_LAYERS, 3)
            )
            ar_index = (
                ar_parameters_reshaped[:, :, 1, None] * ar_modulation
                + 1j * ar_parameters_reshaped[:, :, 2, None]
            )
            hgcdte_n = base_n[None, :] * jnp.exp(
                hgcdte_parameters[:, 0, None]
                + jnp.einsum(
                    "bn,wn->bw",
                    hgcdte_n_log_modulation,
                    spectral_modulation_basis_value,
                )
            )
            hgcdte_k = base_k[None, :] * jnp.exp(
                hgcdte_parameters[:, 1, None]
                + jnp.einsum(
                    "bn,wn->bw",
                    hgcdte_k_log_modulation,
                    spectral_modulation_basis_value,
                )
            )
        else:
            ar_parameters_reshaped = ar_parameters.reshape(
                (ar_parameters.shape[0], AR_LAYERS, 3)
            )
            ar_index = (
                ar_parameters_reshaped[:, :, 1]
                + 1j * ar_parameters_reshaped[:, :, 2]
            )
            hgcdte_n = base_n[None, :] * jnp.exp(hgcdte_parameters[:, 0, None])
            hgcdte_k = base_k[None, :] * jnp.exp(hgcdte_parameters[:, 1, None])

        response_ar_parameters = ar_parameters
        response_hgcdte_n = hgcdte_n
        response_hgcdte_k = hgcdte_k
        response_ice_thickness = ice_thickness
        response_ice_index = ice_index
        response_angle_sine = angle_sine
        response_ar_index = ar_index
        if pupil_mode:
            # Rotate the chief-ray-local pupil cone into the x-z chief-ray plane.
            angle_rad = jnp.asarray(incidence_angle_deg) * (jnp.pi / 180.0)
            sine_angle = jnp.sin(angle_rad)
            cosine_angle = jnp.cos(angle_rad)
            direction_x = pupil_local_x * cosine_angle + pupil_local_z * sine_angle
            direction_y = pupil_local_y
            response_angle_sine = jnp.broadcast_to(
                jnp.sqrt(
                    jnp.maximum(
                        direction_x * direction_x + direction_y * direction_y,
                        0.0,
                    )
                ),
                (ar_parameters.shape[0], pupil_ray_count),
            ).reshape(-1)
            response_ar_parameters = jnp.repeat(
                ar_parameters, pupil_ray_count, axis=0
            )
            response_hgcdte_n = jnp.repeat(hgcdte_n, pupil_ray_count, axis=0)
            response_hgcdte_k = jnp.repeat(hgcdte_k, pupil_ray_count, axis=0)
            response_ice_thickness = jnp.repeat(
                ice_thickness, pupil_ray_count, axis=0
            )
            response_ar_index = jnp.repeat(ar_index, pupil_ray_count, axis=0)
            if (
                fit_ice_n_modulation
                or fit_ice_n_constant
                or ice_n_log_values is not None
                or ice_n_thickness_knots is not None
                or fit_ice_k_constant
            ):
                response_ice_index = jnp.repeat(
                    ice_index, pupil_ray_count, axis=0
                )

        clean_s, coated_s = polarization_response(
            response_ar_parameters,
            response_hgcdte_n,
            response_hgcdte_k,
            response_ice_thickness,
            response_angle_sine,
            "s",
            response_ice_index,
            ar_index=response_ar_index,
        )
        if fixed_normal_incidence:
            # At exactly normal incidence TE and TM admittances are identical.
            # Avoid evaluating the same transfer matrix a second time.
            clean = clean_s
            coated = coated_s
        else:
            clean_p, coated_p = polarization_response(
                response_ar_parameters,
                response_hgcdte_n,
                response_hgcdte_k,
                response_ice_thickness,
                response_angle_sine,
                "p",
                response_ice_index,
                ar_index=response_ar_index,
            )
            clean = 0.5 * (clean_s + clean_p)
            coated = 0.5 * (coated_s + coated_p)
        if pupil_mode:
            row_count = ar_parameters.shape[0]
            group_count = ice_thickness.shape[1]
            clean = jnp.sum(
                clean.reshape((row_count, pupil_ray_count, reference_wavelength_count))
                * pupil_weights[None, :, None],
                axis=1,
            )
            coated = jnp.sum(
                coated.reshape(
                    (row_count, pupil_ray_count, group_count, reference_wavelength_count)
                )
                * pupil_weights[None, :, None, None],
                axis=1,
            )
        if normalize_clean_response:
            relative = coated / clean[:, None, :]
            integrated = jnp.sum(relative * selected_weights, axis=-1)
            current = integrated / jnp.sum(selected_weights, axis=-1)
        else:
            integrated_clean = jnp.sum(clean[:, None, :] * selected_weights, axis=-1)
            integrated = jnp.sum(coated * selected_weights, axis=-1)
            current = integrated / integrated_clean
        reference = current[:, first_by_led]
        baseline = reference[:, led_index]
        fitted = jnp.log(current / baseline) + baseline_offsets[led_index][None, :]
        if fit_led_log_offsets:
            if led_log_offsets is None:
                raise ValueError("LED log offsets are required")
            offsets = jnp.asarray(led_log_offsets)
            fitted = fitted + offsets[
                jnp.asarray(all_sca_index)[:, None], led_index[None, :]
            ] * nonreference_group[None, :]
        return fitted

    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    reset_days = reset_days_for_data(data)
    reset_mask = jnp.asarray(
        np.asarray(data["days"], dtype=float)[None, :] >= reset_days[:, None]
    )

    def residual_sum_squares(
        ice_thickness,
        ar_parameters,
        hgcdte_parameters,
        incidence_angle_deg,
        post_reset_tile_log_offsets=None,
        ice_n_log_modulation=None,
        ice_n_log_constant=None,
        ice_n_log_values=None,
        ice_n_log_thickness_knots=None,
        ice_k_constant=None,
        ar_n_log_modulation=None,
        hgcdte_n_log_modulation=None,
        hgcdte_k_log_modulation=None,
        led_log_offsets=None,
        return_prediction=False,
    ):
        fitted = prediction(
            ice_thickness,
            ar_parameters,
            hgcdte_parameters,
            incidence_angle_deg,
            ice_n_log_modulation=ice_n_log_modulation,
            ice_n_log_constant=ice_n_log_constant,
            ice_n_log_values=ice_n_log_values,
            ice_n_log_thickness_knots=ice_n_log_thickness_knots,
            ice_k_constant=ice_k_constant,
            ar_n_log_modulation=ar_n_log_modulation,
            hgcdte_n_log_modulation=hgcdte_n_log_modulation,
            hgcdte_k_log_modulation=hgcdte_k_log_modulation,
            led_log_offsets=led_log_offsets,
        )
        observed = sums / jnp.maximum(counts, 1.0)
        if post_reset_tile_log_offsets is not None:
            offsets = jnp.asarray(post_reset_tile_log_offsets).reshape(-1)
            if offsets.size == reset_mask.shape[0] * counts.shape[0]:
                offsets = offsets.reshape((reset_mask.shape[0], counts.shape[0]))
            elif reset_mask.shape[0] == 1 and offsets.size == counts.shape[0]:
                offsets = offsets.reshape((1, counts.shape[0]))
            else:
                raise ValueError("Reset offset values do not match the reset layout")
            observed = observed + jnp.sum(
                offsets[:, :, None] * reset_mask[:, None, :], axis=0
            )
        safe_counts = jnp.maximum(counts, 1.0)
        within_group = sum_squared - sums * sums / safe_counts
        total = jnp.sum(within_group + counts * (fitted - observed) ** 2)
        if return_prediction:
            return total, fitted
        return total

    prediction.residual_sum_squares = residual_sum_squares
    return prediction


def _trajectory_state(
    data,
    log_rate_nodes,
    initial_thickness,
    reset_initial_thickness,
    jax,
    jnp,
    *,
    initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
):
    days = jnp.asarray(data["days"])
    node_days = jnp.asarray(data["node_days"])
    reset_days = reset_days_for_data(data)
    integration_times = np.unique(
        np.concatenate(
            (
                [float(np.min(data["days"]))],
                np.asarray(data["node_days"], dtype=float),
                [float(np.max(data["days"]))],
                reset_days,
            )
        )
    )
    integration_times = integration_times[(integration_times >= np.min(data["days"])) & (integration_times <= np.max(data["days"]))]
    profiles = jnp.asarray(log_rate_nodes).reshape((-1, len(data["node_days"])))
    reset_times = jnp.asarray(reset_days)
    integrated, rates, node_integrated, node_rates, reset_integrated = jax.vmap(
        lambda profile: integrate_rate_trajectory(
            jax,
            jnp,
            days,
            node_days,
            jnp.asarray(integration_times),
            profile,
            extra_times=reset_times,
        )
    )(profiles)
    prior, _, _ = _initial_thickness_settings(
        initial_thickness_prior,
        INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
    )
    if prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL:
        initial = jnp.exp(jnp.asarray(initial_thickness)).reshape(-1)
        reset_initial = jnp.exp(jnp.asarray(reset_initial_thickness))
    else:
        initial = jnp.asarray(initial_thickness).reshape(-1)
        reset_initial = jnp.asarray(reset_initial_thickness)
    if reset_initial.ndim == 1:
        reset_initial = reset_initial[None, :]
    reset_initial = reset_initial.reshape((len(reset_days), -1))
    ice = initial[:, None] + integrated
    node_ice = initial[:, None] + node_integrated
    for reset_index, reset_day in enumerate(reset_days):
        post = days >= reset_day
        integrated_since_reset = integrated - reset_integrated[:, reset_index, None]
        node_integrated_since_reset = (
            node_integrated - reset_integrated[:, reset_index, None]
        )
        ice = jnp.where(
            post[None, :],
            reset_initial[reset_index, :, None] + integrated_since_reset,
            ice,
        )
        node_ice = jnp.where(
            node_days[None, :] >= reset_day,
            reset_initial[reset_index, :, None] + node_integrated_since_reset,
            node_ice,
        )
    return ice, rates, node_ice, node_rates


def _direct_thickness_state(data, log_thickness_nodes, jax, jnp):
    """Evaluate a piecewise-linear direct-thickness trajectory."""
    days = jnp.asarray(data["days"])
    node_days = np.asarray(data["node_days"], dtype=float)
    reset_days = reset_days_for_data(data)
    observation_min = float(np.min(data["days"]))
    observation_max = float(np.max(data["days"]))
    active_reset_days = reset_days[
        (reset_days > observation_min) & (reset_days < observation_max)
    ]
    boundaries = np.concatenate(([observation_min], active_reset_days, [observation_max]))
    node_indices = []
    for index in range(boundaries.size - 1):
        left = boundaries[index]
        right = boundaries[index + 1]
        if index == boundaries.size - 2:
            mask = (node_days >= left) & (node_days <= right)
        else:
            mask = (node_days >= left) & (node_days < right)
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            raise ValueError(
                "Direct thickness requires at least two nodes in each reset segment"
            )
        node_indices.append(indices)

    profiles = jnp.exp(jnp.asarray(log_thickness_nodes)).reshape(
        (-1, len(node_days))
    )
    ice = jnp.zeros((profiles.shape[0], days.size))
    node_rates = jnp.zeros_like(profiles)
    rates = jnp.zeros((profiles.shape[0], days.size))

    def node_slopes(values, times):
        slopes = jnp.diff(values) / jnp.diff(times)
        return jnp.concatenate((slopes, slopes[:, -1:]), axis=1)

    for index, indices in enumerate(node_indices):
        times = jnp.asarray(node_days[indices])
        values = profiles[:, indices]
        segment_ice = jax.vmap(lambda profile: jnp.interp(days, times, profile))(values)
        segment_rates = node_slopes(values, times)
        segment_observation_rates = jax.vmap(
            lambda profile: jnp.interp(days, times, profile)
        )(segment_rates)
        left = boundaries[index]
        right = boundaries[index + 1]
        if index == len(node_indices) - 1:
            observation_mask = (np.asarray(data["days"]) >= left) & (
                np.asarray(data["days"]) <= right
            )
        else:
            observation_mask = (np.asarray(data["days"]) >= left) & (
                np.asarray(data["days"]) < right
            )
        observation_mask = jnp.asarray(observation_mask)
        ice = jnp.where(observation_mask[None, :], segment_ice, ice)
        rates = jnp.where(
            observation_mask[None, :], segment_observation_rates, rates
        )
        node_rates = node_rates.at[:, indices].set(segment_rates)
    node_ice = profiles
    return ice, rates, node_ice, node_rates


def _least_squares_prior_residuals(
    values,
    data,
    jnp,
    *,
    smoothness_weight=None,
    entropy_reference_log_rates=None,
    entropy_weight=0.0,
    rate_prior_log_mean=None,
    trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    trajectory_parameterization="rate",
    initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
    initial_thickness_prior_sigma_nm=INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
    thickness_prior_centers=None,
    thickness_prior_sigmas=None,
    absolute_hgcdte_n_scale_prior_sigma_log=ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
    fixed_absolute_hgcdte_n_log_scale=None,
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    ice_n_constant_reference=ICE_INDEX_DEFAULT,
    ice_k_prior_sigma=ICE_K_PRIOR_SIGMA,
    ar_k_prior_sigma=AR_K_PRIOR_SIGMA,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_prior_center_raw=None,
    ar_prior_sigma_raw=None,
    fit_ar_hgcdte_modulation=False,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    hgcdte_n_modulation_prior_sigma_log=None,
    led_log_offset_prior_sigma_log=LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    if smoothness_weight is None:
        smoothness_weight = NP_SMOOTHNESS_WEIGHT
    reference, lower, upper = ar_prior_arrays()
    active = ar_active_parameter_indices()
    reference_raw = jnp.asarray(ar_reference_raw(reference, lower, upper, active))
    if ar_prior_center_raw is None:
        ar_prior_center_raw = reference_raw
    if ar_prior_sigma_raw is None:
        ar_prior_sigma_raw = np.full(len(active), ar_mean_prior_sigma, dtype=float)
    residuals = []

    def add_normal(name, center, scale):
        residuals.append(
            ((jnp.asarray(values[name]) - jnp.asarray(center)) / jnp.asarray(scale)).reshape(-1)
        )

    add_optical_normal = add_normal

    add_optical_normal("ar_tile_raw", ar_prior_center_raw, ar_prior_sigma_raw)
    if trajectory_parameterization == "rate":
        initial_thickness_prior, initial_thickness_prior_sigma_nm, _ = (
            _initial_thickness_settings(
                initial_thickness_prior,
                initial_thickness_prior_sigma_nm,
            )
        )
        thickness_names, offset_names = reset_site_names(
            data, initial_thickness_prior
        )
        thickness_prior_centers = thickness_prior_centers or {}
        thickness_prior_sigmas = thickness_prior_sigmas or {}
        if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL:
            for name in thickness_names:
                add_normal(
                    name,
                    thickness_prior_centers.get(
                        name, np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM)
                    ),
                    thickness_prior_sigmas.get(name, INITIAL_THICKNESS_LOG_SIGMA),
                )
        else:
            for name in thickness_names:
                add_normal(
                    name,
                    thickness_prior_centers.get(name, 0.0),
                    thickness_prior_sigmas.get(name, initial_thickness_prior_sigma_nm),
                )
        if rate_prior_log_mean is None:
            rate_prior_log_mean = np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)
        add_normal(
            "np_log_rate_nodes",
            rate_prior_log_mean,
            trajectory_rate_prior_sigma_log,
        )
    _, offset_names = reset_site_names(
        data,
        initial_thickness_prior,
    )
    for name in offset_names:
        add_normal(name, 0.0, POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA)
    if "led_log_offsets" in values:
        add_optical_normal("led_log_offsets", 0.0, led_log_offset_prior_sigma_log)
    if fixed_absolute_hgcdte_n_log_scale is None:
        add_optical_normal(
            "absolute_hgcdte_n_log_scale",
            0.0,
            absolute_hgcdte_n_scale_prior_sigma_log,
        )
    add_optical_normal("absolute_hgcdte_k_log_scale", 0.0, ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)
    if "ice_n_log_modulation" in values:
        add_optical_normal(
            "ice_n_log_modulation",
            0.0,
            ice_n_modulation_prior_sigma_log,
        )
    if "ice_n_log_constant" in values:
        add_optical_normal(
            "ice_n_log_constant",
            np.log(ice_n_constant_reference / ICE_INDEX_DEFAULT),
            ice_n_constant_prior_sigma_log,
        )
    if "ice_k_constant" in values:
        add_optical_normal("ice_k_constant", 0.0, ice_k_prior_sigma)
    if "ar_k_constant" in values:
        add_optical_normal("ar_k_constant", 0.0, ar_k_prior_sigma)
    if fit_ar_hgcdte_modulation:
        add_optical_normal(
            "ar_n_log_modulation",
            0.0,
            ar_hgcdte_modulation_prior_sigma_log,
        )
        add_optical_normal(
            "hgcdte_n_log_modulation",
            0.0,
            hgcdte_n_modulation_prior_sigma_log,
        )
        add_optical_normal(
            "hgcdte_k_log_modulation",
            0.0,
            ar_hgcdte_modulation_prior_sigma_log,
        )
    likelihood_scale = (
        0.5
        * default_likelihood_weight(data["counts"])
        / int(data["n_tiles"])
        * float(np.sum(data["sum_squared"]))
        / LOG_NOISE_SIGMA**2
    )
    if trajectory_parameterization == "rate" and smoothness_weight:
        curvature_matrix = rate_curvature_matrix(data["node_days"])
        curvature = jnp.einsum(
            "ij,tkj->tki", jnp.asarray(curvature_matrix), jnp.asarray(values["np_log_rate_nodes"])
        ).reshape(-1)
        residuals.append(
            curvature
            * jnp.sqrt(2.0 * likelihood_scale * smoothness_weight / curvature.size)
        )
    if trajectory_parameterization == "rate" and entropy_reference_log_rates is not None and entropy_weight:
        entropy_reference_log_rates = jnp.asarray(entropy_reference_log_rates)
        entropy_residual = _signed_entropy_residual(
            jnp.asarray(values["np_log_rate_nodes"]),
            entropy_reference_log_rates,
            jnp,
        ).reshape(-1)
        residuals.append(
            entropy_residual
            * jnp.sqrt(likelihood_scale * entropy_weight / entropy_residual.size)
        )
    return jnp.concatenate(residuals)


def build_model(
    data,
    context,
    *,
    smoothness_weight=None,
    entropy_reference_log_rates=None,
    entropy_weight=0.0,
    rate_prior_log_mean=None,
    trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    trajectory_parameterization="rate",
    initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
    initial_thickness_prior_sigma_nm=INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
    thickness_prior_centers=None,
    thickness_prior_sigmas=None,
    absolute_hgcdte_n_scale_prior_sigma_log=ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
    fixed_absolute_hgcdte_n_log_scale=None,
    constant_rate=False,
    fit_incidence_angle=True,
    incidence_pupil_half_angle_deg=0.0,
    fit_ice_n_modulation=False,
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    fit_ice_n_constant=False,
    ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    ice_n_constant_reference=ICE_INDEX_DEFAULT,
    ice_n_porosity_parameters=None,
    fit_ice_k_constant=False,
    ice_k_prior_sigma=ICE_K_PRIOR_SIGMA,
    fit_ar_k_constant=False,
    ar_k_prior_sigma=AR_K_PRIOR_SIGMA,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_prior_center_raw=None,
    ar_prior_sigma_raw=None,
    fixed_ar_base_n=None,
    fit_ar_hgcdte_modulation=False,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    hgcdte_n_modulation_prior_sigma_log=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    normalize_clean_response=False,
):
    """Build the independent-tile NumPyro density.

    The default arguments preserve the legacy curvature-regularized model.
    Production MaxEnt fits pass the regularization arguments explicitly.
    """
    if not context.get("absolute_substrate_power", False):
        raise ValueError("Production requires an absolute-power response context")
    if trajectory_parameterization not in ("rate", "thickness"):
        raise ValueError("Trajectory parameterization must be 'rate' or 'thickness'")
    if trajectory_parameterization == "thickness" and constant_rate:
        raise ValueError("Direct thickness cannot be combined with constant_rate")
    if trajectory_parameterization == "thickness" and (smoothness_weight or entropy_weight):
        raise ValueError("Direct thickness diagnostics require zero trajectory regularization")
    if smoothness_weight is None:
        smoothness_weight = NP_SMOOTHNESS_WEIGHT
    if smoothness_weight < 0.0 or entropy_weight < 0.0:
        raise ValueError("Regularization weights must be non-negative")
    initial_thickness_prior, initial_thickness_prior_sigma_nm, _ = (
        _initial_thickness_settings(
            initial_thickness_prior,
            initial_thickness_prior_sigma_nm,
        )
    )
    thickness_names, offset_names = reset_site_names(
        data, initial_thickness_prior
    )
    if (
        not np.isfinite(trajectory_rate_prior_sigma_log)
        or trajectory_rate_prior_sigma_log <= 0.0
    ):
        raise ValueError("Trajectory-rate prior sigma must be positive and finite")
    if (
        not np.isfinite(absolute_hgcdte_n_scale_prior_sigma_log)
        or absolute_hgcdte_n_scale_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "HgCdTe n scale prior sigma must be positive and finite"
        )
    if fixed_absolute_hgcdte_n_log_scale is not None:
        fixed_absolute_hgcdte_n_log_scale = float(
            fixed_absolute_hgcdte_n_log_scale
        )
        if not np.isfinite(fixed_absolute_hgcdte_n_log_scale):
            raise ValueError("Fixed HgCdTe n scale must be finite")
    if (
        not np.isfinite(ice_n_modulation_prior_sigma_log)
        or ice_n_modulation_prior_sigma_log <= 0.0
    ):
        raise ValueError("Ice n modulation prior sigma must be positive and finite")
    if (
        not np.isfinite(ice_n_constant_prior_sigma_log)
        or ice_n_constant_prior_sigma_log <= 0.0
    ):
        raise ValueError("Constant ice n prior sigma must be positive and finite")
    if not np.isfinite(ice_n_constant_reference) or ice_n_constant_reference <= 0.0:
        raise ValueError("Constant ice n prior reference must be positive and finite")
    ice_n_porosity_parameters = validate_ice_n_porosity_parameters(
        ice_n_porosity_parameters
    )
    if not np.isfinite(ice_k_prior_sigma) or ice_k_prior_sigma <= 0.0:
        raise ValueError("Constant ice k prior sigma must be positive and finite")
    if not np.isfinite(ar_k_prior_sigma) or ar_k_prior_sigma <= 0.0:
        raise ValueError("Constant AR k prior sigma must be positive and finite")
    if sum(
        bool(value)
        for value in (
            fit_ice_n_modulation,
            fit_ice_n_constant,
            ice_n_porosity_parameters is not None,
        )
    ) > 1:
        raise ValueError("Ice n models are mutually exclusive")
    if (
        not np.isfinite(ar_hgcdte_modulation_prior_sigma_log)
        or ar_hgcdte_modulation_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "AR/HgCdTe modulation prior sigma must be positive and finite"
        )
    if hgcdte_n_modulation_prior_sigma_log is None:
        hgcdte_n_modulation_prior_sigma_log = ar_hgcdte_modulation_prior_sigma_log
    if (
        not np.isfinite(hgcdte_n_modulation_prior_sigma_log)
        or hgcdte_n_modulation_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "HgCdTe n modulation prior sigma must be positive and finite"
        )
    if (
        not np.isfinite(led_log_offset_prior_sigma_log)
        or led_log_offset_prior_sigma_log <= 0.0
    ):
        raise ValueError("LED log-offset prior sigma must be positive and finite")
    jax, jnp, numpyro, dist = require_numpyro()
    likelihood_weight = default_likelihood_weight(data["counts"])
    data = dict(data)
    data["n_scas"] = int(data["n_scas"])
    data["n_tiles"] = int(data["n_tiles"])
    n_scas = data["n_scas"]
    n_tiles = data["n_tiles"]
    n_nodes = len(data["node_days"])
    thickness_prior_centers = dict(thickness_prior_centers or {})
    thickness_prior_sigmas = dict(thickness_prior_sigmas or {})
    thickness_prior_shape = (n_scas, n_tiles)
    default_thickness_center = (
        np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM)
        if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL
        else 0.0
    )
    default_thickness_sigma = (
        INITIAL_THICKNESS_LOG_SIGMA
        if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL
        else initial_thickness_prior_sigma_nm
    )
    for name in thickness_names:
        thickness_prior_centers[name] = np.broadcast_to(
            np.asarray(
                thickness_prior_centers.get(name, default_thickness_center),
                dtype=float,
            ),
            thickness_prior_shape,
        ).copy()
        thickness_prior_sigmas[name] = np.broadcast_to(
            np.asarray(
                thickness_prior_sigmas.get(name, default_thickness_sigma),
                dtype=float,
            ),
            thickness_prior_shape,
        ).copy()
        if (
            np.any(~np.isfinite(thickness_prior_centers[name]))
            or np.any(~np.isfinite(thickness_prior_sigmas[name]))
            or np.any(thickness_prior_sigmas[name] <= 0.0)
        ):
            raise ValueError(
                f"Thickness prior center and sigma for {name} must be finite; "
                "sigma must be positive"
            )
    rate_prior_shape = (n_scas, n_tiles, n_nodes)
    if rate_prior_log_mean is None:
        rate_prior_log_mean = np.full(
            rate_prior_shape,
            np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
            dtype=float,
        )
    else:
        rate_prior_log_mean = np.broadcast_to(
            np.asarray(rate_prior_log_mean, dtype=float), rate_prior_shape
        ).copy()
    if np.any(~np.isfinite(rate_prior_log_mean)):
        raise ValueError("Trajectory-rate prior means must be finite")
    if entropy_reference_log_rates is not None:
        entropy_reference_log_rates = np.broadcast_to(
            np.asarray(entropy_reference_log_rates, dtype=float), rate_prior_shape
        ).copy()
        if np.any(~np.isfinite(entropy_reference_log_rates)):
            raise ValueError("Entropy reference rates must be finite")
    reset_days = reset_days_for_data(data)
    reset_count = reset_days.size
    n_rows = n_scas * n_tiles
    if n_rows != np.asarray(data["counts"]).shape[0]:
        raise ValueError("Tile statistics do not match the SCA/tile layout")
    active = ar_active_parameter_indices()
    reference_raw = ar_reference_raw(
        ar_prior_arrays()[0],
        ar_prior_arrays()[1],
        ar_prior_arrays()[2],
        active,
    )
    if ar_prior_center_raw is None:
        ar_prior_center_raw = reference_raw.copy()
    else:
        ar_prior_center_raw = np.asarray(ar_prior_center_raw, dtype=float).reshape(-1)
        if ar_prior_center_raw.size != len(active):
            raise ValueError(
                "Custom AR prior center must contain one value per active AR site"
            )
    if ar_prior_sigma_raw is None:
        ar_prior_sigma_raw = np.full(len(active), ar_mean_prior_sigma, dtype=float)
    else:
        ar_prior_sigma_raw = np.broadcast_to(
            np.asarray(ar_prior_sigma_raw, dtype=float), (len(active),)
        ).copy()
    if (
        np.any(~np.isfinite(ar_prior_center_raw))
        or np.any(~np.isfinite(ar_prior_sigma_raw))
        or np.any(ar_prior_sigma_raw <= 0.0)
    ):
        raise ValueError("Custom AR prior center and scales must be finite; scales must be positive")
    predictor = build_prediction_function(
        data,
        context,
        incidence_pupil_half_angle_deg=incidence_pupil_half_angle_deg,
        fixed_normal_incidence=(
            not fit_incidence_angle
            and float(incidence_pupil_half_angle_deg) == 0.0
        ),
        fit_ice_n_modulation=fit_ice_n_modulation,
        fit_ice_n_constant=fit_ice_n_constant,
        ice_n_porosity_parameters=ice_n_porosity_parameters,
        fit_ice_k_constant=fit_ice_k_constant,
        fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
        fit_led_log_offsets=fit_led_log_offsets,
        normalize_clean_response=normalize_clean_response,
    )
    curvature_matrix = jnp.asarray(rate_curvature_matrix(data["node_days"]))
    likelihood_scale = (
        0.5
        * likelihood_weight
        / n_tiles
        * float(np.sum(data["sum_squared"]))
        / LOG_NOISE_SIGMA**2
    )

    def rate_nodes_from_values(values):
        if constant_rate:
            return jnp.broadcast_to(
                values["log_rate_scale"][..., None],
                rate_prior_shape,
            )
        return values["np_log_rate_nodes"]

    observation_min_day = float(np.min(data["days"]))
    observation_max_day = float(np.max(data["days"]))
    active_reset_days = reset_days[
        (reset_days > observation_min_day) & (reset_days < observation_max_day)
    ]
    segment_boundaries = np.concatenate(
        ([observation_min_day], active_reset_days, [observation_max_day])
    )
    segment_integration_times = jnp.asarray(
        np.unique(
            np.concatenate(
                (
                    [observation_min_day],
                    np.asarray(data["node_days"], dtype=float),
                    [observation_max_day],
                    reset_days,
                )
            )
        )
    )
    node_days_array = np.asarray(data["node_days"], dtype=float)
    segment_node_indices = []
    for segment_index in range(segment_boundaries.size - 1):
        left = segment_boundaries[segment_index]
        right = segment_boundaries[segment_index + 1]
        if segment_index == segment_boundaries.size - 2:
            mask = (node_days_array >= left) & (node_days_array <= right)
        else:
            mask = (node_days_array >= left) & (node_days_array < right)
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            raise ValueError(
                "The trajectory requires at least two nodes in each reset segment"
            )
        segment_node_indices.append(indices)

    def segment_rate_means(values, state):
        cross_reset_rate = (state["ice"][:, -1] - state["ice"][:, 0]) / (
            observation_max_day - observation_min_day
        )
        if not active_reset_days.size:
            return (
                cross_reset_rate[:, None],
                cross_reset_rate,
                cross_reset_rate,
            )
        if trajectory_parameterization == "rate":
            profiles = rate_nodes_from_values(values).reshape(
                (-1, len(node_days_array))
            )

            def integrated_at(profile, target_day):
                return integrate_rate_trajectory(
                    jax,
                    jnp,
                    jnp.asarray([target_day]),
                    jnp.asarray(node_days_array),
                    segment_integration_times,
                    profile,
                )[0][0]

            boundary_integrated = jax.vmap(
                lambda profile: jax.vmap(
                    lambda target_day: integrated_at(profile, target_day)
                )(jnp.asarray(segment_boundaries))
            )(profiles)
            segment_accumulation = jnp.diff(boundary_integrated, axis=1)
        else:
            node_ice = state["node_ice"]
            segment_accumulation = jnp.stack(
                [
                    node_ice[:, indices[-1]] - node_ice[:, indices[0]]
                    for indices in segment_node_indices
                ],
                axis=1,
            )
        segment_duration = jnp.asarray(np.diff(segment_boundaries))
        segment_rates = segment_accumulation / segment_duration[None, :]
        reset_aware_rate = jnp.sum(segment_accumulation, axis=1) / jnp.sum(
            segment_duration
        )
        return segment_rates, segment_rates[:, 0], reset_aware_rate

    def state_from_values(values):
        ar_parameters = _physical_ar_parameters(values["ar_tile_raw"], jax, jnp)
        if fixed_ar_base_n is not None:
            fixed_base_n = jnp.asarray(fixed_ar_base_n)
            if fixed_base_n.shape != (AR_LAYERS,):
                raise ValueError("Fixed AR base indices must contain one value per layer")
            ar_parameters = ar_parameters.at[..., 1::3].set(
                jnp.broadcast_to(fixed_base_n, ar_parameters[..., 1::3].shape)
            )
        if fit_ar_k_constant:
            ar_parameters = ar_parameters.at[..., 2::3].set(
                values["ar_k_constant"]
            )
        if trajectory_parameterization == "thickness":
            ice, rates, node_ice, node_rates = _direct_thickness_state(
                data,
                values["np_log_thickness_nodes"],
                jax,
                jnp,
            )
        else:
            ice, rates, node_ice, node_rates = _trajectory_state(
                data,
                rate_nodes_from_values(values),
                values[thickness_names[0]],
                jnp.stack(
                    [values[name] for name in thickness_names[1:]],
                    axis=0,
                )
                if len(thickness_names) > 2
                else values[thickness_names[1]],
                jax,
                jnp,
                initial_thickness_prior=initial_thickness_prior,
            )
        row_ar = ar_parameters.reshape((n_rows, 3 * AR_LAYERS))
        hgcdte = jnp.stack(
            (values["absolute_hgcdte_n_log_scale"], values["absolute_hgcdte_k_log_scale"]),
            axis=-1,
        ).reshape((n_rows, 2))
        reset_offsets = (
            jnp.stack([values[name] for name in offset_names], axis=0)
            if len(offset_names) > 1
            else values[offset_names[0]]
        )
        total_sse, prediction = predictor.residual_sum_squares(
            ice,
            row_ar,
            hgcdte,
            values["incidence_angle_deg"] if fit_incidence_angle else 0.0,
            post_reset_tile_log_offsets=reset_offsets,
            ice_n_log_modulation=(
                values["ice_n_log_modulation"]
                if fit_ice_n_modulation
                else None
            ),
            ice_n_log_constant=(
                values["ice_n_log_constant"]
                if fit_ice_n_constant
                else None
            ),
            ice_k_constant=(
                values["ice_k_constant"]
                if fit_ice_k_constant
                else None
            ),
            ar_n_log_modulation=(
                values["ar_n_log_modulation"].reshape(
                    (n_rows, AR_LAYERS, len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
                )
                if fit_ar_hgcdte_modulation
                else None
            ),
            hgcdte_n_log_modulation=(
                values["hgcdte_n_log_modulation"].reshape(
                    (n_rows, len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
                )
                if fit_ar_hgcdte_modulation
                else None
            ),
            hgcdte_k_log_modulation=(
                values["hgcdte_k_log_modulation"].reshape(
                    (n_rows, len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
                )
                if fit_ar_hgcdte_modulation
                else None
            ),
            led_log_offsets=(
                values["led_log_offsets"]
                if fit_led_log_offsets
                else None
            ),
            return_prediction=True,
        )
        return {
            "ar_parameters": ar_parameters,
            "ice": ice,
            "rates": rates,
            "node_ice": node_ice,
            "node_rates": node_rates,
            "hgcdte": hgcdte,
            "prediction": prediction,
            "total_sse": total_sse,
        }

    def production_residuals(values):
        if fixed_absolute_hgcdte_n_log_scale is not None:
            values = dict(values)
            values["absolute_hgcdte_n_log_scale"] = jnp.full(
                (n_scas, n_tiles),
                fixed_absolute_hgcdte_n_log_scale,
            )
        state = state_from_values(values)
        counts = jnp.asarray(data["counts"])
        observed = jnp.asarray(data["sums"]) / jnp.maximum(counts, 1.0)
        post_mask = jnp.asarray(
            np.asarray(data["days"], dtype=float)[None, :] >= reset_days[:, None]
        )
        reset_offsets = jnp.stack(
            [values[name].reshape(-1) for name in offset_names], axis=0
        )
        observed = observed + jnp.sum(
            reset_offsets[:, :, None] * post_mask[:, None, :], axis=0
        )
        data_residual = (
            jnp.sqrt(likelihood_weight / n_tiles)
            / LOG_NOISE_SIGMA
            * jnp.sqrt(counts)
            * (state["prediction"] - observed)
        ).reshape(-1)
        prior_values = values
        if constant_rate:
            prior_values = dict(values)
            prior_values["np_log_rate_nodes"] = rate_nodes_from_values(values)
        prior_residual = _least_squares_prior_residuals(
            prior_values,
            data,
            jnp,
            smoothness_weight=smoothness_weight,
            entropy_reference_log_rates=entropy_reference_log_rates,
            entropy_weight=entropy_weight,
            rate_prior_log_mean=rate_prior_log_mean,
            trajectory_rate_prior_sigma_log=trajectory_rate_prior_sigma_log,
            trajectory_parameterization=trajectory_parameterization,
            initial_thickness_prior=initial_thickness_prior,
            initial_thickness_prior_sigma_nm=initial_thickness_prior_sigma_nm,
            thickness_prior_centers=thickness_prior_centers,
            thickness_prior_sigmas=thickness_prior_sigmas,
            absolute_hgcdte_n_scale_prior_sigma_log=(
                absolute_hgcdte_n_scale_prior_sigma_log
            ),
            fixed_absolute_hgcdte_n_log_scale=fixed_absolute_hgcdte_n_log_scale,
            ice_n_modulation_prior_sigma_log=ice_n_modulation_prior_sigma_log,
            ice_n_constant_prior_sigma_log=ice_n_constant_prior_sigma_log,
            ice_n_constant_reference=ice_n_constant_reference,
            ice_k_prior_sigma=ice_k_prior_sigma,
            ar_k_prior_sigma=ar_k_prior_sigma,
            ar_mean_prior_sigma=ar_mean_prior_sigma,
            ar_prior_center_raw=ar_prior_center_raw,
            ar_prior_sigma_raw=ar_prior_sigma_raw,
            fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
            ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_modulation_prior_sigma_log,
            hgcdte_n_modulation_prior_sigma_log=hgcdte_n_modulation_prior_sigma_log,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        )
        return jnp.concatenate((data_residual, prior_residual))

    def model():
        values = {
            "ar_tile_raw": numpyro.sample(
                "ar_tile_raw",
                dist.Normal(
                    jnp.asarray(ar_prior_center_raw),
                    jnp.asarray(ar_prior_sigma_raw),
                )
                .expand((n_scas, n_tiles, len(active)))
                .to_event(3),
            ),
            "absolute_hgcdte_k_log_scale": numpyro.sample(
                "absolute_hgcdte_k_log_scale",
                dist.Normal(
                    0.0,
                     ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
                )
                .expand((n_scas, n_tiles))
                .to_event(2),
            ),
        }
        for name in offset_names:
            values[name] = numpyro.sample(
                name,
                dist.Normal(0.0, POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA)
                .expand((n_scas, n_tiles))
                .to_event(2),
            )
        if trajectory_parameterization == "thickness":
            values["np_log_thickness_nodes"] = numpyro.sample(
                "np_log_thickness_nodes",
                dist.Uniform(-20.0, 20.0)
                .expand((n_scas, n_tiles, len(data["node_days"])))
                .to_event(3),
            )
        elif initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL:
            for name in thickness_names:
                values[name] = numpyro.sample(
                    name,
                    dist.Normal(
                        jnp.asarray(thickness_prior_centers[name]),
                        jnp.asarray(thickness_prior_sigmas[name]),
                    )
                    .to_event(2),
                )
        else:
            for name in thickness_names:
                values[name] = numpyro.sample(
                    name,
                    dist.HalfNormal(jnp.asarray(thickness_prior_sigmas[name]))
                    .to_event(2),
                )
        if fixed_absolute_hgcdte_n_log_scale is None:
            values["absolute_hgcdte_n_log_scale"] = numpyro.sample(
                "absolute_hgcdte_n_log_scale",
                dist.Normal(
                    0.0,
                     absolute_hgcdte_n_scale_prior_sigma_log,
                )
                .expand((n_scas, n_tiles))
                .to_event(2),
            )
        else:
            values["absolute_hgcdte_n_log_scale"] = jnp.full(
                (n_scas, n_tiles),
                fixed_absolute_hgcdte_n_log_scale,
            )
            numpyro.deterministic(
                "absolute_hgcdte_n_log_scale",
                values["absolute_hgcdte_n_log_scale"],
            )
        if fit_incidence_angle:
            values["incidence_angle_deg"] = numpyro.sample(
                "incidence_angle_deg",
                dist.Uniform(0.0, INCIDENCE_ANGLE_MAX_DEG),
            )
        if trajectory_parameterization == "thickness":
            pass
        elif constant_rate:
            values["log_rate_scale"] = numpyro.sample(
                "log_rate_scale",
                dist.Normal(
                    jnp.asarray(rate_prior_log_mean[..., 0]),
                    trajectory_rate_prior_sigma_log,
                ).to_event(2),
            )
        else:
            values["np_log_rate_nodes"] = numpyro.sample(
                "np_log_rate_nodes",
                dist.Normal(
                    jnp.asarray(rate_prior_log_mean),
                    trajectory_rate_prior_sigma_log,
                ).to_event(3),
            )
        if fit_ice_n_modulation:
            values["ice_n_log_modulation"] = numpyro.sample(
                "ice_n_log_modulation",
                 dist.Normal(0.0, ice_n_modulation_prior_sigma_log)
                .expand(
                    (
                        n_scas,
                        n_tiles,
                        len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM),
                    )
                )
                .to_event(3),
            )
        if fit_ice_n_constant:
            values["ice_n_log_constant"] = numpyro.sample(
                "ice_n_log_constant",
                dist.Normal(
                    np.log(ice_n_constant_reference / ICE_INDEX_DEFAULT),
                    ice_n_constant_prior_sigma_log,
                )
                .expand((n_scas, n_tiles))
                .to_event(2),
            )
        if fit_ice_k_constant:
            values["ice_k_constant"] = numpyro.sample(
                "ice_k_constant",
                 dist.HalfNormal(ice_k_prior_sigma)
                .expand((n_scas, n_tiles))
                .to_event(2),
            )
        if fit_ar_k_constant:
            values["ar_k_constant"] = numpyro.sample(
                "ar_k_constant",
                 dist.HalfNormal(ar_k_prior_sigma)
                .expand((n_scas, n_tiles, AR_LAYERS))
                .to_event(3),
            )
        if fit_ar_hgcdte_modulation:
            values["ar_n_log_modulation"] = numpyro.sample(
                "ar_n_log_modulation",
                dist.Normal(
                    0.0,
                     ar_hgcdte_modulation_prior_sigma_log,
                )
                .expand(
                    (
                        n_scas,
                        n_tiles,
                        AR_LAYERS,
                        len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM),
                    )
                )
                .to_event(4),
            )
            for name in (
                "hgcdte_n_log_modulation",
                "hgcdte_k_log_modulation",
            ):
                values[name] = numpyro.sample(
                    name,
                    dist.Normal(
                        0.0,
                        (
                            hgcdte_n_modulation_prior_sigma_log
                            if name == "hgcdte_n_log_modulation"
                            else ar_hgcdte_modulation_prior_sigma_log
                         ),
                    )
                    .expand(
                        (
                            n_scas,
                            n_tiles,
                            len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM),
                        )
                    )
                    .to_event(3),
                )
        if fit_led_log_offsets:
            values["led_log_offsets"] = numpyro.sample(
                "led_log_offsets",
                 dist.Normal(0.0, led_log_offset_prior_sigma_log)
                .expand((n_scas, len(data["first_by_led"])))
                .to_event(2),
            )
        state = state_from_values(values)
        segment_rates, pre_reset_rate, reset_aware_rate = segment_rate_means(
            values,
            state,
        )
        cross_reset_rate = (
            (state["ice"][:, -1] - state["ice"][:, 0])
            / (observation_max_day - observation_min_day)
        )
        numpyro.factor(
            "tile_gaussian_log_likelihood",
            -0.5
            * likelihood_weight
            / n_tiles
            * state["total_sse"]
            / LOG_NOISE_SIGMA**2,
        )
        if trajectory_parameterization == "rate" and smoothness_weight:
            curvature = jnp.einsum(
                "ij,tkj->tki", curvature_matrix, rate_nodes_from_values(values)
            )
            numpyro.factor(
                "trajectory_curvature",
                -likelihood_scale * smoothness_weight * jnp.mean(curvature**2),
            )
        if trajectory_parameterization == "rate" and entropy_reference_log_rates is not None and entropy_weight:
            entropy = _relative_entropy(
                rate_nodes_from_values(values),
                jnp.asarray(entropy_reference_log_rates),
                jnp,
            )
            numpyro.factor(
                "trajectory_entropy",
                -likelihood_scale * entropy_weight * jnp.mean(entropy),
            )
        for name, value in state.items():
            if name not in ("total_sse", "prediction"):
                numpyro.deterministic(name, value)
        if fit_ice_n_modulation:
            numpyro.deterministic(
                "ice_n_modulation",
                jnp.exp(values["ice_n_log_modulation"]),
            )
        if fit_ice_n_constant:
            numpyro.deterministic(
                "ice_n_constant",
                ICE_INDEX_DEFAULT * jnp.exp(values["ice_n_log_constant"]),
            )
        numpyro.deterministic("_map_least_squares_prediction", state["prediction"])
        numpyro.deterministic("ice_thickness", state["ice"])
        numpyro.deterministic("instantaneous_rate_nm_day", state["rates"])
        numpyro.deterministic("node_ice_thickness", state["node_ice"])
        numpyro.deterministic("node_instantaneous_rate_nm_day", state["node_rates"])
        if trajectory_parameterization == "rate":
            numpyro.deterministic(
                "np_node_multipliers",
                jnp.exp(
                    rate_nodes_from_values(values)
                    - np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)
                ),
            )
        else:
            numpyro.deterministic(
                "np_node_multipliers",
                state["node_rates"] / TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
            )
        numpyro.deterministic("tile_ice_thickness", state["ice"].reshape((n_scas, n_tiles, -1)))
        numpyro.deterministic("tile_instantaneous_rate_nm_day", state["rates"].reshape((n_scas, n_tiles, -1)))
        numpyro.deterministic("np_node_thickness", state["node_ice"].reshape((n_scas, n_tiles, -1)))
        numpyro.deterministic("tile_initial_rate_nm_day", state["rates"][:, 0].reshape((n_scas, n_tiles)))
        numpyro.deterministic("tile_final_rate_nm_day", state["rates"][:, -1].reshape((n_scas, n_tiles)))
        numpyro.deterministic(
            "tile_mean_rate_nm_day",
            reset_aware_rate.reshape((n_scas, n_tiles)),
        )
        numpyro.deterministic(
            "tile_pre_reset_mean_rate_nm_day",
            pre_reset_rate.reshape((n_scas, n_tiles)),
        )
        numpyro.deterministic(
            "tile_post_reset_mean_rate_nm_day",
            segment_rates[:, min(1, segment_rates.shape[1] - 1)].reshape(
                (n_scas, n_tiles)
            ),
        )
        if active_reset_days.size >= 2:
            numpyro.deterministic(
                "tile_post_third_decon_mean_rate_nm_day",
                segment_rates[:, 2].reshape((n_scas, n_tiles)),
            )
        numpyro.deterministic(
            "tile_cross_reset_mean_rate_nm_day",
            cross_reset_rate.reshape((n_scas, n_tiles)),
        )
        numpyro.deterministic(
            "mean_rate_nm_day",
            jnp.mean(reset_aware_rate),
        )
        numpyro.deterministic(
            "pre_reset_mean_rate_nm_day",
            jnp.mean(pre_reset_rate),
        )
        numpyro.deterministic(
            "post_reset_mean_rate_nm_day",
            jnp.mean(segment_rates[:, min(1, segment_rates.shape[1] - 1)]),
        )
        if active_reset_days.size >= 2:
            numpyro.deterministic(
                "post_third_decon_mean_rate_nm_day",
                jnp.mean(segment_rates[:, 2]),
            )
        numpyro.deterministic(
            "cross_reset_mean_rate_nm_day",
            jnp.mean(cross_reset_rate),
        )

    model.production_residuals = production_residuals
    model.production_state = state_from_values
    model.production_predictor = predictor
    return model


def initial_values_for_data(
    data,
    constant_rate=False,
    trajectory_parameterization="rate",
    fit_incidence_angle=True,
    fixed_absolute_hgcdte_n_log_scale=None,
    fit_ice_n_modulation=False,
    fit_ice_n_constant=False,
    fit_ice_k_constant=False,
    fit_ar_k_constant=False,
    fit_ar_hgcdte_modulation=False,
    fit_led_log_offsets=False,
    initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
    initial_thickness_prior_sigma_nm=INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
):
    if trajectory_parameterization not in ("rate", "thickness"):
        raise ValueError("Trajectory parameterization must be 'rate' or 'thickness'")
    if trajectory_parameterization == "thickness" and constant_rate:
        raise ValueError("Direct thickness cannot be combined with constant_rate")
    initial_thickness_prior, initial_thickness_prior_sigma_nm, _ = (
        _initial_thickness_settings(
            initial_thickness_prior,
            initial_thickness_prior_sigma_nm,
        )
    )
    thickness_names, offset_names = reset_site_names(
        data, initial_thickness_prior
    )
    active = ar_active_parameter_indices()
    reference_raw = ar_reference_raw(*ar_prior_arrays(), active)
    shape = (int(data["n_scas"]), int(data["n_tiles"]))
    values = {
        "ar_tile_raw": np.broadcast_to(reference_raw, shape + (len(active),)).copy(),
        "absolute_hgcdte_n_log_scale": np.zeros(shape),
        "absolute_hgcdte_k_log_scale": np.zeros(shape),
    }
    for name in offset_names:
        values[name] = np.zeros(shape)
    if trajectory_parameterization == "rate":
        if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL:
            thickness_initial = np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM)
        else:
            # Start at the half-normal mean while retaining a zero lower bound.
            thickness_initial = initial_thickness_prior_sigma_nm * np.sqrt(2.0 / np.pi)
        for name in thickness_names:
            values[name] = np.full(shape, thickness_initial)
    if fit_incidence_angle:
        values["incidence_angle_deg"] = float(INCIDENCE_ANGLE_INIT_DEG)
    if fixed_absolute_hgcdte_n_log_scale is not None:
        values.pop("absolute_hgcdte_n_log_scale")
    if trajectory_parameterization == "thickness":
        node_days = np.asarray(data["node_days"], dtype=float)
        reset_days = reset_days_for_data(data)
        initial = INITIAL_THICKNESS_LOG_MEDIAN_NM
        thickness = initial + TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY * (
            node_days - node_days[0]
        )
        for reset_day in reset_days:
            thickness = np.where(
                node_days >= reset_day,
                initial + TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY * (node_days - reset_day),
                thickness,
            )
        values["np_log_thickness_nodes"] = np.broadcast_to(
            np.log(thickness), shape + (len(node_days),)
        ).copy()
    elif constant_rate:
        values["log_rate_scale"] = np.full(
            shape,
            np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
        )
    else:
        values["np_log_rate_nodes"] = np.full(
            shape + (len(data["node_days"]),),
            np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
        )
    if fit_ice_n_modulation:
        values["ice_n_log_modulation"] = np.zeros(
            shape + (len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM),)
        )
    if fit_ice_n_constant:
        values["ice_n_log_constant"] = np.zeros(shape)
    if fit_ice_k_constant:
        values["ice_k_constant"] = np.full(shape, ICE_K_INIT)
    if fit_ar_k_constant:
        values["ar_k_constant"] = np.full(shape + (AR_LAYERS,), AR_K_INIT)
    if fit_ar_hgcdte_modulation:
        values["ar_n_log_modulation"] = np.zeros(
            shape
            + (
                AR_LAYERS,
                len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM),
            )
        )
        for name in (
            "hgcdte_n_log_modulation",
            "hgcdte_k_log_modulation",
        ):
            values[name] = np.zeros(
                shape + (len(ICE_N_MODULATION_NODE_WAVELENGTHS_NM),)
            )
    if fit_led_log_offsets:
        values["led_log_offsets"] = np.zeros(
            (int(data["n_scas"]), len(data["first_by_led"])),
        )
    return values


def map_parameter_scale_values(
    initial_values,
    trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    initial_thickness_prior_sigma_nm=INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
    thickness_prior_sigmas=None,
    absolute_hgcdte_n_scale_prior_sigma_log=ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    ice_k_prior_sigma=ICE_K_PRIOR_SIGMA,
    ar_k_prior_sigma=AR_K_PRIOR_SIGMA,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_prior_sigma_raw=None,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    hgcdte_n_modulation_prior_sigma_log=None,
    led_log_offset_prior_sigma_log=LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    if (
        not np.isfinite(trajectory_rate_prior_sigma_log)
        or trajectory_rate_prior_sigma_log <= 0.0
    ):
        raise ValueError("Trajectory-rate prior sigma must be positive and finite")
    if (
        not np.isfinite(absolute_hgcdte_n_scale_prior_sigma_log)
        or absolute_hgcdte_n_scale_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "HgCdTe n scale prior sigma must be positive and finite"
        )
    if hgcdte_n_modulation_prior_sigma_log is None:
        hgcdte_n_modulation_prior_sigma_log = ar_hgcdte_modulation_prior_sigma_log
    if (
        not np.isfinite(hgcdte_n_modulation_prior_sigma_log)
        or hgcdte_n_modulation_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "HgCdTe n modulation prior sigma must be positive and finite"
        )
    if (
        not np.isfinite(initial_thickness_prior_sigma_nm)
        or initial_thickness_prior_sigma_nm <= 0.0
    ):
        raise ValueError("Initial-thickness prior sigma must be positive and finite")
    if (
        not np.isfinite(led_log_offset_prior_sigma_log)
        or led_log_offset_prior_sigma_log <= 0.0
    ):
        raise ValueError("LED log-offset prior sigma must be positive and finite")
    if not np.isfinite(ice_k_prior_sigma) or ice_k_prior_sigma <= 0.0:
        raise ValueError("Constant ice k prior sigma must be positive and finite")
    thickness_scale = (
        initial_thickness_prior_sigma_nm
        if "initial_thickness_nm" in initial_values
        else INITIAL_THICKNESS_LOG_SIGMA
    )
    thickness_prior_sigmas = thickness_prior_sigmas or {}
    if ar_prior_sigma_raw is None:
        ar_tile_raw_scale = ar_mean_prior_sigma
    else:
        ar_tile_raw_scale = np.broadcast_to(
            np.asarray(ar_prior_sigma_raw, dtype=float),
            np.asarray(initial_values["ar_tile_raw"]).shape,
        ).copy()
        if (
            np.any(~np.isfinite(ar_tile_raw_scale))
            or np.any(ar_tile_raw_scale <= 0.0)
        ):
            raise ValueError("Custom AR prior scales must be finite and positive")
    scales = {
        "ar_tile_raw": ar_tile_raw_scale,
        "initial_thickness_log_nm": thickness_prior_sigmas.get(
            "initial_thickness_log_nm", INITIAL_THICKNESS_LOG_SIGMA
        ),
        "reset_initial_thickness_log_nm": thickness_prior_sigmas.get(
            "reset_initial_thickness_log_nm", INITIAL_THICKNESS_LOG_SIGMA
        ),
        "initial_thickness_nm": thickness_prior_sigmas.get(
            "initial_thickness_nm", thickness_scale
        ),
        "reset_initial_thickness_nm": thickness_prior_sigmas.get(
            "reset_initial_thickness_nm", thickness_scale
        ),
        "incidence_angle_deg": INCIDENCE_ANGLE_MAX_DEG,
        "np_log_rate_nodes": trajectory_rate_prior_sigma_log,
        "log_rate_scale": trajectory_rate_prior_sigma_log,
        "np_log_thickness_nodes": 1.0,
        "post_reset_tile_log_offset": POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
        "post_third_decon_tile_log_offset": POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
        "third_decon_initial_thickness_log_nm": thickness_prior_sigmas.get(
            "third_decon_initial_thickness_log_nm", INITIAL_THICKNESS_LOG_SIGMA
        ),
        "third_decon_initial_thickness_nm": thickness_prior_sigmas.get(
            "third_decon_initial_thickness_nm", thickness_scale
        ),
        "absolute_hgcdte_n_log_scale": absolute_hgcdte_n_scale_prior_sigma_log,
        "absolute_hgcdte_k_log_scale": ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        "ice_n_log_modulation": ice_n_modulation_prior_sigma_log,
        "ice_n_log_constant": ice_n_constant_prior_sigma_log,
        "ice_k_constant": ice_k_prior_sigma,
        "ar_k_constant": ar_k_prior_sigma,
        "ar_n_log_modulation": ar_hgcdte_modulation_prior_sigma_log,
        "hgcdte_n_log_modulation": hgcdte_n_modulation_prior_sigma_log,
        "hgcdte_k_log_modulation": ar_hgcdte_modulation_prior_sigma_log,
        "led_log_offsets": led_log_offset_prior_sigma_log,
    }
    return {
        name: np.broadcast_to(
            np.asarray(scales[name], dtype=float),
            np.asarray(value).shape,
        ).copy()
        for name, value in initial_values.items()
    }


def map_parameter_bounds(initial_values, jax, fixed_parameter_values=None):
    from jax.flatten_util import ravel_pytree

    lower = {}
    upper = {}
    for name, value in initial_values.items():
        shape = np.asarray(value).shape
        lower[name] = np.full(shape, -np.inf)
        upper[name] = np.full(shape, np.inf)
        if name in ("initial_thickness_nm", "reset_initial_thickness_nm"):
            lower[name] = np.zeros(shape)
        elif name == "ice_k_constant":
            lower[name] = np.zeros(shape)
        elif name == "ar_k_constant":
            lower[name] = np.zeros(shape)
        elif name == "np_log_thickness_nodes":
            lower[name] = np.full(shape, -20.0)
            upper[name] = np.full(shape, 20.0)
        elif name == "incidence_angle_deg":
            lower[name] = np.zeros(shape)
            upper[name] = np.full(shape, INCIDENCE_ANGLE_MAX_DEG)
    for name, value in (fixed_parameter_values or {}).items():
        if name not in initial_values:
            raise ValueError(f"Unknown fixed MAP parameter: {name}")
        fixed = np.broadcast_to(
            np.asarray(value, dtype=float), np.asarray(initial_values[name]).shape
        )
        if np.any(~np.isfinite(fixed)):
            raise ValueError(f"Fixed MAP parameter {name} must be finite")
        lower[name] = fixed - 1.0e-12
        upper[name] = fixed + 1.0e-12
    flat_initial, unravel = ravel_pytree({name: jax.numpy.asarray(value) for name, value in initial_values.items()})
    flat_lower, _ = ravel_pytree({name: jax.numpy.asarray(value) for name, value in lower.items()})
    flat_upper, _ = ravel_pytree({name: jax.numpy.asarray(value) for name, value in upper.items()})
    return flat_initial, unravel, flat_lower, flat_upper


def run_least_squares_map_optimization(
    model,
    initial_values,
    jax,
    tolerance=1.0e-9,
    trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    initial_thickness_prior_sigma_nm=INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
    thickness_prior_sigmas=None,
    absolute_hgcdte_n_scale_prior_sigma_log=ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    ice_k_prior_sigma=ICE_K_PRIOR_SIGMA,
    ar_k_prior_sigma=AR_K_PRIOR_SIGMA,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_prior_sigma_raw=None,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    hgcdte_n_modulation_prior_sigma_log=None,
    led_log_offset_prior_sigma_log=LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    fixed_parameter_values=None,
):
    """Run the only production MAP optimizer in prior-width coordinates."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    flat_initial, unravel, flat_lower, flat_upper = map_parameter_bounds(
        initial_values,
        jax,
        fixed_parameter_values=fixed_parameter_values,
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    fixed_parameter_values = fixed_parameter_values or {}
    fixed_mask_tree = {
        name: np.ones_like(value, dtype=bool)
        if name in fixed_parameter_values
        else np.zeros_like(value, dtype=bool)
        for name, value in initial_values.items()
    }
    fixed_mask, _ = ravel_pytree(
        {name: jax.numpy.asarray(value) for name, value in fixed_mask_tree.items()}
    )
    free_mask = ~np.asarray(fixed_mask, dtype=bool)
    if not np.any(free_mask):
        raise ValueError("At least one MAP parameter must remain free")
    free_indices = np.flatnonzero(free_mask)
    free_indices_jax = jax.numpy.asarray(free_indices)
    parameter_scales = map_parameter_scale_values(
        initial_values,
        trajectory_rate_prior_sigma_log=trajectory_rate_prior_sigma_log,
        initial_thickness_prior_sigma_nm=initial_thickness_prior_sigma_nm,
        thickness_prior_sigmas=thickness_prior_sigmas,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
        ice_n_modulation_prior_sigma_log=ice_n_modulation_prior_sigma_log,
        ice_n_constant_prior_sigma_log=ice_n_constant_prior_sigma_log,
        ice_k_prior_sigma=ice_k_prior_sigma,
        ar_k_prior_sigma=ar_k_prior_sigma,
        ar_mean_prior_sigma=ar_mean_prior_sigma,
        ar_prior_sigma_raw=ar_prior_sigma_raw,
        ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_modulation_prior_sigma_log,
        hgcdte_n_modulation_prior_sigma_log=hgcdte_n_modulation_prior_sigma_log,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    scale_pytree = {
        name: jax.numpy.asarray(np.broadcast_to(parameter_scales[name], np.asarray(value).shape))
        for name, value in initial_values.items()
    }
    flat_scales, _ = ravel_pytree(scale_pytree)
    flat_scales = np.asarray(flat_scales, dtype=float)
    scaled_lower = (np.asarray(flat_lower, dtype=float) - flat_initial) / flat_scales
    scaled_upper = (np.asarray(flat_upper, dtype=float) - flat_initial) / flat_scales

    def residuals(scaled_values):
        full_scaled_values = jax.numpy.zeros(flat_initial.shape).at[
            free_indices_jax
        ].set(scaled_values)
        values = unravel(
            jax.numpy.asarray(flat_initial)
            + jax.numpy.asarray(flat_scales) * full_scaled_values
        )
        return model.production_residuals(values)

    def residuals_with_aux(scaled_values):
        residual = residuals(scaled_values)
        return residual, residual

    jacobian_with_residual_jit = jax.jit(
        jax.jacfwd(residuals_with_aux, has_aux=True)
    )
    evaluation_cache = {"values": None, "residual": None, "jacobian": None}

    def scipy_evaluation(scaled_values):
        scaled_values = np.asarray(scaled_values, dtype=float)
        if evaluation_cache["values"] is not None and np.array_equal(
            scaled_values, evaluation_cache["values"]
        ):
            return evaluation_cache["residual"], evaluation_cache["jacobian"]
        jacobian, residual = jacobian_with_residual_jit(
            jax.numpy.asarray(scaled_values)
        )
        evaluation_cache["values"] = scaled_values.copy()
        evaluation_cache["residual"] = np.asarray(residual, dtype=float)
        evaluation_cache["jacobian"] = np.asarray(jacobian, dtype=float)
        return evaluation_cache["residual"], evaluation_cache["jacobian"]

    def scipy_residual(scaled_values):
        return scipy_evaluation(scaled_values)[0]

    def scipy_jacobian(scaled_values):
        return scipy_evaluation(scaled_values)[1]

    initial_scaled = np.zeros(free_indices.size, dtype=float)
    initial_residual = scipy_residual(initial_scaled)
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        bounds=(scaled_lower[free_mask], scaled_upper[free_mask]),
        method="trf",
        x_scale="jac",
        ftol=PRODUCTION_MAP_FTOL,
        xtol=PRODUCTION_MAP_XTOL,
        gtol=float(tolerance),
        max_nfev=PRODUCTION_MAP_STEPS,
        verbose=2,
    )
    optimized_scaled = np.zeros_like(flat_initial)
    optimized_scaled[free_mask] = np.asarray(result.x, dtype=float)
    optimized_flat = flat_initial + flat_scales * optimized_scaled
    optimized_values = {
        name: np.asarray(value)
        for name, value in unravel(jax.numpy.asarray(optimized_flat)).items()
    }
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    jacobian = scipy_jacobian(np.asarray(result.x, dtype=float))
    scaled_gradient = jacobian.T @ final_residual
    scaled_gradient_norm = float(np.max(np.abs(scaled_gradient)))
    from numpyro.infer.util import log_density

    initial_objective = float(
        -log_density(model, (), {}, unravel(jax.numpy.asarray(flat_initial)))[0]
    )
    final_objective = float(
        -log_density(model, (), {}, unravel(jax.numpy.asarray(optimized_flat)))[0]
    )
    gradient_converged = bool(scaled_gradient_norm <= tolerance)
    diagnostics = {
        "optimizer": "scipy least_squares",
        "converged": bool(result.success),
        "gradient_converged": gradient_converged,
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nfev),
        "function_evaluations": int(result.nfev),
        "gradient_norm": float(
            np.max(np.abs(scaled_gradient / flat_scales[free_mask]))
        ),
        "scaled_gradient_norm": scaled_gradient_norm,
        "function_tolerance": float(PRODUCTION_MAP_FTOL),
        "step_tolerance": float(PRODUCTION_MAP_XTOL),
        "gradient_tolerance": float(tolerance),
        "objective": final_objective,
        "initial_objective": initial_objective,
        "parameter_step_norm": float(np.linalg.norm(optimized_flat - flat_initial)),
        "objective_history": np.asarray(
            [0.5 * np.sum(initial_residual**2), 0.5 * np.sum(final_residual**2)]
        ),
        "parameter_scale_min": float(np.min(flat_scales[free_mask])),
        "parameter_scale_max": float(np.max(flat_scales[free_mask])),
    }
    return optimized_values, diagnostics


def load_map_warm_start(
    path,
    initial_values,
    data,
    require_map_only=True,
):
    """Load the fixed one-tile EMCEE template into a MAP start."""
    samples_path = Path(path) / "posterior_samples.npz" if Path(path).is_dir() else Path(path)
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)
    with np.load(samples_path, allow_pickle=False) as samples:
        if (
            require_map_only
            and "map_only" not in samples
        ):
            raise ValueError("Production warm starts must be marked map-only")
        if "map_only" in samples and not bool(np.asarray(samples["map_only"]).item()):
            raise ValueError("Production warm starts must be deterministic map-only archives")
        if "sca_names" in samples and not np.array_equal(
            np.asarray(samples["sca_names"]).astype(str),
            np.asarray(data["sca_names"]).astype(str),
        ):
            raise ValueError("EMCEE warm-start detector does not match the target tile")
        transferred = dict(initial_values)
        for name, target in initial_values.items():
            source_name = name
            transform = None
            require_positive = False
            average_ice_modulation = False
            broadcast_ice_constant = False
            if name == "log_rate_scale" and name not in samples:
                source_name = "np_log_rate_nodes"
            if name == "ice_n_log_constant" and source_name not in samples:
                source_name = "ice_n_log_modulation"
                average_ice_modulation = True
            if name == "ice_n_log_modulation" and source_name not in samples:
                source_name = "ice_n_log_constant"
                broadcast_ice_constant = True
            if name == "initial_thickness_nm" and source_name not in samples:
                source_name = "initial_thickness_log_nm"
                transform = np.exp
            elif (
                name == "reset_initial_thickness_nm"
                and source_name not in samples
            ):
                source_name = "reset_initial_thickness_log_nm"
                transform = np.exp
            elif name == "initial_thickness_log_nm" and source_name not in samples:
                source_name = "initial_thickness_nm"
                transform = np.log
                require_positive = True
            elif (
                name == "reset_initial_thickness_log_nm"
                and source_name not in samples
            ):
                source_name = "reset_initial_thickness_nm"
                transform = np.log
                require_positive = True
            elif (
                name == "third_decon_initial_thickness_nm"
                and source_name not in samples
            ):
                source_name = "third_decon_initial_thickness_log_nm"
                transform = np.exp
            elif (
                name == "third_decon_initial_thickness_log_nm"
                and source_name not in samples
            ):
                source_name = "third_decon_initial_thickness_nm"
                transform = np.log
                require_positive = True
            if source_name not in samples:
                continue
            target_array = np.asarray(target)
            value = np.asarray(samples[source_name], dtype=float)
            if require_positive and np.any(value <= 0.0):
                continue
            if transform is not None:
                value = transform(value)
            if average_ice_modulation:
                target_ndim = target_array.ndim
                while value.ndim > target_ndim + 1 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim == target_ndim + 1:
                    value = np.mean(value, axis=-1)
            if broadcast_ice_constant:
                sample_shape = (1,) + target_array.shape[:-1]
                if value.shape == sample_shape:
                    value = value[0]
                if value.shape == target_array.shape[:-1]:
                    value = np.broadcast_to(
                        value[..., None], target_array.shape
                    ).copy()
            if name == "log_rate_scale" and source_name == "np_log_rate_nodes":
                while value.ndim > target_array.ndim + 1 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim == target_array.ndim + 1:
                    value = value[..., 0]
            else:
                while value.ndim > target_array.ndim and value.shape[0] == 1:
                    value = value[0]
            if value.shape == target_array.shape:
                transferred[name] = value.copy()
            elif (
                value.ndim == target_array.ndim
                and value.shape[0] == 1
                and target_array.shape[0] >= 1
            ):
                transferred[name] = np.broadcast_to(value, target_array.shape).copy()
        return transferred


def _evaluate_model(model, values):
    from numpyro.handlers import substitute, trace

    traced = trace(substitute(model, data=values)).get_trace()
    return {
        name: np.asarray(site["value"])
        for name, site in traced.items()
        if site["type"] in ("sample", "deterministic")
    }


def fit_map_state(config):
    """Fit a map and return its state without writing production outputs."""
    jax, _, _, _ = require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data = load_tile_statistics(
        config.tile_statistics,
        third_decon_day=config.third_decon_day,
    )
    context = load_response_context(
        data["sca_names"],
        sed_weighting=config.sed_weighting,
        qe_response_mode=config.qe_response_mode,
    )
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    initial_values = initial_values_for_data(
        data,
        constant_rate=config.constant_rate,
        trajectory_parameterization=config.trajectory_parameterization,
        initial_thickness_prior=config.initial_thickness_prior,
        initial_thickness_prior_sigma_nm=config.initial_thickness_prior_sigma_nm,
        fit_incidence_angle=config.fit_incidence_angle,
        fixed_absolute_hgcdte_n_log_scale=(
            config.fixed_absolute_hgcdte_n_log_scale
        ),
        fit_ice_n_modulation=config.fit_ice_n_modulation,
        fit_ice_n_constant=config.fit_ice_n_constant,
        fit_ice_k_constant=config.fit_ice_k_constant,
        fit_ar_k_constant=config.fit_ar_k_constant,
        fit_ar_hgcdte_modulation=config.fit_ar_hgcdte_modulation,
        fit_led_log_offsets=config.fit_led_log_offsets,
    )
    if config.map_init_from is not None:
        initial_values = load_map_warm_start(config.map_init_from, initial_values, data)
    if config.map_initial_values is not None:
        for name, value in config.map_initial_values.items():
            if name not in initial_values:
                raise ValueError(f"Unknown MAP initial-value site: {name}")
            value = np.asarray(value, dtype=float)
            if value.shape != np.asarray(initial_values[name]).shape:
                raise ValueError(
                    f"MAP initial-value shape mismatch for {name}: "
                    f"{value.shape} != {np.asarray(initial_values[name]).shape}"
                )
            initial_values[name] = value.copy()
    fixed_parameter_values = {}
    if config.fixed_ice_n_log_constant is not None:
        if "ice_n_log_constant" not in initial_values:
            raise ValueError(
                "A fixed ice n value requires fit_ice_n_constant=True"
            )
        fixed_ice_n_log_constant = float(config.fixed_ice_n_log_constant)
        if not np.isfinite(fixed_ice_n_log_constant):
            raise ValueError("Fixed ice n log constant must be finite")
        initial_values["ice_n_log_constant"] = np.full(
            np.asarray(initial_values["ice_n_log_constant"]).shape,
            fixed_ice_n_log_constant,
        )
        fixed_parameter_values["ice_n_log_constant"] = np.asarray(
            initial_values["ice_n_log_constant"], dtype=float
        ).copy()
    model = build_model(
        data,
        context,
        smoothness_weight=config.smoothness_weight,
        entropy_reference_log_rates=config.entropy_reference_log_rates,
        entropy_weight=config.entropy_weight,
        rate_prior_log_mean=config.rate_prior_log_mean,
        trajectory_rate_prior_sigma_log=config.trajectory_rate_prior_sigma_log,
        trajectory_parameterization=config.trajectory_parameterization,
        initial_thickness_prior=config.initial_thickness_prior,
        initial_thickness_prior_sigma_nm=config.initial_thickness_prior_sigma_nm,
        thickness_prior_centers=config.thickness_prior_centers,
        thickness_prior_sigmas=config.thickness_prior_sigmas,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            config.absolute_hgcdte_n_scale_prior_sigma_log
        ),
        fixed_absolute_hgcdte_n_log_scale=(
            config.fixed_absolute_hgcdte_n_log_scale
        ),
        constant_rate=config.constant_rate,
        fit_incidence_angle=config.fit_incidence_angle,
        incidence_pupil_half_angle_deg=config.incidence_pupil_half_angle_deg,
        fit_ice_n_modulation=config.fit_ice_n_modulation,
        ice_n_modulation_prior_sigma_log=config.ice_n_modulation_prior_sigma_log,
        fit_ice_n_constant=config.fit_ice_n_constant,
        ice_n_constant_prior_sigma_log=config.ice_n_constant_prior_sigma_log,
        ice_n_constant_reference=config.ice_n_constant_reference,
        ice_n_porosity_parameters=config.ice_n_porosity_parameters,
        fit_ice_k_constant=config.fit_ice_k_constant,
        ice_k_prior_sigma=config.ice_k_prior_sigma,
        fit_ar_k_constant=config.fit_ar_k_constant,
        ar_k_prior_sigma=config.ar_k_prior_sigma,
        ar_mean_prior_sigma=config.ar_mean_prior_sigma,
        ar_prior_center_raw=config.ar_prior_center_raw,
        ar_prior_sigma_raw=config.ar_prior_sigma_raw,
        fit_ar_hgcdte_modulation=config.fit_ar_hgcdte_modulation,
        ar_hgcdte_modulation_prior_sigma_log=(
            config.ar_hgcdte_modulation_prior_sigma_log
        ),
        hgcdte_n_modulation_prior_sigma_log=(
            config.hgcdte_n_modulation_prior_sigma_log
        ),
        fit_led_log_offsets=config.fit_led_log_offsets,
        led_log_offset_prior_sigma_log=config.led_log_offset_prior_sigma_log,
        normalize_clean_response=(config.qe_response_mode == "clean_relative"),
    )
    if config.thickness_penalty_plan is not None:
        from fit_wfi_tile136_thickness_penalty import attach_penalty

        attach_penalty(
            model,
            data,
            config.thickness_penalty_plan,
            config.thickness_penalty_lambda_shared,
            config.thickness_penalty_lambda_disagreement,
        )
    map_values, diagnostics = run_least_squares_map_optimization(
        model,
        initial_values,
        jax,
        trajectory_rate_prior_sigma_log=config.trajectory_rate_prior_sigma_log,
        initial_thickness_prior_sigma_nm=config.initial_thickness_prior_sigma_nm,
        thickness_prior_sigmas=config.thickness_prior_sigmas,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            config.absolute_hgcdte_n_scale_prior_sigma_log
        ),
        ice_n_modulation_prior_sigma_log=config.ice_n_modulation_prior_sigma_log,
        ice_n_constant_prior_sigma_log=config.ice_n_constant_prior_sigma_log,
        ice_k_prior_sigma=config.ice_k_prior_sigma,
        ar_k_prior_sigma=config.ar_k_prior_sigma,
        ar_mean_prior_sigma=config.ar_mean_prior_sigma,
        ar_prior_sigma_raw=config.ar_prior_sigma_raw,
        ar_hgcdte_modulation_prior_sigma_log=(
            config.ar_hgcdte_modulation_prior_sigma_log
        ),
        hgcdte_n_modulation_prior_sigma_log=(
            config.hgcdte_n_modulation_prior_sigma_log
        ),
        led_log_offset_prior_sigma_log=config.led_log_offset_prior_sigma_log,
        fixed_parameter_values=fixed_parameter_values,
    )
    evaluated = _evaluate_model(model, map_values)
    return data, context, map_values, diagnostics, evaluated


def ar_quantile_fields():
    fields = []
    for layer in range(AR_LAYERS):
        prefix = f"ar_layer{layer + 1}_"
        for component in ("thickness", "n", "k"):
            fields.extend(f"{prefix}{component}_{suffix}" for suffix in ("p16", "median", "p84"))
    return fields


def _add_ar_quantiles(row, quantiles, sca_index, tile_index):
    for layer in range(AR_LAYERS):
        prefix = f"ar_layer{layer + 1}_"
        for component_index, component in enumerate(("thickness", "n", "k")):
            for percentile_index, suffix in enumerate(("p16", "median", "p84")):
                row[f"{prefix}{component}_{suffix}"] = quantiles[
                    percentile_index, sca_index, tile_index, layer, component_index
                ]


def write_outputs(
    output_dir,
    data,
    evaluated,
    statistics_path,
    diagnostics,
    fit_diagnostics,
    model_metadata=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rate_prior_sigma_log = float(
        (model_metadata or {}).get(
            "trajectory_rate_prior_sigma_log",
            TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        )
    )
    absolute_hgcdte_n_scale_prior_sigma_log = float(
        (model_metadata or {}).get(
            "absolute_hgcdte_n_scale_prior_sigma_log",
            ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
        )
    )
    hgcdte_n_modulation_prior_sigma_log = float(
        (model_metadata or {}).get(
            "hgcdte_n_modulation_prior_sigma_log",
            (model_metadata or {}).get(
                "ar_hgcdte_modulation_prior_sigma_log",
                AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
            ),
        )
    )
    initial_thickness_log_median_nm = float(
        (model_metadata or {}).get(
            "initial_thickness_log_median_nm",
            INITIAL_THICKNESS_LOG_MEDIAN_NM,
        )
    )
    initial_thickness_log_sigma = float(
        (model_metadata or {}).get(
            "initial_thickness_log_sigma",
            INITIAL_THICKNESS_LOG_SIGMA,
        )
    )
    initial_thickness_prior, initial_thickness_prior_sigma_nm, _ = (
        _initial_thickness_settings(
            (model_metadata or {}).get(
                "initial_thickness_prior",
                INITIAL_THICKNESS_PRIOR_LOGNORMAL,
            ),
            (model_metadata or {}).get(
                "initial_thickness_prior_sigma_nm",
                INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
            ),
        )
    )
    if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL and (
        not np.isfinite(initial_thickness_log_median_nm)
        or initial_thickness_log_median_nm <= 0.0
        or not np.isfinite(initial_thickness_log_sigma)
        or initial_thickness_log_sigma <= 0.0
    ):
        raise ValueError(
            "Lognormal initial-thickness metadata must have positive finite median and sigma"
        )
    observed = np.asarray(fit_diagnostics["observed"], dtype=float)
    prediction = np.asarray(fit_diagnostics["prediction"], dtype=float)
    residual = observed - prediction
    counts = np.asarray(data["counts"], dtype=float)
    group_rms_log = float(np.sqrt(np.mean(residual**2)))
    count_weighted_rms_log = float(
        np.sqrt(np.sum(counts * residual**2) / np.sum(counts))
    )
    samples = {
        name: np.asarray(value)[None, ...] for name, value in evaluated.items()
    }
    metadata = {
        "map_only": np.asarray(1),
        "map_objective_history": np.asarray(diagnostics["objective_history"], dtype=float),
        "statistics_path": np.asarray(str(statistics_path)),
        "statistics_fingerprint": np.asarray(statistics_fingerprint(data)),
        "statistics_row_fingerprints": statistics_row_fingerprints(data),
        "tile_ids": np.asarray(data["tile_ids"], dtype=int),
        "sca_names": np.asarray(data["sca_names"]).astype(str),
        "qe_columns": np.asarray(data.get("qe_columns", []), dtype=int),
        "response_wavelengths_nm": np.asarray(
            data.get("response_wavelengths_nm", []), dtype=float
        ),
        "response_wavelength_count": np.asarray(
            int(data.get("response_wavelength_count", 0))
        ),
        "likelihood_weight": np.asarray(default_likelihood_weight(data["counts"])),
        "log_noise_sigma": np.asarray(LOG_NOISE_SIGMA),
        "observation_days": np.asarray(data["days"], dtype=float),
        "observation_led": np.asarray(data["led"], dtype=int),
        "observation_repeat": np.asarray(data["repeat"], dtype=int),
        "observation_program": np.asarray(data.get("program", []), dtype=int),
        "groups": np.asarray(data["groups"], dtype=int),
        "ar_layers": np.asarray(AR_LAYERS),
        "absolute_substrate_power": np.asarray(1),
        "absolute_hgcdte_tile_variation": np.asarray(1),
        "absolute_hgcdte_scale_prior_sigma_log": np.asarray(ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG),
        "absolute_hgcdte_n_scale_prior_sigma_log": np.asarray(
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
        "absolute_hgcdte_k_scale_prior_sigma_log": np.asarray(
            ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG
        ),
        "hgcdte_n_modulation_prior_sigma_log": np.asarray(
            hgcdte_n_modulation_prior_sigma_log
        ),
        "fit_ice_k_constant": np.asarray(
            int((model_metadata or {}).get("fit_ice_k_constant", 0))
        ),
        "ice_k_prior_sigma": np.asarray(
            float((model_metadata or {}).get("ice_k_prior_sigma", ICE_K_PRIOR_SIGMA))
        ),
        "fit_ar_k_constant": np.asarray(
            int((model_metadata or {}).get("fit_ar_k_constant", 0))
        ),
        "ar_k_prior_sigma": np.asarray(
            float((model_metadata or {}).get("ar_k_prior_sigma", AR_K_PRIOR_SIGMA))
        ),
        "sed_mode": np.asarray("core"),
        "sed_weighting": np.asarray((model_metadata or {}).get("sed_weighting", "energy")),
        "qe_response_mode": np.asarray((model_metadata or {}).get("qe_response_mode", "none")),
        "wavelength_stride": np.asarray(1),
        "fully_np_trajectory": np.asarray(1),
        "whiten_np_trajectory": np.asarray(0),
        "trajectory_rate_prior_mean_nm_day": np.asarray(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
        "trajectory_rate_prior_sigma_log": np.asarray(trajectory_rate_prior_sigma_log),
        "tile_initial_thickness_lognormal": np.asarray(
            int(initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL)
        ),
        "tile_initial_thickness_prior": np.asarray(initial_thickness_prior),
        "tile_initial_thickness_prior_sigma_nm": np.asarray(
            initial_thickness_prior_sigma_nm
        ),
        "tile_initial_thickness_log_median_nm": np.asarray(
            initial_thickness_log_median_nm
            if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL
            else np.nan
        ),
        "tile_initial_thickness_log_sigma": np.asarray(
            initial_thickness_log_sigma
            if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL
            else np.nan
        ),
        "ice_reset_day": np.asarray(
            float(reset_days_for_data(data)[0])
        ),
        "fit_post_reset_tile_log_offset": np.asarray(1),
        "post_reset_tile_log_offset_prior_sigma": np.asarray(POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA),
        "fit_incidence_angle": np.asarray(1),
        "incidence_angle_max_deg": np.asarray(INCIDENCE_ANGLE_MAX_DEG),
        "incidence_pupil_half_angle_deg": np.asarray(0.0),
        "input_corrections_already_applied": np.asarray(1),
        "map_optimizer": np.asarray("least-squares"),
        "map_steps": np.asarray(PRODUCTION_MAP_STEPS),
        "map_parameter_scaling": np.asarray("prior"),
        "map_function_tolerance": np.asarray(diagnostics["function_tolerance"]),
        "map_step_tolerance": np.asarray(diagnostics["step_tolerance"]),
        "map_gradient_tolerance": np.asarray(diagnostics["gradient_tolerance"]),
        "map_gradient_converged": np.asarray(int(diagnostics["gradient_converged"])),
        "map_group_rms_log": np.asarray(group_rms_log),
        "map_count_weighted_rms_log": np.asarray(count_weighted_rms_log),
    }
    reset_days = reset_days_for_data(data)
    if reset_days.size > 1:
        metadata["ice_reset_days"] = np.asarray(reset_days, dtype=float)
        metadata["third_decon_day"] = np.asarray(float(reset_days[1]))
        metadata["fit_post_third_decon_tile_log_offset"] = np.asarray(1)
    for name, value in (model_metadata or {}).items():
        if value is not None:
            metadata[str(name)] = np.asarray(value)
    np.savez_compressed(output_dir / "posterior_samples.npz", **samples, **metadata)

    with (output_dir / "tile_group_predictions.csv").open("w", newline="") as handle:
        fields = [
            "detector", "tile_id", "group", "days", "led", "repeat",
            "observed_log_change", "model_log_change", "residual_log",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index in range(observed.shape[0]):
            sca_index, tile_index = divmod(row_index, data["n_tiles"])
            for group_index, group in enumerate(data["groups"]):
                writer.writerow(
                    {
                        "detector": str(data["sca_names"][sca_index]),
                        "tile_id": int(data["tile_ids"][tile_index]),
                        "group": int(group),
                        "days": float(data["days"][group_index]),
                        "led": int(data["led"][group_index]),
                        "repeat": int(data["repeat"][group_index]),
                        "observed_log_change": observed[row_index, group_index],
                        "model_log_change": prediction[row_index, group_index],
                        "residual_log": residual[row_index, group_index],
                    }
                )

    ar_values = np.asarray(samples["ar_parameters"], dtype=float)
    ar_quantiles = np.percentile(ar_values.reshape((ar_values.shape[0], data["n_scas"], data["n_tiles"], AR_LAYERS, 3)), [16, 50, 84], axis=0)
    with (output_dir / "tile_ar_posterior.csv").open("w", newline="") as handle:
        fields = ["detector", "tile_id", "tile_row", "tile_col"] + ar_quantile_fields()
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sca_index, detector in enumerate(data["sca_names"]):
            for tile_index, tile_id in enumerate(data["tile_ids"]):
                tile_metadata = data["metadata"][int(tile_id)]
                row = {
                    "detector": detector,
                    "tile_id": int(tile_id),
                    "tile_row": tile_metadata["tile_row"],
                    "tile_col": tile_metadata["tile_col"],
                }
                _add_ar_quantiles(row, ar_quantiles, sca_index, tile_index)
                writer.writerow(row)

    n_scales = np.asarray(samples["absolute_hgcdte_n_log_scale"], dtype=float)
    k_scales = np.asarray(samples["absolute_hgcdte_k_log_scale"], dtype=float)
    n_quantiles = np.percentile(n_scales, [16, 50, 84], axis=0)
    k_quantiles = np.percentile(k_scales, [16, 50, 84], axis=0)
    with (output_dir / "tile_hgcdte_posterior.csv").open("w", newline="") as handle:
        fields = [
            "detector", "tile_id", "tile_row", "tile_col",
            "n_log_scale_p16", "n_log_scale_median", "n_log_scale_p84",
            "k_log_scale_p16", "k_log_scale_median", "k_log_scale_p84",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sca_index, detector in enumerate(data["sca_names"]):
            for tile_index, tile_id in enumerate(data["tile_ids"]):
                tile_metadata = data["metadata"][int(tile_id)]
                writer.writerow(
                    {
                        "detector": detector,
                        "tile_id": int(tile_id),
                        "tile_row": tile_metadata["tile_row"],
                        "tile_col": tile_metadata["tile_col"],
                        "n_log_scale_p16": n_quantiles[0, sca_index, tile_index],
                        "n_log_scale_median": n_quantiles[1, sca_index, tile_index],
                        "n_log_scale_p84": n_quantiles[2, sca_index, tile_index],
                        "k_log_scale_p16": k_quantiles[0, sca_index, tile_index],
                        "k_log_scale_median": k_quantiles[1, sca_index, tile_index],
                        "k_log_scale_p84": k_quantiles[2, sca_index, tile_index],
                    }
                )

    rate_initial = np.asarray(samples["tile_initial_rate_nm_day"], dtype=float)
    rate_final = np.asarray(samples["tile_final_rate_nm_day"], dtype=float)
    rate_mean = np.asarray(samples["tile_mean_rate_nm_day"], dtype=float)
    rate_pre_reset = np.asarray(
        samples["tile_pre_reset_mean_rate_nm_day"], dtype=float
    )
    rate_post_reset = np.asarray(
        samples["tile_post_reset_mean_rate_nm_day"], dtype=float
    )
    rate_cross_reset = np.asarray(
        samples["tile_cross_reset_mean_rate_nm_day"], dtype=float
    )
    with (output_dir / "tile_rate_posterior.csv").open("w", newline="") as handle:
        fields = [
            "detector", "tile_id", "tile_row", "tile_col",
            "log_rate_offset_p16", "log_rate_offset_median", "log_rate_offset_p84",
            "initial_rate_nm_day_p16", "initial_rate_nm_day_median", "initial_rate_nm_day_p84",
            "final_rate_nm_day_p16", "final_rate_nm_day_median", "final_rate_nm_day_p84",
            "mean_rate_nm_day_p16", "mean_rate_nm_day_median", "mean_rate_nm_day_p84",
            "pre_reset_mean_rate_nm_day_p16", "pre_reset_mean_rate_nm_day_median", "pre_reset_mean_rate_nm_day_p84",
            "post_reset_mean_rate_nm_day_p16", "post_reset_mean_rate_nm_day_median", "post_reset_mean_rate_nm_day_p84",
            "cross_reset_mean_rate_nm_day_p16", "cross_reset_mean_rate_nm_day_median", "cross_reset_mean_rate_nm_day_p84",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sca_index, detector in enumerate(data["sca_names"]):
            for tile_index, tile_id in enumerate(data["tile_ids"]):
                tile_metadata = data["metadata"][int(tile_id)]
                row = {
                    "detector": detector,
                    "tile_id": int(tile_id),
                    "tile_row": tile_metadata["tile_row"],
                    "tile_col": tile_metadata["tile_col"],
                    "log_rate_offset_p16": 0.0,
                    "log_rate_offset_median": 0.0,
                    "log_rate_offset_p84": 0.0,
                }
                for values, prefix in (
                    (rate_initial, "initial_rate_nm_day"),
                    (rate_final, "final_rate_nm_day"),
                    (rate_mean, "mean_rate_nm_day"),
                    (rate_pre_reset, "pre_reset_mean_rate_nm_day"),
                    (rate_post_reset, "post_reset_mean_rate_nm_day"),
                    (rate_cross_reset, "cross_reset_mean_rate_nm_day"),
                ):
                    row[f"{prefix}_p16"] = np.percentile(values[:, sca_index, tile_index], 16)
                    row[f"{prefix}_median"] = np.percentile(values[:, sca_index, tile_index], 50)
                    row[f"{prefix}_p84"] = np.percentile(values[:, sca_index, tile_index], 84)
                writer.writerow(row)

    warmup_prior_summary = []
    initial_reset_prior_summary = []
    if (model_metadata or {}).get("initial_reset_thickness_prior_center_nm") is not None:
        initial_reset_prior_summary = [
            f"- Initial/first-reset thickness prior center: `{(model_metadata or {}).get('initial_reset_thickness_prior_center_nm')}` nm.",
            f"- Initial/first-reset thickness prior log sigma: `{(model_metadata or {}).get('initial_reset_thickness_prior_log_sigma')}`.",
        ]
    if (model_metadata or {}).get("warmup_prior_root") is not None:
        warmup_prior_summary = [
            f"- Warmup-centered thickness prior root: `{(model_metadata or {}).get('warmup_prior_root')}`.",
            f"- Warmup-centered thickness prior log sigma: `{(model_metadata or {}).get('thickness_prior_log_sigma')}`.",
            f"- Warmup-centered thickness prior stages: `{(model_metadata or {}).get('thickness_prior_stages')}`.",
            f"- Third-Decon thickness prior center: `{(model_metadata or {}).get('third_decon_thickness_prior_center_nm')}` nm.",
        ]
    summary = "\n".join(
        [
            "# Production least-squares MAP fit",
            "",
            f"- Statistics: `{Path(statistics_path).resolve()}`.",
            f"- Optimizer: `{diagnostics['optimizer']}`.",
            f"- Evaluations: `{diagnostics['function_evaluations']}`.",
            f"- Objective: `{diagnostics['objective']:.10g}`.",
            f"- Converged: `{diagnostics['converged']}`.",
            f"- Strict gradient convergence: `{diagnostics['gradient_converged']}`.",
            f"- Function/step/gradient tolerances: `{diagnostics['function_tolerance']}` / `{diagnostics['step_tolerance']}` / `{diagnostics['gradient_tolerance']}`.",
            f"- Termination: `{diagnostics['message']}`.",
            f"- Ice reset days: `{reset_days_for_data(data).tolist()}` (physical boundaries; Program 1063 supplies the first when available).",
            f"- SED weighting: `{(model_metadata or {}).get('sed_weighting', 'energy')}`.",
            f"- QE response mode: `{(model_metadata or {}).get('qe_response_mode', 'none')}`.",
            (
                f"- Initial-thickness prior: lognormal median `{initial_thickness_log_median_nm}` nm, "
                f"log sigma `{initial_thickness_log_sigma}`."
                if initial_thickness_prior == INITIAL_THICKNESS_PRIOR_LOGNORMAL
                else (
                    "- Initial-thickness prior: half-normal location `0` nm, "
                    f"sigma `{initial_thickness_prior_sigma_nm}` nm."
                )
            ),
            f"- Trajectory prior: `{(model_metadata or {}).get('trajectory_rate_prior_mean_nm_day', TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)}` nm/day, log sigma `{trajectory_rate_prior_sigma_log}`.",
            f"- Thickness penalty: shared `{(model_metadata or {}).get('thickness_penalty_lambda_shared', 0.0)}`, disagreement `{(model_metadata or {}).get('thickness_penalty_lambda_disagreement', 0.0)}`.",
            f"- Constant ice n prior reference: `{(model_metadata or {}).get('ice_n_constant_reference', ICE_INDEX_DEFAULT)}`; log sigma `{(model_metadata or {}).get('ice_n_constant_prior_sigma_log', ICE_N_CONSTANT_PRIOR_SIGMA_LOG)}`.",
            f"- Constant ice n fixed: `{(model_metadata or {}).get('ice_n_fixed_value', 'not fixed')}`.",
            (
                f"- Ice n relationship: Lorentz-Lorenz porosity with p0 `{(model_metadata or {}).get('ice_n_porosity_p0')}`, "
                f"p_inf `{(model_metadata or {}).get('ice_n_porosity_pinf')}`, "
                f"L `{(model_metadata or {}).get('ice_n_porosity_length_nm')}` nm, "
                f"solid n `{(model_metadata or {}).get('ice_n_porosity_solid_n')}`."
                if (model_metadata or {}).get("ice_n_model") == "lorentz_lorenz_porosity"
                else "- Ice n relationship: not using the fixed porosity relation."
            ),
            f"- Trajectory regularization: `{(model_metadata or {}).get('trajectory_regularization', 'curvature')}`.",
            f"- Trajectory curvature weight: `{(model_metadata or {}).get('trajectory_smoothness_weight', NP_SMOOTHNESS_WEIGHT)}`.",
            f"- Trajectory entropy weight: `{(model_metadata or {}).get('trajectory_entropy_weight', 0.0)}`.",
            f"- HgCdTe n/k scale prior sigma: `{absolute_hgcdte_n_scale_prior_sigma_log}` / `{ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG}`.",
            f"- AR/HgCdTe modulation prior sigma: `{(model_metadata or {}).get('ar_hgcdte_modulation_prior_sigma_log', AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG)}`; HgCdTe n modulation sigma: `{hgcdte_n_modulation_prior_sigma_log}`.",
            f"- Constant ice k fitted: `{bool((model_metadata or {}).get('fit_ice_k_constant', 0))}`; prior sigma: `{(model_metadata or {}).get('ice_k_prior_sigma', ICE_K_PRIOR_SIGMA)}`.",
            f"- Constant AR k fitted: `{bool((model_metadata or {}).get('fit_ar_k_constant', 0))}`; prior sigma: `{(model_metadata or {}).get('ar_k_prior_sigma', AR_K_PRIOR_SIGMA)}`.",
            f"- HgCdTe n scale fixed: `{bool((model_metadata or {}).get('absolute_hgcdte_n_log_scale_fixed', 0))}`.",
            f"- Fixed HgCdTe n log scale: `{(model_metadata or {}).get('fixed_absolute_hgcdte_n_log_scale', 'not fixed')}`.",
            f"- Incidence angle fitted: `{bool((model_metadata or {}).get('fit_incidence_angle', 1))}`.",
            f"- Incidence pupil half-angle: `{(model_metadata or {}).get('incidence_pupil_half_angle_deg', 0.0)}` degrees.",
            f"- LED log offsets fitted: `{bool((model_metadata or {}).get('fit_led_log_offsets', 0))}`.",
            f"- LED log-offset prior sigma: `{(model_metadata or {}).get('led_log_offset_prior_sigma_log', 'not fitted')}`.",
            f"- Group-mean data RMS: `{group_rms_log:.12g}` log.",
            f"- Count-weighted data RMS: `{count_weighted_rms_log:.12g}` log.",
            f"- Mean rates: reset-aware overall `{float(np.mean(rate_mean)):.12g}`, pre-reset `{float(np.mean(rate_pre_reset)):.12g}`, post-reset `{float(np.mean(rate_post_reset)):.12g}`; cross-reset diagnostic (includes the reset jump) `{float(np.mean(rate_cross_reset)):.12g}` nm/day.",
            "- Corrections: already applied in the input statistics; no correction table used.",
            "",
        ]
        + initial_reset_prior_summary
        + warmup_prior_summary
    )
    (output_dir / "map_summary.md").write_text(summary)


def fit_one(config):
    run_start = time.perf_counter()
    data, _, map_values, diagnostics, evaluated = fit_map_state(config)
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(data["counts"], dtype=float)
    reset_days = reset_days_for_data(data)
    offset_names = reset_site_names(data, config.initial_thickness_prior)[1]
    reset_masks = np.asarray(data["days"], dtype=float)[None, :] >= reset_days[:, None]
    reset_offsets = np.stack(
        [np.asarray(map_values[name]).reshape(-1) for name in offset_names], axis=0
    )
    observed += np.sum(reset_offsets[:, :, None] * reset_masks[:, None, :], axis=0)
    fit_diagnostics = {
        "observed": observed,
        "prediction": np.asarray(evaluated["_map_least_squares_prediction"]),
    }
    model_metadata = dict(config.model_metadata or {})
    model_metadata.setdefault(
        "thickness_penalty_enabled",
        int(config.thickness_penalty_plan is not None),
    )
    model_metadata.setdefault(
        "thickness_penalty_lambda_shared",
        float(config.thickness_penalty_lambda_shared),
    )
    model_metadata.setdefault(
        "thickness_penalty_lambda_disagreement",
        float(config.thickness_penalty_lambda_disagreement),
    )
    if config.thickness_penalty_plan is not None:
        model_metadata.setdefault(
            "thickness_penalty_grid_lower_nm",
            float(config.thickness_penalty_plan["grid"][0]),
        )
        model_metadata.setdefault(
            "thickness_penalty_grid_upper_nm",
            float(config.thickness_penalty_plan["grid"][-1]),
        )
        model_metadata.setdefault(
            "thickness_penalty_grid_count",
            int(config.thickness_penalty_plan["grid"].size),
        )
    model_metadata.setdefault(
        "trajectory_rate_prior_sigma_log",
        float(config.trajectory_rate_prior_sigma_log),
    )
    model_metadata.setdefault(
        "absolute_hgcdte_n_scale_prior_sigma_log",
        float(config.absolute_hgcdte_n_scale_prior_sigma_log),
    )
    model_metadata.setdefault(
        "hgcdte_n_modulation_prior_sigma_log",
        float(
            config.hgcdte_n_modulation_prior_sigma_log
            if config.hgcdte_n_modulation_prior_sigma_log is not None
            else config.ar_hgcdte_modulation_prior_sigma_log
        ),
    )
    model_metadata.setdefault(
        "ar_hgcdte_modulation_prior_sigma_log",
        float(config.ar_hgcdte_modulation_prior_sigma_log),
    )
    model_metadata.setdefault("fit_ice_k_constant", int(config.fit_ice_k_constant))
    model_metadata.setdefault("ice_k_prior_sigma", float(config.ice_k_prior_sigma))
    model_metadata.setdefault(
        "ice_n_constant_reference", float(config.ice_n_constant_reference)
    )
    if config.ice_n_porosity_parameters is not None:
        p0, pinf, length_nm, n_solid = validate_ice_n_porosity_parameters(
            config.ice_n_porosity_parameters
        )
        model_metadata.setdefault("ice_n_model", "lorentz_lorenz_porosity")
        model_metadata.setdefault("ice_n_porosity_p0", p0)
        model_metadata.setdefault("ice_n_porosity_pinf", pinf)
        model_metadata.setdefault("ice_n_porosity_length_nm", length_nm)
        model_metadata.setdefault("ice_n_porosity_solid_n", n_solid)
    if config.fixed_ice_n_log_constant is not None:
        model_metadata.setdefault(
            "ice_n_fixed_log_constant", float(config.fixed_ice_n_log_constant)
        )
        model_metadata.setdefault(
            "ice_n_fixed_value",
            float(
                ICE_INDEX_DEFAULT
                * np.exp(config.fixed_ice_n_log_constant)
            ),
        )
    model_metadata.setdefault("fit_ar_k_constant", int(config.fit_ar_k_constant))
    model_metadata.setdefault("ar_k_prior_sigma", float(config.ar_k_prior_sigma))
    model_metadata.setdefault(
        "initial_thickness_prior",
        str(config.initial_thickness_prior).lower().replace("-", "_"),
    )
    model_metadata.setdefault(
        "initial_thickness_prior_sigma_nm",
        float(config.initial_thickness_prior_sigma_nm),
    )
    model_metadata.setdefault(
        "absolute_hgcdte_n_log_scale_fixed",
        int(config.fixed_absolute_hgcdte_n_log_scale is not None),
    )
    if config.fixed_absolute_hgcdte_n_log_scale is not None:
        model_metadata.setdefault(
            "fixed_absolute_hgcdte_n_log_scale",
            float(config.fixed_absolute_hgcdte_n_log_scale),
        )
    model_metadata.setdefault("fit_incidence_angle", int(config.fit_incidence_angle))
    model_metadata.setdefault(
        "incidence_pupil_half_angle_deg",
        float(config.incidence_pupil_half_angle_deg),
    )
    model_metadata.setdefault("fit_led_log_offsets", int(config.fit_led_log_offsets))
    model_metadata.setdefault(
        "led_log_offset_prior_sigma_log",
        float(config.led_log_offset_prior_sigma_log),
    )
    model_metadata.setdefault("sed_weighting", str(config.sed_weighting))
    model_metadata.setdefault("qe_response_mode", str(config.qe_response_mode))
    write_outputs(
        config.output_dir,
        data,
        evaluated,
        config.tile_statistics,
        diagnostics,
        fit_diagnostics,
        model_metadata,
    )
    print(
        f"Wrote {config.output_dir} in {time.perf_counter() - run_start:.2f}s",
        flush=True,
    )
    return map_values, diagnostics
