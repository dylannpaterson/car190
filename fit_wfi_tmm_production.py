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
PRODUCTION_RESET_DAY = 15.064405023148149
PRODUCTION_MAP_STEPS = 10000
PRODUCTION_MAP_FTOL = 1.0e-9
PRODUCTION_MAP_XTOL = 1.0e-9
AR_LAYERS = 3
AR_DESIGN_WAVELENGTH_NM = 1050.0
AR_DIELECTRIC_K_START = 0.0
AR_LOWER = np.asarray([0.01, 1.0, 0.0], dtype=float)
AR_UPPER = np.asarray([10.0, 4.0, 0.5], dtype=float)
AR_MEAN_PRIOR_SIGMA = 10.0
ICE_INDEX_DEFAULT = 1.25
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
AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 0.02
AR_HGCDTE_MODULATION_ANCHOR_NM = 1050.0
INCIDENCE_ANGLE_MAX_DEG = 15.0
INCIDENCE_ANGLE_INIT_DEG = 7.5
PUPIL_RADIAL_NODES, PUPIL_RADIAL_WEIGHTS = np.polynomial.legendre.leggauss(4)
PUPIL_AZIMUTH_COUNT = 8
POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA = 0.1
INITIAL_THICKNESS_LOG_MEDIAN_NM = 10.0
INITIAL_THICKNESS_LOG_SIGMA = 1.5
TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 5.0
TRAJECTORY_RATE_PRIOR_SIGMA_LOG = 1.5
NP_SMOOTHNESS_WEIGHT = 0.1
ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG = 0.25
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


@dataclass(frozen=True)
class ProductionConfig:
    tile_statistics: Path
    output_dir: Path
    map_init_from: Path | None = None
    map_initial_values: dict | None = None
    smoothness_weight: float | None = None
    entropy_reference_log_rates: np.ndarray | None = None
    entropy_weight: float = 0.0
    rate_prior_log_mean: np.ndarray | None = None
    constant_rate: bool = False
    fit_incidence_angle: bool = True
    incidence_pupil_half_angle_deg: float = 0.0
    fit_ice_n_modulation: bool = False
    ice_n_modulation_prior_sigma_log: float = ICE_N_MODULATION_PRIOR_SIGMA_LOG
    ar_mean_prior_sigma: float = AR_MEAN_PRIOR_SIGMA
    fit_ar_hgcdte_modulation: bool = False
    ar_hgcdte_modulation_prior_sigma_log: float = AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG
    model_metadata: dict | None = None


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


def _prepare_response_integration(
    model_wavelengths,
    spectra,
    qe_wavelengths,
):
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
        weighted_output = np.where(supported_qe, _sed_output(spectrum), 0.0)
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


def load_response_context(sca_names):
    """Load the fixed energy-weighted absolute-power response context."""
    spectra = load_sed(SED_URL)
    qe_wavelengths, _, qe_fieldnames = load_qe(QE_URL)
    model_wavelengths = np.unique(
        np.concatenate(
            [np.asarray(spectrum["wavelength"], dtype=float) for spectrum in spectra.values()]
        )
    )
    qe_columns = qe_columns_for_scas(sca_names, qe_fieldnames)
    # Absolute substrate power intentionally omits the measured QE multiplier.
    integration = _prepare_response_integration(
        model_wavelengths, spectra, qe_wavelengths
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
    }


def detector_name_from_statistics_path(path):
    return Path(path).name.split("_tile_log_statistics_", 1)[0].upper()


def load_tile_statistics(path):
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
    return {
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
        "sca_names": np.asarray([detector_name_from_statistics_path(path)]),
        "sca_index": np.asarray([0], dtype=int),
        "n_scas": 1,
        "n_tiles": 1,
    }


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


def integrate_rate_trajectory(jax, jnp, days, node_days, integration_times, log_rates):
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

    return (
        jax.vmap(integrated_at)(days),
        instantaneous_rate(days),
        jax.vmap(integrated_at)(node_days),
        instantaneous_rate(node_days),
    )


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
    fit_ice_n_modulation=False,
    fit_ar_hgcdte_modulation=False,
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
        ar_n_log_modulation=None,
        hgcdte_n_log_modulation=None,
        hgcdte_k_log_modulation=None,
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
        angle_sine = jnp.broadcast_to(
            _incidence_sine(incidence_angle_deg, jnp), (ar_parameters.shape[0],)
        )
        if fit_ice_n_modulation:
            if ice_n_log_modulation is None:
                raise ValueError("Ice n modulation values are required")
            modulation = jnp.asarray(ice_n_log_modulation).reshape(
                (ar_parameters.shape[0], ice_modulation_basis_value.shape[1])
            )
            ice_index = (
                ICE_INDEX_DEFAULT
                * jnp.exp(modulation @ ice_modulation_basis_value.T)
            )[:, None, :].astype(jnp.complex128)
        else:
            ice_index = jnp.asarray(ICE_INDEX_DEFAULT + 0.0j)
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
            if fit_ice_n_modulation:
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
        integrated_clean = jnp.sum(clean[:, None, :] * selected_weights, axis=-1)
        integrated = jnp.sum(coated * selected_weights, axis=-1)
        current = integrated / integrated_clean
        reference = current[:, first_by_led]
        baseline = reference[:, led_index]
        return jnp.log(current / baseline) + baseline_offsets[led_index][None, :]

    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    reset_day = float(data.get("reset_day", PRODUCTION_RESET_DAY))
    reset_mask = jnp.asarray(np.asarray(data["days"], dtype=float) >= reset_day)

    def residual_sum_squares(
        ice_thickness,
        ar_parameters,
        hgcdte_parameters,
        incidence_angle_deg,
        post_reset_tile_log_offsets=None,
        ice_n_log_modulation=None,
        ar_n_log_modulation=None,
        hgcdte_n_log_modulation=None,
        hgcdte_k_log_modulation=None,
        return_prediction=False,
    ):
        fitted = prediction(
            ice_thickness,
            ar_parameters,
            hgcdte_parameters,
            incidence_angle_deg,
            ice_n_log_modulation=ice_n_log_modulation,
            ar_n_log_modulation=ar_n_log_modulation,
            hgcdte_n_log_modulation=hgcdte_n_log_modulation,
            hgcdte_k_log_modulation=hgcdte_k_log_modulation,
        )
        observed = sums / jnp.maximum(counts, 1.0)
        if post_reset_tile_log_offsets is not None:
            offsets = jnp.asarray(post_reset_tile_log_offsets).reshape(-1)
            observed = observed + offsets[:, None] * reset_mask[None, :]
        safe_counts = jnp.maximum(counts, 1.0)
        within_group = sum_squared - sums * sums / safe_counts
        total = jnp.sum(within_group + counts * (fitted - observed) ** 2)
        if return_prediction:
            return total, fitted
        return total

    prediction.residual_sum_squares = residual_sum_squares
    return prediction


def _trajectory_state(data, log_rate_nodes, initial_log, reset_log, jax, jnp):
    days = jnp.asarray(data["days"])
    node_days = jnp.asarray(data["node_days"])
    reset_day = float(data.get("reset_day", PRODUCTION_RESET_DAY))
    integration_times = np.unique(
        np.concatenate(
            (
                [float(np.min(data["days"]))],
                np.asarray(data["node_days"], dtype=float),
                [float(np.max(data["days"]))],
                [reset_day],
            )
        )
    )
    integration_times = integration_times[(integration_times >= np.min(data["days"])) & (integration_times <= np.max(data["days"]))]
    profiles = jnp.asarray(log_rate_nodes).reshape((-1, len(data["node_days"])))
    integrated, rates, node_integrated, node_rates = jax.vmap(
        lambda profile: integrate_rate_trajectory(
            jax,
            jnp,
            days,
            node_days,
            jnp.asarray(integration_times),
            profile,
        )
    )(profiles)
    # The reset integral is evaluated directly for every tile so the reset
    # remains differentiable with respect to its non-parametric rate nodes.
    reset_integrated = jax.vmap(
        lambda profile: integrate_rate_trajectory(
            jax,
            jnp,
            jnp.asarray([reset_day]),
            node_days,
            jnp.asarray(integration_times),
            profile,
        )[0][0]
    )(profiles)
    initial = jnp.exp(jnp.asarray(initial_log)).reshape(-1)
    reset_initial = jnp.exp(jnp.asarray(reset_log)).reshape(-1)
    post = days[None, :] >= reset_day
    integrated = jnp.where(post, integrated - reset_integrated[:, None], integrated)
    node_post = node_days[None, :] >= reset_day
    node_integrated = jnp.where(
        node_post, node_integrated - reset_integrated[:, None], node_integrated
    )
    ice = jnp.where(post, reset_initial[:, None], initial[:, None]) + integrated
    node_ice = jnp.where(node_post, reset_initial[:, None], initial[:, None]) + node_integrated
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
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    fit_ar_hgcdte_modulation=False,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
):
    if smoothness_weight is None:
        smoothness_weight = NP_SMOOTHNESS_WEIGHT
    reference, lower, upper = ar_prior_arrays()
    active = ar_active_parameter_indices()
    reference_raw = jnp.asarray(ar_reference_raw(reference, lower, upper, active))
    residuals = []

    def add_normal(name, center, scale):
        residuals.append(
            ((jnp.asarray(values[name]) - jnp.asarray(center)) / float(scale)).reshape(-1)
        )

    add_normal("ar_tile_raw", reference_raw, ar_mean_prior_sigma)
    add_normal("initial_thickness_log_nm", np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM), INITIAL_THICKNESS_LOG_SIGMA)
    add_normal("reset_initial_thickness_log_nm", np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM), INITIAL_THICKNESS_LOG_SIGMA)
    if rate_prior_log_mean is None:
        rate_prior_log_mean = np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)
    add_normal("np_log_rate_nodes", rate_prior_log_mean, TRAJECTORY_RATE_PRIOR_SIGMA_LOG)
    add_normal("post_reset_tile_log_offset", 0.0, POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA)
    add_normal("absolute_hgcdte_n_log_scale", 0.0, ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)
    add_normal("absolute_hgcdte_k_log_scale", 0.0, ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)
    if "ice_n_log_modulation" in values:
        add_normal(
            "ice_n_log_modulation",
            0.0,
            ice_n_modulation_prior_sigma_log,
        )
    if fit_ar_hgcdte_modulation:
        for name in (
            "ar_n_log_modulation",
            "hgcdte_n_log_modulation",
            "hgcdte_k_log_modulation",
        ):
            add_normal(name, 0.0, ar_hgcdte_modulation_prior_sigma_log)
    likelihood_scale = (
        0.5
        * default_likelihood_weight(data["counts"])
        / int(data["n_tiles"])
        * float(np.sum(data["sum_squared"]))
        / LOG_NOISE_SIGMA**2
    )
    if smoothness_weight:
        curvature_matrix = rate_curvature_matrix(data["node_days"])
        curvature = jnp.einsum(
            "ij,tkj->tki", jnp.asarray(curvature_matrix), jnp.asarray(values["np_log_rate_nodes"])
        ).reshape(-1)
        residuals.append(
            curvature
            * jnp.sqrt(2.0 * likelihood_scale * smoothness_weight / curvature.size)
        )
    if entropy_reference_log_rates is not None and entropy_weight:
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
    constant_rate=False,
    fit_incidence_angle=True,
    incidence_pupil_half_angle_deg=0.0,
    fit_ice_n_modulation=False,
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    fit_ar_hgcdte_modulation=False,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
):
    """Build the independent-tile NumPyro density.

    The default arguments preserve the legacy curvature-regularized model.
    Production MaxEnt fits pass the regularization arguments explicitly.
    """
    if not context.get("absolute_substrate_power", False):
        raise ValueError("Production requires an absolute-power response context")
    if smoothness_weight is None:
        smoothness_weight = NP_SMOOTHNESS_WEIGHT
    if smoothness_weight < 0.0 or entropy_weight < 0.0:
        raise ValueError("Regularization weights must be non-negative")
    if (
        not np.isfinite(ice_n_modulation_prior_sigma_log)
        or ice_n_modulation_prior_sigma_log <= 0.0
    ):
        raise ValueError("Ice n modulation prior sigma must be positive and finite")
    if (
        not np.isfinite(ar_hgcdte_modulation_prior_sigma_log)
        or ar_hgcdte_modulation_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "AR/HgCdTe modulation prior sigma must be positive and finite"
        )
    jax, jnp, numpyro, dist = require_numpyro()
    likelihood_weight = default_likelihood_weight(data["counts"])
    data = dict(data)
    data["n_scas"] = int(data["n_scas"])
    data["n_tiles"] = int(data["n_tiles"])
    n_scas = data["n_scas"]
    n_tiles = data["n_tiles"]
    n_nodes = len(data["node_days"])
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
    reset_day = float(data.get("reset_day", PRODUCTION_RESET_DAY))
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
    predictor = build_prediction_function(
        data,
        context,
        incidence_pupil_half_angle_deg=incidence_pupil_half_angle_deg,
        fit_ice_n_modulation=fit_ice_n_modulation,
        fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
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

    def state_from_values(values):
        ar_parameters = _physical_ar_parameters(values["ar_tile_raw"], jax, jnp)
        ice, rates, node_ice, node_rates = _trajectory_state(
            data,
            rate_nodes_from_values(values),
            values["initial_thickness_log_nm"],
            values["reset_initial_thickness_log_nm"],
            jax,
            jnp,
        )
        row_ar = ar_parameters.reshape((n_rows, 3 * AR_LAYERS))
        hgcdte = jnp.stack(
            (values["absolute_hgcdte_n_log_scale"], values["absolute_hgcdte_k_log_scale"]),
            axis=-1,
        ).reshape((n_rows, 2))
        total_sse, prediction = predictor.residual_sum_squares(
            ice,
            row_ar,
            hgcdte,
            values["incidence_angle_deg"] if fit_incidence_angle else 0.0,
            post_reset_tile_log_offsets=values["post_reset_tile_log_offset"].reshape(-1),
            ice_n_log_modulation=(
                values["ice_n_log_modulation"]
                if fit_ice_n_modulation
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
        state = state_from_values(values)
        counts = jnp.asarray(data["counts"])
        observed = jnp.asarray(data["sums"]) / jnp.maximum(counts, 1.0)
        post_mask = jnp.asarray(np.asarray(data["days"]) >= reset_day)
        observed = observed + (
            jnp.asarray(values["post_reset_tile_log_offset"]).reshape(-1, 1)
            * post_mask[None, :]
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
            ice_n_modulation_prior_sigma_log=ice_n_modulation_prior_sigma_log,
            ar_mean_prior_sigma=ar_mean_prior_sigma,
            fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
            ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_modulation_prior_sigma_log,
        )
        return jnp.concatenate((data_residual, prior_residual))

    def model():
        values = {
            "ar_tile_raw": numpyro.sample(
                "ar_tile_raw",
                dist.Normal(jnp.asarray(reference_raw), ar_mean_prior_sigma)
                .expand((n_scas, n_tiles, len(active)))
                .to_event(3),
            ),
            "initial_thickness_log_nm": numpyro.sample(
                "initial_thickness_log_nm",
                dist.Normal(np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM), INITIAL_THICKNESS_LOG_SIGMA)
                .expand((n_scas, n_tiles))
                .to_event(2),
            ),
            "reset_initial_thickness_log_nm": numpyro.sample(
                "reset_initial_thickness_log_nm",
                dist.Normal(np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM), INITIAL_THICKNESS_LOG_SIGMA)
                .expand((n_scas, n_tiles))
                .to_event(2),
            ),
            "post_reset_tile_log_offset": numpyro.sample(
                "post_reset_tile_log_offset",
                dist.Normal(0.0, POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA)
                .expand((n_scas, n_tiles))
                .to_event(2),
            ),
            "absolute_hgcdte_n_log_scale": numpyro.sample(
                "absolute_hgcdte_n_log_scale",
                dist.Normal(0.0, ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)
                .expand((n_scas, n_tiles))
                .to_event(2),
            ),
            "absolute_hgcdte_k_log_scale": numpyro.sample(
                "absolute_hgcdte_k_log_scale",
                dist.Normal(0.0, ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)
                .expand((n_scas, n_tiles))
                .to_event(2),
            ),
        }
        if fit_incidence_angle:
            values["incidence_angle_deg"] = numpyro.sample(
                "incidence_angle_deg",
                dist.Uniform(0.0, INCIDENCE_ANGLE_MAX_DEG),
            )
        if constant_rate:
            values["log_rate_scale"] = numpyro.sample(
                "log_rate_scale",
                dist.Normal(
                    jnp.asarray(rate_prior_log_mean[..., 0]),
                    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
                ).to_event(2),
            )
        else:
            values["np_log_rate_nodes"] = numpyro.sample(
                "np_log_rate_nodes",
                dist.Normal(
                    jnp.asarray(rate_prior_log_mean),
                    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
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
                        ar_hgcdte_modulation_prior_sigma_log,
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
        state = state_from_values(values)
        numpyro.factor(
            "tile_gaussian_log_likelihood",
            -0.5
            * likelihood_weight
            / n_tiles
            * state["total_sse"]
            / LOG_NOISE_SIGMA**2,
        )
        if smoothness_weight:
            curvature = jnp.einsum(
                "ij,tkj->tki", curvature_matrix, rate_nodes_from_values(values)
            )
            numpyro.factor(
                "trajectory_curvature",
                -likelihood_scale * smoothness_weight * jnp.mean(curvature**2),
            )
        if entropy_reference_log_rates is not None and entropy_weight:
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
        numpyro.deterministic("_map_least_squares_prediction", state["prediction"])
        numpyro.deterministic("ice_thickness", state["ice"])
        numpyro.deterministic("instantaneous_rate_nm_day", state["rates"])
        numpyro.deterministic("node_ice_thickness", state["node_ice"])
        numpyro.deterministic("node_instantaneous_rate_nm_day", state["node_rates"])
        numpyro.deterministic(
            "np_node_multipliers",
            jnp.exp(
                rate_nodes_from_values(values)
                - np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)
            ),
        )
        numpyro.deterministic("tile_ice_thickness", state["ice"].reshape((n_scas, n_tiles, -1)))
        numpyro.deterministic("tile_instantaneous_rate_nm_day", state["rates"].reshape((n_scas, n_tiles, -1)))
        numpyro.deterministic("np_node_thickness", state["node_ice"].reshape((n_scas, n_tiles, -1)))
        numpyro.deterministic("tile_initial_rate_nm_day", state["rates"][:, 0].reshape((n_scas, n_tiles)))
        numpyro.deterministic("tile_final_rate_nm_day", state["rates"][:, -1].reshape((n_scas, n_tiles)))
        numpyro.deterministic(
            "tile_mean_rate_nm_day",
            ((state["ice"][:, -1] - state["ice"][:, 0]) / (jnp.max(jnp.asarray(data["days"])) - jnp.min(jnp.asarray(data["days"]))))
            .reshape((n_scas, n_tiles)),
        )
        numpyro.deterministic(
            "mean_rate_nm_day",
            jnp.mean(
                (state["ice"][:, -1] - state["ice"][:, 0])
                / (jnp.max(jnp.asarray(data["days"])) - jnp.min(jnp.asarray(data["days"])))
            ),
        )

    model.production_residuals = production_residuals
    model.production_state = state_from_values
    model.production_predictor = predictor
    return model


def initial_values_for_data(
    data,
    constant_rate=False,
    fit_incidence_angle=True,
    fit_ice_n_modulation=False,
    fit_ar_hgcdte_modulation=False,
):
    active = ar_active_parameter_indices()
    reference_raw = ar_reference_raw(*ar_prior_arrays(), active)
    shape = (int(data["n_scas"]), int(data["n_tiles"]))
    values = {
        "ar_tile_raw": np.broadcast_to(reference_raw, shape + (len(active),)).copy(),
        "initial_thickness_log_nm": np.full(shape, np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM)),
        "reset_initial_thickness_log_nm": np.full(shape, np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM)),
        "post_reset_tile_log_offset": np.zeros(shape),
        "absolute_hgcdte_n_log_scale": np.zeros(shape),
        "absolute_hgcdte_k_log_scale": np.zeros(shape),
    }
    if fit_incidence_angle:
        values["incidence_angle_deg"] = float(INCIDENCE_ANGLE_INIT_DEG)
    if constant_rate:
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
    return values


def map_parameter_scale_values(
    initial_values,
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
):
    scales = {
        "ar_tile_raw": ar_mean_prior_sigma,
        "initial_thickness_log_nm": INITIAL_THICKNESS_LOG_SIGMA,
        "reset_initial_thickness_log_nm": INITIAL_THICKNESS_LOG_SIGMA,
        "incidence_angle_deg": INCIDENCE_ANGLE_MAX_DEG,
        "np_log_rate_nodes": TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        "log_rate_scale": TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        "post_reset_tile_log_offset": POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
        "absolute_hgcdte_n_log_scale": ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        "absolute_hgcdte_k_log_scale": ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        "ice_n_log_modulation": ice_n_modulation_prior_sigma_log,
        "ar_n_log_modulation": ar_hgcdte_modulation_prior_sigma_log,
        "hgcdte_n_log_modulation": ar_hgcdte_modulation_prior_sigma_log,
        "hgcdte_k_log_modulation": ar_hgcdte_modulation_prior_sigma_log,
    }
    return {
        name: np.full(np.asarray(value).shape, float(scales[name]))
        for name, value in initial_values.items()
    }


def map_parameter_bounds(initial_values, jax):
    from jax.flatten_util import ravel_pytree

    lower = {}
    upper = {}
    for name, value in initial_values.items():
        shape = np.asarray(value).shape
        lower[name] = np.full(shape, -np.inf)
        upper[name] = np.full(shape, np.inf)
        if name == "incidence_angle_deg":
            lower[name] = np.zeros(shape)
            upper[name] = np.full(shape, INCIDENCE_ANGLE_MAX_DEG)
    flat_initial, unravel = ravel_pytree({name: jax.numpy.asarray(value) for name, value in initial_values.items()})
    flat_lower, _ = ravel_pytree({name: jax.numpy.asarray(value) for name, value in lower.items()})
    flat_upper, _ = ravel_pytree({name: jax.numpy.asarray(value) for name, value in upper.items()})
    return flat_initial, unravel, flat_lower, flat_upper


def run_least_squares_map_optimization(
    model,
    initial_values,
    jax,
    tolerance=1.0e-9,
    ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
):
    """Run the only production MAP optimizer in prior-width coordinates."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    flat_initial, unravel, flat_lower, flat_upper = map_parameter_bounds(initial_values, jax)
    flat_initial = np.asarray(flat_initial, dtype=float)
    parameter_scales = map_parameter_scale_values(
        initial_values,
        ice_n_modulation_prior_sigma_log=ice_n_modulation_prior_sigma_log,
        ar_mean_prior_sigma=ar_mean_prior_sigma,
        ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_modulation_prior_sigma_log,
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
        values = unravel(
            jax.numpy.asarray(flat_initial) + jax.numpy.asarray(flat_scales) * scaled_values
        )
        return model.production_residuals(values)

    residual_jit = jax.jit(residuals)
    jacobian_jit = jax.jit(jax.jacfwd(residuals))

    def scipy_residual(scaled_values):
        return np.asarray(residual_jit(jax.numpy.asarray(scaled_values)), dtype=float)

    def scipy_jacobian(scaled_values):
        return np.asarray(jacobian_jit(jax.numpy.asarray(scaled_values)), dtype=float)

    initial_scaled = np.zeros_like(flat_initial)
    initial_residual = scipy_residual(initial_scaled)
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        bounds=(scaled_lower, scaled_upper),
        method="trf",
        x_scale=1.0,
        ftol=PRODUCTION_MAP_FTOL,
        xtol=PRODUCTION_MAP_XTOL,
        gtol=float(tolerance),
        max_nfev=PRODUCTION_MAP_STEPS,
        verbose=2,
    )
    optimized_scaled = np.asarray(result.x, dtype=float)
    optimized_flat = flat_initial + flat_scales * optimized_scaled
    optimized_values = {
        name: np.asarray(value)
        for name, value in unravel(jax.numpy.asarray(optimized_flat)).items()
    }
    final_residual = scipy_residual(optimized_scaled)
    jacobian = scipy_jacobian(optimized_scaled)
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
        "gradient_norm": float(np.max(np.abs(scaled_gradient / flat_scales))),
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
        "parameter_scale_min": float(np.min(flat_scales)),
        "parameter_scale_max": float(np.max(flat_scales)),
    }
    return optimized_values, diagnostics


def load_map_warm_start(
    path,
    initial_values,
    data,
):
    """Load the fixed one-tile EMCEE template into a MAP start."""
    samples_path = Path(path) / "posterior_samples.npz" if Path(path).is_dir() else Path(path)
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)
    with np.load(samples_path, allow_pickle=False) as samples:
        if not bool(np.asarray(samples["map_only"]).item()):
            raise ValueError("Production warm starts must be deterministic map-only archives")
        if "sca_names" in samples and not np.array_equal(
            np.asarray(samples["sca_names"]).astype(str),
            np.asarray(data["sca_names"]).astype(str),
        ):
            raise ValueError("EMCEE warm-start detector does not match the target tile")
        transferred = dict(initial_values)
        for name, target in initial_values.items():
            source_name = name
            if name == "log_rate_scale" and name not in samples:
                source_name = "np_log_rate_nodes"
            if source_name not in samples:
                continue
            target_array = np.asarray(target)
            value = np.asarray(samples[source_name], dtype=float)
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
    data = load_tile_statistics(config.tile_statistics)
    context = load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    initial_values = initial_values_for_data(
        data,
        constant_rate=config.constant_rate,
        fit_incidence_angle=config.fit_incidence_angle,
        fit_ice_n_modulation=config.fit_ice_n_modulation,
        fit_ar_hgcdte_modulation=config.fit_ar_hgcdte_modulation,
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
    model = build_model(
        data,
        context,
        smoothness_weight=config.smoothness_weight,
        entropy_reference_log_rates=config.entropy_reference_log_rates,
        entropy_weight=config.entropy_weight,
        rate_prior_log_mean=config.rate_prior_log_mean,
        constant_rate=config.constant_rate,
        fit_incidence_angle=config.fit_incidence_angle,
        incidence_pupil_half_angle_deg=config.incidence_pupil_half_angle_deg,
        fit_ice_n_modulation=config.fit_ice_n_modulation,
        ice_n_modulation_prior_sigma_log=config.ice_n_modulation_prior_sigma_log,
        ar_mean_prior_sigma=config.ar_mean_prior_sigma,
        fit_ar_hgcdte_modulation=config.fit_ar_hgcdte_modulation,
        ar_hgcdte_modulation_prior_sigma_log=(
            config.ar_hgcdte_modulation_prior_sigma_log
        ),
    )
    map_values, diagnostics = run_least_squares_map_optimization(
        model,
        initial_values,
        jax,
        ice_n_modulation_prior_sigma_log=config.ice_n_modulation_prior_sigma_log,
        ar_mean_prior_sigma=config.ar_mean_prior_sigma,
        ar_hgcdte_modulation_prior_sigma_log=(
            config.ar_hgcdte_modulation_prior_sigma_log
        ),
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
        "sed_mode": np.asarray("core"),
        "sed_weighting": np.asarray("energy"),
        "wavelength_stride": np.asarray(1),
        "fully_np_trajectory": np.asarray(1),
        "whiten_np_trajectory": np.asarray(0),
        "trajectory_rate_prior_mean_nm_day": np.asarray(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
        "trajectory_rate_prior_sigma_log": np.asarray(TRAJECTORY_RATE_PRIOR_SIGMA_LOG),
        "tile_initial_thickness_lognormal": np.asarray(1),
        "tile_initial_thickness_log_median_nm": np.asarray(INITIAL_THICKNESS_LOG_MEDIAN_NM),
        "tile_initial_thickness_log_sigma": np.asarray(INITIAL_THICKNESS_LOG_SIGMA),
        "ice_reset_day": np.asarray(float(data.get("reset_day", PRODUCTION_RESET_DAY))),
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
    }
    for name, value in (model_metadata or {}).items():
        if value is not None:
            metadata[str(name)] = np.asarray(value)
    np.savez_compressed(output_dir / "posterior_samples.npz", **samples, **metadata)

    observed = np.asarray(fit_diagnostics["observed"], dtype=float)
    prediction = np.asarray(fit_diagnostics["prediction"], dtype=float)
    residual = observed - prediction
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
    with (output_dir / "tile_rate_posterior.csv").open("w", newline="") as handle:
        fields = [
            "detector", "tile_id", "tile_row", "tile_col",
            "log_rate_offset_p16", "log_rate_offset_median", "log_rate_offset_p84",
            "initial_rate_nm_day_p16", "initial_rate_nm_day_median", "initial_rate_nm_day_p84",
            "final_rate_nm_day_p16", "final_rate_nm_day_median", "final_rate_nm_day_p84",
            "mean_rate_nm_day_p16", "mean_rate_nm_day_median", "mean_rate_nm_day_p84",
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
                ):
                    row[f"{prefix}_p16"] = np.percentile(values[:, sca_index, tile_index], 16)
                    row[f"{prefix}_median"] = np.percentile(values[:, sca_index, tile_index], 50)
                    row[f"{prefix}_p84"] = np.percentile(values[:, sca_index, tile_index], 84)
                writer.writerow(row)

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
            f"- Ice reset day: `{float(data.get('reset_day', PRODUCTION_RESET_DAY))}` (input Program 1063 boundary when available).",
            f"- SED weighting: `energy`.",
            f"- Initial-thickness prior: median `{INITIAL_THICKNESS_LOG_MEDIAN_NM}` nm, log sigma `{INITIAL_THICKNESS_LOG_SIGMA}`.",
            f"- Trajectory prior: `{TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY}` nm/day, log sigma `{TRAJECTORY_RATE_PRIOR_SIGMA_LOG}`.",
            f"- Trajectory regularization: `{(model_metadata or {}).get('trajectory_regularization', 'curvature')}`.",
            f"- Trajectory curvature weight: `{(model_metadata or {}).get('trajectory_smoothness_weight', NP_SMOOTHNESS_WEIGHT)}`.",
            f"- Trajectory entropy weight: `{(model_metadata or {}).get('trajectory_entropy_weight', 0.0)}`.",
            f"- HgCdTe n/k scale prior sigma: `{ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG}`.",
            f"- Incidence angle fitted: `{bool((model_metadata or {}).get('fit_incidence_angle', 1))}`.",
            f"- Incidence pupil half-angle: `{(model_metadata or {}).get('incidence_pupil_half_angle_deg', 0.0)}` degrees.",
            "- Corrections: already applied in the input statistics; no correction table used.",
            "",
        ]
    )
    (output_dir / "map_summary.md").write_text(summary)


def fit_one(config):
    run_start = time.perf_counter()
    data, _, map_values, diagnostics, evaluated = fit_map_state(config)
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(data["counts"], dtype=float)
    observed += np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1, 1) * (
        np.asarray(data["days"])[None, :] >= float(data.get("reset_day", PRODUCTION_RESET_DAY))
    )
    fit_diagnostics = {
        "observed": observed,
        "prediction": np.asarray(evaluated["_map_least_squares_prediction"]),
    }
    model_metadata = dict(config.model_metadata or {})
    model_metadata.setdefault("fit_incidence_angle", int(config.fit_incidence_angle))
    model_metadata.setdefault(
        "incidence_pupil_half_angle_deg",
        float(config.incidence_pupil_half_angle_deg),
    )
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
