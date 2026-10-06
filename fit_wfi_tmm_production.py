"""Fixed production TMM model for independent WFI tile fits.

The production model deliberately has no exploratory command-line switches.
Each invocation evaluates the corrected sufficient statistics with a fixed
three-layer lossless AR stack, semi-infinite HgCdTe, an energy-weighted SED,
and independent per-tile latent parameters.  EMCEE initialization is handled
by ``run_wfi_tmm_emcee_warmup.py``; this module owns the deterministic MAP fit.
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
AR_LAYERS = 3
AR_DESIGN_WAVELENGTH_NM = 1050.0
AR_DIELECTRIC_K_START = 0.0
AR_LOWER = np.asarray([0.01, 1.0, 0.0], dtype=float)
AR_UPPER = np.asarray([10.0, 4.0, 0.5], dtype=float)
AR_MEAN_PRIOR_SIGMA = 10.0
ICE_INDEX_DEFAULT = 1.25
INCIDENCE_ANGLE_MAX_DEG = 15.0
INCIDENCE_ANGLE_INIT_DEG = 7.5
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
    return {
        "tile_ids": tile_ids,
        "metadata": metadata,
        "counts": counts,
        "sums": sums,
        "sum_squared": sum_squared,
        "groups": np.asarray(groups, dtype=int),
        "led": led,
        "repeat": repeat,
        "days": days,
        "first_by_led": first_by_led,
        "exposure_times": exposure_times,
        "repeat_values": repeat_values,
        "repeat_index": repeat_index,
        "node_days": node_days,
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


def build_prediction_function(data, context):
    """Build the fixed single-ray TE/TM absolute-power predictor."""
    jax, jnp, _, _ = require_numpyro()
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

    def layer_elements(index, thickness, polarization, angle_sine):
        cosine = jnp.sqrt(1.0 - (angle_sine[:, None] / index) ** 2)
        admittance = index * cosine if polarization == "s" else index / cosine
        phase = phase_factor[None, None, :] * index[:, :, None] * thickness[:, :, None] * cosine[:, :, None]
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        return (
            cosine_phase,
            sine_phase / admittance[:, :, None],
            sine_phase * admittance[:, :, None],
            cosine_phase,
        )

    def polarization_response(ar_parameters, hgcdte_parameters, ice_thickness, angle_sine, polarization):
        ar_parameters = ar_parameters.reshape((ar_parameters.shape[0], AR_LAYERS, 3))
        ar_index = ar_parameters[:, :, 1] + 1j * ar_parameters[:, :, 2]
        ar_thickness = 100.0 * ar_parameters[:, :, 0]
        substrate = base_n[None, :] * jnp.exp(hgcdte_parameters[:, 0, None]) + 1j * base_k[None, :] * jnp.exp(hgcdte_parameters[:, 1, None])
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

        ice_index = jnp.asarray(ICE_INDEX_DEFAULT + 0.0j)
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

    def prediction(ice_thickness, ar_parameters, hgcdte_parameters, incidence_angle_deg):
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
        clean_s, coated_s = polarization_response(
            ar_parameters, hgcdte_parameters, ice_thickness, angle_sine, "s"
        )
        clean_p, coated_p = polarization_response(
            ar_parameters, hgcdte_parameters, ice_thickness, angle_sine, "p"
        )
        clean = 0.5 * (clean_s + clean_p)
        coated = 0.5 * (coated_s + coated_p)
        integrated_clean = jnp.sum(clean[:, None, :] * selected_weights, axis=-1)
        integrated = jnp.sum(coated * selected_weights, axis=-1)
        current = integrated / integrated_clean
        reference = current[:, first_by_led]
        baseline = reference[:, led_index]
        return jnp.log(current / baseline) + baseline_offsets[led_index][None, :]

    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    reset_mask = jnp.asarray(np.asarray(data["days"], dtype=float) >= PRODUCTION_RESET_DAY)

    def residual_sum_squares(
        ice_thickness,
        ar_parameters,
        hgcdte_parameters,
        incidence_angle_deg,
        post_reset_tile_log_offsets=None,
        return_prediction=False,
    ):
        fitted = prediction(
            ice_thickness, ar_parameters, hgcdte_parameters, incidence_angle_deg
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
    reset_day = float(PRODUCTION_RESET_DAY)
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


def _least_squares_prior_residuals(values, data, jnp):
    reference, lower, upper = ar_prior_arrays()
    active = ar_active_parameter_indices()
    reference_raw = jnp.asarray(ar_reference_raw(reference, lower, upper, active))
    residuals = []

    def add_normal(name, center, scale):
        residuals.append(
            ((jnp.asarray(values[name]) - jnp.asarray(center)) / float(scale)).reshape(-1)
        )

    add_normal("ar_tile_raw", reference_raw, AR_MEAN_PRIOR_SIGMA)
    add_normal("initial_thickness_log_nm", np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM), INITIAL_THICKNESS_LOG_SIGMA)
    add_normal("reset_initial_thickness_log_nm", np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM), INITIAL_THICKNESS_LOG_SIGMA)
    add_normal("np_log_rate_nodes", np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY), TRAJECTORY_RATE_PRIOR_SIGMA_LOG)
    add_normal("post_reset_tile_log_offset", 0.0, POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA)
    add_normal("absolute_hgcdte_n_log_scale", 0.0, ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)
    add_normal("absolute_hgcdte_k_log_scale", 0.0, ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)
    curvature_matrix = rate_curvature_matrix(data["node_days"])
    curvature = jnp.einsum(
        "ij,tkj->tki", jnp.asarray(curvature_matrix), jnp.asarray(values["np_log_rate_nodes"])
    ).reshape(-1)
    likelihood_scale = (
        0.5
        * default_likelihood_weight(data["counts"])
        / int(data["n_tiles"])
        * float(np.sum(data["sum_squared"]))
        / LOG_NOISE_SIGMA**2
    )
    residuals.append(
        curvature
        * jnp.sqrt(2.0 * likelihood_scale * NP_SMOOTHNESS_WEIGHT / curvature.size)
    )
    return jnp.concatenate(residuals)


def build_model(data, context):
    """Build the fixed independent-tile NumPyro density."""
    if not context.get("absolute_substrate_power", False):
        raise ValueError("Production requires an absolute-power response context")
    jax, jnp, numpyro, dist = require_numpyro()
    likelihood_weight = default_likelihood_weight(data["counts"])
    data = dict(data)
    data["n_scas"] = int(data["n_scas"])
    data["n_tiles"] = int(data["n_tiles"])
    n_scas = data["n_scas"]
    n_tiles = data["n_tiles"]
    n_nodes = len(data["node_days"])
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
    predictor = build_prediction_function(data, context)
    curvature_matrix = jnp.asarray(rate_curvature_matrix(data["node_days"]))
    likelihood_scale = (
        0.5
        * likelihood_weight
        / n_tiles
        * float(np.sum(data["sum_squared"]))
        / LOG_NOISE_SIGMA**2
    )

    def state_from_values(values):
        ar_parameters = _physical_ar_parameters(values["ar_tile_raw"], jax, jnp)
        ice, rates, node_ice, node_rates = _trajectory_state(
            data,
            values["np_log_rate_nodes"],
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
            values["incidence_angle_deg"],
            post_reset_tile_log_offsets=values["post_reset_tile_log_offset"].reshape(-1),
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
        post_mask = jnp.asarray(np.asarray(data["days"]) >= PRODUCTION_RESET_DAY)
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
        prior_residual = _least_squares_prior_residuals(values, data, jnp)
        return jnp.concatenate((data_residual, prior_residual))

    def model():
        values = {
            "ar_tile_raw": numpyro.sample(
                "ar_tile_raw",
                dist.Normal(jnp.asarray(reference_raw), AR_MEAN_PRIOR_SIGMA)
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
            "incidence_angle_deg": numpyro.sample(
                "incidence_angle_deg", dist.Uniform(0.0, INCIDENCE_ANGLE_MAX_DEG)
            ),
            "np_log_rate_nodes": numpyro.sample(
                "np_log_rate_nodes",
                dist.Normal(
                    np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
                    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
                )
                .expand((n_scas, n_tiles, n_nodes))
                .to_event(3),
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
        state = state_from_values(values)
        numpyro.factor(
            "tile_gaussian_log_likelihood",
            -0.5
            * likelihood_weight
            / n_tiles
            * state["total_sse"]
            / LOG_NOISE_SIGMA**2,
        )
        curvature = jnp.einsum("ij,tkj->tki", curvature_matrix, values["np_log_rate_nodes"])
        numpyro.factor(
            "trajectory_curvature",
            -likelihood_scale * NP_SMOOTHNESS_WEIGHT * jnp.mean(curvature**2),
        )
        for name, value in state.items():
            if name not in ("total_sse", "prediction"):
                numpyro.deterministic(name, value)
        numpyro.deterministic("_map_least_squares_prediction", state["prediction"])
        numpyro.deterministic("ice_thickness", state["ice"])
        numpyro.deterministic("instantaneous_rate_nm_day", state["rates"])
        numpyro.deterministic("node_ice_thickness", state["node_ice"])
        numpyro.deterministic("node_instantaneous_rate_nm_day", state["node_rates"])
        numpyro.deterministic(
            "np_node_multipliers",
            jnp.exp(values["np_log_rate_nodes"] - np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)),
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


def initial_values_for_data(data):
    active = ar_active_parameter_indices()
    reference_raw = ar_reference_raw(*ar_prior_arrays(), active)
    shape = (int(data["n_scas"]), int(data["n_tiles"]))
    return {
        "ar_tile_raw": np.broadcast_to(reference_raw, shape + (len(active),)).copy(),
        "initial_thickness_log_nm": np.full(shape, np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM)),
        "reset_initial_thickness_log_nm": np.full(shape, np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM)),
        "incidence_angle_deg": float(INCIDENCE_ANGLE_INIT_DEG),
        "np_log_rate_nodes": np.full(
            shape + (len(data["node_days"]),),
            np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
        ),
        "post_reset_tile_log_offset": np.zeros(shape),
        "absolute_hgcdte_n_log_scale": np.zeros(shape),
        "absolute_hgcdte_k_log_scale": np.zeros(shape),
    }


def map_parameter_scale_values(initial_values):
    scales = {
        "ar_tile_raw": AR_MEAN_PRIOR_SIGMA,
        "initial_thickness_log_nm": INITIAL_THICKNESS_LOG_SIGMA,
        "reset_initial_thickness_log_nm": INITIAL_THICKNESS_LOG_SIGMA,
        "incidence_angle_deg": INCIDENCE_ANGLE_MAX_DEG,
        "np_log_rate_nodes": TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        "post_reset_tile_log_offset": POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
        "absolute_hgcdte_n_log_scale": ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        "absolute_hgcdte_k_log_scale": ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
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
):
    """Run the only production MAP optimizer in prior-width coordinates."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    flat_initial, unravel, flat_lower, flat_upper = map_parameter_bounds(initial_values, jax)
    flat_initial = np.asarray(flat_initial, dtype=float)
    parameter_scales = map_parameter_scale_values(initial_values)
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
        ftol=None,
        xtol=None,
        gtol=float(tolerance),
        max_nfev=PRODUCTION_MAP_STEPS,
        verbose=0,
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
    diagnostics = {
        "optimizer": "scipy least_squares",
        "converged": bool(result.success and scaled_gradient_norm <= tolerance),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nfev),
        "function_evaluations": int(result.nfev),
        "gradient_norm": float(np.max(np.abs(scaled_gradient / flat_scales))),
        "scaled_gradient_norm": scaled_gradient_norm,
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
            if name not in samples:
                continue
            target_array = np.asarray(target)
            value = np.asarray(samples[name], dtype=float)
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
        "ice_reset_day": np.asarray(PRODUCTION_RESET_DAY),
        "fit_post_reset_tile_log_offset": np.asarray(1),
        "post_reset_tile_log_offset_prior_sigma": np.asarray(POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA),
        "fit_incidence_angle": np.asarray(1),
        "incidence_angle_max_deg": np.asarray(INCIDENCE_ANGLE_MAX_DEG),
        "input_corrections_already_applied": np.asarray(1),
        "map_optimizer": np.asarray("least-squares"),
        "map_steps": np.asarray(PRODUCTION_MAP_STEPS),
        "map_parameter_scaling": np.asarray("prior"),
    }
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
            f"- SED weighting: `energy`.",
            f"- Initial-thickness prior: median `{INITIAL_THICKNESS_LOG_MEDIAN_NM}` nm, log sigma `{INITIAL_THICKNESS_LOG_SIGMA}`.",
            f"- Trajectory prior: `{TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY}` nm/day, log sigma `{TRAJECTORY_RATE_PRIOR_SIGMA_LOG}`.",
            f"- HgCdTe n/k scale prior sigma: `{ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG}`.",
            "- Corrections: already applied in the input statistics; no correction table used.",
            "",
        ]
    )
    (output_dir / "map_summary.md").write_text(summary)


def fit_one(config):
    run_start = time.perf_counter()
    jax, _, _, _ = require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data = load_tile_statistics(config.tile_statistics)
    context = load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    initial_values = initial_values_for_data(data)
    if config.map_init_from is not None:
        initial_values = load_map_warm_start(config.map_init_from, initial_values, data)
    model = build_model(data, context)
    map_values, diagnostics = run_least_squares_map_optimization(
        model,
        initial_values,
        jax,
    )
    evaluated = _evaluate_model(model, map_values)
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(data["counts"], dtype=float)
    observed += np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1, 1) * (
        np.asarray(data["days"])[None, :] >= PRODUCTION_RESET_DAY
    )
    fit_diagnostics = {
        "observed": observed,
        "prediction": np.asarray(evaluated["_map_least_squares_prediction"]),
    }
    write_outputs(
        config.output_dir,
        data,
        evaluated,
        config.tile_statistics,
        diagnostics,
        fit_diagnostics,
    )
    print(
        f"Wrote {config.output_dir} in {time.perf_counter() - run_start:.2f}s",
        flush=True,
    )
    return map_values, diagnostics
