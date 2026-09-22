"""NumPyro TMM fit with shared ice and hierarchical SCA/tile AR variation."""

import argparse
import csv
import hashlib
import io
from pathlib import Path

import numpy as np

from analyze_wfi01_all_pixel_soc_model import parse_mast_time
from assess_rcs_spectral_identifiability import (
    QE_URL,
    SED_URL,
    load_qe,
    load_sed,
    read_source,
)
from fit_wfi01_joint_complex_q_soc import prepare_response_integration


ROOT = Path(__file__).resolve().parent
DEFAULT_SINGLE_SCA_TILE_STATISTICS = ROOT / "wfi01_tile_log_statistics_512.csv"
DEFAULT_ALL_SCA_STATISTICS_DIR = ROOT / "wfi_tile_log_statistics_512"
DEFAULT_MODEL_PATH = Path(
    "/Users/paterson.35/Documents/"
    "rgespit-comissioning-ice-simulations/data/ice_optical_constants/"
    "spectral_response_model_SCA1.ecsv"
)
DEFAULT_SINGLE_SCA_OUTPUT = ROOT / "wfi01_tmm_numpyro_tile_pilot"
DEFAULT_ALL_SCA_OUTPUT = ROOT / "wfi_all_sca_tmm_numpyro_tile_production"
DEFAULT_STEP_TIME = "2026-09-14T23:38:00Z"
ASW_OPTICAL_CONSTANTS_URL = (
    "https://sbnarchive.psi.edu/pds4/non_mission/"
    "gbo.ices.mastrapa.lab-spectra/data/nk/amorph_high.tab"
)

REFERENCE_THICKNESS_NM = 122.7
REFERENCE_TIME_DAYS = 4.65
RESET_DAY = 18.0
INITIAL_THICKNESS_INIT_NM = 35.9
TRAJECTORY_SCALE_INIT = 1.14
TRAJECTORY_SCALE_START = 2.0
NOMINAL_HGCDTE = np.asarray([3.6, 0.7], dtype=float)
HGCDTE_LOWER = np.asarray([1.0, 0.0], dtype=float)
HGCDTE_UPPER = np.asarray([6.0, 3.0], dtype=float)
HGCDTE_PRIOR_SIGMA = np.asarray([1.0, 1.0], dtype=float)
CONDUCTIVE_REFERENCE_WAVELENGTH_NM = 1600.0
CONDUCTIVE_EPSILON_INF = 1.0
CONDUCTIVE_LOG_THICKNESS_PRIOR_MEAN = np.log(5.0)
CONDUCTIVE_LOG_PLASMA_STRENGTH_PRIOR_MEAN = np.log(1.0)
CONDUCTIVE_LOG_DAMPING_RATIO_PRIOR_MEAN = np.log(1.0)
CONDUCTIVE_LOG_PRIOR_SIGMA = 1.5
QE_RBF_NODE_WAVELENGTHS_NM = np.asarray([650.0, 950.0, 1250.0, 1600.0])
QE_RBF_WIDTH_NM = 250.0
QE_RBF_PRIOR_SIGMA_LOG = 0.15
AR_LOWER = np.asarray([0.01, 1.0, 0.0], dtype=float)
AR_UPPER = np.asarray([10.0, 4.0, 0.5], dtype=float)
AR_MEAN_PRIOR_SIGMA = 2.0
AR_DIELECTRIC_K_START = 1.0e-4
AR_DESIGN_WAVELENGTH_NM = 1050.0
AR_MAX_LAYERS = 3
AR_SCA_SIGMA_PRIOR = 0.2
AR_TILE_SIGMA_PRIOR = 0.5
NP_ENTROPY_WEIGHT = 0.01
NP_SMOOTHNESS_WEIGHT = 4.64158883361
NP_MAX_NODE_MULTIPLIER = 2.0
PREDICTION_ROW_CHUNK_SIZE = 64
GAUSS_LEGENDRE_NODES, GAUSS_LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(8)
TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG = 0.10
TRAJECTORY_TILE_RATE_NEIGHBOR_SIGMA_LOG = 0.05


def qe_rbf_configuration(
    node_wavelengths_nm=None,
    width_nm=QE_RBF_WIDTH_NM,
    prior_sigma_log=QE_RBF_PRIOR_SIGMA_LOG,
):
    """Validate and return the wavelength-dependent QE basis configuration."""
    if node_wavelengths_nm is None:
        node_wavelengths_nm = QE_RBF_NODE_WAVELENGTHS_NM
    node_wavelengths_nm = np.asarray(node_wavelengths_nm, dtype=float).reshape(-1)
    if (
        node_wavelengths_nm.size < 2
        or not np.all(np.isfinite(node_wavelengths_nm))
        or np.any(node_wavelengths_nm <= 0.0)
        or np.any(np.diff(node_wavelengths_nm) <= 0.0)
    ):
        raise ValueError(
            "QE RBF nodes must contain at least two strictly increasing positive wavelengths"
        )
    if not np.isfinite(width_nm) or width_nm <= 0.0:
        raise ValueError("QE RBF width must be positive and finite")
    if not np.isfinite(prior_sigma_log) or prior_sigma_log <= 0.0:
        raise ValueError("QE RBF prior sigma must be positive and finite")
    return node_wavelengths_nm, float(width_nm), float(prior_sigma_log)


def load_asw_optical_constants(source=ASW_OPTICAL_CONSTANTS_URL):
    """Load the 100 K amorphous-water-ice n/k table in micrometers."""
    values = np.loadtxt(io.StringIO(read_source(source)), ndmin=2)
    if values.shape[1] < 3 or values.shape[0] < 2:
        raise ValueError("ASW optical-constant table must contain wavelength, n, and k")
    wavelengths_nm = 1000.0 * values[:, 0]
    real_index = values[:, 1]
    imaginary_index = values[:, 2]
    if (
        np.any(~np.isfinite(wavelengths_nm))
        or np.any(~np.isfinite(real_index))
        or np.any(~np.isfinite(imaginary_index))
        or np.any(np.diff(wavelengths_nm) <= 0.0)
        or np.any(imaginary_index < 0.0)
    ):
        raise ValueError("ASW optical constants must be finite, ordered, and passive")
    return wavelengths_nm, real_index, imaginary_index


def dielectric_ar_reference(
    ar_layers,
    design_wavelength_nm=AR_DESIGN_WAVELENGTH_NM,
    middle_layer_eighth_wave=False,
    middle_layer_half_wave=False,
):
    """Return a lossless equal-index-ratio matching-stack reference."""
    ar_layers = int(ar_layers)
    if ar_layers < 1 or ar_layers > AR_MAX_LAYERS:
        raise ValueError(f"ar_layers must be between one and {AR_MAX_LAYERS}")
    if not np.isfinite(design_wavelength_nm) or design_wavelength_nm <= 0.0:
        raise ValueError("design_wavelength_nm must be positive and finite")
    if middle_layer_eighth_wave and middle_layer_half_wave:
        raise ValueError("AR middle-layer thickness choices are mutually exclusive")
    if (middle_layer_eighth_wave or middle_layer_half_wave) and ar_layers != 3:
        raise ValueError("A special middle-layer thickness requires three AR layers")
    layer_indices = NOMINAL_HGCDTE[0] ** (
        np.arange(1, ar_layers + 1, dtype=float) / (ar_layers + 1)
    )
    optical_divisors = np.full(ar_layers, 4.0)
    if middle_layer_eighth_wave:
        optical_divisors[1] = 8.0
    elif middle_layer_half_wave:
        optical_divisors[1] = 2.0
    thickness_scales = design_wavelength_nm / (
        optical_divisors * layer_indices * 100.0
    )
    return np.column_stack(
        (
            thickness_scales,
            layer_indices,
            np.full(ar_layers, AR_DIELECTRIC_K_START),
        )
    ).reshape(-1)


def ar_prior_arrays(ar_layers):
    """Return layer-major dielectric AR prior centers and physical bounds."""
    reference = dielectric_ar_reference(ar_layers)
    return (
        reference,
        np.tile(AR_LOWER, ar_layers),
        np.tile(AR_UPPER, ar_layers),
    )


def default_likelihood_weight(counts):
    """Scale pixel SSE to approximately one effective datum per tile/group."""
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ValueError("Likelihood counts must be positive and finite")
    return 1.0 / float(np.median(counts))


def trajectory_rate_per_scale(days):
    """Return the baseline instantaneous rate at the first observed time."""
    days = np.asarray(days, dtype=float)
    if days.size < 2 or not np.all(np.isfinite(days)) or np.ptp(days) <= 0.0:
        raise ValueError("Trajectory days must span at least two finite times")
    first_day = np.min(days)
    return float(
        REFERENCE_THICKNESS_NM
        / (
            2.0
            * np.sqrt(REFERENCE_TIME_DAYS)
            * np.sqrt(RESET_DAY + first_day)
        )
    )


def integrate_rate_trajectory(
    jax,
    jnp,
    days,
    node_days,
    integration_times,
    log_scale,
    log_node_multipliers,
    use_square_root=True,
):
    """Integrate a positive, log-linearly interpolated instantaneous rate."""
    gauss_nodes = jnp.asarray(GAUSS_LEGENDRE_NODES)
    gauss_weights = jnp.asarray(GAUSS_LEGENDRE_WEIGHTS)
    days = jnp.asarray(days)
    node_days = jnp.asarray(node_days)
    integration_times = jnp.asarray(integration_times)
    rate_scale = jnp.exp(log_scale)

    def instantaneous_rate(time):
        if use_square_root:
            base_rate = REFERENCE_THICKNESS_NM / (
                2.0
                * jnp.sqrt(REFERENCE_TIME_DAYS)
                * jnp.sqrt(RESET_DAY + time)
            )
        else:
            base_rate = 1.0
        log_multiplier = jnp.interp(time, node_days, log_node_multipliers)
        return rate_scale * base_rate * jnp.exp(log_multiplier)

    def integrate_interval(left, right):
        left = jnp.asarray(left)[..., None]
        right = jnp.asarray(right)[..., None]
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        return half_width[..., 0] * jnp.sum(
            gauss_weights * instantaneous_rate(
                midpoint + half_width * gauss_nodes
            ),
            axis=-1,
        )

    interval_integrals = jax.vmap(
        integrate_interval
    )(integration_times[:-1], integration_times[1:])
    cumulative_integrals = jnp.concatenate(
        (
            jnp.zeros(1, dtype=interval_integrals.dtype),
            jnp.cumsum(interval_integrals),
        )
    )

    def integrated_at(target_days):
        segment_indices = jnp.searchsorted(
            integration_times, target_days, side="right"
        ) - 1
        segment_indices = jnp.clip(segment_indices, 0, len(integration_times) - 2)
        return cumulative_integrals[segment_indices] + integrate_interval(
            integration_times[segment_indices], target_days
        )

    return (
        integrated_at(days),
        instantaneous_rate(days),
        integrated_at(node_days),
        instantaneous_rate(node_days),
    )


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
            digest.update(
                np.ascontiguousarray(data[key][row], dtype=np.float64).tobytes()
            )
        fingerprints.append(digest.hexdigest())
    return np.asarray(fingerprints)


def is_single_tile_data(data):
    return int(data["n_scas"]) == 1 and int(data["n_tiles"]) == 1


def tile_neighbor_edges(data):
    """Return adjacent selected tile-index pairs using tile row/column metadata."""
    tile_ids = np.asarray(data.get("tile_ids", []), dtype=int)
    metadata = data.get("metadata", {})
    positions = []
    for tile_id in tile_ids:
        tile_metadata = metadata.get(int(tile_id))
        if tile_metadata is None:
            return np.empty((0, 2), dtype=int)
        positions.append(
            (
                int(tile_metadata["tile_row"]),
                int(tile_metadata["tile_col"]),
            )
        )
    edges = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            if sum(
                abs(positions[first][axis] - positions[second][axis])
                for axis in (0, 1)
            ) == 1:
                edges.append((first, second))
    return np.asarray(edges, dtype=int).reshape((-1, 2))


def run_map_initialization(model, initial_values, jax, init_to_value, seed, steps, learning_rate):
    """Use Adam through AutoDelta as an optional NUTS initialization path."""
    from numpyro.infer import SVI, Trace_ELBO, autoguide
    from numpyro.optim import Adam

    guide = autoguide.AutoDelta(
        model,
        init_loc_fn=init_to_value(values=initial_values),
    )
    svi = SVI(model, guide, Adam(step_size=learning_rate), Trace_ELBO())
    losses = []
    best_loss = np.inf
    best_params = None
    state = None
    checkpoint_steps = 100
    base_key = jax.random.PRNGKey(seed)
    for start in range(0, steps, checkpoint_steps):
        result = svi.run(
            jax.random.fold_in(base_key, start),
            min(checkpoint_steps, steps - start),
            progress_bar=False,
            init_state=state,
        )
        state = result.state
        chunk_losses = np.asarray(result.losses)
        losses.extend(chunk_losses.tolist())
        chunk_loss = float(chunk_losses[-1])
        if np.isfinite(chunk_loss) and chunk_loss < best_loss:
            best_loss = chunk_loss
            best_params = jax.tree_util.tree_map(
                lambda value: value.copy(), result.params
            )
    losses = np.asarray(losses)
    if best_params is None:
        raise ValueError("MAP initialization produced a non-finite objective")
    print(
        f"MAP initialization: {steps} steps; best loss={best_loss:.6g}; "
        f"final loss={losses[-1]:.6g}"
    )
    map_values = guide.median(best_params)
    return (
        {**initial_values, **map_values},
        {
            "optimizer": "Adam/AutoDelta",
            "converged": False,
            "status": -1,
            "message": "Checkpointed initializer; convergence was not assessed.",
            "iterations": int(steps),
            "function_evaluations": int(steps),
            "gradient_norm": np.nan,
            "objective": float(best_loss),
            "initial_objective": float(losses[0]),
            "parameter_step_norm": np.nan,
            "objective_history": losses,
        },
    )


def run_map_optimization(model, initial_values, jax, maxiter, tolerance=1.0e-9):
    """Optimize the exact NumPyro joint log density with bounded L-BFGS-B."""
    from numpyro.infer.util import log_density
    from scipy.optimize import minimize

    initial_pytree = {
        name: jax.numpy.asarray(value) for name, value in initial_values.items()
    }
    flat_initial, unravel = jax.flatten_util.ravel_pytree(initial_pytree)

    def lower_bound(name, value):
        shape = np.asarray(value).shape
        if name == "initial_thickness_nm":
            return np.full(shape, 1.0e-8)
        if name == "hgcdte_n":
            return np.full(shape, HGCDTE_LOWER[0])
        if name == "hgcdte_k":
            return np.full(shape, HGCDTE_LOWER[1])
        if name in ("ar_sigma_raw", "ar_sca_sigma", "ar_tile_sigma"):
            return np.full(shape, 1.0e-8)
        return np.full(shape, -np.inf)

    def upper_bound(name, value):
        shape = np.asarray(value).shape
        if name == "initial_thickness_nm":
            return np.full(shape, 50.0 - 1.0e-8)
        if name == "hgcdte_n":
            return np.full(shape, HGCDTE_UPPER[0])
        if name == "hgcdte_k":
            return np.full(shape, HGCDTE_UPPER[1])
        return np.full(shape, np.inf)

    flat_lower, _ = jax.flatten_util.ravel_pytree(
        {
            name: jax.numpy.asarray(lower_bound(name, value))
            for name, value in initial_values.items()
        }
    )
    flat_upper, _ = jax.flatten_util.ravel_pytree(
        {
            name: jax.numpy.asarray(upper_bound(name, value))
            for name, value in initial_values.items()
        }
    )

    def negative_log_joint(flat_values):
        log_joint, _ = log_density(model, (), {}, unravel(flat_values))
        return -log_joint

    value_and_gradient = jax.jit(jax.value_and_grad(negative_log_joint))

    def scipy_objective(flat_values):
        value, gradient = value_and_gradient(jax.numpy.asarray(flat_values))
        return float(value), np.asarray(gradient, dtype=float)

    initial_objective, _ = scipy_objective(np.asarray(flat_initial, dtype=float))
    objective_history = [initial_objective]

    def record_objective(flat_values):
        value, _ = scipy_objective(flat_values)
        objective_history.append(value)

    result = minimize(
        scipy_objective,
        np.asarray(flat_initial, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(np.asarray(flat_lower), np.asarray(flat_upper))),
        callback=record_objective,
        options={
            "maxiter": int(maxiter),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
            "maxls": 50,
        },
    )
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise ValueError("L-BFGS-B MAP optimization produced a non-finite objective")
    map_values = {
        name: np.asarray(value)
        for name, value in unravel(jax.numpy.asarray(result.x)).items()
    }
    gradient_norm = float(np.max(np.abs(np.asarray(result.jac, dtype=float))))
    gradient_converged = gradient_norm <= max(1.0e-6, 100.0 * tolerance)
    diagnostics = {
        "optimizer": "L-BFGS-B",
        "converged": bool(result.success and gradient_converged),
        "gradient_converged": bool(gradient_converged),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_norm": gradient_norm,
        "objective": float(result.fun),
        "initial_objective": float(initial_objective),
        "parameter_step_norm": float(
            np.linalg.norm(result.x - np.asarray(flat_initial, dtype=float))
        ),
        "objective_history": np.asarray(objective_history, dtype=float),
    }
    print(
        f"{diagnostics['optimizer']} MAP: iterations={diagnostics['iterations']}; "
        f"objective={diagnostics['objective']:.6g}; "
        f"gradient_inf_norm={diagnostics['gradient_norm']:.3g}; "
        f"converged={diagnostics['converged']}"
    )
    return map_values, diagnostics


def scope_label(sca_names):
    names = []
    for value in np.asarray(sca_names).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode()
        names.append(str(value))
    if len(names) == 1:
        return names[0]
    return f"{len(names)} WFI SCAs"


def require_numpyro():
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, init_to_value
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The NumPyro TMM fit requires jax and numpyro; use the bulge_env environment"
        ) from error
    return jax, jnp, numpyro, dist, MCMC, NUTS, init_to_value


def select_tile_ids(tile_ids, max_tiles):
    tile_ids = np.asarray(sorted(tile_ids), dtype=int)
    if max_tiles is None or max_tiles >= len(tile_ids):
        return tile_ids
    if max_tiles < 1:
        raise ValueError("max_tiles must be positive")
    indices = np.linspace(0, len(tile_ids) - 1, max_tiles).round().astype(int)
    return tile_ids[np.unique(indices)]


def statistics_paths(path):
    if path.is_dir():
        paths = sorted(path.glob("wfi*_tile_log_statistics_*.csv"))
        if not paths:
            raise ValueError(f"No completed SCA statistics CSVs found in {path}")
        return paths
    return [path]


def detector_name_from_statistics_path(path):
    prefix = path.name.split("_tile_log_statistics_", 1)[0]
    return prefix.upper()


def qe_columns_for_scas(sca_names, qe_fieldnames):
    """Map WFI detector labels to the corresponding public SCU QE columns."""
    field_to_column = {}
    for column, fieldname in enumerate(qe_fieldnames):
        normalized = str(fieldname).strip().upper()
        if normalized in field_to_column:
            raise ValueError(f"Duplicate QE field name: {fieldname}")
        field_to_column[normalized] = column

    columns = []
    for value in np.asarray(sca_names).reshape(-1):
        detector = str(value).strip().upper()
        suffix = detector[3:] if detector.startswith("WFI") else ""
        if not suffix.isdigit() or not 1 <= int(suffix) <= 18:
            raise ValueError(f"Expected a WFI01-WFI18 detector label, got {value!r}")
        fieldname = f"SCU {int(suffix)}"
        if fieldname not in field_to_column:
            raise ValueError(f"QE table does not contain the required column {fieldname}")
        columns.append(field_to_column[fieldname])

    columns = np.asarray(columns, dtype=int)
    if len(np.unique(columns)) != len(columns):
        raise ValueError("SCA statistics contain duplicate detector-to-QE assignments")
    return columns


def load_tile_statistics(path, max_tiles=None, max_groups=None):
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be positive")
    paths = statistics_paths(path)
    sca_counts = []
    sca_sums = []
    sca_sum_squared = []
    tile_ids = None
    metadata = None
    groups = None
    reference = None
    sca_names = []
    for statistics_path in paths:
        with statistics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No tile rows found in {statistics_path}")
        all_tile_ids = sorted({int(row["tile_id"]) for row in rows})
        selected_tile_ids = select_tile_ids(all_tile_ids, max_tiles)
        if tile_ids is None:
            tile_ids = selected_tile_ids
        elif not np.array_equal(tile_ids, selected_tile_ids):
            raise ValueError("SCA statistics do not have the same tile IDs")
        selected = set(tile_ids.tolist())
        rows = [row for row in rows if int(row["tile_id"]) in selected]
        current_groups = sorted({int(row["group"]) for row in rows})
        if max_groups is not None:
            current_groups = current_groups[:max_groups]
            selected_groups = set(current_groups)
            rows = [row for row in rows if int(row["group"]) in selected_groups]
        if groups is None:
            groups = current_groups
        elif groups != current_groups:
            raise ValueError("SCA statistics do not have the same observation groups")
        group_to_index = {group: index for index, group in enumerate(groups)}
        tile_to_index = {tile_id: index for index, tile_id in enumerate(tile_ids)}
        n_tiles = len(tile_ids)
        n_groups = len(groups)
        counts = np.zeros((n_tiles, n_groups), dtype=float)
        sums = np.zeros((n_tiles, n_groups), dtype=float)
        sum_squared = np.zeros((n_tiles, n_groups), dtype=float)
        led = np.zeros(n_groups, dtype=int)
        repeat = np.zeros(n_groups, dtype=int)
        days = np.zeros(n_groups, dtype=float)
        exposure_times = [None] * n_groups
        current_metadata = {}
        for row in rows:
            tile_id = int(row["tile_id"])
            tile_index = tile_to_index[tile_id]
            group = int(row["group"])
            group_index = group_to_index[group]
            counts[tile_index, group_index] = float(row["n_pair_valid"])
            sums[tile_index, group_index] = float(row["sum_log_change"])
            sum_squared[tile_index, group_index] = float(row["sum_squared_log_change"])
            led[group_index] = int(row["led"])
            repeat[group_index] = int(row["repeat"])
            days[group_index] = float(row["days"])
            exposure_times[group_index] = row["exposure_start_time"]
            current_metadata[tile_id] = {
                "tile_row": int(row["tile_row"]),
                "tile_col": int(row["tile_col"]),
                "row_start": int(row["row_start"]),
                "row_stop_exclusive": int(row["row_stop_exclusive"]),
                "col_start": int(row["col_start"]),
                "col_stop_exclusive": int(row["col_stop_exclusive"]),
            }
        if not np.array_equal(current_groups, np.arange(n_groups)):
            raise ValueError(
                "Tile statistics groups must span zero through the final group"
            )
        if np.any(counts <= 0.0):
            raise ValueError("Selected tiles contain empty observation groups")
        if reference is None:
            reference = (led, repeat, days, exposure_times)
            metadata = current_metadata
        else:
            reference_led, reference_repeat, reference_days, reference_times = reference
            if not np.array_equal(led, reference_led) or not np.array_equal(
                repeat, reference_repeat
            ):
                raise ValueError("SCA statistics have inconsistent LED/repeat metadata")
            if not np.allclose(days, reference_days) or exposure_times != reference_times:
                raise ValueError("SCA statistics have inconsistent time metadata")
        sca_counts.append(counts)
        sca_sums.append(sums)
        sca_sum_squared.append(sum_squared)
        sca_names.append(detector_name_from_statistics_path(statistics_path))

    led, repeat, days, exposure_times = reference
    counts = np.concatenate(sca_counts, axis=0)
    sums = np.concatenate(sca_sums, axis=0)
    sum_squared = np.concatenate(sca_sum_squared, axis=0)
    n_scas = len(sca_names)
    n_tiles = len(tile_ids)
    n_groups = len(groups)

    first_by_led = np.asarray(
        [np.flatnonzero(led == led_value)[0] for led_value in range(1, 7)],
        dtype=int,
    )
    after_event = np.asarray(
        [parse_mast_time(value) >= parse_mast_time(DEFAULT_STEP_TIME) for value in exposure_times],
        dtype=bool,
    )
    step_design = after_event.astype(float) - after_event[first_by_led[led - 1]]
    repeat_values = np.asarray(sorted(set(repeat.tolist())), dtype=int)
    repeat_to_index = {repeat_value: index for index, repeat_value in enumerate(repeat_values)}
    repeat_index = np.asarray(
        [repeat_to_index[repeat_value] for repeat_value in repeat],
        dtype=int,
    )
    node_days = np.asarray(
        [np.mean(days[repeat == repeat_value]) for repeat_value in repeat_values],
        dtype=float,
    )
    if len(node_days) < 3 or np.any(np.diff(node_days) <= 0.0):
        raise ValueError("Repeat nodes must contain at least three strictly increasing times")
    return {
        "tile_ids": tile_ids,
        "metadata": metadata,
        "counts": counts,
        "sums": sums,
        "sum_squared": sum_squared,
        "led": led,
        "repeat": repeat,
        "days": days,
        "first_by_led": first_by_led,
        "step_design": step_design,
        "exposure_times": exposure_times,
        "repeat_values": repeat_values,
        "repeat_index": repeat_index,
        "node_days": node_days,
        "sca_names": np.asarray(sca_names),
        "sca_index": np.repeat(np.arange(n_scas, dtype=int), n_tiles),
        "n_scas": n_scas,
        "n_tiles": n_tiles,
    }


def load_response_context(
    model_path,
    sed_mode,
    wavelength_stride,
    sca_names,
    wavelength_step_nm=None,
    ice_optical_constants_source=None,
):
    if wavelength_stride < 1:
        raise ValueError("wavelength_stride must be positive")
    if wavelength_step_nm is not None and (
        not np.isfinite(wavelength_step_nm) or wavelength_step_nm <= 0.0
    ):
        raise ValueError("wavelength_step_nm must be positive and finite")
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, qe_fieldnames = load_qe(QE_URL)
    del model_path  # Retained for compatibility; production TMM does not use its values.
    if wavelength_step_nm is None:
        wavelength_nodes = [np.asarray([qe_wavelengths[0], qe_wavelengths[-1]])]
        for spectrum in spectra.values():
            spectrum_wavelengths = np.asarray(spectrum["wavelength"])
            wavelength_nodes.append(
                spectrum_wavelengths[
                    (spectrum_wavelengths >= qe_wavelengths[0])
                    & (spectrum_wavelengths <= qe_wavelengths[-1])
                ]
            )
        model_wavelengths = np.unique(np.concatenate(wavelength_nodes))
    else:
        model_wavelengths = np.arange(
            qe_wavelengths[0],
            qe_wavelengths[-1] + 0.5 * wavelength_step_nm,
            wavelength_step_nm,
            dtype=float,
        )
        model_wavelengths = model_wavelengths[
            model_wavelengths <= qe_wavelengths[-1]
        ]
        if model_wavelengths.size == 0 or model_wavelengths[-1] < qe_wavelengths[-1]:
            model_wavelengths = np.append(model_wavelengths, qe_wavelengths[-1])
    model_wavelengths = model_wavelengths[::wavelength_stride]
    if model_wavelengths[-1] < qe_wavelengths[-1]:
        model_wavelengths = np.append(model_wavelengths, qe_wavelengths[-1])
    qe_columns = qe_columns_for_scas(sca_names, qe_fieldnames)
    integrations = [
        prepare_response_integration(
            model_wavelengths,
            spectra,
            qe_wavelengths,
            qe_values,
            sed_mode,
            qe_column=int(qe_column),
        )
        for qe_column in qe_columns
    ]
    context = {
        "wavelengths": model_wavelengths,
        "model_weights": np.stack([integration[0] for integration in integrations]),
        "outside_output": np.stack([integration[1] for integration in integrations]),
        "clean_output": np.stack([integration[2] for integration in integrations]),
        "qe_columns": qe_columns,
        "qe_fieldnames": np.asarray(qe_fieldnames),
    }
    if ice_optical_constants_source is not None:
        ice_wavelengths, ice_n, ice_k = load_asw_optical_constants(
            ice_optical_constants_source
        )
        context["ice_n"] = np.interp(
            model_wavelengths, ice_wavelengths, ice_n, left=ice_n[0], right=ice_n[-1]
        )
        context["ice_k"] = np.interp(
            model_wavelengths, ice_wavelengths, ice_k, left=0.0, right=0.0
        )
        context["ice_optical_constants_source"] = np.asarray(
            str(ice_optical_constants_source)
        )
    return context


def build_prediction_function(
    data,
    context,
    row_indices=None,
    fit_hgcdte=False,
    fit_conductive_layer=False,
    fit_qe_rbf=False,
    qe_rbf_node_wavelengths_nm=None,
    qe_rbf_width_nm=QE_RBF_WIDTH_NM,
    use_asw_optical_constants=False,
):
    jax, jnp, _, _, _, _, _ = require_numpyro()
    wavelengths = jnp.asarray(context["wavelengths"])
    model_weights = jnp.asarray(context["model_weights"])
    outside_output = jnp.asarray(context["outside_output"])
    clean_output = jnp.asarray(context["clean_output"])
    n_scas = int(data["n_scas"])
    all_sca_index = np.asarray(data["sca_index"], dtype=int)
    if row_indices is None:
        row_indices = np.arange(len(all_sca_index), dtype=int)
    else:
        row_indices = np.asarray(row_indices, dtype=int).reshape(-1)
        if row_indices.size == 0 or np.any(row_indices < 0) or np.any(
            row_indices >= len(all_sca_index)
        ):
            raise ValueError("Prediction row indices are outside the tile data")
    sca_index = jnp.asarray(all_sca_index[row_indices], dtype=jnp.int32)
    if model_weights.ndim == 2:
        model_weights = jnp.broadcast_to(model_weights[None, :], (n_scas,) + model_weights.shape)
    if outside_output.ndim == 1:
        outside_output = jnp.broadcast_to(outside_output[None, :], (n_scas,) + outside_output.shape)
    if clean_output.ndim == 1:
        clean_output = jnp.broadcast_to(clean_output[None, :], (n_scas,) + clean_output.shape)
    if (
        model_weights.ndim != 3
        or outside_output.ndim != 2
        or clean_output.ndim != 2
        or model_weights.shape[0] != n_scas
        or outside_output.shape[0] != n_scas
        or clean_output.shape[0] != n_scas
    ):
        raise ValueError("Response integration arrays must have one entry per SCA")
    led_index = jnp.asarray(data["led"] - 1, dtype=jnp.int32)
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    step_design = jnp.asarray(data["step_design"])
    row_model_weights = model_weights[sca_index]
    row_outside_output = outside_output[sca_index]
    row_clean_output = clean_output[sca_index]
    if use_asw_optical_constants:
        if "ice_n" not in context or "ice_k" not in context:
            raise ValueError("ASW optical constants are missing from the response context")
        ice_index = jnp.asarray(context["ice_n"]) + 1j * jnp.asarray(
            context["ice_k"]
        )
    else:
        ice_index = 1.25 + 0.0j
    selected_model_weights = row_model_weights[:, led_index, :]
    selected_outside_output = row_outside_output[:, led_index]
    selected_clean_output = row_clean_output[:, led_index]
    all_group_indices = jnp.arange(led_index.shape[0], dtype=jnp.int32)
    if fit_qe_rbf:
        qe_rbf_nodes, qe_rbf_width_nm, _ = qe_rbf_configuration(
            qe_rbf_node_wavelengths_nm,
            qe_rbf_width_nm,
        )
        qe_rbf_nodes = jnp.asarray(qe_rbf_nodes)
        qe_rbf_basis = jnp.exp(
            -0.5
            * (
                (wavelengths[:, None] - qe_rbf_nodes[None, :])
                / qe_rbf_width_nm
            )
            ** 2
        )
        qe_rbf_basis = qe_rbf_basis / jnp.sum(
            qe_rbf_basis, axis=1, keepdims=True
        )

    def layer_matrix(index, thickness):
        index, thickness = jnp.broadcast_arrays(
            jnp.atleast_1d(index), jnp.atleast_1d(thickness)
        )
        phase = (
            2.0
            * jnp.pi
            * index[..., None]
            * thickness[..., None]
            / wavelengths
        )
        cosine = jnp.cos(phase)
        # With n_complex = n + i*k, forward propagation decays as exp(i*phase).
        # The corresponding characteristic matrix therefore uses -i*sin(phase).
        sine = -1j * jnp.sin(phase)
        matrix = jnp.zeros(
            index.shape + (wavelengths.shape[0], 2, 2), dtype=jnp.complex128
        )
        matrix = matrix.at[..., 0, 0].set(cosine)
        matrix = matrix.at[..., 1, 1].set(cosine)
        matrix = matrix.at[..., 0, 1].set(sine / index[..., None])
        matrix = matrix.at[..., 1, 0].set(sine * index[..., None])
        return matrix

    def wavelength_dependent_layer_matrix(index, thickness):
        thickness = jnp.atleast_1d(thickness)
        phase = (
            2.0
            * jnp.pi
            * thickness[..., None]
            * index[None, :]
            / wavelengths[None, :]
        )
        cosine = jnp.cos(phase)
        sine = -1j * jnp.sin(phase)
        matrix = jnp.zeros(
            thickness.shape + (wavelengths.shape[0], 2, 2),
            dtype=jnp.complex128,
        )
        matrix = matrix.at[..., 0, 0].set(cosine)
        matrix = matrix.at[..., 1, 1].set(cosine)
        matrix = matrix.at[..., 0, 1].set(sine / index[None, :])
        matrix = matrix.at[..., 1, 0].set(sine * index[None, :])
        return matrix

    def substrate_index_for_parameters(hgcdte_parameters):
        if not fit_hgcdte:
            return NOMINAL_HGCDTE[0] + 1j * NOMINAL_HGCDTE[1]
        return hgcdte_parameters[0] + 1j * hgcdte_parameters[1]

    def conductive_index_for_parameters(conductive_parameters):
        if not fit_conductive_layer:
            return None, None
        log_thickness, log_plasma_strength, log_damping_ratio = conductive_parameters
        thickness = jnp.exp(log_thickness)
        plasma_strength = jnp.exp(log_plasma_strength)
        damping_ratio = jnp.exp(log_damping_ratio)
        omega_ratio = CONDUCTIVE_REFERENCE_WAVELENGTH_NM / wavelengths
        epsilon = CONDUCTIVE_EPSILON_INF - plasma_strength / (
            omega_ratio**2 + 1j * damping_ratio * omega_ratio
        )
        return jnp.sqrt(epsilon), thickness

    def conductive_layer_matrix(conductive_parameters, n_rows):
        index, thickness = conductive_index_for_parameters(conductive_parameters)
        phase = (
            2.0
            * jnp.pi
            * index[None, :]
            * thickness
            / wavelengths[None, :]
        )
        cosine = jnp.cos(phase)
        sine = -1j * jnp.sin(phase)
        matrix = jnp.zeros(
            (1, wavelengths.shape[0], 2, 2), dtype=jnp.complex128
        )
        matrix = matrix.at[..., 0, 0].set(cosine)
        matrix = matrix.at[..., 1, 1].set(cosine)
        matrix = matrix.at[..., 0, 1].set(sine / index[None, :])
        matrix = matrix.at[..., 1, 0].set(sine * index[None, :])
        return jnp.broadcast_to(
            matrix, (n_rows, wavelengths.shape[0], 2, 2)
        )

    def transmission(matrix, substrate_index):
        denominator = (
            matrix[..., 0, 0]
            + substrate_index * matrix[..., 0, 1]
            + matrix[..., 1, 0]
            + substrate_index * matrix[..., 1, 1]
        )
        amplitude = 2.0 / denominator
        return jnp.real(substrate_index) * jnp.abs(amplitude) ** 2

    def multilayer_matrix(indices, thicknesses):
        n_rows, n_layers = indices.shape
        matrices = layer_matrix(
            indices.reshape((-1,)), thicknesses.reshape((-1,))
        )
        matrices = matrices.reshape(
            (n_rows, n_layers, wavelengths.shape[0], 2, 2)
        )
        result = matrices[:, 0]
        for layer_index in range(1, n_layers):
            result = jnp.einsum(
                "rwab,rwbc->rwac", result, matrices[:, layer_index]
            )
        return result

    def response_terms(
        ar_parameters, substrate_index, conductive_parameters=None
    ):
        ar_parameters = ar_parameters.reshape((ar_parameters.shape[0], -1, 3))
        ar_index = ar_parameters[:, :, 1] + 1j * ar_parameters[:, :, 2]
        ar_matrix = multilayer_matrix(
            ar_index, 100.0 * ar_parameters[:, :, 0]
        )
        if fit_conductive_layer:
            conductive_matrix = conductive_layer_matrix(
                conductive_parameters, ar_parameters.shape[0]
            )
            ar_matrix = jnp.einsum(
                "rwab,rwbc->rwac", ar_matrix, conductive_matrix
            )
        clean = transmission(ar_matrix, substrate_index)
        ar_first = ar_matrix[:, :, 0, 0] + substrate_index * ar_matrix[:, :, 0, 1]
        ar_second = ar_matrix[:, :, 1, 0] + substrate_index * ar_matrix[:, :, 1, 1]
        return clean, ar_first, ar_second

    def response_curves(
        ice_thickness,
        response_terms_value,
        group_indices,
        model_weights_value,
        outside_output_value,
        clean_output_value,
        hgcdte_parameters=None,
        qe_rbf_parameters=None,
    ):
        substrate_index = substrate_index_for_parameters(hgcdte_parameters)
        clean, ar_first, ar_second = response_terms_value
        ice_matrix = (
            wavelength_dependent_layer_matrix(ice_index, ice_thickness)
            if use_asw_optical_constants
            else layer_matrix(ice_index, ice_thickness)
        )
        # Evaluate the denominator of ice @ AR directly instead of materializing
        # the full (tile, group, wavelength, 2, 2) product tensor.
        ice_first = ice_matrix[..., 0, 0] + ice_matrix[..., 1, 0]
        ice_second = ice_matrix[..., 0, 1] + ice_matrix[..., 1, 1]
        if ice_first.ndim == 2:
            ice_first = ice_first[None, :, :]
            ice_second = ice_second[None, :, :]
        denominator = (
            ar_first[:, None, :] * ice_first
            + ar_second[:, None, :] * ice_second
        )
        coated = jnp.real(substrate_index) * jnp.abs(2.0 / denominator) ** 2
        if fit_qe_rbf:
            qe_multiplier = jnp.exp(qe_rbf_basis @ qe_rbf_parameters)
            model_weights_value = (
                model_weights_value * qe_multiplier[None, None, :]
            )
            clean_output_value = (
                jnp.sum(model_weights_value, axis=-1) + outside_output_value
            )
        ratios = coated / clean[:, None, :]
        model_weights = model_weights_value[:, group_indices, :]
        outside_output = outside_output_value[:, group_indices]
        clean_output = clean_output_value[:, group_indices]
        integrated = jnp.sum(ratios * model_weights, axis=-1)
        integrated = integrated + outside_output
        return integrated / clean_output

    def prediction_for_rows(
        ice_thickness,
        ar_parameters,
        acm_step,
        model_weights_value,
        outside_output_value,
        clean_output_value,
        hgcdte_parameters=None,
        conductive_parameters=None,
        qe_rbf_parameters=None,
    ):
        substrate_index = substrate_index_for_parameters(hgcdte_parameters)
        terms = response_terms(
            ar_parameters, substrate_index, conductive_parameters
        )
        response = response_curves(
            ice_thickness,
            terms,
            all_group_indices,
            model_weights_value,
            outside_output_value,
            clean_output_value,
            hgcdte_parameters,
            qe_rbf_parameters,
        )
        current = response
        baseline_ice_thickness = (
            ice_thickness[first_by_led]
            if ice_thickness.ndim == 1
            else ice_thickness[:, first_by_led]
        )
        baseline_response = response_curves(
            baseline_ice_thickness,
            terms,
            first_by_led,
            model_weights_value,
            outside_output_value,
            clean_output_value,
            hgcdte_parameters,
            qe_rbf_parameters,
        )
        baseline_response = baseline_response[:, led_index]
        baseline = baseline_response
        acm_step = jnp.asarray(acm_step)
        observation_acm_step = (
            acm_step[led_index] if acm_step.ndim == 1 else acm_step
        )
        return jnp.log(current / baseline) + observation_acm_step * step_design[None, :]

    def prediction(
        ice_thickness,
        ar_parameters,
        acm_step,
        hgcdte_parameters=None,
        conductive_parameters=None,
        qe_rbf_parameters=None,
    ):
        return prediction_for_rows(
            ice_thickness,
            ar_parameters,
            acm_step,
            selected_model_weights,
            selected_outside_output,
            selected_clean_output,
            hgcdte_parameters,
            conductive_parameters,
            qe_rbf_parameters,
        )

    if all(key in data for key in ("counts", "sums", "sum_squared")):
        selected_counts = np.asarray(data["counts"])[row_indices]
        selected_sums = np.asarray(data["sums"])[row_indices]
        selected_sum_squared = np.asarray(data["sum_squared"])[row_indices]
        n_rows = int(selected_counts.shape[0])
        row_chunk_size = min(
            int(data.get("prediction_row_chunk_size", PREDICTION_ROW_CHUNK_SIZE)),
            n_rows,
        )
        n_chunks = (n_rows + row_chunk_size - 1) // row_chunk_size
        padded_rows = n_chunks * row_chunk_size
        row_padding = padded_rows - n_rows

        def pad_rows(value, mode="edge"):
            return jnp.pad(value, ((0, row_padding),) + ((0, 0),) * (value.ndim - 1), mode=mode)

        padded_counts = pad_rows(jnp.asarray(selected_counts), mode="constant")
        padded_sums = pad_rows(jnp.asarray(selected_sums), mode="constant")
        padded_sum_squared = pad_rows(
            jnp.asarray(selected_sum_squared), mode="constant"
        )
        padded_model_weights = pad_rows(selected_model_weights)
        padded_outside_output = pad_rows(selected_outside_output)
        padded_clean_output = pad_rows(selected_clean_output)

        def residual_sum_squares(
            ice_thickness,
            ar_parameters,
            acm_step,
            hgcdte_parameters=None,
            conductive_parameters=None,
            qe_rbf_parameters=None,
        ):
            ar_rows = ar_parameters.reshape((n_rows, -1))
            padded_ar_rows = pad_rows(ar_rows)
            scan_values = [
                padded_ar_rows.reshape((n_chunks, row_chunk_size, -1)),
                padded_counts.reshape((n_chunks, row_chunk_size, -1)),
                padded_sums.reshape((n_chunks, row_chunk_size, -1)),
                padded_sum_squared.reshape((n_chunks, row_chunk_size, -1)),
                padded_model_weights.reshape(
                    (n_chunks, row_chunk_size) + padded_model_weights.shape[1:]
                ),
                padded_outside_output.reshape(
                    (n_chunks, row_chunk_size) + padded_outside_output.shape[1:]
                ),
                padded_clean_output.reshape(
                    (n_chunks, row_chunk_size) + padded_clean_output.shape[1:]
                ),
            ]
            if ice_thickness.ndim == 2:
                padded_ice_thickness = pad_rows(ice_thickness)
                scan_values.insert(
                    0,
                    padded_ice_thickness.reshape(
                        (n_chunks, row_chunk_size, -1)
                    ),
                )
            scan_values = tuple(scan_values)

            def scan_chunk(total_sse, values):
                if ice_thickness.ndim == 2:
                    (
                        ice_chunk,
                        ar_chunk,
                        counts_chunk,
                        sums_chunk,
                        sum_squared_chunk,
                        model_weights_chunk,
                        outside_output_chunk,
                        clean_output_chunk,
                    ) = values
                else:
                    (
                        ar_chunk,
                        counts_chunk,
                        sums_chunk,
                        sum_squared_chunk,
                        model_weights_chunk,
                        outside_output_chunk,
                        clean_output_chunk,
                    ) = values
                    ice_chunk = ice_thickness
                fitted = prediction_for_rows(
                    ice_chunk,
                    ar_chunk,
                    acm_step,
                    model_weights_chunk,
                    outside_output_chunk,
                    clean_output_chunk,
                    hgcdte_parameters,
                    conductive_parameters,
                    qe_rbf_parameters,
                )
                residual_sse = (
                    sum_squared_chunk
                    - 2.0 * fitted * sums_chunk
                    + counts_chunk * fitted * fitted
                )
                return total_sse + jnp.sum(residual_sse), None

            total_sse, _ = jax.lax.scan(
                scan_chunk,
                jnp.asarray(0.0, dtype=padded_counts.dtype),
                scan_values,
            )
            return total_sse

        prediction.residual_sum_squares = residual_sum_squares

    return prediction


def build_model(
    data,
    context,
    log_noise_sigma,
    likelihood_weight,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_prior_reference=None,
    ar_layers=1,
    np_modulation=False,
    fully_np_trajectory=False,
    np_entropy_weight=NP_ENTROPY_WEIGHT,
    np_smoothness_weight=NP_SMOOTHNESS_WEIGHT,
    np_max_node_multiplier=NP_MAX_NODE_MULTIPLIER,
    ar_sca_sigma_prior=AR_SCA_SIGMA_PRIOR,
    ar_tile_sigma_prior=AR_TILE_SIGMA_PRIOR,
    shared_ar_optical_properties=False,
    fit_ar_absorption=False,
    fit_hgcdte=False,
    fit_conductive_layer=False,
    fit_qe_rbf=False,
    qe_rbf_node_wavelengths_nm=None,
    qe_rbf_width_nm=QE_RBF_WIDTH_NM,
    qe_rbf_prior_sigma_log=QE_RBF_PRIOR_SIGMA_LOG,
    use_asw_optical_constants=False,
    led_dependent_acm_step=False,
    trajectory_rate_prior_mean_nm_day=None,
    trajectory_rate_prior_sigma_log=0.75,
    tile_rate_variation=False,
    tile_rate_prior_sigma_log=TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG,
    tile_rate_neighbor_sigma_log=TRAJECTORY_TILE_RATE_NEIGHBOR_SIGMA_LOG,
    fixed_initial_thickness_nm=None,
):
    jax, jnp, numpyro, dist, _, _, _ = require_numpyro()
    qe_rbf_nodes, qe_rbf_width_nm, qe_rbf_prior_sigma_log = qe_rbf_configuration(
        qe_rbf_node_wavelengths_nm,
        qe_rbf_width_nm,
        qe_rbf_prior_sigma_log,
    )
    ar_reference, ar_lower, ar_upper = ar_prior_arrays(ar_layers)
    ar_parameter_count = len(ar_reference)
    active_parameter_indices = np.arange(ar_parameter_count, dtype=int)
    if not fit_ar_absorption:
        active_parameter_indices = active_parameter_indices[
            active_parameter_indices % 3 != 2
        ]
    active_parameter_count = len(active_parameter_indices)
    active_lower = jnp.asarray(ar_lower[active_parameter_indices])
    active_upper = jnp.asarray(ar_upper[active_parameter_indices])
    if np_modulation and fully_np_trajectory:
        raise ValueError(
            "np_modulation and fully_np_trajectory are mutually exclusive"
        )
    if np_modulation and np_max_node_multiplier <= 1.0:
        raise ValueError("np_max_node_multiplier must be greater than one")
    if ar_mean_prior_sigma <= 0.0:
        raise ValueError("AR mean prior sigma must be positive")
    if ar_sca_sigma_prior <= 0.0 or ar_tile_sigma_prior <= 0.0:
        raise ValueError("AR hierarchy sigma priors must be positive")
    if trajectory_rate_prior_mean_nm_day is not None and (
        not np.isfinite(trajectory_rate_prior_mean_nm_day)
        or trajectory_rate_prior_mean_nm_day <= 0.0
    ):
        raise ValueError("Trajectory rate prior mean must be positive and finite")
    if trajectory_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        trajectory_rate_prior_sigma_log
    ):
        raise ValueError("Trajectory rate prior sigma must be positive and finite")
    if tile_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        tile_rate_prior_sigma_log
    ):
        raise ValueError("Tile trajectory rate prior sigma must be positive and finite")
    if tile_rate_neighbor_sigma_log <= 0.0 or not np.isfinite(
        tile_rate_neighbor_sigma_log
    ):
        raise ValueError(
            "Neighboring tile trajectory rate sigma must be positive and finite"
        )
    if fixed_initial_thickness_nm is not None and (
        not np.isfinite(fixed_initial_thickness_nm)
        or fixed_initial_thickness_nm < 0.0
        or fixed_initial_thickness_nm > 50.0
    ):
        raise ValueError("Fixed initial thickness must be between 0 and 50 nm")
    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    days = jnp.asarray(data["days"])
    node_days = jnp.asarray(data["node_days"])
    integration_times = np.unique(
        np.concatenate(
            (
                [np.min(data["days"])],
                np.asarray(data["node_days"], dtype=float),
                [np.max(data["days"])],
            )
        )
    )
    n_scas = int(data["n_scas"])
    n_tiles = int(data["n_tiles"])
    tile_rate_variation_active = bool(tile_rate_variation and (n_scas * n_tiles > 1))
    neighbor_edges = tile_neighbor_edges(data)
    single_tile = is_single_tile_data(data)
    if n_tiles < 1:
        raise ValueError("At least one tile is required for the likelihood")
    # Keep the sampler curvature comparable when a fit includes more tiles.
    effective_likelihood_weight = likelihood_weight / n_tiles
    null_sse = jnp.sum(sum_squared)
    likelihood_scale = (
        0.5 * effective_likelihood_weight * null_sse / (log_noise_sigma**2)
    )
    if ar_prior_reference is None:
        ar_prior_reference = ar_reference
    ar_prior_reference = np.asarray(ar_prior_reference, dtype=float)
    if ar_prior_reference.shape != ar_reference.shape or np.any(
        ar_prior_reference <= ar_lower
    ) or np.any(ar_prior_reference >= ar_upper):
        raise ValueError("AR prior reference must match the parameter shape and bounds")
    active_prior_reference = ar_prior_reference[active_parameter_indices]
    reference_probability = jnp.asarray(
        (active_prior_reference - ar_lower[active_parameter_indices])
        / (
            ar_upper[active_parameter_indices]
            - ar_lower[active_parameter_indices]
        )
    )
    reference_raw = jnp.log(reference_probability / (1.0 - reference_probability))
    prediction_data = dict(data)
    if tile_rate_variation_active:
        prediction_data["prediction_row_chunk_size"] = 8
    prediction = build_prediction_function(
        prediction_data,
        context,
        fit_hgcdte=fit_hgcdte,
        fit_conductive_layer=fit_conductive_layer,
        fit_qe_rbf=fit_qe_rbf,
        qe_rbf_node_wavelengths_nm=qe_rbf_nodes,
        qe_rbf_width_nm=qe_rbf_width_nm,
        use_asw_optical_constants=use_asw_optical_constants,
    )
    residual_sum_squares = prediction.residual_sum_squares
    thickness_indices = np.flatnonzero(active_parameter_indices % 3 == 0)
    hierarchy_parameter_count = (
        ar_layers if shared_ar_optical_properties else active_parameter_count
    )
    trajectory_prior_rate = (
        trajectory_rate_prior_mean_nm_day
        if trajectory_rate_prior_mean_nm_day is not None
        else trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
    )
    trajectory_prior_scale = trajectory_prior_rate / trajectory_rate_per_scale(
        data["days"]
    )

    def physical_ar_parameters(ar_raw):
        active_parameters = active_lower + (
            active_upper - active_lower
        ) * jax.nn.sigmoid(ar_raw)
        parameters = jnp.zeros(
            ar_raw.shape[:-1] + (ar_parameter_count,), dtype=active_parameters.dtype
        )
        return parameters.at[..., active_parameter_indices].set(active_parameters)

    def log_rate_smoothness(log_rates):
        scaled_node_days = (node_days - node_days[0]) / jnp.median(
            jnp.diff(node_days)
        )
        node_slopes = jnp.diff(log_rates) / jnp.diff(scaled_node_days)
        node_curvature = 2.0 * jnp.diff(node_slopes) / (
            scaled_node_days[2:] - scaled_node_days[:-2]
        )
        return jnp.mean(node_curvature * node_curvature)

    def model():
        ar_mu_raw = numpyro.sample(
            "ar_mu_raw",
            dist.Normal(
                reference_raw,
                ar_mean_prior_sigma * jnp.ones(active_parameter_count),
            ).to_event(1),
        )
        if shared_ar_optical_properties:
            if n_scas == 1:
                if single_tile:
                    ar_raw = ar_mu_raw[None, :]
                else:
                    ar_sigma = numpyro.sample(
                        "ar_sigma_raw",
                        dist.HalfNormal(0.5 * jnp.ones(hierarchy_parameter_count)).to_event(1),
                    )
                    ar_z = numpyro.sample(
                        "ar_z",
                        dist.Normal(0.0, 1.0)
                        .expand((n_tiles, hierarchy_parameter_count))
                        .to_event(2),
                    )
                    thickness_raw = (
                        ar_mu_raw[thickness_indices][None, :]
                        + ar_sigma[None, :] * ar_z
                    )
                    ar_raw = jnp.broadcast_to(
                        ar_mu_raw, (n_tiles, active_parameter_count)
                    )
                    ar_raw = ar_raw.at[:, thickness_indices].set(thickness_raw)
                ar_parameters = physical_ar_parameters(ar_raw)
                ar_parameters_for_prediction = ar_parameters
            else:
                ar_sca_sigma = numpyro.sample(
                    "ar_sca_sigma",
                    dist.HalfNormal(
                        ar_sca_sigma_prior * jnp.ones(hierarchy_parameter_count)
                    ).to_event(1),
                )
                ar_sca_z = numpyro.sample(
                    "ar_sca_z",
                    dist.Normal(0.0, 1.0)
                    .expand((n_scas, hierarchy_parameter_count))
                    .to_event(2),
                )
                sca_thickness_raw = (
                    ar_mu_raw[thickness_indices][None, :]
                    + ar_sca_sigma[None, :] * ar_sca_z
                )
                ar_sca_raw = jnp.broadcast_to(
                    ar_mu_raw[None, :], (n_scas, active_parameter_count)
                )
                ar_sca_raw = ar_sca_raw.at[:, thickness_indices].set(
                    sca_thickness_raw
                )
                ar_tile_sigma = numpyro.sample(
                    "ar_tile_sigma",
                    dist.HalfNormal(
                        ar_tile_sigma_prior * jnp.ones(hierarchy_parameter_count)
                    ).to_event(1),
                )
                ar_tile_z = numpyro.sample(
                    "ar_tile_z",
                    dist.Normal(0.0, 1.0)
                    .expand((n_scas, n_tiles, hierarchy_parameter_count))
                    .to_event(3),
                )
                tile_thickness_raw = (
                    sca_thickness_raw[:, None, :]
                    + ar_tile_sigma[None, None, :] * ar_tile_z
                )
                ar_raw = jnp.broadcast_to(
                    ar_sca_raw[:, None, :],
                    (n_scas, n_tiles, active_parameter_count),
                )
                ar_raw = ar_raw.at[:, :, thickness_indices].set(
                    tile_thickness_raw
                )
                ar_parameters = physical_ar_parameters(ar_raw)
                ar_parameters_for_prediction = ar_parameters.reshape(
                    (-1, ar_parameter_count)
                )
        elif n_scas == 1:
            if single_tile:
                ar_raw = ar_mu_raw[None, :]
            else:
                ar_sigma = numpyro.sample(
                    "ar_sigma_raw",
                    dist.HalfNormal(
                        0.5 * jnp.ones(active_parameter_count)
                    ).to_event(1),
                )
                ar_z = numpyro.sample(
                    "ar_z",
                    dist.Normal(0.0, 1.0)
                    .expand((n_tiles, active_parameter_count))
                    .to_event(2),
                )
                ar_raw = ar_mu_raw[None, :] + ar_sigma[None, :] * ar_z
            ar_parameters = physical_ar_parameters(ar_raw)
            ar_parameters_for_prediction = ar_parameters
        else:
            ar_sca_sigma = numpyro.sample(
                "ar_sca_sigma",
                dist.HalfNormal(
                    ar_sca_sigma_prior * jnp.ones(active_parameter_count)
                ).to_event(1),
            )
            ar_sca_z = numpyro.sample(
                "ar_sca_z",
                dist.Normal(0.0, 1.0)
                .expand((n_scas, active_parameter_count))
                .to_event(2),
            )
            ar_sca_raw = ar_mu_raw[None, :] + ar_sca_sigma[None, :] * ar_sca_z
            ar_tile_sigma = numpyro.sample(
                "ar_tile_sigma",
                dist.HalfNormal(
                    ar_tile_sigma_prior * jnp.ones(active_parameter_count)
                ).to_event(1),
            )
            ar_tile_z = numpyro.sample(
                "ar_tile_z",
                dist.Normal(0.0, 1.0)
                .expand((n_scas, n_tiles, active_parameter_count))
                .to_event(3),
            )
            ar_raw = ar_sca_raw[:, None, :] + ar_tile_sigma[None, None, :] * ar_tile_z
            ar_parameters = physical_ar_parameters(ar_raw)
            ar_parameters_for_prediction = ar_parameters.reshape(
                (-1, ar_parameter_count)
            )
        if fixed_initial_thickness_nm is None:
            d0 = numpyro.sample(
                "initial_thickness_nm",
                dist.TruncatedNormal(35.0, 15.0, low=0.0, high=50.0),
            )
        else:
            d0 = numpyro.deterministic(
                "initial_thickness_nm",
                jnp.asarray(fixed_initial_thickness_nm),
            )
        if fully_np_trajectory:
            log_scale = jnp.asarray(0.0)
            np_log_rate_nodes = numpyro.sample(
                "np_log_rate_nodes",
                dist.Normal(
                    jnp.log(trajectory_prior_rate), trajectory_rate_prior_sigma_log
                )
                .expand((len(data["node_days"]),))
                .to_event(1),
            )
        else:
            log_scale = numpyro.sample(
                "log_trajectory_scale",
                dist.Normal(
                    jnp.log(trajectory_prior_scale), trajectory_rate_prior_sigma_log
                ),
            )
            np_log_rate_nodes = None
        if tile_rate_variation_active:
            tile_log_rate_offsets = numpyro.sample(
                "trajectory_rate_tile_offsets",
                dist.Normal(0.0, tile_rate_prior_sigma_log)
                .expand((n_scas, n_tiles))
                .to_event(2),
            )
            if neighbor_edges.size:
                edge_indices = jnp.asarray(neighbor_edges, dtype=jnp.int32)
                neighbor_difference = (
                    tile_log_rate_offsets[:, edge_indices[:, 0]]
                    - tile_log_rate_offsets[:, edge_indices[:, 1]]
                )
                numpyro.factor(
                    "trajectory_rate_neighbor_smoothness",
                    -0.5
                    * jnp.sum(
                        (neighbor_difference / tile_rate_neighbor_sigma_log) ** 2
                    ),
                )
        else:
            tile_log_rate_offsets = jnp.zeros((n_scas, n_tiles))
        if fit_hgcdte:
            hgcdte_n = numpyro.sample(
                "hgcdte_n",
                dist.TruncatedNormal(
                    NOMINAL_HGCDTE[0],
                    HGCDTE_PRIOR_SIGMA[0],
                    low=HGCDTE_LOWER[0],
                    high=HGCDTE_UPPER[0],
                ),
            )
            hgcdte_k = numpyro.sample(
                "hgcdte_k",
                dist.TruncatedNormal(
                    NOMINAL_HGCDTE[1],
                    HGCDTE_PRIOR_SIGMA[1],
                    low=HGCDTE_LOWER[1],
                    high=HGCDTE_UPPER[1],
                ),
            )
            hgcdte_parameters = jnp.stack((hgcdte_n, hgcdte_k))
        else:
            hgcdte_parameters = None
        if fit_conductive_layer:
            conductive_log_thickness_nm = numpyro.sample(
                "conductive_log_thickness_nm",
                dist.Normal(
                    CONDUCTIVE_LOG_THICKNESS_PRIOR_MEAN,
                    CONDUCTIVE_LOG_PRIOR_SIGMA,
                ),
            )
            conductive_log_plasma_strength = numpyro.sample(
                "conductive_log_plasma_strength",
                dist.Normal(
                    CONDUCTIVE_LOG_PLASMA_STRENGTH_PRIOR_MEAN,
                    CONDUCTIVE_LOG_PRIOR_SIGMA,
                ),
            )
            conductive_log_damping_ratio = numpyro.sample(
                "conductive_log_damping_ratio",
                dist.Normal(
                    CONDUCTIVE_LOG_DAMPING_RATIO_PRIOR_MEAN,
                    CONDUCTIVE_LOG_PRIOR_SIGMA,
                ),
            )
            conductive_parameters = jnp.stack(
                (
                    conductive_log_thickness_nm,
                    conductive_log_plasma_strength,
                    conductive_log_damping_ratio,
                )
            )
            numpyro.deterministic(
                "conductive_thickness_nm",
                jnp.exp(conductive_log_thickness_nm),
            )
            numpyro.deterministic(
                "conductive_plasma_strength",
                jnp.exp(conductive_log_plasma_strength),
            )
            numpyro.deterministic(
                "conductive_damping_ratio",
                jnp.exp(conductive_log_damping_ratio),
            )
        else:
            conductive_parameters = None
        if fit_qe_rbf:
            qe_rbf_log_modulation = numpyro.sample(
                "qe_rbf_log_modulation",
                dist.Normal(0.0, qe_rbf_prior_sigma_log)
                .expand((len(qe_rbf_nodes),))
                .to_event(1),
            )
            qe_rbf_parameters = qe_rbf_log_modulation
            numpyro.deterministic(
                "qe_rbf_modulation",
                jnp.exp(qe_rbf_log_modulation),
            )
        else:
            qe_rbf_parameters = None
        if led_dependent_acm_step:
            acm_step = numpyro.sample(
                "acm_step_log_offset",
                dist.Normal(0.0, 0.05).expand((6,)).to_event(1),
            )
        else:
            acm_step = numpyro.sample(
                "acm_step_log_offset", dist.Normal(0.0, 0.05)
            )
        if fully_np_trajectory:
            log_node_multipliers = np_log_rate_nodes - jnp.log(
                trajectory_prior_rate
            )
            node_multipliers = jnp.exp(log_node_multipliers)
            np_smoothness = log_rate_smoothness(np_log_rate_nodes)
            numpyro.factor(
                "np_smoothness_regularization",
                -likelihood_scale * np_smoothness_weight * np_smoothness,
            )
        elif np_modulation:
            np_node_raw = numpyro.sample(
                "np_node_raw",
                dist.Normal(0.0, 1.0)
                .expand((len(data["repeat_values"]) - 1,))
                .to_event(1),
            )
            log_node_limit = jnp.log(np_max_node_multiplier)
            log_node_multipliers = log_node_limit * jnp.tanh(np_node_raw)
            log_node_multipliers = jnp.concatenate(
                (jnp.zeros(1), log_node_multipliers)
            )
            node_multipliers = jnp.exp(log_node_multipliers)
            np_entropy = jnp.mean(
                node_multipliers * log_node_multipliers
                - node_multipliers
                + 1.0
            )
            np_smoothness = log_rate_smoothness(log_node_multipliers)
            numpyro.factor(
                "np_entropy_regularization",
                -likelihood_scale * np_entropy_weight * np_entropy,
            )
            numpyro.factor(
                "np_smoothness_regularization",
                -likelihood_scale * np_smoothness_weight * np_smoothness,
            )
        else:
            log_node_multipliers = jnp.zeros(len(data["repeat_values"]))
            node_multipliers = jnp.ones(len(data["repeat_values"]))
        (
            integrated_thickness,
            base_instantaneous_rate,
            node_integrated_thickness,
            base_node_instantaneous_rate,
        ) = (
            integrate_rate_trajectory(
                jax,
                jnp,
                days,
                node_days,
                integration_times,
                0.0,
                np_log_rate_nodes
                if fully_np_trajectory
                else log_node_multipliers,
                use_square_root=not fully_np_trajectory,
            )
        )
        row_log_rate_scales = log_scale + tile_log_rate_offsets.reshape(-1)
        row_rate_scales = jnp.exp(row_log_rate_scales)
        tile_ice_thickness = d0 + row_rate_scales[:, None] * integrated_thickness[None, :]
        tile_instantaneous_rate = (
            row_rate_scales[:, None] * base_instantaneous_rate[None, :]
        )
        tile_node_ice_thickness = (
            d0 + row_rate_scales[:, None] * node_integrated_thickness[None, :]
        )
        tile_node_instantaneous_rate = (
            row_rate_scales[:, None] * base_node_instantaneous_rate[None, :]
        )
        if tile_rate_variation_active:
            ice_thickness = tile_ice_thickness
            instantaneous_rate = jnp.mean(tile_instantaneous_rate, axis=0)
            node_ice_thickness = jnp.mean(tile_node_ice_thickness, axis=0)
            node_instantaneous_rate = jnp.mean(tile_node_instantaneous_rate, axis=0)
        else:
            ice_thickness = tile_ice_thickness[0]
            instantaneous_rate = tile_instantaneous_rate[0]
            node_ice_thickness = tile_node_ice_thickness[0]
            node_instantaneous_rate = tile_node_instantaneous_rate[0]
        total_sse = residual_sum_squares(
            ice_thickness,
            ar_parameters_for_prediction,
            acm_step,
            hgcdte_parameters,
            conductive_parameters,
            qe_rbf_parameters,
        )
        numpyro.factor(
            "tile_gaussian_log_likelihood",
            -0.5
            * effective_likelihood_weight
            * total_sse
            / (log_noise_sigma**2),
        )
        numpyro.deterministic("ar_parameters", ar_parameters)
        numpyro.deterministic(
            "ar_mean",
            physical_ar_parameters(ar_mu_raw),
        )
        if n_scas > 1:
            numpyro.deterministic(
                "ar_sca_mean",
                physical_ar_parameters(ar_sca_raw),
            )
        numpyro.deterministic("ice_thickness", ice_thickness)
        numpyro.deterministic("instantaneous_rate_nm_day", instantaneous_rate)
        numpyro.deterministic(
            "node_instantaneous_rate_nm_day", node_instantaneous_rate
        )
        first_group = int(np.argmin(data["days"]))
        last_group = int(np.argmax(data["days"]))
        if tile_rate_variation_active:
            numpyro.deterministic(
                "tile_ice_thickness", tile_ice_thickness
            )
            numpyro.deterministic(
                "tile_instantaneous_rate_nm_day", tile_instantaneous_rate
            )
            numpyro.deterministic(
                "tile_node_thickness", tile_node_ice_thickness
            )
            numpyro.deterministic(
                "tile_node_instantaneous_rate_nm_day",
                tile_node_instantaneous_rate,
            )
            numpyro.deterministic(
                "tile_initial_rate_nm_day", tile_instantaneous_rate[:, first_group]
            )
            numpyro.deterministic(
                "tile_final_rate_nm_day", tile_instantaneous_rate[:, last_group]
            )
            numpyro.deterministic(
                "tile_mean_rate_nm_day",
                (tile_ice_thickness[:, last_group] - tile_ice_thickness[:, first_group])
                / (jnp.max(days) - jnp.min(days)),
            )
        numpyro.deterministic(
            "initial_rate_nm_day", instantaneous_rate[first_group]
        )
        numpyro.deterministic("final_rate_nm_day", instantaneous_rate[last_group])
        if np_modulation or fully_np_trajectory:
            numpyro.deterministic("np_log_node_multipliers", log_node_multipliers)
            numpyro.deterministic("np_node_multipliers", node_multipliers)
            numpyro.deterministic(
                "np_node_thickness",
                node_ice_thickness,
            )
        if tile_rate_variation_active:
            rate_numerator = jnp.mean(
                tile_ice_thickness[:, last_group]
                - tile_ice_thickness[:, first_group]
            )
        else:
            rate_numerator = ice_thickness[last_group] - ice_thickness[first_group]
        numpyro.deterministic(
            "mean_rate_nm_day",
            rate_numerator / (jnp.max(days) - jnp.min(days)),
        )
        if np_modulation:
            numpyro.deterministic("np_entropy", np_entropy)
        if np_modulation or fully_np_trajectory:
            numpyro.deterministic("np_smoothness", np_smoothness)

    return model


def map_start_reference(args, ar_reference, ar_lower, ar_upper):
    map_design_wavelength_nm = getattr(args, "map_design_wavelength_nm", None)
    if map_design_wavelength_nm is None:
        return np.asarray(ar_reference).copy()
    start_reference = dielectric_ar_reference(
        args.ar_layers,
        design_wavelength_nm=map_design_wavelength_nm,
        middle_layer_eighth_wave=getattr(
            args, "map_middle_layer_eighth_wave", False
        ),
        middle_layer_half_wave=getattr(args, "map_middle_layer_half_wave", False),
    )
    if np.any(start_reference <= ar_lower) or np.any(start_reference >= ar_upper):
        raise ValueError("MAP start falls outside the AR parameter bounds")
    return start_reference


def initial_values_for_data(data, args, ar_reference, ar_lower, ar_upper):
    """Build compatible sample-site starts for a selected tile subset."""
    start_reference = map_start_reference(args, ar_reference, ar_lower, ar_upper)
    active_parameter_indices = np.arange(len(ar_reference), dtype=int)
    if not getattr(args, "fit_ar_absorption", False):
        active_parameter_indices = active_parameter_indices[
            active_parameter_indices % 3 != 2
        ]
    reference_raw = np.log(
        (start_reference[active_parameter_indices] - ar_lower[active_parameter_indices])
        / (ar_upper[active_parameter_indices] - start_reference[active_parameter_indices])
    )
    initial_values = {"ar_mu_raw": reference_raw}
    if getattr(args, "fixed_initial_thickness_nm", None) is None:
        initial_values["initial_thickness_nm"] = INITIAL_THICKNESS_INIT_NM
    if getattr(args, "fully_np_trajectory", False):
        trajectory_prior_rate = (
            getattr(args, "trajectory_rate_prior_mean_nm_day", None)
            if getattr(args, "trajectory_rate_prior_mean_nm_day", None) is not None
            else trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
        )
        initial_values["np_log_rate_nodes"] = np.full(
            len(data["node_days"]), np.log(trajectory_prior_rate)
        )
    else:
        initial_values["log_trajectory_scale"] = np.log(TRAJECTORY_SCALE_START)
    if getattr(args, "fit_hgcdte", False):
        initial_values.update(
            {
                "hgcdte_n": float(NOMINAL_HGCDTE[0]),
                "hgcdte_k": float(NOMINAL_HGCDTE[1]),
            }
        )
    if getattr(args, "fit_conductive_layer", False):
        initial_values.update(
            {
                "conductive_log_thickness_nm": CONDUCTIVE_LOG_THICKNESS_PRIOR_MEAN,
                "conductive_log_plasma_strength": CONDUCTIVE_LOG_PLASMA_STRENGTH_PRIOR_MEAN,
                "conductive_log_damping_ratio": CONDUCTIVE_LOG_DAMPING_RATIO_PRIOR_MEAN,
            }
        )
    if getattr(args, "fit_qe_rbf", False):
        qe_rbf_nodes, _, _ = qe_rbf_configuration(
            getattr(args, "qe_rbf_node_wavelengths_nm", None),
            getattr(args, "qe_rbf_width_nm", QE_RBF_WIDTH_NM),
            getattr(args, "qe_rbf_prior_sigma_log", QE_RBF_PRIOR_SIGMA_LOG),
        )
        initial_values["qe_rbf_log_modulation"] = np.zeros(
            len(qe_rbf_nodes)
        )
    initial_values["acm_step_log_offset"] = (
        np.zeros(6) if getattr(args, "led_dependent_acm_step", False) else 0.0
    )
    map_trajectory_rate_nm_day = getattr(args, "map_trajectory_rate_nm_day", None)
    if map_trajectory_rate_nm_day is not None:
        if getattr(args, "fully_np_trajectory", False):
            initial_values["np_log_rate_nodes"] = np.full(
                len(data["node_days"]), np.log(map_trajectory_rate_nm_day)
            )
        else:
            initial_values["log_trajectory_scale"] = np.log(
                map_trajectory_rate_nm_day / trajectory_rate_per_scale(data["days"])
            )
    hierarchy_parameter_count = (
        args.ar_layers
        if args.shared_ar_optical_properties
        else len(active_parameter_indices)
    )
    if is_single_tile_data(data):
        pass
    elif data["n_scas"] == 1:
        initial_values.update(
            {
                "ar_sigma_raw": np.full(hierarchy_parameter_count, 0.2),
                "ar_z": np.zeros((len(data["tile_ids"]), hierarchy_parameter_count)),
            }
        )
    else:
        initial_values.update(
            {
                "ar_sca_sigma": np.full(
                    hierarchy_parameter_count, args.ar_sca_sigma_prior * 0.5
                ),
                "ar_sca_z": np.zeros((data["n_scas"], hierarchy_parameter_count)),
                "ar_tile_sigma": np.full(
                    hierarchy_parameter_count, args.ar_tile_sigma_prior * 0.5
                ),
                "ar_tile_z": np.zeros(
                    (
                        data["n_scas"],
                        data["n_tiles"],
                        hierarchy_parameter_count,
                    )
                ),
            }
        )
    if args.np_modulation:
        initial_values["np_node_raw"] = np.zeros(len(data["repeat_values"]) - 1)
    if getattr(args, "tile_rate_variation", False) and (
        data["n_scas"] * data["n_tiles"] > 1
    ):
        initial_values["trajectory_rate_tile_offsets"] = np.zeros(
            (data["n_scas"], data["n_tiles"])
        )
    return initial_values


def transfer_map_initial_values(map_values, full_initial_values):
    """Transfer reduced-fit sites whose shapes match the full fit."""
    transferred = dict(full_initial_values)
    for name, value in map_values.items():
        if name not in transferred:
            continue
        value = np.asarray(value)
        if value.shape == np.asarray(transferred[name]).shape:
            transferred[name] = value
    return transferred


def quantile(values):
    return np.percentile(np.asarray(values), [16, 50, 84])


def posterior_median_fit_diagnostics(
    data,
    context,
    samples,
    fit_hgcdte=False,
    fit_conductive_layer=False,
    fit_qe_rbf=False,
    qe_rbf_node_wavelengths_nm=None,
    qe_rbf_width_nm=QE_RBF_WIDTH_NM,
    use_asw_optical_constants=False,
):
    """Evaluate the posterior-median parameters against the tile means."""
    _, jnp, _, _, _, _, _ = require_numpyro()
    ice_thickness = np.median(np.asarray(samples["ice_thickness"]), axis=0)
    ar_parameters = np.median(np.asarray(samples["ar_parameters"]), axis=0)
    if ar_parameters.ndim == 3:
        ar_parameters = ar_parameters.reshape((-1, ar_parameters.shape[-1]))
    elif ar_parameters.ndim != 2:
        raise ValueError("Unexpected median AR parameter shape")
    n_rows = int(data["counts"].shape[0])
    if ar_parameters.shape[0] != n_rows:
        raise ValueError("Posterior AR rows do not match the selected tile data")
    acm_step = np.median(
        np.asarray(samples["acm_step_log_offset"]), axis=0
    )
    hgcdte_parameters = None
    if fit_hgcdte:
        hgcdte_parameters = jnp.asarray(
            [
                np.median(np.asarray(samples["hgcdte_n"])),
                np.median(np.asarray(samples["hgcdte_k"])),
            ]
        )
    conductive_parameters = None
    if fit_conductive_layer:
        conductive_parameters = jnp.asarray(
            [
                np.median(np.asarray(samples["conductive_log_thickness_nm"])),
                np.median(np.asarray(samples["conductive_log_plasma_strength"])),
                np.median(np.asarray(samples["conductive_log_damping_ratio"])),
            ]
        )
    qe_rbf_parameters = None
    if fit_qe_rbf:
        qe_rbf_parameters = jnp.asarray(
            np.median(np.asarray(samples["qe_rbf_log_modulation"]), axis=0)
        )
    prediction_chunks = []
    for start in range(0, n_rows, PREDICTION_ROW_CHUNK_SIZE):
        stop = min(start + PREDICTION_ROW_CHUNK_SIZE, n_rows)
        row_indices = np.arange(start, stop, dtype=int)
        prediction_chunks.append(
            np.asarray(
                build_prediction_function(
                    data,
                    context,
                    row_indices=row_indices,
                    fit_hgcdte=fit_hgcdte,
                    fit_conductive_layer=fit_conductive_layer,
                    fit_qe_rbf=fit_qe_rbf,
                    qe_rbf_node_wavelengths_nm=qe_rbf_node_wavelengths_nm,
                    qe_rbf_width_nm=qe_rbf_width_nm,
                    use_asw_optical_constants=use_asw_optical_constants,
                )(
                    jnp.asarray(
                        ice_thickness
                        if ice_thickness.ndim == 1
                        else ice_thickness[row_indices]
                    ),
                    jnp.asarray(ar_parameters[row_indices]),
                    jnp.asarray(acm_step),
                    hgcdte_parameters,
                    conductive_parameters,
                    qe_rbf_parameters,
                )
            )
        )
    prediction = np.concatenate(prediction_chunks, axis=0)
    observed = data["sums"] / data["counts"]
    residual = observed - prediction
    non_reference = np.ones(residual.shape[1], dtype=bool)
    non_reference[data["first_by_led"]] = False
    total_count = float(np.sum(data["counts"]))
    total_sse = float(
        np.sum(
            data["sum_squared"]
            - 2.0 * prediction * data["sums"]
            + data["counts"] * prediction * prediction
        )
    )
    return {
        "group_rms_log": float(np.sqrt(np.mean(residual * residual))),
        "non_reference_group_rms_log": float(
            np.sqrt(np.mean(residual[:, non_reference] ** 2))
        ),
        "count_weighted_group_rms_log": float(
            np.sqrt(np.sum(data["counts"] * residual * residual) / total_count)
        ),
        "pixel_rms_log": float(np.sqrt(total_sse / total_count)),
        "correlation": float(np.corrcoef(observed.ravel(), prediction.ravel())[0, 1]),
        "led_rms_log": np.asarray(
            [
                np.sqrt(np.mean(residual[:, data["led"] == led] ** 2))
                for led in range(1, 7)
            ]
        ),
    }


def ar_quantile_fields(ar_layers):
    fields = []
    for layer in range(ar_layers):
        prefix = "ar_" if ar_layers == 1 else f"ar_layer{layer + 1}_"
        for component in ("thickness", "n", "k"):
            fields.extend(
                f"{prefix}{component}_{percentile}"
                for percentile in ("p16", "median", "p84")
            )
    return fields


def add_tile_ar_quantiles(row, quantiles, sca_index, tile_index, ar_layers):
    for layer in range(ar_layers):
        prefix = "ar_" if ar_layers == 1 else f"ar_layer{layer + 1}_"
        for component_index, component in enumerate(("thickness", "n", "k")):
            for percentile_index, percentile in enumerate(("p16", "median", "p84")):
                row[f"{prefix}{component}_{percentile}"] = quantiles[
                    percentile_index, sca_index, tile_index, layer, component_index
                ]


def add_sca_ar_quantiles(row, quantiles, sca_index, ar_layers):
    for layer in range(ar_layers):
        prefix = "ar_" if ar_layers == 1 else f"ar_layer{layer + 1}_"
        for component_index, component in enumerate(("thickness", "n", "k")):
            for percentile_index, percentile in enumerate(("p16", "median", "p84")):
                row[f"{prefix}{component}_{percentile}"] = quantiles[
                    percentile_index, sca_index, layer, component_index
                ]


def write_outputs(
    output_dir,
    data,
    samples,
    args,
    fit_diagnostics=None,
    sampler_diagnostics=None,
    map_diagnostics=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ar_layers = int(args.ar_layers)
    qe_rbf_nodes, qe_rbf_width_nm, qe_rbf_prior_sigma_log = qe_rbf_configuration(
        getattr(args, "qe_rbf_node_wavelengths_nm", None),
        getattr(args, "qe_rbf_width_nm", QE_RBF_WIDTH_NM),
        getattr(args, "qe_rbf_prior_sigma_log", QE_RBF_PRIOR_SIGMA_LOG),
    )
    metadata = {
        "tile_ids": data["tile_ids"],
        "sca_names": data["sca_names"],
        "qe_columns": data["qe_columns"],
        "ar_layers": ar_layers,
        "wavelength_step_nm": (
            np.nan if args.wavelength_step_nm is None else args.wavelength_step_nm
        ),
        "wavelength_stride": args.wavelength_stride,
        "likelihood_weight": args.likelihood_weight,
        "log_noise_sigma": args.log_noise_sigma,
        "statistics_path": np.asarray(str(args.tile_statistics)),
        "statistics_fingerprint": np.asarray(statistics_fingerprint(data)),
        "statistics_row_fingerprints": statistics_row_fingerprints(data),
        "sed_mode": np.asarray(args.sed_mode),
        "max_tiles": -1 if args.max_tiles is None else args.max_tiles,
        "shared_ar_optical_properties": int(args.shared_ar_optical_properties),
        "fit_ar_absorption": int(args.fit_ar_absorption),
        "fit_hgcdte": int(getattr(args, "fit_hgcdte", False)),
        "fit_conductive_layer": int(getattr(args, "fit_conductive_layer", False)),
        "fit_qe_rbf": int(getattr(args, "fit_qe_rbf", False)),
        "use_asw_optical_constants": int(
            getattr(args, "use_asw_optical_constants", False)
        ),
        "led_dependent_acm_step": int(
            getattr(args, "led_dependent_acm_step", False)
        ),
        "fully_np_trajectory": int(
            getattr(args, "fully_np_trajectory", False)
        ),
        "qe_rbf_node_wavelengths_nm": qe_rbf_nodes,
        "qe_rbf_width_nm": qe_rbf_width_nm,
        "qe_rbf_prior_sigma_log": qe_rbf_prior_sigma_log,
        "trajectory_rate_prior_mean_nm_day": (
            trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
            if args.trajectory_rate_prior_mean_nm_day is None
            else args.trajectory_rate_prior_mean_nm_day
        ),
        "trajectory_rate_prior_sigma_log": args.trajectory_rate_prior_sigma_log,
        "trajectory_rate_definition": np.asarray(
            (
                "fully non-parametric instantaneous rate at each repeat node"
                if getattr(args, "fully_np_trajectory", False)
                else "unmodulated instantaneous rate at the first observed time"
            )
        ),
        "trajectory_rate_reference_day": float(np.min(data["days"])),
        "tile_rate_variation": int(getattr(args, "tile_rate_variation", False)),
        "tile_rate_prior_sigma_log": getattr(
            args, "tile_rate_prior_sigma_log", np.nan
        ),
        "tile_rate_neighbor_sigma_log": getattr(
            args, "tile_rate_neighbor_sigma_log", np.nan
        ),
        "tile_rate_neighbor_count": len(tile_neighbor_edges(data)),
        "map_trajectory_rate_nm_day": (
            np.nan
            if args.map_trajectory_rate_nm_day is None
            else args.map_trajectory_rate_nm_day
        ),
        "map_only": int(getattr(args, "map_only", False)),
        "observation_days": data["days"],
        "observation_led": data["led"],
        "observation_repeat": data["repeat"],
        "observation_exposure_times": np.asarray(data["exposure_times"]),
        "observation_step_design": data["step_design"],
    }
    if map_diagnostics:
        metadata.update(
            {
                "map_optimizer": np.asarray(map_diagnostics["optimizer"]),
                "map_converged": int(map_diagnostics["converged"]),
                "map_status": int(map_diagnostics["status"]),
                "map_message": np.asarray(map_diagnostics["message"]),
                "map_iterations": int(map_diagnostics["iterations"]),
                "map_function_evaluations": int(
                    map_diagnostics["function_evaluations"]
                ),
                "map_gradient_converged": int(
                    map_diagnostics.get("gradient_converged", False)
                ),
                "map_gradient_norm": float(map_diagnostics["gradient_norm"]),
                "map_objective": float(map_diagnostics["objective"]),
                "map_initial_objective": float(
                    map_diagnostics["initial_objective"]
                ),
                "map_parameter_step_norm": float(
                    map_diagnostics["parameter_step_norm"]
                ),
                "map_objective_history": np.asarray(
                    map_diagnostics["objective_history"], dtype=float
                ),
            }
        )
    if sampler_diagnostics:
        metadata.update(
            {
                f"sampler_{key}": value
                for key, value in sampler_diagnostics.items()
            }
        )
    np.savez_compressed(
        output_dir / "posterior_samples.npz",
        **{key: np.asarray(value) for key, value in samples.items()},
        **metadata,
    )
    ar_samples = np.asarray(samples["ar_parameters"])
    if ar_samples.ndim == 3:
        ar_grid_samples = ar_samples[:, None, :, :]
    elif ar_samples.ndim == 4:
        ar_grid_samples = ar_samples
    else:
        raise ValueError("Unexpected AR posterior shape")
    if ar_grid_samples.shape[-1] != 3 * ar_layers:
        raise ValueError("AR posterior shape does not match the selected layer count")
    ar_grid_samples = ar_grid_samples.reshape(
        ar_grid_samples.shape[:-1] + (ar_layers, 3)
    )
    ar_quantiles = np.percentile(ar_grid_samples, [16, 50, 84], axis=0)
    with (output_dir / "tile_ar_posterior.csv").open("w", newline="") as handle:
        fields = [
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
        ] + ar_quantile_fields(ar_layers)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sca_index, detector in enumerate(data["sca_names"]):
            for tile_index, tile_id in enumerate(data["tile_ids"]):
                metadata = data["metadata"][int(tile_id)]
                row = {
                    "detector": detector,
                    "tile_id": int(tile_id),
                    "tile_row": metadata["tile_row"],
                    "tile_col": metadata["tile_col"],
                }
                add_tile_ar_quantiles(
                    row, ar_quantiles, sca_index, tile_index, ar_layers
                )
                writer.writerow(row)

    if "tile_initial_rate_nm_day" in samples:
        tile_initial = np.percentile(
            np.asarray(samples["tile_initial_rate_nm_day"]), [16, 50, 84], axis=0
        )
        tile_final = np.percentile(
            np.asarray(samples["tile_final_rate_nm_day"]), [16, 50, 84], axis=0
        )
        tile_mean = np.percentile(
            np.asarray(samples["tile_mean_rate_nm_day"]), [16, 50, 84], axis=0
        )
        tile_offsets = np.percentile(
            np.asarray(samples["trajectory_rate_tile_offsets"]),
            [16, 50, 84],
            axis=0,
        )
        with (output_dir / "tile_rate_posterior.csv").open("w", newline="") as handle:
            fields = [
                "detector",
                "tile_id",
                "tile_row",
                "tile_col",
                "log_rate_offset_p16",
                "log_rate_offset_median",
                "log_rate_offset_p84",
                "initial_rate_nm_day_p16",
                "initial_rate_nm_day_median",
                "initial_rate_nm_day_p84",
                "final_rate_nm_day_p16",
                "final_rate_nm_day_median",
                "final_rate_nm_day_p84",
                "mean_rate_nm_day_p16",
                "mean_rate_nm_day_median",
                "mean_rate_nm_day_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sca_index, detector in enumerate(data["sca_names"]):
                for tile_index, tile_id in enumerate(data["tile_ids"]):
                    metadata = data["metadata"][int(tile_id)]
                    writer.writerow(
                        {
                            "detector": detector,
                            "tile_id": int(tile_id),
                            "tile_row": metadata["tile_row"],
                            "tile_col": metadata["tile_col"],
                            "log_rate_offset_p16": tile_offsets[0, sca_index, tile_index],
                            "log_rate_offset_median": tile_offsets[1, sca_index, tile_index],
                            "log_rate_offset_p84": tile_offsets[2, sca_index, tile_index],
                            "initial_rate_nm_day_p16": tile_initial[0, sca_index * len(data["tile_ids"]) + tile_index],
                            "initial_rate_nm_day_median": tile_initial[1, sca_index * len(data["tile_ids"]) + tile_index],
                            "initial_rate_nm_day_p84": tile_initial[2, sca_index * len(data["tile_ids"]) + tile_index],
                            "final_rate_nm_day_p16": tile_final[0, sca_index * len(data["tile_ids"]) + tile_index],
                            "final_rate_nm_day_median": tile_final[1, sca_index * len(data["tile_ids"]) + tile_index],
                            "final_rate_nm_day_p84": tile_final[2, sca_index * len(data["tile_ids"]) + tile_index],
                            "mean_rate_nm_day_p16": tile_mean[0, sca_index * len(data["tile_ids"]) + tile_index],
                            "mean_rate_nm_day_median": tile_mean[1, sca_index * len(data["tile_ids"]) + tile_index],
                            "mean_rate_nm_day_p84": tile_mean[2, sca_index * len(data["tile_ids"]) + tile_index],
                        }
                    )

    if "ar_sca_mean" in samples:
        sca_quantiles = np.percentile(
            np.asarray(samples["ar_sca_mean"]), [16, 50, 84], axis=0
        )
        if sca_quantiles.shape[-1] != 3 * ar_layers:
            raise ValueError("SCA AR posterior shape does not match the selected layer count")
        sca_quantiles = sca_quantiles.reshape(
            sca_quantiles.shape[:-1] + (ar_layers, 3)
        )
        with (output_dir / "sca_ar_posterior.csv").open("w", newline="") as handle:
            fields = ["detector"] + ar_quantile_fields(ar_layers)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, detector in enumerate(data["sca_names"]):
                row = {"detector": detector}
                add_sca_ar_quantiles(row, sca_quantiles, index, ar_layers)
                writer.writerow(row)

    if args.np_modulation or getattr(args, "fully_np_trajectory", False):
        node_samples = np.asarray(samples["np_node_multipliers"])
        node_quantiles = np.percentile(node_samples, [16, 50, 84], axis=0)
        with (output_dir / "np_node_posterior.csv").open("w", newline="") as handle:
            node_rate_samples = np.asarray(
                samples["node_instantaneous_rate_nm_day"]
            )
            node_rate_quantiles = np.percentile(node_rate_samples, [16, 50, 84], axis=0)
            fields = [
                "repeat",
                "days",
                "node_multiplier_p16",
                "node_multiplier_median",
                "node_multiplier_p84",
                "instantaneous_rate_nm_day_p16",
                "instantaneous_rate_nm_day_median",
                "instantaneous_rate_nm_day_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, repeat in enumerate(data["repeat_values"]):
                writer.writerow(
                    {
                        "repeat": int(repeat),
                        "days": data["node_days"][index],
                        "node_multiplier_p16": node_quantiles[0, index],
                        "node_multiplier_median": node_quantiles[1, index],
                        "node_multiplier_p84": node_quantiles[2, index],
                        "instantaneous_rate_nm_day_p16": node_rate_quantiles[0, index],
                        "instantaneous_rate_nm_day_median": node_rate_quantiles[1, index],
                        "instantaneous_rate_nm_day_p84": node_rate_quantiles[2, index],
                    }
                )

    trajectory_label = (
        "fully non-parametric positive log-rate trajectory"
        if getattr(args, "fully_np_trajectory", False)
        else "NP-modulated rate-integrated square-root trajectory"
        if args.np_modulation
        else "shared rate-integrated square-root trajectory"
    )
    ar_mean = np.asarray(samples["ar_mean"]).reshape(-1, ar_layers, 3)
    lines = [
        f"# {scope_label(data['sca_names'])} NumPyro hierarchical TMM fit",
        "",
        f"Tile statistics: `{args.tile_statistics}`.",
        f"SCAs used: `{data['sca_names'].tolist()}`.",
        f"QE curves used by SCA: `{dict(zip(data['sca_names'].tolist(), (data['qe_columns'] + 1).tolist()))}`.",
        f"Coating model: `{ar_layers}` dielectric AR layer(s), ordered from air toward HgCdTe.",
        (
            "AR optical properties: direct single-tile parameters; no tile hierarchy."
            if is_single_tile_data(data)
            else "AR optical properties: shared n/k across all tiles; thickness varies hierarchically."
            if args.shared_ar_optical_properties
            else "AR optical properties: thickness, n, and k all vary hierarchically."
        ),
        (
            "AR absorption indices are fitted."
            if args.fit_ar_absorption
            else "AR layers are lossless dielectrics with k fixed to zero."
        ),
        f"Tiles per SCA: `{len(data['tile_ids'])}`; tile IDs: `{data['tile_ids'].tolist()}`.",
        (
            "HgCdTe n/k are fitted as global wavelength-independent optical "
            f"parameters with priors centered at `{NOMINAL_HGCDTE.tolist()}`; "
            f"bounds are `{HGCDTE_LOWER.tolist()}` to `{HGCDTE_UPPER.tolist()}`."
            if getattr(args, "fit_hgcdte", False)
            else f"HgCdTe fixed at n/k = `{NOMINAL_HGCDTE.tolist()}`."
        ),
        (
            "Conductive interlayer: global Drude-like film between the AR stack and HgCdTe; "
            f"reference wavelength `{CONDUCTIVE_REFERENCE_WAVELENGTH_NM}` nm and "
            f"epsilon_inf `{CONDUCTIVE_EPSILON_INF}`."
            if getattr(args, "fit_conductive_layer", False)
            else "No conductive interlayer is included."
        ),
        (
            f"QE is modulated by {len(qe_rbf_nodes)} normalized Gaussian RBF nodes at "
            f"`{qe_rbf_nodes.tolist()}` nm with width "
            f"`{qe_rbf_width_nm}` nm; log-node prior sigma is "
            f"`{qe_rbf_prior_sigma_log}`."
            if getattr(args, "fit_qe_rbf", False)
            else "QE uses the archived ground-measured wavelength weighting without a fitted RBF correction."
        ),
        (
            "Ice uses the published 100 K amorphous-H2O optical constants with "
            f"wavelength-dependent n and k from `{ASW_OPTICAL_CONSTANTS_URL}`."
            if getattr(args, "use_asw_optical_constants", False)
            else "Ice is modeled as a lossless constant-index layer with n=1.25 and k=0."
        ),
        (
            "ACM throughput steps are fitted independently for each LED."
            if getattr(args, "led_dependent_acm_step", False)
            else "One common ACM throughput step is fitted across all LEDs."
        ),
        (
            f"Likelihood pixel weight before selected-tile averaging: `{args.likelihood_weight}` "
            f"({'automatic one-effective-datum normalization' if args.likelihood_weight_automatic else 'user supplied'}); "
            f"effective aggregated SSE weight: `{args.likelihood_weight / len(data['tile_ids'])}`; "
            f"log-noise sigma: `{args.log_noise_sigma}`."
        ),
        f"AR population-mean prior sigma in unconstrained space: `{args.ar_mean_prior_sigma}`.",
        (
            (
                "Fully non-parametric node-rate prior: centered at the default "
                f"`{trajectory_rate_per_scale(data['days']) * TRAJECTORY_SCALE_INIT:.8g}` nm/day."
                if args.trajectory_rate_prior_mean_nm_day is None
                else "Fully non-parametric node-rate prior: centered at "
                f"`{args.trajectory_rate_prior_mean_nm_day}` nm/day."
            )
            if getattr(args, "fully_np_trajectory", False)
            else (
                "Unmodulated initial instantaneous-rate prior: centered at the default "
                f"`{trajectory_rate_per_scale(data['days']) * TRAJECTORY_SCALE_INIT:.8g}` nm/day."
                if args.trajectory_rate_prior_mean_nm_day is None
                else "Unmodulated initial instantaneous-rate prior: centered at "
                f"`{args.trajectory_rate_prior_mean_nm_day}` nm/day."
            )
        ),
        (
            f"Initial ice thickness is fixed at `{args.fixed_initial_thickness_nm}` nm."
            if getattr(args, "fixed_initial_thickness_nm", None) is not None
            else "Initial ice thickness is fitted with a truncated Normal prior over 0 to 50 nm."
        ),
        f"Trajectory-rate prior log-scale sigma: `{args.trajectory_rate_prior_sigma_log}`.",
        "The primary trajectory quantity is a positive instantaneous rate; ice thickness is its time integral.",
        (
            "Fully non-parametric rates are log-linearly interpolated between repeat nodes, "
            "so the positive accumulation trajectory remains monotonic."
            if getattr(args, "fully_np_trajectory", False)
            else "NP node values are log-linearly interpolated rate multipliers, so the rate is continuous at repeat nodes."
        ),
        (
            f"MAP-only trajectory-rate start: `{args.map_trajectory_rate_nm_day}` nm/day."
            if args.map_trajectory_rate_nm_day is not None
            else "MAP-only trajectory-rate start: historical initialization."
        ),
        (
            "Deterministic MAP-only evaluation; no NUTS posterior was sampled."
            if getattr(args, "map_only", False)
            else f"Warmup/samples/chains: `{args.warmup}/{args.samples}/{args.chains}`."
        ),
        (
            f"NUTS dense mass matrix: `{args.dense_mass}`; maximum tree depth: `{args.max_tree_depth}`."
            if not getattr(args, "map_only", False)
            else "NUTS settings were not used."
        ),
        (
            f"Wavelength grid: exact SED nodes / stride `{args.wavelength_stride}`."
            if args.wavelength_step_nm is None
            else f"Wavelength grid step/stride: `{args.wavelength_step_nm}` nm / `{args.wavelength_stride}`."
        ),
        (
            f"MAP optimizer/maximum iterations: `{args.map_optimizer}` / `{args.map_steps}`; "
            f"L-BFGS-B tolerance: `{args.map_tolerance}`."
            if args.map_optimizer == "lbfgs"
            else f"MAP optimizer/steps/learning rate: `{args.map_optimizer}` / "
            f"`{args.map_steps}` / `{args.map_learning_rate}`."
        ),
        f"Ice trajectory: `{trajectory_label}`.",
    ]
    if getattr(args, "tile_rate_variation", False):
        lines.extend(
            [
                "Per-SCA/tile instantaneous-rate variation is enabled.",
                f"Tile log-rate prior sigma: `{args.tile_rate_prior_sigma_log}`.",
                f"Neighboring-tile log-rate sigma: `{args.tile_rate_neighbor_sigma_log}`; selected adjacency edges per SCA: `{len(tile_neighbor_edges(data))}`.",
            ]
        )
    if args.ar_prior_from_map_start:
        lines.append("AR population-mean prior is centered on the physical MAP start.")
    map_init_max_tiles = getattr(args, "map_init_max_tiles", None)
    if map_init_max_tiles is not None:
        lines.append(
            f"Reduced MAP initialization tile count per SCA: `{map_init_max_tiles}`."
        )
    map_design_wavelength_nm = getattr(args, "map_design_wavelength_nm", None)
    if map_design_wavelength_nm is not None:
        middle_layer_start = (
            "lambda/8 middle-layer"
            if getattr(args, "map_middle_layer_eighth_wave", False)
            else "lambda/2 middle-layer"
            if getattr(args, "map_middle_layer_half_wave", False)
            else "quarter-wave"
        )
        lines.append(
            "MAP-only start uses equal log-index steps and "
            f"{middle_layer_start} optical thicknesses at "
            f"`{map_design_wavelength_nm}` nm; the posterior remains unconstrained."
        )
        if getattr(args, "map_middle_layer_eighth_wave", False):
            lines.append(
                "The three-layer MAP start uses lambda/4, lambda/8, lambda/4 "
                "optical thicknesses from air toward HgCdTe."
            )
        elif getattr(args, "map_middle_layer_half_wave", False):
            lines.append(
                "The three-layer MAP start uses lambda/4, lambda/2, lambda/4 "
                "optical thicknesses from air toward HgCdTe."
            )
    if data["n_scas"] > 1:
        lines.extend(
            [
                f"SCA AR sigma prior: `{args.ar_sca_sigma_prior}`; tile AR sigma prior: `{args.ar_tile_sigma_prior}`.",
                (
                    "AR thickness uses global population shrinkage, then SCA-level offsets, "
                    "then tile-level deviations; n/k are shared globally."
                    if args.shared_ar_optical_properties
                    else "AR parameters use global population shrinkage, then SCA-level offsets, then tile-level deviations."
                ),
            ]
        )
    if sampler_diagnostics:
        lines.extend(
            [
                f"Sampler divergences: `{sampler_diagnostics['divergences']}`.",
                f"Mean sampler acceptance probability: `{sampler_diagnostics['mean_accept_probability']:.8g}`.",
                f"Maximum NUTS leapfrog steps: `{sampler_diagnostics['max_num_steps']}`.",
                f"Tree-depth saturated transitions: `{sampler_diagnostics['tree_depth_saturated']}`.",
                f"Maximum finite r_hat: `{sampler_diagnostics['max_r_hat']:.8g}`.",
                f"Minimum effective sample size: `{sampler_diagnostics['min_effective_sample_size']:.8g}`.",
                f"Parameters with r_hat > 1.01: `{sampler_diagnostics['r_hat_gt_1_01']}`.",
                f"Parameters with effective sample size < 100: `{sampler_diagnostics['ess_lt_100']}`.",
                f"Non-finite r_hat diagnostics: `{sampler_diagnostics['r_hat_nonfinite']}`.",
                f"Non-finite effective sample size diagnostics: `{sampler_diagnostics['ess_nonfinite']}`.",
            ]
        )
    if map_diagnostics:
        lines.extend(
            [
                "",
                "## MAP optimization",
                f"- Optimizer: `{map_diagnostics['optimizer']}`.",
                f"- Reported convergence: `{map_diagnostics['converged']}`.",
                (
                    f"- Gradient convergence: `{map_diagnostics['gradient_converged']}`."
                    if "gradient_converged" in map_diagnostics
                    else "- Gradient convergence: `not assessed`."
                ),
                f"- Status: `{map_diagnostics['status']}`; message: `{map_diagnostics['message']}`.",
                f"- Iterations/function evaluations: `{map_diagnostics['iterations']}/{map_diagnostics['function_evaluations']}`.",
                f"- Initial/final objective: `{map_diagnostics['initial_objective']:.10g}` / `{map_diagnostics['objective']:.10g}`.",
                f"- Final gradient infinity norm: `{map_diagnostics['gradient_norm']:.10g}`.",
                f"- Parameter step norm: `{map_diagnostics['parameter_step_norm']:.10g}`.",
            ]
        )
    elif "diagnostic_diverging" in samples:
        divergences = int(np.count_nonzero(samples["diagnostic_diverging"]))
        accept_probability = np.asarray(samples["diagnostic_accept_prob"])
        num_steps = np.asarray(samples["diagnostic_num_steps"])
        lines.extend(
            [
                f"Sampler divergences: `{divergences}`.",
                f"Mean sampler acceptance probability: `{np.mean(accept_probability):.8g}`.",
                f"Maximum NUTS leapfrog steps: `{int(np.max(num_steps))}`.",
            ]
        )
    if args.np_modulation:
        lines.extend(
            [
                f"NP entropy/smoothness weights: `{args.np_entropy_weight}` / `{args.np_smoothness_weight}`.",
                f"NP node multiplier limit: `{args.np_max_node_multiplier}`; first node fixed at one.",
                "NP modulation is regularized in log-node space with generalized relative entropy and time-aware curvature.",
            ]
        )
    if getattr(args, "fully_np_trajectory", False):
        lines.extend(
            [
                f"Fully non-parametric trajectory smoothness weight: `{args.np_smoothness_weight}`.",
                f"Node-rate prior log sigma: `{args.trajectory_rate_prior_sigma_log}`; no square-root rate law is imposed.",
                "Node rates are positive and regularized by time-aware log-rate curvature.",
            ]
        )
    lines.extend(
        [
            "",
            "| Quantity | 16th percentile | Median | 84th percentile |",
            "|---|---:|---:|---:|",
        ]
    )
    summary_quantities = [("Initial thickness (nm)", "initial_thickness_nm")]
    if getattr(args, "fit_hgcdte", False):
        summary_quantities.extend(
            (
                ("HgCdTe refractive index", "hgcdte_n"),
                ("HgCdTe absorption index", "hgcdte_k"),
            )
        )
    if getattr(args, "fit_conductive_layer", False):
        summary_quantities.extend(
            (
                ("Conductive layer thickness (nm)", "conductive_thickness_nm"),
                ("Conductive plasma strength", "conductive_plasma_strength"),
                ("Conductive damping ratio", "conductive_damping_ratio"),
            )
        )
    for label, key in (
        ("Initial instantaneous rate (nm/day)", "initial_rate_nm_day"),
        ("Final instantaneous rate (nm/day)", "final_rate_nm_day"),
        ("Time-averaged instantaneous rate (nm/day)", "mean_rate_nm_day"),
    ):
        if key in samples:
            summary_quantities.append((label, key))
    for label, key in summary_quantities:
        values = np.asarray(samples[key])
        values = quantile(values)
        lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if getattr(args, "fit_qe_rbf", False):
        qe_node_quantiles = np.percentile(
            np.asarray(samples["qe_rbf_modulation"]), [16, 50, 84], axis=0
        )
        for index, wavelength in enumerate(qe_rbf_nodes):
            values = qe_node_quantiles[:, index]
            lines.append(
                f"| QE RBF multiplier at {wavelength:.0f} nm | "
                f"{values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |"
            )
    acm_step_samples = np.asarray(samples["acm_step_log_offset"])
    acm_step_percent = 100.0 * np.expm1(acm_step_samples)
    if acm_step_percent.ndim == 1:
        values = quantile(acm_step_percent)
        lines.append(
            f"| ACM throughput step (%) | {values[0]:.8g} | "
            f"{values[1]:.8g} | {values[2]:.8g} |"
        )
    elif acm_step_percent.ndim == 2 and acm_step_percent.shape[1] == 6:
        values = np.percentile(acm_step_percent, [16, 50, 84], axis=0)
        for led in range(6):
            lines.append(
                f"| ACM throughput step LED {led + 1} (%) | "
                f"{values[0, led]:.8g} | {values[1, led]:.8g} | "
                f"{values[2, led]:.8g} |"
            )
    else:
        raise ValueError("Unexpected ACM step posterior shape")
    if "tile_mean_rate_nm_day" in samples:
        values = quantile(samples["tile_mean_rate_nm_day"])
        lines.append(
            f"| Selected-tile mean rate distribution (nm/day) | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |"
        )
    for layer in range(ar_layers):
        values = quantile(ar_mean[:, layer, 0])
        label = "AR mean thickness scale" if ar_layers == 1 else f"AR layer {layer + 1} mean thickness scale"
        lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if args.np_modulation or getattr(args, "fully_np_trajectory", False):
        for label, key in (
            ("NP smoothness", "np_smoothness"),
        ):
            values = quantile(samples[key])
            lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
        if args.np_modulation:
            values = quantile(samples["np_entropy"])
            lines.append(f"| NP entropy | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if fit_diagnostics is not None:
        lines.extend(
            [
                "",
                "## Posterior median predictive fit",
                "",
                f"- Group-mean RMS: `{fit_diagnostics['group_rms_log']:.10g}` log.",
                f"- Non-reference group-mean RMS: `{fit_diagnostics['non_reference_group_rms_log']:.10g}` log.",
                f"- Count-weighted group-mean RMS: `{fit_diagnostics['count_weighted_group_rms_log']:.10g}` log.",
                f"- All-pixel RMS including within-tile scatter: `{fit_diagnostics['pixel_rms_log']:.10g}` log.",
                f"- Observed/model group-mean correlation: `{fit_diagnostics['correlation']:.10g}`.",
                "",
                "| LED | Group-mean RMS (log) |",
                "|---:|---:|",
            ]
        )
        for led, rms in enumerate(fit_diagnostics["led_rms_log"], start=1):
            lines.append(f"| {led} | {rms:.10g} |")
    if data["n_scas"] > 1:
        model_note = (
            "This all-SCA fit shares one ice trajectory globally and hierarchically "
            "shrinks tile AR thickness toward an SCA-specific population; dielectric "
            "n/k are shared across all tiles."
            if args.shared_ar_optical_properties
            else "This all-SCA fit shares one ice trajectory globally and hierarchically "
            "shrinks tile AR parameters toward an SCA-specific population."
        )
    else:
        model_note = (
            "This is a single-tile fit. The ice trajectory and all AR parameters are "
            "directly shared by the selected tile."
            if is_single_tile_data(data)
            else "This is a hierarchical fit. The ice trajectory is shared globally; "
            f"the {ar_layers} dielectric AR layer(s) vary by tile under hierarchical shrinkage."
        )
    lines.extend(
        [
            "",
            model_note,
            "Longer chains and tile-resolution sensitivity are required before interpreting "
            "the spatial AR field physically.",
        ]
    )
    (output_dir / "numpyro_summary.md").write_text("\n".join(lines) + "\n")


def summarize_sampler_diagnostics(chain_samples, extra_fields, max_tree_depth):
    """Return compact, durable convergence diagnostics for the fit summary."""
    from numpyro.diagnostics import summary

    posterior_summary = summary(chain_samples, group_by_chain=True)
    r_hat = np.concatenate(
        [
            np.asarray(values["r_hat"], dtype=float).reshape(-1)
            for values in posterior_summary.values()
            if "r_hat" in values
        ]
    )
    effective_sample_size = np.concatenate(
        [
            np.asarray(values["n_eff"], dtype=float).reshape(-1)
            for values in posterior_summary.values()
            if "n_eff" in values
        ]
    )
    finite_r_hat = r_hat[np.isfinite(r_hat)]
    finite_effective_sample_size = effective_sample_size[
        np.isfinite(effective_sample_size)
    ]
    num_steps = np.asarray(extra_fields["num_steps"])
    return {
        "divergences": int(np.count_nonzero(extra_fields["diverging"])),
        "mean_accept_probability": float(np.mean(extra_fields["accept_prob"])),
        "max_num_steps": int(np.max(num_steps)),
        "tree_depth_saturated": int(
            np.count_nonzero(num_steps >= 2**max_tree_depth - 1)
        ),
        "max_r_hat": float(np.max(finite_r_hat)) if finite_r_hat.size else np.nan,
        "min_effective_sample_size": (
            float(np.min(finite_effective_sample_size))
            if finite_effective_sample_size.size
            else np.nan
        ),
        "r_hat_gt_1_01": int(np.count_nonzero(finite_r_hat > 1.01)),
        "ess_lt_100": int(np.count_nonzero(finite_effective_sample_size < 100.0)),
        "r_hat_nonfinite": int(np.count_nonzero(~np.isfinite(r_hat))),
        "ess_nonfinite": int(
            np.count_nonzero(~np.isfinite(effective_sample_size))
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tile-statistics",
        type=Path,
        default=DEFAULT_SINGLE_SCA_TILE_STATISTICS,
    )
    sca_group = parser.add_mutually_exclusive_group()
    sca_group.add_argument(
        "--all-sca",
        dest="all_sca",
        action="store_true",
        help="Fit all completed per-SCA statistics under the all-SCA statistics directory.",
    )
    sca_group.add_argument(
        "--single-sca",
        dest="all_sca",
        action="store_false",
        help="Force the legacy single-SCA fit.",
    )
    parser.add_argument(
        "--all-sca-statistics-dir",
        type=Path,
        default=DEFAULT_ALL_SCA_STATISTICS_DIR,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SINGLE_SCA_OUTPUT)
    parser.add_argument(
        "--ar-layers",
        type=int,
        choices=tuple(range(1, AR_MAX_LAYERS + 1)),
        default=3,
        help="Number of stacked dielectric AR layers in the TMM coating model.",
    )
    parser.add_argument(
        "--shared-ar-optical-properties",
        action="store_true",
        help="Share each layer's refractive and absorption indices across all tiles; vary thickness hierarchically.",
    )
    parser.add_argument(
        "--fit-ar-absorption",
        action="store_true",
        help="Fit one absorption index k per AR layer instead of fixing dielectric k to zero.",
    )
    parser.add_argument(
        "--fit-hgcdte",
        action="store_true",
        help="Fit global wavelength-independent HgCdTe n and k instead of fixing them to the nominal constants.",
    )
    parser.add_argument(
        "--fit-conductive-layer",
        action="store_true",
        help=(
            "Fit a global thin Drude-like conductive layer between the AR stack "
            "and HgCdTe."
        ),
    )
    parser.add_argument(
        "--use-asw-optical-constants",
        action="store_true",
        help=(
            "Use wavelength-dependent published 100 K amorphous-H2O n/k "
            "constants for the ice layer instead of n=1.25+0j."
        ),
    )
    parser.add_argument(
        "--led-dependent-acm-step",
        action="store_true",
        help="Fit an independent ACM throughput step for each of the six LEDs.",
    )
    parser.add_argument(
        "--fit-qe-rbf",
        action="store_true",
        help=(
            "Fit smooth wavelength-dependent QE log-modulation nodes."
        ),
    )
    parser.add_argument(
        "--qe-rbf-node-wavelengths-nm",
        type=float,
        nargs="+",
        default=QE_RBF_NODE_WAVELENGTHS_NM.tolist(),
        help="Strictly increasing QE RBF node wavelengths in nm.",
    )
    parser.add_argument(
        "--qe-rbf-width-nm",
        type=float,
        default=QE_RBF_WIDTH_NM,
        help="Gaussian QE RBF width in nm.",
    )
    parser.add_argument(
        "--qe-rbf-prior-sigma-log",
        type=float,
        default=QE_RBF_PRIOR_SIGMA_LOG,
        help="Normal prior sigma for QE RBF log multipliers.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Tiles per SCA to fit; default is all available tiles.",
    )
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument(
        "--wavelength-stride",
        type=int,
        default=1,
        help="Sample the QE-supported wavelength grid every N nm; 1 is the production default.",
    )
    parser.add_argument(
        "--wavelength-step-nm",
        type=float,
        default=None,
        help="Use a uniform base grid at this spacing; omit to evaluate exact SED nodes.",
    )
    parser.add_argument("--log-noise-sigma", type=float, default=0.005)
    parser.add_argument(
        "--trajectory-rate-prior-mean-nm-day",
        type=float,
        default=None,
        help=(
            "Optional center of the unmodulated instantaneous-rate prior at the "
            "first observed time, in nm/day. If omitted, retain the historical "
            "prior center."
        ),
    )
    parser.add_argument(
        "--trajectory-rate-prior-sigma-log",
        type=float,
        default=0.75,
        help="Normal prior sigma for log trajectory scale; 1.0 is deliberately broad.",
    )
    parser.add_argument(
        "--fixed-initial-thickness-nm",
        type=float,
        default=None,
        help="Fix ice thickness at the first observed/reference time instead of fitting it.",
    )
    parser.add_argument(
        "--tile-rate-variation",
        action="store_true",
        help=(
            "Fit a log-rate offset for each selected SCA/tile, with neighboring "
            "tile offsets regularized toward one another."
        ),
    )
    parser.add_argument(
        "--tile-rate-prior-sigma-log",
        type=float,
        default=TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG,
        help="Prior sigma in log rate for each SCA/tile offset.",
    )
    parser.add_argument(
        "--tile-rate-neighbor-sigma-log",
        type=float,
        default=TRAJECTORY_TILE_RATE_NEIGHBOR_SIGMA_LOG,
        help="Prior sigma in log rate for neighboring tile differences.",
    )
    parser.add_argument(
        "--likelihood-weight",
        type=float,
        default=None,
        help=(
            "Pixel-SSE weight before averaging across selected tiles. By default "
            "use the inverse median valid pixel count, making the aggregate "
            "tile/group contribution approximately one effective datum."
        ),
    )
    parser.add_argument(
        "--ar-mean-prior-sigma",
        type=float,
        default=AR_MEAN_PRIOR_SIGMA,
        help="Normal prior sigma for global AR raw parameters; larger values relax the AR reference center.",
    )
    parser.add_argument(
        "--ar-prior-from-map-start",
        action="store_true",
        help="Center the global AR prior on the physical MAP initialization instead of the historical reference.",
    )
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    chain_group = parser.add_mutually_exclusive_group()
    chain_group.add_argument(
        "--parallel-chains",
        action="store_true",
        help="Run chains with pmap on separate host devices.",
    )
    chain_group.add_argument(
        "--vectorized-chains",
        action="store_true",
        help="Run chains as one batched JAX computation on the available device.",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--dense-mass",
        action="store_true",
        help="Adapt a dense NUTS mass matrix; useful for the correlated single-tile optical fit.",
    )
    parser.add_argument(
        "--map-steps",
        type=int,
        default=0,
        help="Run this many full-model AutoDelta MAP steps before NUTS (default: disabled).",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Write a deterministic MAP evaluation after --map-steps instead of running NUTS.",
    )
    parser.add_argument(
        "--map-optimizer",
        choices=("lbfgs", "adam"),
        default="lbfgs",
        help="Optimizer for --map-steps; L-BFGS-B is the deterministic default.",
    )
    parser.add_argument(
        "--map-tolerance",
        type=float,
        default=1.0e-9,
        help="Function/gradient tolerance for the L-BFGS-B MAP optimizer.",
    )
    parser.add_argument(
        "--map-learning-rate",
        type=float,
        default=0.01,
        help="Adam learning rate for optional MAP initialization.",
    )
    parser.add_argument(
        "--map-trajectory-rate-nm-day",
        type=float,
        default=None,
        help=(
            "MAP-only initial unmodulated instantaneous rate at the first observed "
            "time, in nm/day. This changes the optimizer start, not the posterior "
            "prior."
        ),
    )
    parser.add_argument(
        "--map-init-max-tiles",
        type=int,
        default=None,
        help=(
            "Run the MAP initializer first on this many tiles per SCA, then "
            "transfer compatible sites to the full selected-tile fit."
        ),
    )
    parser.add_argument(
        "--map-design-wavelength-nm",
        type=float,
        default=None,
        help=(
            "Use a quarter-wave, equal-log-index optical stack only for MAP "
            "initialization; this does not constrain posterior sampling."
        ),
    )
    map_middle_layer_group = parser.add_mutually_exclusive_group()
    map_middle_layer_group.add_argument(
        "--map-middle-layer-eighth-wave",
        action="store_true",
        help=(
            "For a three-layer MAP start, use lambda/4, lambda/8, lambda/4 "
            "optical thicknesses instead of three quarter-wave layers."
        ),
    )
    map_middle_layer_group.add_argument(
        "--map-middle-layer-half-wave",
        action="store_true",
        help=(
            "For a three-layer MAP start, use lambda/4, lambda/2, lambda/4 "
            "optical thicknesses instead of three quarter-wave layers."
        ),
    )
    parser.add_argument(
        "--ar-sca-sigma-prior",
        type=float,
        default=AR_SCA_SIGMA_PRIOR,
        help="HalfNormal scale for SCA-to-SCA AR offsets in unconstrained parameter space.",
    )
    parser.add_argument(
        "--ar-tile-sigma-prior",
        type=float,
        default=AR_TILE_SIGMA_PRIOR,
        help="HalfNormal scale for tile AR offsets around each SCA.",
    )
    trajectory_model_group = parser.add_mutually_exclusive_group()
    trajectory_model_group.add_argument(
        "--np-modulation",
        action="store_true",
        help="Use entropy-regularized repeat-node modulation of the square-root rate.",
    )
    trajectory_model_group.add_argument(
        "--fully-np-trajectory",
        action="store_true",
        help=(
            "Fit a positive instantaneous rate independently at every repeat node, "
            "with the existing curvature smoothness regularization, instead of using "
            "the square-root trajectory."
        ),
    )
    parser.add_argument("--np-entropy-weight", type=float, default=NP_ENTROPY_WEIGHT)
    parser.add_argument(
        "--np-smoothness-weight",
        type=float,
        default=NP_SMOOTHNESS_WEIGHT,
    )
    parser.add_argument(
        "--np-max-node-multiplier",
        type=float,
        default=NP_MAX_NODE_MULTIPLIER,
    )
    parser.set_defaults(all_sca=None)
    return parser.parse_args()


def main():
    args = parse_args()
    (
        args.qe_rbf_node_wavelengths_nm,
        args.qe_rbf_width_nm,
        args.qe_rbf_prior_sigma_log,
    ) = qe_rbf_configuration(
        args.qe_rbf_node_wavelengths_nm,
        args.qe_rbf_width_nm,
        args.qe_rbf_prior_sigma_log,
    )
    if args.log_noise_sigma <= 0.0 or (
        args.likelihood_weight is not None and args.likelihood_weight <= 0.0
    ):
        raise ValueError("Noise sigma and an explicit likelihood weight must be positive")
    if args.ar_mean_prior_sigma <= 0.0:
        raise ValueError("ar-mean-prior-sigma must be positive")
    if args.trajectory_rate_prior_mean_nm_day is not None and (
        not np.isfinite(args.trajectory_rate_prior_mean_nm_day)
        or args.trajectory_rate_prior_mean_nm_day <= 0.0
    ):
        raise ValueError("trajectory-rate-prior-mean-nm-day must be positive and finite")
    if args.trajectory_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        args.trajectory_rate_prior_sigma_log
    ):
        raise ValueError("trajectory-rate-prior-sigma-log must be positive and finite")
    if args.fixed_initial_thickness_nm is not None and (
        not np.isfinite(args.fixed_initial_thickness_nm)
        or args.fixed_initial_thickness_nm < 0.0
        or args.fixed_initial_thickness_nm > 50.0
    ):
        raise ValueError("fixed-initial-thickness-nm must be between 0 and 50")
    if args.tile_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        args.tile_rate_prior_sigma_log
    ):
        raise ValueError("tile-rate-prior-sigma-log must be positive and finite")
    if args.tile_rate_neighbor_sigma_log <= 0.0 or not np.isfinite(
        args.tile_rate_neighbor_sigma_log
    ):
        raise ValueError("tile-rate-neighbor-sigma-log must be positive and finite")
    if args.warmup < 1 or args.samples < 1 or args.chains < 1 or args.max_tree_depth < 1:
        raise ValueError("Warmup, samples, chains, and max tree depth must be positive")
    if args.map_steps < 0 or args.map_learning_rate <= 0.0:
        raise ValueError("MAP steps must be non-negative and its learning rate must be positive")
    if args.map_tolerance <= 0.0 or not np.isfinite(args.map_tolerance):
        raise ValueError("map-tolerance must be positive and finite")
    if args.map_only and args.map_steps < 1:
        raise ValueError("--map-only requires --map-steps")
    if args.map_trajectory_rate_nm_day is not None and (
        not np.isfinite(args.map_trajectory_rate_nm_day)
        or args.map_trajectory_rate_nm_day <= 0.0
    ):
        raise ValueError("map-trajectory-rate-nm-day must be positive and finite")
    if args.map_init_max_tiles is not None and args.map_init_max_tiles < 1:
        raise ValueError("map-init-max-tiles must be positive")
    if args.map_design_wavelength_nm is not None and args.map_design_wavelength_nm <= 0.0:
        raise ValueError("map-design-wavelength-nm must be positive")
    if (
        args.map_middle_layer_eighth_wave or args.map_middle_layer_half_wave
    ) and args.ar_layers != 3:
        raise ValueError(
            "A special middle-layer thickness requires --ar-layers 3"
        )
    if args.wavelength_step_nm is not None and (
        not np.isfinite(args.wavelength_step_nm) or args.wavelength_step_nm <= 0.0
    ):
        raise ValueError("wavelength-step-nm must be positive and finite")
    if args.np_entropy_weight < 0.0 or args.np_smoothness_weight < 0.0:
        raise ValueError("NP regularization weights must be non-negative")
    if args.np_modulation and args.np_max_node_multiplier <= 1.0:
        raise ValueError("np-max-node-multiplier must be greater than one")
    if args.ar_sca_sigma_prior <= 0.0 or args.ar_tile_sigma_prior <= 0.0:
        raise ValueError("AR hierarchy sigma priors must be positive")
    ar_reference, ar_lower, ar_upper = ar_prior_arrays(args.ar_layers)
    jax, _, numpyro, _, MCMC, NUTS, init_to_value = require_numpyro()
    if args.parallel_chains and args.chains > 1:
        numpyro.set_host_device_count(args.chains)
    jax.config.update("jax_enable_x64", True)
    if args.all_sca is None:
        args.all_sca = (
            args.tile_statistics == DEFAULT_SINGLE_SCA_TILE_STATISTICS
            and args.all_sca_statistics_dir.is_dir()
            and len(
                list(args.all_sca_statistics_dir.glob("wfi*_tile_log_statistics_*.csv"))
            )
            == 18
        )
    if args.all_sca:
        completed_sca_paths = statistics_paths(args.all_sca_statistics_dir)
        if len(completed_sca_paths) != 18:
            raise ValueError(
                "All-SCA TMM fitting requires 18 completed per-SCA statistics CSVs"
            )
    statistics_path = (
        args.all_sca_statistics_dir if args.all_sca else args.tile_statistics
    )
    args.tile_statistics = statistics_path
    if args.all_sca and args.output_dir == DEFAULT_SINGLE_SCA_OUTPUT:
        args.output_dir = DEFAULT_ALL_SCA_OUTPUT
    if args.ar_layers != 3 and args.output_dir in (
        DEFAULT_SINGLE_SCA_OUTPUT,
        DEFAULT_ALL_SCA_OUTPUT,
    ):
        args.output_dir = args.output_dir.with_name(
            f"{args.output_dir.name}_ar{args.ar_layers}layer"
        )
    data = load_tile_statistics(statistics_path, args.max_tiles)
    args.likelihood_weight_automatic = args.likelihood_weight is None
    if args.likelihood_weight_automatic:
        args.likelihood_weight = default_likelihood_weight(data["counts"])
    context = load_response_context(
        args.model_path,
        args.sed_mode,
        args.wavelength_stride,
        data["sca_names"],
        args.wavelength_step_nm,
        ASW_OPTICAL_CONSTANTS_URL if args.use_asw_optical_constants else None,
    )
    data["qe_columns"] = context["qe_columns"]
    initial_values = initial_values_for_data(
        data, args, ar_reference, ar_lower, ar_upper
    )
    ar_prior_reference = (
        map_start_reference(args, ar_reference, ar_lower, ar_upper)
        if args.ar_prior_from_map_start
        else ar_reference
    )
    model = build_model(
        data,
        context,
        args.log_noise_sigma,
        args.likelihood_weight,
        ar_mean_prior_sigma=args.ar_mean_prior_sigma,
        ar_prior_reference=ar_prior_reference,
        ar_layers=args.ar_layers,
        np_modulation=args.np_modulation,
        fully_np_trajectory=args.fully_np_trajectory,
        np_entropy_weight=args.np_entropy_weight,
        np_smoothness_weight=args.np_smoothness_weight,
        np_max_node_multiplier=args.np_max_node_multiplier,
        ar_sca_sigma_prior=args.ar_sca_sigma_prior,
        ar_tile_sigma_prior=args.ar_tile_sigma_prior,
        shared_ar_optical_properties=args.shared_ar_optical_properties,
        fit_ar_absorption=args.fit_ar_absorption,
        fit_hgcdte=args.fit_hgcdte,
        fit_conductive_layer=args.fit_conductive_layer,
        fit_qe_rbf=args.fit_qe_rbf,
        qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
        qe_rbf_width_nm=args.qe_rbf_width_nm,
        qe_rbf_prior_sigma_log=args.qe_rbf_prior_sigma_log,
        use_asw_optical_constants=args.use_asw_optical_constants,
        led_dependent_acm_step=args.led_dependent_acm_step,
        trajectory_rate_prior_mean_nm_day=args.trajectory_rate_prior_mean_nm_day,
        trajectory_rate_prior_sigma_log=args.trajectory_rate_prior_sigma_log,
        tile_rate_variation=args.tile_rate_variation,
        tile_rate_prior_sigma_log=args.tile_rate_prior_sigma_log,
        tile_rate_neighbor_sigma_log=args.tile_rate_neighbor_sigma_log,
        fixed_initial_thickness_nm=args.fixed_initial_thickness_nm,
    )
    map_diagnostics = None
    if (
        args.map_steps
        and args.map_init_max_tiles is not None
        and args.map_init_max_tiles < data["n_tiles"]
    ):
        reduced_data = load_tile_statistics(
            statistics_path, max_tiles=args.map_init_max_tiles
        )
        reduced_model = build_model(
            reduced_data,
            context,
            args.log_noise_sigma,
            args.likelihood_weight,
            ar_mean_prior_sigma=args.ar_mean_prior_sigma,
            ar_prior_reference=ar_prior_reference,
            ar_layers=args.ar_layers,
            np_modulation=args.np_modulation,
            fully_np_trajectory=args.fully_np_trajectory,
            np_entropy_weight=args.np_entropy_weight,
            np_smoothness_weight=args.np_smoothness_weight,
            np_max_node_multiplier=args.np_max_node_multiplier,
            ar_sca_sigma_prior=args.ar_sca_sigma_prior,
            ar_tile_sigma_prior=args.ar_tile_sigma_prior,
            shared_ar_optical_properties=args.shared_ar_optical_properties,
            fit_ar_absorption=args.fit_ar_absorption,
            fit_hgcdte=args.fit_hgcdte,
            fit_conductive_layer=args.fit_conductive_layer,
            fit_qe_rbf=args.fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
            qe_rbf_width_nm=args.qe_rbf_width_nm,
            qe_rbf_prior_sigma_log=args.qe_rbf_prior_sigma_log,
            use_asw_optical_constants=args.use_asw_optical_constants,
            led_dependent_acm_step=args.led_dependent_acm_step,
            trajectory_rate_prior_mean_nm_day=args.trajectory_rate_prior_mean_nm_day,
            trajectory_rate_prior_sigma_log=args.trajectory_rate_prior_sigma_log,
            tile_rate_variation=args.tile_rate_variation,
            tile_rate_prior_sigma_log=args.tile_rate_prior_sigma_log,
            tile_rate_neighbor_sigma_log=args.tile_rate_neighbor_sigma_log,
            fixed_initial_thickness_nm=args.fixed_initial_thickness_nm,
        )
        reduced_initial_values = initial_values_for_data(
            reduced_data, args, ar_reference, ar_lower, ar_upper
        )
        if args.map_optimizer == "lbfgs":
            reduced_map_values, _ = run_map_optimization(
                reduced_model,
                reduced_initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
            )
        else:
            reduced_map_values, _ = run_map_initialization(
                reduced_model,
                reduced_initial_values,
                jax,
                init_to_value,
                args.seed + 1,
                args.map_steps,
                args.map_learning_rate,
            )
        initial_values = transfer_map_initial_values(
            reduced_map_values, initial_values
        )
    if args.map_steps:
        if args.map_optimizer == "lbfgs":
            initial_values, map_diagnostics = run_map_optimization(
                model,
                initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
            )
        else:
            initial_values, map_diagnostics = run_map_initialization(
                model,
                initial_values,
                jax,
                init_to_value,
                args.seed + 1,
                args.map_steps,
                args.map_learning_rate,
            )
    if args.map_only:
        from numpyro.infer import Predictive

        map_posterior = {
            name: jax.numpy.asarray(value)[None, ...]
            for name, value in initial_values.items()
        }
        map_samples = Predictive(model, posterior_samples=map_posterior)(
            jax.random.PRNGKey(args.seed + 2)
        )
        samples = {
            name: np.asarray(value)
            for name, value in {**map_posterior, **map_samples}.items()
        }
        samples["chain_index"] = np.zeros(1, dtype=int)
        fit_diagnostics = posterior_median_fit_diagnostics(
            data,
            context,
            samples,
            fit_hgcdte=args.fit_hgcdte,
            fit_conductive_layer=args.fit_conductive_layer,
            fit_qe_rbf=args.fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
            qe_rbf_width_nm=args.qe_rbf_width_nm,
            use_asw_optical_constants=args.use_asw_optical_constants,
        )
        write_outputs(
            args.output_dir,
            data,
            samples,
            args,
            fit_diagnostics=fit_diagnostics,
            map_diagnostics=map_diagnostics,
        )
        print(f"Wrote {args.output_dir / 'numpyro_summary.md'}")
        print(f"Wrote {args.output_dir / 'tile_ar_posterior.csv'}")
        if data["n_scas"] > 1:
            print(f"Wrote {args.output_dir / 'sca_ar_posterior.csv'}")
        if "tile_initial_rate_nm_day" in samples:
            print(f"Wrote {args.output_dir / 'tile_rate_posterior.csv'}")
        if args.np_modulation or args.fully_np_trajectory:
            print(f"Wrote {args.output_dir / 'np_node_posterior.csv'}")
        return
    kernel = NUTS(
        model,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        dense_mass=args.dense_mass,
        init_strategy=init_to_value(values=initial_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method=(
            "parallel"
            if args.parallel_chains
            else "vectorized"
            if args.vectorized_chains
            else "sequential"
        ),
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(args.seed),
        extra_fields=("accept_prob", "diverging", "num_steps"),
    )
    chain_samples = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    sampler_diagnostics = None
    if args.samples >= 4:
        sampler_diagnostics = summarize_sampler_diagnostics(
            chain_samples,
            extra_fields,
            args.max_tree_depth,
        )
    samples = jax.tree_util.tree_map(
        lambda value: value.reshape((-1,) + value.shape[2:]),
        chain_samples,
    )
    chain_index = np.repeat(np.arange(args.chains), args.samples)
    samples["chain_index"] = chain_index
    for name, value in extra_fields.items():
        samples[f"diagnostic_{name}"] = np.asarray(value).reshape(-1)
    fit_diagnostics = posterior_median_fit_diagnostics(
        data,
        context,
        samples,
        fit_hgcdte=args.fit_hgcdte,
        fit_conductive_layer=args.fit_conductive_layer,
        fit_qe_rbf=args.fit_qe_rbf,
        qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
        qe_rbf_width_nm=args.qe_rbf_width_nm,
        use_asw_optical_constants=args.use_asw_optical_constants,
    )
    write_outputs(
        args.output_dir,
        data,
        samples,
        args,
        fit_diagnostics=fit_diagnostics,
        sampler_diagnostics=sampler_diagnostics,
        map_diagnostics=map_diagnostics,
    )
    if args.samples >= 4:
        mcmc.print_summary()
    else:
        print("Skipped NumPyro summary: at least four samples are required for r_hat.")
    print(f"Wrote {args.output_dir / 'numpyro_summary.md'}")
    print(f"Wrote {args.output_dir / 'tile_ar_posterior.csv'}")
    if data["n_scas"] > 1:
        print(f"Wrote {args.output_dir / 'sca_ar_posterior.csv'}")
    if "tile_initial_rate_nm_day" in samples:
        print(f"Wrote {args.output_dir / 'tile_rate_posterior.csv'}")
    if args.np_modulation or args.fully_np_trajectory:
        print(f"Wrote {args.output_dir / 'np_node_posterior.csv'}")


if __name__ == "__main__":
    main()
