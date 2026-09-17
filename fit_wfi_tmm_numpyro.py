"""NumPyro TMM fit with shared ice and hierarchical SCA/tile AR variation."""

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_rcs_soc_ice_model import read_model_table
from analyze_wfi01_all_pixel_soc_model import parse_mast_time
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
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

REFERENCE_THICKNESS_NM = 122.7
REFERENCE_TIME_DAYS = 4.65
RESET_DAY = 18.0
INITIAL_THICKNESS_INIT_NM = 35.9
TRAJECTORY_SCALE_INIT = 1.14
NOMINAL_HGCDTE = np.asarray([3.6, 0.7], dtype=float)
AR_REFERENCE = np.asarray([2.65870883, 2.88251153, 0.43806530], dtype=float)
AR_LOWER = np.asarray([0.01, 1.0, 0.0], dtype=float)
AR_UPPER = np.asarray([10.0, 4.0, 0.5], dtype=float)
AR_SCA_SIGMA_PRIOR = 0.2
AR_TILE_SIGMA_PRIOR = 0.5
NP_ENTROPY_WEIGHT = 0.01
NP_SMOOTHNESS_WEIGHT = 4.64158883361
NP_MAX_NODE_MULTIPLIER = 2.0


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


def load_tile_statistics(path, max_tiles=None):
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
        "n_scas": n_scas,
        "n_tiles": n_tiles,
    }


def load_response_context(model_path, sed_mode, wavelength_stride):
    model_wavelengths, _, _ = read_model_table(model_path)
    if wavelength_stride < 1:
        raise ValueError("wavelength_stride must be positive")
    model_wavelengths = model_wavelengths[::wavelength_stride]
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    integration = prepare_response_integration(
        model_wavelengths,
        spectra,
        qe_wavelengths,
        qe_values,
        sed_mode,
    )
    return {
        "wavelengths": model_wavelengths,
        "model_weights": integration[0],
        "outside_output": integration[1],
        "clean_output": integration[2],
    }


def build_prediction_function(data, context):
    _, jnp, _, _, _, _, _ = require_numpyro()
    wavelengths = jnp.asarray(context["wavelengths"])
    model_weights = jnp.asarray(context["model_weights"])
    outside_output = jnp.asarray(context["outside_output"])
    clean_output = jnp.asarray(context["clean_output"])
    led_index = jnp.asarray(data["led"] - 1, dtype=jnp.int32)
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    step_design = jnp.asarray(data["step_design"])
    substrate_index = NOMINAL_HGCDTE[0] + 1j * NOMINAL_HGCDTE[1]
    ice_index = 1.25 + 0.0j

    def layer_matrix(index, thickness):
        index, thickness = jnp.broadcast_arrays(
            jnp.atleast_1d(index), jnp.atleast_1d(thickness)
        )
        phase = 2.0 * jnp.pi * index[:, None] * thickness[:, None] / wavelengths[None, :]
        cosine = jnp.cos(phase)
        sine = 1j * jnp.sin(phase)
        matrix = jnp.zeros((index.shape[0], wavelengths.shape[0], 2, 2), dtype=jnp.complex128)
        matrix = matrix.at[:, :, 0, 0].set(cosine)
        matrix = matrix.at[:, :, 1, 1].set(cosine)
        matrix = matrix.at[:, :, 0, 1].set(sine / index[:, None])
        matrix = matrix.at[:, :, 1, 0].set(sine * index[:, None])
        return matrix

    def transmission(matrix):
        denominator = (
            matrix[..., 0, 0]
            + substrate_index * matrix[..., 0, 1]
            + matrix[..., 1, 0]
            + substrate_index * matrix[..., 1, 1]
        )
        amplitude = 2.0 / denominator
        return jnp.real(substrate_index) * jnp.abs(amplitude) ** 2

    def response_curves(ice_thickness, ar_parameters):
        ar_index = ar_parameters[:, 1] + 1j * ar_parameters[:, 2]
        ar_matrix = layer_matrix(ar_index, 100.0 * ar_parameters[:, 0])
        clean = transmission(ar_matrix)
        ice_matrix = layer_matrix(ice_index, ice_thickness)
        total = jnp.einsum("rwab,twbc->trwac", ice_matrix, ar_matrix)
        coated = transmission(total)
        ratios = coated / clean[:, None, :]
        integrated = jnp.einsum("brw,lw->brl", ratios, model_weights)
        integrated = integrated + outside_output[None, None, :]
        return integrated / clean_output[None, None, :]

    def prediction(ice_thickness, ar_parameters, acm_step):
        response = response_curves(ice_thickness, ar_parameters)
        current = response[:, jnp.arange(response.shape[1]), led_index]
        baseline_thickness = ice_thickness[first_by_led[led_index]]
        baseline_response = response_curves(baseline_thickness, ar_parameters)
        baseline = baseline_response[:, jnp.arange(response.shape[1]), led_index]
        return jnp.log(current / baseline) + acm_step * step_design[None, :]

    return prediction


def build_model(
    data,
    context,
    log_noise_sigma,
    likelihood_weight,
    np_modulation=False,
    np_entropy_weight=NP_ENTROPY_WEIGHT,
    np_smoothness_weight=NP_SMOOTHNESS_WEIGHT,
    np_max_node_multiplier=NP_MAX_NODE_MULTIPLIER,
    ar_sca_sigma_prior=AR_SCA_SIGMA_PRIOR,
    ar_tile_sigma_prior=AR_TILE_SIGMA_PRIOR,
):
    jax, jnp, numpyro, dist, _, _, _ = require_numpyro()
    if np_modulation and np_max_node_multiplier <= 1.0:
        raise ValueError("np_max_node_multiplier must be greater than one")
    if ar_sca_sigma_prior <= 0.0 or ar_tile_sigma_prior <= 0.0:
        raise ValueError("AR hierarchy sigma priors must be positive")
    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    days = jnp.asarray(data["days"])
    repeat_index = jnp.asarray(data["repeat_index"], dtype=jnp.int32)
    node_days = jnp.asarray(data["node_days"])
    base_thickness = REFERENCE_THICKNESS_NM * (
        jnp.sqrt((RESET_DAY + days) / REFERENCE_TIME_DAYS)
        - jnp.sqrt(RESET_DAY / REFERENCE_TIME_DAYS)
    )
    node_base_thickness = REFERENCE_THICKNESS_NM * (
        jnp.sqrt((RESET_DAY + node_days) / REFERENCE_TIME_DAYS)
        - jnp.sqrt(RESET_DAY / REFERENCE_TIME_DAYS)
    )
    null_sse = jnp.sum(sum_squared)
    likelihood_scale = 0.5 * likelihood_weight * null_sse / (log_noise_sigma**2)
    reference_probability = jnp.asarray(
        (AR_REFERENCE - AR_LOWER) / (AR_UPPER - AR_LOWER)
    )
    reference_raw = jnp.log(reference_probability / (1.0 - reference_probability))
    prediction = build_prediction_function(data, context)
    n_scas = int(data["n_scas"])
    n_tiles = int(data["n_tiles"])

    def model():
        ar_mu_raw = numpyro.sample(
            "ar_mu_raw",
            dist.Normal(reference_raw, jnp.ones(3)).to_event(1),
        )
        if n_scas == 1:
            ar_sigma = numpyro.sample(
                "ar_sigma_raw",
                dist.HalfNormal(0.5 * jnp.ones(3)).to_event(1),
            )
            ar_z = numpyro.sample(
                "ar_z",
                dist.Normal(0.0, 1.0).expand((n_tiles, 3)).to_event(2),
            )
            ar_raw = ar_mu_raw[None, :] + ar_sigma[None, :] * ar_z
            ar_parameters = AR_LOWER + (AR_UPPER - AR_LOWER) * jax.nn.sigmoid(ar_raw)
            ar_parameters_for_prediction = ar_parameters
        else:
            ar_sca_sigma = numpyro.sample(
                "ar_sca_sigma",
                dist.HalfNormal(ar_sca_sigma_prior * jnp.ones(3)).to_event(1),
            )
            ar_sca_z = numpyro.sample(
                "ar_sca_z",
                dist.Normal(0.0, 1.0).expand((n_scas, 3)).to_event(2),
            )
            ar_sca_raw = ar_mu_raw[None, :] + ar_sca_sigma[None, :] * ar_sca_z
            ar_tile_sigma = numpyro.sample(
                "ar_tile_sigma",
                dist.HalfNormal(ar_tile_sigma_prior * jnp.ones(3)).to_event(1),
            )
            ar_tile_z = numpyro.sample(
                "ar_tile_z",
                dist.Normal(0.0, 1.0)
                .expand((n_scas, n_tiles, 3))
                .to_event(3),
            )
            ar_raw = ar_sca_raw[:, None, :] + ar_tile_sigma[None, None, :] * ar_tile_z
            ar_parameters = AR_LOWER + (AR_UPPER - AR_LOWER) * jax.nn.sigmoid(ar_raw)
            ar_parameters_for_prediction = ar_parameters.reshape((-1, 3))
        d0 = numpyro.sample(
            "initial_thickness_nm",
            dist.TruncatedNormal(35.0, 15.0, low=0.0, high=50.0),
        )
        log_scale = numpyro.sample(
            "log_trajectory_scale", dist.Normal(jnp.log(TRAJECTORY_SCALE_INIT), 0.75)
        )
        acm_step = numpyro.sample("acm_step_log_offset", dist.Normal(0.0, 0.05))
        if np_modulation:
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
            ice_thickness = (
                d0
                + jnp.exp(log_scale)
                * base_thickness
                * node_multipliers[repeat_index]
            )
            scaled_node_days = (node_days - node_days[0]) / jnp.median(
                jnp.diff(node_days)
            )
            node_slopes = jnp.diff(log_node_multipliers) / jnp.diff(scaled_node_days)
            node_curvature = 2.0 * jnp.diff(node_slopes) / (
                scaled_node_days[2:] - scaled_node_days[:-2]
            )
            np_entropy = jnp.mean(
                node_multipliers * log_node_multipliers
                - node_multipliers
                + 1.0
            )
            np_smoothness = jnp.mean(node_curvature * node_curvature)
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
            ice_thickness = d0 + jnp.exp(log_scale) * base_thickness
        fitted = prediction(ice_thickness, ar_parameters_for_prediction, acm_step)
        residual_sse = (
            sum_squared
            - 2.0 * fitted * sums
            + counts * fitted * fitted
        )
        total_sse = jnp.sum(residual_sse)
        numpyro.factor(
            "tile_gaussian_log_likelihood",
            -0.5 * likelihood_weight * total_sse / (log_noise_sigma**2),
        )
        numpyro.deterministic("ar_parameters", ar_parameters)
        numpyro.deterministic("ar_mean", AR_LOWER + (AR_UPPER - AR_LOWER) * jax.nn.sigmoid(ar_mu_raw))
        if n_scas > 1:
            numpyro.deterministic(
                "ar_sca_mean",
                AR_LOWER
                + (AR_UPPER - AR_LOWER) * jax.nn.sigmoid(ar_sca_raw),
            )
        numpyro.deterministic("ice_thickness", ice_thickness)
        if np_modulation:
            numpyro.deterministic("np_log_node_multipliers", log_node_multipliers)
            numpyro.deterministic("np_node_multipliers", node_multipliers)
            numpyro.deterministic(
                "np_node_thickness",
                d0 + jnp.exp(log_scale) * node_base_thickness * node_multipliers,
            )
            first_group = int(np.argmin(data["days"]))
            last_group = int(np.argmax(data["days"]))
            rate_numerator = ice_thickness[last_group] - ice_thickness[first_group]
        else:
            rate_numerator = jnp.exp(log_scale) * (base_thickness[-1] - base_thickness[0])
        numpyro.deterministic(
            "mean_rate_nm_day",
            rate_numerator / (jnp.max(days) - jnp.min(days)),
        )
        if np_modulation:
            numpyro.deterministic("np_entropy", np_entropy)
            numpyro.deterministic("np_smoothness", np_smoothness)

    return model


def quantile(values):
    return np.percentile(np.asarray(values), [16, 50, 84])


def write_outputs(output_dir, data, samples, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "posterior_samples.npz",
        **{key: np.asarray(value) for key, value in samples.items()},
        tile_ids=data["tile_ids"],
        sca_names=data["sca_names"],
    )
    ar_samples = np.asarray(samples["ar_parameters"])
    if ar_samples.ndim == 3:
        ar_grid_samples = ar_samples[:, None, :, :]
    elif ar_samples.ndim == 4:
        ar_grid_samples = ar_samples
    else:
        raise ValueError("Unexpected AR posterior shape")
    ar_quantiles = np.percentile(ar_grid_samples, [16, 50, 84], axis=0)
    with (output_dir / "tile_ar_posterior.csv").open("w", newline="") as handle:
        fields = [
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
            "ar_thickness_p16",
            "ar_thickness_median",
            "ar_thickness_p84",
            "ar_n_p16",
            "ar_n_median",
            "ar_n_p84",
            "ar_k_p16",
            "ar_k_median",
            "ar_k_p84",
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
                        "ar_thickness_p16": ar_quantiles[0, sca_index, tile_index, 0],
                        "ar_thickness_median": ar_quantiles[1, sca_index, tile_index, 0],
                        "ar_thickness_p84": ar_quantiles[2, sca_index, tile_index, 0],
                        "ar_n_p16": ar_quantiles[0, sca_index, tile_index, 1],
                        "ar_n_median": ar_quantiles[1, sca_index, tile_index, 1],
                        "ar_n_p84": ar_quantiles[2, sca_index, tile_index, 1],
                        "ar_k_p16": ar_quantiles[0, sca_index, tile_index, 2],
                        "ar_k_median": ar_quantiles[1, sca_index, tile_index, 2],
                        "ar_k_p84": ar_quantiles[2, sca_index, tile_index, 2],
                    }
                )

    if "ar_sca_mean" in samples:
        sca_quantiles = np.percentile(
            np.asarray(samples["ar_sca_mean"]), [16, 50, 84], axis=0
        )
        with (output_dir / "sca_ar_posterior.csv").open("w", newline="") as handle:
            fields = [
                "detector",
                "ar_thickness_p16",
                "ar_thickness_median",
                "ar_thickness_p84",
                "ar_n_p16",
                "ar_n_median",
                "ar_n_p84",
                "ar_k_p16",
                "ar_k_median",
                "ar_k_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, detector in enumerate(data["sca_names"]):
                writer.writerow(
                    {
                        "detector": detector,
                        "ar_thickness_p16": sca_quantiles[0, index, 0],
                        "ar_thickness_median": sca_quantiles[1, index, 0],
                        "ar_thickness_p84": sca_quantiles[2, index, 0],
                        "ar_n_p16": sca_quantiles[0, index, 1],
                        "ar_n_median": sca_quantiles[1, index, 1],
                        "ar_n_p84": sca_quantiles[2, index, 1],
                        "ar_k_p16": sca_quantiles[0, index, 2],
                        "ar_k_median": sca_quantiles[1, index, 2],
                        "ar_k_p84": sca_quantiles[2, index, 2],
                    }
                )

    if args.np_modulation:
        node_samples = np.asarray(samples["np_node_multipliers"])
        node_quantiles = np.percentile(node_samples, [16, 50, 84], axis=0)
        with (output_dir / "np_node_posterior.csv").open("w", newline="") as handle:
            fields = [
                "repeat",
                "days",
                "node_multiplier_p16",
                "node_multiplier_median",
                "node_multiplier_p84",
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
                    }
                )

    trajectory_label = (
        "NP-modulated square-root trajectory"
        if args.np_modulation
        else "shared square-root trajectory"
    )
    lines = [
        f"# {scope_label(data['sca_names'])} NumPyro hierarchical TMM fit",
        "",
        f"Tile statistics: `{args.tile_statistics}`.",
        f"SCAs used: `{data['sca_names'].tolist()}`.",
        f"Tiles per SCA: `{len(data['tile_ids'])}`; tile IDs: `{data['tile_ids'].tolist()}`.",
        f"HgCdTe fixed at n/k = `{NOMINAL_HGCDTE.tolist()}`.",
        f"Likelihood pixel weight: `{args.likelihood_weight}`; log-noise sigma: `{args.log_noise_sigma}`.",
        f"Warmup/samples/chains: `{args.warmup}/{args.samples}/{args.chains}`.",
        f"Ice trajectory: `{trajectory_label}`.",
    ]
    if data["n_scas"] > 1:
        lines.extend(
            [
                f"SCA AR sigma prior: `{args.ar_sca_sigma_prior}`; tile AR sigma prior: `{args.ar_tile_sigma_prior}`.",
                "AR parameters use global population shrinkage, then SCA-level offsets, then tile-level deviations.",
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
    lines.extend(
        [
            "",
            "| Quantity | 16th percentile | Median | 84th percentile |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("Initial thickness (nm)", "initial_thickness_nm"),
        ("Mean rate (nm/day)", "mean_rate_nm_day"),
        ("AR mean thickness scale", "ar_mean"),
    ):
        values = np.asarray(samples[key])
        if values.ndim > 1:
            values = values[:, 0]
        values = quantile(values)
        lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if args.np_modulation:
        for label, key in (
            ("NP entropy", "np_entropy"),
            ("NP smoothness", "np_smoothness"),
        ):
            values = quantile(samples[key])
            lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if data["n_scas"] > 1:
        model_note = (
            "This all-SCA fit shares one ice trajectory globally and hierarchically "
            "shrinks tile AR parameters toward an SCA-specific population."
        )
    else:
        model_note = (
            "This is a hierarchical fit. The ice trajectory is shared globally; "
            "all three AR parameters vary by tile under hierarchical shrinkage."
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
        "--max-tiles",
        type=int,
        default=None,
        help="Tiles per SCA to fit; default is all available tiles.",
    )
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--wavelength-stride", type=int, default=10)
    parser.add_argument("--log-noise-sigma", type=float, default=0.005)
    parser.add_argument("--likelihood-weight", type=float, default=1.0e-6)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=8)
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
    parser.add_argument(
        "--np-modulation",
        action="store_true",
        help="Use entropy-regularized repeat-node modulation of the square-root ice trajectory.",
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
    if args.log_noise_sigma <= 0.0 or args.likelihood_weight <= 0.0:
        raise ValueError("Noise sigma and likelihood weight must be positive")
    if args.warmup < 1 or args.samples < 1 or args.chains < 1 or args.max_tree_depth < 1:
        raise ValueError("Warmup, samples, chains, and max tree depth must be positive")
    if args.np_entropy_weight < 0.0 or args.np_smoothness_weight < 0.0:
        raise ValueError("NP regularization weights must be non-negative")
    if args.np_modulation and args.np_max_node_multiplier <= 1.0:
        raise ValueError("np-max-node-multiplier must be greater than one")
    if args.ar_sca_sigma_prior <= 0.0 or args.ar_tile_sigma_prior <= 0.0:
        raise ValueError("AR hierarchy sigma priors must be positive")
    jax, _, numpyro, _, MCMC, NUTS, init_to_value = require_numpyro()
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
    data = load_tile_statistics(statistics_path, args.max_tiles)
    context = load_response_context(args.model_path, args.sed_mode, args.wavelength_stride)
    model = build_model(
        data,
        context,
        args.log_noise_sigma,
        args.likelihood_weight,
        np_modulation=args.np_modulation,
        np_entropy_weight=args.np_entropy_weight,
        np_smoothness_weight=args.np_smoothness_weight,
        np_max_node_multiplier=args.np_max_node_multiplier,
        ar_sca_sigma_prior=args.ar_sca_sigma_prior,
        ar_tile_sigma_prior=args.ar_tile_sigma_prior,
    )
    reference_raw = np.log(
        (AR_REFERENCE - AR_LOWER) / (AR_UPPER - AR_REFERENCE)
    )
    initial_values = {
        "ar_mu_raw": reference_raw,
        "initial_thickness_nm": INITIAL_THICKNESS_INIT_NM,
        "log_trajectory_scale": np.log(TRAJECTORY_SCALE_INIT),
        "acm_step_log_offset": 0.0,
    }
    if data["n_scas"] == 1:
        initial_values.update(
            {
                "ar_sigma_raw": np.full(3, 0.2),
                "ar_z": np.zeros((len(data["tile_ids"]), 3)),
            }
        )
    else:
        initial_values.update(
            {
                "ar_sca_sigma": np.full(3, args.ar_sca_sigma_prior * 0.5),
                "ar_sca_z": np.zeros((data["n_scas"], 3)),
                "ar_tile_sigma": np.full(3, args.ar_tile_sigma_prior * 0.5),
                "ar_tile_z": np.zeros((data["n_scas"], data["n_tiles"], 3)),
            }
        )
    if args.np_modulation:
        initial_values["np_node_raw"] = np.zeros(len(data["repeat_values"]) - 1)
    kernel = NUTS(
        model,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        init_strategy=init_to_value(values=initial_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(args.seed))
    samples = mcmc.get_samples(group_by_chain=False)
    write_outputs(args.output_dir, data, samples, args)
    mcmc.print_summary()
    print(f"Wrote {args.output_dir / 'numpyro_summary.md'}")
    print(f"Wrote {args.output_dir / 'tile_ar_posterior.csv'}")
    if data["n_scas"] > 1:
        print(f"Wrote {args.output_dir / 'sca_ar_posterior.csv'}")
    if args.np_modulation:
        print(f"Wrote {args.output_dir / 'np_node_posterior.csv'}")


if __name__ == "__main__":
    main()
