"""Run the single-tile emcee basin search used before production MAP fits."""

import argparse
import sys
from pathlib import Path

import emcee
import numpy as np

import fit_wfi_tmm_production as production


SITE_ORDER = (
    "ar_mu_raw",
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "incidence_angle_deg",
    "log_trajectory_scale",
    "post_reset_tile_log_offset",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
    "complex_q_parameters",
)


def parse_runner_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--walkers", type=int, default=52)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument(
        "--fix-absolute-hgcdte-reference",
        action="store_true",
        help="Fix the absolute-substrate HgCdTe n/k log scales at zero.",
    )
    parser.add_argument(
        "--absolute-hgcdte-scale-prior-sigma-log",
        type=float,
        default=production.ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        help="Log-scale prior sigma for free absolute-substrate HgCdTe n/k scales.",
    )
    parser.add_argument(
        "--no-complex-q",
        dest="fit_complex_q",
        action="store_false",
        help="Run the fresh basin search without phenomenological complex-q parameters.",
    )
    parser.add_argument(
        "--http-sources",
        action="store_true",
        help="Use HTTP for the pinned public SED and QE sources.",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        help="Directory containing the pinned local SED and QE ECSV files.",
    )
    return parser.parse_args()


def build_production_args(
    statistics_path,
    output_dir,
    fix_absolute_hgcdte_reference=False,
    absolute_hgcdte_scale_prior_sigma_log=None,
    fit_complex_q=True,
):
    production._PRODUCTION_STATISTICS_PATH = statistics_path
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = production.parse_args()
    finally:
        sys.argv = saved_argv
    args.all_sca = False
    args.max_tiles = 1
    args.tile_statistics = statistics_path
    args.output_dir = output_dir
    args.fixed_corrections = None
    args.corrections_already_applied = True
    args.constant_rate_trajectory = True
    args.fully_np_trajectory = False
    args.np_modulation = False
    args.constant_rate_modulation = False
    args.tile_rate_variation = False
    args.tile_rate_trajectory_variation = False
    args.independent_tile_trajectories = False
    args.fit_complex_q = bool(fit_complex_q)
    args.complex_q_initial_parameters = production.COMPLEX_Q_INITIAL.copy()
    args.fixed_absolute_hgcdte_scales = bool(fix_absolute_hgcdte_reference)
    if absolute_hgcdte_scale_prior_sigma_log is not None:
        args.absolute_hgcdte_scale_prior_sigma_log = float(
            absolute_hgcdte_scale_prior_sigma_log
        )
    if args.fixed_absolute_hgcdte_scales:
        args.shared_absolute_hgcdte_scales = False
    args.map_steps = 0
    args.map_only = False
    return args


def pack_sites(values, site_sizes, site_order=SITE_ORDER):
    return np.concatenate(
        [np.asarray(values[name], dtype=float).reshape(-1) for name in site_order]
    )


def unpack_sites(theta, site_shapes, site_sizes, site_order=SITE_ORDER):
    values = {}
    start = 0
    for name in site_order:
        stop = start + site_sizes[name]
        values[name] = theta[start:stop].reshape(site_shapes[name])
        start = stop
    if start != theta.size:
        raise ValueError(f"Unexpected emcee parameter width: {theta.size}")
    return values


def run_emcee(
    statistics_path,
    output_dir,
    walkers,
    steps,
    seed,
    fix_absolute_hgcdte_reference=False,
    absolute_hgcdte_scale_prior_sigma_log=None,
    fit_complex_q=True,
    http_sources=False,
    reference_data_dir=None,
):
    if walkers < 2 or steps < 1:
        raise ValueError("walkers must be at least 2 and steps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_production_args(
        statistics_path,
        output_dir,
        fix_absolute_hgcdte_reference=fix_absolute_hgcdte_reference,
        absolute_hgcdte_scale_prior_sigma_log=absolute_hgcdte_scale_prior_sigma_log,
        fit_complex_q=fit_complex_q,
    )
    if reference_data_dir is not None:
        reference_data_dir = Path(reference_data_dir)
        production.SED_URL = str(
            reference_data_dir / "WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
        )
        production.QE_URL = str(
            reference_data_dir / "WFI_Quantum_efficiency.ecsv"
        )
    elif http_sources:
        production.SED_URL = production.SED_URL.replace("https://", "http://", 1)
        production.QE_URL = production.QE_URL.replace("https://", "http://", 1)
    jax, _, _, _, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)

    data = production.load_tile_statistics(
        statistics_path,
        max_tiles=1,
        min_observation_day=args.min_observation_day,
        additional_acm_step_repeat=args.additional_acm_step_repeat,
        pre_reset_acm_step_repeat=args.pre_reset_acm_step_repeat,
    )
    data["prediction_row_chunk_size"] = production.PREDICTION_ROW_CHUNK_SIZE
    context = production.load_response_context(
        args.model_path,
        args.sed_mode,
        args.wavelength_stride,
        data["sca_names"],
        args.wavelength_step_nm,
        args.ice_optical_constants_source
        if args.use_asw_optical_constants
        else None,
        args.use_hgcdte_optical_constants or args.fit_hgcdte_modulation,
        sed_weighting=args.sed_weighting,
        absolute_substrate_power=args.absolute_substrate_power,
    )
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = np.asarray(
        context["wavelengths"], dtype=float
    )
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    args.likelihood_weight_automatic = args.likelihood_weight is None
    args.likelihood_weight = production.default_likelihood_weight(data["counts"])
    ar_reference, ar_lower, ar_upper = production.ar_prior_arrays(args.ar_layers)
    ar_prior_reference = (
        production.map_start_reference(args, ar_reference, ar_lower, ar_upper)
        if args.ar_prior_from_map_start
        else ar_reference
    )
    incidence_chief_ray_tilts_deg = production.load_incidence_chief_ray_tilts(
        args.incidence_chief_ray_map,
        data,
        (args.incidence_chief_ray_x_deg, args.incidence_chief_ray_y_deg),
    )
    model = production.build_model_for_args(
        data,
        context,
        args,
        ar_prior_reference=ar_prior_reference,
        incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
    )
    initial_values = production.initial_values_for_data(
        data, args, ar_reference, ar_lower, ar_upper
    )
    site_order = tuple(name for name in SITE_ORDER if name in initial_values)
    site_shapes = {
        name: np.asarray(initial_values[name]).shape for name in site_order
    }
    site_sizes = {
        name: int(np.asarray(initial_values[name]).size) for name in site_order
    }
    start = pack_sites(initial_values, site_sizes, site_order)
    angle_index = sum(site_sizes[name] for name in site_order[:3])
    q_index = (
        sum(site_sizes[name] for name in site_order[:-1])
        if args.fit_complex_q
        else None
    )

    from numpyro.infer.util import log_density

    def log_prob_jax(theta):
        values = unpack_sites(theta, site_shapes, site_sizes, site_order)
        return log_density(model, (), {}, values)[0]

    log_prob_compiled = jax.jit(log_prob_jax)
    print(
        "Compiled single-tile emcee log posterior at production start: "
        f"{float(log_prob_compiled(jax.numpy.asarray(start))):.6f}",
        flush=True,
    )

    rng = np.random.default_rng(seed)
    ndim = start.size
    jitter_parts = [
        np.full(6, 0.25),
        [0.25],
        [0.35],
        [2.5],
        [0.30],
        [0.08],
    ]
    if not args.fixed_absolute_hgcdte_scales:
        jitter_parts.extend(([0.25], [0.25]))
    if args.fit_complex_q:
        jitter_parts.append(production.COMPLEX_Q_MAP_SCALES)
    jitter = np.concatenate(jitter_parts)
    walkers_start = start[None, :] + rng.normal(size=(walkers, ndim)) * jitter
    walkers_start[:, angle_index] = np.clip(
        walkers_start[:, angle_index],
        1.0e-5,
        production.INCIDENCE_ANGLE_MAX_DEG - 1.0e-5,
    )
    if args.fit_complex_q:
        walkers_start[:, q_index : q_index + 5] = np.clip(
            walkers_start[:, q_index : q_index + 5],
            production.COMPLEX_Q_LOWER + 1.0e-8,
            production.COMPLEX_Q_UPPER - 1.0e-8,
        )

    def log_prob(theta):
        theta = np.asarray(theta, dtype=float)
        if (
            not np.all(np.isfinite(theta))
            or theta[angle_index] <= 0.0
            or theta[angle_index] >= production.INCIDENCE_ANGLE_MAX_DEG
            or (
                args.fit_complex_q
                and (
                    np.any(theta[q_index : q_index + 5] <= production.COMPLEX_Q_LOWER)
                    or np.any(theta[q_index : q_index + 5] >= production.COMPLEX_Q_UPPER)
                )
            )
        ):
            return -np.inf
        try:
            value = float(log_prob_compiled(jax.numpy.asarray(theta)))
        except Exception:
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    initial_log_probability = np.asarray(
        [log_prob(theta) for theta in walkers_start]
    )
    if not np.all(np.isfinite(initial_log_probability)):
        raise RuntimeError("Non-finite initial emcee walker state")
    sampler = emcee.EnsembleSampler(
        walkers,
        ndim,
        log_prob,
        moves=[
            (emcee.moves.StretchMove(a=2.0), 0.8),
            (emcee.moves.DEMove(gamma0=0.1), 0.2),
        ],
    )
    for step, _ in enumerate(
        sampler.sample(walkers_start, iterations=steps, progress=False), start=1
    ):
        if step % 50 == 0 or step == steps:
            print(
                f"Emcee step {step}/{steps}; "
                f"max_logp={np.max(sampler.get_last_sample().log_prob):.6g}; "
                f"mean_acceptance={np.mean(sampler.acceptance_fraction):.4f}",
                flush=True,
            )

    chain = sampler.get_chain()
    log_probability = sampler.get_log_prob()
    best_index = np.unravel_index(np.argmax(log_probability), log_probability.shape)
    best_values = unpack_sites(
        np.asarray(chain[best_index], dtype=float),
        site_shapes,
        site_sizes,
        site_order,
    )
    fingerprint = production.statistics_fingerprint(data)
    np.savez_compressed(
        output_dir / "emcee_chain.npz",
        chain=chain,
        log_probability=log_probability,
        acceptance_fraction=np.asarray(sampler.acceptance_fraction),
        statistics_fingerprint=fingerprint,
    )
    np.savez_compressed(
        output_dir / "emcee_best_sites.npz",
        **{name: np.asarray(value) for name, value in best_values.items()},
    )
    warm_sites = {
        name: np.asarray(value)[None, ...] for name, value in best_values.items()
    }
    if args.fit_complex_q:
        warm_metadata = {
            "complex_q_parameter_names": np.asarray(
                production.COMPLEX_Q_PARAMETER_NAMES
            )
        }
    else:
        warm_metadata = {}
    np.savez_compressed(
        output_dir / "posterior_samples.npz",
        **warm_sites,
        **warm_metadata,
        map_only=np.asarray(1),
        statistics_fingerprint=fingerprint,
        tile_ids=np.asarray(data["tile_ids"], dtype=int),
        sca_names=np.asarray(data["sca_names"]).astype(str),
        statistics_row_fingerprints=production.statistics_row_fingerprints(data),
        input_corrections_already_applied=np.asarray(1),
        fixed_corrections_applied=np.asarray(0),
        fit_complex_q=np.asarray(int(args.fit_complex_q)),
        fixed_absolute_hgcdte_scales=np.asarray(
            int(args.fixed_absolute_hgcdte_scales)
        ),
        absolute_hgcdte_scale_prior_sigma_log=np.asarray(
            args.absolute_hgcdte_scale_prior_sigma_log
        ),
        shared_absolute_hgcdte_scales=np.asarray(
            int(args.shared_absolute_hgcdte_scales)
        ),
        constant_rate_trajectory=np.asarray(1),
        fully_np_trajectory=np.asarray(0),
    )
    summary_lines = [
        [
            "# WFI new-data one-tile free-constant-rate emcee basin search",
            "",
            f"- Tile statistics: `{statistics_path.resolve()}`.",
            f"- Walkers/steps/dimensions: `{walkers}/{len(chain)}/{ndim}`.",
            f"- Mean acceptance fraction: `{np.mean(sampler.acceptance_fraction):.6f}`.",
            f"- Maximum log posterior: `{float(log_probability[best_index]):.10g}`.",
            f"- Statistics fingerprint: `{fingerprint}`.",
            f"- Fixed HgCdTe reference scales: `{args.fixed_absolute_hgcdte_scales}`.",
            f"- HgCdTe n/k log-scale prior sigma: `{args.absolute_hgcdte_scale_prior_sigma_log}`.",
            "",
            "Best-state coordinates:",
            "",
            f"- AR raw: `{np.asarray(best_values['ar_mu_raw']).tolist()}`.",
            "- Initial/reset thickness log nm: "
            f"`{np.asarray(best_values['initial_thickness_log_nm']).tolist()}` / "
            f"`{np.asarray(best_values['reset_initial_thickness_log_nm']).tolist()}`.",
            f"- Incidence angle deg: `{float(np.asarray(best_values['incidence_angle_deg']))}`.",
            f"- Constant rate log scale: `{float(np.asarray(best_values['log_trajectory_scale']))}`.",
            "- Post-reset tile log offset: "
            f"`{np.asarray(best_values['post_reset_tile_log_offset']).tolist()}`.",
            "- HgCdTe n/k log scales: "
            + (
                "`fixed at zero log scale`."
                if args.fixed_absolute_hgcdte_scales
                else (
                    f"`{np.asarray(best_values['absolute_hgcdte_n_log_scale']).tolist()}` / "
                    f"`{np.asarray(best_values['absolute_hgcdte_k_log_scale']).tolist()}`."
                )
            ),
            "",
        ]
    ][0]
    if args.fit_complex_q:
        summary_lines.insert(
            -1,
            f"- Complex-q parameters: `{np.asarray(best_values['complex_q_parameters']).tolist()}`.",
        )
    else:
        summary_lines.insert(-1, "- Complex-q parameters: `disabled`.")
    summary = "\n".join(summary_lines)
    (output_dir / "emcee_summary.md").write_text(summary)


def main():
    args = parse_runner_args()
    run_emcee(
        args.statistics_path,
        args.output_dir,
        args.walkers,
        args.steps,
        args.seed,
        fix_absolute_hgcdte_reference=args.fix_absolute_hgcdte_reference,
        absolute_hgcdte_scale_prior_sigma_log=(
            args.absolute_hgcdte_scale_prior_sigma_log
        ),
        fit_complex_q=args.fit_complex_q,
        http_sources=args.http_sources,
        reference_data_dir=args.reference_data_dir,
    )


if __name__ == "__main__":
    main()
