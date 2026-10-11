"""Run the fixed single-tile EMCEE basin search used by production."""

import numpy as np

import fit_wfi_tmm_production as production


EMCEE_WALKERS = 52
EMCEE_STEPS = 1000
EMCEE_SEED = 20261004

SITE_ORDER = (
    "ar_tile_raw",
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "incidence_angle_deg",
    "log_rate_scale",
    "post_reset_tile_log_offset",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
)
LINEAR_RATE_SITE_ORDER = (
    "ar_tile_raw",
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "incidence_angle_deg",
    "log_rate_start",
    "log_rate_end",
    "post_reset_tile_log_offset",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
)


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
    *,
    trajectory="constant-rate",
    walkers=None,
    steps=None,
    seed=None,
):
    import emcee

    if walkers is None:
        walkers = EMCEE_WALKERS
    if steps is None:
        steps = EMCEE_STEPS
    if seed is None:
        seed = EMCEE_SEED
    if trajectory not in ("constant-rate", "linear-rate-ramp"):
        raise ValueError(f"Unknown EMCEE trajectory: {trajectory}")
    site_order = (
        SITE_ORDER if trajectory == "constant-rate" else LINEAR_RATE_SITE_ORDER
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)

    data = production.load_tile_statistics(
        statistics_path,
    )
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = np.asarray(
        context["wavelengths"], dtype=float
    )
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    model = production.build_model(data, context)
    production_initial_values = production.initial_values_for_data(data)
    initial_values = {
        name: value
        for name, value in production_initial_values.items()
        if name != "np_log_rate_nodes"
    }
    initial_log_rates = production_initial_values["np_log_rate_nodes"]
    if trajectory == "constant-rate":
        initial_values["log_rate_scale"] = initial_log_rates[..., 0]
    else:
        initial_values["log_rate_start"] = initial_log_rates[..., 0]
        initial_values["log_rate_end"] = initial_log_rates[..., -1]
    missing_sites = sorted(set(site_order).difference(initial_values))
    if missing_sites:
        raise ValueError(f"Production EMCEE sites are missing: {missing_sites}")
    site_shapes = {
        name: np.asarray(initial_values[name]).shape for name in site_order
    }
    site_sizes = {
        name: int(np.asarray(initial_values[name]).size) for name in site_order
    }
    start = pack_sites(initial_values, site_sizes, site_order)
    angle_index = sum(site_sizes[name] for name in site_order[:3])

    from numpyro.infer.util import log_density

    def log_prob_jax(theta):
        values = unpack_sites(theta, site_shapes, site_sizes, site_order)
        if trajectory == "constant-rate":
            log_rate_scale = values.pop("log_rate_scale")
            values["np_log_rate_nodes"] = jax.numpy.broadcast_to(
                log_rate_scale[..., None],
                (
                    data["n_scas"],
                    data["n_tiles"],
                    len(data["node_days"]),
                ),
            )
        else:
            log_rate_start = values.pop("log_rate_start")
            log_rate_end = values.pop("log_rate_end")
            node_fraction = (
                jax.numpy.asarray(data["node_days"])
                - float(data["node_days"][0])
            ) / float(data["node_days"][-1] - data["node_days"][0])
            rate_nodes = (
                jax.numpy.exp(log_rate_start)[..., None] * (1.0 - node_fraction)
                + jax.numpy.exp(log_rate_end)[..., None] * node_fraction
            )
            values["np_log_rate_nodes"] = jax.numpy.log(rate_nodes)
        return log_density(model, (), {}, values)[0]

    log_prob_compiled = jax.jit(log_prob_jax)
    print(
        "Compiled single-tile emcee log posterior at production start: "
        f"{float(log_prob_compiled(jax.numpy.asarray(start))):.6f}",
        flush=True,
    )

    rng = np.random.default_rng(seed)
    ndim = start.size
    jitter_by_site = {
        "ar_tile_raw": 0.25,
        "initial_thickness_log_nm": 0.35,
        "reset_initial_thickness_log_nm": 0.35,
        "incidence_angle_deg": 2.5,
        "log_rate_scale": 0.30,
        "log_rate_start": 0.30,
        "log_rate_end": 0.30,
        "post_reset_tile_log_offset": 0.08,
        "absolute_hgcdte_n_log_scale": 0.25,
        "absolute_hgcdte_k_log_scale": 0.25,
    }
    jitter = np.concatenate(
        [
            np.full(site_sizes[name], jitter_by_site[name], dtype=float)
            for name in site_order
        ]
    )
    walkers_start = start[None, :] + rng.normal(
        size=(walkers, ndim)
    ) * jitter
    walkers_start[:, angle_index] = np.clip(
        walkers_start[:, angle_index],
        1.0e-5,
        production.INCIDENCE_ANGLE_MAX_DEG - 1.0e-5,
    )
    def log_prob(theta):
        theta = np.asarray(theta, dtype=float)
        if (
            not np.all(np.isfinite(theta))
            or theta[angle_index] <= 0.0
            or theta[angle_index] >= production.INCIDENCE_ANGLE_MAX_DEG
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
        sampler.sample(
            walkers_start,
            iterations=steps,
            progress=False,
            skip_initial_state_check=True,
        ),
        start=1,
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
    if trajectory == "constant-rate":
        best_rate_start = np.asarray(best_values.pop("log_rate_scale"))
        best_rate_end = best_rate_start
        best_values["np_log_rate_nodes"] = np.broadcast_to(
            best_rate_start[..., None],
            (
                data["n_scas"],
                data["n_tiles"],
                len(data["node_days"]),
            ),
        ).copy()
    else:
        best_rate_start = np.asarray(best_values.pop("log_rate_start"))
        best_rate_end = np.asarray(best_values.pop("log_rate_end"))
        node_fraction = (
            data["node_days"] - data["node_days"][0]
        ) / (data["node_days"][-1] - data["node_days"][0])
        rate_nodes = (
            np.exp(best_rate_start)[..., None] * (1.0 - node_fraction)
            + np.exp(best_rate_end)[..., None] * node_fraction
        )
        best_values["np_log_rate_nodes"] = np.log(rate_nodes)
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
    np.savez_compressed(
        output_dir / "posterior_samples.npz",
        **warm_sites,
        map_only=np.asarray(1),
        statistics_fingerprint=fingerprint,
        tile_ids=np.asarray(data["tile_ids"], dtype=int),
        sca_names=np.asarray(data["sca_names"]).astype(str),
        statistics_row_fingerprints=production.statistics_row_fingerprints(data),
        input_corrections_already_applied=np.asarray(1),
        absolute_hgcdte_scale_prior_sigma_log=np.asarray(
            production.ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG
        ),
        emcee_trajectory=np.asarray(trajectory),
    )
    summary_lines = [
        [
            f"# WFI one-tile {trajectory} EMCEE basin search",
            "",
            f"- Tile statistics: `{statistics_path.resolve()}`.",
            f"- Walkers/steps/dimensions: `{walkers}/{len(chain)}/{ndim}`.",
            f"- Mean acceptance fraction: `{np.mean(sampler.acceptance_fraction):.6f}`.",
            f"- Maximum log posterior: `{float(log_probability[best_index]):.10g}`.",
            f"- Statistics fingerprint: `{fingerprint}`.",
            f"- HgCdTe n/k log-scale prior sigma: `{production.ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG}`.",
            "",
            "Best-state coordinates:",
            "",
            f"- AR raw: `{np.asarray(best_values['ar_tile_raw']).tolist()}`.",
            "- Initial/reset thickness log nm: "
            f"`{np.asarray(best_values['initial_thickness_log_nm']).tolist()}` / "
            f"`{np.asarray(best_values['reset_initial_thickness_log_nm']).tolist()}`.",
            f"- Incidence angle deg: `{float(np.asarray(best_values['incidence_angle_deg']))}`.",
            f"- Start/end rate nm/day: `{np.exp(best_rate_start).tolist()}` / "
            f"`{np.exp(best_rate_end).tolist()}`.",
            "- Post-reset tile log offset: "
            f"`{np.asarray(best_values['post_reset_tile_log_offset']).tolist()}`.",
            "- HgCdTe n/k log scales: "
            f"`{np.asarray(best_values['absolute_hgcdte_n_log_scale']).tolist()}` / "
            f"`{np.asarray(best_values['absolute_hgcdte_k_log_scale']).tolist()}`.",
            "",
        ]
    ][0]
    summary = "\n".join(summary_lines)
    (output_dir / "emcee_summary.md").write_text(summary)
