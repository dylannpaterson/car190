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
FIXED_RATE_SITE_ORDER = (
    "ar_tile_raw",
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "incidence_angle_deg",
    "post_reset_tile_log_offset",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
)


def reset_aware_site_order(base_site_order, reset_events):
    """Append one thickness/grey site per configured reset, without duplicates."""
    reset_layout = {
        "reset_days": np.arange(len(reset_events), dtype=float),
        "reset_event_names": tuple(event["name"] for event in reset_events),
    }
    thickness_names, offset_names = production.reset_site_names(reset_layout)
    reset_sites = set(thickness_names[1:]) | set(offset_names)
    base = tuple(name for name in base_site_order if name not in reset_sites)
    return (*base, *thickness_names[1:], *offset_names)


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
    warmup_steps=0,
    fixed_rate_nm_day=None,
    third_decon_day=None,
    reset_events=None,
    fixed_incidence_angle_deg=None,
    fit_incidence_angle=True,
    trajectory_rate_prior_mean_nm_day=production.TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
    trajectory_rate_prior_sigma_log=production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    absolute_hgcdte_n_scale_prior_sigma_log=(
        production.ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG
    ),
    fixed_absolute_hgcdte_n_log_scale=None,
    fit_ice_n_constant=False,
    ice_n_constant_prior_sigma_log=production.ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    ice_n_constant_reference=production.ICE_INDEX_DEFAULT,
    fixed_ice_n_log_constant=None,
    fit_ice_k_constant=False,
    ice_k_prior_sigma=production.ICE_K_PRIOR_SIGMA,
    fit_ar_k_constant=False,
    ar_k_prior_sigma=production.AR_K_PRIOR_SIGMA,
):
    import emcee

    if walkers is None:
        walkers = EMCEE_WALKERS
    if steps is None:
        steps = EMCEE_STEPS
    if seed is None:
        seed = EMCEE_SEED
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if fixed_rate_nm_day is not None:
        fixed_rate_nm_day = float(fixed_rate_nm_day)
        if trajectory != "constant-rate":
            raise ValueError("fixed_rate_nm_day requires constant-rate trajectory")
        if not np.isfinite(fixed_rate_nm_day) or fixed_rate_nm_day <= 0.0:
            raise ValueError("fixed_rate_nm_day must be positive and finite")
    if fixed_incidence_angle_deg is not None:
        fixed_incidence_angle_deg = float(fixed_incidence_angle_deg)
        if (
            not np.isfinite(fixed_incidence_angle_deg)
            or fixed_incidence_angle_deg < 0.0
            or fixed_incidence_angle_deg > production.INCIDENCE_ANGLE_MAX_DEG
        ):
            raise ValueError(
                "fixed_incidence_angle_deg must be finite in the production angle range"
            )
        if not fit_incidence_angle:
            raise ValueError(
                "fixed_incidence_angle_deg requires fit_incidence_angle=True"
            )
    if (
        not np.isfinite(trajectory_rate_prior_mean_nm_day)
        or trajectory_rate_prior_mean_nm_day <= 0.0
    ):
        raise ValueError("trajectory_rate_prior_mean_nm_day must be positive and finite")
    if (
        not np.isfinite(trajectory_rate_prior_sigma_log)
        or trajectory_rate_prior_sigma_log <= 0.0
    ):
        raise ValueError("trajectory_rate_prior_sigma_log must be positive and finite")
    if (
        not np.isfinite(absolute_hgcdte_n_scale_prior_sigma_log)
        or absolute_hgcdte_n_scale_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "absolute_hgcdte_n_scale_prior_sigma_log must be positive and finite"
        )
    if fixed_absolute_hgcdte_n_log_scale is not None:
        fixed_absolute_hgcdte_n_log_scale = float(
            fixed_absolute_hgcdte_n_log_scale
        )
        if not np.isfinite(fixed_absolute_hgcdte_n_log_scale):
            raise ValueError("fixed_absolute_hgcdte_n_log_scale must be finite")
    if (
        not np.isfinite(ice_n_constant_prior_sigma_log)
        or ice_n_constant_prior_sigma_log <= 0.0
    ):
        raise ValueError("ice_n_constant_prior_sigma_log must be positive and finite")
    if not np.isfinite(ice_n_constant_reference) or ice_n_constant_reference <= 0.0:
        raise ValueError("ice_n_constant_reference must be positive and finite")
    if fixed_ice_n_log_constant is not None:
        fixed_ice_n_log_constant = float(fixed_ice_n_log_constant)
        if not np.isfinite(fixed_ice_n_log_constant):
            raise ValueError("fixed_ice_n_log_constant must be finite")
        if not fit_ice_n_constant:
            raise ValueError(
                "fixed_ice_n_log_constant requires fit_ice_n_constant=True"
            )
    if not np.isfinite(ice_k_prior_sigma) or ice_k_prior_sigma <= 0.0:
        raise ValueError("ice_k_prior_sigma must be positive and finite")
    if not np.isfinite(ar_k_prior_sigma) or ar_k_prior_sigma <= 0.0:
        raise ValueError("ar_k_prior_sigma must be positive and finite")
    if trajectory not in ("constant-rate", "linear-rate-ramp"):
        raise ValueError(f"Unknown EMCEE trajectory: {trajectory}")
    if trajectory == "constant-rate":
        site_order = (
            FIXED_RATE_SITE_ORDER if fixed_rate_nm_day is not None else SITE_ORDER
        )
    else:
        site_order = LINEAR_RATE_SITE_ORDER
    if reset_events is not None:
        site_order = reset_aware_site_order(site_order, reset_events)
    elif third_decon_day is not None:
        site_order = (
            *site_order,
            "third_decon_initial_thickness_log_nm",
            "post_third_decon_tile_log_offset",
        )
    if fit_ice_k_constant:
        site_order = (*site_order, "ice_k_constant")
    if fit_ar_k_constant:
        site_order = (*site_order, "ar_k_constant")
    if fixed_incidence_angle_deg is not None:
        site_order = tuple(
            name for name in site_order if name != "incidence_angle_deg"
        )
    if not fit_incidence_angle:
        site_order = tuple(
            name for name in site_order if name != "incidence_angle_deg"
        )
    if fixed_absolute_hgcdte_n_log_scale is not None:
        site_order = tuple(
            name for name in site_order if name != "absolute_hgcdte_n_log_scale"
        )
    if fixed_ice_n_log_constant is not None:
        site_order = tuple(
            name for name in site_order if name != "ice_n_log_constant"
        )
    elif fit_ice_n_constant:
        site_order = (*site_order, "ice_n_log_constant")
    output_dir.mkdir(parents=True, exist_ok=True)
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)

    data = production.load_tile_statistics(
        statistics_path,
        third_decon_day=(None if reset_events is not None else third_decon_day),
        reset_events=reset_events,
    )
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = np.asarray(
        context["wavelengths"], dtype=float
    )
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    rate_prior_log_mean = np.full(
        (
            int(data["n_scas"]),
            int(data["n_tiles"]),
            len(data["node_days"]),
        ),
        np.log(trajectory_rate_prior_mean_nm_day),
        dtype=float,
    )
    model = production.build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_weight=0.0,
        rate_prior_log_mean=rate_prior_log_mean,
        trajectory_rate_prior_sigma_log=trajectory_rate_prior_sigma_log,
        constant_rate=(trajectory == "constant-rate"),
        fit_incidence_angle=fit_incidence_angle,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
        fixed_absolute_hgcdte_n_log_scale=fixed_absolute_hgcdte_n_log_scale,
        fit_ice_n_constant=fit_ice_n_constant,
        ice_n_constant_prior_sigma_log=ice_n_constant_prior_sigma_log,
        ice_n_constant_reference=ice_n_constant_reference,
        fit_ice_k_constant=fit_ice_k_constant,
        ice_k_prior_sigma=ice_k_prior_sigma,
        fit_ar_k_constant=fit_ar_k_constant,
        ar_k_prior_sigma=ar_k_prior_sigma,
    )
    production_initial_values = production.initial_values_for_data(
        data,
        constant_rate=(trajectory == "constant-rate"),
        fit_incidence_angle=fit_incidence_angle,
        fixed_absolute_hgcdte_n_log_scale=fixed_absolute_hgcdte_n_log_scale,
        fit_ice_n_constant=fit_ice_n_constant,
        fit_ice_k_constant=fit_ice_k_constant,
        fit_ar_k_constant=fit_ar_k_constant,
    )
    initial_values = {
        name: value
        for name, value in production_initial_values.items()
        if name != "np_log_rate_nodes"
    }
    incidence_angle_shape = (
        np.asarray(production_initial_values["incidence_angle_deg"]).shape
        if "incidence_angle_deg" in production_initial_values
        else None
    )
    if fixed_incidence_angle_deg is not None:
        initial_values.pop("incidence_angle_deg")
    if fixed_absolute_hgcdte_n_log_scale is not None:
        initial_values.pop("absolute_hgcdte_n_log_scale", None)
    if fixed_ice_n_log_constant is not None:
        initial_values.pop("ice_n_log_constant", None)
    if trajectory == "constant-rate":
        if fixed_rate_nm_day is not None:
            initial_values.pop("log_rate_scale", None)
    else:
        initial_log_rates = production_initial_values["np_log_rate_nodes"]
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
    parameter_width = start.size
    angle_index = (
        None
        if "incidence_angle_deg" not in site_order
        else sum(
            site_sizes[name]
            for name in site_order[: site_order.index("incidence_angle_deg")]
        )
    )

    from numpyro.infer.util import log_density

    def log_prob_jax(theta):
        values = unpack_sites(theta, site_shapes, site_sizes, site_order)
        if fixed_incidence_angle_deg is not None:
            values["incidence_angle_deg"] = jax.numpy.full(
                incidence_angle_shape,
                fixed_incidence_angle_deg,
            )
        if fixed_ice_n_log_constant is not None:
            values["ice_n_log_constant"] = jax.numpy.full(
                np.asarray(production_initial_values["ice_n_log_constant"]).shape,
                fixed_ice_n_log_constant,
            )
        if trajectory == "constant-rate":
            if fixed_rate_nm_day is None:
                pass
            else:
                values["log_rate_scale"] = jax.numpy.full(
                    (data["n_scas"], data["n_tiles"]),
                    np.log(fixed_rate_nm_day),
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
    log_prob_batch_compiled = jax.jit(jax.vmap(log_prob_jax))

    # Preserve the established walker initializer while also seeding emcee's
    # separate proposal RNG.  Passing the seed only to default_rng leaves the
    # EnsembleSampler proposal sequence random.
    rng = np.random.default_rng(seed)
    sampler_rng = np.random.RandomState(seed)
    ndim = start.size
    jitter_by_site = {
        "ar_tile_raw": 0.25,
        "initial_thickness_log_nm": 0.35,
        "reset_initial_thickness_log_nm": 0.35,
        "third_decon_initial_thickness_log_nm": 0.35,
        "incidence_angle_deg": 2.5,
        "log_rate_scale": 0.30,
        "log_rate_start": 0.30,
        "log_rate_end": 0.30,
        "post_reset_tile_log_offset": 0.08,
        "post_third_decon_tile_log_offset": 0.08,
        "absolute_hgcdte_n_log_scale": absolute_hgcdte_n_scale_prior_sigma_log,
        "absolute_hgcdte_k_log_scale": 0.25,
        "ice_n_log_constant": ice_n_constant_prior_sigma_log,
        "ice_k_constant": 0.005,
        "ar_k_constant": 0.005,
    }
    jitter = np.concatenate(
        [
            np.full(
                site_sizes[name],
                jitter_by_site.get(
                    name,
                    0.35
                    if "initial_thickness" in name
                    else 0.08
                    if name.startswith("post_reset_event_")
                    else 0.25,
                ),
                dtype=float,
            )
            for name in site_order
        ]
    )
    walkers_start = start[None, :] + rng.normal(
        size=(walkers, ndim)
    ) * jitter
    if fit_ice_k_constant:
        ice_k_start = sum(
            site_sizes[name] for name in site_order[: site_order.index("ice_k_constant")]
        )
        ice_k_stop = ice_k_start + site_sizes["ice_k_constant"]
        walkers_start[:, ice_k_start:ice_k_stop] = np.maximum(
            np.abs(walkers_start[:, ice_k_start:ice_k_stop]),
            1.0e-8,
        )
    if fit_ar_k_constant:
        ar_k_start = sum(
            site_sizes[name] for name in site_order[: site_order.index("ar_k_constant")]
        )
        ar_k_stop = ar_k_start + site_sizes["ar_k_constant"]
        walkers_start[:, ar_k_start:ar_k_stop] = np.maximum(
            np.abs(walkers_start[:, ar_k_start:ar_k_stop]),
            1.0e-8,
        )
    if angle_index is not None:
        walkers_start[:, angle_index] = np.clip(
            walkers_start[:, angle_index],
            1.0e-5,
            production.INCIDENCE_ANGLE_MAX_DEG - 1.0e-5,
        )
    def log_prob_vectorized(thetas):
        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim != 2 or thetas.shape[1] != parameter_width:
            raise ValueError(
                f"Expected EMCEE coordinates with shape (n, {parameter_width}), "
                f"got {thetas.shape}"
            )
        valid = np.all(np.isfinite(thetas), axis=1)
        if angle_index is not None:
            valid &= (
                (thetas[:, angle_index] > 0.0)
                & (thetas[:, angle_index] < production.INCIDENCE_ANGLE_MAX_DEG)
            )
        safe_thetas = np.where(valid[:, None], thetas, start[None, :])
        try:
            values = np.asarray(
                log_prob_batch_compiled(jax.numpy.asarray(safe_thetas)),
                dtype=float,
            )
        except Exception:
            return np.full(thetas.shape[0], -np.inf, dtype=float)
        return np.where(valid & np.isfinite(values), values, -np.inf)

    initial_log_probability = log_prob_vectorized(walkers_start)
    if not np.all(np.isfinite(initial_log_probability)):
        raise RuntimeError("Non-finite initial emcee walker state")
    sampler = emcee.EnsembleSampler(
        walkers,
        parameter_width,
        log_prob_vectorized,
        moves=[
            (emcee.moves.StretchMove(a=2.0), 0.8),
            (emcee.moves.DEMove(gamma0=0.1), 0.2),
        ],
        vectorize=True,
    )
    sampler.random_state = sampler_rng.get_state()
    warmup_chain = None
    warmup_log_probability = None
    warmup_acceptance = None
    production_start = walkers_start
    if warmup_steps:
        for step, _ in enumerate(
            sampler.sample(
                walkers_start,
                iterations=warmup_steps,
                progress=False,
                skip_initial_state_check=True,
            ),
            start=1,
        ):
            if step % 50 == 0 or step == warmup_steps:
                print(
                    f"Emcee warmup step {step}/{warmup_steps}; "
                    f"max_logp={np.max(sampler.get_last_sample().log_prob):.6g}; "
                    f"mean_acceptance={np.mean(sampler.acceptance_fraction):.4f}",
                    flush=True,
                )
        production_start = sampler.get_last_sample()
        warmup_chain = sampler.get_chain().copy()
        warmup_log_probability = sampler.get_log_prob().copy()
        warmup_acceptance = np.asarray(sampler.acceptance_fraction).copy()
        sampler.reset()

    for step, _ in enumerate(
        sampler.sample(
            production_start,
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
    if fixed_incidence_angle_deg is not None:
        best_values["incidence_angle_deg"] = np.full(
            incidence_angle_shape,
            fixed_incidence_angle_deg,
        )
    if fixed_ice_n_log_constant is not None:
        best_values["ice_n_log_constant"] = np.full(
            np.asarray(production_initial_values["ice_n_log_constant"]).shape,
            fixed_ice_n_log_constant,
        )
    if trajectory == "constant-rate":
        if fixed_rate_nm_day is None:
            best_rate_start = np.asarray(best_values.pop("log_rate_scale"))
        else:
            best_rate_start = np.asarray(np.log(fixed_rate_nm_day))
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
    reset_days = production.reset_days_for_data(data)
    reset_event_names = production._reset_event_names(data, reset_days.size)
    thickness_names, offset_names = production.reset_site_names(data)
    third_event_day = next(
        (
            float(day)
            for name, day in zip(reset_event_names, reset_days, strict=True)
            if name == "third_decon"
        ),
        np.nan,
    )
    if warmup_chain is not None:
        np.savez_compressed(
            output_dir / "emcee_warmup_chain.npz",
            chain=warmup_chain,
            log_probability=warmup_log_probability,
            acceptance_fraction=warmup_acceptance,
            statistics_fingerprint=fingerprint,
            warmup_steps=np.asarray(warmup_steps),
            fixed_rate_nm_day=np.asarray(
                np.nan if fixed_rate_nm_day is None else fixed_rate_nm_day
            ),
            fixed_incidence_angle_deg=np.asarray(
                np.nan
                if fixed_incidence_angle_deg is None
                else fixed_incidence_angle_deg
            ),
        )
    np.savez_compressed(
        output_dir / "emcee_chain.npz",
        chain=chain,
        log_probability=log_probability,
        acceptance_fraction=np.asarray(sampler.acceptance_fraction),
        statistics_fingerprint=fingerprint,
        warmup_steps=np.asarray(warmup_steps),
        fixed_rate_nm_day=np.asarray(
            np.nan if fixed_rate_nm_day is None else fixed_rate_nm_day
        ),
        fixed_incidence_angle_deg=np.asarray(
            np.nan
            if fixed_incidence_angle_deg is None
            else fixed_incidence_angle_deg
        ),
        third_decon_day=np.asarray(third_event_day),
        reset_event_days=np.asarray(reset_days, dtype=np.float64),
        reset_event_names=np.asarray(reset_event_names, dtype="U64"),
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
        absolute_hgcdte_n_scale_prior_sigma_log=np.asarray(
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
        trajectory_rate_prior_mean_nm_day=np.asarray(
            trajectory_rate_prior_mean_nm_day
        ),
        trajectory_rate_prior_sigma_log=np.asarray(trajectory_rate_prior_sigma_log),
        fit_incidence_angle=np.asarray(int(fit_incidence_angle)),
        fit_ar_hgcdte_modulation=np.asarray(0),
        fit_ice_n_constant=np.asarray(int(fit_ice_n_constant)),
        ice_n_constant_prior_sigma_log=np.asarray(ice_n_constant_prior_sigma_log),
        ice_n_constant_reference=np.asarray(ice_n_constant_reference),
        ice_n_fixed_log_constant=np.asarray(
            np.nan
            if fixed_ice_n_log_constant is None
            else fixed_ice_n_log_constant
        ),
        ice_n_fixed_value=np.asarray(
            np.nan
            if fixed_ice_n_log_constant is None
            else production.ICE_INDEX_DEFAULT * np.exp(fixed_ice_n_log_constant)
        ),
        emcee_trajectory=np.asarray(trajectory),
        warmup_steps=np.asarray(warmup_steps),
        fixed_rate_nm_day=np.asarray(
            np.nan if fixed_rate_nm_day is None else fixed_rate_nm_day
        ),
        fixed_incidence_angle_deg=np.asarray(
            np.nan
            if fixed_incidence_angle_deg is None
            else fixed_incidence_angle_deg
        ),
        ice_reset_day=np.asarray(float(reset_days[0]), dtype=np.float64),
        third_decon_day=np.asarray(third_event_day, dtype=np.float64),
        reset_event_days=np.asarray(reset_days, dtype=np.float64),
        reset_event_names=np.asarray(reset_event_names, dtype="U64"),
        reset_event_thickness_site_names=np.asarray(thickness_names[1:], dtype="U96"),
        reset_event_offset_site_names=np.asarray(offset_names, dtype="U96"),
        reset_event_initial_thickness_log_nm=np.stack(
            [np.asarray(best_values[name], dtype=float).reshape(-1) for name in thickness_names[1:]],
            axis=-1,
        ),
        reset_event_tile_log_offsets=np.stack(
            [np.asarray(best_values[name], dtype=float).reshape(-1) for name in offset_names],
            axis=-1,
        ),
    )
    summary_lines = [
        [
            f"# WFI one-tile {trajectory} EMCEE basin search",
            "",
            f"- Tile statistics: `{statistics_path.resolve()}`.",
            f"- Warmup steps discarded from the saved production chain: `{warmup_steps}`.",
            (
                f"- Constant rate fixed at `{fixed_rate_nm_day:.12g}` nm/day."
                if fixed_rate_nm_day is not None
                else "- Constant rate was free in the EMCEE state."
            ),
            (
                f"- Incidence angle fixed at `{fixed_incidence_angle_deg:.12g}` deg."
                if fixed_incidence_angle_deg is not None
                else "- Incidence angle was free in the EMCEE state."
                if fit_incidence_angle
                else "- Incidence angle was fixed at normal incidence."
            ),
            f"- Walkers/steps/dimensions: `{walkers}/{len(chain)}/{ndim}`.",
            f"- Mean acceptance fraction: `{np.mean(sampler.acceptance_fraction):.6f}`.",
            f"- Maximum log posterior: `{float(log_probability[best_index]):.10g}`.",
            f"- Statistics fingerprint: `{fingerprint}`.",
            f"- HgCdTe n/k log-scale prior sigma: `{absolute_hgcdte_n_scale_prior_sigma_log}` / `{production.ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG}`.",
            (
                f"- Constant ice n fixed at `{production.ICE_INDEX_DEFAULT * np.exp(fixed_ice_n_log_constant):.12g}`."
                if fixed_ice_n_log_constant is not None
                else "- Constant ice n was not fitted in this EMCEE run."
            ),
            f"- Constant ice k fitted: `{fit_ice_k_constant}` with half-normal prior sigma `{ice_k_prior_sigma}`.",
            f"- Constant AR k fitted: `{fit_ar_k_constant}` with half-normal prior sigma `{ar_k_prior_sigma}`.",
            "",
            "Best-state coordinates:",
            "",
            f"- AR raw: `{np.asarray(best_values['ar_tile_raw']).tolist()}`.",
            "- Initial thickness log nm: "
            f"`{np.asarray(best_values['initial_thickness_log_nm']).tolist()}`.",
            f"- Reset-event names: `{list(reset_event_names)}`.",
            "- Reset-event initial thickness log nm: "
            f"`{[np.asarray(best_values[name]).tolist() for name in thickness_names[1:]]}`.",
            "- Reset-event tile log offsets: "
            f"`{[np.asarray(best_values[name]).tolist() for name in offset_names]}`.",
            (
                f"- Incidence angle deg: `{float(np.asarray(best_values['incidence_angle_deg']))}`."
                if "incidence_angle_deg" in best_values
                else "- Incidence angle fixed at normal incidence."
            ),
            f"- Start/end rate nm/day: `{np.exp(best_rate_start).tolist()}` / "
            f"`{np.exp(best_rate_end).tolist()}`.",
            "- HgCdTe n/k log scales: "
            f"`{np.asarray(best_values['absolute_hgcdte_n_log_scale']).tolist()}` / "
            f"`{np.asarray(best_values['absolute_hgcdte_k_log_scale']).tolist()}`.",
            (
                f"- Constant ice k: `{np.asarray(best_values['ice_k_constant']).tolist()}`."
                if fit_ice_k_constant
                else ""
            ),
            (
                f"- Constant AR k: `{np.asarray(best_values['ar_k_constant']).tolist()}`."
                if fit_ar_k_constant
                else ""
            ),
            "",
        ]
    ][0]
    summary = "\n".join(summary_lines)
    (output_dir / "emcee_summary.md").write_text(summary)
