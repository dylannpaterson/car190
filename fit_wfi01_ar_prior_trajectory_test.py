"""Refit WFI01 with selected AR raw coordinates centered on a target MAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


AR_SITE_INDEX = {
    "ar1_thickness": 0,
    "ar1_n": 1,
    "ar2_thickness": 2,
    "ar2_n": 3,
    "ar3_thickness": 4,
    "ar3_n": 5,
}
PRODUCTION_MODEL_VERSION = (
    "wfi01_ar_prior_free_trajectory_constant_ice_n_normal_incidence_rate12"
)
PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 12.0
PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG = 1.5


def _load_ar_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["ar_tile_raw"], dtype=float).reshape(-1)
    if values.size != len(AR_SITE_INDEX):
        raise ValueError("The AR archive does not contain one tile's six active AR sites")
    if np.any(~np.isfinite(values)):
        raise ValueError("The AR prior center contains non-finite values")
    return values


def _map_initial_values(
    statistics_path,
    baseline_path,
    init_archive_path,
    *,
    constant_rate=False,
    fit_ice_n_constant,
):
    data = production.load_tile_statistics(statistics_path)
    initial_values = production.initial_values_for_data(
        data,
        constant_rate=constant_rate,
        fit_incidence_angle=False,
        fit_ice_n_constant=fit_ice_n_constant,
        fit_ar_hgcdte_modulation=True,
    )
    initial_values = production.load_map_warm_start(
        baseline_path,
        initial_values,
        data,
    )
    with np.load(init_archive_path, allow_pickle=False) as archive:
        for name, target in initial_values.items():
            if name not in archive:
                continue
            value = np.asarray(archive[name], dtype=float)
            if value.size != np.asarray(target).size:
                raise ValueError(
                    f"Initialization archive shape mismatch for {name}: "
                    f"{value.shape} != {np.asarray(target).shape}"
                )
            initial_values[name] = value.reshape(np.asarray(target).shape)
    return initial_values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ar-center-archive", type=Path, required=True)
    parser.add_argument(
        "--map-init-archive",
        type=Path,
        help="Optional fixed-rate archive whose fitted state initializes the free fit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ar-site",
        action="append",
        choices=tuple(AR_SITE_INDEX),
        help="AR coordinate to center on the target archive; repeat for multiple sites.",
    )
    parser.add_argument(
        "--sigma-raw",
        type=float,
        default=0.05,
        help="Prior sigma in the bounded-logit AR coordinate.",
    )
    parser.add_argument(
        "--fix-ice-n",
        action="store_true",
        help=(
            "Fix the constant ice real index at ICE_INDEX_DEFAULT (1.25) "
            "using a hard MAP bound."
        ),
    )
    parser.add_argument(
        "--production-warmup",
        action="store_true",
        help=(
            "Run the production constant-rate MAP warmup from the supplied EMCEE "
            "archive before expanding to the free trajectory."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not np.isfinite(args.sigma_raw) or args.sigma_raw <= 0.0:
        raise ValueError("--sigma-raw must be positive and finite")
    ice_n_prior_sigma_log = 0.01

    data = production.load_tile_statistics(args.statistics)
    rate_prior_log_mean = np.full(
        (
            int(data["n_scas"]),
            int(data["n_tiles"]),
            len(data["node_days"]),
        ),
        np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
    )
    active = production.ar_active_parameter_indices()
    reference_raw = production.ar_reference_raw(
        *production.ar_prior_arrays(), active
    )
    prior_center = np.asarray(reference_raw, dtype=float).copy()
    prior_sigma = np.full(prior_center.shape, production.AR_MEAN_PRIOR_SIGMA)
    ar_sites = sorted(set(args.ar_site or ()))
    target_raw = _load_ar_raw(args.ar_center_archive) if ar_sites else None
    for site in ar_sites:
        index = AR_SITE_INDEX[site]
        prior_center[index] = target_raw[index]
        prior_sigma[index] = args.sigma_raw

    model_metadata = {
        "production_model_version": PRODUCTION_MODEL_VERSION,
        "trajectory_regularization": "none",
        "trajectory_smoothness_weight": 0.0,
        "trajectory_entropy_weight": 0.0,
        "constant_rate_map_warmup": 0,
        "trajectory_rate_prior_mean_nm_day": (
            PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY
        ),
        "trajectory_rate_prior_sigma_log": PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        "ar_prior_site_names": np.asarray(ar_sites),
        "ar_prior_center_raw": prior_center,
        "ar_prior_sigma_raw": prior_sigma,
        "ar_prior_center_source": (
            str(args.ar_center_archive.resolve())
            if ar_sites
            else "nominal AR reference"
        ),
        "fit_ice_n_constant": 1,
        "ice_n_fixed_value": (
            None if not args.fix_ice_n else float(production.ICE_INDEX_DEFAULT)
        ),
    }
    map_initial_values = None
    map_init_from = None
    if args.production_warmup:
        if args.map_init_archive is None:
            raise ValueError("--production-warmup requires --map-init-archive")
        warmup_initial_values = _map_initial_values(
            args.statistics,
            args.baseline,
            args.map_init_archive,
            constant_rate=True,
            fit_ice_n_constant=True,
        )
        if args.fix_ice_n:
            # EMCEE uses a broad HgCdTe-n prior; production uses sigma 0.01.
            # Start this nuisance scale at the production prior mode when ice n
            # is fixed, rather than forcing the optimizer through the mismatch.
            warmup_initial_values["absolute_hgcdte_n_log_scale"] = np.zeros_like(
                warmup_initial_values["absolute_hgcdte_n_log_scale"]
            )
        warmup_config = production.ProductionConfig(
            tile_statistics=args.statistics,
            output_dir=args.output_dir,
            map_initial_values=warmup_initial_values,
            constant_rate=True,
            smoothness_weight=0.0,
            rate_prior_log_mean=rate_prior_log_mean,
            trajectory_rate_prior_sigma_log=PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            absolute_hgcdte_n_scale_prior_sigma_log=0.01,
            fit_incidence_angle=False,
            fit_ice_n_constant=True,
            ice_n_constant_prior_sigma_log=ice_n_prior_sigma_log,
            fit_ar_hgcdte_modulation=True,
            ar_hgcdte_modulation_prior_sigma_log=0.10,
            ar_prior_center_raw=prior_center,
            ar_prior_sigma_raw=prior_sigma,
            fixed_ice_n_log_constant=(0.0 if args.fix_ice_n else None),
        )
        warmup_data, _, warmup_values, warmup_diagnostics, _ = (
            production.fit_map_state(warmup_config)
        )
        rate_reference = np.broadcast_to(
            np.asarray(warmup_values["log_rate_scale"], dtype=float)[..., None],
            (
                int(warmup_data["n_scas"]),
                int(warmup_data["n_tiles"]),
                len(warmup_data["node_days"]),
            ),
        ).copy()
        map_initial_values = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in warmup_values.items()
            if name != "log_rate_scale"
        }
        map_initial_values["np_log_rate_nodes"] = rate_reference
        model_metadata.update(
            {
                "constant_rate_map_warmup": 1,
                "constant_rate_map_rate_nm_day": float(
                    np.exp(np.asarray(warmup_values["log_rate_scale"])).reshape(-1)[0]
                ),
                "constant_rate_map_objective": float(warmup_diagnostics["objective"]),
            }
        )
    elif args.map_init_archive is not None:
        map_initial_values = _map_initial_values(
            args.statistics,
            args.baseline,
            args.map_init_archive,
            constant_rate=False,
            fit_ice_n_constant=True,
        )
        if args.fix_ice_n:
            map_initial_values["absolute_hgcdte_n_log_scale"] = np.zeros_like(
                map_initial_values["absolute_hgcdte_n_log_scale"]
            )
    else:
        map_init_from = args.baseline
    config = production.ProductionConfig(
        tile_statistics=args.statistics,
        output_dir=args.output_dir,
        map_init_from=map_init_from,
        map_initial_values=map_initial_values,
        smoothness_weight=0.0,
        entropy_reference_log_rates=(
            rate_prior_log_mean if args.production_warmup else None
        ),
        rate_prior_log_mean=rate_prior_log_mean,
        trajectory_rate_prior_sigma_log=PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        absolute_hgcdte_n_scale_prior_sigma_log=0.01,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        ice_n_constant_prior_sigma_log=ice_n_prior_sigma_log,
        fixed_ice_n_log_constant=(0.0 if args.fix_ice_n else None),
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=0.10,
        ar_prior_center_raw=prior_center,
        ar_prior_sigma_raw=prior_sigma,
        model_metadata=model_metadata,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ar_prior_settings.json").write_text(
        json.dumps(
            {
                "site_names": ar_sites,
                "sigma_raw": float(args.sigma_raw),
                "center_raw": prior_center.tolist(),
                "sigma_by_site_raw": prior_sigma.tolist(),
                "center_source": (
                    str(args.ar_center_archive.resolve())
                    if ar_sites
                    else "nominal AR reference"
                ),
                "fit_ice_n_constant": True,
                "ice_n_fixed_value": (
                    None
                    if not args.fix_ice_n
                    else float(production.ICE_INDEX_DEFAULT)
                ),
            },
            indent=2,
        )
        + "\n"
    )
    production.fit_one(config)


if __name__ == "__main__":
    main()
