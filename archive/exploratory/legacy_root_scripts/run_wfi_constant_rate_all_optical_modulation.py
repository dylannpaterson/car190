"""Fit a diagnostic with ice/AR/HgCdTe wavelength modulation enabled."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


AR_HGCDTE_MODULATION_NAMES = (
    "ar_n_log_modulation",
    "hgcdte_n_log_modulation",
    "hgcdte_k_log_modulation",
)
ICE_MODULATION_NAMES = ("ice_n_log_modulation", "ice_n_log_constant")
DEFAULT_ICE_MODULATION_PRIOR_SIGMA_LOG = 0.01
DEFAULT_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--constant-rate-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nonparametric-init-dir",
        type=Path,
        default=None,
        help="Use an existing all-optical archive as the nonparametric warm start.",
    )
    parser.add_argument(
        "--rate-reference-dir",
        type=Path,
        default=None,
        help=(
            "Use this archive only for the trajectory-prior/MaxEnt rate reference; "
            "the nonparametric initial values still come from --nonparametric-init-dir."
        ),
    )
    parser.add_argument(
        "--skip-constant-rate",
        action="store_true",
        help="Skip the constant-rate stage when --nonparametric-init-dir is given.",
    )
    parser.add_argument(
        "--zero-ice-modulation-warm-start",
        action="store_true",
        help="Reset ice n modulation to zero after loading a nonparametric warm start.",
    )
    parser.add_argument(
        "--modulation-prior-sigma-log",
        type=float,
        default=None,
        help="Override both wavelength-modulation prior sigmas.",
    )
    parser.add_argument(
        "--ice-modulation-prior-sigma-log",
        type=float,
        default=None,
        help="Override the ice/water n(lambda) modulation prior sigma.",
    )
    parser.add_argument(
        "--constant-ice-n",
        action="store_true",
        help="Fit one positive wavelength-independent ice n scale instead of six n(lambda) nodes.",
    )
    parser.add_argument(
        "--ice-constant-prior-sigma-log",
        type=float,
        default=DEFAULT_ICE_MODULATION_PRIOR_SIGMA_LOG,
        help="Prior sigma for the constant ice n log scale.",
    )
    parser.add_argument(
        "--fit-constant-ice-k",
        action="store_true",
        help="Fit one positive wavelength-independent ice extinction coefficient per tile.",
    )
    parser.add_argument(
        "--ice-k-prior-sigma",
        type=float,
        default=production.ICE_K_PRIOR_SIGMA,
        help="Half-normal prior sigma for the constant ice extinction coefficient.",
    )
    parser.add_argument(
        "--fit-constant-ar-k",
        action="store_true",
        help="Fit one positive wavelength-independent AR extinction coefficient per layer and tile.",
    )
    parser.add_argument(
        "--ar-k-prior-sigma",
        type=float,
        default=production.AR_K_PRIOR_SIGMA,
        help="Half-normal prior sigma for constant AR extinction coefficients.",
    )
    parser.add_argument(
        "--ar-hgcdte-modulation-prior-sigma-log",
        type=float,
        default=None,
        help="Override the AR n plus HgCdTe n/k modulation prior sigma.",
    )
    parser.add_argument(
        "--hgcdte-n-modulation-prior-sigma-log",
        type=float,
        default=None,
        help=(
            "Override only the wavelength-dependent HgCdTe n modulation prior; "
            "AR n and HgCdTe k retain the shared modulation prior."
        ),
    )
    parser.add_argument(
        "--disable-ar-hgcdte-modulation",
        action="store_true",
        help="Fit ice-index modulation only; disable AR/HgCdTe spectral modulation.",
    )
    parser.add_argument(
        "--initial-thickness-prior",
        choices=("lognormal", "half_normal"),
        default="half_normal",
        help="Initial and post-reset thickness prior family.",
    )
    parser.add_argument(
        "--initial-thickness-prior-sigma-nm",
        type=float,
        default=5.0,
        help="Scale in nm for the half-normal thickness prior.",
    )
    parser.add_argument(
        "--fix-incidence-angle",
        action="store_true",
        help="Hold incidence angle at the model's zero-angle value.",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.1,
        help="MaxEnt trajectory weight for the nonparametric fit; use 0 to disable it.",
    )
    parser.add_argument(
        "--trajectory-rate-prior-sigma-log",
        type=float,
        default=production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        help="Log sigma for the non-parametric trajectory rate prior.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.0,
        help="Curvature regularization weight for the non-parametric trajectory.",
    )
    parser.add_argument(
        "--fit-led-log-offsets",
        action="store_true",
        help=(
            "Fit one time-independent post-reference log offset per LED; "
            "diagnostic only because LED baselines are normalized."
        ),
    )
    parser.add_argument(
        "--led-log-offset-prior-sigma-log",
        type=float,
        default=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
        help="Prior sigma for diagnostic LED log offsets.",
    )
    parser.add_argument(
        "--sed-weighting",
        choices=("energy", "photon"),
        default="energy",
        help="Interpret the published SED as energy or photon-rate weighting.",
    )
    parser.add_argument(
        "--qe-response-mode",
        choices=("none", "clean_relative"),
        default="none",
        help="Optional clean-QE relative-response diagnostic mode.",
    )
    parser.add_argument(
        "--absolute-hgcdte-n-scale-prior-sigma-log",
        type=float,
        default=production.ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
        help="Prior sigma for the wavelength-independent HgCdTe n scale.",
    )
    parser.add_argument(
        "--constant-rate-only",
        action="store_true",
        help="Write only the constant-rate diagnostic fit and skip the nonparametric stage.",
    )
    parser.add_argument(
        "--direct-thickness",
        action="store_true",
        help=(
            "Fit direct positive thickness nodes instead of rate nodes; requires "
            "--skip-constant-rate and zero trajectory regularization."
        ),
    )
    return parser.parse_args()


def load_constant_rate_initial_values(
    map_path,
    data,
    initial_thickness_prior,
    initial_thickness_prior_sigma_nm,
    fit_ice_n_modulation,
    fit_ice_n_constant,
    fit_ice_k_constant,
    fit_ar_k_constant,
    fit_incidence_angle,
    fit_ar_hgcdte_modulation,
    fit_led_log_offsets,
):
    initial_values = production.initial_values_for_data(
        data,
        constant_rate=True,
        fit_ice_n_modulation=fit_ice_n_modulation,
        fit_ice_n_constant=fit_ice_n_constant,
        fit_ice_k_constant=fit_ice_k_constant,
        fit_ar_k_constant=fit_ar_k_constant,
        fit_incidence_angle=fit_incidence_angle,
        fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
        fit_led_log_offsets=fit_led_log_offsets,
        initial_thickness_prior=initial_thickness_prior,
        initial_thickness_prior_sigma_nm=initial_thickness_prior_sigma_nm,
    )
    values = production.load_map_warm_start(
        map_path,
        initial_values,
        data,
        require_map_only=False,
    )
    modulation_names = ICE_MODULATION_NAMES + ("ice_k_constant",) + (
        AR_HGCDTE_MODULATION_NAMES if fit_ar_hgcdte_modulation else ()
    )
    for name in modulation_names:
        if name in values:
            values[name] = np.asarray(initial_values[name], dtype=float).copy()
    return values


def modulation_metadata(
    ice_prior_sigma_log,
    ar_hgcdte_prior_sigma_log,
    hgcdte_n_modulation_prior_sigma_log,
    constant_ice_n,
    ice_constant_prior_sigma_log,
    fit_ice_k_constant,
    ice_k_prior_sigma,
    fit_ar_k_constant,
    ar_k_prior_sigma,
    fit_ar_hgcdte_modulation,
    fit_led_log_offsets,
    led_log_offset_prior_sigma_log,
    absolute_hgcdte_n_scale_prior_sigma_log,
):
    metadata = {
        "all_optical_modulation_model": (
            "ice_n_ar_n_hgcdte_n_hgcdte_k"
            if fit_ar_hgcdte_modulation
            else "ice_n_only"
        ),
        "modulated_optical_components": (
            "ice_n,ar_n_all_layers,hgcdte_n,hgcdte_k"
            if fit_ar_hgcdte_modulation
            else "ice_n"
        ),
        "unmodulated_optical_components": (
            "ar_k,ice_k"
            if fit_ar_hgcdte_modulation
            else "ar_n,ar_k,hgcdte_n,hgcdte_k,ice_k"
        ),
        "ice_n_model": "constant" if constant_ice_n else "wavelength_basis",
        "fit_ice_n_constant": int(constant_ice_n),
        "ice_n_constant_prior_sigma_log": ice_constant_prior_sigma_log,
        "fit_ice_k_constant": int(fit_ice_k_constant),
        "ice_k_prior_sigma": ice_k_prior_sigma,
        "fit_ar_k_constant": int(fit_ar_k_constant),
        "ar_k_prior_sigma": ar_k_prior_sigma,
        "fit_ar_hgcdte_modulation": int(fit_ar_hgcdte_modulation),
        "hgcdte_n_modulation_prior_sigma_log": (
            ar_hgcdte_prior_sigma_log
            if hgcdte_n_modulation_prior_sigma_log is None
            else hgcdte_n_modulation_prior_sigma_log
        ),
        "fit_led_log_offsets": int(fit_led_log_offsets),
        "led_log_offset_prior_sigma_log": led_log_offset_prior_sigma_log,
        "absolute_hgcdte_n_scale_prior_sigma_log": (
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
    }
    if constant_ice_n:
        metadata["ice_n_constant_reference"] = production.ICE_INDEX_DEFAULT
    else:
        metadata.update(
            {
                "ice_n_modulation_node_wavelengths_nm": (
                    production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM
                ),
                "ice_n_modulation_width_nm": production.ICE_N_MODULATION_WIDTH_NM,
                "ice_n_modulation_prior_sigma_log": ice_prior_sigma_log,
            }
        )
    if fit_ar_hgcdte_modulation:
        metadata.update(
            {
                "ar_hgcdte_modulation_node_wavelengths_nm": (
                    production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM
                ),
                "ar_hgcdte_modulation_width_nm": production.ICE_N_MODULATION_WIDTH_NM,
                "ar_hgcdte_modulation_anchor_nm": production.AR_HGCDTE_MODULATION_ANCHOR_NM,
                "ar_hgcdte_modulation_prior_sigma_log": ar_hgcdte_prior_sigma_log,
            }
        )
    else:
        metadata["ar_hgcdte_modulation_model"] = "disabled"
    return metadata


def load_nonparametric_initial_values(
    warm_start,
    data,
    initial_thickness_prior,
    initial_thickness_prior_sigma_nm,
    fit_ice_n_modulation,
    fit_ice_n_constant,
    fit_ice_k_constant,
    fit_ar_k_constant,
    fit_incidence_angle,
    fit_ar_hgcdte_modulation,
    fit_led_log_offsets,
    rate_reference_dir=None,
):
    initial_values = production.initial_values_for_data(
        data,
        constant_rate=False,
        fit_ice_n_modulation=fit_ice_n_modulation,
        fit_ice_n_constant=fit_ice_n_constant,
        fit_ice_k_constant=fit_ice_k_constant,
        fit_ar_k_constant=fit_ar_k_constant,
        fit_incidence_angle=fit_incidence_angle,
        fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
        fit_led_log_offsets=fit_led_log_offsets,
        initial_thickness_prior=initial_thickness_prior,
        initial_thickness_prior_sigma_nm=initial_thickness_prior_sigma_nm,
    )
    values = production.load_map_warm_start(warm_start, initial_values, data)
    rate_source = warm_start if rate_reference_dir is None else rate_reference_dir
    with np.load(Path(rate_source) / "posterior_samples.npz", allow_pickle=False) as archived:
        if "np_log_rate_nodes" in archived:
            rate_name = "np_log_rate_nodes"
        else:
            rate_name = "log_rate_scale"
        rate = np.asarray(archived[rate_name], dtype=float)
    target = np.asarray(initial_values["np_log_rate_nodes"])
    while rate.ndim > target.ndim and rate.shape[0] == 1:
        rate = rate[0]
    if rate_name == "np_log_rate_nodes":
        if rate.shape != target.shape:
            raise ValueError(
                f"Nonparametric archive rate shape mismatch: {rate.shape} != "
                f"{target.shape}"
            )
        rate_reference = rate.copy()
    else:
        while rate.ndim > target.ndim - 1 and rate.shape[0] == 1:
            rate = rate[0]
        if rate.shape != target.shape[:-1]:
            raise ValueError(
                f"Constant-rate archive rate shape mismatch: {rate.shape} != "
                f"{target.shape[:-1]}"
            )
        rate_reference = np.broadcast_to(rate[..., None], target.shape).copy()
    if rate_reference_dir is None:
        values["np_log_rate_nodes"] = rate_reference.copy()
    return values, rate_reference


def load_direct_thickness_initial_values(
    warm_start,
    data,
    initial_thickness_prior,
    initial_thickness_prior_sigma_nm,
    fit_ice_n_modulation,
    fit_ice_n_constant,
    fit_ice_k_constant,
    fit_ar_k_constant,
    fit_incidence_angle,
    fit_ar_hgcdte_modulation,
    fit_led_log_offsets,
):
    initial_values = production.initial_values_for_data(
        data,
        trajectory_parameterization="thickness",
        fit_ice_n_modulation=fit_ice_n_modulation,
        fit_ice_n_constant=fit_ice_n_constant,
        fit_ice_k_constant=fit_ice_k_constant,
        fit_ar_k_constant=fit_ar_k_constant,
        fit_incidence_angle=fit_incidence_angle,
        fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
        fit_led_log_offsets=fit_led_log_offsets,
        initial_thickness_prior=initial_thickness_prior,
        initial_thickness_prior_sigma_nm=initial_thickness_prior_sigma_nm,
    )
    values = production.load_map_warm_start(warm_start, initial_values, data)
    with np.load(Path(warm_start) / "posterior_samples.npz", allow_pickle=False) as archived:
        if "node_ice_thickness" in archived:
            thickness = np.asarray(archived["node_ice_thickness"], dtype=float)
        else:
            if "initial_thickness_log_nm" not in archived:
                raise ValueError(
                    "Direct-thickness warm start must contain thickness or initial-thickness sites"
                )
            if "log_rate_scale" in archived:
                log_rate = np.asarray(archived["log_rate_scale"], dtype=float)
            elif "np_log_rate_nodes" in archived:
                log_rate = np.asarray(archived["np_log_rate_nodes"], dtype=float)[..., 0]
            else:
                raise ValueError("Direct-thickness warm start must contain a rate site")
            initial = np.exp(np.asarray(archived["initial_thickness_log_nm"], dtype=float))
            reset_initial = np.exp(
                np.asarray(archived["reset_initial_thickness_log_nm"], dtype=float)
            )
            node_days = np.asarray(data["node_days"], dtype=float)
            reset_day = float(data.get("reset_day", production.PRODUCTION_RESET_DAY))
            thickness = np.where(
                node_days[None, None, :] < reset_day,
                initial[..., None]
                + np.exp(log_rate)[..., None]
                * (node_days[None, None, :] - node_days[0]),
                reset_initial[..., None]
                + np.exp(log_rate)[..., None]
                * (node_days[None, None, :] - reset_day),
            )
    target = np.asarray(initial_values["np_log_thickness_nodes"])
    while thickness.ndim > target.ndim and thickness.shape[0] == 1:
        thickness = thickness[0]
    if thickness.shape != target.shape:
        raise ValueError(
            f"Direct-thickness warm-start shape mismatch: {thickness.shape} != {target.shape}"
        )
    if np.any(~np.isfinite(thickness)) or np.any(thickness <= 0.0):
        raise ValueError("Direct-thickness warm start must be finite and positive")
    values["np_log_thickness_nodes"] = np.log(thickness)
    return values


def main():
    args = parse_args()
    fit_ice_n_modulation = not args.constant_ice_n
    fit_ice_n_constant = args.constant_ice_n
    fit_ice_k_constant = args.fit_constant_ice_k
    fit_ar_k_constant = args.fit_constant_ar_k
    fit_incidence_angle = not args.fix_incidence_angle
    fit_ar_hgcdte_modulation = not args.disable_ar_hgcdte_modulation
    fit_led_log_offsets = args.fit_led_log_offsets
    if args.zero_ice_modulation_warm_start and args.constant_ice_n:
        raise ValueError("--zero-ice-modulation-warm-start requires wavelength-node ice n")
    ice_prior_sigma_log = (
        DEFAULT_ICE_MODULATION_PRIOR_SIGMA_LOG
        if args.ice_modulation_prior_sigma_log is None
        and args.modulation_prior_sigma_log is None
        else (
            args.modulation_prior_sigma_log
            if args.ice_modulation_prior_sigma_log is None
            else args.ice_modulation_prior_sigma_log
        )
    )
    ar_hgcdte_prior_sigma_log = (
        DEFAULT_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG
        if args.ar_hgcdte_modulation_prior_sigma_log is None
        and args.modulation_prior_sigma_log is None
        else (
            args.modulation_prior_sigma_log
            if args.ar_hgcdte_modulation_prior_sigma_log is None
            else args.ar_hgcdte_modulation_prior_sigma_log
        )
    )
    for name, value in (
        ("ice modulation", ice_prior_sigma_log),
        ("AR/HgCdTe modulation", ar_hgcdte_prior_sigma_log),
        (
            "HgCdTe n modulation",
            (
                ar_hgcdte_prior_sigma_log
                if args.hgcdte_n_modulation_prior_sigma_log is None
                else args.hgcdte_n_modulation_prior_sigma_log
            ),
        ),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} prior sigma must be positive and finite")
    if (
        not np.isfinite(args.initial_thickness_prior_sigma_nm)
        or args.initial_thickness_prior_sigma_nm <= 0.0
    ):
        raise ValueError("Initial-thickness prior sigma must be positive and finite")
    if (
        not np.isfinite(args.ice_constant_prior_sigma_log)
        or args.ice_constant_prior_sigma_log <= 0.0
    ):
        raise ValueError("Constant ice n prior sigma must be positive and finite")
    if not np.isfinite(args.ice_k_prior_sigma) or args.ice_k_prior_sigma <= 0.0:
        raise ValueError("Constant ice k prior sigma must be positive and finite")
    if not np.isfinite(args.ar_k_prior_sigma) or args.ar_k_prior_sigma <= 0.0:
        raise ValueError("Constant AR k prior sigma must be positive and finite")
    if not np.isfinite(args.entropy_weight) or args.entropy_weight < 0.0:
        raise ValueError("Entropy weight must be finite and non-negative")
    if (
        not np.isfinite(args.trajectory_rate_prior_sigma_log)
        or args.trajectory_rate_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "Trajectory-rate prior sigma must be positive and finite"
        )
    if not np.isfinite(args.smoothness_weight) or args.smoothness_weight < 0.0:
        raise ValueError("Smoothness weight must be finite and non-negative")
    if (
        not np.isfinite(args.led_log_offset_prior_sigma_log)
        or args.led_log_offset_prior_sigma_log <= 0.0
    ):
        raise ValueError("LED log-offset prior sigma must be positive and finite")
    if (
        not np.isfinite(args.absolute_hgcdte_n_scale_prior_sigma_log)
        or args.absolute_hgcdte_n_scale_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "Absolute HgCdTe n scale prior sigma must be positive and finite"
        )
    if args.direct_thickness:
        if not args.skip_constant_rate or args.nonparametric_init_dir is None:
            raise ValueError(
                "--direct-thickness requires --skip-constant-rate and "
                "--nonparametric-init-dir"
            )
        if args.entropy_weight != 0.0 or args.smoothness_weight != 0.0:
            raise ValueError("--direct-thickness requires zero trajectory regularization")

    data = production.load_tile_statistics(args.statistics)
    common_metadata = modulation_metadata(
        ice_prior_sigma_log,
        ar_hgcdte_prior_sigma_log,
        args.hgcdte_n_modulation_prior_sigma_log,
        fit_ice_n_constant,
        args.ice_constant_prior_sigma_log,
        fit_ice_k_constant,
        args.ice_k_prior_sigma,
        fit_ar_k_constant,
        args.ar_k_prior_sigma,
        fit_ar_hgcdte_modulation,
        fit_led_log_offsets,
        args.led_log_offset_prior_sigma_log,
        args.absolute_hgcdte_n_scale_prior_sigma_log,
    )
    common_metadata.update(
        {
            "initial_thickness_prior": args.initial_thickness_prior,
            "initial_thickness_prior_sigma_nm": args.initial_thickness_prior_sigma_nm,
            "fit_incidence_angle": int(fit_incidence_angle),
            "trajectory_rate_prior_sigma_log": args.trajectory_rate_prior_sigma_log,
            "trajectory_smoothness_weight": args.smoothness_weight,
            "trajectory_parameterization": (
                "direct_thickness" if args.direct_thickness else "rate"
            ),
            "sed_weighting": args.sed_weighting,
            "qe_response_mode": args.qe_response_mode,
        }
    )
    constant_output = args.output_dir / "constant_rate"
    if args.skip_constant_rate:
        if args.nonparametric_init_dir is None:
            raise ValueError(
                "--skip-constant-rate requires --nonparametric-init-dir"
            )
        warm_start = args.nonparametric_init_dir
        rate_reference_dir = args.rate_reference_dir
    else:
        constant_metadata = {
            **common_metadata,
            "constant_rate_diagnostic": 1,
            "trajectory_regularization": "none",
        }
        constant_config = production.ProductionConfig(
            tile_statistics=args.statistics,
            output_dir=constant_output,
            map_initial_values=load_constant_rate_initial_values(
                args.constant_rate_map,
                data,
                args.initial_thickness_prior,
                args.initial_thickness_prior_sigma_nm,
                fit_ice_n_modulation,
                fit_ice_n_constant,
                fit_ice_k_constant,
                fit_ar_k_constant,
                fit_incidence_angle,
                fit_ar_hgcdte_modulation,
                fit_led_log_offsets,
            ),
            constant_rate=True,
            smoothness_weight=args.smoothness_weight,
            trajectory_rate_prior_sigma_log=(
                args.trajectory_rate_prior_sigma_log
            ),
            fit_incidence_angle=fit_incidence_angle,
            initial_thickness_prior=args.initial_thickness_prior,
            initial_thickness_prior_sigma_nm=args.initial_thickness_prior_sigma_nm,
            fit_ice_n_modulation=fit_ice_n_modulation,
            ice_n_modulation_prior_sigma_log=ice_prior_sigma_log,
            fit_ice_n_constant=fit_ice_n_constant,
            ice_n_constant_prior_sigma_log=args.ice_constant_prior_sigma_log,
            fit_ice_k_constant=fit_ice_k_constant,
            ice_k_prior_sigma=args.ice_k_prior_sigma,
            fit_ar_k_constant=fit_ar_k_constant,
            ar_k_prior_sigma=args.ar_k_prior_sigma,
            fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
            ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_prior_sigma_log,
            hgcdte_n_modulation_prior_sigma_log=(
                args.hgcdte_n_modulation_prior_sigma_log
            ),
            fit_led_log_offsets=fit_led_log_offsets,
            led_log_offset_prior_sigma_log=args.led_log_offset_prior_sigma_log,
            sed_weighting=args.sed_weighting,
            qe_response_mode=args.qe_response_mode,
            absolute_hgcdte_n_scale_prior_sigma_log=(
                args.absolute_hgcdte_n_scale_prior_sigma_log
            ),
            model_metadata=constant_metadata,
        )
        production.fit_one(constant_config)
        warm_start = constant_output
        if args.constant_rate_only:
            return

    if args.direct_thickness:
        np_values = load_direct_thickness_initial_values(
            warm_start,
            data,
            args.initial_thickness_prior,
            args.initial_thickness_prior_sigma_nm,
            fit_ice_n_modulation,
            fit_ice_n_constant,
            fit_ice_k_constant,
            fit_ar_k_constant,
            fit_incidence_angle,
            fit_ar_hgcdte_modulation,
            fit_led_log_offsets,
        )
        np_metadata = {
            **common_metadata,
            "trajectory_regularization": "none",
            "trajectory_entropy_weight": 0.0,
            "trajectory_rate_reference": "not applicable",
            "nonparametric_warm_start": str(warm_start),
        }
        np_config = production.ProductionConfig(
            tile_statistics=args.statistics,
            output_dir=args.output_dir,
            map_initial_values=np_values,
            smoothness_weight=0.0,
            entropy_weight=0.0,
            trajectory_parameterization="thickness",
            trajectory_rate_prior_sigma_log=args.trajectory_rate_prior_sigma_log,
            constant_rate=False,
            fit_incidence_angle=fit_incidence_angle,
            fit_ice_n_modulation=fit_ice_n_modulation,
            ice_n_modulation_prior_sigma_log=ice_prior_sigma_log,
            fit_ice_n_constant=fit_ice_n_constant,
            ice_n_constant_prior_sigma_log=args.ice_constant_prior_sigma_log,
            fit_ice_k_constant=fit_ice_k_constant,
            ice_k_prior_sigma=args.ice_k_prior_sigma,
            fit_ar_k_constant=fit_ar_k_constant,
            ar_k_prior_sigma=args.ar_k_prior_sigma,
            fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
            ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_prior_sigma_log,
            hgcdte_n_modulation_prior_sigma_log=(
                args.hgcdte_n_modulation_prior_sigma_log
            ),
            fit_led_log_offsets=fit_led_log_offsets,
            led_log_offset_prior_sigma_log=args.led_log_offset_prior_sigma_log,
            sed_weighting=args.sed_weighting,
            qe_response_mode=args.qe_response_mode,
            absolute_hgcdte_n_scale_prior_sigma_log=(
                args.absolute_hgcdte_n_scale_prior_sigma_log
            ),
            initial_thickness_prior=args.initial_thickness_prior,
            initial_thickness_prior_sigma_nm=args.initial_thickness_prior_sigma_nm,
            model_metadata=np_metadata,
        )
        production.fit_one(np_config)
        return

    np_values, rate_reference = load_nonparametric_initial_values(
        warm_start,
        data,
        args.initial_thickness_prior,
        args.initial_thickness_prior_sigma_nm,
        fit_ice_n_modulation,
        fit_ice_n_constant,
        fit_ice_k_constant,
        fit_ar_k_constant,
        fit_incidence_angle,
        fit_ar_hgcdte_modulation,
        fit_led_log_offsets,
        rate_reference_dir=rate_reference_dir if args.skip_constant_rate else None,
    )
    if args.zero_ice_modulation_warm_start:
        np_values["ice_n_log_modulation"] = np.zeros_like(
            np_values["ice_n_log_modulation"]
        )
    np_metadata = {
        **common_metadata,
        "trajectory_regularization": (
            "maxent"
            if args.entropy_weight
            else ("curvature" if args.smoothness_weight else "none")
        ),
        "trajectory_smoothness_weight": args.smoothness_weight,
        "trajectory_entropy_weight": args.entropy_weight,
        "trajectory_rate_prior_sigma_log": args.trajectory_rate_prior_sigma_log,
        "trajectory_entropy_reference": (
            "constant_rate_diagnostic_warmup" if args.entropy_weight else None
        ),
        "trajectory_rate_reference": str(
            rate_reference_dir if args.skip_constant_rate and rate_reference_dir is not None else warm_start
        ),
        "nonparametric_warm_start": str(warm_start),
    }
    np_config = production.ProductionConfig(
        tile_statistics=args.statistics,
        output_dir=args.output_dir,
        map_initial_values=np_values,
        smoothness_weight=args.smoothness_weight,
        entropy_reference_log_rates=(
            rate_reference if args.entropy_weight else None
        ),
        entropy_weight=args.entropy_weight,
        trajectory_rate_prior_sigma_log=args.trajectory_rate_prior_sigma_log,
        rate_prior_log_mean=rate_reference,
        constant_rate=False,
        fit_incidence_angle=fit_incidence_angle,
        fit_ice_n_modulation=fit_ice_n_modulation,
        ice_n_modulation_prior_sigma_log=ice_prior_sigma_log,
        fit_ice_n_constant=fit_ice_n_constant,
        ice_n_constant_prior_sigma_log=args.ice_constant_prior_sigma_log,
        fit_ice_k_constant=fit_ice_k_constant,
        ice_k_prior_sigma=args.ice_k_prior_sigma,
        fit_ar_k_constant=fit_ar_k_constant,
        ar_k_prior_sigma=args.ar_k_prior_sigma,
        fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
        ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_prior_sigma_log,
        hgcdte_n_modulation_prior_sigma_log=(
            args.hgcdte_n_modulation_prior_sigma_log
        ),
        fit_led_log_offsets=fit_led_log_offsets,
        led_log_offset_prior_sigma_log=args.led_log_offset_prior_sigma_log,
        sed_weighting=args.sed_weighting,
        qe_response_mode=args.qe_response_mode,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            args.absolute_hgcdte_n_scale_prior_sigma_log
        ),
        initial_thickness_prior=args.initial_thickness_prior,
        initial_thickness_prior_sigma_nm=args.initial_thickness_prior_sigma_nm,
        model_metadata=np_metadata,
    )
    production.fit_one(np_config)


if __name__ == "__main__":
    main()
