"""Refit a fixed WFI01 AR1/AR2 candidate with only ice evolution free.

This follows ``profile_wfi01_ar12_fixed_real_index_phase.py``.  The candidate
AR1/AR2 thicknesses and every optical/index parameter remain fixed; only the
two initial ice thicknesses, the non-parametric rate nodes, and the reset
offset are reoptimized.  It tests whether a phase-map candidate can recover
through a different ice trajectory without opening the full nuisance model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
from profile_wfi01_ar12_fixed_real_index_phase import (
    _build_model,
    _load_values,
    _metrics,
    _set_ar12,
)


FREE_SITES = (
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "post_reset_tile_log_offset",
    "np_log_rate_nodes",
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ar1-nm", type=float, required=True)
    parser.add_argument("--ar2-nm", type=float, required=True)
    parser.add_argument(
        "--free-init-map",
        type=Path,
        default=None,
        help="Optional archive used only to initialize the free ice sites.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not np.isfinite(args.ar1_nm) or args.ar1_nm <= 0.0:
        raise ValueError("--ar1-nm must be positive and finite")
    if not np.isfinite(args.ar2_nm) or args.ar2_nm <= 0.0:
        raise ValueError("--ar2-nm must be positive and finite")

    data, context, values = _load_values(args.statistics, args.map)
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    with np.load(Path(args.map) / "posterior_samples.npz", allow_pickle=False) as archive:
        ar_parameters = np.asarray(archive["ar_parameters"], dtype=float).reshape(
            -1, 3, 3
        )[0]
    model = _build_model(data, context, values, ar_parameters[:, 1])
    candidate = _set_ar12(values, args.ar1_nm, args.ar2_nm)
    if args.free_init_map is not None:
        with np.load(args.free_init_map, allow_pickle=False) as archive:
            for name in FREE_SITES:
                if name not in archive.files:
                    raise ValueError(
                        f"{args.free_init_map} is missing free initialization site {name}"
                    )
                value = np.asarray(archive[name], dtype=float)
                target_shape = np.asarray(candidate[name]).shape
                while value.ndim > len(target_shape) and value.shape[0] == 1:
                    value = value[0]
                if value.shape != target_shape:
                    raise ValueError(
                        f"{args.free_init_map} site {name} has shape {value.shape}; "
                        f"expected {target_shape}"
                    )
                candidate[name] = value.copy()
    fixed_parameter_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in candidate.items()
        if name not in FREE_SITES
    }
    initial_metrics = _metrics(model, data, candidate)
    optimized, diagnostics = production.run_least_squares_map_optimization(
        model,
        candidate,
        jax,
        trajectory_rate_prior_sigma_log=1.5,
        initial_thickness_prior_sigma_nm=5.0,
        absolute_hgcdte_n_scale_prior_sigma_log=0.01,
        ice_n_constant_prior_sigma_log=0.01,
        ar_hgcdte_modulation_prior_sigma_log=0.10,
        hgcdte_n_modulation_prior_sigma_log=0.10,
        fixed_parameter_values=fixed_parameter_values,
    )
    final_metrics = _metrics(model, data, optimized)
    state = model.production_state(optimized)
    final_ar = np.asarray(state["ar_parameters"], dtype=float).reshape(-1, 3, 3)[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "candidate_fit.npz",
        **{name: np.asarray(value) for name, value in optimized.items()},
        **{name: np.asarray(value) for name, value in state.items()},
    )
    payload = {
        "diagnostic": "wfi01_fixed_ar12_real_index_trajectory_refit",
        "statistics": str(args.statistics.resolve()),
        "map": str(args.map.resolve()),
        "ar1_nm": float(args.ar1_nm),
        "ar2_nm": float(args.ar2_nm),
        "free_initialization_map": (
            None
            if args.free_init_map is None
            else str(args.free_init_map.resolve())
        ),
        "fixed_sites": sorted(fixed_parameter_values),
        "free_sites": list(FREE_SITES),
        "initial_candidate_metrics": initial_metrics,
        "forced_fit_metrics": final_metrics,
        "optimizer": diagnostics,
        "final_ar_parameters": final_ar.tolist(),
        "outputs": {
            "archive": str((args.output_dir / "candidate_fit.npz").resolve()),
            "summary": str((args.output_dir / "summary.json").resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'candidate_fit.npz'}")
    print(
        "Initial candidate objective: "
        f"{initial_metrics['objective']:.9g}; "
        "refit objective: "
        f"{final_metrics['objective']:.9g}; "
        "refit data RMS: "
        f"{final_metrics['data_rms_log']:.9g}"
    )


if __name__ == "__main__":
    main()
