"""Run isolated one-tile WFI trajectory-regularization experiments.

This script deliberately writes outside the production tile-fit directories.
The default WFI02 experiment fits tile 001 while using the final EMCEE samples
from tile 000 as the initialization and, for the MaxEnt variant, as the rate
prior reference.  The target-tile mode runs a short constant-rate EMCEE warmup
first and uses its tail median as the tile-specific reference.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
import run_wfi_tmm_emcee_warmup as emcee_warmup


ROOT = Path(__file__).resolve().parent
DEFAULT_STATISTICS_DIR = ROOT / "20261008/corrected_statistics_program1063_4h"
DEFAULT_OUTPUT = (
    DEFAULT_STATISTICS_DIR
    / "production/wfi02/regularization_experiment/tile_001"
)
VARIANTS = ("baseline", "curvature_off", "maxent")
PRIOR_MODES = (
    "different_tile_emcee",
    "target_tile_constant_map",
    "target_tile_emcee",
)


def tile_statistics_path(statistics_dir, tile_id):
    return (
        Path(statistics_dir)
        / "production/wfi02/tile_statistics"
        / f"wfi02_tile_log_statistics_512_tile_{tile_id:03d}.csv"
    )


def load_data(path):
    data = production.load_tile_statistics(path)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = np.asarray(context["wavelengths"], dtype=float)
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    return data, context


def _emcee_site_layout(data):
    initial = production.initial_values_for_data(data)
    site_shapes = {}
    site_sizes = {}
    for name in emcee_warmup.SITE_ORDER:
        value = (
            initial["np_log_rate_nodes"][..., 0]
            if name == "log_rate_scale"
            else initial[name]
        )
        site_shapes[name] = np.asarray(value).shape
        site_sizes[name] = int(np.asarray(value).size)
    return site_shapes, site_sizes


def tail_emcee_prior(chain_path, source_data, target_data, tail_steps):
    chain_path = Path(chain_path)
    with np.load(chain_path, allow_pickle=False) as archive:
        chain = np.asarray(archive["chain"], dtype=float)
        archived_fingerprint = str(np.asarray(archive["statistics_fingerprint"]).item())
    source_fingerprint = production.statistics_fingerprint(source_data)
    if archived_fingerprint != source_fingerprint:
        raise ValueError(
            "The EMCEE chain does not match the source tile statistics: "
            f"{archived_fingerprint} != {source_fingerprint}"
        )
    if chain.ndim != 3 or chain.shape[0] < tail_steps:
        raise ValueError("The EMCEE chain is shorter than the requested tail")

    site_shapes, site_sizes = _emcee_site_layout(source_data)
    expected_width = sum(site_sizes.values())
    if chain.shape[-1] != expected_width:
        raise ValueError(
            f"Unexpected EMCEE parameter width {chain.shape[-1]}; expected {expected_width}"
        )
    tail = chain[-tail_steps:].reshape(-1, chain.shape[-1])
    median_theta = np.median(tail, axis=0)
    source_sites = emcee_warmup.unpack_sites(
        median_theta,
        site_shapes,
        site_sizes,
    )

    initial_values = production.initial_values_for_data(target_data)
    for name in (
        "ar_tile_raw",
        "initial_thickness_log_nm",
        "reset_initial_thickness_log_nm",
        "incidence_angle_deg",
        "post_reset_tile_log_offset",
        "absolute_hgcdte_n_log_scale",
        "absolute_hgcdte_k_log_scale",
    ):
        target = np.asarray(initial_values[name])
        source = np.asarray(source_sites[name], dtype=float)
        if source.shape != target.shape:
            source = np.broadcast_to(source, target.shape)
        initial_values[name] = np.asarray(source, dtype=float).copy()

    target_rate_shape = np.asarray(initial_values["np_log_rate_nodes"]).shape
    log_rate_reference = np.broadcast_to(
        np.asarray(source_sites["log_rate_scale"], dtype=float)[..., None],
        target_rate_shape,
    ).copy()
    initial_values["np_log_rate_nodes"] = log_rate_reference.copy()
    return {
        "initial_values": initial_values,
        "log_rate_reference": log_rate_reference,
        "tail_steps": int(tail_steps),
        "tail_sample_count": int(tail.shape[0]),
        "median_log_rate": float(np.median(log_rate_reference)),
        "median_rate_nm_day": float(np.exp(np.median(log_rate_reference))),
        "chain_steps": int(chain.shape[0]),
        "chain_walkers": int(chain.shape[1]),
        "chain_width": int(chain.shape[2]),
        "source_statistics_fingerprint": source_fingerprint,
    }


def run_short_constant_rate_emcee(statistics_path, output_dir, steps):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chain_path = output_dir / "emcee_chain.npz"
    if chain_path.is_file():
        return chain_path
    old_steps = emcee_warmup.EMCEE_STEPS
    try:
        emcee_warmup.EMCEE_STEPS = int(steps)
        emcee_warmup.run_emcee(statistics_path, output_dir)
    finally:
        emcee_warmup.EMCEE_STEPS = old_steps
    return chain_path


def refine_constant_rate_prior(target_statistics, target_data, context, prior, output_dir):
    """Refine a transferred EMCEE state with a target-tile constant-rate MAP."""
    initial_values = dict(prior["initial_values"])
    log_rate_nodes = np.asarray(initial_values.pop("np_log_rate_nodes"), dtype=float)
    initial_values["log_rate_scale"] = log_rate_nodes[..., 0].copy()
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    model = production.build_model(
        target_data,
        context,
        smoothness_weight=0.0,
        constant_rate=True,
    )
    map_values, diagnostics = production.run_least_squares_map_optimization(
        model,
        initial_values,
        jax,
    )
    evaluated = production._evaluate_model(model, map_values)
    refined_log_rate = np.broadcast_to(
        np.asarray(map_values["log_rate_scale"], dtype=float)[..., None],
        log_rate_nodes.shape,
    ).copy()
    final_initial_values = dict(map_values)
    final_initial_values.pop("log_rate_scale")
    final_initial_values["np_log_rate_nodes"] = refined_log_rate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "constant_rate_map.npz",
        **{
            name: np.asarray(value)
            for name, value in {**map_values, **evaluated}.items()
        },
    )
    metadata = {
        "statistics": str(Path(target_statistics).resolve()),
        "constant_rate": True,
        "rate_nm_day": float(np.exp(np.asarray(map_values["log_rate_scale"]).reshape(-1)[0])),
        "diagnostics": {
            key: _json_value(value)
            for key, value in diagnostics.items()
            if key != "objective_history"
        },
        "objective_history": _json_value(diagnostics["objective_history"]),
        "initial_rate_nm_day": float(
            np.asarray(evaluated["tile_initial_rate_nm_day"]).reshape(-1)[0]
        ),
        "final_rate_nm_day": float(
            np.asarray(evaluated["tile_final_rate_nm_day"]).reshape(-1)[0]
        ),
        "mean_rate_nm_day": float(
            np.asarray(evaluated["tile_mean_rate_nm_day"]).reshape(-1)[0]
        ),
    }
    (output_dir / "constant_rate_map.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    prior = dict(prior)
    prior["initial_values"] = final_initial_values
    prior["log_rate_reference"] = refined_log_rate
    prior["median_log_rate"] = float(np.log(metadata["rate_nm_day"]))
    prior["median_rate_nm_day"] = metadata["rate_nm_day"]
    prior["rate_prior_source"] = "target_tile_constant_rate_map"
    prior["constant_rate_map"] = metadata
    return prior


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def fit_variant(
    variant,
    output_dir,
    target_statistics,
    target_data,
    context,
    prior,
    maxent_weight,
):
    if variant == "baseline":
        smoothness_weight = production.NP_SMOOTHNESS_WEIGHT
        entropy_weight = 0.0
        entropy_reference = None
        rate_prior_log_mean = None
    elif variant == "curvature_off":
        smoothness_weight = 0.0
        entropy_weight = 0.0
        entropy_reference = None
        rate_prior_log_mean = None
    elif variant == "maxent":
        smoothness_weight = 0.0
        entropy_weight = maxent_weight
        entropy_reference = prior["log_rate_reference"]
        rate_prior_log_mean = prior["log_rate_reference"]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    model = production.build_model(
        target_data,
        context,
        smoothness_weight=smoothness_weight,
        entropy_reference_log_rates=entropy_reference,
        entropy_weight=entropy_weight,
        rate_prior_log_mean=rate_prior_log_mean,
    )
    map_values, diagnostics = production.run_least_squares_map_optimization(
        model,
        prior["initial_values"],
        jax,
    )
    evaluated = production._evaluate_model(model, map_values)
    observed = np.asarray(target_data["sums"], dtype=float) / np.asarray(
        target_data["counts"], dtype=float
    )
    observed += np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1, 1) * (
        np.asarray(target_data["days"])[None, :]
        >= float(target_data.get("reset_day", production.PRODUCTION_RESET_DAY))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    production.write_outputs(
        output_dir,
        target_data,
        evaluated,
        target_statistics,
        diagnostics,
        {
            "observed": observed,
            "prediction": np.asarray(evaluated["_map_least_squares_prediction"]),
        },
    )
    metadata = {
        "variant": variant,
        "target_statistics": str(Path(target_statistics).resolve()),
        "smoothness_weight": float(smoothness_weight),
        "entropy_weight": float(entropy_weight),
        "entropy_reference": "tail_emcee_median" if entropy_reference is not None else None,
        "rate_prior": prior.get(
            "rate_prior_source",
            "tail_emcee_median" if rate_prior_log_mean is not None else "production",
        ),
        "tail_steps": prior["tail_steps"],
        "tail_sample_count": prior["tail_sample_count"],
        "median_log_rate": prior["median_log_rate"],
        "median_rate_nm_day": prior["median_rate_nm_day"],
        "diagnostics": {
            key: _json_value(value)
            for key, value in diagnostics.items()
            if key != "objective_history"
        },
        "objective_history": _json_value(diagnostics["objective_history"]),
        "initial_rate_nm_day": float(
            np.asarray(evaluated["tile_initial_rate_nm_day"]).reshape(-1)[0]
        ),
        "final_rate_nm_day": float(
            np.asarray(evaluated["tile_final_rate_nm_day"]).reshape(-1)[0]
        ),
        "mean_rate_nm_day": float(
            np.asarray(evaluated["tile_mean_rate_nm_day"]).reshape(-1)[0]
        ),
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS_DIR)
    parser.add_argument("--source-tile", type=int, default=0)
    parser.add_argument("--target-tile", type=int, default=1)
    parser.add_argument("--prior-mode", choices=PRIOR_MODES, default="different_tile_emcee")
    parser.add_argument("--tail-steps", type=int, default=25)
    parser.add_argument("--constant-rate-emcee-steps", type=int, default=200)
    parser.add_argument("--maxent-weight", type=float, default=production.NP_SMOOTHNESS_WEIGHT)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--emcee-chain", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prior_mode == "different_tile_emcee" and args.source_tile == args.target_tile:
        raise ValueError("The EMCEE source and target tiles must be different")
    if args.tail_steps < 1:
        raise ValueError("--tail-steps must be positive")
    if args.constant_rate_emcee_steps < 1:
        raise ValueError("--constant-rate-emcee-steps must be positive")
    if args.maxent_weight < 0.0:
        raise ValueError("--maxent-weight must be non-negative")
    variants = tuple(args.variant) if args.variant else VARIANTS
    target_statistics = tile_statistics_path(args.statistics_dir, args.target_tile)
    if not target_statistics.is_file():
        raise FileNotFoundError(target_statistics)
    target_data, context = load_data(target_statistics)
    if args.prior_mode == "target_tile_emcee":
        source_statistics = target_statistics
        source_tile = args.target_tile
        emcee_output = args.output / "tile_specific_constant_rate_emcee"
        emcee_chain = args.emcee_chain or run_short_constant_rate_emcee(
            target_statistics,
            emcee_output,
            args.constant_rate_emcee_steps,
        )
        source_data = target_data
    else:
        source_statistics = tile_statistics_path(args.statistics_dir, args.source_tile)
        source_tile = args.source_tile
        emcee_chain = args.emcee_chain or (
            Path(args.statistics_dir)
            / "production/wfi02/emcee/emcee_chain.npz"
        )
        for path in (source_statistics, emcee_chain):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_data, _ = load_data(source_statistics)
    prior = tail_emcee_prior(
        emcee_chain,
        source_data,
        target_data,
        args.tail_steps,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prior_mode == "target_tile_constant_map":
        prior = refine_constant_rate_prior(
            target_statistics,
            target_data,
            context,
            prior,
            args.output / "constant_rate_map",
        )
    summary = []
    for variant in variants:
        output_dir = args.output / variant
        summary.append(
            fit_variant(
                variant,
                output_dir,
                target_statistics,
                target_data,
                context,
                prior,
                args.maxent_weight,
            )
        )

    summary_path = args.output / "summary.csv"
    fields = (
        "variant",
        "smoothness_weight",
        "entropy_weight",
        "objective",
        "initial_objective",
        "iterations",
        "gradient_converged",
        "initial_rate_nm_day",
        "final_rate_nm_day",
        "mean_rate_nm_day",
    )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in summary:
            diagnostics = result["diagnostics"]
            writer.writerow(
                {
                    "variant": result["variant"],
                    "smoothness_weight": result["smoothness_weight"],
                    "entropy_weight": result["entropy_weight"],
                    "objective": diagnostics["objective"],
                    "initial_objective": diagnostics["initial_objective"],
                    "iterations": diagnostics["iterations"],
                    "gradient_converged": diagnostics["gradient_converged"],
                    "initial_rate_nm_day": result["initial_rate_nm_day"],
                    "final_rate_nm_day": result["final_rate_nm_day"],
                    "mean_rate_nm_day": result["mean_rate_nm_day"],
                }
            )
    (args.output / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "source_statistics": str(source_statistics.resolve()),
                "target_statistics": str(target_statistics.resolve()),
                "emcee_chain": str(emcee_chain.resolve()),
                "source_tile": source_tile,
                "target_tile": args.target_tile,
                "prior_mode": args.prior_mode,
                "prior_tile": source_tile,
                "tail_steps": args.tail_steps,
                "tail_sample_count": prior["tail_sample_count"],
                "median_log_rate": prior["median_log_rate"],
                "median_rate_nm_day": prior["median_rate_nm_day"],
                "constant_rate_emcee_steps": args.constant_rate_emcee_steps,
                "constant_rate_map": prior.get("constant_rate_map"),
                "maxent_weight": args.maxent_weight,
                "variants": list(variants),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
