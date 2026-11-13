"""Profile WFI01's AR1/AR2 thickness phase at a fixed real-index state.

This is a deliberately frozen diagnostic.  It loads a deterministic WFI01
MAP archive, fixes all real-index values to that archive (including ice n),
and evaluates the production residual at a grid of AR1/AR2 thicknesses.
No trajectory, ice thickness, extinction, offset, or spectral nuisance value
is reoptimized.  The output is therefore a phase map for choosing candidates
for a later reduced refit, not a production fit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 12.0
TRAJECTORY_RATE_PRIOR_SIGMA_LOG = 1.5
ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG = 0.01
AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 0.10
ICE_N_CONSTANT_PRIOR_SIGMA_LOG = 0.01


def _positive_finite(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _positive_integer(value):
    value = int(value)
    if value < 2:
        raise argparse.ArgumentTypeError("value must be at least two")
    return value


def _scalar(values, name):
    return float(np.asarray(values[name], dtype=float).reshape(-1)[0])


def _raw_bounded_coordinate(physical_nm, lower, upper):
    serialized = float(physical_nm) / 100.0
    probability = (serialized - float(lower)) / (float(upper) - float(lower))
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"AR thickness {physical_nm} nm is outside the serialized bounds "
            f"{100.0 * lower} to {100.0 * upper} nm"
        )
    return float(np.log(probability / (1.0 - probability)))


def _grid(minimum, maximum, count, anchors):
    values = np.linspace(float(minimum), float(maximum), int(count))
    values = np.concatenate((values, np.asarray(anchors, dtype=float)))
    return np.unique(np.round(values, decimals=12))


def _load_values(statistics_path, map_path):
    data = production.load_tile_statistics(statistics_path)
    context = production.load_response_context(
        data["sca_names"],
        sed_weighting="energy",
        qe_response_mode="none",
    )
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    initial_values = production.initial_values_for_data(
        data,
        constant_rate=False,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        fit_ar_hgcdte_modulation=True,
    )
    values = production.load_map_warm_start(map_path, initial_values, data)
    return data, context, values


def _build_model(data, context, values, ar_base_n):
    rate_shape = np.asarray(values["np_log_rate_nodes"]).shape
    rate_prior_log_mean = np.full(
        rate_shape,
        np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
        dtype=float,
    )
    return production.build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_prior_log_mean,
        entropy_weight=0.0,
        rate_prior_log_mean=rate_prior_log_mean,
        trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG
        ),
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
        ice_n_constant_reference=production.ICE_INDEX_DEFAULT,
        fixed_ar_base_n=np.asarray(ar_base_n, dtype=float),
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
        hgcdte_n_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    )


def _set_ar12(values, ar1_nm, ar2_nm):
    candidate = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in values.items()
    }
    raw = candidate["ar_tile_raw"]
    raw[..., 0] = _raw_bounded_coordinate(
        ar1_nm,
        production.AR_LOWER[0],
        production.AR_UPPER[0],
    )
    raw[..., 2] = _raw_bounded_coordinate(
        ar2_nm,
        production.AR_LOWER[0],
        production.AR_UPPER[0],
    )
    candidate["ar_tile_raw"] = raw
    return candidate


def _metrics(model, data, values):
    residual = np.asarray(model.production_residuals(values), dtype=float)
    state = model.production_state(values)
    prediction = np.asarray(state["prediction"], dtype=float).reshape(-1)
    counts = np.asarray(data["counts"], dtype=float).reshape(-1)
    observed = (
        np.asarray(data["sums"], dtype=float).reshape(-1)
        / np.maximum(counts, 1.0)
    )
    post = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    observed = observed + _scalar(values, "post_reset_tile_log_offset") * post
    data_residual = prediction - observed
    valid = np.ones(data_residual.size, dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    valid_residual = data_residual[valid]
    valid_counts = counts[valid]
    ar = np.asarray(state["ar_parameters"], dtype=float).reshape(-1, 3, 3)[0]
    return {
        "objective": float(0.5 * np.sum(residual**2)),
        "data_rms_log": float(np.sqrt(np.mean(valid_residual**2))),
        "data_count_weighted_rms_log": float(
            np.sqrt(np.sum(valid_counts * valid_residual**2) / np.sum(valid_counts))
        ),
        "data_max_abs_log": float(np.max(np.abs(valid_residual))),
        "initial_ice_thickness_nm": float(
            np.exp(_scalar(values, "initial_thickness_log_nm"))
        ),
        "reset_initial_ice_thickness_nm": float(
            np.exp(_scalar(values, "reset_initial_thickness_log_nm"))
        ),
        "post_reset_tile_log_offset": _scalar(
            values, "post_reset_tile_log_offset"
        ),
        "initial_rate_nm_day": float(
            np.asarray(state["rates"], dtype=float).reshape(-1)[0]
        ),
        "final_rate_nm_day": float(
            np.asarray(state["rates"], dtype=float).reshape(-1)[-1]
        ),
        "ar1_thickness_nm": float(ar[0, 0] * 100.0),
        "ar2_thickness_nm": float(ar[1, 0] * 100.0),
        "ar3_thickness_nm": float(ar[2, 0] * 100.0),
    }


def _write_outputs(output_dir, rows, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ar12_phase_profile.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "ar12_phase_profile.npz",
        ar1_nm=np.asarray(sorted({row["ar1_requested_nm"] for row in rows})),
        ar2_nm=np.asarray(sorted({row["ar2_requested_nm"] for row in rows})),
        objective=np.asarray(
            [row["objective"] for row in rows], dtype=float
        ),
        data_rms_log=np.asarray(
            [row["data_rms_log"] for row in rows], dtype=float
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ar1-min-nm", type=_positive_finite, default=10.0)
    parser.add_argument("--ar1-max-nm", type=_positive_finite, default=300.0)
    parser.add_argument("--ar2-min-nm", type=_positive_finite, default=50.0)
    parser.add_argument("--ar2-max-nm", type=_positive_finite, default=450.0)
    parser.add_argument("--ar1-count", type=_positive_integer, default=21)
    parser.add_argument("--ar2-count", type=_positive_integer, default=21)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.ar1_max_nm <= args.ar1_min_nm:
        raise ValueError("AR1 thickness bounds must be ordered")
    if args.ar2_max_nm <= args.ar2_min_nm:
        raise ValueError("AR2 thickness bounds must be ordered")

    data, context, values = _load_values(args.statistics, args.map)
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    archive_path = Path(args.map) / "posterior_samples.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        ar_parameters = np.asarray(archive["ar_parameters"], dtype=float).reshape(
            -1, 3, 3
        )[0]
    ar_base_n = ar_parameters[:, 1].copy()
    model = _build_model(data, context, values, ar_base_n)

    current_ar1 = float(ar_parameters[0, 0] * 100.0)
    current_ar2 = float(ar_parameters[1, 0] * 100.0)
    ar1_grid = _grid(
        args.ar1_min_nm,
        args.ar1_max_nm,
        args.ar1_count,
        (current_ar1, 149.23307654134867),
    )
    ar2_grid = _grid(
        args.ar2_min_nm,
        args.ar2_max_nm,
        args.ar2_count,
        (current_ar2, 207.52365243043002),
    )

    rows = []
    for ar1_nm in ar1_grid:
        for ar2_nm in ar2_grid:
            candidate = _set_ar12(values, ar1_nm, ar2_nm)
            metrics = _metrics(model, data, candidate)
            rows.append(
                {
                    "ar1_requested_nm": float(ar1_nm),
                    "ar2_requested_nm": float(ar2_nm),
                    **metrics,
                }
            )

    minimum = min(row["objective"] for row in rows)
    for row in rows:
        row["delta_objective"] = row["objective"] - minimum
    rows.sort(key=lambda row: (row["objective"], row["ar1_requested_nm"], row["ar2_requested_nm"]))

    best = rows[:10]
    metadata = {
        "diagnostic": "wfi01_ar12_fixed_real_index_phase_profile",
        "statistics": str(args.statistics.resolve()),
        "map": str(args.map.resolve()),
        "fixed_real_index_reference": str(archive_path.resolve()),
        "fixed_ice_n": float(production.ICE_INDEX_DEFAULT),
        "fixed_ar_base_n": ar_base_n.tolist(),
        "fixed_parameters": [
            "ice_n_log_constant",
            "ar_base_n",
            "ar_n_log_modulation",
            "absolute_hgcdte_n_log_scale",
            "hgcdte_n_log_modulation",
            "all non-AR1/AR2 MAP parameters",
        ],
        "free_parameters": ["AR1 thickness grid coordinate", "AR2 thickness grid coordinate"],
        "ar3_thickness_nm_fixed": float(ar_parameters[2, 0] * 100.0),
        "grid": {
            "ar1_nm": ar1_grid.tolist(),
            "ar2_nm": ar2_grid.tolist(),
        },
        "baseline_ar1_nm": current_ar1,
        "baseline_ar2_nm": current_ar2,
        "minimum_objective": float(minimum),
        "best_points": best,
    }
    csv_path = _write_outputs(args.output_dir, rows, metadata)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'ar12_phase_profile.npz'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    for row in best[:5]:
        print(
            f"AR1={row['ar1_requested_nm']:.6g} nm, "
            f"AR2={row['ar2_requested_nm']:.6g} nm, "
            f"objective={row['objective']:.9g}, "
            f"delta={row['delta_objective']:.9g}, "
            f"data_rms={row['data_rms_log']:.9g}"
        )


if __name__ == "__main__":
    main()
