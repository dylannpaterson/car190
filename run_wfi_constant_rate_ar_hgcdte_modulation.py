"""Fit constant-rate tiles with broad AR/HgCdTe spectral modulation."""

import argparse
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--constant-rate-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prior-sigma-log",
        type=float,
        default=production.AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    )
    return parser.parse_args()


def load_constant_rate_initial_values(map_path, data):
    initial_values = production.initial_values_for_data(
        data,
        constant_rate=True,
        fit_ar_hgcdte_modulation=True,
    )
    with np.load(map_path, allow_pickle=False) as archived:
        values = {}
        for name, target in initial_values.items():
            if name in (
                "ar_n_log_modulation",
                "hgcdte_n_log_modulation",
                "hgcdte_k_log_modulation",
            ):
                values[name] = np.asarray(target, dtype=float).copy()
                continue
            if name not in archived:
                raise KeyError(f"Constant-rate archive is missing {name}")
            value = np.asarray(archived[name], dtype=float)
            target_array = np.asarray(target)
            while value.ndim > target_array.ndim and value.shape[0] == 1:
                value = value[0]
            if value.shape != target_array.shape:
                raise ValueError(
                    f"Constant-rate archive shape mismatch for {name}: "
                    f"{value.shape} != {target_array.shape}"
                )
            values[name] = value.copy()
    return values


def main():
    args = parse_args()
    if not np.isfinite(args.prior_sigma_log) or args.prior_sigma_log <= 0.0:
        raise ValueError("--prior-sigma-log must be positive and finite")
    data = production.load_tile_statistics(args.statistics)
    initial_values = load_constant_rate_initial_values(
        args.constant_rate_map,
        data,
    )
    config = production.ProductionConfig(
        tile_statistics=args.statistics,
        output_dir=args.output_dir,
        map_initial_values=initial_values,
        constant_rate=True,
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=args.prior_sigma_log,
        model_metadata={
            "constant_rate_diagnostic": 1,
            "ar_hgcdte_modulation_model": "anchored_log_basis",
            "ar_hgcdte_modulation_node_wavelengths_nm": (
                production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM
            ),
            "ar_hgcdte_modulation_width_nm": production.ICE_N_MODULATION_WIDTH_NM,
            "ar_hgcdte_modulation_anchor_nm": production.AR_HGCDTE_MODULATION_ANCHOR_NM,
            "ar_hgcdte_modulation_prior_sigma_log": args.prior_sigma_log,
        },
    )
    map_values, diagnostics = production.fit_one(config)
    rate = float(np.exp(np.asarray(map_values["log_rate_scale"]).reshape(-1)[0]))
    print(f"rate_nm_day={rate:.9g}")
    print(f"objective={diagnostics['objective']:.9g}")
    for name in (
        "ar_n_log_modulation",
        "hgcdte_n_log_modulation",
        "hgcdte_k_log_modulation",
    ):
        modulation = np.asarray(map_values[name]).reshape(-1)
        print(f"{name}={modulation.tolist()}")


if __name__ == "__main__":
    main()
