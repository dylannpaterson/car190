"""Fit a constant-rate tile with a smooth n(lambda) modulation about 1.25.

This is an isolated physics diagnostic. It does not change the default
production model or reuse a production output directory.
"""

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
        default=production.ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    )
    return parser.parse_args()


def load_constant_rate_initial_values(map_path, data, prior_sigma_log):
    initial_values = production.initial_values_for_data(
        data,
        constant_rate=True,
        fit_ice_n_modulation=True,
    )
    with np.load(map_path, allow_pickle=False) as archived:
        values = {}
        for name, target in initial_values.items():
            if name == "ice_n_log_modulation":
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
        args.prior_sigma_log,
    )
    config = production.ProductionConfig(
        tile_statistics=args.statistics,
        output_dir=args.output_dir,
        map_initial_values=initial_values,
        constant_rate=True,
        fit_ice_n_modulation=True,
        ice_n_modulation_prior_sigma_log=args.prior_sigma_log,
        model_metadata={
            "constant_rate_diagnostic": 1,
            "ice_n_modulation_model": "multiplicative_log_basis_about_1.25",
            "ice_n_modulation_node_wavelengths_nm": production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM,
            "ice_n_modulation_width_nm": production.ICE_N_MODULATION_WIDTH_NM,
            "ice_n_modulation_prior_sigma_log": args.prior_sigma_log,
        },
    )
    map_values, diagnostics = production.fit_one(config)
    rate = float(np.exp(np.asarray(map_values["log_rate_scale"]).reshape(-1)[0]))
    modulation = np.asarray(map_values["ice_n_log_modulation"]).reshape(-1)
    print(f"rate_nm_day={rate:.9g}")
    print(f"objective={diagnostics['objective']:.9g}")
    print(f"ice_n_modulation_log={modulation.tolist()}")
    print(f"ice_n_modulation_factor={np.exp(modulation).tolist()}")


if __name__ == "__main__":
    main()
