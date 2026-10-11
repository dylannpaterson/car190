"""Test AR initialization and prior-width sensitivity for constant-rate tile 000."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production
from run_wfi_constant_rate_ice_modulation import (
    load_constant_rate_initial_values,
)


DETECTORS = ("wfi01", "wfi02")
CASES = (
    ("nominal_sigma10", 10.0, "nominal"),
    ("map_sigma10", 10.0, "map"),
    ("map_sigma1", 1.0, "map"),
    ("map_sigma025", 0.25, "map"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--constant-rate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def lag_correlation(values):
    values = np.asarray(values, dtype=float)
    if values.size < 3 or np.std(values[:-1]) == 0.0 or np.std(values[1:]) == 0.0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def make_initial_values(data, map_path, source):
    if source == "nominal":
        return production.initial_values_for_data(data, constant_rate=True)
    values = load_constant_rate_initial_values(
        map_path,
        data,
        production.ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    )
    values.pop("ice_n_log_modulation")
    return values


def fit_case(detector, statistics_path, map_path, output_dir, case_name, prior_sigma, source):
    data = production.load_tile_statistics(statistics_path)
    initial_values = make_initial_values(data, map_path, source)
    config = production.ProductionConfig(
        tile_statistics=statistics_path,
        output_dir=output_dir / detector / case_name,
        map_initial_values=initial_values,
        constant_rate=True,
        ar_mean_prior_sigma=prior_sigma,
        model_metadata={
            "ar_prior_sensitivity_diagnostic": 1,
            "ar_mean_prior_sigma": prior_sigma,
            "ar_initialization": source,
        },
    )
    _, _, map_values, diagnostics, evaluated = production.fit_map_state(config)
    residual = (
        np.asarray(evaluated["_map_least_squares_prediction"], dtype=float)[0]
        - np.asarray(data["sums"], dtype=float)[0] / np.asarray(data["counts"], dtype=float)[0]
    )
    post_reset = np.asarray(data["days"]) >= data["reset_day"]
    offset = float(np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1)[0])
    residual -= offset * post_reset

    leds = np.asarray(data["led"], dtype=int)
    days = np.asarray(data["days"], dtype=float)
    per_led_lag = []
    per_led_rms = []
    for led in range(1, 7):
        mask = leds == led
        order = np.argsort(days[mask])
        values = residual[mask][order]
        per_led_lag.append(lag_correlation(values))
        per_led_rms.append(float(np.sqrt(np.mean(values**2))))
    repeats = np.asarray(data["repeat"], dtype=int)
    common = np.asarray(
        [np.mean(residual[repeats == repeat]) for repeat in np.unique(repeats)]
    )
    ar_raw = np.asarray(map_values["ar_tile_raw"], dtype=float).reshape(-1)
    ar_reference = production.ar_prior_arrays()[0][production.ar_active_parameter_indices()]
    ar_z = (ar_raw - ar_reference) / prior_sigma
    ar_parameters = np.asarray(map_values["ar_tile_raw"], dtype=float)
    ar_parameters = np.asarray(evaluated["ar_parameters"], dtype=float).reshape(-1)
    return {
        "detector": detector,
        "case": case_name,
        "prior_sigma": prior_sigma,
        "initialization": source,
        "rate_nm_day": float(np.exp(np.asarray(map_values["log_rate_scale"]).reshape(-1)[0])),
        "objective": float(diagnostics["objective"]),
        "least_squares_cost": float(diagnostics["objective_history"][-1]),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "common_lag1": lag_correlation(common),
        "common_rms": float(np.sqrt(np.mean(common**2))),
        "per_led_lag1": per_led_lag,
        "per_led_rms": per_led_rms,
        "ar_raw": ar_raw.tolist(),
        "ar_z_max": float(np.max(np.abs(ar_z))),
        "ar_prior_cost": float(0.5 * np.sum(ar_z**2)),
        "ar_parameters": ar_parameters.tolist(),
        "incidence_angle_deg": float(
            np.asarray(map_values["incidence_angle_deg"]).reshape(-1)[0]
        ),
        "gradient_converged": bool(diagnostics["gradient_converged"]),
        "scaled_gradient_norm": float(diagnostics["scaled_gradient_norm"]),
    }


def write_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ar_prior_sensitivity.json").write_text(json.dumps(rows, indent=2))
    with (output_dir / "ar_prior_sensitivity.csv").open("w", newline="") as handle:
        fieldnames = (
            "detector",
            "case",
            "prior_sigma",
            "initialization",
            "rate_nm_day",
            "objective",
            "least_squares_cost",
            "rms",
            "common_lag1",
            "common_rms",
            "ar_z_max",
            "ar_prior_cost",
            "incidence_angle_deg",
            "gradient_converged",
            "scaled_gradient_norm",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: row[key]
                for key in fieldnames
            }
            for row in rows
        )

    labels = [case[0] for case in CASES]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metrics = (
        ("rate_nm_day", "final rate [nm/day]"),
        ("rms", "residual RMS"),
        ("common_lag1", "common repeat lag-1 correlation"),
        ("ar_prior_cost", "AR prior cost"),
    )
    for axis, (field, ylabel) in zip(axes.flat, metrics):
        for detector, color, offset in (
            ("wfi01", "#0072b2", -0.18),
            ("wfi02", "#d55e00", 0.18),
        ):
            values = []
            for label in labels:
                row = next(row for row in rows if row["detector"] == detector and row["case"] == label)
                values.append(row[field])
            axis.bar(positions + offset, values, 0.35, label=detector.upper(), color=color)
        axis.set_title(ylabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    figure.savefig(output_dir / "ar_prior_sensitivity.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    rows = []
    for detector in DETECTORS:
        statistics_path = args.statistics_dir / detector / "tile_statistics" / (
            f"{detector}_tile_log_statistics_512_tile_000.csv"
        )
        map_path = (
            args.constant_rate_dir / detector / "fits" / "tile_000" / "constant_rate_map.npz"
        )
        for case_name, prior_sigma, source in CASES:
            row = fit_case(
                detector,
                statistics_path,
                map_path,
                args.output_dir,
                case_name,
                prior_sigma,
                source,
            )
            rows.append(row)
            print(
                f"{detector} {case_name}: rate={row['rate_nm_day']:.9g}, "
                f"rms={row['rms']:.9g}, common_lag={row['common_lag1']:.6g}",
                flush=True,
            )
    write_outputs(rows, args.output_dir)
    print(f"Wrote AR sensitivity results to {args.output_dir}")


if __name__ == "__main__":
    main()
