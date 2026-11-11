"""WFI01 fixed-ice joint fit with reciprocal-wavelength curve curvature priors.

The existing joint fit and optical basis are retained. This experiment adds a
quadratic penalty on the evaluated log-amplitude and phase curvature, integrated
over normalized reciprocal wavelength. The canonical fixed-ice result is used
as the starting point and is never overwritten.
"""

import argparse
import json
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_amber_student_t_detectors as run
from experiment_joint_wfi01_fixed_ice import (
    LedModel,
    TILE_COUNT,
    add_full_basis,
    fixed_output,
    full_basis,
)
from fit_amber_wfi01_student_t import StudentObjective
from support_aligned_modulation import effective_area_support


def weight_label(weight):
    return f"weight_{format(float(weight), '.6g').replace('-', 'm').replace('.', 'p')}"


def parse_weights(value):
    try:
        weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("weights must be comma-separated numbers") from error
    if not weights or any(not np.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise argparse.ArgumentTypeError("curvature weights must be finite and positive")
    if len(set(weights)) != len(weights):
        raise argparse.ArgumentTypeError("curvature weights must be unique")
    return weights


class InverseWavelengthCurvature:
    """Integrated squared curvature operators on a uniform reciprocal grid."""

    def __init__(self, bounds, mode_count, grid_points=513,
                 amplitude_scale=0.30, phase_scale=1.0):
        if grid_points < 5:
            raise ValueError("At least five reciprocal-wavelength grid points are required")
        low, high = map(float, bounds)
        if not 0.0 < low < high:
            raise ValueError("Wavelength support must be positive and increasing")
        if amplitude_scale <= 0.0 or phase_scale <= 0.0:
            raise ValueError("Curvature normalization scales must be positive")

        # u runs from the red support edge to the blue edge, with equal steps.
        self.u = np.linspace(0.0, 1.0, int(grid_points))
        reciprocal = 1.0 / high + self.u * (1.0 / low - 1.0 / high)
        self.wavelengths_nm = 1.0 / reciprocal
        basis = full_basis(self.wavelengths_nm, (low, high))[0]
        if basis.shape[0] != mode_count:
            raise ValueError(f"Expected {mode_count} modes on curvature grid; found {basis.shape[0]}")

        step = self.u[1] - self.u[0]
        # sqrt(step) / step**2 makes the squared residual sum converge to
        # integral (d2 f / du2)**2 du as the grid is refined.
        second_difference = np.diff(basis, n=2, axis=1) / step**1.5
        self.amplitude_operator = second_difference / float(amplitude_scale)
        self.phase_operator = second_difference / float(phase_scale)
        self.mode_count = int(mode_count)

    def evaluate(self, parameters, weight):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (2 * self.mode_count,):
            raise ValueError("Curvature parameters must contain amplitude and phase modes")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("Curvature weight must be finite and nonnegative")
        gradient = np.zeros_like(parameters)
        if weight == 0.0:
            return 0.0, gradient, 0.0, 0.0

        amplitude_residual = parameters[:self.mode_count] @ self.amplitude_operator
        phase_residual = parameters[self.mode_count:] @ self.phase_operator
        amplitude_cost = 0.5 * float(weight) * float(amplitude_residual @ amplitude_residual)
        phase_cost = 0.5 * float(weight) * float(phase_residual @ phase_residual)
        gradient[:self.mode_count] = (
            float(weight) * self.amplitude_operator @ amplitude_residual
        )
        gradient[self.mode_count:] = (
            float(weight) * self.phase_operator @ phase_residual
        )
        return amplitude_cost + phase_cost, gradient, amplitude_cost, phase_cost

def _initial_parameters(detector, valid, source_ids, objective, baseline_fit_dir=None):
    baseline_dir = Path(baseline_fit_dir) if baseline_fit_dir is not None else fixed_output(detector)
    archive_path = baseline_dir / "joint_fit.npz"
    summary_path = baseline_dir / "joint_fit_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"A completed fixed-ice joint baseline is required at {baseline_dir}"
        )
    summary = json.loads(summary_path.read_text())
    if not summary.get("optimizer_success") or summary.get("ice_histories_refitted"):
        raise ValueError("The saved WFI01 baseline must be a successful fixed-ice fit")
    with np.load(archive_path, allow_pickle=False) as archive:
        parameters = np.asarray(archive["parameters"], dtype=float)
        saved_valid = np.asarray(archive["stellar_valid"], dtype=bool)
        saved_ids = np.asarray(archive["source_ids"]).astype(str)
        baseline_prediction = np.asarray(archive["stellar_prediction_mag"], dtype=float) * 1000.0
    np.testing.assert_array_equal(saved_valid, valid)
    np.testing.assert_array_equal(saved_ids, np.asarray(source_ids).astype(str))
    if parameters.shape != objective.coordinate_scale.shape:
        raise ValueError("Saved fixed-ice parameters do not match the current WFI01 objective")
    coordinates = np.concatenate([
        parameters / objective.coordinate_scale,
        [np.log(float(summary["extra_student_t_scale_mmag"]))],
    ])
    return baseline_dir, summary, coordinates, parameters, baseline_prediction


def run_sweep(detector, weights, output_root, maxiter, grid_points,
              baseline_fit_dir=None, ice_history_fit_root=None,
              optical_cache_root=None, relative_response_root=None):
    if detector != "WFI01":
        raise ValueError("This first reciprocal-wavelength pilot is limited to WFI01")

    area, _ = run.canonical_configuration(detector)
    bounds = effective_area_support(area)
    output_root.mkdir(parents=True, exist_ok=True)
    _, _, source_ids, magnitude, error, valid, _, model, _ = run.prepare(
        detector,
        write_inputs=False,
        effective_area_path=area,
        output_dir=output_root,
        full_effective_area_support=True,
        ice_history_fit_root=ice_history_fit_root,
        optical_cache_root=optical_cache_root,
    )
    add_full_basis(model, bounds)
    objective = StudentObjective(model, magnitude, error, valid)
    baseline_dir, baseline_summary, initial, baseline_parameters, saved_prediction = (
        _initial_parameters(detector, valid, source_ids, objective, baseline_fit_dir)
    )
    led = LedModel(bounds, detector=detector, relative_response_root=relative_response_root)
    curvature = InverseWavelengthCurvature(bounds, model.mode_count, grid_points=grid_points)

    base_state = objective.state(initial)
    baseline_prediction = np.asarray(base_state[1], dtype=float)
    if baseline_prediction.shape != saved_prediction.shape:
        raise ValueError("Saved and reconstructed fixed-ice stellar predictions have different shapes")
    baseline_prediction_error = float(np.max(np.abs(baseline_prediction - saved_prediction)))
    if baseline_prediction_error > 1e-5:
        raise ValueError(
            f"Fixed-ice baseline reconstruction differs by {baseline_prediction_error:.6g} mmag"
        )
    base_led_residual, _, _ = led.evaluate(baseline_parameters)
    base_led_chi2 = float(base_led_residual @ base_led_residual)
    _, _, base_amplitude_cost, base_phase_cost = curvature.evaluate(
        baseline_parameters[:2 * model.mode_count], 1.0
    )

    records = [{
        "label": "existing_fixed_ice_baseline",
        "curvature_weight": 0.0,
        "curvature_penalty": 0.0,
        "amplitude_curvature_penalty_at_unit_weight": base_amplitude_cost,
        "phase_curvature_penalty_at_unit_weight": base_phase_cost,
        "car190_chi2": base_led_chi2,
        "stellar_objective_including_existing_regularization": float(base_state[0]),
        "total_objective": float(base_state[0] + 0.5 * base_led_chi2),
        "prediction_change_rms_mmag": 0.0,
        "prediction_change_p95_abs_mmag": 0.0,
        "optimizer_success": True,
        "fit_directory": str(baseline_dir),
    }]
    curves_for_plot = {"existing baseline": (
        curvature.wavelengths_nm,
        baseline_parameters[:model.mode_count] @ full_basis(curvature.wavelengths_nm, bounds)[0],
        baseline_parameters[model.mode_count:2 * model.mode_count]
        @ full_basis(curvature.wavelengths_nm, bounds)[0],
    )}

    for weight in weights:
        label = weight_label(weight)
        output = output_root / label
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing experiment output: {output}")
        output.mkdir(parents=True, exist_ok=True)

        def total(coordinates):
            stellar_value, stellar_gradient = objective(coordinates)
            physical = coordinates[:2 * model.mode_count] * objective.coordinate_scale[:2 * model.mode_count]
            curve_value, curve_gradient, _, _ = curvature.evaluate(physical, weight)
            led_residual, led_jacobian, _ = led.evaluate(physical)
            led_cost = 0.5 * float(led_residual @ led_residual)
            value = stellar_value + curve_value + led_cost
            gradient = stellar_gradient.copy()
            gradient[:2 * model.mode_count] += (
                curve_gradient * objective.coordinate_scale[:2 * model.mode_count]
            )
            gradient[:2 * model.mode_count] += (
                led_jacobian.T @ led_residual
            ) * objective.coordinate_scale[:2 * model.mode_count]
            return value, gradient

        initial_value, initial_gradient = total(initial)
        gradient_checks = []
        check_indices = (0, model.mode_count, 2 * model.mode_count, len(initial) - 1)
        for index in check_indices:
            step = 1e-4
            upper = initial.copy()
            lower = initial.copy()
            upper[index] += step
            lower[index] -= step
            numerical = (total(upper)[0] - total(lower)[0]) / (2.0 * step)
            gradient_checks.append(
                abs(numerical - initial_gradient[index]) / (1.0 + abs(numerical))
            )
        if max(gradient_checks) > 1e-2:
            raise RuntimeError(f"Joint curvature gradient check failed: {gradient_checks}")

        mode = model.mode_count
        fit_bounds = [(-3.0, 3.0)] * mode + [(-10.0, 10.0)] * mode
        fit_bounds += [(-25.0, 25.0)] * model.zero_point_count
        fit_bounds += [(np.log(1e-4), np.log(100.0))]
        for index in (7, 8):
            fit_bounds[index] = (-6.0, 6.0)
        started = time.monotonic()
        calls = 0

        def report(coordinates):
            nonlocal calls
            calls += 1
            if calls % 10 == 0:
                value, _ = total(coordinates)
                physical = coordinates[:2 * mode] * objective.coordinate_scale[:2 * mode]
                curve_cost = curvature.evaluate(physical, weight)[0]
                print(
                    f"{label}: evaluation {calls}; objective={value:.6f}; "
                    f"curvature={curve_cost:.6f}",
                    flush=True,
                )

        result = minimize(
            total,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=fit_bounds,
            callback=report,
            options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-4, "maxls": 30},
        )
        final_value, final_gradient = total(result.x)
        stellar_value, prediction, _, _, _, _ = objective.state(result.x)
        physical = result.x[:-1] * objective.coordinate_scale
        led_residual, _, _ = led.evaluate(physical[:2 * mode])
        curve_value, _, amp_cost, phase_cost = curvature.evaluate(physical[:2 * mode], weight)
        prediction_change = prediction - baseline_prediction
        grid_basis = full_basis(curvature.wavelengths_nm, bounds)[0]
        fitted_curve = (
            curvature.wavelengths_nm,
            physical[:mode] @ grid_basis,
            physical[mode:2 * mode] @ grid_basis,
        )
        curves_for_plot[label] = fitted_curve

        summary = {
            "detector": detector,
            "curvature_weight": float(weight),
            "curvature_definition": (
                "integrated squared second derivative of log-amplitude and phase "
                "over u=normalized(1/wavelength); existing StudentObjective priors retained"
            ),
            "curvature_grid_points": int(grid_points),
            "curvature_amplitude_normalization": 0.30,
            "curvature_phase_normalization_rad": 1.0,
            "baseline_fit_directory": str(baseline_dir),
            "ice_history_fit_root": str(ice_history_fit_root) if ice_history_fit_root else None,
            "optical_cache_root": str(optical_cache_root) if optical_cache_root else None,
            "relative_response_root": str(relative_response_root) if relative_response_root else None,
            "tile_count": TILE_COUNT,
            "stellar_sources": len(source_ids),
            "stellar_measurements": int(valid.sum()),
            "LED_observations": len(led_residual),
            "ice_histories_refitted": False,
            "TMM_parameters_refitted": False,
            "initial_objective": float(initial_value),
            "final_joint_objective_including_curvature": float(final_value),
            "final_stellar_objective_including_existing_regularization": float(stellar_value),
            "curvature_penalty": float(curve_value),
            "log_amplitude_curvature_penalty": float(amp_cost),
            "phase_curvature_penalty": float(phase_cost),
            "car190_chi2": float(led_residual @ led_residual),
            "baseline_car190_chi2": base_led_chi2,
            "extra_student_t_scale_mmag": float(np.exp(result.x[-1])),
            "prediction_change_rms_mmag": float(np.sqrt(np.mean(prediction_change**2))),
            "prediction_change_p95_abs_mmag": float(np.quantile(np.abs(prediction_change), 0.95)),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "iterations": int(result.nit),
            "seconds": time.monotonic() - started,
            "max_gradient_check_relative_difference": float(max(gradient_checks)),
        }
        (output / "joint_fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(
            output / "joint_fit.npz",
            parameters=physical,
            stellar_prediction_mag=prediction / 1000.0,
            led_normalized_residual=led_residual,
            stellar_valid=valid,
            source_ids=np.asarray(source_ids),
            wavelength_nm=model.wavelengths_nm,
            basis=model.basis,
        )
        pd.DataFrame({
            "mode": model.basis_names,
            "log_amplitude": physical[:mode],
            "phase_rad": physical[mode:2 * mode],
        }).to_csv(output / "joint_modulation_parameters.csv", index=False)
        curve_order = np.argsort(curvature.wavelengths_nm)
        pd.DataFrame({
            "wavelength_nm": curvature.wavelengths_nm[curve_order],
            "normalized_inverse_wavelength": curvature.u[curve_order],
            "log_amplitude": fitted_curve[1][curve_order],
            "phase_rad": fitted_curve[2][curve_order],
        }).to_csv(output / "spectral_curves.csv", index=False)
        records.append({
            "label": label,
            "curvature_weight": float(weight),
            "curvature_penalty": float(curve_value),
            "amplitude_curvature_penalty_at_unit_weight": base_amplitude_cost,
            "phase_curvature_penalty_at_unit_weight": base_phase_cost,
            "car190_chi2": float(led_residual @ led_residual),
            "stellar_objective_including_existing_regularization": float(stellar_value),
            "total_objective": float(final_value),
            "prediction_change_rms_mmag": summary["prediction_change_rms_mmag"],
            "prediction_change_p95_abs_mmag": summary["prediction_change_p95_abs_mmag"],
            "optimizer_success": bool(result.success),
            "fit_directory": str(output),
        })
        print(json.dumps(summary, indent=2), flush=True)
        if not result.success:
            raise RuntimeError(f"Curvature fit {label} failed: {result.message}")

    pd.DataFrame(records).to_csv(output_root / "curvature_sweep_summary.csv", index=False)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(curves_for_plot)))
    for (label, (wavelength, amplitude, phase)), color in zip(curves_for_plot.items(), colors, strict=True):
        axes[0].plot(wavelength, 100.0 * np.expm1(amplitude), color=color, label=label)
        axes[1].plot(wavelength, phase, color=color, label=label)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Log-amplitude modulation (%)")
    axes[1].set_ylabel("Phase modulation (rad)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].legend(fontsize=8, ncol=2)
    figure.suptitle("WFI01 fixed-ice joint fit: reciprocal-wavelength curvature sweep")
    figure.savefig(output_root / "reciprocal_wavelength_curvature_sweep.png", dpi=180)
    plt.close(figure)
    print(f"Wrote curvature sweep under {output_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, choices=["WFI01"])
    parser.add_argument("--weights", type=parse_weights, default=parse_weights("0.0001,0.001,0.01"))
    parser.add_argument("--maxiter", type=int, default=700)
    parser.add_argument("--grid-points", type=int, default=513)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--baseline-fit-dir", type=Path)
    parser.add_argument("--ice-history-fit-root", type=Path)
    parser.add_argument("--optical-cache-root", type=Path)
    parser.add_argument("--relative-response-root", type=Path)
    args = parser.parse_args()
    if args.maxiter <= 0:
        parser.error("--maxiter must be positive")
    if args.grid_points < 5:
        parser.error("--grid-points must be at least five")
    output_root = args.output_root or fixed_output(args.detector).with_name(
        "car190_amber_joint_fixed_ice_inverse_wavelength_curvature_car190_anchored"
    )
    run_sweep(args.detector, args.weights, output_root, args.maxiter, args.grid_points,
              baseline_fit_dir=args.baseline_fit_dir,
              ice_history_fit_root=args.ice_history_fit_root,
              optical_cache_root=args.optical_cache_root,
              relative_response_root=args.relative_response_root)


if __name__ == "__main__":
    main()
