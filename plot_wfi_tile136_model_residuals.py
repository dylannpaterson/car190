"""Plot tile-136 data/model trajectories and LED residual diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import fit_wfi_tmm_production as production
from assess_rcs_spectral_identifiability import load_qe, load_sed


RESET_DAY = 12.907397928240743
DETECTORS = ("wfi01", "wfi10")
LED_COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#f0e442", "#56b4e9")
VARIANT_NAMES = (
    ("baseline", "Production baseline"),
    ("mod002", "AR/HgCdTe modulation, sigma=0.02"),
    ("mod010", "AR/HgCdTe modulation, sigma=0.10"),
    ("all025", "All n + HgCdTe k modulation, sigma=0.25"),
    ("all050", "All n + HgCdTe k modulation, sigma=0.50"),
    ("ice001", "Ice n sigma=0.01; AR/HgCdTe sigma=0.25"),
    (
        "ice001_hn5",
        "Ice n sigma=0.01; half-normal d0 sigma=5 nm",
    ),
    (
        "iceconstant_hn5",
        "Constant ice n; angle fixed; half-normal d0 sigma=5 nm",
    ),
    (
        "iceconstant_hn5_freeangle",
        "Constant ice n; angle fitted; half-normal d0 sigma=5 nm",
    ),
    (
        "iceconstant005_fixedangle",
        "Constant ice n; prior sigma=0.05; angle fixed",
    ),
)


def effective_led_wavelengths():
    spectra = load_sed(production.SED_URL)
    qe_wavelengths, _, _ = load_qe(production.QE_URL)
    effective = []
    for band in range(1, 7):
        spectrum = spectra[band]
        wavelengths = np.asarray(spectrum["wavelength"], dtype=float)
        output = np.where(
            (wavelengths >= qe_wavelengths[0])
            & (wavelengths <= qe_wavelengths[-1]),
            production._sed_output(spectrum),
            0.0,
        )
        weights = output * production._trapezoid_sample_weights(wavelengths)
        effective.append(float(np.sum(weights * wavelengths) / np.sum(weights)))
    return np.asarray(effective)


def load_predictions(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    rows.sort(key=lambda row: int(row["group"]))
    return {
        "detector": str(rows[0].get("detector", "WFI")),
        "tile_id": int(rows[0].get("tile_id", 136)),
        "group": np.asarray([int(row["group"]) for row in rows]),
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "repeat": np.asarray([int(row["repeat"]) for row in rows]),
        "observed": np.asarray(
            [float(row["observed_log_change"]) for row in rows]
        ),
        "model": np.asarray([float(row["model_log_change"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }


def _experiment_rms(values, mask=None):
    residual = values["residual"] if mask is None else values["residual"][mask]
    return float(np.sqrt(np.mean(residual**2)))


def _experiment_led_statistics(values):
    nonreference = values["repeat"] != 0
    statistics = []
    for led in range(1, 7):
        mask = nonreference & (values["led"] == led)
        residual = values["residual"][mask]
        statistics.append(
            {
                "led": led,
                "count": int(residual.size),
                "mean_residual_log": float(np.mean(residual)),
                "rms_log": float(np.sqrt(np.mean(residual**2))),
            }
        )
    return statistics


def summarize_experiment(values, output):
    """Write RMS and LED-bias diagnostics for one experiment fit."""
    nonreference = values["repeat"] != 0
    rows = [
        {
            "scope": "all",
            "led": "all",
            "count": int(values["residual"].size),
            "mean_residual_log": float(np.mean(values["residual"])),
            "rms_log": _experiment_rms(values),
        },
        {
            "scope": "nonreference",
            "led": "all",
            "count": int(np.count_nonzero(nonreference)),
            "mean_residual_log": float(np.mean(values["residual"][nonreference])),
            "rms_log": _experiment_rms(values, nonreference),
        },
    ]
    rows.extend(
        {
            "scope": "nonreference",
            "led": str(statistics["led"]),
            "count": statistics["count"],
            "mean_residual_log": statistics["mean_residual_log"],
            "rms_log": statistics["rms_log"],
        }
        for statistics in _experiment_led_statistics(values)
    )
    fields = ("scope", "led", "count", "mean_residual_log", "rms_log")
    with Path(output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_experiment_data_model(values, output):
    """Plot corrected data/model values and residuals for one experiment fit."""
    nonreference = values["repeat"] != 0
    rms = _experiment_rms(values, nonreference)
    label = f"{values['detector']} tile {values['tile_id']}"
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )
    data_axis, residual_axis = axes
    for led, color in enumerate(LED_COLORS, start=1):
        mask = values["led"] == led
        order = np.argsort(values["days"][mask])
        data_axis.scatter(
            values["days"][mask],
            1000.0 * values["observed"][mask],
            color=color,
            s=7,
            alpha=0.35,
            edgecolors="none",
        )
        data_axis.plot(
            values["days"][mask][order],
            1000.0 * values["model"][mask][order],
            color=color,
            linewidth=0.9,
        )
        residual_mask = nonreference & mask
        residual_axis.scatter(
            values["days"][residual_mask],
            1000.0 * values["residual"][residual_mask],
            color=color,
            marker=".",
            s=12,
            edgecolors="none",
        )
    for axis in axes:
        add_common_axis_format(axis)
    data_axis.set_ylabel("log change [10^-3]")
    residual_axis.set_ylabel("observed - model [10^-3]")
    residual_axis.set_xlabel("Days since first exposure")
    data_axis.set_title(
        f"{label}: corrected data versus model\n"
        f"Non-reference residual RMS = {1000.0 * rms:.4f} millilog"
    )
    residual_axis.set_title("Non-reference residuals by LED", fontsize=10)
    data_axis.legend(
        handles=[
            Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="data"),
            Line2D([], [], color="black", linewidth=1.2, label="model"),
            Line2D([], [], color="0.45", linestyle="--", linewidth=0.8, label="ice reset"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    residual_axis.legend(
        handles=[
            Line2D([], [], color=color, linewidth=1.2, label=f"LED {led}")
            for led, color in enumerate(LED_COLORS, start=1)
        ],
        ncol=3,
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_led_summary(values, output):
    """Plot LED mean residuals and RMS values for one experiment fit."""
    statistics = _experiment_led_statistics(values)
    label = f"{values['detector']} tile {values['tile_id']}"
    positions = np.arange(1, 7)
    means = np.asarray([row["mean_residual_log"] for row in statistics])
    rms = np.asarray([row["rms_log"] for row in statistics])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    mean_axis, rms_axis = axes
    mean_axis.axhline(0.0, color="black", linewidth=0.7)
    mean_axis.scatter(
        positions,
        1000.0 * means,
        color="#0072b2",
        s=28,
        edgecolors="none",
    )
    mean_axis.set_title("Non-reference mean residual by LED")
    mean_axis.set_ylabel("mean observed - model [10^-3]")
    mean_axis.set_xlabel("LED number")
    mean_axis.set_xticks(positions)
    mean_axis.grid(alpha=0.2)
    rms_axis.bar(positions, 1000.0 * rms, color=LED_COLORS)
    rms_axis.set_title(
        f"Non-reference residual RMS by LED\nOverall RMS = {1000.0 * _experiment_rms(values, values['repeat'] != 0):.4f} millilog"
    )
    rms_axis.set_ylabel("RMS [10^-3]")
    rms_axis.set_xlabel("LED number")
    rms_axis.set_xticks(positions)
    rms_axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"{label}: LED residual diagnostics")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_residual_histogram(values, output):
    """Plot the non-reference residual distribution and its RMS."""
    residual = values["residual"][values["repeat"] != 0] * 1000.0
    rms = float(np.sqrt(np.mean(residual**2)))
    label = f"{values['detector']} tile {values['tile_id']}"
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.hist(residual, bins=28, color="#0072b2", alpha=0.75, edgecolor="white")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.axvline(rms, color="#d55e00", linestyle="--", linewidth=1.0)
    axis.axvline(-rms, color="#d55e00", linestyle="--", linewidth=1.0)
    axis.set_xlabel("observed - model residual [millilog]")
    axis.set_ylabel("count")
    axis.set_title(f"{label}: residual distribution, RMS = {rms:.4f} millilog")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _experiment_index_curve(knots, knot_values, grid, monotonic_decreasing=False):
    if monotonic_decreasing:
        return np.asarray(
            production.monotonic_cubic_spline_values(
                knots,
                np.asarray(knot_values, dtype=float)[None, :],
                np.asarray(grid, dtype=float)[None, :],
                np,
            )
        ).reshape(-1)
    from scipy.interpolate import CubicSpline

    spline = CubicSpline(knots, knot_values, bc_type="natural")
    return spline(np.clip(grid, knots[0], knots[-1]))


def _porosity_index_curve(grid, p0, pinf, length_nm, n_solid):
    porosity = pinf - (pinf - p0) * np.exp(-np.maximum(grid, 0.0) / length_nm)
    solid_lorentz_lorenz = (n_solid**2 - 1.0) / (n_solid**2 + 2.0)
    effective_lorentz_lorenz = (1.0 - porosity) * solid_lorentz_lorenz
    effective_n = np.sqrt(
        (1.0 + 2.0 * effective_lorentz_lorenz)
        / (1.0 - effective_lorentz_lorenz)
    )
    return porosity, effective_n


def _load_production_trajectory(production_fit_dir):
    production_fit_dir = Path(production_fit_dir)
    with np.load(
        production_fit_dir / "posterior_samples.npz", allow_pickle=False
    ) as archive:
        thickness = np.asarray(
            archive["node_ice_thickness"], dtype=float
        ).reshape(-1)
        rates = np.asarray(
            archive["node_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
    with (production_fit_dir / "tile_group_predictions.csv").open(
        newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    days = np.asarray([float(row["days"]) for row in rows], dtype=float)
    repeats = np.asarray([int(row["repeat"]) for row in rows], dtype=int)
    repeat_values = np.asarray(sorted(set(repeats.tolist())), dtype=int)
    node_days = np.asarray(
        [np.mean(days[repeats == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    if node_days.size != thickness.size or node_days.size != rates.size:
        raise ValueError(
            "The production trajectory archive does not match its node grid: "
            f"days={node_days.size}, thickness={thickness.size}, rates={rates.size}"
        )
    return node_days, thickness, rates


COMPARISON_COLORS = ("#d55e00", "#0072b2", "#009e73", "#cc79a7")
PRODUCTION_TRAJECTORY_LABEL = "production (fixed n = 1.25)"


def _experiment_comparison_label(fit_summary):
    if fit_summary.get("porosity_model", False):
        sigma = fit_summary.get("metrics", {}).get(
            "porosity_solid_n_prior_sigma_log"
        )
        if sigma is not None:
            return f"thickness-based porosity (n prior sigma_log = {float(sigma):g})"
        return "thickness-based porosity model"
    axis = fit_summary.get("ice_n_spline_axis", "thickness")
    trajectory = (
        "free-rate-node trajectory"
        if fit_summary.get("trajectory_parameterization") == "rate_nodes"
        else "constant-rate trajectory"
    )
    return f"{axis}-spline {trajectory}"


def _load_experiment_trajectory(fit_dir):
    fit_dir = Path(fit_dir)
    with (fit_dir / "fit.json").open() as handle:
        fit_summary = json.load(handle)
    with np.load(fit_dir / "spline_map.npz", allow_pickle=False) as archive:
        thickness = np.asarray(archive["ice_thickness"], dtype=float).reshape(-1)
        rates = np.asarray(archive["instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        node_rates = np.asarray(
            archive["node_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
        if fit_summary.get("porosity_model", False):
            index_parameters = (
                float(np.asarray(archive["porosity_p0"]).reshape(-1)[0]),
                float(np.asarray(archive["porosity_pinf"]).reshape(-1)[0]),
                float(np.asarray(archive["porosity_L_nm"]).reshape(-1)[0]),
                float(fit_summary["porosity_solid_n"]),
            )
        else:
            spline_axis = fit_summary.get("ice_n_spline_axis", "thickness")
            knot_name = (
                "ice_n_time_knots_days"
                if spline_axis == "time"
                else "ice_n_thickness_knots_nm"
            )
            index_parameters = (
                spline_axis,
                np.asarray(archive[knot_name], dtype=float),
                np.asarray(archive["ice_n_knot_values"], dtype=float),
                bool(fit_summary.get("monotonic_decreasing", False)),
            )
    with (fit_dir / "group_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    observation_days = np.asarray([float(row["days"]) for row in rows])
    repeats = np.asarray([int(row["repeat"]) for row in rows], dtype=int)
    repeat_values = np.asarray(sorted(set(repeats.tolist())), dtype=int)
    node_days = np.asarray(
        [np.mean(observation_days[repeats == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    if node_days.size != node_rates.size:
        raise ValueError(
            f"The experiment archive has {node_rates.size} node rates for "
            f"{node_days.size} node days: {fit_dir}"
        )
    return {
        "fit_summary": fit_summary,
        "label": _experiment_comparison_label(fit_summary),
        "observation_days": observation_days,
        "thickness": thickness,
        "rates": rates,
        "node_days": node_days,
        "node_rates": node_rates,
        "index_parameters": index_parameters,
    }


def plot_experiment_comparison_rate(fit_dirs, output, production_fit_dir=None):
    """Plot multiple experiment rate trajectories against production."""
    fits = [_load_experiment_trajectory(path) for path in fit_dirs]
    production_trajectory = (
        None
        if production_fit_dir is None
        else _load_production_trajectory(production_fit_dir)
    )
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for index, fit in enumerate(fits):
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        axis.plot(
            fit["node_days"],
            fit["node_rates"],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=fit["label"],
        )
    if production_trajectory is not None:
        production_days, _, production_rates = production_trajectory
        axis.plot(
            production_days,
            production_rates,
            color="#cc79a7",
            linestyle="--",
            linewidth=1.5,
            label=PRODUCTION_TRAJECTORY_LABEL,
        )
    axis.axvline(
        RESET_DAY,
        color="0.45",
        linestyle="--",
        linewidth=0.8,
        label="ice reset",
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("ice accumulation rate [nm/day]")
    axis.set_title("WFI02 tile 136: prior and production trajectory comparison")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_comparison_trajectory(
    fit_dirs,
    output,
    production_fit_dir=None,
):
    """Plot multiple thickness and index trajectories against production."""
    fits = [_load_experiment_trajectory(path) for path in fit_dirs]
    production_trajectory = (
        None
        if production_fit_dir is None
        else _load_production_trajectory(production_fit_dir)
    )
    maximum_thickness = max(
        [float(np.max(fit["thickness"])) for fit in fits]
        + (
            []
            if production_trajectory is None
            else [float(np.max(production_trajectory[1]))]
        )
    )
    spline_axes = {
        parameters[0]
        for fit in fits
        if not fit["fit_summary"].get("porosity_model", False)
        for parameters in [fit["index_parameters"]]
    }
    if len(spline_axes) > 1:
        raise ValueError("Cannot compare thickness- and time-axis splines on one index axis")
    comparison_axis = next(iter(spline_axes), "thickness")
    if comparison_axis == "time":
        grid = np.linspace(
            min(float(np.min(fit["observation_days"])) for fit in fits),
            max(float(np.max(fit["observation_days"])) for fit in fits),
            500,
        )
    else:
        grid = np.linspace(0.0, max(300.0, maximum_thickness), 500)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    thickness_axis, index_axis = axes
    porosity_axis = None
    for index, fit in enumerate(fits):
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        thickness_axis.plot(
            fit["observation_days"],
            fit["thickness"],
            color=color,
            linewidth=1.4,
            label=fit["label"],
        )
        if fit["fit_summary"].get("porosity_model", False):
            p0, pinf, length_nm, n_solid = fit["index_parameters"]
            porosity, index_curve = _porosity_index_curve(
                grid, p0, pinf, length_nm, n_solid
            )
            if porosity_axis is None:
                porosity_axis = index_axis.twinx()
                porosity_axis.set_ylabel("porosity")
                porosity_axis.set_ylim(0.0, 1.0)
            index_axis.plot(
                grid,
                index_curve,
                color=color,
                linewidth=1.5,
                label=fit["label"],
            )
            porosity_axis.plot(
                grid,
                porosity,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"{fit['label']} porosity",
            )
        else:
            spline_axis, knots, knot_values, monotonic_decreasing = fit[
                "index_parameters"
            ]
            index_axis.plot(
                grid,
                _experiment_index_curve(
                    knots,
                    knot_values,
                    grid,
                    monotonic_decreasing=monotonic_decreasing,
                ),
                color=color,
                linewidth=1.5,
                label=fit["label"],
            )
    if production_trajectory is not None:
        production_days, production_thickness, _ = production_trajectory
        thickness_axis.plot(
            production_days,
            production_thickness,
            color="#cc79a7",
            linestyle="--",
            linewidth=1.5,
            label=PRODUCTION_TRAJECTORY_LABEL,
        )
        index_axis.axhline(
            1.25,
            color="#cc79a7",
            linestyle="--",
            linewidth=1.5,
            label=PRODUCTION_TRAJECTORY_LABEL,
        )
    thickness_axis.axvline(RESET_DAY, color="0.45", linestyle="--", linewidth=0.8)
    for axis in axes:
        axis.grid(alpha=0.2)
    thickness_axis.set_ylabel("ice thickness [nm]")
    thickness_axis.set_xlabel("Days since first exposure")
    thickness_axis.set_title("WFI02 tile 136: thickness trajectory comparison")
    thickness_axis.legend(frameon=False)
    index_axis.set_xlabel(
        "days since first exposure"
        if comparison_axis == "time"
        else "ice thickness [nm]"
    )
    index_axis.set_ylabel("ice refractive index n")
    index_axis.set_title(
        "Time-dependent ice index comparison"
        if comparison_axis == "time"
        else "Thickness-dependent ice index comparison"
    )
    if comparison_axis == "time":
        index_axis.set_xlim(float(np.min(grid)), float(np.max(grid)))
    else:
        index_axis.set_xlim(0.0, max(300.0, maximum_thickness))
    index_handles, index_labels = index_axis.get_legend_handles_labels()
    if porosity_axis is not None:
        porosity_handles, porosity_labels = porosity_axis.get_legend_handles_labels()
        index_handles += porosity_handles
        index_labels += porosity_labels
    index_axis.legend(index_handles, index_labels, frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_comparison_diagnostics(
    fit_dirs,
    output_dir,
    production_fit_dir=None,
):
    """Generate rate and trajectory comparisons for multiple experiment fits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "experiment_rate_comparison.png",
        output_dir / "experiment_trajectory_and_ice_n_comparison.png",
    )
    plot_experiment_comparison_rate(
        fit_dirs,
        outputs[0],
        production_fit_dir=production_fit_dir,
    )
    plot_experiment_comparison_trajectory(
        fit_dirs,
        outputs[1],
        production_fit_dir=production_fit_dir,
    )
    return outputs


def plot_experiment_trajectory(fit_dir, output, production_fit_dir=None):
    """Plot fitted thickness/rate and the fitted thickness-dependent ice index."""
    fit_dir = Path(fit_dir)
    with (fit_dir / "fit.json").open() as handle:
        fit_summary = json.load(handle)
    monotonic_decreasing = bool(fit_summary.get("monotonic_decreasing", False))
    porosity_model = bool(fit_summary.get("porosity_model", False))
    spline_axis = fit_summary.get("ice_n_spline_axis", "thickness")
    with np.load(fit_dir / "spline_map.npz", allow_pickle=False) as archive:
        thickness = np.asarray(archive["ice_thickness"], dtype=float).reshape(-1)
        rates = np.asarray(archive["instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        if porosity_model:
            p0 = float(np.asarray(archive["porosity_p0"]).reshape(-1)[0])
            pinf = float(np.asarray(archive["porosity_pinf"]).reshape(-1)[0])
            length_nm = float(np.asarray(archive["porosity_L_nm"]).reshape(-1)[0])
            n_solid = float(fit_summary["porosity_solid_n"])
        else:
            knot_name = (
                "ice_n_time_knots_days"
                if spline_axis == "time"
                else "ice_n_thickness_knots_nm"
            )
            knots = np.asarray(archive[knot_name], dtype=float)
            knot_values = np.asarray(archive["ice_n_knot_values"], dtype=float)
    with (fit_dir / "group_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    label = f"{rows[0]['detector']} tile {rows[0]['tile_id']}"
    production_label = "production (fixed n = 1.25)"
    trajectory_label = (
        "thickness-based porosity model"
        if porosity_model
        else "time-spline ice n model"
        if spline_axis == "time"
        else "free-rate-node trajectory"
        if fit_summary.get("trajectory_parameterization") == "rate_nodes"
        else "constant-rate trajectory"
    )
    observation_days = np.asarray([float(row["days"]) for row in rows])
    if porosity_model or spline_axis == "thickness":
        grid = np.linspace(0.0, max(300.0, float(np.max(thickness))), 500)
    else:
        grid = np.linspace(float(np.min(observation_days)), float(np.max(observation_days)), 500)
    if porosity_model:
        porosity_curve, index_curve = _porosity_index_curve(
            grid, p0, pinf, length_nm, n_solid
        )
    else:
        index_curve = _experiment_index_curve(
            knots,
            knot_values,
            grid,
            monotonic_decreasing=monotonic_decreasing,
        )
    production_trajectory = (
        None
        if production_fit_dir is None
        else _load_production_trajectory(production_fit_dir)
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    thickness_axis, index_axis = axes
    thickness_axis.plot(observation_days, thickness, color="#0072b2", linewidth=1.4, label="ice thickness")
    if production_trajectory is not None:
        production_days, production_thickness, _ = production_trajectory
        thickness_axis.plot(
            production_days,
            production_thickness,
            color="#cc79a7",
            linestyle="--",
            linewidth=1.5,
            label=production_label,
        )
    thickness_axis.axvline(RESET_DAY, color="0.45", linestyle="--", linewidth=0.8)
    thickness_axis.set_ylabel("ice thickness [nm]")
    thickness_axis.set_xlabel("Days since first exposure")
    thickness_axis.set_title(
        f"{label}: {trajectory_label} ({np.mean(rates):.4f} nm/day)"
    )
    thickness_axis.grid(alpha=0.2)
    thickness_axis.legend(frameon=False)
    index_axis.plot(
        grid,
        index_curve,
        color="#d55e00",
        linewidth=1.5,
        label=(
            "thickness-based porosity model"
            if porosity_model
            else "monotonic cubic Hermite spline"
            if monotonic_decreasing
            else "natural cubic spline"
        ),
    )
    if porosity_model:
        porosity_axis = index_axis.twinx()
        porosity_axis.plot(
            grid,
            porosity_curve,
            color="#009e73",
            linestyle="--",
            linewidth=1.2,
            label="porosity",
        )
        porosity_axis.set_ylabel("porosity")
        porosity_axis.set_ylim(0.0, 1.0)
        handles, labels = index_axis.get_legend_handles_labels()
        porosity_handles, porosity_labels = porosity_axis.get_legend_handles_labels()
        index_axis.legend(
            handles + porosity_handles,
            labels + porosity_labels,
            frameon=False,
        )
    else:
        index_axis.scatter(
            knots,
            knot_values,
            color="black",
            s=28,
            zorder=3,
            label="fitted knots",
        )
    if porosity_model or spline_axis == "thickness":
        index_axis.axvline(300.0, color="0.45", linestyle="--", linewidth=0.8)
        index_axis.set_xlabel("ice thickness [nm]")
        index_axis.set_title("Fitted thickness-dependent ice index")
    else:
        index_axis.set_xlabel("Days since first exposure")
        index_axis.set_title("Fitted time-dependent ice index")
    index_axis.set_ylabel("ice refractive index n")
    index_axis.grid(alpha=0.2)
    if not porosity_model:
        index_axis.legend(frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_accumulation_rate(
    fit_dir,
    output,
    production_fit_dir=None,
):
    """Plot the fitted instantaneous and node accumulation rates."""
    fit_dir = Path(fit_dir)
    with (fit_dir / "fit.json").open() as handle:
        fit_summary = json.load(handle)
    with np.load(fit_dir / "spline_map.npz", allow_pickle=False) as archive:
        rates = np.asarray(archive["instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        node_rates = np.asarray(
            archive["node_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
    with (fit_dir / "group_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    days = np.asarray([float(row["days"]) for row in rows], dtype=float)
    repeats = np.asarray([int(row["repeat"]) for row in rows], dtype=int)
    repeat_values = np.asarray(sorted(set(repeats.tolist())), dtype=int)
    node_days = np.asarray(
        [np.mean(days[repeats == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    if node_days.size != node_rates.size:
        raise ValueError(
            f"The trajectory archive has {node_rates.size} node rates for "
            f"{node_days.size} node days"
        )
    production_trajectory = (
        None
        if production_fit_dir is None
        else _load_production_trajectory(production_fit_dir)
    )
    label = f"{rows[0]['detector']} tile {rows[0]['tile_id']}"
    production_label = "production (fixed n = 1.25)"
    model_label = (
        "thickness-based porosity model"
        if fit_summary.get("porosity_model", False)
        else "fitted rate nodes"
    )
    trajectory_label = (
        "free-rate-node trajectory"
        if fit_summary.get("trajectory_parameterization") == "rate_nodes"
        else "constant-rate trajectory"
    )
    order = np.argsort(days)
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.scatter(
        days[order],
        rates[order],
        color="#0072b2",
        s=10,
        alpha=0.38,
        edgecolors="none",
        label="instantaneous fitted rate",
    )
    axis.plot(
        node_days,
        node_rates,
        color="#d55e00",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=model_label,
    )
    if production_trajectory is not None:
        production_days, _, production_rates = production_trajectory
        axis.plot(
            production_days,
            production_rates,
            color="#cc79a7",
            linestyle="--",
            linewidth=1.5,
            label=production_label,
        )
    axis.axvline(RESET_DAY, color="0.45", linestyle="--", linewidth=0.8, label="ice reset")
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("ice accumulation rate [nm/day]")
    axis.set_title(f"{label}: {trajectory_label}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_diagnostics(fit_dir, output_dir, production_fit_dir=None):
    """Generate the existing tile-136-style diagnostics for one experiment."""
    fit_dir = Path(fit_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = load_predictions(fit_dir / "group_predictions.csv")
    summarize_experiment(values, output_dir / "experiment_rms_summary.csv")
    plot_experiment_data_model(values, output_dir / "experiment_data_vs_model.png")
    plot_experiment_led_summary(values, output_dir / "experiment_residual_led_summary.png")
    plot_experiment_residual_histogram(values, output_dir / "experiment_residual_histogram.png")
    plot_experiment_trajectory(
        fit_dir,
        output_dir / "experiment_trajectory_and_ice_n.png",
        production_fit_dir=production_fit_dir,
    )
    plot_experiment_accumulation_rate(
        fit_dir,
        output_dir / "experiment_accumulation_rate.png",
        production_fit_dir=production_fit_dir,
    )
    return sorted(output_dir.iterdir())


def variant_path(root, detector, variant):
    if variant == "baseline":
        return (
            root
            / detector
            / "n_prior_sigma_0p05"
            / "fits"
            / "tile_136"
            / "tile_group_predictions.csv"
        )
    if variant == "all025":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_modulation_prior025"
            / "tile_group_predictions.csv"
        )
    if variant == "all050":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_modulation_prior050"
            / "tile_group_predictions.csv"
        )
    if variant == "ice001":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_iceprior001_arhgcdteprior025"
            / "tile_group_predictions.csv"
        )
    if variant == "ice001_hn5":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_iceprior001_arhgcdteprior025_initialhalfnormal5_correct"
            / "tile_group_predictions.csv"
        )
    if variant == "iceconstant_hn5":
        directory = "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fixedangle"
        return root / detector / "diagnostics" / directory / "tile_group_predictions.csv"
    if variant == "iceconstant_hn5_freeangle":
        directory = {
            "wfi01": "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fromconstant",
            "wfi10": "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fromemcee",
        }[detector]
        return root / detector / "diagnostics" / directory / "tile_group_predictions.csv"
    if variant == "iceconstant005_fixedangle":
        return (
            root
            / detector
            / "diagnostics"
            / "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_sig005_fixedangle"
            / "tile_group_predictions.csv"
        )
    suffix = "" if variant == "mod002" else "_modsigma010"
    return (
        root
        / detector
        / "diagnostics"
        / f"tile_136_hgcdte_modulation_matched_np_nprior005{suffix}"
        / "tile_group_predictions.csv"
    )


def load_all(root):
    fits = {}
    for detector in DETECTORS:
        fits[detector] = {
            variant: load_predictions(variant_path(root, detector, variant))
            for variant, _ in VARIANT_NAMES
        }
    return fits


def add_common_axis_format(axis):
    axis.axvline(RESET_DAY, color="0.45", linestyle="--", linewidth=0.8)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.grid(alpha=0.2)


def plot_data_model(fits, output):
    variant_count = len(VARIANT_NAMES)
    figure, axes = plt.subplots(
        2,
        variant_count,
        figsize=(5 * variant_count, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for detector_index, detector in enumerate(DETECTORS):
        for variant_index, (variant, title) in enumerate(VARIANT_NAMES):
            axis = axes[detector_index, variant_index]
            values = fits[detector][variant]
            for led, color in enumerate(LED_COLORS, start=1):
                mask = values["led"] == led
                order = np.argsort(values["days"][mask])
                axis.scatter(
                    values["days"][mask],
                    1000.0 * values["observed"][mask],
                    color=color,
                    s=7,
                    alpha=0.45,
                    edgecolors="none",
                )
                axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["model"][mask][order],
                    color=color,
                    linewidth=0.9,
                )
            add_common_axis_format(axis)
            axis.set_title(title, fontsize=10)
            if detector_index == 1:
                axis.set_xlabel("Days since first exposure")
            if variant_index == 0:
                axis.set_ylabel(f"{detector.upper()}\nlog change [10^-3]")
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="data"),
        Line2D([], [], color="black", linewidth=1.2, label="model"),
        Line2D([], [], color="0.45", linestyle="--", linewidth=0.8, label="ice reset"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("WFI tile 136: corrected data versus model by LED", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_residuals(fits, output):
    variant_count = len(VARIANT_NAMES)
    figure, axes = plt.subplots(
        2,
        variant_count,
        figsize=(5 * variant_count, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for detector_index, detector in enumerate(DETECTORS):
        for variant_index, (variant, title) in enumerate(VARIANT_NAMES):
            axis = axes[detector_index, variant_index]
            values = fits[detector][variant]
            for led, color in enumerate(LED_COLORS, start=1):
                mask = values["led"] == led
                order = np.argsort(values["days"][mask])
                axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["residual"][mask][order],
                    color=color,
                    marker=".",
                    markersize=2.5,
                    linewidth=0.65,
                )
            add_common_axis_format(axis)
            axis.set_title(title, fontsize=10)
            if detector_index == 1:
                axis.set_xlabel("Days since first exposure")
            if variant_index == 0:
                axis.set_ylabel(f"{detector.upper()}\nobserved - model [10^-3]")
    handles = [
        Line2D([], [], color=color, linewidth=1.2, label=f"LED {led}")
        for led, color in enumerate(LED_COLORS, start=1)
    ]
    axes[0, 0].legend(handles=handles, ncol=3, frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("WFI tile 136: residuals versus time", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(fits, effective_wavelengths, output):
    fieldnames = [
        "detector",
        "variant",
        "group_rms_log",
        "nonreference_rms_log",
        "residual_wavelength_pearson",
        "led_mean_wavelength_pearson",
        *[f"led_{led}_mean_residual_log" for led in range(1, 7)],
        *[f"led_{led}_rms_log" for led in range(1, 7)],
    ]
    with Path(output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for detector in DETECTORS:
            for variant, _ in VARIANT_NAMES:
                values = fits[detector][variant]
                nonreference = values["repeat"] != 0
                residual = values["residual"]
                led = values["led"]
                nonreference_residual = residual[nonreference]
                wavelengths = effective_wavelengths[led[nonreference] - 1]
                led_means = np.asarray(
                    [
                        np.mean(nonreference_residual[led[nonreference] == value])
                        for value in range(1, 7)
                    ]
                )
                row = {
                    "detector": detector.upper(),
                    "variant": variant,
                    "group_rms_log": float(np.sqrt(np.mean(residual**2))),
                    "nonreference_rms_log": float(
                        np.sqrt(np.mean(nonreference_residual**2))
                    ),
                    "residual_wavelength_pearson": float(
                        np.corrcoef(nonreference_residual, wavelengths)[0, 1]
                    ),
                    "led_mean_wavelength_pearson": float(
                        np.corrcoef(led_means, effective_wavelengths)[0, 1]
                    ),
                }
                row.update(
                    {
                        f"led_{led}_mean_residual_log": float(led_means[led - 1])
                        for led in range(1, 7)
                    }
                )
                row.update(
                    {
                        f"led_{led_number}_rms_log": float(
                            np.sqrt(
                                np.mean(
                                    nonreference_residual[
                                        led[nonreference] == led_value
                                    ]
                                    ** 2
                                )
                            )
                        )
                        for led_number, led_value in enumerate(range(1, 7), start=1)
                    }
                )
                writer.writerow(row)


def plot_led_summary(fits, effective_wavelengths, output):
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 8), constrained_layout=True
    )
    for detector_index, detector in enumerate(DETECTORS):
        mean_axis = axes[detector_index, 0]
        rms_axis = axes[detector_index, 1]
        positions = np.arange(1, 7)
        width = 0.25
        for variant_index, (variant, title) in enumerate(VARIANT_NAMES):
            values = fits[detector][variant]
            nonreference = values["repeat"] != 0
            led = values["led"][nonreference]
            residual = values["residual"][nonreference]
            means = np.asarray([np.mean(residual[led == value]) for value in positions])
            rms = np.asarray([np.sqrt(np.mean(residual[led == value] ** 2)) for value in positions])
            mean_axis.plot(
                effective_wavelengths,
                1000.0 * means,
                marker="o",
                linewidth=1.2,
                label=title,
            )
            rms_axis.bar(
                positions + (variant_index - 1) * width,
                1000.0 * rms,
                width,
                label=title,
            )
        mean_axis.axhline(0.0, color="black", linewidth=0.7)
        mean_axis.set_title(f"{detector.upper()} LED mean residual")
        mean_axis.set_ylabel("observed - model [10^-3]")
        mean_axis.set_xlabel("Effective LED wavelength [nm]")
        mean_axis.grid(alpha=0.2)
        rms_axis.set_title(f"{detector.upper()} residual RMS by LED")
        rms_axis.set_ylabel("RMS [10^-3]")
        rms_axis.set_xlabel("LED number")
        rms_axis.set_xticks(positions)
        rms_axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("WFI tile 136: LED residual structure", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-fit-dir",
        type=Path,
        action="append",
        help="Experiment fit directory containing group_predictions.csv and spline_map.npz",
    )
    parser.add_argument(
        "--production-fit-dir",
        type=Path,
        help="Production tile fit directory containing posterior_samples.npz.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment_fit_dir:
        if len(args.experiment_fit_dir) == 1:
            paths = plot_experiment_diagnostics(
                args.experiment_fit_dir[0],
                args.output_dir,
                production_fit_dir=args.production_fit_dir,
            )
        else:
            paths = plot_experiment_comparison_diagnostics(
                args.experiment_fit_dir,
                args.output_dir,
                production_fit_dir=args.production_fit_dir,
            )
        for path in paths:
            print(path)
        return
    fits = load_all(args.root)
    effective_wavelengths = effective_led_wavelengths()
    plot_data_model(fits, args.output_dir / "tile136_data_vs_model.png")
    plot_residuals(fits, args.output_dir / "tile136_residuals_vs_time.png")
    plot_led_summary(
        fits,
        effective_wavelengths,
        args.output_dir / "tile136_residual_led_summary.png",
    )
    summarize(
        fits,
        effective_wavelengths,
        args.output_dir / "tile136_model_residual_summary.csv",
    )
    for path in sorted(args.output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
