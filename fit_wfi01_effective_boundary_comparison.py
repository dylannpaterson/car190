"""Compare the calibrated WFI01 response with a fixed effective complex boundary."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_rcs_soc_ice_model import (
    DEFAULT_MODEL_PATH,
    broadband_response_curves,
    fit_periodic_model,
    read_model_table,
)
from analyze_wfi01_all_pixel_soc_model import (
    ROOT,
    fit_global_soc_sqrt,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_wfi01_joint_complex_q_soc import (
    build_wfi01_response,
    load_effective_q,
)
from fit_wfi01_joint_optical_soc import DEFAULT_STEP_TIME, parse_step_mask


DEFAULT_TILE_STATISTICS = ROOT / "wfi01_tile_log_statistics_512.csv"
DEFAULT_ALL_PIXEL_STATISTICS = ROOT / "wfi01_all_pixel_log_statistics_114.csv"
DEFAULT_EFFECTIVE_Q_TABLE = (
    ROOT / "effective_sca1_reflectance_physical" / "effective_sca1_reflectance_table.csv"
)
DEFAULT_OUTPUT = ROOT / "wfi01_effective_boundary_comparison"


def load_aggregated_tile_statistics(path):
    """Collapse tile sufficient statistics to one exact all-pixel row per group."""
    with path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"No tile statistic rows found in {path}")

    groups = {}
    tile_ids = set()
    for source in source_rows:
        group = int(source["group"])
        count = int(source["n_pair_valid"])
        total = float(source["sum_log_change"])
        total_squared = float(source["sum_squared_log_change"])
        if count <= 0 or not np.isfinite([total, total_squared]).all():
            raise ValueError(f"Invalid tile sufficient statistics for group {group}")
        tile_ids.add(int(source["tile_id"]))
        row = groups.setdefault(
            group,
            {
                "led": int(source["led"]),
                "repeat": int(source["repeat"]),
                "file_set_name": source["file_set_name"],
                "exposure_start_time": source["exposure_start_time"],
                "days": float(source["days"]),
                "pixel_count": 0,
                "sum_log_change": 0.0,
                "sum_squared_log_change": 0.0,
                "tile_count": 0,
            },
        )
        if (
            row["led"] != int(source["led"])
            or row["repeat"] != int(source["repeat"])
            or not np.isclose(row["days"], float(source["days"]))
        ):
            raise ValueError(f"Inconsistent metadata within group {group}")
        row["pixel_count"] += count
        row["sum_log_change"] += total
        row["sum_squared_log_change"] += total_squared
        row["tile_count"] += 1

    rows = []
    for group in sorted(groups):
        source = groups[group]
        count = source["pixel_count"]
        total = source["sum_log_change"]
        total_squared = source["sum_squared_log_change"]
        mean = total / count
        rows.append(
            {
                "led": source["led"],
                "repeat": source["repeat"],
                "group": group,
                "detector": 1,
                "file_set_name": source["file_set_name"],
                "exposure_start_time": source["exposure_start_time"],
                "days": source["days"],
                "pixel_count": count,
                "sum_log_change": total,
                "sum_squared_log_change": total_squared,
                "log_change": mean,
                "within_sum_squares": max(total_squared - count * mean * mean, 0.0),
            }
        )
    expected_groups = list(range(len(rows)))
    if [row["group"] for row in rows] != expected_groups:
        raise ValueError("Tile statistic groups must span zero through the final group")
    tile_counts = {source["tile_count"] for source in groups.values()}
    if len(tile_counts) != 1:
        raise ValueError("Every group must contain the same number of tiles")
    return rows, sorted(tile_ids), tile_counts.pop()


def validate_aggregate_against_all_pixel(rows, path):
    """Check that aggregation preserves the existing all-pixel sufficient statistics."""
    with path.open(newline="") as handle:
        reference_rows = {int(row["group"]): row for row in csv.DictReader(handle)}
    if set(reference_rows) != {row["group"] for row in rows}:
        raise ValueError("Aggregated and all-pixel statistic groups differ")
    for row in rows:
        reference = reference_rows[row["group"]]
        if row["pixel_count"] != int(reference["pixel_count"]):
            raise ValueError(f"Pixel count mismatch in group {row['group']}")
        for key, reference_key in (
            ("sum_log_change", "sum_log_change"),
            ("sum_squared_log_change", "sum_squared_log_change"),
        ):
            if not np.isclose(
                row[key],
                float(reference[reference_key]),
                rtol=2.0e-12,
                atol=1.0e-9,
            ):
                raise ValueError(f"{key} mismatch in group {row['group']}")


def parse_float_list(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty comma-separated list of finite numbers")
    return values


def fit_response(observations, response_curves, d_grid, step_mask, args):
    return fit_global_soc_sqrt(
        observations,
        response_curves,
        d_grid,
        max_scale=args.max_scale,
        scale_steps=args.scale_steps,
        reference_thickness_nm=args.reference_thickness_nm,
        reference_time_days=args.reference_time_days,
        reset_day=args.reset_day,
        refinement_steps=args.refinement_steps,
        step_mask=step_mask,
        initial_thickness_nm=0.0,
        fit_initial_thickness=True,
        initial_thickness_starts=parse_float_list(args.initial_thickness_starts),
        max_initial_thickness_nm=args.max_initial_thickness_nm,
    )


def write_predictions(path, observations, empirical, effective):
    fields = (
        "group",
        "led",
        "repeat",
        "days",
        "observed_percent",
        "empirical_percent",
        "effective_q_percent",
        "empirical_residual_log",
        "effective_q_residual_log",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(observations):
            writer.writerow(
                {
                    "group": row["group"],
                    "led": row["led"],
                    "repeat": row["repeat"],
                    "days": f"{row['days']:.12g}",
                    "observed_percent": f"{100.0 * np.expm1(row['log_change']):.12g}",
                    "empirical_percent": f"{100.0 * np.expm1(empirical['prediction'][index]):.12g}",
                    "effective_q_percent": f"{100.0 * np.expm1(effective['prediction'][index]):.12g}",
                    "empirical_residual_log": f"{row['log_change'] - empirical['prediction'][index]:.12g}",
                    "effective_q_residual_log": f"{row['log_change'] - effective['prediction'][index]:.12g}",
                }
            )


def write_all_led_plot(path, observations, empirical, effective, step_mask):
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for led in range(1, 7):
        indices = [index for index, row in enumerate(observations) if row["led"] == led]
        color = colors(led - 1)
        days = np.asarray([observations[index]["days"] for index in indices])
        observed = 100.0 * np.expm1(
            [observations[index]["log_change"] for index in indices]
        )
        axis.plot(days, observed, "o", color=color, ms=3.5, label=f"LED {led} data")
        axis.plot(
            days,
            100.0 * np.expm1(effective["prediction"][indices]),
            color=color,
            lw=1.7,
            label=f"LED {led} effective q",
        )
        axis.plot(
            days,
            100.0 * np.expm1(empirical["prediction"][indices]),
            color=color,
            ls="--",
            lw=1.1,
            alpha=0.75,
            label=f"LED {led} calibrated",
        )
    if step_mask is not None:
        step_days = [row["days"] for row, selected in zip(observations, step_mask) if selected]
        if step_days:
            axis.axvline(min(step_days), color="black", ls=":", lw=1.0, label="ACM step")
    axis.axhline(0.0, color="black", lw=0.7)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Reference-normalized change (%)")
    axis.set_title("All LEDs: calibrated response versus fixed effective complex boundary")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7, ncol=3)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def fit_summary(fit):
    return {
        "initial_thickness_nm": float(fit["initial_thickness_nm"]),
        "mean_rate_nm_day": float(fit["mean_rate_nm_day"]),
        "final_thickness_nm": float(np.max(fit["thickness_nm"])),
        "acm_step_percent": float(fit["step_percent"]),
        "pixel_rms_log": float(fit["pixel_rms_log"]),
        "nonreference_pixel_rms_log": float(fit["nonreference_pixel_rms_log"]),
        "total_sse": float(fit["total_sse"]),
        "between_sse": float(fit["between_sse"]),
    }


def write_summary(path, args, observations, tile_ids, tiles_per_group, empirical, effective):
    empirical_summary = fit_summary(empirical)
    effective_summary = fit_summary(effective)
    lines = [
        "# WFI01 effective-boundary comparison",
        "",
        "The tile sufficient statistics were aggregated back to the exact all-pixel likelihood and fit with the same square-root trajectory, ACM step, thickness bounds, and optimizer settings.",
        "",
        f"Input tiles: `{len(tile_ids)}`; tiles per group: `{tiles_per_group}`; observations: `{len(observations)}`.",
        f"SED mode: `{args.sed_mode}`; wavelength stride: `{args.wavelength_stride}`.",
        "",
        "The effective-boundary model uses the existing fixed complex-q table: `q = rho * exp(i phase)` with the fitted attenuation term. It has no free optical nuisance parameters and no per-tile AR hierarchy, so this is a response-model sensitivity test rather than a replacement for the spatial NumPyro fit.",
        "",
        "| Model | Initial thickness (nm) | Mean rate (nm/day) | Final thickness (nm) | ACM step (%) | Pixel RMS (log) | Non-reference RMS (log) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (
        ("Calibrated response", empirical_summary),
        ("Fixed effective complex boundary", effective_summary),
    ):
        lines.append(
            f"| {label} | {summary['initial_thickness_nm']:.8f} | "
            f"{summary['mean_rate_nm_day']:.8f} | {summary['final_thickness_nm']:.8f} | "
            f"{summary['acm_step_percent']:.8f} | {summary['pixel_rms_log']:.10f} | "
            f"{summary['nonreference_pixel_rms_log']:.10f} |"
        )
    lines.extend(
        [
            "",
            "The current 64-tile hierarchical NumPyro production fit remains the appropriate spatial baseline; its median rate is recorded in the existing production summary. The rates above isolate the effect of swapping the global response representation while holding the trajectory fit setup fixed.",
            "",
            "## Files",
            "",
            f"- Effective q table: `{args.effective_q_table}`",
            f"- Predictions: `{args.output_dir / 'predictions.csv'}`",
            f"- All-LED plot: `{args.output_dir / 'all_led_fit.png'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-statistics", type=Path, default=DEFAULT_TILE_STATISTICS)
    parser.add_argument("--all-pixel-statistics", type=Path, default=DEFAULT_ALL_PIXEL_STATISTICS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--effective-q-table", type=Path, default=DEFAULT_EFFECTIVE_Q_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--wavelength-stride", type=int, default=20)
    parser.add_argument("--step-time", default=DEFAULT_STEP_TIME)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--scale-steps", type=int, default=201)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--reference-thickness-nm", type=float, default=122.7)
    parser.add_argument("--reference-time-days", type=float, default=4.65)
    parser.add_argument("--reset-day", type=float, default=18.0)
    parser.add_argument("--max-initial-thickness-nm", type=float, default=250.0)
    parser.add_argument("--initial-thickness-starts", default="0,15,40,80,120,160,200,240")
    parser.add_argument("--thickness-grid-steps", type=int, default=1201)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.wavelength_stride < 1 or args.scale_steps < 3 or args.refinement_steps < 0:
        raise ValueError("Stride, scale steps, and refinement steps must be valid")
    if args.thickness_grid_steps < 2:
        raise ValueError("At least two thickness grid points are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations, tile_ids, tiles_per_group = load_aggregated_tile_statistics(
        args.tile_statistics
    )
    validate_aggregate_against_all_pixel(observations, args.all_pixel_statistics)
    step_mask = parse_step_mask(observations, args.step_time)

    full_model_wavelengths, calibrated_thicknesses, empirical_ratios = read_model_table(
        args.model_path
    )
    q_wavelengths, q_values, alpha_values = load_effective_q(args.effective_q_table)
    if not np.allclose(full_model_wavelengths, q_wavelengths):
        raise ValueError("The effective q table must use the model wavelength grid")
    stride = args.wavelength_stride
    model_wavelengths = full_model_wavelengths[::stride]
    q_values = q_values[::stride]
    alpha_values = alpha_values[::stride]
    d_grid = np.linspace(0.0, float(np.max(calibrated_thicknesses)), args.thickness_grid_steps)

    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    empirical_coefficients, empirical_omegas = fit_periodic_model(
        full_model_wavelengths, calibrated_thicknesses, empirical_ratios
    )
    empirical_response = broadband_response_curves(
        (
            model_wavelengths,
            calibrated_thicknesses,
            empirical_coefficients[::stride],
            empirical_omegas[::stride],
        ),
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
    )[args.sed_mode]
    effective_response = build_wfi01_response(
        model_wavelengths,
        q_values,
        alpha_values,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
        args.sed_mode,
        np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    if effective_response is None:
        raise ValueError("The fixed effective-boundary response was invalid")

    empirical = fit_response(observations, empirical_response, d_grid, step_mask, args)
    effective = fit_response(observations, effective_response, d_grid, step_mask, args)
    write_predictions(args.output_dir / "predictions.csv", observations, empirical, effective)
    write_all_led_plot(args.output_dir / "all_led_fit.png", observations, empirical, effective, step_mask)
    write_summary(
        args.output_dir / "comparison.md",
        args,
        observations,
        tile_ids,
        tiles_per_group,
        empirical,
        effective,
    )
    comparison = {
        "tile_count": len(tile_ids),
        "tiles_per_group": tiles_per_group,
        "observation_count": len(observations),
        "wavelength_stride": args.wavelength_stride,
        "calibrated_response": fit_summary(empirical),
        "fixed_effective_boundary": fit_summary(effective),
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        f"Calibrated response: rate={empirical['mean_rate_nm_day']:.8g} nm/day; "
        f"RMS={empirical['pixel_rms_log']:.8g}"
    )
    print(
        f"Fixed effective boundary: rate={effective['mean_rate_nm_day']:.8g} nm/day; "
        f"RMS={effective['pixel_rms_log']:.8g}"
    )
    print(f"Wrote {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
