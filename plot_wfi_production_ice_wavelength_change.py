"""Plot production-TMM sensitivity change between two calendar dates.

The production fit supplies the tile trajectory and optical parameters.  This
script evaluates the fitted coated power on a regular wavelength grid for both
date-specific ice thicknesses and plots their wavelength-dependent ratio.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, time
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production


ROMAN_EFFECTIVE_AREA_PATH = (
    Path(__file__).resolve().parent
    / "external_reference_data/Roman_effarea_v8_SCA05_20240301.ecsv"
)
ROMAN_EFFECTIVE_AREA_URL = (
    "https://raw.githubusercontent.com/RomanSpaceTelescope/"
    "roman-technical-information/main/roman_technical_information/data/"
    "WideFieldInstrument/Imaging/EffectiveAreas/"
    "Roman_effarea_v8_SCA05_20240301.ecsv"
)
ROMAN_FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213")
ROMAN_FILTER_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--date-a", default="2026-09-28")
    parser.add_argument("--date-b", default="2026-10-15")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--wavelength-min-nm", type=float, default=480.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=2700.0)
    parser.add_argument("--wavelength-step-nm", type=float, default=2.0)
    parser.add_argument("--bandpass-file", type=Path, default=ROMAN_EFFECTIVE_AREA_PATH)
    parser.add_argument("--bandpass-output", type=Path)
    return parser.parse_args()


def scalar(archive, name, default=None):
    if name not in archive:
        if default is None:
            raise KeyError(f"Production archive is missing {name}")
        return default
    return np.asarray(archive[name]).reshape(-1)[0].item()


def parse_calendar_date(value):
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"Expected an ISO calendar date, got {value!r}") from error


def target_day(calendar_date, first_exposure):
    target = datetime.combine(
        calendar_date,
        time.min,
        tzinfo=first_exposure.tzinfo,
    )
    return (target - first_exposure).total_seconds() / 86400.0


def thickness_at_day(day_value, node_days, node_thickness, node_rates):
    if day_value < node_days[0]:
        thickness = node_thickness[0] + node_rates[0] * (day_value - node_days[0])
        return float(thickness), "linear extrapolation before first fitted node"
    if day_value > node_days[-1]:
        thickness = node_thickness[-1] + node_rates[-1] * (day_value - node_days[-1])
        return float(thickness), "linear extrapolation at final fitted rate"
    return float(np.interp(day_value, node_days, node_thickness)), "fitted trajectory interpolation"


def load_fit(fit_dir, statistics_path=None):
    fit_dir = Path(fit_dir)
    archive_path = fit_dir / "posterior_samples.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    archive = np.load(archive_path, allow_pickle=False)
    if statistics_path is None:
        statistics_path = Path(str(scalar(archive, "statistics_path")))
    statistics_path = Path(statistics_path)
    if not statistics_path.is_file():
        raise FileNotFoundError(statistics_path)
    data = production.load_tile_statistics(statistics_path)
    return fit_dir, archive, statistics_path, data


def date_thicknesses(archive, data, date_a, date_b):
    first_exposure = datetime.fromisoformat(data["exposure_times"][0].replace("Z", "+00:00"))
    node_days = np.asarray(data["node_days"], dtype=float)
    node_thickness = np.asarray(archive["node_ice_thickness"], dtype=float).reshape(-1)
    node_rates = np.asarray(archive["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
    if node_thickness.size != node_days.size or node_rates.size != node_days.size:
        raise ValueError("Production trajectory arrays do not match the statistics nodes")
    day_a = target_day(date_a, first_exposure)
    day_b = target_day(date_b, first_exposure)
    thickness_a, method_a = thickness_at_day(day_a, node_days, node_thickness, node_rates)
    thickness_b, method_b = thickness_at_day(day_b, node_days, node_thickness, node_rates)
    return {
        "first_exposure": first_exposure,
        "day_a": day_a,
        "day_b": day_b,
        "thickness_a": thickness_a,
        "thickness_b": thickness_b,
        "method_a": method_a,
        "method_b": method_b,
        "node_days_min": float(node_days[0]),
        "node_days_max": float(node_days[-1]),
        "final_rate_nm_day": float(node_rates[-1]),
    }


def load_roman_bandpasses(path):
    """Load normalized Roman effective-area curves from an ECSV file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = list(csv.DictReader(lines, skipinitialspace=True))
    if not rows:
        raise ValueError(f"Roman bandpass file has no data rows: {path}")
    columns = {name.strip(): name for name in rows[0] if name is not None}
    if "Wave" not in columns or any(name not in columns for name in ROMAN_FILTERS):
        raise ValueError("Roman bandpass file is missing required filter columns")
    wavelengths_nm = np.asarray(
        [float(row[columns["Wave"]]) * 1000.0 for row in rows], dtype=float
    )
    if np.any(~np.isfinite(wavelengths_nm)) or np.any(np.diff(wavelengths_nm) <= 0.0):
        raise ValueError("Roman bandpass wavelengths must be finite and increasing")
    bandpasses = {}
    for name in ROMAN_FILTERS:
        values = np.asarray([float(row[columns[name]]) for row in rows], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"Roman bandpass {name} contains invalid effective areas")
        peak = float(np.max(values))
        if peak <= 0.0:
            raise ValueError(f"Roman bandpass {name} has no positive effective area")
        bandpasses[name] = values / peak
    return wavelengths_nm, bandpasses


def integrate_bandpass_response(
    bandpass_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    log_ratio,
):
    """Integrate each effective area curve against the relative response."""
    response_wavelengths_nm = np.asarray(response_wavelengths_nm, dtype=float)
    relative_response = np.exp(np.asarray(log_ratio, dtype=float))
    overlap = (
        (bandpass_wavelengths_nm >= response_wavelengths_nm[0])
        & (bandpass_wavelengths_nm <= response_wavelengths_nm[-1])
    )
    if np.count_nonzero(overlap) < 2:
        raise ValueError("Roman bandpass and sensitivity wavelength grids do not overlap")
    wavelengths_nm = bandpass_wavelengths_nm[overlap]
    interpolated_response = np.interp(
        wavelengths_nm, response_wavelengths_nm, relative_response
    )
    results = []
    for name in ROMAN_FILTERS:
        effective_area = bandpasses[name][overlap]
        integrated_clean = float(np.trapezoid(effective_area, wavelengths_nm))
        integrated_response = float(
            np.trapezoid(effective_area * interpolated_response, wavelengths_nm)
        )
        if integrated_clean <= 0.0:
            raise ValueError(f"Roman bandpass {name} has no overlap with the response")
        effective_wavelength_nm = float(
            np.trapezoid(effective_area * wavelengths_nm, wavelengths_nm)
            / integrated_clean
        )
        results.append(
            {
                "filter": name,
                "effective_wavelength_nm": effective_wavelength_nm,
                "integrated_clean_bandpass": integrated_clean,
                "integrated_bandpass_times_response": integrated_response,
                "relative_bandpass_response": integrated_response / integrated_clean,
            }
        )
    return results


def production_spectral_ratio(
    archive,
    thickness_a,
    thickness_b,
    wavelength_min_nm,
    wavelength_max_nm,
    wavelength_step_nm,
):
    """Evaluate the production TMM on a regular grid for two thicknesses."""
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    native_wavelengths = np.asarray(
        archive["response_wavelengths_nm"], dtype=float
    ).reshape(-1)
    if native_wavelengths.size < 2 or np.any(np.diff(native_wavelengths) <= 0.0):
        raise ValueError("Production wavelength grid is invalid")
    if (
        not np.isfinite(wavelength_min_nm)
        or not np.isfinite(wavelength_max_nm)
        or wavelength_min_nm <= 0.0
        or wavelength_max_nm <= wavelength_min_nm
    ):
        raise ValueError("Wavelength bounds must be positive, finite, and ordered")
    if not np.isfinite(wavelength_step_nm) or wavelength_step_nm <= 0.0:
        raise ValueError("Wavelength step must be positive and finite")
    wavelengths = np.arange(
        wavelength_min_nm,
        wavelength_max_nm,
        float(wavelength_step_nm),
        dtype=float,
    )
    if wavelengths.size == 0 or wavelengths[-1] < wavelength_max_nm:
        wavelengths = np.append(wavelengths, wavelength_max_nm)

    wavelength_count = wavelengths.size
    # Treat each wavelength as a virtual SCA with one reference and one current
    # observation. The single synthetic source channel keeps this evaluation
    # independent of the production LED-integrated SEDs.
    model_weights = np.zeros((wavelength_count, 1, wavelength_count), dtype=float)
    for index in range(wavelength_count):
        model_weights[index, 0, index] = 1.0
    data = {
        "sca_index": np.arange(wavelength_count, dtype=int),
        "led": np.ones(2, dtype=int),
        "first_by_led": np.asarray([0], dtype=int),
        "counts": np.ones((wavelength_count, 2), dtype=float),
        "sums": np.zeros((wavelength_count, 2), dtype=float),
        "sum_squared": np.zeros((wavelength_count, 2), dtype=float),
        "days": np.zeros(2, dtype=float),
        "reset_day": production.PRODUCTION_RESET_DAY,
    }
    hgcdte_n, hgcdte_k = production.hgcdte_reference_optical_constants(wavelengths)
    context = {
        "wavelengths": wavelengths,
        "model_weights": model_weights,
        "hgcdte_n": hgcdte_n,
        "hgcdte_k": hgcdte_k,
        "ice_n_modulation_basis": production.ice_n_modulation_basis(wavelengths),
        "spectral_modulation_basis": production.spectral_modulation_basis(wavelengths),
    }

    fit_ice_n_constant = bool(scalar(archive, "fit_ice_n_constant", 0))
    fit_ice_k_constant = bool(scalar(archive, "fit_ice_k_constant", 0))
    fit_ar_hgcdte_modulation = bool(scalar(archive, "fit_ar_hgcdte_modulation", 0))
    fit_incidence_angle = bool(scalar(archive, "fit_incidence_angle", 0))
    if fit_incidence_angle:
        incidence_angle = float(scalar(archive, "incidence_angle_deg"))
    else:
        incidence_angle = 0.0
    pupil_half_angle = float(scalar(archive, "incidence_pupil_half_angle_deg", 0.0))

    predictor = production.build_prediction_function(
        data,
        context,
        incidence_pupil_half_angle_deg=pupil_half_angle,
        fit_ice_n_constant=fit_ice_n_constant,
        fit_ice_k_constant=fit_ice_k_constant,
        fit_ar_hgcdte_modulation=fit_ar_hgcdte_modulation,
    )
    ar_parameters = np.asarray(archive["ar_parameters"], dtype=float).reshape(-1, 9)[0]
    hgcdte_parameters = np.asarray(archive["hgcdte"], dtype=float).reshape(-1, 2)[0]
    thickness = np.column_stack(
        (
            np.full((wavelength_count, 1), thickness_a, dtype=float),
            np.full((wavelength_count, 1), thickness_b, dtype=float),
        )
    )
    kwargs = {}
    if fit_ice_n_constant:
        kwargs["ice_n_log_constant"] = np.full(
            (wavelength_count, 1),
            float(scalar(archive, "ice_n_log_constant")),
        )
    if fit_ice_k_constant:
        kwargs["ice_k_constant"] = np.full(
            (wavelength_count, 1),
            float(scalar(archive, "ice_k_constant")),
        )
    if fit_ar_hgcdte_modulation:
        ar_modulation = np.asarray(archive["ar_n_log_modulation"], dtype=float).reshape(-1, 3, 6)[0]
        hgcdte_n_modulation = np.asarray(
            archive["hgcdte_n_log_modulation"], dtype=float
        ).reshape(-1, 6)[0]
        hgcdte_k_modulation = np.asarray(
            archive["hgcdte_k_log_modulation"], dtype=float
        ).reshape(-1, 6)[0]
        kwargs.update(
            ar_n_log_modulation=np.broadcast_to(
                ar_modulation, (wavelength_count, 3, 6)
            ),
            hgcdte_n_log_modulation=np.broadcast_to(
                hgcdte_n_modulation, (wavelength_count, 6)
            ),
            hgcdte_k_log_modulation=np.broadcast_to(
                hgcdte_k_modulation, (wavelength_count, 6)
            ),
        )

    prediction = np.asarray(
        jax.jit(predictor)(
            jnp.asarray(thickness),
            jnp.broadcast_to(jnp.asarray(ar_parameters), (wavelength_count, 9)),
            jnp.broadcast_to(jnp.asarray(hgcdte_parameters), (wavelength_count, 2)),
            incidence_angle,
            **{name: jnp.asarray(value) for name, value in kwargs.items()},
        ),
        dtype=float,
    )
    log_ratio = prediction[:, 1]
    if log_ratio.shape != (wavelength_count,):
        raise ValueError("Production TMM returned an unexpected spectral shape")
    return wavelengths, log_ratio, {
        "fit_ice_n_constant": fit_ice_n_constant,
        "fit_ice_k_constant": fit_ice_k_constant,
        "fit_ar_hgcdte_modulation": fit_ar_hgcdte_modulation,
        "incidence_angle_deg": incidence_angle,
        "incidence_pupil_half_angle_deg": pupil_half_angle,
        "ice_n_constant": (
            production.ICE_INDEX_DEFAULT
            * np.exp(float(scalar(archive, "ice_n_log_constant")))
            if fit_ice_n_constant
            else production.ICE_INDEX_DEFAULT
        ),
    }


def write_csv(path, wavelengths, log_ratio):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "wavelength_nm",
                "wavelength_um",
                "relative_sensitivity",
                "relative_log_sensitivity",
                "percent_change",
            ),
        )
        writer.writeheader()
        for wavelength, log_value in zip(wavelengths, log_ratio):
            ratio = float(np.exp(log_value))
            writer.writerow(
                {
                    "wavelength_nm": f"{wavelength:.12g}",
                    "wavelength_um": f"{wavelength / 1000.0:.12g}",
                    "relative_sensitivity": f"{ratio:.12g}",
                    "relative_log_sensitivity": f"{log_value:.12g}",
                    "percent_change": f"{100.0 * (ratio - 1.0):.12g}",
                }
            )


def write_bandpass_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "filter",
        "effective_wavelength_nm",
        "integrated_clean_bandpass",
        "integrated_bandpass_times_response",
        "relative_bandpass_response",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: (
                        result[key]
                        if key == "filter"
                        else f"{result[key]:.12g}"
                    )
                    for key in fieldnames
                }
            )


def write_plot(
    path,
    wavelengths,
    log_ratio,
    details,
    detector,
    tile_id,
    date_a,
    date_b,
    bandpass_wavelengths_nm,
    bandpasses,
    bandpass_results,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    ratio = np.exp(log_ratio)
    wavelength_um = wavelengths / 1000.0
    bandpass_wavelength_um = bandpass_wavelengths_nm / 1000.0

    figure, (axis, bandpass_axis) = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (4.2, 1.6)},
    )
    for name, color in zip(ROMAN_FILTERS, ROMAN_FILTER_COLORS):
        support = bandpasses[name] > 0.01
        if np.any(support):
            axis.axvspan(
                bandpass_wavelength_um[support][0],
                bandpass_wavelength_um[support][-1],
                color=color,
                alpha=0.035,
                zorder=0,
            )
    axis.plot(
        wavelength_um,
        ratio,
        color="#006d77",
        linewidth=2.2,
        label="Production TMM wavelength function",
    )
    bandpass_marker_wavelength_um = np.asarray(
        [result["effective_wavelength_nm"] for result in bandpass_results],
        dtype=float,
    ) / 1000.0
    bandpass_marker_response = np.asarray(
        [result["relative_bandpass_response"] for result in bandpass_results],
        dtype=float,
    )
    axis.scatter(
        bandpass_marker_wavelength_um,
        bandpass_marker_response,
        color="#111111",
        edgecolor="white",
        linewidth=0.8,
        s=42,
        zorder=4,
        label="Bandpass-integrated response",
    )
    for result in bandpass_results:
        axis.annotate(
            result["filter"],
            (
                result["effective_wavelength_nm"] / 1000.0,
                result["relative_bandpass_response"],
            ),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#222222",
        )
    axis.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--", label="No change")
    axis.set_xlim(wavelength_um[0], wavelength_um[-1])
    axis.set_ylabel(f"Relative sensitivity, S({date_b.isoformat()}) / S({date_a.isoformat()})")
    axis.set_title(
        f"Ice-model wavelength sensitivity change: {detector} tile {tile_id:03d}",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="best")
    axis.text(
        0.02,
        0.96,
        (
            f"{date_a.isoformat()}: {details['thickness_a']:.2f} nm\n"
            f"{date_b.isoformat()}: {details['thickness_b']:.2f} nm\n"
            f"n_ice = {details['ice_n_constant']:.5f}"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    for name, color in zip(ROMAN_FILTERS, ROMAN_FILTER_COLORS):
        normalized_area = bandpasses[name]
        bandpass_axis.fill_between(
            bandpass_wavelength_um,
            0.0,
            normalized_area,
            color=color,
            alpha=0.12,
        )
        bandpass_axis.plot(
            bandpass_wavelength_um,
            normalized_area,
            color=color,
            linewidth=1.1,
            label=name,
        )
    bandpass_axis.set_ylim(-0.02, 1.08)
    bandpass_axis.set_ylabel("Normalized\neffective area")
    bandpass_axis.set_xlabel("Wavelength (um)")
    bandpass_axis.set_title(
        "Roman WFI bandpasses, SCA05 effective area",
        loc="left",
        fontsize=10,
    )
    bandpass_axis.grid(alpha=0.22)
    bandpass_axis.legend(
        ncol=4,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    args = parse_args()
    date_a = parse_calendar_date(args.date_a)
    date_b = parse_calendar_date(args.date_b)
    if date_b <= date_a:
        raise ValueError("date-b must be later than date-a")
    fit_dir, archive, statistics_path, data = load_fit(args.fit_dir, args.statistics)
    details = date_thicknesses(archive, data, date_a, date_b)
    bandpass_wavelengths_nm, bandpasses = load_roman_bandpasses(args.bandpass_file)
    wavelengths, log_ratio, optical_details = production_spectral_ratio(
        archive,
        details["thickness_a"],
        details["thickness_b"],
        args.wavelength_min_nm,
        args.wavelength_max_nm,
        args.wavelength_step_nm,
    )
    details.update(optical_details)
    bandpass_results = integrate_bandpass_response(
        bandpass_wavelengths_nm,
        bandpasses,
        wavelengths,
        log_ratio,
    )
    detector = str(scalar(archive, "sca_names"))
    tile_id = int(scalar(archive, "tile_ids"))
    stem = f"ice_response_ratio_{date_b:%Y%m%d}_vs_{date_a:%Y%m%d}"
    output = args.output or fit_dir / "plots" / f"{stem}.png"
    csv_output = args.csv_output or output.with_suffix(".csv")
    metadata_output = args.metadata_output or output.with_suffix(".json")
    bandpass_output = args.bandpass_output or output.with_name(f"{stem}_bandpass_integrals.csv")

    write_csv(csv_output, wavelengths, log_ratio)
    write_bandpass_csv(bandpass_output, bandpass_results)
    write_plot(
        output,
        wavelengths,
        log_ratio,
        details,
        detector,
        tile_id,
        date_a,
        date_b,
        bandpass_wavelengths_nm,
        bandpasses,
        bandpass_results,
    )
    metadata = {
        "fit_dir": str(fit_dir.resolve()),
        "statistics": str(statistics_path.resolve()),
        "detector": detector,
        "tile_id": tile_id,
        "date_a": date_a.isoformat(),
        "date_b": date_b.isoformat(),
        "date_interpretation": "calendar-day midnight in the statistics time basis",
        "first_exposure": details["first_exposure"].isoformat(),
        "day_a_since_first_exposure": details["day_a"],
        "day_b_since_first_exposure": details["day_b"],
        "ice_thickness_a_nm": details["thickness_a"],
        "ice_thickness_b_nm": details["thickness_b"],
        "thickness_method_a": details["method_a"],
        "thickness_method_b": details["method_b"],
        "fitted_node_day_range": [details["node_days_min"], details["node_days_max"]],
        "final_fitted_rate_nm_day": details["final_rate_nm_day"],
        "response_definition": "production TMM coated power ratio at each wavelength; clean power cancels",
        "roman_bandpass_file": str(args.bandpass_file.resolve()),
        "roman_bandpass_source_url": ROMAN_EFFECTIVE_AREA_URL,
        "roman_bandpass_definition": "SCA05 effective area curves normalized independently to each filter peak",
        "roman_bandpass_filters": list(ROMAN_FILTERS),
        "bandpass_integral_definition": "integral(effective_area * relative_response) / integral(effective_area), clipped to the plotted wavelength range",
        "bandpass_integral_output": str(bandpass_output.resolve()),
        "bandpass_integrated_results": bandpass_results,
        "wavelength_grid_count": int(wavelengths.size),
        "wavelength_min_nm": float(wavelengths[0]),
        "wavelength_max_nm": float(wavelengths[-1]),
        "wavelength_step_nm": float(args.wavelength_step_nm),
        **{key: value for key, value in details.items() if key.startswith("fit_") or key.startswith("incidence_") or key == "ice_n_constant"},
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")
    print(f"Wrote {bandpass_output}")
    print(f"Wrote {metadata_output}")
    print(
        f"Ice thickness: {date_a}={details['thickness_a']:.6f} nm; "
        f"{date_b}={details['thickness_b']:.6f} nm"
    )


if __name__ == "__main__":
    main()
