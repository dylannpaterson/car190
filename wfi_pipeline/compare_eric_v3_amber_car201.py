"""Compare Eric's original v3 model and our joint-modulated model on Amber CAR201.

Both models are evaluated through Eric's v3 ``IceResponseModel`` API on the
exact raw common cohort used by the joint CAR190/Amber fit.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import h5py
from astropy.time import Time

import heatmap_car201_allstar_model_correction as workflow
import plot_joint_amber_car190_detailed as amber_plots
import run_amber_student_t_detectors as amber
import run_joint_car190_amber_inverse_wavelength_maxent as joint_fit
from heatmap_car201_allstar_model_correction import DETECTORS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = (
    ROOT / "20261111/corrected_statistics_4h_thinned/production/wavelength_response_bundle"
)
FIT_NAME = "relative_response/experiments/car190_amber_joint_2day_reciprocal_maxent"
DEFAULT_STATISTICS = ROOT / "20261111/corrected_statistics_4h_thinned"
DEFAULT_SOURCE_ZIP = ROOT / "external_reference_data/eric_reader/RST_commissioning-main.zip"
DEFAULT_ERIC_V3_MODEL = BUNDLE_DIR / "eric_v3_reference/ice_response_model_v3.h5"
DEFAULT_JOINT_MODULATED_MODEL = BUNDLE_DIR / "unchanged_reader_port/joint_modulated.h5"
DEFAULT_OUTPUT = BUNDLE_DIR / "unchanged_reader_port/car201"
VARIANTS = ("no_ice_correction", "eric_v3", "joint_modulated")
VARIANT_LABELS = {
    "no_ice_correction": "Amber observed; no ice correction",
    "eric_v3": "Eric's original v3 model",
    "joint_modulated": "Our joint-modulated H5 — unchanged reader",
    "saved_joint_fit_reference": "Our joint fit — full ice histories, before export",
}


def utc_mjd(value: str) -> float:
    """Parse Eric model UTC epochs, which may carry a trailing UTC offset."""
    value = str(value).strip()
    if value.endswith("Z"):
        value = value[:-1]
    if value.endswith("+00:00"):
        value = value[:-6]
    return float(Time(value, format="isot", scale="utc").mjd)


@contextmanager
def load_eric_v3_reader(source_zip: Path):
    """Load Eric's reader/core from the supplied source archive, without vendoring it."""
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    with zipfile.ZipFile(source_zip) as archive, tempfile.TemporaryDirectory(
        prefix="eric_v3_reader_"
    ) as temporary:
        members = archive.namelist()
        core_member = next(
            (name for name in members if name.endswith("/ice_response_core.py")), None
        )
        reader_member = next(
            (name for name in members if name.endswith("/ice_response_model.py")), None
        )
        if core_member is None or reader_member is None:
            raise ValueError(f"{source_zip} does not contain Eric's v3 reader and core")
        source_dir = Path(temporary)
        (source_dir / "ice_response_core.py").write_bytes(archive.read(core_member))
        reader_path = source_dir / "ice_response_model.py"
        reader_path.write_bytes(archive.read(reader_member))
        sys.path.insert(0, str(source_dir))
        try:
            spec = importlib.util.spec_from_file_location(
                "eric_v3_ice_response_model_for_amber", reader_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load Eric v3 reader from {reader_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.path.remove(str(source_dir))
            sys.modules.pop("ice_response_core", None)


def check_joint_fit_sample(detector: str, statistics_dir: Path, fit_name=FIT_NAME):
    """Rebuild and verify the exact saved joint-fit input cohort."""
    source_ids, seds, magnitude, _error, valid, common, _frames = (
        joint_fit.raw_common_training_sample(detector)
    )
    fit_dir = statistics_dir / "production" / detector.lower() / fit_name
    with np.load(fit_dir / "joint_fit.npz", allow_pickle=False) as archive:
        fit_ids = np.asarray(archive["source_ids"]).astype(str)
        fit_valid = np.asarray(archive["stellar_valid"], dtype=bool)
        fit_prediction = np.asarray(archive["stellar_prediction_mag"], dtype=float)
        parameters = np.asarray(archive["parameters"], dtype=float)
        basis = np.asarray(archive["basis"], dtype=float)
    with np.load(fit_dir / "training_inputs.npz", allow_pickle=False) as archive:
        fit_magnitude = np.asarray(archive["magnitude"], dtype=float)
        fit_input_valid = np.asarray(archive["valid"], dtype=bool)
    np.testing.assert_array_equal(source_ids, fit_ids)
    np.testing.assert_array_equal(valid, fit_valid)
    np.testing.assert_array_equal(valid, fit_input_valid)
    np.testing.assert_allclose(magnitude, fit_magnitude, rtol=0.0, atol=1e-10)
    if fit_prediction.shape != magnitude.shape:
        raise ValueError(f"{detector}: saved joint-fit prediction shape differs from inputs")
    pass_groups = np.asarray(
        joint_fit.run.workflow.fit_model.PASS_SET_CONFIG["ep3v2_ep4"]["zero_point_group_ids"],
        dtype=int,
    )
    mode_count = basis.shape[0]
    group_count = int(pass_groups.max())
    zero_point_count = parameters.size - 2 * mode_count
    if zero_point_count != len(amber.FILTERS) * group_count:
        raise ValueError(f"{detector}: saved joint-fit roll-offset layout is not supported")
    zero_points = parameters[2 * mode_count:].reshape(len(amber.FILTERS), group_count)
    roll = np.zeros_like(magnitude)
    for pass_index, group in enumerate(pass_groups):
        if group:
            roll[:, :, pass_index] = (
                -zero_points[:, group - 1] / workflow.LOG_FLUX_PER_MAG
            )[None, :]
    np.testing.assert_allclose(roll[:, :, 0], 0.0, rtol=0.0, atol=0.0)
    summary = json.loads((fit_dir / "summary.json").read_text())
    if not summary.get("optimizer_success"):
        raise RuntimeError(f"{detector}: source joint fit did not report optimizer success")
    return source_ids, seds, magnitude, valid, common, fit_dir, summary, roll, fit_prediction


def effective_area_on_model_grid(detector: str, area_path: Path, wavelengths_nm):
    area_wavelengths, area_by_filter = workflow.mock_model.load_effective_area(area_path)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    supported = np.flatnonzero(np.any(area_by_filter > 0.0, axis=0))
    if not supported.size:
        raise ValueError(f"{detector}: empty effective-area support in {area_path}")
    support_min = area_wavelengths[supported[0]]
    support_max = area_wavelengths[supported[-1]]
    native_step = float(np.median(np.diff(area_wavelengths)))
    lower_overhang = max(0.0, float(wavelengths_nm[0] - support_min))
    upper_overhang = max(0.0, float(support_max - wavelengths_nm[-1]))
    if lower_overhang > 5.0 * native_step + 1e-9 or upper_overhang > 5.0 * native_step + 1e-9:
        raise ValueError(
            f"{detector}: effective-area support {support_min:g}–{support_max:g} nm "
            f"exceeds v3 wavelength grid {wavelengths_nm[0]:g}–{wavelengths_nm[-1]:g} nm"
        )
    result = np.asarray(
        [np.interp(wavelengths_nm, area_wavelengths, band, left=0.0, right=0.0)
         for band in area_by_filter],
        dtype=float,
    )
    if result.shape[0] != len(amber.FILTERS):
        raise ValueError(f"{detector}: effective-area filter ordering/dimension mismatch")
    full_integral = np.trapezoid(area_by_filter, area_wavelengths, axis=1)
    retained_samples = np.asarray(
        [np.interp(area_wavelengths, wavelengths_nm, band, left=0.0, right=0.0)
         for band in result],
        dtype=float,
    )
    retained_integral = np.trapezoid(retained_samples, area_wavelengths, axis=1)
    retained_fractions = {
        band: float(np.clip(retained / full, 0.0, 1.0)) if full > 0.0 else np.nan
        for band, retained, full in zip(amber.FILTERS, retained_integral, full_integral, strict=True)
    }
    if min(retained_fractions.values()) < 0.9999:
        raise ValueError(
            f"{detector}: v3 wavelength-grid truncation omits more than 0.01% "
            "of an effective-area integral"
        )
    support_info = {
        "effective_area_support_min_nm": float(support_min),
        "effective_area_support_max_nm": float(support_max),
        "v3_wavelength_min_nm": float(wavelengths_nm[0]),
        "v3_wavelength_max_nm": float(wavelengths_nm[-1]),
        "lower_support_overhang_nm": lower_overhang,
        "upper_support_overhang_nm": upper_overhang,
        "retained_effective_area_integral_fraction_by_filter": retained_fractions,
        "max_unmodelled_effective_area_integral_fraction": float(
            1.0 - min(retained_fractions.values())
        ),
    }
    return result, support_info


def evaluate_model_on_sample(
    model,
    core,
    detector: str,
    source_ids,
    seds,
    common: pd.DataFrame,
    wavelengths_nm,
    photon_shapes,
    area_by_filter,
    normalize_band=True,
    fitted_photometry=False,
):
    """Integrate v3 ice response over each matched source/filter/time measurement."""
    sca = int(detector[3:])
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    weights = workflow.fit_model.trapezoid_weights(wavelengths_nm)
    source_lookup = {source: index for index, source in enumerate(source_ids)}
    filter_lookup = {name: index for index, name in enumerate(amber.FILTERS)}
    pass_lookup = {name: index for index, name in enumerate(amber.PASSES)}
    shape = (len(source_ids), len(amber.FILTERS), len(amber.PASSES))

    source_index = common["source_id"].map(source_lookup).to_numpy(dtype=int)
    filter_index = common["filter"].map(filter_lookup).to_numpy(dtype=int)
    pass_index = common["pass"].map(pass_lookup).to_numpy(dtype=int)
    if len(set(zip(source_index, filter_index, pass_index))) != len(common):
        raise ValueError(f"{detector}: duplicate source/filter/pass cells in evaluation sample")

    x_pixel = pd.to_numeric(common["x_pixel"], errors="coerce").to_numpy(float)
    y_pixel = pd.to_numeric(common["y_pixel"], errors="coerce").to_numpy(float)
    mjd = pd.to_numeric(common["mjd_amber"], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(x_pixel) | ~np.isfinite(y_pixel) | ~np.isfinite(mjd)):
        raise ValueError(f"{detector}: nonfinite position or time in matched cohort")
    all_sp = np.asarray(model.superpixel(sca, x_pixel, y_pixel), dtype=int)
    unique_sp = np.unique(all_sp)
    sp_for_row = np.searchsorted(unique_sp, all_sp)
    source_row = model._row[unique_sp].astype(int)
    source_col = model._col[unique_sp].astype(int)
    px = int(model.superpixel_px)
    x_center = source_col * px + px // 2
    y_center = source_row * px + px // 2

    # Eric's band_response uses the same no-ice transmission as the spectral
    # integration weight, then integrates the relative ice response over it.
    reflectance = model.reflectance(sca, x_center, y_center, wavelengths_nm)
    _n, _kappa, fresnel = core.optics_at(
        model.wavelength_nm,
        model._n,
        model._kappa,
        model._b,
        wavelengths_nm,
    )
    no_ice_transmission = core.no_ice_transmission(reflectance, fresnel)
    if fitted_photometry:
        with h5py.File(model.path, 'r') as h5:
            photometry = h5['photometry']
            tile_rows = photometry['superpixel_to_tile'][unique_sp]
            tile_values, inverse = np.unique(tile_rows, return_inverse=True)
            native_wave = photometry['wavelength_nm'][()]
            curves = photometry['clean_relative_weight'][tile_values]
        no_ice_transmission = np.asarray([
            np.interp(wavelengths_nm, native_wave, curve) for curve in curves
        ])[inverse]
        normalize_band = False
    if np.any(~np.isfinite(no_ice_transmission)) or np.any(no_ice_transmission <= 0.0):
        raise ValueError(f"{detector}: invalid v3 no-ice transmission")

    reference_mjd = utc_mjd(model.t0_utc)
    relative_hours = (mjd - reference_mjd) * 24.0
    response = np.full(shape, np.nan, dtype=float)
    model_valid_rows = np.zeros(len(common), dtype=bool)

    # Integrate in filter/pass cells. The raw CAR201 data contain several exact
    # exposure times per pass; preserve those times rather than using pass medians.
    for fi, band in enumerate(amber.FILTERS):
        spectrum_area = area_by_filter[fi] * weights
        source_spectral_weights = photon_shapes * spectrum_area[None, :]
        for pi, pass_name in enumerate(amber.PASSES):
            rows = np.flatnonzero((filter_index == fi) & (pass_index == pi))
            if not rows.size:
                continue
            active_sp, row_sp = np.unique(sp_for_row[rows], return_inverse=True)
            unique_hours, row_time = np.unique(relative_hours[rows], return_inverse=True)
            valid_times = np.asarray(model.valid(unique_hours), dtype=bool).reshape(-1)
            if valid_times.size != unique_hours.size:
                raise ValueError(f"{detector}/{band}/{pass_name}: v3 validity axis mismatch")
            model_valid_rows[rows] = valid_times[row_time]

            p_response = model.response(
                sca,
                x_center[active_sp],
                y_center[active_sp],
                unique_hours,
                wavelengths_nm,
                gray=not fitted_photometry,
            )
            row_response = p_response[row_sp, row_time, :]
            row_sources = source_index[rows]
            row_no_ice = no_ice_transmission[active_sp[row_sp]]
            integration_weights = source_spectral_weights[row_sources] * row_no_ice
            denominator = integration_weights.sum(axis=1)
            numerator = np.einsum(
                "rl,rl->r", row_response, integration_weights, optimize=True
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                integrated_response = numerator / denominator if normalize_band else numerator
            good = (
                model_valid_rows[rows]
                & np.isfinite(integrated_response)
                & (integrated_response > 0.0)
                & np.isfinite(denominator)
                & (denominator > 0.0)
            )
            if np.any(np.isfinite(response[row_sources[good], fi, pi])):
                raise ValueError(f"{detector}/{band}/{pass_name}: duplicate response assignment")
            response[row_sources[good], fi, pi] = integrated_response[good]

    if np.any(model_valid_rows & ~np.isfinite(response[source_index, filter_index, pass_index])):
        raise ValueError(f"{detector}: failed to integrate one or more model-valid measurements")

    # Independently verify the custom vectorized integration against the v3 API.
    check_row = next(
        (index for index in range(len(common))
         if model_valid_rows[index] and np.isfinite(response[source_index[index],
                                                            filter_index[index],
                                                            pass_index[index]])),
        None,
    )
    if check_row is None:
        raise ValueError(f"{detector}: no model-valid Amber measurements")
    check_source = source_index[check_row]
    check_filter = filter_index[check_row]
    check_pass = pass_index[check_row]
    check_hour = np.asarray([relative_hours[check_row]], dtype=float)
    check_weight = (
        photon_shapes[check_source] * area_by_filter[check_filter] * weights
    )
    if fitted_photometry:
        check_weight = check_weight * no_ice_transmission[sp_for_row[check_row]]
    api_response = model.band_response(
        sca,
        np.asarray([x_pixel[check_row]]),
        np.asarray([y_pixel[check_row]]),
        check_hour,
        wavelengths_nm,
        check_weight,
        gray=not fitted_photometry,
        no_ice_weight=not fitted_photometry,
    )[0, 0]
    if not normalize_band:
        check_reflection = model.reflectance(
            sca, [x_pixel[check_row]], [y_pixel[check_row]], wavelengths_nm
        )
        api_response *= (
            np.sum(check_weight) if fitted_photometry else np.sum(
                check_weight * core.no_ice_transmission(check_reflection, fresnel)[0]
            )
        )
    np.testing.assert_allclose(
        response[check_source, check_filter, check_pass],
        api_response,
        rtol=2e-9,
        atol=2e-11,
        err_msg=(
            f"{detector}: direct integration {response[check_source, check_filter, check_pass]:.12g} "
            f"disagrees with Eric v3 band_response {api_response:.12g} "
            f"at source {source_ids[check_source]}, {amber.FILTERS[check_filter]}/"
            f"{amber.PASSES[check_pass]}"
        ),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        log_response = np.log(response)
    predicted_delta_mag = -(
        log_response - log_response[:, :, :1]
    ) / workflow.LOG_FLUX_PER_MAG
    model_valid_cells = np.isfinite(response)
    return predicted_delta_mag, model_valid_cells, {
        "checked_band_response_relative_error": float(
            abs(response[check_source, check_filter, check_pass] - api_response)
            / max(abs(api_response), 1e-300)
        ),
        "distinct_superpixels_used": int(unique_sp.size),
        "model_valid_measurements": int(model_valid_rows.sum()),
    }


def summarize_metrics(detector, magnitude, valid, model_valid, variants):
    comparison = valid & valid[:, :, :1]
    comparison[:, :, 0] = False
    comparison &= model_valid
    comparison &= np.isfinite(magnitude)
    comparison &= np.isfinite(magnitude[:, :, :1])
    records = []
    for variant, corrected in variants.items():
        delta_mag = corrected - corrected[:, :, :1]
        flux_change = 100.0 * np.expm1(-workflow.LOG_FLUX_PER_MAG * delta_mag)
        for band_index, band in enumerate(amber.FILTERS):
            selected = comparison[:, band_index, :]
            values = flux_change[:, band_index, :][selected]
            values = values[np.isfinite(values)]
            records.append(metric_record(detector, band, variant, values))
        values = flux_change[comparison]
        values = values[np.isfinite(values)]
        records.append(metric_record(detector, "ALL_FILTERS", variant, values))
    return records, comparison


def metric_record(detector, band, variant, values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values)) if values.size else np.nan
    return {
        "detector": detector,
        "filter": band,
        "variant": variant,
        "matched_source_filter_visit_values": int(values.size),
        "median_residual_flux_change_percent": median,
        "p16_residual_flux_change_percent": float(np.quantile(values, 0.16)) if values.size else np.nan,
        "p84_residual_flux_change_percent": float(np.quantile(values, 0.84)) if values.size else np.nan,
        "rms_residual_flux_change_percent": float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan,
        "robust_rms_residual_flux_change_percent": (
            float(1.4826 * np.median(np.abs(values - median))) if values.size else np.nan
        ),
        "fraction_within_0p5_percent": (
            float(np.mean(np.abs(values) <= 0.5)) if values.size else np.nan
        ),
    }


def plot_comparison(cells: pd.DataFrame, output_dir: Path, detectors, output_suffix: str):
    if cells.empty:
        raise ValueError("No Amber visit-change cells available to plot")
    detector_order = [detector for detector in DETECTORS if detector in set(detectors)]
    colors = plt.get_cmap("tab20")(np.arange(len(detector_order)))
    lower = float(cells["p16_percent"].min())
    upper = float(cells["p84_percent"].max())
    margin = 0.05 * max(upper - lower, 1.0)
    suffix = '' if output_suffix == 'amber_roll_adjusted' else '_without_roll_offsets'
    pdf_path = output_dir / f"car201_comparison{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_variants = VARIANTS + (
            ('saved_joint_fit_reference',) if 'saved_joint_fit_reference' in set(cells['variant']) else ()
        )
        for variant in plot_variants:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
            for band, axis in zip(amber.FILTERS, axes.flat, strict=True):
                axis.axhspan(-0.5, 0.5, color="#65ac68", alpha=0.14)
                axis.axhline(0.0, color="0.4", linewidth=0.6)
                for detector, color in zip(detector_order, colors, strict=True):
                    selected = cells.loc[
                        (cells["variant"] == variant)
                        & (cells["filter"] == band)
                        & (cells["detector"] == detector)
                    ]
                    offset = (detector_order.index(detector) - (len(detector_order) - 1) / 2) / 96.0
                    axis.errorbar(
                        selected["utc_date_number"] + offset,
                        selected["median_percent"],
                        yerr=np.asarray([
                            selected["median_percent"] - selected["p16_percent"],
                            selected["p84_percent"] - selected["median_percent"],
                        ]),
                        fmt="o",
                        color=color,
                        markersize=2.4,
                        elinewidth=0.6,
                        capsize=1.0,
                    )
                axis.set_title(band)
                axis.set_ylim(lower - margin, upper + margin)
                axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                axis.tick_params(axis="x", labelrotation=50, labelsize=8)
                axis.grid(axis="y", alpha=0.2)
            for axis in axes[-1]:
                axis.set_xlabel("UTC date of ending visit")
            handles = [
                Line2D([0], [0], color=color, label=detector)
                for detector, color in zip(detector_order, colors, strict=True)
            ]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                       ncol=min(9, len(handles)), frameon=False)
            fig.suptitle(
                VARIANT_LABELS[variant] + " — exact joint-fit cohort, same-star visit changes",
                y=0.985,
                fontsize=15,
            )
            fig.supylabel("Residual relative flux change from measured EP3 pass 1 (%)")
            roll_note = (
                "Saved joint-fit Amber roll offsets applied unchanged to every variant; these offsets were fit on this cohort."
                if output_suffix == "amber_roll_adjusted"
                else "No fit-derived Amber roll offsets applied."
            )
            fig.text(
                0.5,
                0.025,
                "Points are detector medians; bars are star 16th–84th percentiles; green band is ±0.5%. "
                "Identical unchanged-reader band-response calculation for both H5s; no extra spectral weights. " + roll_note,
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0.025, 0.1, 1, 0.87))
            plot_names = {
                'no_ice_correction': 'uncorrected',
                'eric_v3': 'eric_v3_corrected',
                'joint_modulated': 'joint_modulated_corrected',
                'saved_joint_fit_reference': 'full_history_fit_corrected',
            }
            png_path = output_dir / f"{plot_names[variant]}{suffix}.png"
            fig.savefig(png_path, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--joint-fit-name", default=FIT_NAME,
                        help="Fit path below production/<detector> used for cohort and roll offsets")
    parser.add_argument("--include-fit-reference", action="store_true",
                        help="Include the fitted full-history response as a fourth diagnostic page")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE_ZIP)
    parser.add_argument("--eric-v3-model", type=Path, default=DEFAULT_ERIC_V3_MODEL)
    parser.add_argument(
        "--joint-modulated-model", type=Path, default=DEFAULT_JOINT_MODULATED_MODEL
    )
    parser.add_argument(
        "--detector",
        action="append",
        required=True,
        choices=DETECTORS,
        help="Detector to include; repeat explicitly for multiple detectors.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detectors = list(dict.fromkeys(args.detector))
    for path in (args.eric_v3_model, args.joint_modulated_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_fit_info = []
    all_plot_cells = []
    all_roll_plot_cells = []
    fleet_values = {}
    roll_fleet_values = {}
    prediction_path = args.output_dir / "amber_eric_v3_vs_joint_modulated_measurement_predictions.csv"
    prediction_header = True

    with load_eric_v3_reader(args.source_zip) as reader:
        eric_v3 = reader.IceResponseModel(args.eric_v3_model)
        joint_modulated = reader.IceResponseModel(args.joint_modulated_model)
        lower=max(float(eric_v3.wavelength_nm[0]),float(joint_modulated.wavelength_nm[0]))
        upper=min(float(eric_v3.wavelength_nm[-1]),float(joint_modulated.wavelength_nm[-1]))
        if eric_v3.n_bin != joint_modulated.n_bin:
            raise ValueError("v3 prototypes have different spatial grids")
        wavelengths_nm = np.asarray(eric_v3.wavelength_nm, dtype=float)
        wavelengths_nm = wavelengths_nm[(wavelengths_nm >= lower) & (wavelengths_nm <= upper)]
        print(f"Eric original v3 model: {eric_v3}", flush=True)
        print(f"Our joint-modulated v3 model: {joint_modulated}", flush=True)

        for detector_index, detector in enumerate(detectors, start=1):
            print(f"=== Eric v3 Amber comparison {detector} ({detector_index}/{len(detectors)}) ===",
                  flush=True)
            (
                source_ids, seds, magnitude, valid, common, _fit_dir,
                fit_summary, amber_roll, saved_fit_prediction,
            ) = check_joint_fit_sample(detector, args.statistics_dir, args.joint_fit_name)
            area_path, _ = amber.canonical_configuration(detector)
            area_by_filter, support_info = effective_area_on_model_grid(
                detector, area_path, wavelengths_nm
            )
            photon_shapes, _, fallback_count = workflow.fit_model.reconstruct_sed_photon_shapes(
                source_ids, seds, workflow.SED_GRID_PATH, wavelengths_nm
            )
            if fallback_count:
                raise ValueError(f"{detector}: unexpected SED fallback count {fallback_count}")

            predictions = {}
            model_valid = None
            integration_checks = {}
            for variant, model in (
                ("eric_v3", eric_v3),
                ("joint_modulated", joint_modulated),
            ):
                predicted_delta, this_valid, check = evaluate_model_on_sample(
                    model,
                    reader.core,
                    detector,
                    source_ids,
                    seds,
                    common,
                    wavelengths_nm,
                    photon_shapes,
                    area_by_filter,
                    fitted_photometry=False,
                )
                if model_valid is None:
                    model_valid = this_valid
                else:
                    np.testing.assert_array_equal(model_valid, this_valid)
                predictions[variant] = predicted_delta
                integration_checks[variant] = check
                print(
                    f"{detector} {variant}: {check['model_valid_measurements']} model-valid "
                    f"measurements; band_response check rel.err "
                    f"{check['checked_band_response_relative_error']:.3g}",
                    flush=True,
                )

            common_valid = valid & model_valid
            variants = {
                "no_ice_correction": magnitude,
                "eric_v3": magnitude - predictions["eric_v3"],
                "joint_modulated": magnitude - predictions["joint_modulated"],
            }
            if args.include_fit_reference:
                variants['saved_joint_fit_reference'] = magnitude - saved_fit_prediction + amber_roll
            detector_metrics, comparison_mask = summarize_metrics(
                detector, magnitude, common_valid, model_valid, variants
            )
            for record in detector_metrics:
                record["evaluation"] = "standalone_no_amber_roll"
            all_metrics.extend(detector_metrics)

            roll_adjusted_variants = {
                "no_ice_correction": magnitude - amber_roll,
                "eric_v3": magnitude - predictions["eric_v3"] - amber_roll,
                "joint_modulated": magnitude - predictions["joint_modulated"] - amber_roll,
                "saved_joint_fit_reference": magnitude - saved_fit_prediction,
            }
            roll_metrics, roll_comparison_mask = summarize_metrics(
                detector, magnitude, common_valid, model_valid, roll_adjusted_variants
            )
            np.testing.assert_array_equal(comparison_mask, roll_comparison_mask)
            for record in roll_metrics:
                record["evaluation"] = "amber_roll_adjusted_same_sample"
            all_metrics.extend(roll_metrics)

            plot_valid = comparison_mask.copy()
            plot_valid[:, :, 0] = valid[:, :, 0] & model_valid[:, :, 0]
            rows = amber_plots.amber_variant_cells(
                detector, common, magnitude, plot_valid, variants
            )
            roll_rows = amber_plots.amber_variant_cells(
                detector,
                common,
                magnitude,
                plot_valid,
                {key: roll_adjusted_variants[key] for key in (
                    VARIANTS + (('saved_joint_fit_reference',) if args.include_fit_reference else ())
                )},
            )
            fit_info = {
                "detector": detector,
                "training_stars": int(len(source_ids)),
                "training_measurements": int(valid.sum()),
                "model_valid_measurements": int((valid & model_valid).sum()),
                "matched_source_filter_visit_values": int(comparison_mask.sum()),
                "effective_area_path": str(area_path),
                "effective_area_support_json": json.dumps(support_info, sort_keys=True),
                "training_selection": fit_summary.get("training_selection"),
                "eric_v3_release": eric_v3.version,
                "our_joint_modulated_release": joint_modulated.version,
                "amber_roll_offsets_applied": True,
                "joint_fit_reference_included": True,
                "integration_checks_json": json.dumps(integration_checks, sort_keys=True),
            }
            all_plot_cells.extend(rows)
            all_roll_plot_cells.extend(roll_rows)
            all_fit_info.append(fit_info)

            source_lookup = {source: index for index, source in enumerate(source_ids)}
            filter_lookup = {name: index for index, name in enumerate(amber.FILTERS)}
            pass_lookup = {name: index for index, name in enumerate(amber.PASSES)}
            row_source = common["source_id"].map(source_lookup).to_numpy(int)
            row_filter = common["filter"].map(filter_lookup).to_numpy(int)
            row_pass = common["pass"].map(pass_lookup).to_numpy(int)
            reference_mag = magnitude[row_source, row_filter, np.zeros(len(common), dtype=int)]
            reference_valid = valid[row_source, row_filter, np.zeros(len(common), dtype=int)]
            observed_delta = common["mag_obs"].to_numpy(float) - reference_mag
            row_roll = amber_roll[row_source, row_filter, row_pass]
            row_fit_prediction = saved_fit_prediction[row_source, row_filter, row_pass]
            usable = (
                reference_valid
                & model_valid[row_source, row_filter, row_pass]
                & np.isfinite(predictions["eric_v3"][row_source, row_filter, row_pass])
                & np.isfinite(saved_fit_prediction[row_source, row_filter, row_pass])
                & np.isfinite(predictions["joint_modulated"][row_source, row_filter, row_pass])
                & np.isfinite(reference_mag)
                & (row_pass > 0)
            )
            prediction_frame = pd.DataFrame({
                "detector": detector,
                "source_id": common.loc[usable, "source_id"].to_numpy(str),
                "filter": common.loc[usable, "filter"].to_numpy(str),
                "pass": common.loc[usable, "pass"].to_numpy(str),
                "mjd_utc": common.loc[usable, "mjd_amber"].to_numpy(float),
                "mag_obs": common.loc[usable, "mag_obs"].to_numpy(float),
                "mag_err": common.loc[usable, "mag_err"].to_numpy(float),
                "reference_mag_obs": reference_mag[usable],
                "observed_delta_mag": observed_delta[usable],
                "amber_roll_offset_mag": row_roll[usable],
                "saved_joint_fit_model_delta_mag": row_fit_prediction[usable],
                "eric_v3_model_delta_mag": predictions["eric_v3"][
                    row_source[usable], row_filter[usable], row_pass[usable]
                ],
                "joint_modulated_model_delta_mag": predictions["joint_modulated"][
                    row_source[usable], row_filter[usable], row_pass[usable]
                ],
                "eric_v3_corrected_delta_mag": (
                    observed_delta[usable]
                    - predictions["eric_v3"][row_source[usable], row_filter[usable], row_pass[usable]]
                ),
                "joint_modulated_corrected_delta_mag": (
                    observed_delta[usable]
                    - predictions["joint_modulated"][row_source[usable], row_filter[usable], row_pass[usable]]
                ),
                "no_ice_roll_adjusted_delta_mag": observed_delta[usable] - row_roll[usable],
                "eric_v3_roll_adjusted_delta_mag": (
                    observed_delta[usable]
                    - predictions["eric_v3"][row_source[usable], row_filter[usable], row_pass[usable]]
                    - row_roll[usable]
                ),
                "joint_modulated_roll_adjusted_delta_mag": (
                    observed_delta[usable]
                    - predictions["joint_modulated"][row_source[usable], row_filter[usable], row_pass[usable]]
                    - row_roll[usable]
                ),
                "saved_joint_fit_reference_corrected_delta_mag": (
                    observed_delta[usable] - row_fit_prediction[usable]
                ),
            })
            flux_columns = {
                "observed": "observed_delta_mag",
                "eric_v3_corrected": "eric_v3_corrected_delta_mag",
                "joint_modulated_corrected": "joint_modulated_corrected_delta_mag",
                "no_ice_roll_adjusted": "no_ice_roll_adjusted_delta_mag",
                "eric_v3_roll_adjusted": "eric_v3_roll_adjusted_delta_mag",
                "joint_modulated_roll_adjusted": "joint_modulated_roll_adjusted_delta_mag",
                "saved_joint_fit_reference": "saved_joint_fit_reference_corrected_delta_mag",
            }
            for variant, column in flux_columns.items():
                prediction_frame[f"{variant}_relative_flux_change_percent"] = 100.0 * np.expm1(
                    -workflow.LOG_FLUX_PER_MAG * prediction_frame[column].to_numpy(float)
                )
            metric_columns = {
                "no_ice_correction": "observed_relative_flux_change_percent",
                "eric_v3": "eric_v3_corrected_relative_flux_change_percent",
                "joint_modulated": "joint_modulated_corrected_relative_flux_change_percent",
            }
            for variant, column in metric_columns.items():
                fleet_values.setdefault(("ALL_FILTERS", variant), []).append(
                    prediction_frame[column].to_numpy(float)
                )
                for band, band_frame in prediction_frame.groupby("filter", sort=False):
                    fleet_values.setdefault((band, variant), []).append(
                        band_frame[column].to_numpy(float)
                    )
            roll_metric_columns = {
                "no_ice_correction": "no_ice_roll_adjusted_relative_flux_change_percent",
                "eric_v3": "eric_v3_roll_adjusted_relative_flux_change_percent",
                "joint_modulated": "joint_modulated_roll_adjusted_relative_flux_change_percent",
                "saved_joint_fit_reference": "saved_joint_fit_reference_relative_flux_change_percent",
            }
            for variant, column in roll_metric_columns.items():
                fleet_values_for_variant = roll_fleet_values.setdefault(("ALL_FILTERS", variant), [])
                fleet_values_for_variant.append(prediction_frame[column].to_numpy(float))
                for band, band_frame in prediction_frame.groupby("filter", sort=False):
                    roll_fleet_values.setdefault((band, variant), []).append(
                        band_frame[column].to_numpy(float)
                    )
            prediction_frame.to_csv(
                prediction_path,
                index=False,
                mode="w" if prediction_header else "a",
                header=prediction_header,
            )
            prediction_header = False
            print(
                f"{detector}: {len(source_ids)} stars, {int(valid.sum())} cohort measurements, "
                f"{int(comparison_mask.sum())} same-star visit comparisons",
                flush=True,
            )
            del photon_shapes, predictions, common, seds, magnitude, valid
            gc.collect()

    for (band, variant), chunks in fleet_values.items():
        record = metric_record("ALL_DETECTORS", band, variant, np.concatenate(chunks))
        record["evaluation"] = "standalone_no_amber_roll"
        all_metrics.append(record)
    for (band, variant), chunks in roll_fleet_values.items():
        record = metric_record("ALL_DETECTORS", band, variant, np.concatenate(chunks))
        record["evaluation"] = "amber_roll_adjusted_same_sample"
        all_metrics.append(record)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(
        args.output_dir / "amber_eric_v3_vs_joint_modulated_correction_metrics.csv",
        index=False,
    )
    pd.DataFrame(all_fit_info).to_csv(
        args.output_dir / "amber_eric_v3_vs_joint_modulated_sample_and_model_info.csv",
        index=False,
    )
    cells = pd.DataFrame(all_plot_cells)
    cells.to_csv(
        args.output_dir / "amber_eric_v3_vs_joint_modulated_epoch_percentiles_standalone.csv",
        index=False,
    )
    plot_comparison(cells, args.output_dir, detectors, "standalone_no_amber_roll")
    roll_cells = pd.DataFrame(all_roll_plot_cells)
    roll_cells.to_csv(
        args.output_dir / "amber_eric_v3_vs_joint_modulated_epoch_percentiles_roll_adjusted.csv",
        index=False,
    )
    plot_comparison(roll_cells, args.output_dir, detectors, "amber_roll_adjusted")

    summary = {
        "cohort": "exact raw common CAR201 source/filter/pass rows saved as joint-fit training inputs",
        "detectors": detectors,
        "joint_fit_name": args.joint_fit_name,
        "eric_original_v3_model": str(args.eric_v3_model.resolve()),
        "our_joint_modulated_model": str(args.joint_modulated_model.resolve()),
        "eric_v3_source_zip": str(args.source_zip.resolve()),
        "effective_area": "detector-specific Roman_effarea_v8_SCA##_20240301.ecsv",
        "sed_policy": "the direct SED assignments from the matched joint-fit cohort; no fallback",
        "time_policy": "exact Amber observation MJD, pixel coordinates, and each model's own ice history",
        "correction_policy": (
            "results include standalone v3 corrections and a same-sample comparison applying the "
            "saved joint-fit Amber roll offsets identically to all variants"
        ),
        "joint_photometry_normalization": (
            "Exactly the same unchanged IceResponseModel.band_response call for both H5s: "
            "gray=True, no_ice_weight=True, photon SED * detector effective area * quadrature weights. "
            "No added H5 groups or model-specific weighting are used."
        ),
        "in_sample_control": (
            "the saved joint CAR190/Amber fit prediction is included as a reference; its Amber roll "
            "offsets were fit on this same cohort"
        ),
        "model_valid_policy": "invalid v3 warm-up/out-of-range model times are dropped identically for all variants",
        "amber_dq_policy": "the matched CAR201 parquet rows expose no DQ flag column; no Amber JUMP_DET mask is applied",
        "joint_modulated_prototype_approximation": (
            "Our model copies TMM coefficients from 256x256-pixel cache tiles to Eric's 16-pixel grid "
            "and uses its v3-compatible rank-one deposition approximation; Eric's original v3 model "
            "retains its released reflectance and deposition"
        ),
        "outputs": {
            "metrics": "amber_eric_v3_vs_joint_modulated_correction_metrics.csv",
            "sample_and_model_info": "amber_eric_v3_vs_joint_modulated_sample_and_model_info.csv",
            "standalone_epoch_percentiles": "amber_eric_v3_vs_joint_modulated_epoch_percentiles_standalone.csv",
            "roll_adjusted_epoch_percentiles": "amber_eric_v3_vs_joint_modulated_epoch_percentiles_roll_adjusted.csv",
            "measurement_predictions": "amber_eric_v3_vs_joint_modulated_measurement_predictions.csv",
            "standalone_plot_pdf": "car201_comparison_without_roll_offsets.pdf",
            "roll_adjusted_plot_pdf": "car201_comparison.pdf",
        },
    }
    (args.output_dir / "amber_eric_v3_vs_joint_modulated_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Wrote matched-sample Amber comparison under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
