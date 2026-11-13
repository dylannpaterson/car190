"""Make the standard Amber variants plus CAR190/TMM joint-fit plots."""

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time

import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production
import heatmap_car201_allstar_model_correction as workflow
import mock_fit_wfi_star_reflectance_modulation as mock
import run_amber_student_t_detectors as amber
from car201_photometry_forward_jacobian import CachedPhotometryForward
from experiment_joint_wfi01_fixed_ice import LedModel, add_full_basis, fixed_output
from experiment_joint_wfi01_weekly_ice import RESETS, cadence_output, led_loss
from per_tile_rate_modulation import PerTileRateModulation
from plot_wfi01_free_led_complex_reflection import aligned_phase, reciprocal
from support_aligned_modulation import effective_area_support


ROOT = Path(__file__).resolve().parent
PRODUCTION_ROOT = workflow.PRODUCTION_ROOT
OUTPUT = PRODUCTION_ROOT / (
    "wfi01/spline_relative_response_normal/experiments/"
    "car190_amber_joint_2day_all_detectors_overview"
)
EVENTS = ("2026-10-17T03:30:00", "2026-10-17T23:00:00", "2026-10-30T01:43:00")
TILE_COUNT = 256


def prediction_with_roll(model, theta, thickness_by_tile, star_indices):
    count = model.mode_count
    model.tile_thickness_by_id = thickness_by_tile[:, star_indices]
    forward = CachedPhotometryForward(model)
    response, _ = forward.evaluate(theta[:2 * count])
    zero_points = theta[2 * count:].reshape(len(amber.FILTERS), model.zero_point_group_count)
    roll = np.zeros_like(response)
    for pass_index, group in enumerate(model.pass_zero_point_group_ids):
        if group:
            response[:, :, pass_index] += zero_points[:, group - 1][None, :]
            roll[:, :, pass_index] = -zero_points[:, group - 1][None, :] / workflow.LOG_FLUX_PER_MAG
    return -response / workflow.LOG_FLUX_PER_MAG, roll


def amber_variant_cells(detector, common, magnitude, valid, variants):
    reference = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    dates = common.groupby("pass")["mjd_amber"].median().reindex(amber.PASSES).to_numpy()
    date_numbers = mdates.date2num(Time(dates, format="mjd", scale="utc").to_datetime())
    plot_valid = valid & valid[:, :, :1]
    plot_valid[:, :, 0] = False
    rows = []
    for variant, corrected in variants.items():
        change = corrected - corrected[:, :, :1]
        flux_change = 100 * np.expm1(-workflow.LOG_FLUX_PER_MAG * change)
        for filter_index, band in enumerate(amber.FILTERS):
            for pass_index, epoch in enumerate(amber.PASSES[1:], start=1):
                mask = plot_valid[:, filter_index, pass_index]
                if not mask.any():
                    continue
                values = flux_change[mask, filter_index, pass_index]
                p16, median, p84 = np.quantile(values, [0.16, 0.5, 0.84])
                rows.append({
                    "detector": detector,
                    "filter": band,
                    "pass": epoch,
                    "variant": variant,
                    "matched_stars": int(mask.sum()),
                    "p16_percent": p16,
                    "median_percent": median,
                    "p84_percent": p84,
                    "utc_date_number": date_numbers[pass_index],
                })
    return rows


def plot_amber_variants(cells, output_dir, pdf, out_of_sample=False,
                       cohort_label=None, output_suffix=None, cohort_note=None,
                       variant_labels=None):
    selected = set(cells["detector"].unique())
    detectors = tuple(detector for detector in workflow.DETECTORS if detector in selected)
    colors = plt.get_cmap("tab20")(np.arange(len(detectors)))
    lower = float(cells["p16_percent"].min())
    upper = float(cells["p84_percent"].max())
    margin = 0.05 * max(upper - lower, 1.0)
    labels = {
        "no_ice_correction": "Roll-adjusted data; no ice correction",
        "unmodulated": "Unmodulated optics; baseline ice history",
        "joint_modulated": "Joint modulated optics + fitted 2-day ice",
    }
    if variant_labels is not None:
        labels.update(variant_labels)
    for variant in labels:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
        for band, axis in zip(amber.FILTERS, axes.flat, strict=True):
            axis.axhspan(-0.5, 0.5, color="#65ac68", alpha=0.14)
            axis.axhline(0, color="0.4", linewidth=0.6)
            for index, (detector, color) in enumerate(zip(detectors, colors, strict=True)):
                selected = cells.loc[(cells["variant"] == variant)
                                     & (cells["filter"] == band)
                                     & (cells["detector"] == detector)]
                offset = (index - (len(detectors) - 1) / 2) / 96.0
                axis.errorbar(
                    selected["utc_date_number"] + offset,
                    selected["median_percent"],
                    yerr=np.asarray([
                        selected["median_percent"] - selected["p16_percent"],
                        selected["p84_percent"] - selected["median_percent"],
                    ]),
                    fmt="o", color=color, markersize=2.4, elinewidth=0.6, capsize=1.0,
                )
            axis.set_title(band)
            axis.set_ylim(lower - margin, upper + margin)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            axis.tick_params(axis="x", labelrotation=50, labelsize=8)
            axis.grid(axis="y", alpha=0.2)
        for axis in axes[-1]:
            axis.set_xlabel("UTC date of ending visit")
        handles = [Line2D([0], [0], color=color, label=detector)
                   for detector, color in zip(detectors, colors, strict=True)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                   ncol=9, frameon=False)
        plot_cohort_label = cohort_label or ("held-out stars" if out_of_sample else "fit-cohort stars")
        fig.suptitle(labels[variant] + f" — {plot_cohort_label}, same-star visit changes", y=0.985, fontsize=15)
        fig.supylabel("Relative flux change from measured EP3 pass 1 (%)")
        if cohort_note is not None:
            note = cohort_note
        elif out_of_sample:
            note = ("Stars absent from joint-fit source lists; only direct successful faint-v2/original SEDs used. "
                    "No nearest-neighbor SED substitutions. ")
        else:
            note = ""
        fig.text(0.5, 0.025, note +
                 "Points are detector medians; bars are star 16th–84th percentiles; green band is ±0.5%. "
                 "Identical matched-quality stars and axes across variants. Roll offsets are fit jointly.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0.025, 0.1, 1, 0.87))
        suffix = output_suffix if output_suffix is not None else ("_out_of_sample" if out_of_sample else "")
        path = output_dir / f"amber_{variant}_joint_2day{suffix}.png"
        fig.savefig(path, dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"Wrote {path}", flush=True)


def predict_car190(led, theta, times, thickness_by_tile=None):
    mode = led.basis.shape[0]
    optical = np.asarray(theta[:2 * mode], dtype=float)
    rows = []
    residual_parts = []
    for band_index, tiles in enumerate(led.bands):
        for tile, item in enumerate(tiles):
            a0, b0, reflection, y0, factor, basis, saved, observed, sigma, days, original, wave, ice = item
            a, b = a0, b0
            if thickness_by_tile is not None:
                indices = np.searchsorted(times, days)
                np.testing.assert_allclose(times[indices], days, atol=1e-10, rtol=0)
                current = thickness_by_tile[tile, indices]
                shift = 2 * np.pi * ice * (current - original)[:, None] / wave[None, :]
                cosine, sine = np.cos(shift), -1j * np.sin(shift)
                a = cosine * a0 + sine * ice * b0
                b = cosine * b0 + sine * a0 / ice
            ratio = np.exp(optical[:mode] @ basis + 1j * (optical[mode:] @ basis))
            y = y0 * (1 - reflection * ratio) / (1 + reflection * ratio)
            denominator = a + b * y[None, :]
            rate = np.sum(factor[None, :] / np.abs(denominator) ** 2, axis=1)
            log_rate = np.log(rate)
            baseline = led.references[band_index][tile]
            predicted = saved + (log_rate - log_rate[0]) - (baseline - baseline[0])
            residual_parts.append((predicted - observed) / sigma)
            rows.append(pd.DataFrame({
                "led": band_index + 1,
                "tile": tile,
                "day": days,
                "observed": observed,
                "predicted": predicted,
            }))
    return pd.concat(rows, ignore_index=True), np.concatenate(residual_parts)


def summarize_car190(frame):
    frame = frame.copy()
    frame["day_key"] = np.round(frame["day"], 7)
    result = []
    for (led, day), group in frame.groupby(["led", "day_key"], sort=True):
        row = {"led": led, "day": day, "tile_count": group["tile"].nunique()}
        for name in ("observed", "fixed", "joint"):
            values = group[name].to_numpy(float)
            row[f"{name}_p16"] = np.quantile(values, 0.16)
            row[f"{name}_median"] = np.median(values)
            row[f"{name}_p84"] = np.quantile(values, 0.84)
        result.append(row)
    return pd.DataFrame(result)


def plot_car190(detector, summaries, reference_mjd, path, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    reset_dates = [Time(reference_mjd + day, format="mjd", scale="utc").to_datetime()
                   for day in RESETS]
    event_dates = [Time(event, format="isot", scale="utc").to_datetime() for event in EVENTS]
    for led_number, axis in enumerate(axes.flat, start=1):
        rows = summaries.loc[summaries["led"] == led_number].sort_values("day")
        dates = Time(reference_mjd + rows["day"].to_numpy(), format="mjd", scale="utc").to_datetime()
        axis.scatter(dates, 100 * rows["observed_median"], s=8, color="0.35", alpha=0.7,
                     label="CAR190 corrected data (tile median)")
        stages = np.searchsorted(RESETS, rows["day"].to_numpy(), side="right")
        for stage in range(3):
            selected = stages == stage
            for name, color, label in (("fixed", "#357fa1", "Fixed-ice joint model"),
                                       ("joint", "#d47727", "Joint 2-day model")):
                axis.fill_between(dates[selected], 100 * rows.loc[selected, f"{name}_p16"],
                                  100 * rows.loc[selected, f"{name}_p84"], color=color,
                                  alpha=0.12, linewidth=0)
                axis.plot(dates[selected], 100 * rows.loc[selected, f"{name}_median"],
                          color=color, linewidth=1.3, label=label if led_number == 1 and stage == 0 else None)
        for event in reset_dates:
            axis.axvline(event, color=".35", linestyle=":", linewidth=0.8)
        for event in event_dates:
            axis.axvline(event, color="#b66b2e", linestyle="--", linewidth=0.7, alpha=0.8)
        axis.axhline(0, color=".35", linewidth=0.6)
        axis.set_title(f"LED {led_number}")
        axis.grid(alpha=0.18)
    for axis in axes[:, 0]:
        axis.set_ylabel("Log response change ×100 (%)")
    for axis in axes[-1]:
        axis.set_xlabel("UTC date")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[-1, -1].xaxis.set_major_locator(locator)
    axes[-1, -1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)
    fig.suptitle(f"{detector}: CAR190 corrected data and joint-model response", y=0.995, fontsize=14)
    fig.text(0.5, 0.02,
             "Data markers are tile medians (scatter); model curves and shading are median and 16th–84th "
             "percentiles across tiles. Dotted: decon resets; dashed: confirmed temperature events.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.9))
    fig.savefig(path, dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def plot_complex_reflectance(detector, model, fixed_theta, joint_theta, path, csv_rows, pdf):
    wave = np.asarray(model.wavelengths_nm, dtype=float)
    clean = np.asarray(model.clean_reflection_by_tile, dtype=complex)
    mode = model.mode_count
    fixed_ratio = np.exp(fixed_theta[:mode] @ model.basis
                         + 1j * (fixed_theta[mode:2 * mode] @ model.basis))
    joint_ratio = np.exp(joint_theta[:mode] @ model.basis
                         + 1j * (joint_theta[mode:2 * mode] @ model.basis))
    curves = {
        "Unmodulated clean TMM": clean,
        "Fixed-ice joint modulation": clean * fixed_ratio[None, :],
        "2-day joint modulation": clean * joint_ratio[None, :],
    }
    reference_phase = np.angle(clean[:, -1])
    reference_phase += 2 * np.pi * np.round((np.median(reference_phase) - reference_phase) / (2 * np.pi))
    styles = {
        "Unmodulated clean TMM": ("0.4", "--"),
        "Fixed-ice joint modulation": ("#357fa1", "-"),
        "2-day joint modulation": ("#d47727", "-"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    order = np.argsort(1000 / wave)
    x = 1000 / wave[order]
    for label, reflection in curves.items():
        magnitude = np.percentile(np.abs(reflection), [16, 50, 84], axis=0)
        phase = np.percentile(aligned_phase(reflection, reference_phase), [16, 50, 84], axis=0)
        color, style = styles[label]
        for axis, stats in zip(axes, (magnitude, phase), strict=True):
            axis.fill_between(x, stats[0, order], stats[2, order], color=color, alpha=0.11, linewidth=0)
            axis.plot(x, stats[1, order], color=color, linestyle=style, linewidth=1.5, label=label)
        for index, wavelength in enumerate(wave):
            csv_rows.append({
                "detector": detector,
                "variant": label,
                "wavelength_nm": wavelength,
                **{f"magnitude_p{p}": magnitude[i, index] for i, p in enumerate((16, 50, 84))},
                **{f"phase_p{p}_rad": phase[i, index] for i, p in enumerate((16, 50, 84))},
            })
    led_colors = plt.get_cmap("Dark2")(np.linspace(0, 1, len(mock.LED_CENTERS_NM)))
    for axis in axes:
        axis.set_xlim(x[0], x[-1])
        axis.grid(alpha=0.2)
    for led_number, (center, color) in enumerate(zip(mock.LED_CENTERS_NM, led_colors, strict=True), start=1):
        position = 1000 / center
        if x[0] <= position <= x[-1]:
            for axis in axes:
                axis.axvline(position, color=color, linestyle=":", linewidth=1.0, alpha=0.9, zorder=0)
            axes[0].text(
                position, 0.98, f"LED {led_number}", transform=axes[0].get_xaxis_transform(),
                rotation=90, ha="right", va="top", fontsize=9.5, fontweight="bold", color="black",
                bbox={"facecolor": "white", "edgecolor": color, "linewidth": 0.8, "alpha": 0.95, "pad": 1.5},
                zorder=5, clip_on=False,
            )
    axes[0].set_ylabel(r"Clean reflection magnitude $|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top = axes[0].secondary_xaxis("top", functions=(reciprocal, reciprocal))
    top_ticks = np.asarray([2300, 1700, 1300, 1000, 800, 600, 500], dtype=float)
    top_ticks = top_ticks[(top_ticks >= wave.min()) & (top_ticks <= wave.max())]
    top.set_xticks(top_ticks)
    top.set_xlabel("Wavelength (nm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.suptitle(f"{detector}: clean complex reflection (no ice layer)", y=0.99, fontsize=14)
    fig.text(0.5, 0.025,
             "Lines are tile medians; shading is tile 16th–84th percentiles. "
             "All curves show the clean boundary response; no ice film is included.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.025, 0.07, 1, 0.86))
    fig.savefig(path, dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def process_detector(detector, amber_rows, car_rows, reflect_rows, car_pdf, reflect_pdf):
    fit_dir = cadence_output(detector, 2.0)
    fixed_dir = fixed_output(detector)
    area, _ = amber.canonical_configuration(detector)
    _, common, ids, magnitude, error, valid, _, model, _ = amber.prepare(
        detector, write_inputs=False, effective_area_path=area, output_dir=fit_dir,
        full_effective_area_support=True,
    )
    add_full_basis(model, effective_area_support(area))
    with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
        times = np.asarray(archive["times"], dtype=float)
        baseline_depth = np.asarray(archive["baseline_thickness_nm"], dtype=float)
        fitted_depth = np.asarray(archive["fitted_thickness_nm"], dtype=float)
        joint_theta = np.asarray(archive["shared_parameters"], dtype=float)
    with np.load(fixed_dir / "joint_fit.npz", allow_pickle=False) as archive:
        fixed_theta = np.asarray(archive["parameters"], dtype=float)
        np.testing.assert_array_equal(archive["stellar_valid"], valid)

    reference_mjd = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    star_days = np.array([
        [common.loc[(common["filter"] == band) & (common["pass"] == epoch), "mjd_amber"].median() - reference_mjd
         for epoch in amber.PASSES]
        for band in amber.FILTERS
    ])
    star_indices = np.searchsorted(times, star_days)
    np.testing.assert_allclose(times[star_indices], star_days, atol=1e-10, rtol=0)
    if len(ids) != model.star_count:
        raise ValueError(f"{detector}: Amber source ordering mismatch")

    theta_unmodulated = joint_theta.copy()
    theta_unmodulated[:2 * model.mode_count] = 0.0
    unmodulated_prediction, roll = prediction_with_roll(
        model, theta_unmodulated, baseline_depth, star_indices)
    modulated_prediction, _ = prediction_with_roll(
        model, joint_theta, fitted_depth, star_indices)
    variants = {
        "no_ice_correction": magnitude - roll,
        "unmodulated": magnitude - unmodulated_prediction,
        "joint_modulated": magnitude - modulated_prediction,
    }
    amber_rows.extend(amber_variant_cells(detector, common, magnitude, valid, variants))

    # Build the detector-specific CAR190 optical model and reconstruct both fits.
    led = LedModel(effective_area_support(area), detector=detector)
    mode = led.basis.shape[0]
    if mode != model.mode_count:
        raise ValueError(f"{detector}: Amber and CAR190 modulation bases differ")
    fixed_frame, fixed_residual = predict_car190(led, fixed_theta, times)
    joint_frame, joint_residual = predict_car190(led, joint_theta, times, fitted_depth)
    np.testing.assert_array_equal(fixed_frame[["led", "tile", "day"]].to_numpy(),
                                  joint_frame[["led", "tile", "day"]].to_numpy())
    with np.load(fixed_dir / "joint_fit.npz", allow_pickle=False) as archive:
        saved_fixed_residual = np.asarray(archive["led_normalized_residual"], dtype=float)
    np.testing.assert_allclose(fixed_residual, saved_fixed_residual, atol=2e-5, rtol=2e-6)
    trajectory = PerTileRateModulation(times, baseline_depth, RESETS, spacing_days=2.0)
    with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
        ice_parameters = np.asarray(archive["ice_parameters"], dtype=float)
    reconstructed_depth, jacobian = trajectory.evaluate(ice_parameters, with_jacobian=True)
    np.testing.assert_allclose(reconstructed_depth, fitted_depth, atol=2e-8, rtol=0)
    led_indices = [[np.searchsorted(times, item[9]) for item in tiles] for tiles in led.bands]
    joint_cost, _, _ = led_loss(led, joint_theta[:2 * mode], fitted_depth, jacobian, led_indices)
    np.testing.assert_allclose(0.5 * np.sum(joint_residual ** 2), joint_cost, rtol=2e-7, atol=2e-6)

    frame = fixed_frame[["led", "tile", "day", "observed"]].copy()
    frame["fixed"] = fixed_frame["predicted"].to_numpy()
    frame["joint"] = joint_frame["predicted"].to_numpy()
    car_summary = summarize_car190(frame)
    car_summary.insert(0, "detector", detector)
    car_rows.extend(car_summary.to_dict("records"))
    car_plot_path = fit_dir / "car190_data_vs_joint_models.png"
    plot_car190(detector, car_summary, reference_mjd, car_plot_path, car_pdf)

    # Validate the clean reflection against the full production TMM on tile 136.
    test_tile = 136
    optical_archive = PRODUCTION_ROOT / f"{detector.lower()}/spline_relative_response_normal/fits/tile_{test_tile:03d}/posterior_samples.npz"
    with np.load(optical_archive, allow_pickle=False) as archive:
        cache = Path(str(np.asarray(archive["response_cache"]).reshape(-1)[0]))
    source = mock.source_fit_from_cache(cache, detector, test_tile)
    wave = np.asarray(model.wavelengths_nm, dtype=float)
    base_n, base_k = production.hgcdte_reference_optical_constants(wave)
    state = complex_model.load_state(source, wave, base_n, base_k, production.spectral_modulation_basis(wave))
    first, second, y0, _, sine = mock.clean_stack_boundary(state, wave)
    if abs(sine) > 1e-12:
        raise ValueError(f"{detector}: complex-reflection plot expects normal incidence")
    state_clean = (y0 * first - second) / (y0 * first + second)
    np.testing.assert_allclose(state_clean, model.clean_reflection_by_tile[test_tile], atol=2e-8, rtol=2e-8)
    expected_clean, _ = complex_model.reflection(state, wave, 0.0, "s")
    np.testing.assert_allclose(state_clean, expected_clean, atol=2e-8, rtol=2e-8)

    complex_path = fit_dir / "joint_complex_reflectance.png"
    plot_complex_reflectance(
        detector, model, fixed_theta, joint_theta, complex_path, reflect_rows, reflect_pdf)
    print(f"{detector}: wrote Amber variants, CAR190 fit, and complex-reflection plots", flush=True)
    del led, model, common, magnitude, error, valid, fixed_frame, joint_frame
    gc.collect()


def plot_car190_h5_audit(detector, fit_dir, h5_path, output_dir):
    """Compare actual fitted CAR190 predictions to the unchanged public H5 reader."""
    import h5py
    import compare_eric_v3_amber_car201 as compare

    summary = json.loads((fit_dir / 'summary.json').read_text())
    if summary['detector'] != detector:
        raise ValueError('Detector does not match requested fit')
    with np.load(fit_dir / 'joint_fit.npz') as archive:
        theta = np.asarray(archive['parameters'], float)
    with np.load(fit_dir / 'joint_weekly_ice_fit.npz') as archive:
        times = np.asarray(archive['times'], float)
        history = np.asarray(archive['fitted_thickness_nm'], float)
    with h5py.File(h5_path) as h5:
        nodes = h5['led_bands/wavelength_nm'][()]
        weights = h5['led_bands/weight'][()]
    area, _ = amber.canonical_configuration(detector)
    led = LedModel(effective_area_support(area), detector=detector,
        relative_response_root=Path(summary['relative_response_root']),
        led_wavelength_nm=nodes, led_weights=weights)
    fitted_rows, fitted_residual = predict_car190(led, theta, times, history)
    np.testing.assert_allclose(np.sum(fitted_residual**2), summary['car190_chi2'], rtol=1e-8)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    stats = []
    with compare.load_eric_v3_reader(compare.DEFAULT_SOURCE_ZIP) as reader:
        model = reader.IceResponseModel(h5_path)
        tile = np.arange(256)
        x, y = (tile % 16)*256+128, (tile // 16)*256+128
        reference = compare.utc_mjd(joint_reference := workflow.fit_model.ICE_REFERENCE_UTC)
        h5_reference = compare.utc_mjd(model.t0_utc)
        for bi, band in enumerate(led.bands):
            days = band[0][9]
            hours = (reference-h5_reference+days)*24.
            band_response = model.band_response(int(detector[3:]), x, y, hours,
                nodes[bi], weights[bi], gray=True, no_ice_weight=True)
            gray = model.gray(int(detector[3:]), x, y, hours)
            raw = np.log(band_response)-np.log(band_response[:, :1])
            gray_delta = np.log(gray)-np.log(gray[:, :1])
            for ti, item in enumerate(band):
                first, second, r, y0, factor, basis, saved, observed, sigma, item_days, original, wave, ice = item
                source_baseline = led.references[bi][ti]
                calibration = saved-(source_baseline-source_baseline[0])
                # This is a diagnostic application of the SAME existing source
                # calibration terms in the CAR190 fit, not a new fitted curve.
                usual_calibration = raw[ti]-gray_delta[ti]+calibration
                fit = fitted_rows.loc[(fitted_rows.led == bi+1) & (fitted_rows.tile == ti), 'predicted'].to_numpy()
                np.testing.assert_allclose(item_days, days, atol=1e-10, rtol=0)
                # Diagnostic constant only: distinguishes a reference-level
                # shift from a shape error; it is not written into the model.
                constant = float(np.median(fit-raw[ti]))
                for j, day in enumerate(days):
                    records.append(dict(led=bi+1,tile=ti,day=float(day),observed=float(observed[j]),
                        fit=float(fit[j]),h5=float(raw[ti,j]),
                        h5_source_calibration=float(usual_calibration[j]),
                        h5_constant_aligned=float(raw[ti,j]+constant),
                        sigma=float(sigma[j]),source_calibration=float(calibration[j]),
                        h5_gray_delta=float(gray_delta[ti,j])))
            local = pd.DataFrame(records).query('led == @bi + 1')
            stats.append(dict(led=bi+1,fit_chi2=float(np.sum(((local['fit']-local.observed)/local.sigma)**2)),
                h5_chi2=float(np.sum(((local.h5-local.observed)/local.sigma)**2)),
                h5_source_calibration_chi2=float(np.sum(((local.h5_source_calibration-local.observed)/local.sigma)**2)),
                h5_constant_aligned_chi2=float(np.sum(((local.h5_constant_aligned-local.observed)/local.sigma)**2)),
                h5_vs_fit_rms_percent=float(100*np.sqrt(np.mean((local.h5-local['fit'])**2))),
                h5_source_calibration_vs_fit_rms_percent=float(100*np.sqrt(np.mean((local.h5_source_calibration-local['fit'])**2)))))
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir/'car190_h5_tile_predictions.csv',index=False)
    grouped = frame.groupby(['led','day'],sort=True)
    cells=[]
    columns=['observed','fit','h5','h5_source_calibration','h5_constant_aligned']
    for (number,day),group in grouped:
        row=dict(led=number,day=day)
        for column in columns:
            for q, value in zip((16,50,84),np.percentile(group[column],[16,50,84])):
                row[f'{column}_p{q}']=value
        cells.append(row)
    cells=pd.DataFrame(cells)
    cells.to_csv(output_dir/'car190_h5_epoch_percentiles.csv',index=False)
    pdf_path=output_dir/'car190_h5_vs_data.pdf'
    series=[('fit','#357fa1','Actual joint fit'),('h5','#d47727','H5: public reader'),
            ('h5_source_calibration','#259c66','H5 + existing CAR190 calibration'),
            ('h5_constant_aligned','#9865b5','H5 + diagnostic constant per tile/LED')]
    with PdfPages(pdf_path) as pdf:
        for residual in (False,True):
            fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True)
            for number,ax in enumerate(axes.flat,start=1):
                rows=cells.loc[cells.led==number].sort_values('day')
                dates=Time(reference+rows.day.to_numpy(),format='mjd',scale='utc').to_datetime()
                if not residual:
                    ax.scatter(dates,100*rows.observed_p50,s=9,color='.3',label='Corrected CAR190 data')
                stages=np.searchsorted(RESETS,rows.day.to_numpy(),side='right')
                for column,color,label in series:
                    if residual:
                        residual_rows=frame.loc[frame.led==number].copy()
                        residual_rows['residual']=residual_rows[column]-residual_rows.observed
                        percentiles=residual_rows.groupby('day').residual.quantile([.16,.5,.84]).unstack().reindex(rows.day)
                        median=percentiles[.5].to_numpy()
                        low=percentiles[.16].to_numpy();high=percentiles[.84].to_numpy()
                    else:
                        median=rows[f'{column}_p50'].to_numpy()
                        low=rows[f'{column}_p16'].to_numpy();high=rows[f'{column}_p84'].to_numpy()
                    for stage in np.unique(stages):
                        use=stages==stage
                        ax.plot(dates[use],100*median[use],color=color,lw=1.25,
                            label=label if stage==stages[0] else None)
                        if column in ('fit','h5'):
                            ax.fill_between(dates[use],100*low[use],100*high[use],color=color,alpha=.1,lw=0)
                for day in RESETS:
                    ax.axvline(Time(reference+day,format='mjd').to_datetime(),ls=':',color='.4',lw=.8)
                for event in EVENTS:
                    ax.axvline(Time(event,format='isot').to_datetime(),ls='--',color='#b66b2e',lw=.7,alpha=.65)
                ax.axhline(0,color='.4',lw=.6)
                ax.set_title(f'LED {number}')
                ax.grid(alpha=.2)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            axes[0,0].legend(fontsize=7,loc='best')
            fig.supylabel(('Model minus data' if residual else 'Log response change')+' ×100 (%)')
            fig.supxlabel('UTC date')
            fig.suptitle(f'{detector}: CAR190 '+('residuals' if residual else 'data versus joint fit and exported H5'))
            fig.text(.5,.025,'All curves referenced to the same first exposure of each tile/LED. '
                'Green uses the existing calibration terms already present in the fit; purple adds only a diagnostic constant. '
                'No new offsets were fitted to the observations.',ha='center',fontsize=8)
            fig.tight_layout(rect=(.02,.06,1,.95))
            path=output_dir/('car190_h5_residuals.png' if residual else 'car190_h5_vs_data.png')
            fig.savefig(path,dpi=180);pdf.savefig(fig);plt.close(fig)
            print(f'Wrote {path}',flush=True)
    report=dict(detector=detector,fit_dir=str(fit_dir.resolve()),h5=str(h5_path.resolve()),
        source_fit_reference_utc=joint_reference,h5_reference_utc=model.t0_utc,
        reader_unchanged=True,LED_metrics=stats,pdf=str(pdf_path.resolve()))
    (output_dir/'car190_h5_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", required=True, choices=workflow.DETECTORS)
    parser.add_argument("--complex-only", action="store_true",
                         help="Regenerate only the clean no-ice complex-reflectance plots")
    parser.add_argument('--fit-dir',type=Path)
    parser.add_argument('--h5-model',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    detectors = list(dict.fromkeys(args.detector))
    if args.fit_dir is not None or args.h5_model is not None:
        if args.fit_dir is None or args.h5_model is None or len(detectors)!=1:
            parser.error('H5 CAR190 audit requires one detector, --fit-dir and --h5-model')
        plot_car190_h5_audit(detectors[0],args.fit_dir,args.h5_model,
                            args.output_dir or args.fit_dir/'car190_h5_diagnostics')
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    amber_rows, car_rows, reflect_rows = [], [], []
    with PdfPages(OUTPUT / "complex_reflectance_all_detectors.pdf") as reflect_pdf:
        if args.complex_only:
            for index, detector in enumerate(detectors, start=1):
                print(f"=== Plotting clean complex reflection for {detector} ({index}/{len(detectors)}) ===", flush=True)
                fit_dir = cadence_output(detector, 2.0)
                area, _ = amber.canonical_configuration(detector)
                _, _, _, _, _, _, _, model, _ = amber.prepare(
                    detector, write_inputs=False, effective_area_path=area, output_dir=fit_dir,
                    full_effective_area_support=True)
                add_full_basis(model, effective_area_support(area))
                with np.load(fixed_output(detector) / "joint_fit.npz", allow_pickle=False) as archive:
                    fixed_theta = np.asarray(archive["parameters"], dtype=float)
                with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
                    joint_theta = np.asarray(archive["shared_parameters"], dtype=float)
                test_tile = 136
                optical_archive = PRODUCTION_ROOT / f"{detector.lower()}/spline_relative_response_normal/fits/tile_{test_tile:03d}/posterior_samples.npz"
                with np.load(optical_archive, allow_pickle=False) as archive:
                    cache = Path(str(np.asarray(archive["response_cache"]).reshape(-1)[0]))
                source = mock.source_fit_from_cache(cache, detector, test_tile)
                wave = np.asarray(model.wavelengths_nm, dtype=float)
                base_n, base_k = production.hgcdte_reference_optical_constants(wave)
                state = complex_model.load_state(source, wave, base_n, base_k,
                                                 production.spectral_modulation_basis(wave))
                first, second, y0, _, sine = mock.clean_stack_boundary(state, wave)
                if abs(sine) > 1e-12:
                    raise ValueError(f"{detector}: clean-reflection plot expects normal incidence")
                state_clean = (y0 * first - second) / (y0 * first + second)
                np.testing.assert_allclose(state_clean, model.clean_reflection_by_tile[test_tile],
                                           atol=2e-8, rtol=2e-8)
                rows = []
                plot_complex_reflectance(
                    detector, model, fixed_theta, joint_theta,
                    fit_dir / "joint_complex_reflectance.png", rows, reflect_pdf)
                reflect_rows.extend(rows)
                del model
                gc.collect()
            pd.DataFrame(reflect_rows).to_csv(
                OUTPUT / "complex_reflectance_clean_boundary_wavelength_quantiles.csv", index=False)
            print(f"Clean no-ice complex-reflectance plots written under {OUTPUT}", flush=True)
            return

        with PdfPages(OUTPUT / "car190_data_vs_model_all_detectors.pdf") as car_pdf:
            for index, detector in enumerate(detectors, start=1):
                print(f"=== Plotting {detector} ({index}/{len(detectors)}) ===", flush=True)
                process_detector(detector, amber_rows, car_rows, reflect_rows, car_pdf, reflect_pdf)

    cells = pd.DataFrame(amber_rows)
    cells.to_csv(OUTPUT / "amber_joint_model_epoch_percentiles.csv", index=False)
    car_table = pd.DataFrame(car_rows)
    car_table.to_csv(OUTPUT / "car190_data_model_epoch_percentiles.csv", index=False)
    pd.DataFrame(reflect_rows).to_csv(
        OUTPUT / "complex_reflectance_clean_boundary_wavelength_quantiles.csv", index=False)
    amber_pdf = OUTPUT / "amber_uncorrected_unmodulated_modulated_all_detectors.pdf"
    with PdfPages(amber_pdf) as pdf:
        plot_amber_variants(cells, OUTPUT, pdf)
    print(f"Amber variant plots and all combined PDFs are in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
