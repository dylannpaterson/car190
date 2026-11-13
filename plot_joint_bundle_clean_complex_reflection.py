"""Plot no-ice clean complex reflection with and without joint modulation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import fit_car201_zane_star_reflectance_modulation as fitter
import mock_fit_wfi_star_reflectance_modulation as mock
import plot_joint_amber_car190_detailed as detailed_plotting
from wfi_wavelength_response_bundle import load_bundle


ROOT = Path(__file__).resolve().parent
DEFAULT_STATISTICS = ROOT / "20261111/corrected_statistics_4h_thinned"
DEFAULT_BUNDLE = (
    DEFAULT_STATISTICS
    / "production/wavelength_response_bundle/wfi_all_2nm_rank16_modulated.npz"
)
DEFAULT_OUTPUT = (
    DEFAULT_STATISTICS
    / "production/wavelength_response_bundle/"
    "clean_complex_reflection_no_ice_modulated_vs_unmodulated"
)
LED_CENTERS_NM = np.asarray(mock.LED_CENTERS_NM, dtype=float)


def reciprocal_wavelength(wavelength_inverse_um):
    values = np.asarray(wavelength_inverse_um, dtype=float)
    return np.divide(1000.0, values, out=np.full(values.shape, np.nan), where=values > 0)


def reciprocal(wavelength_nm):
    values = np.asarray(wavelength_nm, dtype=float)
    return np.divide(1000.0, values, out=np.full(values.shape, np.nan), where=values > 0)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_reference(unmodulated):
    reference = np.angle(unmodulated[:, -1])
    return reference + 2.0 * np.pi * np.round(
        (np.median(reference) - reference) / (2.0 * np.pi)
    )


def clean_reflection(source_fit, wavelengths_nm, base_n, base_k, spectral_basis):
    state = fitter.complex_model.load_state(
        source_fit, wavelengths_nm, base_n, base_k, spectral_basis
    )
    first, second, incident_y, _, sine_incidence = mock.clean_stack_boundary(
        state, wavelengths_nm
    )
    if abs(sine_incidence) > 1e-12:
        raise ValueError("Clean-reflection comparison requires the pinned normal-incidence state")
    reflection = (incident_y * first - second) / (incident_y * first + second)
    if not np.isfinite(reflection).all():
        raise ValueError(f"Nonfinite clean reflection from {source_fit}")
    return reflection


def tile_reflections(bundle, detector, wavelengths_nm, base_n, base_k, spectral_basis):
    metadata = bundle["metadata"]
    detector_names = np.asarray(bundle["detectors"]).astype(str)
    tile_ids = np.asarray(bundle["tile_ids"], dtype=int)
    detector_indices = np.flatnonzero(detector_names == detector)
    if detector_indices.size != 256:
        raise ValueError(f"Expected 256 bundle tiles for {detector}, found {detector_indices.size}")
    detector_indices = detector_indices[np.argsort(tile_ids[detector_indices])]
    selected_tile_ids = tile_ids[detector_indices]
    np.testing.assert_array_equal(selected_tile_ids, np.arange(256))

    modulation_detectors = np.asarray(bundle["modulation_detectors"]).astype(str)
    match = np.flatnonzero(modulation_detectors == detector)
    if match.size != 1:
        raise ValueError(f"Bundle modulation metadata does not uniquely identify {detector}")
    modulation_index = int(match[0])
    ratio = (
        np.asarray(bundle["modulation_ratio_real"][modulation_index], dtype=float)
        + 1j * np.asarray(bundle["modulation_ratio_imag"][modulation_index], dtype=float)
    )
    np.testing.assert_allclose(
        np.asarray(bundle["modulation_wavelengths_nm"], dtype=float),
        wavelengths_nm,
        rtol=0,
        atol=0,
    )
    if ratio.shape != wavelengths_nm.shape:
        raise ValueError(f"Modulation ratio wavelength grid differs for {detector}")

    cache_root = Path(metadata["tile_optical_cache_roots"][detector])
    unmodulated = np.empty((256, wavelengths_nm.size), dtype=np.complex128)
    for tile_id in selected_tile_ids:
        cache_path = cache_root / f"tile_{tile_id:03d}/relative_response.npz"
        source_fit = mock.source_fit_from_cache(cache_path, detector, int(tile_id))
        unmodulated[tile_id] = clean_reflection(
            source_fit, wavelengths_nm, base_n, base_k, spectral_basis
        )
    modulated = unmodulated * ratio[None, :]
    return selected_tile_ids, unmodulated, modulated


def summarize(detector, wavelengths_nm, unmodulated, modulated, reference_phase):
    rows = []
    for variant, reflections in (
        ("unmodulated", unmodulated),
        ("joint_modulated", modulated),
    ):
        magnitude = np.percentile(np.abs(reflections), [16, 50, 84], axis=0)
        phase = np.percentile(
            detailed_plotting.aligned_phase(reflections, reference_phase),
            [16, 50, 84], axis=0,
        )
        for index, wavelength in enumerate(wavelengths_nm):
            rows.append({
                "detector": detector,
                "variant": variant,
                "wavelength_nm": float(wavelength),
                **{
                    f"reflection_magnitude_p{percentile}": float(magnitude[row, index])
                    for row, percentile in enumerate((16, 50, 84))
                },
                **{
                    f"reflection_phase_p{percentile}_rad": float(phase[row, index])
                    for row, percentile in enumerate((16, 50, 84))
                },
            })
    return rows


def plot_detector(
    axes, wavelengths_nm, unmodulated, modulated, color,
    reference_phase, detector=None, show_top_axis=False,
):
    x = reciprocal(wavelengths_nm)
    order = np.argsort(x)
    variants = (
        ("Unmodulated clean TMM", unmodulated, "0.35", "--", 0.10, 1.35),
        ("Joint reciprocal-MaxEnt + curvature", modulated, color, "-", 0.16, 1.8),
    )
    for label, reflections, line_color, linestyle, alpha, linewidth in variants:
        magnitude = np.percentile(np.abs(reflections), [16, 50, 84], axis=0)
        phase = np.percentile(
            detailed_plotting.aligned_phase(reflections, reference_phase),
            [16, 50, 84], axis=0,
        )
        for axis, statistics in zip(axes, (magnitude, phase), strict=True):
            axis.fill_between(
                x[order], statistics[0, order], statistics[2, order],
                color=line_color, alpha=alpha, linewidth=0,
            )
            axis.plot(
                x[order], statistics[1, order], color=line_color,
                linestyle=linestyle, linewidth=linewidth, label=label,
            )
    for axis in axes:
        axis.set_xlim(x[order][0], x[order][-1])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"Clean reflection magnitude $|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    if show_top_axis:
        top_axis = axes[0].secondary_xaxis(
            "top", functions=(reciprocal_wavelength, reciprocal_wavelength)
        )
        ticks_nm = np.asarray([2300.0, 1700.0, 1300.0, 1000.0, 800.0, 600.0, 500.0])
        ticks_nm = ticks_nm[
            (ticks_nm >= wavelengths_nm.min()) & (ticks_nm <= wavelengths_nm.max())
        ]
        top_axis.set_xticks(ticks_nm)
        top_axis.set_xlabel("Wavelength (nm)")
    if detector is not None:
        axes[0].set_title(detector)


def add_labeled_led_markers(axes, wavelengths_nm):
    led_colors = plt.get_cmap("Dark2")(
        np.linspace(0.0, 1.0, len(LED_CENTERS_NM))
    )
    low, high = reciprocal(wavelengths_nm).min(), reciprocal(wavelengths_nm).max()
    for led_number, (center, color) in enumerate(
        zip(LED_CENTERS_NM, led_colors, strict=True), start=1
    ):
        position = reciprocal(center)
        if not low <= position <= high:
            continue
        for axis in axes:
            axis.axvline(
                position, color=color, linestyle=":", linewidth=1.2,
                alpha=1.0, zorder=0,
            )
        axes[0].text(
            position,
            0.98,
            f"LED {led_number}",
            transform=axes[0].get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.0,
                "alpha": 1.0,
                "pad": 1.8,
            },
            zorder=5,
            clip_on=False,
        )


def plot_saved_detector_fit(fit_dir, detector, output_dir):
    """Use this plotter on one explicit new fit, rather than the old fleet bundle."""
    from experiment_joint_wfi01_fixed_ice import full_basis
    from run_amber_student_t_detectors import canonical_configuration
    from support_aligned_modulation import effective_area_support

    summary_path = fit_dir / 'summary.json'
    archive_path = fit_dir / 'joint_fit.npz'
    summary = json.loads(summary_path.read_text())
    if summary['detector'] != detector:
        raise ValueError('Requested detector does not match the saved fit')
    if not summary.get('optimizer_success'):
        raise ValueError('Requested fit did not converge')
    with np.load(archive_path, allow_pickle=False) as archive:
        theta = np.asarray(archive['parameters'], float)
        saved_wave = np.asarray(archive['wavelengths_nm'], float)
        saved_basis = np.asarray(archive['basis'], float)
    area_path, _ = canonical_configuration(detector)
    bounds = effective_area_support(area_path)
    np.testing.assert_allclose(full_basis(saved_wave, bounds)[0], saved_basis, atol=0, rtol=0)
    wave = np.arange(np.floor(bounds[0]), np.ceil(bounds[1])+1.)
    basis = full_basis(wave, bounds)[0]
    count = len(basis)
    ratio = np.exp(theta[:count] @ basis + 1j*(theta[count:2*count] @ basis))
    base_n, base_k = fitter.production.hgcdte_reference_optical_constants(wave)
    spectral = fitter.production.spectral_modulation_basis(wave)
    cache_root = Path(summary['optical_cache_root'])
    original = []
    for tile in range(256):
        cache_path = cache_root / f'tile_{tile:03d}/relative_response.npz'
        source = mock.source_fit_from_cache(cache_path, detector, tile)
        original.append(clean_reflection(source, wave, base_n, base_k, spectral))
    original = np.asarray(original)
    modulated = original * ratio[None, :]
    reference = phase_reference(original)
    area_wave, areas = mock.load_effective_area(area_path)
    band_index = list(mock.FILTERS).index('F213')
    supported = area_wave[areas[band_index] > 0.]
    f213_min, f213_max = float(supported[0]), float(supported[-1])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = detector.lower()
    pdf_path = output_dir / f'{stem}_modulation.pdf'
    with PdfPages(pdf_path) as pdf:
        for suffix, limits in [('', None), ('_f213', (f213_min, f213_max))]:
            selected = np.ones(len(wave), bool) if limits is None else (wave >= limits[0]) & (wave <= limits[1])
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            plot_detector(axes, wave[selected], original[:, selected], modulated[:, selected],
                          '#d47727', reference, show_top_axis=True)
            add_labeled_led_markers(axes, wave[selected])
            for ax in axes:
                ax.axvspan(float(reciprocal(f213_max)), float(reciprocal(f213_min)),
                           color='#b7a755', alpha=.12)
            axes[0].legend(loc='best')
            fig.suptitle(f'{detector}: NEW joint fit — unmodulated vs fitted complex reflectance'
                         + (' — F213' if limits else ''), fontsize=12)
            fig.text(.5, .02, 'Dashed: pinned unmodulated TMM; orange: this fit’s modulation. '
                     'Shading: 16th–84th percentiles over 256 tiles; pale yellow: F213 support. No ice layer.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0,.05,1,.93))
            path = output_dir / f'{stem}_complex_reflectance{suffix}.png'
            fig.savefig(path, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
            print(f'Wrote {path}', flush=True)

        # The detector-global complex multiplier is the most direct answer to
        # "how much modulation happened?", independent of tile spread.
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        amplitude = 100*(np.abs(ratio)-1)
        phase = theta[count:2*count] @ basis
        axes[0].plot(wave, amplitude, color='#d47727', lw=1.8)
        axes[1].plot(wave, phase, color='#d47727', lw=1.8)
        for ax in axes:
            ax.axhline(0, color='.4', lw=.7)
            ax.axvspan(f213_min, f213_max, color='#b7a755', alpha=.16)
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Change in reflection magnitude (%)')
        axes[1].set_ylabel('Added reflection phase (rad)')
        axes[1].set_xlabel('Wavelength (nm)')
        fig.suptitle(f'{detector}: modulation applied by the NEW joint fit')
        fig.text(.5,.02, 'Magnitude change = 100 × (|r_mod / r_unmod| − 1); phase = arg(r_mod / r_unmod). '
                 'Yellow region: F213 support. These are reflection changes, not photometric residuals.',
                 ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.95))
        path = output_dir / f'{stem}_modulation_amount.png'
        fig.savefig(path, dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
        print(f'Wrote {path}', flush=True)
    pd.DataFrame(dict(wavelength_nm=wave, reflection_magnitude_change_percent=amplitude,
                      added_phase_rad=phase, multiplier_real=ratio.real,
                      multiplier_imag=ratio.imag)).to_csv(output_dir/f'{stem}_modulation_amount.csv', index=False)
    summarize_rows = summarize(detector, wave, original, modulated, reference)
    pd.DataFrame(summarize_rows).to_csv(output_dir/f'{stem}_complex_reflectance_percentiles.csv', index=False)
    selected = (wave >= f213_min) & (wave <= f213_max)
    provenance = dict(detector=detector, fit_dir=str(fit_dir.resolve()),
        source_fit_sha256=sha256_file(archive_path), response_convention=summary.get('response_convention'),
        detectors_plotted=[detector], tiles=256, ice_layer_included=False,
        f213_support_nm=[f213_min,f213_max],
        f213_reflection_magnitude_change_percent_min_max=[float(amplitude[selected].min()),float(amplitude[selected].max())],
        f213_added_phase_rad_min_max=[float(phase[selected].min()),float(phase[selected].max())],
        pdf=str(pdf_path.resolve()))
    (output_dir/'plot_summary.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fit-dir', type=Path,
                        help='Plot one explicit saved joint fit rather than the historical fleet bundle')
    parser.add_argument('--detector', choices=tuple(f'WFI{i:02d}' for i in range(1,19)))
    args = parser.parse_args(argv)
    if args.fit_dir is not None:
        if args.detector is None:
            parser.error('--fit-dir requires --detector')
        output = args.fit_dir/'complex_reflectance' if args.output_dir == DEFAULT_OUTPUT else args.output_dir
        plot_saved_detector_fit(args.fit_dir, args.detector, output)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.bundle)
    metadata = bundle["metadata"]
    if metadata.get("modulation_fit_family") != "joint-car190-amber-2day":
        raise ValueError("Input bundle does not contain the current joint CAR190/Amber modulation")
    wavelengths_nm = np.asarray(bundle["modulation_wavelengths_nm"], dtype=float)
    base_n, base_k = fitter.production.hgcdte_reference_optical_constants(wavelengths_nm)
    spectral_basis = fitter.production.spectral_modulation_basis(wavelengths_nm)
    detectors = np.asarray(bundle["modulation_detectors"]).astype(str).tolist()
    colors = plt.get_cmap("tab20")(np.arange(len(detectors)))
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    all_rows = []
    pdf_path = args.output_dir / "clean_complex_reflection_by_detector.pdf"
    with PdfPages(pdf_path) as pdf:
        for detector_index, (detector, color) in enumerate(zip(detectors, colors, strict=True), start=1):
            print(f"=== Clean no-ice reflection {detector} ({detector_index}/{len(detectors)}) ===", flush=True)
            tile_ids, unmodulated, modulated = tile_reflections(
                bundle, detector, wavelengths_nm, base_n, base_k, spectral_basis
            )
            reference_phase = phase_reference(unmodulated)
            all_rows.extend(summarize(
                detector, wavelengths_nm, unmodulated, modulated, reference_phase
            ))
            plot_detector(
                axes, wavelengths_nm, unmodulated, modulated, color, reference_phase
            )

            detector_figure, detector_axes = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True,
                gridspec_kw={"height_ratios": [1.0, 0.9], "hspace": 0.08},
            )
            plot_detector(
                detector_axes,
                wavelengths_nm,
                unmodulated,
                modulated,
                "#d47727",
                reference_phase,
                show_top_axis=True,
            )
            add_labeled_led_markers(detector_axes, wavelengths_nm)
            detector_axes[0].legend(loc="best", fontsize=8)
            detector_figure.suptitle(
                f"{detector}: clean complex reflection, no ice layer", y=0.99
            )
            detector_figure.text(
                0.5, 0.015,
                "Lines are tile medians; shading is the tile 16th–84th percentile range "
                f"({len(tile_ids)} tiles). Modulation is applied to the clean boundary response.",
                ha="center", fontsize=8,
            )
            detector_figure.subplots_adjust(
                top=0.87, bottom=0.12, left=0.11, right=0.97, hspace=0.08
            )
            pdf.savefig(detector_figure, bbox_inches="tight")
            plt.close(detector_figure)
            print(f"{detector}: reconstructed clean reflection for {len(tile_ids)} tiles", flush=True)

    for axis in axes:
        axes[0].set_ylabel(r"Clean reflection magnitude $|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top_axis = axes[0].secondary_xaxis(
        "top", functions=(reciprocal_wavelength, reciprocal_wavelength)
    )
    top_ticks = np.asarray([2300.0, 1700.0, 1300.0, 1000.0, 800.0, 600.0, 500.0])
    top_ticks = top_ticks[
        (top_ticks >= wavelengths_nm.min()) & (top_ticks <= wavelengths_nm.max())
    ]
    top_axis.set_xticks(top_ticks)
    top_axis.set_xlabel("Wavelength (nm)")
    add_labeled_led_markers(axes, wavelengths_nm)
    handles = [
        Line2D([0], [0], color=color, label=detector)
        for detector, color in zip(detectors, colors, strict=True)
    ]
    handles.extend([
        Line2D([0], [0], color="0.35", linestyle="--", label="Unmodulated"),
        Line2D([0], [0], color="0.15", linestyle="-", label="Joint modulation"),
    ])
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=10, fontsize=8)
    figure.suptitle("Clean complex reflection across WFI detectors — no ice coating", y=0.995)
    figure.text(
        0.5, 0.02,
        "Dashed: unmodulated TMM; solid: joint reciprocal-MaxEnt + curvature modulation. "
        "Shading: tile 16th–84th percentiles, not fit uncertainty.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.07, 0.99, 0.88))
    png_path = args.output_dir / "all_detector_clean_complex_reflection.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    csv_path = args.output_dir / "clean_complex_reflection_tile_percentiles.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    summary_path = args.output_dir / "plot_summary.json"
    summary_path.write_text(json.dumps({
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256": sha256_file(args.bundle),
        "detectors": detectors,
        "tiles_per_detector": 256,
        "wavelength_samples": int(wavelengths_nm.size),
        "modulation_fit_family": metadata["modulation_fit_family"],
        "reciprocal_maxent_weight": float(
            metadata["modulation_fit_sources"][detectors[0]]["reciprocal_maxent_weight"]
        ),
        "curvature_weight": float(
            metadata["modulation_fit_sources"][detectors[0]]["curvature_weight"]
        ),
        "ice_layer_included": False,
        "plot": str(png_path),
        "detector_pdf": str(pdf_path),
        "tile_percentile_csv": str(csv_path),
    }, indent=2) + "\n")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
