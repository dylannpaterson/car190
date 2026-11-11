"""Map clean, ice-free, unmodulated complex reflection across WFI16 tiles."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production
import mock_fit_wfi_star_reflectance_modulation as mock_model


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = (
    ROOT
    / "20261020/corrected_statistics_4h_thinned/production/relative_response_cache/"
    "tile_size_256/sensitivity_hgcdte_n_sigma_0p25/wfi16_400nm"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "20261105/corrected_statistics_4h_thinned/production/wfi16/"
    "spline_relative_response_normal/experiments/"
    "wfi16_clean_complex_reflectance_full_detector"
)
LED_CENTERS_NM = np.asarray(
    [
        623.520885362797,
        852.802649193115,
        1038.5847494059,
        1269.7582849979,
        1481.24145201185,
        1670.16888078615,
    ],
    dtype=float,
)


def _tile_id_from_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        detector = str(np.asarray(archive["detector"]).reshape(-1)[0])
        tile_id = int(np.asarray(archive["tile_id"]).reshape(-1)[0])
    if detector.upper() != "WFI16":
        raise ValueError(f"Unexpected optical cache detector {detector} in {path}")
    return tile_id


def _aligned_unwrapped_phase(reflections):
    phase = np.unwrap(np.angle(reflections)[:, ::-1], axis=1)[:, ::-1]
    long_wave_anchor = phase[:, -1]
    common_anchor = np.median(long_wave_anchor)
    turns = np.round((common_anchor - long_wave_anchor) / (2.0 * np.pi))
    return phase + turns[:, None] * (2.0 * np.pi)


def _tile_map(values, tile_ids):
    result = np.full((16, 16), np.nan, dtype=float)
    for tile_id, value in zip(tile_ids, values):
        result[int(tile_id) // 16, int(tile_id) % 16] = value
    return result


def _plot_tile_maps(
    path,
    values_by_tile_wavelength,
    tile_ids,
    wavelengths_nm,
    value_label,
    cmap,
):
    figure, axes = plt.subplots(2, 3, figsize=(13.8, 8.2), constrained_layout=True)
    finite = values_by_tile_wavelength[np.isfinite(values_by_tile_wavelength)]
    lower, upper = np.percentile(finite, [1.0, 99.0])
    if np.isclose(lower, upper):
        lower, upper = float(np.min(finite)), float(np.max(finite))
    for led_index, (axis, wavelength_nm) in enumerate(
        zip(axes.flat, wavelengths_nm)
    ):
        image = axis.imshow(
            _tile_map(values_by_tile_wavelength[:, led_index], tile_ids),
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
        )
        axis.set_title(f"LED{led_index + 1}: {wavelength_nm:.1f} nm")
        axis.set_xticks([0, 4, 8, 12, 15])
        axis.set_yticks([0, 4, 8, 12, 15])
        axis.set_xlabel("Tile column")
        axis.set_ylabel("Tile row")
        axis.set_xlim(-0.5, 15.5)
        axis.set_ylim(15.5, -0.5)
    figure.colorbar(image, ax=axes.ravel().tolist(), label=value_label, shrink=0.86)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optical-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wavelength-min-nm", type=float, default=500.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=2300.0)
    parser.add_argument("--wavelength-step-nm", type=float, default=5.0)
    args = parser.parse_args(argv)
    if (
        args.wavelength_min_nm <= 0.0
        or args.wavelength_max_nm <= args.wavelength_min_nm
        or args.wavelength_step_nm <= 0.0
    ):
        parser.error("Wavelength limits and step must be positive and increasing")

    cache_paths = sorted(args.optical_cache_root.glob("tile_*/relative_response.npz"))
    tile_ids = np.asarray([_tile_id_from_cache(path) for path in cache_paths], dtype=int)
    order = np.argsort(tile_ids)
    cache_paths = [cache_paths[index] for index in order]
    tile_ids = tile_ids[order]
    if tile_ids.size != 256 or not np.array_equal(tile_ids, np.arange(256)):
        raise ValueError(
            f"Expected the complete WFI16 16x16 tile cache; found {tile_ids.size} tiles"
        )

    wavelengths_nm = np.arange(
        args.wavelength_min_nm,
        args.wavelength_max_nm + 0.5 * args.wavelength_step_nm,
        args.wavelength_step_nm,
    )
    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths_nm)
    # A zero spectral basis explicitly disables the additional fitted AR/HgCdTe
    # wavelength modulation. Evaluate only the clean (zero-ice) stack.
    no_modulation_basis = np.zeros(
        (
            wavelengths_nm.size,
            len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM),
        ),
        dtype=float,
    )
    reflections = np.empty((tile_ids.size, wavelengths_nm.size), dtype=complex)
    state_sources = []
    for index, (tile_id, cache_path) in enumerate(zip(tile_ids, cache_paths)):
        source_fit = mock_model.source_fit_from_cache(
            cache_path, "WFI16", int(tile_id)
        )
        state_sources.append(str(source_fit))
        state = complex_model.load_state(
            source_fit,
            wavelengths_nm,
            base_n,
            base_k,
            no_modulation_basis,
        )
        clean_reflection, _ = complex_model.reflection(
            state, wavelengths_nm, 0.0, "s"
        )
        reflections[index] = clean_reflection

    magnitude = np.abs(reflections)
    phase = _aligned_unwrapped_phase(reflections)
    magnitude_stats = np.percentile(magnitude, [16.0, 50.0, 84.0], axis=0)
    phase_stats = np.percentile(phase, [16.0, 50.0, 84.0], axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = args.output_dir / "wfi16_clean_complex_reflectance_tile_spectra.csv"
    with spectra_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tile_id",
                "tile_row",
                "tile_col",
                "wavelength_nm",
                "reflection_real",
                "reflection_imag",
                "reflection_magnitude",
                "phase_unwrapped_rad",
            ]
        )
        for tile_index, tile_id in enumerate(tile_ids):
            for wavelength_index, wavelength_nm in enumerate(wavelengths_nm):
                value = reflections[tile_index, wavelength_index]
                writer.writerow(
                    [
                        int(tile_id),
                        int(tile_id // 16),
                        int(tile_id % 16),
                        wavelength_nm,
                        value.real,
                        value.imag,
                        magnitude[tile_index, wavelength_index],
                        phase[tile_index, wavelength_index],
                    ]
                )

    spectrum_plot = args.output_dir / "wfi16_clean_complex_reflectance_tile_spectra.png"
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)
    for tile_index in range(tile_ids.size):
        axes[0].plot(wavelengths_nm, magnitude[tile_index], color="0.35", alpha=0.045, linewidth=0.55)
        axes[1].plot(wavelengths_nm, phase[tile_index], color="0.35", alpha=0.045, linewidth=0.55)
    axes[0].fill_between(
        wavelengths_nm,
        magnitude_stats[0],
        magnitude_stats[2],
        color="#4c9f70",
        alpha=0.22,
        label="tile p16–p84",
    )
    axes[0].plot(wavelengths_nm, magnitude_stats[1], color="#24734a", linewidth=2.0, label="tile median")
    axes[1].fill_between(
        wavelengths_nm,
        phase_stats[0],
        phase_stats[2],
        color="#5b77b5",
        alpha=0.22,
        label="tile p16–p84; phase branches aligned at long λ",
    )
    axes[1].plot(wavelengths_nm, phase_stats[1], color="#35518e", linewidth=2.0, label="tile median")
    for axis in axes:
        for led_center in LED_CENTERS_NM:
            if wavelengths_nm[0] <= led_center <= wavelengths_nm[-1]:
                axis.axvline(led_center, color="0.6", linestyle=":", linewidth=0.7)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[0].set_ylabel(r"$|r_{\rm clean}(\lambda)|$")
    axes[1].set_ylabel("Unwrapped clean reflection phase (rad)")
    axes[1].set_xlabel("Wavelength (nm)")
    figure.suptitle(
        "WFI16 clean, ice-free, zero-spectral-modulation TMM reflection\n"
        "256 tile-specific pinned optical states",
        y=0.99,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.94))
    figure.savefig(spectrum_plot, dpi=180, bbox_inches="tight")
    plt.close(figure)

    map_wavelengths = np.asarray(
        [
            623.520885362797,
            852.802649193115,
            1038.5847494059,
            1269.7582849979,
            1481.24145201185,
            1670.16888078615,
        ]
    )
    wavelength_indices = np.asarray(
        [np.argmin(np.abs(wavelengths_nm - value)) for value in map_wavelengths],
        dtype=int,
    )
    map_wavelengths = wavelengths_nm[wavelength_indices]
    magnitude_maps = magnitude[:, wavelength_indices]
    phase_maps = phase[:, wavelength_indices]
    magnitude_map_path = args.output_dir / "wfi16_clean_reflection_magnitude_tile_maps.png"
    phase_map_path = args.output_dir / "wfi16_clean_reflection_phase_tile_maps.png"
    _plot_tile_maps(
        magnitude_map_path,
        magnitude_maps,
        tile_ids,
        map_wavelengths,
        r"$|r_{\rm clean}|$",
        "viridis",
    )
    _plot_tile_maps(
        phase_map_path,
        phase_maps,
        tile_ids,
        map_wavelengths,
        "Phase (rad; long-wavelength branches aligned)",
        "coolwarm",
    )

    summary_rows = []
    for led_number, (wavelength_nm, wavelength_index) in enumerate(
        zip(map_wavelengths, wavelength_indices), start=1
    ):
        summary_rows.append(
            {
                "led_number": led_number,
                "wavelength_nm": float(wavelength_nm),
                "magnitude_p16": float(magnitude_stats[0, wavelength_index]),
                "magnitude_median": float(magnitude_stats[1, wavelength_index]),
                "magnitude_p84": float(magnitude_stats[2, wavelength_index]),
                "magnitude_min": float(np.min(magnitude[:, wavelength_index])),
                "magnitude_max": float(np.max(magnitude[:, wavelength_index])),
                "phase_p16_rad": float(phase_stats[0, wavelength_index]),
                "phase_median_rad": float(phase_stats[1, wavelength_index]),
                "phase_p84_rad": float(phase_stats[2, wavelength_index]),
                "phase_range_rad": float(
                    np.max(phase[:, wavelength_index])
                    - np.min(phase[:, wavelength_index])
                ),
            }
        )
    spread_csv = args.output_dir / "wfi16_clean_complex_reflectance_tile_spread.csv"
    with spread_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "detector": "WFI16",
        "tile_count": int(tile_ids.size),
        "tile_id_layout": "tile_id = 16*row + col; zero-based rows and columns",
        "wavelength_range_nm": [float(wavelengths_nm[0]), float(wavelengths_nm[-1])],
        "complex_reflection_state": "clean, zero ice thickness, no additional AR/HgCdTe wavelength modulation",
        "incidence_angle_deg": 0.0,
        "optical_cache_root": str(args.optical_cache_root),
        "tile_optical_source_fits": state_sources,
        "map_wavelengths_nm": map_wavelengths.tolist(),
        "spectrum_plot": str(spectrum_plot),
        "magnitude_maps": str(magnitude_map_path),
        "phase_maps": str(phase_map_path),
        "spectrum_csv": str(spectra_path),
        "spread_csv": str(spread_csv),
    }
    summary_path = args.output_dir / "wfi16_clean_complex_reflectance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for path in (spectrum_plot, magnitude_map_path, phase_map_path, spectra_path, spread_csv, summary_path):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
