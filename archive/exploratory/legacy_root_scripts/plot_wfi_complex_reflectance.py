"""Plot the model complex reflection coefficient at a tile's reference epoch."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production


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


def _archive_scalar(archive, name):
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"Expected a scalar {name} in the archive")
    return value.reshape(-1)[0]


def _tile_reference_thickness(path, detector, tile_id, requested_day):
    with np.load(path, allow_pickle=False) as archive:
        tile_ids = np.asarray(archive["tile_ids"], dtype=int).reshape(-1)
        sca_names = np.asarray(archive["sca_names"]).astype(str).reshape(-1)
        if tile_id not in tile_ids:
            raise ValueError(f"Tile {tile_id} is not present in {path}")
        tile_index = int(np.flatnonzero(tile_ids == tile_id)[0])
        if (
            tile_index >= sca_names.size
            or sca_names[tile_index].upper() != detector.upper()
        ):
            raise ValueError(
                f"Fit archive does not identify tile {tile_id} as {detector}"
            )

        days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
        if days.size == 0:
            raise ValueError("Fit archive contains no observation epochs")
        target_day = float(days[0] if requested_day is None else requested_day)
        observation_index = int(np.argmin(np.abs(days - target_day)))
        thickness = np.asarray(archive["tile_ice_thickness"], dtype=float)
        thickness = thickness.reshape((-1, days.size))
        if thickness.shape[0] != tile_ids.size:
            raise ValueError("Tile thickness array does not match the archived tile IDs")
        return float(days[observation_index]), float(thickness[tile_index, observation_index])


def _source_fit_from_cache(path, detector, tile_id):
    with np.load(path, allow_pickle=False) as archive:
        cached_detector = str(_archive_scalar(archive, "detector"))
        cached_tile = int(_archive_scalar(archive, "tile_id"))
        source_fit = Path(str(_archive_scalar(archive, "source_fit")))
    if cached_detector.upper() != detector.upper() or cached_tile != tile_id:
        raise ValueError(
            f"Optical cache is for {cached_detector} tile {cached_tile}, "
            f"not {detector} tile {tile_id}"
        )
    if not source_fit.is_file():
        raise FileNotFoundError(f"Optical source fit from cache is missing: {source_fit}")
    return source_fit


def _unwrapped_phase(reflection):
    # Unwrap from long to short wavelengths, matching the reversed wavelength axis.
    phase = np.unwrap(np.angle(reflection)[::-1])[::-1]
    return phase - phase[-1]


def plot_complex_reflectance(
    detector,
    tile_id,
    reference_day,
    thickness_nm,
    state,
    wavelengths_nm,
    reflection,
    output_dir,
    optical_source,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    magnitude = np.abs(reflection)
    phase = _unwrapped_phase(reflection)
    visible_leds = (LED_CENTERS_NM >= wavelengths_nm[0]) & (
        LED_CENTERS_NM <= wavelengths_nm[-1]
    )
    led_wavelengths = LED_CENTERS_NM[visible_leds]
    led_magnitude = np.interp(led_wavelengths, wavelengths_nm, magnitude)
    led_phase = np.interp(led_wavelengths, wavelengths_nm, phase)

    figure, (magnitude_axis, phase_axis) = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9], "hspace": 0.08},
    )
    curve_color = "#45a77d"
    led_color = "#3274c8"
    magnitude_axis.plot(
        wavelengths_nm,
        magnitude,
        color=curve_color,
        linewidth=2.7,
        label="Production TMM ice model",
    )
    magnitude_axis.scatter(
        led_wavelengths,
        led_magnitude,
        facecolors="none",
        edgecolors=led_color,
        linewidths=2.0,
        s=78,
        zorder=4,
        label="Model at RCS LED centres",
    )
    phase_axis.plot(wavelengths_nm, phase, color=curve_color, linewidth=2.7)
    phase_axis.scatter(
        led_wavelengths,
        led_phase,
        facecolors="none",
        edgecolors=led_color,
        linewidths=2.0,
        s=78,
        zorder=4,
    )

    for led_number, led_wavelength in enumerate(LED_CENTERS_NM, start=1):
        if wavelengths_nm[0] <= led_wavelength <= wavelengths_nm[-1]:
            for axis in (magnitude_axis, phase_axis):
                axis.axvline(
                    led_wavelength,
                    color="0.55",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.7,
                    zorder=0,
                )
            magnitude_axis.text(
                led_wavelength,
                0.98,
                f"LED{led_number}",
                transform=magnitude_axis.get_xaxis_transform(),
                rotation=90,
                ha="left",
                va="top",
                fontsize=8,
                color="0.4",
            )

    for axis in (magnitude_axis, phase_axis):
        axis.grid(True, color="0.87", linewidth=0.8)
        axis.set_xlim(wavelengths_nm[-1], wavelengths_nm[0])
    xticks_nm = np.asarray([2300.0, 1700.0, 1300.0, 1000.0, 800.0, 600.0, 500.0])
    xticks_nm = xticks_nm[
        (xticks_nm >= wavelengths_nm[0]) & (xticks_nm <= wavelengths_nm[-1])
    ]
    magnitude_axis.set_xticks(xticks_nm)
    magnitude_axis.set_ylabel(r"$|r_{\rm ref}|$")
    phase_axis.set_ylabel(r"$\arg(r_{\rm ref})$ (rad)")
    magnitude_axis.xaxis.tick_top()
    magnitude_axis.xaxis.set_label_position("top")
    magnitude_axis.tick_params(
        axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False
    )
    magnitude_axis.set_xlabel("Wavelength (nm)", labelpad=8)
    phase_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    magnitude_axis.legend(loc="best", frameon=True, fontsize=9)
    figure.suptitle(
        f"{detector}, tile row {tile_id // 16}, col {tile_id % 16}: "
        "complex reflection at the reference epoch\n"
        f"constant-index production ice (n={state['ice_index']:.4f}, k=0), "
        f"d={thickness_nm:.2f} nm at day {reference_day:.5f}, "
        f"incidence={state['incidence_angle_deg']:.2f}°",
        y=0.99,
        fontsize=13,
    )
    figure.text(
        0.5,
        0.015,
        "Phase is unwrapped from long to short wavelength and zeroed at the "
        "long-wavelength endpoint; hollow markers are model samples at the six LED centres.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(top=0.84, bottom=0.09, left=0.09, right=0.985)

    stem = f"{detector.lower()}_tile_{tile_id:03d}_complex_reflectance_reference"
    figure_path = output_dir / f"{stem}.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as handle:
        fields = [
            "wavelength_nm",
            "reflection_real",
            "reflection_imag",
            "reflection_magnitude",
            "reflectance",
            "phase_wrapped_rad",
            "phase_unwrapped_rad",
            "is_led_center",
            "led_number",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        nearest_led = np.argmin(
            np.abs(wavelengths_nm[:, None] - LED_CENTERS_NM[None, :]), axis=1
        )
        led_error = np.abs(wavelengths_nm - LED_CENTERS_NM[nearest_led])
        for index, wavelength_nm in enumerate(wavelengths_nm):
            is_led_center = bool(led_error[index] <= 0.5)
            writer.writerow(
                {
                    "wavelength_nm": wavelength_nm,
                    "reflection_real": reflection[index].real,
                    "reflection_imag": reflection[index].imag,
                    "reflection_magnitude": magnitude[index],
                    "reflectance": magnitude[index] ** 2,
                    "phase_wrapped_rad": np.angle(reflection[index]),
                    "phase_unwrapped_rad": phase[index],
                    "is_led_center": is_led_center,
                    "led_number": int(nearest_led[index] + 1) if is_led_center else "",
                }
            )

    summary_path = output_dir / f"{stem}_summary.json"
    summary = {
        "detector": detector,
        "tile_id": int(tile_id),
        "tile_row_zero_based": int(tile_id // 16),
        "tile_col_zero_based": int(tile_id % 16),
        "reference_day": float(reference_day),
        "ice_thickness_nm": float(thickness_nm),
        "ice_index_real": float(state["ice_index"]),
        "ice_index_imag": 0.0,
        "incidence_angle_deg": float(state["incidence_angle_deg"]),
        "polarization": "s (equal to p at the normal incidence used here)",
        "wavelength_range_nm": [float(wavelengths_nm[0]), float(wavelengths_nm[-1])],
        "optical_source_fit": str(optical_source),
        "led_centers_nm": LED_CENTERS_NM[visible_leds].tolist(),
        "plot": str(figure_path),
        "spectrum_csv": str(csv_path),
        "mean_reflectance": float(np.mean(magnitude**2)),
        "mean_reflection_magnitude": float(np.mean(magnitude)),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return figure_path, csv_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-fit-archive", type=Path, required=True)
    parser.add_argument("--optical-cache-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI16")
    parser.add_argument("--tile-id", type=int, default=136)
    parser.add_argument(
        "--reference-day",
        type=float,
        default=None,
        help="Fit-relative day; defaults to the first archived observation epoch.",
    )
    parser.add_argument("--wavelength-min-nm", type=float, default=500.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=2300.0)
    parser.add_argument("--wavelength-samples", type=int, default=1901)
    args = parser.parse_args(argv)

    if args.wavelength_min_nm <= 0.0 or args.wavelength_max_nm <= args.wavelength_min_nm:
        parser.error("Wavelength limits must be positive and increasing")
    if args.wavelength_samples < 2:
        parser.error("At least two wavelength samples are required")

    reference_day, thickness_nm = _tile_reference_thickness(
        args.trajectory_fit_archive,
        args.detector,
        args.tile_id,
        args.reference_day,
    )
    optical_source = _source_fit_from_cache(
        args.optical_cache_archive, args.detector, args.tile_id
    )
    wavelengths_nm = np.linspace(
        args.wavelength_min_nm,
        args.wavelength_max_nm,
        args.wavelength_samples,
    )
    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths_nm)
    spectral_basis = production.spectral_modulation_basis(wavelengths_nm)
    state = complex_model.load_state(
        optical_source,
        wavelengths_nm,
        base_n,
        base_k,
        spectral_basis,
    )
    _, reflection = complex_model.reflection(
        state, wavelengths_nm, thickness_nm, "s"
    )
    paths = plot_complex_reflectance(
        args.detector.upper(),
        args.tile_id,
        reference_day,
        thickness_nm,
        state,
        wavelengths_nm,
        reflection,
        args.output_dir,
        optical_source,
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
