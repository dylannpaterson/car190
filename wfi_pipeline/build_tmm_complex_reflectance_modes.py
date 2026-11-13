"""Export TMM clean reflection as Eric-style complex PCA modes in HDF5.

The output is a reflectance-mode product, not a complete time-dependent v3 release.
Its ``reflectance`` datasets use the v3 core convention:
``r = mean + amplitudes @ modes``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import svd

import fit_car201_zane_star_reflectance_modulation as fitter
import mock_fit_wfi_star_reflectance_modulation as mock
from wfi_wavelength_response_bundle import load_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/wfi_all_2nm_rank16_modulated.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/tmm_complex_reflectance_modes_v1.h5"
)
DEFAULT_MODULATED_OUTPUT = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/tmm_joint_modulated_reflectance_modes_v1.h5"
)
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
TILE_GRID_SIZE = 16
TILE_SIZE_PIXELS = 256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_reflection(source_fit, wavelengths_nm, base_n, base_k, spectral_basis):
    state = fitter.complex_model.load_state(
        source_fit, wavelengths_nm, base_n, base_k, spectral_basis
    )
    first, second, incident_y, _, sine_incidence = mock.clean_stack_boundary(
        state, wavelengths_nm
    )
    if abs(sine_incidence) > 1e-12:
        raise ValueError(f"Expected normal incidence in cached TMM state {source_fit}")
    result = (incident_y * first - second) / (incident_y * first + second)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite clean TMM reflection in {source_fit}")
    return result


def _write_dataset(group, name, values, **kwargs):
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 4)
    kwargs.setdefault("shuffle", True)
    group.create_dataset(name, data=values, **kwargs)


def build(args):
    if args.mode_count < 1:
        raise ValueError("Mode count must be positive")
    if args.wavelength_step_nm <= 0 or args.wavelength_max_nm <= args.wavelength_min_nm:
        raise ValueError("Wavelength limits and step must be positive and increasing")
    if not np.isclose(
        (args.wavelength_max_nm - args.wavelength_min_nm) / args.wavelength_step_nm,
        round((args.wavelength_max_nm - args.wavelength_min_nm) / args.wavelength_step_nm),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("The requested wavelength range must divide evenly by the step")
    output_path = args.output or (
        DEFAULT_OUTPUT if args.variant == "unmodulated" else DEFAULT_MODULATED_OUTPUT
    )
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing mode product: {output_path}")

    detectors = list(dict.fromkeys(value.upper() for value in args.detector))
    unknown = sorted(set(detectors) - set(DETECTORS))
    if unknown:
        raise ValueError(f"Unknown WFI detectors: {unknown}")
    bundle = load_bundle(args.bundle)
    metadata = bundle["metadata"]
    wavelengths_nm = np.arange(
        args.wavelength_min_nm,
        args.wavelength_max_nm + args.wavelength_step_nm * 0.5,
        args.wavelength_step_nm,
        dtype=float,
    )
    base_n, base_k = fitter.production.hgcdte_reference_optical_constants(wavelengths_nm)
    spectral_basis = fitter.production.spectral_modulation_basis(wavelengths_nm)
    modulation_wave = np.asarray(bundle["modulation_wavelengths_nm"], dtype=float)
    modulation_detectors = np.asarray(bundle["modulation_detectors"]).astype(str)
    modulation_ratios = (
        np.asarray(bundle["modulation_ratio_real"], dtype=float)
        + 1j * np.asarray(bundle["modulation_ratio_imag"], dtype=float)
    )

    bundle_detectors = np.asarray(bundle["detectors"]).astype(str)
    tile_ids_all = np.asarray(bundle["tile_ids"], dtype=int)
    detector_indices = []
    tile_records = []
    source_fits = []
    reflection_rows = []
    cache_roots = metadata["tile_optical_cache_roots"]
    for detector_index, detector in enumerate(detectors, start=1):
        selected = np.flatnonzero(bundle_detectors == detector)
        if selected.size != TILE_GRID_SIZE**2:
            raise ValueError(f"{detector} has {selected.size} tiles in the source bundle")
        selected = selected[np.argsort(tile_ids_all[selected])]
        tile_ids = tile_ids_all[selected]
        np.testing.assert_array_equal(tile_ids, np.arange(TILE_GRID_SIZE**2))
        detector_indices.extend(selected.tolist())
        modulation_index = int(np.flatnonzero(modulation_detectors == detector)[0])
        ratio_native = modulation_ratios[modulation_index]
        ratio = (
            np.interp(wavelengths_nm, modulation_wave, ratio_native.real)
            + 1j * np.interp(wavelengths_nm, modulation_wave, ratio_native.imag)
        )
        cache_root = Path(cache_roots[detector])
        for tile_id in tile_ids:
            cache_path = cache_root / f"tile_{tile_id:03d}/relative_response.npz"
            source_fit = mock.source_fit_from_cache(cache_path, detector, int(tile_id))
            reflection = _clean_reflection(
                source_fit, wavelengths_nm, base_n, base_k, spectral_basis
            )
            if args.variant == "joint_modulated":
                reflection = reflection * ratio
            reflection_rows.append(reflection.astype(np.complex64))
            source_fits.append(str(source_fit.resolve()))
            row, col = divmod(int(tile_id), TILE_GRID_SIZE)
            tile_records.append((int(detector[3:]), row, col, int(tile_id)))
        print(f"Loaded {args.variant} TMM reflection for {detector} "
              f"(256 tiles, {detector_index}/{len(detectors)})",
              flush=True)

    reflections = np.asarray(reflection_rows, dtype=np.complex64)
    if reflections.shape != (len(detector_indices), wavelengths_nm.size):
        raise ValueError(f"Unexpected TMM reflection array shape: {reflections.shape}")
    if args.mode_count >= min(reflections.shape):
        raise ValueError("Mode count must be smaller than both sample and wavelength counts")

    mean = reflections.mean(axis=0, dtype=np.complex128).astype(np.complex64)
    centered = reflections - mean[None, :]
    print(f"Computing global complex SVD: {centered.shape}, rank {args.mode_count}", flush=True)
    left, singular_values, right = svd(
        centered,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
        overwrite_a=True,
    )
    rank = int(args.mode_count)
    amplitudes = (left[:, :rank] * singular_values[:rank]).astype(np.complex64)
    modes = np.asarray(right[:rank], dtype=np.complex64)
    reconstruction = mean[None, :] + amplitudes @ modes
    residual = reflections.astype(np.complex128) - reconstruction.astype(np.complex128)
    full = reflections.astype(np.complex128)
    relative_error = np.linalg.norm(residual, axis=1) / np.linalg.norm(full, axis=1)
    absolute_error = np.abs(residual)
    total_variance = float(np.sum(np.asarray(singular_values, dtype=np.float64) ** 2))
    explained_fraction = float(
        np.sum(np.asarray(singular_values[:rank], dtype=np.float64) ** 2) / total_variance
    )
    metrics = {
        "rank": rank,
        "explained_variance_fraction": explained_fraction,
        "relative_curve_error_p50": float(np.percentile(relative_error, 50)),
        "relative_curve_error_p95": float(np.percentile(relative_error, 95)),
        "relative_curve_error_max": float(np.max(relative_error)),
        "absolute_complex_error_p95": float(np.percentile(absolute_error, 95)),
        "absolute_complex_error_max": float(np.max(absolute_error)),
        "max_amplitude_reconstruction_abs_error": float(
            np.max(np.abs(reflections - reconstruction))
        ),
    }
    tile_map = np.asarray(tile_records, dtype=np.int16)
    output_metadata = {
        "product_type": "TMM clean complex-reflectance principal modes",
        "mode_convention": "r(lambda) = mean(lambda) + amplitudes @ modes(lambda)",
        "decomposition": "global complex SVD over all selected detector/tile clean-reflection curves",
        "mean_centering": "one global complex mean across all selected tiles",
        "amplitude_type": "complex per-mode coefficient; directly compatible with ice_response_core.reflectance",
        "variant": args.variant,
        "source_reflection": (
            "unmodulated TMM clean stack boundary reflection, viewed from ice; no ice layer"
            if args.variant == "unmodulated" else
            "joint-modulated TMM clean stack boundary reflection, viewed from ice; no ice layer"
        ),
        "joint_modulation_is_baked_into_modes": args.variant == "joint_modulated",
        "source_bundle": str(args.bundle.resolve()),
        "source_bundle_sha256": sha256_file(args.bundle),
        "source_tile_cache_roots": {detector: cache_roots[detector] for detector in detectors},
        "source_fit_paths_dataset": "/superpixels/source_fit_path",
        "detectors": detectors,
        "tiles_per_detector": TILE_GRID_SIZE**2,
        "tile_size_pixels": TILE_SIZE_PIXELS,
        "superpixel_bins_per_side": TILE_GRID_SIZE,
        "wavelength_min_nm": float(wavelengths_nm[0]),
        "wavelength_max_nm": float(wavelengths_nm[-1]),
        "wavelength_step_nm": float(args.wavelength_step_nm),
        "mode_metrics": metrics,
        "joint_bundle_mode_fit_family": metadata.get("modulation_fit_family"),
        "joint_modulation_reference_group": "/joint_modulation",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema"] = "rcs_complex_reflectance_modes/1"
        handle.attrs["mode_product_only"] = True
        handle.attrs["variant"] = args.variant
        handle.attrs["compatible_reflectance_expression"] = "mean + amplitudes @ modes"
        handle.attrs["n_bin"] = TILE_GRID_SIZE
        handle.attrs["superpixel_px"] = TILE_SIZE_PIXELS
        reflectance = handle.create_group("reflectance")
        _write_dataset(reflectance, "wavelength_nm", wavelengths_nm.astype(np.float32))
        _write_dataset(reflectance, "mean", mean)
        _write_dataset(reflectance, "modes", modes)
        _write_dataset(reflectance, "amplitudes", amplitudes)
        _write_dataset(
            reflectance,
            "singular_values",
            np.asarray(singular_values[:rank], dtype=np.float64),
        )
        reflectance.attrs["fit_sample_count"] = int(reflections.shape[0])
        reflectance.attrs["explained_variance_fraction"] = explained_fraction

        geometry = handle.create_group("superpixels")
        _write_dataset(geometry, "sca", tile_map[:, 0])
        _write_dataset(geometry, "row", tile_map[:, 1])
        _write_dataset(geometry, "col", tile_map[:, 2])
        _write_dataset(geometry, "tile_id", tile_map[:, 3])
        _write_dataset(
            geometry,
            "x_pixel_center",
            (tile_map[:, 2] * TILE_SIZE_PIXELS + TILE_SIZE_PIXELS // 2).astype(np.int32),
        )
        _write_dataset(
            geometry,
            "y_pixel_center",
            (tile_map[:, 1] * TILE_SIZE_PIXELS + TILE_SIZE_PIXELS // 2).astype(np.int32),
        )
        geometry.create_dataset(
            "source_fit_path",
            data=np.asarray(source_fits, dtype=h5py.string_dtype("utf-8")),
        )
        geometry.attrs["sca_pixel_side"] = 4096
        geometry.attrs["grid_bins_per_side"] = TILE_GRID_SIZE
        geometry.attrs["tile_size_pixels"] = TILE_SIZE_PIXELS
        geometry.attrs["tile_layout"] = "SCA-major, then row-major; tile_id=row*16+col"

        joint = handle.create_group("joint_modulation")
        joint_wavelengths = np.asarray(
            bundle["modulation_wavelengths_nm"], dtype=np.float32
        )
        mod_detectors = np.asarray(bundle["modulation_detectors"]).astype(str)
        requested_indices = [int(np.flatnonzero(mod_detectors == d)[0]) for d in detectors]
        ratios = (
            np.asarray(bundle["modulation_ratio_real"], dtype=np.float32)
            + 1j * np.asarray(bundle["modulation_ratio_imag"], dtype=np.float32)
        )[requested_indices]
        joint.create_dataset(
            "detectors", data=np.asarray(detectors, dtype=h5py.string_dtype("utf-8"))
        )
        _write_dataset(joint, "wavelength_nm", joint_wavelengths)
        _write_dataset(joint, "ratio", ratios)
        joint.attrs["description"] = (
            "Current bundle detector-global complex multiplier; already applied to the joint-modulated PCA input."
            if args.variant == "joint_modulated" else
            "Optional current bundle detector-global complex multiplier; not included in the TMM PCA modes."
        )
        joint.attrs["applied_to_mode_training_curves"] = args.variant == "joint_modulated"

        provenance = handle.create_group("provenance")
        provenance.attrs["metadata_json"] = json.dumps(output_metadata, sort_keys=True)
        provenance.attrs["source_bundle_path"] = str(args.bundle.resolve())
        provenance.attrs["source_bundle_sha256"] = output_metadata["source_bundle_sha256"]
        provenance.attrs["tmm_cache_roots_json"] = json.dumps(
            output_metadata["source_tile_cache_roots"], sort_keys=True
        )
        handle.flush()

    print(f"Wrote mode product {output_path}")
    print(json.dumps(metrics, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--variant", choices=("unmodulated", "joint_modulated"), default="unmodulated",
        help="TMM clean reflection alone or after applying the current detector-global joint modulation.",
    )
    parser.add_argument("--mode-count", type=int, default=24)
    parser.add_argument("--wavelength-min-nm", type=float, default=475.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=2350.0)
    parser.add_argument("--wavelength-step-nm", type=float, default=1.0)
    parser.add_argument(
        "--detector", action="append", required=True, choices=DETECTORS,
        help="Detector to include; repeat explicitly for multiple detectors.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing mode product")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
