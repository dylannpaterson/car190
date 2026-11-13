"""Assemble a local Eric-v3-loadable prototype from TMM modes and fitted ice."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from astropy.time import Time
from scipy.linalg import svd


ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/wavelength_response_bundle"
)
DEFAULT_BASE = BUNDLE_DIR / "eric_v3_reference/ice_response_model_v3.h5"
DEFAULT_REFLECTANCE = BUNDLE_DIR / "tmm_joint_modulated_reflectance_modes_v1.h5"
DEFAULT_ICE_HISTORY = BUNDLE_DIR / "wfi_joint_2day_ice_history_reference_v1.h5"
DEFAULT_OUTPUT = BUNDLE_DIR / "ice_response_model_tmm_joint_iceside_20261111_prototype.h5"
TMM_ICE_N_REFERENCE = 1.25


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def utc_mjd(value):
    value = h5_string(value).strip()
    if value.endswith("Z"):
        value = value[:-1]
    if value.endswith("+00:00"):
        value = value[:-6]
    return float(Time(value, format="isot", scale="utc").mjd)


def write_compressed(group, name, values, dtype=None, chunks=None):
    values = np.asarray(values, dtype=dtype)
    return group.create_dataset(
        name,
        data=values,
        chunks=chunks,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def tmm_air_to_ice_reflection_modes(
    air_mean, air_modes, air_amplitudes, tile_ice_indices, mode_count
):
    """Transform air-side TMM reflection to the ice-side reflection v3 models.

    The complex reflection produced by the TMM cache is viewed from air:
    ``r_air = (1 - Y_stack) / (1 + Y_stack)`` at normal incidence. Eric's film
    equation needs ``r_ice = (n_ice - Y_stack) / (n_ice + Y_stack)``. The
    transform is nonlinear, so reconstruct tile curves, transform, then refit
    a shared complex PCA basis.
    """
    air_mean = np.asarray(air_mean, dtype=np.complex128)
    air_modes = np.asarray(air_modes, dtype=np.complex128)
    air_amplitudes = np.asarray(air_amplitudes, dtype=np.complex128)
    tile_ice_indices = np.asarray(tile_ice_indices, dtype=float).reshape(-1)
    air_reflection = air_mean[None, :] + air_amplitudes @ air_modes
    if air_reflection.shape[0] != tile_ice_indices.size:
        raise ValueError("TMM reflection curves and ice-index map have different tile counts")
    if np.any(np.abs(1.0 + air_reflection) < 1e-8):
        raise ValueError("Air-side TMM reflection has a pole in the admittance transform")
    stack_admittance = (1.0 - air_reflection) / (1.0 + air_reflection)
    ice_side_reflection = (
        tile_ice_indices[:, None] - stack_admittance
    ) / (tile_ice_indices[:, None] + stack_admittance)

    mean = ice_side_reflection.mean(axis=0, dtype=np.complex128)
    centered = ice_side_reflection - mean[None, :]
    left, singular_values, right = svd(
        centered,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
        overwrite_a=True,
    )
    rank = min(int(mode_count), singular_values.size)
    amplitudes = left[:, :rank] * singular_values[:rank]
    modes = right[:rank]
    reconstruction = mean[None, :] + amplitudes @ modes
    residual = ice_side_reflection - reconstruction
    metrics = {
        "rank": rank,
        "explained_complex_variance_fraction": float(
            np.sum(singular_values[:rank] ** 2) / np.sum(singular_values**2)
        ),
        "relative_curve_error_p50_p95_max": np.percentile(
            np.linalg.norm(residual, axis=1)
            / np.maximum(np.linalg.norm(ice_side_reflection, axis=1), 1e-15),
            [50, 95, 100],
        ).tolist(),
        "absolute_complex_error_max": float(np.max(np.abs(residual))),
    }
    return (
        mean.astype(np.complex64),
        modes.astype(np.complex64),
        amplitudes.astype(np.complex64),
        singular_values[:rank],
        metrics,
    )


def build(args):
    for path in (args.base_v3_h5, args.reflectance_modes, args.ice_history_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite candidate model: {args.output}")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite candidate model: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmm_map = {}
    with h5py.File(args.reflectance_modes, "r") as modes_file:
        mode_attrs = dict(modes_file.attrs)
        reflectance_variant = h5_string(mode_attrs.get("variant", "unmodulated"))
        mode_reflectance = modes_file["reflectance"]
        wavelengths_nm = np.asarray(mode_reflectance["wavelength_nm"], dtype=float)
        mean = np.asarray(mode_reflectance["mean"], dtype=np.complex64)
        modes = np.asarray(mode_reflectance["modes"], dtype=np.complex64)
        coarse_amplitudes = np.asarray(mode_reflectance["amplitudes"], dtype=np.complex64)
        singular_values = np.asarray(mode_reflectance["singular_values"], dtype=float)
        coarse_geometry = modes_file["superpixels"]
        coarse_sca = np.asarray(coarse_geometry["sca"], dtype=int)
        coarse_row = np.asarray(coarse_geometry["row"], dtype=int)
        coarse_col = np.asarray(coarse_geometry["col"], dtype=int)
        coarse_tile_id = np.asarray(coarse_geometry["tile_id"], dtype=int)
        coarse_bins = int(mode_attrs["n_bin"])
        cache_roots = json.loads(
            h5_string(modes_file["provenance"].attrs["tmm_cache_roots_json"])
        )
        for i, key in enumerate(zip(coarse_sca, coarse_row, coarse_col, strict=True)):
            if key in tmm_map:
                raise ValueError(f"Duplicate TMM mode-product superpixel coordinate: {key}")
            tmm_map[key] = i
        joint_modulation = {
            "detectors": modes_file["joint_modulation/detectors"][()],
            "wavelength_nm": np.asarray(modes_file["joint_modulation/wavelength_nm"], dtype=float),
            "ratio": np.asarray(modes_file["joint_modulation/ratio"], dtype=np.complex64),
        }

    coarse_ice_index = np.empty(coarse_sca.size, dtype=float)
    for index, (sca, tile_id) in enumerate(zip(coarse_sca, coarse_tile_id, strict=True)):
        detector = f"WFI{int(sca):02d}"
        cache_path = (
            Path(cache_roots[detector])
            / f"tile_{int(tile_id):03d}/relative_response.npz"
        )
        with np.load(cache_path, allow_pickle=False) as cache:
            ice_n_log = float(
                np.asarray(cache["ice_n_log_constant"], dtype=float).reshape(-1)[0]
            )
            incidence_angle = float(
                np.asarray(cache["incidence_angle_deg"], dtype=float).reshape(-1)[0]
            )
        if abs(ice_n_log) > 1e-12:
            raise ValueError(
                f"{detector}/tile {tile_id}: Eric v3 HDF5 conversion requires one shared TMM ice n; "
                f"cache ice_n_log_constant={ice_n_log:g}"
            )
        if abs(incidence_angle) > 1e-12:
            raise ValueError(
                f"{detector}/tile {tile_id}: Eric v3 HDF5 conversion requires normal-incidence TMM; "
                f"cache incidence angle={incidence_angle:g} deg"
            )
        coarse_ice_index[index] = TMM_ICE_N_REFERENCE * np.exp(ice_n_log)

    air_curves = (
        mean.astype(np.complex128)[None, :]
        + coarse_amplitudes.astype(np.complex128) @ modes.astype(np.complex128)
    )
    clean_relative_weight = np.ones(air_curves.shape, dtype=float)
    if reflectance_variant == 'joint_modulated':
        detector_names = np.asarray(joint_modulation['detectors']).astype(str)
        for sca in np.unique(coarse_sca):
            detector = f'WFI{int(sca):02d}'
            di = int(np.flatnonzero(detector_names == detector)[0])
            ratio = joint_modulation['ratio'][di]
            wave = joint_modulation['wavelength_nm']
            ratio = np.interp(wavelengths_nm, wave, ratio.real) + 1j*np.interp(
                wavelengths_nm, wave, ratio.imag
            )
            selected = coarse_sca == sca
            unmodulated_air = air_curves[selected] / ratio[None, :]
            clean_relative_weight[selected] = np.abs(
                (1 + air_curves[selected]) / (1 + unmodulated_air)
            )**2
    del air_curves

    mean, modes, coarse_amplitudes, singular_values, reflection_transform_metrics = (
        tmm_air_to_ice_reflection_modes(
            mean, modes, coarse_amplitudes, coarse_ice_index, modes.shape[0]
        )
    )

    with h5py.File(args.ice_history_reference, "r") as history_file:
        history_metadata = dict(history_file.attrs)
        reference_utc = h5_string(history_file.attrs["ice_reference_utc"])
        reset_days = np.asarray(
            history_file.attrs["reset_days_from_reference"], dtype=float
        )
        history_group = history_file["v3_rank1_deposition_candidate"]
        time_hours = np.asarray(history_group["time_hours"], dtype=float)
        episode = np.asarray(history_group["episode"], dtype=np.int8)
        blocks = np.asarray(history_group["blocks"], dtype=float)
        multipliers = np.asarray(history_group["multipliers"], dtype=float)
        tau_h = np.asarray(history_group["tau_h"], dtype=float)
        offset_episodes = np.asarray(history_group["offset_episodes"], dtype=int)
        coarse_log_a = np.asarray(history_group["log_a"], dtype=float)
        coarse_d_off = np.asarray(history_group["d_off"], dtype=float)
        coarse_rate = np.asarray(history_group["rate_nm_per_day"], dtype=float)
        rank1_metrics = json.loads(history_group.attrs["fit_metrics_json"])
        timestamp_utc = np.asarray(
            history_file["bundle_observation_axis/timestamp_utc"].asstr()[()]
        )

    source_bundle_path = Path(h5_string(history_metadata["source_bundle"]))
    with np.load(source_bundle_path, allow_pickle=False) as source_bundle:
        bundle_detectors = np.asarray(source_bundle["detectors"]).astype(str)
        bundle_tile_ids = np.asarray(source_bundle["tile_ids"], dtype=int)
        post_reset_offsets = np.asarray(
            source_bundle["post_reset_tile_log_offset"], dtype=float
        )
        if "post_third_decon_tile_log_offset" not in source_bundle:
            raise ValueError("The source bundle has no fitted third-decon gray offsets")
        post_third_offsets = np.asarray(
            source_bundle["post_third_decon_tile_log_offset"], dtype=float
        )
    if not (
        bundle_detectors.shape
        == bundle_tile_ids.shape
        == post_reset_offsets.shape
        == post_third_offsets.shape
    ):
        raise ValueError("Source-bundle gray arrays do not share the tile axis")
    if reset_days.shape != (2,) or np.any(np.diff(reset_days) <= 0.0):
        raise ValueError(f"Expected two ordered fitted gray resets, received {reset_days}")
    bundle_gray_index = {}
    for index, (detector, tile_id) in enumerate(
        zip(bundle_detectors, bundle_tile_ids, strict=True)
    ):
        key = (int(str(detector)[3:]), int(tile_id))
        if key in bundle_gray_index:
            raise ValueError(f"Duplicate source-bundle tile for fitted gray factors: {key}")
        bundle_gray_index[key] = index

    if blocks.shape != (multipliers.size, 3):
        raise ValueError("v3 deposition block and multiplier dimensions do not match")
    if time_hours.shape != episode.shape or time_hours.shape != timestamp_utc.shape:
        raise ValueError("Deposition time, episode, and timestamp axes differ")
    if coarse_d_off.shape[0] != coarse_log_a.size or coarse_rate.shape[0] != coarse_log_a.size:
        raise ValueError("Coarse deposition arrays do not share a superpixel map")
    if coarse_rate.shape[1] != multipliers.size:
        raise ValueError("Per-pixel v3 rate map does not match deposition block count")

    # Match rates' time reference to the copied v3 instrument history. The
    # current RCS bundle is referenced a few minutes before the base release.
    with h5py.File(args.base_v3_h5, "r") as base:
        if h5_string(base.attrs.get("schema", "")) != "rcs_response_model/1":
            raise ValueError("Base file is not an Eric v3 response-model HDF5")
        base_bins = int(base.attrs["n_bin"])
        base_reflectance_wave = np.asarray(base["reflectance/wavelength_nm"], dtype=float)
        base_t0 = h5_string(base["deposition"].attrs["t0_utc"])
        if not np.array_equal(base_reflectance_wave, wavelengths_nm):
            raise ValueError("Base v3 and TMM mode-product wavelength grids differ")
        if base_bins % coarse_bins != 0:
            raise ValueError(f"Base n_bin={base_bins} is not divisible by TMM n_bin={coarse_bins}")
        factor = base_bins // coarse_bins
        fine_sca = np.asarray(base["superpixels/sca"], dtype=int)
        fine_row = np.asarray(base["superpixels/row"], dtype=int)
        fine_col = np.asarray(base["superpixels/col"], dtype=int)
        fine_count = fine_sca.size
        if fine_row.shape != (fine_count,) or fine_col.shape != (fine_count,):
            raise ValueError("Base v3 superpixel coordinate arrays have incompatible shapes")
        fine_keys = list(zip(fine_sca, fine_row, fine_col, strict=True))
        fine_index = np.fromiter(
            (
                tmm_map[(int(sca), int(row // factor), int(col // factor))]
                for sca, row, col in fine_keys
            ),
            dtype=np.int64,
            count=fine_count,
        )
        if np.any((fine_row < 0) | (fine_row >= base_bins) | (fine_col < 0) | (fine_col >= base_bins)):
            raise ValueError("Base v3 row/column indices lie outside its superpixel grid")
        fine_tile_id = (fine_row // factor) * coarse_bins + fine_col // factor
        gray_indices = np.fromiter(
            (
                bundle_gray_index[(int(sca), int(tile_id))]
                for sca, tile_id in zip(fine_sca, fine_tile_id, strict=True)
            ),
            dtype=np.int64,
            count=fine_count,
        )
        first_log_gray = post_reset_offsets[gray_indices]
        second_log_gray = post_third_offsets[gray_indices]
        fitted_gray = np.column_stack(
            (np.expm1(first_log_gray), np.exp(second_log_gray) - np.exp(first_log_gray))
        ).astype(np.float64)
        if not np.isfinite(fitted_gray).all():
            raise ValueError("Nonfinite fitted gray factors while expanding to Eric superpixels")
        gray_time_shift_hours = (utc_mjd(base_t0) - utc_mjd(reference_utc)) * 24.0

    with h5py.File(args.base_v3_h5, "r") as base, h5py.File(args.output, "w") as out:
        for key, value in base.attrs.items():
            out.attrs[key] = value
        for name in base:
            if name not in ("reflectance", "deposition"):
                base.copy(name, out)

        out.attrs["release_version"] = f"tmm_{reflectance_variant}_20261111_prototype"
        out.attrs["release_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out.attrs["n_modes"] = int(modes.shape[0])
        out.attrs["release_note"] = (
            f"Experimental local prototype: {reflectance_variant} TMM ice-side reflectance modes and a rank-one "
            "v3-compatible approximation to the current joint 2-day per-tile ice history. "
            "Not an official release; see prototype_provenance."
        )

        instrument = out["instrument"]
        # Fitted gray and deposition epochs share reference_utc. Translate only
        # the original calibration-step UTC boundaries; the current fit's gray
        # reset boundaries are already relative to the new model origin.
        if "gray_period_hours" in instrument:
            instrument["gray_period_hours"][...] = reset_days * 24.0
        if "step_boundary_hours" in instrument:
            instrument["step_boundary_hours"][...] = (
                np.asarray(instrument["step_boundary_hours"], dtype=float)
                + gray_time_shift_hours
            )
        instrument["gray"][...] = fitted_gray
        instrument.attrs["fitted_gray_source_bundle"] = str(source_bundle_path.resolve())
        instrument.attrs["time_origin_shift_hours_for_joint_prototype"] = float(
            gray_time_shift_hours
        )

        film = out["film"]
        film["n"][...] = TMM_ICE_N_REFERENCE
        film["kappa"][...] = 0.0
        film["b"][...] = (1.0 - TMM_ICE_N_REFERENCE) / (1.0 + TMM_ICE_N_REFERENCE)
        film.attrs["n_snell_ref"] = TMM_ICE_N_REFERENCE
        film.attrs["chromatic_snell"] = False
        film.attrs["tmm_conversion_note"] = (
            "Matches the source cached TMM: lossless n=1.25, normal incidence. "
            "The original Eric-v3 dispersive film optics are not used for this TMM model."
        )
        out["superpixels/cos_theta"][...] = 1.0

        reflectance = out.create_group("reflectance")
        reflectance.attrs["note"] = (
            "r(lambda) = mean + amplitudes @ modes; clean TMM-stack reflection "
            "transformed to Eric's ice-side convention"
        )
        reflectance.attrs["source_variant"] = (
            f"{reflectance_variant} TMM clean stack, ice-side reflection, no ice layer"
        )
        write_compressed(reflectance, "wavelength_nm", wavelengths_nm.astype(np.float32))
        write_compressed(reflectance, "mean", mean)
        write_compressed(reflectance, "modes", modes)
        write_compressed(reflectance, "singular_values", singular_values.astype(np.float64))
        labels = np.asarray([f"TMM_PC_{i + 1:02d}" for i in range(modes.shape[0])],
                            dtype=h5py.string_dtype("utf-8"))
        reflectance.create_dataset("mode_labels", data=labels)
        amplitude_dataset = reflectance.create_dataset(
            "amplitudes",
            shape=(fine_count, modes.shape[0]),
            dtype=np.complex64,
            chunks=(min(4096, fine_count), modes.shape[0]),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        for start in range(0, fine_count, 4096):
            stop = min(start + 4096, fine_count)
            amplitude_dataset[start:stop] = coarse_amplitudes[fine_index[start:stop]]

        deposition = out.create_group("deposition")
        deposition.attrs["t0_utc"] = reference_utc + "+00:00"
        deposition.attrs["v3_parameterization"] = (
            "single spatial log_a map times shared 2-day rate-block multipliers, with per-episode d_off"
        )
        deposition.attrs["current_joint_history_reference_utc"] = reference_utc
        deposition.attrs["time_origin_shift_hours_from_base_release"] = float(
            gray_time_shift_hours
        )
        write_compressed(deposition, "time_hours", time_hours)
        write_compressed(deposition, "episode", episode)
        deposition.create_dataset("timestamp_utc", data=timestamp_utc.astype(h5py.string_dtype("utf-8")))
        write_compressed(deposition, "blocks", blocks)
        write_compressed(deposition, "multipliers", multipliers)
        write_compressed(deposition, "tau_h", tau_h)
        write_compressed(deposition, "offset_episodes", offset_episodes)
        write_compressed(deposition, "log_a", coarse_log_a[fine_index].astype(np.float32))
        write_compressed(deposition, "d_off", coarse_d_off[fine_index].astype(np.float32))
        rate_dataset = deposition.create_dataset(
            "rate_nm_per_day",
            shape=(fine_count, coarse_rate.shape[1]),
            dtype=np.float32,
            chunks=(min(2048, fine_count), coarse_rate.shape[1]),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        for start in range(0, fine_count, 2048):
            stop = min(start + 2048, fine_count)
            rate_dataset[start:stop] = coarse_rate[fine_index[start:stop]].astype(np.float32)
        deposition.attrs["rank1_history_fit_metrics_json"] = json.dumps(rank1_metrics, sort_keys=True)

        joint_group = out.create_group("joint_modulation_reference")
        with h5py.File(args.reflectance_modes, "r") as mode_file:
            j = mode_file["joint_modulation"]
            joint_group.create_dataset("detectors", data=j["detectors"][()])
            for key in ("wavelength_nm", "ratio"):
                write_compressed(joint_group, key, j[key][()])
        joint_group.attrs["already_applied_to_reflectance_modes"] = (
            reflectance_variant == "joint_modulated"
        )

        photometry = out.create_group('photometry')
        photometry.attrs['normalization'] = 'modulated_clean_power / unmodulated_clean_power'
        photometry.attrs['usage'] = (
            'For reproduction of the stellar fit, integrate photon_SED * effective_area * '
            'clean_relative_weight * response(gray=False), without renormalizing separately '
            'at each position. Apply saved Amber roll offsets separately. '
            'This group stores response curves only, no AR parameters.'
        )
        write_compressed(photometry, 'wavelength_nm', wavelengths_nm)
        write_compressed(photometry, 'clean_relative_weight', clean_relative_weight)
        write_compressed(photometry, 'superpixel_to_tile', fine_index.astype(np.int32))

        provenance = out.create_group("prototype_provenance")
        provenance.attrs["base_v3_h5"] = str(args.base_v3_h5.resolve())
        provenance.attrs["base_v3_h5_sha256"] = sha256_file(args.base_v3_h5)
        provenance.attrs["reflectance_mode_product"] = str(args.reflectance_modes.resolve())
        provenance.attrs["reflectance_mode_product_sha256"] = sha256_file(args.reflectance_modes)
        provenance.attrs["reflectance_variant"] = reflectance_variant
        provenance.attrs["ice_history_reference"] = str(args.ice_history_reference.resolve())
        provenance.attrs["ice_history_reference_sha256"] = sha256_file(args.ice_history_reference)
        provenance.attrs["tmm_coarse_bins_per_side"] = coarse_bins
        provenance.attrs["base_bins_per_side"] = base_bins
        provenance.attrs["coarse_amplitudes_upsampled_by_nearest_tile_copy"] = factor
        provenance.attrs["source_reflection_side"] = "air"
        provenance.attrs["target_reflection_side"] = "ice"
        provenance.attrs["air_to_ice_reflection_transform"] = (
            "Y_stack=(1-r_air)/(1+r_air); r_ice=(n_ice-Y_stack)/(n_ice+Y_stack)"
        )
        provenance.attrs["tmm_ice_n_reference"] = TMM_ICE_N_REFERENCE
        provenance.attrs["tmm_incidence_angle_deg"] = 0.0
        provenance.attrs["ice_side_reflection_pca_metrics_json"] = json.dumps(
            reflection_transform_metrics, sort_keys=True
        )
        provenance.attrs["deposition_rank1_metrics_json"] = json.dumps(rank1_metrics, sort_keys=True)
        provenance.attrs["quality_note"] = (
            "Air-side TMM reflectance modes are transformed to Eric's ice-side convention and refit "
            "at the same rank. The resulting 256x256-pixel TMM tile coefficients are copied to the "
            "finer base superpixels. Deposition remains a rank-one approximation to the saved joint "
            "history; its residual must be reviewed before any release use."
        )
        out.flush()
    print(f"Wrote local v3-loadable prototype {args.output}")
    print(f"n_bin={base_bins}, n_modes={modes.shape[0]}, superpixels={fine_count}, blocks={len(multipliers)}")
    print(json.dumps(reflection_transform_metrics, indent=2))
    print(json.dumps(rank1_metrics, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-v3-h5", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--reflectance-modes", type=Path, default=DEFAULT_REFLECTANCE)
    parser.add_argument("--ice-history-reference", type=Path, default=DEFAULT_ICE_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing prototype output")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
