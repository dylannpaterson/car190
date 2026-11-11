"""Heatmap of all-star CAR201 observations corrected by saved detector models.

No model coefficients are refit. Successful faint-v2 SEDs are preferred, then
successful original SEDs; remaining sources borrow the SED of the nearest
successful same-detector source in the eight-band CAR201 magnitude colors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from astropy.time import Time

import fit_car201_zane_star_reflectance_modulation as fit_model
import mock_fit_wfi_star_reflectance_modulation as mock_model


ROOT = Path(__file__).resolve().parent
CAR201_ROOT = ROOT.parent / "CAR201"
PRODUCTION_ROOT = ROOT / "20261106/corrected_statistics_4h_thinned/production"
PHOTOMETRY_DIR = CAR201_ROOT / "zanes_photometry/20261106"
FAINT_SED_PATH = CAR201_ROOT / "sed/v2/faint-posteriors-v2.h5"
ORIGINAL_SED_PATH = CAR201_ROOT / "sed/v2/car201-btsettl-posteriors.h5"
SED_GRID_PATH = CAR201_ROOT / "sed/v2/btsettl-spectral-grid.h5"
FILTERS = tuple(mock_model.FILTERS)
DETECTORS = tuple(f"WFI{i:02d}" for i in range(1, 19))
PASSES = tuple(fit_model.PASS_SET_CONFIG["ep3v2_ep4"]["passes"])
LOG_FLUX_PER_MAG = float(mock_model.LOG_FLUX_PER_MAG)
TILE_SIZE_PIXELS = 256
TILE_GRID_SIZE = 16
DETECTOR_PIXELS = 4096
DEFAULT_OUTPUT_PREFIX = (
    PRODUCTION_ROOT
    / "wfi01/spline_relative_response_normal/experiments/"
    "car201_zane_wfi01_ep3v2_ep4_faint_v2_tile_matched/"
    "car201_allstar_corrected_within_0p5"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--unmodulated", action="store_true")
    return parser.parse_args(argv)


def _decode_source_id(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("ascii").rstrip("\x00")
    return str(value)


def _load_successful_seds(paths, source_ids):
    wanted = set(source_ids)
    result = {}
    for path, source_name in paths:
        with h5py.File(path, "r") as archive:
            ids = [_decode_source_id(value) for value in archive["source_id"][:]]
            states = np.asarray(archive["state"][:], dtype=int)
            band_values = (
                archive["bands"][:]
                if "bands" in archive
                else archive["band_count"][:]
            )
            candidates = [
                (index, source_id)
                for index, (source_id, state) in enumerate(zip(ids, states, strict=True))
                if source_id in wanted and state == 1 and source_id not in result
            ]
            candidates.sort()
            if not candidates:
                continue
            indices = [index for index, _ in candidates]
            samples = np.asarray(archive["samples"][indices], dtype=float)
            for sample_index, (archive_index, source_id) in enumerate(candidates):
                result[source_id] = {
                    "parameters_median": np.median(samples[sample_index], axis=0),
                    "state": 1,
                    "bands": int(band_values[archive_index]),
                    "chi2": float(archive["chi2"][archive_index]),
                    "sed_source": source_name,
                    "sed_donor_id": source_id,
                    "nn_distance": 0.0,
                }
    return result


def _load_detector_frames(detector, start_mjd, end_mjd):
    frames = []
    for pass_name in PASSES:
        frame = fit_model.load_pass(
            PHOTOMETRY_DIR / f"{pass_name}_master.parquet",
            start_mjd,
            end_mjd,
            detector,
            include_invalid=True,
        )
        missing = {"x_pixel", "y_pixel"} - set(frame.columns)
        if missing:
            raise ValueError(f"{pass_name}/{detector} lacks coordinates {sorted(missing)}")
        frames.append(frame)
    return frames


def _reference_magnitude_vectors(frames, source_ids):
    """Use the earliest pass with the most valid bands for each source."""
    vectors = {
        source_id: np.full(len(FILTERS), np.nan, dtype=float)
        for source_id in source_ids
    }
    best_band_count = {source_id: 0 for source_id in source_ids}
    for frame in frames:
        valid = frame.loc[
            np.isfinite(pd.to_numeric(frame["mag_obs"], errors="coerce"))
            & np.isfinite(pd.to_numeric(frame["mag_err"], errors="coerce"))
            & (pd.to_numeric(frame["mag_err"], errors="coerce") > 0.0)
        ]
        if valid.empty:
            continue
        matrix = valid.pivot(index="source_id", columns="filter", values="mag_obs")
        matrix = matrix.reindex(columns=FILTERS)
        for source_id, row in matrix.iterrows():
            if source_id not in vectors:
                continue
            values = row.to_numpy(dtype=float)
            band_count = int(np.count_nonzero(np.isfinite(values)))
            if band_count > best_band_count[source_id]:
                vectors[source_id] = values
                best_band_count[source_id] = band_count
    return vectors, best_band_count


def _nearest_sed_donor(query, donor_ids, reference_vectors, sed_by_id):
    query = np.asarray(query, dtype=float)
    query_valid = np.isfinite(query)
    scores = []
    for donor_id in donor_ids:
        donor = np.asarray(reference_vectors[donor_id], dtype=float)
        common = query_valid & np.isfinite(donor)
        count = int(np.count_nonzero(common))
        if count == 0:
            continue
        if count >= 2:
            query_color = query[common] - np.median(query[common])
            donor_color = donor[common] - np.median(donor[common])
            scale = 0.25
        else:
            query_color = query[common]
            donor_color = donor[common]
            scale = 1.0
        distance = float(np.sqrt(np.mean(np.square((query_color - donor_color) / scale))))
        # Prefer posterior fits supported by more external photometric bands if
        # two candidates have effectively identical CAR201 colors.
        sed_bands = int(sed_by_id[donor_id]["bands"])
        scores.append((distance, -sed_bands, donor_id, count))
    if not scores:
        raise ValueError("No successful same-detector SED donor overlaps the source colors")
    distance, _negative_bands, donor_id, overlap_count = min(scores)
    return donor_id, distance, overlap_count


def _assign_seds_for_detector(detector, frames, source_ids, sed_by_id):
    reference_vectors, band_counts = _reference_magnitude_vectors(frames, source_ids)
    known = [source_id for source_id in source_ids if source_id in sed_by_id]
    if not known:
        raise ValueError(f"No successful posterior donors are available for {detector}")
    high_quality = [source_id for source_id in known if sed_by_id[source_id]["bands"] >= 4]
    donors = high_quality if len(high_quality) >= 10 else known

    assignments = {}
    records = []
    for source_id in source_ids:
        if source_id in sed_by_id:
            assignments[source_id] = sed_by_id[source_id]
            item = sed_by_id[source_id]
            records.append(
                {
                    "detector": detector,
                    "source_id": source_id,
                    "sed_source": item["sed_source"],
                    "sed_donor_id": item["sed_donor_id"],
                    "sed_bands": item["bands"],
                    "nearest_neighbor_distance": 0.0,
                    "magnitude_bands_available": band_counts[source_id],
                }
            )
            continue
        donor_id, distance, overlap_count = _nearest_sed_donor(
            reference_vectors[source_id], donors, reference_vectors, sed_by_id
        )
        donor = sed_by_id[donor_id]
        assignments[source_id] = {
            **donor,
            "sed_source": "nearest_neighbor_magnitudes",
            "sed_donor_id": donor_id,
            "nn_distance": distance,
        }
        records.append(
            {
                "detector": detector,
                "source_id": source_id,
                "sed_source": "nearest_neighbor_magnitudes",
                "sed_donor_id": donor_id,
                "sed_bands": donor["bands"],
                "nearest_neighbor_distance": distance,
                "nearest_neighbor_shared_filters": overlap_count,
                "magnitude_bands_available": band_counts[source_id],
            }
        )
    return assignments, records


def _make_wavelength_grid(stride_nm, effective_area_path=None, full_effective_area_support=False):
    ea_wavelengths, effective_area = mock_model.load_effective_area(
        effective_area_path or mock_model.DEFAULT_EFFECTIVE_AREA
    )
    if full_effective_area_support:
        supported=np.flatnonzero(np.any(effective_area>0.,axis=0))
        if not len(supported) or supported[0]==0 or supported[-1]==len(ea_wavelengths)-1:
            raise ValueError("Effective-area support must have zero-throughput samples on both sides")
        lower,upper=ea_wavelengths[supported[0]],ea_wavelengths[supported[-1]]
        zero_lower,zero_upper=ea_wavelengths[supported[0]-1],ea_wavelengths[supported[-1]+1]
        start=np.floor(zero_lower/stride_nm)*stride_nm
        stop=np.ceil(zero_upper/stride_nm)*stride_nm
        grid=np.unique(np.concatenate((np.arange(start,stop+.5*stride_nm,stride_nm),
                                        [zero_lower,lower,upper,zero_upper])))
        area=np.asarray([np.interp(grid,ea_wavelengths,band) for band in effective_area],dtype=float)
        if np.any(area[:,0]!=0.) or np.any(area[:,-1]!=0.):
            raise ValueError("Integration grid does not bracket the full effective-area support")
        return grid,area
    full_grid = np.arange(400.0, 2321.0 + 0.5 * stride_nm, stride_nm)
    area = np.asarray(
        [np.interp(full_grid, ea_wavelengths, band) for band in effective_area],
        dtype=float,
    )
    supported = np.any(area > 0.0, axis=0)
    return full_grid[supported], area[:, supported]


def _tile_histories(detector, target_days_by_filter, history_root=None):
    history_root = (
        Path(history_root)
        if history_root is not None
        else PRODUCTION_ROOT / detector.lower() / "spline_relative_response_normal/fits"
    )
    archives = sorted(history_root.glob("tile_*/posterior_samples.npz"))
    if len(archives) != TILE_GRID_SIZE**2:
        raise ValueError(f"Expected 256 fixed ice histories for {detector}, found {len(archives)}")
    days_by_id = [None] * (TILE_GRID_SIZE**2)
    thickness_by_id = [None] * (TILE_GRID_SIZE**2)
    reset_days = None
    target_flat = np.asarray(target_days_by_filter, dtype=float).reshape(-1)
    for path in archives:
        tile_id = int(path.parent.name.split("_")[1])
        with np.load(path, allow_pickle=False) as archive:
            days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
            thickness = np.asarray(archive["tile_ice_thickness"], dtype=float)
            thickness = thickness.reshape((-1, days.size))[0]
            local_resets = np.sort(
                np.asarray(
                    archive.get("ice_reset_days", archive.get("ice_reset_day", [])),
                    dtype=float,
                ).reshape(-1)
            )
        if reset_days is None:
            reset_days = local_resets
        elif not np.allclose(reset_days, local_resets, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"Fixed ice reset days disagree for {detector}")
        days_by_id[tile_id] = days
        thickness_by_id[tile_id] = mock_model.interpolate_fixed_history(
            days, thickness, reset_days, target_flat
        )
    return np.asarray(thickness_by_id, dtype=float).reshape(
        (TILE_GRID_SIZE**2, len(FILTERS), len(PASSES))
    )


def _source_tile_ids(source_ids, frames):
    source_lookup = {source_id: i for i, source_id in enumerate(source_ids)}
    filter_lookup = {filter_name: i for i, filter_name in enumerate(FILTERS)}
    pass_lookup = {pass_name: i for i, pass_name in enumerate(PASSES)}
    default_by_source = np.full(len(source_ids), -1, dtype=np.int16)
    tile_ids = np.full(
        (len(source_ids), len(FILTERS), len(PASSES)), -1, dtype=np.int16
    )
    for pass_name, frame in zip(PASSES, frames, strict=True):
        pass_index = pass_lookup[pass_name]
        for row in frame.itertuples(index=False):
            source_index = source_lookup.get(row.source_id)
            filter_index = filter_lookup.get(row.filter)
            if source_index is None or filter_index is None:
                continue
            x_pixel = float(row.x_pixel) if pd.notna(row.x_pixel) else np.nan
            y_pixel = float(row.y_pixel) if pd.notna(row.y_pixel) else np.nan
            if not np.isfinite(x_pixel) or not np.isfinite(y_pixel):
                continue
            if not (0.0 <= x_pixel < DETECTOR_PIXELS and 0.0 <= y_pixel < DETECTOR_PIXELS):
                raise ValueError(f"Out-of-range source coordinate for {row.source_id}")
            tile_id = int(y_pixel // TILE_SIZE_PIXELS) * TILE_GRID_SIZE + int(
                x_pixel // TILE_SIZE_PIXELS
            )
            tile_ids[source_index, filter_index, pass_index] = tile_id
            if default_by_source[source_index] < 0:
                default_by_source[source_index] = tile_id
    if np.any(default_by_source < 0):
        raise ValueError("At least one source has no valid detector coordinates")
    for source_index, default_tile in enumerate(default_by_source):
        tile_ids[source_index][tile_ids[source_index] < 0] = default_tile
    return tile_ids


def _optical_tile_arrays(detector, wavelengths_nm, optical_cache_root):
    production = fit_model.production
    complex_model = fit_model.complex_model
    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths_nm)
    spectral_basis = production.spectral_modulation_basis(wavelengths_nm)
    tile_count = TILE_GRID_SIZE**2
    clean_reflection = np.empty((tile_count, wavelengths_nm.size), dtype=complex)
    clean_transfer = np.empty_like(clean_reflection)
    clean_power = np.empty((tile_count, wavelengths_nm.size), dtype=float)
    incident_y = np.empty(tile_count, dtype=float)
    substrate_y = np.empty_like(clean_reflection)
    incidence_sine = np.empty(tile_count, dtype=float)
    ice_index = np.empty(tile_count, dtype=float)
    for tile_id in range(tile_count):
        cache = optical_cache_root / f"tile_{tile_id:03d}/relative_response.npz"
        source_fit = mock_model.source_fit_from_cache(cache, detector, tile_id)
        state = complex_model.load_state(
            source_fit, wavelengths_nm, base_n, base_k, spectral_basis
        )
        first, second, local_incident_y, local_substrate_y, local_sine = (
            mock_model.clean_stack_boundary(state, wavelengths_nm)
        )
        reflection = (local_incident_y * first - second) / (
            local_incident_y * first + second
        )
        power = np.real(local_substrate_y / local_incident_y) * np.abs(
            2.0 * local_incident_y
            / (local_incident_y * first + second)
        ) ** 2
        clean_reflection[tile_id] = reflection
        clean_transfer[tile_id] = first
        clean_power[tile_id] = power
        incident_y[tile_id] = local_incident_y
        substrate_y[tile_id] = local_substrate_y
        incidence_sine[tile_id] = local_sine
        ice_index[tile_id] = state["ice_index"]
        if (tile_id + 1) % 64 == 0:
            print(f"  {detector}: prepared optical tile {tile_id + 1}/256", flush=True)
    return (
        clean_reflection,
        clean_transfer,
        clean_power,
        incident_y,
        substrate_y,
        incidence_sine,
        ice_index,
    )


def _load_fit_parameters(detector, fit_dir, summary):
    parameters = pd.read_csv(fit_dir / "car201_actual_star_modulation_parameters.csv")
    expected_names = list(summary["modulation"]["mode_names"])
    parameters = parameters.set_index("mode").loc[expected_names]
    coefficients = np.concatenate(
        (
            parameters["fit_log_amplitude"].to_numpy(float),
            parameters["fit_phase_rad"].to_numpy(float),
        )
    )
    zero_points = pd.read_csv(fit_dir / "car201_actual_roll_zero_points.csv")
    return coefficients, zero_points


def _predict_detector(
    detector,
    frames,
    source_ids,
    sed_assignments,
    sed_grid_path,
    fit_dir,
    wavelengths_nm,
    effective_area,
    ice_reference_mjd,
    unmodulated=False,
    return_model=False,
    effective_area_path=None,
    full_effective_area_support=False,
    ice_history_fit_root=None,
    optical_cache_root=None,
):
    with (fit_dir / "car201_actual_star_modulation_fit_summary.json").open() as handle:
        summary = json.load(handle)
    stride = float(summary["modulation"]["wavelength_stride_nm"])
    expected_wavelengths, expected_area = _make_wavelength_grid(stride, effective_area_path=effective_area_path,
                                                               full_effective_area_support=full_effective_area_support)
    if not np.array_equal(expected_wavelengths, wavelengths_nm):
        raise ValueError(f"Wavelength grid mismatch for {detector}")
    if expected_area.shape != effective_area.shape:
        raise ValueError(f"Effective-area grid mismatch for {detector}")

    mjd_by_filter = np.empty((len(FILTERS), len(PASSES)), dtype=float)
    for pass_index, frame in enumerate(frames):
        for filter_index, filter_name in enumerate(FILTERS):
            times = frame.loc[frame["filter"] == filter_name, "mjd"].to_numpy(float)
            mjd_by_filter[filter_index, pass_index] = float(np.median(times))
    target_days_by_filter = mjd_by_filter - ice_reference_mjd
    tile_thickness = _tile_histories(
        detector, target_days_by_filter, history_root=ice_history_fit_root
    )
    tile_ids = _source_tile_ids(source_ids, frames)
    source_seds = {source_id: sed_assignments[source_id] for source_id in source_ids}
    star_photon_shapes, _sed_parameters, _fallback_count = (
        fit_model.reconstruct_sed_photon_shapes(
            source_ids, source_seds, sed_grid_path, wavelengths_nm
        )
    )

    optical_root = (
        Path(optical_cache_root)
        if optical_cache_root is not None
        else Path(summary["ice_history"]["optical_state_source"])
    )
    optical_arrays = _optical_tile_arrays(detector, wavelengths_nm, optical_root)
    model = fit_model.RealPhotometryFit(
        wavelengths_nm,
        effective_area,
        star_photon_shapes,
        tile_thickness,
        tile_ids,
        *optical_arrays,
        summary["modulation"]["basis"],
        pass_zero_point_group_ids=summary["modulation"]["zeropoint_model"][
            "pass_group_ids"
        ],
    )
    model.detector = detector
    coefficients, zero_points = _load_fit_parameters(detector, fit_dir, summary)
    if unmodulated:
        coefficients = np.zeros_like(coefficients)
    if coefficients.size != 2 * model.mode_count:
        raise ValueError(f"Saved modulation coefficients do not match basis for {detector}")
    predicted_log_response = model.modeled_log_response(coefficients)
    group_names = summary["modulation"]["zeropoint_model"]["group_names"]
    pass_group_ids = summary["modulation"]["zeropoint_model"]["pass_group_ids"]
    for pass_index, group_id in enumerate(pass_group_ids):
        if int(group_id) == 0:
            continue
        offset_group = str(group_names[int(group_id)]).removesuffix("_reference")
        for filter_index, filter_name in enumerate(FILTERS):
            selected = zero_points.loc[
                (zero_points["filter"] == filter_name)
                & (zero_points["offset_group"] == offset_group),
                "fit_log_flux_offset",
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Missing unique {offset_group} zero-point for {detector}/{filter_name}"
                )
            predicted_log_response[:, filter_index, pass_index] += float(
                selected.iloc[0]
            )
    predicted_delta_mag = -predicted_log_response / LOG_FLUX_PER_MAG
    return (predicted_delta_mag, model) if return_model else predicted_delta_mag


def _make_source_rows(detector, frames, source_ids, predicted_delta_mag):
    source_lookup = {source_id: i for i, source_id in enumerate(source_ids)}
    filter_lookup = {filter_name: i for i, filter_name in enumerate(FILTERS)}
    records = []
    for pass_index, (pass_name, frame) in enumerate(zip(PASSES, frames, strict=True)):
        mag = pd.to_numeric(frame["mag_obs"], errors="coerce")
        err = pd.to_numeric(frame["mag_err"], errors="coerce")
        valid = np.isfinite(mag) & np.isfinite(err) & (err > 0.0)
        selected = frame.loc[valid].copy()
        for row in selected.itertuples(index=False):
            source_index = source_lookup[row.source_id]
            filter_index = filter_lookup[row.filter]
            records.append(
                {
                    "detector": detector,
                    "source_id": row.source_id,
                    "filter": row.filter,
                    "pass": pass_name,
                    "pass_index": pass_index,
                    "mag_obs": float(row.mag_obs),
                    "mag_err": float(row.mag_err),
                    "dq_flag": int(row.dq_flag) if pd.notna(row.dq_flag) else -1,
                    "model_delta_mag": float(
                        predicted_delta_mag[source_index, filter_index, pass_index]
                    ),
                    "day_from_fixed_ice_reference": float(row.mjd - Time(
                        fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc"
                    ).mjd),
                }
            )
    return pd.DataFrame.from_records(records)


def _assign_relative_changes(observations):
    observations = observations.sort_values(
        ["detector", "source_id", "filter", "pass_index"]
    ).copy()
    grouped = observations.groupby(["detector", "source_id", "filter"], sort=False)
    observations["anchor_pass"] = grouped["pass"].transform("first")
    observations["anchor_mag"] = grouped["mag_obs"].transform("first")
    observations["anchor_model_mag"] = grouped["model_delta_mag"].transform("first")
    observations["observations_per_source_filter"] = grouped["mag_obs"].transform("size")
    observations["is_anchor_observation"] = grouped.cumcount() == 0
    observations["raw_delta_mag"] = observations["mag_obs"] - observations["anchor_mag"]
    observations["model_delta_from_anchor_mag"] = (
        observations["model_delta_mag"] - observations["anchor_model_mag"]
    )
    observations["corrected_delta_mag"] = (
        observations["raw_delta_mag"] - observations["model_delta_from_anchor_mag"]
    )
    observations["raw_flux_change_percent"] = np.expm1(
        -LOG_FLUX_PER_MAG * observations["raw_delta_mag"].to_numpy(float)
    ) * 100.0
    observations["corrected_flux_change_percent"] = np.expm1(
        -LOG_FLUX_PER_MAG * observations["corrected_delta_mag"].to_numpy(float)
    ) * 100.0
    return observations


def _heatmap_table(observations):
    records = []
    repeated = observations.loc[
        observations["observations_per_source_filter"] >= 2
    ]
    for detector in DETECTORS:
        for filter_name in FILTERS:
            for pass_name in PASSES:
                selected = repeated.loc[
                    (repeated["detector"] == detector)
                    & (repeated["filter"] == filter_name)
                    & (repeated["pass"] == pass_name)
                ]
                count = int(len(selected))
                within = int(
                    np.count_nonzero(
                        np.abs(selected["corrected_flux_change_percent"].to_numpy(float))
                        <= 0.5
                    )
                )
                records.append(
                    {
                        "filter": filter_name,
                        "detector": detector,
                        "pass": pass_name,
                        "observation_count": count,
                        "within_0p5_percent_count": within,
                        "proportion_within_0p5_percent": (
                            within / count if count else np.nan
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _plot_heatmap(table, output_path):
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 11.0), sharex=True, sharey=True)
    detector_indices = np.arange(len(DETECTORS))
    pass_labels = [
        name.replace("ep3_pass", "EP3-").replace("ep4_pass", "EP4-")
        for name in PASSES
    ]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    image = None
    for filter_name, axis in zip(FILTERS, axes.flat, strict=True):
        rows = table.loc[table["filter"] == filter_name]
        matrix = np.full((len(DETECTORS), len(PASSES)), np.nan, dtype=float)
        counts = np.zeros_like(matrix, dtype=int)
        for _, row in rows.iterrows():
            detector_index = DETECTORS.index(row["detector"])
            pass_index = PASSES.index(row["pass"])
            matrix[detector_index, pass_index] = row["proportion_within_0p5_percent"]
            counts[detector_index, pass_index] = row["observation_count"]
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            origin="upper",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            interpolation="nearest",
        )
        for detector_index in range(len(DETECTORS)):
            for pass_index in range(len(PASSES)):
                value = matrix[detector_index, pass_index]
                if np.isfinite(value):
                    text_color = "white" if value < 0.48 else "black"
                    axis.text(
                        pass_index,
                        detector_index,
                        f"{100.0 * value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=text_color,
                    )
        axis.set_title(filter_name)
        axis.set_xticks(np.arange(len(PASSES)), labels=pass_labels, rotation=45, ha="right")
        axis.set_yticks(detector_indices, labels=DETECTORS)
        axis.set_xticks(np.arange(-0.5, len(PASSES), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(DETECTORS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.6)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_xlim(-0.5, len(PASSES) - 0.5)
        axis.set_ylim(len(DETECTORS) - 0.5, -0.5)
    cbar_axis = fig.add_axes([0.95, 0.19, 0.012, 0.65])
    cbar = fig.colorbar(image, cax=cbar_axis)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    cbar.set_label("Observations corrected to within ±0.5%")
    fig.suptitle(
        "All-star CAR201 model-correction success by epoch, detector, and filter\n"
        "Each cell is the fraction of repeated-source observations within ±0.5% after correction",
        y=0.99,
    )
    fig.supxlabel("Epoch / visit", y=0.065)
    fig.supylabel("Detector")
    fig.text(
        0.5,
        0.015,
        "All finite photometry rows with positive errors are included, regardless of fit/DQ/outlier selection. "
        "Sources with only one valid observation in a filter cannot measure a change and are excluded; "
        "each repeated source/filter is anchored to its first valid pass. Cell counts are in the CSV.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.055, right=0.94, bottom=0.15, top=0.89, wspace=0.16, hspace=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    first_summary_path = (
        PRODUCTION_ROOT
        / "wfi01/spline_relative_response_normal/experiments/"
        "car201_zane_wfi01_ep3v2_ep4_faint_v2_tile_matched/"
        "car201_actual_star_modulation_fit_summary.json"
    )
    with first_summary_path.open() as handle:
        reference_summary = json.load(handle)
    start_utc, end_utc = reference_summary["date_window_utc"]
    start_mjd = float(Time(start_utc, format="isot", scale="utc").mjd)
    end_mjd = float(Time(end_utc, format="isot", scale="utc").mjd)
    ice_reference_mjd = float(
        Time(fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    )

    detector_frames = {}
    detector_source_ids = {}
    detector_reference_vectors = {}
    all_source_ids = set()
    for detector in DETECTORS:
        frames = _load_detector_frames(detector, start_mjd, end_mjd)
        source_ids = sorted(
            set().union(*(set(frame["source_id"].astype(str)) for frame in frames))
        )
        valid_source_ids = set()
        for frame in frames:
            mag = pd.to_numeric(frame["mag_obs"], errors="coerce")
            err = pd.to_numeric(frame["mag_err"], errors="coerce")
            valid_source_ids.update(
                frame.loc[np.isfinite(mag) & np.isfinite(err) & (err > 0.0), "source_id"]
            )
        source_ids = sorted(valid_source_ids)
        detector_frames[detector] = frames
        detector_source_ids[detector] = source_ids
        all_source_ids.update(source_ids)
        print(f"Loaded {detector}: {len(source_ids)} stars with finite measurements", flush=True)

    sed_by_id = _load_successful_seds(
        ((FAINT_SED_PATH, "faint_v2"), (ORIGINAL_SED_PATH, "original_successful")),
        all_source_ids,
    )
    sed_assignment_records = []
    sed_assignments_by_detector = {}
    for detector in DETECTORS:
        assignments, records = _assign_seds_for_detector(
            detector,
            detector_frames[detector],
            detector_source_ids[detector],
            sed_by_id,
        )
        sed_assignments_by_detector[detector] = assignments
        sed_assignment_records.extend(records)
        imputed_count = sum(
            record["sed_source"] == "nearest_neighbor_magnitudes" for record in records
        )
        print(f"{detector}: assigned nearest-neighbor SEDs to {imputed_count} stars", flush=True)

    stride = float(reference_summary["modulation"]["wavelength_stride_nm"])
    wavelengths_nm, effective_area = _make_wavelength_grid(stride)
    all_observation_frames = []
    validation_records = []
    for detector in DETECTORS:
        print(f"Predicting saved-model response for all {detector} sources", flush=True)
        fit_dir = (
            PRODUCTION_ROOT
            / detector.lower()
            / "spline_relative_response_normal/experiments"
            / f"car201_zane_{detector.lower()}_ep3v2_ep4_faint_v2_tile_matched"
        )
        source_ids = detector_source_ids[detector]
        predicted = _predict_detector(
            detector,
            detector_frames[detector],
            source_ids,
            sed_assignments_by_detector[detector],
            SED_GRID_PATH,
            fit_dir,
            wavelengths_nm,
            effective_area,
            ice_reference_mjd,
            unmodulated=args.unmodulated,
        )
        observations = _make_source_rows(
            detector, detector_frames[detector], source_ids, predicted
        )
        saved_path = fit_dir / "car201_actual_star_modulation_measurements.csv"
        saved = pd.read_csv(saved_path, dtype={"source_id": str})
        source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
        filter_lookup = {name: index for index, name in enumerate(FILTERS)}
        pass_lookup = {name: index for index, name in enumerate(PASSES)}
        model_lookup = {}
        for _, saved_row in saved.iterrows():
            saved_prediction = float(saved_row["fitted_delta_mag_including_zeropoint"])
            if np.isfinite(saved_prediction):
                model_lookup[
                    (
                        str(saved_row["source_id"]),
                        str(saved_row["filter"]),
                        str(saved_row["pass"]),
                    )
                ] = saved_prediction
        observations["saved_fit_delta_mag"] = [
            model_lookup.get((row.source_id, row.filter, row.pass_name), np.nan)
            for row in observations.rename(columns={"pass": "pass_name"})[
                ["source_id", "filter", "pass_name"]
            ].itertuples(index=False, name="Observation")
        ]
        comparable = observations.loc[observations["saved_fit_delta_mag"].notna()].copy()
        keys = ["source_id", "filter"]
        current_groups = comparable.groupby(keys, sort=False)["model_delta_mag"]
        saved_groups = comparable.groupby(keys, sort=False)["saved_fit_delta_mag"]
        comparable["current_centered"] = current_groups.transform(
            lambda values: values - values.iloc[0]
        )
        comparable["saved_centered"] = saved_groups.transform(
            lambda values: values - values.iloc[0]
        )
        validation_errors = np.abs(
            comparable["current_centered"].to_numpy(float)
            - comparable["saved_centered"].to_numpy(float)
        )
        max_error = float(np.max(validation_errors)) if validation_errors.size else 0.0
        validation_records.append(
            {"detector": detector, "matched_saved_rows": len(comparable), "max_abs_delta_mag": max_error}
        )
        if max_error > 1.0e-5 and not args.unmodulated:
            print(
                comparable.assign(abs_error=validation_errors)
                .sort_values("abs_error", ascending=False)
                .loc[:, ["source_id", "filter", "pass", "current_centered", "saved_centered", "abs_error"]]
                .head(5)
                .to_string(index=False),
                flush=True,
            )
            raise RuntimeError(
                f"Expanded all-star predictor disagrees with saved {detector} predictions "
                f"by {max_error:.6g} mag"
            )
        print(f"{detector}: saved-model prediction check max Δ={max_error:.3g} mag", flush=True)
        all_observation_frames.append(observations)

    observations = _assign_relative_changes(
        pd.concat(all_observation_frames, ignore_index=True)
    )
    table = _heatmap_table(observations)
    prefix = args.output_prefix
    plot_path = prefix.with_suffix(".png")
    table_path = prefix.with_name(prefix.name + "_cells.csv")
    observations_path = prefix.with_name(prefix.name + "_all_observations.csv")
    seds_path = prefix.with_name(prefix.name + "_sed_assignments.csv")
    validation_path = prefix.with_name(prefix.name + "_prediction_validation.csv")
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")
    table.to_csv(table_path, index=False)
    observations.to_csv(observations_path, index=False)
    pd.DataFrame.from_records(sed_assignment_records).to_csv(seds_path, index=False)
    pd.DataFrame.from_records(validation_records).to_csv(validation_path, index=False)
    _plot_heatmap(table, plot_path)
    metadata_path.write_text(
        json.dumps(
            {
                "detectors": list(DETECTORS),
                "filters": list(FILTERS),
                "passes": list(PASSES),
                "fit_parameters_refit": False,
                "spectral_modulation_disabled": args.unmodulated,
                "sed_policy": "successful faint-v2 posterior; else successful original posterior; else nearest same-detector successful SED in first/highest-coverage eight-band CAR201 magnitude color vector",
                "observation_selection": "all finite valid catalog magnitudes with finite positive errors; no likelihood, DQ, saturation, or outlier mask",
                "change_reference": "first valid observed pass separately for each detector/source/filter",
                "heatmap_denominator": "observations in source/filter groups with at least two valid measurements; includes each group's reference observation",
                "success_definition": "absolute corrected relative flux change <= 0.5 percent",
                "plot": str(plot_path),
                "cell_summary_csv": str(table_path),
                "all_observations_csv": str(observations_path),
                "sed_assignments_csv": str(seds_path),
                "saved_prediction_validation_csv": str(validation_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {plot_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {observations_path}")
    print(f"Wrote {seds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
