"""Export the fitted joint 2-day tile histories as an Eric-model fit reference."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import svd
from astropy.time import Time

import run_joint_car190_amber_inverse_wavelength_maxent as joint_fit
from wfi_wavelength_response_bundle import load_bundle


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/wfi_all_2nm_rank16_modulated.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/wfi_joint_2day_ice_history_reference_v1.h5"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_array(group, name, data, **kwargs):
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 4)
    kwargs.setdefault("shuffle", True)
    group.create_dataset(name, data=data, **kwargs)


def deposition_factorizations(times_days, episodes, thickness_nm, multi_rank=16):
    """Fit v3's rank-one spatial-rate form and a diagnostic multi-mode extension."""
    hours = np.asarray(times_days, dtype=float) * 24.0
    episodes = np.asarray(episodes, dtype=np.int8)
    thickness = np.asarray(thickness_nm, dtype=float)
    left_indices = np.flatnonzero(episodes[:-1] == episodes[1:])
    dt_hours = np.diff(hours)[left_indices]
    if np.any(dt_hours <= 0.0):
        raise ValueError("Observation time axis is not strictly increasing")
    rates = np.diff(thickness, axis=1)[:, left_indices] / dt_hours[None, :]
    if np.any(rates < -1e-8):
        raise ValueError("Joint thickness histories decrease within an ice-growth episode")
    rates = np.maximum(rates, 0.0)
    weights = np.sqrt(dt_hours)
    weighted_rates = rates * weights[None, :]
    left, singular, right = svd(
        weighted_rates, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    rank = min(int(multi_rank), singular.size)
    cumulative_variance = np.cumsum(singular.astype(float) ** 2) / np.sum(singular.astype(float) ** 2)
    blocks = np.column_stack((episodes[left_indices], hours[left_indices], hours[left_indices + 1]))

    # Nonnegative alternating least squares produces the single spatial pattern
    # and shared time function accepted by the current v3 core.
    spatial = np.maximum(left[:, 0] * singular[0], 1e-15)
    time_factor = np.maximum(right[0] / weights, 1e-15)
    for _ in range(100):
        time_factor = np.maximum(
            (spatial[:, None] * weighted_rates).sum(axis=0)
            / (spatial @ spatial)
            / weights,
            1e-15,
        )
        weighted_time = time_factor * weights
        spatial = np.maximum(
            (weighted_rates * weighted_time[None, :]).sum(axis=1)
            / (weighted_time @ weighted_time),
            1e-15,
        )
    scale = float(np.median(time_factor))
    spatial *= scale
    time_factor /= scale
    rank1_rates = spatial[:, None] * time_factor[None, :]
    rank1_rate_error = np.linalg.norm(
        (rates - rank1_rates) * weights[None, :], axis=1
    ) / np.maximum(np.linalg.norm(weighted_rates, axis=1), 1e-15)

    def integrate_rate_modes(rate_modes):
        # rate_modes is (rank, interval) in nm/hour per unit spatial coefficient.
        accumulated = np.zeros((rate_modes.shape[0], hours.size), dtype=float)
        interval_column = {int(left_index): i for i, left_index in enumerate(left_indices)}
        for episode in np.unique(episodes):
            nodes = np.flatnonzero(episodes == episode)
            if not nodes.size:
                continue
            accumulated[:, nodes[0]] = 0.0
            for start, stop in zip(nodes[:-1], nodes[1:]):
                column = interval_column[int(start)]
                accumulated[:, stop] = (
                    accumulated[:, start]
                    + rate_modes[:, column] * (hours[stop] - hours[start])
                )
        return accumulated

    rank1_growth = integrate_rate_modes(time_factor[None, :])[0]
    n_episodes = int(episodes.max()) + 1

    def fit_offsets(spatial_modes, growth_modes):
        offsets = np.empty((thickness.shape[0], n_episodes), dtype=float)
        reconstructed = np.empty_like(thickness)
        for episode in range(n_episodes):
            nodes = np.flatnonzero(episodes == episode)
            growth = spatial_modes @ growth_modes[:, nodes]
            residual = thickness[:, nodes] - growth
            offsets[:, episode] = residual.mean(axis=1)
            reconstructed[:, nodes] = growth + offsets[:, episode, None]
        return offsets, reconstructed

    rank1_offsets, rank1_thickness = fit_offsets(spatial[:, None], rank1_growth[None, :])
    rank1_abs_error = np.abs(thickness - rank1_thickness)
    rank1_per_tile_max = np.max(rank1_abs_error, axis=1)
    rank1_metrics = {
        "rate_weighted_relative_error_p50_p95_max": np.percentile(
            rank1_rate_error, [50, 95, 100]
        ).tolist(),
        "thickness_absolute_error_nm_p50_p95_max": np.percentile(
            rank1_abs_error, [50, 95, 100]
        ).tolist(),
        "per_tile_max_thickness_error_nm_p50_p95_max": np.percentile(
            rank1_per_tile_max, [50, 95, 100]
        ).tolist(),
    }

    # A signed rank-K extension is not accepted by the current v3 core, but it
    # quantifies how much temporal/spatial rank the 2-day histories actually need.
    spatial_modes = left[:, :rank] * singular[:rank]
    temporal_modes = right[:rank] / weights[None, :]
    growth_modes = integrate_rate_modes(temporal_modes)
    rankk_offsets, rankk_thickness = fit_offsets(spatial_modes, growth_modes)
    rankk_abs_error = np.abs(thickness - rankk_thickness)
    rankk_rate_error = np.linalg.norm(
        (rates - spatial_modes @ temporal_modes) * weights[None, :], axis=1
    ) / np.maximum(np.linalg.norm(weighted_rates, axis=1), 1e-15)
    rankk_metrics = {
        "rank": rank,
        "weighted_rate_variance_fraction": float(cumulative_variance[rank - 1]),
        "rate_weighted_relative_error_p50_p95_max": np.percentile(
            rankk_rate_error, [50, 95, 100]
        ).tolist(),
        "thickness_absolute_error_nm_p50_p95_max": np.percentile(
            rankk_abs_error, [50, 95, 100]
        ).tolist(),
        "per_tile_max_thickness_error_nm_p50_p95_max": np.percentile(
            np.max(rankk_abs_error, axis=1), [50, 95, 100]
        ).tolist(),
    }
    return {
        "time_hours": hours,
        "episodes": episodes,
        "blocks": blocks,
        "rank1": {
            "log_a": np.log(spatial),
            "d_off_nm": rank1_offsets,
            "multipliers": time_factor,
            "rate_nm_per_day": spatial[:, None] * time_factor[None, :] * 24.0,
            "metrics": rank1_metrics,
        },
        "rankk": {
            "spatial_rate_coefficients": spatial_modes,
            "temporal_rate_functions_nm_per_hour": temporal_modes,
            "d_off_nm": rankk_offsets,
            "metrics": rankk_metrics,
        },
    }


def deposition_factorizations_on_blocks(
    times_days, episodes, thickness_nm, spacing_days=2.0, multi_rank=16
):
    """Fit v3-style rate blocks on a compact 2-day grid to bundle histories."""
    days = np.asarray(times_days, dtype=float)
    hours = days * 24.0
    episodes = np.asarray(episodes, dtype=np.int8)
    thickness = np.asarray(thickness_nm, dtype=float)
    if spacing_days <= 0.0:
        raise ValueError("Deposition block spacing must be positive")

    block_rows = []
    rate_parts = []
    duration_parts = []
    episode_edges = {}
    episode_values = np.unique(episodes)
    for episode in episode_values:
        nodes = np.flatnonzero(episodes == episode)
        if nodes.size < 2:
            raise ValueError(f"Episode {episode} has fewer than two observation samples")
        start_day, end_day = float(days[nodes[0]]), float(days[nodes[-1]])
        edges = np.arange(start_day, end_day, spacing_days, dtype=float)
        if edges.size == 0 or not np.isclose(edges[0], start_day, rtol=0.0, atol=1e-10):
            edges = np.r_[start_day, edges]
        if end_day - edges[-1] > 1e-10:
            edges = np.r_[edges, end_day]
        else:
            edges[-1] = end_day
        edge_hours = edges * 24.0
        source_hours = hours[nodes]
        edge_thickness = np.stack([
            np.interp(edge_hours, source_hours, thickness[tile, nodes])
            for tile in range(thickness.shape[0])
        ])
        dt = np.diff(edge_hours)
        if np.any(dt <= 0.0):
            raise ValueError(f"Episode {episode} has non-increasing deposition block edges")
        rates = np.diff(edge_thickness, axis=1) / dt[None, :]
        if np.any(rates < -1e-7):
            raise ValueError(f"Joint thickness decreases within episode {episode}")
        rate_parts.append(np.maximum(rates, 0.0))
        duration_parts.append(dt)
        block_rows.extend(
            (int(episode), float(left), float(right))
            for left, right in zip(edge_hours[:-1], edge_hours[1:], strict=True)
        )
        episode_edges[int(episode)] = (edge_hours, len(dt))

    rates = np.concatenate(rate_parts, axis=1)
    durations = np.concatenate(duration_parts)
    weights = np.sqrt(durations)
    weighted_rates = rates * weights[None, :]
    left, singular, right = svd(
        weighted_rates, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    rank = min(int(multi_rank), singular.size)
    cumulative_variance = np.cumsum(singular.astype(float) ** 2) / np.sum(
        singular.astype(float) ** 2
    )

    spatial = np.maximum(left[:, 0] * singular[0], 1e-15)
    multipliers = np.maximum(right[0] / weights, 1e-15)
    for _ in range(100):
        multipliers = np.maximum(
            (spatial[:, None] * weighted_rates).sum(axis=0)
            / (spatial @ spatial)
            / weights,
            1e-15,
        )
        weighted_time = multipliers * weights
        spatial = np.maximum(
            (weighted_rates * weighted_time[None, :]).sum(axis=1)
            / (weighted_time @ weighted_time),
            1e-15,
        )
    scale = float(np.median(multipliers))
    spatial *= scale
    multipliers /= scale
    rank1_rates = spatial[:, None] * multipliers[None, :]
    rank1_rate_error = np.linalg.norm(
        (rates - rank1_rates) * weights[None, :], axis=1
    ) / np.maximum(np.linalg.norm(weighted_rates, axis=1), 1e-15)

    def cumulative_modes(rate_modes):
        cumulative = np.zeros((rate_modes.shape[0], hours.size), dtype=float)
        block_offset = 0
        for episode in episode_values:
            nodes = np.flatnonzero(episodes == episode)
            edge_hours, count = episode_edges[int(episode)]
            episode_rates = rate_modes[:, block_offset:block_offset + count]
            interval_start = edge_hours[:-1]
            interval_end = edge_hours[1:]
            prefix = np.concatenate([
                np.zeros((rate_modes.shape[0], 1)),
                np.cumsum(episode_rates * np.diff(edge_hours)[None, :], axis=1)[:, :-1],
            ], axis=1)
            node_hours = hours[nodes]
            interval = np.clip(
                np.searchsorted(interval_start, node_hours, side="right") - 1,
                0,
                count - 1,
            )
            elapsed = np.clip(
                np.minimum(node_hours, interval_end[interval]) - interval_start[interval],
                0.0,
                None,
            )
            cumulative[:, nodes] = prefix[:, interval] + episode_rates[:, interval] * elapsed[None, :]
            block_offset += count
        return cumulative

    def fit_offsets(spatial_modes, growth_modes):
        offsets = np.empty((thickness.shape[0], episode_values.size), dtype=float)
        reconstructed = np.empty_like(thickness)
        for column, episode in enumerate(episode_values):
            nodes = np.flatnonzero(episodes == episode)
            growth = spatial_modes @ growth_modes[:, nodes]
            residual = thickness[:, nodes] - growth
            offsets[:, column] = residual.mean(axis=1)
            reconstructed[:, nodes] = growth + offsets[:, column, None]
        return offsets, reconstructed

    rank1_growth = cumulative_modes(multipliers[None, :])
    rank1_offsets, rank1_thickness = fit_offsets(spatial[:, None], rank1_growth)
    rank1_abs_error = np.abs(thickness - rank1_thickness)
    rank1_metrics = {
        "rate_weighted_relative_error_p50_p95_max": np.percentile(
            rank1_rate_error, [50, 95, 100]
        ).tolist(),
        "thickness_absolute_error_nm_p50_p95_max": np.percentile(
            rank1_abs_error, [50, 95, 100]
        ).tolist(),
        "per_tile_max_thickness_error_nm_p50_p95_max": np.percentile(
            np.max(rank1_abs_error, axis=1), [50, 95, 100]
        ).tolist(),
    }

    spatial_modes = left[:, :rank] * singular[:rank]
    temporal_modes = right[:rank] / weights[None, :]
    rankk_growth = cumulative_modes(temporal_modes)
    rankk_offsets, rankk_thickness = fit_offsets(spatial_modes, rankk_growth)
    rankk_abs_error = np.abs(thickness - rankk_thickness)
    rankk_rate_error = np.linalg.norm(
        (rates - spatial_modes @ temporal_modes) * weights[None, :], axis=1
    ) / np.maximum(np.linalg.norm(weighted_rates, axis=1), 1e-15)
    rankk_metrics = {
        "rank": rank,
        "weighted_rate_variance_fraction": float(cumulative_variance[rank - 1]),
        "rate_weighted_relative_error_p50_p95_max": np.percentile(
            rankk_rate_error, [50, 95, 100]
        ).tolist(),
        "thickness_absolute_error_nm_p50_p95_max": np.percentile(
            rankk_abs_error, [50, 95, 100]
        ).tolist(),
        "per_tile_max_thickness_error_nm_p50_p95_max": np.percentile(
            np.max(rankk_abs_error, axis=1), [50, 95, 100]
        ).tolist(),
    }
    return {
        "time_hours": hours,
        "episodes": episodes,
        "blocks": np.asarray(block_rows, dtype=float),
        "rank1": {
            "log_a": np.log(spatial),
            "d_off_nm": rank1_offsets,
            "multipliers": multipliers,
            "rate_nm_per_day": spatial[:, None] * multipliers[None, :] * 24.0,
            "fitted_thickness_nm": rank1_thickness,
            "metrics": rank1_metrics,
        },
        "rankk": {
            "spatial_rate_coefficients": spatial_modes,
            "temporal_rate_functions_nm_per_hour": temporal_modes,
            "d_off_nm": rankk_offsets,
            "fitted_thickness_nm": rankk_thickness,
            "metrics": rankk_metrics,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing export")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing history reference: {args.output}")

    bundle = load_bundle(args.bundle)
    metadata = bundle["metadata"]
    history_sources = metadata["joint_ice_history_fit_summaries"]
    detectors = np.asarray(bundle["modulation_detectors"]).astype(str).tolist()
    reference_utc = joint_fit.run.workflow.fit_model.ICE_REFERENCE_UTC
    reference_mjd = float(Time(reference_utc, format="isot", scale="utc").mjd)
    histories = []
    baseline_histories = []
    control_parameters = []
    time_days_by_detector = []
    provenance = {}
    for detector in detectors:
        source = history_sources[detector]
        archive_path = Path(source["archive"])
        summary_path = Path(source["summary"])
        with np.load(archive_path, allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=float)
            baseline = np.asarray(archive["baseline_thickness_nm"], dtype=float)
            fitted = np.asarray(archive["fitted_thickness_nm"], dtype=float)
            params = np.asarray(archive["ice_parameters"], dtype=float)
        if baseline.shape != fitted.shape or baseline.shape != (256, times.size):
            raise ValueError(f"Invalid joint history array shapes for {detector}")
        if params.shape[0] != 256:
            raise ValueError(f"Invalid joint rate-control shape for {detector}: {params.shape}")
        if not np.isfinite(times).all() or not np.isfinite(fitted).all() or not np.isfinite(params).all():
            raise ValueError(f"Nonfinite joint ice history for {detector}")
        histories.append(fitted)
        baseline_histories.append(baseline)
        control_parameters.append(params)
        time_days_by_detector.append(times)
        provenance[detector] = {
            "summary": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "archive": str(archive_path.resolve()),
            "archive_sha256": sha256_file(archive_path),
            "rate_control_spacing_days": 2.0,
        }
    if len({array.size for array in time_days_by_detector}) != 1:
        raise ValueError("Joint detectors have incompatible per-tile history-node counts")

    bundle_days = np.asarray(bundle["observation_days"], dtype=float)
    bundle_times = reference_mjd + bundle_days
    bundle_utc = np.asarray(bundle["observation_start_time"]).astype(str)
    if bundle_utc.size != bundle_days.size:
        raise ValueError("The bundle observation-time labels do not match its time grid")
    reset_days = np.asarray(
        [float(bundle["ice_reset_day"]), float(bundle["third_decon_day"])],
        dtype=float,
    )
    episode = np.searchsorted(reset_days, bundle_days, side="right").astype(np.int8)
    deposition_fit = deposition_factorizations_on_blocks(
        bundle_days,
        episode,
        np.asarray(bundle["tile_ice_thickness"], dtype=float),
        spacing_days=2.0,
        multi_rank=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5:
        h5.attrs["schema"] = "rcs_joint_ice_history_reference/1"
        h5.attrs["product_type"] = "reference histories for fitting a low-rank deposition model"
        h5.attrs["v3_deposition_compatible_parameters_included"] = False
        h5.attrs["rate_control_spacing_days"] = 2.0
        h5.attrs["ice_reference_utc"] = reference_utc
        h5.attrs["reset_days_from_reference"] = reset_days
        h5.attrs["source_bundle"] = str(args.bundle.resolve())
        h5.attrs["source_bundle_sha256"] = sha256_file(args.bundle)

        superpixels = h5.create_group("superpixels")
        string_dtype = h5py.string_dtype("utf-8")
        superpixels.create_dataset(
            "detector", data=np.asarray(bundle["detectors"]).astype(str).astype(string_dtype)
        )
        for name in (
            "sca_numbers", "tile_ids", "tile_rows", "tile_cols",
            "tile_center_x_mm", "tile_center_y_mm",
        ):
            write_array(superpixels, name, np.asarray(bundle[name]))
        superpixels.attrs["grid_bins_per_side"] = 16
        superpixels.attrs["tile_size_pixels"] = 256
        superpixels.attrs["tile_order"] = "SCA-major, row-major tile ID"

        observation = h5.create_group("bundle_observation_axis")
        write_array(observation, "time_days_from_reference", bundle_days)
        write_array(observation, "time_hours_from_reference", bundle_days * 24.0)
        observation.create_dataset("timestamp_utc", data=bundle_utc.astype(string_dtype))
        write_array(observation, "episode", episode)
        write_array(observation, "tile_thickness_nm", np.asarray(bundle["tile_ice_thickness"]))
        write_array(
            observation,
            "tile_rate_nm_per_day",
            np.asarray(bundle["tile_instantaneous_rate_nm_day"]),
        )

        joint = h5.create_group("joint_2day_fit_nodes")
        joint.create_dataset("detectors", data=np.asarray(detectors, dtype=string_dtype))
        write_array(joint, "time_days_from_reference", np.asarray(time_days_by_detector))
        write_array(joint, "time_hours_from_reference", np.asarray(time_days_by_detector) * 24.0)
        write_array(joint, "baseline_thickness_nm", np.asarray(baseline_histories))
        write_array(joint, "fitted_thickness_nm", np.asarray(histories))
        write_array(joint, "ice_rate_control_parameters", np.asarray(control_parameters))
        joint.attrs["rate_control_spacing_days"] = 2.0
        joint.attrs["reset_days_from_reference"] = reset_days

        rank1 = h5.create_group("v3_rank1_deposition_candidate")
        write_array(rank1, "time_hours", deposition_fit["time_hours"])
        write_array(rank1, "episode", deposition_fit["episodes"])
        write_array(rank1, "blocks", deposition_fit["blocks"])
        write_array(rank1, "multipliers", deposition_fit["rank1"]["multipliers"])
        write_array(rank1, "tau_h", np.empty(0, dtype=float))
        write_array(rank1, "offset_episodes", np.empty(0, dtype=np.int16))
        write_array(rank1, "log_a", deposition_fit["rank1"]["log_a"])
        write_array(rank1, "d_off", deposition_fit["rank1"]["d_off_nm"])
        write_array(rank1, "rate_nm_per_day", deposition_fit["rank1"]["rate_nm_per_day"])
        rank1.attrs["current_v3_deposition_group_fields"] = True
        rank1.attrs["fit_metrics_json"] = json.dumps(
            deposition_fit["rank1"]["metrics"], sort_keys=True
        )
        rank1.attrs["warning"] = (
            "Rank-one approximation to the independent 2-day tile histories; inspect fit_metrics_json "
            "before using as a replacement deposition model."
        )
        # Store the evaluated rank-one trajectory for direct comparison with the
        # current bundle history and v3-core validation.
        write_array(rank1, "fitted_thickness_nm", deposition_fit["rank1"]["fitted_thickness_nm"])

        lowrank = h5.create_group("deposition_lowrank_extension")
        write_array(lowrank, "blocks", deposition_fit["blocks"])
        write_array(lowrank, "time_hours", deposition_fit["time_hours"])
        write_array(lowrank, "episode", deposition_fit["episodes"])
        write_array(
            lowrank,
            "spatial_rate_coefficients",
            deposition_fit["rankk"]["spatial_rate_coefficients"],
        )
        write_array(
            lowrank,
            "temporal_rate_functions_nm_per_hour",
            deposition_fit["rankk"]["temporal_rate_functions_nm_per_hour"],
        )
        write_array(lowrank, "d_off_nm", deposition_fit["rankk"]["d_off_nm"])
        lowrank.attrs["fit_metrics_json"] = json.dumps(
            deposition_fit["rankk"]["metrics"], sort_keys=True
        )
        lowrank.attrs["warning"] = (
            "Signed multi-mode spatial x temporal-rate factorization for future low-rank deposition "
            "fitting; this is not accepted by the unmodified v3 core."
        )

        group = h5.create_group("provenance")
        group.attrs["source_fit_summaries_json"] = json.dumps(provenance, sort_keys=True)
        group.attrs["note"] = (
            "These are fitted 2-day per-tile reference trajectories. They are not serialized as Eric v3 "
            "deposition parameters without approximation. The HDF5 contains a tested rank-one v3-style "
            "candidate and a higher-rank diagnostic extension; check their residuals before release use."
        )
    print(f"Wrote joint-ice history reference {args.output}")
    print("observation samples", bundle_days.size, "joint control nodes", time_days_by_detector[0].size)


if __name__ == "__main__":
    main()
