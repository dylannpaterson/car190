"""Measure spatial compression of tile complex-reflectance PCA amplitudes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wfi_wavelength_response_bundle import tile_geometry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODE_PRODUCT = (
    ROOT
    / "20261111/production/pipeline_runs/relative_response_refactor_validation/production/"
    "wavelength_response_bundle/intermediates/h5/tmm_joint_modulated_reflectance_modes.h5"
)
DEFAULT_FINAL_H5 = (
    ROOT
    / "20261113/production/pipeline_runs/relative_response_24h_update/production/"
    "wavelength_response_bundle/final/joint_modulated_v2.h5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "20261113/production/pipeline_runs/relative_response_24h_update/production/"
    "plots/relative_response/reflectance_spatial_compression"
)
SPATIAL_K = (2, 4, 8, 12, 16, 24, 32, 48, 64)
TILE_BINS = 16


def dct_spatial_basis():
    """Return row-major orthonormal 2-D DCT modes, lowest spatial frequency first."""
    one_d = np.empty((TILE_BINS, TILE_BINS), dtype=float)
    coordinates = np.arange(TILE_BINS, dtype=float)
    for frequency in range(TILE_BINS):
        normalization = np.sqrt(1.0 / TILE_BINS) if frequency == 0 else np.sqrt(2.0 / TILE_BINS)
        one_d[frequency] = normalization * np.cos(
            np.pi * (coordinates + 0.5) * frequency / TILE_BINS
        )
    pairs = sorted(
        ((row_frequency, col_frequency)
         for row_frequency in range(TILE_BINS)
         for col_frequency in range(TILE_BINS)),
        key=lambda pair: (
            pair[0] ** 2 + pair[1] ** 2,
            pair[0] + pair[1],
            pair[0],
            pair[1],
        ),
    )
    basis = np.column_stack(
        [
            np.outer(one_d[row_frequency], one_d[col_frequency]).reshape(-1)
            for row_frequency, col_frequency in pairs
        ]
    )
    return basis


def _complex_curve_errors(reference, reconstructed):
    errors = np.linalg.norm(reference - reconstructed, axis=1)
    relative = errors / np.maximum(np.linalg.norm(reference, axis=1), 1.0e-15)
    return {
        "relative_curve_error_p50_p95_p99_max": np.percentile(
            relative, [50, 95, 99, 100]
        ).tolist(),
        "absolute_complex_curve_error_p50_p95_p99_max": np.percentile(
            errors, [50, 95, 99, 100]
        ).tolist(),
    }, relative


def _compression_sizes(detector_count, tile_count_per_detector, mode_count, k):
    complex_bytes = np.dtype(np.complex64).itemsize
    dense_tile = detector_count * tile_count_per_detector * mode_count * complex_bytes
    dct_coefficients = detector_count * k * mode_count * complex_bytes
    spatial_svd = detector_count * k * (tile_count_per_detector + mode_count) * complex_bytes
    return {
        "tile_amplitude_table_bytes": int(dense_tile),
        "dct_coefficient_bytes": int(dct_coefficients),
        "spatial_svd_factor_bytes": int(spatial_svd),
    }


def _plot_error_size(path, rows, expanded_h5_bytes=None):
    figure, axis = plt.subplots(figsize=(9, 6))
    for method, color, marker in (
        ("2D DCT", "tab:blue", "o"),
        ("spatial SVD", "tab:orange", "s"),
    ):
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["spatial_modes"],
        )
        x = np.asarray([row["encoded_bytes"] for row in selected], dtype=float) / 2**20
        p50 = [row["relative_curve_error_p50"] for row in selected]
        p95 = [row["relative_curve_error_p95"] for row in selected]
        axis.plot(x, p50, color=color, marker=marker, label=f"{method} p50")
        axis.plot(x, p95, color=color, marker=marker, linestyle="--", label=f"{method} p95")
    tile_bytes = rows[0]["tile_amplitude_table_bytes"] / 2**20
    axis.axvline(tile_bytes, color="0.3", linestyle=":", label="exact tile amplitudes")
    if expanded_h5_bytes is not None:
        axis.axvline(expanded_h5_bytes / 2**20, color="0.5", linestyle="-.",
                     label="expanded H5 amplitude storage")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Amplitude/factor payload (MiB; basis assumed implicit)")
    axis.set_ylabel("Relative complex-reflectance curve error")
    axis.set_title("Spatial compression of complex reflectance amplitudes")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_detector_errors(path, detector_rows, selected_k=32):
    selected = [
        row for row in detector_rows
        if row["method"] == "2D DCT" and row["spatial_modes"] == selected_k
    ]
    if not selected:
        return
    selected.sort(key=lambda row: row["detector"])
    labels = [row["detector"] for row in selected]
    p50 = [row["relative_curve_error_p50"] for row in selected]
    p95 = [row["relative_curve_error_p95"] for row in selected]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    axis.bar(x, p50, color="tab:blue", alpha=0.75, label="p50")
    axis.scatter(x, p95, color="tab:red", marker="D", label="p95")
    axis.set_xticks(x, labels, rotation=45)
    axis.set_ylabel("Relative complex-reflectance curve error")
    axis.set_title(f"Per-detector 2D-DCT error with {selected_k} spatial modes/SCA")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_example_map(path, original, dct_reconstruction, detector, mode_index, k):
    original_map = original[:, mode_index].reshape(TILE_BINS, TILE_BINS)
    fitted_map = dct_reconstruction[:, mode_index].reshape(TILE_BINS, TILE_BINS)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    max_abs = max(float(np.max(np.abs(original_map.real))),
                  float(np.max(np.abs(original_map.imag))))
    for axis, values, title in (
        (axes[0], original_map.real, "real coefficient"),
        (axes[1], fitted_map.real, f"{k}-mode DCT"),
        (axes[2], np.abs(original_map - fitted_map), "absolute complex error"),
    ):
        image = axis.imshow(values, origin="lower", cmap="coolwarm" if axis is not axes[2] else "magma",
                            vmin=-max_abs if axis is not axes[2] else None,
                            vmax=max_abs if axis is not axes[2] else None)
        axis.set_title(title)
        axis.set_xlabel("Tile column")
        axis.set_ylabel("Tile row")
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(f"{detector} complex-reflectance amplitude PC {mode_index + 1}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _focal_plane_coordinates(detectors):
    x_mm = []
    y_mm = []
    for detector in sorted(np.unique(detectors)):
        for tile_id in range(TILE_BINS**2):
            tile_row, tile_col = divmod(tile_id, TILE_BINS)
            geometry = tile_geometry(
                detector,
                tile_id,
                tile_row,
                tile_col,
                tile_row * 256,
                (tile_row + 1) * 256,
                tile_col * 256,
                (tile_col + 1) * 256,
            )
            x_mm.append(geometry["tile_center_x_mm"])
            y_mm.append(geometry["tile_center_y_mm"])
    return np.asarray(x_mm, dtype=float), np.asarray(y_mm, dtype=float)


def _scatter_focal_map(axis, x_mm, y_mm, values, title, cmap="viridis", vmin=None, vmax=None):
    artist = axis.scatter(
        x_mm, y_mm, c=values, s=11, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("MPA x (mm)")
    axis.set_ylabel("MPA y (mm)")
    axis.set_title(title)
    axis.grid(alpha=0.15)
    return artist


def _plot_focal_plane_amplitude_norm(
    path, x_mm, y_mm, original, spatial_svd, spatial_modes
):
    original_norm = np.linalg.norm(original, axis=1)
    fitted_norm = np.linalg.norm(spatial_svd, axis=1)
    difference = np.linalg.norm(original - spatial_svd, axis=1)
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    panels = (
        (original_norm, "Current amplitude-vector norm", "viridis", None, None),
        (fitted_norm, f"Per-SCA spatial SVD (rank {spatial_modes})", "viridis", None, None),
        (difference, "Amplitude-vector reconstruction error", "magma", 0.0, None),
    )
    for axis, (values, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        artist = _scatter_focal_map(axis, x_mm, y_mm, values, title, cmap, vmin, vmax)
        figure.colorbar(artist, ax=axis, shrink=0.78, label="Complex amplitude norm")
    figure.suptitle("Focal-plane complex-reflectance amplitudes (all 64 spectral modes)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_focal_plane_pc(
    path, x_mm, y_mm, original, spatial_svd, mode_index=0, spatial_modes=16
):
    original_mode = original[:, mode_index]
    fitted_mode = spatial_svd[:, mode_index]
    panels = (
        (original_mode.real, "Current: real part", "coolwarm"),
        (fitted_mode.real, "Spatial SVD: real part", "coolwarm"),
        ((original_mode - fitted_mode).real, "Real-part error", "coolwarm"),
        (original_mode.imag, "Current: imaginary part", "coolwarm"),
        (fitted_mode.imag, "Spatial SVD: imaginary part", "coolwarm"),
        ((original_mode - fitted_mode).imag, "Imaginary-part error", "coolwarm"),
    )
    shared_limit = max(
        float(np.max(np.abs(original_mode.real))),
        float(np.max(np.abs(fitted_mode.real))),
        float(np.max(np.abs(original_mode.imag))),
        float(np.max(np.abs(fitted_mode.imag))),
    )
    error_limit = max(
        float(np.max(np.abs((original_mode - fitted_mode).real))),
        float(np.max(np.abs((original_mode - fitted_mode).imag))),
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 11))
    for axis, (values, title, cmap) in zip(axes.reshape(-1), panels, strict=True):
        if title.endswith("error"):
            vmin, vmax = -error_limit, error_limit
        else:
            vmin, vmax = -shared_limit, shared_limit
        artist = _scatter_focal_map(
            axis, x_mm, y_mm, values, title, cmap=cmap, vmin=vmin, vmax=vmax
        )
        figure.colorbar(artist, ax=axis, shrink=0.78)
    figure.suptitle(
        f"Focal-plane complex-reflectance amplitude PC {mode_index + 1} "
        f"(spatial SVD rank {spatial_modes})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(mode_product, output_dir, final_h5=None, spatial_modes=SPATIAL_K):
    mode_product = Path(mode_product).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    with h5py.File(mode_product, "r") as h5:
        reflectance = h5["reflectance"]
        mean = np.asarray(reflectance["mean"], dtype=np.complex64)
        modes = np.asarray(reflectance["modes"], dtype=np.complex64)
        amplitudes = np.asarray(reflectance["amplitudes"], dtype=np.complex64)
        sca_numbers = np.asarray(h5["superpixels/sca"], dtype=int)
        detectors = np.asarray([f"WFI{int(sca):02d}" for sca in sca_numbers])
        tile_ids = np.asarray(h5["superpixels/tile_id"], dtype=int)
        rows = np.asarray(h5["superpixels/row"], dtype=int)
        cols = np.asarray(h5["superpixels/col"], dtype=int)
        mode_metadata = json.loads(
            h5["provenance"].attrs.get("metadata_json", "{}")
        )
    if amplitudes.shape != (detectors.size, modes.shape[0]):
        raise ValueError("Amplitude maps do not align with the tile and spectral-mode axes")
    if modes.shape[1] != mean.size:
        raise ValueError("Reflectance mean and modes use different wavelength axes")
    if set(np.unique(detectors).tolist()) != {f"WFI{n:02d}" for n in range(1, 19)}:
        raise ValueError("Spatial analysis expects all 18 WFI detectors")
    basis = dct_spatial_basis()
    selected_k = sorted(set(int(k) for k in spatial_modes if 1 <= int(k) <= TILE_BINS**2))
    if not selected_k:
        raise ValueError("At least one valid spatial mode count is required")
    preview_k = min(selected_k, key=lambda value: abs(value - 16))
    focal_x_mm, focal_y_mm = _focal_plane_coordinates(detectors)

    per_detector_rows = []
    tile_error_distributions = {
        method: {k: [] for k in selected_k}
        for method in ("2D DCT", "spatial SVD")
    }
    example = None
    focal_original = []
    focal_svd = []
    for detector in sorted(np.unique(detectors)):
        indices = np.flatnonzero(detectors == detector)
        order = np.argsort(tile_ids[indices])
        indices = indices[order]
        detector_tiles = tile_ids[indices]
        np.testing.assert_array_equal(detector_tiles, np.arange(TILE_BINS**2))
        if not np.array_equal(rows[indices], detector_tiles // TILE_BINS) or not np.array_equal(
            cols[indices], detector_tiles % TILE_BINS
        ):
            raise ValueError(f"{detector}: tile IDs are not row-major on the 16x16 grid")
        tile_amplitudes = amplitudes[indices]
        focal_original.append(tile_amplitudes)
        tile_reflection = mean[None, :] + tile_amplitudes @ modes
        spatial_u, singular, spatial_vh = np.linalg.svd(tile_amplitudes, full_matrices=False)
        dct_coefficients = basis.T @ tile_amplitudes
        for k in selected_k:
            dct_reconstruction = basis[:, :k] @ dct_coefficients[:k]
            dct_reflection = mean[None, :] + dct_reconstruction @ modes
            dct_metrics, dct_relative = _complex_curve_errors(tile_reflection, dct_reflection)
            tile_error_distributions["2D DCT"][k].append(dct_relative)
            per_detector_rows.append({
                "detector": detector,
                "method": "2D DCT",
                "spatial_modes": k,
                "relative_curve_error_p50": float(dct_metrics["relative_curve_error_p50_p95_p99_max"][0]),
                "relative_curve_error_p95": float(dct_metrics["relative_curve_error_p50_p95_p99_max"][1]),
                "relative_curve_error_max": float(dct_metrics["relative_curve_error_p50_p95_p99_max"][3]),
            })

            svd_reconstruction = (
                spatial_u[:, :k] * singular[None, :k]
            ) @ spatial_vh[:k]
            if k == preview_k:
                focal_svd.append(svd_reconstruction)
            svd_reflection = mean[None, :] + svd_reconstruction @ modes
            svd_metrics, svd_relative = _complex_curve_errors(tile_reflection, svd_reflection)
            tile_error_distributions["spatial SVD"][k].append(svd_relative)
            per_detector_rows.append({
                "detector": detector,
                "method": "spatial SVD",
                "spatial_modes": k,
                "relative_curve_error_p50": float(svd_metrics["relative_curve_error_p50_p95_p99_max"][0]),
                "relative_curve_error_p95": float(svd_metrics["relative_curve_error_p50_p95_p99_max"][1]),
                "relative_curve_error_max": float(svd_metrics["relative_curve_error_p50_p95_p99_max"][3]),
            })
            if detector == "WFI01" and k == min(selected_k, key=lambda value: abs(value - 32)):
                example = (detector, k, tile_amplitudes, dct_reconstruction)

    aggregate_rows = []
    detector_count = len(np.unique(detectors))
    for method in ("2D DCT", "spatial SVD"):
        for k in selected_k:
            relative = np.concatenate(tile_error_distributions[method][k])
            encoded_bytes = (
                detector_count * k * modes.shape[0] * amplitudes.dtype.itemsize
                if method == "2D DCT"
                else detector_count * k * (TILE_BINS**2 + modes.shape[0]) * amplitudes.dtype.itemsize
            )
            aggregate_rows.append({
                "method": method,
                "spatial_modes": k,
                "encoded_bytes": int(encoded_bytes),
                "tile_amplitude_table_bytes": int(amplitudes.nbytes),
                "relative_curve_error_p50": float(np.percentile(relative, 50)),
                "relative_curve_error_p95": float(np.percentile(relative, 95)),
                "relative_curve_error_max": float(np.max(relative)),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "spatial_mode_error_size.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    detector_path = output_dir / "spatial_mode_error_by_detector.csv"
    with detector_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_detector_rows[0]))
        writer.writeheader()
        writer.writerows(per_detector_rows)

    expanded_h5_bytes = None
    if final_h5 is not None and Path(final_h5).is_file():
        with h5py.File(final_h5, "r") as h5:
            expanded_h5_bytes = int(
                h5["reflectance/amplitudes"].id.get_storage_size()
            )
    focal_original = np.concatenate(focal_original, axis=0)
    focal_svd = np.concatenate(focal_svd, axis=0)
    _plot_error_size(
        output_dir / "spatial_basis_error_vs_size.png",
        aggregate_rows,
        expanded_h5_bytes,
    )
    _plot_detector_errors(output_dir / "dct_error_by_detector.png", per_detector_rows)
    _plot_focal_plane_amplitude_norm(
        output_dir / f"focal_plane_amplitude_norm_rank{preview_k}.png",
        focal_x_mm,
        focal_y_mm,
        focal_original,
        focal_svd,
        preview_k,
    )
    _plot_focal_plane_pc(
        output_dir / f"focal_plane_complex_pc1_rank{preview_k}.png",
        focal_x_mm,
        focal_y_mm,
        focal_original,
        focal_svd,
        mode_index=0,
        spatial_modes=preview_k,
    )
    if example is not None:
        _plot_example_map(
            output_dir / "example_wfi01_pc1_dct_map.png",
            example[2], example[3], example[0], 0, example[1],
        )

    summary = {
        "mode_product": str(mode_product),
        "mode_product_file_bytes": mode_product.stat().st_size,
        "detector_count": int(np.unique(detectors).size),
        "tiles_per_detector": TILE_BINS**2,
        "spectral_mode_count": int(modes.shape[0]),
        "wavelength_count": int(mean.size),
        "tile_amplitude_table_bytes": int(amplitudes.nbytes),
        "expanded_fine_amplitude_raw_bytes": int(
            amplitudes.nbytes * 256
        ),
        "expanded_h5_amplitudes_stored_bytes": expanded_h5_bytes,
        "dct_basis": "per-SCA orthonormal 2D DCT modes sorted by spatial frequency",
        "spatial_svd": "per-SCA optimal complex truncated SVD of tile-by-spectral-mode amplitudes",
        "mode_product_metrics": mode_metadata.get("mode_metrics", {}),
        "spatial_mode_counts_tested": selected_k,
        "tables": [str(csv_path), str(detector_path)],
        "plots": [
            str(output_dir / "spatial_basis_error_vs_size.png"),
            str(output_dir / "dct_error_by_detector.png"),
            str(output_dir / f"focal_plane_amplitude_norm_rank{preview_k}.png"),
            str(output_dir / f"focal_plane_complex_pc1_rank{preview_k}.png"),
            str(output_dir / "example_wfi01_pc1_dct_map.png"),
        ],
    }
    (output_dir / "spatial_mode_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-product", type=Path, default=DEFAULT_MODE_PRODUCT)
    parser.add_argument("--final-h5", type=Path, default=DEFAULT_FINAL_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = analyze(args.mode_product, args.output_dir, final_h5=args.final_h5)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
