"""Export the preserved unmodulated TMM bundle's tile ice history for v3 comparison."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from export_joint_ice_history_reference import (
    ROOT,
    deposition_factorizations_on_blocks,
    sha256_file,
    write_array,
)
from wfi_wavelength_response_bundle import load_bundle


DEFAULT_BUNDLE = (
    ROOT
    / "20261106/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/wfi_all_2nm_rank16.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/wfi_unmodulated_ice_history_reference_v1.h5"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing reference: {args.output}")

    bundle = load_bundle(args.bundle)
    metadata = bundle["metadata"]
    days = np.asarray(bundle["observation_days"], dtype=float)
    thickness = np.asarray(bundle["tile_ice_thickness"], dtype=float)
    timestamps = np.asarray(bundle["observation_start_time"]).astype(str)
    if timestamps.size != days.size:
        raise ValueError("Unmodulated bundle time labels do not match its observation axis")
    reset_days = np.asarray(
        [float(bundle["ice_reset_day"]), float(bundle["third_decon_day"])], dtype=float
    )
    episode = np.searchsorted(reset_days, days, side="right").astype(np.int8)
    deposition_fit = deposition_factorizations_on_blocks(
        days, episode, thickness, spacing_days=2.0, multi_rank=16
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5:
        h5.attrs["schema"] = "rcs_unmodulated_ice_history_reference/1"
        h5.attrs["product_type"] = "preserved unmodulated TMM tile histories for fitting a v3 deposition model"
        h5.attrs["v3_deposition_compatible_parameters_included"] = False
        h5.attrs["ice_reference_utc"] = str(timestamps[0])
        h5.attrs["reset_days_from_reference"] = reset_days
        h5.attrs["source_bundle"] = str(args.bundle.resolve())
        h5.attrs["source_bundle_sha256"] = sha256_file(args.bundle)
        h5.attrs["source_model_variant"] = "unmodulated TMM; preserved production ice history"

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
        write_array(observation, "time_days_from_reference", days)
        write_array(observation, "time_hours_from_reference", days * 24.0)
        observation.create_dataset("timestamp_utc", data=timestamps.astype(string_dtype))
        write_array(observation, "episode", episode)
        write_array(observation, "tile_thickness_nm", thickness)
        write_array(
            observation,
            "tile_rate_nm_per_day",
            np.asarray(bundle["tile_instantaneous_rate_nm_day"]),
        )

        rank1 = h5.create_group("v3_rank1_deposition_candidate")
        for key in (
            "time_hours", "episodes", "blocks",
        ):
            out_name = "episode" if key == "episodes" else key
            write_array(rank1, out_name, deposition_fit[key])
        for key in ("log_a", "d_off_nm", "multipliers", "rate_nm_per_day", "fitted_thickness_nm"):
            out_name = "d_off" if key == "d_off_nm" else key
            write_array(rank1, out_name, deposition_fit["rank1"][key])
        write_array(rank1, "tau_h", np.empty(0, dtype=float))
        write_array(rank1, "offset_episodes", np.empty(0, dtype=np.int16))
        rank1.attrs["current_v3_deposition_group_fields"] = True
        rank1.attrs["fit_metrics_json"] = json.dumps(
            deposition_fit["rank1"]["metrics"], sort_keys=True
        )
        rank1.attrs["warning"] = "Rank-one fit to the original unmodulated TMM history; inspect residuals."

        lowrank = h5.create_group("deposition_lowrank_extension")
        for key in (
            "time_hours", "episodes", "blocks",
        ):
            out_name = "episode" if key == "episodes" else key
            write_array(lowrank, out_name, deposition_fit[key])
        for key in (
            "spatial_rate_coefficients", "temporal_rate_functions_nm_per_hour",
            "d_off_nm", "fitted_thickness_nm",
        ):
            out_name = "d_off_nm" if key == "d_off_nm" else key
            write_array(lowrank, out_name, deposition_fit["rankk"][key])
        lowrank.attrs["fit_metrics_json"] = json.dumps(
            deposition_fit["rankk"]["metrics"], sort_keys=True
        )
        lowrank.attrs["warning"] = "Multi-mode low-rank deposition extension; not supported by unmodified v3 core."

        provenance = h5.create_group("provenance")
        provenance.attrs["source_bundle_metadata_json"] = json.dumps(metadata, sort_keys=True)
    print(f"Wrote unmodulated-history reference {args.output}")
    print(json.dumps(deposition_fit["rank1"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
