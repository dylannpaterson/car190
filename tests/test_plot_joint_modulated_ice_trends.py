import json

import numpy as np
import pytest

from wfi_pipeline.plot_joint_modulated_ice_trends import (
    extract_detector_history,
    load_joint_modulated_plot_bundle,
    validate_joint_modulated_bundle,
)


def test_joint_bundle_validation_requires_joint_ice_provenance():
    bundle = {
        "metadata": {
            "trajectory_definition": "MAP thickness from relative-response fits",
            "joint_ice_history_fit_roots": {},
        },
        "clean_reflectance_modulation_included": True,
    }

    with pytest.raises(ValueError, match=r"joint CAR190 \+ Amber"):
        validate_joint_modulated_bundle(bundle)


def test_detector_history_is_sorted_by_bundle_tile_id():
    bundle = {
        "detectors": np.asarray(["WFI01", "WFI18", "WFI01"]),
        "tile_ids": np.asarray([2, 0, 0]),
        "tile_rows": np.asarray([0, 0, 0]),
        "tile_cols": np.asarray([2, 0, 0]),
        "observation_days": np.asarray([0.0, 1.0]),
        "observation_led": np.asarray([1, 1]),
        "observation_repeat": np.asarray([0, 1]),
        "tile_ice_thickness": np.asarray(
            [
                [20.0, 22.0],
                [100.0, 101.0],
                [10.0, 12.0],
            ]
        ),
        "tile_instantaneous_rate_nm_day": np.asarray(
            [
                [2.0, 2.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        ),
    }

    data = extract_detector_history(bundle, "wfi01")

    np.testing.assert_array_equal(data["tile_ids"], [0, 2])
    np.testing.assert_array_equal(data["thickness"], [[10.0, 12.0], [20.0, 22.0]])
    np.testing.assert_array_equal(data["rates"], [[2.0, 2.0], [2.0, 2.0]])


def test_plot_bundle_loader_reads_trajectory_fields_only(tmp_path):
    path = tmp_path / "modulated.npz"
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "trajectory_definition": (
                        "joint CAR190 + Amber MAP thickness from independent positive "
                        "2-day per-tile rate controls"
                    ),
                    "joint_ice_history_fit_roots": {"WFI01": "joint-fit"},
                    "modulation_fit_family": "joint-car190-amber-2day",
                }
            )
        ),
        "bundle_version": np.asarray(3, dtype=np.int16),
        "detectors": np.asarray(["WFI01"]),
        "tile_ids": np.asarray([0]),
        "tile_rows": np.asarray([0]),
        "tile_cols": np.asarray([0]),
        "observation_days": np.asarray([0.0, 1.0]),
        "observation_led": np.asarray([1, 1]),
        "observation_repeat": np.asarray([0, 1]),
        "tile_ice_thickness": np.asarray([[10.0, 20.0]]),
        "tile_instantaneous_rate_nm_day": np.asarray([[10.0, 10.0]]),
        "ice_reset_day": np.asarray(0.5),
        "clean_power": np.ones((1, 1000, 500), dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)

    bundle = load_joint_modulated_plot_bundle(path)

    assert bundle["metadata"]["modulation_fit_family"] == "joint-car190-amber-2day"
    assert bundle["clean_reflectance_modulation_included"]
    assert "clean_power" not in bundle
