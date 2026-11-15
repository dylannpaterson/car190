import json

import numpy as np
import pytest

from build_wfi_wavelength_response_bundle import (
    discover_cache_roots,
    validate_tile_ids,
)
from wfi_wavelength_response_bundle import (
    BUNDLE_VERSION,
    bundle_summary,
    evaluate,
    load_bundle,
    nearest_tile,
    tile_index,
    tile_geometry,
)


def _write_detector_cache_marker(root, detector):
    cache_root = root / f"{detector.lower()}_2nm_rank16" / "tile_000"
    cache_root.mkdir(parents=True)
    np.savez_compressed(cache_root / "spectral_response.npz", detector=np.asarray(detector))
    return cache_root.parent


def test_cache_root_discovers_detectors_and_rejects_partial_all_detector_set(tmp_path):
    cache_root = tmp_path / "cache"
    wfi01 = _write_detector_cache_marker(cache_root, "WFI01")
    wfi02 = _write_detector_cache_marker(cache_root, "WFI02")

    assert discover_cache_roots(cache_root) == [
        ("WFI01", wfi01),
        ("WFI02", wfi02),
    ]
    with pytest.raises(FileNotFoundError, match="missing: WFI03"):
        discover_cache_roots(cache_root, require_all=True)
    with pytest.raises(ValueError, match="WFI01 is incomplete"):
        validate_tile_ids("WFI01", [0], 256, require_complete=True)


def test_tile_geometry_uses_mpa_positive_x_direction():
    geometry = tile_geometry(
        "WFI01",
        tile_id=44,
        tile_row=2,
        tile_col=12,
        row_start=512,
        row_stop_exclusive=768,
        col_start=3072,
        col_stop_exclusive=3328,
    )
    assert np.isclose(geometry["tile_center_x_mm"], -33.66)
    assert np.isclose(geometry["tile_center_y_mm"], -1.93)
    assert geometry["active_pixel_center_col"] == 3200.0
    assert geometry["active_pixel_center_row"] == 640.0


def test_bundle_load_evaluate_and_nearest_tile(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez_compressed(
        path,
        bundle_version=np.asarray(BUNDLE_VERSION),
        metadata_json=np.asarray(json.dumps({"test": True})),
        detectors=np.asarray(["WFI01", "WFI02"]),
        sca_numbers=np.asarray([1, 2]),
        response_rank=np.asarray(1),
        wavelengths_nm=np.asarray([480.0, 500.0, 520.0]),
        ice_thickness_nm=np.asarray([0.0, 2.0, 4.0]),
        tile_ids=np.asarray([44, 44]),
        tile_rows=np.asarray([2, 2]),
        tile_cols=np.asarray([12, 12]),
        row_starts=np.asarray([512, 512]),
        row_stops_exclusive=np.asarray([768, 768]),
        col_starts=np.asarray([3072, 3072]),
        col_stops_exclusive=np.asarray([3328, 3328]),
        tile_n_rows=np.asarray([256, 256]),
        tile_n_cols=np.asarray([256, 256]),
        pixel_center_rows=np.asarray([640.0, 640.0]),
        pixel_center_cols=np.asarray([3200.0, 3200.0]),
        active_pixel_center_rows=np.asarray([640.0, 640.0]),
        active_pixel_center_cols=np.asarray([3200.0, 3200.0]),
        sca_center_x_mm=np.asarray([-22.14, -22.29]),
        sca_center_y_mm=np.asarray([12.15, -37.03]),
        sca_rotation_deg=np.asarray([0.0, 0.0]),
        tile_local_x_mm=np.asarray([-11.52, -11.52]),
        tile_local_y_mm=np.asarray([-14.08, -14.08]),
        tile_center_x_mm=np.asarray([-33.66, -33.81]),
        tile_center_y_mm=np.asarray([-1.93, -51.11]),
        clean_power=np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        response_left=np.asarray(
            [[[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]]]
        ),
        response_right=np.asarray(
            [[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]]
        ),
        observation_days=np.asarray([0.0, 1.5, 2.5]),
        observation_start_time=np.asarray(["a", "b", "c"]),
        observation_led=np.asarray([1, 2, 3]),
        observation_repeat=np.asarray([0, 1, 2]),
        observation_program=np.asarray([1039, 1063, 1063]),
        groups=np.asarray([0, 1, 2]),
        tile_ice_thickness=np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        tile_instantaneous_rate_nm_day=np.ones((2, 3)),
        post_reset_tile_log_offset=np.asarray([0.1, -0.2]),
        post_reset_tile_throughput_factor=np.exp([0.1, -0.2]),
        ice_reset_day=np.asarray(1.0),
        third_decon_day=np.asarray(2.0),
        post_third_decon_tile_log_offset=np.asarray([0.2, 0.3]),
        post_third_decon_tile_throughput_factor=np.exp([0.2, 0.3]),
        reset_event_days=np.asarray([1.0, 2.0, 2.5]),
        reset_event_names=np.asarray(
            ["second_decon", "third_decon", "fourth_decon"]
        ),
        reset_event_tile_log_offsets=np.asarray(
            [[0.1, 0.2, 0.05], [-0.2, 0.3, 0.04]]
        ),
        reset_event_tile_throughput_factors=np.exp(
            [[0.1, 0.2, 0.05], [-0.2, 0.3, 0.04]]
        ),
    )
    bundle = load_bundle(path)
    assert bundle["reset_events_included"] is True
    assert bundle_summary(bundle)["reset_events"][-1] == {
        "name": "fourth_decon",
        "day": 2.5,
    }
    clean, response = evaluate(
        bundle, 44, [1.0], [490.0, 510.0], detector="WFI01"
    )
    np.testing.assert_allclose(clean, [1.5, 2.5])
    np.testing.assert_allclose(response, [[0.75, 1.25]])
    assert nearest_tile(bundle, -33.66, -1.93, detector="WFI01")["tile_id"] == 44
    assert tile_index(bundle, 44, detector="WFI02") == 1
    with pytest.raises(KeyError, match="ambiguous"):
        evaluate(bundle, 44, [1.0], [490.0, 510.0])

    legacy_path = tmp_path / "legacy_bundle.npz"
    with np.load(path, allow_pickle=False) as source:
        legacy = {name: np.asarray(source[name]) for name in source.files}
    for name, value in list(legacy.items()):
        if value.ndim > 0 and value.shape[0] == 2:
            legacy[name] = value[:1]
    legacy.pop("detectors")
    legacy.pop("sca_numbers")
    legacy["bundle_version"] = np.asarray(1)
    legacy["detector"] = np.asarray("WFI01")
    legacy["sca_number"] = np.asarray(1)
    np.savez_compressed(legacy_path, **legacy)
    legacy_bundle = load_bundle(legacy_path)
    assert tile_index(legacy_bundle, 44) == 0
