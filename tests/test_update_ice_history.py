import csv

import h5py
import numpy as np

from wfi_pipeline.update_ice_history import (
    build_tile_led_lookup,
    find_bundle_root,
    h5_fit_batch_size,
    lookup_tile_history,
    rate_spline_knot_days,
    update_reader_h5,
    write_tile_rate_summary,
)


class FakeEricModel:
    n_bin = 256
    superpixel_px = 16
    wavelength_nm = np.asarray([500.0, 600.0])
    _mean = np.ones(2, dtype=complex)
    _modes = np.ones((1, 2), dtype=complex)
    _n = np.ones(2)
    _kappa = np.zeros(2)
    _b = np.zeros(2, dtype=complex)
    _n_ref = 1.25
    _snell = False
    led_nodes_nm = np.tile(np.asarray([500.0, 600.0]), (6, 1))
    led_weight = np.ones((6, 2))

    def __init__(self):
        self._sca = np.ones(256, dtype=int)
        self._row = np.repeat(np.arange(16), 16)
        self._col = np.tile(np.arange(16), 16)
        self._cos_theta = np.ones(256)
        self._amp = np.zeros((256, 1), dtype=complex)


class FakeEricCore:
    @staticmethod
    def reflectance(amplitudes, wavelength_grid, mean, modes, wavelengths):
        return np.ones((len(amplitudes), len(wavelengths)), dtype=complex)

    @staticmethod
    def optics_at(wavelength_grid, n, kappa, b, wavelengths):
        return np.ones_like(wavelengths), np.zeros_like(wavelengths), np.zeros_like(
            wavelengths, dtype=complex
        )

    @staticmethod
    def no_ice_transmission(reflectance, fresnel):
        return np.ones_like(reflectance, dtype=float)

    @staticmethod
    def relative_response(
        reflectance, wavelengths, thickness, cos_theta, wavelength_grid,
        n, kappa, b, n_ref, snell,
    ):
        return np.broadcast_to(
            np.exp(-0.01 * thickness[:, None, :]),
            (len(reflectance), len(wavelengths), thickness.shape[1]),
        ).copy()


def test_h5_led_lookup_uses_cached_core_formula_without_tmm():
    grid = np.asarray([0.0, 2.0, 4.0])
    lookup = build_tile_led_lookup(
        FakeEricModel(), FakeEricCore, "WFI01", 0, grid
    )
    assert lookup.shape == (6, 3)
    np.testing.assert_allclose(lookup[:, 0], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(lookup[:, 1], -0.02, atol=1.0e-12)
    np.testing.assert_allclose(lookup[:, 2], -0.04, atol=1.0e-12)


def test_h5_lookup_recovers_full_thickness_history_from_six_led_statistics():
    grid = np.arange(0.0, 102.0, 2.0)
    slopes = np.asarray([0.0005, 0.0008, 0.0011, 0.0015, 0.0020, 0.0027])
    curves = slopes[:, None] * grid[None, :]
    repeat_values = np.asarray([0, 1, 2])
    node_days = np.asarray([0.01, 1.01, 2.01])
    led = np.tile(np.arange(1, 7), 3)
    repeat = np.repeat(repeat_values, 6)
    days = np.repeat(node_days, 6) + np.tile(np.arange(6), 3) * 0.0001
    groups = np.arange(led.size)
    counts = np.full((1, led.size), 100.0)
    first_by_led = np.arange(6)
    expected_node_ice = np.asarray([10.0, 22.0, 34.0])
    baseline = curves[np.arange(6), np.searchsorted(grid, 10.0)]
    sums = np.zeros_like(counts)
    for index, (led_value, repeat_value) in enumerate(zip(led, repeat, strict=True)):
        led_index = int(led_value) - 1
        thickness = expected_node_ice[int(repeat_value)]
        predicted = np.interp(thickness, grid, curves[led_index]) - baseline[led_index]
        sums[0, index] = counts[0, index] * predicted

    data = {
        "days": days,
        "node_days": node_days,
        "repeat_values": repeat_values,
        "repeat": repeat,
        "repeat_index": np.repeat(np.arange(3), 6),
        "led": led,
        "groups": groups,
        "first_by_led": first_by_led,
        "counts": counts,
        "sums": sums,
        "sum_squared": sums**2 / counts,
        "reset_days": np.asarray([50.0, 80.0]),
        "reset_day": 50.0,
        "third_decon_day": 80.0,
    }
    base_archive = {
        "led_log_offsets": np.zeros((1, 6)),
        "post_reset_tile_log_offset": np.asarray([0.0]),
        "post_third_decon_tile_log_offset": np.asarray([0.0]),
        "observation_days": days,
        "tile_ice_thickness": np.asarray(
            [[[10.0] * 6 + [22.0] * 6 + [34.0] * 6]]
        ),
        "initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "reset_initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
    }

    history, rates, node_history, records, _fit_values = lookup_tile_history(
        data, curves, grid, base_archive
    )

    np.testing.assert_allclose(node_history, expected_node_ice, atol=0.5, rtol=0.0)
    np.testing.assert_allclose(
        history.reshape(3, 6),
        np.broadcast_to(expected_node_ice[:, None], (3, 6)),
        atol=0.5,
        rtol=0.0,
    )
    assert records[-1]["repeat"] == 2
    assert np.isfinite(rates).all()


def test_48h_rate_knots_keep_resets_and_batch_size_adapts_to_cadence():
    resets = np.asarray([12.907397928240743, 38.79784510416666])
    thinned = {
        "days": np.linspace(0.0, 60.0, 815),
        "node_days": np.linspace(0.1, 59.9, 136),
        "reset_days": resets,
        "counts": np.ones((1, 815)),
    }
    unthinned = {
        **thinned,
        "node_days": np.linspace(0.01, 59.99, 2600),
        "counts": np.ones((1, 16000)),
    }

    knots = rate_spline_knot_days(thinned, spacing_days=2.0)
    np.testing.assert_array_equal(knots[np.isin(knots, resets)], resets)
    assert np.max(np.diff(knots)) <= 2.0 + 1.0e-12
    assert h5_fit_batch_size(thinned, knots) == 256
    assert h5_fit_batch_size(unthinned, knots) < 256


def test_extended_rate_knot_grid_keeps_regular_spacing_and_passes_data_end():
    data = {
        "node_days": np.asarray([0.5, 4.7]),
        "reset_days": np.asarray([2.0]),
    }

    knots = rate_spline_knot_days(
        data,
        spacing_days=1.0,
        extend_past_end=True,
    )

    np.testing.assert_array_equal(
        knots,
        np.asarray([0.5, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5]),
    )
    assert 4.7 not in knots
    assert knots[-1] > data["node_days"][-1]
    assert np.max(np.diff(knots)) <= 1.0 + 1.0e-12


def test_extended_rate_knot_grid_adds_one_extra_step_when_endpoint_aligns():
    data = {
        "node_days": np.asarray([0.5, 4.5]),
        "reset_days": np.asarray([2.0]),
    }

    knots = rate_spline_knot_days(
        data,
        spacing_days=1.0,
        extend_past_end=True,
    )

    assert 4.5 in knots
    assert knots[-1] == 5.5


def test_reader_h5_writer_preserves_arbitrary_reset_event_count(tmp_path):
    base_h5 = tmp_path / "base.h5"
    output_h5 = tmp_path / "updated.h5"
    event_names = np.asarray(
        ["second_decon", "third_decon", "fourth_decon", "fifth_decon"]
    )
    event_days = np.asarray([1.0, 2.0, 3.0, 4.0])
    event_offsets = np.asarray([[0.1, 0.05, -0.02, 0.03]])
    with h5py.File(base_h5, "w") as archive:
        archive.create_group("deposition")
        instrument = archive.create_group("instrument")
        instrument.create_dataset("gray", data=np.zeros((256, 2), dtype=float))
        instrument.create_dataset("gray_period_hours", data=np.asarray([24.0, 48.0]))

    episode_rate = np.ones((256, 3), dtype=float)
    factorization = {
        "time_hours": np.asarray([0.0, 24.0, 48.0, 72.0]),
        "episodes": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "blocks": np.asarray([[0, 3]], dtype=np.int32),
        "rank1": {
            "multipliers": np.ones((1,), dtype=float),
            "log_a": np.zeros((256, 1), dtype=float),
            "d_off_nm": np.zeros((256, 1), dtype=float),
            "rate_nm_per_day": episode_rate,
            "metrics": {},
        },
    }
    bundle = {
        "detectors": np.asarray(["WFI01"]),
        "tile_ids": np.asarray([0]),
        "reset_event_names": event_names,
        "reset_event_tile_log_offsets": event_offsets,
    }

    update_reader_h5(
        base_h5,
        output_h5,
        FakeEricModel(),
        bundle,
        factorization,
        np.asarray(["2026-09-12T01:25:50.714"]),
        event_days * 24.0,
        0.0,
    )

    cumulative = np.cumsum(event_offsets[0])
    previous = np.r_[0.0, cumulative[:-1]]
    expected_gray = np.exp(cumulative) - np.exp(previous)
    with h5py.File(output_h5, "r") as archive:
        assert archive["instrument/gray_period_hours"].shape == (4,)
        assert archive["instrument/gray"].shape == (256, 4)
        np.testing.assert_allclose(archive["instrument/gray_period_hours"], event_days * 24.0)
        np.testing.assert_allclose(archive["instrument/gray"][0], expected_gray)


def test_tile_rate_summary_writes_one_rate_column_per_reset_episode(tmp_path):
    data = {
        "days": np.arange(9, dtype=float),
        "reset_days": np.asarray([2.0, 4.0, 6.0]),
        "reset_event_names": (
            "second_decon", "third_decon", "fourth_decon"
        ),
        "metadata": {0: {"tile_row": 0, "tile_col": 0}},
    }
    path = tmp_path / "tile_rate_posterior.csv"

    write_tile_rate_summary(path, "WFI01", 0, data, np.arange(1.0, 10.0))

    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["mean_rate_before_second_decon_nm_day"]) == 1.5
    assert float(row["mean_rate_after_second_decon_nm_day"]) == 3.5
    assert float(row["mean_rate_after_third_decon_nm_day"]) == 5.5
    assert float(row["mean_rate_after_fourth_decon_nm_day"]) == 8.0


def test_bundle_root_uses_h5_provenance_when_h5_was_copied(tmp_path):
    original_bundle_root = tmp_path / "run" / "production" / "wavelength_response_bundle"
    original_bundle_root.mkdir(parents=True)
    (original_bundle_root / "wfi_all_2nm_rank16.npz").touch()
    modulated = original_bundle_root / "wfi_all_2nm_rank16_modulated.npz"
    modulated.touch()
    copied_h5 = tmp_path / "modeling" / "joint_modulated_v2.h5"
    copied_h5.parent.mkdir()
    with h5py.File(copied_h5, "w") as h5:
        instrument = h5.create_group("instrument")
        instrument.attrs["fitted_gray_source_bundle"] = str(modulated)

    assert find_bundle_root(copied_h5) == original_bundle_root
