import numpy as np

from wfi_pipeline.update_ice_history import (
    build_tile_led_lookup,
    lookup_tile_history,
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
