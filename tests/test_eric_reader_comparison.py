import numpy as np
import pandas as pd

from wfi_pipeline import compare_eric_v3_amber_car201 as comparison


class RecordingReaderModel:
    t0_utc = "2000-01-01T00:00:00"
    wavelength_nm = np.asarray([500.0, 600.0])
    _n = _kappa = _b = None

    def __init__(self):
        self.response_calls = []
        self.band_response_calls = []

    def valid(self, hours):
        return np.asarray(hours) >= 0.0

    def response(self, sca, x, y, hours, wavelengths, *, gray):
        self.response_calls.append(
            (sca, np.asarray(x).copy(), np.asarray(y).copy(), np.asarray(hours).copy(), gray)
        )
        values = 1.0 + np.asarray(hours, dtype=float)[None, :, None] * 0.01
        return np.broadcast_to(
            values, (len(x), len(hours), len(wavelengths))
        ).copy()

    def reflectance(self, sca, x, y, wavelengths):
        return np.full((len(x), len(wavelengths)), 0.9)

    def band_response(
        self, sca, x, y, hours, wavelengths, weight, *, gray, no_ice_weight
    ):
        self.band_response_calls.append(
            {
                "sca": sca,
                "x": np.asarray(x).copy(),
                "y": np.asarray(y).copy(),
                "hours": np.asarray(hours).copy(),
                "weight": np.asarray(weight).copy(),
                "gray": gray,
                "no_ice_weight": no_ice_weight,
            }
        )
        return np.asarray([[1.0 + float(np.asarray(hours)[0]) * 0.01]])

    def superpixel(self, sca, x, y):
        return np.arange(len(x), dtype=int)


class RecordingReaderCore:
    @staticmethod
    def optics_at(wavelength_nm, n, kappa, b, query_wavelengths):
        return None, None, np.ones_like(query_wavelengths, dtype=float)

    @staticmethod
    def no_ice_transmission(reflectance, fresnel):
        return np.ones_like(reflectance, dtype=float)


def test_standard_comparison_caches_reader_spectra_and_checks_public_band_response():
    model = RecordingReaderModel()
    core = RecordingReaderCore()
    source_ids = np.asarray(["star-a", "star-b"])
    first_pass, second_pass = comparison.amber.PASSES[:2]
    common = pd.DataFrame(
        {
            "source_id": ["star-a", "star-a", "star-b"],
            "filter": ["F062", "F062", "F062"],
            "pass": [first_pass, second_pass, first_pass],
            "x_pixel": [10.0, 11.0, 20.0],
            "y_pixel": [30.0, 31.0, 40.0],
            "mjd_amber": [51544.0, 51544.0, 51543.9],
        }
    )
    wavelengths = np.asarray([500.0, 600.0])
    photon_shapes = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    area_by_filter = np.ones((len(comparison.amber.FILTERS), 2))

    _prediction, _valid, details = comparison.evaluate_model_on_sample(
        model,
        core=core,
        detector="WFI01",
        source_ids=source_ids,
        seds=None,
        common=common,
        wavelengths_nm=wavelengths,
        photon_shapes=photon_shapes,
        area_by_filter=area_by_filter,
    )

    assert "IceResponseModel.response" in details["evaluation_method"]
    assert details["unique_reader_response_time_calls"] == 1
    assert details["unique_position_time_spectra"] == 2
    assert details["band_response_validation_calls"] == 2
    assert details["model_valid_measurements"] == 2
    assert len(model.response_calls) == 1
    assert model.response_calls[0][0] == 1
    assert model.response_calls[0][-1] is True
    np.testing.assert_array_equal(model.response_calls[0][1], [10.0, 11.0])
    assert len(model.band_response_calls) == 2
    assert all(call["sca"] == 1 for call in model.band_response_calls)
    assert all(call["gray"] is True for call in model.band_response_calls)
    assert all(call["no_ice_weight"] is True for call in model.band_response_calls)
    np.testing.assert_allclose(model.band_response_calls[0]["weight"], [50.0, 100.0])
    np.testing.assert_allclose(model.band_response_calls[1]["weight"], [50.0, 100.0])
