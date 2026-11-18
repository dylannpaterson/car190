from pathlib import Path

import numpy as np
import pytest

import fit_wfi_tmm_production as production
import fit_wfi_relative_response as relative_fit
import wfi_relative_response as relative_response
from wfi_relative_response import (
    CACHE_VERSION,
    MODEL_PROVENANCE_FIELDS,
    PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
    evaluate_spectral_response_cache,
    integrate_spectral_response_cache,
    load_warm_values,
    load_spectral_response_cache,
    load_response_cache,
    load_optical_values,
    resolve_portable_path,
    statistics_fingerprint,
    validate_response_cache_provenance,
    write_response_cache,
    write_spectral_response_cache,
    _prior_residuals,
    _led_log_offset_values,
)


def _write_provenance_fit(path, statistics_path, recorded_statistics_path=None):
    values = {
        "statistics_path": np.asarray(
            str((recorded_statistics_path or statistics_path).resolve())
        ),
        "statistics_fingerprint": np.asarray(statistics_fingerprint(statistics_path)),
    }
    for name in MODEL_PROVENANCE_FIELDS:
        values[name] = np.asarray(
            {
                "sed_mode": "core",
                "sed_weighting": "energy",
                "map_optimizer": "least-squares",
                "map_parameter_scaling": "prior",
                "trajectory_regularization": "maxent",
            }.get(name, 1.0)
        )
    np.savez_compressed(path, **values)


def test_external_cache_provenance_paths_relocate_with_repository(tmp_path, monkeypatch):
    repository = tmp_path / "new_checkout"
    relative_statistics = Path(
        "20261109/corrected_statistics_4h_thinned/production/wfi01/"
        "tile_statistics/tile_000.csv"
    )
    relative_fit = Path(
        "20261109/corrected_statistics_4h_thinned/production/wfi01/"
        "fits/tile_000/posterior_samples.npz"
    )
    statistics = repository / relative_statistics
    fit = repository / relative_fit
    statistics.parent.mkdir(parents=True)
    fit.parent.mkdir(parents=True)
    statistics.write_text(
        "group,n_pair_valid,sum_log_change,sum_squared_log_change\n"
        "0,4,0.1,0.2\n",
        encoding="utf-8",
    )

    old_root = Path("/old_machine/workspace/MAST_2026-09-14T1134")
    old_statistics = old_root / relative_statistics
    old_fit = old_root / relative_fit
    _write_provenance_fit(
        fit,
        statistics,
        recorded_statistics_path=old_statistics,
    )
    monkeypatch.setattr(relative_response, "REPOSITORY_ROOT", repository)

    provenance = relative_response.source_provenance(old_fit, old_statistics)
    cache_path = repository / "response_cache.npz"
    np.savez_compressed(
        cache_path,
        detector=np.asarray("WFI01"),
        tile_id=np.asarray(0),
        source_fit=np.asarray(str(old_fit)),
        source_statistics=np.asarray(str(old_statistics)),
        source_fit_sha256=np.asarray(provenance["source_fit_sha256"]),
        source_statistics_sha256=np.asarray(provenance["source_statistics_sha256"]),
        source_statistics_fingerprint=np.asarray(
            provenance["source_statistics_fingerprint"]
        ),
        source_model_fingerprint=np.asarray(provenance["source_model_fingerprint"]),
    )

    validated = validate_response_cache_provenance(cache_path)
    assert validated["source_fit"] == relative_fit.as_posix()
    assert validated["source_statistics"] == relative_statistics.as_posix()
    from run_wfi_relative_response_production import cache_source_paths

    resolved_fit, resolved_statistics = cache_source_paths(cache_path)
    assert resolved_fit == fit.resolve()
    assert resolved_statistics == statistics.resolve()
    assert resolve_portable_path(old_fit, repository) == fit.resolve()
    missing_repository = tmp_path / "checkout_without_source_files"
    assert resolve_portable_path(old_fit, missing_repository) == (
        missing_repository / relative_fit
    ).resolve()

    from mock_fit_wfi_star_reflectance_modulation import source_fit_from_cache

    assert source_fit_from_cache(cache_path, "WFI01", 0) == fit.resolve()


def test_spectral_response_cache_round_trip(tmp_path):
    wavelengths = np.linspace(480.0, 2700.0, 9)
    thickness = np.linspace(0.0, 500.0, 7)
    relative = (
        0.001 * thickness[:, None]
        + 0.00001 * wavelengths[None, :]
        + 0.0000001 * thickness[:, None] * wavelengths[None, :]
    )
    spectral = {
        "wavelengths_nm": wavelengths,
        "ice_thickness_nm": thickness,
        "clean_power": 0.5 + wavelengths / 5000.0,
        "relative_log_response": relative,
    }
    path = write_spectral_response_cache(
        tmp_path / "spectral_response.npz",
        "WFI01",
        44,
        spectral,
        "/workspace/source-fit.npz",
        rank=3,
    )
    cache = load_spectral_response_cache(path, detector="WFI01", tile_id=44)
    clean, response = evaluate_spectral_response_cache(
        cache,
        np.asarray([125.0, 375.0]),
        np.asarray([757.5, 1867.5]),
    )
    expected_clean = np.interp([757.5, 1867.5], wavelengths, spectral["clean_power"])
    query_thickness = np.asarray([125.0, 375.0])
    expected_response = (
        0.001 * query_thickness[:, None]
        + 0.00001 * np.asarray([757.5, 1867.5])[None, :]
        + 0.0000001 * query_thickness[:, None] * np.asarray([757.5, 1867.5])[None, :]
    )
    np.testing.assert_allclose(clean, expected_clean, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(response, expected_response, rtol=0.0, atol=2.0e-7)


def test_spectral_response_cache_rejects_out_of_range_queries(tmp_path):
    spectral = {
        "wavelengths_nm": np.asarray([480.0, 500.0]),
        "ice_thickness_nm": np.asarray([0.0, 2.0]),
        "clean_power": np.ones(2),
        "relative_log_response": np.zeros((2, 2)),
    }
    path = write_spectral_response_cache(
        tmp_path / "spectral_response.npz",
        "WFI01",
        0,
        spectral,
        "/workspace/source-fit.npz",
        rank=1,
    )
    cache = load_spectral_response_cache(path)
    with pytest.raises(ValueError, match="outside"):
        evaluate_spectral_response_cache(cache, [3.0], [490.0])
    with pytest.raises(ValueError, match="outside"):
        evaluate_spectral_response_cache(cache, [1.0], [501.0])


def test_spectral_response_cache_integrates_multiple_filters(tmp_path):
    wavelengths = np.asarray([480.0, 500.0, 520.0, 540.0])
    thickness = np.asarray([0.0, 2.0, 4.0])
    relative = 0.0005 * thickness[:, None] * wavelengths[None, :]
    spectral = {
        "wavelengths_nm": wavelengths,
        "ice_thickness_nm": thickness,
        "clean_power": np.ones(4),
        "relative_log_response": relative,
    }
    path = write_spectral_response_cache(
        tmp_path / "spectral_response.npz",
        "WFI01",
        0,
        spectral,
        "/workspace/source-fit.npz",
        rank=1,
    )
    cache = load_spectral_response_cache(path)
    weights = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    result = integrate_spectral_response_cache(cache, [2.0], wavelengths, weights)
    expected = np.log(
        np.asarray(
            [
                np.trapezoid(np.exp(relative[1, :]) * weights[0], wavelengths)
                / np.trapezoid(weights[0], wavelengths),
                np.trapezoid(np.exp(relative[1, :]) * weights[1], wavelengths)
                / np.trapezoid(weights[1], wavelengths),
            ]
        )
    )
    np.testing.assert_allclose(result[0], expected, rtol=0.0, atol=2.0e-7)


def test_relative_response_cache_records_and_validates_current_provenance(tmp_path):
    statistics = tmp_path / "tile_statistics.csv"
    statistics.write_text(
        "tile_id,group,n_pair_valid,sum_log_change,sum_squared_log_change\n"
        "0,0,2,0.1,0.2\n"
    )
    fit = tmp_path / "posterior_samples.npz"
    _write_provenance_fit(fit, statistics)
    context = {
        "wavelengths": np.asarray([480.0, 500.0]),
        "model_weights": np.ones((1, 1, 2)),
        "qe_columns": np.asarray([0]),
    }
    data = {
        "tile_ids": np.asarray([0]),
        "sca_names": np.asarray(["WFI01"]),
    }
    cache_path = tmp_path / "relative_response.npz"
    write_response_cache(
        cache_path,
        data,
        context,
        {
            "ar_tile_raw": np.zeros(9),
            "absolute_hgcdte_n_log_scale": np.asarray(0.0),
            "absolute_hgcdte_k_log_scale": np.asarray(0.0),
            "ice_n_log_constant": np.asarray(0.0),
            "ar_n_log_modulation": np.zeros((3, 6)),
            "hgcdte_n_log_modulation": np.zeros(6),
            "hgcdte_k_log_modulation": np.zeros(6),
            "incidence_angle_deg": np.asarray(0.0),
        },
        np.asarray([0.0, 2.0]),
        np.zeros((6, 2)),
        fit,
        statistics,
        0.1,
    )
    with np.load(cache_path, allow_pickle=False) as source:
        assert int(np.asarray(source["cache_version"]).item()) == CACHE_VERSION
        assert "source_model_fingerprint" in source
    load_response_cache(cache_path, data, context)
    validate_response_cache_provenance(cache_path, fit, statistics)

    statistics.write_text(
        "tile_id,group,n_pair_valid,sum_log_change,sum_squared_log_change\n"
        "0,1,2,0.3,0.4\n"
    )
    with pytest.raises(ValueError, match="statistics"):
        validate_response_cache_provenance(cache_path, fit, statistics)


def test_optical_values_default_missing_incidence_to_normal(tmp_path):
    path = tmp_path / "fixed_normal_fit.npz"
    np.savez_compressed(
        path,
        ar_tile_raw=np.zeros(9),
        absolute_hgcdte_n_log_scale=np.asarray(0.0),
        absolute_hgcdte_k_log_scale=np.asarray(0.0),
        ice_n_log_constant=np.asarray(0.0),
        ar_n_log_modulation=np.zeros((3, 6)),
        hgcdte_n_log_modulation=np.zeros(6),
        hgcdte_k_log_modulation=np.zeros(6),
    )
    values = load_optical_values(path)
    assert float(np.asarray(values["incidence_angle_deg"]).item()) == 0.0


def test_optical_values_load_fixed_sites_from_response_cache(tmp_path):
    cache_path = tmp_path / "relative_response.npz"
    np.savez_compressed(
        cache_path,
        ice_n_log_constant=np.asarray(0.3),
        ar_n_log_modulation=np.full((3, 6), 0.4),
        hgcdte_n_log_modulation=np.full(6, 0.5),
        hgcdte_k_log_modulation=np.full(6, 0.6),
    )
    fit_path = tmp_path / "spline_fit.npz"
    np.savez_compressed(
        fit_path,
        ar_tile_raw=np.zeros(9),
        absolute_hgcdte_n_log_scale=np.asarray(0.1),
        absolute_hgcdte_k_log_scale=np.asarray(0.2),
        response_cache=np.asarray(str(cache_path)),
    )
    values = load_optical_values(fit_path)
    np.testing.assert_allclose(values["ice_n_log_constant"], 0.3)
    np.testing.assert_allclose(values["ar_n_log_modulation"], 0.4)
    np.testing.assert_allclose(values["hgcdte_n_log_modulation"], 0.5)
    np.testing.assert_allclose(values["hgcdte_k_log_modulation"], 0.6)


def test_warm_values_use_current_rate_prior_not_source_trajectory(tmp_path, monkeypatch):
    initial = {
        "ar_tile_raw": np.zeros(9),
        "initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "reset_initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "post_reset_tile_log_offset": np.zeros((1, 1)),
        "absolute_hgcdte_n_log_scale": np.zeros((1, 1)),
        "absolute_hgcdte_k_log_scale": np.zeros((1, 1)),
        "incidence_angle_deg": np.asarray(0.0),
        "np_log_rate_nodes": np.zeros((1, 1, 3)),
    }
    monkeypatch.setattr(
        "wfi_relative_response.production.initial_values_for_data",
        lambda data, constant_rate=False: {
            name: np.asarray(value).copy() for name, value in initial.items()
        },
    )
    source = tmp_path / "source_fit.npz"
    np.savez_compressed(
        source,
        ar_tile_raw=np.full(9, 2.0),
        absolute_hgcdte_n_log_scale=np.asarray([[0.1]]),
        absolute_hgcdte_k_log_scale=np.asarray([[0.2]]),
        ice_n_log_constant=np.asarray([[0.3]]),
        ar_n_log_modulation=np.full((1, 1, 3, 6), 0.4),
        hgcdte_n_log_modulation=np.full((1, 1, 6), 0.5),
        hgcdte_k_log_modulation=np.full((1, 1, 6), 0.6),
        incidence_angle_deg=np.asarray(0.0),
        node_instantaneous_rate_nm_day=np.asarray([3.0, 4.0, 5.0]),
    )
    values = load_warm_values(
        {"node_days": np.asarray([0.0, 1.0, 2.0])},
        source,
        tmp_path / "old_statistics.csv",
    )
    np.testing.assert_allclose(
        values["np_log_rate_nodes"],
        np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
    )
    np.testing.assert_allclose(values["ar_tile_raw"], 2.0)
    np.testing.assert_allclose(values["absolute_hgcdte_n_log_scale"], 0.1)


def test_warm_values_override_third_decon_initial_thickness_start(tmp_path, monkeypatch):
    initial = {
        "ar_tile_raw": np.zeros(9),
        "initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "reset_initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "third_decon_initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "post_reset_tile_log_offset": np.zeros((1, 1)),
        "post_third_decon_tile_log_offset": np.zeros((1, 1)),
        "absolute_hgcdte_n_log_scale": np.zeros((1, 1)),
        "absolute_hgcdte_k_log_scale": np.zeros((1, 1)),
        "incidence_angle_deg": np.asarray(0.0),
        "np_log_rate_nodes": np.zeros((1, 1, 3)),
    }
    monkeypatch.setattr(
        "wfi_relative_response.production.initial_values_for_data",
        lambda data, constant_rate=False: {
            name: np.asarray(value).copy() for name, value in initial.items()
        },
    )
    source = tmp_path / "source_fit.npz"
    np.savez_compressed(
        source,
        ar_tile_raw=np.zeros(9),
        absolute_hgcdte_n_log_scale=np.asarray([[0.1]]),
        absolute_hgcdte_k_log_scale=np.asarray([[0.2]]),
        incidence_angle_deg=np.asarray(0.0),
        node_instantaneous_rate_nm_day=np.asarray([3.0, 4.0, 5.0]),
    )
    values = load_warm_values(
        {"node_days": np.asarray([0.0, 1.0, 2.0])},
        source,
        tmp_path / "old_statistics.csv",
        third_decon_initial_thickness_start_nm=40.0,
    )
    np.testing.assert_allclose(
        values["third_decon_initial_thickness_log_nm"], np.log(40.0)
    )


def test_configured_warmup_prior_uses_each_reset_events_own_sigma(tmp_path):
    events = (
        {
            "name": "second_decon",
            "day": 12.0,
            "thickness_prior": {"median_nm": 10.0, "log_sigma": 1.5},
            "h5_prior": {"center": "configured_median", "log_sigma": 1.5},
        },
        {
            "name": "third_decon",
            "day": 38.0,
            "thickness_prior": {"median_nm": 10.0, "log_sigma": 1.5},
            "relative_response_warmup": {"start_nm": 40.0, "log_sigma": 0.1},
            "h5_prior": {"center": "base_h5", "log_sigma": 0.1},
        },
        {
            "name": "fourth_decon",
            "day": 55.0,
            "thickness_prior": {"median_nm": 20.0, "log_sigma": 0.8},
            "relative_response_warmup": {"start_nm": 25.0, "log_sigma": 0.25},
            "h5_prior": {"center": "base_h5", "log_sigma": 0.5},
        },
    )
    data = {
        "reset_days": np.asarray([12.0, 38.0, 55.0]),
        "reset_event_names": tuple(event["name"] for event in events),
        "reset_events": events,
    }
    thickness_names, _offset_names = production.reset_site_names(data)
    archive_path = tmp_path / "fits/tile_000/constant_rate_map.npz"
    archive_path.parent.mkdir(parents=True)
    np.savez_compressed(
        archive_path,
        **{
            name: np.asarray([np.log(value)])
            for name, value in zip(
                thickness_names[1:], (12.0, 42.0, 27.0), strict=True
            )
        },
    )

    centers, sigmas = relative_fit.warmup_thickness_prior(
        tmp_path, 0, data, prior_sigma_log=None, prior_stages="configured"
    )

    assert set(centers) == set(thickness_names)
    assert sigmas[thickness_names[1]] == pytest.approx(1.5)
    assert centers[thickness_names[1]] == pytest.approx(np.log(10.0))
    assert centers[thickness_names[2]] == pytest.approx(np.log(42.0))
    assert sigmas[thickness_names[2]] == pytest.approx(0.1)
    assert centers[thickness_names[3]] == pytest.approx(np.log(27.0))
    assert sigmas[thickness_names[3]] == pytest.approx(0.25)


def test_third_decon_relative_prior_includes_third_initial_thickness():
    jax, jnp, _, _ = production.require_numpyro()
    data = {
        "days": np.arange(7.0),
        "node_days": np.arange(7.0),
        "reset_days": np.asarray([2.0, 5.0]),
    }
    values = {
        "initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "reset_initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "third_decon_initial_thickness_log_nm": np.asarray([[np.log(10.0)]]),
        "np_log_rate_nodes": np.zeros((1, 1, 7)),
        "post_reset_tile_log_offset": np.zeros((1, 1)),
        "post_third_decon_tile_log_offset": np.zeros((1, 1)),
    }
    residual = np.asarray(
        _prior_residuals(data, values, np.zeros((1, 1, 7)), jnp)
    )
    assert residual.size == 12


def test_led_offsets_apply_only_to_nonreference_groups():
    import jax.numpy as jnp

    data = {
        "led": np.asarray([1, 2, 1, 2]),
        "repeat": np.asarray([0, 0, 1, 2]),
    }
    values = {"led_log_offsets": np.asarray([[0.1, -0.2]])}
    np.testing.assert_allclose(
        np.asarray(_led_log_offset_values(data, values, jnp)),
        [[0.0, 0.0, 0.1, -0.2]],
    )
