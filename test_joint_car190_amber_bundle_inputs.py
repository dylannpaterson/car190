import json
from pathlib import Path

import numpy as np

import build_modulated_wfi_wavelength_response_bundle as builder
import run_amber_student_t_detectors as amber
import support_aligned_modulation
from experiment_joint_wfi01_fixed_ice import full_basis
from per_tile_rate_modulation import PerTileRateModulation


def test_joint_modulation_loader_reconstructs_detector_curve(tmp_path, monkeypatch):
    detector = "WFI01"
    root = Path(tmp_path)
    joint_dir = builder._joint_experiment_dir(
        root, detector, "car190_amber_joint_per_tile_2day_rate_test"
    )
    fixed_dir = builder._joint_experiment_dir(
        root, detector, "car190_amber_joint_fixed_ice_full_reflection_test"
    )
    joint_dir.mkdir(parents=True)
    fixed_dir.mkdir(parents=True)
    area = root / "Roman_effarea_v8_SCA01_20240301.ecsv"
    area.write_text("fake effective area")
    area.with_suffix(".provenance.json").write_text(json.dumps({"source": "fixture"}))
    monkeypatch.setattr(amber, "canonical_configuration", lambda _detector: (area, root))
    old = root / "old-amber-inputs"
    old.mkdir()
    (old / "car201_actual_star_modulation_fit_summary.json").write_text(json.dumps({
        "ice_history": {"optical_state_source": str(root / "optical-state")},
        "modulation": {
            "maxent_regularization": {"weight": 1000.0},
            "phase_smoothness_regularization": {"weight": 100.0},
        },
    }))
    monkeypatch.setattr(amber, "paths", lambda _detector: (old, root))
    monkeypatch.setattr(support_aligned_modulation, "effective_area_support", lambda _path: (474.0, 2333.0))

    fit_wave = np.arange(470.0, 2701.0, 2.0)
    basis = full_basis(fit_wave, (474.0, 2333.0))[0]
    modes = basis.shape[0]
    theta = np.r_[np.linspace(-0.04, 0.05, modes), np.linspace(0.03, -0.02, modes)]
    fixed_summary = {"detector": detector, "optimizer_success": True, "LED_observations": 200960}
    joint_summary = {
        "detector": detector,
        "optimizer_success": True,
        "rate_control_spacing_days": 2.0,
        "ice_parameters_per_tile": 33,
        "total_ice_parameters": 8448,
        "stellar_stars": 500,
        "stellar_measurements": 30000,
        "reset_days": [12.9, 38.8],
    }
    (fixed_dir / "joint_fit_summary.json").write_text(json.dumps(fixed_summary))
    (joint_dir / "summary.json").write_text(json.dumps(joint_summary))
    np.savez_compressed(fixed_dir / "joint_fit.npz", wavelength_nm=fit_wave, basis=basis)
    np.savez_compressed(joint_dir / "joint_weekly_ice_fit.npz", shared_parameters=theta)

    target_wave = np.array([470.0, 500.0, 623.520885362797, 1000.0, 2333.0, 2700.0])
    result = builder._load_detector_modulation(root, detector, target_wave,
                                               fit_kind=builder.JOINT_MODULATION_FIT)
    expected = np.exp(theta[:modes] @ full_basis(target_wave, (474.0, 2333.0))[0]
                      + 1j * (theta[modes:] @ full_basis(target_wave, (474.0, 2333.0))[0]))
    np.testing.assert_allclose(result["ratio"], expected)
    assert result["fit_provenance"]["tile_specific_ice_history_refitted"] is True
    assert result["fit_provenance"]["modulation_zero_at_led_centers"] is False
    assert result["anchor_error"] > 0


def test_joint_ice_history_loader_retains_fit_and_computes_rates(tmp_path, monkeypatch):
    import experiment_joint_wfi01_weekly_ice as weekly_fit

    times = np.arange(9.0)
    resets = np.array([3.0, 6.0])
    monkeypatch.setattr(weekly_fit, "RESETS", resets)
    baseline = np.array([
        [1.0, 3.0, 5.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0],
        [2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 1.0, 2.0, 5.0],
    ])
    # The public archive is tile-major with 256 rows; fill all rows with a
    # valid baseline but vary one representative tile for checking the output.
    full_baseline = np.tile(baseline[:1], (256, 1))
    full_baseline[1] = baseline[1]
    model = PerTileRateModulation(times, full_baseline, resets, spacing_days=2.0)
    params = np.zeros((256, model.parameter_count))
    fitted = model.evaluate(params)
    archive_path = Path(tmp_path) / "joint.npz"
    np.savez_compressed(archive_path, times=times, baseline_thickness_nm=full_baseline,
                        fitted_thickness_nm=fitted, ice_parameters=params)
    modulation_item = {
        "joint_ice_archive_path": archive_path,
        "joint_ice_summary": {"reset_days": resets.tolist()},
    }
    monkeypatch.setattr(builder, "_load_joint_detector_modulation",
                        lambda *_args, **_kwargs: modulation_item)

    result = builder._load_joint_ice_history(tmp_path, "WFI01")
    np.testing.assert_allclose(result["thickness"], full_baseline)
    assert result["rates"].shape == full_baseline.shape
    assert np.all(result["rates"] >= 0)
    np.testing.assert_allclose(result["rates"][0], [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(result["rates"][1],
                               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 3.0, 3.0])


def test_joint_history_samples_onto_existing_car190_axis():
    joint_times = np.arange(5.0)
    joint_history = {
        "times": joint_times,
        "thickness": np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                [10.0, 20.0, 30.0, 40.0, 50.0]]),
        "rates": np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                            [10.0, 10.0, 10.0, 10.0, 10.0]]),
        "archive_path": Path("joint.npz"),
    }
    car190_days = np.array([0.0, 2.0, 4.0])
    thickness, rate = builder._sample_joint_tile_ice(joint_history, 1, car190_days)
    np.testing.assert_array_equal(thickness, [10.0, 30.0, 50.0])
    np.testing.assert_array_equal(rate, [10.0, 10.0, 10.0])
