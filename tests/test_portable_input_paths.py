from pathlib import Path

from wfi_pipeline import (
    assess_rcs_spectral_identifiability as spectral_inputs,
    compare_eric_v3_amber_car201 as h5_reader_inputs,
    fit_wfi_tmm_production as tmm_inputs,
    heatmap_car201_allstar_model_correction as car201_inputs,
    run_amber_student_t_detectors as amber_inputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_data_defaults_are_resolved_inside_the_repository():
    paths = (
        Path(tmm_inputs.SED_URL),
        Path(tmm_inputs.QE_URL),
        Path(spectral_inputs.SED_URL),
        Path(spectral_inputs.QE_URL),
        Path(amber_inputs.AMBER),
        Path(car201_inputs.PHOTOMETRY_DIR),
        Path(car201_inputs.FAINT_SED_PATH),
        Path(car201_inputs.ORIGINAL_SED_PATH),
        Path(car201_inputs.SED_GRID_PATH),
        Path(h5_reader_inputs.DEFAULT_SOURCE_ZIP),
    )
    for path in paths:
        assert path.resolve().is_relative_to(REPOSITORY_ROOT)

    assert "car201_data" in amber_inputs.AMBER.parts
    assert "car201_data" in car201_inputs.PHOTOMETRY_DIR.parts
    assert "external_reference_data" in h5_reader_inputs.DEFAULT_SOURCE_ZIP.parts
