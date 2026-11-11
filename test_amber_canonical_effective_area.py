import pytest

import run_amber_student_t_detectors as run


@pytest.mark.parametrize("detector",["WFI01","WFI05","WFI18"])
def test_default_run_selects_own_detector_and_separate_output(tmp_path,monkeypatch,detector):
    monkeypatch.setattr(run,"DETECTOR_EFFECTIVE_AREA_DIR",tmp_path)
    number=int(detector[3:])
    area=tmp_path/f"Roman_effarea_v8_SCA{number:02d}_20240301.ecsv"
    area.write_text("reference fixture")
    selected,output=run.canonical_configuration(detector)
    assert selected==area.resolve()
    assert output.name.endswith(f"_sca{number:02d}_effective_area")
    assert output!=run.paths(detector)[1]


def test_missing_detector_curve_never_falls_back_to_sca05(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"DETECTOR_EFFECTIVE_AREA_DIR",tmp_path)
    (tmp_path/"Roman_effarea_v8_SCA05_20240301.ecsv").write_text("only SCA05 exists")
    with pytest.raises(FileNotFoundError,match="not used as a fallback"):
        run.canonical_configuration("WFI01")


def test_explicit_historical_override_remains_possible(tmp_path):
    area=tmp_path/"sca05.ecsv"
    area.write_text("legacy table")
    selected,output=run.canonical_configuration("WFI01",area,tmp_path/"control")
    assert selected==area.resolve()
    assert output==(tmp_path/"control").resolve()
