import hashlib
from pathlib import Path

import pytest

import upload_zane_gaussian_code_s3 as uploader


def package(tmp_path):
    names=("fit_car201_zane_star_reflectance_modulation.py","run_gaussian_reproduction.py",
           "README.md","INPUTS.json","requirements.txt","reference_results/WFI01/model.csv")
    checksums=[]
    for name in names:
        file=tmp_path/name
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(name)
        checksums.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {name}\n")
    (tmp_path/"SHA256SUMS").write_text("".join(checksums))
    return names


def test_dry_run_uses_code_subdirectory_and_preserves_nested_files(tmp_path,monkeypatch):
    names=package(tmp_path)
    commands=[]
    monkeypatch.setattr(uploader.subprocess,"run",lambda command,check:commands.append(command))
    uploader.main(["--source-dir",str(tmp_path),"--dry-run"])
    command=commands[0]
    assert "s3://roman-s3-romcom/rges-pit/ice_model/zane_gaussian_reproduction_code/" in command
    assert "reference_results/WFI01/model.csv" in command
    assert command[-1]=="--dryrun"
    assert "--delete" not in command
    assert "--profile" not in command


def test_modified_file_rejected_before_upload(tmp_path,monkeypatch):
    package(tmp_path)
    (tmp_path/"README.md").write_text("changed")
    monkeypatch.setattr(uploader.subprocess,"run",lambda *args,**kwargs:pytest.fail("Unexpected AWS call"))
    with pytest.raises(ValueError,match="Checksum mismatch"):
        uploader.main(["--source-dir",str(tmp_path)])


def test_manifest_cannot_escape_package(tmp_path):
    package(tmp_path)
    with (tmp_path/"SHA256SUMS").open("a") as stream:
        stream.write("0  ../outside.txt\n")
    with pytest.raises(ValueError,match="escapes package"):
        uploader.validate_source(tmp_path)
