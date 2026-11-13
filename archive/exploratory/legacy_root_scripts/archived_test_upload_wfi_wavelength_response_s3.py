from pathlib import Path
import pytest

import upload_wfi_wavelength_response_s3 as uploader


def test_existing_cli_contract_is_preserved():
    args = uploader.parse_args(
        ["--bucket", "roman-s3-romcom", "--prefix", "rges-pit/ice_model", "--dry-run"]
    )
    assert args.bucket == "roman-s3-romcom"
    assert args.prefix == "rges-pit/ice_model"
    assert args.dry_run
    assert not hasattr(args, "aws_profile")


def test_refresh_checksums_updates_replaced_bundle(tmp_path):
    files = (
        "wfi_all_2nm_rank16.npz",
        "wfi_wavelength_response_bundle.py",
        "README.md",
        "WFI_WAVELENGTH_RESPONSE_BUNDLE.md",
        "MANIFEST.txt",
        "upload_wfi_wavelength_response_s3.py",
    )
    for name in files:
        (Path(tmp_path) / name).write_text(name)
    (Path(tmp_path) / "SHA256SUMS").write_text(
        "0  wfi_all_2nm_rank16.npz\n"
        "0  wfi_wavelength_response_bundle.py\n"
        "0  README.md\n"
        "0  WFI_WAVELENGTH_RESPONSE_BUNDLE.md\n"
        "0  MANIFEST.txt\n"
        "0  upload_wfi_wavelength_response_s3.py\n"
    )

    assert uploader.refresh_checksums(tmp_path) == 6
    source, checked = uploader.validate_source(tmp_path)
    assert source == Path(tmp_path).resolve()
    assert checked == 6


def test_destination_uri():
    assert uploader.destination_uri("bucket", "/prefix/") == "s3://bucket/prefix"


def test_stale_manifest_entries_for_absent_package_files_are_dropped(tmp_path):
    (Path(tmp_path) / "wfi_all_2nm_rank16.npz").write_text("bundle")
    (Path(tmp_path) / "upload_wfi_wavelength_response_s3.py").write_text("script")
    (Path(tmp_path) / "SHA256SUMS").write_text(
        "old  wfi_all_2nm_rank16.npz\nold  wfi_wavelength_response_bundle.py\n"
    )

    assert uploader.refresh_checksums(tmp_path) == 2
    _source, checked = uploader.validate_source(tmp_path)
    assert checked == 2


def test_main_keeps_existing_sync_command(tmp_path, monkeypatch):
    required = (
        "wfi_all_2nm_rank16.npz",
        "wfi_wavelength_response_bundle.py",
        "README.md",
        "WFI_WAVELENGTH_RESPONSE_BUNDLE.md",
        "MANIFEST.txt",
    )
    for name in required:
        (Path(tmp_path) / name).write_text(name)
    commands = []

    def fake_run(command, check):
        commands.append((command, check))

    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    uploader.main(
        [
            "--source-dir",
            str(tmp_path),
            "--bucket",
            "roman-s3-romcom",
            "--prefix",
            "rges-pit/ice_model",
            "--dry-run",
        ]
    )

    command, check = commands[0]
    assert check is True
    assert command[0:4] == ["aws", "s3", "sync", str(Path(tmp_path).resolve())]
    assert "s3://roman-s3-romcom/rges-pit/ice_model" in command
    assert "--profile" not in command


def test_dated_model_manifest_entry_is_not_uploaded_or_deleted(tmp_path):
    original = Path(tmp_path) / "wfi_all_2nm_rank16.npz"
    original.write_text("original model")
    dated = Path(tmp_path) / "old_dated_model.npz"
    dated.write_text("keep this local model")
    (Path(tmp_path) / "SHA256SUMS").write_text(
        "old  wfi_all_2nm_rank16.npz\nold  old_dated_model.npz\n"
    )
    with pytest.raises(ValueError, match="two stable model filenames"):
        uploader.refresh_checksums(tmp_path)
    assert dated.read_text() == "keep this local model"
