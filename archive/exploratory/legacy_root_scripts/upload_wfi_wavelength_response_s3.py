"""Upload the staged WFI wavelength-response export to an S3 prefix.

The script uses the AWS CLI's normal Nexus credential chain, refreshes the
staged SHA256SUMS file, validates it, and never deletes remote objects.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess


KNOWN_FILES = (
    "wfi_all_2nm_rank16.npz",
    "wfi_all_2nm_rank16_modulated.npz",
    "wfi_wavelength_response_bundle.py",
    "README.md",
    "WFI_WAVELENGTH_RESPONSE_BUNDLE.md",
    "WFI_MODULATED_WAVELENGTH_RESPONSE_BUNDLE.md",
    "MANIFEST.txt",
    "SHA256SUMS",
)
MODEL_FILES = {"wfi_all_2nm_rank16.npz", "wfi_all_2nm_rank16_modulated.npz"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Staged export directory; defaults to this script's directory.",
    )
    parser.add_argument("--bucket", required=True, help="Destination S3 bucket name.")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Destination key prefix, without the s3://bucket/ part.",
    )
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--region")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the AWS sync plan without uploading objects.",
    )
    return parser.parse_args(argv)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_paths(source_dir):
    source_dir = Path(source_dir)
    paths = []
    seen = set()
    checksum_path = Path(source_dir) / "SHA256SUMS"
    if checksum_path.is_file():
        for line in checksum_path.read_text().splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2:
                continue
            relative = fields[1].lstrip(" *")
            if (
                relative != "SHA256SUMS"
                and relative not in seen
                and (source_dir / relative).is_file()
            ):
                paths.append(relative)
                seen.add(relative)
    for name in KNOWN_FILES:
        if name != "SHA256SUMS" and name not in seen and (source_dir / name).is_file():
            paths.append(name)
            seen.add(name)
    upload_name = "upload_wfi_wavelength_response_s3.py"
    if (source_dir / upload_name).is_file() and upload_name not in seen:
        paths.append(upload_name)
    unexpected = [name for name in paths if Path(name).suffix == ".npz" and name not in MODEL_FILES]
    if unexpected:
        raise ValueError(
            "Only the two stable model filenames may be uploaded; unexpected manifest entries: "
            + ", ".join(unexpected)
        )
    return paths


def refresh_checksums(source_dir):
    source_dir = Path(source_dir).resolve()
    rows = []
    for relative in checksum_paths(source_dir):
        path = (source_dir / relative).resolve()
        if source_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        rows.append(f"{sha256(path)}  {relative}")
    if not rows:
        raise ValueError("No files available to checksum")
    (source_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n")
    return len(rows)


def validate_source(source_dir):
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    missing = [name for name in ("wfi_all_2nm_rank16.npz", "SHA256SUMS") if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Staged export is missing required files: {', '.join(missing)}"
        )
    checked = 0
    for line_number, line in enumerate(
        (source_dir / "SHA256SUMS").read_text().splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(None, 1)
        except ValueError as error:
            raise ValueError(f"Invalid SHA256SUMS line {line_number}") from error
        relative = relative.lstrip(" *")
        path = (source_dir / relative).resolve()
        if source_dir not in path.parents:
            raise ValueError(f"Checksum path escapes source directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
        checked += 1
    if checked == 0:
        raise ValueError("SHA256SUMS contains no checksums")
    return source_dir, checked


def destination_uri(bucket, prefix):
    bucket = str(bucket).strip()
    prefix = str(prefix).strip().strip("/")
    if not bucket or not prefix:
        raise ValueError("bucket and prefix must be non-empty")
    return f"s3://{bucket}/{prefix}"


def upload(source_dir, destination, aws_cli="aws", region=None, dry_run=False):
    command = [str(aws_cli)]
    if region:
        command.extend(("--region", str(region)))
    command.extend(("s3", "sync", str(source_dir), destination, "--no-progress"))
    if not dry_run:
        command.append("--only-show-errors")
    command.extend(("--exclude", "*"))
    for relative in checksum_paths(source_dir):
        command.extend(("--include", relative))
    command.extend(("--include", "SHA256SUMS", "--exclude", "__pycache__/*", "--exclude", "*.pyc"))
    if dry_run:
        command.append("--dryrun")
    print(f"Source: {source_dir}")
    print(f"Destination: {destination}")
    print(f"AWS command: {shlex.join(command)}")
    subprocess.run(command, check=True)


def main(argv=None):
    args = parse_args(argv)
    source_dir = Path(args.source_dir).resolve()
    refreshed = refresh_checksums(source_dir)
    source_dir, checked = validate_source(source_dir)
    destination = destination_uri(args.bucket, args.prefix)
    print(f"Refreshed {refreshed} SHA256 entries")
    print(f"Validated {checked} SHA256 entries")
    upload(
        source_dir,
        destination,
        aws_cli=args.aws_cli,
        region=args.region,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
