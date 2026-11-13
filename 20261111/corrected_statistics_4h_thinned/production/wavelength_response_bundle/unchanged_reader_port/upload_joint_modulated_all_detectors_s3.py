"""Upload the current all-detector joint-modulated H5 to the model S3 prefix.

Uses the AWS CLI's standard credential chain. Only this H5 object is uploaded;
no other remote objects are synchronized or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "joint_modulated_all_detectors_current_gray_v2.h5"
DEFAULT_SOURCE = SCRIPT_DIR / MODEL_NAME
DEFAULT_BUCKET = "roman-s3-romcom"
DEFAULT_PREFIX = "rges-pit/ice_model"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--key", default=MODEL_NAME, help="S3 object key below --prefix.")
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--region")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ask the AWS CLI to show the upload without transferring the file.",
    )
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != "rcs_response_model/1":
            raise ValueError(f"Unexpected H5 schema in {path}")
        if int(h5.attrs.get("n_bin", -1)) != 256:
            raise ValueError(f"Expected n_bin=256 in {path}")
        required = ("reflectance", "deposition", "instrument", "superpixels")
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(f"H5 is missing required model groups: {missing}")
        scas = np.unique(np.asarray(h5["superpixels/sca"], dtype=int))
        if not np.array_equal(scas, np.arange(1, 19)):
            raise ValueError(f"Expected all WFI detectors 1–18; found SCA values {scas}")
    return path


def destination_uri(bucket, prefix, key):
    bucket = str(bucket).strip()
    prefix = str(prefix).strip().strip("/")
    key = str(key).strip().lstrip("/")
    key_path = PurePosixPath(key)
    if not bucket or not prefix or not key or ".." in key_path.parts:
        raise ValueError("bucket, prefix, and a safe relative key are required")
    return f"s3://{bucket}/{prefix}/{key_path.as_posix()}"


def upload(source, destination, aws_cli="aws", region=None, dry_run=False):
    command = [str(aws_cli)]
    if region:
        command.extend(("--region", str(region)))
    command.extend((
        "s3", "cp", str(source), destination,
        "--content-type", "application/x-hdf5",
        "--no-progress",
    ))
    if dry_run:
        command.append("--dryrun")
    else:
        command.append("--only-show-errors")
    print(f"Source: {source}")
    print(f"Destination: {destination}")
    print(f"SHA256: {sha256_file(source)}")
    print(f"AWS command: {shlex.join(command)}")
    subprocess.run(command, check=True)


def main(argv=None):
    args = parse_args(argv)
    source = validate_model(args.source)
    destination = destination_uri(args.bucket, args.prefix, args.key)
    upload(
        source,
        destination,
        aws_cli=args.aws_cli,
        region=args.region,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
