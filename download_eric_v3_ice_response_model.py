"""Download Eric's v3 release HDF5 and verify it against its S3 manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


DEFAULT_OUTPUT_DIR = Path.cwd()
BUCKET = "smdc-rom-com-ops"
PREFIX = "rrn-ops/activities/car-190-rcs-spectro-trending/ice_response_model/v3"
MODEL_NAME = "ice_response_model_v3.h5"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aws_s3_copy(uri, destination):
    try:
        subprocess.run(
            ["aws", "s3", "cp", uri, str(destination), "--only-show-errors"],
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("AWS CLI is required on Nexus to download the v3 model") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"AWS CLI failed to download {uri}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing local HDF5 file")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / MODEL_NAME
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest_temp = manifest_path.with_suffix(".json.partial")
    aws_s3_copy(f"s3://{BUCKET}/{PREFIX}/MANIFEST.json", manifest_temp)
    manifest = json.loads(manifest_temp.read_text())
    expected_hash = str(manifest["sha256"])
    expected_size = int(manifest["size_bytes"])
    if model_path.is_file() and not args.force:
        if (model_path.stat().st_size == expected_size
                and sha256_file(model_path) == expected_hash):
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            manifest_temp.unlink(missing_ok=True)
            print(f"Verified existing v3 model: {model_path}")
            return
        manifest_temp.unlink(missing_ok=True)
        raise FileExistsError(f"Existing model differs from the v3 manifest: {model_path}; use --force")

    temporary = model_path.with_suffix(model_path.suffix + ".partial")
    try:
        aws_s3_copy(f"s3://{BUCKET}/{PREFIX}/{MODEL_NAME}", temporary)
        actual_hash = sha256_file(temporary)
        actual_size = temporary.stat().st_size
        if actual_hash != expected_hash or actual_size != expected_size:
            raise ValueError(
                f"Downloaded v3 model does not match manifest: size={actual_size}, sha256={actual_hash}"
            )
        temporary.replace(model_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_temp.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified v3 model: {model_path}")


if __name__ == "__main__":
    main()
