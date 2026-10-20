"""List Eric RCS Trend v4 S3 objects and optionally inspect one HDF5 file.

The normal mode only calls ``aws s3api list-objects-v2`` and downloads nothing.
With ``--inspect-first-cal`` or ``--inspect-key``, exactly one HDF5 object is
downloaded to a temporary directory, its layout is printed, and the temporary
file is removed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


DEFAULT_S3_URI = (
    "s3://smdc-rom-com-ops/rrn-ops/activities/"
    "car-190-rcs-spectro-trending/rcs_trend_v4/"
)


def parse_s3_uri(uri):
    parsed = urlparse(str(uri))
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def run_aws(arguments, aws_cli="aws", aws_profile=None):
    command = [str(aws_cli)]
    if aws_profile:
        command.extend(("--profile", str(aws_profile)))
    command.extend(str(argument) for argument in arguments)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"AWS CLI executable was not found: {aws_cli!r}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"AWS command failed ({result.returncode}): "
            f"{' '.join(command)}\n{detail}"
        )
    return result.stdout


def list_objects(s3_uri, aws_cli="aws", aws_profile=None):
    bucket, prefix = parse_s3_uri(s3_uri)
    payload = json.loads(
        run_aws(
            (
                "s3api",
                "list-objects-v2",
                "--bucket",
                bucket,
                "--prefix",
                prefix,
                "--output",
                "json",
            ),
            aws_cli=aws_cli,
            aws_profile=aws_profile,
        )
    )
    objects = []
    for item in payload.get("Contents", ()):
        key = str(item.get("Key", ""))
        if not key or key == prefix:
            continue
        relative = key[len(prefix) :] if key.startswith(prefix) else key
        objects.append(
            {
                "key": key,
                "relative": relative,
                "size_bytes": int(item.get("Size", 0)),
                "last_modified": item.get("LastModified", ""),
                "etag": item.get("ETag", ""),
                "storage_class": item.get("StorageClass", ""),
            }
        )
    objects.sort(key=lambda item: item["key"])
    return bucket, prefix, objects


def short_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def inspect_hdf5(path):
    if h5py is None:
        raise RuntimeError(
            "HDF5 inspection requires h5py; the S3 inventory listing does not."
        )
    print(f"\nHDF5: {path}")
    with h5py.File(path, "r") as handle:
        print("Attributes:")
        for key in sorted(handle.attrs):
            print(f"  {key} = {short_value(handle.attrs[key])}")
        print("Groups and datasets:")

        def visitor(name, item):
            if isinstance(item, h5py.Group):
                print(f"  Group: {name or '/'}")
                for key in sorted(item.attrs):
                    print(f"    attr {key} = {short_value(item.attrs[key])}")
                return
            if isinstance(item, h5py.Dataset):
                print(
                    f"  {name}: shape={item.shape} dtype={item.dtype} "
                    f"chunks={item.chunks} compression={item.compression}"
                )
                for key in sorted(item.attrs):
                    print(f"    attr {key} = {short_value(item.attrs[key])}")

        handle.visititems(visitor)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-uri", default=DEFAULT_S3_URI)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eric_rcs_s3_inventory.json"),
        help="Inventory JSON output path",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=40,
        help="Maximum object rows printed to stdout; JSON always contains all",
    )
    inspect = parser.add_mutually_exclusive_group()
    inspect.add_argument(
        "--inspect-first-cal",
        action="store_true",
        help="Download and inspect the first cal/*.h5 object",
    )
    inspect.add_argument(
        "--inspect-key",
        help="Download and inspect a relative key such as cal/example_cal.h5",
    )
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--aws-profile", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bucket, prefix, objects = list_objects(
        args.s3_uri,
        aws_cli=args.aws_cli,
        aws_profile=args.aws_profile,
    )
    cal_objects = [
        item
        for item in objects
        if item["relative"].startswith("cal/")
        and Path(item["relative"]).suffix.lower() in {".h5", ".hdf5"}
    ]
    payload = {
        "s3_uri": args.s3_uri,
        "bucket": bucket,
        "prefix": prefix,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_count": len(objects),
        "cal_hdf5_count": sum(
            Path(item["relative"]).suffix.lower() in {".h5", ".hdf5"}
            for item in cal_objects
        ),
        "objects": objects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"S3 prefix: {args.s3_uri}")
    print(f"Objects: {len(objects)}; cal HDF5: {payload['cal_hdf5_count']}")
    for item in objects[: max(args.max_display, 0)]:
        print(
            f"{item['size_bytes']:>12}  {item['last_modified']}  "
            f"{item['relative']}"
        )
    if len(objects) > args.max_display:
        print(f"... {len(objects) - args.max_display} more; see {args.output}")
    print(f"Wrote {args.output}")

    inspect_key = args.inspect_key
    if args.inspect_first_cal:
        if not cal_objects:
            raise ValueError("No cal/*.h5 object is available")
        inspect_key = cal_objects[0]["relative"]
    if inspect_key:
        if inspect_key.startswith(prefix):
            key = inspect_key
        else:
            key = prefix + inspect_key.lstrip("/")
        object_keys = {item["key"] for item in objects}
        if key not in object_keys:
            raise ValueError(f"Object is not listed below the prefix: {key}")
        with tempfile.TemporaryDirectory(prefix=".eric_rcs_inspect_") as directory:
            path = Path(directory) / Path(key).name
            run_aws(
                (
                    "s3",
                    "cp",
                    f"s3://{bucket}/{key}",
                    str(path),
                    "--only-show-errors",
                ),
                aws_cli=args.aws_cli,
                aws_profile=args.aws_profile,
            )
            inspect_hdf5(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
