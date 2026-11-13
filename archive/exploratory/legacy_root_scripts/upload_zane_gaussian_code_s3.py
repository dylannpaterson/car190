"""Upload the unzipped Gaussian reproduction code using normal Nexus AWS credentials."""

import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

DEFAULT_BUCKET="roman-s3-romcom"
DEFAULT_PREFIX="rges-pit/ice_model/zane_gaussian_reproduction_code"


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def validate_source(source_dir):
    source_dir=Path(source_dir).resolve()
    required=("fit_car201_zane_star_reflectance_modulation.py","run_gaussian_reproduction.py",
              "README.md","INPUTS.json","requirements.txt","SHA256SUMS")
    for name in required:
        if not (source_dir/name).is_file():raise FileNotFoundError(source_dir/name)
    names=[]
    for number,line in enumerate((source_dir/"SHA256SUMS").read_text().splitlines(),1):
        if not line.strip():continue
        fields=line.split(None,1)
        if len(fields)!=2:raise ValueError(f"Invalid SHA256SUMS line {number}")
        expected,relative=fields
        relative=relative.lstrip(" *")
        path=(source_dir/relative).resolve()
        if source_dir not in path.parents:raise ValueError(f"Path escapes package: {relative}")
        if not path.is_file():raise FileNotFoundError(path)
        if sha256(path)!=expected:raise ValueError(f"Checksum mismatch: {relative}")
        if relative not in names:names.append(relative)
    if not names:raise ValueError("No package checksums found")
    for name in required:
        if name!="SHA256SUMS" and name not in names:
            raise ValueError(f"Required package file is missing from checksums: {name}")
    return source_dir,names


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,default=Path(__file__).resolve().parent,
                        help="Unzipped code-package directory; defaults to this script's directory")
    parser.add_argument("--bucket",default=DEFAULT_BUCKET)
    parser.add_argument("--prefix",default=DEFAULT_PREFIX)
    parser.add_argument("--aws-cli",default="aws")
    parser.add_argument("--region")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    source,names=validate_source(args.source_dir)
    bucket=args.bucket.strip()
    prefix=args.prefix.strip().strip("/")
    if not bucket or not prefix:raise ValueError("Bucket and prefix must be nonempty")
    destination=f"s3://{bucket}/{prefix}/"
    command=[args.aws_cli]
    if args.region:command.extend(["--region",args.region])
    command.extend(["s3","sync",str(source),destination,"--no-progress","--exclude","*"])
    for name in names+["SHA256SUMS"]:command.extend(["--include",name])
    command.extend(["--exclude","__pycache__/*","--exclude","*.pyc"])
    if args.dry_run:command.append("--dryrun")
    else:command.append("--only-show-errors")
    print(f"Validated {len(names)} package checksums",flush=True)
    print(f"Destination: {destination}",flush=True)
    print(f"AWS command: {shlex.join(command)}",flush=True)
    subprocess.run(command,check=True)


if __name__=="__main__":
    main()
