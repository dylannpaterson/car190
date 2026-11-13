"""Collect the Gaussian fitter's local import closure and portable run recipe."""

import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parent
PRODUCTION=ROOT/"20261106/corrected_statistics_4h_thinned/production"
OUTPUT=PRODUCTION/"wavelength_response_bundle/zane_gaussian_reproduction_code"


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            result.update(chunk)
    return result.hexdigest()


def local_modules():
    pending=["fit_car201_zane_star_reflectance_modulation"]
    found={}
    while pending:
        name=pending.pop()
        if name in found:continue
        source=ROOT/(name+".py")
        imports=set()
        for node in ast.walk(ast.parse(source.read_text())):
            names=([item.name.split(".")[0] for item in node.names] if isinstance(node,ast.Import)
                   else [(node.module or "").split(".")[0]] if isinstance(node,ast.ImportFrom) else [])
            for item in names:
                if (ROOT/(item+".py")).is_file():imports.add(item)
        found[name]=sorted(imports)
        pending.extend(imports-found.keys())
    return found


LAUNCHER='''"""Reproduce the saved Zane Gaussian fit using the copied code only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root",type=Path,required=True,
                        help="Root containing 20261020/ and 20261106/ input trees")
    parser.add_argument("--car201-root",type=Path,
                        help="CAR201 root; defaults to a CAR201 sibling of workspace-root")
    parser.add_argument("--detector",default="WFI01",choices=[f"WFI{i:02d}" for i in range(1,19)])
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--check-inputs",action="store_true",help="Check inputs and imports without fitting")
    parser.add_argument("--verify-input-hashes",action="store_true")
    parser.add_argument("--fit-systematic-floor",action="store_true",
                        help="Optional later fitted-floor experiment; NOT the original Gaussian configuration")
    args=parser.parse_args()
    import numpy as np
    import fit_car201_zane_star_reflectance_modulation as fitter
    import mock_fit_wfi_star_reflectance_modulation as mock
    manifest=json.loads((HERE/"INPUTS.json").read_text())
    root=args.workspace_root.resolve()
    car=(args.car201_root or root.parent/"CAR201").resolve()
    config=manifest["detectors"][args.detector]
    bases={"workspace":root,"car201":car}
    required=manifest["global_files"]+config["files"]
    for entry in required:
        path=bases[entry["root"]]/entry["path"]
        if not path.is_file():raise FileNotFoundError(path)
        if args.verify_input_hashes:
            digest=hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
            if digest.hexdigest()!=entry["sha256"]:raise ValueError(f"Input checksum mismatch: {path}")
    # Original optical caches store absolute source-fit paths. Resolve those
    # through the selected workspace without editing any cache or fitted data.
    source_paths=manifest["optical_source_paths"]
    def portable_source_fit(path,detector,tile_id):
        with np.load(path,allow_pickle=False) as archive:
            cached_detector=str(np.asarray(archive["detector"]).reshape(-1)[0])
            cached_tile=int(np.asarray(archive["tile_id"]).reshape(-1)[0])
            embedded=str(np.asarray(archive["source_fit"]).reshape(-1)[0])
        if cached_detector.upper()!=detector.upper() or cached_tile!=tile_id:
            raise ValueError("Optical cache detector/tile mismatch")
        relative=source_paths[detector.upper()][str(tile_id)]
        original=Path(manifest["original_workspace_root"])/relative
        if Path(embedded)!=original:
            raise ValueError(f"Cache source provenance changed: {embedded}")
        source=root/relative
        if not source.is_file():raise FileNotFoundError(source)
        return source
    mock.source_fit_from_cache=portable_source_fit
    fitter.source_fit_from_cache=portable_source_fit
    lower=args.detector.lower()
    output=(args.output_dir or root/"20261106/corrected_statistics_4h_thinned/production"/lower
            /"spline_relative_response_normal/experiments"/f"car201_zane_{lower}_gaussian_reproduction")
    command=["--detector",args.detector,
             "--photometry-dir",str(car/"zanes_photometry/20261106"),
             "--sed-posteriors",str(car/"sed/v2/faint-posteriors-v2.h5"),
             "--sed-grid",str(car/"sed/v2/btsettl-spectral-grid.h5"),
             "--effective-area",str(HERE/"external_reference_data/Roman_effarea_v8_SCA05_20240301.ecsv"),
             "--optical-cache-root",str(root/config["optical_cache_root"]),
             "--ice-history-fit-root",str(root/config["ice_history_fit_root"]),
             "--pass-set","ep3v2_ep4","--allow-partial-source-coverage","--exclude-repeat-outliers",
             "--basis","per-gap","--wavelength-stride-nm","5",
             "--amplitude-bound","0.30","--phase-bound-rad","1.0",
             "--maxent-weight","1000","--phase-smoothness-weight","100",
             "--output-dir",str(output)]
    for item in config["manual_measurement_exclusions"]:
        command.extend(["--exclude-measurement",item["source_id"],item["filter"],item["pass"]])
    if args.fit_systematic_floor:command.append("--fit-systematic-floor")
    print(f"Validated {len(required)} input files for {args.detector}; code imported from {Path(fitter.__file__).resolve()}")
    print("Fit arguments:",json.dumps(command))
    print("Expected original fit:",json.dumps(config["expected_fit"]))
    if args.check_inputs:
        print("Preflight only: no fit run")
        return
    return fitter.main(command)


if __name__=="__main__":
    raise SystemExit(main())
'''


README='''# Zane Gaussian photometry-fit reproduction code

This is a snapshot of the main Gaussian fitter and **all nine local Python
modules in its transitive import closure**, copied without modification.
It reproduces the stellar-photometry fitting stage given the fixed optical
states and LED-derived ice histories. It does not rerun their upstream LED fits.

## Contents

- `fit_car201_zane_star_reflectance_modulation.py` and eight supporting modules.
- `run_gaussian_reproduction.py`: explicit input/settings launcher and preflight.
- `upload_zane_gaussian_code_s3.py`: upload the unzipped package from Nexus.
- `external_reference_data/`: the reference effective area, LED SED and QE tables.
- `INPUTS.json`: required external-data files, SHA256 hashes, fixed input roots,
  and reference result counts for every detector.
- `reference_results/WFI##/`: original no-floor fit summaries, modulation
  coefficients, and roll offsets for comparison (not the photometry itself).
- `LOCAL_IMPORTS.json`, `ENVIRONMENT.json`, `requirements.txt`, `SHA256SUMS`.

The photometry, stellar SED HDF5 files, optical caches, source optical-fit
archives, and ice-history archives are **not copied** into this code package.
Their exact paths and hashes are in `INPUTS.json`.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

`ENVIRONMENT.json` records the Python and installed package versions used when
this snapshot was assembled. Requirements are not a platform-specific GPU lock;
CPU JAX can be used. Preserve the recorded environment where practical when
comparing optimizer results. Runtime BLAS/backend differences can change final
rounding or termination; compare predictions as well as coefficients.

## Run from any directory

The workspace root is the directory containing the `20261020/` and `20261106/`
data trees. CAR201 is expected as its sibling unless `--car201-root` is supplied.
For example, if data are laid out as `/data/MAST/...` and `/data/CAR201/...`:

```bash
python /path/to/zane_gaussian_reproduction_code/run_gaussian_reproduction.py \\
  --workspace-root /data/MAST --car201-root /data/CAR201 \\
  --detector WFI01 --check-inputs --verify-input-hashes
```

Then run the actual fit:

```bash
python /path/to/zane_gaussian_reproduction_code/run_gaussian_reproduction.py \\
  --workspace-root /data/MAST --car201-root /data/CAR201 \\
  --detector WFI01 --output-dir /data/MAST/20261106/corrected_statistics_4h_thinned/production/wfi01/spline_relative_response_normal/experiments/car201_zane_wfi01_gaussian_reproduction
```

Choose an explicit detector WFI01–WFI18. The launcher resolves absolute paths
stored inside optical-cache metadata against the selected workspace; it does
not rewrite caches or change the forward model. The copied fitting modules
otherwise remain byte-identical to the source snapshot. Directly running the
fitter's `--help` also works, but its historical defaults should not be used to
reconstruct the current experiment: use the launcher or explicit arguments.

## Exact original configuration

Photometry: Zane's `20261106` EP3-v2/EP4 master catalogs, 11 visits, Sep 28 through
Oct 31. SEDs: successful `sed/v2/faint-posteriors-v2.h5` posteriors, reconstructed
with the v2 BT-Settl grid. Tile assignment is per measurement, using x/y in
256-pixel tiles. Pinned per-tile optical states and `20261106` spline ice
histories are fixed inputs. The effective-area table is the included SCA05
reference curve, sampled on the fitter's 5-nm grid.

Source coverage is the union across visits. Stars with any saturated measurement
on a detector are excluded entirely; DO_NOT_USE rows are excluded. The repeat
outlier screen removes deviations greater than max(0.25 mag, 4 times the catalog
error), measured about the same star/filter/roll/tile group's median, with at
least four group observations. This is the implemented threshold; historical
CLI help text describing eight sigma is stale. Each source/filter needs at
least two usable visits. No SED chi-square or band-count quality cut is applied.

Modulation: one detector-wide per-gap complex multiplier on clean reflection,
anchored to unity at LED wavelengths. Amplitude coefficient bound +/-0.30;
phase bound +/-1 rad. MaxEnt amplitude weight 1000; phase weight 100. One free
post-roll offset per filter, beginning at EP4 pass 2. One constant magnitude per
star/filter is calculated as the catalog-error-weighted mean of observed minus
predicted magnitude over its usable visits. The likelihood compares individual
measurements, not population medians.

**The original Gaussian fits have no extra systematic floor.** To select the
later optional floor experiment, add `--fit-systematic-floor`. That is a
different fit; it should not be compared as a reproduction of the original.
The Amber Student-t fits are a separate workflow and are not this package's
target. Cross-roll offsets are unidentifiable without bridging stars in the
original implementation; compare predicted changes rather than treating those
offsets as measured calibration values.

## Compare against the original outputs

Check `car201_actual_star_modulation_fit_summary.json`, the modulation parameter
CSV, and the roll-offset CSV against `reference_results/<detector>/`. For WFI01,
the reference includes 611 SED-matched sources, 574 contributing sources and
37,526 likelihood measurements. The no-floor reduced chi-square is about 79.23.

`SHA256SUMS` checks this entire code package. Input-data hashes are separate in
`INPUTS.json`; use the preflight command above to check those without fitting.

## Upload the unzipped code from Nexus

From this directory:

```bash
python upload_zane_gaussian_code_s3.py --source-dir . --dry-run
python upload_zane_gaussian_code_s3.py --source-dir .
```

The default destination is
`s3://roman-s3-romcom/rges-pit/ice_model/zane_gaussian_reproduction_code/`.
This preserves subdirectories and uploads only checksum-listed package files
plus SHA256SUMS, rather than the ZIP. It does not upload the referenced large
input datasets or delete remote files. The normal AWS CLI/Nexus credentials
are used. The destination can be overridden with `--bucket` and `--prefix`.
'''


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    imports=local_modules()
    for name in imports:
        shutil.copy2(ROOT/(name+".py"),OUTPUT/(name+".py"))
    refs=OUTPUT/"external_reference_data"
    refs.mkdir(exist_ok=True)
    for name in ("Roman_effarea_v8_SCA05_20240301.ecsv","WFI_RCS_LEDP_spectral_energy_distribution.ecsv","WFI_Quantum_efficiency.ecsv"):
        shutil.copy2(ROOT/"external_reference_data"/name,refs/name)
    (OUTPUT/"run_gaussian_reproduction.py").write_text(LAUNCHER)
    shutil.copy2(ROOT/"upload_zane_gaussian_code_s3.py",OUTPUT/"upload_zane_gaussian_code_s3.py")
    (OUTPUT/"README.md").write_text(README)
    (OUTPUT/"LOCAL_IMPORTS.json").write_text(json.dumps(imports,indent=2)+"\n")
    dependencies=["numpy","scipy","pandas","matplotlib","astropy","h5py","extinction","pyarrow","jax","numpyro"]
    environment={"python":sys.version,"platform":platform.platform(),"packages":{}}
    for name in dependencies+["jaxlib"]:
        try:environment["packages"][name]=metadata.version(name)
        except metadata.PackageNotFoundError:environment["packages"][name]="not installed"
    (OUTPUT/"ENVIRONMENT.json").write_text(json.dumps(environment,indent=2)+"\n")
    (OUTPUT/"requirements.txt").write_text("\n".join(dependencies)+"\n")
    hashes={}
    def entry(path,base,role):
        path=Path(path).resolve()
        if path not in hashes:hashes[path]=sha256(path)
        return {"root":base,"path":str(path.relative_to(ROOT if base=="workspace" else ROOT.parent/"CAR201")),
                "role":role,"size_bytes":path.stat().st_size,"sha256":hashes[path]}
    global_files=[]
    car=ROOT.parent/"CAR201"
    passes=[f"ep3_pass{i}" for i in range(1,7)]+[f"ep4_pass{i}" for i in range(1,6)]
    for epoch in passes:global_files.append(entry(car/"zanes_photometry/20261106"/(epoch+"_master.parquet"),"car201","photometry"))
    for name in ("faint-posteriors-v2.h5","btsettl-spectral-grid.h5"):
        global_files.append(entry(car/"sed/v2"/name,"car201","stellar spectrum input"))
    manifest={"original_workspace_root":str(ROOT),"global_files":global_files,"detectors":{},"optical_source_paths":{}}
    for index in range(1,19):
        detector=f"WFI{index:02d}"
        lower=detector.lower()
        fit=PRODUCTION/lower/"spline_relative_response_normal/experiments"/f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched"
        summary=json.loads((fit/"car201_actual_star_modulation_fit_summary.json").read_text())
        optical=Path(summary["ice_history"]["optical_state_source"])
        history=PRODUCTION/lower/"spline_relative_response_normal/fits"
        files=[]
        mapping={}
        for tile in range(256):
            cache=optical/f"tile_{tile:03d}/relative_response.npz"
            with np.load(cache,allow_pickle=False) as archive:
                source=Path(str(np.asarray(archive["source_fit"]).reshape(-1)[0]))
            mapping[str(tile)]=str(source.relative_to(ROOT))
            files.extend((entry(cache,"workspace","pinned optical cache"),
                          entry(source,"workspace","optical source fit referenced by cache"),
                          entry(history/f"tile_{tile:03d}/posterior_samples.npz","workspace","fixed ice history")))
        manifest["optical_source_paths"][detector]=mapping
        manifest["detectors"][detector]={"optical_cache_root":str(optical.relative_to(ROOT)),
            "ice_history_fit_root":str(history.relative_to(ROOT)),"files":files,
            "manual_measurement_exclusions":summary["manual_measurement_exclusions"],
            "expected_fit":{"sed_matched_stars":summary["matched_source_count"],
                            "contributing_stars":summary["active_source_count_with_usable_photometry"],
                            "likelihood_measurements":summary["data_quality"]["rows_used_in_likelihood"],
                            "reduced_chi2":summary["fit"]["reduced_chi2_approx"]}}
        reference=OUTPUT/"reference_results"/detector
        reference.mkdir(parents=True,exist_ok=True)
        for name in ("car201_actual_star_modulation_fit_summary.json","car201_actual_star_modulation_parameters.csv","car201_actual_roll_zero_points.csv"):
            shutil.copy2(fit/name,reference/name)
        print(f"Indexed inputs and reference results: {detector}",flush=True)
    (OUTPUT/"INPUTS.json").write_text(json.dumps(manifest,indent=2)+"\n")
    paths=sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name!="SHA256SUMS"
                 and "__pycache__" not in path.parts)
    (OUTPUT/"SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in paths))
    archive=str(OUTPUT)+".zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in paths+[OUTPUT/"SHA256SUMS"]:
            zipped.write(path,arcname=str(Path(OUTPUT.name)/path.relative_to(OUTPUT)))
    print(f"Copied {len(imports)} local Python modules; {len(paths)} checksummed package files")
    print(OUTPUT)
    print(archive)


if __name__=="__main__":
    main()
