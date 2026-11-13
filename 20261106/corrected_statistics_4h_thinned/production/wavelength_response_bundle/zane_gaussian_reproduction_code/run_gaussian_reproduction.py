"""Reproduce the saved Zane Gaussian fit using the copied code only."""
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
