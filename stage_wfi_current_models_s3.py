"""Stage both current optical models and shared documentation for Nexus upload."""

import json
from pathlib import Path
import shutil

import numpy as np

import upload_wfi_wavelength_response_s3 as uploader
from wfi_wavelength_response_bundle import load_bundle


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "20261106/corrected_statistics_4h_thinned/production/wavelength_response_bundle"
STAGE = MODEL_ROOT / "s3_export"
MODELS = {
    "wfi_all_2nm_rank16.npz": MODEL_ROOT / "wfi_all_2nm_rank16.npz",
    "wfi_all_2nm_rank16_modulated.npz": MODEL_ROOT / "wfi_all_2nm_rank16_joint_car190_amber_2day_modulated.npz",
}


def main():
    for source in MODELS.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    STAGE.mkdir(parents=True, exist_ok=True)
    manifest = [
        "Current best Roman WFI ice models: unmodulated and modulated",
        "S3 destination: s3://roman-s3-romcom/rges-pit/ice_model/",
        "Stable remote filenames; dated source provenance retained locally and in each archive.",
        "",
    ]
    for name, source in MODELS.items():
        bundle = load_bundle(source)
        metadata = bundle["metadata"]
        if name.endswith("_modulated.npz"):
            if metadata.get("modulation_fit_family") != "joint-car190-amber-2day":
                raise ValueError("The canonical modulated export must use the joint CAR190/Amber 2-day fit")
            sources = metadata["modulation_fit_sources"]
            if len(sources) != 18 or any(
                item.get("photometry") not in ("Amber + CAR190", "Amber CAR201 + CAR190")
                or item.get("nu") != 4.
                or item.get("tile_specific_ice_history_refitted") is not True
                or item.get("rate_control_spacing_days") != 2.0
                for item in sources.values()
            ):
                raise ValueError("Incomplete joint-fit provenance in current modulated archive")
            if not bundle.get("trajectory_included") or metadata.get("modulation_zero_at_led_centers") is not False:
                raise ValueError("Joint bundle must include updated ice histories and record free LED modulation")
        wavelengths = np.asarray(bundle["wavelengths_nm"])
        thickness = np.asarray(bundle["ice_thickness_nm"])
        manifest.extend([
            f"File: {name}",
            f"Local source: {source.relative_to(ROOT)}",
            f"Bundle version: {int(np.asarray(bundle['bundle_version']).item())}",
            f"Detectors: {', '.join(sorted(set(np.asarray(bundle['detectors']).astype(str))))}",
            f"Tile count: {len(bundle['tile_ids'])}",
            f"Wavelength range: {wavelengths.min():g}–{wavelengths.max():g} nm",
            f"Thickness range: {thickness.min():g}–{thickness.max():g} nm",
            f"Response rank: {int(np.asarray(bundle['response_rank']).item())}",
            f"Modulation fit family: {metadata.get('modulation_fit_family', 'unmodulated')}",
            *( ["Joint constraints: CAR190 production Gaussian plus Amber's CAR201 photometry (Student-t, nu=4)",
                 "Ice trajectory: joint independent per-tile positive-rate controls at 2-day cadence",
                 f"Amber observations: {sum(int(item['amber_measurements']) for item in metadata['modulation_fit_sources'].values())}",
                 f"CAR190 observations: {sum(int(item['car190_observations']) for item in metadata['modulation_fit_sources'].values())}",
                 f"Ice parameter count: {sum(int(item['total_ice_parameters']) for item in metadata['modulation_fit_sources'].values())}",
                 f"LED-center modulation is free; max |ratio-1|: {max(metadata['led_center_max_abs_complex_deviation_by_detector'].values()):.6g}"]
               if name.endswith("_modulated.npz") else [] ),
            "Grey reset/decon offsets retained as auxiliary provenance; not CAR201 roll offsets.",
            "",
        ])
        shutil.copy2(source, STAGE / name)
    for name in ("wfi_wavelength_response_bundle.py", "upload_wfi_wavelength_response_s3.py"):
        shutil.copy2(ROOT / name, STAGE / name)
    for name in ("WFI_WAVELENGTH_RESPONSE_BUNDLE.md", "WFI_MODULATED_WAVELENGTH_RESPONSE_BUNDLE.md"):
        shutil.copy2(ROOT / "docs" / name, STAGE / name)
    (STAGE / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    (STAGE / "README.md").write_text('''# Current best WFI ice-response models

This directory publishes both models together at
`s3://roman-s3-romcom/rges-pit/ice_model/`:

- `wfi_all_2nm_rank16.npz`: unmodulated optical response.
- `wfi_all_2nm_rank16_modulated.npz`: optical response from the joint CAR190 +
  Amber's CAR201 photometry fit. It contains detector-specific complex-reflection
  modulation and jointly refitted per-tile ice histories using 2-day positive rate
  controls. Amber uses Student-t nu=4; CAR190 uses its production group weighting.
  LED-center modulation is free in this joint model and recorded per detector.
- `wfi_wavelength_response_bundle.py`: shared loader/interpolator/integrator.
- The two model documents, manifest, uploader, and SHA256 checksums.

Both archives describe tile-specific ice histories and wavelength/thickness
response surfaces before downstream SED, filter, or QE integration. The
modulated bundle's 2-day joint-fit histories are sampled on the original
785-point CAR190 bundle time axis; bundle arrays and coordinate grids retain the
existing format. The joint fit used 603,235 Amber CAR201 measurements, 3,617,280 CAR190
group observations, and 152,064 tile-trajectory parameters across WFI01–WFI18.
No stellar SEDs, per-star normalizations, or fitted roll zero-points are embedded.
Grey reset/decon offsets are retained from the original production fits.
Modulation and updated-trajectory provenance are recorded in the metadata.
Effective-area curves were fit inputs; the exported response remains
wavelength-resolved before downstream SED/filter/QE integration.

The joint fit is an in-sample sensitivity model: its starting ice histories were
derived from CAR190, and CAR190 also contributes to the joint likelihood. The
archive metadata records this model family and both fit sources.

## Minimal use

```python
import numpy as np
from wfi_wavelength_response_bundle import load_bundle, evaluate

bundle = load_bundle("wfi_all_2nm_rank16_modulated.npz")
clean_power, relative_log_response = evaluate(
    bundle, tile_id=136, ice_thickness_nm=np.array([0.0, 50.0, 100.0]),
    wavelengths_nm=np.array([1000.0, 1500.0, 2000.0]), detector="WFI01",
)
relative_response = np.exp(relative_log_response)
```

See the model documents for time-history access, integration conventions and
limitations. In particular, saved CAR201 fit diagnostics included roll offsets
and a different clean-power weighting; they do not measure identical performance
for integrations of this portable optical product. Trajectory uncertainties are
not included. `MANIFEST.txt` and archive metadata retain source provenance.

## Upload from Nexus

Run inside this staged directory using your normal AWS/Nexus credentials:

```bash
python upload_wfi_wavelength_response_s3.py --source-dir . \\
  --bucket roman-s3-romcom --prefix rges-pit/ice_model --dry-run
```

Remove `--dry-run` to upload both current models and accompanying files.
Matching remote keys are updated; unrelated remote keys are not deleted.
''')
    count = uploader.refresh_checksums(STAGE)
    uploader.validate_source(STAGE)
    print(f"Staged both models and {count} checksummed files in {STAGE}")


if __name__ == "__main__":
    main()
