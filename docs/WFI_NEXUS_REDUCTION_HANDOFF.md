# WFI Nexus Reduction Handoff

The calibrated-data reducer runs on Nexus only. Keep its runtime, MAST access,
S3 reads, and telemetry queries on Nexus; the local model-pipeline refactor
starts from the corrected-statistics directory transferred from the dated
Nexus run. This document records the operator command and artifact contract; it
does not replace or change the Nexus implementation.

## Run on Nexus

Run the maintained Nexus checkout with its normal credential and software
environment. For the production 256-pixel tile layout, explicitly pass
`--tile-size 256`:

```bash
./run_wfi_reduction_pipeline.py --tile-size 256
```

The default reduces WFI01–WFI18 sequentially; use `--detector WFI##` for one
detector, or `--workers N` to reduce several detectors concurrently. The
pipeline's default run root is date-named (`YYYYMMDD`); pass `--run-root PATH`
only when the desired dated data root is elsewhere. A single-detector example is:

```bash
./run_wfi_reduction_pipeline.py \
  --detector WFI01 \
  --tile-size 256 \
  --run-root /path/to/wfi_data/20261113
```

The Nexus installation uses the local `mast_api_token.txt`, the Nexus MAST
supplement, the CAR190 calibrated-product S3 prefix, explicit PD2
photodiode-telemetry queries, and the `WFI_MCE_ADC_WTSM00` detector-temperature
stream. Images are streamed into memory; there is no persistent S3 image cache.
The side-swap, photodiode, ACM, and count-space temperature/decon corrections
are applied during this reduction stage.

## Handoff artifacts

Each dated run root contains:

```text
<run-root>/
  raw_statistics/          # uncorrected detector-count sufficient statistics
  statistics/              # side-swap-corrected, photodiode-scaled statistics
  corrected_statistics/    # count-space corrections already applied
  corrections/             # side-swap and count-correction tables/provenance
  telemetry/               # PD2 and WTSM00 telemetry used by the run
  diagnostics/             # reduction/correction diagnostics
  all_wfi_rcs_flat_index.csv
  *_rrn_inventory.csv
  pipeline.log
  pipeline_manifest.json
```

Transfer the dated run root (or at minimum its `corrected_statistics/` folder
and manifest) to the local model environment. Use the manifest to verify the
source run and the selected tile size. The canonical correction input for model
fitting is the corrected statistics; the same count-correction table must not
be applied again.

The local pipeline's default production input is the 4-hour, 256-pixel derived
directory. It is made from the transferred corrected statistics without
changing them:

```bash
python wfi_pipeline/thin_wfi_statistics.py \
  --source-dir /path/to/20261113/corrected_statistics \
  --output-dir /path/to/20261113/corrected_statistics_4h_thinned \
  --spacing-hours 4 \
  --tile-size 256
```

This thinning preserves all Program 1039 rows and selects complete Program 1063
six-LED cycles. It does not average the sufficient statistics or apply another
correction. Keep the original `corrected_statistics/` tree as the canonical
audit input and use the output manifest to record which thinned input was fit.

## Nexus dependencies and troubleshooting boundary

Use the Nexus deployment's existing `run_wfi_reduction_pipeline.py` and its
installed MAST/ASDF/telemetry dependencies. Do not try to run the reduction
from the local model checkout: Nexus credentials, MAST connectivity, and the
reduction runtime are intentionally not part of the local fit pipeline. A
failed or partial reduction is not a valid handoff; require a complete
`pipeline_manifest.json` and all expected detector corrected-statistics files
before transferring the run.
